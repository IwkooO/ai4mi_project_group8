#!/usr/bin/env python3

# MIT License

# Copyright (c) 2025 Hoel Kervadec

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


import torch
from torch import einsum
import torch.nn as nn
import torch.nn.functional as F

from utils import simplex, sset


class CrossEntropy():
    def __init__(self, **kwargs):
        # Self.idk is used to filter out some classes of the target mask. Use fancy indexing
        self.idk = kwargs['idk']
        print(f"Initialized {self.__class__.__name__} with {kwargs}")

    def __call__(self, pred_softmax, weak_target):
        assert pred_softmax.shape == weak_target.shape
        assert simplex(pred_softmax)
        assert sset(weak_target, [0, 1])

        log_p = (pred_softmax[:, self.idk, ...] + 1e-10).log()
        mask = weak_target[:, self.idk, ...].float()

        loss = - einsum("bkwh,bkwh->", mask, log_p)
        loss /= mask.sum() + 1e-10

        return loss


class PartialCrossEntropy(CrossEntropy):
    def __init__(self, **kwargs):
        super().__init__(idk=[1], **kwargs)


class DilatedBalancedCrossEntropy(nn.Module):
    def __init__(self, dilation_radius=3, idk=None, eps=1e-6, background_class=0):
        """
        dilation_radius: radius for dilation (in voxels) of each class mask
        idk: iterable of class indices to include 
        eps: small epsilon to avoid div by zero
        background_class: which class index is the background class
        """
        super().__init__()
        self.dilation_radius = dilation_radius
        self.idk = idk
        self.eps = eps
        self.background_class = background_class
        print(f"Initialized {self.__class__.__name__}")

    def _get_dilation_kernel(self, device, ndim):
        """
        Create a cube / ball kernel for dilation of given radius
        ndim: number of spatial dims (2 or 3)
        """
        r = self.dilation_radius
        if ndim == 2:
            K = 2 * r + 1
            kernel = torch.ones((1, 1, K, K), dtype=torch.float32, device=device) # create a 2D square kernel of ones
        elif ndim == 3:
            K = 2 * r + 1
            kernel = torch.ones((1, 1, K, K, K), dtype=torch.float32, device=device)
        else:
            raise ValueError(f"Unsupported ndim {ndim} for dilation kernel")
        return kernel

    def forward(self, pred_softmax, weak_target):
        """
        pred_softmax: Tensor, shape (B, C, *spatial), softmax probabilities over classes
        weak_target: same shape, one-hot (or probability 0/1) targets
        """
        assert pred_softmax.shape == weak_target.shape, "Shape mismatch"
        B, C = pred_softmax.shape[:2]
        spatial = pred_softmax.shape[2:]
        ndim = len(spatial)

        device = pred_softmax.device
        dtype = pred_softmax.dtype

        kernel = self._get_dilation_kernel(device, ndim)

        mask = weak_target.float()


        pad_sizes = []
        for _ in range(ndim):
            pad_sizes.extend([self.dilation_radius, self.dilation_radius])
  
        mask_padded = F.pad(mask, pad_sizes, mode='constant', value=0)


        if ndim == 2:
            dilated = F.conv2d(mask_padded, weight=kernel.repeat(C,1,1,1), groups=C)
        else:  # ndim == 3 if we try 3D training
            dilated = F.conv3d(mask_padded, weight=kernel.repeat(C,1,1,1,1), groups=C)

        weight_map = torch.zeros_like(dilated, dtype=dtype, device=device)

        if self.idk is None:
            class_indices = list(range(C))
        else:
            class_indices = self.idk

        spatial_dims = tuple(range(2, 2 + ndim))
        sums = dilated.sum(dim=spatial_dims, keepdim=True)  

        for cls in class_indices:
            weight_map[:, cls, ...] = dilated[:, cls, ...] / (sums[:, cls, ...] + self.eps)

        # Optionally, downweight background class:
        # if self.background_class is not None:
        #     weight_map[:, self.background_class, ...] = weight_map[:, self.background_class, ...] * 0.0

        log_p = (pred_softmax + self.eps).log()
        ce_map = - weak_target.float() * log_p
        weighted_ce = ce_map * weight_map
        loss = weighted_ce.sum()

        total_weight = weight_map.sum()
        loss = loss / (total_weight + self.eps)

        return loss
    

class DiceLoss:
    def __init__(self, **kwargs):
        """
        kwargs may include:
          - idk: list or iterable of class indices to include in the dice loss
          - smooth: smoothing constant (to avoid zero division)
        """
        self.idk = kwargs.get('idk', None)
        self.smooth = kwargs.get('smooth', 1e-6)
        print(f"Initialized {self.__class__.__name__} with {kwargs}")

    def __call__(self, pred_softmax, weak_target):
        """
        pred_softmax: Tensor, shape (B, C, ...) — softmax probabilities over classes
        weak_target: same shape (one-hot / binary per class), values 0 or 1
        """
        assert pred_softmax.shape == weak_target.shape, "Shape mismatch"
        pred_softmax = pred_softmax.type(torch.float32)
        weak_target = weak_target.type(torch.float32)

        if self.idk is not None:
            pred_sel = pred_softmax[:, self.idk, ...]
            target_sel = weak_target[:, self.idk, ...]
        else:
            pred_sel = pred_softmax
            target_sel = weak_target

        B, K = pred_sel.shape[:2]
        spatial_dims = tuple(range(2, pred_sel.dim()))

        intersection = (pred_sel * target_sel).sum(dim=spatial_dims)
        sum_preds = pred_sel.sum(dim=spatial_dims) 
        sum_targets = target_sel.sum(dim=spatial_dims)


        denom = sum_preds + sum_targets
        dice = (2.0 * intersection + self.smooth) / (denom + self.smooth)

        # Handle classes with no ground-truth: set dice to 1 to ignore
        empty_mask = (sum_targets <= 0.0)
        if empty_mask.any():
            dice = dice.masked_fill(empty_mask, 1.0)

        loss_per_class = 1.0 - dice
        loss = loss_per_class.mean()

        return loss
    

class FocalTverskyLoss:
    def __init__(self, **kwargs):
        """
        Keyword arguments:
          - idk: list or iterable of class indices to include in the loss
          - alpha: weight for false positives (or false negatives, depending on convention)
          - beta: weight for false negatives (or false positives)
          - gamma: focal parameter (controls how much harder examples are emphasized)
          - smooth: a small constant to avoid division by zero
        """
        self.idk = kwargs.get('idk', None)
        self.alpha = kwargs.get('alpha', 0.7)
        self.beta = kwargs.get('beta', 0.3)
        self.gamma = kwargs.get('gamma', 1.0)
        self.smooth = kwargs.get('smooth', 1e-6)
        self.eps = 1e-8  # for numerical stability
        print(f"Initialized {self.__class__.__name__} with {kwargs}")

    def __call__(self, pred_softmax, weak_target):
        """
        pred_softmax: Tensor, shape (B, C, …), softmax over classes assumed
        weak_target: same shape, one-hot / binary per class mask (0/1)
        """
        assert pred_softmax.shape == weak_target.shape, "Shape mismatch between prediction and target"
        pred = pred_softmax.type(torch.float32)
        target = weak_target.type(torch.float32)

        if self.idk is not None:
            pred_sel = pred[:, self.idk, ...]
            target_sel = target[:, self.idk, ...]
        else:
            pred_sel = pred
            target_sel = target

        # Clamp predictions to avoid extreme values
        pred_sel = torch.clamp(pred_sel, self.eps, 1.0 - self.eps)
        target_sel = torch.clamp(target_sel, 0.0, 1.0)

        B, K = pred_sel.shape[:2]
        spatial_dims = tuple(range(2, pred_sel.dim()))

        intersection = (pred_sel * target_sel).sum(dim=spatial_dims)
        false_pos = ((1.0 - target_sel) * pred_sel).sum(dim=spatial_dims)
        false_neg = (target_sel * (1.0 - pred_sel)).sum(dim=spatial_dims)

        denom = intersection + self.alpha * false_pos + self.beta * false_neg + self.smooth
        denom = torch.clamp(denom, min=self.eps)

        tversky = (intersection + self.smooth) / denom
        tversky = torch.clamp(tversky, self.eps, 1.0 - self.eps) # Clamp tversky to valid range to avoid issues with pow
        
        # Use safe power operation
        if self.gamma == 1.0:
            loss_per_class = 1.0 - tversky
        else:
            base = torch.clamp(1.0 - tversky, self.eps, 1.0)
            loss_per_class = torch.pow(base, float(self.gamma))

        # Handle classes with no ground-truth: set loss to zero to ignore
        sum_targets = target_sel.sum(dim=spatial_dims)
        empty_mask = (sum_targets <= 0.0)
        if empty_mask.any():
            loss_per_class = loss_per_class.masked_fill(empty_mask, 0.0)
        
        loss_per_class = torch.where(torch.isnan(loss_per_class) | torch.isinf(loss_per_class), 
                                   torch.zeros_like(loss_per_class), loss_per_class)
        
        loss = loss_per_class.mean()

        return loss
    
class CE_DiceLoss:
    def __init__(self, ce_weight=1.0, dice_weight=1.0, **kwargs):
        self.ce_weight = float(ce_weight)
        self.dice_weight = float(dice_weight)
        self.ce = CrossEntropy(**kwargs)
        self.dice = DiceLoss(**kwargs)

    def __call__(self, pred_softmax, weak_target):
        ce_loss = self.ce(pred_softmax, weak_target)
        dice_loss = self.dice(pred_softmax, weak_target)
        return self.ce_weight * ce_loss + self.dice_weight * dice_loss