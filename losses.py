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


from torch import einsum

from regularizers import reg_l1w, reg_l2w
from utils import simplex, sset


class CrossEntropy():
    def __init__(self, idk=None, reg_fn=None, reg_weight=0.0):
        # Self.idk is used to filter out some classes of the target mask. Use fancy indexing
        if idk is None or len(idk) == 0:
            raise ValueError("`idk` must be a non-empty sequence of class indices.")
        self.idk = idk
        self.reg_fn = reg_fn
        self.reg_weight = reg_weight
        print(f"Initialized {self.__class__.__name__} with {self.idk=}, {self.reg_fn=}, {self.reg_weight=}")

    def __call__(self, pred_softmax, weak_target, net=None):
        assert pred_softmax.shape == weak_target.shape
        assert simplex(pred_softmax)
        assert sset(weak_target, [0, 1])

        log_p = (pred_softmax[:, self.idk, ...] + 1e-10).log()
        mask = weak_target[:, self.idk, ...].float()

        loss = - einsum("bkwh,bkwh->", mask, log_p)
        loss /= mask.sum() + 1e-10

        if self.reg_fn is not None and self.reg_weight > 0:
            if self.reg_fn in [reg_l1w, reg_l2w]:
                reg_term = self.reg_fn(net=net)
            else:
                reg_term = self.reg_fn(probs=pred_softmax, target=weak_target, idk=self.idk)
            loss = loss + self.reg_weight * reg_term

        return loss


class PartialCrossEntropy(CrossEntropy):
    def __init__(self, reg_fn=None, reg_weight=0.0):
        super().__init__(idk=[1], reg_fn=reg_fn, reg_weight=reg_weight)
