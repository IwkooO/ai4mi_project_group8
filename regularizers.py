from __future__ import annotations
from typing import Sequence, Optional
import torch
import torch.nn as nn
import scipy.ndimage as ndi
import numpy as np

EPS: float = 1e-8

def _param_iter(net: nn.Module, exclude_bn_bias: bool = True):
    for p in net.parameters():
        if not p.requires_grad:
            continue
        if exclude_bn_bias and p.dim() == 1:
            continue
        yield p

def reg_l2w(*, net: nn.Module, exclude_bn_bias: bool = True, **_) -> torch.Tensor:
    terms = [p.pow(2).mean() for p in _param_iter(net, exclude_bn_bias=exclude_bn_bias)]
    return torch.stack(terms).mean() if terms else torch.tensor(0.0, device=next(net.parameters()).device)

def reg_l1w(*, net: nn.Module, exclude_bn_bias: bool = True, **_) -> torch.Tensor:
    terms = [p.abs().mean() for p in _param_iter(net, exclude_bn_bias=exclude_bn_bias)]
    return torch.stack(terms).mean() if terms else torch.tensor(0.0, device=next(net.parameters()).device)

# CE regularizers

def reg_ce_focal(
    *,
    probs: torch.Tensor,
    target: torch.Tensor,
    idk: Optional[Sequence[int]] = None,
    gamma: float = 2.0,
    alpha: Optional[float] = None,
    **_,
) -> torch.Tensor:
    assert idk is not None, "idk must be specified for reg_ce_focal"
    p = probs[:, idk, ...]
    t = target[:, idk, ...].float()

    logp = (p.clamp_min(EPS)).log()
    pt = (p * t).sum(dim=1)
    logpt = (logp * t).sum(dim=1)

    focal = ((1.0 - pt).clamp_min(EPS)).pow(gamma) * (-logpt)
    if alpha is not None:
        focal = alpha * focal

    denom = t.sum() + EPS
    return focal.sum() / denom


# non-CE regularizers

def reg_tv(*, probs: torch.Tensor, **_) -> torch.Tensor:
    dx = probs[:, :, 1:, :] - probs[:, :, :-1, :]
    dy = probs[:, :, :, 1:] - probs[:, :, :, :-1]
    return dx.abs().mean() + dy.abs().mean()


def reg_kl_uniform(*, probs: torch.Tensor, **_) -> torch.Tensor:
    plogp = (probs * (probs.clamp_min(EPS)).log()).sum(dim=1)
    return plogp.mean()

def _sdf_from_one_hot(target_oh: torch.Tensor) -> torch.Tensor:

    B, K, H, W = target_oh.shape
    sdf = torch.empty((B, K, H, W), dtype=torch.float32)
    tgt_np = target_oh.detach().cpu().numpy().astype(np.bool_)
    for b in range(B):
        for c in range(K):
            mask = tgt_np[b, c]              # True for class c pixels
            if mask.any():
                dist_out = ndi.distance_transform_edt(~mask)
                dist_in  = ndi.distance_transform_edt(mask)
                phi = dist_out - dist_in
            else:
                phi = np.zeros((H, W), dtype=np.float32)
            sdf[b, c] = torch.from_numpy(phi.astype(np.float32))
    return sdf


def reg_surface_sdf(
    *,
    probs: torch.Tensor,
    target: torch.Tensor,
    idk=None,
    **_,
) -> torch.Tensor:
    p = probs[:, idk, ...]
    t = target[:, idk, ...].float()

    phi = _sdf_from_one_hot(t).to(p.device)  # (B, K, H, W)

    loss_map = (p * phi).sum(dim=1)

    B, H, W = loss_map.shape
    return loss_map.sum() / (B * H * W)