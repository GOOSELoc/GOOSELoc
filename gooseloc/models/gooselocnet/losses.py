"""Training objective (paper, Eq. 6): KL to a soft Gaussian position target plus
a soft circular cross-entropy on the 72 heading bins."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def soft_gaussian_target(patch_rc: torch.Tensor, grid: int = 64, sigma: float = 1.5) -> torch.Tensor:
    """``(B, 2)`` sub-patch ``(row, col)`` -> ``(B, grid, grid)`` normalized Gaussian."""
    idx = torch.arange(grid, dtype=torch.float32, device=patch_rc.device)
    gy, gx = torch.meshgrid(idx, idx, indexing="ij")
    d2 = (gy[None] - patch_rc[:, 0, None, None]) ** 2 + (gx[None] - patch_rc[:, 1, None, None]) ** 2
    heat = torch.exp(-0.5 * d2 / sigma ** 2)
    return heat / heat.sum(dim=(1, 2), keepdim=True).clamp_min(1e-12)


def position_loss(scores: torch.Tensor, patch_rc: torch.Tensor, sigma: float = 1.5) -> torch.Tensor:
    """Batch-mean ``KL(target || softmax(scores))`` over the patch grid."""
    B, G = scores.shape[0], scores.shape[-1]
    target = soft_gaussian_target(patch_rc, G, sigma).reshape(B, -1)
    return F.kl_div(F.log_softmax(scores.reshape(B, -1), dim=1), target, reduction="batchmean")


def heading_loss(logits: torch.Tensor, target_deg: torch.Tensor, sigma_deg: float = 10.0) -> torch.Tensor:
    """Soft circular cross-entropy with a wrapped-Gaussian label centred on the target.

    Bin ``k`` is centred at ``k * 360 / K`` (same as the decoder). The soft label
    lets the head keep mass on both theta and theta + 180 degrees.
    """
    B, K = logits.shape
    centres = torch.arange(K, device=logits.device, dtype=torch.float32) * (360.0 / K)
    th = (target_deg.float().to(logits.device) % 360.0).view(B, 1)
    d = (centres.view(1, K) - th + 180.0) % 360.0 - 180.0
    soft = torch.softmax(-(d ** 2) / (2.0 * sigma_deg ** 2), dim=1)
    return -(soft * F.log_softmax(logits.float(), dim=1)).sum(dim=1).mean()


def total_loss(out: dict, batch: dict, *, sigma_patches: float = 1.5, sigma_deg: float = 10.0,
               lambda_heading: float = 0.5, patch_px: float = 16.0) -> dict:
    """Loss on a model output and a :mod:`gooseloc.data.torch_dataset` batch."""
    patch_rc = batch["gt_xy"].to(out["scores"].device).flip(-1) / patch_px
    l_pos = position_loss(out["scores"], patch_rc, sigma_patches)
    l_head = heading_loss(out["heading_logits"], batch["gt_heading_deg"], sigma_deg)
    return {"loss": l_pos + lambda_heading * l_head, "position": l_pos, "heading": l_head}
