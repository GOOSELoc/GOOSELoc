"""Frozen foundation encoders: DINOv3 (images) and Utonia (LiDAR) with LoRA.

Utonia is imported lazily so the image-only variant runs without spconv.
Install it with ``pip install git+https://github.com/Pointcept/Utonia`` (see
``docs/MODEL.md``).
"""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn

DEFAULT_VIT = "vit_large_patch16_dinov3.lvd1689m"   # DINOv3 ViT-L/16, 1024-d
DEFAULT_UTONIA = "utonia"


def build_vit(name: str = DEFAULT_VIT, img_size: int = 1024, pretrained: bool = True) -> nn.Module:
    """DINOv3 backbone via timm (one model serves the square tile and the cameras)."""
    import timm

    return timm.create_model(name, pretrained=pretrained, num_classes=0,
                             dynamic_img_size=True, img_size=img_size)


def _import_utonia():
    try:
        import utonia
    except ImportError as e:  # pragma: no cover - depends on the environment
        raise ImportError("the LiDAR branch needs Utonia (and spconv), see docs/MODEL.md") from e
    return utonia


def build_utonia(name: str = DEFAULT_UTONIA, *, enc_patch_size: int = 1024,
                 enable_flash: bool = False) -> nn.Module:
    """Utonia point encoder with the settings used in the paper (V100-safe, no flash-attn)."""
    utonia = _import_utonia()
    cfg = {"enc_patch_size": [enc_patch_size] * 5, "enable_flash": enable_flash}
    return utonia.model.load(name, custom_config=cfg)


def build_utonia_transform(scale: float = 0.2):
    """Utonia's outdoor preprocessing: ``scale`` rescales metric coordinates
    (+-150 m -> +-30 m); no centre shift, the ego vehicle stays at the origin."""
    return _import_utonia().transform.default(scale, apply_z_positive=False)


class LoRALinear(nn.Module):
    """``base(x) + (alpha / r) * B A x`` around a frozen linear layer (B zero-initialized)."""

    def __init__(self, base: nn.Linear, rank: int = 8, alpha: int = 16) -> None:
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)
        self.lora_A = nn.Parameter(torch.zeros(rank, base.in_features))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, rank))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        self.scaling = alpha / rank

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base(x) + (x @ self.lora_A.t() @ self.lora_B.t()) * self.scaling


def apply_lora(module: nn.Module, rank: int, alpha: int,
               targets: tuple[str, ...] = ("qkv", "proj", "fc1", "fc2")) -> int:
    """Wrap every child ``nn.Linear`` named in ``targets`` with LoRA; returns the count."""
    n = 0
    for mod in module.modules():
        for cname, child in list(mod.named_children()):
            if isinstance(child, nn.Linear) and cname in targets:
                setattr(mod, cname, LoRALinear(child, rank, alpha))
                n += 1
    return n


def encode_point_cloud(utonia: nn.Module, transform, points: torch.Tensor, device: torch.device,
                       n_sectors: int) -> tuple[torch.Tensor, torch.Tensor | None]:
    """One sweep ``(N, 3)`` -> global descriptor ``(1224,)`` and ``(K, 1224)`` sector descriptors.

    Features of the two deepest pooling levels are concatenated and all levels
    are upcast to the input points (Utonia's outdoor recipe). Sectors bin the
    points by azimuth ``atan2(y, x)`` into ``n_sectors`` equal wedges and
    mean-pool each; an empty wedge falls back to the global mean. Sector
    features are detached so the heading loss does not shape the encoder.
    """
    pts = points.detach().cpu().numpy().astype(np.float32, copy=False)
    n = pts.shape[0]
    point = transform({"coord": pts, "color": np.zeros((n, 3), np.float32),
                       "normal": np.zeros((n, 3), np.float32)})
    for k in list(point.keys()):
        if isinstance(point[k], torch.Tensor):
            point[k] = point[k].to(device, non_blocking=True)
    point = utonia(point)
    for _ in range(2):
        if "pooling_parent" not in point.keys():
            break
        parent, inverse = point.pop("pooling_parent"), point.pop("pooling_inverse")
        parent.feat = torch.cat([parent.feat, point.feat[inverse]], dim=-1)
        point = parent
    while "pooling_parent" in point.keys():
        parent, inverse = point.pop("pooling_parent"), point.pop("pooling_inverse")
        parent.feat = point.feat[inverse]
        point = parent
    feat = point.feat.float()
    desc = feat.mean(dim=0)
    if n_sectors <= 0:
        return desc, None
    coord = point.coord.float()
    fdet = feat.detach()
    bearing = torch.atan2(coord[:, 1], coord[:, 0])
    idx = ((bearing + math.pi) / (2 * math.pi) * n_sectors).long().clamp(0, n_sectors - 1)
    gmean = fdet.mean(dim=0)
    sectors = torch.stack([fdet[idx == k].mean(dim=0) if (idx == k).any() else gmean
                           for k in range(n_sectors)])
    return desc, sectors
