"""The upstream Boosting3DoF model (``models_kitti.Model``) set up for GOOSELoc.

The upstream sources in ``upstream/`` (see ``setup.sh``) are imported
unchanged. :func:`import_upstream` applies the GOOSELoc accommodations at
import time:

1. ``utils.get_meter_per_pixel`` returns the resolution of the fed tile,
   0.40 m/px (upstream hard-codes the KITTI Google-Maps value, ~0.196). It
   sizes the ground-plane homography and the per-level correlation grids.
2. ``utils.get_camera_height`` returns 2.0 m (upstream KITTI: 1.65 m).
3. ``Model.sat2world`` is replaced by a device-agnostic copy with the same
   math (the original calls ``.cuda()`` and cannot run on CPU).
4. Upstream ``cross_attention.py`` imports ``turtle`` (unused), which fails
   on Python builds without tkinter; an empty stand-in is registered then.

Additionally :func:`build_model` switches off the autograd anomaly detection
that ``Model.__init__`` enables globally, and can skip the ImageNet VGG-16
download when the weights come from a checkpoint anyway.
"""

from __future__ import annotations

import sys
import warnings
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
import torch
from data import (
    CAMERA,
    CAMERA_HEIGHT_M,
    GRD_H,
    GRD_W,
    ROTATION_RANGE_DEG,
    SAT_M_PER_PX,
    SAT_SIZE,
    SHIFT_RANGE_M,
)

UPSTREAM_DIR = Path(__file__).resolve().parent / "upstream"
UPSTREAM_COMMIT = "2dc4e93c64cda8c2ea2841cb58394ddbe16a8aaa"


@dataclass(frozen=True)
class ShiConfig:
    """Arguments of the upstream ``Model`` (KITTI-3DoF defaults, GOOSELoc ranges)."""

    rotation_range: float = ROTATION_RANGE_DEG
    shift_range_lat: float = SHIFT_RANGE_M
    shift_range_lon: float = SHIFT_RANGE_M
    level: int = 3
    N_iters: int = 2
    Optimizer: str = "TransV1G2SP"
    proj: str = "CrossAttn"
    use_uncertainty: int = 1
    coe_heading_aug: float = 0.0   # read only by an upstream path not used here


def data_config() -> dict:
    """Input geometry the released models were trained with (stored in checkpoints)."""
    return {
        "camera": CAMERA, "grd_hw": [GRD_H, GRD_W], "sat_size": SAT_SIZE,
        "sat_m_per_px": SAT_M_PER_PX, "camera_height_m": CAMERA_HEIGHT_M,
        "tile_rotation": "heading - 90 + prior_offset (CCW)",
    }


def _sat2world(self, satmap_sidelength):
    """Device-agnostic copy of upstream ``Model.sat2world`` (same math).

    Adapted from Boosting3DoFAccuracy ``models_kitti.py``,
    Copyright (c) 2023 Yujiao Shi, MIT License.
    """
    import utils as shi_utils

    device = next(self.parameters()).device
    i = j = torch.arange(0, satmap_sidelength, device=device)
    ii, jj = torch.meshgrid(i, j, indexing="ij")
    uv = torch.stack([jj, ii], dim=-1).float()
    u0 = v0 = satmap_sidelength // 2
    uv_center = uv - torch.tensor([u0, v0], device=device)
    meter_per_pixel = shi_utils.get_meter_per_pixel()
    meter_per_pixel *= shi_utils.get_process_satmap_sidelength() / satmap_sidelength
    R = torch.tensor([[0.0, 1.0], [1.0, 0.0]], device=device)
    XZ = torch.einsum("ij, hwj -> hwi", meter_per_pixel * R, uv_center)
    Y = torch.zeros_like(XZ[..., 0:1])
    ones = torch.ones_like(Y)
    return torch.cat([XZ[:, :, :1], Y, XZ[:, :, 1:], ones], dim=-1)


def import_upstream(upstream_dir: Path = UPSTREAM_DIR) -> ModuleType:
    """Import upstream ``models_kitti`` with the GOOSELoc accommodations applied."""
    if not (upstream_dir / "models_kitti.py").exists():
        raise FileNotFoundError(f"{upstream_dir} has no upstream code; run setup.sh first")
    # Upstream uses top-level module names (utils, VGG, ...); refuse to mix
    # them with an unrelated module of the same name.
    existing = sys.modules.get("utils")
    if existing is not None and Path(getattr(existing, "__file__", "") or "").parent != upstream_dir:
        raise RuntimeError(f"a different top-level 'utils' module is already imported "
                           f"({getattr(existing, '__file__', '?')})")
    if str(upstream_dir) not in sys.path:
        sys.path.insert(0, str(upstream_dir))

    import utils as shi_utils

    # Patch before importing the model so every user of utils sees the values.
    shi_utils.get_meter_per_pixel = lambda lat=None, zoom=None, scale=None: SAT_M_PER_PX
    shi_utils.Camera_height = CAMERA_HEIGHT_M
    shi_utils.get_camera_height = lambda: CAMERA_HEIGHT_M

    try:
        import turtle  # noqa: F401
    except ImportError:   # no tkinter; upstream only does `from turtle import forward`
        sys.modules["turtle"] = ModuleType("turtle")
        sys.modules["turtle"].forward = None
    with warnings.catch_warnings():   # timm.models.layers deprecation, harmless
        warnings.simplefilter("ignore", FutureWarning)
        import models_kitti

    models_kitti.Model.sat2world = _sat2world
    return models_kitti


@contextmanager
def _without_imagenet_weights() -> Iterator[None]:
    """Make upstream's ``vgg16(pretrained=True)`` build an uninitialised VGG-16."""
    import torchvision

    original = torchvision.models.vgg16
    torchvision.models.vgg16 = lambda *args, **kwargs: original(weights=None)
    try:
        yield
    finally:
        torchvision.models.vgg16 = original


def build_model(config: ShiConfig | None = None, *, imagenet_init: bool = True,
                device: str | torch.device = "cpu") -> torch.nn.Module:
    """Construct the upstream model.

    ``imagenet_init`` initialises both VGG-16 encoders from ImageNet, as for
    training (downloads the torchvision weights once); disable it when a
    checkpoint is loaded afterwards.
    """
    models_kitti = import_upstream()
    args = SimpleNamespace(**asdict(config or ShiConfig()))
    if imagenet_init:
        net = models_kitti.Model(args)
    else:
        with _without_imagenet_weights():
            net = models_kitti.Model(args)
    torch.autograd.set_detect_anomaly(False)   # enabled globally by Model.__init__
    return net.to(torch.device(device))


def load_model(checkpoint: str | Path, device: str | torch.device = "cpu") -> torch.nn.Module:
    """Load a released checkpoint (``shi_imu_prior_s*.pt``) in eval mode."""
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=True)
    cfg = ckpt["config"]
    if cfg["data"] != data_config():
        raise ValueError(f"checkpoint was trained with input geometry {cfg['data']}, "
                         f"data.py provides {data_config()}")
    net = build_model(ShiConfig(**cfg["model"]), imagenet_init=False)
    net.load_state_dict(ckpt["state_dict"], strict=True)
    return net.to(torch.device(device)).eval()


@torch.no_grad()
def predict_batch(net: torch.nn.Module, batch: dict) -> tuple[np.ndarray, ...]:
    """Upstream test-mode inference on a :class:`data.ShiGOOSELoc` batch.

    Returns ``(u_m, v_m, theta_deg)``: the translation estimate in metres
    right/up of the centre of the fed tile and the residual rotation. Upstream
    rotates the translation estimate by the *ground-truth* residual before
    returning it (the frame of its labels); passing zero instead keeps the
    estimate in the frame of the fed tile, so no ground truth is used.
    """
    device = next(net.parameters()).device
    sat = batch["sat_map"].to(device)
    grd = batch["grd_img"].to(device)
    k = batch["camera_k"].to(device)
    zero = torch.zeros(sat.shape[0], 1, device=device)
    u, v, theta = net.CVattn_rot_corr(sat, grd, k, gt_heading=zero, mode="test")
    return tuple(t.detach().cpu().numpy().astype(np.float64).reshape(-1) for t in (u, v, theta))
