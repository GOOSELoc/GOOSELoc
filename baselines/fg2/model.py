# SPDX-License-Identifier: GPL-3.0-or-later
# Part of the GOOSELoc FG² baseline (see LICENSE in this directory).
"""Upstream FG² (KITTI model) set up for GOOSELoc: construction, checkpoints,
per-camera inference and the confidence-weighted camera fusion.

The upstream sources in ``upstream/`` (see ``setup.sh``) are imported as they
are; they read ``./config.ini`` at import time, so imports run from inside
that directory. Importing this module does not import upstream (or mmcv).
"""

from __future__ import annotations

import math
import os
import sys
import warnings
import zlib
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import NamedTuple

import numpy as np
import torch
from data import GRID_SIZE_M, SAT_SIZE, fg2_to_north_up

from gooseloc.data import geometry

UPSTREAM_DIR = Path(__file__).resolve().parent / "upstream"
UPSTREAM_URL = "https://github.com/vita-epfl/FG2"
UPSTREAM_COMMIT = "a55d9d13eb029ab2cccf21cff31b728c78d443b2"
CHECKPOINT_FORMAT = "gooseloc-fg2/1"

# FG²-KITTI hyper-parameters (upstream config.ini) with the GOOSELoc BEV grid.
MODEL_CONFIG = {"grd_bev_res": 41, "grd_height_res": 11, "sat_bev_res": 41,
                "grid_size_h": GRID_SIZE_M, "grid_size_v": 25.0}
NUM_MATCHES = 1024        # correspondences sampled for weighted Procrustes
UPSTREAM_SEED = 0         # [RandomSeed] of upstream config.ini
GROUND_QUERY_SHAPE = (MODEL_CONFIG["grd_bev_res"] // 2 + 1, MODEL_CONFIG["grd_bev_res"], 1024)


@contextmanager
def _in_upstream() -> Iterator[None]:
    if not (UPSTREAM_DIR / "models" / "model_kitti.py").is_file():
        raise FileNotFoundError(f"{UPSTREAM_DIR} is missing; run baselines/fg2/setup.sh first")
    previous = os.getcwd()
    os.chdir(UPSTREAM_DIR)
    try:
        yield
    finally:
        os.chdir(previous)


def import_upstream() -> SimpleNamespace:
    """Upstream FG²'s KITTI model and the utilities used here."""
    # Upstream switches torch to deterministic algorithms on import; cuBLAS
    # then needs a fixed workspace (as in upstream's kitti_train_test.py).
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    if str(UPSTREAM_DIR) not in sys.path:
        sys.path.insert(0, str(UPSTREAM_DIR))
    with _in_upstream():
        from models.model_kitti import CVM
        from models.modules import DinoExtractor
        from utils.utils import create_metric_grid, weighted_procrustes_2d
    return SimpleNamespace(CVM=CVM, DinoExtractor=DinoExtractor,
                           create_metric_grid=create_metric_grid,
                           weighted_procrustes_2d=weighted_procrustes_2d)


def _cpu_deformable_attention() -> None:
    """Make mmcv's CPU deformable attention ignore non-finite sampling points.

    FG²'s ground grid contains points in the camera plane (depth 0), whose
    projections are not finite. mmcv's CUDA kernel skips such sampling
    locations; its CPU fallback (``grid_sample``) turns them into NaN. Moving
    them outside the image makes ``grid_sample``'s zero padding skip them too.
    """
    import mmcv.ops.multi_scale_deform_attn as msda

    original = msda.multi_scale_deformable_attn_pytorch
    if getattr(original, "skips_non_finite", False):
        return

    def patched(value, spatial_shapes, sampling_locations, attention_weights):
        sampling_locations = torch.nan_to_num(sampling_locations, nan=-2.0, posinf=-2.0, neginf=-2.0)
        return original(value, spatial_shapes, sampling_locations, attention_weights)

    patched.skips_non_finite = True
    msda.multi_scale_deformable_attn_pytorch = patched


def build_model(device: str | torch.device) -> tuple[torch.nn.Module, torch.nn.Module]:
    """DINOv2 feature extractor and FG² ``CVM``, constructed like the paper's runs.

    ``CVM`` draws its fixed ground BEV queries (``grd_grid_queries``) from
    torch's RNG, which upstream seeds with 0 on import, right after
    ``DinoExtractor`` has consumed its share for initialisation. The queries
    are not trained (on GPU the attribute is not even registered as a
    parameter) and therefore missing from every checkpoint, so this order is
    kept; release checkpoints also store them (see :func:`load_checkpoint`).
    Here the queries are a constant tensor on every device. On CPU the DINOv2
    backbone runs in float32 instead of float16 and the deformable attention
    gets the fix of :func:`_cpu_deformable_attention`; the paper used GPUs.
    """
    up = import_upstream()
    device = torch.device(device)
    torch.manual_seed(UPSTREAM_SEED)
    dino = up.DinoExtractor().to(device)
    model = up.CVM(device=device, **MODEL_CONFIG).to(device)
    queries = model.grd_grid_queries
    if isinstance(queries, torch.nn.Parameter):      # registered (CPU): keep it fixed
        queries.requires_grad_(False)
    else:                                            # plain attribute (GPU): drop autograd
        model.grd_grid_queries = queries.detach()
    if device.type == "cpu":
        dino.dinov2_vitl14.float()
        dino.amp_dtype = torch.float32
        _cpu_deformable_attention()
    return dino, model


def load_checkpoint(model: torch.nn.Module, path: str | Path) -> dict:
    """Load FG² weights into ``model`` and return the checkpoint's config.

    Accepts the released checkpoint files and plain
    state dicts (e.g. the FG² authors' KITTI ``model.pt``; config ``{}``).
    """
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    if ckpt.get("format") == CHECKPOINT_FORMAT:
        state, queries, config = ckpt["model"], ckpt["ground_queries"], ckpt["config"]
    else:
        state, queries, config = ckpt, None, {}
    state = {k.removeprefix("module."): v for k, v in state.items()}
    missing, unexpected = model.load_state_dict(state, strict=False)
    missing = [k for k in missing if k != "grd_grid_queries"]   # a parameter only on CPU
    if missing or unexpected:
        raise RuntimeError(f"{path}: missing keys {missing}, unexpected keys {unexpected}")
    if queries is not None:
        current = model.grd_grid_queries
        if not torch.equal(current.detach().cpu(), queries):
            warnings.warn("constructed ground queries differ from the checkpoint's; "
                          "using the checkpoint's", stacklevel=2)
        with torch.no_grad():
            current.copy_(queries.to(current.device))
    return config


def scene_generator(scene_id: str, seed: int, device: str | torch.device) -> torch.Generator:
    """Random stream for the match sampling of one scene (order-independent)."""
    g = torch.Generator(device=device)
    g.manual_seed((zlib.crc32(scene_id.encode()) + 7919 * seed) & 0xFFFFFFFF)
    return g


class ViewPrediction(NamedTuple):
    """FG²'s estimate for one camera view."""

    t_px: np.ndarray      # translation in 504-view pixels, [y_off, x_off] (like gt_loc)
    yaw_rad: float        # heading relative to the view's up-axis, clockwise positive
    confidence: float     # total matching probability of the sampled correspondences


class FG2Localizer:
    """Per-view FG² inference: matching, match sampling and weighted Procrustes."""

    def __init__(self, dino: torch.nn.Module, model: torch.nn.Module,
                 device: str | torch.device, num_matches: int = NUM_MATCHES) -> None:
        up = import_upstream()
        self.dino, self.model = dino.eval(), model.eval()
        self.device = torch.device(device)
        self.num_matches = num_matches
        self._procrustes = up.weighted_procrustes_2d
        self._sat_grid = up.create_metric_grid(
            GRID_SIZE_M, MODEL_CONFIG["sat_bev_res"], batch_size=1)[0].to(self.device)
        self._grd_grid = up.create_metric_grid(
            GRID_SIZE_M, MODEL_CONFIG["grd_bev_res"], batch_size=1, only_front=True)[0].to(self.device)

    @torch.no_grad()
    def predict_views(self, sat: torch.Tensor, grd: torch.Tensor, camera_k: torch.Tensor,
                      generator: torch.Generator | None = None) -> list[ViewPrediction | None]:
        """Estimates for a batch of views (``None`` where Procrustes is degenerate)."""
        sat, grd, camera_k = sat.to(self.device), grd.to(self.device), camera_k.to(self.device)
        scores, _ = self.model(self.dino(grd), self.dino(sat), camera_k)
        n_grd = scores.shape[2]
        flat = scores.flatten(1)
        out: list[ViewPrediction | None] = []
        for probs in flat.split(1):
            idx = torch.multinomial(probs, self.num_matches, generator=generator)
            sat_pts = self._sat_grid[torch.div(idx, n_grd, rounding_mode="trunc")]
            grd_pts = self._grd_grid[idx % n_grd]
            w = probs.gather(1, idx)
            rot, t, _ = self._procrustes(sat_pts, grd_pts, use_weights=True, use_mask=True, w=w)
            if t is None:
                out.append(None)
                continue
            rot = rot[0].cpu().numpy()
            out.append(ViewPrediction(t_px=(t[0, 0] / GRID_SIZE_M * SAT_SIZE).cpu().numpy(),
                                      yaw_rad=float(np.arctan2(rot[1, 0], rot[0, 0])),
                                      confidence=float(w.sum())))
        return out


def fuse(views: Sequence[ViewPrediction | None], view_rotation_deg: Sequence[float],
         tile_rotation_deg: float) -> tuple[np.ndarray, float] | None:
    """Confidence-weighted fusion of per-camera estimates (paper appendix).

    Every view's position is mapped back to the north-up tile and averaged,
    weighted by the view's confidence. All views estimate the same relative
    heading (heading - tile_rotation); like the paper's evaluation, these are
    averaged linearly in radians with the same weights (no wrap-around issue
    in ``imu_prior``, where they stay within a few degrees of zero). Returns
    ``((x, y) in the north-up tile, compass heading)``, or ``None`` if every
    view failed. With one camera this is that camera's estimate.
    """
    kept = [(fg2_to_north_up(v.t_px, rot), v.yaw_rad, v.confidence)
            for v, rot in zip(views, view_rotation_deg) if v is not None]
    if not kept:
        return None
    total = sum(c for _, _, c in kept) + 1e-9
    xy = sum(xy * c for xy, _, c in kept) / total
    yaw = sum(yaw * c for _, yaw, c in kept) / total
    heading = float(geometry.wrap_deg_360(tile_rotation_deg + math.degrees(yaw)))
    return xy, heading
