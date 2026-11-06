"""Augmentations. Only the LiDAR augmentation of the paper is provided."""

from __future__ import annotations

import math

import numpy as np


def lidar_augment(
    points: np.ndarray,
    rng: np.random.Generator,
    *,
    max_dropout: float = 0.20,
    jitter_sigma_m: float = 0.05,
    scale_range: float = 0.03,
    yaw_range_deg: float = 2.0,
) -> np.ndarray:
    """LiDAR augmentation used to train our baseline.

    Random dropout of up to 20 % of the points, per-point Gaussian xyz jitter
    (sigma 0.05 m), global isotropic scaling of +-3 % and a random yaw rotation
    of +-2 degrees. Operates on ``(N, C)`` arrays whose first three columns are
    xyz; further columns (intensity) are carried along unchanged.
    """
    pts = np.array(points, dtype=np.float32, copy=True)
    drop_p = float(rng.uniform(0.0, max_dropout))
    if drop_p > 0.0:
        pts = pts[rng.random(pts.shape[0]) > drop_p]
    if pts.shape[0] > 0 and jitter_sigma_m > 0:
        pts[:, :3] += rng.normal(0.0, jitter_sigma_m, size=(pts.shape[0], 3)).astype(np.float32)
    pts[:, :3] *= float(rng.uniform(1.0 - scale_range, 1.0 + scale_range))
    yaw = math.radians(float(rng.uniform(-yaw_range_deg, yaw_range_deg)))
    c, s = math.cos(yaw), math.sin(yaw)
    x, y = pts[:, 0].copy(), pts[:, 1].copy()
    pts[:, 0] = c * x - s * y
    pts[:, 1] = s * x + c * y
    return np.ascontiguousarray(pts, dtype=np.float32)
