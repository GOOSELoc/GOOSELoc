# SPDX-License-Identifier: GPL-3.0-or-later
# Part of the GOOSELoc FG² baseline (see LICENSE in this directory).
"""GOOSELoc scenes as inputs of upstream FG²'s KITTI model, and back.

FG² (Xia et al., CVPR 2025) localizes one forward-facing ground image inside
an aerial image. Upstream's KITTI pipeline consumes batches of
``(sat, grd, camera_k, gt_loc, R)``; this module produces exactly those
tensors from :class:`gooseloc.GOOSELoc` scenes, so the upstream model, losses
and Procrustes solver are used unchanged.

Conventions (identical to the runs reported in the paper):

Camera views
    Camera ``c`` has azimuth ``delta_c`` (front/right/back/left = 0/90/180/270
    degrees clockwise from the vehicle's forward axis). Its *view* of a scene
    is the aerial tile rotated counter-clockwise by
    ``view_rotation = tile_rotation + delta_c``, where ``tile_rotation`` is the
    GOOSELoc protocol rotation (0 in ``north_up``; heading + prior offset in
    ``imu_prior``). The extra ``delta_c`` turns camera ``c`` into a virtual
    forward camera: it looks roughly "up" in its view.

Aerial input
    The full 1024 x 1024 north-up tile is rotated once by ``view_rotation``
    about its centre (bicubic, black corners) and resized (bicubic) to
    504 x 504, i.e. ~0.41 m/px.

Ground input
    The released 2448 x 1000 image is cropped to rows ``[138, 861)``
    (2448 x 723, the aspect ratio of FG²-KITTI) and resized (bicubic) to
    1232 x 364. Pixels stay square.

Intrinsics
    Native pinhole intrinsics of each camera (``gooseloc/resources/calibration.yaml``),
    shifted by the release crop (sensor rows ``[524, 1524)``) and the crop
    above, then scaled by 1232 / 2448.

Labels (FG²-KITTI convention)
    ``gt_loc = [[y_off, x_off]]`` with ``off = 252 - p``, where ``p`` is the
    vehicle position ``(x, y)`` in pixels of the 504 x 504 view.
    ``R = [[cos a, -sin a], [sin a, cos a]]`` with
    ``a = heading + delta_c - view_rotation = heading - tile_rotation``, the
    camera's heading relative to the view's up-axis, clockwise positive (the
    same for all cameras, and equal to GOOSELoc's ``gt_heading_deg``).
    FG² predicts ``(R, t)`` with ``t`` in metres of a ``GRID_SIZE_M`` grid laid
    over the 504 view; ``t / GRID_SIZE_M * 504`` has the layout of ``gt_loc``.

Training draws one random camera per example. In the ``imu_prior`` setting the
random prior offset (uniform in +-10 degrees) perturbs the view orientation;
in ``north_up`` the same +-10 degree perturbation is applied explicitly
(FG²'s KITTI recipe, as for the paper's runs). Evaluation is deterministic and
uses the benchmark's fixed per-scene prior offsets.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import NamedTuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from gooseloc import GOOSELoc
from gooseloc.constants import CAMERA_AZIMUTH_DEG, CAMERA_HW, CAMERAS, TILE_SIZE_PX
from gooseloc.data import geometry
from gooseloc.data.io import aerial_path, camera_path, read_image

SAT_SIZE = 504                     # aerial input side length (px)
GROUND_HW = (364, 1232)            # ground input (height, width), FG²-KITTI
GROUND_CROP_ROWS = (138, 861)      # rows of the released image fed to FG² (723 rows)
RELEASE_CROP_TOP = 524             # released images are sensor rows [524, 1524)
# Metric extent of FG²'s BEV grids, in metres. The paper's runs used 200 (the
# tile actually spans 204.8 m); keep it for the released checkpoints.
GRID_SIZE_M = 200.0

# Native pinhole intrinsics (fx, fy, cx, cy) of the 2448 x 2048 sensors, as in
# gooseloc/resources/calibration.yaml. Fixed here because the released
# checkpoints were trained with exactly these values.
NATIVE_INTRINSICS: dict[str, tuple[float, float, float, float]] = {
    "front": (1127.30058, 1126.67571, 1222.59179, 994.31072),
    "right": (1127.21924, 1126.03821, 1257.15040, 966.80746),
    "back": (1126.08407, 1125.28735, 1262.14856, 964.26557),
    "left": (1130.75573, 1129.50384, 1271.58781, 981.54998),
}


class View(NamedTuple):
    """One camera view of a scene, in FG²-KITTI's batch layout."""

    sat: torch.Tensor        # (3, 504, 504) float in [0, 1]
    grd: torch.Tensor        # (3, 364, 1232) float in [0, 1]
    camera_k: torch.Tensor   # (3, 3) float32
    gt_loc: torch.Tensor     # (1, 2) float32, [y_off, x_off] in 504-view pixels
    rot_gt: torch.Tensor     # (2, 2) float32


# -- per-view building blocks ---------------------------------------------------

def _to_tensor(image: Image.Image) -> torch.Tensor:
    """RGB image to a (3, H, W) float tensor in [0, 1]."""
    return torch.from_numpy(np.array(image, dtype=np.uint8)).permute(2, 0, 1).contiguous().float().div(255.0)


def camera_intrinsics(camera: str) -> torch.Tensor:
    """3x3 intrinsics of ``camera`` for the 1232 x 364 ground input."""
    fx, fy, cx, cy = NATIVE_INTRINSICS[camera]
    s = GROUND_HW[1] / CAMERA_HW[1]
    cy_input = (cy - RELEASE_CROP_TOP - GROUND_CROP_ROWS[0]) * s
    k = np.array([[fx * s, 0.0, cx * s], [0.0, fy * s, cy_input], [0.0, 0.0, 1.0]], dtype=np.float32)
    return torch.from_numpy(k)


def ground_input(image: Image.Image) -> torch.Tensor:
    """Released camera image to FG²'s (3, 364, 1232) ground input."""
    if image.size != (CAMERA_HW[1], CAMERA_HW[0]):
        raise ValueError(f"expected a {CAMERA_HW[1]}x{CAMERA_HW[0]} camera image, got {image.size}")
    top, bottom = GROUND_CROP_ROWS
    image = image.crop((0, top, image.width, bottom))
    return _to_tensor(image.resize((GROUND_HW[1], GROUND_HW[0]), Image.BICUBIC))


def aerial_input(tile: Image.Image, view_rotation_deg: float) -> torch.Tensor:
    """North-up tile to FG²'s (3, 504, 504) aerial input of one view."""
    if tile.size != (TILE_SIZE_PX, TILE_SIZE_PX):
        raise ValueError(f"expected a {TILE_SIZE_PX}x{TILE_SIZE_PX} aerial tile, got {tile.size}")
    view = geometry.rotate_tile(tile, view_rotation_deg)
    return _to_tensor(view.resize((SAT_SIZE, SAT_SIZE), Image.BICUBIC))


def fg2_labels(gt_xy_north_up, heading_deg: float, camera: str,
               view_rotation_deg: float) -> tuple[torch.Tensor, torch.Tensor]:
    """Ground truth of one view as FG²'s ``(gt_loc, R)``."""
    p = geometry.rotate_xy(np.asarray(gt_xy_north_up, dtype=np.float64), view_rotation_deg)
    off = SAT_SIZE / 2.0 - p * (SAT_SIZE / TILE_SIZE_PX)
    gt_loc = torch.tensor([[off[1], off[0]]], dtype=torch.float32)
    a = math.radians((float(heading_deg) + CAMERA_AZIMUTH_DEG[camera]) - view_rotation_deg)
    c, s = math.cos(a), math.sin(a)
    return gt_loc, torch.tensor([[c, -s], [s, c]], dtype=torch.float32)


def fg2_to_north_up(t_px, view_rotation_deg: float) -> np.ndarray:
    """FG² translation of one view to the vehicle position in the north-up tile.

    ``t_px`` is the predicted ``t`` converted to 504-view pixels
    (``t / GRID_SIZE_M * 504``), laid out like ``gt_loc``: ``[y_off, x_off]``.
    Returns ``(x, y)`` in pixels of the north-up 1024 tile.
    """
    t_px = np.asarray(t_px, dtype=np.float64)
    xy_view = (SAT_SIZE / 2.0 - t_px[::-1]) * (TILE_SIZE_PX / SAT_SIZE)
    xy_nu, _ = geometry.to_north_up(xy_view, None, view_rotation_deg)
    return xy_nu


def make_view(tile: Image.Image, image: Image.Image, camera: str, tile_rotation_deg: float,
              gt_xy_north_up, heading_deg: float) -> tuple[View, float]:
    """Build one camera view. Returns the view and its ``view_rotation_deg``."""
    view_rotation = tile_rotation_deg + CAMERA_AZIMUTH_DEG[camera]
    gt_loc, rot_gt = fg2_labels(gt_xy_north_up, heading_deg, camera, view_rotation)
    view = View(aerial_input(tile, view_rotation), ground_input(image),
                camera_intrinsics(camera), gt_loc, rot_gt)
    return view, view_rotation


# -- datasets -------------------------------------------------------------------

class _FG2Scenes(Dataset):
    """Scene access shared by the training and evaluation datasets."""

    def __init__(self, root, split, *, setting: str, cameras: Sequence[str], training: bool,
                 seed: int, **kwargs) -> None:
        self.base = GOOSELoc(root, split, setting=setting, modalities=(), cameras=cameras,
                             training=training, seed=seed, **kwargs)
        self.setting = setting
        self.cameras = tuple(cameras)

    def __len__(self) -> int:
        return len(self.base)

    @property
    def scene_ids(self) -> list[str]:
        return self.base.scene_ids

    def _scene(self, idx: int):
        row = self.base.scenes.iloc[idx]
        gt_nu = np.array([row.gt_px_x, row.gt_px_y], dtype=np.float64)
        tile = read_image(aerial_path(self.base.root, row.scene_id))
        return row, gt_nu, tile

    def _camera(self, scene_id: str, camera: str) -> Image.Image:
        return read_image(camera_path(self.base.root, scene_id, camera))


class FG2TrainDataset(_FG2Scenes):
    """Training examples: one random camera per access, as FG²-KITTI 5-tuples.

    Returns ``(sat, grd, camera_k, gt_loc, R)`` (see :class:`View`). Arguments
    other than ``north_up_jitter_deg`` are passed to :class:`gooseloc.GOOSELoc`.
    Compatible with :func:`gooseloc.data.torch_dataset.worker_init_fn`.
    """

    def __init__(self, root=None, split="train", *, setting: str = "north_up",
                 cameras: Sequence[str] = CAMERAS, north_up_jitter_deg: float = 10.0,
                 seed: int = 0, **kwargs) -> None:
        super().__init__(root, split, setting=setting, cameras=cameras, training=True,
                         seed=seed, **kwargs)
        self.north_up_jitter_deg = float(north_up_jitter_deg)
        self.seed = int(seed)
        self._rng = np.random.default_rng([self.seed, 1])

    def reseed(self, seed: int) -> None:
        """Reset the random state (called per DataLoader worker)."""
        self.base.reseed(seed)
        self._rng = np.random.default_rng([seed, 1])

    def __getitem__(self, idx: int) -> View:
        row, gt_nu, tile = self._scene(idx)
        rotation = self.base.tile_rotation_deg(row)        # imu_prior: heading + random offset
        if self.setting == "north_up" and self.north_up_jitter_deg > 0:
            rotation += float(self._rng.uniform(-self.north_up_jitter_deg, self.north_up_jitter_deg))
        camera = self.cameras[int(self._rng.integers(len(self.cameras)))]
        view, _ = make_view(tile, self._camera(row.scene_id, camera), camera, rotation,
                            gt_nu, float(row.heading_deg))
        return view


class FG2EvalDataset(_FG2Scenes):
    """Evaluation scenes with all requested cameras, deterministic.

    Each item is a dict with ``scene_id``, ``cameras``, ``tile_rotation_deg``
    (protocol rotation of the scene, float), ``view_rotation_deg`` (C,) float64,
    the stacked views ``sat`` (C, 3, 504, 504), ``grd`` (C, 3, 364, 1232),
    ``camera_k`` (C, 3, 3), ``gt_loc`` (C, 1, 2), ``rot_gt`` (C, 2, 2), and the
    ground truth ``gt_xy_north_up`` (2,) and ``heading_deg``. Arguments are
    passed to :class:`gooseloc.GOOSELoc`; use ``imu_prior_range_deg=0`` for an
    exact heading prior (how the paper's runs selected checkpoints on dev).
    """

    def __init__(self, root=None, split="test", *, setting: str = "north_up",
                 cameras: Sequence[str] = CAMERAS, **kwargs) -> None:
        super().__init__(root, split, setting=setting, cameras=cameras, training=False,
                         seed=0, **kwargs)

    def __getitem__(self, idx: int) -> dict:
        row, gt_nu, tile = self._scene(idx)
        rotation = self.base.tile_rotation_deg(row)
        views, view_rotations = [], []
        for camera in self.cameras:
            view, view_rotation = make_view(tile, self._camera(row.scene_id, camera), camera,
                                            rotation, gt_nu, float(row.heading_deg))
            views.append(view)
            view_rotations.append(view_rotation)
        out = {name: torch.stack(tensors) for name, tensors in zip(View._fields, zip(*views))}
        out.update(
            scene_id=row.scene_id,
            cameras=self.cameras,
            tile_rotation_deg=float(rotation),
            view_rotation_deg=torch.tensor(view_rotations, dtype=torch.float64),
            gt_xy_north_up=torch.from_numpy(gt_nu),
            heading_deg=float(row.heading_deg),
        )
        return out
