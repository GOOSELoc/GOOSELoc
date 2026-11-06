"""GOOSELoc samples in the input format of Shi et al. (Boosting3DoFAccuracy).

The upstream KITTI-3DoF model consumes ``(sat_map, camera_k, grd_img,
gt_shift_u, gt_shift_v, gt_heading)``. :class:`ShiGOOSELoc` builds exactly
these tensors on top of :class:`gooseloc.GOOSELoc`, for the IMU-prior setting
(the method needs a bounded heading prior):

``sat_map`` (3, 512, 512), float in [0, 1]
    The whole north-up 1024 px tile, rotated counter-clockwise about its
    centre by ``tile_rotation_deg = (heading - 90) + prior_offset`` (bicubic,
    black fill) and then downsampled to 512 px, i.e. 0.40 m/px. Shi's
    ground-plane geometry has the camera looking along image *right*, hence
    the extra -90 degrees compared to the GOOSELoc IMU-prior tile, in which
    the vehicle faces up.
``camera_k`` (3, 3)
    Front-camera intrinsics for the 1024 x 256 input.
``grd_img`` (3, 256, 1024), float in [0, 1]
    The 2448 x 1000 front image resized non-uniformly to 1024 x 256, the
    input shape of the upstream model (upstream resizes KITTI the same way).
``gt_shift_u``, ``gt_shift_v`` (1,)
    Vehicle offset from the tile centre in metres, right- and up-positive,
    divided by ``SHIFT_RANGE_M``. Measured in the *aligned* tile (rotated by
    ``heading - 90`` only); the fed tile adds the prior offset on top, which
    is how upstream defines its labels. Values reach about 1.41 for GT pixels
    near the tile corners.
``gt_heading`` (1,)
    The prior offset (the residual rotation the model has to recover)
    divided by ``ROTATION_RANGE_DEG``.

Predictions are mapped back to the official north-up format with
:func:`to_north_up`.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from importlib import resources
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms.functional as TF
import yaml
from PIL import Image
from torch.utils.data import Dataset

from gooseloc import GOOSELoc
from gooseloc.constants import GSD_M, IMU_PRIOR_RANGE_DEG, TILE_SIZE_PX
from gooseloc.data import geometry

SAT_SIZE = 512                                  # upstream satellite input side length
SAT_M_PER_PX = GSD_M * TILE_SIZE_PX / SAT_SIZE  # 0.40: the full tile, downsampled
GRD_H, GRD_W = 256, 1024                        # upstream ground input shape
SHIFT_RANGE_M = 51.2      # label normalisation; the correlation searches +-1.5x (76.8 m)
ROTATION_RANGE_DEG = IMU_PRIOR_RANGE_DEG        # label normalisation of the residual
CAMERA_HEIGHT_M = 2.0     # front camera above ground (estimate, see calibration.yaml)
CAMERA = "front"
ALIGN_DEG = -90.0         # tile rotation on top of the heading: forward = image right

_TILE_CENTRE = TILE_SIZE_PX / 2


def camera_k(camera: str = CAMERA) -> np.ndarray:
    """3x3 intrinsics of the 1024 x 256 ground input.

    Native sensor intrinsics from ``calibration.yaml``, shifted by the
    vertical crop of the released 2448 x 1000 images and scaled per axis to
    1024 x 256.
    """
    path = resources.files("gooseloc") / "resources" / "calibration.yaml"
    calib = yaml.safe_load(path.read_text())
    p = calib["cameras"][camera]
    width, height = calib["image"]["released_size_wh"]
    crop_top = calib["image"]["released_crop_top_px"]
    sx, sy = GRD_W / width, GRD_H / height
    return np.array([
        [p["fx"] * sx, 0.0, p["cx"] * sx],
        [0.0, p["fy"] * sy, (p["cy"] - crop_top) * sy],
        [0.0, 0.0, 1.0],
    ], dtype=np.float32)


def tile_rotation_deg(heading_deg: float, prior_offset_deg: float) -> float:
    """CCW rotation applied to the north-up tile: ``(heading - 90) + offset``."""
    # Keep this evaluation order: it is the one the released models were run with.
    return (float(heading_deg) + ALIGN_DEG) + float(prior_offset_deg)


def encode_labels(gt_xy_north_up, heading_deg: float,
                  prior_offset_deg: float) -> tuple[float, float, float]:
    """Ground truth to the upstream labels ``(gt_shift_u, gt_shift_v, gt_heading)``.

    ``gt_xy_north_up`` is the vehicle position in pixels of the north-up tile.
    """
    dx = float(gt_xy_north_up[0]) - _TILE_CENTRE
    dy = float(gt_xy_north_up[1]) - _TILE_CENTRE
    # CCW image rotation about the centre (as geometry.rotate_xy), by the
    # aligned rotation only; scalar math keeps the labels bit-reproducible.
    t = math.radians(float(heading_deg) + ALIGN_DEG)
    ax = dx * math.cos(t) + dy * math.sin(t)
    ay = -dx * math.sin(t) + dy * math.cos(t)
    u = ax * GSD_M / SHIFT_RANGE_M        # right-positive
    v = -ay * GSD_M / SHIFT_RANGE_M       # up-positive (image y points down)
    return u, v, float(prior_offset_deg) / ROTATION_RANGE_DEG


def to_north_up(pred_u_m, pred_v_m, pred_theta_deg,
                tile_rotation: float | np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Model output to the official prediction format.

    ``pred_u_m``/``pred_v_m`` are the translation estimate in metres right/up
    of the centre of the tile that was fed (see :func:`model.predict_batch`),
    ``pred_theta_deg`` the estimated residual rotation and ``tile_rotation``
    the CCW rotation the tile was given. Returns ``(xy, heading_deg)``: pixels
    of the north-up 1024 px tile and the compass heading.
    """
    u = np.asarray(pred_u_m, dtype=np.float64)
    v = np.asarray(pred_v_m, dtype=np.float64)
    xy_tile = np.stack([_TILE_CENTRE + u / GSD_M, _TILE_CENTRE - v / GSD_M], axis=-1)
    # In the fed tile the vehicle faces image right, rotated CCW by the
    # residual theta, i.e. it points 90 - theta degrees clockwise of the up-axis.
    rel_heading = 90.0 - np.asarray(pred_theta_deg, dtype=np.float64)
    return geometry.to_north_up(xy_tile, rel_heading, tile_rotation)


class ShiGOOSELoc(Dataset):
    """GOOSELoc split as Boosting3DoF input tensors (IMU-prior setting).

    Each sample is a dict with the upstream tensors (see the module
    docstring) plus ``scene_id`` and ``tile_rotation_deg`` (float64), which
    :func:`to_north_up` needs to map predictions back.

    Parameters
    ----------
    root, split, terrain, season, scene_ids:
        As for :class:`gooseloc.GOOSELoc`.
    training:
        Draw a fresh prior offset uniformly in +-``ROTATION_RANGE_DEG`` on
        every access instead of the fixed per-scene benchmark offset.
        Defaults to ``split == "train"``.
    seed:
        Seed of the training offsets.
    """

    def __init__(
        self,
        root: str | Path | None = None,
        split: str | Iterable[str] = "test",
        *,
        training: bool | None = None,
        seed: int = 0,
        terrain: str | None = None,
        season: str | Iterable[str] | None = None,
        scene_ids: Iterable[str] | None = None,
    ) -> None:
        # North-up tiles: the rotation convention differs from the GOOSELoc
        # IMU-prior tiles, so the tile is rotated here, once, from the original.
        self.base = GOOSELoc(
            root, split, setting="north_up", modalities=("aerial", "cameras"),
            cameras=(CAMERA,), camera_size=(GRD_H, GRD_W), training=False,
            terrain=terrain, season=season, scene_ids=scene_ids,
        )
        self.training = (split == "train") if training is None else bool(training)
        self._rng = np.random.default_rng(seed)
        self.camera_k = torch.from_numpy(camera_k(CAMERA))

    def __len__(self) -> int:
        return len(self.base)

    @property
    def scenes(self):
        return self.base.scenes

    def prior_offset_deg(self, row) -> float:
        """Heading prior error of this access (benchmark offset unless training)."""
        if self.training:
            return float(self._rng.uniform(-ROTATION_RANGE_DEG, ROTATION_RANGE_DEG))
        return float(row.imu_prior_offset_deg)

    def __getitem__(self, idx: int) -> dict:
        row = self.base.scenes.iloc[idx]
        sample = self.base[idx]
        offset = self.prior_offset_deg(row)
        rotation = tile_rotation_deg(row.heading_deg, offset)

        tile = Image.fromarray(sample["aerial"]).rotate(rotation, resample=Image.BICUBIC)
        tile = tile.resize((SAT_SIZE, SAT_SIZE), Image.BICUBIC)
        u, v, theta = encode_labels((row.gt_px_x, row.gt_px_y), row.heading_deg, offset)
        return {
            "sat_map": TF.to_tensor(tile),
            "camera_k": self.camera_k,
            "grd_img": TF.to_tensor(sample["cameras"][0]),
            "gt_shift_u": torch.tensor([u], dtype=torch.float32),
            "gt_shift_v": torch.tensor([v], dtype=torch.float32),
            "gt_heading": torch.tensor([theta], dtype=torch.float32),
            "tile_rotation_deg": torch.tensor(rotation, dtype=torch.float64),
            "scene_id": str(row.scene_id),
        }
