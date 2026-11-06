# SPDX-License-Identifier: GPL-3.0-only
"""GOOSELoc samples in the input format of CCVPE (Xia et al., T-PAMI 2023).

CCVPE consumes one 360 degree ground panorama and a north-up aerial patch. We
place each surround camera in its 90 degree azimuth sector and concatenate
them in the order front, right, back, left into one equirectangular-style
pseudo-panorama (front at column 0, azimuth growing clockwise with the column).
The DOP20 tile is resized to 512 x 512 px (0.40 m/px) and kept north-up; no
heading prior is used (``north_up`` setting only).

:class:`CCVPEGOOSELoc` returns, per scene, exactly the tensors of CCVPE's VIGOR
pipeline (``datasets.VIGORDataset``) that the paper's models were trained and
evaluated on:

``grd`` (3, 320, 640)
    Pseudo-panorama, ImageNet-normalised. Each camera is first resized to
    512 x 1216 (PIL bicubic) and then to a 320 x 160 strip (bilinear); the
    two-stage resize follows the training pipeline.
``sat`` (3, 512, 512)
    The north-up 1024 px tile, bilinearly downsampled, ImageNet-normalised.
``gt`` (1, 512, 512)
    Gaussian (sigma 4 px) centred on the vehicle at ``0.5 * (x, y)``, where
    ``(x, y)`` is the ground-truth position in 1024 px tile coordinates.
``gt_with_ori`` (20, 512, 512)
    ``gt`` split over the two nearest of 20 orientation bins of 18 degrees
    (CCVPE convention); all zeros outside training, as upstream.
``orientation`` (2, 512, 512)
    Constant ``(cos, sin)`` of the compass heading.
``angle``
    Compass heading in degrees, ``[0, 360)``.

The upstream tuple order is ``(grd, sat, gt, gt_with_ori, orientation, city,
angle)``; ``city`` is replaced by ``scene_id``.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from gooseloc import GOOSELoc
from gooseloc.constants import GSD_M, TILE_SIZE_PX

CAMERA_SIZE = (512, 1216)      # first-stage camera resize (height, width), PIL bicubic
PANO_H, PANO_W = 320, 640      # pseudo-panorama; each camera fills a 320 x 160 strip
SAT_SIZE = 512                 # CCVPE aerial input and heatmap side length
SAT_SCALE = TILE_SIZE_PX / SAT_SIZE      # 2.0: 1024 px tile coordinates per heatmap pixel
SAT_M_PER_PX = GSD_M * SAT_SCALE         # 0.40 m per heatmap pixel
SIGMA_PX = 4.0                 # Gaussian location target, as upstream
N_ORI_BINS = 20                # upstream orientation bins of 18 degrees

_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def _to_float(img: np.ndarray) -> torch.Tensor:
    """``(..., H, W, 3)`` uint8 to contiguous ``(..., 3, H, W)`` float32 in ``[0, 1]``.

    Contiguous (not channels-last) memory matters: ``F.interpolate`` picks a
    different CPU kernel for channels-last input, which changes the rounding.
    """
    t = torch.from_numpy(np.ascontiguousarray(img))
    return t.movedim(-1, -3).contiguous().to(torch.float32).div(255.0)


def pseudo_panorama(cameras: np.ndarray) -> torch.Tensor:
    """Front/right/back/left images ``(4, H, W, 3)`` uint8 to a ``(3, 320, 640)`` panorama in ``[0, 1]``.

    Every camera covers its 90 degree azimuth sector: column ``c`` looks
    ``c / 640 * 360 - 45`` degrees clockwise of the vehicle's forward axis.
    """
    cams = _to_float(cameras)
    strips = [F.interpolate(cams[i].unsqueeze(0), size=(PANO_H, PANO_W // 4),
                            mode="bilinear", align_corners=False)[0] for i in range(cams.shape[0])]
    return torch.cat(strips, dim=2)


def heatmap_to_tile_xy(row, col) -> tuple[np.ndarray, np.ndarray]:
    """Heatmap pixel indices to ``(x, y)`` in 1024 px north-up tile coordinates.

    Inverse of the label convention above (heatmap index = 0.5 * tile coordinate).
    """
    return np.asarray(col, dtype=np.float64) * SAT_SCALE, np.asarray(row, dtype=np.float64) * SAT_SCALE


class CCVPEGOOSELoc(Dataset):
    """A GOOSELoc split as CCVPE samples (dicts, see the module docstring).

    ``training`` (default: ``split == "train"``) only enables the
    ``gt_with_ori`` target; there is no data augmentation.
    """

    def __init__(
        self,
        root: str | Path | None = None,
        split: str | Iterable[str] = "test",
        *,
        training: bool | None = None,
        scene_ids: Iterable[str] | None = None,
    ) -> None:
        self.base = GOOSELoc(root, split, setting="north_up", modalities=("aerial", "cameras"),
                             camera_size=CAMERA_SIZE, training=training, scene_ids=scene_ids)
        self.training = self.base.training
        self._yy, self._xx = torch.meshgrid(torch.arange(float(SAT_SIZE)), torch.arange(float(SAT_SIZE)),
                                            indexing="ij")

    def __len__(self) -> int:
        return len(self.base)

    @property
    def scene_ids(self) -> list[str]:
        return self.base.scene_ids

    def __getitem__(self, idx: int) -> dict:
        s = self.base[idx]
        sat = F.interpolate(_to_float(s["aerial"]).unsqueeze(0), size=(SAT_SIZE, SAT_SIZE),
                            mode="bilinear", align_corners=False)[0]
        sample = {
            "grd": (pseudo_panorama(s["cameras"]) - _MEAN) / _STD,
            "sat": (sat - _MEAN) / _STD,
            "scene_id": s["scene_id"],
        }
        return sample | self.labels(s["gt_xy"], s["heading_deg"])

    def labels(self, gt_xy, heading_deg) -> dict:
        """Location and orientation targets from the 1024 px position and the compass heading."""
        angle = float(heading_deg) % 360.0
        px, py = float(gt_xy[0]) * 0.5, float(gt_xy[1]) * 0.5
        gauss = torch.exp(-((self._yy - py) ** 2 + (self._xx - px) ** 2) / (2.0 * SIGMA_PX**2))
        orientation = torch.empty(2, SAT_SIZE, SAT_SIZE)
        orientation[0].fill_(math.cos(math.radians(angle)))
        orientation[1].fill_(math.sin(math.radians(angle)))
        gt_with_ori = torch.zeros(N_ORI_BINS, SAT_SIZE, SAT_SIZE)
        if self.training:
            index = int(angle // 18)
            ratio = (angle % 18) / 18.0
            if index == 0:
                gt_with_ori[0] = gauss * (1 - ratio)
                gt_with_ori[19] = gauss * ratio
            else:
                gt_with_ori[20 - index] = gauss * (1 - ratio)
                gt_with_ori[20 - index - 1] = gauss * ratio
        return {"gt": gauss.unsqueeze(0), "gt_with_ori": gt_with_ori,
                "orientation": orientation, "angle": angle}
