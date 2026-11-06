"""GOOSELoc samples in the input format of HC-Net (Wang et al., NeurIPS 2023).

HC-Net matches a quasi bird's-eye view (BEV), obtained from a ground panorama
with its spherical transform, against an aerial patch and regresses the
homography between the two. We build the same four-camera pseudo-panorama as
for CCVPE (front, right, back, left, each in its 90 degree azimuth sector;
320 x 640 px) and project it to a 512 x 512 BEV with HC-Net's own
``get_BEV_projection`` (FoV 170 degrees, their VIGOR setting). The DOP20 tile
is resized to 512 x 512 px (0.40 m/px) and stays north-up in both settings.

Orientation is set by rolling the panorama before the projection. Panorama
columns grow clockwise in azimuth, the BEV's up-axis shows the centre column,
and rolling by ``+k`` columns turns the BEV ``k / 640 * 360`` degrees
clockwise, which is the angle HC-Net's orientation output reports. With the
compass heading ``h``, the roll ``640 * (0.5 + (h - 45) / 360)`` north-aligns
the panorama, and

=============  ==========================================  ================
mode           panorama roll                               target ``ori_gt``
=============  ==========================================  ================
training       north-aligned + u, u ~ U(-180, 180)         u
``north_up``   constant 240 columns (no heading used)      -h
``imu_prior``  north-aligned according to the prior        prior offset
               heading ``h + imu_prior_offset_deg``
=============  ==========================================  ================

Rolls are whole columns (0.5625 degrees); targets are computed after rounding
and are exact. One model is trained per seed and evaluated under both
settings, so ``setting`` only matters outside training.

Each sample is a dict:

``bev`` (3, 512, 512)
    Projected pseudo-panorama, raw 0..255 floats (HC-Net normalises itself).
    Each camera is first resized to 512 x 1216 (PIL bicubic) and then to a
    320 x 160 strip (bilinear), as in the training pipeline.
``sat`` (3, 512, 512)
    North-up tile, bilinearly downsampled, raw 0..255 floats.
``gt_xy`` (2,)
    Vehicle position ``(x, y)`` in 512 px aerial coordinates (half the 1024 px tile coordinates).
``ori_gt`` ()
    Orientation target in degrees, see the table.
``roll``
    Applied panorama roll in columns (needed to recover the compass heading).
``scene_id``
"""

from __future__ import annotations

import sys
import types
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from gooseloc import GOOSELoc
from gooseloc.constants import GSD_M, SETTINGS, TILE_SIZE_PX
from gooseloc.data.geometry import wrap_deg_180, wrap_deg_360

HERE = Path(__file__).resolve().parent
UPSTREAM = HERE / "upstream"

CAMERA_SIZE = (512, 1216)      # first-stage camera resize (height, width), PIL bicubic
PANO_H, PANO_W = 320, 640      # pseudo-panorama; each camera fills a 320 x 160 strip
SAT_SIZE = 512                 # HC-Net image size (BEV and aerial)
SAT_SCALE = TILE_SIZE_PX / SAT_SIZE      # 2.0: 1024 px tile coordinates per aerial pixel
SAT_M_PER_PX = GSD_M * SAT_SCALE         # 0.40 m
BEV_FOV_DEG = 170.0            # spherical transform field of view, as upstream for VIGOR
TRAIN_ROLL_RANGE_DEG = 180.0   # training rolls cover the full circle (upstream VIGOR: 45)


def import_upstream() -> types.ModuleType:
    """Put ``upstream/`` on ``sys.path`` and return HC-Net's ``models.utils.utils``.

    Upstream imports scikit-image only for an image writer that is never
    called here; a stub stands in when it is not installed.
    """
    if not (UPSTREAM / "models" / "network.py").exists():
        raise FileNotFoundError(f"{UPSTREAM} not found; run {HERE / 'setup.sh'} first")
    if str(UPSTREAM) not in sys.path:
        sys.path.insert(0, str(UPSTREAM))
    try:
        import skimage.io  # noqa: F401
    except ImportError:
        def imsave(*args, **kwargs):
            raise ImportError("scikit-image is not installed")
        io = types.ModuleType("skimage.io")
        io.imsave = imsave
        sk = types.ModuleType("skimage")
        sk.io = io
        sys.modules.update({"skimage": sk, "skimage.io": io})
    from models.utils import utils
    return utils


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


def north_roll(heading_deg: float) -> float:
    """Fractional column roll that north-aligns the panorama of a vehicle with this compass heading."""
    return PANO_W * (0.5 + (heading_deg - 45.0) / 360.0)


def roll_and_target(heading_deg: float, offset_deg: float) -> tuple[int, float]:
    """Whole-column roll and its exact target for a panorama turned ``offset_deg`` cw of north-aligned."""
    k_north = north_roll(heading_deg)
    k = int(round(k_north + PANO_W * offset_deg / 360.0)) % PANO_W
    return k, float(wrap_deg_180((k - k_north) * 360.0 / PANO_W))


def compass_heading(roll, ori_deg) -> np.ndarray:
    """Compass heading from the applied roll and HC-Net's orientation output.

    Inverse of :func:`roll_and_target`: ``compass_heading(k, target) == heading``.
    """
    return wrap_deg_360((np.asarray(roll, dtype=np.float64) - north_roll(0.0)) * 360.0 / PANO_W
                        - np.asarray(ori_deg, dtype=np.float64))


class HCNetGOOSELoc(Dataset):
    """A GOOSELoc split as HC-Net samples (dicts, see the module docstring).

    ``training`` (default: ``split == "train"``) draws a random panorama roll
    per access from torch's global generator (seeded per DataLoader worker).
    """

    def __init__(
        self,
        root: str | Path | None = None,
        split: str | Iterable[str] = "test",
        *,
        setting: str = "north_up",
        training: bool | None = None,
        scene_ids: Iterable[str] | None = None,
    ) -> None:
        if setting not in SETTINGS:
            raise ValueError(f"unknown setting {setting!r}; expected one of {SETTINGS}")
        # The tile stays north-up in both settings; the prior enters through the panorama roll.
        self.base = GOOSELoc(root, split, setting="north_up", modalities=("aerial", "cameras"),
                             camera_size=CAMERA_SIZE, training=training, scene_ids=scene_ids)
        self.setting = setting
        self.training = self.base.training
        utils = import_upstream()
        self.grid = utils.get_BEV_projection(np.zeros((PANO_H, PANO_W, 3), dtype=np.uint8),
                                             SAT_SIZE, SAT_SIZE, Fov=BEV_FOV_DEG, dty=0, dy=0)

    def __len__(self) -> int:
        return len(self.base)

    @property
    def scene_ids(self) -> list[str]:
        return self.base.scene_ids

    def roll(self, idx: int, heading_deg: float) -> tuple[int, float]:
        """Panorama roll (columns) and orientation target of scene ``idx``."""
        h = float(heading_deg) % 360.0
        if self.training:
            offset = float(torch.empty(1).uniform_(-TRAIN_ROLL_RANGE_DEG, TRAIN_ROLL_RANGE_DEG))
        elif self.setting == "imu_prior":
            offset = float(self.base.scenes["imu_prior_offset_deg"].iat[idx])
        else:
            offset = -h  # constant roll: k_north(h) - h / 360 * 640 = 240
        return roll_and_target(h, offset)

    def labels(self, idx: int, sample: dict) -> dict:
        """``roll``, ``gt_xy`` and ``ori_gt`` of scene ``idx`` from its GOOSELoc sample (no images needed)."""
        k, ori_gt = self.roll(idx, sample["heading_deg"])
        gx, gy = float(sample["gt_xy"][0]), float(sample["gt_xy"][1])
        return {
            "gt_xy": torch.tensor([gx * SAT_SIZE / TILE_SIZE_PX, gy * SAT_SIZE / TILE_SIZE_PX],
                                  dtype=torch.float32),
            "ori_gt": torch.tensor(ori_gt, dtype=torch.float32),
            "roll": k,
        }

    def __getitem__(self, idx: int) -> dict:
        s = self.base[idx]
        sat = F.interpolate(_to_float(s["aerial"]).unsqueeze(0), size=(SAT_SIZE, SAT_SIZE),
                            mode="bilinear", align_corners=False)[0] * 255.0
        labels = self.labels(idx, s)
        pano = torch.roll(pseudo_panorama(s["cameras"]) * 255.0, shifts=labels["roll"], dims=2)
        bev = F.grid_sample(pano.unsqueeze(0), self.grid.unsqueeze(0), align_corners=True)[0]
        return {"bev": bev, "sat": sat, "scene_id": s["scene_id"]} | labels
