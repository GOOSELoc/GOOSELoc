"""Framework-agnostic GOOSELoc dataset.

Returns plain numpy arrays, so it works with any training framework and in
the separate baseline environments. :mod:`gooseloc.data.torch_dataset` wraps
it for PyTorch.

Example::

    from gooseloc import GOOSELoc

    ds = GOOSELoc("data/GOOSELoc", split="train", setting="north_up")
    s = ds[0]
    s["aerial"]          # (1024, 1024, 3) uint8, the tile the model sees
    s["cameras"]         # (4, 1000, 2448, 3) uint8, front/right/back/left
    s["lidar"]           # (N, 4) float32, x/y/z/intensity in the vehicle frame
    s["gt_xy"]           # (2,) vehicle position (x, y) in pixels of s["aerial"]
    s["gt_heading_deg"]  # heading clockwise from the up-axis of s["aerial"]
"""

from __future__ import annotations

import zlib
from collections.abc import Callable, Iterable, Sequence
from pathlib import Path

import numpy as np
import pandas as pd

from gooseloc.constants import (
    CAMERAS,
    IMU_PRIOR_RANGE_DEG,
    MODALITIES,
    SETTINGS,
    TILE_SIZE_PX,
)
from gooseloc.data import geometry
from gooseloc.data.augment import lidar_augment
from gooseloc.data.io import (
    aerial_path,
    camera_path,
    lidar_path,
    read_image,
    read_lidar,
    resolve_root,
)
from gooseloc.data.metadata import load_scenes, select

META_COLUMNS = (
    "split", "date", "season", "terrain", "trajectory_id", "lat", "lon", "utm_e", "utm_n",
    "tile_utm_e0", "tile_utm_n0", "dop20_date", "test_buffered", "gnss_ins_offset_m",
)


class GOOSELoc:
    """One GOOSELoc split under one heading setting.

    Parameters
    ----------
    root:
        Dataset root (see :mod:`gooseloc.data.io`); defaults to ``$GOOSELOC_ROOT``, else ``data/GOOSELoc``.
    split:
        ``train``, ``dev``, ``test``, ``test_buffered``, ``all`` or a list.
    setting:
        ``north_up``: the aerial tile is north-aligned and the model must
        recover the full 360 degree heading. ``imu_prior``: the tile is rotated
        by a noisy heading prior (true heading + offset, offset uniform in
        +-``imu_prior_range_deg``), so the vehicle faces roughly up and only
        the residual has to be recovered. During training the offset is drawn
        anew on every access; otherwise the fixed per-scene offset from the
        scene table is used, which is the benchmark protocol.
    modalities:
        Subset of ``("aerial", "cameras", "lidar")`` to load.
    cameras:
        Subset (and order) of ``("front", "right", "back", "left")``.
    camera_size:
        Optional ``(height, width)`` to resize camera images to (bicubic).
    lidar_max_points:
        Randomly subsample sweeps to at most this many points. In training the
        subset changes per access, otherwise it is fixed per scene.
    lidar_aug:
        Apply :func:`gooseloc.data.augment.lidar_augment` (training only).
    training:
        Enables random prior offsets and augmentation. Defaults to
        ``split == "train"``.
    terrain, season, scene_ids:
        Optional filters (see :func:`gooseloc.data.metadata.select`).
    transform:
        Optional callable applied to the finished sample dict.
    """

    def __init__(
        self,
        root: str | Path | None = None,
        split: str | Iterable[str] = "train",
        *,
        setting: str = "north_up",
        modalities: Sequence[str] = MODALITIES,
        cameras: Sequence[str] = CAMERAS,
        camera_size: tuple[int, int] | None = None,
        lidar_max_points: int | None = None,
        lidar_aug: bool = False,
        training: bool | None = None,
        imu_prior_range_deg: float = IMU_PRIOR_RANGE_DEG,
        terrain: str | None = None,
        season: str | Iterable[str] | None = None,
        scene_ids: Iterable[str] | None = None,
        transform: Callable[[dict], dict] | None = None,
        seed: int = 0,
        scenes: pd.DataFrame | None = None,
    ) -> None:
        if setting not in SETTINGS:
            raise ValueError(f"unknown setting {setting!r}; expected one of {SETTINGS}")
        unknown = set(modalities) - set(MODALITIES)
        if unknown:
            raise ValueError(f"unknown modalities {sorted(unknown)}; expected {MODALITIES}")
        unknown = set(cameras) - set(CAMERAS)
        if unknown:
            raise ValueError(f"unknown cameras {sorted(unknown)}; expected {CAMERAS}")

        self.root = resolve_root(root)
        self.split = split
        self.setting = setting
        self.modalities = tuple(modalities)
        self.cameras = tuple(cameras)
        self.camera_size = tuple(camera_size) if camera_size is not None else None
        self.lidar_max_points = lidar_max_points
        self.training = (split == "train") if training is None else bool(training)
        self.lidar_aug = bool(lidar_aug)
        self.imu_prior_range_deg = float(imu_prior_range_deg)
        self.transform = transform
        self.seed = int(seed)
        self._rng = np.random.default_rng(self.seed)

        table = scenes if scenes is not None else load_scenes(self.root)
        self.scenes = select(table, split, terrain=terrain, season=season, scene_ids=scene_ids)
        if len(self.scenes) == 0:
            raise ValueError("no scenes match the requested split/filters")

    # -- bookkeeping ----------------------------------------------------------

    def __len__(self) -> int:
        return len(self.scenes)

    def __repr__(self) -> str:
        return (f"GOOSELoc(split={self.split!r}, setting={self.setting!r}, n={len(self)}, "
                f"modalities={self.modalities}, training={self.training})")

    @property
    def scene_ids(self) -> list[str]:
        return self.scenes["scene_id"].tolist()

    def reseed(self, seed: int) -> None:
        """Reset the random state (called per DataLoader worker)."""
        self._rng = np.random.default_rng(seed)

    def _scene_rng(self, scene_id: str) -> np.random.Generator:
        """Generator fixed per scene, for deterministic evaluation."""
        return np.random.default_rng((zlib.crc32(scene_id.encode()) + 7919 * self.seed) & 0xFFFFFFFF)

    # -- protocol -------------------------------------------------------------

    def tile_rotation_deg(self, row) -> float:
        """CCW rotation applied to the north-up tile for this scene and access."""
        if self.setting == "north_up":
            return 0.0
        if self.training:
            offset = float(self._rng.uniform(-self.imu_prior_range_deg, self.imu_prior_range_deg))
        else:
            # The stored offsets were drawn uniformly in +-IMU_PRIOR_RANGE_DEG;
            # scaling them keeps the per-scene draw for other prior widths.
            offset = float(row.imu_prior_offset_deg)
            if self.imu_prior_range_deg != IMU_PRIOR_RANGE_DEG:
                offset *= self.imu_prior_range_deg / IMU_PRIOR_RANGE_DEG
        return float(row.heading_deg) + offset

    # -- loading --------------------------------------------------------------

    def _load_lidar(self, scene_id: str) -> np.ndarray:
        pts = read_lidar(lidar_path(self.root, scene_id))
        rng = self._rng if self.training else self._scene_rng(scene_id)
        if self.lidar_max_points is not None and pts.shape[0] > self.lidar_max_points:
            pts = pts[rng.choice(pts.shape[0], self.lidar_max_points, replace=False)]
        if self.training and self.lidar_aug:
            pts = lidar_augment(pts, self._rng)
        return np.ascontiguousarray(pts, dtype=np.float32)

    def __getitem__(self, idx: int) -> dict:
        row = self.scenes.iloc[idx]
        sid = row.scene_id
        rotation = self.tile_rotation_deg(row)
        gt_nu = np.array([row.gt_px_x, row.gt_px_y], dtype=np.float64)
        gt_xy = geometry.rotate_xy(gt_nu, rotation)

        sample: dict = {
            "scene_id": sid,
            "setting": self.setting,
            "gt_xy": gt_xy.astype(np.float32),
            "gt_heading_deg": np.float32(self._relative_heading(row.heading_deg, rotation)),
            "tile_rotation_deg": np.float32(rotation),
            "gt_xy_north_up": gt_nu.astype(np.float32),
            "heading_deg": np.float32(row.heading_deg),
            "meta": {k: row[k].item() if isinstance(row[k], np.generic) else row[k] for k in META_COLUMNS},
        }
        if "aerial" in self.modalities:
            tile = read_image(aerial_path(self.root, sid))
            if tile.size != (TILE_SIZE_PX, TILE_SIZE_PX):
                raise RuntimeError(f"aerial tile of {sid} has size {tile.size}")
            sample["aerial"] = np.array(geometry.rotate_tile(tile, rotation))
        if "cameras" in self.modalities:
            sample["cameras"] = np.stack([
                np.asarray(read_image(camera_path(self.root, sid, cam), self.camera_size))
                for cam in self.cameras
            ])
        if "lidar" in self.modalities:
            sample["lidar"] = self._load_lidar(sid)
        if self.transform is not None:
            sample = self.transform(sample)
        return sample

    def _relative_heading(self, heading_deg: float, rotation: float) -> float:
        rel = float(heading_deg) - rotation
        if self.setting == "north_up":
            return float(geometry.wrap_deg_360(rel))
        return float(geometry.wrap_deg_180(rel))
