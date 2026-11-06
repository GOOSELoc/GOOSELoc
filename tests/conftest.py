"""Shared fixtures: a tiny synthetic GOOSELoc root built from the real scene table."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from gooseloc.constants import CAMERAS
from gooseloc.data.io import resolve_root
from gooseloc.data.metadata import load_scenes

N_PER_SPLIT = 3


@pytest.fixture(scope="session")
def scenes():
    return load_scenes()


@pytest.fixture(scope="session")
def mini_root(tmp_path_factory, scenes) -> Path:
    """Fake sensor files (small cameras, random LiDAR) for a few scenes per split.

    The aerial tile is black with a white 4x4 marker at the GT pixel, so tests
    can find the vehicle in rotated tiles.
    """
    root = tmp_path_factory.mktemp("gooseloc_root")
    rng = np.random.default_rng(0)
    sub = scenes.groupby("split", group_keys=False).head(N_PER_SPLIT)
    (root / "metadata").mkdir()
    sub.to_csv(root / "metadata" / "scenes.csv", index=False)
    for d in ("aerial", "lidar", *(f"cameras/{c}" for c in CAMERAS)):
        (root / d).mkdir(parents=True)
    for row in sub.itertuples():
        tile = np.zeros((1024, 1024, 3), np.uint8)
        tile[row.gt_px_y - 2:row.gt_px_y + 2, row.gt_px_x - 2:row.gt_px_x + 2] = 255
        Image.fromarray(tile).save(root / "aerial" / f"{row.scene_id}.png")
        for i, cam in enumerate(CAMERAS):
            Image.fromarray(np.full((100, 245, 3), 40 * i, np.uint8)).save(
                root / "cameras" / cam / f"{row.scene_id}.png")
        rng.normal(size=(5000, 4)).astype(np.float32).tofile(root / "lidar" / f"{row.scene_id}.bin")
    return root


@pytest.fixture(scope="session")
def data_root() -> Path:
    """The real dataset at ``$GOOSELOC_ROOT`` or ``data/GOOSELoc`` (tests marked ``data``)."""
    try:
        return resolve_root(None)
    except ValueError:
        pytest.skip("no dataset: GOOSELOC_ROOT not set and no data/GOOSELoc")
