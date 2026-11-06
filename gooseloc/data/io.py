"""File layout of a GOOSELoc download and readers for each modality.

::

    <root>/
    ├── metadata/scenes.csv           scene table (also bundled with the package)
    ├── aerial/<scene_id>.jpg         1024 x 1024 RGB, 0.20 m/px, north-up
    ├── cameras/<cam>/<scene_id>.jpg  2448 x 1000 RGB, cam in front/right/back/left
    └── lidar/<scene_id>.bin          float32 (N, 4): x, y, z, intensity

Images may be stored as JPEG or PNG; each image path is resolved to whichever
of the extensions in :data:`gooseloc.constants.IMAGE_EXTS` exists on disk.

The dataset root is, in this order: the ``root`` argument, the
``GOOSELOC_ROOT`` environment variable, or ``data/GOOSELoc`` (relative to the
working directory, then to the repository checkout) if it exists.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
from PIL import Image

from gooseloc.constants import CAMERAS, DEFAULT_ROOT, IMAGE_EXTS

ROOT_ENV = "GOOSELOC_ROOT"
_REPO_DIR = Path(__file__).resolve().parents[2]


def default_root_candidates() -> list[Path]:
    """Where ``data/GOOSELoc`` is looked for when no root is given."""
    return [Path.cwd() / DEFAULT_ROOT, _REPO_DIR / DEFAULT_ROOT]


def resolve_root(root: str | Path | None) -> Path:
    """``root`` argument, else ``$GOOSELOC_ROOT``, else an existing ``data/GOOSELoc``."""
    if root is None:
        root = os.environ.get(ROOT_ENV)
    if root is None:
        for candidate in default_root_candidates():
            if candidate.is_dir():
                return candidate
        raise ValueError(
            f"pass root=..., set the {ROOT_ENV} environment variable, "
            f"or download the dataset into {DEFAULT_ROOT}/"
        )
    return Path(root).expanduser()


def _image_path(stem: Path) -> Path:
    """``stem`` with the first image extension that exists (the first one if none does)."""
    for ext in IMAGE_EXTS:
        path = stem.with_name(stem.name + ext)
        if path.exists():
            return path
    return stem.with_name(stem.name + IMAGE_EXTS[0])


def aerial_path(root: Path, scene_id: str) -> Path:
    return _image_path(root / "aerial" / str(scene_id))


def camera_path(root: Path, scene_id: str, camera: str) -> Path:
    if camera not in CAMERAS:
        raise ValueError(f"unknown camera {camera!r}; expected one of {CAMERAS}")
    return _image_path(root / "cameras" / camera / str(scene_id))


def lidar_path(root: Path, scene_id: str) -> Path:
    return root / "lidar" / f"{scene_id}.bin"


def read_image(path: Path, size_hw: tuple[int, int] | None = None) -> Image.Image:
    """Open an image as RGB, optionally resized (bicubic) to ``(height, width)``."""
    with Image.open(path) as im:
        im = im.convert("RGB")
        if size_hw is not None and im.size != (size_hw[1], size_hw[0]):
            im = im.resize((size_hw[1], size_hw[0]), Image.BICUBIC)
        return im


def read_lidar(path: Path) -> np.ndarray:
    """LiDAR sweep as ``float32 (N, 4)``: x forward, y left, z up (m), intensity."""
    return np.fromfile(path, dtype=np.float32).reshape(-1, 4)
