"""The scene table: one row per GOOSELoc scene.

``scenes.csv`` is the single source of truth for splits, ground truth and the
benchmark protocol. It ships with the dataset (``<root>/metadata/scenes.csv``)
and is bundled with this package, so evaluation and statistics work without
downloading any sensor data. See ``docs/DATASET.md`` for the column reference.
"""

from __future__ import annotations

from collections.abc import Iterable
from importlib import resources
from pathlib import Path

import pandas as pd

from gooseloc.constants import SEASONS, SPLITS, SUBSETS, TERRAINS

_DTYPES = {
    "scene_id": str, "split": str, "date": str, "season": str, "terrain": str,
    "dop20_date": str,
}


def bundled_scenes_path() -> Path:
    return Path(str(resources.files("gooseloc") / "resources" / "scenes.csv"))


def load_scenes(root: str | Path | None = None) -> pd.DataFrame:
    """Load the scene table.

    Uses ``<root>/metadata/scenes.csv`` if ``root`` is given and contains one,
    otherwise the copy bundled with the package.
    """
    path = bundled_scenes_path()
    if root is not None:
        candidate = Path(root).expanduser() / "metadata" / "scenes.csv"
        if candidate.exists():
            path = candidate
    df = pd.read_csv(path, dtype=_DTYPES, float_precision="round_trip")
    df["test_buffered"] = df["test_buffered"].astype(bool)
    return df


def select(
    scenes: pd.DataFrame,
    split: str | Iterable[str] | None = None,
    *,
    terrain: str | None = None,
    season: str | Iterable[str] | None = None,
    scene_ids: Iterable[str] | None = None,
) -> pd.DataFrame:
    """Filter the scene table.

    ``split`` is one of ``train``, ``dev``, ``test``, ``test_buffered`` (the
    leakage-free test subset, appendix of the paper), ``all``/``None``, or a
    list of those.
    """
    mask = pd.Series(True, index=scenes.index)
    if split is not None and split != "all":
        names = [split] if isinstance(split, str) else list(split)
        split_mask = pd.Series(False, index=scenes.index)
        for name in names:
            if name in SPLITS:
                split_mask |= scenes["split"] == name
            elif name == "test_buffered":
                split_mask |= (scenes["split"] == "test") & scenes["test_buffered"]
            else:
                raise ValueError(f"unknown split {name!r}; expected one of {SPLITS + SUBSETS}")
        mask &= split_mask
    if terrain is not None:
        if terrain not in TERRAINS:
            raise ValueError(f"unknown terrain {terrain!r}; expected one of {TERRAINS}")
        mask &= scenes["terrain"] == terrain
    if season is not None:
        seasons = [season] if isinstance(season, str) else list(season)
        unknown = set(seasons) - set(SEASONS)
        if unknown:
            raise ValueError(f"unknown season(s) {sorted(unknown)}; expected one of {SEASONS}")
        mask &= scenes["season"].isin(seasons)
    if scene_ids is not None:
        mask &= scenes["scene_id"].isin(set(map(str, scene_ids)))
    return scenes[mask].reset_index(drop=True)
