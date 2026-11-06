"""Official GOOSELoc evaluation.

A submission is a CSV with one row per scene::

    scene_id,x,y,heading_deg

``x``/``y`` are the predicted vehicle position in pixels of the **north-up**
aerial tile (see :mod:`gooseloc.data.geometry`), ``heading_deg`` the predicted
compass heading. ``heading_deg`` may be omitted for position-only methods.
Predictions made in a rotated tile (IMU-prior setting) are mapped back with
:func:`gooseloc.data.geometry.to_north_up` first, so both settings share one
format and one ground truth.

Errors are the Euclidean distance in metres (0.20 m/px) and the shortest-arc
heading difference in degrees. Results are reported for the whole split and per
terrain, per season and, for ``test``, on the leakage-free ``test_buffered``
subset.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from gooseloc.constants import GSD_M, SEASONS, TERRAINS
from gooseloc.data.geometry import angular_error_deg
from gooseloc.data.metadata import load_scenes, select
from gooseloc.eval.metrics import localization_metrics, orientation_metrics


def load_predictions(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path, dtype={"scene_id": str})
    missing = {"scene_id", "x", "y"} - set(df.columns)
    if missing:
        raise ValueError(f"prediction file lacks columns {sorted(missing)}")
    return df


def per_scene_errors(predictions: pd.DataFrame, split: str = "test", *,
                     scenes: pd.DataFrame | None = None,
                     allow_partial: bool = False) -> pd.DataFrame:
    """Join predictions with the ground truth and compute per-scene errors.

    Returns ``scene_id, loc_err_m, ori_err_deg`` (``ori_err_deg`` is NaN when
    no heading was predicted).
    """
    gt = select(scenes if scenes is not None else load_scenes(), split)
    pred = predictions.drop_duplicates("scene_id")
    if len(pred) != len(predictions):
        raise ValueError("duplicate scene ids in the predictions")
    df = gt[["scene_id", "gt_px_x", "gt_px_y", "heading_deg"]].merge(
        pred, on="scene_id", how="left", suffixes=("_gt", ""))
    n_missing = int(df["x"].isna().sum())
    if n_missing and not allow_partial:
        raise ValueError(f"{n_missing}/{len(df)} scenes of split {split!r} have no prediction")
    df = df[df["x"].notna()]
    loc = np.hypot(df["x"] - df["gt_px_x"], df["y"] - df["gt_px_y"]) * GSD_M
    has_heading = "heading_deg" in pred.columns
    ori = (angular_error_deg(df["heading_deg"], df["heading_deg_gt"]) if has_heading
           else np.full(len(df), np.nan))
    return pd.DataFrame({"scene_id": df["scene_id"].to_numpy(),
                         "loc_err_m": np.asarray(loc, dtype=np.float64),
                         "ori_err_deg": np.asarray(ori, dtype=np.float64)})


def summarize(errors: pd.DataFrame, *, scenes: pd.DataFrame | None = None,
              groups: bool = True) -> dict[str, dict]:
    """Aggregate per-scene errors overall and per group.

    ``errors`` needs ``scene_id`` and ``loc_err_m``; ``ori_err_deg`` is optional.
    """
    meta = scenes if scenes is not None else load_scenes()
    df = errors.merge(meta[["scene_id", "split", "terrain", "season", "test_buffered"]],
                      on="scene_id", how="left")
    if df["terrain"].isna().any():
        raise ValueError("errors contain unknown scene ids")
    if "ori_err_deg" not in df:
        df["ori_err_deg"] = np.nan

    def block(sub: pd.DataFrame) -> dict:
        return localization_metrics(sub["loc_err_m"]) | orientation_metrics(sub["ori_err_deg"])

    out = {"all": block(df)}
    if groups:
        for t in TERRAINS:
            if (df["terrain"] == t).any():
                out[t] = block(df[df["terrain"] == t])
        for s in SEASONS:
            if (df["season"] == s).any():
                out[s] = block(df[df["season"] == s])
        if (df["split"] == "test").all() and df["test_buffered"].any():
            out["test_buffered"] = block(df[df["test_buffered"]])
    return out


def evaluate(predictions: pd.DataFrame | str | Path, split: str = "test", **kwargs) -> dict:
    """Evaluate a submission. Returns ``{"summary": ..., "per_scene": DataFrame}``."""
    if not isinstance(predictions, pd.DataFrame):
        predictions = load_predictions(predictions)
    scenes = kwargs.pop("scenes", None)
    if scenes is None:
        scenes = load_scenes()
    errors = per_scene_errors(predictions, split, scenes=scenes, **kwargs)
    return {"summary": summarize(errors, scenes=scenes), "per_scene": errors}


def format_summary(summary: dict[str, dict]) -> str:
    """Plain-text table of a :func:`summarize` result."""
    has_ori = any("ori_mean_deg" in v for v in summary.values())
    head = f"{'group':<14}{'n':>6}{'loc mean':>10}{'loc med':>9}{'R@5m':>7}{'R@10m':>7}{'R@25m':>7}"
    if has_ori:
        head += f"{'ori mean':>10}{'ori med':>9}{'R@5°':>7}{'R@10°':>7}"
    lines = [head, "-" * len(head)]
    for name, m in summary.items():
        line = (f"{name:<14}{m['n']:>6}{m['loc_mean_m']:>10.2f}{m['loc_median_m']:>9.2f}"
                f"{m['loc_recall@5m']:>7.3f}{m['loc_recall@10m']:>7.3f}{m['loc_recall@25m']:>7.3f}")
        if has_ori:
            line += (f"{m['ori_mean_deg']:>10.2f}{m['ori_median_deg']:>9.2f}"
                     f"{m['ori_recall@5deg']:>7.3f}{m['ori_recall@10deg']:>7.3f}")
        lines.append(line)
    return "\n".join(lines)
