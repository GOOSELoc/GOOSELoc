"""Metric definitions (localization in metres, orientation in degrees)."""

from __future__ import annotations

import numpy as np

LOC_THRESHOLDS_M = (5.0, 10.0, 25.0, 50.0)
ORI_THRESHOLDS_DEG = (5.0, 10.0, 30.0)


def localization_metrics(err_m) -> dict[str, float]:
    e = np.asarray(err_m, dtype=np.float64)
    out = {"n": int(e.size), "loc_mean_m": float(e.mean()), "loc_median_m": float(np.median(e))}
    for t in LOC_THRESHOLDS_M:
        out[f"loc_recall@{t:g}m"] = float((e <= t).mean())
    return out


def orientation_metrics(err_deg) -> dict[str, float]:
    e = np.asarray(err_deg, dtype=np.float64)
    if e.size == 0 or np.isnan(e).all():
        return {}
    if np.isnan(e).any():
        raise ValueError("orientation errors must be all set or all missing")
    out = {"ori_mean_deg": float(e.mean()), "ori_median_deg": float(np.median(e))}
    for t in ORI_THRESHOLDS_DEG:
        out[f"ori_recall@{t:g}deg"] = float((e <= t).mean())
    return out
