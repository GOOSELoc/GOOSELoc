# SPDX-License-Identifier: GPL-3.0-or-later
# Part of the GOOSELoc FG² baseline (see LICENSE in this directory).
"""Run a released FG² checkpoint on a GOOSELoc split.

Writes the official prediction CSV (``scene_id,x,y,heading_deg``: pixels of
the north-up 1024 px tile and compass heading), ready for ``gooseloc-eval``::

    python predict.py --checkpoint fg2_surround_imu_prior_s0.pt --root data/GOOSELoc \\
        --split test --out fg2_surround_imu_prior_s0_test.csv
    gooseloc-eval fg2_surround_imu_prior_s0_test.csv --split test

The checkpoint fixes the setting (``north_up`` / ``imu_prior``) and the cameras
(``front``: the front camera; ``surround``: all four, fused by confidence).
FG² samples its correspondences at random; the samples are seeded per scene
(``--seed``), so a run is reproducible on a given device.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from data import FG2EvalDataset
from model import FG2Localizer, build_model, fuse, load_checkpoint, scene_generator
from torch.utils.data import DataLoader, Subset

from gooseloc.data.geometry import TILE_CENTRE

log = logging.getLogger("fg2.predict")


def predict(checkpoint: str | Path, root: str | Path | None = None, split: str = "test", *,
            device: str = "cuda", num_workers: int = 4, seed: int = 0,
            limit: int | None = None) -> pd.DataFrame:
    """Predictions for ``split`` as a DataFrame in the official format."""
    dino, model = build_model(device)
    config = load_checkpoint(model, checkpoint)
    if not config:
        raise ValueError(f"{checkpoint} is not a GOOSELoc FG² release checkpoint")
    localizer = FG2Localizer(dino, model, device)
    dataset = FG2EvalDataset(root, split, setting=config["setting"], cameras=config["cameras"])
    log.info("%s: %s, setting=%s, cameras=%s, %d scenes", Path(checkpoint).name,
             config["variant"], config["setting"], ",".join(config["cameras"]), len(dataset))
    if limit is not None:
        dataset = Subset(dataset, range(min(limit, len(dataset))))
    loader = DataLoader(dataset, batch_size=None, shuffle=False, num_workers=num_workers)

    rows, n_failed = [], 0
    for i, s in enumerate(loader):
        views = localizer.predict_views(s["sat"], s["grd"], s["camera_k"],
                                        scene_generator(s["scene_id"], seed, device))
        fused = fuse(views, s["view_rotation_deg"].tolist(), s["tile_rotation_deg"])
        if fused is None:          # every view degenerate: fall back to the tile centre / prior
            n_failed += 1
            fused = np.array([TILE_CENTRE, TILE_CENTRE]), s["tile_rotation_deg"] % 360.0
        (x, y), heading = fused
        rows.append((s["scene_id"], float(x), float(y), float(heading)))
        if i % 100 == 0:
            log.info("scene %d/%d", i + 1, len(dataset))
    if n_failed:
        log.warning("%d scenes without a valid Procrustes solution (tile centre written)", n_failed)
    return pd.DataFrame(rows, columns=["scene_id", "x", "y", "heading_deg"])


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", type=Path, required=True, help="fg2_<variant>_<setting>_s<seed>.pt")
    p.add_argument("--root", type=Path, default=None, help="dataset root (default: $GOOSELOC_ROOT, else data/GOOSELoc)")
    p.add_argument("--split", default="test", help="dev, test or test_buffered")
    p.add_argument("--out", type=Path, required=True, help="output CSV")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=0, help="seed of the correspondence sampling")
    p.add_argument("--limit", type=int, default=None, help="only the first N scenes (smoke test)")
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    df = predict(args.checkpoint, args.root, args.split, device=args.device,
                 num_workers=args.num_workers, seed=args.seed, limit=args.limit)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    log.info("wrote %d predictions to %s", len(df), args.out)


if __name__ == "__main__":
    main()
