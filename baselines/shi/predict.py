"""Run a released Shi et al. checkpoint on a GOOSELoc split (IMU-prior setting).

Writes the official prediction CSV (``scene_id,x,y,heading_deg``: pixels of
the north-up 1024 px tile and compass heading), ready for ``gooseloc-eval``::

    python predict.py --checkpoint shi_imu_prior_s0.pt --root data/GOOSELoc \\
        --split test --out shi_s0_test.csv
    gooseloc-eval shi_s0_test.csv --split test
"""

from __future__ import annotations

import argparse
import logging
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from data import ShiGOOSELoc, to_north_up
from model import load_model, predict_batch
from torch.utils.data import DataLoader, Subset

log = logging.getLogger("shi.predict")


def predict(checkpoint: str | Path, root: str | Path | None = None, split: str = "test", *,
            device: str = "cpu", batch_size: int = 4, num_workers: int = 4,
            limit: int | None = None) -> pd.DataFrame:
    """Predictions for ``split`` as a DataFrame in the official format."""
    net = load_model(checkpoint, device)
    dataset = ShiGOOSELoc(root, split, training=False)
    if limit is not None:
        dataset = Subset(dataset, range(min(limit, len(dataset))))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    rows = []
    for i, batch in enumerate(loader):
        u, v, theta = predict_batch(net, batch)
        xy, heading = to_north_up(u, v, theta, batch["tile_rotation_deg"].numpy())
        rows.append(pd.DataFrame({"scene_id": batch["scene_id"], "x": xy[:, 0], "y": xy[:, 1],
                                  "heading_deg": heading}))
        if i % 50 == 0:
            log.info("batch %d/%d", i + 1, len(loader))
    return pd.concat(rows, ignore_index=True)


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", type=Path, required=True, help="shi_imu_prior_s{seed}.pt")
    p.add_argument("--root", type=Path, default=None, help="GOOSELoc root (default: $GOOSELOC_ROOT, else data/GOOSELoc)")
    p.add_argument("--split", default="test", help="train, dev, test or test_buffered")
    p.add_argument("--out", type=Path, required=True, help="prediction CSV to write")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--limit", type=int, default=None,
                   help="only the first N scenes (smoke test; evaluate with --allow-partial)")
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    warnings.filterwarnings("ignore", message="torch.meshgrid")   # upstream, default "ij" is intended

    df = predict(args.checkpoint, args.root, args.split, device=args.device,
                 batch_size=args.batch_size, num_workers=args.num_workers, limit=args.limit)
    if not np.isfinite(df[["x", "y", "heading_deg"]].to_numpy()).all():
        raise RuntimeError("non-finite predictions")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False, float_format="%.4f")
    log.info("wrote %d predictions to %s", len(df), args.out)


if __name__ == "__main__":
    main()
