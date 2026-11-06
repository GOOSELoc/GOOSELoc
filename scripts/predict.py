"""Run a GOOSELocNet checkpoint on a split and write an official prediction file.

    python scripts/predict.py --checkpoint data/GOOSELoc/checkpoints/gooselocnet/gooselocnet_rgbl_imu_prior_s0.pt \
        --split test --out predictions/rgbl_imu_prior_s0_test.csv

The setting (north-up / IMU prior) is stored in the checkpoint. The output is
``scene_id,x,y,heading_deg`` in the north-up tile frame and is evaluated right
away (same as ``gooseloc-eval``). ``--imu-prior-range`` re-scales the fixed
per-scene prior errors for the robustness study (paper appendix).
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from gooseloc.data.torch_dataset import GOOSELocTorch, make_loader  # noqa: E402
from gooseloc.eval.benchmark import evaluate, format_summary  # noqa: E402
from gooseloc.models.gooselocnet import dataset_kwargs, load_checkpoint, predict_batch  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--root", default=None, help="dataset root (default: $GOOSELOC_ROOT, else data/GOOSELoc)")
    p.add_argument("--split", default="test")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--imu-prior-range", type=float, default=10.0)
    p.add_argument("--lidar-seed", type=int, default=0,
                   help="seed of the fixed per-scene 8,000-point LiDAR subset")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--no-amp", action="store_true")
    args = p.parse_args()

    model, ck = load_checkpoint(args.checkpoint, device=args.device)
    ds = GOOSELocTorch(args.root, args.split, setting=ck["setting"],
                       imu_prior_range_deg=args.imu_prior_range, seed=args.lidar_seed,
                       **dataset_kwargs(model.config, training=False))
    loader = make_loader(ds, batch_size=args.batch_size, num_workers=args.num_workers)
    print(f"{args.checkpoint.name}: {model.config.variant} / {ck['setting']} on {len(ds)} {args.split} scenes")

    parts, t0 = [], time.time()
    for i, batch in enumerate(loader):
        parts.append(pd.DataFrame(predict_batch(model, batch, amp=not args.no_amp)))
        if (i + 1) % 20 == 0:
            n = sum(len(x) for x in parts)
            print(f"  {n}/{len(ds)} scenes ({n / (time.time() - t0):.1f}/s)", flush=True)
    pred = pd.concat(parts, ignore_index=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    pred.to_csv(args.out, index=False, float_format="%.4f")
    print(f"wrote {args.out}")
    print(format_summary(evaluate(pred, args.split)["summary"]))


if __name__ == "__main__":
    main()
