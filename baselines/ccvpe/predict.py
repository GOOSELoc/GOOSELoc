# SPDX-License-Identifier: GPL-3.0-only
"""Run a released CCVPE checkpoint on a GOOSELoc split and write an official prediction CSV.

    python baselines/ccvpe/predict.py --checkpoint ccvpe_north_up_s0.pt \\
        --root data/GOOSELoc --split test --out ccvpe_s0_test.csv
    gooseloc-eval ccvpe_s0_test.csv --split test

The position is the arg-max of CCVPE's 512 x 512 location heatmap, mapped to
1024 px north-up tile coordinates; the heading is read from the orientation
field at that pixel. Both are already in the official frame (the tile stays
north-up). Run ``setup.sh`` first.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset

HERE = Path(__file__).resolve().parent
UPSTREAM = HERE / "upstream"
sys.path.insert(0, str(HERE))

from data import CCVPEGOOSELoc, heatmap_to_tile_xy  # noqa: E402


def _import_upstream():
    if not (UPSTREAM / "models.py").exists():
        raise FileNotFoundError(f"{UPSTREAM} not found; run {HERE / 'setup.sh'} first")
    if str(UPSTREAM) not in sys.path:
        sys.path.insert(0, str(UPSTREAM))
    import efficientnet_pytorch.model as effnet  # upstream's vendored copy
    import models
    return models, effnet


def build_model(device: str | torch.device = "cpu") -> torch.nn.Module:
    """``CVM_VIGOR`` with circular padding, as trained for the paper.

    The ImageNet EfficientNet weights that upstream downloads on construction
    are skipped; a checkpoint overwrites them anyway.
    """
    models, effnet = _import_upstream()
    orig = effnet.load_pretrained_weights
    effnet.load_pretrained_weights = lambda *args, **kwargs: None
    try:
        model = models.CVM_VIGOR(device, circular_padding=True)
    finally:
        effnet.load_pretrained_weights = orig
    return model.to(device)


def load_checkpoint(path: str | Path, device: str | torch.device = "cpu") -> tuple[torch.nn.Module, dict]:
    """Model with the weights of a released checkpoint, in eval mode, and its config."""
    ckpt = torch.load(path, map_location=device, weights_only=True)
    model = build_model(device)
    model.load_state_dict(ckpt["state_dict"], strict=True)
    return model.eval(), ckpt["config"]


@torch.no_grad()
def predict(model: torch.nn.Module, loader: DataLoader, device: str | torch.device = "cpu") -> pd.DataFrame:
    """Predictions ``scene_id, x, y, heading_deg`` (north-up 1024 px tile, compass heading)."""
    rows = []
    for batch in loader:
        out = model(batch["grd"].to(device), batch["sat"].to(device))
        heatmap = out[1].cpu().numpy()   # (B, 1, 512, 512)
        ori = out[2].cpu().numpy()       # (B, 2, 512, 512), unit (cos, sin) field
        for b, sid in enumerate(batch["scene_id"]):
            _, r, c = np.unravel_index(heatmap[b].argmax(), heatmap[b].shape)
            x, y = heatmap_to_tile_xy(r, c)
            heading = math.degrees(math.atan2(float(ori[b, 1, r, c]), float(ori[b, 0, r, c]))) % 360.0
            rows.append((sid, float(x), float(y), heading))
    return pd.DataFrame(rows, columns=["scene_id", "x", "y", "heading_deg"])


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", type=Path, required=True, help="ccvpe_north_up_s{seed}.pt")
    p.add_argument("--root", type=Path, default=None, help="GOOSELoc root (default: $GOOSELOC_ROOT, else data/GOOSELoc)")
    p.add_argument("--split", default="test", help="train, dev, test or test_buffered")
    p.add_argument("--out", type=Path, required=True, help="prediction CSV to write")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--limit", type=int, default=None,
                   help="only the first N scenes, for a quick check (evaluate with --allow-partial)")
    args = p.parse_args(argv)

    model, config = load_checkpoint(args.checkpoint, args.device)
    if config.get("setting", "north_up") != "north_up":
        raise ValueError(f"CCVPE checkpoints are north_up only, got {config['setting']!r}")
    ds = CCVPEGOOSELoc(args.root, args.split, training=False)
    if args.limit is not None:
        ds = Subset(ds, range(min(args.limit, len(ds))))
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
                        pin_memory=str(args.device).startswith("cuda"))
    preds = predict(model, loader, args.device)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    preds.to_csv(args.out, index=False)
    print(f"wrote {len(preds)} predictions to {args.out}")


if __name__ == "__main__":
    main()
