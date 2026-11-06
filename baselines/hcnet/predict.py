"""Run a released HC-Net checkpoint on a GOOSELoc split and write an official prediction CSV.

    python baselines/hcnet/predict.py --checkpoint hcnet_imu_prior_s0.pt \\
        --root data/GOOSELoc --split test --out hcnet_imu_s0_test.csv
    gooseloc-eval hcnet_imu_s0_test.csv --split test

HC-Net regresses the homography from the BEV to the aerial image; the
position is the projected BEV centre and the orientation the angle of the
projected BEV up-axis (6 GRU iterations). The aerial tile is north-up in both
settings, so the position only needs scaling to the 1024 px tile; the compass
heading is recovered from the orientation output and the panorama roll. Run
``setup.sh`` first.
"""

from __future__ import annotations

import argparse
import sys
import types
import warnings
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from data import SAT_SCALE, SAT_SIZE, HCNetGOOSELoc, compass_heading, import_upstream  # noqa: E402

GRU_ITERS = 6

# Upstream wraps its update step in the deprecated torch.cuda.amp.autocast (disabled).
warnings.filterwarnings("ignore", message=r".*torch\.cuda\.amp\.autocast.*", category=FutureWarning)


def hcnet_args() -> types.SimpleNamespace:
    """Architecture flags of upstream's ``models/config/VIGOR/train-vigor.json``."""
    return types.SimpleNamespace(gpuid=[0], mixed_precision=False, iters_lev0=GRU_ITERS, lev0=True,
                                 flow=True, p_siamese=True, CNN16=True, dropout=0.0)


def build_model(device: str | torch.device = "cpu") -> torch.nn.Module:
    """HC-Net without fetching the ImageNet EfficientNet weights (a checkpoint overwrites them)."""
    import_upstream()
    import models.efficientnet_pytorch.model as effnet
    from models.network import HCNet

    orig = effnet.load_pretrained_weights
    effnet.load_pretrained_weights = lambda *args, **kwargs: None
    try:
        model = HCNet(hcnet_args())
    finally:
        effnet.load_pretrained_weights = orig
    return model.to(device)


def load_checkpoint(path: str | Path, device: str | torch.device = "cpu") -> tuple[torch.nn.Module, dict]:
    """Model with the weights of a released checkpoint, in eval mode, and its config."""
    ckpt = torch.load(path, map_location=device, weights_only=True)
    model = build_model(device)
    model.load_state_dict(ckpt["state_dict"], strict=True)
    return model.eval(), ckpt["config"]


def project(four_point: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Corner offsets ``(B, 2, 2, 2)`` to position and orientation.

    Returns the projected BEV centre ``(B, 2)`` in 512 px aerial coordinates
    and the angle of the projected BEV up-axis ``(B,)`` in degrees.
    """
    from models.utils.utils import get_homograpy

    b = four_point.shape[0]
    H = get_homograpy(four_point, [b, 1, SAT_SIZE, SAT_SIZE])
    # BEV centre and a point SAT_SIZE / 8 above it, homogeneous (rows x, y, 1).
    pts = torch.tensor([[SAT_SIZE / 2.0, SAT_SIZE / 2.0],
                        [SAT_SIZE / 2.0, SAT_SIZE / 2.0 - SAT_SIZE / 8.0],
                        [1.0, 1.0]], device=four_point.device)
    x = H.bmm(pts.unsqueeze(0).expand(b, -1, -1))
    x = x / x[:, 2, :].unsqueeze(1)
    dx = x[:, 0, 1] - x[:, 0, 0]
    dy = x[:, 1, 0] - x[:, 1, 1]
    return x[:, 0:2, 0], -torch.rad2deg(torch.atan2(dx, dy))


@torch.no_grad()
def predict(model: torch.nn.Module, loader: DataLoader, device: str | torch.device = "cpu") -> pd.DataFrame:
    """Predictions ``scene_id, x, y, heading_deg`` (north-up 1024 px tile, compass heading)."""
    frames = []
    for batch in loader:
        four = model(batch["bev"].to(device), batch["sat"].to(device), iters_lev0=GRU_ITERS, test_mode=True)
        pos, ori = project(four)
        pos, ori = pos.cpu().double().numpy(), ori.cpu().double().numpy()
        frames.append(pd.DataFrame({
            "scene_id": batch["scene_id"],
            "x": pos[:, 0] * SAT_SCALE,
            "y": pos[:, 1] * SAT_SCALE,
            "heading_deg": compass_heading(batch["roll"].numpy(), ori),
        }))
    return pd.concat(frames, ignore_index=True)


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", type=Path, required=True, help="hcnet_{north_up,imu_prior}_s{seed}.pt")
    p.add_argument("--setting", choices=("north_up", "imu_prior"), default=None,
                   help="evaluation protocol (default: the checkpoint's)")
    p.add_argument("--root", type=Path, default=None, help="GOOSELoc root (default: $GOOSELOC_ROOT, else data/GOOSELoc)")
    p.add_argument("--split", default="test", help="train, dev, test or test_buffered")
    p.add_argument("--out", type=Path, required=True, help="prediction CSV to write")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--limit", type=int, default=None,
                   help="only the first N scenes, for a quick check (evaluate with --allow-partial)")
    args = p.parse_args(argv)

    model, config = load_checkpoint(args.checkpoint, args.device)
    setting = args.setting or config["setting"]
    ds = HCNetGOOSELoc(args.root, args.split, setting=setting, training=False)
    if args.limit is not None:
        ds = Subset(ds, range(min(args.limit, len(ds))))
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
                        pin_memory=str(args.device).startswith("cuda"))
    preds = predict(model, loader, args.device)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    preds.to_csv(args.out, index=False)
    print(f"wrote {len(preds)} {setting} predictions to {args.out}")


if __name__ == "__main__":
    main()
