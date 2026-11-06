"""Train GOOSELocNet (paper recipe).

    torchrun --nproc_per_node=4 scripts/train.py \
        --config configs/gooselocnet/rgbl_imu_prior.yaml \
        --seed 0 --out runs/rgbl_imu_prior_s0

AdamW with three parameter groups (projections lr 1e-3; heads lr 5e-4; norms,
embeddings, temperature and gains without weight decay), cosine schedule with
one warm-up epoch, fp16 autocast, gradient accumulation to an effective batch
of 16. Every ``eval_every`` epochs the dev split is evaluated with the official
protocol; ``last.pt`` is always written, ``best.pt`` on a new best dev median
position error. The paper reports ``best.pt`` for the IMU-prior setting and
``last.pt`` for north-up (``report_checkpoint`` in the config).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import yaml
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from gooseloc.data.torch_dataset import GOOSELocTorch, make_loader  # noqa: E402
from gooseloc.eval.benchmark import per_scene_errors  # noqa: E402
from gooseloc.models.gooselocnet import (  # noqa: E402
    GOOSELocNet,
    GOOSELocNetConfig,
    dataset_kwargs,
    predict_batch,
    save_checkpoint,
    total_loss,
)
from gooseloc.models.gooselocnet.inference import model_inputs  # noqa: E402

NO_DECAY = ("cam_pe", "lidar_pe", "fusion_query", "yaw_cam_az", "yaw_lidar_az")
PROJECTIONS = ("proj_sat", "proj_cam", "proj_lidar")


def param_groups(model: GOOSELocNet, cfg: dict) -> list[dict]:
    proj, head, no_decay = [], [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if n.split(".")[0] in PROJECTIONS:
            proj.append(p)
        elif p.ndim <= 1 or n in NO_DECAY:
            no_decay.append(p)
        else:
            head.append(p)
    return [{"params": proj, "lr": cfg["lr_proj"], "weight_decay": cfg["weight_decay"]},
            {"params": head, "lr": cfg["lr_head"], "weight_decay": cfg["weight_decay"]},
            {"params": no_decay, "lr": cfg["lr_head"], "weight_decay": 0.0}]


def lr_multiplier(step: int, warmup: int, total: int) -> float:
    if step < warmup:
        return (step + 1) / warmup
    progress = min(max((step - warmup) / max(1, total - warmup), 0.0), 1.0)
    return 0.5 * (1.0 + math.cos(math.pi * progress))


@torch.no_grad()
def evaluate_dev(model, loader, scenes, rank: int, world: int, amp: bool) -> dict:
    model.eval()
    rows = []
    for batch in loader:
        p = predict_batch(model, batch, amp=amp)
        rows += [dict(zip(p, vals)) for vals in zip(*p.values())]
    if world > 1:
        gathered = [None] * world
        dist.all_gather_object(gathered, rows)
        rows = [r for part in gathered for r in part]
    model.train()
    if rank != 0:
        return {}
    import pandas as pd
    pred = pd.DataFrame(rows).drop_duplicates("scene_id")   # DistributedSampler pads
    err = per_scene_errors(pred, "dev", scenes=scenes)
    return {"n": len(err), "loc_median_m": float(err.loc_err_m.median()),
            "loc_mean_m": float(err.loc_err_m.mean()),
            "ori_median_deg": float(err.ori_err_deg.median()),
            "ori_mean_deg": float(err.ori_err_deg.mean())}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--root", default=None, help="dataset root (default: $GOOSELOC_ROOT, else data/GOOSELoc)")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--resume", type=Path, default=None, help="last.pt of an interrupted run")
    p.add_argument("--max-steps", type=int, default=None, help="debug: stop each epoch early")
    args = p.parse_args()
    cfg = yaml.safe_load(args.config.read_text())

    ddp = "RANK" in os.environ
    if ddp:
        dist.init_process_group("nccl")
        rank, world = dist.get_rank(), dist.get_world_size()
        local = int(os.environ.get("LOCAL_RANK", rank))
        torch.cuda.set_device(local)
        device = torch.device("cuda", local)
    else:
        rank, world = 0, 1
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed = args.seed + rank
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = True
    log = print if rank == 0 else (lambda *a, **k: None)

    model_cfg = GOOSELocNetConfig(variant=cfg["variant"], **cfg.get("model", {}))
    train_ds = GOOSELocTorch(args.root, "train", setting=cfg["setting"], seed=seed,
                             **dataset_kwargs(model_cfg, training=True))
    dev_ds = GOOSELocTorch(args.root, "dev", setting=cfg["setting"],
                           **dataset_kwargs(model_cfg, training=False))
    bs = int(cfg["batch_size_per_gpu"])
    accum = max(1, int(cfg["effective_batch_size"]) // (bs * world))
    train_sampler = DistributedSampler(train_ds, shuffle=True, seed=args.seed, drop_last=True) if ddp else None
    dev_sampler = DistributedSampler(dev_ds, shuffle=False) if ddp else None
    nw = int(cfg.get("num_workers", 4))
    train_loader = make_loader(train_ds, batch_size=bs, sampler=train_sampler, num_workers=nw,
                               shuffle=train_sampler is None, drop_last=True)
    dev_loader = make_loader(dev_ds, batch_size=bs, sampler=dev_sampler, num_workers=nw)

    model = GOOSELocNet(model_cfg).to(device)
    log(f"{cfg['variant']}/{cfg['setting']}: {sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6:.2f} M "
        f"trainable parameters | {len(train_ds)} train / {len(dev_ds)} dev scenes | "
        f"{world} GPU(s) x batch {bs} x accum {accum}")
    core = model
    if ddp:
        model = DDP(model, device_ids=[device.index], find_unused_parameters=False)
    opt = torch.optim.AdamW(param_groups(core, cfg), lr=cfg["lr_proj"])
    base_lrs = [g["lr"] for g in opt.param_groups]
    amp = bool(cfg.get("amp", True)) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp)

    steps = len(train_loader) if args.max_steps is None else min(len(train_loader), args.max_steps)
    warmup = max(1, int(cfg["warmup_epochs"]) * steps)
    total = max(warmup + 1, int(cfg["epochs"]) * steps)
    start_epoch, step_global, best = 0, 0, float("inf")
    if args.resume is not None:
        ck = torch.load(args.resume, map_location="cpu", weights_only=False)
        core.load_trainable_state_dict(ck["model"])
        opt.load_state_dict(ck["optimizer"])
        if amp and "scaler" in ck:
            scaler.load_state_dict(ck["scaler"])
        start_epoch, step_global, best = ck["epoch"] + 1, ck["global_step"], ck["best_loc_median_m"]

    args.out.mkdir(parents=True, exist_ok=True)
    if rank == 0:
        (args.out / "config.yaml").write_text(yaml.safe_dump(cfg | {"seed": args.seed, "world": world}))
    history = []
    for epoch in range(start_epoch, int(cfg["epochs"])):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        model.train()
        t0, running = time.time(), 0.0
        for step, batch in enumerate(train_loader):
            if step >= steps:
                break
            for g, base in zip(opt.param_groups, base_lrs):
                g["lr"] = base * lr_multiplier(step_global, warmup, total)
            if step % accum == 0:
                opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
                losses = total_loss(model(**model_inputs(batch, device)), batch,
                                    sigma_patches=cfg["sigma_patches"], sigma_deg=cfg["heading_sigma_deg"],
                                    lambda_heading=cfg["lambda_heading"])
            # The last accumulation window of an epoch may be shorter.
            rem = steps % accum
            window = rem if (rem and step >= steps - rem) else accum
            scaler.scale(losses["loss"] / window).backward()
            if (step + 1) % accum == 0 or step + 1 == steps:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_([p for g in opt.param_groups for p in g["params"]],
                                               cfg["grad_clip"])
                scaler.step(opt)
                scaler.update()
            step_global += 1
            running += float(losses["loss"])
            if rank == 0 and (step + 1) % 100 == 0:
                log(f"epoch {epoch:02d} step {step + 1}/{steps} loss {running / (step + 1):.4f} "
                    f"tau {core.tau.item():.4f}")
        log(f"epoch {epoch:02d} done in {time.time() - t0:.0f}s, loss {running / max(1, steps):.4f}")

        if (epoch + 1) % int(cfg["eval_every"]) == 0 or epoch + 1 == int(cfg["epochs"]):
            metrics = evaluate_dev(core, dev_loader, dev_ds.scenes, rank, world, amp)
            if rank == 0:
                history.append({"epoch": epoch, **metrics})
                (args.out / "history.json").write_text(json.dumps(history, indent=2))
                is_best = metrics["loc_median_m"] < best
                best = min(best, metrics["loc_median_m"])
                extra = {"setting": cfg["setting"], "seed": args.seed, "epoch": epoch,
                         "global_step": step_global, "dev_metrics": metrics, "best_loc_median_m": best,
                         "optimizer": opt.state_dict(), "scaler": scaler.state_dict()}
                save_checkpoint(args.out / "last.pt", core, **extra)
                if is_best:
                    save_checkpoint(args.out / "best.pt", core, **extra)
                log(f"dev: {json.dumps(metrics)}{'  (new best)' if is_best else ''}")
    log(f"done; the paper reports {cfg['report_checkpoint']}.pt of this run")
    if ddp:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
