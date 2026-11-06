"""Data settings, checkpoint I/O and prediction for GOOSELocNet."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from gooseloc.data.geometry import to_north_up
from gooseloc.models.gooselocnet.model import (
    GOOSELocNet,
    GOOSELocNetConfig,
    heading_from_logits,
    patch_to_xy,
)

CAMERA_SIZE = (512, 1216)        # (H, W): 32 x 76 ViT patches, keeps the 2.45 aspect
LIDAR_MAX_POINTS = 8000


def dataset_kwargs(config: GOOSELocNetConfig, *, training: bool) -> dict:
    """Arguments for :class:`gooseloc.data.torch_dataset.GOOSELocTorch` matching the paper."""
    modalities = ["aerial"]
    if config.use_cameras:
        modalities.append("cameras")
    if config.use_lidar:
        modalities.append("lidar")
    return {"modalities": tuple(modalities), "camera_size": CAMERA_SIZE,
            "lidar_max_points": LIDAR_MAX_POINTS, "lidar_aug": training and config.use_lidar,
            "training": training}


def model_inputs(batch: dict, device: torch.device) -> dict:
    return {"aerial": batch["aerial"].to(device, non_blocking=True),
            "cameras": batch["cameras"].to(device, non_blocking=True) if "cameras" in batch else None,
            "lidar": batch.get("lidar")}


def save_checkpoint(path: str | Path, model: GOOSELocNet, **extra) -> None:
    """Release format: config + trainable weights (the frozen encoders reload from their hubs)."""
    torch.save({"format": "gooselocnet-v1", "config": model.config.to_dict(),
                "model": model.trainable_state_dict(), **extra}, path)


def load_checkpoint(path: str | Path, *, device: str | torch.device = "cpu",
                    pretrained: bool = True) -> tuple[GOOSELocNet, dict]:
    ck = torch.load(path, map_location="cpu", weights_only=False)
    if ck.get("format") != "gooselocnet-v1":
        raise ValueError(f"{path} is not a GOOSELocNet release checkpoint")
    model = GOOSELocNet(GOOSELocNetConfig(**ck["config"]), pretrained=pretrained)
    model.load_trainable_state_dict(ck["model"])
    return model.to(device).eval(), ck


@torch.no_grad()
def predict_batch(model: GOOSELocNet, batch: dict, *, amp: bool = True) -> dict[str, np.ndarray]:
    """Predictions of one batch in the north-up tile frame (official submission format)."""
    device = model.device
    ctx = torch.autocast("cuda", dtype=torch.float16, enabled=amp and device.type == "cuda")
    with ctx:
        out = model(**model_inputs(batch, device))
    xy = patch_to_xy(out["pred_patch"].float(), model.config.tile_size, model.config.grid)
    rel = heading_from_logits(out["heading_logits"])
    rot = batch["tile_rotation_deg"].numpy()
    xy_nu, heading = to_north_up(xy.cpu().numpy(), rel.cpu().numpy(), rot)
    return {"scene_id": np.asarray(batch["scene_id"]), "x": xy_nu[:, 0], "y": xy_nu[:, 1],
            "heading_deg": heading}
