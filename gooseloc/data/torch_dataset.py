"""PyTorch wrapper around :class:`gooseloc.data.dataset.GOOSELoc`.

Images become float tensors in ``[0, 1]`` (channels first), LiDAR sweeps stay
variable-length and are collated into a list::

    from gooseloc.data.torch_dataset import GOOSELocTorch, make_loader

    ds = GOOSELocTorch("data/GOOSELoc", split="train", setting="imu_prior",
                       camera_size=(512, 1216), lidar_max_points=8000, lidar_aug=True)
    loader = make_loader(ds, batch_size=8, shuffle=True, num_workers=8)
    batch = next(iter(loader))
    batch["aerial"]    # (B, 3, 1024, 1024)
    batch["cameras"]   # (B, 4, 3, 512, 1216)
    batch["lidar"]     # list of B tensors (N_i, 4)
"""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Sampler

from gooseloc.data.dataset import GOOSELoc

_LIST_KEYS = ("scene_id", "setting", "lidar", "meta")


def _to_tensor(sample: dict) -> dict:
    out = dict(sample)
    if "aerial" in out:
        out["aerial"] = torch.from_numpy(np.ascontiguousarray(out["aerial"])).permute(2, 0, 1).float().div_(255.0)
    if "cameras" in out:
        out["cameras"] = torch.from_numpy(np.ascontiguousarray(out["cameras"])).permute(0, 3, 1, 2).float().div_(255.0)
    if "lidar" in out:
        out["lidar"] = torch.from_numpy(out["lidar"])
    for key in ("gt_xy", "gt_xy_north_up"):
        out[key] = torch.from_numpy(np.asarray(out[key], dtype=np.float32))
    for key in ("gt_heading_deg", "tile_rotation_deg", "heading_deg"):
        out[key] = torch.tensor(float(out[key]), dtype=torch.float32)
    return out


class GOOSELocTorch(GOOSELoc, Dataset):
    """:class:`GOOSELoc` returning torch tensors. Same arguments."""

    def __getitem__(self, idx: int) -> dict:
        return _to_tensor(GOOSELoc.__getitem__(self, idx))


def collate(batch: list[dict]) -> dict:
    """Stack fixed-size entries; keep ids, LiDAR sweeps and metadata as lists."""
    out: dict = {}
    for key in batch[0]:
        if key in _LIST_KEYS or not isinstance(batch[0][key], torch.Tensor):
            out[key] = [b[key] for b in batch]
        else:
            out[key] = torch.stack([b[key] for b in batch], dim=0)
    return out


def worker_init_fn(worker_id: int) -> None:
    """Give every worker its own random stream (prior offsets, LiDAR augmentation)."""
    info = torch.utils.data.get_worker_info()
    if info is None:
        return
    info.dataset.reseed((info.seed + 9973 * info.dataset.seed + worker_id) % 2**32)


def make_loader(
    dataset: GOOSELocTorch,
    *,
    batch_size: int = 8,
    shuffle: bool | None = None,
    num_workers: int = 4,
    sampler: Sampler | None = None,
    drop_last: bool | None = None,
    pin_memory: bool | None = None,
    **kwargs,
) -> DataLoader:
    """DataLoader with the GOOSELoc collate function and per-worker seeding.

    ``shuffle`` and ``drop_last`` default to ``dataset.training``; ``pin_memory``
    defaults to whether CUDA is available.
    """
    pin_memory = torch.cuda.is_available() if pin_memory is None else pin_memory
    shuffle = dataset.training if shuffle is None else shuffle
    drop_last = dataset.training if drop_last is None else drop_last
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle if sampler is None else False,
        sampler=sampler, num_workers=num_workers, collate_fn=collate,
        worker_init_fn=worker_init_fn, pin_memory=pin_memory, drop_last=drop_last,
        persistent_workers=num_workers > 0, **kwargs,
    )
