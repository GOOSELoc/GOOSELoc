# GOOSELocNet

Our RGB / LiDAR-only / RGB+L baseline. Code:
[`gooseloc/models/gooselocnet/`](../gooseloc/models/gooselocnet/), one training
config per paper row in [`configs/gooselocnet/`](../configs/gooselocnet/).

Frozen DINOv3 (aerial tile, cameras) and Utonia (LiDAR, with LoRA) encoders;
the ground queries are correlated with the aerial patch tokens into a heat map
for the position, and a 72-bin classifier predicts the heading.

## Installation

```bash
pip install -e ".[model]"           # torch, timm, pyyaml
# LiDAR variants additionally need Utonia, spconv and torch_scatter (CUDA builds matching
# your torch; example for torch 2.7.1 + CUDA 12.6):
pip install addict spconv-cu126
pip install torch_scatter -f https://data.pyg.org/whl/torch-2.7.1+cu126.html
# Utonia's setup.py imports pkg_resources, hence setuptools<81 and no build isolation:
pip install "setuptools<81"
pip install --no-build-isolation --no-deps git+https://github.com/Pointcept/Utonia.git@9a13cbf
```

flash-attn is not needed. DINOv3 and Utonia weights are downloaded on first use
(see their model cards for the licenses).

## Run a checkpoint

The checkpoints are part of the dataset download (see the [README](../README.md#download)):

```bash
hf download Skyy93/GOOSELoc --repo-type dataset --local-dir data/GOOSELoc --include "checkpoints/gooselocnet/*"
```

`data/GOOSELoc/checkpoints/gooselocnet/` holds `gooselocnet_{rgb,lidar,rgbl}_{north_up,imu_prior}_s{0,1,2}.pt`.

```bash
python scripts/predict.py --checkpoint data/GOOSELoc/checkpoints/gooselocnet/gooselocnet_rgbl_imu_prior_s0.pt \
    --split test --out rgbl_imu_prior_s0_test.csv
gooseloc-eval rgbl_imu_prior_s0_test.csv --split test
```

```python
import torch

from gooseloc.data.torch_dataset import GOOSELocTorch, make_loader
from gooseloc.models.gooselocnet import dataset_kwargs, load_checkpoint, predict_batch

if __name__ == "__main__":  # needed for worker processes on macOS and Windows
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, ck = load_checkpoint("data/GOOSELoc/checkpoints/gooselocnet/gooselocnet_rgb_imu_prior_s0.pt",
                                device=device)
    ds = GOOSELocTorch(split="test", setting=ck["setting"],
                       **dataset_kwargs(model.config, training=False))
    for batch in make_loader(ds, batch_size=8):
        pred = predict_batch(model, batch)      # north-up x, y, heading_deg
```

## Training

```bash
torchrun --nproc_per_node=4 scripts/train.py --config configs/gooselocnet/rgbl_imu_prior.yaml \
    --seed 0 --out runs/rgbl_imu_prior_s0
```

One RGB+L run takes about 13.5 h on 4 V100 32 GB GPUs.

## Expected results (test, mean ± std over 3 seeds)

| setting | variant | loc mean (m) | loc median (m) | ori mean (°) | ori median (°) |
|---|---|---|---|---|---|
| North-up | RGB | 29.8±0.1 | 25.5±0.1 | 49.0±5.3 | 27.9±3.3 |
| North-up | LiDAR-only | 28.7±0.3 | 24.9±0.8 | 64.3±3.6 | 38.5±5.4 |
| North-up | RGB+L | 27.1±0.3 | 22.2±0.6 | 46.3±1.1 | 24.8±0.3 |
| IMU ±10° | RGB | 25.1±0.1 | 20.7±0.2 | 4.6±0.1 | 3.9±0.1 |
| IMU ±10° | LiDAR-only | 21.2±0.2 | 16.1±0.1 | 4.2±0.1 | 3.5±0.1 |
| IMU ±10° | RGB+L | 19.5±0.3 | 14.1±0.1 | 4.6±0.1 | 3.8±0.1 |

The LiDAR variants use a random 8,000-point subset per scene
(`predict.py --lidar-seed`), so single runs vary by up to about 1 m.
