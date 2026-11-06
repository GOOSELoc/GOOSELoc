# FG² on GOOSELoc

[FG²](https://github.com/vita-epfl/FG2) (Xia & Alahi, CVPR 2025) with each
camera as a forward view of its own rotated aerial tile; the per-camera poses
are fused. Variants `front` and `surround`, north-up and IMU-prior setting.
This folder is GPL-3.0 like upstream (see [LICENSE](LICENSE)).

## Setup

```bash
bash baselines/fg2/setup.sh          # upstream FG² @ a55d9d1 + goose.patch -> baselines/fg2/upstream/
conda env create -f baselines/fg2/environment.yml
conda activate gooseloc-fg2
MMCV_WITH_OPS=1 FORCE_CUDA=1 TORCH_CUDA_ARCH_LIST="7.0" \
    pip install mmcv==2.1.0 --no-binary mmcv --no-build-isolation   # CUDA ops, ~20-40 min
pip install -e .                     # the gooseloc package (repository root)
```

The frozen DINOv2 weights (~1.2 GB) are downloaded on first use.

## Run a checkpoint

Download the checkpoints (from the repository root):

```bash
hf download Skyy93/GOOSELoc --repo-type dataset --local-dir data/GOOSELoc --include "checkpoints/fg2/*"
```


```bash
cd baselines/fg2
python predict.py --checkpoint ../../data/GOOSELoc/checkpoints/fg2/fg2_surround_imu_prior_s0.pt \
    --split test --out fg2_surround_imu_prior_s0_test.csv
gooseloc-eval fg2_surround_imu_prior_s0_test.csv --split test
```

The test split takes about 15 min (`front`) and 1 h (`surround`) on one V100.

## Expected results

Loc mean / median in m, ori mean / median in °. FG² samples its
correspondences at random (`--seed`), so a rerun differs by a few tenths.

| Checkpoint | s0 | s1 | s2 |
|---|---|---|---|
| `fg2_front_north_up` | 37.62 / 35.23 m, 92.04 / 93.26° | 38.93 / 35.69 m, 91.75 / 93.51° | 36.58 / 34.22 m, 92.09 / 93.44° |
| `fg2_surround_north_up` | 34.25 / 32.01 m, 92.50 / 92.70° | 34.16 / 31.97 m, 92.58 / 93.26° | 34.84 / 32.43 m, 92.41 / 93.27° |
| `fg2_front_imu_prior` | 29.49 / 25.14 m, 6.63 / 4.84° | 29.15 / 24.48 m, 6.47 / 4.77° | 28.89 / 24.02 m, 6.66 / 4.87° |
| `fg2_surround_imu_prior` | 25.20 / 20.71 m, 5.22 / 4.50° | 24.56 / 19.51 m, 4.95 / 4.19° | 25.26 / 20.48 m, 5.09 / 4.49° |

## Citation

```bibtex
@inproceedings{xia2025fg,
  title     = {FG\^{} 2: Fine-Grained Cross-View Localization by Fine-Grained Feature Matching},
  author    = {Xia, Zimin and Alahi, Alexandre},
  booktitle = {Proceedings of the Computer Vision and Pattern Recognition Conference},
  pages     = {6362--6372},
  year      = {2025}
}
```
