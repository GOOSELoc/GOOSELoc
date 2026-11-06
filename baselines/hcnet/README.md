# HC-Net on GOOSELoc

[HC-Net](https://github.com/xlwangDev/HC-Net) (Wang et al., NeurIPS 2023)
with the four surround cameras as one pseudo-panorama, projected to a
bird's-eye view; north-up and IMU-prior setting. Upstream has no license
file: `setup.sh` clones it on your machine and none of its code is in this
repository. Our adapter is MIT.

## Setup

```bash
pip install -e .                                   # from the repository root
pip install -r baselines/hcnet/requirements.txt
bash baselines/hcnet/setup.sh                      # fetches upstream @ 62329dd
```

## Run a checkpoint

Download the checkpoints (from the repository root):

```bash
hf download Skyy93/GOOSELoc --repo-type dataset --local-dir data/GOOSELoc --include "checkpoints/hcnet/*"
```

The checkpoint name selects the setting.

```bash
python baselines/hcnet/predict.py --checkpoint data/GOOSELoc/checkpoints/hcnet/hcnet_imu_prior_s0.pt \
    --split test --out hcnet_imu_prior_s0_test.csv
gooseloc-eval hcnet_imu_prior_s0_test.csv --split test
```

## Expected results

Loc in m, ori in °. The IMU-prior test mean of seed 2 includes one scene on
which the homography degenerates (several km off).

| setting | split | seed | loc mean | loc median | ori mean | ori median |
|---|---|---|---|---|---|---|
| north-up | test | 0 | 31.49 | 28.85 | 86.86 | 84.03 |
| north-up | test | 1 | 31.71 | 28.60 | 87.63 | 85.82 |
| north-up | test | 2 | 32.20 | 28.45 | 86.60 | 83.46 |
| **north-up** | **test** | **mean** | **31.8±0.4** | **28.6±0.2** | **87.0±0.5** | **84.4±1.2** |
| north-up | dev | mean | 31.3±0.2 | 27.3±0.6 | 85.7±1.5 | 84.2±4.8 |
| IMU ±10° | test | 0 | 29.89 | 27.13 | 16.48 | 12.58 |
| IMU ±10° | test | 1 | 30.09 | 27.60 | 13.51 | 11.09 |
| IMU ±10° | test | 2 | 35.64 | 26.97 | 15.12 | 11.43 |
| **IMU ±10°** | **test** | **mean** | **31.9±3.3** | **27.2±0.3** | **15.0±1.5** | **11.7±0.8** |
| IMU ±10° | dev | mean | 29.8±0.5 | 26.2±0.5 | 12.4±0.6 | 8.9±0.5 |

## Citation

```bibtex
@inproceedings{wang2023hcnet,
  title     = {Fine-Grained Cross-View Geo-Localization Using a Correlation-Aware Homography Estimator},
  author    = {Wang, Xiaolong and Xu, Runsen and Cui, Zhuofan and Wan, Zeyu and Zhang, Yu},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
  volume    = {36},
  year      = {2023}
}
```
