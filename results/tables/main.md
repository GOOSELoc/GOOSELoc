# Baseline localization (m) and orientation (°), mean ± std over 3 seeds (Table 3 of the paper)

| Setting | Method | dev loc mean | dev loc med | dev ori mean | dev ori med | test loc mean | test loc med | test ori mean | test ori med |
|---|---|---|---|---|---|---|---|---|---|
| North-up | Predict center | 38.9 | 40.9 | 90.0 | 90.0 | 39.7 | 41.5 | 90.0 | 90.0 |
| North-up | CCVPE (surround) | 33.6±0.4 | 25.7±1.3 | 37.2±8.5 | 16.7±12.2 | 45.4±2.9 | 37.0±2.9 | 52.8±5.0 | 35.6±17.0 |
| North-up | HC-Net (surround) | 31.3±0.2 | 27.3±0.6 | 85.7±1.5 | 84.2±4.8 | 31.8±0.4 | 28.6±0.2 | 87.0±0.5 | 84.4±1.2 |
| North-up | FG² (front) | 32.8±0.9 | 29.3±0.2 | 89.9±0.4 | 91.3±0.4 | 37.7±1.2 | 35.1±0.8 | 92.0±0.2 | 93.4±0.1 |
| North-up | FG² (surround) | 30.4±0.2 | 28.0±0.3 | 89.7±0.1 | 90.4±0.1 | 34.4±0.4 | 32.1±0.3 | 92.5±0.1 | 93.1±0.3 |
| North-up | Ours (RGB) | 25.9±0.1 | 21.8±0.2 | 30.2±4.0 | 11.5±0.9 | 29.8±0.1 | 25.5±0.1 | 49.0±5.3 | 27.9±3.3 |
| North-up | Ours (LiDAR-only) | 23.9±0.4 | 19.8±0.4 | 58.3±4.0 | 16.5±2.0 | 28.7±0.3 | 24.9±0.8 | 64.3±3.6 | 38.5±5.4 |
| North-up | Ours (RGB+L) | 23.1±0.3 | 18.8±0.2 | 26.2±1.0 | 10.2±0.4 | 27.1±0.3 | 22.2±0.6 | 46.3±1.1 | 24.8±0.3 |
| IMU ±10° | Shi et al. (front) | 32.4 | 22.7 | 3.3 | 2.5 | 35.4 | 26.6 | 4.1 | 3.2 |
| IMU ±10° | HC-Net (surround) | 29.8±0.5 | 26.2±0.5 | 12.4±0.6 | 8.9±0.5 | 31.9±3.3 | 27.2±0.3 | 15.0±1.5 | 11.7±0.8 |
| IMU ±10° | FG² (front) | 23.1±0.4 | 17.0±0.4 | 5.2±0.2 | 3.7±0.1 | 29.2±0.3 | 24.6±0.6 | 6.6±0.1 | 4.8±0.1 |
| IMU ±10° | FG² (surround) | 19.8±0.2 | 14.5±0.3 | 4.3±0.1 | 3.6±0.1 | 25.0±0.4 | 20.2±0.6 | 5.1±0.1 | 4.4±0.2 |
| IMU ±10° | Ours (RGB) | 21.1±0.0 | 16.3±0.1 | 3.8±0.1 | 3.0±0.1 | 25.1±0.1 | 20.7±0.2 | 4.6±0.1 | 3.9±0.1 |
| IMU ±10° | Ours (LiDAR-only) | 16.9±0.2 | 11.3±0.2 | 3.4±0.0 | 2.6±0.1 | 21.2±0.2 | 16.1±0.1 | 4.2±0.1 | 3.5±0.1 |
| IMU ±10° | Ours (RGB+L) | 16.2±0.2 | 10.5±0.3 | 3.7±0.1 | 2.8±0.0 | 19.5±0.3 | 14.1±0.1 | 4.6±0.1 | 3.8±0.1 |

Shi et al. is reported without standard deviation, as in the paper.

Re-running the evaluation, e.g. on the per-scene results in `results/per_scene/` or with the
released checkpoints, gives slightly different numbers. This is expected: GPU training and
inference are not fully deterministic, and the camera images were anonymized for the release.
