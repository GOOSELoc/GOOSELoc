# Full test vs. buffered test (n=840)

| Setting | Method | full loc (mean) | full ori (median) | buffered loc (mean) | buffered ori (median) |
|---|---|---|---|---|---|
| North-up | CCVPE (surround) | 45.4±2.9 | 35.5±17.0 | 46.3±2.8 | 39.9±13.0 |
| North-up | HC-Net (surround) | 31.8±0.4 | 84.4±1.2 | 31.8±0.2 | 83.6±2.1 |
| North-up | FG² (surround) | 34.4±0.4 | 93.1±0.3 | 34.2±0.2 | 92.8±0.2 |
| North-up | Ours (RGB) | 29.8±0.1 | 27.9±3.3 | 28.8±0.1 | 35.1±10.6 |
| North-up | Ours (LiDAR-only) | 28.7±0.3 | 38.5±5.4 | 29.2±0.4 | 49.3±6.4 |
| North-up | Ours (RGB+L) | 27.1±0.3 | 24.8±0.3 | 26.2±0.7 | 29.6±1.8 |
| IMU ±10° | Shi et al. (front) | 35.5±0.5 | 3.1±0.2 | 34.1±0.4 | 3.5±0.1 |
| IMU ±10° | HC-Net (surround) | 31.9±3.3 | 11.7±0.8 | 33.8±6.6 | 12.5±0.6 |
| IMU ±10° | FG² (surround) | 25.0±0.4 | 4.4±0.2 | 25.2±0.5 | 4.9±0.3 |
| IMU ±10° | Ours (RGB) | 25.1±0.1 | 3.9±0.1 | 24.5±0.2 | 4.2±0.2 |
| IMU ±10° | Ours (LiDAR-only) | 21.2±0.2 | 3.5±0.1 | 20.9±0.1 | 3.8±0.1 |
| IMU ±10° | Ours (RGB+L) | 19.5±0.3 | 3.8±0.1 | 18.7±0.2 | 4.0±0.2 |
