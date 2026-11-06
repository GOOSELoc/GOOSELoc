# Performance by terrain (test, median ± std over 3 seeds)

| Setting | Method | paved loc | paved ori | unpaved loc | unpaved ori |
|---|---|---|---|---|---|
| North-up | FG² (surround) | 27.6±0.4 | 89.9±0.2 | 36.1±0.4 | 98.0±0.9 |
| North-up | Ours (RGB) | 25.5±0.7 | 28.6±3.3 | 25.5±0.3 | 27.6±4.2 |
| North-up | Ours (LiDAR-only) | 23.2±0.6 | 35.6±7.3 | 26.5±0.9 | 40.8±3.9 |
| North-up | Ours (RGB+L) | 21.2±0.3 | 24.8±1.6 | 22.9±0.7 | 25.3±1.1 |
| IMU ±10° | FG² (surround) | 13.8±0.1 | 4.0±0.2 | 27.0±0.7 | 4.7±0.2 |
| IMU ±10° | Ours (RGB) | 19.5±0.5 | 3.9±0.1 | 21.5±0.2 | 3.8±0.0 |
| IMU ±10° | Ours (LiDAR-only) | 13.4±0.1 | 3.5±0.2 | 19.1±0.5 | 3.5±0.1 |
| IMU ±10° | Ours (RGB+L) | 12.2±0.3 | 3.8±0.1 | 15.4±0.3 | 3.8±0.1 |
