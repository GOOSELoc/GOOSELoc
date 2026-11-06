# Results

The per-scene errors of every method, setting, split and seed of the paper.

| path | content |
|---|---|
| `per_scene/<method>/<setting>_<split>_s<seed>.csv` | per-scene errors `scene_id, loc_err_m, ori_err_deg` |
| `summary_only.csv` | nine cells that were only logged as summaries (FG² seed 0 dev/front, CCVPE dev) |
| `robustness/<method>_prior<deg>.csv` | GOOSELocNet seed 0, test, IMU prior error scaled to ±0/5/10/20/45° |
| `tables/*.md` | main results (with dev), terrain, season, buffered test, heading-prior robustness |

Methods: `ours_rgb`, `ours_lidar`, `ours_rgbl` (GOOSELocNet), `fg2_front`,
`fg2_surround`, `ccvpe`, `hcnet`, `shi`. Settings: `north_up`, `imu_prior`.
