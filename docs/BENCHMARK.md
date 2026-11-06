# Benchmark protocol

## Task

Given the ground observation of one scene (any subset of the four surround
cameras and the LiDAR sweep) and its 204.8 m x 204.8 m aerial crop, predict the
vehicle's position in the crop and its heading.

## Settings

| setting | aerial input | heading to predict |
|---|---|---|
| `north_up` | north-aligned crop, no orientation information | full 360° compass heading |
| `imu_prior` | crop rotated by a noisy heading prior (true heading + offset, offset uniform in ±10°) | residual within ±10° |

The IMU-prior setting reflects deployment, where a coarse heading from an IMU or
compass is available; north-up is the harder stress test. During training the
prior offset may be drawn freely (the data loader draws a new uniform ±10° offset
on every access). **For dev and test the offset is fixed per scene**
(`imu_prior_offset_deg` in the scene table); `GOOSELoc(..., training=False)`
applies it automatically. Methods that do not consume a rotated tile (e.g. HC-Net,
which rolls its panorama instead) use the same per-scene prior.

## Data use

* Train on `train`; select hyper-parameters and checkpoints on `dev`; report `test`.
* Report the mean ± standard deviation over three training seeds.
* `test_buffered` is a diagnostic subset (no aerial overlap with training), not a
  separate benchmark.
* Test ground truth is public; please do not tune on it.

## Metrics

* **Localization error**: Euclidean distance between predicted and true
  position, in metres (0.20 m/px). Report mean and median; recall at 5/10/25 m is
  printed as well.
* **Orientation error**: absolute shortest-arc difference to the true heading,
  in degrees. Report mean and median. In the north-up setting the mean is
  dominated by front/back (≈180°) flips, so report both.

Results are additionally broken down by terrain (paved / unpaved), season and,
on test, the buffered subset.

## Submission format

A CSV file with one row per scene of the split:

```
scene_id,x,y,heading_deg
1657197242779529338,512.31,488.02,221.4
...
```

`x`, `y` are pixel coordinates in the **north-up** crop (see
[DATASET.md](DATASET.md#coordinate-conventions)) and `heading_deg` is the compass
heading, in both settings. If your model predicts in the rotated IMU-prior tile,
convert with `gooseloc.data.geometry.to_north_up`. `heading_deg` may be omitted
by position-only methods.

```bash
gooseloc-eval predictions.csv --split test --json summary.json --per-scene errors.csv
```

```python
from gooseloc import evaluate
res = evaluate("predictions.csv", split="test")
res["summary"]["all"]["loc_median_m"], res["summary"]["unpaved"]["ori_median_deg"]
```

A trivial reference: always predicting the tile centre gives 39.7 m mean /
41.5 m median on test (orientation at chance, 90°).
