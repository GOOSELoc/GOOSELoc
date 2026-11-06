# The GOOSELoc dataset

GOOSELoc pairs 9,356 ground scenes recorded with the MuCAR-3 research vehicle
around Munich (July 2022 – July 2023, all four seasons) with official 20 cm
DOP20 orthophotos of the Bavarian surveying administration. Each scene has four
surround-view camera images, one 128-beam LiDAR sweep, an aerial crop and an
INS ground-truth pose. Roughly half of the scenes are on paved roads, the other
half on unpaved forest tracks, field paths and rural tracks.

* [Download and layout](#download-and-layout)
* [The scene table](#the-scene-table-scenescsv)
* [Coordinate conventions](#coordinate-conventions)
* [Sensors](#sensors)
* [Ground truth and aerial reference](#ground-truth-and-aerial-reference)
* [Splits](#splits)
* [License](#license)

## Download and layout

GOOSELoc is published on the Hugging Face Hub as
[Skyy93/GOOSELoc](https://huggingface.co/datasets/Skyy93/GOOSELoc). The code
looks for it in `data/GOOSELoc` inside the repository by default:

```bash
pip install -U huggingface_hub
hf download Skyy93/GOOSELoc --repo-type dataset --local-dir data/GOOSELoc
```

Use `--include` to fetch only what you need:

| part | size | `--include` |
|---|---:|---|
| metadata, checksums, README, license | < 10 MB | `"metadata/*" "CHECKSUMS.sha256" "README.md" "LICENSE.md"` |
| aerial crops | 4.5 GB | `"aerial/*"` |
| LiDAR sweeps | 26 GB | `"lidar/*"` |
| cameras (all four) | 40 GB | `"cameras/*"` (or `"cameras/front/*"`) |
| checkpoints of the paper | 4.1 GB | `"checkpoints/*"` (or `"checkpoints/<method>/*"`) |

```
data/GOOSELoc/
├── README.md, LICENSE.md, CHECKSUMS.sha256
├── metadata/
│   ├── scenes.csv                       one row per scene (also bundled with the Python package)
│   ├── splits/{train,dev,test,test_buffered}.txt
│   └── calibration.yaml                 camera intrinsics and sensor frames (preliminary)
├── aerial/<scene_id>.jpg                1024 x 1024 RGB, 0.20 m/px, north-up
├── cameras/{front,right,back,left}/<scene_id>.jpg   2448 x 1000 RGB
├── lidar/<scene_id>.bin                 float32 (N, 4): x, y, z, intensity
└── checkpoints/{gooselocnet,fg2,ccvpe,hcnet,shi}/   checkpoints of the paper
```

Images are read as `.jpg` or, if that file does not exist, `.png`
(`gooseloc.constants.IMAGE_EXTS`), so both encodings of the release work
with the same code.

The dataset root is taken from, in this order, the `root=...` / `--root`
argument, the `GOOSELOC_ROOT` environment variable, and `data/GOOSELoc`
(relative to the working directory, then to the repository). Check a download with

```bash
gooseloc-check                # all files present?
gooseloc-check --deep         # ... and decodable with the right shapes
gooseloc-check --checksums    # ... and matching CHECKSUMS.sha256 (sensor files)
```

The evaluation and all statistics need just `metadata/` (which ships with the
Python package), a LiDAR-only method needs `aerial/` and `lidar/`. For a
partial download, check only what you fetched, e.g.
`gooseloc-check --split test --modalities aerial lidar --checksums`.

## The scene table (`scenes.csv`)

| column | type | description |
|---|---|---|
| `scene_id` | str | GOOSE scene id; equals the capture timestamp in nanoseconds |
| `split` | str | `train`, `dev` or `test` |
| `timestamp_ns` | int | capture time (UTC, ns since epoch) |
| `date` | str | capture date (UTC), `YYYY-MM-DD` |
| `season` | str | meteorological season of `date`: `winter` (Dec–Feb), `spring`, `summer`, `autumn` |
| `terrain` | str | manually verified `paved` / `unpaved` label |
| `trajectory_id` | int | drive segment (scenes less than 60 s apart); segments are cut at split borders |
| `lat`, `lon` | float | ground-truth position (WGS84), INS pose (RTK-corrected where available) |
| `utm_e`, `utm_n` | float | the same in ETRS89 / UTM 32N (EPSG:25832), metres |
| `heading_deg` | float | ground-truth compass heading, degrees clockwise from north, `[0, 360)` |
| `gnss_lat`, `gnss_lon` | float | raw GNSS fix at the same instant (a realistic, degraded position prior) |
| `gnss_ins_offset_m` | float | distance between GNSS fix and INS pose; `<= 5 m` for every test scene |
| `gt_px_x`, `gt_px_y` | int | **localization target**: vehicle position in pixels of the north-up aerial crop |
| `tile_utm_e0`, `tile_utm_n0` | float | UTM coordinates of the aerial crop's top-left corner |
| `imu_prior_offset_deg` | float | fixed heading-prior error of the IMU-prior setting, uniform in ±10° (used for dev/test) |
| `dop20_date` | str | flight date of the orthophoto (all June–August 2024/2025) |
| `test_buffered` | bool | test scene whose aerial crop shares no pixel with any training crop |

```python
from gooseloc import load_scenes, select
scenes = load_scenes()                                     # bundled copy, no download needed
test_offroad = select(scenes, "test", terrain="unpaved", season=["winter", "spring"])
```

## Coordinate conventions

All conventions are implemented once, in
[`gooseloc/data/geometry.py`](../gooseloc/data/geometry.py), and tested in
`tests/test_geometry.py`.

**Aerial pixels.** Continuous `(x, y)` with the origin at the top-left corner
of the tile, `x` to the right (east), `y` down (south); one pixel is 0.20 m.
The centre of pixel `(i, j)` is at `(i + 0.5, j + 0.5)`. A UTM coordinate maps
to `x = (E - tile_utm_e0) / 0.2`, `y = (tile_utm_n0 - N) / 0.2`
(`geometry.utm_to_tile_px`). `gt_px_x/y` agree with this mapping of `utm_e/n`
to within half a pixel (the crops were cut at integer pixel offsets).

**Heading.** Compass convention: 0° = north, 90° = east, clockwise.

**Tile rotation.** In the IMU-prior setting the tile is rotated
counter-clockwise about its centre (512, 512) by
`tile_rotation_deg = heading_deg + offset`, so that the vehicle faces
approximately up. Positions map with `geometry.rotate_xy`, and a prediction made
in the rotated tile goes back to the north-up frame with
`geometry.to_north_up(xy, relative_heading, tile_rotation_deg)`. The *relative*
heading is the vehicle's forward direction measured clockwise from the tile's
up-axis; in the IMU-prior setting the ground truth is `-offset`.

**LiDAR.** Vehicle frame: `x` forward, `y` left, `z` up, metres.
`geometry.vehicle_to_tile_px` projects vehicle-frame points into the north-up
tile given the pose (see `gooseloc.viz.plot_aerial` for an overlay).

**Cameras.** Always ordered `front, right, back, left`, i.e. viewing azimuths
0°, 90°, 180°, 270° clockwise from the vehicle's forward axis.

## Sensors

| | |
|---|---|
| Cameras | 4 roof-mounted Basler surround cameras at 90° spacing, 2448 x 1000 px (crop of the 2448 x 2048 sensor) |
| LiDAR | roof-mounted 128-beam rotating Velodyne, one sweep per scene, median 200k points (30k in open terrain to 380k under canopy), most returns within ~66 m |
| INS | Oxford RT3000v3 with RTK-GNSS correction (NTRIP) |
| Aerial | Bavaria DOP20, 20 cm/px, leaf-on flights June–August 2024 (95 %) and August 2025 |

Camera intrinsics are in `metadata/calibration.yaml`.

## Ground truth and aerial reference

The pose of each scene comes from the Oxford RT3000v3 INS: the RTK-corrected pose
where available, otherwise its raw GNSS fix (the share per split is in the
appendix of the paper). It is projected to UTM 32N. The aerial crop is cut at native resolution from 1 km x 1 km
DOP20 tiles (composited where a crop straddles tile borders); it is north-aligned
and the vehicle is placed uniformly at random in the central 512 x 512 px, so it
is at least 51.2 m from any edge but never at a fixed location. The orthophotos
are flown independently of the drives, so every pair carries a temporal gap
(vegetation, snow, construction).

## Splits

The recording area is covered with a regular grid of 1 km x 1 km cells and
every cell, with all its scenes, is assigned to one split; the assignment was
searched to balance size (≈ 70/10/20), terrain and season. The test split keeps
only scenes whose GNSS fix and INS pose agree within 5 m.

| split | winter | spring | summer | autumn | paved | total |
|---|---:|---:|---:|---:|---:|---:|
| train | 1,387 | 1,976 | 1,341 | 1,726 | 2,988 | 6,430 |
| dev | 154 | 304 | 103 | 503 | 400 | 1,064 |
| test | 332 | 608 | 532 | 390 | 900 | 1,862 |
| total | 1,873 | 2,888 | 1,976 | 2,619 | 4,288 | 9,356 |

Drives that cross a cell border are cut, so aerial crops near borders can
overlap across splits. `test_buffered` (840 scenes) removes every test scene
whose crop could share a pixel with a training crop.

## License

The two sources keep their own licenses; there is no single license for the
dataset as a whole (full terms in `LICENSE.md` of the dataset).

* Ground data (`cameras/`, `lidar/`) and `metadata/`:
  [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/), like GOOSE.
  Modified versions have to be shared under the same license.
* Aerial crops (`aerial/`): [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/),
  with the required attribution
  **Datenquelle: Bayerische Vermessungsverwaltung – www.geodaten.bayern.de**.
  The crops were cut from the DOP20 tiles, composited at tile borders and
  JPEG-encoded, and are otherwise unchanged.
* Checkpoints: the license of the method's code (GOOSELocNet and Shi et al.
  MIT, FG² and CCVPE GPL-3.0; HC-Net has no upstream license, see
  [baselines/hcnet](../baselines/hcnet)).
