# Image baselines

The four cross-view localization methods from the paper, with our checkpoints.
Our own baseline, GOOSELocNet, is in the package (see [docs/MODEL.md](../docs/MODEL.md)).

| folder | method | input | settings | checkpoints | license |
|---|---|---|---|---|---|
| [`fg2/`](fg2) | FG² (Xia & Alahi, CVPR 2025) | front camera / 4 cameras as virtual forward views | north-up, IMU prior | 12 (front/surround x 2 settings x 3 seeds) | GPL-3.0 |
| [`ccvpe/`](ccvpe) | CCVPE (Xia et al., T-PAMI 2023) | 4-camera pseudo-panorama | north-up | 3 | GPL-3.0 |
| [`hcnet/`](hcnet) | HC-Net (Wang et al., NeurIPS 2023) | 4-camera pseudo-panorama → spherical BEV | north-up, IMU prior | 6 (one model per seed, both protocols) | MIT (adapter only; upstream has no license) |
| [`shi/`](shi) | Shi et al. (ICCV 2023) | front camera | IMU prior | 3 | MIT |

Each folder has a `setup.sh` that fetches the upstream code at the commit the
paper used, a data adapter (`data.py`) and `predict.py`, which turns a
checkpoint into a prediction file for `gooseloc-eval`. Each method runs in its
own environment; install the `gooseloc` package there with `pip install -e .`
from the repository root. The checkpoints are part of the dataset download, in
`data/GOOSELoc/checkpoints/<method>/` (see the [README](../README.md#download)).
