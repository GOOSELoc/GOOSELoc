# CCVPE on GOOSELoc

[CCVPE](https://github.com/tudelft-iv/CCVPE) (Xia, Booij, Kooij; T-PAMI 2023)
with the four surround cameras as one pseudo-panorama and the aerial tile
resized to 512 px; north-up setting only. This folder is GPL-3.0 like
upstream (see [LICENSE](LICENSE)).

## Setup

```bash
pip install -e .                                   # from the repository root
pip install -r baselines/ccvpe/requirements.txt
bash baselines/ccvpe/setup.sh                      # fetches upstream @ 3028aa5
```

## Run a checkpoint

Download the checkpoints (from the repository root):

```bash
hf download Skyy93/GOOSELoc --repo-type dataset --local-dir data/GOOSELoc --include "checkpoints/ccvpe/*"
```


```bash
python baselines/ccvpe/predict.py --checkpoint data/GOOSELoc/checkpoints/ccvpe/ccvpe_north_up_s0.pt \
    --split test --out ccvpe_north_up_s0_test.csv
gooseloc-eval ccvpe_north_up_s0_test.csv --split test
```

## Expected results

North-up; loc in m, ori in °.

| split | seed | loc mean | loc median | ori mean | ori median |
|---|---|---|---|---|---|
| test | 0 | 44.48 | 38.10 | 69.95 | 54.15 |
| test | 1 | 48.70 | 39.28 | 56.38 | 31.82 |
| test | 2 | 43.09 | 33.70 | 49.29 | 20.68 |
| **test** | **mean** | **45.4±2.9** | **37.0±2.9** | **58.5±10.5** | **35.5±17.0** |
| dev | mean | 33.6±0.4 | 25.7±1.3 | 37.2±8.5 | 16.7±12.2 |

## Citation

```bibtex
@article{xia2023convolutional,
  title   = {Convolutional Cross-View Pose Estimation},
  author  = {Xia, Zimin and Booij, Olaf and Kooij, Julian F. P.},
  journal = {IEEE Transactions on Pattern Analysis and Machine Intelligence},
  volume  = {46},
  number  = {5},
  pages   = {3813--3831},
  year    = {2024}
}
```
