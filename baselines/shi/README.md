# Shi et al. on GOOSELoc

[Shi et al.](https://github.com/YujiaoShi/Boosting3DoFAccuracy) (ICCV 2023)
with the front camera and the aerial tile resized to 512 px; IMU-prior
setting only. MIT, like upstream.

## Setup

```bash
cd baselines/shi
bash setup.sh                                      # fetches upstream @ 2dc4e93
conda env create -f environment.yml && conda activate gooseloc-shi
pip install -e ../..                               # the gooseloc package
```

## Run a checkpoint

Download the checkpoints (from the repository root):

```bash
hf download Skyy93/GOOSELoc --repo-type dataset --local-dir data/GOOSELoc --include "checkpoints/shi/*"
```


```bash
python predict.py --checkpoint ../../data/GOOSELoc/checkpoints/shi/shi_imu_prior_s0.pt \
    --split test --out shi_imu_prior_s0_test.csv
gooseloc-eval shi_imu_prior_s0_test.csv --split test
```

## Expected results

IMU prior ±10°; loc in m, ori in °.

| split | seed | loc mean | loc median | ori mean | ori median |
|---|---|---|---|---|---|
| test | 0 | 34.95 | 25.97 | 4.17 | 3.33 |
| test | 1 | 35.78 | 27.17 | 3.98 | 3.07 |
| test | 2 | 35.79 | 27.54 | 3.95 | 3.04 |
| **test** | **mean** | **35.5±0.5** | **26.9±0.8** | **4.0±0.1** | **3.1±0.2** |
| dev | mean | 32.3±0.5 | 22.5±1.2 | 3.3±0.2 | 2.4±0.2 |

## Citation

```bibtex
@inproceedings{shi2023boosting,
  title     = {Boosting 3-DoF Ground-to-Satellite Camera Localization Accuracy via Geometry-Guided Cross-View Transformer},
  author    = {Shi, Yujiao and Wu, Fei and Perincherry, Akhil and Vora, Ankit and Li, Hongdong},
  booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
  pages     = {21516--21526},
  year      = {2023}
}
```
