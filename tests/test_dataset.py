import numpy as np
import pytest

from gooseloc import GOOSELoc
from gooseloc.constants import IMU_PRIOR_RANGE_DEG
from gooseloc.data import geometry as g


def _marker_xy(aerial: np.ndarray) -> np.ndarray:
    ys, xs = np.nonzero(aerial[..., 0] > 64)
    return np.array([xs.mean() + 0.5, ys.mean() + 0.5])


def test_north_up_sample(mini_root):
    ds = GOOSELoc(mini_root, "dev", setting="north_up")
    s = ds[0]
    row = ds.scenes.iloc[0]
    assert s["aerial"].shape == (1024, 1024, 3) and s["aerial"].dtype == np.uint8
    assert s["cameras"].shape == (4, 100, 245, 3)
    assert s["lidar"].shape == (5000, 4) and s["lidar"].dtype == np.float32
    np.testing.assert_array_equal(s["gt_xy"], [row.gt_px_x, row.gt_px_y])
    assert s["gt_heading_deg"] == pytest.approx(row.heading_deg, abs=1e-4)
    assert s["tile_rotation_deg"] == 0.0
    np.testing.assert_allclose(_marker_xy(s["aerial"]), s["gt_xy"], atol=0.6)


def test_imu_prior_uses_fixed_offsets(mini_root):
    ds = GOOSELoc(mini_root, "test", setting="imu_prior")
    row = ds.scenes.iloc[1]
    a, b = ds[1], ds[1]
    assert a["tile_rotation_deg"] == pytest.approx(row.heading_deg + row.imu_prior_offset_deg, abs=1e-4)
    assert a["gt_heading_deg"] == pytest.approx(-row.imu_prior_offset_deg, abs=1e-4)
    np.testing.assert_array_equal(a["aerial"], b["aerial"])
    np.testing.assert_allclose(a["gt_xy"], g.rotate_xy(a["gt_xy_north_up"], a["tile_rotation_deg"]), atol=1e-3)
    np.testing.assert_allclose(_marker_xy(a["aerial"]), a["gt_xy"], atol=0.8)
    back, heading = g.to_north_up(a["gt_xy"], a["gt_heading_deg"], a["tile_rotation_deg"])
    np.testing.assert_allclose(back, a["gt_xy_north_up"], atol=1e-3)
    assert float(heading) == pytest.approx(row.heading_deg, abs=1e-3)


def test_imu_prior_training_draws_random_offsets(mini_root):
    ds = GOOSELoc(mini_root, "train", setting="imu_prior", modalities=("aerial",), seed=3)
    assert ds.training
    rel = [float(ds[0]["gt_heading_deg"]) for _ in range(20)]
    assert max(abs(r) for r in rel) <= IMU_PRIOR_RANGE_DEG + 1e-4
    assert len(set(np.round(rel, 4))) > 1


def test_prior_range_scaling(mini_root):
    ds10 = GOOSELoc(mini_root, "test", setting="imu_prior", modalities=())
    ds20 = GOOSELoc(mini_root, "test", setting="imu_prior", modalities=(), imu_prior_range_deg=20.0)
    assert float(ds20[0]["gt_heading_deg"]) == pytest.approx(2 * float(ds10[0]["gt_heading_deg"]), abs=1e-4)


def test_lidar_subsampling_is_deterministic_in_eval(mini_root):
    ds = GOOSELoc(mini_root, "dev", modalities=("lidar",), lidar_max_points=1000)
    np.testing.assert_array_equal(ds[0]["lidar"], ds[0]["lidar"])
    assert ds[0]["lidar"].shape == (1000, 4)
    tr = GOOSELoc(mini_root, "train", modalities=("lidar",), lidar_max_points=1000, lidar_aug=True)
    assert not np.array_equal(tr[0]["lidar"], tr[0]["lidar"])


def test_options(mini_root):
    ds = GOOSELoc(mini_root, "dev", cameras=("front",), camera_size=(50, 120), modalities=("cameras",))
    s = ds[0]
    assert s["cameras"].shape == (1, 50, 120, 3) and "aerial" not in s and "lidar" not in s
    assert len(GOOSELoc(mini_root, ["dev", "test"], modalities=())) == 6
    with pytest.raises(ValueError):
        GOOSELoc(mini_root, "dev", setting="vehicle_aligned")
    with pytest.raises(ValueError):
        GOOSELoc(mini_root, "dev", cameras=("rear",))


def test_torch_loader(mini_root):
    torch = pytest.importorskip("torch")
    from gooseloc.data.torch_dataset import GOOSELocTorch, make_loader

    ds = GOOSELocTorch(mini_root, "train", setting="imu_prior", camera_size=(64, 152), lidar_max_points=500)
    batch = next(iter(make_loader(ds, batch_size=2, num_workers=0, pin_memory=False)))
    assert batch["aerial"].shape == (2, 3, 1024, 1024) and batch["aerial"].dtype == torch.float32
    assert float(batch["aerial"].max()) <= 1.0
    assert batch["cameras"].shape == (2, 4, 3, 64, 152)
    assert isinstance(batch["lidar"], list) and batch["lidar"][0].shape == (500, 4)
    assert batch["gt_xy"].shape == (2, 2) and len(batch["scene_id"]) == 2


@pytest.mark.data
def test_real_data_sample(data_root):
    ds = GOOSELoc(data_root, "test", setting="imu_prior")
    s = ds[0]
    assert s["aerial"].shape == (1024, 1024, 3)
    assert s["cameras"].shape == (4, 1000, 2448, 3)
    assert s["lidar"].ndim == 2 and s["lidar"].shape[1] == 4
