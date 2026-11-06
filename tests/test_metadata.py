import zlib

import numpy as np
import pytest

from gooseloc.constants import GSD_M, IMU_PRIOR_RANGE_DEG, SEASONS, TERRAINS
from gooseloc.data.geometry import utm_to_tile_px
from gooseloc.data.metadata import select


def test_split_sizes_match_paper(scenes):
    assert len(scenes) == 9356
    assert scenes["scene_id"].is_unique
    counts = scenes["split"].value_counts().to_dict()
    assert counts == {"train": 6430, "dev": 1064, "test": 1862}
    assert len(select(scenes, "test_buffered")) == 840
    assert int((scenes["terrain"] == "paved").sum()) == 4288


def test_categorical_values(scenes):
    assert set(scenes["season"]) == set(SEASONS)
    assert set(scenes["terrain"]) == set(TERRAINS)
    assert not scenes.isna().any().any()


def test_gt_pixel_in_central_region(scenes):
    for col in ("gt_px_x", "gt_px_y"):
        assert scenes[col].between(256, 767).all()


def test_georeference_matches_gt_pixel(scenes):
    xy = utm_to_tile_px(scenes.utm_e, scenes.utm_n, scenes.tile_utm_e0, scenes.tile_utm_n0)
    err = np.abs(xy - scenes[["gt_px_x", "gt_px_y"]].to_numpy())
    assert err.max() <= 0.5 + 1e-6          # integer crop offsets, <= half a pixel
    assert err.max() * GSD_M <= 0.1 + 1e-6


def test_imu_prior_offsets_are_the_paper_protocol(scenes):
    off = scenes["imu_prior_offset_deg"]
    assert off.abs().max() <= IMU_PRIOR_RANGE_DEG
    for sid, value in scenes[["scene_id", "imu_prior_offset_deg"]].head(50).itertuples(index=False):
        seed = zlib.crc32(sid.encode()) & 0xFFFFFFFF
        expected = np.random.default_rng(seed).uniform(-IMU_PRIOR_RANGE_DEG, IMU_PRIOR_RANGE_DEG)
        assert value == pytest.approx(expected, abs=1e-9)


def test_test_split_is_pose_verified(scenes):
    assert (select(scenes, "test")["gnss_ins_offset_m"] <= 5.0).all()


def test_select_filters(scenes):
    sub = select(scenes, ["dev", "test"], terrain="unpaved", season=["winter", "summer"])
    assert set(sub.split) <= {"dev", "test"} and set(sub.terrain) == {"unpaved"}
    assert set(sub.season) == {"winter", "summer"}
    with pytest.raises(ValueError):
        select(scenes, "val")
