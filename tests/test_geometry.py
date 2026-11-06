import numpy as np
import pytest
from PIL import Image

from gooseloc.data import geometry as g


def test_rotate_xy_inverse():
    xy = np.array([[300.0, 700.0], [512.0, 512.0], [10.5, 1000.25]])
    for a in (0.0, 7.3, 90.0, -135.0):
        np.testing.assert_allclose(g.rotate_xy(g.rotate_xy(xy, a), -a), xy, atol=1e-9)


@pytest.mark.parametrize("angle", [0.0, 37.0, 90.0, 181.5, -60.0])
def test_rotate_xy_matches_pil_rotation(angle):
    """A marker at xy lands at rotate_xy(xy) after rotate_tile."""
    xy = np.array([700.0, 330.0])
    img = np.zeros((1024, 1024), np.uint8)
    img[328:332, 698:702] = 255                      # 4x4 marker centred on (700, 330)
    rot = np.asarray(g.rotate_tile(Image.fromarray(img), angle), dtype=float)
    ys, xs = np.nonzero(rot > 64)
    centroid = np.array([xs.mean() + 0.5, ys.mean() + 0.5])   # pixel centres
    np.testing.assert_allclose(centroid, g.rotate_xy(xy, angle), atol=0.6)


def test_heading_up_after_rotating_by_heading():
    """Rotating the tile CCW by the heading makes the vehicle's forward point up."""
    heading = 60.0                                    # facing ENE
    fwd = g.vehicle_to_tile_px(np.array([[10.0, 0.0]]), heading, (512.0, 512.0))[0]
    up = g.rotate_xy(fwd, heading)
    assert up[0] == pytest.approx(512.0, abs=1e-6) and up[1] < 512.0


def test_to_north_up_round_trip():
    xy_nu = np.array([[400.0, 600.0]])
    heading, rot = 123.0, 128.5
    xy_rot = g.rotate_xy(xy_nu, rot)
    rel = g.wrap_deg_180(heading - rot)
    back, h = g.to_north_up(xy_rot, rel, rot)
    np.testing.assert_allclose(back, xy_nu, atol=1e-9)
    assert float(h[0] if np.ndim(h) else h) == pytest.approx(heading)


def test_vehicle_frame_conventions():
    o = (512.0, 512.0)
    # Facing north: forward is up (smaller y), left is west (smaller x).
    fwd, left = g.vehicle_to_tile_px(np.array([[1.0, 0.0], [0.0, 1.0]]), 0.0, o)
    assert fwd[1] < o[1] and fwd[0] == pytest.approx(o[0])
    assert left[0] < o[0] and left[1] == pytest.approx(o[1])


def test_utm_round_trip():
    xy = np.array([[100.25, 900.5]])
    utm = g.tile_px_to_utm(xy, 690000.0, 5330000.0)
    np.testing.assert_allclose(g.utm_to_tile_px(utm[:, 0], utm[:, 1], 690000.0, 5330000.0), xy)


def test_angular_error():
    np.testing.assert_allclose(g.angular_error_deg([359.0, 10.0, 180.0], [1.0, 350.0, 0.0]), [2.0, 20.0, 180.0])
