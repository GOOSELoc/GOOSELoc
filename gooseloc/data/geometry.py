"""Coordinate conventions, in one place.

Aerial tile
    Pixel coordinates are continuous ``(x, y)`` with the origin at the top-left
    corner of the tile, ``x`` to the right (east) and ``y`` down (south) for a
    north-up tile; the centre of pixel ``(i, j)`` is at ``(i + 0.5, j + 0.5)``.
    One pixel is 0.20 m. The tile is ``TILE_SIZE_PX`` wide and rotations are
    about its centre ``(512, 512)``.

Heading
    Compass convention: degrees clockwise from north, in ``[0, 360)``. In a
    (possibly rotated) tile, the *relative* heading is the vehicle's forward
    direction measured clockwise from the tile's up-axis.

Tile rotation
    ``tile_rotation_deg`` is the counter-clockwise rotation applied to the
    north-up tile (PIL convention). Rotating by the vehicle heading makes the
    vehicle face up. In the IMU-prior setting the tile is rotated by
    ``heading + prior_offset``, so the relative heading the model has to
    recover is ``-prior_offset``.

UTM
    ETRS89 / UTM 32N (EPSG:25832). ``tile_utm_e0`` / ``tile_utm_n0`` are the
    easting / northing of the north-up tile's top-left corner.
"""

from __future__ import annotations

import numpy as np
from PIL import Image

from gooseloc.constants import GSD_M, TILE_SIZE_PX

TILE_CENTRE = TILE_SIZE_PX / 2.0


def wrap_deg_360(a):
    """Wrap angles to ``[0, 360)``."""
    return np.mod(a, 360.0)


def wrap_deg_180(a):
    """Wrap angles to ``[-180, 180)``."""
    return np.mod(np.asarray(a, dtype=np.float64) + 180.0, 360.0) - 180.0


def angular_error_deg(pred, gt):
    """Absolute shortest-arc difference in degrees, in ``[0, 180]``."""
    return np.abs(wrap_deg_180(np.asarray(pred, dtype=np.float64) - np.asarray(gt, dtype=np.float64)))


def rotate_xy(xy, angle_ccw_deg, centre: float = TILE_CENTRE) -> np.ndarray:
    """Map tile pixel coordinates through a CCW image rotation about the centre.

    ``xy`` has shape ``(..., 2)``. Matches :func:`rotate_tile`: the pixel content
    at ``xy`` in the input tile is found at the returned position afterwards.
    """
    xy = np.asarray(xy, dtype=np.float64)
    th = np.radians(angle_ccw_deg)
    c, s = np.cos(th), np.sin(th)
    rx, ry = xy[..., 0] - centre, xy[..., 1] - centre
    return np.stack([rx * c + ry * s + centre, -rx * s + ry * c + centre], axis=-1)


def rotate_tile(tile: Image.Image, angle_ccw_deg: float) -> Image.Image:
    """Rotate a tile CCW about its centre (bicubic, black fill, size kept)."""
    if angle_ccw_deg == 0.0:
        return tile
    return tile.rotate(float(angle_ccw_deg), resample=Image.BICUBIC)


def to_north_up(xy, rel_heading_deg, tile_rotation_deg):
    """Convert a prediction in a rotated tile back to the north-up tile.

    Returns ``(xy_north_up, compass_heading_deg)``. ``rel_heading_deg`` may be
    ``None`` for position-only predictions.
    """
    xy_nu = rotate_xy(xy, -np.asarray(tile_rotation_deg, dtype=np.float64))
    if rel_heading_deg is None:
        return xy_nu, None
    return xy_nu, wrap_deg_360(np.asarray(rel_heading_deg) + np.asarray(tile_rotation_deg))


def utm_to_tile_px(utm_e, utm_n, tile_utm_e0, tile_utm_n0) -> np.ndarray:
    """UTM coordinates to continuous pixel coordinates of the north-up tile."""
    x = (np.asarray(utm_e) - np.asarray(tile_utm_e0)) / GSD_M
    y = (np.asarray(tile_utm_n0) - np.asarray(utm_n)) / GSD_M
    return np.stack([x, y], axis=-1)


def tile_px_to_utm(xy, tile_utm_e0, tile_utm_n0) -> np.ndarray:
    """Continuous north-up tile pixel coordinates to UTM ``(easting, northing)``."""
    xy = np.asarray(xy, dtype=np.float64)
    return np.stack([np.asarray(tile_utm_e0) + xy[..., 0] * GSD_M,
                     np.asarray(tile_utm_n0) - xy[..., 1] * GSD_M], axis=-1)


def vehicle_to_tile_px(points_xy, heading_deg, origin_xy) -> np.ndarray:
    """Project vehicle-frame points (x forward, y left; metres) into the north-up tile.

    ``origin_xy`` is the vehicle's pixel position, ``heading_deg`` its compass
    heading. Useful for overlaying LiDAR on the aerial image.
    """
    p = np.asarray(points_xy, dtype=np.float64)
    th = np.radians(heading_deg)
    east = np.sin(th) * p[..., 0] - np.cos(th) * p[..., 1]
    north = np.cos(th) * p[..., 0] + np.sin(th) * p[..., 1]
    return np.stack([origin_xy[0] + east / GSD_M, origin_xy[1] - north / GSD_M], axis=-1)
