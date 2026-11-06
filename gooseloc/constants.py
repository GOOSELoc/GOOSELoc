"""Benchmark-wide constants. Every other module reads these from here."""

from __future__ import annotations

# --- Aerial reference (Bavaria DOP20) ----------------------------------------
GSD_M = 0.20                       # ground sampling distance, metres per pixel
TILE_SIZE_PX = 1024                # aerial crop side length in pixels
TILE_EXTENT_M = TILE_SIZE_PX * GSD_M   # 204.8 m
GT_REGION_PX = (256, 768)          # GT pixel is sampled in [256, 768) on both axes
UTM_EPSG = 25832                   # ETRS89 / UTM zone 32N (Bavaria's native grid)

# --- Ground sensors -----------------------------------------------------------
# Fixed camera order used everywhere (arrays, configs, models). The azimuth is
# the camera's viewing direction measured clockwise from the vehicle's forward
# axis, i.e. "right" looks 90 degrees clockwise of forward.
CAMERAS: tuple[str, ...] = ("front", "right", "back", "left")
CAMERA_AZIMUTH_DEG: dict[str, float] = {"front": 0.0, "right": 90.0, "back": 180.0, "left": 270.0}
CAMERA_HW = (1000, 2448)           # on-disk image height, width
LIDAR_COLUMNS = ("x", "y", "z", "intensity")

# --- Files on disk -------------------------------------------------------------
DEFAULT_ROOT = "data/GOOSELoc"     # where the dataset is downloaded to by default
IMAGE_EXTS: tuple[str, ...] = (".jpg", ".png")  # tried in this order for every image

# --- Benchmark protocol ------------------------------------------------------
SPLITS: tuple[str, ...] = ("train", "dev", "test")
SUBSETS: tuple[str, ...] = ("test_buffered",)
SETTINGS: tuple[str, ...] = ("north_up", "imu_prior")
IMU_PRIOR_RANGE_DEG = 10.0         # heading prior error is uniform in [-10, +10] degrees
SEASONS: tuple[str, ...] = ("winter", "spring", "summer", "autumn")
TERRAINS: tuple[str, ...] = ("paved", "unpaved")
MODALITIES: tuple[str, ...] = ("aerial", "cameras", "lidar")

# Meteorological seasons, by calendar month (UTC timestamp of the scene).
MONTH_TO_SEASON: dict[int, str] = {
    12: "winter", 1: "winter", 2: "winter",
    3: "spring", 4: "spring", 5: "spring",
    6: "summer", 7: "summer", 8: "summer",
    9: "autumn", 10: "autumn", 11: "autumn",
}
