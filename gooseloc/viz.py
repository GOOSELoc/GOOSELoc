"""Plotting helpers (needs matplotlib: ``pip install gooseloc[viz]``)."""

from __future__ import annotations

import numpy as np

from gooseloc.constants import CAMERAS, GSD_M
from gooseloc.data import geometry


def _arrow(ax, xy, heading_rel_deg, color, length=60.0, **kw):
    th = np.radians(heading_rel_deg)
    ax.annotate("", xy=(xy[0] + length * np.sin(th), xy[1] - length * np.cos(th)), xytext=tuple(xy),
                arrowprops=dict(arrowstyle="-|>", color=color, lw=2), **kw)


def plot_aerial(sample: dict, ax=None, *, prediction: dict | None = None, heatmap=None,
                lidar: bool = True):
    """Aerial tile as the model sees it, with GT (gold) and optional prediction (red).

    ``prediction`` holds ``x``, ``y`` (pixels of the delivered tile) and
    optionally ``heading_deg`` (relative to its up-axis). ``heatmap`` is an
    optional ``(G, G)`` posterior that is stretched over the tile. With
    ``lidar=True`` and a LiDAR sweep in the sample, the points are overlaid at
    the GT pose.
    """
    import matplotlib.pyplot as plt

    ax = ax or plt.gca()
    ax.imshow(sample["aerial"])
    size = sample["aerial"].shape[0]
    gt = sample["gt_xy"]
    if lidar and "lidar" in sample:
        pts = sample["lidar"]
        pts = pts[np.hypot(pts[:, 0], pts[:, 1]) < 60.0][::4]
        uv = geometry.vehicle_to_tile_px(pts[:, :2], float(sample["gt_heading_deg"]), gt)
        ax.scatter(uv[:, 0], uv[:, 1], s=0.2, c=pts[:, 2], cmap="cool", alpha=0.5)
    if heatmap is not None:
        ax.imshow(np.asarray(heatmap), extent=(0, size, size, 0), cmap="viridis", alpha=0.45)
    ax.plot(*gt, marker="*", color="gold", markersize=16, markeredgecolor="black")
    _arrow(ax, gt, float(sample["gt_heading_deg"]), "gold")
    if prediction is not None:
        p = (prediction["x"], prediction["y"])
        ax.plot(*p, marker="o", color="red", markersize=9, markerfacecolor="none", markeredgewidth=2)
        if "heading_deg" in prediction:
            _arrow(ax, p, float(prediction["heading_deg"]), "red")
    m = sample.get("meta", {})
    ax.set_title(f"{sample['scene_id']}  {m.get('terrain', '')} / {m.get('season', '')}  "
                 f"({sample.get('setting', '')})", fontsize=9)
    ax.set_xlim(0, size)
    ax.set_ylim(size, 0)
    ax.set_axis_off()
    return ax


def plot_cameras(sample: dict, axes=None, cameras=CAMERAS):
    """The surround images in a row."""
    import matplotlib.pyplot as plt

    imgs = sample["cameras"]
    if axes is None:
        _, axes = plt.subplots(1, len(imgs), figsize=(4 * len(imgs), 2))
    for ax, img, name in zip(np.atleast_1d(axes), imgs, cameras):
        ax.imshow(img)
        ax.set_title(name, fontsize=9)
        ax.set_axis_off()
    return axes


def plot_lidar_bev(sample: dict, ax=None, radius_m: float = 60.0):
    """Bird's-eye view of the LiDAR sweep in the vehicle frame (forward = up)."""
    import matplotlib.pyplot as plt

    ax = ax or plt.gca()
    pts = sample["lidar"]
    keep = np.hypot(pts[:, 0], pts[:, 1]) < radius_m
    ax.scatter(-pts[keep, 1], pts[keep, 0], s=0.1, c=pts[keep, 2], cmap="viridis")
    ax.set_aspect("equal")
    ax.set_xlim(-radius_m, radius_m)
    ax.set_ylim(-radius_m, radius_m)
    ax.set_title("LiDAR (vehicle frame, forward up)", fontsize=9)
    return ax


def plot_sample(sample: dict, prediction: dict | None = None, heatmap=None):
    """Overview figure: aerial tile, LiDAR bird's-eye view and the four cameras."""
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(16, 9))
    grid = fig.add_gridspec(2, 4, height_ratios=[3, 1])
    plot_aerial(sample, fig.add_subplot(grid[0, :2]), prediction=prediction, heatmap=heatmap)
    if "lidar" in sample:
        plot_lidar_bev(sample, fig.add_subplot(grid[0, 2:]))
    if "cameras" in sample:
        plot_cameras(sample, [fig.add_subplot(grid[1, i]) for i in range(len(sample["cameras"]))])
    fig.text(0.01, 0.01, f"1 px = {GSD_M:.2f} m. Aerial imagery: Datenquelle: Bayerische "
             "Vermessungsverwaltung – www.geodaten.bayern.de (CC BY 4.0)", fontsize=7)
    fig.tight_layout()
    return fig
