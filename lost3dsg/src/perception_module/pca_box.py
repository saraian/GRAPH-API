#!/usr/bin/env python3
"""Gravity-canonical PCA oriented box for an object point cloud.

Numpy-only on purpose: importable (and testable) without ROS or the VLM stack.
"""
import numpy as np


def pca_oriented_box(pts):
    """Fit an oriented box to Nx3 map-frame points (z = up).

    PCA is run on the horizontal (xy) projection only, so the box stays
    gravity-aligned: yaw is the orientation of the dominant horizontal axis,
    the z extent is taken straight from the points. This canonicalises the
    PCA sign/order ambiguity (a full 3D PCA fit of a single-view slab would
    tilt the box toward the observed face).

    Returns a dict: center (3,), yaw (rad, in [-pi/2, pi/2)), extents (3,)
    = full sizes along the box's own x/y/z axes. None if fewer than 3 points.
    """
    pts = np.asarray(pts, dtype=float)
    if pts.ndim != 2 or pts.shape[0] < 3 or pts.shape[1] != 3:
        return None

    xy = pts[:, :2]
    mean_xy = xy.mean(axis=0)
    centered = xy - mean_xy
    cov = np.cov(centered.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    major = eigvecs[:, np.argmax(eigvals)]

    yaw = np.arctan2(major[1], major[0])
    # 180-degree flip is the same box: canonicalise to [-pi/2, pi/2)
    if yaw >= np.pi / 2:
        yaw -= np.pi
    elif yaw < -np.pi / 2:
        yaw += np.pi

    c, s = np.cos(yaw), np.sin(yaw)
    rot = np.array([[c, s], [-s, c]])          # world -> box frame
    local = centered @ rot.T
    ext_xy = local.max(axis=0) - local.min(axis=0)
    mid_local = (local.max(axis=0) + local.min(axis=0)) / 2.0
    center_xy = mean_xy + mid_local @ rot       # box center back in world xy

    z_min, z_max = pts[:, 2].min(), pts[:, 2].max()

    return {
        "center": np.array([center_xy[0], center_xy[1], (z_min + z_max) / 2.0]),
        "yaw": float(yaw),
        "extents": np.array([ext_xy[0], ext_xy[1], z_max - z_min]),
    }


def extent_similarity(extents_a, extents_b):
    """Orientation-invariant size similarity of two oriented boxes, in [0, 1].

    Compares sorted extents (so a 90-degree viewpoint change scores 1.0, which
    world-axis AABB IoU cannot do) as the geometric mean of per-axis ratios.
    Returns 0.0 when either box is missing or degenerate. Intended as an
    association signal alongside centroid distance — not a replacement for it.
    """
    if not extents_a or not extents_b:
        return 0.0
    a = np.sort(np.asarray(extents_a, dtype=float))
    b = np.sort(np.asarray(extents_b, dtype=float))
    if a.shape != (3,) or b.shape != (3,) or a[0] <= 0 or b[0] <= 0:
        return 0.0
    ratios = np.minimum(a, b) / np.maximum(a, b)
    return float(ratios.prod() ** (1.0 / 3.0))


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    # 2.0 x 0.5 x 1.0 box rotated 30 degrees about z, centered at (3, -1, 0.5)
    half = np.array([1.0, 0.25, 0.5])
    local = rng.uniform(-1, 1, size=(2000, 3)) * half
    a = np.deg2rad(30)
    rot = np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]])
    world = local @ rot.T + np.array([3.0, -1.0, 0.5])

    box = pca_oriented_box(world)
    assert box is not None
    assert abs(np.rad2deg(box["yaw"]) - 30) < 2, box["yaw"]
    assert np.allclose(box["center"], [3.0, -1.0, 0.5], atol=0.05), box["center"]
    assert np.allclose(box["extents"], 2 * half, atol=0.1), box["extents"]
    # degenerate input
    assert pca_oriented_box(world[:2]) is None

    # extent similarity: identical boxes seen from rotated viewpoints score 1.0
    assert extent_similarity([2.0, 0.5, 1.0], [0.5, 1.0, 2.0]) == 1.0
    assert extent_similarity([2.0, 0.5, 1.0], [2.0, 0.5, 1.0]) == 1.0
    s = extent_similarity([2.0, 0.5, 1.0], [1.0, 0.25, 0.5])  # half-size box
    assert abs(s - 0.5) < 1e-9, s
    assert extent_similarity(None, [1, 1, 1]) == 0.0
    assert extent_similarity([0.0, 1, 1], [1, 1, 1]) == 0.0
    print("pca_box self-check OK:", {k: np.round(v, 3) if isinstance(v, np.ndarray) else round(v, 3) for k, v in box.items()})
