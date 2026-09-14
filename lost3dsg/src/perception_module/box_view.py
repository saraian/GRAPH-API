#!/usr/bin/env python3
"""Box geometry and the one visibility rule shared by every place a 3D box is drawn
into an image (the ROS /image_with_bb overlay and the simulator window).

ROS-free on purpose: the habitat host process imports it without a ROS install.

Visibility: a box is tested at its 8 corners and its centre against the depth image
of the frame. A point is visible when it lands inside the image and is not behind
the surface the sensor sees at that pixel (tolerance: 10 cm + 5 % of range, for
sensor noise and for the box standing slightly proud of the object). No visible
point -> the box is out of view or occluded and is not drawn; fewer than 5 of 9 ->
partially visible (callers draw it thin).
# ponytail: corner depth test, not a mesh occlusion query — right for an overlay;
# use the simulator's semantic sensor (instance ids) when visibility feeds evaluation.
"""
import numpy as np

# corner index = 4*ix + 2*iy + iz (x-major), so these 12 edges hold for AABB and OBB alike
BOX_EDGES = [(0, 1), (2, 3), (4, 5), (6, 7), (0, 2), (1, 3), (4, 6), (5, 7), (0, 4), (1, 5), (2, 6), (3, 7)]
_AABB_KEYS = {"x_min", "x_max", "y_min", "y_max", "z_min", "z_max"}


def _normalise_axis_yaw(yaw):
    """Return an equivalent box-axis angle in [-pi/2, pi/2)."""
    return float((float(yaw) + np.pi / 2.0) % np.pi - np.pi / 2.0)


def aabb_from_points(points_xyz):
    """Return the exact axis-aligned bounds of finite 3D points.

    The perception pipeline has already removed invalid depth and statistical outliers before
    calling this function. A bounding box is an enclosure, so trimming another 5 percent here
    is incorrect: it makes the visual box miss valid points and makes the merge geometry
    describe a smaller object than the one that was measured.
    """
    pts = np.asarray(points_xyz, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 3 or pts.shape[0] == 0:
        return None
    if not np.all(np.isfinite(pts)):
        return None
    lo = np.min(pts, axis=0)
    hi = np.max(pts, axis=0)
    if np.any((hi - lo) <= 1e-6):
        return None
    return {
        "x_min": float(lo[0]), "x_max": float(hi[0]),
        "y_min": float(lo[1]), "y_max": float(hi[1]),
        "z_min": float(lo[2]), "z_max": float(hi[2]),
    }


def pca_oriented_box(points_xyz, min_anisotropy=1.2):
    """Fit a yaw-only, enclosing PCA box to finite 3D points.

    This is the one orientation implementation shared by perception, persistence and the
    renderers. PCA is performed on the complete filtered XY point set, not on a top-surface
    subset and not by choosing a discretised rectangle angle. The returned extents are exact
    min/max projections in the PCA frame, so every input point is inside the box.

    A nearly isotropic XY cloud has no stable principal axis. Returning ``None`` for that case
    is intentional; callers retain the exact AABB instead of publishing a made-up orientation.
    """
    pts = np.asarray(points_xyz, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 3 or pts.shape[0] < 3:
        return None
    if not np.all(np.isfinite(pts)):
        return None

    xy = pts[:, :2]
    centred = xy - np.mean(xy, axis=0)
    covariance = np.cov(centred, rowvar=False)
    if covariance.shape != (2, 2) or not np.all(np.isfinite(covariance)):
        return None
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)
    minor_value = max(float(eigenvalues[order[0]]), 0.0)
    major_value = max(float(eigenvalues[order[1]]), 0.0)
    if major_value <= 1e-12:
        return None
    if major_value / max(minor_value, 1e-12) < float(min_anisotropy) ** 2:
        return None

    major = eigenvectors[:, order[1]]
    yaw = _normalise_axis_yaw(np.arctan2(major[1], major[0]))

    def projections(theta):
        c, s = np.cos(theta), np.sin(theta)
        u = xy[:, 0] * c + xy[:, 1] * s
        v = -xy[:, 0] * s + xy[:, 1] * c
        return u, v, c, s

    u, v, c, s = projections(yaw)
    lo_u, hi_u = float(np.min(u)), float(np.max(u))
    lo_v, hi_v = float(np.min(v)), float(np.max(v))
    lo_z, hi_z = float(np.min(pts[:, 2])), float(np.max(pts[:, 2]))

    # The major eigenvector should already be the long axis. Keep this guard for numerical
    # ties and for callers using unusual point sets, while preserving the same exact points.
    if hi_v - lo_v > hi_u - lo_u:
        yaw = _normalise_axis_yaw(yaw + np.pi / 2.0)
        u, v, c, s = projections(yaw)
        lo_u, hi_u = float(np.min(u)), float(np.max(u))
        lo_v, hi_v = float(np.min(v)), float(np.max(v))

    du, dv, dz = hi_u - lo_u, hi_v - lo_v, hi_z - lo_z
    if min(du, dv, dz) <= 1e-6 or du / max(dv, 1e-12) < float(min_anisotropy):
        return None
    uc, vc = (lo_u + hi_u) / 2.0, (lo_v + hi_v) / 2.0
    return {
        "yaw": float(yaw),
        "oriented_center": [float(uc * c - vc * s), float(uc * s + vc * c),
                             float((lo_z + hi_z) / 2.0)],
        "oriented_extents": [float(du), float(dv), float(dz)],
    }


def enclosing_box_from_points(points_xyz, min_anisotropy=1.2, include_orientation=True):
    """Build one exact AABB and, when stable, one exact PCA-oriented box."""
    box = aabb_from_points(points_xyz)
    if box is None:
        return None
    if include_orientation:
        oriented = pca_oriented_box(points_xyz, min_anisotropy=min_anisotropy)
        if oriented is not None:
            box.update(oriented)
            box["has_orientation"] = True
            return box
    box["has_orientation"] = False
    return box


def box_corners_map(bbox):
    """(8 corners in the map frame, oriented?) — the PCA-oriented box when the dict
    carries one, else the AABB; ([], False) when the dict has neither."""
    b = bbox or {}
    if b.get("oriented_extents") and b.get("oriented_center") is not None:
        ex, ey, ez = (float(v) / 2.0 for v in b["oriented_extents"])
        c, s = np.cos(float(b.get("yaw", 0.0))), np.sin(float(b.get("yaw", 0.0)))
        centre = np.asarray(b["oriented_center"], dtype=np.float64)
        return [centre + np.array([c * sx - s * sy, s * sx + c * sy, sz])
                for sx in (-ex, ex) for sy in (-ey, ey) for sz in (-ez, ez)], True
    if _AABB_KEYS <= set(b):
        return [np.array([x, y, z], dtype=np.float64) for x in (b["x_min"], b["x_max"])
                for y in (b["y_min"], b["y_max"]) for z in (b["z_min"], b["z_max"])], False
    return [], False


def project_visible(cam_xyz, depth, fx, fy, cx, cy, w, h, tol_abs=0.10, tol_rel=0.05):
    """cam_xyz: (9,3) — the 8 corners then the centre — in the OPTICAL frame
    (x right, y down, z forward). Returns (corner pixels, number of visible points),
    or (None, 0) when any point is behind the camera."""
    pts = np.asarray(cam_xyz, dtype=np.float64)
    if np.any(pts[:, 2] <= 0.05):
        return None, 0
    us = (fx * pts[:, 0] / pts[:, 2] + cx).astype(int)
    vs = (fy * pts[:, 1] / pts[:, 2] + cy).astype(int)
    n_vis = 0
    for u, v, z in zip(us, vs, pts[:, 2]):
        if 0 <= u < w and 0 <= v < h:
            seen = float(depth[v, u]) if depth is not None else 0.0
            n_vis += (not np.isfinite(seen)) or seen <= 0 or z <= seen + tol_abs + tol_rel * z
    return list(zip(us[:8].tolist(), vs[:8].tolist())), int(n_vis)


if __name__ == "__main__":
    aabb = {"x_min": 1.5, "x_max": 2.5, "y_min": -0.5, "y_max": 0.5, "z_min": -0.5, "z_max": 0.5}
    corners, oriented = box_corners_map(aabb)
    assert len(corners) == 8 and not oriented and box_corners_map({}) == ([], False)
    obb = dict(aabb, yaw=0.0, oriented_extents=[1, 1, 1], oriented_center=[2, 0, 0])
    assert np.allclose(sorted(map(tuple, box_corners_map(obb)[0])), sorted(map(tuple, corners)))
    # optical frame: the AABB 2 m ahead (x forward -> z forward)
    cam = [np.array([-c[1], -c[2], c[0]]) for c in corners]
    cam.append(np.mean(cam, axis=0))
    w, h, f = 640, 480, 320.0
    far, wall = np.full((h, w), 10.0), np.full((h, w), 0.8)
    assert project_visible(cam, far, f, f, 320, 240, w, h)[1] == 9
    assert project_visible(cam, wall, f, f, 320, 240, w, h)[1] == 0
    half = far.copy(); half[:, : 2 * w // 3] = 0.8  # noqa: E702
    assert 0 < project_visible(cam, half, f, f, 320, 240, w, h)[1] < 9
    assert project_visible(cam, None, f, f, 320, 240, w, h)[1] == 9
    assert project_visible([[0, 0, -1.0]] * 9, far, f, f, 320, 240, w, h) == (None, 0)
    print("box_view OK")
