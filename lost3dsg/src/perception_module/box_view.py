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
