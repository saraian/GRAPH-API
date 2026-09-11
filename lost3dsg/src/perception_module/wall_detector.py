#!/usr/bin/env python3
"""Wall detection from the depth image.

WHY THIS EXISTS, AND WHY room_manager's WALLS ARE NOT A SUBSTITUTE
------------------------------------------------------------------
`room_manager.py:924` already writes 53-229 entries per bundle under `rooms[].walls`, from
`_polygon_edges(best.polygon)` -- the outer contour of the room mask taken from the 2D
occupancy projection. Those are not detections:

    An edge there means "observed free space stopped here", NOT "a surface exists here".

`room_manager.py:337-338` marks unknown cells occupied (`unknown_is_obstacle`, default True at
:107, no yaml override), so the exploration frontier, the sensor's range limit, a sofa and a
wall all generate identical geometry. Interior structure is removed twice: `_fill_room_holes`
(:839) erases it from the mask and the outer-contour filter (:831-833, `hierarchy[i][3] == -1`)
discards what survives. The result carries no height and no thickness.

This node is a DIFFERENT MEASUREMENT, not a better use of the same one. Only pixels with a
finite in-range depth return contribute, so unobserved space produces nothing; and each segment
carries the height band it was observed over, which a polygon edge cannot express.

Owner's ruling, 2026-08-31, presented as an interactive choice by the orchestrator against the
alternatives of deleting the file or filing it as dead: keep wall detection as a capability and
point it at the depth image instead of a laser this stack does not have.

WHAT THE PREVIOUS VERSION DID, AND ALL FOUR REASONS IT NEVER DELIVERED A WALL (GA-29)
-------------------------------------------------------------------------------------
1. Never launched. `live_stack_container.sh` starts habitat_feed_node, object_manager_6 and
   perception_2. No launch file, script or yaml names this one.
2. It subscribed to `/scan_raw` as a LaserScan. Nothing publishes `/scan_raw`; the feed
   publishes no LaserScan at all. The stack is RGB-D.
3. It was written for `object_manager_3.py`, superseded and deleted (GA-27).
4. It emitted `[x1, y1, x2, y2]` while `object_manager_6.walls_callback` reads
   `w["start"]["x"]`. A list indexed by a string raises TypeError, and the callback wraps its
   body in `except Exception: print(...)` -- so it would have failed silently on the first
   message even had 1-3 been fixed.

This version publishes the schema the consumer actually reads.
"""
import json
import math
import sys
import threading
import time
import traceback
from collections import deque

import numpy as np

# ROS and cv_utils are imported inside the node rather than at module level, so the geometry
# below stays importable — and therefore testable — without a ROS environment. `--selfcheck`
# runs on the host with only numpy.

# A wall is tall. Points below the lower bound are floor and clutter; above the upper bound is
# ceiling. The band is in METRES ABOVE THE MAP FRAME'S z=0, which rtabmap puts at the floor.
HEIGHT_BAND_M = (0.4, 2.0)
# A wall must be observed over at least this much height. A table edge or a sofa back clears the
# lower bound but not this: it is the discriminant a 2D polygon edge cannot make.
MIN_VERTICAL_EXTENT_M = 0.8
DEPTH_RANGE_M = (0.3, 8.0)        # outside this the depth return is not trustworthy
PIXEL_STRIDE = 4                  # subsample; a wall is not a fine structure


def _walls_cfg():
    """Tuning from config.walls, with the previous literals as the fallback. Imported
    lazily so the geometry above stays runnable on a host with no ROS package path."""
    try:
        from config import CFG
        return CFG.get("walls", {}) or {}
    except Exception:
        return {}


_WCFG = _walls_cfg()
MIN_INTERVAL_S = float(_WCFG.get("min_interval_s", 0.5))
HEIGHT_BAND_M = tuple(float(v) for v in _WCFG.get("height_band_m", HEIGHT_BAND_M))
DEPTH_RANGE_M = tuple(float(v) for v in _WCFG.get("depth_range_m", DEPTH_RANGE_M))
PIXEL_STRIDE = int(_WCFG.get("pixel_stride", PIXEL_STRIDE))
MIN_VERTICAL_EXTENT_M = float(_WCFG.get("min_vertical_extent_m", MIN_VERTICAL_EXTENT_M))
LINE_INLIER_TOL_M = float(_WCFG.get("line_inlier_tolerance_m", 0.05))
DEPTH_NOISE_PER_M = float(_WCFG.get("depth_noise_per_m", 0.006))
MIN_INLIERS = int(_WCFG.get("min_inliers", 60))
MIN_INLIERS_FLOOR = int(_WCFG.get("min_inliers_floor", 20))
INLIER_REFERENCE_RANGE_M = float(_WCFG.get("inlier_reference_range_m", 2.0))
MIN_SEGMENT_LEN_M = float(_WCFG.get("min_segment_length_m", 0.5))
MAX_INLIER_RMS_M = float(_WCFG.get("max_inlier_rms_m", 0.05))
MAX_SEGMENTS = int(_WCFG.get("max_segments", 12))
GRID_RESOLUTION_M = float(_WCFG.get("grid_resolution_m", 0.05))
VERTICAL_BANDS = int(_WCFG.get("vertical_bands", 4))
MIN_VERTICAL_BANDS = int(_WCFG.get("min_vertical_bands", 3))
MIN_CELL_POINTS = int(_WCFG.get("min_cell_points", 3))
HOUGH_THRESHOLD = int(_WCFG.get("hough_threshold", 8))
MAX_LINE_GAP_M = float(_WCFG.get("max_line_gap_m", 0.20))
# A wall is a *surface*, not merely a tall collection of points that happens to be
# collinear in the floor projection.  Once a candidate vertical plane has been
# fitted, test its occupancy in its intrinsic (along-wall, height) coordinates.
# This is deliberately coarser than the 5 cm Hough raster: it measures surface
# support and tolerates individual depth holes, without turning a few shelf/leg
# returns into a wall.
SUPPORT_CELL_M = float(_WCFG.get("support_cell_m", 0.20))
MIN_ALONG_COVERAGE = float(_WCFG.get("min_along_coverage", 0.60))
MIN_HEIGHT_COVERAGE = float(_WCFG.get("min_height_coverage", 0.70))
MIN_SURFACE_COVERAGE = float(_WCFG.get("min_surface_coverage", 0.30))
# Parameters of the landmark association in plane space.  They are intentionally
# tighter than room_manager's multi-frame association: this stage only removes
# duplicate Hough hypotheses from one depth frame.
PLANE_CLUSTER_ANGLE_DEG = float(_WCFG.get("plane_cluster_angle_deg", 4.0))
PLANE_CLUSTER_DISTANCE_M = float(_WCFG.get("plane_cluster_distance_m", 0.08))
PLANE_CLUSTER_MIN_OVERLAP = float(_WCFG.get("plane_cluster_min_overlap", 0.35))

if (len(HEIGHT_BAND_M) != 2 or HEIGHT_BAND_M[0] >= HEIGHT_BAND_M[1] or
        len(DEPTH_RANGE_M) != 2 or DEPTH_RANGE_M[0] >= DEPTH_RANGE_M[1]):
    raise ValueError("walls height_band_m/depth_range_m must be increasing pairs")
if PIXEL_STRIDE < 1:
    raise ValueError("walls.pixel_stride must be >= 1")
if not 1 <= VERTICAL_BANDS <= 16 or not 1 <= MIN_VERTICAL_BANDS <= VERTICAL_BANDS:
    raise ValueError("walls vertical bands must satisfy 1 <= minimum <= total <= 16")


def _adaptive_min_inliers(range_m):
    """Keep the support requirement metric as angular sampling becomes sparser."""
    if not np.isfinite(range_m) or range_m <= INLIER_REFERENCE_RANGE_M:
        return MIN_INLIERS
    scaled = MIN_INLIERS * (INLIER_REFERENCE_RANGE_M / range_m) ** 2
    return max(MIN_INLIERS_FLOOR, min(MIN_INLIERS, int(round(scaled))))


def _voxel_downsample(xy, z, ranges, voxel_m):
    """Average points in 3-D metric voxels, removing repeated temporal samples."""
    if len(xy) == 0 or voxel_m <= 0.0:
        return xy, z, ranges
    xyz = np.column_stack((xy, z)).astype(np.float64, copy=False)
    keys = np.floor(xyz / voxel_m).astype(np.int64)
    _, inverse = np.unique(keys, axis=0, return_inverse=True)
    counts = np.bincount(inverse).astype(np.float64)
    out = np.column_stack([
        np.bincount(inverse, weights=xyz[:, axis]) / counts for axis in range(3)
    ])
    out_ranges = np.bincount(inverse, weights=ranges) / counts
    return out[:, :2].astype(np.float32), out[:, 2].astype(np.float32), out_ranges.astype(np.float32)


class _TemporalPointBuffer:
    """Bounded sliding window of map-frame wall candidates from recent depth frames."""
    def __init__(self, window_s, max_frames, voxel_m, floor_reset_m):
        self.window_s = max(0.0, float(window_s))
        self.max_frames = max(1, int(max_frames))
        self.voxel_m = max(0.0, float(voxel_m))
        self.floor_reset_m = max(0.0, float(floor_reset_m))
        self.frames = deque()
        self.floor_z = None

    def add(self, stamp_s, xy, z, ranges, floor_z, camera_xy=None):
        if (self.floor_z is not None and self.floor_reset_m > 0.0 and
                abs(float(floor_z) - self.floor_z) > self.floor_reset_m):
            self.frames.clear()
        self.floor_z = float(floor_z)
        camera_xy = (None if camera_xy is None else
                     np.asarray(camera_xy, dtype=np.float64).reshape(2))
        self.frames.append((float(stamp_s), np.asarray(xy), np.asarray(z),
                            np.asarray(ranges), camera_xy))
        while len(self.frames) > self.max_frames:
            self.frames.popleft()
        if self.window_s > 0.0:
            cutoff = float(stamp_s) - self.window_s
            while self.frames and self.frames[0][0] < cutoff:
                self.frames.popleft()
        all_xy = np.concatenate([frame[1] for frame in self.frames], axis=0)
        all_z = np.concatenate([frame[2] for frame in self.frames], axis=0)
        all_ranges = np.concatenate([frame[3] for frame in self.frames], axis=0)
        xy_ds, z_ds, range_ds = _voxel_downsample(all_xy, all_z, all_ranges, self.voxel_m)
        return xy_ds, z_ds, range_ds, len(self.frames)

    def camera_positions(self):
        return [frame[4].copy() for frame in self.frames if frame[4] is not None]


def _distinct_viewpoint_count(camera_positions, separation_m=0.15):
    kept = []
    for point in camera_positions:
        point = np.asarray(point, dtype=np.float64)
        if not any(float(np.linalg.norm(point - old)) < separation_m for old in kept):
            kept.append(point)
    return len(kept)


def _organized_surface_masks(depth, fx, fy, cx, cy, rotation,
                             max_depth_jump_ratio=0.08,
                             vertical_normal_tolerance_deg=20.0):
    """Classify locally connected depth patches by their map-frame normal.

    Returns masks for vertical surfaces (wall candidates) and horizontal surfaces
    (floor-datum candidates). Invalid pixels and depth discontinuities are excluded.
    """
    depth = np.asarray(depth, dtype=np.float64)
    h, w = depth.shape
    ys, xs = np.indices((h, w), dtype=np.float64)
    xs *= PIXEL_STRIDE
    ys *= PIXEL_STRIDE
    with np.errstate(invalid="ignore", over="ignore"):
        xyz = np.stack(((xs - cx) * depth / fx,
                        (ys - cy) * depth / fy, depth), axis=-1)
        dx = np.zeros_like(xyz)
        dy = np.zeros_like(xyz)
        dx[:, 1:-1] = xyz[:, 2:] - xyz[:, :-2]
        dy[1:-1, :] = xyz[2:, :] - xyz[:-2, :]
        normals = np.cross(dx, dy)
        norm = np.linalg.norm(normals, axis=2)
    valid = np.isfinite(depth) & np.isfinite(norm) & (norm > 1e-10)
    jump = np.maximum(0.02, max_depth_jump_ratio * depth)
    valid[:, 1:-1] &= (np.abs(depth[:, 2:] - depth[:, :-2]) <= 2.0 * jump[:, 1:-1])
    valid[1:-1, :] &= (np.abs(depth[2:, :] - depth[:-2, :]) <= 2.0 * jump[1:-1, :])
    unit = np.zeros_like(normals)
    unit[valid] = normals[valid] / norm[valid, None]
    map_normals = unit @ np.asarray(rotation, dtype=np.float64).T
    nz = np.abs(map_normals[:, :, 2])
    tolerance = math.radians(vertical_normal_tolerance_deg)
    return valid & (nz <= math.sin(tolerance)), valid & (nz >= math.cos(tolerance))


def _estimate_floor_z(points_map, floor_mask, camera_z, min_points=20):
    """Robust local floor datum from pixels whose normals are near vertical."""
    z = np.asarray(points_map)[np.asarray(floor_mask, dtype=bool), 2]
    z = z[np.isfinite(z) & (z < float(camera_z) - 0.20) &
          (z > float(camera_z) - 2.5)]
    if z.size < int(min_points):
        return None, int(z.size)
    # Choose the densest 5 cm height bin, then trim within that physical surface.
    bins = np.floor(z / 0.05).astype(np.int64)
    values, counts = np.unique(bins, return_counts=True)
    mode = values[int(np.argmax(counts))]
    support = z[np.abs(z - (mode + 0.5) * 0.05) <= 0.075]
    return float(np.median(support)), int(support.size)


def _refine_vertical_wall(pts, zs, min_inliers=MIN_INLIERS):
    """Fit and validate one gravity-aligned planar wall patch.

    In a gravity-aligned ``map`` frame a vertical plane has normal ``(nx, ny,
    0)``.  TLS estimates that normal from its horizontal trace, while the
    occupancy test below verifies that the inliers cover a 2-D patch in the
    plane's own coordinates (distance along the trace and height).  This is a
    constrained plane fit, rather than the old "long top-down line" heuristic.
    """
    if len(pts) < min_inliers:
        return None
    centre = pts.mean(axis=0)
    _, _, vh = np.linalg.svd(pts - centre, full_matrices=False)
    direction = vh[0]
    t = (pts - centre) @ direction
    # Robust endpoints make isolated depth outliers unable to extend a wall.
    lo, hi = np.percentile(t, [2.0, 98.0])
    length = float(hi - lo)
    zlo, zhi = np.percentile(zs, [5.0, 95.0])
    if length < MIN_SEGMENT_LEN_M or zhi - zlo < MIN_VERTICAL_EXTENT_M:
        return None

    # Surface support in the fitted vertical plane.  A real wall populates both
    # axes; a sparse vertical edge, a ladder, or points from unrelated objects
    # aligned in bird's-eye view does not.
    cell = max(SUPPORT_CELL_M, 1e-6)
    nt = max(1, int(math.ceil(length / cell)))
    nz = max(1, int(math.ceil((zhi - zlo) / cell)))
    ti = np.clip(((t - lo) / cell).astype(np.int32), 0, nt - 1)
    zi = np.clip(((zs - zlo) / cell).astype(np.int32), 0, nz - 1)
    occupied = np.zeros((nz, nt), dtype=bool)
    occupied[zi, ti] = True
    along_coverage = float(occupied.any(axis=0).mean())
    height_coverage = float(occupied.any(axis=1).mean())
    surface_coverage = float(occupied.mean())
    if (along_coverage < MIN_ALONG_COVERAGE or
            height_coverage < MIN_HEIGHT_COVERAGE or
            surface_coverage < MIN_SURFACE_COVERAGE):
        return None

    normal = np.array([-direction[1], direction[0]])
    rms = float(np.sqrt(np.mean(((pts - centre) @ normal) ** 2)))
    if rms > MAX_INLIER_RMS_M:
        return None
    return (centre + direction * lo, centre + direction * hi,
            float(zlo), float(zhi), int(len(pts)), rms)


def _same_plane_landmark(first, second):
    """Whether two finite segments are duplicate observations of one plane.

    A canonical wall landmark is the vertical plane ``n·(x,y)=rho`` plus a
    finite interval along its tangent.  Comparing angle and signed normal
    distance alone would collapse distinct parallel walls; requiring interval
    overlap prevents that and also preserves doorway-separated wall pieces.
    """
    a0, a1 = first[:2]
    b0, b1 = second[:2]
    avec = a1 - a0
    alen = float(np.linalg.norm(avec))
    bvec = b1 - b0
    blen = float(np.linalg.norm(bvec))
    if alen < 1e-9 or blen < 1e-9:
        return False
    adir, bdir = avec / alen, bvec / blen
    if abs(float(adir @ bdir)) < math.cos(math.radians(PLANE_CLUSTER_ANGLE_DEG)):
        return False
    normal = np.array([-adir[1], adir[0]])
    if abs(float((((b0 + b1) - (a0 + a1)) * 0.5) @ normal)) > PLANE_CLUSTER_DISTANCE_M:
        return False
    bproj = (np.stack((b0, b1)) - a0) @ adir
    overlap = min(alen, float(bproj.max())) - max(0.0, float(bproj.min()))
    return overlap / min(alen, blen) >= PLANE_CLUSTER_MIN_OVERLAP


def _consolidate_plane_landmarks(xy, z, candidates):
    """Cluster overlapping Hough candidates in (theta, rho, interval) space.

    Each cluster is re-estimated from the union of its source points, so the
    published segment is a single TLS plane estimate, not one selected raster
    stroke.  ``candidates`` contains ``(segment, source_indices)`` pairs.
    """
    clusters = []
    for segment, source_idx in sorted(
            candidates, key=lambda item: (-item[0][4], -float(np.linalg.norm(item[0][1] - item[0][0])))):
        for cluster in clusters:
            if _same_plane_landmark(cluster["segment"], segment):
                cluster["indices"].append(source_idx)
                merged_idx = np.unique(np.concatenate(cluster["indices"]))
                refined = _refine_vertical_wall(
                    xy[merged_idx], z[merged_idx], min_inliers=MIN_INLIERS_FLOOR)
                # The original cluster remains valid if a partial overlap has
                # too little 2-D support after de-duplication.
                if refined is not None:
                    cluster["segment"] = refined
                break
        else:
            clusters.append({"segment": segment, "indices": [source_idx]})
    return [cluster["segment"] for cluster in clusters[:MAX_SEGMENTS]]


def _fit_segments_grid_hough(xy, z, ranges=None):
    """Extract walls from a metric 2.5D grid, then refine them against the source points.

    Pixel density affects cell counts but not the size of the Hough problem. Requiring support
    in several vertical bands rejects low furniture before line extraction. Hough supplies only
    candidates; PCA/TLS and robust percentiles produce the published geometry.
    """
    if len(xy) < MIN_INLIERS_FLOOR:
        return []
    import cv2

    xy = np.asarray(xy, dtype=np.float32)
    z = np.asarray(z, dtype=np.float32)
    ranges = (np.asarray(ranges, dtype=np.float32) if ranges is not None
              else np.full(len(xy), INLIER_REFERENCE_RANGE_M, dtype=np.float32))
    if ranges.shape != (len(xy),):
        raise ValueError("ranges must contain one value per point")
    origin = np.floor(xy.min(axis=0) / GRID_RESOLUTION_M) * GRID_RESOLUTION_M
    ij = np.floor((xy - origin) / GRID_RESOLUTION_M).astype(np.int32)
    width, height = int(ij[:, 0].max()) + 1, int(ij[:, 1].max()) + 1
    # A corrupt transform/depth value must not allocate an arbitrarily large raster.
    if width <= 0 or height <= 0 or width * height > 4_000_000:
        return []

    flat = ij[:, 1] * width + ij[:, 0]
    counts = np.bincount(flat, minlength=width * height)
    band_h = (HEIGHT_BAND_M[1] - HEIGHT_BAND_M[0]) / max(1, VERTICAL_BANDS)
    zb = np.clip(((z - HEIGHT_BAND_M[0]) / band_h).astype(np.int32),
                 0, VERTICAL_BANDS - 1)
    masks = np.zeros(width * height, dtype=np.uint16)
    np.bitwise_or.at(masks, flat, np.left_shift(np.uint16(1), zb.astype(np.uint16)))
    band_counts = np.zeros_like(masks, dtype=np.uint8)
    for bit in range(VERTICAL_BANDS):
        band_counts += ((masks >> bit) & 1).astype(np.uint8)
    structural = ((counts >= MIN_CELL_POINTS) &
                  (band_counts >= MIN_VERTICAL_BANDS)).reshape(height, width)
    if structural.sum() < 2:
        return []

    image = structural.astype(np.uint8) * 255
    # Close only one-cell sampling holes. Door-sized gaps remain gaps.
    image = cv2.morphologyEx(image, cv2.MORPH_CLOSE,
                             np.ones((3, 3), np.uint8), iterations=1)
    lines = cv2.HoughLinesP(
        image, 1, np.pi / 180.0, threshold=HOUGH_THRESHOLD,
        minLineLength=max(2, round(MIN_SEGMENT_LEN_M / GRID_RESOLUTION_M)),
        maxLineGap=max(0, round(MAX_LINE_GAP_M / GRID_RESOLUTION_M)))
    if lines is None:
        return []

    # Suppress duplicate raster strokes before touching the (larger) source point set.
    raw_lines = []
    for x0, y0, x1, y1 in sorted(lines[:, 0],
                                  key=lambda q: -math.hypot(q[2] - q[0], q[3] - q[1])):
        a = origin + GRID_RESOLUTION_M * (np.array([x0, y0], np.float32) + 0.5)
        b = origin + GRID_RESOLUTION_M * (np.array([x1, y1], np.float32) + 0.5)
        direction = b - a
        line_len = float(np.linalg.norm(direction))
        if line_len < 1e-6:
            continue
        direction /= line_len
        normal = np.array([-direction[1], direction[0]], dtype=np.float32)
        midpoint = (a + b) * 0.5
        if any(abs(float(direction @ old_dir)) > math.cos(math.radians(5.0)) and
               abs(float((midpoint - old_mid) @ old_normal)) < GRID_RESOLUTION_M * 1.5 and
               abs(float((midpoint - old_mid) @ old_dir)) <=
               (line_len + float(np.linalg.norm(old_b - old_a))) * 0.5 + MAX_LINE_GAP_M
               for old_mid, old_dir, old_normal, old_a, old_b in raw_lines):
            continue
        raw_lines.append((midpoint, direction, normal, a, b))
        if len(raw_lines) >= MAX_SEGMENTS * 3:
            break

    candidates = []
    for midpoint, direction, normal, a, b in raw_lines:
        line_len = float(np.linalg.norm(b - a))
        along = (xy - a) @ direction
        distance = np.abs((xy - a) @ normal)
        tolerances = np.maximum(
            GRID_RESOLUTION_M, LINE_INLIER_TOL_M + DEPTH_NOISE_PER_M * ranges)
        near_idx = np.flatnonzero(distance <= tolerances)
        candidate_range = float(np.median(ranges[near_idx])) if len(near_idx) else float("inf")
        required_inliers = _adaptive_min_inliers(candidate_range)
        if len(near_idx) < required_inliers:
            continue
        # Extend the Hough seed over its connected support, but never bridge a door/gap.
        order = np.argsort(along[near_idx])
        ordered_idx = near_idx[order]
        ordered_t = along[ordered_idx]
        cuts = np.flatnonzero(np.diff(ordered_t) > MAX_LINE_GAP_M) + 1
        groups = np.split(ordered_idx, cuts)
        seed_t = line_len * 0.5
        group = min(groups, key=lambda g: abs(float(np.median(along[g])) - seed_t))
        if len(group) < required_inliers:
            continue
        pts, zs = xy[group], z[group]

        refined = _refine_vertical_wall(pts, zs, min_inliers=required_inliers)
        if refined is not None:
            candidates.append((refined, group))

    return _consolidate_plane_landmarks(xy, z, candidates)


def _node_class():
    """Build the node class once ROS is importable. Kept out of module scope so the geometry
    above stays testable on a host with no ROS: `python3 wall_detector.py --selfcheck`."""
    import tf2_ros
    from cv_bridge import CvBridge
    from cv_utils import _apply_transform, _get_R_and_T, _pixels_to_points_habitat_camera
    from geometry_msgs.msg import PoseWithCovarianceStamped
    from rclpy.node import Node
    from rclpy.qos import qos_profile_sensor_data
    from sensor_msgs.msg import CameraInfo, Image
    from std_msgs.msg import String
    from tf2_ros import TransformException
    from visualization_msgs.msg import Marker, MarkerArray

    class WallDetector(Node):
        def __init__(self):
            super().__init__("wall_detector")
            self.bridge = CvBridge()
            self.camera_info = None
            self._last_fit = 0.0
            self._pending = None
            self._deferred_depth = None
            self._floor_z = None
            self._floor_seen_at = None
            self._pose_std_m = None
            self._temporal = _TemporalPointBuffer(
                _WCFG.get("temporal_window_s", 2.5),
                _WCFG.get("temporal_max_frames", 8),
                _WCFG.get("temporal_voxel_m", 0.035),
                _WCFG.get("floor_reset_m", 0.60))
            self._wake = threading.Event()
            self._stop = threading.Event()
            self._worker = threading.Thread(target=self._fit_worker, daemon=True)
            self._stats = {
                "depth_frames": 0, "tf_failures": 0, "jobs_queued": 0,
                "fits_completed": 0, "valid_depth_points": 0,
                "height_band_points": 0, "walls_last_fit": 0,
                "normal_wall_points": 0, "temporal_frames": 0,
                "temporal_points": 0, "floor_z": None, "floor_source": "none",
                "wall_confidence_mean": None, "wall_confidence_max": None,
                "last_fit_ms": None, "last_reason": "waiting_for_camera_info",
            }
            self._worker.start()
            self.tf_buffer = tf2_ros.Buffer()
            self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
            self.create_subscription(CameraInfo, "/camera/camera_info", self._info_cb,
                                     qos_profile_sensor_data)
            self.create_subscription(Image, "/camera/depth", self._depth_cb,
                                     qos_profile_sensor_data)
            self.create_subscription(
                PoseWithCovarianceStamped,
                str(_WCFG.get("pose_covariance_topic", "/rtabmap/localization_pose")),
                self._pose_cb, qos_profile_sensor_data)
            # Topic and String/JSON envelope unchanged: object_manager_6 already subscribes and
            # its callback is wired. Only the schema inside is corrected -- see reason 4 above.
            self.pub = self.create_publisher(String, "/detected_wall_segments", 10)
            self.status_pub = self.create_publisher(String, "/wall_detector/status", 10)
            self.marker_pub = self.create_publisher(
                MarkerArray, "/detected_wall_markers", 10)
            self.create_timer(1.0, self._publish_status)
            # Depth and its same-stamp TF can be delivered in either executor order. Retry the
            # exact stamped lookup after TF has had a chance to enter the buffer.
            self.create_timer(0.05, self._retry_deferred_depth)
            self.get_logger().info(
                "wall_detector: depth-derived. Waiting for /camera/depth and /camera/camera_info.")

        def _info_cb(self, msg):
            self.camera_info = msg
            if self._stats["last_reason"] == "waiting_for_camera_info":
                self._stats["last_reason"] = "waiting_for_depth"

        def _pose_cb(self, msg):
            covariance = np.asarray(msg.pose.covariance, dtype=np.float64).reshape(6, 6)
            xy = covariance[:2, :2]
            eigenvalues = np.linalg.eigvalsh(xy)
            self._pose_std_m = math.sqrt(max(0.0, float(eigenvalues[-1])))
            self._stats["pose_std_m"] = round(self._pose_std_m, 4)

        def _publish_status(self):
            status = dict(self._stats)
            status["camera_info_received"] = self.camera_info is not None
            status["fit_pending"] = self._pending is not None
            status["depth_waiting_for_tf"] = self._deferred_depth is not None
            status["method"] = "temporal_normals_grid_hough_tls"
            self.status_pub.publish(String(data=json.dumps(status)))

        def _depth_cb(self, msg):
            self._stats["depth_frames"] += 1
            if self.camera_info is None:
                self._stats["last_reason"] = "waiting_for_camera_info"
                return
            if self._deferred_depth is not None:
                return             # preserve the exact frame whose TF we are waiting for
            # Rate limit BEFORE any work. Dropping the frame here costs nothing; dropping it
            # after the unprojection has already spent the CPU this guard exists to save.
            now = time.monotonic()
            if now - self._last_fit < MIN_INTERVAL_S:
                return
            if self._pending is not None:
                self._stats["last_reason"] = "fit_pending"
                return          # a fit is already queued; this frame is redundant
            self._last_fit = now
            try:
                t = self.tf_buffer.lookup_transform("map", msg.header.frame_id, msg.header.stamp)
            except TransformException as exc:
                self._stats["tf_failures"] += 1
                self._stats["last_reason"] = "waiting_for_timestamped_transform"
                self._deferred_depth = (msg, now, str(exc))
                return
            self._process_depth(msg, t)

        def _retry_deferred_depth(self):
            deferred = self._deferred_depth
            if deferred is None or self._pending is not None:
                return
            msg, queued_at, first_error = deferred
            try:
                transform = self.tf_buffer.lookup_transform(
                    "map", msg.header.frame_id, msg.header.stamp)
            except TransformException:
                if time.monotonic() - queued_at >= 1.0:
                    self._deferred_depth = None
                    self._stats["last_reason"] = "timestamped_transform_timeout"
                    self.get_logger().warning(
                        f"no map<-{msg.header.frame_id} at stamp after 1 s: {first_error}")
                return
            self._deferred_depth = None
            self._stats["last_reason"] = "timestamped_transform_recovered"
            self._process_depth(msg, transform)

        def _process_depth(self, msg, t):
            depth = np.asarray(self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough"),
                               dtype=np.float32)
            finite_depth = depth[np.isfinite(depth)]
            if finite_depth.size == 0:
                self._stats["last_reason"] = "no_finite_depth"
                return
            if float(finite_depth.max()) > 100.0:   # millimetres, as some encodings publish
                depth = depth / 1000.0
            depth = depth[::PIXEL_STRIDE, ::PIXEL_STRIDE]

            ys, xs = np.nonzero(np.isfinite(depth) &
                                (depth >= DEPTH_RANGE_M[0]) & (depth <= DEPTH_RANGE_M[1]))
            self._stats["valid_depth_points"] = int(len(xs))
            if len(xs) < MIN_INLIERS:
                self._stats["last_reason"] = "too_few_valid_depth_points"
                return
            k = self.camera_info.k
            pts_cam = _pixels_to_points_habitat_camera(
                xs * PIXEL_STRIDE, ys * PIXEL_STRIDE, depth[ys, xs], k[0], k[4], k[2], k[5])
            pts_map = _apply_transform(pts_cam, t)
            rotation, translation = _get_R_and_T(t)

            use_normals = bool(_WCFG.get("use_normal_prefilter", True))
            if use_normals:
                wall_mask, floor_mask = _organized_surface_masks(
                    depth, k[0], k[4], k[2], k[5], rotation,
                    float(_WCFG.get("normal_max_depth_jump_ratio", 0.08)),
                    float(_WCFG.get("vertical_normal_tolerance_deg", 20.0)))
                # Connected image regions reject isolated normals before the metric fit.
                import cv2
                count, labels = cv2.connectedComponents(wall_mask.astype(np.uint8), 8)
                component_min = max(1, int(_WCFG.get("normal_min_component_px", 12)))
                sizes = np.bincount(labels.ravel(), minlength=count)
                keep_labels = np.flatnonzero(sizes >= component_min)
                keep_labels = keep_labels[keep_labels != 0]
                wall_mask = np.isin(labels, keep_labels)
                wall_pixels = wall_mask[ys, xs]
                floor_pixels = floor_mask[ys, xs]
            else:
                wall_pixels = np.ones(len(xs), dtype=bool)
                floor_pixels = np.zeros(len(xs), dtype=bool)

            now = time.monotonic()
            floor_z, floor_points = _estimate_floor_z(
                pts_map, floor_pixels, float(translation[2]),
                int(_WCFG.get("floor_min_points", 20)))
            if floor_z is not None:
                self._floor_z, self._floor_seen_at = floor_z, now
                floor_source = "depth_normals"
            elif (self._floor_z is not None and self._floor_seen_at is not None and
                  now - self._floor_seen_at <= float(_WCFG.get("floor_cache_s", 10.0))):
                floor_z = self._floor_z
                floor_source = "cached_depth_normals"
            else:
                camera_height = float(_WCFG.get("floor_fallback_camera_height_m", 1.5))
                floor_z = float(translation[2]) - camera_height
                floor_source = "configured_camera_height"
            self._stats["floor_z"] = round(float(floor_z), 3)
            self._stats["floor_source"] = floor_source
            self._stats["floor_points"] = floor_points

            relative_z = pts_map[:, 2] - floor_z
            band = (wall_pixels & (relative_z >= HEIGHT_BAND_M[0]) &
                    (relative_z <= HEIGHT_BAND_M[1]))
            band_points = int(band.sum())
            self._stats["height_band_points"] = band_points
            self._stats["normal_wall_points"] = int(wall_pixels.sum())
            if band_points < MIN_INLIERS_FLOOR:
                self._stats["last_reason"] = "too_few_connected_vertical_surface_points"
                return
            # The fit is the only expensive step, and it runs OFF the executor thread.
            # rclpy.spin() is single-threaded, so a 164 ms fit inside this callback also
            # blocks this node's OWN /tf subscription -- the buffer the next lookup_transform
            # above reads. Blocking here makes the detector fail its own TF lookups.
            # numpy releases the GIL inside the einsum, so the worker really does run beside
            # the executor rather than interleaving with it.
            xy, z, ranges, frame_count = self._temporal.add(
                now, pts_map[band][:, :2].copy(), pts_map[band][:, 2].copy(),
                depth[ys[band], xs[band]].copy(), floor_z, translation[:2])
            self._stats["temporal_frames"] = frame_count
            self._stats["temporal_points"] = int(len(xy))
            self._pending = (xy, z, ranges, frame_count, float(floor_z), floor_source,
                             self._temporal.camera_positions(), self._pose_std_m)
            self._stats["jobs_queued"] += 1
            self._stats["last_reason"] = "fit_queued"
            self._wake.set()

        def _fit_worker(self):
            """One fit at a time. A frame arriving mid-fit REPLACES the pending one instead
            of queueing: the newest depth frame is the only one worth fitting, and an
            unbounded queue would turn a CPU shortage into a memory leak and a growing lag."""
            while not self._stop.is_set():
                if not self._wake.wait(timeout=0.2):
                    continue
                self._wake.clear()
                job, self._pending = self._pending, None
                if job is None:
                    continue
                try:
                    fit_start = time.perf_counter()
                    segments = _fit_segments_grid_hough(job[0], job[1], job[2])
                    walls = segments_to_wall_dicts(
                        segments, temporal_frames=job[3], floor_z=job[4],
                        floor_source=job[5], evidence_points=(job[0], job[1], job[2]),
                        camera_positions=job[6], pose_std_m=job[7])
                except Exception:
                    self._stats["last_reason"] = "fit_exception"
                    # A worker thread that dies silently leaves a node that looks healthy and
                    # publishes nothing. Log with the traceback and keep the thread alive.
                    self.get_logger().error(f"wall fit failed: {traceback.format_exc()}")
                    continue
                self._stats["last_fit_ms"] = round(
                    (time.perf_counter() - fit_start) * 1000.0, 2)
                self._stats["fits_completed"] += 1
                self._stats["walls_last_fit"] = len(walls)
                confidences = [float(wall["confidence"]) for wall in walls]
                self._stats["wall_confidence_mean"] = (
                    round(float(np.mean(confidences)), 4) if confidences else None)
                self._stats["wall_confidence_max"] = (
                    round(float(np.max(confidences)), 4) if confidences else None)
                self._stats["last_reason"] = "walls_found" if walls else "fit_completed_no_walls"
                # Empty is a measurement too: without it, consumers display the last wall set
                # forever after the camera moves to a view containing no supported wall.
                self.pub.publish(String(data=json.dumps(walls)))
                self._publish_markers(walls)

        def _publish_markers(self, walls):
            """Publish vertical wall rectangles that RViz can render directly."""
            array = MarkerArray()
            clear = Marker()
            clear.header.frame_id = "map"
            clear.header.stamp = self.get_clock().now().to_msg()
            clear.action = Marker.DELETEALL
            array.markers.append(clear)
            for marker_id, wall in enumerate(walls):
                x0, y0 = wall["start"]["x"], wall["start"]["y"]
                x1, y1 = wall["end"]["x"], wall["end"]["y"]
                z0, z1 = wall["z_min"], wall["z_max"]
                yaw = math.atan2(y1 - y0, x1 - x0)
                marker = Marker()
                marker.header = clear.header
                marker.ns = "detected_walls"
                marker.id = marker_id
                marker.type = Marker.CUBE
                marker.action = Marker.ADD
                marker.pose.position.x = (x0 + x1) * 0.5
                marker.pose.position.y = (y0 + y1) * 0.5
                marker.pose.position.z = (z0 + z1) * 0.5
                marker.pose.orientation.z = math.sin(yaw * 0.5)
                marker.pose.orientation.w = math.cos(yaw * 0.5)
                marker.scale.x = max(0.01, math.hypot(x1 - x0, y1 - y0))
                marker.scale.y = max(0.03, GRID_RESOLUTION_M)
                marker.scale.z = max(0.01, z1 - z0)
                marker.color.r, marker.color.g = 1.0, 0.35
                marker.color.b, marker.color.a = 0.05, 0.55
                array.markers.append(marker)
            self.marker_pub.publish(array)

        def destroy_node(self):
            self._stop.set()
            self._wake.set()
            self._worker.join(timeout=1.0)
            return super().destroy_node()

    return WallDetector


def _wall_confidence(segment, temporal_frames=1, evidence_points=None,
                     camera_positions=None, pose_std_m=None):
    """Calibratable evidence score; no single density-dependent term dominates it."""
    p0, p1, zmin, zmax, n, rms = segment
    length = float(np.linalg.norm(p1 - p0))
    direction = (p1 - p0) / max(length, 1e-9)
    normal = np.array([-direction[1], direction[0]])
    vertical = float(zmax - zmin)
    length_score = min(1.0, length / max(2.0 * MIN_SEGMENT_LEN_M, 1e-6))
    vertical_score = min(1.0, vertical / max(1.5 * MIN_VERTICAL_EXTENT_M, 1e-6))
    rms_score = max(0.0, 1.0 - float(rms) / max(MAX_INLIER_RMS_M, 1e-6))
    point_score = min(1.0, int(n) / max(2.0 * MIN_INLIERS_FLOOR, 1.0))
    distinct_views = _distinct_viewpoint_count(
        camera_positions or [], float(_WCFG.get("viewpoint_separation_m", 0.15)))
    temporal_support = distinct_views if camera_positions else int(temporal_frames)
    temporal_score = min(1.0, temporal_support /
                         max(1, int(_WCFG.get("confidence_target_frames", 4))))
    scores = [(0.15, length_score), (0.15, vertical_score), (0.20, rms_score),
              (0.10, point_score), (0.15, temporal_score)]
    evidence = {
        "length": round(length_score, 4), "vertical": round(vertical_score, 4),
        "planarity": round(rms_score, 4), "point_support": round(point_score, 4),
        "temporal_support": round(temporal_score, 4),
    }
    if evidence_points is not None:
        xy, zs, ranges = (np.asarray(value) for value in evidence_points)
        along = (xy - p0) @ direction
        distance = np.abs((xy - p0) @ normal)
        tolerances = np.maximum(
            GRID_RESOLUTION_M, LINE_INLIER_TOL_M + DEPTH_NOISE_PER_M * ranges)
        selected = ((along >= 0.0) & (along <= length) & (distance <= tolerances) &
                    (zs >= zmin) & (zs <= zmax))
        cell = max(SUPPORT_CELL_M, 1e-6)
        nt = max(1, int(math.ceil(length / cell)))
        nz = max(1, int(math.ceil((zmax - zmin) / cell)))
        occupied = np.zeros((nz, nt), dtype=bool)
        if np.any(selected):
            ti = np.clip((along[selected] / cell).astype(np.int32), 0, nt - 1)
            zi = np.clip(((zs[selected] - zmin) / cell).astype(np.int32), 0, nz - 1)
            occupied[zi, ti] = True
        along_coverage = float(occupied.any(axis=0).mean())
        height_coverage = float(occupied.any(axis=1).mean())
        surface_coverage = float(occupied.mean())
        coverage_score = float(np.mean((along_coverage, height_coverage, surface_coverage)))
        median_range = float(np.median(ranges[selected])) if np.any(selected) else None
        evidence.update({
            "along_coverage": round(along_coverage, 4),
            "height_coverage": round(height_coverage, 4),
            "surface_coverage": round(surface_coverage, 4),
            "median_range_m": round(median_range, 3) if median_range is not None else None,
        })
        scores.append((0.15, coverage_score))
    if camera_positions:
        midpoint = (p0 + p1) * 0.5
        incidence = []
        for camera in camera_positions:
            ray = midpoint - np.asarray(camera, dtype=np.float64)
            ray /= max(float(np.linalg.norm(ray)), 1e-9)
            incidence.append(abs(float(ray @ normal)))
        incidence_score = float(np.mean(incidence))
        evidence["view_incidence"] = round(incidence_score, 4)
        evidence["distinct_viewpoints"] = distinct_views
        scores.append((0.10, incidence_score))
    if pose_std_m is not None and np.isfinite(pose_std_m):
        pose_score = max(0.0, 1.0 - float(pose_std_m) /
                         max(float(_WCFG.get("max_pose_std_m", 0.25)), 1e-6))
        evidence["pose_std_m"] = round(float(pose_std_m), 4)
        evidence["pose_quality"] = round(pose_score, 4)
        scores.append((0.10, pose_score))
    weight = sum(item[0] for item in scores)
    confidence = sum(w * score for w, score in scores) / max(weight, 1e-9)
    return round(float(confidence), 4), evidence


def segments_to_wall_dicts(segments, temporal_frames=1, floor_z=None, floor_source=None,
                           evidence_points=None, camera_positions=None, pose_std_m=None):
    """The schema object_manager_6.walls_callback actually reads: w["start"]["x"] and
    w["end"]["x"]. The previous version emitted a flat [x1, y1, x2, y2], which raises
    TypeError there -- silently, inside that callback's `except Exception: print(...)`.

    z_min/z_max are the field a polygon edge cannot carry, and the reason this is a different
    measurement rather than a better use of the same one.

    SALVAGED FROM walls_rooms.py, deleted 2026-08-31 by owner's ruling: its
    `_polygon_to_segments` emitted exactly this nested schema. So `walls_callback` was written
    against THAT mixin, and the old wall_detector was the odd one out — two producers, one
    schema each, and the consumer matched the one that was never wired. This function converged
    on the same shape independently, by reading the consumer; the dead file confirms it."""
    walls = []
    for segment in segments:
        p0, p1, zmin, zmax, n, rms = segment
        direction = p1 - p0
        direction /= max(float(np.linalg.norm(direction)), 1e-9)
        normal = np.array([-direction[1], direction[0]])
        offset = float(normal @ ((p0 + p1) * 0.5))
        # Equivalent signs describe the same plane.  Canonicalise them so a
        # consumer can compare models directly across frames.
        if (offset < -1e-9 or
                (abs(offset) <= 1e-9 and
                 (normal[0] < 0.0 or
                  (abs(normal[0]) <= 1e-9 and normal[1] < 0.0)))):
            normal, offset = -normal, -offset
        confidence, evidence = _wall_confidence(
            segment, temporal_frames, evidence_points, camera_positions, pose_std_m)
        wall = {
            "start": {"x": float(p0[0]), "y": float(p0[1])},
            "end": {"x": float(p1[0]), "y": float(p1[1])},
            "z_min": zmin, "z_max": zmax,
            "n_points": n, "inlier_rms_m": round(rms, 4),
            "confidence": confidence,
            "evidence": evidence,
            "temporal_frames": int(temporal_frames),
            # Formal, gravity-constrained vertical-plane representation:
            # normal.x*x + normal.y*y = offset_m.
            "plane": {"normal": {"x": round(float(normal[0]), 6),
                                  "y": round(float(normal[1]), 6)},
                      "offset_m": round(offset, 4)},
            "source": "depth",
            "method": "temporal_normals_grid_hough_tls"}
        if floor_z is not None:
            wall["floor_z"] = round(float(floor_z), 4)
        if floor_source is not None:
            wall["floor_source"] = str(floor_source)
        walls.append(wall)
    return walls


def _selfcheck():
    """Runs on the host with only numpy. The second case is the whole point of the node."""
    rng = np.random.default_rng(1)

    # A wall: 3 m long, observed from 0.5 m to 1.9 m.
    n = 4000
    xy = np.column_stack([rng.uniform(0, 3, n), rng.normal(0, 0.01, n)])
    z = rng.uniform(0.5, 1.9, n)
    segs = _fit_segments_grid_hough(xy, z)
    assert len(segs) == 1, f"one wall expected, got {len(segs)}"
    p0, p1, zmin, zmax, npts, rms = segs[0]
    assert math.hypot(*(p1 - p0)) > 2.5, "the wall should span its length"
    assert zmax - zmin > MIN_VERTICAL_EXTENT_M, "vertical extent should be recovered"
    assert rms < LINE_INLIER_TOL_M, f"a plane should fit tightly, rms={rms}"

    # Two raster strokes of the same physical plane consolidate, whereas a
    # parallel plane and a doorway-separated interval stay distinct.
    duplicate = (p0 + np.array([0.01, 0.015]), p1 + np.array([0.01, 0.015]),
                 zmin, zmax, npts, rms)
    assert _same_plane_landmark(segs[0], duplicate)
    assert len(_consolidate_plane_landmarks(
        xy, z, [(segs[0], np.arange(n)), (duplicate, np.arange(n))])) == 1
    parallel = (p0 + np.array([0.0, 0.15]), p1 + np.array([0.0, 0.15]),
                zmin, zmax, npts, rms)
    assert not _same_plane_landmark(segs[0], parallel)
    separated = (p0 + np.array([4.0, 0.0]), p1 + np.array([4.0, 0.0]),
                 zmin, zmax, npts, rms)
    assert not _same_plane_landmark(segs[0], separated)

    # A sofa back: same footprint, observed over 0.3 m of height. A 2D polygon edge cannot tell
    # this from the wall above -- that is exactly what room_manager's edges cannot do, and
    # refusing it here is why this node is worth having.
    z_low = rng.uniform(0.45, 0.75, n)
    assert _fit_segments_grid_hough(xy, z_low) == [], "a low obstacle must NOT be reported as a wall"

    # Unobserved space contributes nothing: no points, no walls, no silhouette.
    assert _fit_segments_grid_hough(np.zeros((0, 2)), np.zeros(0)) == []

    # Collinear walls separated by a door must not be joined across the opening.
    left_x = rng.uniform(0.0, 1.1, n // 2)
    right_x = rng.uniform(2.0, 3.2, n // 2)
    xy_door = np.column_stack([np.r_[left_x, right_x], rng.normal(0, 0.01, n)])
    z_door = rng.uniform(0.5, 1.9, n)
    door_segments = _fit_segments_grid_hough(xy_door, z_door)
    assert len(door_segments) == 2, f"door must split collinear walls, got {len(door_segments)}"
    assert all(math.hypot(*(p1 - p0)) < 1.5 for p0, p1, *_ in door_segments)

    # A diagonal chain has the same top-down line and vertical extent as a wall,
    # but covers only a one-dimensional trace in the fitted plane (for example a
    # stair rail or coincidental returns).  Plane-patch support must reject it.
    t = rng.uniform(0.0, 3.0, n)
    xy_chain = np.column_stack([t, rng.normal(0, 0.01, n)])
    z_chain = 0.45 + 1.45 * t / 3.0 + rng.normal(0, 0.01, n)
    assert _fit_segments_grid_hough(xy_chain, z_chain) == [], "a 1-D vertical trace is not a wall"

    wall = segments_to_wall_dicts(segs)[0]
    assert wall["start"].keys() >= {"x", "y"}
    assert abs(math.hypot(wall["plane"]["normal"]["x"],
                          wall["plane"]["normal"]["y"]) - 1.0) < 1e-5
    print("wall_detector selfcheck OK: plane patch found; low, sparse and empty inputs refused")


def _benchmark():
    """Small repeatable algorithm benchmark; excludes ROS conversion and TF lookup."""
    rng = np.random.default_rng(4)
    for n in (4000, 8000, 20000):
        xy = np.column_stack((rng.uniform(0, 5, n), rng.normal(0, 0.015, n)))
        z = rng.uniform(0.4, 2.0, n)
        _fit_segments_grid_hough(xy, z)  # warm imports and native-library thread pools
        samples = []
        for _ in range(7):
            start = time.perf_counter()
            _fit_segments_grid_hough(xy, z)
            samples.append((time.perf_counter() - start) * 1000.0)
        print(f"{n:5d} points  grid_hough  median={np.median(samples):6.2f} ms "
              f"p95={np.percentile(samples, 95):6.2f} ms")


def main(args=None):
    if "--selfcheck" in (args or sys.argv):
        return _selfcheck()
    if "--benchmark" in (args or sys.argv):
        return _benchmark()
    import rclpy
    rclpy.init(args=args)
    node = _node_class()()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
