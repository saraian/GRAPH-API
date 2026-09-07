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
RANSAC_ITERS = int(_WCFG.get("ransac_iters", 60))
RANSAC_TOL_M = 0.05               # inlier distance to the fitted line
MIN_INLIERS = 60
MIN_SEGMENT_LEN_M = 0.5
MAX_SEGMENTS = int(_WCFG.get("max_segments", 12))
METHOD = str(_WCFG.get("method", "grid_hough"))
GRID_RESOLUTION_M = float(_WCFG.get("grid_resolution_m", 0.05))
VERTICAL_BANDS = int(_WCFG.get("vertical_bands", 4))
MIN_VERTICAL_BANDS = int(_WCFG.get("min_vertical_bands", 3))
MIN_CELL_POINTS = int(_WCFG.get("min_cell_points", 3))
HOUGH_THRESHOLD = int(_WCFG.get("hough_threshold", 8))
MAX_LINE_GAP_M = float(_WCFG.get("max_line_gap_m", 0.20))


def _fit_segments_ransac(xy, z):
    """Top-down RANSAC line fitting. -> list of (p0, p1, z_min, z_max, n_inliers, rms).

    Returns segments in the order found (longest support first, since each pass takes the
    largest inlier set remaining).
    """
    out = []
    remaining = np.ones(len(xy), dtype=bool)
    rng = np.random.default_rng(0)          # deterministic: a run must be reproducible
    for _ in range(MAX_SEGMENTS):
        idx = np.flatnonzero(remaining)
        if len(idx) < MIN_INLIERS:
            break
        # All RANSAC_ITERS hypotheses at once. They are independent -- scoring them in a
        # Python loop cost 435 ms/frame on 8k points and starved rtabmap of a core; the
        # arithmetic was never the problem, the 720 sequential numpy calls were.
        # Determinism is kept (same seeded rng) but the DRAW ORDER differs from the loop
        # version, so a given frame can yield a different equally-valid segment set.
        pairs = rng.choice(idx, size=(RANSAC_ITERS, 2), replace=True)
        pa, pb = xy[pairs[:, 0]], xy[pairs[:, 1]]
        d = pb - pa
        n = np.hypot(d[:, 0], d[:, 1])
        ok = n >= 1e-6                       # a pair that drew the same point twice
        if not ok.any():
            break
        d = d[ok] / n[ok, None]
        pa = pa[ok]
        normals = np.stack([-d[:, 1], d[:, 0]], axis=1)          # (H, 2)
        # (H, N): distance from every candidate point to every hypothesis line.
        dist = np.abs(np.einsum("nj,hj->hn", xy[idx], normals)
                      - np.einsum("hj,hj->h", pa, normals)[:, None])
        counts = (dist <= RANSAC_TOL_M).sum(axis=1)
        h = int(counts.argmax())
        if counts[h] < MIN_INLIERS:
            break
        best_inliers = idx[dist[h] <= RANSAC_TOL_M]
        best_dir, best_p = d[h], pa[h]
        pts = xy[best_inliers]
        t = (pts - best_p) @ best_dir
        p0, p1 = best_p + best_dir * t.min(), best_p + best_dir * t.max()
        length = math.hypot(*(p1 - p0))
        zs = z[best_inliers]
        extent = float(zs.max() - zs.min())
        normal = np.array([-best_dir[1], best_dir[0]])
        rms = float(np.sqrt(np.mean(((pts - best_p) @ normal) ** 2)))
        remaining[best_inliers] = False
        # A surface, not a silhouette: it must be long enough AND tall enough. Dropping the
        # vertical test would readmit exactly what the polygon edges already give.
        if length >= MIN_SEGMENT_LEN_M and extent >= MIN_VERTICAL_EXTENT_M:
            out.append((p0, p1, float(zs.min()), float(zs.max()), int(len(best_inliers)), rms))
    return out


def _fit_segments_grid_hough(xy, z):
    """Extract walls from a metric 2.5D grid, then refine them against the source points.

    Pixel density affects cell counts but not the size of the Hough problem. Requiring support
    in several vertical bands rejects low furniture before line extraction. Hough supplies only
    candidates; PCA/TLS and robust percentiles produce the published geometry.
    """
    if len(xy) < MIN_INLIERS:
        return []
    import cv2

    xy = np.asarray(xy, dtype=np.float32)
    z = np.asarray(z, dtype=np.float32)
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
        near_idx = np.flatnonzero(distance <= max(RANSAC_TOL_M, GRID_RESOLUTION_M))
        if len(near_idx) < MIN_INLIERS:
            continue
        # Extend the Hough seed over its connected support, but never bridge a door/gap.
        order = np.argsort(along[near_idx])
        ordered_idx = near_idx[order]
        ordered_t = along[ordered_idx]
        cuts = np.flatnonzero(np.diff(ordered_t) > MAX_LINE_GAP_M) + 1
        groups = np.split(ordered_idx, cuts)
        seed_t = line_len * 0.5
        group = min(groups, key=lambda g: abs(float(np.median(along[g])) - seed_t))
        if len(group) < MIN_INLIERS:
            continue
        pts, zs = xy[group], z[group]

        # Total least squares removes the one-degree angular quantisation of Hough.
        centre = pts.mean(axis=0)
        _, _, vh = np.linalg.svd(pts - centre, full_matrices=False)
        fitted_dir = vh[0]
        t = (pts - centre) @ fitted_dir
        lo, hi = np.percentile(t, [2.0, 98.0])
        p0, p1 = centre + fitted_dir * lo, centre + fitted_dir * hi
        length = float(hi - lo)
        zlo, zhi = np.percentile(zs, [5.0, 95.0])
        fitted_normal = np.array([-fitted_dir[1], fitted_dir[0]])
        rms = float(np.sqrt(np.mean(((pts - centre) @ fitted_normal) ** 2)))
        if length >= MIN_SEGMENT_LEN_M and zhi - zlo >= MIN_VERTICAL_EXTENT_M:
            candidates.append((p0, p1, float(zlo), float(zhi), int(len(group)), rms))

    # Hough commonly returns several nearly identical strokes for a thick raster line.
    candidates.sort(key=lambda s: (-s[4], -float(np.linalg.norm(s[1] - s[0]))))
    out = []
    for candidate in candidates:
        cp0, cp1 = candidate[:2]
        cmid, cdir = (cp0 + cp1) * 0.5, cp1 - cp0
        clen = float(np.linalg.norm(cdir))
        cdir /= max(clen, 1e-9)
        duplicate = False
        for accepted in out:
            ap0, ap1 = accepted[:2]
            adir = ap1 - ap0
            adir /= max(float(np.linalg.norm(adir)), 1e-9)
            amid = (ap0 + ap1) * 0.5
            angle_cos = abs(float(cdir @ adir))
            line_distance = abs(float((cmid - amid) @ np.array([-adir[1], adir[0]])))
            axial_distance = abs(float((cmid - amid) @ adir))
            alen = float(np.linalg.norm(ap1 - ap0))
            if (angle_cos > math.cos(math.radians(5.0)) and
                    line_distance < GRID_RESOLUTION_M * 1.5 and
                    axial_distance <= (clen + alen) * 0.5 + MAX_LINE_GAP_M):
                duplicate = True
                break
        if not duplicate:
            out.append(candidate)
            if len(out) >= MAX_SEGMENTS:
                break
    return out


def _fit_segments(xy, z):
    if METHOD == "ransac":
        return _fit_segments_ransac(xy, z)
    if METHOD != "grid_hough":
        raise ValueError(f"unknown walls.method={METHOD!r}")
    return _fit_segments_grid_hough(xy, z)


def _node_class():
    """Build the node class once ROS is importable. Kept out of module scope so the geometry
    above stays testable on a host with no ROS: `python3 wall_detector.py --selfcheck`."""
    import tf2_ros
    from cv_bridge import CvBridge
    from cv_utils import _apply_transform, _pixels_to_points_habitat_camera
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
            self._wake = threading.Event()
            self._stop = threading.Event()
            self._worker = threading.Thread(target=self._fit_worker, daemon=True)
            self._stats = {
                "depth_frames": 0, "tf_failures": 0, "jobs_queued": 0,
                "fits_completed": 0, "valid_depth_points": 0,
                "height_band_points": 0, "walls_last_fit": 0,
                "last_fit_ms": None, "last_reason": "waiting_for_camera_info",
            }
            self._worker.start()
            self.tf_buffer = tf2_ros.Buffer()
            self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
            self.create_subscription(CameraInfo, "/camera/camera_info", self._info_cb,
                                     qos_profile_sensor_data)
            self.create_subscription(Image, "/camera/depth", self._depth_cb,
                                     qos_profile_sensor_data)
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

        def _publish_status(self):
            status = dict(self._stats)
            status["camera_info_received"] = self.camera_info is not None
            status["fit_pending"] = self._pending is not None
            status["depth_waiting_for_tf"] = self._deferred_depth is not None
            status["method"] = METHOD
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
            if np.nanmax(depth) > 100.0:            # millimetres, as some encodings publish
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

            band = ((pts_map[:, 2] >= HEIGHT_BAND_M[0]) & (pts_map[:, 2] <= HEIGHT_BAND_M[1]))
            band_points = int(band.sum())
            self._stats["height_band_points"] = band_points
            if band_points < MIN_INLIERS:
                self._stats["last_reason"] = "too_few_points_in_height_band"
                return
            # The fit is the only expensive step, and it runs OFF the executor thread.
            # rclpy.spin() is single-threaded, so a 164 ms fit inside this callback also
            # blocks this node's OWN /tf subscription -- the buffer the next lookup_transform
            # above reads. Blocking here makes the detector fail its own TF lookups.
            # numpy releases the GIL inside the einsum, so the worker really does run beside
            # the executor rather than interleaving with it.
            self._pending = (pts_map[band][:, :2].copy(), pts_map[band][:, 2].copy())
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
                    walls = segments_to_wall_dicts(_fit_segments(job[0], job[1]))
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


def segments_to_wall_dicts(segments):
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
    return [{"start": {"x": float(p0[0]), "y": float(p0[1])},
             "end": {"x": float(p1[0]), "y": float(p1[1])},
             "z_min": zmin, "z_max": zmax,
             "n_points": n, "inlier_rms_m": round(rms, 4),
             "source": "depth"}
            for p0, p1, zmin, zmax, n, rms in segments]


def _selfcheck():
    """Runs on the host with only numpy. The second case is the whole point of the node."""
    rng = np.random.default_rng(1)

    # A wall: 3 m long, observed from 0.5 m to 1.9 m.
    n = 4000
    xy = np.column_stack([rng.uniform(0, 3, n), rng.normal(0, 0.01, n)])
    z = rng.uniform(0.5, 1.9, n)
    segs = _fit_segments(xy, z)
    assert len(segs) == 1, f"one wall expected, got {len(segs)}"
    p0, p1, zmin, zmax, npts, rms = segs[0]
    assert math.hypot(*(p1 - p0)) > 2.5, "the wall should span its length"
    assert zmax - zmin > MIN_VERTICAL_EXTENT_M, "vertical extent should be recovered"
    assert rms < RANSAC_TOL_M, f"a plane should fit tightly, rms={rms}"

    # A sofa back: same footprint, observed over 0.3 m of height. A 2D polygon edge cannot tell
    # this from the wall above -- that is exactly what room_manager's edges cannot do, and
    # refusing it here is why this node is worth having.
    z_low = rng.uniform(0.45, 0.75, n)
    assert _fit_segments(xy, z_low) == [], "a low obstacle must NOT be reported as a wall"

    # Unobserved space contributes nothing: no points, no walls, no silhouette.
    assert _fit_segments(np.zeros((0, 2)), np.zeros(0)) == []

    # Collinear walls separated by a door must not be joined across the opening.
    left_x = rng.uniform(0.0, 1.1, n // 2)
    right_x = rng.uniform(2.0, 3.2, n // 2)
    xy_door = np.column_stack([np.r_[left_x, right_x], rng.normal(0, 0.01, n)])
    z_door = rng.uniform(0.5, 1.9, n)
    door_segments = _fit_segments_grid_hough(xy_door, z_door)
    assert len(door_segments) == 2, f"door must split collinear walls, got {len(door_segments)}"
    assert all(math.hypot(*(p1 - p0)) < 1.5 for p0, p1, *_ in door_segments)

    assert segments_to_wall_dicts(segs)[0]["start"].keys() >= {"x", "y"}
    print("wall_detector selfcheck OK: wall found, low obstacle refused, empty input empty")


def _benchmark():
    """Small repeatable algorithm benchmark; excludes ROS conversion and TF lookup."""
    rng = np.random.default_rng(4)
    for n in (4000, 8000, 20000):
        xy = np.column_stack((rng.uniform(0, 5, n), rng.normal(0, 0.015, n)))
        z = rng.uniform(0.4, 2.0, n)
        for name, fn in (("grid_hough", _fit_segments_grid_hough),
                         ("ransac", _fit_segments_ransac)):
            fn(xy, z)  # warm imports and native-library thread pools
            samples = []
            for _ in range(7):
                start = time.perf_counter()
                fn(xy, z)
                samples.append((time.perf_counter() - start) * 1000.0)
            print(f"{n:5d} points  {name:10s}  median={np.median(samples):6.2f} ms "
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
