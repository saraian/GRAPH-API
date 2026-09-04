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
3. It was written for `object_manager_3.py`, which is in `old/`.
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


def _fit_segments(xy, z):
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

    class WallDetector(Node):
        def __init__(self):
            super().__init__("wall_detector")
            self.bridge = CvBridge()
            self.camera_info = None
            self._last_fit = 0.0
            self._pending = None
            self._wake = threading.Event()
            self._stop = threading.Event()
            self._worker = threading.Thread(target=self._fit_worker, daemon=True)
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
            self.get_logger().info(
                "wall_detector: depth-derived. Waiting for /camera/depth and /camera/camera_info.")

        def _info_cb(self, msg):
            self.camera_info = msg

        def _depth_cb(self, msg):
            if self.camera_info is None:
                return
            # Rate limit BEFORE any work. Dropping the frame here costs nothing; dropping it
            # after the unprojection has already spent the CPU this guard exists to save.
            now = time.monotonic()
            if now - self._last_fit < MIN_INTERVAL_S:
                return
            if self._pending is not None:
                return          # a fit is already queued; this frame is redundant
            self._last_fit = now
            try:
                t = self.tf_buffer.lookup_transform("map", msg.header.frame_id, msg.header.stamp)
            except TransformException as exc:
                # No pose means no map-frame geometry. Say so and drop the frame: a wall placed
                # with a guessed transform is worse than no wall.
                self.get_logger().warning(f"no map<-{msg.header.frame_id} at stamp: {exc}")
                return

            depth = np.asarray(self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough"),
                               dtype=np.float32)
            if np.nanmax(depth) > 100.0:            # millimetres, as some encodings publish
                depth = depth / 1000.0
            depth = depth[::PIXEL_STRIDE, ::PIXEL_STRIDE]

            ys, xs = np.nonzero(np.isfinite(depth) &
                                (depth >= DEPTH_RANGE_M[0]) & (depth <= DEPTH_RANGE_M[1]))
            if len(xs) < MIN_INLIERS:
                return
            k = self.camera_info.k
            pts_cam = _pixels_to_points_habitat_camera(
                xs * PIXEL_STRIDE, ys * PIXEL_STRIDE, depth[ys, xs], k[0], k[4], k[2], k[5])
            pts_map = _apply_transform(pts_cam, t)

            band = ((pts_map[:, 2] >= HEIGHT_BAND_M[0]) & (pts_map[:, 2] <= HEIGHT_BAND_M[1]))
            if band.sum() < MIN_INLIERS:
                return
            # The fit is the only expensive step, and it runs OFF the executor thread.
            # rclpy.spin() is single-threaded, so a 164 ms fit inside this callback also
            # blocks this node's OWN /tf subscription -- the buffer the next lookup_transform
            # above reads. Blocking here makes the detector fail its own TF lookups.
            # numpy releases the GIL inside the einsum, so the worker really does run beside
            # the executor rather than interleaving with it.
            self._pending = (pts_map[band][:, :2].copy(), pts_map[band][:, 2].copy())
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
                    walls = segments_to_wall_dicts(_fit_segments(job[0], job[1]))
                except Exception:
                    # A worker thread that dies silently leaves a node that looks healthy and
                    # publishes nothing. Log with the traceback and keep the thread alive.
                    self.get_logger().error(f"wall fit failed: {traceback.format_exc()}")
                    continue
                if walls:
                    self.pub.publish(String(data=json.dumps(walls)))

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

    assert segments_to_wall_dicts(segs)[0]["start"].keys() >= {"x", "y"}
    print("wall_detector selfcheck OK: wall found, low obstacle refused, empty input empty")


def main(args=None):
    if "--selfcheck" in (args or sys.argv):
        return _selfcheck()
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
