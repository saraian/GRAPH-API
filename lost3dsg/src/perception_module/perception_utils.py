import os
import numpy as np


def room_frame_due(frames, xy, stride_m, cap, stamp=None):
    """Is another tagged view of this room due? (GA-350, ported verbatim from GRAPH-API 3a5a818.)

    One frame on entry, then one per `stride_m` of in-room travel, at most `cap`.
    `frames` is the list already captured (each with an optional "pose"), `xy` the
    robot's current ground position or None, `stamp` the current image's timestamp.

    The same image is never a new view: the camera stream and the pose stream tick at
    different rates, so a travel-triggered capture can arrive while the latest frame is
    still the one already saved. Frames are named by image stamp, so saving it again
    silently overwrote the earlier view and left the list pointing two entries at one
    file (run 20260826_0849: 4 captures, 3 files).

    Travel is measured from the last frame that HAS a pose, not simply the last frame.
    The entry view is often saved before the first agent pose reaches this node (the
    pose is published at the end of a detection cycle, the descriptions that trigger
    the entry frame during it), so it is stored unposed; anchoring on it would leave
    the room stuck at one view forever.

    Without a current pose only the entry frame is taken: we cannot tell travel from
    standing still, and a burst of near-identical views from one spot would bias a
    majority vote over them rather than sampling the room.
    """
    if cap <= 0 or len(frames) >= cap:
        return False
    if not frames:
        return True
    if stamp is not None and frames[-1].get("stamp") == stamp:
        return False
    if xy is None:
        return False
    posed = [frame["pose"] for frame in frames if frame.get("pose")]
    if not posed:
        return True          # every view so far is unposed and we have a pose now
    return float(np.hypot(xy[0] - posed[-1]["x"], xy[1] - posed[-1]["y"])) >= stride_m


def compute_fov_volume_from_depth(
    depth_image,
    camera_info,
    node,
    depth_threshold=None,
    stride=4,
    output_frame=None,
    stamp=None,
):
    from cv_utils import ROS2Duration, _apply_transform, _pixels_to_points_habitat_camera
    from rclpy.time import Time
    from config import CFG, world_frame

    output_frame = output_frame or world_frame()
    camera_frame = camera_info.header.frame_id or CFG["frames"]["camera"]

    try:
        k = camera_info.k
        stamp = stamp if stamp is not None else camera_info.header.stamp
        fx, fy, cx, cy = k[0], k[4], k[2], k[5]

        if fx <= 0 or fy <= 0:
            node.get_logger().warn("Invalid camera intrinsics for FOV computation")
            return None

        depth_m = depth_image.astype(np.float32) / 1000.0 if depth_image.dtype == np.uint16 else depth_image.astype(np.float32)

        # The original 1.8 m default was tuned for the Habitat camera and made the
        # physical Tiago's view volume empty: most of the objects in front of the
        # robot are farther away than that.  Keep an explicit function argument as
        # an optional upper bound, but let the active configuration choose a larger
        # sensor range when the caller does not provide one.
        configured_max_depth = float(CFG['perception'].get('fov_max_depth_m', 1.8))
        max_depth = configured_max_depth if depth_threshold is None else min(
            float(depth_threshold), configured_max_depth)
        sampled_depth = depth_m[::stride, ::stride]
        valid_mask = (sampled_depth > 0.1) & (sampled_depth < max_depth) & np.isfinite(sampled_depth)

        if not np.any(valid_mask):
            node.get_logger().warn("No valid depth values found for FOV computation")
            return None

        ys_small, xs_small = np.nonzero(valid_mask)
        ys = ys_small * stride
        xs = xs_small * stride
        zs = depth_m[ys, xs]

        points_habitat = _pixels_to_points_habitat_camera(xs, ys, zs, fx, fy, cx, cy)

        if output_frame == camera_frame:
            points_out = points_habitat
        else:
            # LAT-1. ONE lookup for the whole cloud, then the transform applied to the
            # (N,3) array. This branch used to call `_transform_point_xyz` PER POINT, and
            # each of those did its own `lookup_transform` -- thousands of TF lookups for
            # one frame's FOV, all resolving the same transform at the same stamp.
            #
            # NOT bit-identical, and the plan's word for it was wrong: the per-point path
            # computed `R.dot(p) + T` and `_apply_transform` computes `p.dot(R.T) + T`. That
            # is the same product written the other way round, so the two differ only in
            # floating-point accumulation order. MEASURED on the smoke suite's fixed 45-degree
            # transform: max absolute deviation 6.8e-14 m on coordinates up to 1e3 m. The
            # value read from this function is the min/max over the cloud in metres, fed to
            # containment tests with metre-scale thresholds, so a 1e-13 m shift cannot move
            # any decision. Same R, same T, same stamp; one lookup instead of N.
            #
            # The lookup stays inside the function's existing try/except, so a TF failure
            # still warns and returns None exactly as before, and it now fails once rather
            # than on the first of N points.
            trans = node.tf_buffer.lookup_transform(
                output_frame, camera_frame, Time.from_msg(stamp) if hasattr(stamp, "sec") else stamp,
                timeout=ROS2Duration(seconds=CFG["tf"]["lookup_timeout"]))
            points_out = _apply_transform(points_habitat, trans)

        return {
            "x_min": float(points_out[:, 0].min()),
            "x_max": float(points_out[:, 0].max()),
            "y_min": float(points_out[:, 1].min()),
            "y_max": float(points_out[:, 1].max()),
            "z_min": float(points_out[:, 2].min()),
            "z_max": float(points_out[:, 2].max()),
        }
    except Exception as exc:
        node.get_logger().warn(f"FOV computation failed: {exc}")
        return None


def get_project_root(file_path: str) -> str:
    current_dir = os.path.dirname(os.path.abspath(file_path))
    return os.path.abspath(os.path.join(current_dir, "../.."))
