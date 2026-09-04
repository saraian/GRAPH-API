import os
import numpy as np


def compute_fov_volume_from_depth(
    depth_image,
    camera_info,
    node,
    depth_threshold=4.0,
    stride=4,
    output_frame="map",
):
    from cv_utils import ROS2Duration, _apply_transform, _pixels_to_points_habitat_camera
    from config import CFG

    try:
        k = camera_info.k
        stamp = camera_info.header.stamp
        fx, fy, cx, cy = k[0], k[4], k[2], k[5]

        if fx <= 0 or fy <= 0:
            node.get_logger().warn("Invalid camera intrinsics for FOV computation")
            return None

        depth_m = depth_image.astype(np.float32) / 1000.0 if depth_image.dtype == np.uint16 else depth_image.astype(np.float32)
        max_depth = min(depth_threshold, 1.8)
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

        if output_frame == "habitat_camera_optical":
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
                output_frame, "habitat_camera_optical", stamp,
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
