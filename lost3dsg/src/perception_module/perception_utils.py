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
    from cv_utils import _transform_point_xyz, _pixels_to_points_habitat_camera

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

        if output_frame == "habitat_camera":
            points_out = points_habitat
        else:
            points_out = np.asarray([
                _transform_point_xyz(tuple(point), "habitat_camera", output_frame, stamp=stamp, node=node)
                for point in points_habitat
            ])

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
