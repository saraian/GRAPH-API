#!/usr/bin/env python3
import hashlib
import os
import urllib.parse
import re
import numpy as np
from sensor_msgs.msg import CameraInfo
from visualization_msgs.msg import Marker, MarkerArray
from matplotlib.colors import to_rgb
import cv2
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2
from std_msgs.msg import Header, String
from sensor_msgs.msg import PointField
import json
from geometry_msgs.msg import Point
from utils import statistical_outlier_removal, get_distinct_color
from box_view import BOX_EDGES, box_corners_map, project_visible
from config import CFG, vlm_completion_kwargs, world_frame
import struct
from openai import OpenAI
import base64
from rclpy.duration import Duration
from rclpy.time import Time
from rclpy.duration import Duration as ROS2Duration
from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy
from builtin_interfaces.msg import Time as TimeMsg
file_path = os.path.abspath(__file__)
import time
import itertools
import tf2_ros

def _filter_object_points(mask, depth_image, fx, fy, cx, cy,
                           max_points_per_obj=20000,
                           remove_outliers=True, sor_k=15, sor_std=1.5):
    mask = np.array(mask)
    mask2d = mask[:, :, 0] if mask.ndim == 3 else mask
    ys, xs = np.nonzero(mask2d.astype(bool))

    if len(xs) == 0:
        return None

    if len(xs) > max_points_per_obj:
        idx = np.linspace(0, len(xs) - 1, max_points_per_obj).astype(int)
        xs, ys = xs[idx], ys[idx]

    zs = depth_image[ys, xs].astype(np.float64)
    valid = np.isfinite(zs) & (zs > 0.0)
    if not valid.any():
        return None
    xs, ys, zs = xs[valid], ys[valid], zs[valid]

    depth_min, depth_max = _depth_bounds(zs)
    keep = (zs >= depth_min) & (zs <= depth_max)
    xs, ys, zs = xs[keep], ys[keep], zs[keep]
    if len(xs) == 0:
        return None

    pts = _pixels_to_points_habitat_camera(xs, ys, zs, fx, fy, cx, cy)

    if remove_outliers and len(pts) > 20:
        keep_idx = statistical_outlier_removal(pts, k=sor_k, std_ratio=sor_std)
        pts = pts[keep_idx]
        if len(pts) == 0:
            return None

    return pts


def _pixels_to_points_habitat_camera(xs, ys, zs, fx, fy, cx, cy):
    xs = np.asarray(xs, dtype=np.float64)
    ys = np.asarray(ys, dtype=np.float64)
    zs = np.asarray(zs, dtype=np.float64)

    # Standard pinhole camera coordinates:
    # X points right, Y points down, Z points forward from the camera.
    X = (xs - cx) * zs / fx
    Y = (ys - cy) * zs / fy
    Z = zs

    return np.column_stack([X, Y, Z]).astype(np.float32)


def _get_R_and_T(trans):
    t = trans.transform
    T = np.array([t.translation.x, t.translation.y, t.translation.z], dtype=np.float64)

    qx = float(t.rotation.x)
    qy = float(t.rotation.y)
    qz = float(t.rotation.z)
    qw = float(t.rotation.w)

    n = qx*qx + qy*qy + qz*qz + qw*qw
    if n < 1e-12:
        R = np.eye(3, dtype=np.float64)
        return R, T

    s = 2.0 / n

    xx = qx * qx * s
    yy = qy * qy * s
    zz = qz * qz * s
    xy = qx * qy * s
    xz = qx * qz * s
    yz = qy * qz * s
    wx = qw * qx * s
    wy = qw * qy * s
    wz = qw * qz * s

    R = np.array([
        [1.0 - (yy + zz), xy - wz,         xz + wy],
        [xy + wz,         1.0 - (xx + zz), yz - wx],
        [xz - wy,         yz + wx,         1.0 - (xx + yy)],
    ], dtype=np.float64)

    return R, T


def _transform_point_xyz(pt_xyz, source_frame, target_frame, stamp=None, timeout=None, node=None, tf_buffer=None):
    if timeout is None:
        timeout = CFG["tf"]["lookup_timeout"]
    if target_frame == source_frame:
        return np.array(pt_xyz).reshape(3)

    tf_buffer = tf_buffer or getattr(node, 'tf_buffer', None)
    if tf_buffer is None:
        raise ValueError("tf_buffer or node with tf_buffer required")

    lookup_time = stamp if stamp is not None else Time()

    try:
        trans = tf_buffer.lookup_transform(target_frame, source_frame, lookup_time,
                                            timeout=ROS2Duration(seconds=timeout))
    except tf2_ros.ExtrapolationException as e:
        # Dato ormai troppo vecchio (o troppo nel futuro oltre il buffer): non aspettare, fallisci subito
        raise RuntimeError(f"TF unavailable (extrapolation) for {source_frame}->{target_frame} "
                            f"al tempo {lookup_time}: {e}")
    except Exception as e:
        raise RuntimeError(f"TF lookup fallito per {source_frame}->{target_frame} al tempo {lookup_time}: {e}")

    R, T = _get_R_and_T(trans)
    return R.dot(np.array(pt_xyz)) + T

def _clear_markers(topic, node=None, publisher=None):
    ma = MarkerArray()
    m = Marker(); m.action = Marker.DELETEALL
    ma.markers.append(m)
    if publisher:
        publisher.publish(ma)
    elif node:
        pub = node.create_publisher(MarkerArray, topic,
                                    QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL))
        time.sleep(0.05)
        pub.publish(ma)
    else:
        raise ValueError("Either node or publisher must be provided")


def overlay_mask_on_image(image, mask, color_rgb=(0.0, 1.0, 0.0), alpha=0.5):
    mask_bool = mask.astype(bool)
    color_bgr = (np.array(color_rgb[::-1]) * 255).astype(np.uint8)
    image[mask_bool] = (image[mask_bool] * (1 - alpha) + color_bgr * alpha).astype(np.uint8)
    return image


def _pack_rgb(color_rgba):
    r = int(color_rgba.r * 255) & 0xFF
    g = int(color_rgba.g * 255) & 0xFF
    b = int(color_rgba.b * 255) & 0xFF
    return struct.unpack('f', struct.pack('I', (r << 16) | (g << 8) | b))[0]


def _depth_bounds(depth_values):
    arr = np.array(depth_values)
    median = np.median(arr)
    mad = np.median(np.abs(arr - median))
    return (median - 4.5*mad, median + 4.5*mad) if mad > 0.001 else (median*0.5, median*1.5)


def _robust_bounds_from_points(points_xyz, lower_q=5.0, upper_q=95.0):
    pts = np.asarray(points_xyz, dtype=np.float64)
    if pts.size == 0:
        return None, None

    lower = np.percentile(pts, lower_q, axis=0)
    upper = np.percentile(pts, upper_q, axis=0)
    return lower, upper


def mask_list_to_pointcloud2(
    masks,
    depth_image,
    camera_info,
    node,
    labels=None,
    topic="/pcl_objects",
    max_points_per_obj=20000,
    publisher=None,
    labels_publisher=None,
    transform=None,
):
    """Per-object coloured cloud. With `transform` (map<-optical of this frame, the
    same one the boxes are lifted with) the cloud is published in the map frame;
    without it, in CFG frames.camera — which must then be the OPTICAL frame, or the
    cloud renders rotated (depth into the height axis) and looks absent in rviz."""
    if not isinstance(camera_info, CameraInfo):
        raise TypeError("camera_info must be CameraInfo")

    labels = labels or [f"obj_{i}" for i in range(len(masks))]
    fx, fy, cx, cy = camera_info.k[0], camera_info.k[4], camera_info.k[2], camera_info.k[5]
    # Depth pixels are expressed in the OPTICAL pinhole frame; the config default
    # is the optical frame for exactly that reason.
    camera_frame = CFG["frames"]["camera"]

    current_points, id_to_label = [], {}

    for obj_idx, mask in enumerate(masks):
        mask = np.array(mask)
        mask2d = mask[:, :, 0] if mask.ndim == 3 else mask
        ys, xs = np.nonzero(mask2d.astype(bool))

        if len(xs) == 0:
            continue

        if len(xs) > max_points_per_obj:
            idx = np.linspace(0, len(xs) - 1, max_points_per_obj).astype(int)
            xs, ys = xs[idx], ys[idx]

        zs = depth_image[ys, xs].astype(np.float32)
        valid = np.isfinite(zs) & (zs > 0.0)
        if not valid.any():
            node.get_logger().warn(f"Mask {obj_idx}: no valid depth, skip")
            continue

        xs, ys, zs = xs[valid], ys[valid], zs[valid]

        # unused — superseded by _pixels_to_points_habitat_camera below
        # x = (xs - cx) * zs / fx
        # y = (ys - cy) * zs / fy
        pts = _pixels_to_points_habitat_camera(xs, ys, zs, fx, fy, cx, cy)

        if len(pts) == 0:
            continue
        if transform is not None:
            pts = _apply_transform(pts, transform)

        unique_id = node.pcl_object_id_counter
        rgb_packed = _pack_rgb(get_distinct_color(unique_id))

        for pt in pts:
            current_points.append((float(pt[0]), float(pt[1]), float(pt[2]), rgb_packed, unique_id))

        id_to_label[unique_id] = labels[obj_idx]
        node.pcl_object_id_counter += 1

    if not current_points:
        node.get_logger().warn("mask_list_to_pointcloud2_debug: no points to publish")
        return

    fields = [
        PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
        PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
        PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
        PointField(name="rgb", offset=12, datatype=PointField.FLOAT32, count=1),
        PointField(name="object_id", offset=16, datatype=PointField.INT32, count=1),
    ]

    if transform is not None:
        header = Header(stamp=camera_info.header.stamp, frame_id=world_frame())
    else:
        header = Header(stamp=camera_info.header.stamp, frame_id=camera_frame)
    cloud_msg = point_cloud2.create_cloud(header, fields, current_points)

    qos = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
    (publisher or node.create_publisher(PointCloud2, topic, qos)).publish(cloud_msg)

    try:
        lbl_msg = String()
        lbl_msg.data = json.dumps(id_to_label)
        (labels_publisher or node.create_publisher(String, topic + "_labels", qos)).publish(lbl_msg)
    except Exception:
        node.get_logger().warn("Could not publish pcl labels")
    

def publish_individual_pointclouds_by_id(masks, depth_image, camera_info, node, labels=None,
                                          frame_id="camera_link", topic_prefix="/pcl_id",
                                          max_points_per_obj=20000, remove_outliers=True,
                                          sor_k=15, sor_std=1.5, publishers_dict=None,
                                          id_counter_start=0, timestamp=None):
    if not isinstance(camera_info, CameraInfo):
        raise TypeError('camera_info must be CameraInfo')

    labels       = labels or [f"obj_{i}" for i in range(len(masks))]
    fx, fy, cx, cy = camera_info.k[0], camera_info.k[4], camera_info.k[2], camera_info.k[5]
    #camera_frame = "head_front_camera_color_optical_frame"
    camera_frame = CFG["frames"]["camera"]
    palette      = [to_rgb(c) for c in ('red','green','blue','magenta','cyan','yellow','orange','purple','brown','pink')]
    qos          = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
    stamp        = timestamp or node.get_clock().now().to_msg()
    fields       = [
        PointField(name='x',         offset=0,  datatype=PointField.FLOAT32, count=1),
        PointField(name='y',         offset=4,  datatype=PointField.FLOAT32, count=1),
        PointField(name='z',         offset=8,  datatype=PointField.FLOAT32, count=1),
        PointField(name='rgb',       offset=12, datatype=PointField.FLOAT32, count=1),
        PointField(name='object_id', offset=16, datatype=PointField.INT32,   count=1),
    ]
    published_count = 0

    for obj_idx, mask in enumerate(masks):
        obj_id = id_counter_start + obj_idx
        pts = _filter_object_points(mask, depth_image, fx, fy, cx, cy,
                             max_points_per_obj=max_points_per_obj,
                             remove_outliers=remove_outliers, sor_k=sor_k, sor_std=sor_std)
        if pts is None:
            node.get_logger().warn(f"{labels[obj_idx]}: no valid points after filtering")
            continue
        
        # Transform if needed
        if frame_id != camera_frame:
            transformed = []
            for pt in pts:
                try:
                    transformed.append(_transform_point_xyz(pt, camera_frame, frame_id, node=node))
                except Exception as e:
                    node.get_logger().warn(f"Transform failed: {e}")
            pts = np.array(transformed) if transformed else np.empty((0, 3))

        if len(pts) == 0:
            continue

        # Outlier removal
        if remove_outliers and len(pts) > 20:
            pts = pts[statistical_outlier_removal(pts, k=sor_k, std_ratio=sor_std)]

        if len(pts) == 0:
            node.get_logger().warn(f"Obj {obj_id} ({labels[obj_idx]}): no points after filtering")
            continue

        r, g, b = (int(c * 255) for c in palette[obj_id % len(palette)])
        rgb_packed = struct.unpack('f', struct.pack('I', (r << 16) | (g << 8) | b))[0]
        points = [(float(p[0]), float(p[1]), float(p[2]), rgb_packed, obj_id) for p in pts]

        # Publish
        topic_name = re.sub(r'[^a-zA-Z0-9_]', '_', labels[obj_idx]) + f"_{obj_id}"
        if publishers_dict is not None and topic_name not in publishers_dict:
            publishers_dict[topic_name] = node.create_publisher(PointCloud2, topic_name, qos)
        pub = (publishers_dict or {}).get(topic_name) or node.create_publisher(PointCloud2, topic_name, qos)

        pub.publish(point_cloud2.create_cloud(Header(stamp=stamp, frame_id=frame_id), fields, points))
        published_count += 1

    node.get_logger().info(f"publish_individual_pointclouds_by_id: {published_count} clouds published")
    return published_count


def points_list_to_rviz_3d(points, node, centroid_marker_pub=None, labels=None,
                            frame_id=None, topic="/centroid_markers", marker_scale=0.06,
                            stamp=None):
    frame_id = frame_id or world_frame()
    if centroid_marker_pub is None:
        centroid_marker_pub = node.create_publisher(
            MarkerArray, topic, QoSProfile(depth=10, durability=DurabilityPolicy.TRANSIENT_LOCAL))

    if not hasattr(node, '_centroid_marker_id_counter'):
        node._centroid_marker_id_counter = 0

    ma    = MarkerArray()
    stamp = stamp if stamp is not None else node.get_clock().now().to_msg()   # <-- unica riga cambiata

    for i, point in enumerate(points):
        if point is None:
            continue
        uid   = node._centroid_marker_id_counter
        # unused — markers carry no text; restore if a TEXT_VIEW_FACING label marker is added
        # label = labels[i] if labels and i < len(labels) else f"obj_{i}"
        m = Marker()
        m.header.frame_id = frame_id
        m.header.stamp    = stamp
        m.ns, m.id, m.type, m.action = "centroid_spheres", uid, Marker.SPHERE, Marker.ADD
        m.pose.position.x = float(point[0])   # <-- fix del bug precedente, già discusso
        m.pose.position.y = float(point[1])
        m.pose.position.z = float(point[2])
        m.pose.orientation.w = 1.0
        m.scale.x = m.scale.y = m.scale.z = marker_scale
        m.color    = get_distinct_color(uid)
        m.lifetime = Duration(seconds=0).to_msg()
        ma.markers.append(m)
        node._centroid_marker_id_counter += 1

    if ma.markers:
        centroid_marker_pub.publish(ma)


def init_bbox_publisher(node):
    qos = QoSProfile(depth=10, durability=DurabilityPolicy.VOLATILE,
                     reliability=ReliabilityPolicy.BEST_EFFORT)
    bbox_pub     = node.create_publisher(MarkerArray, '/bbox_marker',      qos)
    centroid_pub = node.create_publisher(MarkerArray, '/centroid_markers', qos)
    time.sleep(0.5)
    return bbox_pub, centroid_pub

def _apply_transform(pts, transform):
    """Applica in un colpo solo una trasformazione tf2 già risolta a un array (N,3)."""
    R, T = _get_R_and_T(transform)
    pts = np.asarray(pts, dtype=np.float64)
    return pts.dot(R.T) + T


def draw_cloud(img, points_map, camera_info, transform, max_points=6000, radius=1):
    """Project map-frame points into this frame and dot them in. GA-218.

    THE SAME PROJECTION `draw_boxes_3d` USES, on a point set instead of eight corners: the
    map<-optical transform of THIS frame, then the pinhole. Reusing it is the point -- a
    second implementation of the projection would drift from the first, and a cloud drawn
    with a stale or mismatched transform looks plausible while being wrong, which is the
    failure mode a debugging view must not have.

    Coloured by HEIGHT, not by a flat tint: a uniform overlay hides whether the cloud is
    lying on the floor or floating, and floating geometry is exactly what a bad localisation
    produces. Points BEHIND the camera are dropped rather than wrapped -- a negative z
    through a pinhole projects to a plausible pixel on the wrong side of the image.
    """
    if points_map is None or len(points_map) == 0 or transform is None:
        return img
    pts = np.asarray(points_map, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] < 3:
        return img
    pts = pts[:, :3]
    if len(pts) > max_points:
        # Deterministic stride, not a random draw: the same cloud should dot the same way on
        # consecutive frames, or the overlay shimmers and looks like motion that is not there.
        pts = pts[:: max(1, len(pts) // max_points)][:max_points]

    # _get_R_and_T is what draw_boxes_3d itself uses, and `(p - T) @ R` is its exact
    # expression for map -> optical. Written by hand the first time as a `transform_to_matrix`
    # that DOES NOT EXIST -- checked before applying rather than discovered at runtime.
    R, T = _get_R_and_T(transform)
    cam = (pts - T) @ R
    front = cam[:, 2] > 0.05
    cam = cam[front]
    if not len(cam):
        return img
    k = camera_info.k
    fx, fy, cx, cy = float(k[0]), float(k[4]), float(k[2]), float(k[5])
    u = (fx * cam[:, 0] / cam[:, 2] + cx).astype(np.int32)
    v = (fy * cam[:, 1] / cam[:, 2] + cy).astype(np.int32)
    h, w = img.shape[:2]
    inside = (u >= 0) & (u < w) & (v >= 0) & (v < h)
    u, v, z = u[inside], v[inside], pts[front][inside][:, 2]
    if not len(u):
        return img
    lo, hi = float(z.min()), float(z.max())
    span = max(hi - lo, 1e-6)
    shade = ((z - lo) / span * 255).astype(np.uint8)
    for uu, vv, ss in zip(u, v, shade):
        img[max(0, vv - radius):vv + radius + 1, max(0, uu - radius):uu + radius + 1] = (
            int(255 - ss), int(ss), 90)
    return img


# How strongly a 3D box's faces are tinted. Deliberately low: the point is a depth cue,
# not a highlight, and the camera image underneath has to stay readable through it.
BOX_FACE_ALPHA = 0.13


def draw_boxes_3d(img, bboxes_3d, labels, camera_info, transform, depth=None, min_visible_points=1,
                  tol_abs=0.10, tol_rel=0.05):
    """Draw each 3D box (map frame) as a wireframe in the image it was measured from.

    `transform` is the map<-optical transform of this very frame (the one the points
    were lifted with), so this is the exact inverse of _apply_transform followed by the
    pinhole. Orange = PCA-oriented box, blue = axis-aligned fallback. With `depth`
    the box is subject to the same visibility rule as the simulator overlay
    (box_view.project_visible): not drawn when out of view or occluded, thin when
    only partially visible.
    """
    fx, fy, cx, cy = camera_info.k[0], camera_info.k[4], camera_info.k[2], camera_info.k[5]
    h, w = img.shape[:2]
    R, T = _get_R_and_T(transform)
    for bbox, label in zip(bboxes_3d, labels):
        corners, oriented = box_corners_map(bbox)
        if not corners:
            continue
        cam = (np.asarray(corners + [np.mean(corners, axis=0)]) - T) @ R   # R^T (p - T): map -> optical
        px, n_vis = project_visible(cam, depth, fx, fy, cx, cy, w, h, tol_abs=tol_abs, tol_rel=tol_rel)
        if n_vis < min_visible_points:
            continue
        colour = (0, 165, 255) if oriented else (255, 160, 0)
        thick = 2 if n_vis >= 5 else 1

        # Faces, lightly shaded, UNDER the wireframe.
        #
        # A wireframe alone reads as a flat tangle once a few boxes overlap: there is no
        # cue for which face is toward the camera, so two boxes at different depths look
        # like one lattice. A low-alpha fill gives the solid back without hiding the
        # image behind it -- the edges are still drawn on top at full strength, so
        # nothing that was legible before becomes less so.
        #
        # Corner order comes from box_corners_map's nested comprehension,
        # `for sx in (-ex, ex) for sy in (-ey, ey) for sz in (-ez, ez)`, so the index is
        # 4*ix + 2*iy + iz. Each quad below is therefore a genuine face in cyclic order;
        # listing them in the wrong order would fill bow-ties rather than faces.
        if n_vis >= min_visible_points:
            quads = np.array([[px[0], px[1], px[3], px[2]],    # x-
                              [px[4], px[5], px[7], px[6]],    # x+
                              [px[0], px[1], px[5], px[4]],    # y-
                              [px[2], px[3], px[7], px[6]],    # y+
                              [px[0], px[2], px[6], px[4]],    # z-
                              [px[1], px[3], px[7], px[5]]],   # z+
                             dtype=np.int32)
            overlay = img.copy()
            cv2.fillPoly(overlay, quads, colour, cv2.LINE_AA)
            cv2.addWeighted(overlay, BOX_FACE_ALPHA, img, 1.0 - BOX_FACE_ALPHA, 0, dst=img)

        for i, j in BOX_EDGES:
            cv2.line(img, px[i], px[j], colour, thick, cv2.LINE_AA)
        top = min(px, key=lambda p: p[1])
        cv2.putText(img, str(label), (top[0], max(14, top[1] - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour, thick, cv2.LINE_AA)
    return img


def mask_list_to_centroid_and_bbox(mask_list, labels, depth_image, camera_info, node,
                                    bbox_marker_pub=None, centroid_marker_pub=None,
                                    max_points_per_obj=20000, remove_outliers=True,
                                    sor_k=30, sor_std=1.5, transform=None,
                                    output_frame=None, points_out=None):
    """`points_out`, when a list, receives one entry per mask: the map-frame points the
    box was measured from, or None where no box was produced. The PCA stage reads them
    instead of re-running the projection and the outlier removal on the same mask."""
    output_frame = output_frame or world_frame()
    fx, fy, cx, cy = camera_info.k[0], camera_info.k[4], camera_info.k[2], camera_info.k[5]
    camera_frame = CFG["frames"]["camera"]
    centroids_3d, bboxes_3d, all_markers = [], [], []
    stamp = camera_info.header.stamp

    if not hasattr(node, '_bbox_marker_id_counter'):
        node._bbox_marker_id_counter = 0

    for mask_idx, mask in enumerate(mask_list):
        label = labels[mask_idx] if labels and mask_idx < len(labels) else f"obj_{mask_idx}"

        pts = _filter_object_points(
            mask, depth_image, fx, fy, cx, cy,
            max_points_per_obj=max_points_per_obj,
            remove_outliers=remove_outliers,
            sor_k=sor_k, sor_std=sor_std
        )

        if pts is None:
            node.get_logger().warn(f"{label}: no valid points after filtering")
            centroids_3d.append(None)
            bboxes_3d.append(None)
            if points_out is not None:
                points_out.append(None)
            continue

        try:
            if output_frame == camera_frame:
                pts_map = pts
            elif transform is not None:
                pts_map = _apply_transform(pts, transform)
            else:
                pts_map = np.array([
                    _transform_point_xyz(tuple(p), camera_frame, output_frame, stamp=stamp, node=node)
                    for p in pts
                ])


            centroid_map = np.mean(pts_map, axis=0)

            if centroid_marker_pub is not None:
                points_list_to_rviz_3d(
                    [centroid_map],
                    node,
                    centroid_marker_pub=centroid_marker_pub,
                    labels=[label],
                    frame_id=output_frame,
                    marker_scale=0.05
                )

            # The points have already passed the depth/MAD and SOR filters above. A box is an
            # enclosure, so do not apply another 5/95 percentile trim here: that was the reason
            # valid mask-supported points fell outside both the merge and visualisation boxes.
            mins_map = np.min(pts_map, axis=0)
            maxs_map = np.max(pts_map, axis=0)

            if np.any((maxs_map - mins_map) <= 1e-4):
                raise ValueError("degenerate bbox after filtered-point enclosure")

            bbox_dict = {
                "x_min": float(mins_map[0]), "x_max": float(maxs_map[0]),
                "y_min": float(mins_map[1]), "y_max": float(maxs_map[1]),
                "z_min": float(mins_map[2]), "z_max": float(maxs_map[2]),
            }
            corners_map = np.array(list(itertools.product(*zip(mins_map, maxs_map))))

            # ALL THREE appends together, as the LAST statements of the try. The centroid used
            # to be appended before the two raises above, so on an empty or degenerate box
            # the except path's None made it TWO entries for one mask and every later
            # centroid was read against the wrong detection by _archive_detections
            # (positional). Found by agent1 in review 2026-09-06, shown red-first on a
            # one-pixel-row mask, GA-327. Kept adjacent so that nothing inserted above them
            # can ever leave the three lists at different lengths.
            centroids_3d.append(tuple(float(v) for v in centroid_map))
            bboxes_3d.append(bbox_dict)
            if points_out is not None:
                points_out.append(pts_map)

        except Exception as e:
            node.get_logger().warn(f"{label}: transform to map failed: {e}")
            centroids_3d.append(None)
            bboxes_3d.append(None)
            if points_out is not None:
                points_out.append(None)
            continue

        if bbox_marker_pub is not None:
            m = Marker()
            m.header.frame_id = output_frame
            m.header.stamp = stamp
            m.ns = "bbox_markers"
            m.id = node._bbox_marker_id_counter
            m.type = Marker.SPHERE_LIST
            m.action = Marker.ADD
            m.scale.x = m.scale.y = m.scale.z = 0.02
            m.color = get_distinct_color(node._bbox_marker_id_counter)
            m.lifetime = Duration(seconds=0).to_msg()
            m.points = [Point(x=float(p[0]), y=float(p[1]), z=float(p[2])) for p in corners_map]
            all_markers.append(m)
            node._bbox_marker_id_counter += 1

            if 'pts_map' in locals():
                del pts_map

    if bbox_marker_pub is not None and all_markers:
        ma = MarkerArray()
        ma.markers = all_markers
        bbox_marker_pub.publish(ma)
        node.get_logger().info(f"Published {len(all_markers)} bbox markers on /bbox_marker")

    node.get_logger().info(f"Total bboxes_3d={len(bboxes_3d)}")

    return centroids_3d, bboxes_3d

def _make_marker(frame_id, stamp, ns, mid, mtype, scale, color, position, lifetime_sec=0):
    """Helper to build a basic RViz Marker."""
    m = Marker()
    m.header.frame_id, m.header.stamp = frame_id, stamp
    m.ns, m.id, m.type, m.action = ns, mid, mtype, Marker.ADD
    m.pose.position.x, m.pose.position.y, m.pose.position.z = position
    m.pose.orientation.w = 1.0
    if isinstance(scale, (int, float)):
        m.scale.x = m.scale.y = m.scale.z = scale
    if hasattr(color, 'r'):
        m.color.r = float(color.r)
        m.color.g = float(color.g)
        m.color.b = float(color.b)
        m.color.a = float(color.a)
    else:
        m.color.r = float(color[0])
        m.color.g = float(color[1])
        m.color.b = float(color[2])
        m.color.a = float(color[3])
    m.lifetime = Duration(seconds=lifetime_sec).to_msg()
    return m


def _make_text_marker(frame_id, stamp, ns, mid, text, position, scale=0.08, color=(1,1,1,1)):
    m = _make_marker(frame_id, stamp, ns, mid, Marker.TEXT_VIEW_FACING, scale, color,
                     (position[0], position[1], position[2] + 0.1))
    m.text = text
    return m


def _centroid_from_bbox(bbox):
    return [(bbox[f"{k}_min"] + bbox[f"{k}_max"]) / 2.0 for k in "xyz"]


def _stamp_from_seconds(timestamp_sec):
    stamp = TimeMsg()
    sec = int(timestamp_sec)
    nanosec = int(round((timestamp_sec - sec) * 1e9))
    if nanosec >= 1_000_000_000:
        sec += 1
        nanosec -= 1_000_000_000
    stamp.sec = sec
    stamp.nanosec = nanosec
    return stamp


def _stable_marker_id(obj, fallback_index):
    """Return a repeatable RViz id for a persistent object.

    The object-manager and merge paths can enumerate the same world-model list in
    different orders.  Using the list index therefore makes RViz associate a
    label with the wrong box for one update (and makes a moving label look late).
    Persistent objects have a stable ``object_id``; the bbox fallback keeps legacy
    objects renderable when an old run did not store one.
    """
    object_id = getattr(obj, "object_id", None)
    if object_id is None or not str(object_id).strip():
        bbox = getattr(obj, "bbox", None) or {}
        identity = [getattr(obj, "label", "")]
        identity.extend(bbox.get(key) for key in ("x_min", "x_max", "y_min", "y_max", "z_min", "z_max"))
        object_id = "|".join(str(value) for value in identity)

    digest = hashlib.sha1(str(object_id).encode("utf-8")).digest()
    marker_id = int.from_bytes(digest[:4], byteorder="big") & 0x7FFFFFFF
    return marker_id or (fallback_index + 1)


def _publish_centroid_markers(node, objects, pub, ns, color, label_suffix="",
                              prefer_fused=False):
    if not pub:
        return
    ma = MarkerArray()
    clear = Marker()
    clear.header.frame_id = world_frame()
    clear.header.stamp = node.get_clock().now().to_msg()
    clear.action = Marker.DELETEALL
    ma.markers.append(clear)
    used_ids = set()
    for i, obj in enumerate(objects):
        bbox = (getattr(obj, "fused_bbox", None) if prefer_fused else None) or obj.bbox
        if bbox is None:
            continue
        obj_stamp = getattr(obj, "last_perception_time", None)
        stamp = _stamp_from_seconds(obj_stamp) if obj_stamp else node.get_clock().now().to_msg()
        cx, cy, cz = _centroid_from_bbox(bbox)
        marker_id = _stable_marker_id(obj, i)
        while marker_id in used_ids:
            marker_id = (marker_id + 1) & 0x7FFFFFFF or 1
        used_ids.add(marker_id)
        ma.markers.append(_make_marker(world_frame(), stamp, ns, marker_id,
                                       Marker.SPHERE, 0.08, color, (cx, cy, cz)))
        ma.markers.append(_make_text_marker(world_frame(), stamp, ns+"_labels", marker_id,
                                            str(obj.label).replace(' ', '') + label_suffix,
                                            (cx, cy, cz)))
    # Publish even when the list is empty: DELETEALL removes labels belonging to
    # objects deleted or merged since the previous update.
    pub.publish(ma)


def publish_pov_volume(node, pov_volume, considered_volume_pub=None):
    m = Marker()
    m.header.frame_id = world_frame()
    m.header.stamp    = node.get_clock().now().to_msg()
    m.ns, m.id, m.type, m.action = "pov_volume", 0, Marker.CUBE, Marker.ADD
    m.pose.position.x = (pov_volume["x_min"] + pov_volume["x_max"]) / 2
    m.pose.position.y = (pov_volume["y_min"] + pov_volume["y_max"]) / 2
    m.pose.position.z = (pov_volume["z_min"] + pov_volume["z_max"]) / 2
    m.pose.orientation.w = 1.0
    m.scale.x = pov_volume["x_max"] - pov_volume["x_min"]
    m.scale.y = pov_volume["y_max"] - pov_volume["y_min"]
    m.scale.z = pov_volume["z_max"] - pov_volume["z_min"]
    m.color.r, m.color.g, m.color.b, m.color.a = 1.0, 0.2, 0.2, 0.15
    m.lifetime = Duration(seconds=0).to_msg()
    ma = MarkerArray(); ma.markers.append(m)
    considered_volume_pub.publish(ma)


def publish_persistent_centroids(node, wm, persistent_centroids_pub=None):
    _publish_centroid_markers(node, wm.persistent_perceptions, persistent_centroids_pub,
                              "persistent_centroids", (0.0, 1.0, 0.0, 1.0),
                              prefer_fused=True)


def publish_uncertain_centroids(node, uncertain_objects, uncertain_centroids_pub):
    _publish_centroid_markers(node, uncertain_objects, uncertain_centroids_pub,
                              "uncertain_centroids", (1.0, 0.6, 0.0, 1.0), label_suffix="[?]")

def _publish_bbox_markers(node, objects, pub, ns, color, prefer_fused=False):
    if not pub:
        return
    ma = MarkerArray()
    for i, obj in enumerate(objects):
        bbox = (getattr(obj, "fused_bbox", None) if prefer_fused else None) or obj.bbox
        if bbox is None:
            continue
        obj_stamp = getattr(obj, "last_perception_time", None)
        stamp = _stamp_from_seconds(obj_stamp) if obj_stamp else node.get_clock().now().to_msg()
        cx, cy, cz = _centroid_from_bbox(bbox)
        m = _make_marker(world_frame(), stamp, ns, i * 2, Marker.CUBE, None, color, (cx, cy, cz))
        m.scale.x = bbox["x_max"] - bbox["x_min"]
        m.scale.y = bbox["y_max"] - bbox["y_min"]
        m.scale.z = bbox["z_max"] - bbox["z_min"]
        ma.markers.append(m)
    if ma.markers:
        pub.publish(ma)


def publish_persistent_bboxes(node, wm, persistent_bboxes_pub=None):
    _publish_bbox_markers(node, wm.persistent_perceptions, persistent_bboxes_pub,
                          "persistent_bboxes", (0.0, 1.0, 0.0, 0.3), prefer_fused=True)


def publish_uncertain_bboxes(node, uncertain_objects, uncertain_bbox_pub):
    _publish_bbox_markers(node, uncertain_objects, uncertain_bbox_pub,
                          "uncertain_bboxes", (1.0, 0.6, 0.0, 0.4))


# ── VLM ──────────────────────────────────────────────────────────────────────
'''
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
api_path = os.path.join(script_dir, "api_1.txt")

with open(api_path, "r") as f:
    groq_api_key = f.read().strip()

groq_client = Groq(api_key=groq_api_key)

def vlm_call(prompt, encoded_image):
    resp = groq_client.chat.completions.create(
        # Ho cambiato il modello qui sotto. 
        # "llama-3.2-90b-vision-preview" also works if the quota allows
        model="llama-3.2-11b-vision-preview", 
        messages=[{"role": "user", "content": [
            {"type": "text",      "text": prompt},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{encoded_image}"}}
        ]}]
    )
    return resp.choices[0].message.content

'''

_client = None


_LOCAL_HOSTS = frozenset({"localhost", "127.0.0.1", "::1", "0.0.0.0", "host.docker.internal"})

_KEY_SOURCES = ("config vlm.api_key", "$OPENAI_API_KEY", "$REGOLO_API_KEY",
                "$OPENROUTER_API_KEY", "the legacy api.txt beside cv_utils.py")


def _endpoint_is_local(base_url):
    """A local server ignores the key entirely; a remote one does not.

    Split out and named so the distinction is testable without constructing a client, and so
    the reason the placeholder survives is written down rather than inferred from a string.
    """
    try:
        host = urllib.parse.urlparse(str(base_url)).hostname
    except Exception:
        return False
    return host in _LOCAL_HOSTS


def _resolve_api_key():
    key = (
        CFG.get("vlm", {}).get("api_key")
        or os.environ.get("OPENAI_API_KEY")
        or os.environ.get("REGOLO_API_KEY")
        or os.environ.get("OPENROUTER_API_KEY", "")
    )
    if not key:
        legacy = os.path.join(os.path.dirname(__file__), "api.txt")
        if os.path.exists(legacy):
            key = open(legacy).read().strip()
    if key:
        return key

    # GA-136. This chain ended `return key or "ollama"`, so with no key anywhere the client
    # was handed the literal string "ollama" as its credential.
    #
    # The placeholder is NOT arbitrary and is kept: the DEFAULT base_url is
    # http://localhost:11434/v1 and config.py:18 documents the fallback -- "else 'ollama'
    # (local server ignores it)". Deleting it outright would break the documented default.
    #
    # What it must not do is travel to a REMOTE endpoint. Every config that has actually run
    # points at https://api.regolo.ai/v1, which authenticates: there, "ollama" produces a 401
    # that reads as "the key you configured was rejected" when the truth is that no key was
    # ever found. The misdirection is the defect, not the placeholder -- a wrong credential
    # and an absent one are different faults and were indistinguishable at the call site.
    base_url = CFG.get("vlm", {}).get("base_url", "")
    if _endpoint_is_local(base_url):
        return "ollama"
    raise RuntimeError(
        f"no VLM API key found for {base_url!r}, which is not a local endpoint. "
        f"Searched, in order: {', '.join(_KEY_SOURCES)}. Refusing to send the literal "
        f'string "ollama" as a credential: the 401 it produces reads as a rejected key '
        f"rather than a missing one."
    )


def _vlm_client():
    # Lazy: no api.txt read and no client construction at import time.
    global _client
    if _client is None:
        _client = OpenAI(
            base_url=CFG["vlm"]["base_url"],
            api_key=_resolve_api_key(),
            timeout=CFG["vlm"]["timeout"],
            max_retries=0,  # vlm_call owns retries (cfg vlm.retries); SDK backoff just adds latency
        )
    return _client


def _emit_vlm_trace(trace_fn, event):
    """Send request telemetry to the owning node without making logging fatal.

    ``cv_utils`` is also used by small offline tests and by the legacy perception node, so
    telemetry is an optional callback rather than a hard dependency on an rclpy logger.
    A broken logger must never turn a successful model response into a failed perception
    cycle.
    """
    if trace_fn is None:
        return
    try:
        trace_fn(event)
    except Exception:
        pass


def _header_value(headers, name):
    """Read an HTTP header from both dicts and case-insensitive header mappings."""
    if not headers:
        return None
    wanted = name.lower()
    try:
        for key, value in headers.items():
            if str(key).lower() == wanted:
                return value
    except Exception:
        return None
    return None


def vlm_call(prompt, encoded_image, timeout=None, response_format=None, image_detail=None,
             trace_fn=None, request_kind="vlm"):
    """One VLM round-trip. Transport failures (timeout, malformed envelope)
    retry up to cfg vlm.retries times, then raise — never silently degraded.
    A well-formed response is returned as-is (may be empty: a semantic outcome
    the callers already handle).

    `timeout` (seconds) bounds THIS call; None keeps the client's cfg vlm.timeout.
    GA-303: the crop describer passes cfg vlm.crop_timeout here.

    `response_format` and `image_detail` are optional so existing label/crop callers
    keep byte-for-byte request semantics. The unified scene caller uses both to ask
    the configured OpenAI-compatible endpoint for one strict JSON object containing
    the boxes and attributes for the entire frame.
    """
    last_err = None
    attempts_total = CFG["vlm"]["retries"] + 1
    call_started = time.perf_counter()
    for attempt in range(CFG["vlm"]["retries"] + 1):
        # GA-288. BACKOFF, because there was none. Run 20260903_135823 died at 18 cycles on
        # a DNS blip ("Temporary failure in name resolution") that lasted under a second:
        # three attempts fired back-to-back inside that second, all failed, and the raise
        # below propagated through the timer callback into executor.spin(). 1 s / 2 s / 4 s
        # lets a transient fault pass; a real outage still exhausts the attempts and raises.
        backoff_s = min(2 ** (attempt - 1), 8) if attempt else 0.0
        if backoff_s:
            time.sleep(backoff_s)
        attempt_started = time.perf_counter()
        try:
            image_url = {"url": f"data:image/png;base64,{encoded_image}"}
            if image_detail is not None:
                image_url["detail"] = image_detail
            request = {
                "model": CFG["vlm"]["model"],
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": image_url,
                            }
                        ],
                    }
                ],
            }
            if timeout is not None:
                request["timeout"] = timeout
            if response_format is not None:
                request["response_format"] = response_format
            # Pass provider-specific completion options through the OpenAI-compatible
            # client.  In particular, Qwen-style endpoints otherwise ignore the YAML
            # `enable_thinking: false` setting and may spend most of the request budget
            # generating hidden reasoning before returning the structured scene result.
            request.update(vlm_completion_kwargs())
            agent = _vlm_client().chat.completions.create(**request)
            if not getattr(agent, "choices", None) or agent.choices[0].message is None:
                raise RuntimeError(f"malformed VLM response: {agent!r:.200}")
            message = agent.choices[0].message
            refusal = getattr(message, "refusal", None)
            if refusal:
                raise RuntimeError(f"VLM refused the image request: {refusal}")
            if not message.content:
                raise RuntimeError("VLM returned an empty response")

            attempt_ms = (time.perf_counter() - attempt_started) * 1000.0
            usage = getattr(agent, "usage", None)
            _emit_vlm_trace(trace_fn, {
                "request_kind": request_kind,
                "status": "ok",
                "attempt": attempt + 1,
                "attempts_total": attempts_total,
                # This is the client-observed request/response round trip. Regolo does not
                # currently expose a separate server-compute duration in the response.
                "request_ms": round(attempt_ms, 1),
                "call_ms": round((time.perf_counter() - call_started) * 1000.0, 1),
                "backoff_ms": round(backoff_s * 1000.0, 1),
                "model": CFG["vlm"]["model"],
                "prompt_chars": len(prompt),
                "image_b64_chars": len(encoded_image),
                "image_detail": image_detail,
                "structured_response": response_format is not None,
                "prompt_tokens": getattr(usage, "prompt_tokens", None),
                "completion_tokens": getattr(usage, "completion_tokens", None),
                "total_tokens": getattr(usage, "total_tokens", None),
                "request_id": getattr(agent, "_request_id", None),
            })
            return message.content
        except Exception as e:
            attempt_ms = (time.perf_counter() - attempt_started) * 1000.0
            response = getattr(e, "response", None)
            headers = getattr(response, "headers", None)
            _emit_vlm_trace(trace_fn, {
                "request_kind": request_kind,
                "status": "error",
                "attempt": attempt + 1,
                "attempts_total": attempts_total,
                "request_ms": round(attempt_ms, 1),
                "call_ms": round((time.perf_counter() - call_started) * 1000.0, 1),
                "backoff_ms": round(backoff_s * 1000.0, 1),
                "model": CFG["vlm"]["model"],
                "prompt_chars": len(prompt),
                "image_b64_chars": len(encoded_image),
                "image_detail": image_detail,
                "structured_response": response_format is not None,
                "http_status": getattr(e, "status_code", None),
                "retry_after": _header_value(headers, "retry-after"),
                "request_id": _header_value(headers, "x-request-id"),
                "error_type": type(e).__name__,
                "error": str(e)[:300],
            })
            last_err = e
    last_detail = f"{type(last_err).__name__}: {str(last_err)[:240]}" if last_err else "unknown"
    raise RuntimeError(
        f"VLM unreachable after {attempts_total} attempts; last_error={last_detail}"
    ) from last_err

def numpy_to_base64(img, fmt='.png'):
    _, buf = cv2.imencode(fmt, img)
    return base64.b64encode(buf).decode('utf-8')


def create_pointcloud2_msg(points, frame_id):
    """Crea un messaggio PointCloud2 colorato partendo da una lista [x, y, z, r, g, b]"""
    from std_msgs.msg import Header
    fields = [
        PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
        PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
        PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
        PointField(name='rgb', offset=12, datatype=PointField.UINT32, count=1),
    ]
    packed_points = []
    for p in points:
        x, y, z, r, g, b = p
        rgb = (int(r) << 16) | (int(g) << 8) | int(b)
        packed_points.append([x, y, z, rgb])
        
    header = Header()
    header.frame_id = frame_id
    return point_cloud2.create_cloud(header, fields, packed_points)
