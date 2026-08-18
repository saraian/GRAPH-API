#!/usr/bin/env python3
import os
import numpy as np

import perception_config
from sensor_msgs.msg import CameraInfo
from visualization_msgs.msg import Marker, MarkerArray
from matplotlib.colors import to_rgb
import cv2
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2
from std_msgs.msg import Header, String, ColorRGBA
from sensor_msgs.msg import PointField
import json
from geometry_msgs.msg import Point
from utils import statistical_outlier_removal, get_distinct_color
import struct
from openai import OpenAI
import base64
from rclpy.duration import Duration
from rclpy.time import Time
from rclpy.duration import Duration as ROS2Duration
from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy
import ollama
file_path = os.path.abspath(__file__)
from groq import Groq
import time
import itertools


def _get_R_and_T(trans):
    t = trans.transform
    T = np.array([t.translation.x, t.translation.y, t.translation.z])
    qx, qy, qz, qw = t.rotation.x, t.rotation.y, t.rotation.z, t.rotation.w
    R = 2 * np.array([
        [qw**2 + qx**2 - 0.5,  qx*qy - qw*qz,  qw*qy + qx*qz],
        [qw*qz + qx*qy,        qw**2 + qy**2 - 0.5,  qy*qz - qw*qx],
        [qx*qz - qw*qy,        qw*qx + qy*qz,  qw**2 + qz**2 - 0.5]
    ])
    return R, T


def _transform_point_xyz(pt_xyz, source_frame, target_frame, timeout=1.0, node=None, tf_buffer=None):
    if target_frame == source_frame:
        return np.array(pt_xyz).reshape(3)

    if node is not None:
        trans = getattr(node, 'current_transforms', {}).get((source_frame, target_frame))
        if trans:
            R, T = _get_R_and_T(trans)
            return R.dot(np.array(pt_xyz)) + T

    tf_buffer = tf_buffer or getattr(node, 'tf_buffer', None)
    if tf_buffer is None:
        raise ValueError("tf_buffer or node with tf_buffer required")

    # MODIFICA QUI: Usiamo Time() vuoto per chiedere "l'ultima trasformata disponibile"
    # invece di Time(seconds=0) che in simulazione punta al passato e fallisce.
    try:
        trans = tf_buffer.lookup_transform(
            target_frame, 
            source_frame,
            Time(),  
            timeout=ROS2Duration(seconds=timeout)
        )
    except Exception as e:
        if node:
            node.get_logger().error(f"Errore TF critico: impossibile trasformare da {source_frame} a {target_frame}. Errore: {e}")
        raise e

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


def mask_list_to_pointcloud2(masks, depth_image, camera_info, node, labels=None,
                              topic="/pcl_objects", max_points_per_obj=20000,
                              remove_outliers=True, sor_k=30, sor_std=1.0,
                              publisher=None, labels_publisher=None):
    if not isinstance(camera_info, CameraInfo):
        raise TypeError('camera_info must be CameraInfo')

    labels       = labels or [f"obj_{i}" for i in range(len(masks))]
    fx, fy, cx, cy = camera_info.k[0], camera_info.k[4], camera_info.k[2], camera_info.k[5]
    #camera_frame = "head_front_camera_color_optical_frame"
    camera_frame= "habitat_camera"
    current_points, id_to_label = [], {}

    for obj_idx, mask in enumerate(masks):
        mask = np.array(mask)
        mask2d = mask[:, :, 0] if mask.ndim == 3 else mask
        ys, xs = np.nonzero(mask2d.astype(bool))

        if len(xs) == 0:
            continue
        if len(xs) > max_points_per_obj:
            idx = np.linspace(0, len(xs)-1, max_points_per_obj).astype(int)
            xs, ys = xs[idx], ys[idx]

        # Depth values + bounds
        zs = depth_image[ys, xs].astype(float)
        valid = (zs > 0) & np.isfinite(zs)
        if not valid.any():
            node.get_logger().warn(f"Mask {obj_idx}: no valid depth, skip")
            continue

        depth_min, depth_max = _depth_bounds(zs[valid])
        in_range = valid & (zs >= depth_min) & (zs <= depth_max)
        xs, ys, zs = xs[in_range], ys[in_range], zs[in_range]

        if len(xs) == 0:
            continue

        # Project to camera-frame 3D
        Xs = (xs - cx) * zs / fx
        Ys = (ys - cy) * zs / fy
        points = np.column_stack([Xs, Ys, zs])

        # Statistical outlier removal
        if remove_outliers and len(points) > 20:
            mask_sor = statistical_outlier_removal(points, k=sor_k, std_ratio=sor_std)
            points   = points[mask_sor]

        if len(points) == 0:
            node.get_logger().warn(f"Obj {obj_idx}: all points removed by filter")
            continue

        # Color + unique ID
        unique_id  = node.pcl_object_id_counter
        rgb_packed = _pack_rgb(get_distinct_color(unique_id))

        # Transform to map frame (vectorised where possible)
        mapped = []
        for pt in points:
            try:
                p = _transform_point_xyz(pt, camera_frame, "map", node=node)
                mapped.append((*p, rgb_packed, unique_id))
            except Exception as e:
                node.get_logger().warn(f"Transform failed for point {pt}: {e}")

        current_points.extend(mapped)
        id_to_label[unique_id] = labels[obj_idx]
        node.pcl_object_id_counter += 1

    if not current_points:
        node.get_logger().warn("mask_list_to_pointcloud2: no points to publish")
        return

    # Build and publish PointCloud2
    fields = [
        PointField(name='x',         offset=0,  datatype=PointField.FLOAT32, count=1),
        PointField(name='y',         offset=4,  datatype=PointField.FLOAT32, count=1),
        PointField(name='z',         offset=8,  datatype=PointField.FLOAT32, count=1),
        PointField(name='rgb',       offset=12, datatype=PointField.FLOAT32, count=1),
        PointField(name='object_id', offset=16, datatype=PointField.INT32,   count=1),
    ]
    header = Header(stamp=node.get_clock().now().to_msg(), frame_id="map")
    cloud_msg = point_cloud2.create_cloud(header, fields, current_points)

    qos = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
    (publisher or node.create_publisher(PointCloud2, topic, qos)).publish(cloud_msg)

    try:
        lbl_msg = String(); lbl_msg.data = json.dumps(id_to_label)
        (labels_publisher or node.create_publisher(String, topic+"_labels", qos)).publish(lbl_msg)
    except Exception:
        node.get_logger().warn("Could not publish pcl labels")

    node.get_logger().info(f"Published {len(current_points)} points, {len(id_to_label)} objects")

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
    camera_frame = "habitat_camera"
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
        mask   = np.array(mask)
        mask2d = mask[:, :, 0] if mask.ndim == 3 else mask
        ys, xs = np.nonzero(mask2d.astype(bool))

        if len(xs) == 0:
            continue
        if len(xs) > max_points_per_obj:
            idx = np.linspace(0, len(xs)-1, max_points_per_obj).astype(int)
            xs, ys = xs[idx], ys[idx]

        # Color
        col = palette[obj_idx % len(palette)]
        rgb_packed = struct.unpack('f', struct.pack('I',
            (int(col[0]*255) << 16) | (int(col[1]*255) << 8) | int(col[2]*255)))[0]

        # Depth filter
        zs    = depth_image[ys, xs].astype(float)
        valid = (zs > 0) & np.isfinite(zs)
        if not valid.any():
            continue
        depth_min, depth_max = _depth_bounds(zs[valid])
        keep = valid & (zs >= depth_min) & (zs <= depth_max)
        xs, ys, zs = xs[keep], ys[keep], zs[keep]

        if len(xs) == 0:
            continue

        # Project to 3D
        Xs = (xs - cx) * zs / fx
        Ys = (ys - cy) * zs / fy
        pts = np.column_stack([Xs, Ys, zs])

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
                            frame_id="map", topic="/centroid_markers", marker_scale=0.06):
    if centroid_marker_pub is None:
        centroid_marker_pub = node.create_publisher(
            MarkerArray, topic, QoSProfile(depth=10, durability=DurabilityPolicy.TRANSIENT_LOCAL))

    if not hasattr(node, '_centroid_marker_id_counter'):
        node._centroid_marker_id_counter = 0

    ma    = MarkerArray()
    stamp = node.get_clock().now().to_msg()

    for i, point in enumerate(points):
        if point is None:
            continue
        uid   = node._centroid_marker_id_counter
        label = labels[i] if labels and i < len(labels) else f"obj_{i}"
        m = Marker()
        m.header.frame_id = frame_id
        m.header.stamp    = stamp
        m.ns, m.id, m.type, m.action = "centroid_spheres", uid, Marker.SPHERE, Marker.ADD
        m.pose.position.x, m.pose.position.y, m.pose.position.z = point
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
  
def mask_list_to_centroid_and_bbox(mask_list, labels, depth_image, camera_info, node,
                                    bbox_marker_pub=None, centroid_marker_pub=None,
                                    max_points_per_obj=20000, remove_outliers=True,
                                    sor_k=30, sor_std=1.0):
    fx, fy, cx, cy = camera_info.k[0], camera_info.k[4], camera_info.k[2], camera_info.k[5]
    #camera_frame   = "head_front_camera_color_optical_frame"
    camera_frame = "habitat_camera"
    centroids_3d, bboxes_3d, all_markers = [], [], []

    if not hasattr(node, '_bbox_marker_id_counter'):
        node._bbox_marker_id_counter = 0

    for mask_idx, mask in enumerate(mask_list):
        label = labels[mask_idx] if labels and mask_idx < len(labels) else f"obj_{mask_idx}"

        mask  = np.array(mask)
        mask2d = mask[:, :, 0] if mask.ndim == 3 else mask
        ys, xs = np.nonzero(mask2d.astype(bool))

        if len(xs) == 0:
            centroids_3d.append(None); bboxes_3d.append(None); continue

        if len(xs) > max_points_per_obj:
            idx = np.linspace(0, len(xs)-1, max_points_per_obj).astype(int)
            xs, ys = xs[idx], ys[idx]

        # Depth filter
        zs    = depth_image[ys, xs].astype(float)
        valid = (zs > 0) & np.isfinite(zs)
        if not valid.any():
            centroids_3d.append(None); bboxes_3d.append(None); continue

        depth_min, depth_max = _depth_bounds(zs[valid])
        keep = valid & (zs >= depth_min) & (zs <= depth_max)
        xs, ys, zs = xs[keep], ys[keep], zs[keep]

        if len(xs) == 0:
            centroids_3d.append(None); bboxes_3d.append(None); continue

        # Project to 3D
        pts = np.column_stack([(xs - cx) * zs / fx, (ys - cy) * zs / fy, zs])

        if remove_outliers and len(pts) > 20:
            pts = pts[statistical_outlier_removal(pts, k=sor_k, std_ratio=sor_std)]

        if len(pts) == 0:
            centroids_3d.append(None); bboxes_3d.append(None); continue

        # Centroid
        centroid = np.mean(pts, axis=0)
        centroids_3d.append(tuple(centroid))

        if centroid_marker_pub is not None:
            c_map = _transform_point_xyz(tuple(centroid), camera_frame, "map", node=node)
            points_list_to_rviz_3d([c_map], node, centroid_marker_pub=centroid_marker_pub,
                                   labels=[label], frame_id="map", marker_scale=0.05)

        # BBox corners → map frame
        mins, maxs = pts.min(axis=0), pts.max(axis=0)
        corners = list(itertools.product(*zip(mins, maxs)))  # 8 corners
        try:
            corners_map = np.array([_transform_point_xyz(c, camera_frame, "map", node=node) for c in corners])
            bbox_dict = {f"{ax}_{k}": float(fn(corners_map[:, i]))
                        for i, ax in enumerate("xyz")
                        for k, fn in [("min", np.min), ("max", np.max)]}
            bboxes_3d.append(bbox_dict)
        except Exception as e:
            node.get_logger().warn(f"BBox transform failed for mask {mask_idx}: {e}, using camera frame")
            bbox_dict = {f"{k}_{ax}": float(fn(pts[:, i]))
                        for i, ax in enumerate("xyz")
                        for k, fn in [("min", np.min), ("max", np.max)]}
            bboxes_3d.append(bbox_dict)
            continue

        # Bbox marker
        m = Marker()
        m.header.frame_id = "map"
        m.header.stamp    = node.get_clock().now().to_msg()
        m.ns, m.id        = "bbox_markers", node._bbox_marker_id_counter
        m.type, m.action  = Marker.SPHERE_LIST, Marker.ADD
        m.scale.x = m.scale.y = m.scale.z = 0.02
        m.color   = get_distinct_color(node._bbox_marker_id_counter)
        m.lifetime = Duration(seconds=0).to_msg()
        m.points  = [Point(x=float(p[0]), y=float(p[1]), z=float(p[2])) for p in corners_map]
        all_markers.append(m)
        node._bbox_marker_id_counter += 1

    if bbox_marker_pub is not None and all_markers:
        ma = MarkerArray(); ma.markers = all_markers
        bbox_marker_pub.publish(ma)
        node.get_logger().info(f"Published {len(all_markers)} bbox markers on /bbox_marker")

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


def _publish_centroid_markers(node, objects, pub, ns, color, label_suffix=""):
    if not pub:
        return
    ma, stamp = MarkerArray(), node.get_clock().now().to_msg()
    for i, obj in enumerate(objects):
        if obj.bbox is None:
            continue
        cx, cy, cz = _centroid_from_bbox(obj.bbox)
        ma.markers.append(_make_marker("map", stamp, ns, i, Marker.SPHERE, 0.08, color, (cx, cy, cz)))
        ma.markers.append(_make_text_marker("map", stamp, ns+"_labels", i+10000,
                                            obj.label.replace(' ', '') + label_suffix, (cx, cy, cz)))
    if ma.markers:
        pub.publish(ma)


def publish_pov_volume(node, pov_volume, considered_volume_pub=None):
    m = Marker()
    m.header.frame_id = "map"
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
                              "persistent_centroids", (0.0, 1.0, 0.0, 1.0))


def publish_uncertain_centroids(node, uncertain_objects, uncertain_centroids_pub):
    _publish_centroid_markers(node, uncertain_objects, uncertain_centroids_pub,
                              "uncertain_centroids", (1.0, 0.6, 0.0, 1.0), label_suffix="[?]")

def _publish_bbox_markers(node, objects, pub, ns, color):
    if not pub:
        return
    ma, stamp = MarkerArray(), node.get_clock().now().to_msg()
    for i, obj in enumerate(objects):
        if obj.bbox is None:
            continue
        cx, cy, cz = _centroid_from_bbox(obj.bbox)
        m = _make_marker("map", stamp, ns, i * 2, Marker.CUBE, None, color, (cx, cy, cz))
        m.scale.x = obj.bbox["x_max"] - obj.bbox["x_min"]
        m.scale.y = obj.bbox["y_max"] - obj.bbox["y_min"]
        m.scale.z = obj.bbox["z_max"] - obj.bbox["z_min"]
        ma.markers.append(m)
    if ma.markers:
        pub.publish(ma)


def publish_persistent_bboxes(node, wm, persistent_bboxes_pub=None):
    _publish_bbox_markers(node, wm.persistent_perceptions, persistent_bboxes_pub,
                          "persistent_bboxes", (0.0, 1.0, 0.0, 0.3))


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
        # Puoi usare anche "llama-3.2-90b-vision-preview" se hai abbastanza quota
        model="llama-3.2-11b-vision-preview", 
        messages=[{"role": "user", "content": [
            {"type": "text",      "text": prompt},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{encoded_image}"}}
        ]}]
    )
    return resp.choices[0].message.content

'''
class VLMUnavailable(RuntimeError):
    """The provider could not be reached or answered with a malformed envelope.

    Deliberately distinct from "the model had nothing to say". A connection
    problem is not a semantic result, and collapsing the two hides outages.
    """


_client = None


def _get_client():
    """Build the client on first use, not at import.

    Reading the key at module scope means a missing file takes down every
    importer of cv_utils, including perception.py, before any function runs and
    with no way for a caller to guard it. Anyone who never makes a VLM call now
    never needs a key.
    """
    global _client
    if _client is not None:
        return _client
    key = os.environ.get(perception_config.get("vlm", "api_key_env") or "OPENAI_API_KEY")
    if not key:
        key_file = perception_config.get("vlm", "api_key_file") or \
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "api.txt")
        if os.path.isfile(key_file):
            with open(key_file) as f:
                key = f.read().strip()
    if not key:
        raise VLMUnavailable(
            "No API key. Set the env var named by vlm.api_key_env, or point "
            "vlm.api_key_file at a file, or disable descriptions with "
            "vlm.enabled: false in perception_config.yaml.")
    kwargs = {"api_key": key, "timeout": perception_config.get("vlm", "timeout_s"),
              "max_retries": 0}
    base_url = perception_config.get("vlm", "base_url")
    if base_url:
        kwargs["base_url"] = base_url
    _client = OpenAI(**kwargs)
    return _client


def vlm_call(prompt, encoded_image):
    """Describe an image region. Returns the text, or None if the model had none.

    Raises VLMUnavailable when the provider is unreachable or returns an
    envelope we cannot read, after the configured retries. That case must stay
    loud: it is an outage, not an empty answer.
    """
    if not perception_config.get("vlm", "enabled"):
        return None
    retries = int(perception_config.get("vlm", "max_retries"))
    last = None
    for attempt in range(retries + 1):
        try:
            agent = _get_client().chat.completions.create(
                model=perception_config.get("vlm", "model"),
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/png;base64,{encoded_image}"}
                            }
                        ],
                    }
                ]
            )
        except VLMUnavailable:
            raise
        except Exception as e:              # timeout, connection, provider error
            last = e
            continue
        choices = getattr(agent, "choices", None)
        if not choices or getattr(choices[0], "message", None) is None:
            # A well-formed reply always carries choices. Missing ones mean a
            # moderation block or an error envelope, i.e. transport, not content.
            last = VLMUnavailable(f"provider returned no choices: {agent!r}"[:300])
            continue
        content = choices[0].message.content
        # Well-formed and empty is a real answer: the model had nothing to add.
        return content if content and content.strip() else None
    raise VLMUnavailable(f"VLM call failed after {retries + 1} attempt(s): {last}")

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
