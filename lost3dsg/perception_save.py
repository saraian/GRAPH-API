#!/usr/bin/env python3

# Suppress warnings
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
import os
import json
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow warnings
import rclpy
from rclpy.node import Node
from rclpy.duration import Duration
from rclpy.qos import QoSProfile, DurabilityPolicy
from sensor_msgs.msg import Image, JointState
from std_msgs.msg import String, Header
from datetime import datetime
from sensor_msgs.msg import PointCloud2
import tf2_ros
from cv_bridge import CvBridge
import numpy as np
import cv2
from cv_utils import _clear_markers
import time
from dataclasses import dataclass
from typing import Tuple
from collections import Counter
from matplotlib.colors import to_rgb
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging
from datetime import datetime
import re
import itertools

# Project imports
from cv_utils import *
import utils
from utils import draw_detections, apply_nms
from models import DINO, VitSam, OWLv2
from lost3dsg.msg import Centroid, CentroidArray, Bbox3d, Bbox3dArray
from object_info import Object
from world_model import wm
from lost3dsg.msg import ObjectDescription, ObjectDescriptionArray
from std_msgs.msg import Bool
import torch
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
from cv_utils import _transform_point_xyz, init_bbox_publisher

def compute_fov_volume_from_depth(depth_image, camera_info, node, depth_threshold=4.0):
    """
    Compute the 3D FOV volume from the entire depth image.

    Projects all valid depth pixels to 3D, transforms to map frame,
    then takes min/max to get the bounding volume.

    Args:
        depth_image: The depth image (numpy array)
        camera_info: Camera calibration info (CameraInfo message)
        node: ROS node for TF lookups and logging
        depth_threshold: Maximum depth to consider (meters)

    Returns:
        dict: FOV volume with x_min, x_max, y_min, y_max, z_min, z_max in map frame
              or None if computation fails
    """
    from cv_utils import _transform_point_xyz

    try:
        # Get camera intrinsics
        K = camera_info.k
        fx, fy, cx, cy = K[0], K[4], K[2], K[5]

        h, w = depth_image.shape[:2]

        # Convert depth to meters if needed
        if depth_image.dtype == np.uint16:
            depth_m = depth_image.astype(np.float32) / 1000.0
        else:
            depth_m = depth_image.astype(np.float32)

        # Get valid depth mask - FILTER points beyond 1.8m
        MAX_DEPTH_FOR_FOV = 1.8  # Maximum depth to consider for FOV (meters)
        valid_mask = (depth_m > 0.1) & (depth_m < min(depth_threshold, MAX_DEPTH_FOR_FOV)) & np.isfinite(depth_m)

        if not np.any(valid_mask):
            node.get_logger().warn("No valid depth values found for FOV computation")
            return None

        # Get coordinates of valid pixels
        ys, xs = np.nonzero(valid_mask)
        zs = depth_m[valid_mask]

        # Project all valid pixels to 3D in camera frame
        Xs = (xs - cx) * zs / fx
        Ys = (ys - cy) * zs / fy

        # Get min/max in camera frame
        x_min_cam, x_max_cam = float(Xs.min()), float(Xs.max())
        y_min_cam, y_max_cam = float(Ys.min()), float(Ys.max())
        z_min_cam, z_max_cam = float(zs.min()), float(zs.max())

        # Transform the 8 corners of the camera-frame box to map frame
        camera_frame = camera_info.header.frame_id

        corners_camera = [
            (x_min_cam, y_min_cam, z_min_cam),
            (x_max_cam, y_min_cam, z_min_cam),
            (x_min_cam, y_max_cam, z_min_cam),
            (x_max_cam, y_max_cam, z_min_cam),
            (x_min_cam, y_min_cam, z_max_cam),
            (x_max_cam, y_min_cam, z_max_cam),
            (x_min_cam, y_max_cam, z_max_cam),
            (x_max_cam, y_max_cam, z_max_cam),
        ]

        try:
            corners_map = [
                _transform_point_xyz(p, camera_frame, "map", node=node)
                for p in corners_camera
            ]

            corners_map_array = np.array(corners_map)

            fov_map = {
                "x_min": float(corners_map_array[:, 0].min()),
                "x_max": float(corners_map_array[:, 0].max()),
                "y_min": float(corners_map_array[:, 1].min()),
                "y_max": float(corners_map_array[:, 1].max()),
                "z_min": float(corners_map_array[:, 2].min()),
                "z_max": float(corners_map_array[:, 2].max())
            }

            node.get_logger().info(f"FOV volume (map frame): X[{fov_map['x_min']:.2f}, {fov_map['x_max']:.2f}], "
                                   f"Y[{fov_map['y_min']:.2f}, {fov_map['y_max']:.2f}], "
                                   f"Z[{fov_map['z_min']:.2f}, {fov_map['z_max']:.2f}]")

            return fov_map

        except Exception as e:
            node.get_logger().warn(f"Could not transform FOV to map frame: {e}")
            return None

    except Exception as e:
        node.get_logger().error(f"Error computing FOV volume: {e}")
        return None


# ── Project paths & file logger ──────────────────────────────────────────────
_dir = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(_dir, "../.."))

log_dir = os.path.join(PROJECT_ROOT, "output")
os.makedirs(log_dir, exist_ok=True)

module_logger = logging.getLogger('perception_module')
module_logger.setLevel(logging.DEBUG)
_fh = logging.FileHandler(os.path.join(log_dir, f"perception_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"))
_fh.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(message)s'))
module_logger.addHandler(_fh)

_UNKNOWN_FIELDS = {"description", "color", "material", "shape"}
@dataclass
class Detection:
    bbox: Tuple[float, float, float, float]
    label: str
    score: float
    mask: np.ndarray


class DetectObjects(Node):

    _LOG_METHODS = {
        'info':  ('info',    'info'),
        'warn':  ('warn',    'warning'),
        'error': ('error',   'error'),
        'debug': ('debug',   'debug'),
    }

    def __init__(self):
        super().__init__('detection_node')

        self.file_logger = module_logger
        self.file_logger.info("=== DetectObjects Node Initialized ===")

        self.bridge = CvBridge()
        self.detector = OWLv2()
        self.vitsam = VitSam(utils.ENCODER_VITSAM_PATH, utils.DECODER_VITSAM_PATH)
        self.COLORS = ['red', 'green', 'blue', 'magenta', 'gray', 'yellow'] * 3

        self.tf_buffer = tf2_ros.Buffer(cache_time=Duration(seconds=30.0))
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        qos_l = QoSProfile(depth=10, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        qos   = QoSProfile(depth=10)

        self.pub_image          = self.create_publisher(Image,                "/image_with_bb",        qos_l)
        self.pub_crop           = self.create_publisher(Image,                "/cropped_image",         qos)
        self.centroid_pub       = self.create_publisher(CentroidArray,        "/centroids_custom",      qos_l)
        self.bbox_pub           = self.create_publisher(Bbox3dArray,          '/bbox_3d',               qos)
        self.movement_detected_pub = self.create_publisher(Bool,              '/robot_movement_detected', qos)
        self.pub_object_descriptions = self.create_publisher(ObjectDescriptionArray, '/object_descriptions', qos)
        self.pcl_objects_pub    = self.create_publisher(PointCloud2,          '/pcl_objects',           qos_l)
        self.pcl_objects_labels_pub = self.create_publisher(String,           '/pcl_objects_labels',    qos_l)

        self.bbox_marker_pub, self.centroid_marker_pub = init_bbox_publisher(self)
        self.camera_data = utils.SyncedCameraData(self, sync_tolerance_ms=150)

        self.create_subscription(JointState, "joint_states", self.joint_callback, qos)
        self.head_joints       = ['head_1_joint', 'head_2_joint']
        self.base_joints       = ['wheel_left_joint', 'wheel_right_joint']
        self.position_threshold = 0.005
        self.last_joint_positions = {}

        # Robot movement state
        self.is_stationary             = True
        self.time_stationary_start     = None
        self.last_detection_time       = None
        self.first_detection_done      = False
        self.robot_has_moved_once      = False
        self.min_stationary_after_movement = 3.0
        self.processing_interrupted    = False

        # Test-phase manual trigger
        self.manual_trigger_requested  = False
        self.waiting_for_input         = False

        self.object_list               = []
        self.filtered_objects          = []
        self.publish_individual_objects = False
        self.pcl_object_id_counter     = 0
        self.individual_pcl_publishers = {}

        self.clear_accumulated_markers()
        self.get_logger().info(f"Log saved in: {_fh.baseFilename}")

    def log_both(self, level: str, message: str):
        """Log to both ROS logger and file logger."""
        ros_lvl, file_lvl = self._LOG_METHODS.get(level, ('info', 'info'))
        
        # ROS logger - usa metodi diretti invece di getattr
        logger = self.get_logger()
        if ros_lvl == 'debug':
            logger.debug(message)
        elif ros_lvl == 'info':
            logger.info(message)
        elif ros_lvl == 'warn' or ros_lvl == 'warning':
            logger.warn(message)
        elif ros_lvl == 'error':
            logger.error(message)
        elif ros_lvl == 'fatal':
            logger.fatal(message)
        
        # File logger - getattr va bene qui
        getattr(self.file_logger, file_lvl)(message)

    def clear_accumulated_markers(self):
        """Clear centroid/bbox markers and reset their ID counters."""
        for topic, publisher, counter_attr in [
            ("/centroid_markers", self.centroid_marker_pub, '_centroid_marker_id_counter'),
            ("/bbox_marker",      self.bbox_marker_pub,     '_bbox_marker_id_counter'),
        ]:
            try:
                _clear_markers(topic, node=self, publisher=publisher)
            except Exception as e:
                self.get_logger().warn(f"Error clearing {topic}: {e}")
            setattr(self, counter_attr, 0)

        self.get_logger().info("Markers cleared and counters reset")
   
    def color_pcl(self, detections, data):
        """Generate and publish colored PointCloud2 from masks."""
        camera_info = data['camera_info']
        depth_img   = data['depth']

        mask_list = [(det.mask[:, :, 0].astype(np.uint8) * 255) for det in detections]
        labels    = [det.label for det in detections]

        self.log_both("info", f"color_pcl: {len(mask_list)} masks → PointCloud2")

        try:
            mask_list_to_pointcloud2(
                mask_list, depth_img, camera_info,
                node=self, labels=labels, topic="/pcl_objects",
                max_points_per_obj=1500,
                publisher=self.pcl_objects_pub,
                labels_publisher=self.pcl_objects_labels_pub,
            )
        except Exception as e:
            self.get_logger().error(f"Error publishing aggregated PointCloud2: {e}")
            return

        if not self.publish_individual_objects:
            return

        try:
            published = publish_individual_pointclouds_by_id(
                mask_list, depth_img, camera_info,
                node=self, labels=labels, frame_id="map",
                topic_prefix="/pcl_id",
                publishers_dict=self.individual_pcl_publishers,
                id_counter_start=self.pcl_object_id_counter,
                timestamp=camera_info.header.stamp,
            )
            self.pcl_object_id_counter += published
        except Exception as e:
            self.get_logger().error(f"Error in publish_individual_pointclouds_by_id: {e}")


    def run_detection(self, camera_data):
        """Run full detection pipeline: VLM → OWLv2 → NMS → SAM → PCL."""
        self.log_both("info", "=== START DETECTION ===")

        prompt_path = os.path.join(os.path.dirname(file_path), "object_identification_prompt.txt")
        objects_to_identify = vlm_call(
            open(prompt_path).read(),
            numpy_to_base64(camera_data['rgb'])
        )

        objects_to_identify = objects_to_identify.replace('[', '').replace(']', '').replace('"', '').replace('\n', '').strip()

        labels = [l.strip() for l in objects_to_identify.split(',')]
        self.log_both("info", f"VLM labels ({len(labels)}): {labels}")

        self.detector.set_classes(labels)
        bboxs, labels, scores = self.detector.predict(camera_data['rgb'], box_threshold=0.20, text_threshold=0.2)
        self.log_both("info", f"OWLv2 detected {len(labels)} objects: {labels}")

        if len(bboxs) > 0:
            bboxs, labels, scores = apply_nms(bboxs, labels, scores, iou_threshold=0.5)
            self.log_both("info", f"After NMS: {len(labels)} objects: {labels}")

        # ================= INIZIO FILTRO PORTA PIU' VICINA =================
        depth_img = camera_data['depth']
        h, w = depth_img.shape[:2]
        
        closest_door_idx = -1
        min_door_distance = float('inf')
        
        # 1. Trova l'indice della porta con la distanza minima al centro del bounding box
        for i, (bbox, label) in enumerate(zip(bboxs, labels)):
            if "door" in label.lower():
                x0, y0, x1, y1 = bbox
                # Calcola il punto centrale del bounding box
                cx = min(max(int((x0 + x1) / 2), 0), w - 1)
                cy = min(max(int((y0 + y1) / 2), 0), h - 1)
                
                # Leggi la profondità in quel punto
                dist = float(depth_img[cy, cx])
                
                # Ignora i pixel non validi (di solito hanno valore 0 o NaN nei sensori depth)
                if dist <= 0.0 or np.isnan(dist):
                    dist = float('inf')
                    
                if dist < min_door_distance:
                    min_door_distance = dist
                    closest_door_idx = i

        # 2. Ricostruisci le liste eliminando le porte extra
        if closest_door_idx != -1:
            filtered_bboxs, filtered_labels, filtered_scores = [], [], []
            for i, (bbox, label, score) in enumerate(zip(bboxs, labels, scores)):
                if "door" in label.lower():
                    # Mantieni la porta SOLO se è quella più vicina
                    if i == closest_door_idx:
                        filtered_bboxs.append(bbox)
                        filtered_labels.append(label)
                        filtered_scores.append(score)
                else:
                    # Mantieni tutti gli altri oggetti (non porte)
                    filtered_bboxs.append(bbox)
                    filtered_labels.append(label)
                    filtered_scores.append(score)
                    
            # Sovrascrivi le variabili originali
            bboxs = filtered_bboxs
            labels = filtered_labels
            scores = filtered_scores
            self.log_both("info", f"After Door Filter: {len(labels)} objects left.")
        # ================= FINE FILTRO =================

        if not self.is_stationary:
            self.log_both("warn", "Robot moved after OWLv2 — interrupting")
            self.processing_interrupted = True
            return []
            
        detections = []
        for bbox, label_name, score in zip(bboxs, labels, scores):
            masks, _ = self.vitsam(camera_data['rgb'].copy(), bbox)
            for mask in masks:
                mask_np = (mask.cpu().numpy() if hasattr(mask, 'cpu') else np.array(mask)).transpose(1, 2, 0)
                detections.append(Detection(bbox=tuple(bbox), label=label_name, score=score, mask=mask_np))

        self.pcl_object_id_counter = 0
        self.color_pcl(detections, camera_data)
        self.log_both("info", f"Detection complete: {len(detections)} objects")
        return detections


    _UNKNOWN_FIELDS = {"description", "color", "material", "shape"}

    def process_crop_vlm(self, crop_info):
        """Describe a cropped object via VLM, returns structured dict."""
        if crop_info is None:
            return None

        label = crop_info['label']
        prompt_path = os.path.join(os.path.dirname(file_path), "visual_prompt.txt")
        prompt = open(prompt_path).read().strip().replace("{LABEL}", label)

        _default = {k: "unknown" for k in _UNKNOWN_FIELDS} | {"label": label, "json_answer": "{}"}

        try:
            raw = vlm_call(prompt, numpy_to_base64(crop_info['cropped']))
            cleaned = re.sub(r'```json|```', '', raw).strip()
            match = re.search(r'\{.*\}', cleaned, re.DOTALL)
            obj_data = json.loads(match.group(0) if match else '{}').get("objects", [{}])[0]

            return _default | {k: obj_data.get(k, "unknown") for k in _UNKNOWN_FIELDS} | {"json_answer": match.group(0) if match else "{}"}
        except Exception as e:
            self.get_logger().error(f"VLM error for {label}: {e}")
            return _default

    def publish_objects(self):
        self.processing_interrupted = False
        self.log_both("info", "=== New perception cycle started ===")

        if not self.is_stationary:
            self.get_logger().warn("Robot moving at the start, canceling processing")
            return

        # --- 1. Get synced data ---
        camera_data = self.camera_data.get_synced_data()
        if camera_data is None:
            self.get_logger().warn("Could not get synced camera data, waiting ...")
            return

        image_raw   = camera_data['rgb']
        depth       = camera_data['depth']
        camera_info = camera_data['camera_info']
        camera_frame = camera_data['camera_frame']

        self.current_transforms = {(camera_frame, "map"): camera_data['transform']}

        # --- 2. Run detection ---
        detections = self.run_detection(camera_data)

        if self.processing_interrupted or not self.is_stationary:
            self.get_logger().error("Processing interrupted: robot moving during detection")
            return

        if not detections:
            self._publish_empty(depth, camera_info)
            return

        # --- 3. Visualizations ---
        self._save_visualizations(image_raw, depth, detections)

        img_msg = self.bridge.cv2_to_imgmsg(draw_detections(image_raw.copy(), detections), "bgr8")
        img_msg.header.stamp    = self.get_clock().now().to_msg()
        img_msg.header.frame_id = camera_info.header.frame_id
        self.pub_image.publish(img_msg)

        # --- 4. Assign instance labels ---
        label_counts = Counter(det.label for det in detections)
        label_seen   = Counter()
        for det in detections:
            label_seen[det.label] += 1
            det.instance_label = f"{det.label}#{label_seen[det.label]}" if label_counts[det.label] > 1 else det.label
        instance_labels = [det.instance_label for det in detections]

        # --- 5. 3D centroids and bboxes ---
        self.clear_accumulated_markers()
        all_masks = [det.mask[:, :, 0] for det in detections]
        centroids_3d, bboxes_3d = mask_list_to_centroid_and_bbox(
            all_masks, instance_labels, depth, camera_info, node=self,
            bbox_marker_pub=self.bbox_marker_pub, centroid_marker_pub=self.centroid_marker_pub
        )

        # --- 6. Crops + VLM ---
        crops_data = self._prepare_crops(detections, image_raw)
        self._publish_crops(crops_data)

        with ThreadPoolExecutor(max_workers=min(4, len(crops_data))) as executor:
            futures     = {executor.submit(self.process_crop_vlm, c): i for i, c in enumerate(crops_data)}
            vlm_results = [None] * len(crops_data)
            for future in as_completed(futures):
                i, result = futures[future], future.result()
                if result:
                    vlm_results[i] = result

        # --- 7. Publish Bbox3dArray ---
        msg        = self._make_header_msg(Bbox3dArray)
        fov_volume = compute_fov_volume_from_depth(depth, camera_info, self)
        if fov_volume:
            for k, v in fov_volume.items():
                setattr(msg, f"fov_{k}", v)

        _unknown = lambda: {"description": "unknown", "color": "unknown", "material": "unknown", "shape": "unknown"}
        descriptions = []
        for idx, (det, bbox_3d) in enumerate(zip(detections, bboxes_3d)):
            result = vlm_results[idx] or {}
            desc   = {k: result.get(k, "unknown") for k in ("description", "color", "material", "shape")}
            descriptions.append(desc)

            if bbox_3d:
                box_msg = Bbox3d()
                box_msg.label = det.instance_label
                for k, v in bbox_3d.items():
                    setattr(box_msg, k, v)
                msg.boxes.append(box_msg)
        self.bbox_pub.publish(msg)

        # --- 8. Publish ObjectDescriptionArray ---
        desc_array = self._make_header_msg(ObjectDescriptionArray)
        for det, desc in zip(detections, descriptions):
            o = ObjectDescription()
            o.label = det.instance_label
            for k, v in desc.items():
                setattr(o, k, v)
            desc_array.descriptions.append(o)
        self.pub_object_descriptions.publish(desc_array)

        # --- 9. Update World Model ---
        wm.actual_perceptions.clear()
        for det, centroid, bbox, desc in zip(detections, centroids_3d, bboxes_3d, descriptions):
            wm.add_actual_perception(Object(det.label, centroid, bbox, **desc))

        perceptions_path = "/root/exchange/lost3dsg/output/actual_perceptions.json"
        os.makedirs(os.path.dirname(perceptions_path), exist_ok=True)
        with open(perceptions_path, "w") as f:
            json.dump([
                {"label": o.label,
                "centroid": o.centroid.tolist() if hasattr(o.centroid, 'tolist') else list(o.centroid or []),
                "bbox": o.bbox, **{k: getattr(o, k) for k in ("description", "color", "material", "shape")}}
                for o in wm.actual_perceptions
            ], f, indent=4)

        self.waiting_for_input = False
        self.log_both("info", "=== PUBLISH_OBJECTS COMPLETED ===")


    # ── Helpers ──────────────────────────────────────────────────────────────────

    def _make_header_msg(self, msg_type):
        msg = msg_type()
        msg.header = Header(stamp=self.get_clock().now().to_msg(), frame_id="map")
        return msg

    def _publish_empty(self, depth, camera_info):
        """Publish empty messages when no detections are found."""
        self.pcl_objects_pub.publish(PointCloud2(
            header=Header(stamp=self.get_clock().now().to_msg(), frame_id="map"),
            height=1, width=0
        ))
        self.pub_object_descriptions.publish(self._make_header_msg(ObjectDescriptionArray))

        empty_bboxes = self._make_header_msg(Bbox3dArray)
        fov = compute_fov_volume_from_depth(depth, camera_info, self)
        if fov:
            for k, v in fov.items():
                setattr(empty_bboxes, f"fov_{k}", v)
        self.bbox_pub.publish(empty_bboxes)
        self.waiting_for_input = False

    def _save_visualizations(self, image_raw, depth, detections):
        """Save bbox, depth, and PCL overlay images to disk."""
        ts  = datetime.now().strftime("%Y%m%d_%H%M%S")
        viz = os.path.join(PROJECT_ROOT, "output/visualizations")

        cv2.imwrite(os.path.join(viz, f"bbox_{ts}.jpg"), draw_detections(image_raw.copy(), detections))

        depth_norm = cv2.applyColorMap(cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8), cv2.COLORMAP_JET)
        os.makedirs(os.path.join(viz, "depth"), exist_ok=True)
        cv2.imwrite(os.path.join(viz, "depth", f"depth_{ts}.jpg"), depth_norm)

        COLORS_BGR = [(255,0,0),(0,255,0),(0,0,255),(255,255,0),(255,0,255),(0,255,255)]
        overlay = image_raw.copy()
        for idx, det in enumerate(detections):
            bgr   = COLORS_BGR[idx % len(COLORS_BGR)]
            mask2 = (det.mask[:, :, 0] * 255).astype(np.uint8)
            overlay_mask_on_image(overlay, mask2, color_rgb=(bgr[2]/255, bgr[1]/255, bgr[0]/255), alpha=0.4)
            m = cv2.moments(mask2)
            if m['m00']:
                cv2.putText(overlay, det.label, (int(m['m10']/m['m00']), int(m['m01']/m['m00'])),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, bgr, 2)
        os.makedirs(os.path.join(viz, "pointclouds"), exist_ok=True)
        cv2.imwrite(os.path.join(viz, "pointclouds", f"pcl_overlay_{ts}.jpg"), overlay)

    def _prepare_crops(self, detections, image_raw):
        """Crop each detection from the source image, save to disk, return list of crop dicts."""
        h, w      = image_raw.shape[:2]
        ts        = datetime.now().strftime("%Y%m%d_%H%M%S")
        crops_dir = os.path.join(PROJECT_ROOT, "output/cropped_images")
        os.makedirs(crops_dir, exist_ok=True)
        crops = []

        for idx, det in enumerate(detections):
            x0, y0, x1, y1 = (max(0, min(int(v), lim - 1)) for v, lim in
                            zip(det.bbox, (w, h, w, h)))
            x1, y1 = max(x0 + 1, min(x1, w)), max(y0 + 1, min(y1, h))
            crop = image_raw[y0:y1, x0:x1].copy()

            if crop.size == 0:
                self.get_logger().warn(f"Invalid crop for {det.label}")
                crops.append(None)
                continue

            bordered = crop.copy()
            cv2.rectangle(bordered, (0, 0), (bordered.shape[1]-1, bordered.shape[0]-1), (0, 255, 0), 2)
            cv2.imwrite(os.path.join(crops_dir, f"crop_{det.instance_label.replace(' ','_')}_{ts}_{idx}.jpg"), bordered)
            crops.append({'cropped': crop, 'label': det.instance_label, 'idx': idx})

        return crops

    def _publish_crops(self, crops_data):
        for crop in filter(None, crops_data):
            try:
                msg = self.bridge.cv2_to_imgmsg(crop['cropped'], encoding="bgr8")
                msg.header.stamp    = self.get_clock().now().to_msg()
                msg.header.frame_id = "camera"
                self.pub_crop.publish(msg)
            except Exception as e:
                self.get_logger().error(f"Crop publish error for {crop['label']}: {e}")
        

    def joint_callback(self, msg):
        """Monitor joint states and detect robot movement."""
        tracked = self.head_joints + self.base_joints
        deltas = []

        for j in tracked:
            if j in msg.name:
                pos = msg.position[msg.name.index(j)]
                if j in self.last_joint_positions:
                    deltas.append(abs(pos - self.last_joint_positions[j]))
                self.last_joint_positions[j] = pos

        if not deltas:
            return

        moving = (sum(deltas) / len(deltas)) >= self.position_threshold

        if moving and self.is_stationary:
            self.is_stationary = False
            self.processing_interrupted = True
            self.time_stationary_start  = None
            self.log_both('warn', "Movement detected — timer reset.")

            if not self.robot_has_moved_once:
                self.robot_has_moved_once = True
                m = Bool(); m.data = True
                self.movement_detected_pub.publish(m)
                self.log_both('warn', "First movement published to /robot_movement_detected")

        elif not moving and not self.is_stationary:
            self.is_stationary         = True
            self.time_stationary_start = self.get_clock().now()
            self.log_both('info', f"Robot stopped — detection in {self.min_stationary_after_movement}s")


def main(args=None):
    rclpy.init(args=args)
    node = DetectObjects()

    def _run_perception(label=""):
        """Run perception and reset timing state."""
        if label:
            node.log_both('info', label)
        node.publish_objects()
        t = node.get_clock().now()
        node.first_detection_done  = True
        node.last_detection_time   = t
        node.time_stationary_start = t

    def timer_callback():
        # Manual trigger has priority
        if node.manual_trigger_requested:
            if node.camera_data.get_synced_data() is not None:
                _run_perception("=== MANUAL PERCEPTION TRIGGERED ===")
                node.manual_trigger_requested = False
            else:
                node.get_logger().warn("Manual trigger: camera data not ready, retrying...")
            return

        # First detection
        if not node.first_detection_done:
            if node.camera_data.get_synced_data() is not None:
                _run_perception("First detection: data available — starting perception.")
            else:
                node.file_logger.debug("Waiting for synced data...")
            return

        # Re-detection after stationary interval
        if not (node.is_stationary and node.time_stationary_start):
            return

        elapsed = (node.get_clock().now() - node.time_stationary_start).nanoseconds / 1e9
        if elapsed >= node.min_stationary_after_movement:
            _run_perception("Robot stationary long enough — starting new perception cycle.")

    node.create_timer(0.5, timer_callback)

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()