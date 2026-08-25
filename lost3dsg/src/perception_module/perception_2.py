#!/usr/bin/env python3
import json
import logging
import os
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from threading import Lock

# Ensure local sibling packages and directories (e.g. cloud/) are on sys.path
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import rclpy  # noqa: E402
import tf2_ros  # noqa: E402
import torch  # noqa: E402
from config import CFG  # noqa: E402
from cv_bridge import CvBridge  # noqa: E402
from geometry_msgs.msg import PoseStamped  # noqa: E402
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup, ReentrantCallbackGroup  # noqa: E402
from rclpy.duration import Duration  # noqa: E402
from rclpy.executors import MultiThreadedExecutor  # noqa: E402
from rclpy.logging import LoggingSeverity  # noqa: E402
from rclpy.node import Node  # noqa: E402
from rclpy.qos import DurabilityPolicy, QoSProfile  # noqa: E402
from sensor_msgs.msg import Image, PointCloud2  # noqa: E402
from std_msgs.msg import Bool, String  # noqa: E402
from tf2_ros import TransformException  # noqa: E402

# Compat shim for older transforms3d/tf_transformations on NumPy >= 1.24
if not hasattr(np, "float"):
    np.float = float  # type: ignore[attr-defined]

import utils  # noqa: E402
from cloud import get_perception_backend  # noqa: E402
from cv_utils import (  # noqa: E402
    _clear_markers,
    draw_boxes_3d,
    init_bbox_publisher,
    mask_list_to_centroid_and_bbox,
    mask_list_to_pointcloud2,
    numpy_to_base64,
    publish_individual_pointclouds_by_id,
    vlm_call,
)
from detection_pipeline import DetectionPipelineMixin  # noqa: E402
from input_output import PerceptionIOMixin  # noqa: E402
from models import OWLv2, VitSam  # noqa: E402
from object_info import Object  # noqa: E402
from perception_utils import compute_fov_volume_from_depth, get_project_root  # noqa: E402
from tf_transformations import euler_from_quaternion, quaternion_inverse, quaternion_multiply  # noqa: E402
from utils import draw_detections  # noqa: E402
from vlm_call import VlmClient  # noqa: E402
from world_model import wm  # noqa: E402

from lost3dsg.msg import Bbox3d, Bbox3dArray, ObjectDescription, ObjectDescriptionArray  # noqa: E402

if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True

PROJECT_ROOT = get_project_root(__file__)
LOG_DIR = os.path.join(PROJECT_ROOT, "output")
os.makedirs(LOG_DIR, exist_ok=True)

module_logger = logging.getLogger("perception_module")
module_logger.setLevel(logging.DEBUG)
if not module_logger.handlers:
    file_handler = logging.FileHandler(
        os.path.join(LOG_DIR, f"perception_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    )
    file_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    module_logger.addHandler(file_handler)
else:
    file_handler = module_logger.handlers[0]

DESCRIPTION_FIELDS = ("description", "color", "material", "shape")

# ponytail: fixed cap for images sent to the VLM; make it a CFG["vlm"] knob if a
# model ever needs finer input. The base64 payload dominates vlm_ms, not the answer.
VLM_IMAGE_MAX_SIDE = 512


def _encode_for_vlm(img):
    h, w = img.shape[:2]
    scale = VLM_IMAGE_MAX_SIDE / float(max(h, w))
    if scale < 1.0:
        img = cv2.resize(img, (max(1, round(w * scale)), max(1, round(h * scale))), interpolation=cv2.INTER_AREA)
    return numpy_to_base64(img)


class DetectObjectsNode(Node, DetectionPipelineMixin, PerceptionIOMixin):
    LOG_METHODS = {
        "debug": (LoggingSeverity.DEBUG, "debug"),
        "info": (LoggingSeverity.INFO, "info"),
        "warn": (LoggingSeverity.WARN, "warning"),
        "error": (LoggingSeverity.ERROR, "error"),
    }

    def __init__(self):
        super().__init__("detection_node")
        self.sensor_cb_group = ReentrantCallbackGroup()
        self.perception_cb_group = MutuallyExclusiveCallbackGroup()
        self._perception_lock = Lock()

        self.file_logger = module_logger
        self.file_logger.info("=== DetectObjectsNode initialized ===")

        self.bridge = CvBridge()
        self.perception_backend = get_perception_backend(CFG)
        backend_type = CFG.get("perception", {}).get("backend", "local").lower()
        if backend_type == "local":
            self.detector = OWLv2()
            self.vitsam = VitSam(utils.ENCODER_VITSAM_PATH, utils.DECODER_VITSAM_PATH)
        else:
            self.detector = None
            self.vitsam = None
            self.file_logger.info(f"Using Cloud Perception Backend: {backend_type}")
        self.vlm = VlmClient(vlm_call_fn=vlm_call, image_encoder_fn=_encode_for_vlm)

        self.tf_buffer = tf2_ros.Buffer(cache_time=Duration(seconds=30.0))
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self._init_publishers()
        self._init_subscribers()
        self._init_state()

        self._io_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="perception_io")
        self.clear_accumulated_markers()
        self._create_timers()

        self.get_logger().info(f"Log saved in: {file_handler.baseFilename}")

    def _init_publishers(self):
        qos_latched = QoSProfile(depth=10, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        qos_default = 10

        self.pub_image = self.create_publisher(Image, "/image_with_bb", qos_latched)
        self.pub_crop = self.create_publisher(Image, "/cropped_image", qos_default)
        self.bbox_pub = self.create_publisher(Bbox3dArray, "/bbox_3d", qos_default)
        self.pub_object_descriptions = self.create_publisher(ObjectDescriptionArray, "/object_descriptions", qos_default)
        self.pcl_objects_pub = self.create_publisher(PointCloud2, "/pcl_objects", qos_latched)
        self.pcl_objects_labels_pub = self.create_publisher(String, "/pcl_objects_labels", qos_latched)
        self.movement_detected_pub = self.create_publisher(Bool, "/robot_movement_detected", qos_default)
        self.agent_pose_pub = self.create_publisher(PoseStamped, "/agent_camera_pose", qos_default)
        self.bbox_marker_pub, self.centroid_marker_pub = init_bbox_publisher(self)

    def _init_subscribers(self):
        self.camera_data = utils.SyncedCameraData(self, sync_tolerance_ms=2000)
        
    def _init_state(self):
        self.head_joints = ["head_1_joint", "head_2_joint", "habitat_camera"]
        self.base_joints = ["wheel_left_joint", "wheel_right_joint"]
        self.position_threshold = 0.05
        self.last_joint_positions = {}
        self.is_stationary = True
        self.time_stationary_start = None
        self.last_detection_time = None
        self.first_detection_done = False
        self.robot_has_moved_once = False
        self.min_stationary_after_movement = 0.5
        self.processing_interrupted = False
        self.manual_trigger_requested = False
        self.waiting_for_input = False
        self.publish_individual_objects = False
        self.pcl_object_id_counter = 0
        self.individual_pcl_publishers = {}
        self.robot_base_frame = "base_link"  

    def _create_timers(self):
        self.create_timer(0.5, self._perception_timer_callback, callback_group=self.perception_cb_group)
        self.create_timer(1.0, self.joint_callback, callback_group=self.perception_cb_group)
        self.get_logger().info("Perception timers created")


    def log_both(self, level, message):
        try:
            self.get_logger().info(f"[{level.upper()}] {message}")
        except Exception as exc:
            self.get_logger().error(f"Errore: {exc}")

        try:
            if level == "debug":
                self.file_logger.debug(message)
            elif level == "warn":
                self.file_logger.warning(message)
            elif level == "error":
                self.file_logger.error(message)
            else:
                self.file_logger.info(message)
        except Exception as exc:
            self.get_logger().error(f"Errore: {exc}")

    def _abort_if_moving(self, stage):
        if not self.is_stationary:
            self.processing_interrupted = True
            self.log_both("warn", f"Robot moved during {stage}: perception aborted")
            return True
        return False

    def _run_perception_cycle(self, reason=""):
        if not self._perception_lock.acquire(blocking=False):
            self.log_both("debug", "Perception cycle skipped: previous cycle is still running")
            return

        try:
            if reason:
                self.log_both("info", reason)
            self.log_both(
                "info",
                f"_run_perception_cycle: stationary={self.is_stationary}, first_done={self.first_detection_done}, manual={self.manual_trigger_requested}",
            )
            self.publish_objects()
            now = self.get_clock().now()
            self.first_detection_done = True
            self.last_detection_time = now
            self.time_stationary_start = now
        except Exception as exc:
            self.log_both("error", f"Unhandled perception-cycle error: {exc}")
        finally:
            self._perception_lock.release()

    def _perception_timer_callback(self):
        self.log_both(
            "debug",
            f"timer: manual={self.manual_trigger_requested}, first_done={self.first_detection_done}, stationary={self.is_stationary}, stationary_start={self.time_stationary_start}",
        )

        if self.manual_trigger_requested:
            self.log_both("info", "timer: manual trigger requested")
            if self.camera_data.get_synced_data() is not None:
                self._run_perception_cycle("=== MANUAL PERCEPTION TRIGGERED ===")
                self.manual_trigger_requested = False
            else:
                self.log_both("info", "timer: manual trigger waiting for synced camera data")
            return

        if not self.first_detection_done:
            self.log_both("info", "timer: first detection not done yet")
            if self.camera_data.get_synced_data() is not None:
                self._run_perception_cycle("First detection: data available — starting perception.")
            else:
                self.log_both("info", "timer: synced camera data not ready yet")
            return

        if not self.is_stationary:
            self.log_both("debug", "timer: robot moving, skip perception")
            return

        if self.time_stationary_start is None:
            self.time_stationary_start = self.get_clock().now()
            self.log_both("info", "timer: stationary start initialized")
            return

        elapsed = (self.get_clock().now() - self.time_stationary_start).nanoseconds / 1e9
        self.log_both("debug", f"timer: stationary for {elapsed:.3f}s")

        if elapsed >= self.min_stationary_after_movement:
            self._run_perception_cycle("Robot stationary long enough — starting new perception cycle.")

    def clear_accumulated_markers(self):
        for topic, publisher, counter_attr in [
            ("/centroid_markers", self.centroid_marker_pub, "_centroid_marker_id_counter"),
            ("/bbox_marker", self.bbox_marker_pub, "_bbox_marker_id_counter"),
        ]:
            try:
                _clear_markers(topic, node=self, publisher=publisher)
            except Exception as exc:
                self.get_logger().warn(f"Error clearing {topic}: {exc}")
            setattr(self, counter_attr, 0)

    def color_pcl(self, detections, camera_data):
        camera_info = camera_data["camera_info"]
        depth_img = camera_data["depth"]
        mask_list = [(det.mask[:, :, 0].astype(np.uint8) * 255) for det in detections]
        labels = [det.label for det in detections]

        mask_list_to_pointcloud2(
            mask_list,
            depth_img,
            camera_info,
            node=self,
            labels=labels,
            topic="/pcl_objects",
            max_points_per_obj=1000,
            publisher=self.pcl_objects_pub,
            labels_publisher=self.pcl_objects_labels_pub,
            transform=camera_data.get("transform"),   # map frame, like the boxes
        )

        if not self.publish_individual_objects:
            return

        published = publish_individual_pointclouds_by_id(
            mask_list,
            depth_img,
            camera_info,
            node=self,
            labels=labels,
            frame_id="map",
            topic_prefix="/pcl_id",
            publishers_dict=self.individual_pcl_publishers,
            id_counter_start=self.pcl_object_id_counter,
            timestamp=camera_data.get("timestamp", camera_info.header.stamp),
        )
        self.pcl_object_id_counter += published

    def process_crop_vlm(self, crop_info):
        if crop_info is None:
            return None
        prompt_path = CFG["paths"]["visual_prompt"] or os.path.join(
            os.path.dirname(__file__), "prompts", "visual_prompt.txt")
        yaw = crop_info.get("yaw", 0.0)
        distance = crop_info.get("distance", 1.0)
        image_id = crop_info.get("image_id", "")
        return self.vlm.call_crop_full(
            prompt_path, crop_info["label"], crop_info["cropped"],
            yaw=yaw, distance=distance, image_id=image_id
        )

    # -------------------------------------------------------------------------
    # TODO (Lazy Two-Stage Crop Refinement & Property Separation):
    # - Stage 1 (Hot Detection Cycle): Detector produces primary class noun (e.g. "chair").
    # - Stage 2 (Lazy on Admission/Ambiguity): When an object is admitted or contested
    #   by the ontological layer (found.admission), trigger this asynchronous crop VLM
    #   query to refine the noun (e.g. "office chair") and extract extended traits.
    # - Standard properties ("color", "material", "shape", "description") remain in the
    #   primary metadata schema, while extended attributes ("style", "affordances", "state")
    #   populate the instance attribute set for deep ontological alignment.
    # -------------------------------------------------------------------------
    def lazy_refine_object_crop(self, obj, crop_image):
        """Asynchronous / Lazy refinement of object semantics and attributes."""
        pass

    def publish_objects(self):
        self.processing_interrupted = False
        self.log_both("info", "publish_objects entered")

        if not self.is_stationary:
            self.get_logger().warn("Robot moving at the start, canceling processing")
            return

        camera_data = self.camera_data.get_synced_data()
        if camera_data is None:
            self.get_logger().warn("Could not get synced camera data, waiting ...")
            return

        image_raw = camera_data["rgb"]
        depth = camera_data["depth"]
        camera_info = camera_data["camera_info"]
        cycle_stamp = camera_data.get("timestamp", None) or (
            camera_info.header.stamp if hasattr(camera_info, "header") else self.get_clock().now().to_msg()
        )

        # TF-dependent: calcolata subito, finché lo stamp è ancora nel buffer TF
        fov_volume = compute_fov_volume_from_depth(depth, camera_info, self)

        self.log_both("info", "publish_objects: before run_detection")
        detections = self.run_detection(camera_data)
        self.log_both("info", f"publish_objects: after run_detection, detections={len(detections)}")

        if self.processing_interrupted or not self.is_stationary:
            self.get_logger().error("Processing interrupted: robot moving during detection")
            return
        if not detections:
            self._publish_image_with_bb(image_raw, [], [], camera_info, camera_data["transform"], cycle_stamp, depth)
            self.publish_empty_state(depth, camera_info, cycle_stamp)
            return

        self._io_executor.submit(self.save_visualizations, image_raw.copy(), depth.copy(), list(detections), PROJECT_ROOT)

        self._assign_instance_labels(detections)
        centroids_3d, bboxes_3d = self._compute_3d_geometry(detections, depth, camera_info, camera_data["transform"])
        self._add_pca_orientation(detections, bboxes_3d, depth, camera_info, camera_data["transform"])
        self._publish_image_with_bb(image_raw, detections, bboxes_3d, camera_info, camera_data["transform"], cycle_stamp, depth)
        crops_data = self.prepare_crops(detections, image_raw, PROJECT_ROOT)
        self.publish_crops(crops_data)
        self._attach_crop_embeddings(detections, crops_data)
        vlm_results = self._run_crop_vlm_batch(crops_data)
        descriptions = self._build_descriptions(detections, vlm_results)
        self._publish_bbox_array(detections, bboxes_3d, fov_volume, cycle_stamp)
        self._publish_description_array(detections, descriptions, cycle_stamp)
        self._update_world_model(detections, centroids_3d, bboxes_3d, descriptions)
        self._queue_perceptions_json()
        self._publish_agent_pose(cycle_stamp)
        self.waiting_for_input = False
        self.log_both("info", "publish_objects completed")

    def _publish_image_with_bb(self, image_raw, detections, bboxes_3d, camera_info, transform, stamp, depth=None):
        """/image_with_bb shows the 3D boxes projected back into the frame they were
        measured from, under the same visibility rule as the simulator overlay; the
        flat 2D rectangle is kept only for detections that got no 3D box. Published
        on every cycle, empty ones included, so a subscriber that joins late (rviz)
        sees the latest frame instead of "No image"."""
        drawn = image_raw.copy()
        if detections:
            flat = [det for det, box in zip(detections, bboxes_3d) if not box]
            if flat:
                draw_detections(drawn, flat)
            draw_boxes_3d(drawn, bboxes_3d, [det.instance_label for det in detections], camera_info, transform, depth)
        img_msg = self.bridge.cv2_to_imgmsg(drawn, "bgr8")
        img_msg.header.stamp = stamp
        img_msg.header.frame_id = camera_info.header.frame_id
        self.pub_image.publish(img_msg)

    def _assign_instance_labels(self, detections):
        label_counts = Counter(det.label for det in detections)
        label_seen = Counter()
        for det in detections:
            label_seen[det.label] += 1
            det.instance_label = f"{det.label}#{label_seen[det.label]}" if label_counts[det.label] > 1 else det.label

    def _compute_3d_geometry(self, detections, depth, camera_info, transform):
        self.clear_accumulated_markers()
        all_masks = [det.mask[:, :, 0] for det in detections]
        instance_labels = [det.instance_label for det in detections]
        return mask_list_to_centroid_and_bbox(
            all_masks, instance_labels, depth, camera_info,
            node=self,
            bbox_marker_pub=self.bbox_marker_pub,
            centroid_marker_pub=self.centroid_marker_pub,
            transform=transform,
        )

    def _attach_crop_embeddings(self, detections, crops_data):
        """CLIP image embedding per object, reusing the OWLv2 backbone already on
        the GPU (OWLv2 is CLIP-based) — no extra model load. Sets det.clip_embedding
        and queues the per-cycle sidecar dump (output/clip_embeddings.json)."""
        # If detections already have clip_embedding from cloud backend, dump and return
        if any(getattr(d, "clip_embedding", None) is not None for d in detections):
            snapshot = {det.instance_label: det.clip_embedding for det in detections if getattr(det, "clip_embedding", None)}
            if snapshot:
                self._io_executor.submit(self.write_clip_embeddings, snapshot)
            return

        if not self.detector:
            return

        valid = [c for c in crops_data if c is not None]
        embeddings = {}
        if valid:
            try:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                images = [cv2.cvtColor(c["cropped"], cv2.COLOR_BGR2RGB) for c in valid]
                inputs = self.detector.processor(images=images, return_tensors="pt")
                pixel_values = inputs["pixel_values"].to(self.detector.device)
                with torch.inference_mode():
                    feats = self.detector.model.owlv2.get_image_features(pixel_values=pixel_values)
                # transformers returns a tensor or BaseModelOutputWithPooling depending on version
                if not torch.is_tensor(feats):
                    feats = feats.pooler_output
                feats = torch.nn.functional.normalize(feats, dim=-1).cpu().numpy()
                embeddings = {c["idx"]: [round(float(v), 5) for v in feats[i]] for i, c in enumerate(valid)}
            except Exception as exc:
                self.log_both("warn", f"Crop embedding computation failed: {exc}")
            finally:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        for idx, det in enumerate(detections):
            det.clip_embedding = embeddings.get(idx)

        snapshot = {det.instance_label: det.clip_embedding for det in detections if det.clip_embedding}
        if snapshot:
            self._io_executor.submit(self.write_clip_embeddings, snapshot)

    def write_clip_embeddings(self, embeddings):
        path = os.path.join(PROJECT_ROOT, "output", "clip_embeddings.json")
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "w") as f:
                json.dump(embeddings, f)
        except Exception as exc:
            self.log_both("error", f"CLIP embedding dump failed: {exc}")

    def _run_crop_vlm_batch(self, crops_data):
        results = {}
        valid_crops = [crop for crop in crops_data if crop is not None]
        if not valid_crops:
            return results

        concurrency = int(CFG.get("vlm", {}).get("crop_concurrency", 4))
        max_workers = min(concurrency, len(valid_crops))
        timeout = float(CFG.get("vlm", {}).get("crop_timeout", 15.0))

        with ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="crop_vlm") as executor:
            futures = {executor.submit(self.process_crop_vlm, crop): crop["label"] for crop in valid_crops}
            for future in as_completed(futures):
                label = futures[future]
                try:
                    results[label] = future.result(timeout=timeout) or {}
                except Exception as exc:
                    self.get_logger().error(f"Crop VLM future failed for {label}: {exc}")
                    results[label] = {}
        return results

    def _build_descriptions(self, detections, vlm_results):
        descriptions = []
        for det in detections:
            res = (vlm_results.get(det.instance_label, {}) or {})
            d = {field: res.get(field, "unknown") for field in DESCRIPTION_FIELDS}
            d["confirmed"] = getattr(det, "is_confirmed", True)
            if "provenance" in res:
                d["provenance"] = res["provenance"]
            descriptions.append(d)
        return descriptions

    def _publish_bbox_array(self, detections, bboxes_3d, fov_volume, cycle_stamp):
        msg = self.make_header_msg(Bbox3dArray, stamp=cycle_stamp, frame_id="map")
        if fov_volume:
            for key, value in fov_volume.items():
                setattr(msg, f"fov_{key}", value)

        for det, bbox_3d in zip(detections, bboxes_3d):
            if not bbox_3d:
                continue
            box_msg = Bbox3d()
            box_msg.label = det.instance_label
            for key, value in bbox_3d.items():
                # Only copy keys the msg actually has: bbox dicts may carry
                # extra fields (added before the msg grows a matching one) and
                # a blind setattr would raise AttributeError on the msg slots.
                if hasattr(box_msg, key):
                    setattr(box_msg, key, value)
            msg.boxes.append(box_msg)
        self.bbox_pub.publish(msg)

    def _publish_description_array(self, detections, descriptions, cycle_stamp):
        desc_array = self.make_header_msg(ObjectDescriptionArray, stamp=cycle_stamp, frame_id="map")
        for det, desc in zip(detections, descriptions):
            obj_msg = ObjectDescription()
            obj_msg.label = det.instance_label
            for key, value in desc.items():
                setattr(obj_msg, key, value)
            desc_array.descriptions.append(obj_msg)
        self.pub_object_descriptions.publish(desc_array)

    def _update_world_model(self, detections, centroids_3d, bboxes_3d, descriptions):
        wm.actual_perceptions.clear()
        for det, centroid, bbox, desc in zip(detections, centroids_3d, bboxes_3d, descriptions):
            obj = Object(det.label, centroid, bbox, **desc)
            # distinct from obj.embedding (the 300-d word2vec description vector)
            obj.clip_embedding = getattr(det, "clip_embedding", None)
            wm.add_actual_perception(obj)

    def _publish_agent_pose(self, cycle_stamp):
        try:
            lookup_time = rclpy.time.Time.from_msg(cycle_stamp)
            t = self.tf_buffer.lookup_transform(
                "map",
                self.robot_base_frame,
                lookup_time,
            )
        except TransformException as ex:
            self.log_both("warn", f"Could not get agent pose (map -> {self.robot_base_frame}): {ex}")
            return

        pose_msg = PoseStamped()
        pose_msg.header.stamp = cycle_stamp
        pose_msg.header.frame_id = "map"
        pose_msg.pose.position.x = t.transform.translation.x
        pose_msg.pose.position.y = t.transform.translation.y
        pose_msg.pose.position.z = t.transform.translation.z
        pose_msg.pose.orientation = t.transform.rotation

        self.agent_pose_pub.publish(pose_msg)
        self.log_both("debug", "Agent pose published on /agent_camera_pose")

    def _queue_perceptions_json(self):
        perceptions_snapshot = [
            {
                "label": obj.label,
                "centroid": obj.centroid.tolist() if hasattr(obj.centroid, "tolist") else list(obj.centroid or []),
                "bbox": obj.bbox,
                **{field: getattr(obj, field) for field in DESCRIPTION_FIELDS},
            }
            for obj in wm.actual_perceptions
        ]
        self._io_executor.submit(self.write_perceptions_json, perceptions_snapshot)

    def joint_callback(self):
        tracked_joints = self.head_joints + self.base_joints
        motion_scores = []

        for joint_name in tracked_joints:
            try:
                from_frame_rel = "map"
                t = self.tf_buffer.lookup_transform(
                    joint_name,
                    from_frame_rel,
                    rclpy.time.Time()
                )
                position = t.transform

                self.get_logger().info(
                    "Joint refreshed: "
                    f"joint={joint_name} "
                    f"position={position} "
                )

                if joint_name in self.last_joint_positions:
                    previous_position = self.last_joint_positions[joint_name]

                    dx = position.translation.x - previous_position.translation.x
                    dy = position.translation.y - previous_position.translation.y
                    dz = position.translation.z - previous_position.translation.z

                    q_prev = [
                        previous_position.rotation.x,
                        previous_position.rotation.y,
                        previous_position.rotation.z,
                        previous_position.rotation.w,
                    ]
                    q_curr = [
                        position.rotation.x,
                        position.rotation.y,
                        position.rotation.z,
                        position.rotation.w,
                    ]

                    q_delta = quaternion_multiply(quaternion_inverse(q_prev), q_curr)
                    roll, pitch, yaw = euler_from_quaternion(q_delta)

                    linear_mag = (dx * dx + dy * dy + dz * dz) ** 0.5
                    angular_mag = abs(roll) + abs(pitch) + abs(yaw)

                    motion_score = linear_mag + angular_mag
                    motion_scores.append(motion_score)

                    self.get_logger().info(
                        "Joint delta: "
                        f"joint={joint_name} "
                        f"linear={linear_mag:.6f} "
                        f"angular={angular_mag:.6f} "
                        f"score={motion_score:.6f}"
                    )

                self.last_joint_positions[joint_name] = position

            except TransformException as ex:
                self.get_logger().info(
                    f"Could not transform {joint_name} to {from_frame_rel}: {ex}"
                )

        if not motion_scores:
            return

        moving = (sum(motion_scores) / len(motion_scores)) >= self.position_threshold
        self.log_both(
            "debug",
            f"joint: moving={moving}, stationary={self.is_stationary}, scores={motion_scores}"
        )

        if moving and self.is_stationary:
            self.is_stationary = False
            self.processing_interrupted = True
            self.time_stationary_start = None
            self.log_both("warn", "Movement detected — timer reset.")

            movement_msg = Bool()
            movement_msg.data = True
            self.movement_detected_pub.publish(movement_msg)
            self.log_both("warn", "Movement published to /robot_movement_detected")

        elif not moving and not self.is_stationary:
            self.is_stationary = True
            self.time_stationary_start = self.get_clock().now()
            self.log_both(
                "info",
                f"Robot stopped — detection in {self.min_stationary_after_movement}s"
            )
            movement_msg = Bool()
            movement_msg.data = False
            self.movement_detected_pub.publish(movement_msg)
            self.log_both("info", "Stop published to /robot_movement_detected")

    def destroy_node(self):
        try:
            self._io_executor.shutdown(wait=True)
        except Exception:
            pass
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = DetectObjectsNode()
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)

    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
