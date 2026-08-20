#!/usr/bin/env python3
import logging
import os
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from threading import Lock

import numpy as np
import rclpy
import tf2_ros
from tf2_ros import TransformException
from tf2_ros.transform_listener import TransformListener
import torch
from cv_bridge import CvBridge
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup, ReentrantCallbackGroup
from rclpy.duration import Duration
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile
from rclpy.logging import LoggingSeverity
from sensor_msgs.msg import Image, JointState, PointCloud2
from std_msgs.msg import Bool, String
from visualization_msgs.msg import MarkerArray
from geometry_msgs.msg import PoseStamped

# Compat shim for older transforms3d/tf_transformations on NumPy >= 1.24
if not hasattr(np, "float"):
    np.float = float  # type: ignore[attr-defined]

from tf_transformations import quaternion_inverse, quaternion_multiply, euler_from_quaternion

import utils
from cv_utils import _clear_markers, init_bbox_publisher, vlm_call, numpy_to_base64,mask_list_to_centroid_and_bbox, mask_list_to_pointcloud2, publish_individual_pointclouds_by_id
from detection_pipeline import DetectionPipelineMixin
from lost3dsg.msg import Bbox3d, Bbox3dArray, ObjectDescription, ObjectDescriptionArray
from models import OWLv2, VitSam
from object_info import Object
from perception_utils import compute_fov_volume_from_depth, get_project_root
from input_output import PerceptionIOMixin
from vlm_call import VlmClient
from world_model import wm
from utils import draw_detections

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
        self.detector = OWLv2()
        self.vitsam = VitSam(utils.ENCODER_VITSAM_PATH, utils.DECODER_VITSAM_PATH)
        self.vlm = VlmClient(vlm_call_fn=vlm_call, image_encoder_fn=numpy_to_base64)

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
        prompt_path = os.path.join(os.path.dirname(__file__), "visual_prompt.txt")
        return self.vlm.call_crop_full(prompt_path, crop_info["label"], crop_info["cropped"])

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
            self.publish_empty_state(depth, camera_info, cycle_stamp)
            return

        self._io_executor.submit(self.save_visualizations, image_raw.copy(), depth.copy(), list(detections), PROJECT_ROOT)
        drawn = draw_detections(image_raw.copy(), detections)
        img_msg = self.bridge.cv2_to_imgmsg(drawn, "bgr8")
        img_msg.header.stamp = cycle_stamp
        img_msg.header.frame_id = camera_info.header.frame_id
        self.pub_image.publish(img_msg)

        self._assign_instance_labels(detections)
        centroids_3d, bboxes_3d = self._compute_3d_geometry(detections, depth, camera_info, camera_data["transform"])
        crops_data = self.prepare_crops(detections, image_raw, PROJECT_ROOT)
        self.publish_crops(crops_data)
        vlm_results = self._run_crop_vlm_batch(crops_data)
        descriptions = self._build_descriptions(detections, vlm_results)
        self._publish_bbox_array(detections, bboxes_3d, fov_volume, cycle_stamp)
        self._publish_description_array(detections, descriptions, cycle_stamp)
        self._update_world_model(detections, centroids_3d, bboxes_3d, descriptions)
        self._queue_perceptions_json()
        self._publish_agent_pose(cycle_stamp)
        self.waiting_for_input = False
        self.log_both("info", "publish_objects completed")

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

    def _run_crop_vlm_batch(self, crops_data):
        results = {}
        valid_crops = [crop for crop in crops_data if crop is not None]
        if not valid_crops:
            return results

        max_workers = min(4, len(valid_crops))
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(self.process_crop_vlm, crop): crop["label"] for crop in valid_crops}
            for future in as_completed(futures):
                label = futures[future]
                try:
                    results[label] = future.result() or {}
                except Exception as exc:
                    self.get_logger().error(f"Crop VLM future failed for {label}: {exc}")
                    results[label] = {}
        return results

    def _build_descriptions(self, detections, vlm_results):
        return [
            {field: (vlm_results.get(det.instance_label, {}) or {}).get(field, "unknown") for field in DESCRIPTION_FIELDS}
            for det in detections
        ]

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
            wm.add_actual_perception(Object(det.label, centroid, bbox, **desc))

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
