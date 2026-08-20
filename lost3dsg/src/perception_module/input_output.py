import json
import os
import re
from datetime import datetime

import cv2
from sensor_msgs.msg import PointCloud2
from std_msgs.msg import Header
from lost3dsg.msg import Bbox3dArray, ObjectDescriptionArray

from perception_utils import compute_fov_volume_from_depth
from utils import draw_detections


class PerceptionIOMixin:
    def make_header_msg(self, msg_type, stamp=None, frame_id="map"):
        msg = msg_type()
        msg.header = Header(
            stamp=stamp if stamp is not None else self.get_clock().now().to_msg(),
            frame_id=frame_id,
        )
        return msg

    def save_crop_file(self, path, image):
        try:
            cv2.imwrite(path, image)
        except Exception as exc:
            self.log_both("error", f"Background crop save failed ({path}): {exc}")

    def write_perceptions_json(self, perceptions_snapshot):
        perceptions_path = "/root/exchange/lost3dsg/output/actual_perceptions.json"
        try:
            os.makedirs(os.path.dirname(perceptions_path), exist_ok=True)
            with open(perceptions_path, "w") as file_obj:
                json.dump(perceptions_snapshot, file_obj, indent=4)
        except Exception as exc:
            self.log_both("error", f"Background JSON dump failed: {exc}")

    def publish_empty_state(self, depth, camera_info, cycle_stamp=None):
        stamp = cycle_stamp if cycle_stamp is not None else self.get_clock().now().to_msg()
        self.pcl_objects_pub.publish(PointCloud2(header=Header(stamp=stamp, frame_id="map"), height=1, width=0))
        self.pub_object_descriptions.publish(self.make_header_msg(ObjectDescriptionArray, stamp=stamp, frame_id="map"))

        empty_bboxes = self.make_header_msg(Bbox3dArray, stamp=stamp, frame_id="map")
        fov = compute_fov_volume_from_depth(depth, camera_info, self)
        if fov:
            for key, value in fov.items():
                setattr(empty_bboxes, f"fov_{key}", value)
        self.bbox_pub.publish(empty_bboxes)
        self.waiting_for_input = False

    def save_visualizations(self, image_raw, depth, detections, project_root):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        visualization_dir = os.path.join(project_root, "output/visualizations")
        os.makedirs(visualization_dir, exist_ok=True)

        cv2.imwrite(os.path.join(visualization_dir, f"bbox_{timestamp}.jpg"), draw_detections(image_raw.copy(), detections))

        depth_norm = cv2.applyColorMap(
            cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX).astype("uint8"),
            cv2.COLORMAP_JET,
        )
        depth_dir = os.path.join(visualization_dir, "depth")
        os.makedirs(depth_dir, exist_ok=True)
        cv2.imwrite(os.path.join(depth_dir, f"depth_{timestamp}.jpg"), depth_norm)

    def prepare_crops(self, detections, image_raw, project_root):
        height, width = image_raw.shape[:2]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        crops_dir = os.path.join(project_root, "output/cropped_images")
        os.makedirs(crops_dir, exist_ok=True)

        crops = []
        for idx, det in enumerate(detections):
            x0 = max(0, min(int(det.bbox[0]), width - 1))
            y0 = max(0, min(int(det.bbox[1]), height - 1))
            x1 = max(0, min(int(det.bbox[2]), width))
            y1 = max(0, min(int(det.bbox[3]), height))
            x1 = max(x0 + 1, x1)
            y1 = max(y0 + 1, y1)

            crop = image_raw[y0:y1, x0:x1].copy()
            if crop.size == 0:
                self.get_logger().warn(f"Invalid crop for {det.instance_label}")
                crops.append(None)
                continue

            bordered = crop.copy()
            cv2.rectangle(bordered, (0, 0), (bordered.shape[1] - 1, bordered.shape[0] - 1), (0, 255, 0), 2)
            safe_label = re.sub(r"[^A-Za-z0-9_.-]+", "_", det.instance_label)
            crop_path = os.path.join(crops_dir, f"crop_{safe_label}_{timestamp}_{idx}.jpg")
            self._io_executor.submit(self.save_crop_file, crop_path, bordered.copy())
            crops.append({"cropped": crop, "label": det.instance_label, "idx": idx})
        return crops

    def publish_crops(self, crops_data):
        for crop in filter(None, crops_data):
            try:
                msg = self.bridge.cv2_to_imgmsg(crop["cropped"], encoding="bgr8")
                msg.header.stamp = self.get_clock().now().to_msg()
                msg.header.frame_id = "camera"
                self.pub_crop.publish(msg)
            except Exception as exc:
                self.get_logger().error(f"Crop publish error for {crop['label']}: {exc}")
