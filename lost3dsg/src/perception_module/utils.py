from object_info import Object
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from rclpy.duration import Duration as ROS2Duration
import numpy as np
from cv_bridge import CvBridge
import cv2, os, colorsys, time
from sensor_msgs.msg import Image, CameraInfo
from scipy.spatial import KDTree
from std_msgs.msg import ColorRGBA
import config 
from rclpy.time import Time

bridge = CvBridge()

file_path = os.path.abspath(__file__)
ENCODER_VITSAM_PATH = config.CFG["paths"]["vitsam_encoder"] or os.path.join(os.path.dirname(file_path), "utils", "l2_encoder.onnx")
DECODER_VITSAM_PATH = config.CFG["paths"]["vitsam_decoder"] or os.path.join(os.path.dirname(file_path), "utils", "l2_decoder.onnx")


class SyncedCameraData:
    """
    Manages the reception of RGB, Depth, CameraInfo and Transform from the robot.
    ALWAYS updates with the most recent frames (no temporal synchronization).
    """
    def __init__(self, node, sync_tolerance_ms=50):
        """
        Args:
            node: ROS2 Node instance
            sync_tolerance_ms: Not used (kept for compatibility)
        """
        self.node = node
        self.bridge = CvBridge()
        self.sync_tolerance_sec = float(sync_tolerance_ms) / 1000.0
        self.default_camera_frame = config.CFG["frames"]["camera"]

        # Data cache - ALWAYS UPDATED with the most recent messages
        self.cached_rgb = None
        self.cached_depth = None
        self.cached_camera_info = None
        self.cached_transform = None
        self.all_ready = False
        # Wall-clock receive time. Image stamps are the robot clock; this PC can
        # be minutes ahead, so stamp-vs-get_clock().now() is not frame age.
        self._rgb_received_mono = None

        # QoS for real robot sensor topics
        # GA-164. depth=1, not 10, and the callback comment three screens down says why:
        # "ALWAYS updates with the most recent RGB". A depth of 10 defeats that -- the node
        # drains a QUEUE of ten frames in arrival order, each one immediately superseded by
        # the next, so `cached_rgb` lags the sensor by up to the queue depth.
        #
        # MEASURED, run 20260901_035141 at 1280x960: frames reached get_synced_data a MEDIAN
        # 7.14 s stale (p90 10.90 s, max 34.05 s) against its 1.0 s freshness limit, so ALL
        # 1650 were rejected and `publish_objects` ran ZERO times in 17 minutes. At 640x480
        # the same check rejected 292 and 17 in the two previous runs and 428 and 29 cycles
        # still ran -- so this is a backlog that 4x the pixels turned from a tax into a wall.
        #
        # depth=1 means the middleware keeps only the newest sample and the node reads the
        # present rather than catching up on the past.
        qos_sensor = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST
        )

        # Persistent subscriptions
        self.node.get_logger().info("Subscribing to camera topics...")
        #node.create_subscription(Image, '/head_front_camera/rgb/image_raw', self._rgb_callback, qos_sensor)
        #node.create_subscription(Image, '/head_front_camera/depth/image_raw', self._depth_callback, qos_sensor)
        #node.create_subscription(CameraInfo, '/head_front_camera/rgb/camera_info', self._camera_info_callback, qos_sensor)
        node.create_subscription(Image, '/camera/rgb', self._rgb_callback, qos_sensor)
        node.create_subscription(Image, '/camera/depth', self._depth_callback, qos_sensor)
        node.create_subscription(CameraInfo, '/camera/camera_info', self._camera_info_callback, qos_sensor)
        self.node.get_logger().info("Subscriptions created!")

    def _rgb_callback(self, msg):
        """ALWAYS updates with the most recent RGB"""
        first_time = self.cached_rgb is None
        self.cached_rgb = msg  # Always update!
        self._rgb_received_mono = time.monotonic()
        if first_time:
            self.node.get_logger().info("RGB received (first frame)")
        # Always try to get the transform
        self._try_get_transform()

    def _depth_callback(self, msg):
        """ALWAYS updates with the most recent Depth"""
        first_time = self.cached_depth is None
        self.cached_depth = msg  # Always update!
        if first_time:
            self.node.get_logger().info("Depth received (first frame)")
            self._check_all_ready()
        self._try_get_transform()

    def _camera_info_callback(self, msg):
        """Keep the latest CameraInfo header stamp aligned with the current frame."""
        first_time = self.cached_camera_info is None
        self.cached_camera_info = msg
        if first_time:
            self.node.get_logger().info("CameraInfo received")
        self._check_all_ready()
        self._try_get_transform()

    def _try_get_transform(self):
        if self.cached_rgb is None:
            return
        if not hasattr(self.node, 'tf_buffer'):
            return
        camera_frame = self.cached_rgb.header.frame_id or self.default_camera_frame
        target_frame = config.world_frame()
        try:
            lookup_time = Time.from_msg(self.cached_rgb.header.stamp)
            transform = self.node.tf_buffer.lookup_transform(
                target_frame,
                camera_frame,
                lookup_time,
                timeout=ROS2Duration(seconds=config.CFG["tf"]["lookup_timeout"])
            )
            first_time = self.cached_transform is None
            self.cached_transform = transform
            if first_time:
                self.node.get_logger().info(
                    f"Transform received (first): {target_frame} <- {camera_frame}")
                self._check_all_ready()
        except Exception as e:
            # Per le bbox 3D preferiamo una posa esatta al timestamp del frame RGB:
            # when unavailable, invalidate the cache so the frame is dropped.
            self.cached_transform = None

            # GA-95. The TF-at-image-stamp principle above is CORRECT and stays: a box is
            # back-projected with the pose the camera actually had when the shutter opened.
            # What was missing is the exit. Only `cached_transform` was cleared, never
            # `cached_rgb`, so the SAME frame was re-looked-up on every tick -- and once its
            # stamp falls out of the TF buffer the lookup is unrecoverable BY DEFINITION,
            # because the data it needs has been evicted. In run A that produced 2271
            # retries of one dead frame over 38 minutes, each one logging at INFO from the
            # caller, while the first (and only) warn here had already been suppressed by
            # _transform_error_logged.
            #
            # A frame older than the buffer's cache window can never be transformed again.
            # Say so ONCE with the numbers, then DROP IT so the next frame gets a turn.
            #
            # Age is vs the newest TF in this tree (robot clock), not this PC's wall
            # clock. A TIAGo and the workstation can disagree by minutes; that is not
            # "the shutter opened 180s ago".
            try:
                stamp_s = Time.from_msg(self.cached_rgb.header.stamp).nanoseconds / 1e9
                latest = self.node.tf_buffer.get_latest_common_time(
                    target_frame, camera_frame)
                age = (latest.nanoseconds / 1e9) - stamp_s
            except Exception:
                age = None

            cache_s = float(config.CFG["tf"].get("buffer_cache_s", 30.0))
            if age is not None and age > cache_s:
                self.node.get_logger().warn(
                    f"Dropping frame: its stamp is {age:.1f}s old and the TF buffer holds "
                    f"only {cache_s:.0f}s, so this lookup can never succeed. "
                    f"Discarding it so the next frame is tried. ({e})")
                self.cached_rgb = None
                self.cached_depth = None
                self.all_ready = False
                # Re-arm the one-shot warn: the NEXT frame's failure is a new fact, and
                # suppressing it was half of why this went unnoticed for 38 minutes.
                if hasattr(self, '_transform_error_logged'):
                    del self._transform_error_logged
                return

            if not hasattr(self, '_transform_error_logged'):
                self.node.get_logger().warn(f"Transform not available: {e}")
                self._transform_error_logged = True

    def _check_all_ready(self):
        """Checks if we have ALL the data"""
        if (self.cached_rgb is not None and
            self.cached_depth is not None and
            self.cached_camera_info is not None and
            self.cached_transform is not None):
            if not self.all_ready:
                self.node.get_logger().info("OK - All data ready!")
                self.all_ready = True

    def get_synced_data(self, max_age=None):
        # GA-164: reachable, not hardcoded. 1.0 s was a literal default that no config could
        # reach -- the same class as the three unreachable settings found tonight -- and it
        # is the exact threshold that rejected every frame of run 035141. Raising it is a
        # real trade and should be made deliberately: TF-at-image-stamp keeps an old frame
        # GEOMETRICALLY correct, but a frame seconds old describes a scene the robot may
        # have left, and the TF buffer only holds 30 s.
        if max_age is None:
            max_age = float(config.CFG["perception"].get("max_frame_age_s", 1.0))
        if self.cached_transform is None:
            self._try_get_transform()

        missing = []
        if self.cached_rgb is None:
            missing.append("rgb")
        if self.cached_depth is None:
            missing.append("depth")
        if self.cached_camera_info is None:
            missing.append("camera_info")
        if self.cached_transform is None:
            missing.append("transform")
        if missing:
            self.node.get_logger().info(f"Synced data not ready, missing: {', '.join(missing)}")
            return None

        # Freshness is how long THIS process has held the frame, not stamp vs the
        # PC clock. Robot image stamps can be minutes behind wall time here.
        rgb_stamp = Time.from_msg(self.cached_rgb.header.stamp)
        received = self._rgb_received_mono
        age = (time.monotonic() - received) if received is not None else 0.0
        if age > max_age:
            self.node.get_logger().warn(f"Cached frame too old ({age:.2f}s), discarding")
            return None

        depth_stamp = None
        if hasattr(self.cached_depth, "header"):
            depth_stamp = Time.from_msg(self.cached_depth.header.stamp)
            stamp_delta = abs((rgb_stamp - depth_stamp).nanoseconds) / 1e9
            if stamp_delta > self.sync_tolerance_sec:
                self.node.get_logger().warn(
                    f"RGB/depth not synchronised ({stamp_delta:.3f}s), discarding the frame"
                )
                return None

        try:
            rgb_cv = self.bridge.imgmsg_to_cv2(self.cached_rgb, 'bgr8')
            depth_array = depth_to_metres(
                self.bridge.imgmsg_to_cv2(self.cached_depth, desired_encoding='passthrough'))

            if config.simulation:
                depth_array = np.nan_to_num(depth_array, nan=0.0, posinf=0.0, neginf=0.0)

            result = {
                'rgb': rgb_cv,
                'depth': depth_array,
                'camera_info': self.cached_camera_info,
                'transform': self.cached_transform,
                'timestamp': self.cached_rgb.header.stamp,
                'camera_frame': self.cached_rgb.header.frame_id
            }

            # Invalida dopo il consumo, per forzare l'attesa di un nuovo frame
            self.cached_rgb = None
            self.cached_depth = None
            self.cached_camera_info = None
            self.cached_transform = None

            return result

        except Exception as e:
            self.node.get_logger().error(f"Data conversion error: {e}")
            return None


def depth_image_to_point_cloud(depth_image, camera_intrinsics):
    """Convert depth image to 3D point cloud"""
    height, width = depth_image.shape

    v, u = np.indices((height, width))

    x = (u - camera_intrinsics[0, 2]) * depth_image / camera_intrinsics[0, 0]
    y = (v - camera_intrinsics[1, 2]) * depth_image / camera_intrinsics[1, 1]
    z = depth_image

    points = np.dstack((x, y, z)).reshape(-1, 3)

    return points


def depth_to_metres(raw):
    """GA-42. The unit comes from the ENCODING (REP 118: 16UC1 is millimetres, 32FC1 is
    metres), not from the frame's largest pixel: `max() > 20.0` divided a whole metre-valued
    frame by 1000 on one far or infinite reading."""
    raw = np.asarray(raw)
    depth = raw.astype(float)
    return depth / 1000.0 if raw.dtype == np.uint16 else depth


def statistical_outlier_removal(points_xyz, k=20, std_ratio=2.0):
    """
    Removes statistical outliers based on the mean distance from the k nearest neighbors.

    Args:
        points_xyz: Numpy array (N, 3) with XYZ coordinates
        k: Number of neighbors to consider
        std_ratio: Standard deviation multiplier for the threshold

    Returns:
        mask: Boolean array (N,) where True = valid point
    """

    # <= k, not < k (reviewed 2026-09-07): with exactly k points the k+1 query pads a
    # neighbour with inf, every mean distance is inf, the threshold is nan and NOTHING is
    # kept -- the detection lost its 3D box. Measured: n=30, k=30 -> kept 0/30.
    if len(points_xyz) <= k:
        return np.ones(len(points_xyz), dtype=bool)

    tree = KDTree(points_xyz)
    # workers=-1: the same query on every core. kNN distances are deterministic, so the
    # kept set is identical; only the wall time changes (22 cores in the run container).
    distances, _ = tree.query(points_xyz, k=k+1, workers=-1)  # +1 because it includes the point itself
    mean_distances = distances[:, 1:].mean(axis=1)  # Exclude the point itself (distance 0)

    global_mean = mean_distances.mean()
    global_std = mean_distances.std()
    threshold = global_mean + std_ratio * global_std

    mask = mean_distances < threshold
    return mask

def get_distinct_color(index):
    """
    Generate a distinct color for each index using HSV.
    Returns a ColorRGBA.
    """
    hue = (index * 0.618033988749895) % 1.0  # Golden ratio for uniform distribution
    saturation = 0.8
    value = 0.9
    r, g, b = colorsys.hsv_to_rgb(hue, saturation, value)
    return ColorRGBA(r=r, g=g, b=b, a=1.0)

def apply_nms(bboxs, labels, scores, iou_threshold=0.5, containment_threshold=None):
    """
    Apply CLASS-AWARE Non-Maximum Suppression to remove overlapping bounding boxes.
    NMS is applied SEPARATELY for each class, so boxes of different classes are never suppressed.
    This prevents removing a "book" just because it overlaps with a "table".

    Args:
        bboxs: List of bounding boxes [x1, y1, x2, y2]
        labels: List of corresponding labels
        scores: List of confidence scores
        iou_threshold: IoU threshold to consider two boxes as overlapping

    Returns:
        bboxs_filtered, labels_filtered, scores_filtered
    """
    if len(bboxs) == 0:
        return [], [], []

    if containment_threshold is None:
        try:
            from config import CFG
            containment_threshold = float(
                (CFG.get("perception", {}) or {}).get("containment_threshold", 0.85))
        except Exception:
            containment_threshold = 0.85

    # Convert to numpy arrays for easier manipulation
    bboxs = np.array(bboxs)
    scores = np.array(scores)
    labels = np.array(labels)

    # Apply NMS separately for each unique class
    unique_labels = np.unique(labels)
    all_keep_indices = []

    for label in unique_labels:
        # Get indices for this class only
        class_mask = labels == label
        class_indices = np.where(class_mask)[0]

        if len(class_indices) == 0:
            continue

        class_bboxs = bboxs[class_indices]
        class_scores = scores[class_indices]

        # Calculate areas
        x1 = class_bboxs[:, 0]
        y1 = class_bboxs[:, 1]
        x2 = class_bboxs[:, 2]
        y2 = class_bboxs[:, 3]
        areas = (x2 - x1) * (y2 - y1)

        # Sort by score (descending)
        order = class_scores.argsort()[::-1]

        keep = []
        while len(order) > 0:
            # Take element with highest score
            i = order[0]
            keep.append(i)

            if len(order) == 1:
                break

            # Calculate IoU with all other boxes OF THE SAME CLASS
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            w = np.maximum(0.0, xx2 - xx1)
            h = np.maximum(0.0, yy2 - yy1)
            intersection = w * h

            # IoU = intersection / union
            # GA-19: epsilon, as the cloud twin has. Two zero-area boxes gave 0/0 = nan,
            # and `nan <= threshold` is False, so a box was suppressed by one it does
            # not overlap. A degenerate box now has IoU 0 and survives NMS on its own.
            iou = intersection / np.maximum(areas[i] + areas[order[1:]] - intersection, 1e-9)

            # GA-276. CONTAINMENT, because IoU IS BLIND TO NESTING.
            # A small box wholly inside a large one has IoU = area_small/area_large: at a 5x
            # size difference that is 0.2, far below any sane threshold, so it survives.
            # MEASURED on run 20260902_221606: 61 same-class pairs in one frame where one box
            # is >90% contained in the other, and ALL 61 have IoU < 0.5 -- median IoU 0.185
            # against median IoS 0.934. That is the concentric stack of five `bed` boxes and
            # four `nightstand` boxes the operator saw on the live overlay.
            #
            # IoS = intersection / area of the SMALLER box. 1.0 means fully contained.
            # Boxes are visited in DESCENDING score order, so the survivor is always the
            # stronger detection and the suppressed one is always the weaker -- on the
            # measured pairs the contained box scored 0.15-0.28 against 0.49-0.60.
            #
            # THE RISK, and it is real: 2D containment is not 3D containment. Two same-class
            # objects at different depths -- a far chair seen "inside" a near chair's box --
            # nest in the image while being distinct in the world. This trades that rare
            # false merge against a measured, constant flood of duplicates. Set
            # perception.containment_threshold to 1.01 to disable it without a code edit.
            smaller = np.minimum(areas[i], areas[order[1:]])
            ios = np.where(smaller > 0, intersection / np.maximum(smaller, 1e-9), 0.0)

            # Keep only boxes below BOTH thresholds
            inds = np.where((iou <= iou_threshold) & (ios <= containment_threshold))[0]
            order = order[inds + 1]

        # Map back to original indices
        class_keep_indices = class_indices[keep]
        all_keep_indices.extend(class_keep_indices.tolist())

    # Sort by original order to maintain consistency
    all_keep_indices = sorted(all_keep_indices)

    # Return only the kept boxes
    bboxs_filtered = bboxs[all_keep_indices].tolist()
    labels_filtered = labels[all_keep_indices].tolist()
    scores_filtered = scores[all_keep_indices].tolist()

    return bboxs_filtered, labels_filtered, scores_filtered

def rectangles_overlap(rect1, rect2):
    """Check if two rectangles overlap."""
    x1_min, y1_min, x1_max, y1_max = rect1
    x2_min, y2_min, x2_max, y2_max = rect2
    
    return not (x1_max < x2_min or x2_max < x1_min or 
                y1_max < y2_min or y2_max < y1_min)

# Distinct, high-contrast BGR fills for mask overlays. Deliberately not a colormap over the
# label string: two adjacent objects of the same class would then get the same colour and the
# overlay would show one blob where the segmenter found two. Cycled by DETECTION INDEX, so
# neighbours always differ.
_MASK_COLOURS = [(60, 60, 230), (60, 200, 60), (230, 140, 40), (200, 60, 200),
                 (40, 210, 210), (230, 90, 140), (120, 200, 60), (60, 140, 230)]


def draw_masks(img, detections, alpha=0.40):
    """Paint each detection's SAM mask over the frame. GA-214.

    The live view showed boxes and labels but never the masks, so the one stage whose output
    is hardest to judge from numbers -- segmentation -- was the one stage you could not
    watch. A box tells you the detector fired; only the mask tells you whether it grabbed the
    object, half of it, or the wall behind it.

    Blended, not replaced: at alpha 0.40 the underlying pixels stay visible, so a mask that
    has slipped off its object is obvious rather than hidden under an opaque patch. A
    detection with no mask is SKIPPED silently -- that is a normal state for a box the
    segmenter declined, and drawing a rectangle in its place would imply a mask that is not
    there.
    """
    overlay = None
    for i, det in enumerate(detections):
        m = getattr(det, "mask", None)
        if m is None:
            continue
        m = np.asarray(m)
        if m.ndim == 3:
            m = m[0] if m.shape[0] in (1, 3) else m[..., 0]
        if m.shape[:2] != img.shape[:2] or not m.any():
            continue
        if overlay is None:
            overlay = img.copy()
        overlay[m.astype(bool)] = _MASK_COLOURS[i % len(_MASK_COLOURS)]
    if overlay is not None:
        cv2.addWeighted(overlay, alpha, img, 1.0 - alpha, 0, dst=img)
    return img


def draw_detections(img, detections):
    occupied_regions = [] 
    
    for detection in detections:
        x1, y1, x2, y2 = map(int, detection.bbox)
        
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Unified scene boxes carry no calibrated confidence. Showing 1.00 would
        # invent one; omit the suffix when the producing model supplied no score.
        score = getattr(detection, "score", None)
        text = detection.label if score is None else f"{detection.label}: {score:.2f}"
        (text_width, text_height), baseline = cv2.getTextSize(
            text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2
        )
        
        positions = [
            (x1, y1 - text_height - 5),         
            (x1, y2 + text_height + 5),       
            (x2 + 5, y1),                    
            (x1 - text_width - 5, y1),        
            (x1, y1 + text_height + 5),          
            ((x1 + x2) // 2 - text_width // 2, y1 - text_height - 5)  
        ]
        
        final_pos = positions[0]  
        for pos_x, pos_y in positions:
            text_rect = (
                pos_x, 
                pos_y - text_height - baseline - 5,
                pos_x + text_width,
                pos_y
            )
            
            overlaps = False
            for occupied in occupied_regions:
                if rectangles_overlap(text_rect, occupied):
                    overlaps = True
                    break
            
            if (text_rect[0] >= 0 and text_rect[1] >= 0 and 
                text_rect[2] < img.shape[1] and text_rect[3] < img.shape[0] and 
                not overlaps):
                final_pos = (pos_x, pos_y)
                occupied_regions.append(text_rect)
                break

        text_x, text_y = final_pos
        cv2.rectangle(
            img, 
            (text_x, text_y - text_height - baseline - 5), 
            (text_x + text_width, text_y), 
            (0, 255, 0), 
            -1
        )
        cv2.putText(
            img, 
            text, 
            (text_x, text_y - 5), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            0.5, 
            (0, 0, 0), 
            2
        )
    
    path_file = os.path.dirname(os.path.abspath(__file__))
    cv2.imwrite(os.path.join(path_file, "../assets/debug_detections.jpg"), img)
    return img

def compute_iou_3d(bbox1, bbox2, min_size=0.01):
    """
    Compute IoU 3D between two bounding boxes with minimum size expansion.
    Args:
        bbox1, bbox2: dict with keys 'x_min', 'y_min', '
            'z_min', 'x_max', 'y_max', 'z_max'
        min_size: minimum size for each dimension
    Returns:
        iou: float between 0 and 1  
    """

    # Espandi bbox se troppo sottili
    def expand_if_needed(bbox, min_size):
        expanded = bbox.copy()
        for axis in ['x', 'y', 'z']:
            size = bbox[f"{axis}_max"] - bbox[f"{axis}_min"]
            if size < min_size:
                center = (bbox[f"{axis}_min"] + bbox[f"{axis}_max"]) / 2
                expanded[f"{axis}_min"] = center - min_size / 2
                expanded[f"{axis}_max"] = center + min_size / 2
        return expanded

    bbox1_exp = expand_if_needed(bbox1, min_size)
    bbox2_exp = expand_if_needed(bbox2, min_size)

    # Calcola intersezione
    x_inter_min = max(bbox1_exp["x_min"], bbox2_exp["x_min"])
    x_inter_max = min(bbox1_exp["x_max"], bbox2_exp["x_max"])
    y_inter_min = max(bbox1_exp["y_min"], bbox2_exp["y_min"])
    y_inter_max = min(bbox1_exp["y_max"], bbox2_exp["y_max"])
    z_inter_min = max(bbox1_exp["z_min"], bbox2_exp["z_min"])
    z_inter_max = min(bbox1_exp["z_max"], bbox2_exp["z_max"])

    if x_inter_max < x_inter_min or y_inter_max < y_inter_min or z_inter_max < z_inter_min:
        return 0.0

    inter_volume = (x_inter_max - x_inter_min) * (y_inter_max - y_inter_min) * (z_inter_max - z_inter_min)

    volume1 = (bbox1_exp["x_max"] - bbox1_exp["x_min"]) * (bbox1_exp["y_max"] - bbox1_exp["y_min"]) * (bbox1_exp["z_max"] - bbox1_exp["z_min"])
    volume2 = (bbox2_exp["x_max"] - bbox2_exp["x_min"]) * (bbox2_exp["y_max"] - bbox2_exp["y_min"]) * (bbox2_exp["z_max"] - bbox2_exp["z_min"])

    union_volume = volume1 + volume2 - inter_volume

    if union_volume == 0:
        return 0.0

    return inter_volume / union_volume


def bbox_to_dict(bbox):
    if bbox is None:
        return None
    return {
        "x_min": float(bbox.get("x_min")),
        "x_max": float(bbox.get("x_max")),
        "y_min": float(bbox.get("y_min")),
        "y_max": float(bbox.get("y_max")),
        "z_min": float(bbox.get("z_min")),
        "z_max": float(bbox.get("z_max")),
    }

def object_to_dict( obj: Object) -> dict:
    return {
        "label": getattr(obj, "label", ""),
        "color": getattr(obj, "color", ""),
        "material": getattr(obj, "material", ""),
        "shape": getattr(obj, "shape", ""),
        "description": getattr(obj, "description", ""),
        "centroid": getattr(obj, "centroid", None),
        "bbox": bbox_to_dict(getattr(obj, "bbox", None)),
    }
