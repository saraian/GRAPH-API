#!/usr/bin/env python3
import json
import logging
import os
import sys
import time
import urllib.request
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from functools import partial
from threading import Lock

# Ensure local sibling packages and directories (e.g. cloud/) are on sys.path
# The simulator host's control surface: the dashboard pushes overlay toggles to its
# /set_config, and its own overlay reads them every frame. Polling /get_config here is
# what makes the same toggle reach /image_with_bb, the feed the bridge actually serves.
# GA-270. From config; see config.py "services". Third literal of the same shape found in
# this sweep -- the other two each cost a run.
try:
    from config import CFG as _P2_CFG
except ImportError:
    _P2_CFG = {}
_P2_SVC = (_P2_CFG.get("services", {}) or {}) if isinstance(_P2_CFG, dict) else {}
FEED_HOST = os.environ.get("FEED_HOST") or (
    f"http://{_P2_SVC.get('feed_host', '127.0.0.1')}:{_P2_SVC.get('feed_port', 7790)}")
_VIS_POLL_SECONDS = 2.0

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import rclpy  # noqa: E402
import tf2_ros  # noqa: E402
import torch  # noqa: E402
from config import CFG  # noqa: E402
from detection_archive import (  # noqa: E402
    DetectionArchive, frame_id_from_stamp, resolve_archive_dir)
from config import visibility as visibility_cfg  # noqa: E402
from cv_bridge import CvBridge  # noqa: E402
from geometry_msgs.msg import PoseStamped  # noqa: E402
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup, ReentrantCallbackGroup  # noqa: E402
from rclpy.duration import Duration  # noqa: E402
from rclpy.executors import MultiThreadedExecutor  # noqa: E402
from rclpy.logging import LoggingSeverity  # noqa: E402
from rclpy.node import Node  # noqa: E402
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy  # noqa: E402
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
    draw_cloud,
    init_bbox_publisher,
    mask_list_to_centroid_and_bbox,
    mask_list_to_pointcloud2,
    numpy_to_base64,
    publish_individual_pointclouds_by_id,
    vlm_call,
)
from detection_pipeline import DetectionPipelineMixin  # noqa: E402
from input_output import PerceptionIOMixin  # noqa: E402
from models import VitSam, write_vitsam_status  # noqa: E402
from object_info import Object  # noqa: E402
from perception_utils import compute_fov_volume_from_depth, get_project_root  # noqa: E402
from tf_transformations import euler_from_quaternion, quaternion_inverse, quaternion_multiply  # noqa: E402
from utils import draw_detections, draw_masks  # noqa: E402
from sensor_msgs_py import point_cloud2 as _pc2  # noqa: E402
from vlm_call import VlmClient  # noqa: E402
from world_model import wm  # noqa: E402

from lost3dsg.msg import Bbox3d, Bbox3dArray, ObjectDescription, ObjectDescriptionArray  # noqa: E402

# Do not probe torch CUDA during module import. The local VitSAM path is ONNX-based and
# selects its provider in models.VitSam; probing here used to emit a misleading CUDA warning
# before the node had even selected its backend. The optional OWLv2 path configures its own
# device when it is instantiated.

PROJECT_ROOT = get_project_root(__file__)
LOG_DIR = os.path.join(PROJECT_ROOT, "output")
os.makedirs(LOG_DIR, exist_ok=True)

module_logger = logging.getLogger("perception_module")
module_logger.setLevel(logging.DEBUG)
if not module_logger.handlers:
    # GA-79: delay=True opens the file on the FIRST EMIT, not at import. This created a
    # timestamped log the moment anything imported this module -- and the pre-flight gate
    # imports all four nodes on every run, so the probe that certifies the tree was writing
    # into the tree each time it passed. A probe with a filesystem side effect is a probe
    # that cannot be run freely, and that one is the one we most want to run freely.
    #
    # stdlib does it; no lazy-handler wrapper needed.
    file_handler = logging.FileHandler(
        os.path.join(LOG_DIR, f"perception_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
        delay=True,
    )
    file_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    module_logger.addHandler(file_handler)
else:
    file_handler = module_logger.handlers[0]

DESCRIPTION_FIELDS = ("description", "color", "material", "shape")

# WN1. Where the latency record lives -- the same two paths detection_pipeline writes, kept
# as one constant so the cycle-time stamp and the detection-span stamp land in one file.
_METRICS_OUTPUT_ROOT = os.environ.get("GRAPH_API_OUTPUT_DIR", "/root/exchange/output")
LATENCY_JSON_PATHS = (
    "/tmp/perception_latencies.json",
    os.path.join(_METRICS_OUTPUT_ROOT, "perception_latencies.json"),
)
# GA-334. The per-cycle series beside the snapshot: one JSON line per completed cycle, the
# snapshot's keys plus `t`, `cycle`, `frame_id`, `n_detections`. graph_api_bridge._cycle_seq
# counts its lines as the cycle number; until 2026-09-07 nothing wrote it.
LATENCY_JSONL_PATHS = tuple(p[:-len(".json")] + ".jsonl" for p in LATENCY_JSON_PATHS)


def _truncate_cycle_series():
    """GA-334. A fresh series per node start: the bridge counts lines, and /tmp outlives
    the run, so an earlier run's rows would otherwise inflate this one's cycle number
    (the bridge also resets its tally when the file shrinks)."""
    for target_path in LATENCY_JSONL_PATHS:
        try:
            os.makedirs(os.path.dirname(target_path), exist_ok=True)
            open(target_path, "w").close()
        except OSError:
            pass


def _append_cycle_row(row):
    line = json.dumps(row) + "\n"
    for target_path in LATENCY_JSONL_PATHS:
        try:
            with open(target_path, "a") as f:
                f.write(line)
        except OSError:
            pass

# ponytail: fixed cap for images sent to the VLM; make it a CFG["vlm"] knob if a
# model ever needs finer input. The base64 payload dominates vlm_ms, not the answer.
VLM_IMAGE_MAX_SIDE = 512

# GA-353. Depth of the ground-truth semantic frame cache, in frames. Derived, not tuned:
# the feed host renders at most ~11 f/s at 1280x960 (measured 89 ms/frame, config note) and
# the archive looks the frame up after the detection span, p95 ~9.4 s on 192014, so the frame
# must survive 11 x 9.4 = ~103 arrivals; 120 with margin. Compressed blobs (~130 KB each) make
# that ~16 MB, where 8 DECODED frames were already 39 MB. Logged at startup and carried in the
# per-cycle row as `gt_semantic_hit`, so a bundle can show whether the depth held.
GT_SEMANTIC_CACHE_FRAMES = 120


def _encode_for_vlm(img):
    h, w = img.shape[:2]
    scale = VLM_IMAGE_MAX_SIDE / float(max(h, w))
    if scale < 1.0:
        img = cv2.resize(img, (max(1, round(w * scale)), max(1, round(h * scale))), interpolation=cv2.INTER_AREA)
    return numpy_to_base64(img)


def _description_status(res):
    """W6. Which route produced a description record: the four-way split the run can count.

    `unanswered` -- nothing landed for this object this cycle (describer still in flight,
    or the result was refused by the W2 box check). `call_failed` / `parse_failed` /
    `model_abstained` / `ok` -- read from the provenance the VLM client now stamps on every
    answer (crop_context's classify_description_result vocabulary). Grid cells carry no
    provenance, so they fall to the content test, same vocabulary. An empty record is
    `unanswered`, never `model_abstained`: a description that never arrived is not a refusal.
    """
    # Reviewed 2026-09-07: a harvested result that carries ONLY the origin stamp is a call
    # that produced nothing (a failed grid cell), not a model abstention.
    if not res or set(res) <= {"origin"}:
        return "unanswered"
    if not res:
        return "unanswered"
    prov = res.get("provenance") or {}
    status = prov.get("status")
    if status:
        return status
    err = prov.get("error")
    if err:
        return err                    # older records carried only the error key
    if str(res.get("description", "")).strip().lower() in ("", "unknown", "none", "n/a"):
        return "model_abstained"
    return "ok"


def _bbox_iou(a, b):
    """W2. IoU of two (x0, y0, x1, y1) pixel boxes; 0.0 when either is not one.

    Used to decide whether a deferred VLM result was computed for the detection
    it is about to decorate. 0.0 on a malformed box REFUSES the result: the cost
    of a wrongly-dropped description is one more abstention, the cost of a
    wrongly-attached one is a wrong attribute on the map.
    """
    try:
        ax0, ay0, ax1, ay1 = a
        bx0, by0, bx1, by1 = b
    except (TypeError, ValueError):
        return 0.0
    iw = min(ax1, bx1) - max(ax0, bx0)
    ih = min(ay1, by1) - max(ay0, by0)
    if iw <= 0.0 or ih <= 0.0:
        return 0.0
    inter = iw * ih
    union = (ax1 - ax0) * (ay1 - ay0) + (bx1 - bx0) * (by1 - by0) - inter
    return inter / union if union > 0.0 else 0.0


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
            # The unified VLM response supplies the 2D boxes, so this path does
            # not need to load a separate OWLv2 detector.
            self.detector = None
            try:
                self.vitsam = VitSam(utils.ENCODER_VITSAM_PATH, utils.DECODER_VITSAM_PATH)
            except Exception:
                # Do not leave the host-side feed waiting for its timeout when model startup
                # fails before VitSam can publish the final status itself.
                write_vitsam_status("failed")
                raise
            self.file_logger.info("Using unified whole-scene VLM boxes with local VitSAM")
        else:
            self.detector = None
            self.vitsam = None
            self.file_logger.info(f"Using Cloud Perception Backend: {backend_type}")
        self.vlm = VlmClient(vlm_call_fn=vlm_call, image_encoder_fn=_encode_for_vlm,
                             crop_call_fn=partial(vlm_call, timeout=CFG["vlm"]["crop_timeout"]))
        # GA-215. DEBUG OVERLAY: publish the annotated frame at every perception stage, as
        # each result appears, rather than once at the end of the cycle. Off by default --
        # it costs an encode and a publish per stage, and a measured run should not pay for
        # a debugging view. Env wins over config so it can be flipped for one run.
        self._debug_overlay = (
            os.environ.get("PERCEPTION_DEBUG", "").lower() in ("1", "true", "yes", "on")
            or bool(CFG.get("perception", {}).get("debug_overlay", False)))
        if self._debug_overlay:
            self.log_both("info", "[DEBUG] stage-by-stage overlay ON: /image_with_bb is "
                                  "republished after detection, segmentation and geometry")
            # GA-218. rtabmap's cloud, subscribed HERE because this node holds the transform
            # the projection needs. Latest-only (depth 1): an old cloud drawn against a new
            # frame is worse than no cloud, and queueing them would guarantee exactly that.
            self._latest_cloud = None
            self.create_subscription(
                PointCloud2, "/rtabmap/cloud_map", self._on_cloud_map,
                QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT,
                           history=HistoryPolicy.KEEP_LAST))
            self.log_both("info", "[DEBUG] subscribed to /rtabmap/cloud_map for the voxel overlay")

        # GA-210. ONE executor for the life of the node, so a description outlives the cycle
        # that asked for it. Per-call executors joined on exit, which is what made the
        # describer synchronous.
        self._vlm_executor = ThreadPoolExecutor(
            max_workers=int(CFG.get("vlm", {}).get("crop_concurrency", 4)),
            thread_name_prefix="crop_vlm")
        self._vlm_pending = {}
        # W2. For every pending label, the frame and 2D box of the crop the description
        # is being computed FROM. Set at submit, popped at harvest, injected into the
        # result so `_build_descriptions` can refuse one whose object is no longer current.
        self._vlm_origin = {}

        # GA-95: one source for the cache window. utils.CameraData drops a frame whose stamp
        # is older than this, so the two must not drift apart.
        self.tf_buffer = tf2_ros.Buffer(
            cache_time=Duration(seconds=float(CFG["tf"].get("buffer_cache_s", 30.0))))
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self._init_publishers()
        self._init_subscribers()
        self._init_state()

        self._io_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="perception_io")
        self._cycle_count = 0
        _truncate_cycle_series()
        self._undeliverable_fields = set()
        self.clear_accumulated_markers()
        self._create_timers()

        self.get_logger().info(f"Log saved in: {file_handler.baseFilename}")


        # GA-167. Built ONCE, here, in __init__. It previously sat inside process_crop_vlm --
        # once PER CROP -- because I anchored the edit on a line I assumed was in __init__ and
        # asserted only that the anchor was UNIQUE. It was unique and it was in the wrong
        # function. A unique anchor is not a correct anchor.
        #
        # The cost was not cosmetic: cycle 1 of run 042828 called _archive_detections before
        # any crop had been processed, found no archive, and returned silently -- ELEVEN
        # detections never recorded. And rebuilding the object 29 times reset _frames_written,
        # so the frame dedupe was not running at all; three frames for three frame_ids held
        # only because rewriting the same path is idempotent. It LOOKED like it worked.
        self.detection_archive = DetectionArchive(
            resolve_archive_dir(CFG, PROJECT_ROOT),
            enabled=bool(CFG["archive"]["per_detection"]),
            logger=self.get_logger())
        if self.detection_archive.enabled:
            self.log_both("info", f"[ARCHIVE] per-detection archiving ON -> "
                                  f"{self.detection_archive.root}")
            # GT-ONLY. Subscribed ONLY when archiving is on, so the runtime path cannot
            # acquire a ground-truth channel as a side effect of anything else. Keyed by
            # exact stamp: a GT label from a neighbouring frame would be worse than none.
            self._gt_semantic = {}
            from sensor_msgs.msg import CompressedImage as _CompressedImage
            # GA-353: the subscription queue must not be the shallow cache in disguise; the
            # cache depth is the bound, the queue only has to keep up with the feed rate.
            self.create_subscription(_CompressedImage, "/gt/semantic_instance",
                                     self._gt_semantic_callback, 30)
            self.log_both("info", f"[GT] semantic frame cache: {GT_SEMANTIC_CACHE_FRAMES} frames "
                                  f"(compressed, decoded at lookup)")

        # The launcher opens the feed socket before ROS so habitat_feed_node can connect, but
        # it must not release the first simulator frame until the actual perception process has
        # finished VitSAM warmup AND completed its own subscriptions/timers setup.  The marker is
        # atomically written into the shared run directory by models.write_vitsam_status().
        write_vitsam_status("ready")
        self.get_logger().info("Perception startup ready; releasing the Habitat feed gate")

    def _on_cloud_map(self, msg):
        """Keep the newest cloud as an (N,3) array. GA-218.

        Decoded once on arrival rather than per drawn frame: the cloud changes far less often
        than the camera does, and decoding it three times per cycle to draw three stages
        would put the cost in the wrong place.
        """
        try:
            # Same call room_manager already uses in production against this topic.
            raw = list(_pc2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True))
            # `len(None)` on an empty cloud would raise into the handler below and log a
            # decode failure that never happened -- an empty cloud is a normal state early in
            # a run, not an error.
            self._latest_cloud = (
                np.array([[float(q[0]), float(q[1]), float(q[2])] for q in raw],
                         dtype=np.float32) if raw else None)
        except Exception as exc:
            self.get_logger().warn(f"[DEBUG] cloud decode failed: {exc}")
            self._latest_cloud = None

    def _init_publishers(self):
        qos_latched = QoSProfile(depth=10, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        qos_default = 10

        self.pub_image = self.create_publisher(Image, "/image_with_bb", qos_latched)
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
        # GA-236. The frames to watch come from CONFIG, not a TIAGo literal.
        #
        # This was ["head_1_joint", "head_2_joint", "habitat_camera"] plus two wheel joints.
        # Four of those five are TIAGo frames that do not exist in a Habitat deployment, so
        # joint_callback failed four lookups every second -- 383 logged occurrences of
        # head_1_joint alone in a 61-minute run.
        #
        # The noise was not the real fault. The failures are caught and the loop continues,
        # so `motion_scores` ended up with ONE entry and
        #     moving = sum(motion_scores) / len(motion_scores) >= threshold
        # computed the mean of a five-source design from a single reading, with nothing
        # anywhere reporting that four fifths of the intended evidence was missing. A
        # statistic that looks like an aggregate and is one measurement.
        _frames_cfg = (CFG.get("frames", {}) or {})
        self.head_joints = list(_frames_cfg.get("motion_watch") or ["habitat_camera"])
        self.base_joints = list(_frames_cfg.get("motion_watch_base") or [])
        self._motion_absent_logged = False
        self.position_threshold = 0.05
        # GA-359. The pose source, and the gate that refuses to place a box on a stale
        # localisation. Under `rtabmap` the localiser publishes /localization_pose ONLY while
        # localised; a cycle with no pose newer than `localization_max_age_s` is SKIPPED and
        # COUNTED (`cycles_skipped_unlocalised` in the latency row). Never a GT fallback: a box
        # placed with the simulator's true pose in an rtabmap run is the contamination a12
        # exists to refuse (rule 14, rule 11: hold, do not invent). Under `simulator` the gate
        # is off and the counter stays 0 -- measured, not absent.
        self.pose_source = os.environ.get("FEED_POSE_SOURCE", "simulator").strip().lower()
        if self.pose_source not in ("simulator", "rtabmap"):
            raise ValueError(f"FEED_POSE_SOURCE must be 'simulator' or 'rtabmap', got {self.pose_source!r}")
        self.localization_max_age_s = float(
            (CFG.get("frames", {}) or {}).get("localization_max_age_s", 5.0))
        self._last_localization_time = None      # monotonic seconds of the newest /localization_pose
        self.cycles_skipped_unlocalised = 0
        if self.pose_source == "rtabmap":
            from geometry_msgs.msg import PoseWithCovarianceStamped as _PoseCov
            # rtabmap's nodes run in the `rtabmap` namespace (om6 reads /rtabmap/map the same
            # way); the absolute /localization_pose name would never receive a message (rule 49,
            # simulator lane, 2026-09-07). NB: with rtabmap's default
            # pub_loc_pose_only_when_localizing=false the pose is published every frame whether
            # localised or not, and this AGE gate cannot trip -- it is live only when the launch
            # sets that parameter true (simulator's line); a run must report which.
            self.localization_pose_topic = str(
                (CFG.get("frames", {}) or {}).get("localization_pose_topic", "/rtabmap/localization_pose"))
            self.create_subscription(_PoseCov, self.localization_pose_topic, self._localization_pose_callback, 10)
            self.log_both("info", f"[POSE] pose_source=rtabmap: cycles run only with a {self.localization_pose_topic} "
                                  f"younger than {self.localization_max_age_s:.1f} s")
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
        # Save/publish the physical camera pose. This is deliberately distinct
        # from frames.camera, the optical frame used for RGB-D projection.
        self.agent_pose_frame = (CFG.get("frames", {}) or {}).get(
            "agent_pose", "habitat_camera"
        )

    def _create_timers(self):
        self.create_timer(0.5, self._perception_timer_callback, callback_group=self.perception_cb_group)
        self.create_timer(1.0, self.joint_callback, callback_group=self.perception_cb_group)
        self.get_logger().info("Perception timers created")


    def log_both(self, level, message):
        try:
            # The ROS record must carry the SEVERITY, not merely mention it. This used to
            # call .info() for every level with the level as a text prefix, so
            # log_both("error", ...) produced an INFO record -- and ten consecutive
            # perception-cycle crashes were invisible to any error-level scan of the node's
            # output while sitting in plain sight in the file log. Rule 2: assert what
            # answered, not that something answered.
            # GA-164b: each severity is emitted from its OWN call site. rclpy caches a
            # log call site by caller location and raises "Logger severity cannot be
            # changed between calls" when the same site is used for a second distinct
            # severity -- so funnelling four levels through one `emit(...)` line threw on
            # every level change. 238 of those in run 035141, exactly 2 per cycle.
            #
            # NOT THE CAUSE OF THAT RUN'S FAILURE, and worth stating so nobody re-fixes it:
            # the same error appears 839 times in run 184822, which detected normally. It
            # is noise that made the log unreadable, not the thing that stopped the cycle.
            text = f"[{level.upper()}] {message}"
            if level == "debug":
                self.get_logger().debug(text)
            elif level == "warn":
                self.get_logger().warn(text)
            elif level == "error":
                self.get_logger().error(text)
            else:
                self.get_logger().info(text)
        except Exception as exc:
            self.get_logger().error(f"Error: {exc}")

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
            self.get_logger().error(f"Error: {exc}")

    def _abort_if_moving(self, stage):
        if not self.is_stationary:
            self.processing_interrupted = True
            self.log_both("warn", f"Robot moved during {stage}: perception aborted")
            return True
        return False

    def _localization_pose_callback(self, _msg):
        self._last_localization_time = time.monotonic()

    def _localised(self):
        """GA-359. True when a cycle may place boxes: always under `simulator`; under `rtabmap`
        only while a /localization_pose younger than localization_max_age_s has arrived."""
        if getattr(self, "pose_source", "simulator") != "rtabmap":
            return True
        t = getattr(self, "_last_localization_time", None)
        return t is not None and (time.monotonic() - t) <= self.localization_max_age_s

    def _run_perception_cycle(self, reason=""):
        if not self._localised():
            # Skipped, not deferred with a stale pose. Logged at info once per skip because the
            # count is the evidence (rule 5); the hold continues on the feed side by itself.
            self.cycles_skipped_unlocalised += 1
            self.log_both("info", f"[POSE] cycle skipped: not localised (no /localization_pose within "
                                  f"{self.localization_max_age_s:.1f} s); skipped so far "
                                  f"{self.cycles_skipped_unlocalised}")
            return
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
        # Rule 14, owner's ruling 2026-08-31: the handler that stood here is REMOVED, not
        # made louder. It caught every cycle error, logged it and continued, so the node
        # reported healthy through ten consecutive total failures while publishing nothing
        # -- and three lanes spent an evening looking for the cause in correctly-wired code
        # downstream. A perception cycle that cannot complete is a missing component, and a
        # missing component must stop the run rather than produce an empty one that looks
        # like a quiet scene.
        #
        # The lock release stays: `finally` without `except` propagates the exception.
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

    def process_crop_grid(self, batch):
        """GA-209. One VLM call for a batch of crops. -> {"__grid__": {label: result}}

        WHY THIS EXISTS. The provider is not the bottleneck -- measured 2026-09-01, regolo
        gemma4-31b 0.55 s/image against openrouter qwen2.5-vl-72b 0.56 s, so a bigger model
        buys nothing. The cost is the STRAGGLER: one call in five hung near the 15 s timeout
        against a 0.55 s median, and a concurrent batch finishes when its slowest member
        does, so a five-crop cycle measured vlm_ms = 13,713. Measured fix, same provider:
        five separate calls at concurrency 8 took 1.69 s, one 3x2 grid took 0.88 s. 1.91x,
        by removing the parallelism rather than by tuning it.

        FALLBACK IS PER-CELL, NOT PER-BATCH. If the reply omits or garbles some cells, only
        those go back through the single-crop path. A grid that half-worked must cost a
        retry on the missing half, never a confident description attached to the wrong
        object -- which is why crop_grid.parse joins on the printed cell number instead of
        on array position.
        """
        import crop_grid

        labels = [c["label"] for c in batch]
        images = [c["cropped"] for c in batch]
        out = {}
        try:
            canvas, rows, cols, _cell = crop_grid.compose(images)
            if canvas is None:
                raise ValueError("empty grid")
            prompt = crop_grid.build_prompt(labels, rows, cols)
            reply = self.vlm.call_image_prompt(canvas, prompt)
            cells, missing = crop_grid.parse(reply, len(batch))
        except Exception as exc:
            self.get_logger().warn(
                f"[VLM] grid call failed ({type(exc).__name__}: {exc}); "
                f"falling back to {len(batch)} single-crop call(s)")
            cells, missing = [None] * len(batch), list(range(len(batch)))

        for i, cell in enumerate(cells):
            if cell is not None:
                out[labels[i]] = dict(cell, label=labels[i])
        if missing:
            self.log_both("info", f"[VLM] grid answered {len(batch) - len(missing)}/"
                                  f"{len(batch)} cells; {len(missing)} fall back to single calls")
            for i in missing:
                try:
                    out[labels[i]] = self.process_crop_vlm(batch[i]) or {}
                except Exception as exc:
                    self.get_logger().error(f"Crop VLM fallback failed for {labels[i]}: {exc}")
                    out[labels[i]] = {}
        return {"__grid__": out}

    # -------------------------------------------------------------------------
    # TODO (Lazy Two-Stage Crop Refinement & Property Separation):
    # - Stage 1 (Hot Detection Cycle): Detector produces primary class noun (e.g. "chair").
    # - Stage 2 (Lazy on Admission/Ambiguity): When an object is admitted or contested
    #   by the ontological layer, trigger this asynchronous crop VLM
    #   query to refine the noun (e.g. "office chair") and extract extended traits.
    # - Standard properties ("color", "material", "shape", "description") remain in the
    #   primary metadata schema, while extended attributes ("style", "affordances", "state")
    #   populate the instance attribute set for deep ontological alignment.
    # -------------------------------------------------------------------------
    def lazy_refine_object_crop(self, obj, crop_image):
        """Asynchronous / Lazy refinement of object semantics and attributes."""
        pass

    def publish_objects(self):
        # WN1. The TRUE cycle wall time, entry to completion of this method. `total_ms`
        # covers only run_detection's own span and was logged as "total cycle" -- ~2x off
        # against the publish_objects wall. Interrupted/empty cycles do not write one:
        # a cycle_ms exists only for a cycle that completed.
        t_cycle = time.time()
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
            # LAT-2: hand over the FOV computed at the top of this cycle instead of
            # letting the empty-state path project the whole depth image a second time.
            self.publish_empty_state(depth, camera_info, cycle_stamp, fov_volume=fov_volume)
            return

        # H12: save_visualizations no longer takes the root — it resolves the same
        # bundle root the crops and the per-cycle JSON use.
        self._io_executor.submit(self.save_visualizations, image_raw.copy(), depth.copy(), list(detections))

        # Per-stage wall time of everything after run_detection, written into the latency
        # record as `stages_ms`. Until 2026-09-06 the cycle had ONE number for this half
        # (cycle_ms - total_ms) and the log's timestamps were the only way to split it.
        stages = {}
        t_stage = time.time()

        def _mark(name):
            nonlocal t_stage
            now = time.time()
            stages[name] = round((now - t_stage) * 1000.0, 1)
            t_stage = now

        # GA-215: the detector has answered and the masks exist. Show them NOW -- everything
        # below takes time, and until today none of it was visible until all of it finished.
        self._debug_stage("detector+masks", image_raw, detections, camera_info, cycle_stamp)
        self._assign_instance_labels(detections)
        self._debug_stage("labelled", image_raw, detections, camera_info, cycle_stamp)
        _mark("labels")
        centroids_3d, bboxes_3d = self._compute_3d_geometry(detections, depth, camera_info, camera_data["transform"])
        self._debug_stage("3d-geometry", image_raw, detections, camera_info, cycle_stamp,
                          bboxes_3d=bboxes_3d, transform=camera_data["transform"], depth=depth)
        _mark("geometry")
        self._add_pca_orientation(detections, bboxes_3d, depth, camera_info, camera_data["transform"])
        _mark("pca")
        self._publish_image_with_bb(image_raw, detections, bboxes_3d, camera_info, camera_data["transform"], cycle_stamp, depth)
        _mark("image_with_bb")
        # H12: prepare_crops resolves the bundle root itself; PROJECT_ROOT is no longer
        # threaded through. W2: the frame key is mandatory provenance.
        crops_data = self.prepare_crops(detections, image_raw,
                                        frame_id_from_stamp(cycle_stamp))
        _mark("crops")
        # GA-172: archived AFTER prepare_crops so the row can carry `crop_meta`, and still
        # BEFORE the VLM batch so a describer failure cannot cost the record of what was
        # detected. It used to run before prepare_crops, so crop_meta DID NOT YET EXIST when
        # the row was written -- every row carried null, and the four-way status that
        # separates `unanswerable` from `model_abstained` was unrecoverable. The status was
        # computed correctly and thrown away one line too early.
        self._archive_detections(detections, bboxes_3d, centroids_3d, image_raw,
                                 camera_data.get("transform"), cycle_stamp,
                                 crops_data=crops_data, depth=depth)
        _mark("archive")
        self._attach_crop_embeddings(detections, crops_data)
        _mark("embeddings")
        # All perception backends consume the same whole-frame VLM reply. Reuse its
        # attributes here so the VLM workflow remains one request regardless of where
        # detection and segmentation are supplied.
        vlm_results = self._unified_scene_description_results(detections)
        descriptions = self._build_descriptions(detections, vlm_results, crops_data)
        _mark("describer_queue")
        self._publish_bbox_array(detections, bboxes_3d, fov_volume, cycle_stamp)
        self._publish_description_array(detections, descriptions, cycle_stamp)
        self._publish_late_descriptions(cycle_stamp)
        self._update_world_model(detections, centroids_3d, bboxes_3d, descriptions)
        self._queue_perceptions_json()
        self._publish_agent_pose(cycle_stamp)
        _mark("publish")
        self._record_cycle_ms(time.time() - t_cycle, stages=stages,
                              frame_id=frame_id_from_stamp(cycle_stamp),
                              n_detections=len(detections))
        self.waiting_for_input = False
        self.log_both("info", "publish_objects completed")

    def _record_cycle_ms(self, cycle_seconds, stages=None, frame_id=None, n_detections=None):
        """WN1. Stamp the completed cycle's wall time into the latency record.

        `total_ms` (run_detection's own span) stays for its existing readers; `cycle_ms`
        is the number any latency claim must quote. Same two paths the detection
        pipeline writes, so one file carries both. GA-334: the same record, plus the
        cycle's identity, is also appended as one line to the `.jsonl` series beside it,
        on the io executor so the cycle does not pay for the write.
        """
        lat = getattr(self, "latest_latencies", None)
        if not isinstance(lat, dict):
            lat = {}
            self.latest_latencies = lat
        lat["cycle_ms"] = round(cycle_seconds * 1000.0, 1)
        if stages:
            lat["stages_ms"] = dict(stages)
        lat["last_updated"] = time.time()
        for target_path in LATENCY_JSON_PATHS:
            try:
                os.makedirs(os.path.dirname(target_path), exist_ok=True)
                with open(target_path, "w") as f:
                    json.dump(lat, f, indent=2)
            except Exception:
                pass
        self._cycle_count = getattr(self, "_cycle_count", 0) + 1
        # GA-353: None when archiving is off (not measured), else whether this cycle's exact
        # semantic frame was still cached when the archive looked. Only a real bool is a
        # measurement; anything else reads as "not measured".
        hit = getattr(self, "_last_gt_semantic_hit", None)
        # GA-359: which authority placed this cycle's boxes, and how many cycles were refused
        # for want of a fresh localisation up to now. Both additive keys.
        skipped = getattr(self, "cycles_skipped_unlocalised", None)
        row = dict(lat, t=lat["last_updated"], cycle=self._cycle_count,
                   frame_id=frame_id, n_detections=n_detections,
                   gt_semantic_hit=(hit if isinstance(hit, bool) else None),
                   pose_source=getattr(self, "pose_source", None) if isinstance(getattr(self, "pose_source", None), str) else None,
                   localization_pose_topic=(getattr(self, "localization_pose_topic", None)
                                            if isinstance(getattr(self, "localization_pose_topic", None), str) else None),
                   cycles_skipped_unlocalised=(skipped if isinstance(skipped, int) else None))
        # WRITTEN SYNCHRONOUSLY, NOT QUEUED. Owner 2026-09-11: "measured time is critical,
        # especially perception loop latency." This row IS that measurement, and going through
        # _io_executor lost it: MEASURED across the archive, 3 of the 6 bundles whose runs
        # completed cycles have no perception_latencies.jsonl at all. A queued task is dropped
        # when the node exits abruptly -- which is how every run that died on a node ends -- so
        # the timing series went missing exactly in the runs whose timing needs explaining.
        # The cost is one short append per cycle against a cycle that takes seconds.
        _append_cycle_row(row)

    # last /get_config answer and when it was fetched; rebound per instance on use
    _vis_live = {}
    _vis_at = 0.0

    def _live_visibility(self):
        """Whatever the dashboard last pushed to the simulator host's /set_config.

        The host keeps that state in memory and its own overlay reads it every frame;
        polling it here is what makes the same toggle reach /image_with_bb, which is
        the feed the bridge actually serves. Throttled, and best-effort: with no host
        reachable this returns the last answer (or {}) and the yaml defaults stand.

        Rule 14, argued rather than flagged -- the swallow has two halves and only one
        is a designed default:

        * **No answer yet** -> `{}` -> `visibility_cfg({})` resolves entirely from the
          yaml. That is the documented default and is genuinely best-effort.
        * **A stale answer** -> the last slider position the host ever pushed keeps
          gating the boxes for the rest of the run, with nothing recording that it is
          stale. This is a known ceiling, and it matters more than it did when these
          values only drove a viewer: they now gate what is DRAWN AS EVIDENCE.

        The upgrade path is a timestamp beside `_vis_at` and treating an answer older
        than some age as absent -- not a wider `except`.
        """
        now = time.monotonic()
        if now - self._vis_at > _VIS_POLL_SECONDS:
            self._vis_at = now
            try:
                with urllib.request.urlopen(f"{FEED_HOST}/get_config", timeout=0.5) as resp:
                    self._vis_live = json.loads(resp.read()).get("config") or {}
            except Exception:
                pass
        return self._vis_live

    def _debug_stage(self, stage, image_raw, detections, camera_info, stamp,
                     bboxes_3d=None, transform=None, depth=None):
        """Publish what is known RIGHT NOW. GA-215; no-op unless debug overlay is on.

        Each stage overwrites the same topic, so the viewer shows one frame being refined
        rather than four competing streams. The stage name is burned into the image: without
        it a half-finished frame is indistinguishable from a finished one that found less,
        which is the whole failure this view exists to make visible.
        """
        if not self._debug_overlay:
            return
        try:
            drawn = image_raw.copy()
            # GA-218: the cloud goes down FIRST, so masks, boxes and labels stay readable on
            # top of it. Needs this frame's transform, so it only appears from the geometry
            # stage onward -- the earlier stages have no transform to project with, and
            # drawing with a neighbouring frame's transform is the mistake this avoids.
            cloud = getattr(self, "_latest_cloud", None)
            if cloud is not None and transform is not None:
                draw_cloud(drawn, cloud, camera_info, transform)
            if detections:
                draw_masks(drawn, detections)
                # 2D boxes at every stage: they are what the DETECTOR said, and keeping them
                # visible after the 3D boxes appear is how a box that failed to lift shows up.
                draw_detections(drawn, detections)
                if bboxes_3d and transform is not None:
                    min_vis, tol_abs, tol_rel = visibility_cfg(self._live_visibility())
                    draw_boxes_3d(drawn, bboxes_3d,
                                  [getattr(d, "instance_label", None) or d.label for d in detections],
                                  camera_info, transform, depth,
                                  min_visible_points=min_vis, tol_abs=tol_abs, tol_rel=tol_rel)
            _n_cloud = 0 if getattr(self, "_latest_cloud", None) is None else len(self._latest_cloud)
            cv2.putText(drawn, f"[{stage}] {len(detections)} det  {_n_cloud} cloud pts",
                        (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3)
            cv2.putText(drawn, f"[{stage}] {len(detections)} det  {_n_cloud} cloud pts",
                        (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (60, 230, 60), 1)
            msg = self.bridge.cv2_to_imgmsg(drawn, "bgr8")
            msg.header.stamp = stamp
            msg.header.frame_id = camera_info.header.frame_id
            self.pub_image.publish(msg)
        except Exception as exc:
            # A debugging view must never take the cycle down with it.
            self.get_logger().warn(f"[DEBUG] stage '{stage}' overlay failed: {exc}")

    def _publish_image_with_bb(self, image_raw, detections, bboxes_3d, camera_info, transform, stamp, depth=None):
        """/image_with_bb shows the 3D boxes projected back into the frame they were
        measured from, under the same visibility rule as the simulator overlay; the
        flat 2D rectangle is kept only for detections that got no 3D box. Published
        on every cycle, empty ones included, so a subscriber that joins late (rviz)
        sees the latest frame instead of "No image"."""
        drawn = image_raw.copy()
        if detections:
            # GA-214: MASKS FIRST, so boxes and labels stay legible on top of the fill.
            # Gated on the `seg` flag the viewer already sends through /set_config, and ON by
            # default: the segmenter's output is the hardest stage to judge from numbers, and
            # a live view that hides it leaves the one thing you cannot check afterwards.
            _viz = self._live_visibility() or {}
            _seg = str(_viz.get("seg", "1")).lower() not in ("0", "false", "off", "no")
            if _seg:
                draw_masks(drawn, detections)
            flat = [det for det, box in zip(detections, bboxes_3d) if not box]
            if flat:
                draw_detections(drawn, flat)
            # The overlay gate, resolved from the yaml and from whatever the dashboard
            # last pushed to the host -- so /image_with_bb obeys the same rule as the
            # simulator's own overlay, which is what this method's docstring claims.
            min_vis, tol_abs, tol_rel = visibility_cfg(self._live_visibility())
            draw_boxes_3d(drawn, bboxes_3d, [det.instance_label for det in detections], camera_info, transform, depth,
                          min_visible_points=min_vis, tol_abs=tol_abs, tol_rel=tol_rel)
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
        points = []
        centroids_3d, bboxes_3d = mask_list_to_centroid_and_bbox(
            all_masks, instance_labels, depth, camera_info,
            node=self,
            bbox_marker_pub=self.bbox_marker_pub,
            centroid_marker_pub=self.centroid_marker_pub,
            transform=transform,
            points_out=points,
        )
        # The map-frame points each box came from, kept for _add_pca_orientation. It used
        # to re-run _filter_object_points (projection + k=30 outlier removal on up to
        # 20,000 points) per detection -- the same call on the same mask with the same
        # parameters, so the same points. Measured on 20260906_203744: the two stages
        # were 1.19 s and the bulk of 1.85 s in a 14-detection cycle.
        for det, pts_map in zip(detections, points):
            det.points_map = pts_map
        return centroids_3d, bboxes_3d

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
        # H12: the same root every other writer in this seam uses — the bundle, when set.
        from input_output import resolve_output_root
        path = os.path.join(resolve_output_root(), "clip_embeddings.json")
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "w") as f:
                json.dump(embeddings, f)
        except Exception as exc:
            self.log_both("error", f"CLIP embedding dump failed: {exc}")

    def _run_crop_vlm_batch(self, crops_data):
        """Submit this cycle's crops, return the ones that have ALREADY finished. GA-210.

        THE CYCLE NO LONGER WAITS FOR THE DESCRIBER. The previous version used futures and
        then joined on them -- `with ThreadPoolExecutor(...)` blocks on exit -- so a single
        slow crop stalled the whole pipeline. Measured on run 20260901_151714: vlm_ms 13,713
        for five crops against a provider median of 0.55 s, i.e. the batch waiting on one
        straggler, in a cycle that took 54 s end to end.

        WHY IT IS SAFE TO NOT WAIT, and this is the part that needed no new machinery: an
        object with no description yet is ALREADY a supported state everywhere downstream.
        `_build_descriptions` fills an absent entry with {}; the association layer's
        description term is only computed when BOTH embeddings exist and drops out with the
        weights renormalised otherwise; the admission gate treats missing evidence as
        ABSTAIN, never as a low score; and `Hypothesis` persists across sweeps, so a pair
        that cannot be decided now is re-decided when the description arrives. The object is
        held out of a merge decision by the evidence rules that already exist -- it is not
        hidden from the loop, it simply carries nothing to compare on yet.

        THE COST, stated: a description lands one or more cycles after its detection, so an
        object may be admitted before it is describable. The gate's answer to thin evidence
        is to abstain, so the failure direction is MORE ABSTENTIONS, not wrong admissions.

        The executor is per-node, not per-call, so work outlives the cycle that queued it.
        """
        valid_crops = [crop for crop in crops_data if crop is not None]

        # Harvest whatever finished since the last cycle, whenever it was submitted.
        results = {}
        still_pending = {}
        for label, fut in self._vlm_pending.items():
            if fut.done():
                try:
                    r = fut.result() or {}
                    # A grid future is shared by every label in its batch and returns all of
                    # them at once; a single-crop future returns just this label's result.
                    # The marker keeps the two apart explicitly -- sniffing the dict's keys
                    # would misread a description that happened to contain the label.
                    if isinstance(r, dict) and "__grid__" in r:
                        results[label] = r["__grid__"].get(label, {}) or {}
                    else:
                        results[label] = r
                except Exception as exc:
                    self.get_logger().error(f"Crop VLM future failed for {label}: {exc}")
                    results[label] = {}
                # W2. Attach the origin recorded at submit time -- frame and 2D box of the
                # crop this result was computed from -- so the consumer can tell a deferred
                # answer for THIS object from one for whatever answered to the same string
                # when it left. Popped, not read: a harvested origin is spent.
                results[label]["origin"] = self._vlm_origin.pop(label)
            else:
                still_pending[label] = fut
        self._vlm_pending = still_pending

        if not valid_crops:
            if results:
                self.log_both("info", f"[VLM] {len(results)} deferred description(s) landed; "
                                      f"{len(self._vlm_pending)} still in flight")
            return results

        # BOUNDED. Without a cap a describer slower than the cycle rate queues without limit
        # and the backlog only grows -- the object would never get its description AND the
        # memory would climb. Dropping the oldest is visible in the log; an unbounded queue
        # is not.
        max_pending = int(CFG.get("vlm", {}).get("max_pending_crops", 64))
        fresh = [c for c in valid_crops if c["label"] not in self._vlm_pending]
        room = max(0, max_pending - len(self._vlm_pending))
        if len(fresh) > room:
            self.log_both("warn", f"[VLM] backlog at {max_pending}; queueing {room} of "
                                  f"{len(fresh)} crop(s) (describer slower than detection)")
            fresh = fresh[:room]

        grid_cells = int(CFG.get("vlm", {}).get("grid_cells", 0))
        if grid_cells > 1 and len(fresh) > 1:
            # GA-209. One request per batch instead of one per crop. Every label in a batch
            # holds the SAME future, and the harvest pulls its own cell out of the shared
            # result -- so the existing per-label bookkeeping is untouched.
            import crop_grid
            for idx in crop_grid.plan(len(fresh), grid_cells):
                batch = [fresh[i] for i in idx]
                fut = self._vlm_executor.submit(self.process_crop_grid, batch)
                for c in batch:
                    self._vlm_pending[c["label"]] = fut
                    self._vlm_remember_origin(c)
            self.log_both("info", f"[VLM] grid: {len(fresh)} crop(s) in "
                                  f"{len(crop_grid.plan(len(fresh), grid_cells))} request(s)")
        else:
            for crop in fresh:
                self._vlm_pending[crop["label"]] = self._vlm_executor.submit(
                    self.process_crop_vlm, crop)
                self._vlm_remember_origin(crop)

        self.log_both("info", f"[VLM] queued {len(valid_crops)} crop(s), "
                              f"{len(results)} landed, {len(self._vlm_pending)} in flight")
        return results

    def _unified_scene_description_results(self, detections):
        """Adapt same-call scene attributes to the existing description seam."""
        results = {}
        model = CFG.get("vlm", {}).get("model", "unknown")
        unknown = {"", "unknown", "none", "n/a"}
        for det in detections:
            values = {
                field: str(getattr(det, field, "unknown") or "unknown")
                for field in DESCRIPTION_FIELDS
            }
            has_answer = any(value.strip().lower() not in unknown for value in values.values())
            values["provenance"] = {
                "model": model,
                "status": "ok" if has_answer else "model_abstained",
                "source": "unified_scene_call",
            }
            results[det.instance_label] = values
        return results

    def _vlm_remember_origin(self, crop):
        """W2. Record where a pending description came FROM: frame + 2D box.

        Direct indexing on purpose: a crop dict without `frame`/`bbox` is a broken
        producer, and defaulting it here would rebuild the exact silent
        misattribution this bookkeeping exists to prevent.
        """
        self._vlm_origin[crop["label"]] = {"frame": crop["frame"], "bbox": crop["bbox"]}

    def _build_descriptions(self, detections, vlm_results, crops_data=None):
        """GA-277. `crop_path` rides along so the admission gate can SEE the object.

        The gate is handed geometry -- label, bbox, room -- and no pixels, so a VLM check on a
        held decision had nothing to look at. The crop is already written to disk for the
        describer; this carries WHERE, not the image, so the seam stays a small message.
        Keyed by instance_label because that is what both sides already agree on -- and
        REFUSED when the label string no longer names the same object (W2): a harvested
        result carries the frame and 2D box of its crop, and is dropped unless that box
        overlaps the detection it would decorate. Same object, robot moved -> kept;
        same string, different object -> refused and logged, and the object is described
        by its own fresh submission instead.
        """
        by_label = {}
        for c in (crops_data or []):
            if c and c.get("label") and c.get("path"):
                by_label[c["label"]] = c["path"]
        min_iou = float(CFG.get("vlm", {}).get("stale_result_min_iou", 0.1))
        descriptions = []
        # GA-108. A deferred answer that does not belong to this cycle's namesake is not
        # dropped any more: it is delivered to the object it was taken FROM, by origin frame
        # and box, on /object_descriptions_late (see _publish_late_descriptions). Measured on
        # run 20260906_223701: 496 answers landed, 163 were refused here, and 204 of 212
        # objects ended "unknown" -- the answer for an object never reached that object.
        self._late_descriptions = []
        seen = set()
        for det in detections:
            seen.add(det.instance_label)
            res = (vlm_results.get(det.instance_label, {}) or {})
            origin = res.get("origin")
            if origin is not None and _bbox_iou(det.bbox, origin["bbox"]) < min_iou:
                self.log_both(
                    "info",
                    f"[VLM] deferred description for {det.instance_label} was taken in frame "
                    f"{origin['frame']} for a box that does not overlap this detection "
                    f"(< {min_iou} IoU); delivered to its origin object instead (GA-108)")
                self._late_descriptions.append((det.instance_label, origin, res))
                res = {}
            d = {field: res.get(field, "unknown") for field in DESCRIPTION_FIELDS}
            # W6. The route this description took, so a run can split the "unknown"
            # population into call_failed / parse_failed / model_abstained / unanswered
            # instead of reporting one conflated rate.
            d["status"] = _description_status(res)
            d["confirmed"] = getattr(det, "is_confirmed", True)
            # Empty string, not absent: the msg field always exists, and "" reads as "no crop
            # was written for this detection" rather than as a missing key nobody set.
            d["crop_path"] = by_label.get(det.instance_label, "")
            if "provenance" in res:
                d["provenance"] = res["provenance"]
            descriptions.append(d)
        # Answers whose label is not in this cycle at all (the object left the view): the
        # old code never looked at them. Same delivery.
        for label, res in vlm_results.items():
            if label in seen or not isinstance(res, dict) or res.get("origin") is None:
                continue
            self._late_descriptions.append((label, res["origin"], res))
        return descriptions

    def _publish_late_descriptions(self, cycle_stamp):
        """GA-108. Deferred describer answers, addressed by the crop's origin frame and box."""
        late = getattr(self, "_late_descriptions", None) or []
        if not late:
            return
        pub = getattr(self, "pub_object_descriptions_late", None)
        if pub is None:
            self.pub_object_descriptions_late = pub = self.create_publisher(
                ObjectDescriptionArray, "/object_descriptions_late", 10)
        arr = self.make_header_msg(ObjectDescriptionArray, stamp=cycle_stamp, frame_id="map")
        for label, origin, res in late:
            m = ObjectDescription()
            m.label = label
            for field in DESCRIPTION_FIELDS:
                setattr(m, field, str(res.get(field, "unknown")))
            m.status = _description_status(res)
            m.origin_frame = str(origin.get("frame") or "")
            m.origin_bbox_2d = [float(v) for v in origin.get("bbox") or (0.0, 0.0, 0.0, 0.0)]
            arr.descriptions.append(m)
        pub.publish(arr)
        self.log_both("info", f"[VLM] {len(late)} late description(s) delivered by origin (GA-108)")
        self._late_descriptions = []

    def _gt_semantic_callback(self, msg):
        """Cache habitat's per-pixel instance frame, keyed by its EXACT stamp.

        A small bounded cache rather than a single slot: the semantic frame and the cycle
        that consumes it are not guaranteed to arrive in lockstep, and matching on "the
        latest one" would silently pair a detection with a different frame's ground truth --
        which is the failure this whole night has been about, in the one place where it
        would corrupt the measurement rather than the map.
        """
        key = frame_id_from_stamp(msg.header.stamp)
        if key is None:
            return
        # GA-353. The cache holds the COMPRESSED blob (gt_codec's run-length form, ~2.7% of the
        # raw 32SC1 frame, ~130 KB at 1280x960) and decodes at lookup, so it can be deep. It
        # held 8 DECODED frames, and the archive's exact-stamp lookup runs after the detection
        # span (median 3.1 s): at run H's stalled 0.16 f/s the frame was still there (485/506
        # joined); once GA-335 let the feed run at 3 f/s it was evicted 9 frames later, and run
        # 152446 joined 33 of 770 ("no semantic frame" x735). Rule 15: the feed fix armed this.
        self._gt_semantic[key] = bytes(msg.data)
        while len(self._gt_semantic) > GT_SEMANTIC_CACHE_FRAMES:
            self._gt_semantic.pop(next(iter(self._gt_semantic)))

    def _gt_semantic_for(self, frame_id):
        """-> the decoded semantic frame for EXACTLY this frame_id, or None. Never a
        neighbouring frame: a GT label from another frame is worse than none."""
        blob = getattr(self, "_gt_semantic", {}).get(frame_id)
        if blob is None:
            return None
        import gt_codec
        arr = gt_codec.decode(blob)
        if arr is None:
            self.log_both("warn", f"[GT] semantic frame {frame_id} could not be decoded; ignored")
        return arr

    def _archive_detections(self, detections, bboxes_3d, centroids_3d, image_raw,
                            transform, cycle_stamp, crops_data=None, depth=None):
        """Write the frame and one row per detection. No-op unless archiving is enabled.

        Placed AFTER the 3D geometry so the row carries the 3D box and centroid too, and
        BEFORE the VLM batch so a failure in the describer cannot cost us the record of what
        was detected. Everything written here already exists in memory; this adds a write,
        not a computation.
        """
        arch = getattr(self, "detection_archive", None)
        if arch is None or not arch.enabled:
            return
        frame_id = frame_id_from_stamp(cycle_stamp)
        if frame_id is None:
            return
        arch.record_frame(frame_id, image_raw)
        # GA-279. The depth THIS detection was measured from, archived beside the RGB. Every
        # measurement-provenance question so far has died on its absence, and the re-render
        # workaround does not validate.
        arch.record_depth(frame_id, depth)
        # EXACT stamp match only. If the semantic frame for THIS frame_id is not held, the
        # rows carry no GT and say why -- never the nearest available frame.
        semantic = self._gt_semantic_for(frame_id)
        # GA-353. Read by the per-cycle latency row so a bundle measures its own GT join per
        # cycle instead of discovering it from the archive afterwards.
        self._last_gt_semantic_hit = semantic is not None
        cam = None
        try:
            t = transform.transform.translation
            cam = [float(t.x), float(t.y), float(t.z)]
        except Exception:
            cam = None
        # GA-230. The ORIENTATION beside the position. Without it a bundle records where the
        # camera stood and not where it pointed, and a 3D box cannot be drawn back onto the
        # frame it came from except by solving for the rotation from the frame's own
        # detections -- which works, at 0.75 deg median, but only on frames carrying four or
        # more correspondences. Four floats remove that dependency entirely.
        _cam_quat = None
        try:
            _r = transform.transform.rotation
            _cam_quat = [float(_r.x), float(_r.y), float(_r.z), float(_r.w)]
        except Exception:
            _cam_quat = None
        for i, det in enumerate(detections):
            arch.record_detection(
                frame_id, det, camera_position=cam, camera_transform=_cam_quat,
                centroid=(centroids_3d[i] if centroids_3d is not None
                          and i < len(centroids_3d) else None),
                bbox_3d=(bboxes_3d[i] if bboxes_3d is not None
                         and i < len(bboxes_3d) else None),
                crop_meta=((crops_data[i] or {}).get("crop_meta")
                           if crops_data is not None and i < len(crops_data) else None),
                stamp=frame_id,
                room_id=getattr(self, "current_room_id", None),
                semantic_frame=semantic)

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
                if key in ("yaw", "oriented_center", "oriented_extents"):
                    continue          # handled below, with the has_orientation flag
                if hasattr(box_msg, key):
                    setattr(box_msg, key, value)

            # GA-100: the oriented box, and a flag saying whether it is real. This guard is
            # what the generic hasattr loop above could not express -- it would have written
            # yaw=0.0 for an object with no orientation and left the reader unable to tell
            # that from a genuine yaw of zero.
            oc, oe = bbox_3d.get("oriented_center"), bbox_3d.get("oriented_extents")
            if oc is not None and oe is not None and len(oc) == 3 and len(oe) == 3:
                box_msg.has_orientation = True
                box_msg.yaw = float(bbox_3d.get("yaw", 0.0))
                box_msg.oriented_center = [float(v) for v in oc]
                box_msg.oriented_extents = [float(v) for v in oe]
            else:
                box_msg.has_orientation = False

            # GA-186: the detector's own 2D box, so the object manager can tell a duplicate
            # detection of one object from two objects sharing a frame. `det.bbox` is the
            # (x1, y1, x2, y2) the archive already writes as `bbox_2d`; carried here because
            # co-visibility is a hard negative and must never be evaluated on a guess.
            det_box_2d = getattr(det, "bbox", None)
            if det_box_2d is not None and len(det_box_2d) == 4:
                box_msg.has_bbox_2d = True
                box_msg.bbox_2d = [float(v) for v in det_box_2d]
            else:
                box_msg.has_bbox_2d = False

            # GA-190: the appearance embedding, already computed and until now only dumped
            # to a sidecar. `clip_embedding` is None whenever the backend could not embed
            # that crop -- a degenerate mask, for instance -- and None must stay absent
            # rather than become a zero vector.
            det_embed = getattr(det, "clip_embedding", None)
            if det_embed:
                box_msg.has_clip_embedding = True
                box_msg.clip_embedding = [float(v) for v in det_embed]
            else:
                box_msg.has_clip_embedding = False
            msg.boxes.append(box_msg)
        self.bbox_pub.publish(msg)

    def _publish_description_array(self, detections, descriptions, cycle_stamp):
        desc_array = self.make_header_msg(ObjectDescriptionArray, stamp=cycle_stamp, frame_id="map")
        for det, desc in zip(detections, descriptions):
            obj_msg = ObjectDescription()
            obj_msg.label = det.instance_label
            # Only copy keys the msg actually has -- the same guard _publish_bbox_array
            # already carries, and for the same reason. `_build_descriptions` adds
            # `confirmed` (always) and `provenance` (sometimes), and ObjectDescription.msg
            # declares only label/description/color/material/shape. The blind setattr here
            # raised `'ObjectDescription' object has no attribute 'confirmed'` on EVERY
            # cycle, which the handler in _run_perception_cycle absorbed -- so neither
            # topic was ever published and the object manager waited for messages nobody
            # sent.
            for key, value in desc.items():
                if hasattr(obj_msg, key):
                    setattr(obj_msg, key, value)
                else:
                    self._note_undeliverable_field(key)
            desc_array.descriptions.append(obj_msg)
        self.pub_object_descriptions.publish(desc_array)

    def _note_undeliverable_field(self, key):
        """Report, once per key, a description field no consumer can carry.

        Dropping it silently is what the guard next door does, and it is the reason this
        was cheap to write and expensive to find. A field that is produced every cycle and
        delivered nowhere is either a msg that needs the field or a producer that should
        stop computing it -- and neither question gets asked if nothing says it happened.
        """
        if key in self._undeliverable_fields:
            return
        self._undeliverable_fields.add(key)
        self.log_both("warn",
                      f"description field '{key}' has no slot in ObjectDescription.msg and no "
                      f"kwarg on Object -- produced every cycle, delivered nowhere")

    def _update_world_model(self, detections, centroids_3d, bboxes_3d, descriptions):
        wm.actual_perceptions.clear()
        for det, centroid, bbox, desc in zip(detections, centroids_3d, bboxes_3d, descriptions):
            # Same mismatch as the publisher above, and latent only because the publisher
            # crashed first: Object.__init__ takes description/color/material/shape and
            # would raise TypeError on `confirmed`.
            obj = Object(det.label, centroid, bbox,
                         **{k: v for k, v in desc.items() if k in DESCRIPTION_FIELDS})
            # W6: the status rides the world-model object so the per-cycle snapshot can
            # carry the four-way split. Additive attribute: no reader tests for it.
            obj.status = desc.get("status", "")
            # distinct from obj.embedding (the 300-d word2vec description vector)
            obj.clip_embedding = getattr(det, "clip_embedding", None)
            wm.add_actual_perception(obj)

    def _publish_agent_pose(self, cycle_stamp):
        try:
            lookup_time = rclpy.time.Time.from_msg(cycle_stamp)
            t = self.tf_buffer.lookup_transform(
                "map",
                self.agent_pose_frame,
                lookup_time,
            )
        except TransformException as ex:
            self.log_both(
                "warn",
                f"Could not get camera pose (map -> {self.agent_pose_frame}): {ex}",
            )
            return

        pose_msg = PoseStamped()
        pose_msg.header.stamp = cycle_stamp
        pose_msg.header.frame_id = "map"
        pose_msg.pose.position.x = t.transform.translation.x
        pose_msg.pose.position.y = t.transform.translation.y
        pose_msg.pose.position.z = t.transform.translation.z
        pose_msg.pose.orientation = t.transform.rotation

        self.agent_pose_pub.publish(pose_msg)
        self.log_both(
            "debug",
            f"Camera pose ({self.agent_pose_frame}) published on /agent_camera_pose",
        )

    def _queue_perceptions_json(self):
        perceptions_snapshot = [
            {
                "label": obj.label,
                "centroid": obj.centroid.tolist() if hasattr(obj.centroid, "tolist") else list(obj.centroid or []),
                "bbox": obj.bbox,
                "status": getattr(obj, "status", ""),
                **{field: getattr(obj, field) for field in DESCRIPTION_FIELDS},
            }
            for obj in wm.actual_perceptions
        ]
        self._io_executor.submit(self.write_perceptions_json, perceptions_snapshot)

    def joint_callback(self):
        tracked_joints = self.head_joints + self.base_joints
        motion_scores = []
        _missing = []

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
                # ONCE, at debug. This ran at 1 Hz per missing frame and produced hundreds of
                # INFO lines carrying the text of an error that was expected.
                if not self._motion_absent_logged:
                    self.get_logger().debug(
                        f"motion watch: {joint_name} has no transform to {from_frame_rel} "
                        f"({ex}); it is excluded from the motion average for this run")
                _missing.append(joint_name)

        # The mean must say what it is a mean OF. A five-source design reporting one source
        # is not wrong here -- habitat_camera is the thing that moves -- but silence about
        # the other four is how a mean-of-one passes for an aggregate.
        if _missing and not self._motion_absent_logged:
            self._motion_absent_logged = True
            self.log_both("warn", f"motion watch: {len(tracked_joints) - len(_missing)} of "
                                  f"{len(tracked_joints)} frames resolve; absent: "
                                  f"{', '.join(_missing)}. The motion average uses only the "
                                  f"frames that resolve.")
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
        # H-item from the review: the DESCRIBER executor was never shut down, so a slow
        # crop call (60 s stragglers measured against a 0.55 s median) delayed node exit
        # by its full remaining runtime. Queued descriptions are CANCELLED — nobody
        # will read them in a dying node — and running calls are not waited on: they are
        # stateless HTTP requests whose results land in a future nobody harvests.
        try:
            self._vlm_executor.shutdown(wait=False, cancel_futures=True)
        except TypeError:
            # py<3.9 has no cancel_futures; wait=False alone still unblocks the exit.
            self._vlm_executor.shutdown(wait=False)
        except Exception:
            pass
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    try:
        node = DetectObjectsNode()
    except Exception:
        # Wake the host-side startup gate immediately if node construction fails anywhere
        # before the final ready marker, rather than making it wait for its full timeout.
        write_vitsam_status("failed")
        rclpy.shutdown()
        raise
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
