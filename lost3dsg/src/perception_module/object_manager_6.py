#!/usr/bin/env python3
"""
Object Manager Service - Semantic and Spatial Tracking of Perceived Objects
Tracks objects and automatically transitions from EXPLORATION to TRACKING when
an object is seen again in a different position.
Includes Topological Semantic Mapping (Room Manager) with Scene Graph generation.

Room changes are handled by the Room Manager using detected wall segments and
its normal scene-evaluation logic.
"""
import hashlib
import json
import logging
import os
import subprocess
import sys
import threading
import time
import urllib.parse
from collections import deque
from datetime import datetime, timezone

import numpy as np
import rclpy
import requests
from association import AssocObject, Observation, search_radius
from builtin_interfaces.msg import Time as TimeMsg
from config import CFG
from cv_utils import publish_persistent_bboxes
from geometry_msgs.msg import PoseStamped
from hooks import DecisionLog, load_hooks
from nav_msgs.msg import Path
from nlp_utils import get_embedding, lost_similarity, lost_similarity_detailed, world2vec
from object_services import (
    ObjectServices,
    ensure_relations,
    infer_spatial_relations,
    save_persistent_perceptions,
    save_uncertain_objects,
    synchronized_world_model,
)
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy
from room_manager import RoomManager
from std_msgs.msg import Bool, String
from tf2_ros import Buffer, TransformListener

# Explicit, not `import *`. Only the names this file does NOT define itself:
# publish_persistent_centroids, publish_pov_volume and publish_uncertain_* are defined
# BELOW and also in cv_utils with different bodies, so importing them here would swap a
# local implementation for a five-line wrapper (GA-77). While the star imports stood,
# ruff's F family was blind on this file -- which is why GA-22's two crashes read as
# `F405 may be undefined` instead of `F821 undefined name`.
from utils import compute_iou_3d
from visualization_msgs.msg import Marker, MarkerArray
from world_model import wm

from lost3dsg.msg import Bbox3dArray, ObjectDescriptionArray
from lost3dsg.srv import (
    ObjectTrackingService,
    UpdateObject,
)

# ============= EXPLORATION PARAMETERS (config.yaml: association) =============
EXPLORATION_IOU_THRESHOLD = CFG["association"]["exploration_iou_threshold"]
SIM_THRESHOLD = CFG["association"]["sim_threshold"]
TRACKING_IOU_THRESHOLD = CFG["association"]["tracking_iou_threshold"]
VOLUME_EXPANSION_RATIO = CFG["association"]["volume_expansion_ratio"]
EXPLORATION_FRAME_LIMIT = CFG["association"]["exploration_frame_limit"]
OBJECT_STABILITY_TIMEOUT = CFG["association"]["object_stability_timeout"]
POV_SCALE_FACTOR = CFG["association"]["pov_scale_factor"]
MAX_VOLUME_THRESHOLD = CFG["association"]["max_volume_threshold"]
BBOX_REDUCTION_RATIO = CFG["association"]["bbox_reduction_ratio"]
# GA-12: the exploration->tracking move distance was the literal 0.35 at its one use site,
# the only destructive threshold in this block with no config key.
TRANSITION_MOVE_DISTANCE_M = CFG["association"].get("transition_move_distance_m", 0.35)
# GA-48: the re-evaluation radius was ALWAYS this literal. `MAX_MATCH_DISTANCE or 2.0`
# evaluated to 2.0 for every shipped run, because max_match_distance_m has always been
# 0.0 -- so the key that appeared to govern the neighbour radius never did. It is its own
# value and now has its own key, default 2.0 to reproduce that behaviour exactly.
REEVALUATION_RADIUS = CFG["association"].get("reevaluation_radius_m", 2.0)
REEVALUATION_DEBOUNCE_S = float(CFG["association"].get("reevaluation_debounce_s", 2.0))
REEVALUATION_MAX_FANOUT = int(CFG["association"].get("reevaluation_max_fanout", 12))
# GA-289. The tracking path (check_tracking_transition) ran `lost_similarity` over EVERY stable
# object with no geometry consulted at all, and a win needed no evidence: two identical labels
# score exactly 1.000 on nothing else measurable, and `>` keeps the FIRST such object at a tie.
# Measured 2026-09-03 over the six runs of the day (GA-190's rows): 368 comparisons, 95% of
# them beyond 1.0 m, and all four winners at 1.000 -- a `faucet` seen in the second bathroom
# attached to `faucet#1` in the first, 9.8 m away. Era (rule 34): geometry-blind since the
# first import (3353c96, 2026-07-04; FOUND from 82745b2); the label-only 1.000 became
# reachable when GA-101 dropped absent terms from the divisor (2026-09-01) -- before that the
# `continue` on a missing embedding made this loop unreachable in every run (GA-90).
#
# Same criterion as the merge path (D1: locality before similarity): the object's own
# covariance shell via association.search_radius. Objects with no covariance yet get their
# extent plus this configured radius, by the orchestrator's ruling of 2026-09-03 (session
# 9fee2a6d). The same `merge_min_evidence` that guards the sweep now guards the win here.
# ponytail: frustum culling (behind-the-camera pruning) is deliberately NOT built -- unmeasured,
# and it would give om6 a camera-pose dependency; it stays the third arm of the D1 switch.
TRACKING_FALLBACK_RADIUS_M = float(CFG["association"].get("tracking_fallback_radius_m", 1.0))
TRACKING_MIN_EVIDENCE = int(CFG["association"].get("merge_min_evidence", 1))


def _bbox_centre(b):
    if not b:
        return None
    try:
        return ((b["x_min"] + b["x_max"]) / 2.0, (b["y_min"] + b["y_max"]) / 2.0,
                (b["z_min"] + b["z_max"]) / 2.0)
    except (KeyError, TypeError):
        return None


def _centre_distance(centre, other_bbox):
    """GA-190: the cheap discriminator, measured. A few arithmetic operations against a
    composite similarity over four terms -- measuring costs a fraction of what it measures."""
    o = _bbox_centre(other_bbox)
    if centre is None or o is None:
        return None
    return ((centre[0] - o[0]) ** 2 + (centre[1] - o[1]) ** 2 + (centre[2] - o[2]) ** 2) ** 0.5


def tracking_reach_m(obj, fallback_m=None):
    """GA-289: how far a persistent object may sit from a detection and still be compared.

    Reuses association.search_radius -- the 99% chi-squared shell of the object's OWN
    position covariance plus its bounding-box half-diagonal -- so the tracking path and the
    merge path agree on what "near enough to compare" means. With no sightings recorded the
    shell is unmeasurable; the ruled fallback is the extent plus a CONFIG radius rather than a
    distance invented here. Returns (metres, basis) so the row can say which one applied.
    """
    if fallback_m is None:
        fallback_m = TRACKING_FALLBACK_RADIUS_M
    ao = AssocObject(object_id=getattr(obj, "object_id", None) or getattr(obj, "label", None),
                     bbox=getattr(obj, "bbox", None), centroid=getattr(obj, "centroid", None),
                     observations=getattr(obj, "observations", None) or [])
    reach, basis = search_radius(ao, None)
    if ao.covariance is None:
        return reach + float(fallback_m), "extent + fallback radius (no covariance)"
    return reach, basis


def _half_diagonal_m(bbox):
    if not bbox:
        return 0.0
    try:
        return 0.5 * ((bbox["x_max"] - bbox["x_min"]) ** 2 + (bbox["y_max"] - bbox["y_min"]) ** 2
                      + (bbox["z_max"] - bbox["z_min"]) ** 2) ** 0.5
    except (KeyError, TypeError):
        return 0.0


def locality_ok(bbox, obj, threshold):
    """GA-04: 3D overlap gate, applied BEFORE any attribute comparison.

    Contract G1 invariant 3: locality is evaluated before attribute similarity, and an
    object failing the gate is never compared on attributes. Absent geometry is not
    locality evidence, so a candidate without a box does not pass -- the exploration
    branch already excluded those, and D14 prefers strict.
    """
    if bbox is None or getattr(obj, "bbox", None) is None:
        return False
    return compute_iou_3d(bbox, obj.bbox) >= threshold


def bbox_center_distance(a, b):
    return float(np.linalg.norm([
        (a["x_min"] + a["x_max"] - b["x_min"] - b["x_max"]) / 2.0,
        (a["y_min"] + a["y_max"] - b["y_min"] - b["y_max"]) / 2.0,
        (a["z_min"] + a["z_max"] - b["z_min"] - b["z_max"]) / 2.0,
    ]))

file_path = os.path.abspath(__file__)
current_dir = os.path.dirname(file_path)
PROJECT_ROOT = current_dir.split('/install/')[0] if '/install/' in current_dir else os.path.abspath(os.path.join(current_dir, "../.."))

# world2vec is imported explicitly above -- loaded once in nlp_utils.

# Setup path per il file sintetico di operazioni
log_dir = os.path.join(PROJECT_ROOT, "output")
os.makedirs(log_dir, exist_ok=True)
SYNTHETIC_LOG_FILE = os.path.join(log_dir, "operations.txt")
AGENT_POSES_LOG_FILE = os.path.join(log_dir, "agent_poses.json")
AGENT_POSES_SERIES_FILE = os.path.join(log_dir, "agent_poses.jsonl")
# How often the two O(n) views of the pose series may be rebuilt. Both are derived
# from agent_poses.jsonl, so a lag here loses nothing.
AGENT_POSES_SNAPSHOT_PERIOD = 5.0   # agent_poses.json whole-array rewrite
AGENT_PATH_PUBLISH_PERIOD = 1.0     # /agent_path nav_msgs/Path for RViz
# GA-186: how many sightings one object keeps. Bounded because the co-visibility channel
# walks the two lists pairwise, so an unbounded list makes one comparison quadratic in the
# number of frames an object was visible for -- and run 20260901_055513 had objects present
# across 202 cycles. The covariance and the view spread both converge long before 64.
MAX_OBSERVATIONS_PER_OBJECT = int(CFG["association"].get("max_observations_per_object", 64))
# GA-83: how long /bbox_3d may be silent before the node says so. Long enough not to fire
# between ordinary detection cycles, short enough that a dead producer is in the log within
# a minute rather than in a process table an hour later.
INPUT_SILENCE_TIMEOUT = CFG["association"].get("input_silence_timeout_s", 60.0)
# GA-94: how many CONSECUTIVE silent checks before this node ends the run. GA-83 stated
# the fact and left the conclusion to a reader; run A proved there is no reader -- om6
# announced "the producer may have stopped" at 17:53:35 and then idled 36 more minutes
# while the run was already dead.
INPUT_SILENCE_MAX_STRIKES = CFG["association"].get("input_silence_max_strikes", 3)
# GA-94b: how many robot STOPS must pass with no detection before the producer is called
# dead. Detection only happens when the robot stops, so stops -- not seconds -- are the unit
# in which "the producer had its chance" is measurable.
INPUT_SILENCE_MIN_STOPS = CFG["association"].get("input_silence_min_stops", 3)
# must match the bridge's own default (BRIDGE_PORT=8081); :8080 is the FOUND dashboard server
# GA-267b. BRIDGE_PORT, honoured -- the SAME hardcoded 8081 that silenced the feed host's
# belief poller, in a second place and with a far worse consequence.
#
# The bridge moves to 8091 whenever 8081 is taken, which it is on this machine. With this
# pinned to 8081 every Graph API POST failed, so `add_new_object` never received an
# object_id, no `link` record was written, and NOTHING entered the world model. Run
# 20260902_172331 logged 4 admit decisions, 0 links and no persistent_perception.json, then
# ended itself: "no /bbox_3d for 202s ... 0 objects in the map". The decisions were real and
# the map was empty, and the grade counter alone could not tell the difference.
# GA-270. Config first, env override second. The literal that used to live here cost this
# run its entire world model; see config.py "services" for why addresses are configuration.
_SVC = (CFG.get("services", {}) or {})
GRAPH_API_BASE_URL = os.environ.get("GRAPH_API_BASE_URL") or (
    f"http://{_SVC.get('bridge_host', '127.0.0.1')}:"
    f"{os.environ.get('BRIDGE_PORT') or _SVC.get('bridge_port', 8081)}")
GRAPH_API_TIMEOUT = float(os.environ.get("GRAPH_API_TIMEOUT", "10.0"))
GRAPH_API_AUTOSTART = os.environ.get("GRAPH_API_AUTOSTART", "1").lower() not in {"0", "false", "no"}
SYNC_BUFFER_LIMIT = 20

def _launch_graph_api_bridge():
    bridge_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "graph_api_bridge.py")
    if not os.path.exists(bridge_path):
        print(f"[WARN] graph_api_bridge.py not found: {bridge_path}")
        return None

    proc = subprocess.Popen(
        [sys.executable, bridge_path],
        env={
            **os.environ,
            # The bridge and RoomManager must read the same live output
            # directory even when one process is launched from src and the
            # other from install.
            "LOST3DSG_OUTPUT_DIR": os.path.join(PROJECT_ROOT, "output"),
        },
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    def _pipe_logs():
        try:
            if proc.stdout is None:
                return
            for line in proc.stdout:
                print(f"[BRIDGE] {line}", end="")
        except Exception as e:
            print(f"[WARN] Error reading graph_api_bridge log: {e}")

    threading.Thread(target=_pipe_logs, daemon=True).start()
    return proc


def _graph_api_is_running(base_url):
    try:
        response = requests.get(f"{base_url.rstrip('/')}/rooms", timeout=0.5)
        return response.ok
    except requests.RequestException:
        return False


# ============= HELPER FUNCTIONS =============

def create_object_key(label, material, color, description):
    """Create a unique key for an object based on attributes."""
    key_dict = {
        "label": label if label else "",
        "material": material if material else "",
        "color": color if color else "",
        "description": description if description else ""
    }
    return json.dumps(key_dict, sort_keys=True)

def create_object_id(label, material, color, description):
    key = create_object_key(label, material, color, description)
    return hashlib.sha256(key.encode("utf-8")).hexdigest()

def shrink_bbox(bbox, ratio):
    x_center = (bbox["x_min"] + bbox["x_max"]) / 2.0
    y_center = (bbox["y_min"] + bbox["y_max"]) / 2.0
    z_center = (bbox["z_min"] + bbox["z_max"]) / 2.0

    x_size = (bbox["x_max"] - bbox["x_min"]) * (1.0 - ratio)
    y_size = (bbox["y_max"] - bbox["y_min"]) * (1.0 - ratio)
    z_size = (bbox["z_max"] - bbox["z_min"]) * (1.0 - ratio)

    return {
        "x_min": x_center - x_size / 2.0,
        "x_max": x_center + x_size / 2.0,
        "y_min": y_center - y_size / 2.0,
        "y_max": y_center + y_size / 2.0,
        "z_min": z_center - z_size / 2.0,
        "z_max": z_center + z_size / 2.0
    }

def bbox_volume(bbox):
    return ((bbox["x_max"] - bbox["x_min"]) *
            (bbox["y_max"] - bbox["y_min"]) *
            (bbox["z_max"] - bbox["z_min"]))

def compute_pov_volume(bboxes_list, expansion_ratio=VOLUME_EXPANSION_RATIO):
    """Compute the POV volume that contains all detections."""
    if not bboxes_list:
        return None

    processed_bboxes = []
    for bbox in bboxes_list:
        vol = bbox_volume(bbox)
        if vol > MAX_VOLUME_THRESHOLD:
            processed_bbox = shrink_bbox(bbox, BBOX_REDUCTION_RATIO)
        else:
            processed_bbox = bbox
        processed_bboxes.append(processed_bbox)

    x_min = min(bbox["x_min"] for bbox in processed_bboxes)
    x_max = max(bbox["x_max"] for bbox in processed_bboxes)
    y_min = min(bbox["y_min"] for bbox in processed_bboxes)
    y_max = max(bbox["y_max"] for bbox in processed_bboxes)
    z_min = min(bbox["z_min"] for bbox in processed_bboxes)
    z_max = max(bbox["z_max"] for bbox in processed_bboxes)

    x_size = x_max - x_min
    y_size = y_max - y_min
    z_size = z_max - z_min

    x_expansion = x_size * expansion_ratio
    y_expansion = y_size * expansion_ratio
    z_expansion = z_size * expansion_ratio

    MIN_EXPANSION = 0.1
    x_expansion = max(x_expansion, MIN_EXPANSION)
    y_expansion = max(y_expansion, MIN_EXPANSION)
    z_expansion = max(z_expansion, MIN_EXPANSION)

    pov_z_min = z_min - z_expansion
    pov_z_max = z_max

    return {
        "x_min": x_min - x_expansion,
        "x_max": x_max + x_expansion,
        "y_min": y_min - y_expansion,
        "y_max": y_max + y_expansion,
        "z_min": pov_z_min,
        "z_max": pov_z_max
    }

def shrink_pov_volume(pov_volume, scale_factor=POV_SCALE_FACTOR):
    """Shrink POV volume uniformly in all directions around its center."""
    if not pov_volume or scale_factor >= 1.0:
        return pov_volume
    
    center_x = (pov_volume['x_min'] + pov_volume['x_max']) / 2.0
    center_y = (pov_volume['y_min'] + pov_volume['y_max']) / 2.0
    center_z = (pov_volume['z_min'] + pov_volume['z_max']) / 2.0
    
    half_x = ((pov_volume['x_max'] - pov_volume['x_min']) / 2.0) * scale_factor
    half_y = ((pov_volume['y_max'] - pov_volume['y_min']) / 2.0) * scale_factor
    half_z = ((pov_volume['z_max'] - pov_volume['z_min']) / 2.0) * scale_factor
    
    return {
        "x_min": center_x - half_x,
        "x_max": center_x + half_x,
        "y_min": center_y - half_y,
        "y_max": center_y + half_y,
        "z_min": center_z - half_z,
        "z_max": center_z + half_z
    }

def expand_bbox_for_search(bbox, expansion_ratio=VOLUME_EXPANSION_RATIO):
    """Expand a bounding box proportionally to its size."""
    x_size = bbox["x_max"] - bbox["x_min"]
    y_size = bbox["y_max"] - bbox["y_min"]
    z_size = bbox["z_max"] - bbox["z_min"]
    
    x_expansion = x_size * expansion_ratio
    y_expansion = y_size * expansion_ratio
    z_expansion = z_size * expansion_ratio
    
    return {
        "x_min": bbox["x_min"] - x_expansion,
        "x_max": bbox["x_max"] + x_expansion,
        "y_min": bbox["y_min"] - y_expansion,
        "y_max": bbox["y_max"] + y_expansion,
        "z_min": bbox["z_min"] - z_expansion,
        "z_max": bbox["z_max"] + z_expansion
    }

# save_uncertain_objects moved to object_services, which owns `uncertain_objects`.
# It was defined here and called from BOTH modules, and object_services cannot import this
# one (this module imports it), so every call from there raised NameError -- GA-22.


def save_agent_poses(agent_poses):
    """Save the accumulated agent poses (with timestamp) to a JSON file.

    Rewrites the whole array, so it costs O(n) per pose: fine at the current ~10 poses
    per run, but quadratic the moment poses are logged per rendered frame (~2,160 a run
    would be ~2.3M entry-writes, on the callback thread). append_agent_pose below is the
    append-only path that has to carry that rate; this file stays as the compatible
    whole-array view for anything already reading it.
    """
    try:
        with open(AGENT_POSES_LOG_FILE, "w") as f:
            json.dump(agent_poses, f, indent=2)
    except Exception as e:
        print(f"[WARN] Error saving agent_poses.json: {e}")


def append_agent_pose(entry):
    """Append one pose to the JSONL series — O(1) per pose, one line each.

    The analysis series: `agent_poses.jsonl` keeps every pose ever received, in order,
    and is never rewritten. Paper-side denominators (which GT objects entered the
    frustum) need every viewpoint, not the handful the perception cycle happens to
    publish at.
    """
    try:
        with open(AGENT_POSES_SERIES_FILE, "a") as f:
            f.write(json.dumps(entry) + "\n")
    except Exception as e:
        print(f"[WARN] Error appending agent_poses.jsonl: {e}")


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


def _utc_iso_from_seconds(timestamp_sec):
    return datetime.fromtimestamp(timestamp_sec, tz=timezone.utc).isoformat()


def _stamp_key(stamp_msg):
    return (int(stamp_msg.sec), int(stamp_msg.nanosec))


def _stamp_key_str(stamp_msg):
    sec, nanosec = _stamp_key(stamp_msg)
    return f"{sec}.{nanosec:09d}"


def publish_agent_path(node, agent_poses, pub):
    """Publish all accumulated agent poses together as a nav_msgs/Path (for RViz)."""
    path_msg = Path()
    path_msg.header.frame_id = "map"
    if agent_poses:
        last_timestamp = agent_poses[-1].get("timestamp")
        path_msg.header.stamp = _stamp_from_seconds(last_timestamp) if last_timestamp is not None else node.get_clock().now().to_msg()
    else:
        path_msg.header.stamp = node.get_clock().now().to_msg()

    for entry in agent_poses:
        pose_stamped = PoseStamped()
        pose_stamped.header.frame_id = "map"
        timestamp_sec = entry.get("timestamp")
        pose_stamped.header.stamp = _stamp_from_seconds(timestamp_sec) if timestamp_sec is not None else node.get_clock().now().to_msg()
        pose_stamped.pose.position.x = entry["x"]
        pose_stamped.pose.position.y = entry["y"]
        pose_stamped.pose.position.z = entry["z"]
        pose_stamped.pose.orientation.x = entry["qx"]
        pose_stamped.pose.orientation.y = entry["qy"]
        pose_stamped.pose.orientation.z = entry["qz"]
        pose_stamped.pose.orientation.w = entry["qw"]
        path_msg.poses.append(pose_stamped)

    pub.publish(path_msg)

def publish_persistent_centroids(node, wm, pub):
    marker_array = MarkerArray()
    for i, obj in enumerate(wm.persistent_perceptions):
        if obj.bbox is None or "door" in obj.label.lower():
             continue
        marker = Marker()
        marker.header.frame_id = "map"
        marker.header.stamp = _stamp_from_seconds(
            getattr(obj, "last_perception_time", None)
        ) if getattr(obj, "last_perception_time", None) else node.get_clock().now().to_msg()
        marker.id = i
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        marker.pose.orientation.w = 1.0
        marker.pose.position.x = (obj.bbox['x_min'] + obj.bbox['x_max']) / 2.0
        marker.pose.position.y = (obj.bbox['y_min'] + obj.bbox['y_max']) / 2.0
        marker.pose.position.z = (obj.bbox['z_min'] + obj.bbox['z_max']) / 2.0
        marker.scale.x = marker.scale.y = marker.scale.z = 0.1
        marker.color.a = 1.0
        marker.color.r, marker.color.g, marker.color.b = 0.0, 1.0, 0.0
        marker_array.markers.append(marker)
    pub.publish(marker_array)

def publish_uncertain_bboxes(node, uncertain_objects, pub):
    marker_array = MarkerArray()
    for i, obj in enumerate(uncertain_objects):
        if obj.bbox is None or "door" in obj.label.lower():
             continue
        marker = Marker()
        marker.header.frame_id = "map"
        marker.header.stamp = _stamp_from_seconds(
            getattr(obj, "last_perception_time", None)
        ) if getattr(obj, "last_perception_time", None) else node.get_clock().now().to_msg()
        marker.id = i
        marker.type = Marker.CUBE
        marker.action = Marker.ADD
        marker.pose.orientation.w = 1.0
        marker.pose.position.x = (obj.bbox['x_min'] + obj.bbox['x_max']) / 2.0
        marker.pose.position.y = (obj.bbox['y_min'] + obj.bbox['y_max']) / 2.0
        marker.pose.position.z = (obj.bbox['z_min'] + obj.bbox['z_max']) / 2.0
        marker.scale.x = obj.bbox['x_max'] - obj.bbox['x_min']
        marker.scale.y = obj.bbox['y_max'] - obj.bbox['y_min']
        marker.scale.z = obj.bbox['z_max'] - obj.bbox['z_min']
        marker.color.a = 0.5
        marker.color.r, marker.color.g, marker.color.b = 1.0, 0.5, 0.0
        marker_array.markers.append(marker)
    pub.publish(marker_array)

def publish_uncertain_centroids(node, uncertain_objects, pub):
    marker_array = MarkerArray()
    for i, obj in enumerate(uncertain_objects):
        if obj.bbox is None or "door" in obj.label.lower(): 
            continue
        marker = Marker()
        marker.header.frame_id = "map"
        marker.header.stamp = _stamp_from_seconds(
            getattr(obj, "last_perception_time", None)
        ) if getattr(obj, "last_perception_time", None) else node.get_clock().now().to_msg()
        marker.id = i
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        marker.pose.orientation.w = 1.0
        marker.pose.position.x = (obj.bbox['x_min'] + obj.bbox['x_max']) / 2.0
        marker.pose.position.y = (obj.bbox['y_min'] + obj.bbox['y_max']) / 2.0
        marker.pose.position.z = (obj.bbox['z_min'] + obj.bbox['z_max']) / 2.0
        marker.scale.x = marker.scale.y = marker.scale.z = 0.1
        marker.color.a = 1.0
        marker.color.r, marker.color.g, marker.color.b = 1.0, 0.5, 0.0
        marker_array.markers.append(marker)
    pub.publish(marker_array)

def publish_pov_volume(node, pov_volume, pub):
    marker_array = MarkerArray()
    marker = Marker()
    marker.header.frame_id = "map"
    marker.header.stamp = node.get_clock().now().to_msg()
    marker.id = 0
    marker.type = Marker.CUBE
    marker.action = Marker.ADD
    marker.pose.orientation.w = 1.0
    
    marker.pose.position.x = (pov_volume['x_min'] + pov_volume['x_max']) / 2.0
    marker.pose.position.y = (pov_volume['y_min'] + pov_volume['y_max']) / 2.0
    marker.pose.position.z = (pov_volume['z_min'] + pov_volume['z_max']) / 2.0
    
    marker.scale.x = pov_volume['x_max'] - pov_volume['x_min']
    marker.scale.y = pov_volume['y_max'] - pov_volume['y_min']
    marker.scale.z = pov_volume['z_max'] - pov_volume['z_min']
    
    marker.color.a = 0.2
    marker.color.r, marker.color.g, marker.color.b = 0.0, 0.0, 1.0
    
    marker_array.markers.append(marker)
    pub.publish(marker_array)


# ============= MAIN SERVICE NODE =============

class ObjectManagerService(Node):
    def __init__(self):
        super().__init__('object_tracking_service_node')
        self.kb_instance_counters = {}
        self.get_logger().info("=== ObjectManagerService Initialized ===")
        self.get_logger().info(f"Log sintetico operazioni: {SYNTHETIC_LOG_FILE}")
        
        with open(SYNTHETIC_LOG_FILE, "a") as f:
            f.write(f"\n{'='*50}\n")
            f.write(f"NEW RUN: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"{'='*50}\n")

        self.exploration_mode = True
        # Never initialised: `self.tracking_step_counter += 1` at the top of
        # object_tracking_callback is reachable only once exploration_mode is False, and the
        # transition that clears it sets the counter to 1 -- so the attribute exists by
        # ORDERING, guarded by an invariant nobody states. Any future path that leaves
        # exploration without going through that branch is an AttributeError on the first
        # tracking frame. One line removes the dependence on the invariant.
        self.tracking_step_counter = 0
        self.seen_again = False
        self.latest_bboxes = {}
        self.latest_fov_volume = None
        # W6. Run-cumulative tally of description statuses (ok / model_abstained /
        # parse_failed / call_failed / unanswered), written to the decision log once per
        # tracking cycle. Initialised here so the description loop never KeyErrors.
        self._vlm_status_counts = {}
        self.uncertain_objects = []
        self.exploration_frame_counter = 0
        self.robot_has_moved = False
        # When the current motion began, as a perception stamp (sec + nanosec/1e9), or None
        # while stationary. A buffered pair is judged by WHEN IT WAS OBSERVED, not by what
        # the robot is doing when it arrives -- see _try_process.
        self._moving_since = None
        self._dropped_moving_pairs = 0
        # Arrival counters. The two callbacks used to buffer unconditionally with a single
        # silent `stamp is None` exit, so "every message arrived unusable" and "no message
        # arrived" produced identical evidence -- which is the pair of possibilities three
        # lanes could not separate on runs 7 and 8. These make the next run state which.
        self._n_desc_msgs = 0
        self._n_bbox_msgs = 0
        self._n_desc_no_stamp = 0
        self._n_bbox_no_stamp = 0
        self._last_nopair_state = None
        # GA-83: when a bbox last arrived, and whether the silence has been reported. The
        # detector died four cycles in and the stack ran 55 more minutes at 322% CPU with
        # nothing to process -- visible only in the container's process table, because
        # nothing in the log or the bundle said the producer was gone. A node with no input
        # cannot know why, but it can say that it has none.
        self._last_bbox_at = None
        self._input_silence_reported = False
        self._input_silence_strikes = 0
        self._stops_since_input = 0

        self.latest_descriptions = None
        self.latest_bboxes_msg = None

        self.agent_poses = []
        self.agent_pose_history = deque(maxlen=2000)
        self.latest_agent_pose = None
        self._last_pose_snapshot = 0.0
        self._last_path_publish = 0.0
        self._pending_descriptions = {}
        self._pending_bboxes = {}
        
        # --- INIT ROOM MANAGER ---
        self.room_manager = RoomManager(
            w2v_model=world2vec,
            node=self,
            map_topic='/rtabmap/map',
            cloud_map_topic='/rtabmap/cloud_map',
        )
        self.object_services = ObjectServices(self.room_manager)
        self.last_room_check_time = time.time()

        # Extension seam (config `hooks`, see hooks.py): admission filter, node
        # refiner and the re-evaluation queue. Blueprints unless configured.
        self.filter_hook, self.refiner_hook, self.reeval = load_hooks(CFG)
        self._reeval_last = {}            # object_id -> monotonic time of its last queueing (GA-11)
        self._reeval_debounced = 0        # counters, so a run can report how often the bounds bit
        self._reeval_fanout_capped = 0
        self.decision_log = DecisionLog(CFG["hooks"]["decisions_log"] or os.path.join(log_dir, "hook_decisions.jsonl"))
        self.get_logger().info(f"hooks: filter={self.filter_hook.name} refiner={self.refiner_hook.name} "
                               f"log={self.decision_log.path}")

        self.wall_sub = self.create_subscription(
            String,
            '/detected_wall_segments',
            self.walls_callback,
            10
        )
        from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy

        qos_poly = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT, # o BEST_EFFORT se la rete è lenta
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )

        qos_latch = QoSProfile(depth=10, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        qos_standard = QoSProfile(depth=10)
        
        
        # Publishers
        self.kb_add_pub = self.create_publisher(String, '/kb/add_fact', 10)
        self.persistent_bbox_pub=self.object_services.persistent_bbox_pub
        self.persistent_centroids_pub = self.object_services.persistent_centroids_pub
        self.considered_volume_pub = self.object_services.considered_volume_pub
        self.uncertain_bboxes_pub = self.object_services.uncertain_bboxes_pub
        self.uncertain_centroids_pub = self.object_services.uncertain_centroids_pub
        self.uncertain_objects = self.object_services.uncertain_objects
        self.tracking_activated_pub = self.create_publisher(Bool, '/tracking_mode_activated', qos_standard)
        
        self.agent_path_pub = self.create_publisher(Path, '/agent_path', qos_latch)
        
        #self.room_area_pub = self.create_publisher(MarkerArray, '/room_areas_array', qos_latch)
        
        # Tell Perception about room changes
        self.room_pub = self.create_publisher(String, '/current_room', 10)
       
        # Service server
        self.srv = self.create_service(
            ObjectTrackingService,
            'object_tracking_service',
            self.object_tracking_callback
        )

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.graph_api_base_url = GRAPH_API_BASE_URL.rstrip('/')
        self.graph_api_timeout = GRAPH_API_TIMEOUT
        self._graph_api_process = None
        self.get_logger().info(f"Graph API base URL: {self.graph_api_base_url}")
        self.get_logger().info('Object Tracking Service ready')

        # Subscribers
        self.create_subscription(Bool, "/robot_movement_detected", self.movement_callback, qos_poly)

        self.create_subscription(ObjectDescriptionArray, '/object_descriptions', self._descriptions_callback, qos_standard)
        # GA-108: describer answers that arrived after their cycle, addressed by the crop's
        # origin frame and 2D box. They go to the object whose SIGHTING matches, not to the
        # next cycle's namesake (run 20260906_223701: 204 of 212 objects ended "unknown").
        self.create_subscription(ObjectDescriptionArray, '/object_descriptions_late',
                                 self._late_descriptions_callback, qos_standard)
        self._n_late_applied = 0
        self._n_late_unmatched = 0
        self.create_subscription(Bbox3dArray, '/bbox_3d', self._bboxes_callback, qos_standard)
        self.create_subscription(PoseStamped, '/agent_camera_pose', self._agent_pose_callback, qos_standard)
        self.get_logger().info("Subscribing to /object_descriptions, /bbox_3d and /agent_camera_pose")
        self.get_logger().info(f"Node name={self.get_name()} ns={self.get_namespace()}")

        self._bbox_timer = self.create_timer(2.0, self.periodic_bbox_publisher)
        self._input_watchdog = self.create_timer(INPUT_SILENCE_TIMEOUT, self._check_input_silence)
        #self._uncertain_cleanup_timer = self.create_timer(5.0, self._cleanup_uncertain_by_time)

    @synchronized_world_model
    def reassign_objects_to_rooms(self):
        """Synchronize persistent object assignments after map resegmentation."""
        changed = self.room_manager.reassign_objects_by_geometry(wm.persistent_perceptions)
        if not changed:
            return

        for obj, old_room, new_room in changed:
            self.object_services.db.on_object_room_changed(
                obj, old_room, new_room, step=self.object_services.tracking_step_counter
            )
            # GA-11: a room change is a change to this object; queue it for a second look.
            self._note_update(getattr(obj, "object_id", None) or obj.label, reason="room_changed")
            self.get_logger().info(
                f"Object '{obj.label}' riassegnato: {old_room} -> {new_room}"
            )

        publish_persistent_bboxes(self, wm, self.persistent_bbox_pub)
        publish_persistent_centroids(self, wm, self.persistent_centroids_pub)
        save_persistent_perceptions(self.object_services)

    def movement_callback(self, msg):
        # Sincronizza lo stato reale: True se si muove, False se è fermo
        was_moving = self.robot_has_moved
        self.robot_has_moved = msg.data
        if was_moving and not msg.data:
            # GA-94b: a STOP is the event that produces a detection. Counted so the input
            # watchdog can measure silence in stops rather than in seconds -- see
            # _check_input_silence.
            self._stops_since_input = getattr(self, "_stops_since_input", 0) + 1
        if msg.data:
            if not was_moving:
                now = self.get_clock().now().to_msg()
                self._moving_since = now.sec + now.nanosec * 1e-9
            # The pending buffers are NOT cleared here any more.
            #
            # A perception cycle is triggered while the robot is stationary, but its VLM
            # round-trip takes 1.6-2.3 s, so the descriptions and boxes for a STATIONARY
            # observation arrive well after it was taken. Wiping on the arrival of a
            # movement message therefore discarded observations that were correctly made
            # while stopped, purely because the robot had set off again in the meantime --
            # and on a driving run that is nearly all of them.
            #
            # `latest_bboxes` is still cleared: it is the live view of what is in front of
            # the robot right now, and that really is invalidated by motion.
            self.latest_bboxes.clear()
            self.object_services.log_both('warn', "[MOVEMENT] Robot is moving -> room creation blocked")
        else:
            self._moving_since = None
            self.object_services.log_both('info', "[MOVEMENT] Robot has stopped -> room creation allowed")
            # Motion has ended: anything buffered from before it began is still valid.
            self._try_process()

    def _agent_pose_callback(self, msg):
        stamp = msg.header.stamp
        timestamp_sec = stamp.sec + stamp.nanosec * 1e-9

        entry = {
            "timestamp": timestamp_sec,
            "datetime": _utc_iso_from_seconds(timestamp_sec),
            "x": msg.pose.position.x,
            "y": msg.pose.position.y,
            "z": msg.pose.position.z,
            "qx": msg.pose.orientation.x,
            "qy": msg.pose.orientation.y,
            "qz": msg.pose.orientation.z,
            "qw": msg.pose.orientation.w,
        }
        self.agent_poses.append(entry)
        self.agent_pose_history.append(entry)
        self.latest_agent_pose = entry

        # Lossless and O(1): this is the record the paper's frustum denominator reads.
        append_agent_pose(entry)

        # The other two touch the WHOLE array on every pose — save_agent_poses rewrites
        # the file, publish_agent_path rebuilds every PoseStamped — so at per-frame pose
        # rate they turn this callback quadratic. Neither needs per-pose freshness: the
        # JSONL above is the lossless series, and RViz is happy with a 1 Hz path.
        now = time.monotonic()
        if now - self._last_pose_snapshot >= AGENT_POSES_SNAPSHOT_PERIOD:
            self._last_pose_snapshot = now
            save_agent_poses(self.agent_poses)
        if now - self._last_path_publish >= AGENT_PATH_PUBLISH_PERIOD:
            self._last_path_publish = now
            publish_agent_path(self, self.agent_poses, self.agent_path_pub)

    def _closest_agent_pose(self, timestamp_sec):
        if timestamp_sec is None:
            return self.latest_agent_pose
        if not self.agent_pose_history:
            return self.latest_agent_pose

        return min(
            self.agent_pose_history,
            key=lambda entry: abs(float(entry.get("timestamp", timestamp_sec)) - float(timestamp_sec))
        )

    def _scan_acc(self):
        """Per-cycle accumulator for the tracking scan. Reset by the cycle, not by the call."""
        if getattr(self, "_scan_stats", None) is None:
            self._scan_stats = {"calls": 0, "visited": 0, "skipped_unstable": 0,
                                "scored": 0, "distances": [], "winner_distance": None,
                                "winners": 0,
                                # GA-289: what the two new gates did, so the fix is measured
                                # from the same row that measured the defect.
                                "pruned_locality": 0, "locality_unmeasured": 0,
                                "reach_fallback": 0, "refused_evidence": 0}
        self._scan_stats["calls"] += 1
        return self._scan_stats

    def flush_scan_summary(self, frame_id=None):
        """Emit ONE row summarising this cycle's tracking scan, then reset.

        The same shape as `not_offered_summary` on the merge path, and for the same reason:
        a path that records nothing cannot be measured, and every performance claim about it
        stays a code reading. One row per cycle, never one per comparison.
        """
        st = getattr(self, "_scan_stats", None)
        self._scan_stats = None
        if not st or not st["calls"]:
            return
        d = sorted(st["distances"])

        def pct(q):
            if not d:
                return None
            return d[min(len(d) - 1, max(0, int(round(q / 100.0 * (len(d) - 1)))))]

        try:
            self.decision_log.write(
                "tracking_scan_summary", "<cycle>",
                frame=frame_id,
                detections=st["calls"],
                objects_visited=st["visited"],
                skipped_unstable=st["skipped_unstable"],
                comparisons_scored=st["scored"],
                winners=st["winners"],
                winner_distance=st["winner_distance"],
                # GA-289. Skipped before similarity because the object was outside its own
                # reach; distance unmeasurable (no bbox) so NOT pruned; reach came from the
                # fallback radius (no covariance yet); would-be winners refused on evidence.
                pruned_locality=st["pruned_locality"],
                locality_unmeasured=st["locality_unmeasured"],
                reach_fallback=st["reach_fallback"],
                refused_evidence=st["refused_evidence"],
                # the distribution of the LOSERS is the number that decides whether a
                # locality gate would have saved anything here
                loser_distance_min=(round(d[0], 3) if d else None),
                loser_distance_p50=(round(pct(50), 3) if d else None),
                loser_distance_p90=(round(pct(90), 3) if d else None),
                loser_distance_max=(round(d[-1], 3) if d else None),
                # How many comparisons a locality gate at each radius would have skipped.
                # Reported against FIXED radii rather than a configured constant, because
                # `MAX_MATCH_DISTANCE` no longer exists -- GA-49 removed `max_match_distance_m`
                # from config, so the brief's "it is consulted after the scan" is out of date:
                # there is nothing to consult. The distribution lets the reader pick a radius
                # after seeing the data rather than before.
                would_prune={f"{r}m": sum(1 for x in d if x > r)
                             for r in (0.5, 1.0, 1.5, 2.0, 3.0)})
        except Exception as exc:
            self.get_logger().warn(f"tracking_scan_summary not written: {exc}")

    @synchronized_world_model
    def check_tracking_transition(self, label_base, color, material, description_embedding, bbox):
        best_match = None
        highest_similarity = -1.0
        # GA-190: measure this scan. The merge path is legible -- it logs what candidate
        # selection excluded, which is why 87% pruning could be MEASURED there tonight -- and
        # this path logs nothing at all, so its cost is invisible. D1 says locality gates
        # before similarity; here `lost_similarity` runs over EVERY persistent object and
        # MAX_MATCH_DISTANCE is consulted only on the winner. That is certainly a contract
        # deviation; whether it costs anything is unknown, and a handful of comparisons per
        # detection would make it a tidiness issue rather than a performance one.
        #
        # Cheap by construction: a centroid distance is a few arithmetic operations against a
        # composite similarity over four terms including a word2vec embedding, so measuring
        # costs a fraction of a percent of what it measures. Accumulated here, ONE summary
        # row per cycle -- hook_decisions.jsonl is already 1.8 GB and 99.98% merge_refused.
        _scan = self._scan_acc()
        _new_c = _bbox_centre(bbox)
        _new_half = _half_diagonal_m(bbox)

        # GA-107's named site, still iterating the LIVE list: this scan runs on the
        # executor thread with no lock held (object_tracking_callback cannot hold it --
        # see snapshot()'s docstring), while the HTTP surface can trigger a merge and
        # REMOVE from this list mid-pass; a removal under a live list iterator silently
        # skips the NEXT object, which is then never considered for association in
        # this pass. The snapshot is the pass's consistent view: it judges the world
        # as it was when the pass began.
        for obj in wm.snapshot():
            _scan["visited"] += 1
            # GA-12: default is NOW, not the epoch -- an unstamped object is not yet stable.
            _ct = getattr(obj, 'creation_time', None)
            if (time.time() - (time.time() if _ct is None else _ct)) < OBJECT_STABILITY_TIMEOUT:
                # Skipped INSIDE the loop, so it is visited to be discarded: it belongs
                # outside the candidate set, and the count says how often that costs a visit.
                _scan["skipped_unstable"] += 1
                continue
            _d = _centre_distance(_new_c, getattr(obj, "bbox", None))
            if _d is not None:
                _scan["distances"].append(_d)
                # GA-289. D1: locality BEFORE similarity. The object's own reach (covariance
                # shell, or extent + fallback) plus the detection's half-diagonal, the same
                # symmetric test the merge path applies. Pruned pairs are counted, not scored.
                _reach, _basis = tracking_reach_m(obj)
                if "fallback" in _basis:
                    _scan["reach_fallback"] += 1
                if _d > _reach + _new_half:
                    _scan["pruned_locality"] += 1
                    continue
            else:
                # No distance is measurable (no bbox on one side): the discriminator is
                # absent, so this ABSTAINS from pruning rather than assuming near or far.
                _scan["locality_unmeasured"] += 1

            obj_label_base = obj.label.split('#')[0] if '#' in obj.label else obj.label

            if not hasattr(obj, "embedding") or obj.embedding is None:
                obj.embedding = get_embedding(world2vec, obj.description)
            # FIX (same bug as in the two association loops): a missing description
            # embedding is absent evidence, not a reason to skip. With "unknown"
            # descriptions (VLM down) every embedding is None, so this `continue`
            # made the EXPLORATION -> TRACKING transition impossible: the stack never
            # reached updates, the uncertain pool or merging.
            similarity, _ev = lost_similarity_detailed(
                world2vec, label_base, obj_label_base, color, obj.color,
                material, obj.material, description_embedding, obj.embedding)

            _scan["scored"] += 1
            if similarity > SIM_THRESHOLD and similarity > highest_similarity:
                # GA-289. Evidence before a win, AFTER the similarity test on purpose: a
                # low-score loser must keep its own reason. A 1.000 on the label alone is
                # GA-101's zero-evidence merge on this path, and it does not win here either.
                if _ev["optional_count"] < TRACKING_MIN_EVIDENCE:
                    _scan["refused_evidence"] += 1
                    continue
                highest_similarity = similarity
                best_match = obj
                _scan["winner_distance"] = _d

        if best_match:
            # GA-190: counted here, not in the loop. Set inside the loop it would count every
            # improvement to the running best rather than the one match this detection made,
            # and the summary would report more winners than detections. My own check caught
            # that this was never incremented at all.
            _scan["winners"] += 1
            print(f"[BEST MATCH FOUND] Detected: '{label_base}' -> Best in memory: '{best_match.label}' (Score: {highest_similarity:.3f})")
            
            if best_match.bbox is None:
                return False, None, 0.0

            old_x = (best_match.bbox['x_min'] + best_match.bbox['x_max']) / 2.0
            old_y = (best_match.bbox['y_min'] + best_match.bbox['y_max']) / 2.0
            old_z = (best_match.bbox['z_min'] + best_match.bbox['z_max']) / 2.0
            new_x = (bbox['x_min'] + bbox['x_max']) / 2.0
            new_y = (bbox['y_min'] + bbox['y_max']) / 2.0
            new_z = (bbox['z_min'] + bbox['z_max']) / 2.0
            
            distance = np.sqrt((new_x - old_x)**2 + (new_y - old_y)**2 + (new_z - old_z)**2)
            iou = compute_iou_3d(bbox, best_match.bbox)
            
            if distance > TRANSITION_MOVE_DISTANCE_M and iou < EXPLORATION_IOU_THRESHOLD:
                self.object_services.log_both('warn', f"[TRACKING TRANSITION] Object '{best_match.label}' is the best match but moved! (Dist: {distance:.2f}m), the iou was {iou}")
                return True, best_match, distance
        
        return False, None, 0.0

    @synchronized_world_model
    def update_spatial_relations(self):
        for obj in wm.persistent_perceptions:
            ensure_relations(obj)
            for key in obj.relations:
                obj.relations[key].clear()

        for obj in wm.persistent_perceptions:
            ensure_relations(obj)
            # unused — room_id was fetched but never consumed in this loop
            # room_id = getattr(obj, "room_id", None)

        for i, obj_a in enumerate(wm.persistent_perceptions):
            for j, obj_b in enumerate(wm.persistent_perceptions):
                if i == j:
                    continue
                for _, pred, target in infer_spatial_relations(obj_a, obj_b):
                    obj_a.relations[pred].add(target)

    @synchronized_world_model
    def publish_kb_relation_facts(self):
        facts = []

        for obj in wm.persistent_perceptions:
            ensure_relations(obj)

            if not getattr(obj, "object_id", None):
                continue

            for pred, targets in obj.relations.items():
                for target in targets:
                    facts.append(f"{obj.object_id} {pred} {target}")

        return facts

    def _late_descriptions_callback(self, msg):
        """GA-108. Hand each late description to the object seen in its origin frame."""
        for d in getattr(msg, "descriptions", []) or []:
            origin_frame = str(getattr(d, "origin_frame", "") or "")
            if not origin_frame:
                continue
            obj = self._object_for_sighting(origin_frame, list(getattr(d, "origin_bbox_2d", []) or []),
                                            str(getattr(d, "label", "") or ""))
            if obj is None:
                self._n_late_unmatched += 1
                self.object_services.log_both(
                    'info', f"[VLM-late] no object sighted in frame {origin_frame} matches "
                            f"{d.label}; dropped ({self._n_late_unmatched} unmatched so far)")
                continue
            changed = []
            for field in ("description", "color", "material", "shape"):
                new = str(getattr(d, field, "") or "").strip()
                cur = str(getattr(obj, field, "") or "").strip().lower()
                # Only fill what is still unknown: a late answer must never overwrite a
                # description that a later, better-overlapping sighting already supplied.
                if new and new.lower() != "unknown" and (not cur or cur == "unknown"):
                    setattr(obj, field, new)
                    changed.append(field)
            if changed:
                self._n_late_applied += 1
                self.object_services.log_both(
                    'info', f"[VLM-late] {obj.label} ({getattr(obj, 'object_id', '?')}) described from "
                            f"frame {origin_frame}: {', '.join(changed)} ({self._n_late_applied} applied)")
                try:
                    self._note_update(getattr(obj, "object_id", None), reason="described")
                except Exception as exc:   # a re-evaluation trigger must not lose the description
                    self.object_services.log_both('warn', f"[VLM-late] re-evaluation not queued: {exc}")

    def _cycle_bbox_2d_for(self, obj):
        """This cycle's detector box for `obj`, from the object's own bbox dict when the
        update path kept it, else from the cycle's incoming boxes by label (GA-316 residual:
        194 of 212 run C objects had no 2D box on any sighting, so co-visibility could not
        conclude even where a shared frame existed)."""
        box = (getattr(obj, "bbox", None) or {}).get("bbox_2d")
        if box is not None:
            return box
        label = getattr(obj, "label", None)
        for entry in (getattr(self, "latest_bboxes", None) or {}).values():
            if entry.get("label") == label and (entry.get("bbox") or {}).get("bbox_2d") is not None:
                return entry["bbox"]["bbox_2d"]
        return None

    @staticmethod
    def _frame_seconds(frame_id):
        """perception's frame id ("<sec>_<nanosec>") or om6's float seconds -> float seconds."""
        if isinstance(frame_id, (int, float)):
            return float(frame_id)
        txt = str(frame_id)
        if "_" in txt:
            sec, _, nsec = txt.partition("_")
            try:
                return int(sec) + int(nsec) * 1e-9
            except ValueError:
                return None
        try:
            return float(txt)
        except ValueError:
            return None

    def _object_for_sighting(self, origin_frame, origin_bbox_2d, label):
        """The world-model object whose sighting in `origin_frame` has `origin_bbox_2d`.

        Frame match is by time (1 ms), because perception keys frames as "<sec>_<nanosec>" and
        this node stores float seconds. Within the frame the 2D box decides (IoU >= 0.3); a
        sighting with no 2D box falls back to the label base, and only if it is the sole
        candidate -- two same-label objects in one frame with no box stay unmatched.
        """
        from association import _iou_2d
        t = self._frame_seconds(origin_frame)
        if t is None:
            return None
        base = label.split('#')[0].strip().lower()
        best, best_iou, by_label = None, 0.0, []
        for obj in list(wm.persistent_perceptions):
            for obs in getattr(obj, "observations", None) or []:
                ft = self._frame_seconds(getattr(obs, "frame_id", None))
                if ft is None or abs(ft - t) > 1e-3:
                    continue
                box = getattr(obs, "bbox_2d", None)
                if box is not None and origin_bbox_2d and len(origin_bbox_2d) == 4:
                    iou = _iou_2d(box, origin_bbox_2d)
                    if iou > best_iou:
                        best, best_iou = obj, iou
                elif str(getattr(obj, "label", "")).split('#')[0].strip().lower() == base:
                    by_label.append(obj)
        if best is not None and best_iou >= 0.3:
            return best
        if len(by_label) == 1:
            return by_label[0]
        return None

    def _record_sighting(self, obj, perception_timestamp):
        """Append one Observation to `obj`, or none at all. GA-186.

        The camera position comes from `latest_agent_pose`, which is the pose stream the
        bundle already records. WITHOUT A POSE THERE IS NO OBSERVATION: a sighting whose
        camera position was guessed would put a fabricated bearing into the appearance
        channel and a fabricated range into the covariance, and both would look exactly
        like a measurement. Abstaining here is what makes every channel downstream able to
        say "not evaluated" instead of "evaluated and equal".

        `MAX_OBSERVATIONS_PER_OBJECT` bounds the memory: an object seen in 200 frames does
        not need 200 records to establish its covariance or its view spread, and the list
        is walked pairwise by the co-visibility channel. The OLDEST are dropped, keeping
        the most recent views, because those are the ones a current comparison is about.
        """
        pose = getattr(self, "latest_agent_pose", None)
        if pose is None:
            return
        centroid = getattr(obj, "centroid", None)
        if centroid is None:
            return
        try:
            obs = Observation(
                frame_id=perception_timestamp,
                camera_position=(pose["x"], pose["y"], pose["z"]),
                centroid=centroid,
                stamp=perception_timestamp,
                # GA-186: this frame's DETECTOR box, carried in the bbox dict since the
                # message grew `has_bbox_2d`. Absent stays None -- co-visibility abstains
                # on a missing box and vetoes only on a measured disjoint one.
                bbox_2d=self._cycle_bbox_2d_for(obj),
                # GA-190: this view's appearance embedding, so the appearance channel
                # compares MEASURED crops rather than a shape descriptor derived from the
                # box the two objects already agree on.
                appearance=(getattr(obj, "bbox", None) or {}).get("clip_embedding"),
            )
        except (KeyError, TypeError, ValueError) as e:
            # A malformed pose or centroid is a real defect and must not be silently
            # replaced by a default; but it must also not kill the tracking callback for
            # every other object in the frame. Logged loudly, counted, never substituted.
            self.object_services.log_both(
                'error', f"[ASSOC] sighting non registrata per '{getattr(obj, 'label', '?')}': {e}")
            return
        if not hasattr(obj, "observations") or obj.observations is None:
            obj.observations = []
        obj.observations.append(obs)
        if len(obj.observations) > MAX_OBSERVATIONS_PER_OBJECT:
            del obj.observations[:-MAX_OBSERVATIONS_PER_OBJECT]

    def publish_kb_facts(self, current_perception_objects):
        facts = []

        for obj in current_perception_objects:
            if not getattr(obj, "object_id", None):
                continue

            cls = obj.label.split('#')[0].replace(' ', '_')
            inst = obj.object_id

            facts.append(f"{inst} rdf:type {cls}")

            if getattr(obj, "color", "unknown") != "unknown":
                facts.append(f"{inst} hasColor {obj.color}")

            if getattr(obj, "material", "unknown") != "unknown":
                facts.append(f"{inst} hasMaterial {obj.material}")

        return facts

    def object_tracking_callback(self, request, response):
        if not self.exploration_mode:
            self.tracking_step_counter += 1

        if self.robot_has_moved:
            self.object_services.log_both('warn', "Robot is moving — data discarded by object_tracking_callback")
            response.status = "moving"
            response.num_objects = len(wm.persistent_perceptions)
            response.tracking_mode_activated = False
            return response

        in_exploration = self.exploration_mode
        current_perception_objects = []
        objects_modified = False
        tracking_activated = False

        bbox_header = getattr(request.bboxes, "header", None)
        if bbox_header is not None and (bbox_header.stamp.sec != 0 or bbox_header.stamp.nanosec != 0):
            perception_stamp = bbox_header.stamp
        else:
            perception_stamp = self.get_clock().now().to_msg()
        perception_timestamp = perception_stamp.sec + perception_stamp.nanosec * 1e-9

        if in_exploration:
            self.exploration_frame_counter += 1

        self.latest_bboxes = {}

        if request.bboxes.fov_x_max != 0 or request.bboxes.fov_y_max != 0 or request.bboxes.fov_z_max != 0:
            self.latest_fov_volume = {
                "x_min": request.bboxes.fov_x_min,
                "x_max": request.bboxes.fov_x_max,
                "y_min": request.bboxes.fov_y_min,
                "y_max": request.bboxes.fov_y_max,
                "z_min": request.bboxes.fov_z_min,
                "z_max": request.bboxes.fov_z_max
            }
        else:
            self.latest_fov_volume = None

        for box in request.bboxes.boxes:
            x_min, x_max = min(box.x_min, box.x_max), max(box.x_min, box.x_max)
            y_min, y_max = min(box.y_min, box.y_max), max(box.y_min, box.y_max)
            z_min, z_max = min(box.z_min, box.z_max), max(box.z_min, box.z_max)

            bbox_data = {
                "x_min": x_min, "x_max": x_max,
                "y_min": y_min, "y_max": y_max,
                "z_min": z_min, "z_max": z_max
            }
            # GA-100: carry the oriented box through, and ONLY when the publisher says it
            # measured one. Reading yaw unconditionally would put a fabricated 0.0 into
            # every bbox dict and make an unoriented object indistinguishable from one
            # measured as axis-aligned -- box_view and the merge keeper both read these keys.
            if getattr(box, "has_orientation", False):
                bbox_data["yaw"] = float(box.yaw)
                bbox_data["oriented_center"] = [float(v) for v in box.oriented_center]
                bbox_data["oriented_extents"] = [float(v) for v in box.oriented_extents]
            # GA-186: the detector's 2D box, and ONLY when the publisher says it carried
            # one. Same rule as the oriented box above -- an absent 2D box read as
            # (0,0,0,0) is a real box at the image origin as far as any reader can tell,
            # and co-visibility would then compute a confident overlap of 0.0 and VETO.
            if getattr(box, "has_bbox_2d", False):
                bbox_data["bbox_2d"] = [float(v) for v in box.bbox_2d]
            # GA-190: same rule, third field. Only when the publisher says it carried one.
            if getattr(box, "has_clip_embedding", False):
                bbox_data["clip_embedding"] = [float(v) for v in box.clip_embedding]
            temp_key = create_object_key(box.label, "", "", "")
            self.latest_bboxes[temp_key] = {
                "bbox": bbox_data, "label": box.label,
                "color": "", "material": "", "description": ""
            }

        current_time = time.time()

        if getattr(self, 'last_room_check_time', 0) == 0:
            self.last_room_check_time = current_time

        if (current_time - self.last_room_check_time) > 5.0:
            if len(request.descriptions.descriptions) > 0:
                old_room_id = self.room_manager.current_room_id
                self.room_manager.evaluate_scene(request.descriptions.descriptions, wm.persistent_perceptions)

                if self.room_manager.current_room_id != old_room_id:
                    room_msg = String()
                    room_msg.data = self.room_manager.current_room_id
                    self.room_pub.publish(room_msg)
                    self.object_services.log_both('info', f"Room change detected! Signalled Perception for: {self.room_manager.current_room_id}")

            self.last_room_check_time = current_time

        for description in request.descriptions.descriptions:
            label = description.label
            label_base = label.split('#')[0] if '#' in label else label
            color = description.color
            material = description.material
            description_text = description.description
            # GA-277. getattr, not attribute access: a bundle replayed against an OLDER
            # interface has no such field, and the seam must not raise on a message shape
            # that was valid when it was recorded.
            crop_path = getattr(description, "crop_path", "") or ""
            # W6. Same rule, same reason: the status field postdates older interfaces.
            # Tallied per run so the "unknown" population can be split into its routes
            # (ok / model_abstained / parse_failed / call_failed / unanswered) from the log.
            status = getattr(description, "status", "") or "unanswered"
            self._vlm_status_counts[status] = self._vlm_status_counts.get(status, 0) + 1

            description_embedding = get_embedding(world2vec, description_text)

            old_key = create_object_key(label, "", "", "")
            if old_key not in self.latest_bboxes:
                continue

            bbox = self.latest_bboxes[old_key]["bbox"]
            new_key = create_object_key(label, material, color, description_text)

            del self.latest_bboxes[old_key]
            self.latest_bboxes[new_key] = {
                "bbox": bbox, "label": label,
                "color": color, "material": material, "description": description_text,
                "status": status
            }

            already_seen = False
            transition = False

            if in_exploration:
                transition, obj, distance = self.check_tracking_transition(
                    label_base, color, material, description_embedding, bbox
                )

                if transition:
                    self.object_services.log_both('warn', "[TRANSITION] Switching from EXPLORATION to TRACKING mode")
                    self.exploration_mode = False
                    self.tracking_step_counter = 1
                    tracking_activated = True
                    in_exploration = False
                    self.exploration_frame_counter = 0

                    msg = Bool()
                    msg.data = True
                    self.tracking_activated_pub.publish(msg)

                    update_response = self.modify_existing_object(obj, bbox, description_embedding)
                    if update_response.success:
                        matching_obj = next(
                            (o for o in wm.snapshot()
                             if getattr(o, "object_id", None) == update_response.object_id),
                            obj,
                        )
                        current_perception_objects.append(matching_obj)
                        objects_modified = True
                        self._note_update(update_response.object_id)
                        # GA-10: only a SUCCESSFUL update means the detection was absorbed.
                        # This was set unconditionally, so a failed update dropped the
                        # detection silently -- it never reached filter_hook.judge and no
                        # decision row was written. A refused move (GA-24) arrives here, so
                        # the failure is now a visible proposal rather than a lost object.
                        already_seen = True
                        continue
                    else:
                        self.object_services.log_both('warn', f"Update failed for {obj.label}: {update_response.message}")
                    # GA-10, transition branch: the `continue` stood HERE, after the if/else,
                    # so a refused update still skipped the admission seam below and the
                    # comment above it was true of the other branch only. `in_exploration`
                    # is already False, which skips the exploration loop; `transition` skips
                    # the tracking loop (it would re-run the update that just failed).

            if in_exploration:
                for obj in wm.snapshot():
                    # GA-04: locality first. Previously the overlap test was conjoined with the
                    # similarity test below, so attributes were compared against every object in
                    # the map before geometry could rule any of them out.
                    if not locality_ok(bbox, obj, EXPLORATION_IOU_THRESHOLD):
                        continue

                    obj_label_base = obj.label.split('#')[0] if '#' in obj.label else obj.label
                    if not hasattr(obj, "embedding"):
                        obj.embedding = get_embedding(world2vec, obj.description)
                    # FIX: a missing description embedding is not a reason to
                    # skip the candidate — lost_similarity now treats it as
                    # absent evidence. Skipping here left undescribed objects
                    # unmatched forever (every re-detection became a new node).

                    similarity = lost_similarity(
                        world2vec, label_base, obj_label_base, color, obj.color,
                        material, obj.material, description_embedding, obj.embedding
                    )

                    # GA-04: `obj.bbox is not None` and the overlap recomputation that stood
                    # here are gone -- locality_ok above guarantees both for every candidate
                    # that reaches this point, so the acceptance test is now attributes only.
                    #
                    # GA-05: the unknown-description shortcut that stood here is deleted.
                    # It accepted the first candidate overlapping by 10% whenever EITHER
                    # side's description was "unknown", discarding the similarity computed
                    # three lines above -- so a chair standing at an undescribed table was
                    # declared to BE the table. lost_similarity already handles the unknown
                    # case correctly, by dropping unevidenced terms and renormalising; that
                    # is what the FIX comment above describes. Overlap is LOCALITY evidence
                    # and is used as the gate below, never as the score.
                    #
                    # Deleted with it, deliberately: that branch also back-filled an
                    # undescribed stored object's description, colour, material and
                    # embedding from the detection. Nothing else back-fills, so an object
                    # described once keeps that description and a genuinely unknown one
                    # stays unknown until it is re-detected as new. The loss is accepted
                    # (owner's ruling, 30 Aug): the back-fill was triggered by the very
                    # condition that made the match unsafe -- copying attributes across a
                    # match established on the ABSENCE of evidence. If back-filling is
                    # wanted it belongs on a match established ON evidence, which is the
                    # re-evaluation path (D15), not the association loop.
                    if similarity > SIM_THRESHOLD:
                        already_seen = True
                        current_perception_objects.append(obj)
                        # GA-07: the confirm_stationary guard that stood here is DELETED,
                        # with its twin below and the config key. It read
                        # `self.object_services.is_moving`, which is defined NOWHERE in the
                        # tree, so the getattr default False was taken on every arrival and
                        # `not confirm_stationary or not is_moving` was always True. This
                        # deletion therefore changes NO runtime behaviour -- the else branch
                        # has never executed once.
                        #
                        # It was inert in the PERMISSIVE direction, which is worse than
                        # absent: the config advertised that established boxes are protected
                        # during motion, and they never were. Reading the real
                        # `robot_has_moved` here cannot fix it either -- this code sits after
                        # an early return that fires when it is true, so it is provably False
                        # at this point. Real protection belongs ABOVE that early return and
                        # is a different change from this one.
                        obj.bbox = bbox
                        objects_modified = True
                        # GA-11: an in-place box write is a change to THIS object; queue it.
                        self._note_update(getattr(obj, "object_id", None) or obj.label, reason="box_written")
                        break

            elif not transition:
                best_match = None
                best_score = 0

                for obj in wm.snapshot():
                    # GA-04: in TRACKING mode there was NO locality gate at all --
                    # TRACKING_IOU_THRESHOLD's only use raised the score to 1.0, and the
                    # centre-distance guard below was disabled by its own shipped config. So
                    # acceptance was similarity alone, against every object in the map at any
                    # distance: two chairs in different rooms matched on the label term.
                    #
                    # A candidate with no bbox no longer matches here. It did before, on
                    # similarity alone; the exploration branch always excluded it, and D14
                    # prefers strict. Deliberate behaviour change, owner-approved 30 Aug.
                    if not locality_ok(bbox, obj, TRACKING_IOU_THRESHOLD):
                        continue

                    if not hasattr(obj, "embedding"):
                        obj.embedding = get_embedding(world2vec, obj.description)
                    # FIX: a missing description embedding is not a reason to
                    # skip the candidate — lost_similarity now treats it as
                    # absent evidence. Skipping here left undescribed objects
                    # unmatched forever (every re-detection became a new node).

                    obj_label_base = obj.label.split('#')[0] if '#' in obj.label else obj.label
                    similarity = lost_similarity(
                        world2vec, label_base, obj_label_base, color, obj.color,
                        material, obj.material, description_embedding, obj.embedding
                    )

                    # GA-05: the unknown-description override that stood here is deleted. It
                    # set similarity to 1.0 -- a PERFECT attribute score -- whenever either
                    # side's description was "unknown" and the boxes overlapped, so a lamp
                    # standing on an undescribed sofa took the sofa's node, and no later
                    # candidate could beat 1.0. Same defect as the exploration branch, by a
                    # different route. TRACKING_IOU_THRESHOLD has no use site in this file
                    # after this deletion; GA-04 gives it its single correct one, as the
                    # locality gate at the top of this loop.
                    #
                    # The centre-distance guard that stood here is gone with
                    # association.max_match_distance_m. D14 already settled that the distance
                    # cutoff is REPLACED by a locality test, so removing it executes a decision
                    # rather than making one -- and leaving a second, disabled locality
                    # mechanism beside a live one is a config key that reads as if it does
                    # something. The gate at the top of this loop is the locality test now.
                    if similarity > SIM_THRESHOLD and similarity > best_score:
                        best_score = similarity
                        best_match = obj

                if best_match:
                    # GA-10: `already_seen = True` stood here, before the update was even
                    # attempted, so a failure still suppressed the admission block below.
                    # It now follows the update's success, as in the transition branch.
                    # GA-07, second of the two dead guards -- see the exploration branch
                    # above. `target_bbox` resolved to `bbox` on every arrival because
                    # is_moving was always False, so passing `bbox` directly is behaviour-
                    # identical and removes a name that suggested a choice was being made.
                    update_response = self.modify_existing_object(best_match, bbox, description_embedding)
                    if update_response.success:
                        matching_obj = next(
                            (o for o in wm.snapshot()
                             if getattr(o, "object_id", None) == update_response.object_id),
                            best_match,
                        )
                        current_perception_objects.append(matching_obj)
                        objects_modified = True
                        self._note_update(update_response.object_id)
                        already_seen = True
                    else:
                        self.object_services.log_both('warn', f"Update failed for {best_match.label}: {update_response.message}")

            if not already_seen:
                # Admission seam: the configured Filter sees exactly what would be
                # sent to the Graph API and may refuse it (blueprint: never does).
                proposal = {
                    "label": label, "bbox": bbox, "color": color, "material": material,
                    "description": description_text,
                    "room_id": self.room_manager.assign_room_by_geometry(bbox),
                    # GA-277. The gate sees geometry and no pixels, so an image-based check
                    # on a borderline decision had nothing to look at. The PATH, not the
                    # image: the seam stays a small message and the reader opens the file.
                    "crop_path": crop_path,
                }
                decision = self.filter_hook.judge(proposal)
                self.decision_log.write("admission", label, filter=self.filter_hook.name, outcome=decision.outcome,
                                        reason=decision.reason, annotation=decision.annotation)
                if not decision.admitted:
                    self.object_services.log_both('warn', f"[{self.filter_hook.name}] refused {label}: {decision.reason}")
                    continue
                new_obj = self.add_new_object(
                    label, bbox, description_text, color, material,
                    description_embedding, in_exploration, proposal["room_id"]
                )
                if new_obj is not None:
                    # Join the decision to the object it produced. The admission line above is
                    # written BEFORE the object exists -- add_new_object only learns object_id
                    # from the Graph API's POST response -- and refused proposals never get one,
                    # so the link cannot live on that line and must be a second, later record.
                    #
                    # Without it the only key shared by the decision log and the world model is
                    # the label, which joined 9 of 11 objects on the 26 Aug run. That gap is why
                    # the analysis falls back to label + centroid on a 0.1 m grid for "distinct
                    # object", making a published denominator a function of a rounding constant
                    # (54 at 0.1 m, 33 at 1.0 m, same run).
                    #
                    # decision_id is an opaque string the hook put in its own annotation dict:
                    # generic seam data, nothing imported from any particular filter.
                    decision_id = (decision.annotation or {}).get("decision_id")
                    linked_id = getattr(new_obj, "object_id", None)
                    if decision_id and linked_id:
                        self.decision_log.write("link", linked_id,
                                                decision_id=decision_id, label=label)

                    # GA-192: carry the ONTOLOGY TYPE the filter just resolved onto the
                    # object. The hook already computed it -- `entity` is the aligned class
                    # and `alignment.status` says whether the alignment holds -- and it was
                    # being written to the decision log and then dropped, so the association
                    # layer's ontology channel abstained on EVERY pair for want of a type
                    # that had already been derived one call earlier.
                    #
                    # Read out of the generic annotation dict, exactly as `decision_id` is:
                    # this stays seam data and imports nothing from any particular filter.
                    # ALIGNMENT IS REQUIRED BEFORE THE TYPE IS USABLE -- an unaligned guess
                    # would let the channel compare two labels as if the ontology had
                    # endorsed them, which is the one thing the design says it must not do.
                    ann = decision.annotation or {}
                    aligned = (ann.get("alignment") or {}).get("status") == "aligned"
                    new_obj.onto_type = ann.get("entity") if aligned else None
                    # GA-240, owner ruling 2026-09-02. A PROVISIONAL admission -- the filter's
                    # hold or no-grounds -- enters the map and must be unusable by the
                    # ontological layer until the association/core level resolves it.
                    #
                    # Enforced HERE, at the one place the flag is set, because
                    # `association.channel_ontology` already abstains unless BOTH sides are
                    # aligned: "one or both sides unaligned; alignment is required before
                    # use". So clearing this single flag is exactly "not usable in the
                    # ontological layers", and it cannot be forgotten by a consumer that
                    # never learns about a new field -- there is no new field for the channel
                    # to check.
                    #
                    # `ontologically_usable` is recorded beside it so a bundle SAYS why the
                    # object is unaligned. Without it, a provisional object and a genuinely
                    # unalignable one are indistinguishable after the fact, which is the
                    # class of ambiguity this project keeps paying for.
                    new_obj.provisional = bool(getattr(decision, "provisional", False))
                    new_obj.ontologically_usable = not new_obj.provisional
                    new_obj.onto_aligned = bool(aligned and new_obj.onto_type
                                                and new_obj.ontologically_usable)

                    current_perception_objects.append(new_obj)
                    objects_modified = True

        for obj in current_perception_objects:
            obj.last_perception_time = perception_timestamp
            # GA-186. Every object in this list was seen in THIS perception frame, so this
            # loop is exactly the co-visibility relation: same frame_id means the two were
            # observed together. Recorded here rather than in each match branch because
            # there are four of those and a sighting missed in one of them is invisible.
            self._record_sighting(obj, perception_timestamp)
        pov_volume = getattr(self, 'latest_fov_volume', None)

        if not in_exploration:
            uncertain_deleted = False

            # FIX: scaled_pov_volume was computed here and never used — the
            # unscaled pov_volume was passed below, so pov_scale_factor had no
            # effect (invisible while it defaulted to 1.0). Now the scaled
            # volume is actually applied.
            if pov_volume:
                pov_volume = shrink_pov_volume(pov_volume, POV_SCALE_FACTOR)

            uncertain_deleted = self.delete_uncertain_objects(pov_volume)

            if uncertain_deleted:
                objects_modified = True

            # GA-297. The disappearance-removal path, wired at last -- same place as
            # delete_uncertain_objects, so the decision "does it run during exploration?"
            # inherits the existing answer: NO, tracking only. Exploration is map-building;
            # deleting during it would count misses against objects the robot has not had a
            # chance to re-look at. Guarded on a real POV volume (a None pov must not reach
            # the service -- it refuses, but the caller should not ask) and on a config
            # switch, default ON, so the instrumented run can measure the deletion rate and
            # an A/B run can turn it off. Recorded EVERY call, including deleted_count=0:
            # "the path ran and deleted nothing" is a different fact from "the path never
            # ran", and until today they were indistinguishable in every bundle.
            if pov_volume and CFG["association"].get("delete_undetected", True):
                result = self.delete_undetected_objects(
                    pov_volume, current_perception_objects,
                    bool(request.descriptions.descriptions))
                try:
                    self.decision_log.write(
                        "disappearance_removal", "<cycle>",
                        frame=getattr(self, "_current_frame_id", None),
                        deleted_count=int((result or {}).get("deleted_count", 0)),
                        deleted_labels=(result or {}).get("deleted_labels", []),
                        error=(None if result is not None else "graph_api_call_failed"))
                except Exception as exc:
                    self.object_services.log_both(
                        "warn", f"[GA-297] removal row not written: {exc}")
                if result and int(result.get("deleted_count", 0)) > 0:
                    objects_modified = True

        self.latest_bboxes.clear()

        merged_any = self.merge_duplicate_objects()
        if merged_any:
            objects_modified = True

        if objects_modified:
            publish_persistent_bboxes(self, wm, self.persistent_bbox_pub)
            publish_persistent_centroids(self, wm, self.persistent_centroids_pub)
            publish_uncertain_bboxes(self, self.uncertain_objects, self.uncertain_bboxes_pub)
            publish_uncertain_centroids(self, self.uncertain_objects, self.uncertain_centroids_pub)
            self.room_manager.update_current_room_semantics(wm.persistent_perceptions)
            save_uncertain_objects(self)
            self.update_spatial_relations()
            save_persistent_perceptions(self.object_services)
        
        facts = self.publish_kb_facts(current_perception_objects)
        relation_facts = self.publish_kb_relation_facts()
        print("RELATION_FACTS =", relation_facts)

        for fact in facts + relation_facts:
            msg = String()
            msg.data = fact
            self.kb_add_pub.publish(msg)

        response.status = "tracking_activated" if tracking_activated else "success"
        response.num_objects = len(wm.persistent_perceptions)
        response.tracking_mode_activated = tracking_activated

        # GA-190: one row per cycle, after every detection in this callback has scanned.
        # Flushed here rather than inside check_tracking_transition so the unit is the CYCLE,
        # matching `not_offered_summary` on the merge path.
        self.flush_scan_summary(frame_id=getattr(self, "_current_frame_id", None))

        # W6: one row per cycle with the RUN-CUMULATIVE split of description statuses, so
        # the "unknown" population decomposes into call_failed / parse_failed /
        # model_abstained / unanswered from the same log that counts everything else.
        # Cumulative, not per-cycle: deltas between consecutive rows give the cycle's own.
        counts = getattr(self, "_vlm_status_counts", None)
        if counts:
            try:
                self.decision_log.write(
                    "vlm_description_status", "<cycle>",
                    frame=getattr(self, "_current_frame_id", None),
                    **dict(sorted(counts.items())))
            except Exception as exc:
                self.object_services.log_both(
                    "warn", f"[VLM-STATUS] tally row not written: {exc}")

        return response

    def _graph_api_url(self, path):
        return f"{self.graph_api_base_url}{path}"

    def _serialize_embedding(self, embedding):
        if embedding is None:
            return []
        if hasattr(embedding, 'astype'):
            return embedding.astype(np.float32).tolist()
        if hasattr(embedding, 'tolist'):
            return embedding.tolist()
        return [float(x) for x in embedding]

    def _call_graph_api(self, method, path, json_body=None, expected_status=None):
        url = self._graph_api_url(path)
        try:
            response = requests.request(method=method, url=url, json=json_body, timeout=self.graph_api_timeout)
        except requests.RequestException as e:
            raise RuntimeError(f"Graph API non raggiungibile ({method} {url}): {e}")

        if expected_status is None:
            ok = 200 <= response.status_code < 300
        elif isinstance(expected_status, (list, tuple, set)):
            ok = response.status_code in expected_status
        else:
            ok = response.status_code == expected_status

        if not ok:
            detail = response.text
            try:
                payload = response.json()
                if isinstance(payload, dict):
                    detail = payload.get('detail') or payload.get('message') or payload
            except Exception:
                pass
            raise RuntimeError(f"Graph API error {response.status_code} on {method} {path}: {detail}")

        try:
            return response.json()
        except Exception:
            return {}

    def add_new_object(self, label, bbox, description, color, material,
                       description_embedding=None, in_exploration=False, room_id=None):

        serialized_embedding=self._serialize_embedding(description_embedding)
        assigned_room = str(room_id) if room_id is not None else self.room_manager.current_room_id
        self.room_manager.init_room_node(assigned_room)
        payload = {
            "label": label,
            "description": description,
            "color": color,
            "material": material,
            "room_id": assigned_room,
            "x_min": bbox["x_min"],
            "x_max": bbox["x_max"],
            "y_min": bbox["y_min"],
            "y_max": bbox["y_max"],
            "z_min": bbox["z_min"],
            "z_max": bbox["z_max"],
            **({"yaw": bbox["yaw"], "oriented_center": list(bbox["oriented_center"]),
                "oriented_extents": list(bbox["oriented_extents"])}
               if bbox.get("oriented_extents") and "yaw" in bbox else {}),   # GA-312
            "in_exploration": in_exploration,
            "description_embedding": serialized_embedding,
        }

        try:
            result = self._call_graph_api("POST", "/objects", json_body=payload)
        except RuntimeError as e:
            self.get_logger().error(f"Add object failed via Graph API: {e}")
            return None

        object_id = result.get("object_id", label)

        for obj in reversed(wm.persistent_perceptions):
            if getattr(obj, "object_id", None) == object_id or (obj.label == label and obj.bbox == bbox):
                return obj

        self.get_logger().warn(f"Object {label} created via the API but not found in memory")
        return None

    def modify_existing_object(self, best_match, bbox, description_embedding=None):
        payload = {
            "description": best_match.description,
            "color": best_match.color,
            "material": best_match.material,
            "x_min": bbox["x_min"],
            "x_max": bbox["x_max"],
            "y_min": bbox["y_min"],
            "y_max": bbox["y_max"],
            "z_min": bbox["z_min"],
            "z_max": bbox["z_max"],
            **({"yaw": bbox["yaw"], "oriented_center": list(bbox["oriented_center"]),
                "oriented_extents": list(bbox["oriented_extents"])}
               if bbox.get("oriented_extents") and "yaw" in bbox else {}),   # GA-312
            "description_embedding": self._serialize_embedding(description_embedding),
        }

        response = UpdateObject.Response()
        try:
            stable_id = getattr(best_match, "object_id", None) or best_match.label
            encoded_id = urllib.parse.quote(stable_id, safe='')
            result = self._call_graph_api("PATCH", f"/objects/{encoded_id}", json_body=payload)
            response.success = bool(result.get("success", False))
            response.message = str(result.get("message", ""))
            response.object_id = str(result.get("object_id", stable_id))
            response.distance = float(result.get("distance", 0.0))
            response.iou = float(result.get("iou", 0.0))
            response.replaced = bool(result.get("replaced", False))
        except RuntimeError as e:
            response.success = False
            response.message = str(e)
            response.object_id = getattr(best_match, "object_id", None) or best_match.label
            response.distance = 0.0
            response.iou = 0.0
            response.replaced = False
            self.get_logger().error(f"Update object failed via Graph API: {e}")
        return response

    def merge_duplicate_objects(self):
        payload = {
            # GA-06: from config, and strictly stricter than the match gate. These were
            # 0.8 and 0.75 against a sim_threshold of 0.85, so merge fused pairs the
            # association loop had just refused. object_services asserts the ordering at
            # load; these are read from the same block.
            "max_distance": CFG["association"].get("merge_max_distance_m", 0.8),
            "min_similarity": CFG["association"].get(
                "merge_min_similarity", SIM_THRESHOLD + (1.0 - SIM_THRESHOLD) / 2.0),
            "dry_run": False,
        }

        try:
            result = self._call_graph_api("POST", "/merge", json_body=payload)
            if result.get("pending"):
                # GA-183: HTTP 202 -- the bridge dispatched the request and stopped waiting.
                # The merge may still land; the next cycle re-reads the world model.
                self.get_logger().warn(f"Merge dispatched, not confirmed: {result.get('message')}")
            merged = int(result.get("merged_count", 0)) > 0
            if merged:
                # GA-11. A merge changes the survivor more than anything else does; it never
                # triggered a second look. The service already reports each pair's keeper.
                try:
                    # The bridge returns the parsed list as "merge_log"; "merge_log_json" is the
                    # ROS field name and never reached this dict, so no survivor was ever queued.
                    for pair in result.get("merge_log") or json.loads(result.get("merge_log_json") or "[]"):
                        kid = (pair.get("keeper") or {}).get("object_id") or pair.get("keeper_id")
                        if kid:
                            self._note_update(kid, reason="merged")
                except (TypeError, ValueError) as exc:
                    self.get_logger().warn(f"GA-11: merge log unreadable, survivors not queued: {exc}")
            return merged
        except RuntimeError as e:
            self.get_logger().error(f"Merge objects failed via Graph API: {e}")
            return False

    def delete_undetected_objects(self, pov_volume, current_perception_objects, description_received):
        """GA-297. Client half of the disappearance-removal path.

        -> the bridge's result dict ({deleted_count, deleted_labels, ...}) so the caller
        can MEASURE the deletion rate, or None when the call failed. Was `> 0` bool --
        and had no caller at all, which is why the map has been accumulation-only for
        confirmed objects: the service half (object_services
        _cb_delete_unseen_objects, miss counter toward MAX_MISSES_BEFORE_DELETE) has
        been complete and dormant since it was written.
        """
        payload = {
            "pov_volume_flat": [
                pov_volume['x_min'], pov_volume['x_max'],
                pov_volume['y_min'], pov_volume['y_max'],
                pov_volume['z_min'], pov_volume['z_max'],
            ] if pov_volume else [],
            # GA-26: identity, not label -- seeing one chair must not clear every chair's
            # tally. The srv field keeps its name; it carries the object_id when there is one.
            "current_labels": [getattr(o, "object_id", None) or o.label for o in current_perception_objects],
            "check_uncertain": False,
        }

        try:
            return self._call_graph_api("POST", "/delete_objects", json_body=payload)
        except RuntimeError as e:
            self.get_logger().error(f"Delete undetected objects failed via Graph API: {e}")
            return None

    def delete_uncertain_objects(self, pov_volume):
        payload = {
            "pov_volume_flat": [
                pov_volume['x_min'], pov_volume['x_max'],
                pov_volume['y_min'], pov_volume['y_max'],
                pov_volume['z_min'], pov_volume['z_max'],
            ] if pov_volume else [],
            "current_labels": [],
            "check_uncertain": True,
        }

        try:
            result = self._call_graph_api("POST", "/delete_objects", json_body=payload)
            return int(result.get("uncertain_removed_count", 0)) > 0
        except RuntimeError as e:
            self.get_logger().error(f"Delete uncertain objects failed via Graph API: {e}")
            return False
    
    def _cleanup_uncertain_by_time(self):
        now = time.time()
        expiry = 120.0 # secondi
        to_remove = [obj for obj in self.uncertain_objects
                     if now - (now if getattr(obj, 'creation_time', None) is None else obj.creation_time) > expiry]  # GA-12: unstamped = not expired
        if to_remove:
            for obj in to_remove:
                self.uncertain_objects.remove(obj)
                self.object_services.log_both('info', f"[UNCERTAIN CLEANUP] Removed '{obj.label}' (expired)")
    
    def _descriptions_callback(self, msg):
        self._n_desc_msgs += 1
        stamp = getattr(getattr(msg, "header", None), "stamp", None)
        if stamp is None:
            self._n_desc_no_stamp += 1
            self.object_services.log_both(
                'warn', f"[SYNC] /object_descriptions message with no usable stamp "
                        f"({self._n_desc_no_stamp} of {self._n_desc_msgs})")
            return
        self._pending_descriptions[_stamp_key(stamp)] = msg
        while len(self._pending_descriptions) > SYNC_BUFFER_LIMIT:
            self._pending_descriptions.pop(next(iter(self._pending_descriptions)))
        self._try_process()

    def _check_input_silence(self):
        """No detections for a while: say so once, then END THE RUN if it persists.

        GA-83 shipped the first half of this and deliberately stopped there, on the
        reasoning that "this node cannot tell a dead producer from a robot standing
        still, and guessing would be the fallback-as-fact shape". GA-94 retires that
        reasoning on evidence, in two steps.

        FIRST, THE PREMISE WAS FALSE. `perception_2._publish_bbox_array` ends in an
        UNCONDITIONAL `self.bbox_pub.publish(msg)` -- a cycle that detects nothing still
        publishes an empty Bbox3dArray. So a stationary robot staring at a blank wall
        still feeds this topic every cycle. Silence on /bbox_3d does NOT mean "nothing to
        see"; it means the producer stopped. The distinction GA-83 refused to guess at is
        one the code already makes.

        SECOND, THE SILENCE WAS NOT READ BY ANYONE. In run A this node printed exactly
        the line GA-83 designed, at 17:53:35, and then idled for 36 more minutes on a run
        whose rtabmap had already been killed. A fact stated to a log nobody is reading
        is not a safeguard; it is a record of how long the waste went on.

        So: report once as before, then exit non-zero after INPUT_SILENCE_MAX_STRIKES
        consecutive silent checks, with the announcement as the last line.
        """
        if self._last_bbox_at is None:
            return
        silent_for = time.time() - self._last_bbox_at
        if silent_for < INPUT_SILENCE_TIMEOUT:
            self._input_silence_reported = False
            self._input_silence_strikes = 0
            return
        # GA-94b. THE ORIGINAL PREMISE IS NO LONGER TRUE. This watchdog was written when a
        # silent /bbox_3d meant a broken producer. Detection is gated on the robot STOPPING,
        # and with dwell=0 the robot barely stops -- run 042828 had four cycles and three
        # stop events in nine minutes, so a 118-second gap between incidental halts is
        # NORMAL, not a dead producer. The guard fired correctly on a premise that had
        # changed underneath it, and ended a healthy run: perception's last line at that
        # moment was "CameraInfo received", no traceback, zero stale rejects.
        #
        # So silence is measured in the unit that actually produces input: STOPS. If the
        # robot has stopped MIN_STOPS_BEFORE_DEAD times since the last detection arrived,
        # the producer had its chance and did not take it -- that is a dead producer at any
        # dwell. Time alone is not, any more.
        stops = getattr(self, "_stops_since_input", 0)
        if stops < INPUT_SILENCE_MIN_STOPS:
            if not self._input_silence_reported:
                self._input_silence_reported = True
                self.object_services.log_both(
                    'warn',
                    f"[INPUT] no /bbox_3d for {silent_for:.0f}s, but only {stops} robot stop(s) "
                    f"since the last one (need {INPUT_SILENCE_MIN_STOPS} to call it dead). "
                    f"Detection is gated on stopping; with a low dwell this is expected.")
            return
        self._input_silence_strikes += 1
        if not self._input_silence_reported:
            self._input_silence_reported = True
            self.object_services.log_both(
                'warn',
                f"[INPUT] no /bbox_3d for {silent_for:.0f}s "
                f"({self._n_bbox_msgs} received in total, {len(wm.persistent_perceptions)} objects "
                f"in the map). The producer may have stopped; this node has nothing to process.")
        if self._input_silence_strikes >= INPUT_SILENCE_MAX_STRIKES:
            self.object_services.log_both(
                'error',
                f"[INPUT] ENDING THE RUN: no /bbox_3d for {silent_for:.0f}s across "
                f"{self._input_silence_strikes} consecutive checks "
                f"({self._n_bbox_msgs} received in total, {len(wm.persistent_perceptions)} objects "
                f"in the map). The producer is gone and this node cannot make progress.")
            try:
                if hasattr(self, "room_manager"):
                    self.room_manager.finalize_current_room(wm.persistent_perceptions)
            except Exception as exc:
                self.object_services.log_both('error', f"[INPUT] room finalize failed on exit: {exc}")
            for h in list(logging.getLogger().handlers):
                try:
                    h.flush()
                except Exception:
                    pass
            sys.stdout.flush()
            sys.stderr.flush()
            # os._exit, not sys.exit: this runs on a MultiThreadedExecutor WORKER thread,
            # where SystemExit unwinds that thread only and leaves the process spinning --
            # which is the exact failure being fixed. The log is flushed above first.
            os._exit(1)

    def _bboxes_callback(self, msg):
        self._n_bbox_msgs += 1
        self._last_bbox_at = time.time()
        self._input_silence_reported = False
        self._stops_since_input = 0          # GA-94b: input arrived; the stop count restarts
        stamp = getattr(getattr(msg, "header", None), "stamp", None)
        if stamp is None:
            self._n_bbox_no_stamp += 1
            self.object_services.log_both(
                'warn', f"[SYNC] /bbox_3d message with no usable stamp "
                        f"({self._n_bbox_no_stamp} of {self._n_bbox_msgs})")
            return
        self._pending_bboxes[_stamp_key(stamp)] = msg
        while len(self._pending_bboxes) > SYNC_BUFFER_LIMIT:
            self._pending_bboxes.pop(next(iter(self._pending_bboxes)))
        self._try_process()

    def _try_process(self):
        # A pair is judged by WHEN IT WAS OBSERVED, not by what the robot is doing when it
        # arrives. This used to return early and clear BOTH buffers whenever the robot was
        # in motion -- which, given the 1.6-2.3 s perception round-trip, discarded stationary
        # observations because the robot had set off again before they came back.
        #
        # The guard's real intent is preserved: an observation whose own stamp falls inside
        # the current motion was taken while moving and is still refused.
        common_keys = sorted(set(self._pending_descriptions) & set(self._pending_bboxes))
        if not common_keys:
            # Say WHY there is no pair: one side empty, or two non-empty sides that do not
            # intersect. Those are different faults and silence conflates them.
            #
            # Once per change of state, not once per callback. Logging every arrival is
            # what produced a 1.7 GB om6.log -- a diagnostic nobody can open is not a
            # diagnostic, and the counters below carry the same information in one line.
            state = (len(self._pending_descriptions), len(self._pending_bboxes))
            if state != self._last_nopair_state:
                self._last_nopair_state = state
                self.object_services.log_both(
                    'warn',
                    f"[SYNC] no pair: desc={sorted(self._pending_descriptions)[-3:]} "
                    f"({state[0]} buffered, {self._n_desc_msgs} received) "
                    f"bbox={sorted(self._pending_bboxes)[-3:]} "
                    f"({state[1]} buffered, {self._n_bbox_msgs} received)")
            return
        self._last_nopair_state = None

        for stamp_key in common_keys:
            bboxes_msg = self._pending_bboxes[stamp_key]
            if self._moving_since is not None:
                observed_at = stamp_key[0] + stamp_key[1] * 1e-9
                if observed_at >= self._moving_since:
                    # Taken during this motion, so it can never become valid -- the stamp
                    # does not change when the robot stops. Drop it once, here, rather than
                    # leaving it to be re-examined and re-logged on every later call.
                    self._pending_descriptions.pop(stamp_key, None)
                    self._pending_bboxes.pop(stamp_key, None)
                    self._dropped_moving_pairs += 1
                    self.object_services.log_both(
                        'warn',
                        f"[SYNC] pair {_stamp_key_str(bboxes_msg.header.stamp)} observed during "
                        f"motion -- discarded (total {self._dropped_moving_pairs})")
                    continue

            descriptions = self._pending_descriptions.pop(stamp_key, None)
            bboxes = self._pending_bboxes.pop(stamp_key, None)
            if descriptions is None or bboxes is None:
                continue

            self.object_services.log_both(
                'debug',
                f"[SYNC] Processing matched perception stamp {_stamp_key_str(descriptions.header.stamp)}"
            )

            request = ObjectTrackingService.Request()
            request.descriptions = descriptions
            request.bboxes = bboxes

            response = ObjectTrackingService.Response()
            self.object_tracking_callback(request, response)

    def walls_callback(self, msg):
        # The `except Exception` that stood here printed to stdout and continued, so a
        # malformed or renamed wall message left current_room_walls silently empty and
        # the run looked clean. wall_detector was rewritten for depth and this is its
        # FIRST live flight: a swallowed exception here is precisely what would make a
        # broken first flight indistinguishable from a room with no walls in it.
        #
        # No handler. json.loads and the "start"/"end"/"x"/"y" lookups now raise, and a
        # raise in a subscription callback is visible in the node's log with a traceback
        # naming the field that was wrong. Rule 14: a missing component crashes.
        new_walls = json.loads(msg.data)
        self.room_manager.init_room_node(self.room_manager.current_room_id)
        for w in new_walls:
            start_x, start_y = w["start"]["x"], w["start"]["y"]
            end_x, end_y = w["end"]["x"], w["end"]["y"]
            self.room_manager.current_room_walls.append([start_x, start_y, end_x, end_y])

    # --- re-evaluation seam (hooks.Reevaluation / hooks.Refiner) ---
    @staticmethod
    def _node_dict(obj):
        return {"object_id": getattr(obj, "object_id", None) or obj.label, "label": obj.label,
                "color": obj.color, "material": obj.material, "description": obj.description,
                "bbox": obj.bbox, "room_id": getattr(obj, "room_id", None)}

    def _find_node(self, object_id):
        return next((o for o in wm.snapshot()
                     if (getattr(o, "object_id", None) or o.label) == object_id), None)

    def _neighbours(self, node, radius):
        return [o for o in wm.snapshot()
                if o is not node and o.bbox is not None and node.bbox is not None
                and bbox_center_distance(node.bbox, o.bbox) <= radius]

    def _note_update(self, object_id, reason="updated", now=None):
        """A node changed: it and its spatial neighbours (within the association gate, 2 m
        when the gate is off) deserve a second look. The trigger policy — which neighbours,
        graph-distance instead of metres — is the queue subclass's to refine.

        GA-11, the three missing triggers: this is now also called after a MERGE (for each
        surviving object), after an IN-PLACE BOX WRITE, and after a ROOM CHANGE. `reason` is
        recorded so the drain log can say which event caused the second look.

        Debounce and fan-out bound, both config knobs with stated costs (config.py, GA-11):
        an object re-queued inside `reevaluation_debounce_s` of its last queueing is skipped,
        and at most `reevaluation_max_fanout` neighbours are queued per event, so one churning
        object cannot flood the queue and one update cannot re-examine a whole room.
        """
        import time as _t
        now = _t.monotonic() if now is None else now
        last = self._reeval_last.get(object_id)
        if last is not None and now - last < REEVALUATION_DEBOUNCE_S:
            self._reeval_debounced += 1
            return
        node = self._find_node(object_id)
        if node is None:
            return
        self._reeval_last[object_id] = now
        radius = REEVALUATION_RADIUS
        # GA-11: queue the node that CHANGED, not only its neighbours. `on_update` adds
        # the neighbour ids and never the subject -- hooks.py's own self-test pins that
        # exactly: after `on_update("a", ["b","c"])` the queue holds b and c, never a.
        # So every update re-examined the neighbourhood of a node and never the node
        # itself, which is the one thing known to have new evidence.
        #
        # Fixed HERE and deliberately NOT in hooks.py. hooks.py ships the generic
        # blueprint that FOUND extends; widening on_update's contract would change it
        # for every subclass and break the blueprint's self-test. WHICH nodes deserve a
        # second look is the caller's trigger policy, which is what this method is.
        self.reeval.mark(object_id, reason)
        neighbours = [self._node_dict(o)["object_id"] for o in self._neighbours(node, radius)]
        if len(neighbours) > REEVALUATION_MAX_FANOUT:
            self._reeval_fanout_capped += len(neighbours) - REEVALUATION_MAX_FANOUT
            neighbours = neighbours[:REEVALUATION_MAX_FANOUT]
        self.reeval.on_update(object_id, neighbours)

    def _drain_reevaluations(self):
        """Periodic: hand every queued node, with its neighbours, to the Refiner and log
        what it proposes. Proposals are recorded, not applied — applying one is an
        UpdateObject with the revised fields, wired once the policy is settled."""
        for object_id, reason in self.reeval.drain():
            node = self._find_node(object_id)
            if node is None:
                continue
            neighbours = [self._node_dict(o) for o in self._neighbours(node, REEVALUATION_RADIUS)]
            try:
                revision = self.refiner_hook.refine(self._node_dict(node), neighbours)
            except Exception as exc:
                self.get_logger().warn(f"[{self.refiner_hook.name}] refine failed for {object_id}: {exc}")
                continue
            if revision:
                self.decision_log.write("revision", object_id, refiner=self.refiner_hook.name, reason=reason, revision=revision)

    @synchronized_world_model
    def periodic_bbox_publisher(self):
        self._drain_reevaluations()
        if self.robot_has_moved:
           return
        if len(wm.persistent_perceptions) > 0:
            publish_persistent_bboxes(self, wm, self.persistent_bbox_pub)
            publish_persistent_centroids(self, wm, self.persistent_centroids_pub)
        if len(self.uncertain_objects) > 0:
            publish_uncertain_bboxes(self, self.uncertain_objects, self.uncertain_bboxes_pub)
            publish_uncertain_centroids(self, self.uncertain_objects, self.uncertain_centroids_pub)
        msg = MarkerArray()
        from geometry_msgs.msg import Point
        for i, data in enumerate(self.room_manager.scene_graph.values()):
            poly = data.get("polygon", [])
            if len(poly) >= 3:
                m = Marker()
                m.header.frame_id = "map"
                m.header.stamp = self.get_clock().now().to_msg()
                m.id = i
                m.type = Marker.LINE_STRIP
                m.action = Marker.ADD
                m.pose.orientation.w = 1.0
                m.scale.x = 0.15
                m.color.r, m.color.g, m.color.b, m.color.a = 0.0, 1.0, 0.5, 1.0
                for pt in poly:
                    p = Point()
                    p.x, p.y, p.z = float(pt[0]), float(pt[1]), 0.05
                    m.points.append(p)
                p = Point()
                p.x, p.y, p.z = float(poly[0][0]), float(poly[0][1]), 0.05
                m.points.append(p)
                msg.markers.append(m)
        #if hasattr(self, 'room_area_pub'):
        #    self.room_area_pub.publish(msg)

from rclpy.executors import MultiThreadedExecutor  # noqa: E402, I001  (late on purpose: used by main() only, and the module must import under rosstub on the host)


def main(args=None):
    rclpy.init(args=args)
    bridge_process = None
    if GRAPH_API_AUTOSTART and GRAPH_API_BASE_URL.startswith(("http://127.0.0.1", "http://localhost")):
        if not _graph_api_is_running(GRAPH_API_BASE_URL):
            bridge_process = _launch_graph_api_bridge()
    service_node = ObjectManagerService()

    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(service_node)
    executor.add_node(service_node.object_services)

    try:
        executor.spin()
    except KeyboardInterrupt:
        from datetime import datetime
        print(f"\nOBJECT MANAGER chiuso ({datetime.now().strftime('%Y-%m-%d %H:%M:%S')})")
        print("Saving the last room...")

        if hasattr(service_node, 'room_manager'):
            service_node.room_manager.finalize_current_room(wm.persistent_perceptions)

    finally:
        executor.shutdown()
        if bridge_process is not None and bridge_process.poll() is None:
            bridge_process.terminate()
        service_node.object_services.destroy_node()
        service_node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
