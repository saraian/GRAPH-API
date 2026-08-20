#!/usr/bin/env python3
"""
Object Manager Service - Semantic and Spatial Tracking of Perceived Objects
Tracks objects and automatically transitions from EXPLORATION to TRACKING when
an object is seen again in a different position.
Includes Topological Semantic Mapping (Room Manager) with Scene Graph generation.

Room changes are handled by the Room Manager using detected wall segments and
its normal scene-evaluation logic.
"""
import rclpy, json, os, time, threading, re, subprocess, sys
import urllib.parse
from collections import deque
from rclpy.node import Node
from rclpy.duration import Duration
from rclpy.qos import QoSProfile, DurabilityPolicy
import numpy as np
import requests
from openai import OpenAI
from lost3dsg.msg import ObjectDescriptionArray, Bbox3dArray
from lost3dsg.srv import (
    ObjectTrackingService,
    AddObject, RemoveObject, UpdateObject, MergeObjects, DeleteObjects, QueryObjects,
)
import uuid
from object_services import (
    ObjectServices,
    save_persistent_perceptions,
    ensure_relations,
    infer_spatial_relations,
)
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import Bool
from sensor_msgs.msg import PointCloud2
from object_info import Object
from world_model import wm
import gensim.downloader as api
from utils import *
from room_manager import RoomManager
from nlp_utils import *
from datetime import datetime
from cv_utils import *
from map_database import MapDatabase
from gensim.models import KeyedVectors
from std_msgs.msg import String
from tf2_ros import Buffer, TransformListener
import json
import hashlib
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path
from builtin_interfaces.msg import Time as TimeMsg
from datetime import timezone

# ============= EXPLORATION PARAMETERS =============
EXPLORATION_IOU_THRESHOLD = 0.10
SIM_THRESHOLD = 0.85
TRACKING_IOU_THRESHOLD = 0.3
VOLUME_EXPANSION_RATIO = 0.01
EXPLORATION_FRAME_LIMIT = 10 # Numero di frame in exploration prima di passare a tracking
OBJECT_STABILITY_TIMEOUT = 3.0 # Secondi minimi di vita prima di poter essere 'MOVED'
POV_SCALE_FACTOR = 1.0
MAX_VOLUME_THRESHOLD = 0.5
BBOX_REDUCTION_RATIO = 0.30

# Load OpenAI API key & Paths
file_path = os.path.abspath(__file__)
current_dir = os.path.dirname(file_path)
PROJECT_ROOT = current_dir.split('/install/')[0] if '/install/' in current_dir else os.path.abspath(os.path.join(current_dir, "../.."))

world2vec = KeyedVectors.load_word2vec_format(
    '/root/gensim-data/word2vec-google-news-300/word2vec-google-news-300.gz',
    binary=True,
    limit=200000 # <--- ECCO LA MAGIA CHE SALVA LA RAM!
)

# Setup path per il file sintetico di operazioni
log_dir = os.path.join(PROJECT_ROOT, "output")
os.makedirs(log_dir, exist_ok=True)
SYNTHETIC_LOG_FILE = os.path.join(log_dir, "operations.txt")
AGENT_POSES_LOG_FILE = os.path.join(log_dir, "agent_poses.json")
GRAPH_API_BASE_URL = os.environ.get("GRAPH_API_BASE_URL", "http://127.0.0.1:8080")
GRAPH_API_TIMEOUT = float(os.environ.get("GRAPH_API_TIMEOUT", "10.0"))
SYNC_BUFFER_LIMIT = 20

def _launch_graph_api_bridge():
    bridge_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "graph_api_bridge.py")
    if not os.path.exists(bridge_path):
        print(f"[WARN] graph_api_bridge.py non trovato: {bridge_path}")
        return None

    proc = subprocess.Popen(
        [sys.executable, bridge_path],
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
            print(f"[WARN] Errore lettura log graph_api_bridge: {e}")

    threading.Thread(target=_pipe_logs, daemon=True).start()
    return proc


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

def save_uncertain_objects(node):
    """Save uncertain_objects to a text file."""
    output_dir = os.path.join(PROJECT_ROOT, "output")
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, "uncertain_objects.txt")

    with open(save_path, "w") as f:
        f.write("=" * 80 + "\n")
        f.write(f"UNCERTAIN OBJECTS - Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 80 + "\n\n")

        if not node.uncertain_objects:
            f.write("No uncertain objects at the moment.\n")
        else:
            f.write(f"Total uncertain objects: {len(node.uncertain_objects)}\n\n")

            for i, obj in enumerate(node.uncertain_objects, 1):
                f.write(f"{i}. {obj.label}\n")
                if obj.bbox:
                    x_center = (obj.bbox['x_min'] + obj.bbox['x_max']) / 2.0
                    y_center = (obj.bbox['y_min'] + obj.bbox['y_max']) / 2.0
                    z_center = (obj.bbox['z_min'] + obj.bbox['z_max']) / 2.0
                    f.write(f"   Center position: X={x_center:.3f}, Y={y_center:.3f}, Z={z_center:.3f}\n")
                    

def save_agent_poses(agent_poses):
    """Save the accumulated agent poses (with timestamp) to a JSON file."""
    try:
        with open(AGENT_POSES_LOG_FILE, "w") as f:
            json.dump(agent_poses, f, indent=2)
    except Exception as e:
        print(f"[WARN] Errore salvataggio agent_poses.json: {e}")


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
            f.write(f"NUOVO AVVIO: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"{'='*50}\n")

        self.exploration_mode = True
        self.seen_again = False
        self.latest_bboxes = {}
        self.latest_fov_volume = None
        self.uncertain_objects = []
        self.exploration_frame_counter = 0
        self.robot_has_moved = False

        self.latest_descriptions = None
        self.latest_bboxes_msg = None

        self.agent_poses = []
        self.agent_pose_history = deque(maxlen=2000)
        self.latest_agent_pose = None
        self._pending_descriptions = {}
        self._pending_bboxes = {}
        
        # --- INIT ROOM MANAGER ---
        self.room_manager = RoomManager(
            w2v_model=world2vec,
            node=self,
            cloud_map_topic='/rtabmap/cloud_map',
        )
        self.object_services = ObjectServices(self.room_manager)
        self.last_room_check_time = time.time()
        
        self.wall_sub = self.create_subscription(
            String,
            '/detected_wall_segments',
            self.walls_callback,
            10
        )
        from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

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
        
        # Avvisa Perception dei cambi di stanza
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
        self.create_subscription(Bbox3dArray, '/bbox_3d', self._bboxes_callback, qos_standard)
        self.create_subscription(PoseStamped, '/agent_camera_pose', self._agent_pose_callback, qos_standard)
        self.get_logger().info("Subscribing to /object_descriptions, /bbox_3d and /agent_camera_pose")
        self.get_logger().info(f"Node name={self.get_name()} ns={self.get_namespace()}")

        self._bbox_timer = self.create_timer(2.0, self.periodic_bbox_publisher)
        #self._uncertain_cleanup_timer = self.create_timer(5.0, self._cleanup_uncertain_by_time)
    def movement_callback(self, msg):
        # Sincronizza lo stato reale: True se si muove, False se è fermo
        self.robot_has_moved = msg.data
        if msg.data:
            self._pending_descriptions.clear()
            self._pending_bboxes.clear()
            self.latest_bboxes.clear()
            self.object_services.log_both('warn', "[MOVEMENT] Robot is moving -> Blocco stanze attivato")
        else:
            self.object_services.log_both('info', "[MOVEMENT] Robot has stopped -> Creazione stanze permessa")

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

        save_agent_poses(self.agent_poses)
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

    def check_tracking_transition(self, label_base, color, material, description_embedding, bbox):
        best_match = None
        highest_similarity = -1.0

        for obj in wm.persistent_perceptions:
            if (time.time() - getattr(obj, 'creation_time', 0)) < OBJECT_STABILITY_TIMEOUT:
                continue
            
            obj_label_base = obj.label.split('#')[0] if '#' in obj.label else obj.label

            if not hasattr(obj, "embedding") or obj.embedding is None:
                obj.embedding = get_embedding(world2vec, obj.description)
            
            if obj.embedding is None or description_embedding is None:
                continue
            
            similarity = lost_similarity(world2vec, label_base, obj_label_base, color, obj.color,
                                         material, obj.material, description_embedding, obj.embedding)
            
            if similarity > SIM_THRESHOLD and similarity > highest_similarity:
                highest_similarity = similarity
                best_match = obj

        if best_match:
            print(f"[BEST MATCH FOUND] Rilevato: '{label_base}' -> Best Memoria: '{best_match.label}' (Score: {highest_similarity:.3f})")
            
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
            
            if distance > 0.35 and iou < EXPLORATION_IOU_THRESHOLD:
                self.object_services.log_both('warn', f"[TRACKING TRANSITION] Object '{best_match.label}' is the best match but moved! (Dist: {distance:.2f}m), the iou was {iou}")
                return True, best_match, distance
        
        return False, None, 0.0

    def update_spatial_relations(self):
        for obj in wm.persistent_perceptions:
            ensure_relations(obj)
            for key in obj.relations:
                obj.relations[key].clear()

        for obj in wm.persistent_perceptions:
            ensure_relations(obj)
            room_id = getattr(obj, "room_id", None)

        for i, obj_a in enumerate(wm.persistent_perceptions):
            for j, obj_b in enumerate(wm.persistent_perceptions):
                if i == j:
                    continue
                for _, pred, target in infer_spatial_relations(obj_a, obj_b):
                    obj_a.relations[pred].add(target)

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
            self.object_services.log_both('warn', "Robot in movimento — dati scartati da object_tracking_callback")
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
                    self.object_services.log_both('info', f"Cambio stanza rilevato! Inviato segnale a Perception per: {self.room_manager.current_room_id}")

            self.last_room_check_time = current_time

        for description in request.descriptions.descriptions:
            label = description.label
            label_base = label.split('#')[0] if '#' in label else label
            color = description.color
            material = description.material
            description_text = description.description

            description_embedding = get_embedding(world2vec, description_text)

            old_key = create_object_key(label, "", "", "")
            if old_key not in self.latest_bboxes:
                continue

            bbox = self.latest_bboxes[old_key]["bbox"]
            new_key = create_object_key(label, material, color, description_text)

            del self.latest_bboxes[old_key]
            self.latest_bboxes[new_key] = {
                "bbox": bbox, "label": label,
                "color": color, "material": material, "description": description_text
            }

            already_seen = False

            if in_exploration:
                transition, obj, distance = self.check_tracking_transition(
                    label_base, color, material, description_embedding, bbox
                )

                if transition:
                    self.object_services.log_both('warn', f"[TRANSITION] Switching from EXPLORATION to TRACKING mode")
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
                            (o for o in wm.persistent_perceptions
                             if getattr(o, "object_id", None) == update_response.object_id),
                            obj,
                        )
                        current_perception_objects.append(matching_obj)
                        objects_modified = True
                    else:
                        self.object_services.log_both('warn', f"Update fallito per {obj.label}: {update_response.message}")
                    already_seen = True
                    continue

                for obj in wm.persistent_perceptions:
                    obj_label_base = obj.label.split('#')[0] if '#' in obj.label else obj.label
                    if not hasattr(obj, "embedding"):
                        obj.embedding = get_embedding(world2vec, obj.description)
                    if obj.embedding is None:
                        continue

                    similarity = lost_similarity(
                        world2vec, label_base, obj_label_base, color, obj.color,
                        material, obj.material, description_embedding, obj.embedding
                    )

                    if obj.bbox is not None:
                        iou = compute_iou_3d(bbox, obj.bbox)

                        current_is_unknown = (description_text.lower() == "unknown")
                        stored_is_unknown = (obj.description.lower() == "unknown")

                        if (current_is_unknown or stored_is_unknown) and iou >= EXPLORATION_IOU_THRESHOLD:
                            already_seen = True
                            current_perception_objects.append(obj)
                            obj.bbox = bbox
                            if stored_is_unknown and not current_is_unknown:
                                obj.description = description_text
                                obj.color = color
                                obj.material = material
                                obj.embedding = description_embedding
                            objects_modified = True
                            break

                        if similarity > SIM_THRESHOLD and iou >= EXPLORATION_IOU_THRESHOLD:
                            already_seen = True
                            current_perception_objects.append(obj)
                            obj.bbox = bbox
                            objects_modified = True
                            break

            else:
                best_match = None
                best_score = 0

                for obj in wm.persistent_perceptions:
                    if not hasattr(obj, "embedding"):
                        obj.embedding = get_embedding(world2vec, obj.description)
                    if obj.embedding is None:
                        continue

                    obj_label_base = obj.label.split('#')[0] if '#' in obj.label else obj.label
                    similarity = lost_similarity(
                        world2vec, label_base, obj_label_base, color, obj.color,
                        material, obj.material, description_embedding, obj.embedding
                    )

                    if description_text.lower() == "unknown" or obj.description.lower() == "unknown":
                        if obj.bbox is not None:
                            iou = compute_iou_3d(bbox, obj.bbox)
                            if iou >= TRACKING_IOU_THRESHOLD:
                                similarity = 1.0

                    if similarity > SIM_THRESHOLD and similarity > best_score:
                        best_score = similarity
                        best_match = obj

                if best_match:
                    already_seen = True
                    update_response = self.modify_existing_object(best_match, bbox, description_embedding)
                    if update_response.success:
                        matching_obj = next(
                            (o for o in wm.persistent_perceptions
                             if getattr(o, "object_id", None) == update_response.object_id),
                            best_match,
                        )
                        current_perception_objects.append(matching_obj)
                        objects_modified = True
                    else:
                        self.object_services.log_both('warn', f"Update fallito per {best_match.label}: {update_response.message}")

            if not already_seen:
                new_obj = self.add_new_object(
                    label, bbox, description_text, color, material,
                    description_embedding, in_exploration,
                    self.room_manager.assign_room_by_geometry(bbox)
                )
                if new_obj is not None:
                    current_perception_objects.append(new_obj)
                    objects_modified = True

        for obj in current_perception_objects:
            obj.last_perception_time = perception_timestamp
        pov_volume = getattr(self, 'latest_fov_volume', None)

        if not in_exploration:
            uncertain_deleted = False

            if pov_volume:
                scaled_pov_volume = shrink_pov_volume(pov_volume, POV_SCALE_FACTOR)

            uncertain_deleted = self.delete_uncertain_objects(pov_volume)

            if uncertain_deleted:
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

        self.get_logger().warn(f"Oggetto {label} creato via API ma non ritrovato in memoria")
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
            "max_distance": 0.8,
            "min_similarity": 0.75,
            "dry_run": False,
        }

        try:
            result = self._call_graph_api("POST", "/merge", json_body=payload)
            return int(result.get("merged_count", 0)) > 0
        except RuntimeError as e:
            self.get_logger().error(f"Merge objects failed via Graph API: {e}")
            return False

    def delete_undetected_objects(self, pov_volume, current_perception_objects, description_received):
        payload = {
            "pov_volume_flat": [
                pov_volume['x_min'], pov_volume['x_max'],
                pov_volume['y_min'], pov_volume['y_max'],
                pov_volume['z_min'], pov_volume['z_max'],
            ] if pov_volume else [],
            "current_labels": [o.label for o in current_perception_objects],
            "check_uncertain": False,
        }

        try:
            result = self._call_graph_api("POST", "/delete_objects", json_body=payload)
            return int(result.get("deleted_count", 0)) > 0
        except RuntimeError as e:
            self.get_logger().error(f"Delete undetected objects failed via Graph API: {e}")
            return False

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
                     if now - getattr(obj, 'creation_time', 0) > expiry]
        if to_remove:
            for obj in to_remove:
                self.uncertain_objects.remove(obj)
                self.object_services.log_both('info', f"[UNCERTAIN CLEANUP] Rimosso '{obj.label}' (tempo scaduto)")
    
    def _descriptions_callback(self, msg):
        stamp = getattr(getattr(msg, "header", None), "stamp", None)
        if stamp is None:
            return
        self._pending_descriptions[_stamp_key(stamp)] = msg
        while len(self._pending_descriptions) > SYNC_BUFFER_LIMIT:
            self._pending_descriptions.pop(next(iter(self._pending_descriptions)))
        self._try_process()

    def _bboxes_callback(self, msg):
        stamp = getattr(getattr(msg, "header", None), "stamp", None)
        if stamp is None:
            return
        self._pending_bboxes[_stamp_key(stamp)] = msg
        while len(self._pending_bboxes) > SYNC_BUFFER_LIMIT:
            self._pending_bboxes.pop(next(iter(self._pending_bboxes)))
        self._try_process()

    def _try_process(self):
        if self.robot_has_moved:
            self._pending_descriptions.clear()
            self._pending_bboxes.clear()
            return

        common_keys = sorted(set(self._pending_descriptions) & set(self._pending_bboxes))
        if not common_keys:
            return

        for stamp_key in common_keys:
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
        try:
            new_walls = json.loads(msg.data)
            self.room_manager.init_room_node(self.room_manager.current_room_id)
            for w in new_walls:
                start_x, start_y = w["start"]["x"], w["start"]["y"]
                end_x, end_y = w["end"]["x"], w["end"]["y"]
                self.room_manager.current_room_walls.append([start_x, start_y, end_x, end_y])
        except Exception as e:
            print(f"[ERROR] Errore nel salvataggio muri: {e}")

    def periodic_bbox_publisher(self):
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

from rclpy.executors import MultiThreadedExecutor

def main(args=None):
    rclpy.init(args=args)
    service_node = ObjectManagerService()

    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(service_node)
    executor.add_node(service_node.object_services)

    try:
        executor.spin()
    except KeyboardInterrupt:
        from datetime import datetime
        print(f"\nOBJECT MANAGER chiuso ({datetime.now().strftime('%Y-%m-%d %H:%M:%S')})")
        print("Salvataggio dell'ultima stanza in corso...")

        if hasattr(service_node, 'room_manager'):
            service_node.room_manager.finalize_current_room(wm.persistent_perceptions)

    finally:
        executor.shutdown()
        service_node.object_services.destroy_node()
        service_node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
