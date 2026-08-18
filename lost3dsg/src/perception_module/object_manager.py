#!/usr/bin/env python3
"""
Object Manager - Semantic and Spatial Tracking of Perceived Objects

"""
import rclpy, json, os, time, logging, threading
from rclpy.node import Node
from rclpy.duration import Duration
from rclpy.qos import QoSProfile, DurabilityPolicy
import numpy as np
from openai import OpenAI
from lost3dsg.msg import ObjectDescriptionArray, Bbox3dArray
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import Bool
from object_info import Object
from debug_utils import TrackingLogger
from world_model import wm
import gensim.downloader as api
from utils import *
from nlp_utils import *
from datetime import datetime
from cv_utils import *
from map_database import MapDatabase

# =============  EXPLORATION PARAMETERS =============
EXPLORATION_IOU_THRESHOLD = 0.18 # IoU threshold to match objects during exploration phase
EXPLORATION_MODE = True  # Flag to indicate if we are in exploration mode

SIM_THRESHOLD = 0.75
TRACKING_IOU_THRESHOLD = 0.3  # IoU threshold to consider objects as "moved" in tracking mode
VOLUME_EXPANSION_RATIO = 0.01 # 1% expansion relative to object dimensions


# Load OpenAI API key
# Setup file logger and project paths
file_path = os.path.abspath(__file__)
# Find project root: if in install dir, go to workspace root, else go up to find CMakeLists.txt
current_dir = os.path.dirname(file_path)
PROJECT_ROOT = current_dir.split('/install/')[0] if '/install/' in current_dir else os.path.abspath(os.path.join(current_dir, "../.."))
# Metti il percorso esatto in base a dove si trova realmente il file

# Built lazily by cv_utils._get_client(); reading a key at import time takes
# down every importer of this module when the file is absent.
client = None


# Loaded lazily on first use by nlp_utils.semantic_model(); importing this
# module no longer downloads 1.6 GB before the node can start.
world2vec = None


# Setup file logger and project paths
current_dir = os.path.dirname(file_path)
PROJECT_ROOT = current_dir.split('/install/')[0] if '/install/' in current_dir else os.path.abspath(os.path.join(current_dir, "../.."))
log_dir = os.path.join(PROJECT_ROOT, "output")
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, f"object_manager_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
file_handler = logging.FileHandler(log_file)
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(message)s'))

# Setup module logger
module_logger = logging.getLogger('object_manager_module')
module_logger.setLevel(logging.DEBUG)
module_logger.addHandler(file_handler)

# Dedicated tracking file for important operations
tracking_log_file = os.path.join(log_dir, f"tracking_operations_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")

# Initialize TrackingLogger
tracking_logger = TrackingLogger(tracking_log_file)


def create_object_key(label, material, color, description):
    """
    Create a unique key for an object based on label, material, color, and description.

    Args:
        label: Object label
        material: Object material
        color: Object color
        description: Object description

    Returns:
        str: Unique key (JSON string)
    """
    key_dict = {
        "label": label if label else "",
        "material": material if material else "",
        "color": color if color else "",
        "description": description if description else ""
    }
    return json.dumps(key_dict, sort_keys=True)


def find_best_matching_key(target_label, target_material, target_color, target_description,
                           target_embedding, keys_dict, word2vec_model, threshold=0.0):
    """
    Find the best key in keys_dict matching the target object using lost_similarity.

    Args:
        target_label, target_material, target_color, target_description: Target object attributes
        target_embedding: Embedding of the target object's description
        keys_dict: Dictionary containing composite keys to search
        word2vec_model: Word2Vec model
        threshold: Minimum similarity threshold

    Returns:
        tuple: (best_key, best_similarity) or (None, 0.0) if no match
    """
    best_key = None
    best_similarity = threshold

    for key in keys_dict.keys():
        key_data = json.loads(key)

        # Get embedding for the key description
        key_embedding = get_embedding(semantic_model(), key_data["description"]) if key_data["description"] else None

        # Compute lost_similarity
        similarity = lost_similarity(
            word2vec_model,
            target_label, key_data["label"],
            target_color, key_data["color"],
            target_material, key_data["material"],
            target_embedding, key_embedding
        )

        if similarity > best_similarity:
            best_similarity = similarity
            best_key = key

    return best_key, best_similarity


def compute_pov_volume(bboxes_list, expansion_ratio=VOLUME_EXPANSION_RATIO):
    """
    Compute the POV (Point of View) volume that contains all detections.

    This volume represents what the camera is looking at in this frame.
    Persistent objects whose centroids are inside this volume but are not seen
    must be removed.

    Args:
        bboxes_list: list of bbox dicts (x_min, x_max, y_min, y_max, z_min, z_max)
        expansion_ratio: proportional expansion ratio (default 0.2 = 20%)

    Returns:
        dict: Expanded POV volume, or None if the list is empty
    """
    if not bboxes_list:
        return None

    # Max volume to trigger bbox reduction (in cubic meters)
    MAX_VOLUME_THRESHOLD = 0.5  # If bbox > 0.5 m³, shrink by 30%
    BBOX_REDUCTION_RATIO = 0.30  # 30% reduction

    def shrink_bbox(bbox, ratio):
        """Shrink a bbox by ratio% while keeping its center."""
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
        """Compute bbox volume."""
        return ((bbox["x_max"] - bbox["x_min"]) *
                (bbox["y_max"] - bbox["y_min"]) *
                (bbox["z_max"] - bbox["z_min"]))

    # Process bboxes: if too large, shrink them
    processed_bboxes = []
    for bbox in bboxes_list:
        vol = bbox_volume(bbox)
        if vol > MAX_VOLUME_THRESHOLD:
            # Bbox too large; use reduced version for extrema
            processed_bbox = shrink_bbox(bbox, BBOX_REDUCTION_RATIO)
            print(f"   [POV] Bbox volume {vol:.3f} m³ > {MAX_VOLUME_THRESHOLD} m³ → reduced by {BBOX_REDUCTION_RATIO*100:.0f}%")
        else:
            processed_bbox = bbox
        processed_bboxes.append(processed_bbox)

    # Find limits covering ALL processed bboxes
    x_min = min(bbox["x_min"] for bbox in processed_bboxes)
    x_max = max(bbox["x_max"] for bbox in processed_bboxes)
    y_min = min(bbox["y_min"] for bbox in processed_bboxes)
    y_max = max(bbox["y_max"] for bbox in processed_bboxes)
    z_min = min(bbox["z_min"] for bbox in processed_bboxes)
    z_max = max(bbox["z_max"] for bbox in processed_bboxes)

    # Compute POV dimensions
    x_size = x_max - x_min
    y_size = y_max - y_min
    z_size = z_max - z_min

    # Proportional expansion
    x_expansion = x_size * expansion_ratio
    y_expansion = y_size * expansion_ratio
    z_expansion = z_size * expansion_ratio

    # Minimum expansion to avoid tiny volumes (e.g., single object)
    MIN_EXPANSION = 0.1 # minimum 10 cm
    x_expansion = max(x_expansion, MIN_EXPANSION)
    y_expansion = max(y_expansion, MIN_EXPANSION)
    z_expansion = max(z_expansion, MIN_EXPANSION)

    # Optimize Z to cover the full seen depth without exceeding it.
    # z_max is the farthest depth; do not expand beyond z_max.

    pov_z_min = z_min - z_expansion
    pov_z_max = z_max  # Do not expand beyond the farthest seen depth

    print(f"   [POV] Optimized Z volume: [{pov_z_min:.3f}m, {pov_z_max:.3f}m] (max seen depth: {z_max:.3f}m)")

    return {
        "x_min": x_min - x_expansion,
        "x_max": x_max + x_expansion,
        "y_min": y_min - y_expansion,
        "y_max": y_max + y_expansion,
        "z_min": pov_z_min,
        "z_max": pov_z_max
    }


def expand_bbox_for_search(bbox, expansion_ratio=VOLUME_EXPANSION_RATIO):
    """
    Expand a bounding box proportionally to its size.
    
    Args:
        bbox: dict with keys x_min, x_max, y_min, y_max, z_min, z_max
        expansion_ratio: expansion percentage relative to dimensions (default 0.2 = 20%)

    Returns:
        dict: expanded bounding box
    """
    # Compute current dimensions
    x_size = bbox["x_max"] - bbox["x_min"]
    y_size = bbox["y_max"] - bbox["y_min"]
    z_size = bbox["z_max"] - bbox["z_min"]
    
    # Compute proportional expansion for each axis
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


def bbox_intersects_volume(bbox, volume):
    """
    Check if a bounding box intersects a search volume.

    Args:
        bbox: bounding box of the persistent object
        volume: expanded search volume

    Returns:
        bool: True if they intersect
    """
    if bbox is None:
        return False

    # Check no overlap (then negate)
    no_overlap = (
        bbox["x_max"] < volume["x_min"] or bbox["x_min"] > volume["x_max"] or
        bbox["y_max"] < volume["y_min"] or bbox["y_min"] > volume["y_max"] or
        bbox["z_max"] < volume["z_min"] or bbox["z_min"] > volume["z_max"]
    )

    return not no_overlap


def bbox_centroid_in_volume(bbox, volume):
    """
    Check whether the centroid of a bounding box lies inside a volume.

    More conservative than bbox_intersects_volume: large objects (e.g., a sofa)
    that barely intersect the POV will not be removed if their centroid is outside.

    Args:
        bbox: object bounding box (dict with x_min, x_max, y_min, y_max, z_min, z_max)
        volume: search volume (dict with x_min, x_max, y_min, y_max, z_min, z_max)

    Returns:
        bool: True if the bbox centroid is fully inside the volume
    """
    if bbox is None:
        return False

    # Compute bbox centroid
    centroid_x = (bbox["x_min"] + bbox["x_max"]) / 2.0
    centroid_y = (bbox["y_min"] + bbox["y_max"]) / 2.0
    centroid_z = (bbox["z_min"] + bbox["z_max"]) / 2.0

    # Check whether the centroid is inside the volume
    is_inside = (
        volume["x_min"] <= centroid_x <= volume["x_max"] and
        volume["y_min"] <= centroid_y <= volume["y_max"] and
        volume["z_min"] <= centroid_z <= volume["z_max"]
    )

    return is_inside


def save_persistent_perceptions(node):
    """
    Save persistent_perceptions to JSON to persist across node runs.
    """
    output_dir = os.path.join(PROJECT_ROOT, "output")
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, "persistent_perception.json")

    data = []
    for obj in wm.persistent_perceptions:
        data.append({
            "label": obj.label,
            "description": obj.description,
            "color": obj.color,
            "material": obj.material,
            "shape": obj.shape,
            "bbox": obj.bbox
        })

    with open(save_path, "w") as f:
        json.dump(data, f, indent=4)

    # ROS2_MIGRATION
    msg = f"Saved {len(data)} objects to persistent_perception.json"
    node.log_both('info', msg)
    labels = [obj["label"] for obj in data]
    node.log_both('info', f"Saved object labels: {labels}")

    # Detailed file log
    for obj_data in data:
        node.log_both('debug', f"  - {obj_data['label']}: {obj_data['description']}")


def save_scene_graph(node, step, is_exploration=False):
    """
    Generate and save a 3D scene graph image for the current step.

    Args:
        node: ROS node for logging
        step: Current step number (exploration or tracking)
        is_exploration: If True, this is an exploration step, else tracking step
    """
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend for saving
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    output_dir = os.path.join(PROJECT_ROOT, "output")
    os.makedirs(output_dir, exist_ok=True)

    objects = []
    for obj in wm.persistent_perceptions:
        objects.append({
            "label": obj.label,
            "color": obj.color,
            "material": obj.material,
            "bbox": obj.bbox
        })
    
    # Save JSON data for better analysis
    prefix = "exploration" if is_exploration else "tracking"
    json_path = os.path.join(output_dir, f"scene_graph_{prefix}_{step:03d}.json")
    with open(json_path, 'w') as f:
        json.dump({
            "step": step,
            "mode": prefix,
            "num_objects": len(objects),
            "objects": objects
        }, f, indent=2)
    node.log_both('info', f"[SCENE GRAPH] Saved JSON: {json_path}")

    mode_str = "Exploration" if is_exploration else "Tracking"
    if len(objects) == 0:
        node.log_both('info', f"[SCENE GRAPH] No objects to visualize for {mode_str} step {step}")
        return

    # Set up figure
    fig = plt.figure(figsize=(14, 10), facecolor='#F9F7F7')
    ax = fig.add_subplot(111, projection='3d')
    ax.set_facecolor('#F9F7F7')

    # Style panes
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor('#DDE6ED')
    ax.yaxis.pane.set_edgecolor('#DDE6ED')
    ax.zaxis.pane.set_edgecolor('#DDE6ED')

    # Grid style
    ax.xaxis._axinfo["grid"]['color'] = '#DDE6ED'
    ax.yaxis._axinfo["grid"]['color'] = '#DDE6ED'
    ax.zaxis._axinfo["grid"]['color'] = '#DDE6ED'
    ax.xaxis._axinfo["grid"]['linewidth'] = 0.5
    ax.yaxis._axinfo["grid"]['linewidth'] = 0.5
    ax.zaxis._axinfo["grid"]['linewidth'] = 0.5

    node_color = '#7EB5D6'

    # Get centroids
    centroids = []
    for obj in objects:
        bbox = obj['bbox']
        centroid = np.array([
            (bbox['x_min'] + bbox['x_max']) / 2,
            (bbox['y_min'] + bbox['y_max']) / 2,
            (bbox['z_min'] + bbox['z_max']) / 2
        ])
        centroids.append(centroid)
    centroids = np.array(centroids)

    # Root position
    scene_center = centroids.mean(axis=0)
    z_max = max(c[2] for c in centroids) + 0.3
    root_pos = np.array([scene_center[0], scene_center[1], z_max])

    # Draw root node
    ax.scatter(*root_pos, s=500, c='#4A6572', marker='o', zorder=10,
               edgecolors='#F9F7F7', linewidth=2.5)
    ax.text(root_pos[0], root_pos[1], root_pos[2] + 0.08, 'SCENE',
            ha='center', va='bottom', fontsize=13, fontweight='bold', color='#344955')

    # Draw objects (senza label individuali per ridurre affollamento)
    for i, obj in enumerate(objects):
        centroid = centroids[i]

        # Edge
        ax.plot3D([root_pos[0], centroid[0]],
                  [root_pos[1], centroid[1]],
                  [root_pos[2], centroid[2]],
                  color='#9DB2BF', linewidth=1.5, alpha=0.5, linestyle='-')

        # Node
        ax.scatter(*centroid, s=400, c=node_color, marker='o', zorder=10,
                   edgecolors='#F9F7F7', linewidth=2)

        # Label
        ax.text(centroid[0], centroid[1], centroid[2] + 0.1,
                obj['label'],
                ha='center', va='bottom', fontsize=10, fontweight='semibold',
                color='#2C3E50')

    # Labels
    ax.set_xlabel('X (m)', fontsize=11, color='#526D82')
    ax.set_ylabel('Y (m)', fontsize=11, color='#526D82')
    ax.set_zlabel('Z (m)', fontsize=11, color='#526D82')

    # Title
    title_mode = "Exploration" if is_exploration else "Tracking"
    ax.set_title(f'3D Scene Graph - {title_mode} Step {step}\n{len(objects)} objects',
                 fontsize=14, fontweight='bold', color='#27374D', pad=20)

    # Equal aspect ratio
    all_points = np.vstack([centroids, root_pos.reshape(1, -1)])
    max_range = np.array([all_points[:, 0].max() - all_points[:, 0].min(),
                          all_points[:, 1].max() - all_points[:, 1].min(),
                          all_points[:, 2].max() - all_points[:, 2].min()]).max() / 2.0

    mid_x = (all_points[:, 0].max() + all_points[:, 0].min()) * 0.5
    mid_y = (all_points[:, 1].max() + all_points[:, 1].min()) * 0.5
    mid_z = (all_points[:, 2].max() + all_points[:, 2].min()) * 0.5

    ax.set_xlim(mid_x - max_range - 0.2, mid_x + max_range + 0.2)
    ax.set_ylim(mid_y - max_range - 0.2, mid_y + max_range + 0.2)
    ax.set_zlim(0, mid_z + max_range + 0.3)

    ax.view_init(elev=30, azim=45)

    plt.tight_layout()

    # Save
    prefix = "exploration" if is_exploration else "tracking"
    output_path = os.path.join(output_dir, f"scene_graph_{prefix}_{step:03d}.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)

    node.log_both('info', f"[SCENE GRAPH] Saved: {output_path}")


def save_uncertain_objects(node):
    """
    Save uncertain_objects to a text file.
    Tracks potentially moved or duplicated objects.
    """
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
                f.write(f"   Description: {obj.description}\n")
                f.write(f"   Color: {obj.color}\n")
                f.write(f"   Material: {obj.material}\n")

                if obj.bbox:
                    x_center = (obj.bbox['x_min'] + obj.bbox['x_max']) / 2.0
                    y_center = (obj.bbox['y_min'] + obj.bbox['y_max']) / 2.0
                    z_center = (obj.bbox['z_min'] + obj.bbox['z_max']) / 2.0

                    x_size = obj.bbox["x_max"] - obj.bbox["x_min"]
                    y_size = obj.bbox["y_max"] - obj.bbox["y_min"]
                    z_size = obj.bbox["z_max"] - obj.bbox["z_min"]

                    f.write(f"   Center position: X={x_center:.3f}, Y={y_center:.3f}, Z={z_center:.3f}\n")
                    f.write(f"   Dimensions: {x_size:.3f}m x {y_size:.3f}m x {z_size:.3f}m\n")
                else:
                    f.write(f"   Bbox: NOT AVAILABLE\n")

                f.write("\n" + "-" * 80 + "\n\n")

    node.log_both('info', f"Saved {len(node.uncertain_objects)} uncertain objects in uncertain_objects.txt")


class DebugTracer:
    """Trace all major code steps for debugging."""
    def __init__(self, output_dir):
        self.output_dir = output_dir
        self.trace_data = {
            "execution_timestamp": None,
            "phase": None,
            "steps": []
        }
    
    def start_trace(self, phase):
        """Initialize trace for a new execution phase."""
        self.trace_data = {
            "execution_timestamp": datetime.now().isoformat(),
            "phase": phase,
            "steps": []
        }
    
    def add_step(self, step_name, details):
        """Add a step to the trace."""
        self.trace_data["steps"].append({
            "timestamp": datetime.now().isoformat(),
            "step": step_name,
            "details": details
        })
    
    def save(self):
        """Save trace to debug_trace.json (overwrites)."""
        output_path = os.path.join(self.output_dir, "debug_trace.json")
        with open(output_path, "w") as f:
            json.dump(self.trace_data, f, indent=2)


class ObjectManagerNode(Node):
    def __init__(self):
        super().__init__('object_description_listener_node')

        self.file_logger = module_logger
        self.file_logger.info("=== ObjectManagerNode Initialized ===")
        self.get_logger().info(f"Log saved in: {log_file}")

        self.exploration_mode = True
        self.seen_again = False
        self.latest_bboxes = {}
        self.latest_fov_volume = None
        self.uncertain_objects = []
        self.tracking_step_counter = 0
        self.exploration_step_counter = 0
        self.db = MapDatabase(db_path=os.path.join(log_dir, "tiago_temporal_map_1.db"))
        self.debug_tracer = DebugTracer(log_dir)

        qos_latch = QoSProfile(depth=10, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        qos_standard = QoSProfile(depth=10)
        self.robot_has_moved = False
        
        self.create_subscription(Bool, "/robot_movement_detected", self.movement_callback, qos_standard)
        self.persistent_bbox_pub = self.create_publisher(MarkerArray, '/persistent_bbox', qos_latch)
        self.persistent_centroids_pub = self.create_publisher(MarkerArray, '/persistent_centroids', qos_latch)
        self.considered_volume_pub = self.create_publisher(MarkerArray, '/considered_volume', qos_standard)
        self.uncertain_bboxes_pub = self.create_publisher(MarkerArray, '/uncertain_object', qos_standard)
        self.uncertain_centroids_pub = self.create_publisher(MarkerArray, '/uncertain_centroids', qos_standard)

        self.create_subscription(Bbox3dArray, "/bbox_3d", self.bbox_callback, qos_standard)
        self.create_subscription(ObjectDescriptionArray, "/object_descriptions", self.description_callback, qos_standard)

        self._bbox_timer = self.create_timer(2.0, self.periodic_bbox_publisher)

        exploration_thread = threading.Thread(target=self.wait_for_exploration_end, daemon=True)
        exploration_thread.start()

    def movement_callback(self, msg):
        """Callback to receive robot movement notification from perception."""
        self.log_both('info', f"[MOVEMENT] Received movement message: {msg.data}")
        if msg.data and not self.robot_has_moved:
            self.robot_has_moved = True
            self.log_both('warn', "[MOVEMENT] ✓ Robot has moved at least once - can now transition to TRACKING")

    def log_both(self, level, message):
        """Log to both ROS and file logger."""
        if level == 'info':
            self.get_logger().info(message)
            self.file_logger.info(message)
        elif level == 'warn':
            self.get_logger().warn(message)
            self.file_logger.warning(message)
        elif level == 'error':
            self.get_logger().error(message)
            self.file_logger.error(message)
        elif level == 'debug':
            self.get_logger().debug(message)
            self.file_logger.debug(message)

        if level in ['info', 'warn']:
            prefix = "[INFO] " if level == 'info' else "[WARN] "
            print(f"{prefix}{message}")

    @staticmethod
    def _stdin_ready():
        """True if a line is waiting on stdin. Never blocks, safe without a TTY."""
        try:
            if not sys.stdin or not sys.stdin.isatty():
                return False
            return bool(select.select([sys.stdin], [], [], 0.0)[0])
        except Exception:
            return False

    def wait_for_exploration_end(self):
        """Wait for exploration phase to end."""
        self.log_both('warn', "=" * 60)
        self.log_both('warn', "[EXPLORATION] The robot is in EXPLORATION.")
        self.log_both('warn', "[EXPLORATION] Will switch to TRACKING when an object is seen again.")
        self.log_both('warn', "=" * 60)

        # The loop polls self.seen_again, which the detection path sets when an
        # object is re-observed. The blocking input() that used to be here
        # defeated that: the thread parked inside it and never re-evaluated the
        # condition, so the automatic switch this banner advertises never fired
        # and a human had to press ENTER anyway. With no TTY it raised EOFError
        # and the thread died, taking the switch with it.
        #
        # The prompt is now opt-in (exploration.interactive_gate) and read
        # without blocking, so both exits work: automatic on re-observation,
        # manual whenever an operator wants it.
        interactive = perception_config.get("exploration", "interactive_gate")
        if interactive:
            print("Press ENTER to stop exploration and switch to TRACKING mode...")
        while not self.seen_again:
            if interactive and self._stdin_ready():
                sys.stdin.readline()
                self.seen_again = True
                self.log_both('warn', "[EXPLORATION] Manual interruption received - switching to TRACKING mode...")
                break
            time.sleep(0.1)

        self.log_both('warn', "=" * 60)
        self.log_both('warn', "[EXPLORATION] Exploration phase ENDED!")
        self.log_both('warn', "[TRACKING] Tracking mode ACTIVATED")
        self.log_both('warn', "=" * 60)
        print("\n\n\n")

        self.exploration_mode = False

        publish_persistent_bboxes(self, wm, self.persistent_bbox_pub)
        publish_persistent_centroids(self, wm, self.persistent_centroids_pub)

    def add_new_object(self, label, bbox, description_text, color, material, description_embedding, in_exploration):
        """
        Add a new detected object to persistent_perceptions.

        Args:
            label: Object label
            bbox: Bounding box dict
            description_text: Description text
            color: Object color
            material: Object material
            description_embedding: Embedding of description
            in_exploration: Boolean flag for exploration mode

        Returns:
            new_obj: The newly created object
        """
        new_obj = Object(label, None, bbox, description_text, color, material)
        new_obj.embedding = description_embedding
        wm.persistent_perceptions.append(new_obj)

         # AGGIUNGI (3 righe):
        phase = "exploration" if in_exploration else "tracking"
        step  = self.exploration_step_counter if in_exploration else self.tracking_step_counter
        self.db.on_new_object(new_obj, phase=phase, step=step)

        x_size = bbox["x_max"] - bbox["x_min"]
        y_size = bbox["y_max"] - bbox["y_min"]
        z_size = bbox["z_max"] - bbox["z_min"]
        volume = x_size * y_size * z_size

        mode_tag = "[EXPLORATION]" if in_exploration else f"[TRACKING STEP {self.tracking_step_counter}]"
        print(f"{mode_tag} New object '{label}' added to persistent_perceptions (bbox volume: {volume:.3f} m³)")

        if not in_exploration:
            tracking_logger.log_new_object(new_obj, case_type="NEW DETECTION")

        save_persistent_perceptions(self)

        if in_exploration:
            self.exploration_step_counter += 1
            save_scene_graph(self, self.exploration_step_counter, is_exploration=True)

        return new_obj

    def modify_existing_object(self, best_match, bbox, description_embedding):
        """
        Modify an existing tracked object based on new detection.

        Args:
            best_match: The best matching persistent object
            bbox: New bounding box
            description_embedding: New description embedding

        Returns:
            tuple: (updated_obj, distance, iou)
        """
        old_bbox = best_match.bbox
        iou = compute_iou_3d(bbox, old_bbox)
        print(f"IoU with existing position: {iou:.3f} (threshold: {TRACKING_IOU_THRESHOLD})")

        # Calculate distance moved
        old_x = (old_bbox['x_min'] + old_bbox['x_max']) / 2.0
        old_y = (old_bbox['y_min'] + old_bbox['y_max']) / 2.0
        old_z = (old_bbox['z_min'] + old_bbox['z_max']) / 2.0
        new_x = (bbox['x_min'] + bbox['x_max']) / 2.0
        new_y = (bbox['y_min'] + bbox['y_max']) / 2.0
        new_z = (bbox['z_min'] + bbox['z_max']) / 2.0
        distance = np.sqrt((new_x - old_x)**2 + (new_y - old_y)**2 + (new_z - old_z)**2)

        # CASE 2: High IoU → Same position, just update bbox
        if iou >= TRACKING_IOU_THRESHOLD:
            print(f"CASE 2: Object recognized in the same position (IoU={iou:.3f})")
            best_match.bbox = bbox
            return best_match, distance, iou

        # CASE 3: Low IoU → Object moved far
        else:
            print(f"\nOLD POSITION: X={old_x:.3f}, Y={old_y:.3f}, Z={old_z:.3f}")
            print(f"NEW POSITION: X={new_x:.3f}, Y={new_y:.3f}, Z={new_z:.3f}")
            print(f"Movement distance: {distance:.3f} meters")

            # Remove from old position
            if best_match in wm.persistent_perceptions:
                wm.persistent_perceptions.remove(best_match)
                self.db.on_object_moved(
                best_match, old_bbox=best_match.bbox, new_bbox=bbox,
                distance=distance, iou=iou,
                step=self.tracking_step_counter
            )
                print(f"🔄 MODIFICATION: '{best_match.label}' removed from old position")

            # Add to uncertain if moved far (>0.8m)
            if distance > 0.8:
                if best_match not in self.uncertain_objects:
                    self.uncertain_objects.append(best_match)
                    self.db.on_uncertain_added(best_match, step=self.tracking_step_counter)

                    print(f"📝 '{best_match.label}' added to UNCERTAIN_OBJECTS (movement={distance:.3f}m)")
                    tracking_logger.log_uncertain_added(best_match.label, "Large displacement detected", distance,
                                                       bbox=best_match.bbox, step_number=self.tracking_step_counter,
                                                       obj=best_match, case_type="MOVED >0.8m")

            # Create new object at new position
            updated_obj = Object(best_match.label, None, bbox, best_match.description, best_match.color, best_match.material)
            updated_obj.embedding = description_embedding
            wm.persistent_perceptions.append(updated_obj)
            print(f"✓ '{best_match.label}' reinserted with new position in persistent_perceptions")

            tracking_logger.log_position_change(best_match.label, old_bbox, bbox, distance,
                                              step_number=self.tracking_step_counter, obj=updated_obj,
                                              case_type="POSITION UPDATE")

            return updated_obj, distance, iou

    def delete_undetected_objects(self, pov_volume, current_perception_objects, description_received):
        """
        Delete objects not detected in current frame within POV.

        Args:
            pov_volume: POV volume from depth camera
            current_perception_objects: List of detected objects this frame
            description_received: Boolean if objects were detected

        Returns:
            bool: True if objects were modified
        """
        if not pov_volume:
            print(f"No FOV volume available from depth camera - unable to calculate POV")
            return False

        print(f"[POV] Using FOV from depth camera: X[{pov_volume['x_min']:.2f}, {pov_volume['x_max']:.2f}], "
              f"Y[{pov_volume['y_min']:.2f}, {pov_volume['y_max']:.2f}], "
              f"Z[{pov_volume['z_min']:.2f}, {pov_volume['z_max']:.2f}]")
        publish_pov_volume(self, pov_volume, self.considered_volume_pub)

        objects_to_remove = []

        if not description_received:
            # No objects detected - remove those in POV
            for obj in wm.persistent_perceptions:
                if obj.bbox and bbox_centroid_in_volume(obj.bbox, pov_volume):
                    objects_to_remove.append(obj)
                    print(f"DELETION: '{obj.label}' is IN POV but NOT SEEN → Will be REMOVED")
        else:
            # Objects detected - remove unmatched ones in POV
            for obj in wm.persistent_perceptions:
                if obj not in current_perception_objects:
                    if obj.bbox and bbox_centroid_in_volume(obj.bbox, pov_volume):
                        objects_to_remove.append(obj)

        if objects_to_remove:
            print(f"\nDELETING {len(objects_to_remove)} OBJECTS - TRACKING STEP {self.tracking_step_counter}")
            for obj in objects_to_remove:
                tracking_logger.log_deletion(obj.label, "Object in POV but not detected", bbox=obj.bbox,
                                            step_number=self.tracking_step_counter, obj=obj, case_type="NOT SEEN IN POV")
                wm.persistent_perceptions.remove(obj)
                self.db.on_object_deleted(obj, reason="not seen in POV",
                                  step=self.tracking_step_counter)

            save_persistent_perceptions(self)
            return True
        else:
            print(f"✓ No objects to remove in this step")
            return False

    def delete_uncertain_objects(self, pov_volume):
        """
        Remove uncertain objects whose zones have been verified in POV.

        Args:
            pov_volume: POV volume from depth camera

        Returns:
            bool: True if uncertain objects were modified
        """
        print(f"\n{'─'*60}")
        print(f"[TRACKING STEP {self.tracking_step_counter}] Verifying uncertain objects with POV VOLUME...")
        print(f"{'─'*60}")

        uncertain_to_remove = []

        if pov_volume:
            for uncertain_obj in self.uncertain_objects:
                if uncertain_obj.bbox and bbox_centroid_in_volume(uncertain_obj.bbox, pov_volume):
                    uncertain_to_remove.append(uncertain_obj)
                    print(f"✅ DELETION: '{uncertain_obj.label}' (ORANGE) in POV - zone VERIFIED → REMOVE")

        if uncertain_to_remove:
            print(f"\nDELETING {len(uncertain_to_remove)} UNCERTAIN OBJECTS - TRACKING STEP {self.tracking_step_counter}")
            for uncertain_obj in uncertain_to_remove:
                tracking_logger.log_deletion(uncertain_obj.label, "Uncertain zone verified", bbox=uncertain_obj.bbox,
                                            step_number=self.tracking_step_counter, obj=uncertain_obj,
                                            case_type="UNCERTAIN ZONE VERIFIED")
                self.uncertain_objects.remove(uncertain_obj)
            return True
        else:
            if self.uncertain_objects:
                print(f"No uncertain object to remove in this step")
            return False

    def description_callback(self, msg):
        """Process object descriptions from perception module."""
        # Start debug trace
        self.debug_tracer.start_trace("DESCRIPTION_CALLBACK")
        self.debug_tracer.add_step("CALLBACK_START", {
            "num_descriptions": len(msg.descriptions),
            "timestamp": datetime.now().isoformat()
        })

        if not hasattr(self, 'tracking_step_counter'):
            self.tracking_step_counter = 0
        if not hasattr(self, 'exploration_step_counter'):
            self.exploration_step_counter = 0

        # Publish whenever there is something to show. Gating this on the mode
        # meant the belief was invisible for the entire exploration phase and
        # only appeared after the manual switch, so an unattended run built a
        # correct belief nobody ever saw. The mode is about delete/verify
        # semantics, not about visibility.
        if not self.exploration_mode or perception_config.get(
                "exploration", "publish_during_exploration"):
            self.tracking_step_counter += 1

        in_exploration = self.exploration_mode
        mode_str = "EXPLORATION" if in_exploration else f"TRACKING STEP {self.tracking_step_counter}"
        print(f"\n{'='*60}\nMode: {mode_str}\n{'='*60}")

        self.debug_tracer.add_step("MODE_DETERMINED", {
            "mode": mode_str,
            "in_exploration": in_exploration,
            "tracking_step": self.tracking_step_counter
        })

        current_perception_objects = []
        objects_modified = False
        matched_bboxes = []

        for idx, description in enumerate(msg.descriptions):
            label = description.label
            label_base = label.split('#')[0] if '#' in label else label
            color = description.color
            material = description.material
            description_text = description.description

            self.debug_tracer.add_step(f"PROCESSING_DESCRIPTION_{idx}", {
                "label": label,
                "label_base": label_base,
                "color": color,
                "material": material
            })

            print(f"Processing: '{label}'")
            description_embedding = get_embedding(semantic_model(), description_text)

            # Find matching bbox
            old_key = create_object_key(label, "", "", "")
            if old_key not in self.latest_bboxes:
                self.debug_tracer.add_step(f"NO_BBOX_MATCH_{idx}", {
                    "label": label,
                    "reason": "No matching bbox found"
                })
                print(f"  No matching bbox found for '{label}' → skipped")
                continue

            bbox = self.latest_bboxes[old_key]["bbox"]
            new_key = create_object_key(label, material, color, description_text)

            del self.latest_bboxes[old_key]
            self.latest_bboxes[new_key] = {
                "bbox": bbox,
                "label": label,
                "color": color,
                "material": material,
                "description": description_text
            }

            matched_bboxes.append(bbox)
            already_seen = False

            self.debug_tracer.add_step(f"BBOX_FOUND_{idx}", {
                "label": label,
                "bbox": bbox
            })

            # ======== EXPLORATION LOGIC ========
            if in_exploration:
                self.debug_tracer.add_step(f"EXPLORATION_MATCHING_{idx}", {
                    "label": label,
                    "num_persistent": len(wm.persistent_perceptions)
                })
                print(f"  Comparing with {len(wm.persistent_perceptions)} persistent objects...")
                for obj in wm.persistent_perceptions:
                    obj_label_base = obj.label.split('#')[0] if '#' in obj.label else obj.label
                    similarity = lost_similarity(world2vec, label_base, obj_label_base, color, obj.color,
                                               material, obj.material, description_embedding, obj.embedding)

                    if similarity > SIM_THRESHOLD and obj.bbox is not None:
                        iou = compute_iou_3d(bbox, obj.bbox)
                        if iou >= EXPLORATION_IOU_THRESHOLD:
                            already_seen = True
                            current_perception_objects.append(obj)
                            obj.bbox = bbox
                            print(f"  ✓ MATCH: '{label}' = '{obj.label}' (sim={similarity:.3f}, IoU={iou:.3f})")

                            self.debug_tracer.add_step(f"EXPLORATION_MATCH_FOUND_{idx}", {
                                "new_label": label,
                                "existing_label": obj.label,
                                "similarity": similarity,
                                "iou": iou
                            })

                            if not self.seen_again and self.robot_has_moved:
                                self.seen_again = True
                                self.log_both('warn', f"[EXPLORATION] Object seen again + robot moved → Switching to TRACKING...")
                            break

            # ======== TRACKING LOGIC ========
            else:
                search_volume = expand_bbox_for_search(bbox, VOLUME_EXPANSION_RATIO)
                self.debug_tracer.add_step(f"TRACKING_MATCHING_{idx}", {
                    "label": label,
                    "num_persistent": len(wm.persistent_perceptions)
                })
                print(f"  Searching in {len(wm.persistent_perceptions)} persistent objects...")

                best_match = None
                best_score = 0

                for obj in wm.persistent_perceptions:
                    if not hasattr(obj, "embedding"):
                        obj.embedding = get_embedding(semantic_model(), obj.description)
                    if obj.embedding is None:
                        continue

                    obj_label_base = obj.label.split('#')[0] if '#' in obj.label else obj.label
                    similarity = lost_similarity(world2vec, label_base, obj_label_base, color, obj.color,
                                               material, obj.material, description_embedding, obj.embedding)

                    if similarity > SIM_THRESHOLD and similarity > best_score:
                        best_score = similarity
                        best_match = obj

                if best_match:
                    print(f"  Best match: '{best_match.label}' (sim={best_score:.2f})")
                    already_seen = True

                    updated_obj, distance, iou = self.modify_existing_object(best_match, bbox, description_embedding)
                    current_perception_objects.append(updated_obj)
                    objects_modified = True

                    self.debug_tracer.add_step(f"TRACKING_MATCH_FOUND_{idx}", {
                        "detected_label": label,
                        "matched_label": best_match.label,
                        "similarity": best_score,
                        "distance_moved": distance,
                        "iou": iou
                    })
                else:
                    print(f"  No semantic match found")
                    self.debug_tracer.add_step(f"NO_SEMANTIC_MATCH_{idx}", {
                        "label": label
                    })

            # ======== ADD NEW OBJECT ========
            if not already_seen:
                print(f"  NEW OBJECT: Adding '{label}'")
                new_obj = self.add_new_object(label, bbox, description_text, color, material,
                                             description_embedding, in_exploration)
                current_perception_objects.append(new_obj)
                objects_modified = True

                self.debug_tracer.add_step(f"NEW_OBJECT_ADDED_{idx}", {
                    "label": label,
                    "color": color,
                    "material": material
                })

        # ======== DELETION LOGIC (TRACKING ONLY) ========
        if not in_exploration:
            description_received = len(msg.descriptions) > 0
            pov_volume = self.latest_fov_volume

            self.debug_tracer.add_step("DELETION_LOGIC_START", {
                "description_received": description_received,
                "pov_volume_available": pov_volume is not None
            })

            if pov_volume:
                objects_modified_delete = self.delete_undetected_objects(pov_volume, current_perception_objects, description_received)
                objects_modified = objects_modified or objects_modified_delete

                objects_modified_uncertain = self.delete_uncertain_objects(pov_volume)
                objects_modified = objects_modified or objects_modified_uncertain

                self.debug_tracer.add_step("DELETION_LOGIC_COMPLETE", {
                    "objects_deleted": objects_modified_delete,
                    "uncertain_deleted": objects_modified_uncertain
                })

        # ======== PUBLISH & SAVE ========
        if objects_modified and not in_exploration:
            publish_persistent_bboxes(self, wm, self.persistent_bbox_pub)
            publish_persistent_centroids(self, wm, self.persistent_centroids_pub)
            publish_uncertain_bboxes(self, self.uncertain_objects, self.uncertain_bboxes_pub)
            publish_uncertain_centroids(self, self.uncertain_objects, self.uncertain_centroids_pub)
            save_uncertain_objects(self)

            self.debug_tracer.add_step("VISUALIZATION_PUBLISHED", {
                "num_persistent": len(wm.persistent_perceptions),
                "num_uncertain": len(self.uncertain_objects)
            })

        if not in_exploration:
            save_scene_graph(self, self.tracking_step_counter)

        self.debug_tracer.add_step("CALLBACK_END", {
            "final_persistent_count": len(wm.persistent_perceptions),
            "final_uncertain_count": len(self.uncertain_objects)
        })

        # Save debug trace (overwrites)
        self.debug_tracer.save()

        self.latest_bboxes.clear()
        self.latest_fov_volume = None

    def bbox_callback(self, msg):
        """Callback for 3D bounding boxes."""
        self.latest_bboxes = {}

        if msg.fov_x_max != 0 or msg.fov_y_max != 0 or msg.fov_z_max != 0:
            self.latest_fov_volume = {
                "x_min": msg.fov_x_min,
                "x_max": msg.fov_x_max,
                "y_min": msg.fov_y_min,
                "y_max": msg.fov_y_max,
                "z_min": msg.fov_z_min,
                "z_max": msg.fov_z_max
            }

        for box in msg.boxes:
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
                "bbox": bbox_data,
                "label": box.label,
                "color": "",
                "material": "",
                "description": ""
            }

    def periodic_bbox_publisher(self):
        """Periodically publishes persistent and uncertain bounding boxes."""
        if not self.exploration_mode:
            if len(wm.persistent_perceptions) > 0:
                publish_persistent_bboxes(self, wm, self.persistent_bbox_pub)
                publish_persistent_centroids(self, wm, self.persistent_centroids_pub)

            if len(self.uncertain_objects) > 0:
                publish_uncertain_bboxes(self, self.uncertain_objects, self.uncertain_bboxes_pub)
                publish_uncertain_centroids(self, self.uncertain_objects, self.uncertain_centroids_pub)
                save_uncertain_objects(self)


def main(args=None):
    rclpy.init(args=args)
    node = ObjectManagerNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        print(f"OBJECT MANAGER closed {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    finally:
        try:
            node.close_json_writers()
        except Exception:
            pass
        node.destroy_node()
        rclpy.shutdown()
        tracking_logger.close()


if __name__ == "__main__":
    main()