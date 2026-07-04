#!/usr/bin/env python3
"""
Object Manager Service - Semantic and Spatial Tracking of Perceived Objects
Tracks objects and automatically transitions from EXPLORATION to TRACKING when
an object is seen again in a different position.
Includes Topological Semantic Mapping (Room Manager) with Scene Graph generation.
"""
import rclpy, json, os, time, threading, re
from rclpy.node import Node
from rclpy.duration import Duration
from rclpy.qos import QoSProfile, DurabilityPolicy
import numpy as np
from openai import OpenAI
from lost3dsg.msg import ObjectDescriptionArray, Bbox3dArray
from lost3dsg.srv import ObjectTrackingService
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import Bool
from object_info import Object
from world_model import wm
import gensim.downloader as api
from utils import *
from nlp_utils import *
from datetime import datetime
from cv_utils import *
from map_database import MapDatabase
from gensim.models import KeyedVectors
from std_msgs.msg import String
from tf2_ros import Buffer, TransformListener

# =============  EXPLORATION PARAMETERS =============
EXPLORATION_IOU_THRESHOLD = 0.10
SIM_THRESHOLD = 0.85
TRACKING_IOU_THRESHOLD = 0.3
VOLUME_EXPANSION_RATIO = 0.01
EXPLORATION_FRAME_LIMIT = 10  # Numero di frame in exploration prima di passare a tracking
OBJECT_STABILITY_TIMEOUT = 3.0  # Secondi minimi di vita prima di poter essere 'MOVED'
POV_SCALE_FACTOR = 1.0

# Load OpenAI API key & Paths
file_path = os.path.abspath(__file__)
current_dir = os.path.dirname(file_path)
PROJECT_ROOT = current_dir.split('/install/')[0] if '/install/' in current_dir else os.path.abspath(os.path.join(current_dir, "../.."))

world2vec = KeyedVectors.load_word2vec_format(
    '/root/gensim-data/word2vec-google-news-300/word2vec-google-news-300.gz', 
    binary=True,
    limit=200000  # <--- ECCO LA MAGIA CHE SALVA LA RAM!
)
# Setup path per il file sintetico di operazioni
log_dir = os.path.join(PROJECT_ROOT, "output")
os.makedirs(log_dir, exist_ok=True)
SYNTHETIC_LOG_FILE = os.path.join(log_dir, "operations.txt")

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


def compute_pov_volume(bboxes_list, expansion_ratio=VOLUME_EXPANSION_RATIO):
    """Compute the POV volume that contains all detections."""
    if not bboxes_list:
        return None

    MAX_VOLUME_THRESHOLD = 0.5
    BBOX_REDUCTION_RATIO = 0.30

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


def bbox_centroid_in_volume(bbox, volume):
    """Check whether the centroid of a bounding box lies inside a volume."""
    if bbox is None:
        return False

    centroid_x = (bbox["x_min"] + bbox["x_max"]) / 2.0
    centroid_y = (bbox["y_min"] + bbox["y_max"]) / 2.0
    centroid_z = (bbox["z_min"] + bbox["z_max"]) / 2.0

    is_inside = (
        volume["x_min"] <= centroid_x <= volume["x_max"] and
        volume["y_min"] <= centroid_y <= volume["y_max"] and
        volume["z_min"] <= centroid_z <= volume["z_max"]
    )

    return is_inside


def save_persistent_perceptions(node):
    """Save persistent_perceptions to JSON."""
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
            "bbox": obj.bbox,
            "room_id": getattr(obj, "room_id", "unknown")
        })

    with open(save_path, "w") as f:
        json.dump(data, f, indent=4)

    msg = f"Saved {len(data)} objects to persistent_perception.json"
    node.log_both('info', msg)

def save_scene_graph(node, step, is_exploration=False):
    """Generate and save a 3D scene graph image for the current step."""
    import matplotlib
    matplotlib.use('Agg')
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
            "bbox": obj.bbox,
            "room_id": getattr(obj, "room_id", "unknown")
        })
    
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

    # Plot logic (omitted for brevity, you can keep your existing plot generation logic here if desired)
    # ...


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


def publish_persistent_bboxes(node, wm, pub):
    marker_array = MarkerArray()
    for i, obj in enumerate(wm.persistent_perceptions):
        if obj.bbox is None or "door" in obj.label.lower():
             continue
        marker = Marker()
        marker.header.frame_id = "map"
        marker.header.stamp = node.get_clock().now().to_msg()
        marker.id = i
        marker.type = Marker.CUBE
        marker.action = Marker.ADD
        marker.pose.position.x = (obj.bbox['x_min'] + obj.bbox['x_max']) / 2.0
        marker.pose.position.y = (obj.bbox['y_min'] + obj.bbox['y_max']) / 2.0
        marker.pose.position.z = (obj.bbox['z_min'] + obj.bbox['z_max']) / 2.0
        marker.scale.x = obj.bbox['x_max'] - obj.bbox['x_min']
        marker.scale.y = obj.bbox['y_max'] - obj.bbox['y_min']
        marker.scale.z = obj.bbox['z_max'] - obj.bbox['z_min']
        marker.color.a = 0.5
        marker.color.r, marker.color.g, marker.color.b = 0.0, 1.0, 0.0
        marker_array.markers.append(marker)
    pub.publish(marker_array)

def publish_persistent_centroids(node, wm, pub):
    marker_array = MarkerArray()
    for i, obj in enumerate(wm.persistent_perceptions):
        if obj.bbox is None or "door" in obj.label.lower():
             continue
        marker = Marker()
        marker.header.frame_id = "map"
        marker.header.stamp = node.get_clock().now().to_msg()
        marker.id = i
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
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
        marker.header.stamp = node.get_clock().now().to_msg()
        marker.id = i
        marker.type = Marker.CUBE
        marker.action = Marker.ADD
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
        marker.header.stamp = node.get_clock().now().to_msg()
        marker.id = i
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
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


# ============= ROOM MANAGER & SCENE GRAPH =============

class RoomManager:
    def __init__(self, w2v_model, similarity_threshold=0.55):
        self.config = config
        self.w2v = w2v_model
        self.similarity_threshold = similarity_threshold
        self.room_counter = 1
        self.current_room_id = "room_0"
        self.current_room_walls = []
        self.rooms = [] 
        self.last_closed_room_polygon = None
        
        # Cache pesi: se un oggetto non è qui, chiameremo il VLM
        self.weights_cache = {
            "chair": 0.2,
            "table": 0.5,
            "wall": 0.1,
        }
        
        # --- SCENE GRAPH ---
        self.scene_graph = {}
        self.init_room_node(self.current_room_id)
        # Buffer per salvare i muri visti in questa stanza
        self.current_room_walls = []
        self.current_room_wall_segments = []
        self.closed_room_centroids = []  # Centroidi degli hull già chiusi

        # Iscrizione al topic dei muri rilevati
     

    def init_room_node(self, room_id):
        """Crea un nuovo nodo padre (Stanza) nel grafo."""
        self.scene_graph[room_id] = {
            "semantic_label": "Unknown",
            "embedding": None,
            "boundaries": {  # Manteniamo questo per compatibilità
                "x_min": float('inf'), "x_max": float('-inf'),
                "y_min": float('inf'), "y_max": float('-inf'),
                "z_min": float('inf'), "z_max": float('-inf')
            },
            "polygon": [],   # <--- NUOVO: Conterrà i punti del bordo stanza
            "objects": []
        }

    def update_room_geometry(self, room_id, bbox):
        """Espande la geometria della stanza in base agli oggetti e ai muri rilevati."""
        if not bbox: return
        b = self.scene_graph[room_id]["boundaries"]
        b["x_min"], b["x_max"] = min(b["x_min"], bbox["x_min"]), max(b["x_max"], bbox["x_max"])
        b["y_min"], b["y_max"] = min(b["y_min"], bbox["y_min"]), max(b["y_max"], bbox["y_max"])
        b["z_min"], b["z_max"] = min(b["z_min"], bbox["z_min"]), max(b["z_max"], bbox["z_max"])

    def assign_room_by_geometry(self, bbox):
        """Assegna un oggetto alla stanza verificandone il centroide."""
        if not bbox: return self.current_room_id

        cx = (bbox["x_min"] + bbox["x_max"]) / 2.0
        cy = (bbox["y_min"] + bbox["y_max"]) / 2.0

        for r_id, data in self.scene_graph.items():
            if r_id == self.current_room_id:
                continue 
                
            poly = data.get("polygon", [])
            if len(poly) >= 3:
                # Usa matplotlib Path per controllare se il punto (cx, cy) è dentro il poligono
                from matplotlib.path import Path
                if Path(poly).contains_point([cx, cy]):
                    return r_id
            else:
                # Fallback di sicurezza se la stanza non ha un poligono valido
                b = data["boundaries"]
                if (b["x_min"] <= cx <= b["x_max"]) and (b["y_min"] <= cy <= b["y_max"]):
                    return r_id
                
        return self.current_room_id

    def ask_vlm_room_info(self, room_objects_labels, encoded_image=None):
        """
        Interroga il VLM per ottenere nome e descrizione della stanza.
        Supporta sia testo che immagini codificate in Base64.
        """
        if not room_objects_labels:
            return "Unknown_Room", "Stanza senza oggetti rilevanti."

        labels_str = ", ".join(set(room_objects_labels))
        
        try:
            # Caricamento API Key
            api_key_path = "/root/exchange/lost3dsg/src/perception_module/api.txt"
            with open(api_key_path, "r") as f:
                api_key = f.read().strip()

            client = OpenAI(base_url="https://api.regolo.ai/v1", api_key=api_key)

            # Costruzione del prompt
            text_prompt = (
                f"Analizza questi oggetti rilevati: {labels_str}.\n"
                "Identifica il tipo di stanza e fornisci una descrizione.\n"
                "Rispondi SOLO in formato JSON: {\"label\": \"nome\", \"description\": \"descrizione\"}"
            )

            # Preparazione contenuto del messaggio (Multimodale)
            content = [{"type": "text", "text": text_prompt}]
            
            if encoded_image:
                content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{encoded_image}"}
                })

            response = client.chat.completions.create(
                model="gemma4-31b", # Assicurati che questo modello supporti la Vision su Regolo
                messages=[{"role": "user", "content": content}],
                response_format={"type": "json_object"},
                timeout=15.0
            )

            # Estrazione e pulizia del contenuto
            raw_content = response.choices[0].message.content.strip()
            
            # Rimuove eventuali blocchi ```json ... ``` se presenti
            clean_json = re.sub(r'^```json\s*|```$', '', raw_content, flags=re.MULTILINE)
            
            data = json.loads(clean_json)
            return data.get("label", "Unknown_Room"), data.get("description", "")

        except Exception as e:
            print(f"🔴 [VLM ERROR] Fallimento chiamata: {e}")
            return "Unknown_Room", f"Fallback per oggetti: {labels_str}"

            
    def evaluate_scene(self, current_scene_objects, persistent_objects, encoded_image=None, vlm_callback=None):
        self.last_known_objects = persistent_objects 
        
        if not current_scene_objects: 
            return self.current_room_id
            
        raw_labels = [obj.label.lower() for obj in current_scene_objects]
        clean_labels = list(set([l.split('#')[0].replace(" ", "_") for l in raw_labels]))
        
        return self.current_room_id


    
    def save_room_to_json(self, room_id, label, description, objects, walls):
        output_dir = os.path.join(PROJECT_ROOT, "output")
        os.makedirs(output_dir, exist_ok=True)
        json_path = os.path.join(output_dir, "room.json")
        
        room_data = {
            "room_id": room_id,
            "semantic_label": label,
            "description": description,
            "objects": objects,
            "walls": walls
        }
        
        updated = False
        for i, r in enumerate(self.rooms):
            if r["room_id"] == room_id:
                self.rooms[i] = room_data
                updated = True
                break
        if not updated:
            self.rooms.append(room_data)

        with open(json_path, "w") as f:
            json.dump(self.rooms, f, indent=4)
        
        # AGGIUNTA QUESTA STAMPA
        print(f"✅ [DEBUG SALVATAGGIO] Dati stanza '{room_id}' salvati correttamente in: {json_path}")

    def finalize_current_room(self, persistent_objects):
        room_id = self.current_room_id
        print(f"🛑 [FINALIZZAZIONE] Chiamata finalize per: {room_id}")
        
        # 1. Recupera oggetti della stanza
        room_objs = [o for o in persistent_objects if getattr(o, 'room_id', 'unknown') == room_id]
        
        # 2. DEFINIZIONE CORRETTA (Risolve NameError)
        room_labels = [o.label.split('#')[0].lower().replace(" ", "_") for o in room_objs]
        
        semantic_name = "Unknown_Room"
        description = "Nessuna descrizione (VLM non disponibile o stanza vuota)."

        if room_labels:
            vlm_name, vlm_desc = self.ask_vlm_room_info(room_labels)
            if vlm_name and vlm_name != "Unknown_Room":
                semantic_name = vlm_name
            if vlm_desc:
                description = vlm_desc
        
        # 3. Aggiorna Scene Graph
        if room_id in self.scene_graph:
            self.scene_graph[room_id]["semantic_label"] = semantic_name
            self.scene_graph[room_id]["description"] = description
            objs_to_save = self.scene_graph[room_id]["objects"]
        else:
            objs_to_save = room_labels

        # 4. SALVATAGGIO
        self.save_room_to_json(
            room_id=room_id, 
            label=semantic_name, 
            description=description, 
            objects=objs_to_save, 
            walls=self.current_room_walls
        )

# ============= MAIN SERVICE NODE =============

class ObjectManagerService(Node):
    def __init__(self):
        super().__init__('object_tracking_service_node')

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
        self.tracking_step_counter = 0
        self.exploration_step_counter = 0
        self.exploration_frame_counter = 0
        self.db = MapDatabase(db_path=os.path.join(log_dir, "tiago_temporal_map_5.db"))
        self.robot_has_moved = False

        self.latest_descriptions = None
        self.latest_bboxes_msg = None
        
        # --- INIT ROOM MANAGER ---
        self.room_manager = RoomManager(w2v_model=world2vec)
        self.last_room_check_time = time.time()
        self.wall_sub = self.create_subscription(
            String,
             '/detected_wall_segments',
            self.walls_callback,
            10
        )
        from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

        qos_poly = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE, # o BEST_EFFORT se la rete è lenta
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )

        self.poly_sub = self.create_subscription(
            String,
            '/room_polygons_sync',
            self.poly_callback,
            qos_poly # Usa il profilo invece di un semplice intero
        )
        qos_latch = QoSProfile(depth=10, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        qos_standard = QoSProfile(depth=10)
        
        # Publishers
        # Publishers
        self.persistent_bbox_pub = self.create_publisher(MarkerArray, '/persistent_bbox', qos_latch)
        self.persistent_centroids_pub = self.create_publisher(MarkerArray, '/persistent_centroids', qos_latch)
        self.considered_volume_pub = self.create_publisher(MarkerArray, '/considered_volume', qos_latch)
        self.uncertain_bboxes_pub = self.create_publisher(MarkerArray, '/uncertain_object', qos_latch)
        self.uncertain_centroids_pub = self.create_publisher(MarkerArray, '/uncertain_centroids', qos_latch)
        self.tracking_activated_pub = self.create_publisher(Bool, '/tracking_mode_activated', qos_standard)
        
        # ---> AGGIUNGI QUESTA RIGA QUI SOTTO <---
        self.room_area_pub = self.create_publisher(MarkerArray, '/room_areas_array', qos_latch)
        
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
        self.get_logger().info('Object Tracking Service ready')

        # Subscribers
        self.create_subscription(Bool, "/robot_movement_detected", self.movement_callback, qos_standard)

        self.create_subscription(ObjectDescriptionArray, '/object_descriptions', self._descriptions_callback, qos_standard)
        self.create_subscription(Bbox3dArray, '/bbox_3d', self._bboxes_callback, qos_standard)

        self._bbox_timer = self.create_timer(2.0, self.periodic_bbox_publisher)
        self._uncertain_cleanup_timer = self.create_timer(5.0, self._cleanup_uncertain_by_time)
        # Aggiungilo vicino a self.wall_sub
        self.poly_sub = self.create_subscription(
            String,
            '/room_polygons_sync',
            self.poly_callback,
            10
        )
    
    def log_operation(self, message):
        timestamp = datetime.now().strftime('%H:%M:%S')
        with open(SYNTHETIC_LOG_FILE, "a") as f:
            f.write(f"[{timestamp}] {message}\n")

    def log_both(self, level, message):
        if level == 'info':
            self.get_logger().info(message)
        elif level == 'warn':
            self.get_logger().warn(message)
        elif level == 'error':
            self.get_logger().error(message)
        elif level == 'debug':
            self.get_logger().debug(message)

        if level in ['info', 'warn']:
            prefix = "[INFO] " if level == 'info' else "[WARN] "
            print(f"{prefix}{message}")

    def movement_callback(self, msg):
        # Sincronizza lo stato reale: True se si muove, False se è fermo
        self.robot_has_moved = msg.data
        if msg.data:
            self._topic_descriptions = None
            self._topic_bboxes = None
            self.latest_bboxes.clear()
            self.log_both('warn', "[MOVEMENT] Robot is moving -> Blocco stanze attivato")
        else:
            self.log_both('info', "[MOVEMENT] Robot has stopped -> Creazione stanze permessa")

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
            print(f"🔍 [BEST MATCH FOUND] Rilevato: '{label_base}' -> Best Memoria: '{best_match.label}' (Score: {highest_similarity:.3f})")
            
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
                self.log_both('warn', f"🔴 [TRACKING TRANSITION] Object '{best_match.label}' is the best match but moved! (Dist: {distance:.2f}m), the iou was {iou}")
                return True, best_match, distance
        
        return False, None, 0.0


    def object_tracking_callback(self, request, response):
        if not self.exploration_mode:
            self.tracking_step_counter += 1

        if self.robot_has_moved:
            self.log_both('warn', "Robot in movimento — dati scartati da object_tracking_callback")
            response.status = "moving"
            response.num_objects = len(wm.persistent_perceptions)
            response.tracking_mode_activated = False
            return response
        in_exploration = self.exploration_mode
        current_perception_objects = []
        objects_modified = False
        tracking_activated = False

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

        # --- VALUTAZIONE STANZA PERIODICA (SOLO PER NOME E DESCRIZIONE) ---
        current_time = time.time()
        
        if getattr(self, 'last_room_check_time', 0) == 0:
             self.last_room_check_time = current_time

        # Valuta la scena ogni 5 secondi per aggiornare la semantica, SENZA cercare porte
        if (current_time - self.last_room_check_time) > 5.0:
            if len(request.descriptions.descriptions) > 0:
                old_room_id = self.room_manager.current_room_id
                
                self.room_manager.evaluate_scene(request.descriptions.descriptions, wm.persistent_perceptions)
                
                # Se per cause esterne (es. walls_callback) l'ID è cambiato, avvisa Perception
                if self.room_manager.current_room_id != old_room_id:
                    from std_msgs.msg import String 
                    room_msg = String()
                    room_msg.data = self.room_manager.current_room_id
                    self.room_pub.publish(room_msg)
                    self.log_both('info', f"🚪 Cambio stanza rilevato! Inviato segnale a Perception per: {self.room_manager.current_room_id}")
                
            self.last_room_check_time = current_time
        # -------------------------------------------------------------------
    
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
                    self.log_both('warn', f"🔴 [TRANSITION] Switching from EXPLORATION to TRACKING mode")
                    self.exploration_mode = False
                    self.tracking_step_counter = 1
                    tracking_activated = True
                    in_exploration = False
                    self.exploration_frame_counter = 0
                    
                    msg = Bool()
                    msg.data = True
                    self.tracking_activated_pub.publish(msg)
                    
                    updated_obj, dist, iou = self.modify_existing_object(obj, bbox, description_embedding)
                    current_perception_objects.append(updated_obj)
                    objects_modified = True
                    already_seen = True
                    continue

                for obj in wm.persistent_perceptions:
                    obj_label_base = obj.label.split('#')[0] if '#' in obj.label else obj.label
                    if not hasattr(obj, "embedding"):
                        obj.embedding = get_embedding(world2vec, obj.description)
                    if obj.embedding is None:
                        continue
                    
                    similarity = lost_similarity(world2vec, label_base, obj_label_base, color, obj.color,
                                            material, obj.material, description_embedding, obj.embedding)
                    
                    if obj.bbox is not None:
                        iou = compute_iou_3d(bbox, obj.bbox)
                        print(f"DEBUG-EXPLORATION: Comparing current {label_base} with stored {obj.label}")
                        print(f"   -> Similarity: {similarity:.3f} (Soglia: {SIM_THRESHOLD})")
                        print(f"   -> IoU 3D: {iou:.3f} (Soglia: {EXPLORATION_IOU_THRESHOLD})")

                        # FIX: se UNA QUALSIASI descrizione è "unknown", usa solo IoU
                        current_is_unknown = (description_text.lower() == "unknown")
                        stored_is_unknown = (obj.description.lower() == "unknown")
                        
                        if (current_is_unknown or stored_is_unknown) and iou >= EXPLORATION_IOU_THRESHOLD:
                            print(f"   -> RISULTATO: Oggetto CONFERMATO (match spaziale, una descrizione è 'unknown').")
                            already_seen = True
                            current_perception_objects.append(obj)
                            obj.bbox = bbox
                            # Aggiorna la description se quella in memoria è "unknown"
                            if stored_is_unknown and not current_is_unknown:
                                obj.description = description_text
                                obj.color = color
                                obj.material = material
                                obj.embedding = description_embedding
                            break

                        if similarity > SIM_THRESHOLD and iou >= EXPLORATION_IOU_THRESHOLD:
                            print(f"   -> RISULTATO: Oggetto CONFERMATO nella stessa posizione.")
                            already_seen = True
                            current_perception_objects.append(obj)
                            obj.bbox = bbox
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
                    similarity = lost_similarity(world2vec, label_base, obj_label_base, color, obj.color,
                                            material, obj.material, description_embedding, obj.embedding)

                    # Se una delle due descrizioni è "unknown", usa match spaziale
                    if description_text.lower() == "unknown" or obj.description.lower() == "unknown":
                        if obj.bbox is not None:
                            iou = compute_iou_3d(bbox, obj.bbox)
                            if iou >= TRACKING_IOU_THRESHOLD:
                                similarity = 1.0  # forza match

                    if similarity > SIM_THRESHOLD and similarity > best_score:
                        best_score = similarity
                        best_match = obj

                if best_match:
                    already_seen = True
                    updated_obj, distance, iou = self.modify_existing_object(best_match, bbox, description_embedding)
                    current_perception_objects.append(updated_obj)
                    objects_modified = True

            if not already_seen:
                new_obj = self.add_new_object(label, bbox, description_text, color, material,
                                             description_embedding, in_exploration)
                current_perception_objects.append(new_obj)
                objects_modified = True

        pov_volume = getattr(self, 'latest_fov_volume', None)

        if not in_exploration:
            if self.robot_has_moved:
                pass

            if pov_volume:
                publish_pov_volume(self, pov_volume, self.considered_volume_pub)
                scaled_pov_volume = shrink_pov_volume(pov_volume, POV_SCALE_FACTOR)
                
                detected_labels = {obj.label for obj in current_perception_objects}
                
                to_remove = []
                for obj in list(wm.persistent_perceptions):
                    if bbox_centroid_in_volume(obj.bbox, scaled_pov_volume):
                        if obj.label not in detected_labels:
                            to_remove.append(obj)

                if to_remove:
                    for obj in to_remove:
                        self.get_logger().warn(f"🗑️ [DELETE] L'oggetto '{obj.label}' è sparito dal FOV. Rimozione dalla memoria.")
                        wm.persistent_perceptions.remove(obj)
                        if hasattr(self, 'db'):
                            self.db.on_object_deleted(obj, reason="sparito dal FOV", step=self.tracking_step_counter)
                        self.log_operation(f"🗑️ ELIMINATO: '{obj.label}' (room: {getattr(obj, 'room_id', 'unknown')}) - sparito dal FOV")
                    objects_modified = True
            # Pulisci anche gli uncertain_objects se rientrano nel FOV
            self.delete_uncertain_objects(pov_volume)

        if objects_modified:
            publish_persistent_bboxes(self, wm, self.persistent_bbox_pub)
            publish_persistent_centroids(self, wm, self.persistent_centroids_pub)
            publish_uncertain_bboxes(self, self.uncertain_objects, self.uncertain_bboxes_pub)
            publish_uncertain_centroids(self, self.uncertain_objects, self.uncertain_centroids_pub)
            save_uncertain_objects(self)
            save_persistent_perceptions(self)  # <--- AGGIUNGI QUESTA RIGA QUI
            
        self.latest_bboxes.clear()
        
        self.merge_duplicate_objects()
        
        response.status = "tracking_activated" if tracking_activated else "success"
        response.num_objects = len(wm.persistent_perceptions)
        response.tracking_mode_activated = tracking_activated

        return response
    
    def add_new_object(self, label, bbox, description_text, color, material, description_embedding, in_exploration):
        new_obj = Object(label, None, bbox, description_text, color, material)
        new_obj.embedding = description_embedding
        new_obj.creation_time = time.time()
        
        # --- NUOVO: Assegnazione Spaziale ---
        # Forza l'assegnazione alla stanza attuale del robot
        assigned_room = self.room_manager.current_room_id 
        new_obj.room_id = assigned_room
        
        # Aggiorna la geometria della stanza attuale
        self.room_manager.update_room_geometry(assigned_room, bbox)
        
        if new_obj.label not in self.room_manager.scene_graph[assigned_room]["objects"]:
            self.room_manager.scene_graph[assigned_room]["objects"].append(new_obj.label)
        # ------------------------------------
        
        wm.persistent_perceptions.append(new_obj)

        phase = "exploration" if in_exploration else "tracking"
        step = self.exploration_step_counter if in_exploration else self.tracking_step_counter
        
        self.db.on_new_object(new_obj, phase=phase, step=step)

        x_size = bbox["x_max"] - bbox["x_min"]
        y_size = bbox["y_max"] - bbox["y_min"]
        z_size = bbox["z_max"] - bbox["z_min"]
        volume = x_size * y_size * z_size

        mode_tag = "[EXPLORATION]" if in_exploration else f"[TRACKING STEP {self.tracking_step_counter}]"
        self.log_both('info', f"{mode_tag} New object '{label}' in {new_obj.room_id} (vol: {volume:.3f} m³)")
        
        save_persistent_perceptions(self)

        if in_exploration:
            self.exploration_step_counter += 1
        # --- LOG OPERAZIONI: NUOVO OGGETTO ---
        try:
            with open('/root/exchange/output/operations.txt', 'a') as f:
                timestamp = datetime.now().strftime('%H:%M:%S')
                room_id = self.room_manager.current_room_id if hasattr(self, 'room_manager') else "Unknown"
                cx = (bbox["x_min"] + bbox["x_max"]) / 2.0
                cy = (bbox["y_min"] + bbox["y_max"]) / 2.0
                cz = (bbox["z_min"] + bbox["z_max"]) / 2.0
                f.write(f"[{timestamp}] 🟢 AGGIUNTO: {label} in {room_id} a pos({cx:.2f}, {cy:.2f}, {cz:.2f})\n")
        except Exception as e:
            self.get_logger().error(f"Impossibile scrivere su operations.txt: {e}")

        return new_obj

    def modify_existing_object(self, best_match, bbox, description_embedding):
        if "door" in best_match.label.lower():
            best_match.bbox = bbox
            return best_match, 0.0, 1.0

        old_bbox = best_match.bbox
        iou = compute_iou_3d(bbox, old_bbox)

        old_x = (old_bbox['x_min'] + old_bbox['x_max']) / 2.0
        old_y = (old_bbox['y_min'] + old_bbox['y_max']) / 2.0
        old_z = (old_bbox['z_min'] + old_bbox['z_max']) / 2.0
        new_x = (bbox['x_min'] + bbox['x_max']) / 2.0
        new_y = (bbox['y_min'] + bbox['y_max']) / 2.0
        new_z = (bbox['z_min'] + bbox['z_max']) / 2.0
        distance = np.sqrt((new_x - old_x)**2 + (new_y - old_y)**2 + (new_z - old_z)**2)

        # FIX: Sotto i 0.5m aggiorna solo il bbox senza notificare lo spostamento (filtro rumore)
        if distance < 0.5 or iou >= TRACKING_IOU_THRESHOLD:
            best_match.bbox = bbox
            self.room_manager.update_room_geometry(getattr(best_match, 'room_id', self.room_manager.current_room_id), bbox)
            return best_match, distance, iou

        if (time.time() - getattr(best_match, 'creation_time', 0)) < OBJECT_STABILITY_TIMEOUT:
            best_match.bbox = bbox 
            return best_match, distance, iou

        else:
            if best_match in wm.persistent_perceptions:
                wm.persistent_perceptions.remove(best_match)
                self.db.on_object_moved(best_match, old_bbox=best_match.bbox, new_bbox=bbox,
                                       distance=distance, iou=iou, step=self.tracking_step_counter)

            if distance > 0.8:
                if best_match not in self.uncertain_objects:
                    self.uncertain_objects.append(best_match)
                    self.db.on_uncertain_added(best_match, step=self.tracking_step_counter)

            updated_obj = Object(best_match.label, None, bbox, best_match.description,
                               best_match.color, best_match.material)
            updated_obj.embedding = description_embedding
            
            new_room = self.room_manager.assign_room_by_geometry(bbox)
            updated_obj.room_id = new_room
            self.room_manager.update_room_geometry(new_room, bbox)
            
            if updated_obj.label not in self.room_manager.scene_graph[new_room]["objects"]:
                self.room_manager.scene_graph[new_room]["objects"].append(updated_obj.label)
                
            wm.persistent_perceptions.append(updated_obj)
            self.log_operation(f"[SPOSTAMENTO] '{best_match.label}' si è mosso di {distance:.2f}m")

            return updated_obj, distance, iou

    
    def delete_undetected_objects(self, pov_volume, current_perception_objects, description_received):
        # 1. CONTROLLO VOLUME CON ALLARME
        if not pov_volume:
            print("❌ ERRORE CRITICO TF: pov_volume è vuoto! Il robot non sa dove sta guardando (Controlla il frame in lookup_transform). Cancellazione annullata.")
            return False

        # Se arriva qui, il volume funziona! Lo pubblichiamo per visualizzarlo su RViz
        try:
            publish_pov_volume(self, pov_volume, self.considered_volume_pub)
        except Exception as e:
            print(f"⚠️ Impossibile pubblicare il volume visivo: {e}")

        objects_to_remove = []

        # 2. LOGICA DI CANCELLAZIONE SPAZIALE (Come volevi tu)
        for obj in list(wm.persistent_perceptions):
            
            # Se lo vediamo in questo momento, azzeriamo l'incertezza
            if obj in current_perception_objects:
                obj.not_seen_in_pov_frames = 0
                continue
                
            # Se l'oggetto NON è visto, ma STIAMO guardando nel suo volume
            if obj.bbox and bbox_centroid_in_volume(obj.bbox, pov_volume):
                
                if not hasattr(obj, 'not_seen_in_pov_frames'):
                    obj.not_seen_in_pov_frames = 0
                    
                obj.not_seen_in_pov_frames += 1
                
                # Se abbiamo guardato lì per 5 frame e non c'è, è sparito davvero
                if obj.not_seen_in_pov_frames >= 5:
                    objects_to_remove.append(obj)

        # 3. RIMOZIONE FISICA
        if objects_to_remove:
            for obj in objects_to_remove:
                print(f"🗑️ [CANCELLATO] L'oggetto '{obj.label}' non è più presente nel volume osservato! RIMOSSO.")
                
                if obj in wm.persistent_perceptions:
                    wm.persistent_perceptions.remove(obj)
                
                if hasattr(self, 'db'):
                    self.db.on_object_deleted(obj, reason="not seen in POV", step=self.tracking_step_counter)

            save_persistent_perceptions(self)
            return True
            
        return False

    def delete_uncertain_objects(self, pov_volume):
        uncertain_to_remove = []

        if pov_volume:
            for uncertain_obj in self.uncertain_objects:
                if uncertain_obj.bbox and bbox_centroid_in_volume(uncertain_obj.bbox, pov_volume):
                    uncertain_to_remove.append(uncertain_obj)

        if uncertain_to_remove:
            for uncertain_obj in uncertain_to_remove:
                self.uncertain_objects.remove(uncertain_obj)
            return True
        return False
    
    def _cleanup_uncertain_by_time(self):
        now = time.time()
        expiry = 120.0  # secondi
        to_remove = [obj for obj in self.uncertain_objects
                     if now - getattr(obj, 'creation_time', 0) > expiry]
        if to_remove:
            for obj in to_remove:
                self.uncertain_objects.remove(obj)
                self.log_both('info', f"🧹 [UNCERTAIN CLEANUP] Rimosso '{obj.label}' (tempo scaduto)")
    
    def _descriptions_callback(self, msg):
        self._topic_descriptions = msg
        self._try_process()

    def _bboxes_callback(self, msg):
        self._topic_bboxes = msg
        self._try_process()

    def _try_process(self):
        if self.robot_has_moved:
            self._topic_descriptions = None
            self._topic_bboxes = None
            return

        if getattr(self, '_topic_descriptions', None) is None or getattr(self, '_topic_bboxes', None) is None:
            return

        descriptions = self._topic_descriptions
        bboxes = self._topic_bboxes

        self._topic_descriptions = None
        self._topic_bboxes = None

        request = ObjectTrackingService.Request()
        request.descriptions = descriptions
        request.bboxes = bboxes

        response = ObjectTrackingService.Response()
        self.object_tracking_callback(request, response)

    def check_polygon_overlap(self, poly1, poly2):
        from shapely.geometry import Polygon
        try:
            p1 = Polygon(poly1)
            p2 = Polygon(poly2)
            
            if not p1.is_valid or not p2.is_valid:
                return 0.0 # Nel dubbio restituisci 0, non 1, altrimenti crea falsi ritorni!
                
            intersection_area = p1.intersection(p2).area
            area_p1 = p1.area 
            
            if area_p1 == 0: return 0.0
            
            # --- FIX: Intersezione fratto l'area della vista corrente ---
            # Se la fetta che vedo è tutta dentro room_0, restituirà 1.0 (100%)
            return intersection_area / area_p1 
            
        except Exception as e:
            print(f"Errore calcolo overlap: {e}")
            return 0.0
    

    def walls_callback(self, msg):
        try:
            new_walls = json.loads(msg.data)
            for w in new_walls:
                start_x, start_y = w["start"]["x"], w["start"]["y"]
                end_x, end_y = w["end"]["x"], w["end"]["y"]
                self.room_manager.current_room_walls.append([start_x, start_y, end_x, end_y])
        except Exception as e:
            print(f"🔴 Errore nel salvataggio muri: {e}")
        
    def poly_callback(self, msg):
        try:
            data = json.loads(msg.data)
            vertices = data["polygon"]
            
            if len(vertices) < 3:
                return

            import numpy as np
            from shapely.geometry import Polygon

            # Calcoliamo le dimensioni dell'area esatta passata da Perception
            pts_array = np.array(vertices)
            min_x, max_x = np.min(pts_array[:, 0]), np.max(pts_array[:, 0])
            min_y, max_y = np.min(pts_array[:, 1]), np.max(pts_array[:, 1])
            width_x = max_x - min_x
            width_y = max_y - min_y
            area = Polygon(vertices).area

            # Filtro per evitare che frammenti minuscoli facciano impazzire la logica
            if area > 10.0 and width_x > 2.0 and width_y > 2.0:
                best_overlap = 0.0
                best_room_id = None
                
                for r_id, r_data in self.room_manager.scene_graph.items():
                    poly = r_data.get("polygon", [])
                    if len(poly) >= 3:
                        overlap = self.check_polygon_overlap(vertices, poly)
                        if overlap > best_overlap:
                            best_overlap = overlap
                            best_room_id = r_id
                            
                print(f"📊 MAX OVERLAP: {best_overlap*100:.2f}% con {best_room_id if best_room_id else 'Nessuna'}")
                if best_overlap >= 0.40:
                    # --- NUOVO CONTROLLO MOVIMENTO ---
                    if getattr(self, 'robot_has_moved', False):
                        # Se il robot si muove, non aggiorniamo per non deformare con dati sporchi
                        pass
                    else:
                        if best_room_id == self.room_manager.current_room_id:
                            # --- FIX DEFORMAZIONE: Facciamo l'unione dei poligoni ---
                            try:
                                old_poly = Polygon(self.room_manager.scene_graph[best_room_id]["polygon"])
                                new_poly = Polygon(vertices)
                                if old_poly.is_valid and new_poly.is_valid:
                                    merged_poly = old_poly.union(new_poly).convex_hull
                                    if merged_poly.geom_type == 'Polygon':
                                        self.room_manager.scene_graph[best_room_id]["polygon"] = list(merged_poly.exterior.coords)
                            except Exception as e:
                                print(f"Errore unione poligoni: {e}")
                        else:
                            print(f"🔄 [RITORNO] Bentornato in {best_room_id}!")
                            self.room_manager.current_room_id = best_room_id
                            
                            from std_msgs.msg import String 
                            room_msg = String()
                            room_msg.data = best_room_id
                            self.room_pub.publish(room_msg)
                else:
                    if not any(d.get("polygon") for d in self.room_manager.scene_graph.values()):
                        print("🏠 Prima stanza rilevata. Imposto il poligono esatto in memoria.")
                        self.room_manager.scene_graph[self.room_manager.current_room_id]["polygon"] = vertices
                    else:
                        # --- NUOVO CONTROLLO MOVIMENTO INSERITO QUI ---
                        if getattr(self, 'robot_has_moved', False):
                            print("🛑 [BLOCCO] Ignoro la nuova stanza: il robot è in movimento!")
                        else:
                            print(f"🚀 [CAMBIO STANZA] Overlap basso. Triggering room transition!")
                            self.trigger_room_transition(vertices)

        except Exception as e:
            print(f"🔴 Errore in poly_callback: {e}")
            
    def publish_single_room_area(self, room_id, poly):
        """
        Pubblica il poligono di una singola stanza sul suo topic dedicato
        mantenendolo visibile su RViz grazie al QoS TRANSIENT_LOCAL.
        """
        from visualization_msgs.msg import Marker
        from geometry_msgs.msg import Point
        from rclpy.qos import QoSProfile, DurabilityPolicy

        # Inizializza il dizionario dei publisher se non esiste
        if not hasattr(self, 'room_area_publishers'):
            self.room_area_publishers = {}

        # Crea un publisher dedicato per questa stanza con QoS TRANSIENT_LOCAL
        if room_id not in self.room_area_publishers:
            qos_profile = QoSProfile(
                depth=10,
                durability=DurabilityPolicy.TRANSIENT_LOCAL
            )
            topic_name = f'/room_areas/{room_id}'
            self.room_area_publishers[room_id] = self.create_publisher(
                Marker, 
                topic_name, 
                qos_profile
            )

        m = Marker()
        m.header.frame_id = "map"
        m.header.stamp = self.get_clock().now().to_msg()
        m.ns = "room_polygons"
        
        # Assegna un ID univoco basato sul numero della stanza (es. room_0 -> 0)
        try:
            # Estrae l'ultimo numero dal nome della stanza
            import re
            match = re.search(r'\d+', room_id)
            m.id = int(match.group()) if match else hash(room_id) % 10000
        except Exception:
            m.id = hash(room_id) % 10000
            
        m.type = Marker.LINE_STRIP
        m.action = Marker.ADD
        m.pose.orientation.w = 1.0
        m.scale.x = 0.08  # Spessore della linea del perimetro

        # Genera un colore deterministico ma diverso per ogni stanza
        room_num = m.id
        m.color.r = float((room_num * 123 % 255) / 255.0)
        m.color.g = float((room_num * 456 % 255) / 255.0)
        m.color.b = float((room_num * 789 % 255) / 255.0)
        m.color.a = 0.8  # Leggera trasparenza

        # Inserimento dei vertici del poligono
        if len(poly) > 0:
            for pt in poly:
                p = Point()
                # pt[0] è x, pt[1] è y. Alziamo leggermente z (0.05) per evitare z-fighting col suolo
                p.x, p.y, p.z = float(pt[0]), float(pt[1]), 0.05
                m.points.append(p)
                
            # Chiude il loop tornando al primo punto
            p_start = Point()
            p_start.x, p_start.y, p_start.z = float(poly[0][0]), float(poly[0][1]), 0.05
            m.points.append(p_start)

        # Pubblica sul topic specifico della stanza
        self.room_area_publishers[room_id].publish(m)
        self.get_logger().info(f"✅ Pubblicata area aggiornata per {room_id} su /room_areas/{room_id}")

        
    def trigger_room_transition(self, new_room_polygon):
        """Gestisce il passaggio a una nuova stanza quando non c'è overlap"""
        import time
        current_time = time.time()
        
        # Evitiamo transizioni troppo frequenti (debounce di 10 secondi)
        if (current_time - getattr(self, 'last_room_transition_time', 0)) > 10.0:
            old_room_id = self.room_manager.current_room_id
            
            # --- CALCOLO NUOVO ID ROBUSTO (0, 1, 2...) ---
            # Guardiamo le stanze esistenti nel scene_graph per decidere il prossimo numero
            existing_ids = self.room_manager.scene_graph.keys()
            numeric_ids = []
            for s in existing_ids:
                nums = re.findall(r'\d+', s)
                if nums:
                    numeric_ids.append(int(nums[0]))
            
            # Se abbiamo room_0, il max è 0, quindi il prossimo è 1.
            next_id_val = max(numeric_ids) + 1 if numeric_ids else 1
            new_room_id = f"room_{next_id_val}"
            # ----------------------------------------------

            self.get_logger().info(f"🧱 [WALL DETECTED] Esco dalla stanza {old_room_id}. Creazione {new_room_id}...")
            
            # 1. Salva e finalizza gli oggetti della stanza precedente
            self.room_manager.finalize_current_room(wm.persistent_perceptions)

            # 2. Aggiorna lo stato del Room Manager
            self.room_manager.room_counter = next_id_val # Sincronizziamo il contatore interno
            self.room_manager.current_room_id = new_room_id
            self.room_manager.init_room_node(new_room_id)
            
            # 3. Assegna il poligono (mura chiuse) alla nuova stanza
            self.room_manager.scene_graph[new_room_id]["polygon"] = new_room_polygon
            self.room_manager.last_closed_room_polygon = new_room_polygon
            
            # 4. Pulisci i dati temporanei dei muri per la nuova sessione
            self.room_manager.current_room_walls = []
            self.room_manager.current_room_wall_segments = []
            
            self.last_room_transition_time = current_time
            self.get_logger().info(f"✨ [CAMBIO STANZA] Inizio mappatura {new_room_id}")
            
            # 5. Notifica PERCEPTION del cambio stanza (fondamentale per non trascinare i muri)
            from std_msgs.msg import String 
            room_msg = String()
            room_msg.data = new_room_id
            self.room_pub.publish(room_msg)

            
    def periodic_bbox_publisher(self):
        if self.robot_has_moved:
           return
        # 1. (lascia invariata la pubblicazione degli oggetti) ...
        if len(wm.persistent_perceptions) > 0:
            publish_persistent_bboxes(self, wm, self.persistent_bbox_pub)
            publish_persistent_centroids(self, wm, self.persistent_centroids_pub)

        if len(self.uncertain_objects) > 0:
            publish_uncertain_bboxes(self, self.uncertain_objects, self.uncertain_bboxes_pub)
            publish_uncertain_centroids(self, self.uncertain_objects, self.uncertain_centroids_pub)
            
        # 2. DISEGNA LE PIANTINE DELLE STANZE SU RVIZ
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
                m.scale.x = 0.15 # Spessore della linea del perimetro
                m.color.r, m.color.g, m.color.b, m.color.a = 0.0, 1.0, 0.5, 1.0 
                
                # Disegna i bordi esatti della stanza
                for pt in poly:
                    p = Point()
                    p.x, p.y, p.z = float(pt[0]), float(pt[1]), 0.05
                    m.points.append(p)
                    
                # Chiudi il poligono ritornando al primo punto
                p = Point()
                p.x, p.y, p.z = float(poly[0][0]), float(poly[0][1]), 0.05
                m.points.append(p)
                
                msg.markers.append(m)
                
        if hasattr(self, 'room_area_pub'):
            self.room_area_pub.publish(msg)
    
    def merge_duplicate_objects(self):
        """Unisce oggetti in persistent_perceptions che sono vicini spazialmente
        e hanno alta similarità semantica."""
        MAX_DISTANCE = 0.8
        MIN_SIMILARITY = 0.75
        merged_any = False

        objects = list(wm.persistent_perceptions)
        to_remove = set()

        print("══════════════════════════════════════════════")
        print(f"🔍 MERGE CHECK: {len(objects)} oggetti in memoria")
        print("══════════════════════════════════════════════")

        for i in range(len(objects)):
            if objects[i] in to_remove:
                continue
            for j in range(i + 1, len(objects)):
                if objects[j] in to_remove:
                    continue

                a, b = objects[i], objects[j]
                if a.bbox is None or b.bbox is None:
                    continue

                # Centroids
                ax = (a.bbox['x_min'] + a.bbox['x_max']) / 2.0
                ay = (a.bbox['y_min'] + a.bbox['y_max']) / 2.0
                az = (a.bbox['z_min'] + a.bbox['z_max']) / 2.0
                bx = (b.bbox['x_min'] + b.bbox['x_max']) / 2.0
                by = (b.bbox['y_min'] + b.bbox['y_max']) / 2.0
                bz = (b.bbox['z_min'] + b.bbox['z_max']) / 2.0

                a_label = a.label.split('#')[0] if '#' in a.label else a.label
                b_label = b.label.split('#')[0] if '#' in b.label else b.label

                print(f"\n📐 CONFRONTO [{i}]{a_label} vs [{j}]{b_label}:")
                print(f"   Pos A: ({ax:.2f}, {ay:.2f}, {az:.2f})")
                print(f"   Pos B: ({bx:.2f}, {by:.2f}, {bz:.2f})")

                # Semantic similarity (check first)
                if not hasattr(a, 'embedding') or a.embedding is None:
                    a.embedding = get_embedding(world2vec, a.description)
                if not hasattr(b, 'embedding') or b.embedding is None:
                    b.embedding = get_embedding(world2vec, b.description)
                if a.embedding is None or b.embedding is None:
                    print(f"   ⚠️ Embedding mancante, skip")
                    continue

                sim = lost_similarity(world2vec, a_label, b_label,
                                    a.color, b.color,
                                    a.material, b.material,
                                    a.embedding, b.embedding)

                print(f"   Sim semantiche:")
                print(f"     Label: '{a_label}' vs '{b_label}'")
                print(f"     Colore: '{a.color}' vs '{b.color}'")
                print(f"     Materiale: '{a.material}' vs '{b.material}'")
                print(f"     Similarità: {sim:.3f} (soglia: {MIN_SIMILARITY})")

                if sim < MIN_SIMILARITY:
                # Fallback: se stessa label e alto overlap spaziale, fonde comunque
                    if a_label == b_label:
                        iou = compute_iou_3d(a.bbox, b.bbox)
                        if iou >= 0.5:
                            print(f"   ⚠️ Stessa label + IoU alto ({iou:.3f}), forzo merge")
                            sim = 1.0
                    if sim < MIN_SIMILARITY:
                        print(f"   ❌ SIMILARITÀ BASSA ({sim:.2f} < {MIN_SIMILARITY})")
                        continue

                # Distance check (second)
                dist = np.sqrt((ax - bx)**2 + (ay - by)**2 + (az - bz)**2)
                print(f"   Distanza: {dist:.3f}m (soglia: {MAX_DISTANCE}m)")

                if dist > MAX_DISTANCE:
                    print(f"   ❌ TROPPO LONTANI ({dist:.2f}m > {MAX_DISTANCE}m)")
                    continue

                # Merge bboxes: media
                merged_bbox = {
                    'x_min': (a.bbox['x_min'] + b.bbox['x_min']) / 2.0,
                    'x_max': (a.bbox['x_max'] + b.bbox['x_max']) / 2.0,
                    'y_min': (a.bbox['y_min'] + b.bbox['y_min']) / 2.0,
                    'y_max': (a.bbox['y_max'] + b.bbox['y_max']) / 2.0,
                    'z_min': (a.bbox['z_min'] + b.bbox['z_min']) / 2.0,
                    'z_max': (a.bbox['z_max'] + b.bbox['z_max']) / 2.0,
                }

                # Keep the one with better description
                a_unknown = a.description.lower() == 'unknown'
                b_unknown = b.description.lower() == 'unknown'
                if a_unknown and not b_unknown:
                    keeper, discard = b, a
                elif b_unknown and not a_unknown:
                    keeper, discard = a, b
                else:
                    keeper, discard = a, b

                keeper.bbox = merged_bbox
                to_remove.add(discard)

                vol_a = ((a.bbox['x_max'] - a.bbox['x_min']) *
                        (a.bbox['y_max'] - a.bbox['y_min']) *
                        (a.bbox['z_max'] - a.bbox['z_min']))
                vol_b = ((b.bbox['x_max'] - b.bbox['x_min']) *
                        (b.bbox['y_max'] - b.bbox['y_min']) *
                        (b.bbox['z_max'] - b.bbox['z_min']))
                vol_m = ((merged_bbox['x_max'] - merged_bbox['x_min']) *
                        (merged_bbox['y_max'] - merged_bbox['y_min']) *
                        (merged_bbox['z_max'] - merged_bbox['z_min']))

                print(f"   ✅ MERGE!")
                print(f"     Volume A: {vol_a:.3f}m³ | Volume B: {vol_b:.3f}m³ → Media: {vol_m:.3f}m³")
                print(f"     Tenuto: '{keeper.label}' | Rimosso: '{discard.label}'")
                print(f"     Desc keeper: '{keeper.description[:40]}...'")
                print(f"     Bbox unito: x[{merged_bbox['x_min']:.2f},{merged_bbox['x_max']:.2f}] "
                    f"y[{merged_bbox['y_min']:.2f},{merged_bbox['y_max']:.2f}] "
                    f"z[{merged_bbox['z_min']:.2f},{merged_bbox['z_max']:.2f}]")

        if to_remove:
            print(f"\n🗑️ RIMOZIONE: {len(to_remove)} oggetti duplicati:")
            for obj in to_remove:
                print(f"   - {obj.label} @ "
                    f"({(obj.bbox['x_min']+obj.bbox['x_max'])/2:.2f}, "
                    f"{(obj.bbox['y_min']+obj.bbox['y_max'])/2:.2f})")
                wm.persistent_perceptions.remove(obj)
            save_persistent_perceptions(self)
            publish_persistent_bboxes(self, wm, self.persistent_bbox_pub)
            publish_persistent_centroids(self, wm, self.persistent_centroids_pub)
            merged_any = True
        else:
            print(f"\n✅ NESSUN duplicato trovato.")

        print("══════════════════════════════════════════════\n")
        return merged_any
def main(args=None):
    rclpy.init(args=args)
    service_node = ObjectManagerService()

    try:
        rclpy.spin(service_node)
    except KeyboardInterrupt:
        from datetime import datetime
        print(f"\nOBJECT MANAGER SERVICE chiuso ({datetime.now().strftime('%Y-%m-%d %H:%M:%S')})")
        print("Salvataggio dell'ultima stanza in corso...")
        
        # Salva i dati della stanza corrente usando i persistent_perceptions globali
        if hasattr(service_node, 'room_manager'):
            service_node.room_manager.finalize_current_room(wm.persistent_perceptions)
        
        # Forza anche un ultimo salvataggio globale delle percezioni
        save_persistent_perceptions(service_node)

    finally:
        service_node.destroy_node()
        # --- MODIFICA QUESTA PARTE ---
        if rclpy.ok():
            rclpy.shutdown()
        # -----------------------------

if __name__ == '__main__':
    main()