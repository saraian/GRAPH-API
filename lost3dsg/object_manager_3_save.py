#!/usr/bin/env python3
"""
Object Manager Service - Semantic and Spatial Tracking of Perceived Objects
Tracks objects and automatically transitions from EXPLORATION to TRACKING when
an object is seen again in a different position.
Includes Topological Semantic Mapping (Room Manager) with Scene Graph generation.
"""
import rclpy, json, os, time
from rclpy.node import Node
from rclpy.qos import QoSProfile, DurabilityPolicy
import numpy as np
from openai import OpenAI
from lost3dsg.msg import ObjectDescriptionArray, Bbox3dArray
from lost3dsg.srv import ObjectTrackingService
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import Bool
from object_info import Object
from world_model import wm
from utils import *
from nlp_utils import *
from datetime import datetime
from cv_utils import *
from map_database import MapDatabase
from gensim.models import KeyedVectors

# =============  EXPLORATION PARAMETERS =============
EXPLORATION_IOU_THRESHOLD = 0.18
SIM_THRESHOLD = 0.75
TRACKING_IOU_THRESHOLD = 0.3
VOLUME_EXPANSION_RATIO = 0.01
EXPLORATION_FRAME_LIMIT = 10  # Numero di frame in exploration prima di passare a tracking
OBJECT_STABILITY_TIMEOUT = 5.0  # Secondi minimi di vita prima di poter essere 'MOVED'
POV_SCALE_FACTOR = 0.6 # Scale POV volume uniformly (0.8 = 20% smaller in all directions)

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
        if obj.bbox is None: continue
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
        if obj.bbox is None: continue
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
        if obj.bbox is None: continue
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
        if obj.bbox is None: continue
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
        self.w2v = w2v_model
        self.similarity_threshold = similarity_threshold
        self.room_counter = 1
        self.current_room_id = f"room_{self.room_counter}"
        
        # Cache pesi: se un oggetto non è qui, chiameremo il VLM
        self.weights_cache = {
            "chair": 0.2,
            "table": 0.5,
            "wall": 0.1,
            "door": 0.1 # Le porte sono trigger, non servono tanto semanticamente
        }
        
        # --- SCENE GRAPH ---
        self.scene_graph = {}
        self.init_room_node(self.current_room_id)

    def init_room_node(self, room_id):
        """Crea un nuovo nodo padre (Stanza) nel grafo."""
        self.scene_graph[room_id] = {
            "semantic_label": "Unknown",
            "embedding": None,
            "boundaries": {  # Limiti geometrici inizializzati a valori estremi
                "x_min": float('inf'), "x_max": float('-inf'),
                "y_min": float('inf'), "y_max": float('-inf'),
                "z_min": float('inf'), "z_max": float('-inf')
            },
            "objects": []  # Nodi figli (label o ID degli oggetti)
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
            b = data["boundaries"]
            # Controlliamo X e Y per tolleranza
            if (b["x_min"] <= cx <= b["x_max"]) and (b["y_min"] <= cy <= b["y_max"]):
                return r_id
                
        # Fallback: se non cade in nessuna stanza nota, lo assegna a quella corrente
        return self.current_room_id

    def ask_vlm_room_info(self, room_objects_labels):
        """Interroga il VLM per dare un nome semantico alla stanza e una descrizione."""
        if not room_objects_labels: return "Unknown_Room", ""
        labels_str = ", ".join(set(room_objects_labels))
        
        prompt = (
            f"Analizza questa lista di oggetti presenti in un ambiente: {labels_str}.\n"
            "Fornisci due informazioni in formato JSON puro:\n"
            "1. 'label': Il nome della stanza in inglese (es. Kitchen, Bedroom, Office, Bathroom, Living Room).\n"
            "2. 'description': Una breve descrizione della stanza in 1 o 2 frasi, dedotta dagli oggetti presenti.\n"
            "Rispondi ESCLUSIVAMENTE in formato JSON puro: {\"label\": \"nome\", \"description\": \"descrizione\"}"
        )
        try:
            with open(os.path.join(os.path.dirname(file_path), "api.txt"), "r") as f:
                 api_key = f.read().strip()
            client = OpenAI(api_key=api_key)
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                response_format={ "type": "json_object" }
            )
            data = json.loads(response.choices[0].message.content)
            return data.get("label", "Unknown_Room"), data.get("description", "")
        except Exception as e:
            print(f"🔴 ERRORE VLM Room Info: {e}")
            return "Unknown_Room", ""


    def get_object_weights_from_vlm(self, labels, encoded_image=None, vlm_callback=None):
        labels_str = ", ".join(labels)
        prompt = (
            f"Analizza questa lista di oggetti: {labels_str}.\n"
            "Assegna a ogni oggetto un peso da 0.1 a 3.0 per definire l'identità della stanza.\n"
            "- PESO ALTO (2.0 - 3.0): Oggetti unici (es. 'plant', 'fridge', 'cactus', 'monitor').\n"
            "- PESO BASSO (0.1 - 0.5): Oggetti generici (es. 'table', 'chair', 'book', 'wall').\n"
            "Rispondi ESCLUSIVAMENTE in formato JSON puro: {\"label\": peso}"
        )
        try:
            with open(os.path.join(os.path.dirname(file_path), "api.txt"), "r") as f:
                 api_key = f.read().strip()

            client = OpenAI(api_key=api_key)
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                response_format={ "type": "json_object" }
            )
            
            weights = json.loads(response.choices[0].message.content)
            print("\n⚖️  [VLM WEIGHTS] Nuovi pesi calcolati:")
            for k, v in weights.items():
                print(f"   - {k.upper()}: {float(v):.1f}")
            return weights
            
        except Exception as e:
            print(f"🔴 ERRORE VLM Weights: {e}")
            return {}

    def compute_scene_embedding(self, labels, dynamic_weights):
        vectors = []
        weights = []
        
        for label in labels:
            base_label = label.split('#')[0].lower().replace(" ", "_")
            if base_label in self.w2v:
                vectors.append(self.w2v[base_label])
                
                # Prendi il peso dal VLM. Fallback: cache statica, Fallback estremo: 1.0
                w = dynamic_weights.get(base_label, self.weights_cache.get(base_label, 1.0))
                weights.append(float(w))
        
        if not vectors: return None
        
        scene_vector = np.average(vectors, axis=0, weights=weights)
        return scene_vector / np.linalg.norm(scene_vector)

    def evaluate_scene(self, current_scene_objects, persistent_objects, encoded_image=None, vlm_callback=None):
        if not current_scene_objects: 
            return self.current_room_id
            
        raw_labels = [obj.label.lower() for obj in current_scene_objects]
        clean_labels = list(set([l.split('#')[0].replace(" ", "_") for l in raw_labels]))
        
        # 1. TRIGGER FISICO: DOOR DETECTION
        door_detected = any("door" in l for l in clean_labels)

        if door_detected:
            old_room = self.current_room_id
            print(f"🚪 [DOOR DETECTED] Trigger attivato. Finalizzazione stanza: {old_room}")

            # A. Recupera tutti gli oggetti
            room_objs = [o for o in persistent_objects if getattr(o, 'room_id', old_room) == old_room]
            room_labels = [o.label.split('#')[0].lower().replace(" ", "_") for o in room_objs]

            if room_labels:
                # B. Chiedi al VLM i Pesi e le Info (Label + Descrizione)
                dynamic_weights = self.get_object_weights_from_vlm(room_labels)
                semantic_name, description = self.ask_vlm_room_info(room_labels)
                
                # C. Calcola l'Embedding usando i pesi dinamici appena estratti
                final_embedding = self.compute_scene_embedding(room_labels, dynamic_weights)
                
                # D. Aggiorna il grafo in RAM
                self.scene_graph[old_room]["embedding"] = final_embedding
                self.scene_graph[old_room]["semantic_label"] = semantic_name
                self.scene_graph[old_room]["description"] = description
                
                print(f"🧠 [VLM] Stanza {old_room} classificata come: {semantic_name}")
                print(f"📝 [VLM] Descrizione: {description}")
                
                # E. Salva nel nuovo file room.json e nel grafo generale
                self.save_room_to_json(old_room, semantic_name, description, final_embedding, room_labels)
                
                output_sg = os.path.join(PROJECT_ROOT, "output", "scene_graph_data.json")
                try:
                    sg_copy = json.loads(json.dumps(self.scene_graph, default=lambda x: x.tolist() if isinstance(x, np.ndarray) else x))
                    with open(output_sg, "w") as f:
                        json.dump(sg_copy, f, indent=4)
                except Exception as e:
                    print(f"Errore salvataggio Scene Graph: {e}")

            # F. Crea la nuova stanza
            self.room_counter += 1
            self.current_room_id = f"room_{self.room_counter}"
            self.init_room_node(self.current_room_id)
            print(f"✨ [CAMBIO STANZA] Inizio mappatura {self.current_room_id}")
            print("-" * 55)
            
            return self.current_room_id

    def save_room_to_json(self, room_id, label, description, embedding, objects):
        output_dir = os.path.join(PROJECT_ROOT, "output")
        os.makedirs(output_dir, exist_ok=True)
        json_path = os.path.join(output_dir, "room.json")
        
        # Rimosso il campo "embedding" per non intasare il JSON
        room_data = {
            "room_id": room_id,
            "semantic_label": label,
            "description": description,
            "objects": objects
        }
        
        existing_data = []
        if os.path.exists(json_path):
            try:
                with open(json_path, "r") as f:
                    existing_data = json.loads(f.read())
            except Exception:
                pass
                
        # Aggiorna se esiste, altrimenti aggiungi
        updated = False
        for i, r in enumerate(existing_data):
            if r["room_id"] == room_id:
                existing_data[i] = room_data
                updated = True
                break
                
        if not updated:
            existing_data.append(room_data)
            
        with open(json_path, "w") as f:
            json.dump(existing_data, f, indent=4)
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
        self.db = MapDatabase(db_path=os.path.join(log_dir, "tiago_temporal_map_4.db"))
        self.robot_has_moved = False

        self.latest_descriptions = None
        self.latest_bboxes_msg = None
        
        # --- INIT ROOM MANAGER ---
        self.room_manager = RoomManager(w2v_model=world2vec)
        self.last_room_check_time = time.time()

        qos_latch = QoSProfile(depth=10, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        qos_standard = QoSProfile(depth=10)
        
        # Publishers
        self.persistent_bbox_pub = self.create_publisher(MarkerArray, '/persistent_bbox', qos_latch)
        self.persistent_centroids_pub = self.create_publisher(MarkerArray, '/persistent_centroids', qos_latch)
        self.considered_volume_pub = self.create_publisher(MarkerArray, '/considered_volume', qos_standard)
        self.uncertain_bboxes_pub = self.create_publisher(MarkerArray, '/uncertain_object', qos_standard)
        self.uncertain_centroids_pub = self.create_publisher(MarkerArray, '/uncertain_centroids', qos_standard)
        self.tracking_activated_pub = self.create_publisher(Bool, '/tracking_mode_activated', qos_standard)

        # Service server
        self.srv = self.create_service(
            ObjectTrackingService,
            'object_tracking_service',
            self.object_tracking_callback
        )
        self.get_logger().info('Object Tracking Service ready')

        # Subscribers
        self.create_subscription(Bool, "/robot_movement_detected", self.movement_callback, qos_standard)

        self.create_subscription(ObjectDescriptionArray, '/object_descriptions', self._descriptions_callback, qos_standard)
        self.create_subscription(Bbox3dArray, '/bbox_3d', self._bboxes_callback, qos_standard)

        self._bbox_timer = self.create_timer(2.0, self.periodic_bbox_publisher)
        
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
        self.log_both('info', f"[MOVEMENT] Robot moved: {msg.data}")
        if msg.data and not self.robot_has_moved:
            self.robot_has_moved = True
            self.log_both('warn', "[MOVEMENT] ✓ Robot has moved - ready for tracking transition")

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
            
            if distance > 0.6 and iou < EXPLORATION_IOU_THRESHOLD:
                self.log_both('warn', f"🔴 [TRACKING TRANSITION] Object '{best_match.label}' is the best match but moved! (Dist: {distance:.2f}m)")
                return True, best_match, distance
        
        return False, None, 0.0


    def object_tracking_callback(self, request, response):
        if not self.exploration_mode:
            self.tracking_step_counter += 1
        
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

        # --- VALUTAZIONE STANZA PERIODICA (ogni 5 secondi) ---
        current_time = time.time()
        if (current_time - getattr(self, 'last_room_check_time', 0)) > 5.0:
            if len(request.descriptions.descriptions) > 0:
                self.room_manager.evaluate_scene(request.descriptions.descriptions, wm.persistent_perceptions)
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
                    self.log_both('warn', "🔴 [TRANSITION] Switching from EXPLORATION to TRACKING mode")
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
                    if similarity > SIM_THRESHOLD and obj.bbox is not None:
                        iou = compute_iou_3d(bbox, obj.bbox)
                        if iou >= EXPLORATION_IOU_THRESHOLD:
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

        if in_exploration and self.exploration_frame_counter >= EXPLORATION_FRAME_LIMIT and len(request.descriptions.descriptions) > 0:
            self.log_both('warn', "🔴 [TRANSITION] Exploration frame limit reached - switching to TRACKING mode")
            self.exploration_mode = False
            self.tracking_step_counter = 1
            tracking_activated = True
            in_exploration = False
            self.exploration_frame_counter = 0
            
            msg = Bool()
            msg.data = True
            self.tracking_activated_pub.publish(msg)

        if not in_exploration:
            description_received = len(request.descriptions.descriptions) > 0
            pov_volume = self.latest_fov_volume

            if pov_volume:
                scaled_pov_volume = shrink_pov_volume(pov_volume, POV_SCALE_FACTOR)
                objects_modified_delete = self.delete_undetected_objects(scaled_pov_volume, current_perception_objects, description_received)
                objects_modified = objects_modified or objects_modified_delete
                objects_modified_uncertain = self.delete_uncertain_objects(scaled_pov_volume)
                objects_modified = objects_modified or objects_modified_uncertain

        if objects_modified and not in_exploration:
            publish_persistent_bboxes(self, wm, self.persistent_bbox_pub)
            publish_persistent_centroids(self, wm, self.persistent_centroids_pub)
            publish_uncertain_bboxes(self, self.uncertain_objects, self.uncertain_bboxes_pub)
            publish_uncertain_centroids(self, self.uncertain_objects, self.uncertain_centroids_pub)
            save_uncertain_objects(self)
            
        self.latest_bboxes.clear()
        self.latest_fov_volume = None

        response.status = "tracking_activated" if tracking_activated else "success"
        response.num_objects = len(wm.persistent_perceptions)
        response.tracking_mode_activated = tracking_activated

        return response

    def add_new_object(self, label, bbox, description_text, color, material, description_embedding, in_exploration):
        new_obj = Object(label, None, bbox, description_text, color, material)
        new_obj.embedding = description_embedding
        new_obj.creation_time = time.time()
        
        # --- NUOVO: Assegnazione Spaziale ---
        assigned_room = self.room_manager.assign_room_by_geometry(bbox)
        new_obj.room_id = assigned_room
        
        # Aggiorna la geometria della stanza e aggiunge l'oggetto al Scene Graph
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

        return new_obj

    def modify_existing_object(self, best_match, bbox, description_embedding):
        old_bbox = best_match.bbox
        iou = compute_iou_3d(bbox, old_bbox)

        old_x = (old_bbox['x_min'] + old_bbox['x_max']) / 2.0
        old_y = (old_bbox['y_min'] + old_bbox['y_max']) / 2.0
        old_z = (old_bbox['z_min'] + old_bbox['z_max']) / 2.0
        new_x = (bbox['x_min'] + bbox['x_max']) / 2.0
        new_y = (bbox['y_min'] + bbox['y_max']) / 2.0
        new_z = (bbox['z_min'] + bbox['z_max']) / 2.0
        distance = np.sqrt((new_x - old_x)**2 + (new_y - old_y)**2 + (new_z - old_z)**2)

        if iou >= TRACKING_IOU_THRESHOLD:
            best_match.bbox = bbox
            # Aggiorniamo la geometria in caso di movimenti lievi
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
            
            # RI-ASSEGNAZIONE GEOMETRICA IN CASO DI SPOSTAMENTO NETTO
            new_room = self.room_manager.assign_room_by_geometry(bbox)
            updated_obj.room_id = new_room
            self.room_manager.update_room_geometry(new_room, bbox)
            if updated_obj.label not in self.room_manager.scene_graph[new_room]["objects"]:
                self.room_manager.scene_graph[new_room]["objects"].append(updated_obj.label)
                
            wm.persistent_perceptions.append(updated_obj)

            self.log_operation(f"[SPOSTAMENTO] '{best_match.label}' si è mosso di {distance:.2f}m")

            return updated_obj, distance, iou

    def delete_undetected_objects(self, pov_volume, current_perception_objects, description_received):
        if not pov_volume:
            return False

        publish_pov_volume(self, pov_volume, self.considered_volume_pub)
        objects_to_remove = []

        if not description_received:
            for obj in wm.persistent_perceptions:
                if obj.bbox and bbox_centroid_in_volume(obj.bbox, pov_volume):
                    objects_to_remove.append(obj)
        else:
            for obj in wm.persistent_perceptions:
                if obj not in current_perception_objects:
                    if obj.bbox and bbox_centroid_in_volume(obj.bbox, pov_volume):
                        objects_to_remove.append(obj)

        if objects_to_remove:
            for obj in objects_to_remove:
                self.log_operation(f"[CANCELLATO] '{obj.label}' rimosso")
                wm.persistent_perceptions.remove(obj)
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

    def periodic_bbox_publisher(self):
        if not self.exploration_mode:
            if len(wm.persistent_perceptions) > 0:
                publish_persistent_bboxes(self, wm, self.persistent_bbox_pub)
                publish_persistent_centroids(self, wm, self.persistent_centroids_pub)

            if len(self.uncertain_objects) > 0:
                publish_uncertain_bboxes(self, self.uncertain_objects, self.uncertain_bboxes_pub)
                publish_uncertain_centroids(self, self.uncertain_objects, self.uncertain_centroids_pub)

    def _descriptions_callback(self, msg):
        self._topic_descriptions = msg
        self._try_process()

    def _bboxes_callback(self, msg):
        self._topic_bboxes = msg
        self._try_process()

    def _try_process(self):
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

def main(args=None):
    rclpy.init(args=args)
    service_node = ObjectManagerService()

    try:
        rclpy.spin(service_node)
    except KeyboardInterrupt:
        print(f"\nOBJECT MANAGER SERVICE chiuso ({datetime.now().strftime('%Y-%m-%d %H:%M:%S')})")
    finally:
        service_node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()