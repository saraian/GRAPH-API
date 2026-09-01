#!/usr/bin/env python3
"""
Object Manager Service - Semantic and Spatial Tracking of Perceived Objects
Tracks objects and automatically transitions from EXPLORATION to TRACKING when
an object is seen again in a different position.
Includes Topological Semantic Mapping (Room Manager).
"""
import rclpy, json, os, time, threading
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

# =============  EXPLORATION PARAMETERS =============
EXPLORATION_IOU_THRESHOLD = 0.18
SIM_THRESHOLD = 0.75
TRACKING_IOU_THRESHOLD = 0.3
VOLUME_EXPANSION_RATIO = 0.01
EXPLORATION_FRAME_LIMIT = 10  # Numero di frame in exploration prima di passare a tracking
# Aggiungi questa riga sotto EXPLORATION_FRAME_LIMIT
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

    if len(objects) == 0:
        return

    fig = plt.figure(figsize=(14, 10), facecolor='#F9F7F7')
    ax = fig.add_subplot(111, projection='3d')
    ax.set_facecolor('#F9F7F7')

    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False

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
    scene_center = centroids.mean(axis=0)
    z_max = max(c[2] for c in centroids) + 0.3
    root_pos = np.array([scene_center[0], scene_center[1], z_max])

    ax.scatter(*root_pos, s=500, c='#4A6572', marker='o', zorder=10)
    ax.text(root_pos[0], root_pos[1], root_pos[2] + 0.08, 'SCENE',
            ha='center', va='bottom', fontsize=13, fontweight='bold', color='#344955')

    for i, obj in enumerate(objects):
        centroid = centroids[i]
        ax.plot3D([root_pos[0], centroid[0]],
                  [root_pos[1], centroid[1]],
                  [root_pos[2], centroid[2]],
                  color='#9DB2BF', linewidth=1.5, alpha=0.5)

        ax.scatter(*centroid, s=400, c='#7EB5D6', marker='o', zorder=10)
        ax.text(centroid[0], centroid[1], centroid[2] + 0.1,
                obj['label'],
                ha='center', va='bottom', fontsize=10, fontweight='semibold',
                color='#2C3E50')

    ax.set_xlabel('X (m)', fontsize=11, color='#526D82')
    ax.set_ylabel('Y (m)', fontsize=11, color='#526D82')
    ax.set_zlabel('Z (m)', fontsize=11, color='#526D82')

    plt.tight_layout()

    output_path = os.path.join(output_dir, f"scene_graph_{prefix}_{step:03d}.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)


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
        if obj.bbox is None:
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
        if obj.bbox is None:
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
        if obj.bbox is None:
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
        if obj.bbox is None:
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
    
    # Calcolo della posizione usando direttamente pov_volume
    marker.pose.position.x = (pov_volume['x_min'] + pov_volume['x_max']) / 2.0
    marker.pose.position.y = (pov_volume['y_min'] + pov_volume['y_max']) / 2.0
    marker.pose.position.z = (pov_volume['z_min'] + pov_volume['z_max']) / 2.0
    
    # Calcolo della scala (dimensioni) usando direttamente pov_volume
    marker.scale.x = pov_volume['x_max'] - pov_volume['x_min']
    marker.scale.y = pov_volume['y_max'] - pov_volume['y_min']
    marker.scale.z = pov_volume['z_max'] - pov_volume['z_min']
    
    marker.color.a = 0.2
    marker.color.r, marker.color.g, marker.color.b = 0.0, 0.0, 1.0
    
    marker_array.markers.append(marker)
    pub.publish(marker_array)

def get_object_weights_from_vlm(labels, encoded_image):
    labels_str = ", ".join(labels)
    prompt = (
            f"Analizza questa lista di oggetti: {labels_str}.\n"
            "Assegna a ogni oggetto un peso da 0.1 a 3.0 per definire l'identità della stanza.\n"
            "- PESO ALTO (2.0 - 3.0): Oggetti unici, dispositivi elettronici e accessori specifici (es. 'plant', 'fridge', 'monitor', 'headphones', 'laptop').\n"
            "- PESO BASSO (0.1 - 0.5): Oggetti generici, cancelleria e mobili base (es. 'table', 'chair', 'book', 'wall', 'paper').\n"
            "Rispondi ESCLUSIVAMENTE in formato JSON puro: {\"label\": peso}"
        )
    
    try:
        response_text = vlm_call(prompt, encoded_image)
        
        # Estrazione JSON
        json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
        if json_match:
            weights_dict = json.loads(json_match.group())
            
            # --- DEBUG PRINT ---
            print("\n⚖️  [VLM WEIGHTS] Nuovi pesi assegnati:")
            for obj, weight in weights_dict.items():
                star_count = int(weight) * "⭐" # Visualizzazione rapida dell'importanza
                print(f"   - {obj.upper()}: {weight:.1f} {star_count}")
            print("-" * 30)
            # -------------------
            
            return weights_dict
        else:
            print(f"⚠️ [VLM ERROR] Nessun JSON trovato nella risposta: {response_text}")
            return {}
    except Exception as e:
        print(f"❌ [VLM EXCEPTION] Errore: {e}")
        return {}

# ============= ROOM MANAGER =============
import numpy as np
import json
import re

import numpy as np
import json
import re

class RoomManager:
    def __init__(self, w2v_model, similarity_threshold=0.55):
        self.w2v = w2v_model
        self.similarity_threshold = similarity_threshold
        self.room_counter = 1
        self.current_room_id = f"room_{self.room_counter}"
        self.current_room_embedding = None
        self.current_room_labels = set()
        
        # Cache pesi: se un oggetto non è qui, chiameremo il VLM
        self.weights_cache = {
            "chair": 0.2,
            "table": 0.5,
            "wall": 0.1
        }

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
            from openai import OpenAI
            with open(os.path.join(os.path.dirname(file_path), "api.txt"), "r") as f:
                 api_key = f.read().strip()

            client = OpenAI(api_key=api_key)
            
            # Chiamata SOLO TESTO (ignora l'immagine) e forza il JSON
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                response_format={ "type": "json_object" }
            )
            
            response_text = response.choices[0].message.content
            print(f"RISPOSTA RAW VLM: {response_text}")

            import json
            weights = json.loads(response_text)
            
            print("\n⚖️  [VLM WEIGHTS] Nuovi pesi calcolati:")
            for k, v in weights.items():
                print(f"   - {k.upper()}: {float(v):.1f} {'⭐' * int(v)}")
            return weights
            
        except Exception as e:
            print(f"🔴 ERRORE VLM: {e}")
            return {}

    def compute_scene_embedding(self, labels):
        vectors = []
        weights = []
        debug_parts = []
        
        for label in labels:
            base_label = label.split('#')[0].lower().replace(" ", "_")
            if base_label in self.w2v:
                vectors.append(self.w2v[base_label])
                w = self.weights_cache.get(base_label, 1.0)
                weights.append(w)
                debug_parts.append(f"{base_label}({w:.1f})")
        
        if not vectors: return None
            
        print(f"🧬 [EMBEDDING] Media ponderata: {', '.join(debug_parts)}")
        scene_vector = np.average(vectors, axis=0, weights=weights)
        return scene_vector / np.linalg.norm(scene_vector)

    def evaluate_scene(self, current_scene_objects, persistent_objects, encoded_image=None, vlm_callback=None):
        if not current_scene_objects:
            return self.current_room_id
            
        # 1. PREPARA LE ETICHETTE (Vista Attuale vs Tutto il Persistent)
        raw_labels = [obj.label for obj in current_scene_objects]
        clean_labels = list(set([l.split('#')[0].lower().replace(" ", "_") for l in raw_labels]))
        
        mem_labels = [obj.label for obj in persistent_objects] if persistent_objects else []
        clean_mem_labels = list(set([l.split('#')[0].lower().replace(" ", "_") for l in mem_labels]))
        
        # 2. AGGIORNAMENTO PESI VLM 
        all_labels = list(set(clean_labels + clean_mem_labels))
        unknown = [l for l in all_labels if l not in self.weights_cache]
        
        if unknown:
            print(f"🧠 Calcolo nuovi pesi VLM per oggetti mancanti: {unknown}")
            from cv_utils import vlm_call
            try:
                new_w = self.get_object_weights_from_vlm(unknown, "", vlm_call)
                if new_w:
                    for k, v in new_w.items(): 
                        self.weights_cache[k.lower()] = float(v)
            except Exception as e:
                print(f" Errore VLM in evaluate_scene: {e}")
            
            for label in unknown:
                if label not in self.weights_cache:
                    print(f"FALLBACK: Nessun peso dal VLM per '{label}'. Assegno valore di emergenza.")
                    if label in ["table", "chair", "wall", "floor", "ceiling"]:
                        self.weights_cache[label] = 0.5
                    else:
                        self.weights_cache[label] = 2.0

        # 3. CREA GLI EMBEDDING DA CONFRONTARE
        new_scene_embedding = self.compute_scene_embedding(raw_labels)
        memory_embedding = self.compute_scene_embedding(mem_labels)

        if new_scene_embedding is None: 
            return self.current_room_id 

        # 4. INIZIALIZZAZIONE (Se persistent perception è vuoto, siamo all'inizio)
        if not mem_labels or memory_embedding is None:
            print(f"🏠 [INIZIALIZZAZIONE] Stanza: {self.current_room_id}")
            return self.current_room_id

        # 5. CONFRONTO TOPOLOGICO
        similarity = np.dot(memory_embedding, new_scene_embedding)
        
        print(f"\n🔍 [TOPOLOGICAL CONTROL] {self.current_room_id}")
        
        vista_con_pesi = []
        for raw_lbl in raw_labels:
            clean_lbl = raw_lbl.split('#')[0].lower().replace(" ", "_")
            peso = self.weights_cache.get(clean_lbl, "N/A")
            vista_con_pesi.append(f"{raw_lbl}({peso})")
            
        print(f"👀 VISTA ATTUALE (Ultime percezioni): {', '.join(vista_con_pesi)}")
        print(f"🧠 MEMORIA STANZA (Tutto il Persistent): {', '.join(mem_labels)}") 
        print(f"📊 Similarità: {similarity:.3f} (Soglia: {self.similarity_threshold})")

       # --- NUOVO CONTROLLO: Anchor Object (Oggetto Ancora) ---
        has_anchor_object = False
        anchor_label = ""
        ANCHOR_DIST_THRESHOLD = 0.50  # Alzata a 50cm per tollerare il rumore

        for current_obj_desc in current_scene_objects:
            current_label_base = current_obj_desc.label.split('#')[0].lower()
            curr_bbox = getattr(current_obj_desc, 'bbox', None) 
            
            if not curr_bbox: continue

            curr_cx = (curr_bbox['x_min'] + curr_bbox['x_max']) / 2.0
            curr_cy = (curr_bbox['y_min'] + curr_bbox['y_max']) / 2.0

            for persistent_obj in persistent_objects:
                pers_label_base = persistent_obj.label.split('#')[0].lower()
                
                if current_label_base == pers_label_base:
                    if persistent_obj.bbox:
                        pers_cx = (persistent_obj.bbox['x_min'] + persistent_obj.bbox['x_max']) / 2.0
                        pers_cy = (persistent_obj.bbox['y_min'] + persistent_obj.bbox['y_max']) / 2.0
                        
                        dist = np.sqrt((curr_cx - pers_cx)**2 + (curr_cy - pers_cy)**2)
                        
                        # LOG DI DEBUG (opzionale ma consigliato per capire perché fallisce)
                        # print(f"DEBUG: Anchor candidate '{current_label_base}' dist: {dist:.3f}m")

                        if dist < ANCHOR_DIST_THRESHOLD:
                            has_anchor_object = True
                            anchor_label = persistent_obj.label
                            break
            if has_anchor_object: break

        # Decisione finale
        if similarity < self.similarity_threshold and not has_anchor_object:
            # ==========================================================
            # 1. LOGICA CAMBIO STANZA (Eseguita solo se similarity < soglia E no anchor)
            # ==========================================================
            self.room_counter += 1
            old_room = self.current_room_id
            self.current_room_id = f"room_{self.room_counter}"
            print(f"🚪 [CAMBIO STANZA] {old_room} -> {self.current_room_id} ✨")
            
            # --- AZZERAMENTO TOTALE MEMORIA (Spostato qui dentro!) ---        
            try:
                with open('persistent_perception.json', 'w') as f:
                    json.dump([], f)
                
                from world_model import wm
                # Pulisco gli oggetti nel World Model
                if hasattr(wm, 'objects'):
                    wm.objects.clear() if isinstance(wm.objects, (list, dict)) else None
                
                # Pulisco le percezioni persistenti
                if hasattr(wm, 'persistent_perceptions'):
                    wm.persistent_perceptions.clear() if isinstance(wm.persistent_perceptions, (list, dict)) else None
                        
                print("🧹 [CLEANUP] Memoria oggetti e persistent_perception.json azzerati!")
            except Exception as e:
                print(f"🔴 Errore durante il cleanup: {e}")

        else:
            # ==========================================================
            # 2. LOGICA PERMANENZA (Eseguita se similarity è alta O c'è un anchor)
            # ==========================================================
            if has_anchor_object and similarity < self.similarity_threshold:
                print(f"⚓ [ANCHOR DETECTED] Trovato '{anchor_label}' in posizione identica.")
                print(f"   Forza permanenza in {self.current_room_id} nonostante similarity bassa ({similarity:.3f})")
            else:
                print(f"✅ [COERENTE] Resto in {self.current_room_id} (Similarity: {similarity:.3f})")
            
            # NOTA: Qui NON c'è alcun cleanup, quindi la memoria viene mantenuta.
            
        print("-" * 55)
        return self.current_room_id

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
        self.db = MapDatabase(db_path=os.path.join(log_dir, "tiago_temporal_map_3.db"))
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

        # 1. Cerca il miglior match semantico in tutta la memoria
        for obj in wm.persistent_perceptions:
            # Salta oggetti troppo "giovani"
            if (time.time() - getattr(obj, 'creation_time', 0)) < OBJECT_STABILITY_TIMEOUT:
               continue
            
            obj_label_base = obj.label.split('#')[0] if '#' in obj.label else obj.label

            if not hasattr(obj, "embedding") or obj.embedding is None:
                obj.embedding = get_embedding(world2vec, obj.description)
            
            if obj.embedding is None or description_embedding is None:
                continue
            
            similarity = lost_similarity(world2vec, label_base, obj_label_base, color, obj.color,
                                       material, obj.material, description_embedding, obj.embedding)
            
            # Teniamo traccia solo del match migliore che superi la soglia
            if similarity > SIM_THRESHOLD and similarity > highest_similarity:
                highest_similarity = similarity
                best_match = obj

        # 2. Se abbiamo trovato un best match, controlliamo se si è spostato
        if best_match:
            print(f"🔍 [BEST MATCH FOUND] Detected: '{label_base}' -> Best in memory: '{best_match.label}' (Score: {highest_similarity:.3f})")
            
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
            
            # Transizione solo se il MIGLIOR match è effettivamente in un'altra posizione
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

        # --- SPOSTATO QUI: VALUTAZIONE STANZA PERIODICA (ogni 5 secondi) ---
        # Viene eseguito PRIMA che il ciclo sottostante fonda i nuovi oggetti nel persistent_perceptions!
        current_time = time.time()
        if (current_time - getattr(self, 'last_room_check_time', 0)) > 5.0:
            if len(request.descriptions.descriptions) > 0:
                # Confrontiamo i nuovi arrivi con lo storico incontaminato
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
                    self.log_both('warn', f"🔴 [TRANSITION] Switching from EXPLORATION to TRACKING mode")
                    self.log_operation("!!! TRANSITION: Modalità passata da EXPLORATION a TRACKING !!!")
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
            self.log_both('warn', f"🔴 [TRANSITION] Exploration frame limit ({EXPLORATION_FRAME_LIMIT}) reached - switching to TRACKING mode")
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
                # FIX: Riduciamo il volume QUI, in modo che la logica usi lo spazio corretto!
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
        #ASSEGNAZIONE TEMPO DI CREAZIONE 
        new_obj.creation_time = time.time()
        # --- ASSEGNAZIONE ROOM ID ---
        new_obj.room_id = self.room_manager.current_room_id
        
        wm.persistent_perceptions.append(new_obj)

        phase = "exploration" if in_exploration else "tracking"
        step = self.exploration_step_counter if in_exploration else self.tracking_step_counter
        
        # Nel tuo map_database potrai in futuro usare kwargs per passare il room_id:
        # self.db.on_new_object(new_obj, phase=phase, step=step, room_id=new_obj.room_id)
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
            # Mantiene la stanza di origine
            updated_obj.room_id = getattr(best_match, 'room_id', self.room_manager.current_room_id)
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
                self.db.on_object_deleted(obj, reason="not seen in POV",
                                  step=self.tracking_step_counter)

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