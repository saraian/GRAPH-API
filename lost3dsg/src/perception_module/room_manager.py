#!/usr/bin/env python3

import rclpy, json, os, time, threading, re
from rclpy.node import Node
from rclpy.duration import Duration
from rclpy.qos import QoSProfile, DurabilityPolicy
import numpy as np
from openai import OpenAI
from lost3dsg.msg import ObjectDescriptionArray, Bbox3dArray
from lost3dsg.srv import (
    ObjectTrackingService,
    AddObject, RemoveObject, UpdateObject, MergeObjects, DeleteObjects,
)
import uuid
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import Bool
from object_info import Object
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
        
        # Cache pesi: se un oggetto non Ã¨ qui, chiameremo il VLM
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
        self.closed_room_centroids = []  # Centroidi degli hull giÃ  chiusi

        # Iscrizione al topic dei muri rilevati
     

    def init_room_node(self, room_id):
        """Crea un nuovo nodo padre (Stanza) nel grafo."""
        self.scene_graph[room_id] = {
            "semantic_label": "Unknown",
            "embedding": None,
            "boundaries": {  # Manteniamo questo per compatibilitÃ 
                "x_min": float('inf'), "x_max": float('-inf'),
                "y_min": float('inf'), "y_max": float('-inf'),
                "z_min": float('inf'), "z_max": float('-inf')
            },
            "polygon": [],   # <--- NUOVO: ConterrÃ  i punti del bordo stanza
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
                # Usa matplotlib Path per controllare se il punto (cx, cy) Ã¨ dentro il poligono
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
            print(f"ðŸ”´ [VLM ERROR] Fallimento chiamata: {e}")
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
        print(f"âœ… [DEBUG SALVATAGGIO] Dati stanza '{room_id}' salvati correttamente in: {json_path}")

    def finalize_current_room(self, persistent_objects):
        room_id = self.current_room_id
        print(f"ðŸ›‘ [FINALIZZAZIONE] Chiamata finalize per: {room_id}")
        
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