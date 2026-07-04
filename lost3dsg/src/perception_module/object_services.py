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
    AddObject, RemoveObject, UpdateObject, MergeObjects, DeleteObjects, QueryObjects,
)
import uuid
from visualization_msgs.msg import Marker, MarkerArray
from room_manager import RoomManager
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
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

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

log_dir = os.path.join(PROJECT_ROOT, "output")
os.makedirs(log_dir, exist_ok=True)
SYNTHETIC_LOG_FILE = os.path.join(log_dir, "operations.txt")

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
    

class ObjectServices(Node):
    def __init__(self, room_manager):
        super().__init__('object_services_node')
        self.room_manager=room_manager
        self.get_logger().info("=== ObjectServices Initialized ===")
        self.get_logger().info(f"Log sintetico operazioni: {SYNTHETIC_LOG_FILE}")

        self.tracking_step_counter = 0
        self.exploration_step_counter = 0
        
        self.db = MapDatabase(db_path=os.path.join(log_dir, "tiago_temporal_map_5.db"))
        qos_latch = QoSProfile(depth=10, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.persistent_bbox_pub = self.create_publisher(MarkerArray, '/persistent_bbox', qos_latch)
        self.persistent_centroids_pub = self.create_publisher(MarkerArray, '/persistent_centroids', qos_latch)
        self.considered_volume_pub = self.create_publisher(MarkerArray, '/considered_volume', qos_latch)
        self.uncertain_bboxes_pub = self.create_publisher(MarkerArray, '/uncertain_object', qos_latch)
        self.uncertain_centroids_pub = self.create_publisher(MarkerArray, '/uncertain_centroids', qos_latch)
        self.uncertain_objects = []
        
        with open(SYNTHETIC_LOG_FILE, "a") as f:
            f.write(f"\n{'='*50}\n")
            f.write(f"NUOVO AVVIO: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"{'='*50}\n")
        
        # --- INIT ROOM MANAGER ---
        
       
        self.create_service(AddObject,        '/graph/add_object',          self._cb_add_object)
        self.create_service(RemoveObject,     '/graph/remove_object',       self._cb_remove_object)
        self.create_service(DeleteObjects,     '/graph/delete_objects',       self._cb_delete_unseen_objects)
        self.create_service(UpdateObject,     '/graph/update_object',       self._cb_update_object)
        self.create_service(MergeObjects,     '/graph/merge_objects',       self._cb_merge_objects)
        self.create_service(QueryObjects,     '/graph/query_objects',      self._cb_query_objects)

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.get_logger().info('Object Tracking Service ready')

    
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


    def _cb_delete_unseen_objects(self, request, response):
        try:
            import json

            flat = list(request.pov_volume_flat)
            if len(flat) == 6:
                pov_volume = {
                    'x_min': flat[0], 'x_max': flat[1],
                    'y_min': flat[2], 'y_max': flat[3],
                    'z_min': flat[4], 'z_max': flat[5],
                }
            else:
                pov_volume = None

            check_uncertain = getattr(request, 'check_uncertain', False)
            current_labels  = set(request.current_labels)

            deleted_labels           = []
            uncertain_removed_count  = 0

            if not check_uncertain:
                if not pov_volume:
                    print("❌ ERRORE CRITICO TF: pov_volume è vuoto! "
                        "Il robot non sa dove sta guardando "
                        "(Controlla il frame in lookup_transform). "
                        "Cancellazione annullata.")
                    response.success = False
                    response.message = "pov_volume vuoto, cancellazione annullata"
                    return response

                try:
                    publish_pov_volume(self, pov_volume, self.considered_volume_pub)
                except Exception as e:
                    print(f"⚠️ Impossibile pubblicare il volume visivo: {e}")

                objects_to_remove = []

                for obj in list(wm.persistent_perceptions):

                    # Visto in questo frame → azzera contatore
                    if obj.label in current_labels:
                        obj.not_seen_in_pov_frames = 0
                        continue

                    if obj.bbox and bbox_centroid_in_volume(obj.bbox, pov_volume):
                        if not hasattr(obj, 'not_seen_in_pov_frames'):
                            obj.not_seen_in_pov_frames = 0
                        obj.not_seen_in_pov_frames += 1

                        if obj.not_seen_in_pov_frames >= 5:
                            objects_to_remove.append(obj)

                for obj in objects_to_remove:
                    print(f"🗑️ [CANCELLATO] L'oggetto '{obj.label}' non è più "
                        f"presente nel volume osservato! RIMOSSO.")

                    if obj in wm.persistent_perceptions:
                        wm.persistent_perceptions.remove(obj)

                    room_id = getattr(obj, 'room_id', None)
                    if room_id and room_id in self.room_manager.scene_graph:
                        objs = self.room_manager.scene_graph[room_id]["objects"]
                        if obj.label in objs:
                            objs.remove(obj.label)

                    if hasattr(self, 'db'):
                        self.db.on_object_deleted(
                            obj,
                            reason="not seen in POV",
                            step=self.tracking_step_counter
                        )

                    deleted_labels.append(obj.label)

                if deleted_labels:
                    save_persistent_perceptions(self)

                    try:
                        with open('/root/exchange/output/operations.txt', 'a') as f:
                            timestamp = datetime.now().strftime('%H:%M:%S')
                            for lbl in deleted_labels:
                                f.write(f"[{timestamp}] 🗑️ CANCELLATO (non visto): {lbl}\n")
                    except Exception as e:
                        self.get_logger().error(f"Impossibile scrivere su operations.txt: {e}")

            if check_uncertain:
                uncertain_to_remove = []

                if pov_volume:
                    for uncertain_obj in self.uncertain_objects:
                        if (uncertain_obj.bbox and
                                bbox_centroid_in_volume(uncertain_obj.bbox, pov_volume)):
                            uncertain_to_remove.append(uncertain_obj)

                for uncertain_obj in uncertain_to_remove:
                    self.uncertain_objects.remove(uncertain_obj)
                    print(f"🗑️ [UNCERTAIN RIMOSSO] '{uncertain_obj.label}'")

                    try:
                        with open('/root/exchange/output/operations.txt', 'a') as f:
                            timestamp = datetime.now().strftime('%H:%M:%S')
                            f.write(f"[{timestamp}] ⚠️ UNCERTAIN RIMOSSO: {uncertain_obj.label}\n")
                    except Exception as e:
                        self.get_logger().error(f"Impossibile scrivere su operations.txt: {e}")

                uncertain_removed_count = len(uncertain_to_remove)

            response.success                 = True
            response.deleted_count           = len(deleted_labels)
            response.uncertain_removed_count = uncertain_removed_count
            response.deleted_labels_json     = json.dumps(deleted_labels)
            response.message                 = (
                f"{len(deleted_labels)} oggetti cancellati, "
                f"{uncertain_removed_count} uncertain rimossi"
            )

        except Exception as e:
            self.get_logger().error(f"_cb_delete_unseen_objects failed: {e}")
            response.success                 = False
            response.message                 = str(e)
            response.deleted_count           = 0
            response.uncertain_removed_count = 0
            response.deleted_labels_json     = "[]"

        return response
    
    def _cb_merge_objects(self, request, response):
        try:
            import json

            MAX_DISTANCE   = request.max_distance   if request.max_distance   > 0.0 else 0.8
            MIN_SIMILARITY = request.min_similarity if request.min_similarity > 0.0 else 0.75
            dry_run        = getattr(request, 'dry_run', False)

            objects = list(wm.persistent_perceptions)
            to_remove       = set()
            to_remove_pairs = []
            merge_log       = []

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

                    # Embedding lazy
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
                        if a_label == b_label:
                            iou = compute_iou_3d(a.bbox, b.bbox)
                            if iou >= 0.5:
                                print(f"   ⚠️ Stessa label + IoU alto ({iou:.3f}), forzo merge")
                                sim = 1.0
                        if sim < MIN_SIMILARITY:
                            print(f"   ❌ SIMILARITÀ BASSA ({sim:.2f} < {MIN_SIMILARITY})")
                            continue

                    dist = np.sqrt((ax - bx)**2 + (ay - by)**2 + (az - bz)**2)
                    print(f"   Distanza: {dist:.3f}m (soglia: {MAX_DISTANCE}m)")

                    if dist > MAX_DISTANCE:
                        print(f"   ❌ TROPPO LONTANI ({dist:.2f}m > {MAX_DISTANCE}m)")
                        continue

                    merged_bbox = {
                        'x_min': (a.bbox['x_min'] + b.bbox['x_min']) / 2.0,
                        'x_max': (a.bbox['x_max'] + b.bbox['x_max']) / 2.0,
                        'y_min': (a.bbox['y_min'] + b.bbox['y_min']) / 2.0,
                        'y_max': (a.bbox['y_max'] + b.bbox['y_max']) / 2.0,
                        'z_min': (a.bbox['z_min'] + b.bbox['z_min']) / 2.0,
                        'z_max': (a.bbox['z_max'] + b.bbox['z_max']) / 2.0,
                    }

                    a_unknown = a.description.lower() == 'unknown'
                    b_unknown = b.description.lower() == 'unknown'
                    if a_unknown and not b_unknown:
                        keeper, discard = b, a
                    else:
                        keeper, discard = a, b

                    vol_a = ((a.bbox['x_max']-a.bbox['x_min']) *
                            (a.bbox['y_max']-a.bbox['y_min']) *
                            (a.bbox['z_max']-a.bbox['z_min']))
                    vol_b = ((b.bbox['x_max']-b.bbox['x_min']) *
                            (b.bbox['y_max']-b.bbox['y_min']) *
                            (b.bbox['z_max']-b.bbox['z_min']))
                    vol_m = ((merged_bbox['x_max']-merged_bbox['x_min']) *
                            (merged_bbox['y_max']-merged_bbox['y_min']) *
                            (merged_bbox['z_max']-merged_bbox['z_min']))

                    print(f"   ✅ MERGE!")
                    print(f"     Volume A: {vol_a:.3f}m³ | Volume B: {vol_b:.3f}m³ → Media: {vol_m:.3f}m³")
                    print(f"     Tenuto: '{keeper.label}' | Rimosso: '{discard.label}'")
                    print(f"     Desc keeper: '{keeper.description[:40]}...'")
                    print(f"     Bbox unito: x[{merged_bbox['x_min']:.2f},{merged_bbox['x_max']:.2f}] "
                        f"y[{merged_bbox['y_min']:.2f},{merged_bbox['y_max']:.2f}] "
                        f"z[{merged_bbox['z_min']:.2f},{merged_bbox['z_max']:.2f}]")

                    to_remove.add(discard)
                    to_remove_pairs.append((keeper, discard, merged_bbox))
                    merge_log.append({
                        "keeper":      keeper.label,
                        "discarded":   discard.label,
                        "distance":    round(dist, 3),
                        "similarity":  round(sim, 3),
                        "merged_bbox": merged_bbox,
                    })

            if to_remove_pairs and not dry_run:
                print(f"\n🗑️ RIMOZIONE: {len(to_remove_pairs)} oggetti duplicati:")

                for keeper, discard, merged_bbox in to_remove_pairs:
                    keeper.bbox = merged_bbox

                    if discard in wm.persistent_perceptions:
                        cx = (discard.bbox['x_min'] + discard.bbox['x_max']) / 2.0
                        cy = (discard.bbox['y_min'] + discard.bbox['y_max']) / 2.0
                        print(f"   - {discard.label} @ ({cx:.2f}, {cy:.2f})")
                        wm.persistent_perceptions.remove(discard)

                    discard_room = getattr(discard, 'room_id', None)
                    if discard_room and discard_room in self.room_manager.scene_graph:
                        objs = self.room_manager.scene_graph[discard_room]["objects"]
                        if discard.label in objs:
                            objs.remove(discard.label)

                    self.room_manager.update_room_geometry(
                        getattr(keeper, 'room_id', self.room_manager.current_room_id),
                        merged_bbox
                    )

                save_persistent_perceptions(self)
                publish_persistent_bboxes(self, wm, self.persistent_bbox_pub)
                publish_persistent_centroids(self, wm, self.persistent_centroids_pub)

                try:
                    with open('/root/exchange/output/operations.txt', 'a') as f:
                        timestamp = datetime.now().strftime('%H:%M:%S')
                        for keeper, discard, _ in to_remove_pairs:
                            f.write(f"[{timestamp}] 🔗 MERGE: '{discard.label}' → '{keeper.label}'\n")
                except Exception as e:
                    self.get_logger().error(f"Impossibile scrivere su operations.txt: {e}")

            elif not to_remove_pairs:
                print(f"\n✅ NESSUN duplicato trovato.")

            print("══════════════════════════════════════════════\n")

            # ── Risposta ──────────────────────────────────────────────────────
            response.success        = True
            response.merged_count   = len(to_remove_pairs)
            response.merge_log_json = json.dumps(merge_log)
            response.message        = (
                f"{'[DRY RUN] ' if dry_run else ''}"
                f"{len(to_remove_pairs)} merge(s) "
                f"{'simulati' if dry_run else 'eseguiti'}"
            )

        except Exception as e:
            self.get_logger().error(f"_cb_merge_objects failed: {e}")
            response.success        = False
            response.message        = str(e)
            response.merged_count   = 0
            response.merge_log_json = "[]"

        return response
        

    def _cb_add_object(self, request, response):
        try:
            bbox = {
                "x_min": request.x_min, "x_max": request.x_max,
                "y_min": request.y_min, "y_max": request.y_max,
                "z_min": request.z_min, "z_max": request.z_max,
            }
            label       = request.label
            description = request.description
            color       = request.color
            material    = request.material

            new_obj = Object(label, None, bbox, description, color, material)
            new_obj.creation_time = time.time()

            raw_embedding = getattr(request, 'description_embedding', None)

            if raw_embedding is None:
                self.log_both(
                    'warn',
                    f"[EMBEDDING] Campo description_embedding assente per '{label}' "
                    f"(descrizione='{description}')"
                )
                new_obj.embedding = None
            else:
                embedding = np.asarray(raw_embedding, dtype=np.float32).flatten()
                print(type(raw_embedding), len(raw_embedding))
                if embedding.size == 0:
                    self.log_both(
                        'warn',
                        f"[EMBEDDING] Embedding vuoto serializzato per '{label}' "
                        f"(descrizione='{description}')"
                    )
                    new_obj.embedding = None
                else:
                    if embedding.size != 300:
                        self.log_both(
                            'warn',
                            f"[EMBEDDING] Embedding con dimensione inattesa per '{label}': "
                            f"{embedding.size}"
                        )
                    new_obj.embedding = embedding.tolist()

            if hasattr(request, 'room_id') and request.room_id:
                assigned_room = request.room_id
            else:
                assigned_room = self.room_manager.current_room_id

            new_obj.room_id = assigned_room
            self.room_manager.update_room_geometry(assigned_room, bbox)

            if label not in self.room_manager.scene_graph[assigned_room]["objects"]:
                self.room_manager.scene_graph[assigned_room]["objects"].append(label)

            wm.persistent_perceptions.append(new_obj)

            in_exploration = getattr(request, 'in_exploration', False)
            phase = "exploration" if in_exploration else "tracking"
            step  = self.exploration_step_counter if in_exploration else self.tracking_step_counter
            self.db.on_new_object(new_obj, phase=phase, step=step)

            x_size = bbox["x_max"] - bbox["x_min"]
            y_size = bbox["y_max"] - bbox["y_min"]
            z_size = bbox["z_max"] - bbox["z_min"]
            volume = x_size * y_size * z_size

            mode_tag = "[EXPLORATION]" if in_exploration else f"[TRACKING STEP {self.tracking_step_counter}]"
            self.log_both('info', f"{mode_tag} New object '{label}' in {assigned_room} (vol: {volume:.3f} m³)")

            save_persistent_perceptions(self)

            if in_exploration:
                self.exploration_step_counter += 1

            try:
                with open('/root/exchange/output/operations.txt', 'a') as f:
                    timestamp = datetime.now().strftime('%H:%M:%S')
                    cx = (bbox["x_min"] + bbox["x_max"]) / 2.0
                    cy = (bbox["y_min"] + bbox["y_max"]) / 2.0
                    cz = (bbox["z_min"] + bbox["z_max"]) / 2.0
                    f.write(f"[{timestamp}] 🟢 AGGIUNTO: {label} in {assigned_room} "
                            f"a pos({cx:.2f}, {cy:.2f}, {cz:.2f})\n")
            except Exception as e:
                self.get_logger().error(f"Impossibile scrivere su operations.txt: {e}")

            response.success   = True
            response.message   = f"Object '{label}' added to {assigned_room}"
            response.object_id = new_obj.label  

        except Exception as e:
            self.get_logger().error(f"_cb_add_object failed: {e}")
            response.success = False
            response.message = str(e)
            response.object_id = ""

        return response
    

    def _cb_remove_object(self, request, response):
        try:
            pov_volume = {
                "x_min": request.pov_x_min,
                "x_max": request.pov_x_max,
                "y_min": request.pov_y_min,
                "y_max": request.pov_y_max,
                "z_min": request.pov_z_min,
                "z_max": request.pov_z_max,
            }

            uncertain_to_remove = []

            if pov_volume:
                for obj in self.uncertain_objects:
                    if obj.bbox and bbox_centroid_in_volume(obj.bbox, pov_volume):
                        uncertain_to_remove.append(obj)

            for obj in uncertain_to_remove:
                self.uncertain_objects.remove(obj)

            save_uncertain_objects(self)

            response.success = True
            response.message = f"Removed {len(uncertain_to_remove)} uncertain objects"

        except Exception as e:
            response.success = False
            response.message = str(e)

        return response


    def _cb_update_object(self, request, response):
        try:
            obj_id = request.object_id
            all_labels = [o.label for o in wm.persistent_perceptions]
            self.get_logger().info(f"[DEBUG UPDATE] Cerco '{obj_id}' tra: {all_labels}")
            best_match = next((o for o in wm.persistent_perceptions if o.label == obj_id), None)
            if best_match is None:
                response.success = False
                response.message = f"Object {obj_id} not found"
                response.object_id = ""
                response.distance = 0.0
                response.iou = 0.0
                response.replaced = False
                return response

            bbox = {
                "x_min": request.x_min,
                "x_max": request.x_max,
                "y_min": request.y_min,
                "y_max": request.y_max,
                "z_min": request.z_min,
                "z_max": request.z_max,
            }

            if hasattr(request, "description") and request.description:
                best_match.description = request.description
            if hasattr(request, "color") and request.color:
                best_match.color = request.color
            if hasattr(request, "material") and request.material:
                best_match.material = request.material

            description_embedding = getattr(request, "description_embedding", None)

            updated_obj = best_match
            distance = 0.0
            iou = 0.0

            if getattr(request, "update_bbox", False):
                if "door" in best_match.label.lower():
                    best_match.bbox = bbox
                    updated_obj = best_match
                    distance = 0.0
                    iou = 1.0

                else:
                    old_bbox = best_match.bbox
                    iou = compute_iou_3d(bbox, old_bbox)

                    old_x = (old_bbox["x_min"] + old_bbox["x_max"]) / 2.0
                    old_y = (old_bbox["y_min"] + old_bbox["y_max"]) / 2.0
                    old_z = (old_bbox["z_min"] + old_bbox["z_max"]) / 2.0
                    new_x = (bbox["x_min"] + bbox["x_max"]) / 2.0
                    new_y = (bbox["y_min"] + bbox["y_max"]) / 2.0
                    new_z = (bbox["z_min"] + bbox["z_max"]) / 2.0
                    distance = np.sqrt((new_x - old_x) ** 2 + (new_y - old_y) ** 2 + (new_z - old_z) ** 2)

                    if distance < 0.5 or iou >= TRACKING_IOU_THRESHOLD:
                        best_match.bbox = bbox
                        self.room_manager.update_room_geometry(
                            getattr(best_match, "room_id", self.room_manager.current_room_id),
                            bbox
                        )
                        updated_obj = best_match

                    elif (time.time() - getattr(best_match, "creation_time", 0)) < OBJECT_STABILITY_TIMEOUT:
                        best_match.bbox = bbox
                        updated_obj = best_match

                    else:
                        if best_match in wm.persistent_perceptions:
                            wm.persistent_perceptions.remove(best_match)
                            self.db.on_object_moved(
                                best_match,
                                old_bbox=best_match.bbox,
                                new_bbox=bbox,
                                distance=distance,
                                iou=iou,
                                step=self.tracking_step_counter
                            )

                        if distance > 0.8:
                            if best_match not in self.uncertain_objects:
                                self.uncertain_objects.append(best_match)
                                self.db.on_uncertain_added(best_match, step=self.tracking_step_counter)

                        updated_obj = Object(
                            best_match.label,
                            None,
                            bbox,
                            best_match.description,
                            best_match.color,
                            best_match.material
                        )
                        updated_obj.embedding = description_embedding

                        new_room = self.room_manager.assign_room_by_geometry(bbox)
                        updated_obj.room_id = new_room
                        self.room_manager.update_room_geometry(new_room, bbox)

                        if updated_obj.label not in self.room_manager.scene_graph[new_room]["objects"]:
                            self.room_manager.scene_graph[new_room]["objects"].append(updated_obj.label)

                        wm.persistent_perceptions.append(updated_obj)
                        self.log_operation(f"[SPOSTAMENTO] '{best_match.label}' si è mosso di {distance:.2f}m")

            save_persistent_perceptions(self)

            replaced = updated_obj is not best_match

            self.log_both("info", f"UPDATE {obj_id} dist={distance:.2f}m iou={iou:.2f} replaced={replaced}")

            try:
                with open("/root/exchange/output/operations.txt", "a") as f:
                    timestamp = datetime.now().strftime("%H:%M:%S")
                    cx = (bbox["x_min"] + bbox["x_max"]) / 2.0
                    cy = (bbox["y_min"] + bbox["y_max"]) / 2.0
                    cz = (bbox["z_min"] + bbox["z_max"]) / 2.0
                    tag = "SPOSTATO" if replaced else "AGGIORNATO"
                    f.write(f"{timestamp} {tag} {obj_id} a pos=({cx:.2f}, {cy:.2f}, {cz:.2f}) dist={distance:.2f}m iou={iou:.2f}\n")
            except Exception as e:
                self.get_logger().error(f"Impossibile scrivere su operations.txt: {e}")

            response.success = True
            response.message = f"Object {obj_id} updated dist={distance:.2f}m, iou={iou:.2f}"
            response.object_id = updated_obj.label
            response.distance = float(distance)
            response.iou = float(iou)
            response.replaced = replaced
            return response

        except Exception as e:
            self.get_logger().error(f"_cb_update_object failed: {e}")
            response.success = False
            response.message = str(e)
            response.object_id = ""
            response.distance = 0.0
            response.iou = 0.0
            response.replaced = False
            return response

    def _cb_query_objects(self, request, response):
        try:
            pool = self.uncertain_objects if request.uncertain_only else wm.persistent_perceptions

            results = list(pool)
            if request.room_id:
                results = [o for o in results
                        if getattr(o, 'room_id', '') == request.room_id]
            if request.label_filter:
                results = [o for o in results
                        if request.label_filter.lower() in o.label.lower()]

            response.object_ids = [o.label for o in results]
            response.serialized_json = json.dumps([{
                "label":       o.label,
                "description": o.description,
                "color":       o.color,
                "material":    o.material,
                "bbox":        o.bbox,
                "room_id":     getattr(o, 'room_id', 'unknown')
            } for o in results])
            response.success = True
        except Exception as e:
            response.success = False
            response.serialized_json = "[]"
        return response

        
def main(args=None):
    rclpy.init(args=args)
    room_manager = RoomManager(w2v_model=world2vec)
    service_node = ObjectServices(room_manager)

    try:
        rclpy.spin(service_node)
    except KeyboardInterrupt:
        from datetime import datetime
        print(f"\nOBJECT SERVICES chiuso ({datetime.now().strftime('%Y-%m-%d %H:%M:%S')})")
        
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
