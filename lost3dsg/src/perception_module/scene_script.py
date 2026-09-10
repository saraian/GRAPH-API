#!/usr/bin/env python3

"""Compila ed esegue un piano Habitat usando un solo file Python.

Il piano in scripts/*.json contiene target_point, non coordinate inventate.
Questo programma genera i punti dalla collision mesh, calcola l'altezza del
template e invia a Habitat soltanto pose 3D già compilate.

Esempi:
    python3 scene_script.py --list-points
    python3 scene_script.py scripts/organize_objects_v2.json
"""

import argparse
from collections import Counter
from io import BytesIO
import json
import math
import os
from pathlib import Path
import re
import struct
import time
from PIL import Image as PILImage
import uuid
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import habitat_sim
import magnum as mn
import numpy as np
from config import CFG, habitat_value


DEFAULT_SCENE = os.path.join(
    CFG["habitat"]["dataset_root"], "hm3d-val-habitat-v0.2",
    "00814-p53SfW6mjZe", "p53SfW6mjZe.basis.glb",
)
DEFAULT_SCENE_DATASET = habitat_value("scene_dataset")
DEFAULT_OBJECTS = os.path.join(CFG["habitat"]["dataset_root"], "habitat_objects", "configs")
DEFAULT_NAVMESH = ""
VALID_ACTIONS = frozenset({"spawn", "move", "remove", "wait"})
MAX_STEPS = 64
MAX_COMPILED_STEPS = MAX_STEPS * 2 - 1
MAX_WAIT_SECONDS = 200.0
DEFAULT_SETTLE_SECONDS = 10.0
MAX_SEMANTIC_SURFACE_SAMPLES = 80


def load_openrouter_config():
    """Load optional OpenRouter settings from openrouter.txt.

    Environment variables take precedence over this file. The file uses a
    small dotenv-like format and is intentionally kept outside the code so it
    can contain the API key without changing this script.
    """
    config_path = Path(__file__).with_name("openrouter.txt")
    settings = {}
    try:
        lines = config_path.read_text(encoding="utf-8").splitlines()
    except FileNotFoundError:
        return settings
    except OSError as exc:
        raise RuntimeError(f"Impossibile leggere {config_path}: {exc}") from exc

    for line in lines:
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].lstrip()
        key, separator, value = line.partition("=")
        if not separator:
            continue
        key = key.strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        settings[key] = value
    return settings


def setting(name, config, default=""):
    """Return an environment setting, falling back to openrouter.txt."""
    return os.environ.get(name, config.get(name, default)).strip()


def request_llm_json(request_obj, service_name, endpoint, retry_transient=False):
    """Perform an LLM HTTP request, retrying transient OpenRouter failures."""
    try:
        configured_attempts = int(os.environ.get("OPENROUTER_HTTP_RETRIES", "5"))
    except ValueError:
        configured_attempts = 5
    try:
        base_delay = float(os.environ.get("OPENROUTER_RETRY_BASE_SECONDS", "3"))
    except ValueError:
        base_delay = 3.0
    max_attempts = min(10, max(1, configured_attempts)) if retry_transient else 1
    base_delay = min(30.0, max(0.1, base_delay))
    retryable_statuses = {408, 409, 429, 500, 502, 503, 504}
    for attempt in range(1, max_attempts + 1):
        try:
            with urlopen(request_obj, timeout=300) as response:
                return json.loads(response.read().decode("utf-8"))
        except HTTPError as exc:
            try:
                details = exc.read().decode("utf-8", errors="replace")[:1000]
            except OSError:
                details = str(exc)
            if exc.code not in retryable_statuses or attempt == max_attempts:
                raise RuntimeError(
                    f"Impossibile contattare {service_name} su {endpoint}: "
                    f"HTTP {exc.code}. Risposta: {details}"
                ) from exc
            retry_after = None
            if exc.headers is not None:
                try:
                    retry_after = float(exc.headers.get("Retry-After", ""))
                except (TypeError, ValueError):
                    retry_after = None
            delay = min(
                60.0,
                retry_after if retry_after is not None
                else base_delay * (2 ** (attempt - 1)),
            )
            print(
                f"{service_name} HTTP {exc.code}: nuovo tentativo "
                f"{attempt + 1}/{max_attempts} tra {delay:g}s.",
                flush=True,
            )
            time.sleep(delay)
        except Exception as exc:
            raise RuntimeError(
                f"Impossibile contattare {service_name} su {endpoint}: {exc}. "
                + (
                    "Controlla OPENROUTER_API_KEY, OPENROUTER_URL e il modello."
                    if retry_transient else
                    "Controlla che Ollama sia avviato e che il modello sia installato."
                )
            ) from exc
    raise AssertionError("ciclo retry LLM terminato senza risultato")


def resolve_object_configs_dir(objects_dir):
    """Return the directory containing Habitat ``*.object_config.json`` files.

    HabitatSim's ``load_configs`` is not recursive. Object datasets commonly
    expose a root directory with separate ``configs/`` and ``meshes/``
    subdirectories, so accept either the configs directory or the dataset
    root and resolve the former automatically.
    """
    requested = str(objects_dir or "").strip()
    if not requested:
        return requested
    if not os.path.isdir(requested):
        return requested
    if any(name.endswith(".object_config.json") for name in os.listdir(requested)):
        return requested
    configs = os.path.join(requested, "configs")
    if os.path.isdir(configs):
        return configs
    return requested
# Un mobile non e' rappresentato da un solo punto centrale: su un bancone il
# centro puo' coincidere con lavabo, fornelli o altra geometria occupata. Si
# conservano alcuni campioni distribuiti, poi verificati con l'ingombro del
# template prima di mostrarli al planner.
MAX_TARGET_POINTS_PER_SURFACE = 10
# Il semantic GLB contiene etichette corrette, ma la sua gerarchia puo' non
# coincidere con la geometria renderizzata nella build Habitat in uso.  Per i
# punti destinati agli script preferiamo quindi le osservazioni semantic+depth
# prodotte dal simulatore stesso.
RENDERED_SUPPORT_GRID_SIZE = 6
RENDERED_SUPPORT_YAWS = (0.0, 90.0, 180.0, 270.0)
RENDERED_SUPPORT_PIXEL_STRIDE = 4
# Un supporto utile è un tavolo o ripiano vicino al piano calpestabile; le
# facce superiori dei soffitti hanno la stessa normale ma non sono superfici
# su cui mettere un oggetto nella stanza. La navmesh serve come riferimento
# locale del pavimento, che non viene mai proposto come destinazione.
# Altezza massima di un piano di arredo sopra il pavimento navigabile. Un
# valore piccolo permetterebbe solo il pavimento; un valore troppo grande
# accetterebbe tetti, bordi superiori delle pareti e geometria esterna.
MAX_SUPPORT_HEIGHT_ABOVE_NAVMESH = 1.5
MIN_SUPPORT_HEIGHT_ABOVE_NAVMESH = 0.30
MAX_SUPPORT_HORIZONTAL_NAVMESH_DISTANCE = 0.35
MAX_SEMANTIC_COLLISION_HEIGHT_ERROR = 0.12
MAX_FOOTPRINT_HEIGHT_ERROR = 0.045
FOOTPRINT_SAMPLE_FRACTION = 0.42
SUPPORT_CATEGORIES = (
    "table", "desk", "shelf", "shelving", "counter", "cabinet",
    "dresser", "chest of drawers", "nightstand", "bedside", "sideboard",
    "console", "workbench", "kitchen island", "tv stand", "wardrobe",
    "bench", "stool", "piano", "couch", "sofa",
)

def is_support_category(category):
    """Distingue mobili di supporto dagli oggetti *sopra* un supporto."""
    normalized = str(category).strip().lower()
    if any(phrase in normalized for phrase in (
        "on shelf", "on table", "on desk", "table lamp", "desk lamp",
    )):
        return False
    return any(re.search(r"\b" + re.escape(name) + r"\b", normalized)
               for name in SUPPORT_CATEGORIES)


NON_SUPPORT_CATEGORIES = (
    "floor", "ground", "ceiling", "wall", "door", "window", "curtain",
    "picture", "painting", "lamp", "light", "person", "human", "plant",
)


def is_geometric_support_candidate(category):
    """Filtro aperto: la geometria decide, escludendo superfici impossibili."""
    normalized = str(category).strip().lower()
    if not normalized:
        return False
    if any(phrase in normalized for phrase in (
        "on shelf", "on table", "on desk", "table lamp", "desk lamp",
    )):
        return False
    return not any(re.search(r"\b" + re.escape(name) + r"\b", normalized)
                   for name in NON_SUPPORT_CATEGORIES)


def point_matches_support_constraint(point, constraint):
    """Match a point against a semantic constraint emitted by the LLM."""
    if not isinstance(constraint, dict):
        return False
    category = str(constraint.get("category", "")).strip().lower()
    rooms = {
        str(item).strip().lower()
        for item in constraint.get("room_categories", [])
        if str(item).strip()
    }
    if category and str(point.get("category", "")).strip().lower() != category:
        return False
    if rooms:
        point_rooms = {
            str(item).strip().lower()
            for item in point.get("room_categories", [])
            if str(item).strip()
        }
        if not rooms & point_rooms:
            return False
    return True


NUMBER_WORDS = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    "uno": 1, "una": 1, "due": 2, "tre": 3, "quattro": 4,
    "cinque": 5, "sei": 6, "sette": 7, "otto": 8, "nove": 9,
    "dieci": 10,
}


def _requested_object_count(request):
    """Extract an explicit object count from common Italian/English requests."""
    normalized = str(request or "").strip().lower()
    number = r"\d+|" + "|".join(sorted(NUMBER_WORDS, key=len, reverse=True))
    match = re.search(
        rf"\b({number})\b(?:\s+\w+){{0,3}}\s+\b(?:objects?|oggetti?)\b",
        normalized,
    )
    if match is None:
        return None
    token = match.group(1)
    return int(token) if token.isdigit() else NUMBER_WORDS[token]


def _expected_positioned_action_count(request):
    """Infer an exact spawn+move count when the request makes it unambiguous."""
    count = _requested_object_count(request)
    if count is None:
        return None
    normalized = str(request or "").lower()
    asks_move = bool(re.search(
        r"\b(move|moves|moved|sposta|spostare|muovi|muovere|trasferisci|trasferire|porta|portare)\b",
        normalized,
    ))
    asks_all = bool(re.search(r"\b(all|every|tutti|tutte|ciascun[oa]?)\b", normalized))
    return count * 2 if asks_move and asks_all else count


def normalize_empty_support_constraints(plan):
    """Repair only a count mismatch made entirely of unconstrained entries.

    Empty constraints are interchangeable. Non-empty semantic constraints are
    deliberately left untouched because their chronological association cannot
    be recovered safely after the model returns the wrong array length.
    """
    if not isinstance(plan, dict) or not isinstance(plan.get("steps"), list):
        return False
    constraints = plan.get("support_constraints")
    if not isinstance(constraints, list):
        return False
    required = sum(
        step.get("action") in {"spawn", "move"}
        for step in plan["steps"] if isinstance(step, dict)
    )
    if len(constraints) == required:
        return False

    def is_empty_constraint(item):
        if not isinstance(item, dict):
            return False
        category = str(item.get("category", "")).strip()
        rooms = item.get("room_categories", [])
        return (
            not category
            and isinstance(rooms, list)
            and not any(str(room).strip() for room in rooms)
        )

    if not all(is_empty_constraint(item) for item in constraints):
        return False
    plan["support_constraints"] = [
        {"category": "", "room_categories": []} for _ in range(required)
    ]
    return True


def validate_plan_intent(plan, request):
    """Reject structurally valid plans that do not fulfil the user request."""
    if not isinstance(plan, dict) or not isinstance(plan.get("steps"), list):
        raise ValueError("piano LLM privo di steps")
    steps = plan["steps"]
    positioned = [step for step in steps if step.get("action") in {"spawn", "move"}]
    constraints = plan.get("support_constraints", [])
    if len(constraints) != len(positioned):
        raise ValueError(
            "support_constraints deve avere esattamente un elemento per ogni spawn/move"
        )

    spawn_steps = [step for step in steps if step.get("action") == "spawn"]
    requested_count = _requested_object_count(request)
    if requested_count is not None and len(spawn_steps) != requested_count:
        raise ValueError(
            f"la richiesta richiede {requested_count} oggetti, ma il piano ne crea "
            f"{len(spawn_steps)}"
        )

    normalized = str(request or "").lower()
    asks_move = bool(re.search(
        r"\b(move|moves|moved|sposta|spostare|muovi|muovere|trasferisci|trasferire|porta|portare)\b",
        normalized,
    ))
    asks_all = bool(re.search(r"\b(all|every|tutti|tutte|ciascun[oa]?)\b", normalized))
    if asks_move and asks_all:
        spawned_names = {str(step.get("name", "")).strip() for step in spawn_steps}
        moved_names = {
            str(step.get("object", "")).strip()
            for step in steps if step.get("action") == "move"
        }
        missing = sorted(name for name in spawned_names if name and name not in moved_names)
        if missing:
            raise ValueError(
                "la richiesta richiede di spostare tutti gli oggetti; move mancanti per: "
                + ", ".join(missing)
            )

    wait_match = re.search(
        r"\b(?:at\s+least|almeno)\s+(\d+(?:\.\d+)?)\s*(?:seconds?|secondi?)\b",
        normalized,
    )
    if wait_match and any(step.get("action") == "move" for step in steps):
        first_move = next(i for i, step in enumerate(steps) if step.get("action") == "move")
        waited = sum(
            float(step.get("seconds", 0.0))
            for step in steps[:first_move] if step.get("action") == "wait"
        )
        required = float(wait_match.group(1))
        if waited + 1e-9 < required:
            raise ValueError(
                f"la richiesta richiede almeno {required:g}s prima dei move; "
                f"il piano ne attende {waited:g}"
            )


def placement_rank(point, object_size):
    """Prefer conventional supports and points far from their observed edges."""
    x, _, z = valid_position(point.get("surface_point"), "surface_point")
    bounds = point.get("support_bounds") or []
    edge_clearance = -float("inf")
    centrality = -float("inf")
    if (
        isinstance(bounds, (list, tuple)) and len(bounds) == 2
        and all(isinstance(item, (list, tuple)) and len(item) == 3 for item in bounds)
    ):
        low, high = bounds
        half_x, half_z = float(object_size[0]) / 2.0, float(object_size[2]) / 2.0
        edge_clearance = min(
            x - float(low[0]) - half_x, float(high[0]) - x - half_x,
            z - float(low[2]) - half_z, float(high[2]) - z - half_z,
        )
        center_x = (float(low[0]) + float(high[0])) / 2.0
        center_z = (float(low[2]) + float(high[2])) / 2.0
        centrality = -math.hypot(x - center_x, z - center_z)
    category = str(point.get("category", "")).strip().lower()
    semantic_quality = 2 if is_support_category(category) else 0
    if category in {"surface", "object", "unknown"}:
        semantic_quality = -1
    return semantic_quality, edge_clearance, centrality


def assign_valid_placements(plan, points, sim, objects_dir, margin=0.01):
    """Assign geometry-valid placements; the LLM supplies only semantic constraints."""
    constraints = plan.get("support_constraints", [])
    if not isinstance(constraints, list):
        raise ValueError("support_constraints deve essere un array")
    templates = available_template_names(objects_dir)
    object_templates = {}
    object_placements = {}
    used_placements = set()
    valid_cache = {}
    placement_index = 0

    def valid_for_template(template):
        if template in valid_cache:
            return valid_cache[template]
        handle, offset = template_handle_and_support_offset(sim, template, objects_dir)
        object_size = template_object_size(sim, handle)
        valid = {}
        for point_id, point in points.items():
            surface = collision_support_point(sim, point.get("surface_point"))
            if surface is None:
                continue
            position = [surface[0], surface[1] + offset + margin, surface[2]]
            if (
                footprint_support_quality(sim, handle, position, surface) is not None
                and target_has_visible_view(sim, handle, position)
            ):
                valid[point_id] = (
                    point, surface, position, placement_rank(point, object_size),
                    object_size,
                )
        valid_cache[template] = valid
        return valid

    def has_free_footprint(item):
        """Keep placed AABBs apart while allowing a large support to be reused."""
        _, surface, position, _, object_size = item
        for placed in object_placements.values():
            if abs(float(surface[1]) - float(placed["surface_y"])) > 0.08:
                continue
            other_position, other_size = placed["position"], placed["size"]
            overlap_x = abs(float(position[0]) - float(other_position[0])) < (
                (float(object_size[0]) + float(other_size[0])) / 2.0 + 0.03
            )
            overlap_z = abs(float(position[2]) - float(other_position[2])) < (
                (float(object_size[2]) + float(other_size[2])) / 2.0 + 0.03
            )
            if overlap_x and overlap_z:
                return False
        return True

    for index, step in enumerate(plan.get("steps", [])):
        action = str(step.get("action", "")).strip().lower()
        if action not in {"spawn", "move"}:
            continue
        old = None
        if action == "spawn":
            template = resolve_template_name(step.get("template"), templates)
            if template is None:
                raise ValueError(f"step {index}: template non disponibile")
            name = str(step.get("name") or f"{template}_1").strip()
            object_templates[name] = template
        else:
            name = str(step.get("object") or "").strip()
            template = object_templates.get(name)
            if template is None:
                raise ValueError(f"step {index}: oggetto '{name}' non disponibile")
            old = object_placements.pop(name, None)
        old_placement_id = old["placement_id"] if old else None
        constraint = constraints[placement_index] if placement_index < len(constraints) else {}
        candidates = [
            (point_id, item) for point_id, item in valid_for_template(template).items()
            if point_matches_support_constraint(item[0], constraint)
            and has_free_footprint(item)
            and item[0].get("placement_id") != old_placement_id
            and (
                not plan.get("distinct_destinations", False)
                or item[0].get("placement_id") not in used_placements
            )
        ]
        if not candidates:
            raise ValueError(
                f"step {index}: nessun placement valido per {template} "
                f"con vincolo {json.dumps(constraint, ensure_ascii=False)}"
            )
        point_id, (point, surface, position, _, object_size) = max(
            candidates, key=lambda item: (item[1][3], str(item[0]))
        )
        # placement_id identifica la superficie semantica, ma una superficie
        # può avere più punti geometrici. Conserviamo anche il punto preciso
        # già validato, evitando che semantic_placement_point() ne scelga un
        # altro durante compile_plan().
        step["target_point"] = point_id
        step["placement_id"] = point["placement_id"]
        used_placements.add(point["placement_id"])
        object_placements[name] = {
            "placement_id": point["placement_id"],
            "position": position,
            "surface_y": surface[1],
            "size": object_size,
        }
        placement_index += 1


def semantic_region_categories_at(sim, position):
    """Restituisce le categorie delle stanze che contengono un punto."""
    semantic_scene = getattr(sim, "semantic_scene", None)
    regions = list(getattr(semantic_scene, "regions", None) or [])
    getter = getattr(semantic_scene, "get_regions_for_point", None)
    if getter is None:
        return []
    try:
        found = list(getter(mn.Vector3(position)) or [])
    except (TypeError, ValueError, RuntimeError):
        return []

    labels = set()
    region_by_id = {str(getattr(region, "id", index)): region
                    for index, region in enumerate(regions)}
    for reference in found:
        region = reference if hasattr(reference, "category") else None
        if region is None:
            try:
                index = int(reference)
                if 0 <= index < len(regions):
                    region = regions[index]
            except (TypeError, ValueError):
                region = region_by_id.get(str(reference))
        if region is None:
            continue
        category = getattr(region, "category", None)
        try:
            label = str(category.name()).strip().lower()
        except (AttributeError, TypeError):
            label = str(category or "").strip().lower()
        if label:
            labels.add(label)
    return sorted(labels)


def semantic_asset_paths(scene):
    """Trova gli asset semantic.glb/.txt corrispondenti alla scena."""
    configured = os.environ.get("HABITAT_SEMANTIC_MESH", "").strip()
    if configured:
        mesh = Path(configured)
    else:
        scene_path = Path(scene)
        mesh = Path(str(scene_path).replace(
            "hm3d-val-habitat-v0.2", "hm3d-val-semantic-annots-v0.2"
        ).replace(".basis.glb", ".semantic.glb"))
    text = mesh.with_suffix(".txt")
    return mesh, text


def _read_glb(scene_mesh):
    raw = Path(scene_mesh).read_bytes()
    if raw[:4] != b"glTF":
        raise RuntimeError(f"mesh semantica non GLB: {scene_mesh}")
    offset = 12
    json_chunk = None
    binary_chunk = b""
    while offset + 8 <= len(raw):
        length, chunk_type = struct.unpack_from("<II", raw, offset)
        chunk = raw[offset + 8:offset + 8 + length]
        if chunk_type == 0x4E4F534A:
            json_chunk = json.loads(chunk.rstrip(b" \t\r\n\x00").decode("utf-8"))
        elif chunk_type == 0x004E4942:
            binary_chunk = chunk
        offset += 8 + length
    if json_chunk is None:
        raise RuntimeError(f"GLB senza chunk JSON: {scene_mesh}")
    return json_chunk, binary_chunk


def _glb_accessor(gltf, binary, accessor_index):
    accessor = gltf["accessors"][accessor_index]
    view = gltf["bufferViews"][accessor["bufferView"]]
    component_formats = {5121: "B", 5123: "H", 5125: "I", 5126: "f"}
    component_sizes = {5121: 1, 5123: 2, 5125: 4, 5126: 4}
    component_type = accessor["componentType"]
    type_count = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4}[accessor["type"]]
    count = accessor["count"]
    stride = view.get("byteStride", component_sizes[component_type] * type_count)
    start = view.get("byteOffset", 0) + accessor.get("byteOffset", 0)
    fmt = "<" + component_formats[component_type] * type_count
    values_out = []
    for index in range(count):
        values_out.append(struct.unpack_from(fmt, binary, start + index * stride))
    if accessor.get("normalized") and component_type != 5126:
        maximum = {5121: 255.0, 5123: 65535.0, 5125: 4294967295.0}[component_type]
        values_out = [tuple(value / maximum for value in row) for row in values_out]
    return values_out


def _glb_image_rgb(gltf, binary, image_index, cache):
    if image_index not in cache:
        image = gltf["images"][image_index]
        view = gltf["bufferViews"][image["bufferView"]]
        start = view.get("byteOffset", 0)
        end = start + view["byteLength"]
        cache[image_index] = PILImage.open(BytesIO(binary[start:end])).convert("RGB")
    return cache[image_index]


def _gltf_node_matrix(node):
    """Restituisce la trasformazione locale glTF come matrice NumPy."""
    if "matrix" in node:
        return np.asarray(node["matrix"], dtype=np.float64).reshape((4, 4), order="F")
    translation = np.asarray(node.get("translation", (0.0, 0.0, 0.0)), dtype=np.float64)
    scale = np.asarray(node.get("scale", (1.0, 1.0, 1.0)), dtype=np.float64)
    x, y, z, w = (float(value) for value in node.get("rotation", (0.0, 0.0, 0.0, 1.0)))
    transform = np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w), 0.0],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w), 0.0],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y), 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ], dtype=np.float64)
    transform[:3, :3] *= scale[np.newaxis, :]
    transform[:3, 3] = translation
    return transform


def _gltf_mesh_transforms(gltf):
    """Mappa ogni mesh alle sue istanze e trasformazioni mondo."""
    nodes = gltf.get("nodes", [])
    scenes = gltf.get("scenes", [])
    scene_index = int(gltf.get("scene", 0))
    roots = scenes[scene_index].get("nodes", []) if scene_index < len(scenes) else []
    if not roots:
        children = {child for node in nodes for child in node.get("children", [])}
        roots = [index for index in range(len(nodes)) if index not in children]
    transforms = {}

    def visit(node_index, parent):
        node = nodes[node_index]
        world = parent @ _gltf_node_matrix(node)
        if "mesh" in node:
            transforms.setdefault(int(node["mesh"]), []).append((node_index, world))
        for child_index in node.get("children", []):
            visit(int(child_index), world)

    for root_index in roots:
        visit(int(root_index), np.identity(4, dtype=np.float64))
    return transforms


def _transform_position(matrix, position):
    point = matrix @ np.array([position[0], position[1], position[2], 1.0])
    return point[:3] / point[3] if abs(point[3]) > 1e-12 else point[:3]


def _hm3d_semantic_to_habitat(position):
    """Converte l'asset semantico HM3D Z-up nel mondo Habitat Y-up.

    La configurazione HM3D dichiara up=+Z e front=+Y; Habitat usa up=+Y e
    front=-Z. Il parser GLB manuale non passa dall'AssetManager e deve quindi
    applicare esplicitamente la stessa rotazione usata dal simulatore.
    """
    x, y, z = (float(value) for value in position)
    return np.array([x, z, -y], dtype=np.float64)


def semantic_mesh_supports(scene):
    """Restituisce AABB mondo per istanza usando i colori della mesh GLB."""
    mesh_path, text_path = semantic_asset_paths(scene)
    if not mesh_path.is_file() or not text_path.is_file():
        raise RuntimeError(
            f"asset semantici mancanti: {mesh_path} e/o {text_path}"
        )
    color_to_category = {}
    pattern = re.compile(r'^\s*\d+\s*,\s*([0-9A-Fa-f]{6})\s*,\s*"([^"]+)"')
    for line in text_path.read_text(encoding="utf-8", errors="replace").splitlines():
        match = pattern.match(line)
        if match:
            color_to_category[tuple(int(match.group(1)[i:i + 2], 16) for i in (0, 2, 4))] = match.group(2).lower()

    gltf, binary = _read_glb(mesh_path)
    supports = {}
    image_cache = {}
    mesh_transforms = _gltf_mesh_transforms(gltf)
    for mesh_index, mesh in enumerate(gltf.get("meshes", [])):
        instances = mesh_transforms.get(mesh_index, [(-1, np.identity(4, dtype=np.float64))])
        for primitive in mesh.get("primitives", []):
            attrs = primitive.get("attributes", {})
            if "POSITION" not in attrs:
                continue
            positions = _glb_accessor(gltf, binary, attrs["POSITION"])
            indices = (
                [int(row[0]) for row in _glb_accessor(gltf, binary, primitive["indices"])]
                if "indices" in primitive else list(range(len(positions)))
            )

            texture = None
            if "TEXCOORD_0" in attrs and "material" in primitive:
                texcoords = _glb_accessor(gltf, binary, attrs["TEXCOORD_0"])
                material = gltf.get("materials", [])[primitive["material"]]
                texture = material.get("pbrMetallicRoughness", {}).get("baseColorTexture")
                if texture is not None:
                    texture_info = gltf.get("textures", [])[texture["index"]]
                    image = _glb_image_rgb(gltf, binary, texture_info["source"], image_cache)
                    for node_index, transform in instances:
                        for start in range(0, len(indices) - 2, 3):
                            triangle = indices[start:start + 3]
                            uv = tuple(sum(float(texcoords[index][axis]) for index in triangle) / 3.0 for axis in (0, 1))
                            u = min(0.999999, max(0.0, uv[0]))
                            v = min(0.999999, max(0.0, uv[1]))
                            rgb = image.getpixel((int(u * image.width), int((1.0 - v) * image.height)))
                            category = color_to_category.get(rgb)
                            if not is_geometric_support_candidate(category):
                                continue
                            # In HM3D ogni colore identifica un'istanza
                            # semantica. La sua mesh può essere divisa fra più
                            # nodi glTF, che vanno riuniti nello stesso oggetto.
                            entry = supports.setdefault(rgb, {
                                "category": category, "color": rgb,
                                "surface_group": f"semantic:{''.join(f'{channel:02x}' for channel in rgb)}",
                                "min": [float("inf")] * 3,
                                "max": [float("-inf")] * 3,
                                "surface_samples": [],
                            })
                            triangle_points = [
                                _hm3d_semantic_to_habitat(
                                    _transform_position(transform, positions[index])
                                )
                                for index in triangle
                            ]
                            for point in triangle_points:
                                for axis in range(3):
                                    entry["min"][axis] = min(entry["min"][axis], float(point[axis]))
                                    entry["max"][axis] = max(entry["max"][axis], float(point[axis]))
                            normal = np.cross(
                                triangle_points[1] - triangle_points[0],
                                triangle_points[2] - triangle_points[0],
                            )
                            normal_length = float(np.linalg.norm(normal))
                            if (
                                normal_length > 1e-8
                                # Una faccia rivolta verso il basso e' il lato
                                # inferiore di mensole e piani: appoggiarvi un
                                # oggetto lo lascia sospeso sotto il mobile.
                                and float(normal[1]) / normal_length >= 0.90
                                and len(entry["surface_samples"]) < MAX_SEMANTIC_SURFACE_SAMPLES
                            ):
                                entry["surface_samples"].append(
                                    [float(value) for value in np.mean(triangle_points, axis=0)]
                                )
                    continue
    return list(supports.values())


def simulator_semantic_supports(sim):
    """Legge gli AABB mondo dal SemanticScene di Habitat, se disponibili.

    Gli AABB ricavati a mano dal GLB non includono necessariamente le
    trasformazioni dei nodi glTF. Habitat invece espone qui gli oggetti
    semantici gia' trasformati nel frame della scena.
    """
    semantic_scene = getattr(sim, "semantic_scene", None)
    objects = getattr(semantic_scene, "objects", None)
    if not objects:
        return []
    supports = []
    for semantic_object in objects:
        if semantic_object is None:
            continue
        category = getattr(semantic_object, "category", None)
        category_name = ""
        try:
            category_name = str(category.name())
        except (AttributeError, TypeError):
            category_name = str(category or "")
        if not is_geometric_support_candidate(category_name):
            continue
        aabb = getattr(semantic_object, "aabb", None)
        if aabb is None:
            continue
        try:
            minimum, maximum = values(aabb.min), values(aabb.max)
        except (AttributeError, TypeError, ValueError):
            continue
        if not all(math.isfinite(value) for value in minimum + maximum):
            continue
        if max(maximum[axis] - minimum[axis] for axis in range(3)) <= 1e-6:
            continue
        supports.append({
            "category": category_name.strip().lower(),
            "semantic_id": getattr(semantic_object, "id", None),
            "min": minimum,
            "max": maximum,
        })
    return supports


def values(v):
    return [float(v[0]), float(v[1]), float(v[2])]


def valid_position(position, label="position"):
    """Restituisce una posa sicura e serializzabile oppure solleva ValueError."""
    if not isinstance(position, (list, tuple)) or len(position) != 3:
        raise ValueError(f"{label} deve contenere esattamente tre numeri")
    try:
        result = [float(value) for value in position]
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} deve contenere solo numeri") from exc
    if not all(math.isfinite(value) for value in result):
        raise ValueError(f"{label} deve contenere solo numeri finiti")
    if any(abs(value) > 100.0 for value in result):
        raise ValueError(f"{label} contiene coordinate fuori dal limite +/-100 m")
    return result


def available_template_names(objects_dir):
    objects_dir = resolve_object_configs_dir(objects_dir)
    if not os.path.isdir(objects_dir):
        return []
    return sorted({
        Path(name).name.replace(".object_config.json", "")
        for name in os.listdir(objects_dir)
        if name.endswith(".object_config.json")
    }, key=str.lower)


def resolve_template_name(template, templates):
    """Risolve un nome LLM senza accettare corrispondenze ambigue."""
    if not isinstance(template, str) or not template.strip():
        return None
    wanted = Path(template.strip()).name.lower().replace(".object_config.json", "")
    matches = [name for name in templates if name.lower() == wanted]
    return matches[0] if len(matches) == 1 else None


def scene_bounds(sim):
    low, high = sim.pathfinder.get_bounds()
    if all(float(low[i]) == float(high[i]) for i in range(3)):
        bb = sim.get_active_scene_graph().get_root_node().cumulative_bb
        low, high = bb.min, bb.max
    return values(low), values(high)


def load_navmesh(sim, scene, navmesh=""):
    """Carica la navmesh esplicitamente: Habitat non la carica sempre da solo."""
    if sim.pathfinder.is_loaded:
        return None
    candidate = navmesh or os.environ.get("HABITAT_NAVMESH", "")
    if not candidate:
        scene_path = Path(scene)
        if scene_path.name.endswith(".basis.glb"):
            candidate = str(scene_path.with_name(scene_path.name[:-4] + ".navmesh"))
        elif scene_path.suffix == ".glb":
            candidate = str(scene_path.with_suffix(".navmesh"))
    if not candidate or not Path(candidate).is_file():
        raise RuntimeError(
            "Navmesh non caricata. Specifica --navmesh o HABITAT_NAVMESH "
            "con il file .navmesh della scena."
        )
    if not sim.pathfinder.load_nav_mesh(candidate):
        raise RuntimeError(f"Impossibile caricare la navmesh: {candidate}")
    return candidate


def _quat_to_rotmat(rotation):
    """Matrice mondo<-camera per un quaternion Magnum/Habitat."""
    qx, qy, qz, qw = (float(rotation.x), float(rotation.y),
                      float(rotation.z), float(rotation.w))
    return np.array([
        [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
        [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
        [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)],
    ], dtype=np.float64)


def _agent_rotation_coeffs(rotation):
    """Adatta un Quaternion Magnum all'unico formato accettato da AgentState.

    Habitat-Sim Python non accetta direttamente mn.Quaternion in
    Agent.set_state(), ma il quaternion resta Magnum per tutti i calcoli di
    posa; qui estraiamo soltanto la rappresentazione xyzw richiesta dal
    wrapper Python dell'agente.
    """
    return np.array([
        float(rotation.vector.x), float(rotation.vector.y),
        float(rotation.vector.z), float(rotation.scalar),
    ], dtype=np.float64)


def _semantic_categories(sim):
    """Associa gli ID dell'immagine semantic alle categorie Habitat."""
    categories = {}
    objects = getattr(getattr(sim, "semantic_scene", None), "objects", []) or []
    for object_index, obj in enumerate(objects):
        if obj is None:
            continue
        try:
            category = str(obj.category.name()).strip().lower()
        except (AttributeError, TypeError):
            category = ""
        if not category:
            continue
        # Nei dataset HM3D ``obj.id`` è una stringa come ``table_44``, mentre
        # il semantic sensor restituisce l'indice dell'oggetto nel vettore
        # SemanticScene.objects. È la stessa convenzione usata dagli esempi
        # ufficiali Habitat-Sim (semantic_scene.objects[pixel_id]).
        categories[int(object_index)] = category
        for identifier in (getattr(obj, "id", None), getattr(obj, "semantic_id", None)):
            if identifier is not None:
                # HM3D puo' esporre voci di servizio come "Unknown_0": non
                # sono valori che il semantic sensor restituisce come uint32.
                try:
                    categories[int(identifier)] = category
                except (TypeError, ValueError):
                    continue
    return categories


def rendered_semantic_supports(sim):
    """Ricava piani d'appoggio dalle immagini depth+semantic di Habitat.

    Ogni punto nasce da un pixel realmente visibile e dalla sua profondita':
    questo evita di usare una faccia del semantic.glb che non corrisponde al
    piano mostrato dal renderer/collision mesh. I campioni sono raggruppati
    per semantic id, dunque una stessa superficie non diventa decine di target.
    """
    categories = _semantic_categories(sim)
    if not categories:
        return []
    low, high = sim.pathfinder.get_bounds()
    low, high = np.asarray(values(low), dtype=np.float64), np.asarray(values(high), dtype=np.float64)
    agent = sim.initialize_agent(0)
    poses = []
    seen_poses = set()
    for x in np.linspace(low[0], high[0], RENDERED_SUPPORT_GRID_SIZE):
        for z in np.linspace(low[2], high[2], RENDERED_SUPPORT_GRID_SIZE):
            floor = sim.pathfinder.snap_point(mn.Vector3(float(x), float((low[1] + high[1]) / 2.0), float(z)))
            if not all(math.isfinite(float(value)) for value in (floor.x, floor.y, floor.z)):
                continue
            key = (round(float(floor.x), 1), round(float(floor.z), 1))
            if key not in seen_poses:
                seen_poses.add(key)
                poses.append(floor)

    supports = {}
    # Gli stessi parametri usati dagli spec creati in main().
    hfov = math.radians(90.0)
    for floor in poses:
        for yaw in RENDERED_SUPPORT_YAWS:
            state = agent.get_state()
            state.position = floor
            rotation = mn.Quaternion.rotation(
                mn.Deg(yaw), mn.Vector3(0.0, 1.0, 0.0)
            )
            state.rotation = _agent_rotation_coeffs(rotation)
            agent.set_state(state, reset_sensors=True, infer_sensor_states=True)
            observations = sim.get_sensor_observations()
            depth = observations.get("depth_sensor")
            semantic = observations.get("semantic_sensor")
            sensor_state = agent.get_state().sensor_states.get("depth_sensor")
            if depth is None or semantic is None or sensor_state is None:
                continue
            height, width = depth.shape[:2]
            fx = (width / 2.0) / math.tan(hfov / 2.0)
            fy, cx, cy = fx, width / 2.0, height / 2.0
            rotation = _quat_to_rotmat(sensor_state.rotation)
            origin = np.asarray(values(sensor_state.position), dtype=np.float64)

            def world_point(row, column):
                distance = float(depth[row, column])
                if not math.isfinite(distance) or distance <= 0.05 or distance > 8.0:
                    return None
                # Camera Habitat: X destra, Y alto, -Z in avanti.
                camera = np.array([
                    (column - cx) * distance / fx,
                    -(row - cy) * distance / fy,
                    -distance,
                ], dtype=np.float64)
                return origin + rotation @ camera

            for row in range(2, height - 2, RENDERED_SUPPORT_PIXEL_STRIDE):
                for column in range(2, width - 2, RENDERED_SUPPORT_PIXEL_STRIDE):
                    semantic_id = int(semantic[row, column])
                    category = categories.get(semantic_id, "")
                    if not is_geometric_support_candidate(category):
                        continue
                    point = world_point(row, column)
                    right, down = world_point(row, column + 2), world_point(row + 2, column)
                    if point is None or right is None or down is None:
                        continue
                    normal = np.cross(right - point, down - point)
                    normal_size = float(np.linalg.norm(normal))
                    # Orientiamo la normale verso la camera: il verso grezzo
                    # del prodotto vettoriale dipende dalla convenzione degli
                    # assi immagine. Un piano utilizzabile deve essere visto
                    # dall'alto e avere quindi una normale orientata +Y.
                    if normal_size > 1e-8 and float(np.dot(normal, origin - point)) < 0.0:
                        normal = -normal
                    if (
                        normal_size <= 1e-8
                        or float(origin[1]) <= float(point[1]) + 0.05
                        or float(normal[1]) / normal_size < 0.85
                    ):
                        continue
                    snapped = sim.pathfinder.snap_point(mn.Vector3(*point))
                    if not all(math.isfinite(float(value)) for value in (snapped.x, snapped.y, snapped.z)):
                        continue
                    if math.hypot(float(snapped.x) - point[0], float(snapped.z) - point[2]) > MAX_SUPPORT_HORIZONTAL_NAVMESH_DISTANCE:
                        continue
                    support_height = point[1] - float(snapped.y)
                    if not MIN_SUPPORT_HEIGHT_ABOVE_NAVMESH <= support_height <= MAX_SUPPORT_HEIGHT_ABOVE_NAVMESH:
                        continue
                    # Uno stesso oggetto semantico può contenere più ripiani.
                    # Separarli per quota evita di mediare piani distinti.
                    support_key = (semantic_id, round(float(point[1]) / 0.08))
                    entry = supports.setdefault(support_key, {
                        "category": category,
                        "surface_group": f"rendered:{semantic_id}:{support_key[1]}",
                        "surface_samples": [],
                        "min": point.copy(), "max": point.copy(),
                    })
                    if len(entry["surface_samples"]) < MAX_SEMANTIC_SURFACE_SAMPLES:
                        entry["surface_samples"].append(point.tolist())
                    entry["min"] = np.minimum(entry["min"], point)
                    entry["max"] = np.maximum(entry["max"], point)
    result = []
    for entry in supports.values():
        if entry["surface_samples"]:
            entry["min"] = entry["min"].tolist()
            entry["max"] = entry["max"].tolist()
            result.append(entry)
    return result


def collision_support_point(sim, position, height_tolerance=MAX_SEMANTIC_COLLISION_HEIGHT_ERROR):
    """Restituisce il vero piano collisionale sotto la X/Z semantica."""
    x, expected_y, z = valid_position(position, "support position")
    origin = mn.Vector3(x, expected_y + 0.50, z)
    hits = sim.cast_ray(habitat_sim.geo.Ray(origin, mn.Vector3(0.0, -1.0, 0.0)))
    if not hits.has_hits():
        return None
    candidates = []
    stage_id = getattr(habitat_sim, "stage_id", None)
    for hit in hits.hits:
        if stage_id is not None and int(hit.object_id) != int(stage_id):
            continue
        hit_y = float(hit.point.y)
        if abs(hit_y - expected_y) > float(height_tolerance):
            continue
        normal = getattr(hit, "normal", None)
        if normal is not None:
            try:
                normal_length = math.sqrt(sum(float(normal[index]) ** 2 for index in range(3)))
                if normal_length <= 1e-8 or float(normal.y) / normal_length < 0.75:
                    continue
            except (AttributeError, TypeError, ValueError, IndexError):
                continue
        candidates.append(hit)
    if not candidates:
        return None
    support = min(candidates, key=lambda hit: abs(float(hit.point.y) - expected_y))
    return [float(support.point.x), float(support.point.y), float(support.point.z)]


def generate_points(sim, step=0.25, margin=0.01, scene=None):
    try:
        step = float(step)
        margin = float(margin)
    except (TypeError, ValueError) as exc:
        raise ValueError("step e margin devono essere numeri") from exc
    if not math.isfinite(step) or step <= 0:
        raise ValueError("step deve essere un numero finito maggiore di zero")
    if not math.isfinite(margin) or margin < 0:
        raise ValueError("margin deve essere un numero finito non negativo")
    if not sim.pathfinder.is_loaded:
        raise ValueError("navmesh non caricata")
    # La prima scelta e' il renderer Habitat: semantic id, profondita' e posa
    # della camera condividono certamente lo stesso frame della scena.
    semantic_objects = rendered_semantic_supports(sim)
    semantic_source = "rendered semantic+depth"
    if not semantic_objects:
        semantic_objects = simulator_semantic_supports(sim)
        semantic_source = "Habitat SemanticScene"
    if not any(item.get("surface_samples") for item in semantic_objects):
        semantic_objects = semantic_mesh_supports(
            scene or os.environ.get("HABITAT_SCENE", "")
        )
        semantic_source = "semantic GLB fallback"
    if not semantic_objects:
        raise RuntimeError("nessun oggetto semantico supporto trovato nella mesh")

    points = {}
    point_index = 0
    diagnostic = Counter()
    categories_seen = Counter()
    categories_support = Counter()
    aabb_samples = []
    for semantic_object in semantic_objects:
        diagnostic["semantic_objects"] += 1
        category_name = str(semantic_object["category"]).strip().lower()
        if category_name:
            categories_seen[category_name] += 1
        if not is_geometric_support_candidate(category_name):
            continue
        diagnostic["support_categories"] += 1
        categories_support[category_name] += 1
        minimum = mn.Vector3(semantic_object["min"])
        maximum = mn.Vector3(semantic_object["max"])
        width = float(maximum.x - minimum.x)
        depth = float(maximum.z - minimum.z)
        if len(aabb_samples) < 12:
            aabb_samples.append({
                "category": category_name,
                "min": values(minimum),
                "max": values(maximum),
                "size": [width, float(maximum.y - minimum.y), depth],
            })
        samples = semantic_object.get("surface_samples", [])
        if not samples:
            diagnostic["no_horizontal_semantic_triangle"] += 1
            continue
        seen_samples = set()
        valid_samples = []
        for sample in samples:
            x, surface_y, z = valid_position(sample, "semantic surface")
            collision_point = collision_support_point(sim, [x, surface_y, z])
            if collision_point is None:
                diagnostic["no_matching_collision_surface"] += 1
                continue
            x, surface_y, z = collision_point
            # Triangoli adiacenti producono molti centroidi quasi coincidenti.
            sample_key = (round(x / max(step, 0.1)), round(z / max(step, 0.1)), round(surface_y, 2))
            if sample_key in seen_samples:
                continue
            seen_samples.add(sample_key)
            floor = sim.pathfinder.snap_point(mn.Vector3(x, surface_y, z))
            if not all(math.isfinite(float(value)) for value in (floor.x, floor.y, floor.z)):
                diagnostic["invalid_navmesh"] += 1
                continue
            if math.hypot(float(floor.x) - x, float(floor.z) - z) > MAX_SUPPORT_HORIZONTAL_NAVMESH_DISTANCE:
                diagnostic["far_from_navmesh"] += 1
                continue
            height = surface_y - float(floor.y)
            if not MIN_SUPPORT_HEIGHT_ABOVE_NAVMESH <= height <= MAX_SUPPORT_HEIGHT_ABOVE_NAVMESH:
                diagnostic["height_filter"] += 1
                continue
            valid_samples.append([x, surface_y, z])
        if not valid_samples:
            continue
        # Conserviamo piu' posizioni distribuite sullo stesso supporto. Un
        # unico centro non e' generale: sui counter spesso coincide con un
        # lavabo o con i fornelli. Il campionamento farthest-point evita sia
        # centinaia di centroidi quasi uguali sia il bias verso il primo lato
        # osservato dal renderer.
        center = np.mean(np.asarray(valid_samples, dtype=np.float64), axis=0)
        representatives = [min(
            valid_samples,
            key=lambda sample: (sample[0] - center[0]) ** 2 + (sample[2] - center[2]) ** 2,
        )]
        remaining = [sample for sample in valid_samples if sample is not representatives[0]]
        while remaining and len(representatives) < MAX_TARGET_POINTS_PER_SURFACE:
            next_sample = max(
                remaining,
                key=lambda sample: min(
                    (sample[0] - chosen[0]) ** 2 + (sample[2] - chosen[2]) ** 2
                    for chosen in representatives
                ),
            )
            # I campioni piu' vicini dello step configurato descrivono di
            # fatto la stessa posa e non aggiungono scelta utile.
            min_distance_sq = min(
                (next_sample[0] - chosen[0]) ** 2
                + (next_sample[2] - chosen[2]) ** 2
                for chosen in representatives
            )
            if min_distance_sq < max(step, 0.1) ** 2:
                break
            representatives.append(next_sample)
            remaining.remove(next_sample)

        surface_group = semantic_object.get("surface_group")
        for x, surface_y, z in representatives:
            point_id = f"surface_{point_index:04d}"
            room_categories = set(semantic_region_categories_at(
                sim, [x, surface_y, z]
            ))
            floor = sim.pathfinder.snap_point(mn.Vector3(x, surface_y, z))
            if all(math.isfinite(float(value)) for value in (floor.x, floor.y, floor.z)):
                room_categories.update(semantic_region_categories_at(
                    sim, [float(floor.x), float(floor.y) + 0.1, float(floor.z)]
                ))
            points[point_id] = {
                "id": point_id,
                "surface_point": [x, surface_y, z],
                "normal": [0.0, 1.0, 0.0],
                "category": category_name,
                "room_categories": sorted(room_categories),
                "surface_group": surface_group or point_id,
                "support_bounds": [values(minimum), values(maximum)],
                "semantic_color": semantic_object.get("color"),
                "margin": margin,
            }
            point_index += 1
            diagnostic["points"] += 1
    print(
        "Support diagnostic: "
        + json.dumps({
            "counts": dict(diagnostic),
            "source": semantic_source,
            "support_categories": dict(categories_support),
            "categories_sample": dict(categories_seen.most_common(30)),
            "aabb_samples": aabb_samples,
        }, ensure_ascii=False),
        flush=True,
    )
    if not points:
        raise ValueError(
            "nessuna superficie semantica verificabile; "
            "consultare il report 'Support diagnostic' precedente"
        )
    add_semantic_surface_ids(points)
    return points


def add_semantic_surface_ids(points):
    """Aggiunge riferimenti semantici senza esporre la geometria alla LLM."""
    support_keys = {}
    for point in points.values():
        group = str(point.get("surface_group") or point["id"])
        parts = group.split(":")
        # I gruppi rendered condividono il semantic id anche quando sono
        # ripiani diversi: la quota resta invece parte della superficie.
        support_key = ":".join(parts[:2]) if parts[0] == "rendered" and len(parts) >= 2 else group
        support_keys.setdefault(support_key, []).append(point)

    ordered_supports = sorted(
        support_keys.items(),
        key=lambda item: (str(item[1][0].get("category", "")), item[0]),
    )
    for support_index, (_, support_points) in enumerate(ordered_supports, 1):
        category = str(support_points[0].get("category", "support")).strip().lower()
        safe_category = re.sub(r"[^a-z0-9]+", "_", category).strip("_") or "support"
        support_id = f"{safe_category}_{support_index:02d}"
        surface_groups = {}
        for point in support_points:
            surface_groups.setdefault(str(point.get("surface_group") or point["id"]), []).append(point)
        for surface_index, (_, grouped_points) in enumerate(sorted(surface_groups.items()), 1):
            surface_id = f"{support_id}_surface_{surface_index:02d}"
            for point in grouped_points:
                point["support_id"] = support_id
                point["surface_id"] = surface_id
                point["placement_id"] = f"{support_id}::{surface_id}"


def semantic_surface_catalog(points):
    """Restituisce solo il catalogo semantico destinato alla LLM."""
    grouped = {}
    for point in points.values():
        support_id = point.get("support_id")
        surface_id = point.get("surface_id")
        if not support_id or not surface_id:
            continue
        support = grouped.setdefault(support_id, {
            "support_id": support_id,
            "type": str(point.get("category", "")),
            "room_categories": sorted(set(point.get("room_categories", []))),
            "surfaces": {},
        })
        support["room_categories"] = sorted(set(support["room_categories"]) | set(point.get("room_categories", [])))
        support["surfaces"].setdefault(surface_id, {
            "surface_id": surface_id,
            "placement_id": f"{support_id}::{surface_id}",
            "type": "horizontal_surface",
        })
    result = []
    for support in grouped.values():
        support["surfaces"] = list(support["surfaces"].values())
        result.append(support)
    return sorted(result, key=lambda item: item["support_id"])


def semantic_target_point(points, support_id, surface_id):
    """Risolve un riferimento semantico nel miglior punto geometrico disponibile."""
    candidates = [
        point for point in points.values()
        if point.get("support_id") == support_id and point.get("surface_id") == surface_id
    ]
    if not candidates:
        raise ValueError(f"superficie semantica non trovata: {support_id}/{surface_id}")
    return sorted(candidates, key=lambda point: str(point.get("id", "")))[0]["id"]


def semantic_placement_point(points, placement_id):
    """Risolve un placement_id unico nel miglior punto geometrico disponibile."""
    candidates = [
        point for point in points.values()
        if point.get("placement_id") == placement_id
    ]
    if not candidates:
        raise ValueError(f"placement_id non trovato: {placement_id}")
    return sorted(candidates, key=lambda point: str(point.get("id", "")))[0]["id"]


def template_handle_and_support_offset(sim, template_name, objects_dir):
    manager = sim.get_object_template_manager()
    configs_dir = resolve_object_configs_dir(objects_dir)
    if configs_dir and os.path.isdir(configs_dir):
        manager.load_configs(configs_dir)
    handles = manager.get_template_handles()
    requested = Path(str(template_name)).name.lower()
    if requested.endswith(".object_config.json"):
        requested = requested[:-len(".object_config.json")]
    handle = next(
        (h for h in handles
         if Path(str(h)).name.lower().replace(".object_config.json", "") == requested),
        None,
    )
    if handle is None:
        raise ValueError(f"template non trovato: {template_name}")
    obj = sim.get_rigid_object_manager().add_object_by_template_handle(handle)
    if obj is None:
        raise ValueError(f"template non istanziabile: {template_name}")
    apply_global_object_scale(obj)
    # L'AABB di collisione puo' differire dalla mesh renderizzata (come la
    # banana di example_objects). Per una posa visivamente corretta usiamo il
    # bounding box del sottoalbero visuale, espresso nel frame locale root.
    support_offset = -float(obj.root_scene_node.cumulative_bb.min.y)
    sim.get_rigid_object_manager().remove_object_by_id(obj.object_id)
    if not math.isfinite(support_offset) or support_offset < 0:
        raise ValueError(f"template con origine/AABB non valida: {template_name}")
    return str(handle), support_offset


def apply_global_object_scale(obj):
    """Applica la scala globale anche ai controlli temporanei del planner."""
    try:
        factor = float(os.environ.get("HABITAT_OBJECT_SCALE", "1.0"))
    except (TypeError, ValueError):
        factor = 1.0
    if not math.isfinite(factor) or factor <= 0:
        raise ValueError("HABITAT_OBJECT_SCALE deve essere un numero positivo")
    if abs(factor - 1.0) > 1e-6:
        # In questa build ManagedRigidObject.scale e' un Vector3 read-only;
        # la trasformazione va applicata al SceneNode radice.
        obj.root_scene_node.scale(mn.Vector3(factor))


def configured_object_scale():
    """Restituisce il fattore globale usato dal planner."""
    try:
        factor = float(os.environ.get("HABITAT_OBJECT_SCALE", "1.0"))
    except (TypeError, ValueError) as exc:
        raise ValueError("HABITAT_OBJECT_SCALE deve essere un numero positivo") from exc
    if not math.isfinite(factor) or factor <= 0:
        raise ValueError("HABITAT_OBJECT_SCALE deve essere un numero positivo")
    return factor


def template_object_size(sim, template_handle):
    """Return the scaled collision AABB size of a temporary template instance."""
    manager = sim.get_rigid_object_manager()
    obj = manager.add_object_by_template_handle(str(template_handle))
    if obj is None:
        raise ValueError(f"template non istanziabile: {template_handle}")
    try:
        apply_global_object_scale(obj)
        size = obj.aabb.size()
        return [float(size.x), float(size.y), float(size.z)]
    finally:
        manager.remove_object_by_id(obj.object_id)


def footprint_support_quality(
    sim, template_handle, position, surface,
    height_tolerance=MAX_FOOTPRINT_HEIGHT_ERROR,
):
    """Validate support beneath the centre, edges and corners of an object.

    Returns a small quality value when all nine rays hit the same horizontal
    stage surface. ``None`` means that some part of the footprint overhangs,
    intersects another height, or lacks collision support.
    """
    manager = sim.get_rigid_object_manager()
    obj = manager.add_object_by_template_handle(str(template_handle))
    if obj is None:
        return None
    try:
        apply_global_object_scale(obj)
        obj.motion_type = habitat_sim.physics.MotionType.KINEMATIC
        obj.translation = mn.Vector3(valid_position(position))
        size = obj.aabb.size()
        half_x = max(0.01, FOOTPRINT_SAMPLE_FRACTION * float(size.x))
        half_z = max(0.01, FOOTPRINT_SAMPLE_FRACTION * float(size.z))
        expected_y = float(valid_position(surface, "support surface")[1])
        stage_id = getattr(habitat_sim, "stage_id", None)
        deviations = []
        for dx, dz in (
            (0.0, 0.0),
            (-half_x, -half_z), (-half_x, half_z),
            (half_x, -half_z), (half_x, half_z),
            (-half_x, 0.0), (half_x, 0.0),
            (0.0, -half_z), (0.0, half_z),
        ):
            origin = mn.Vector3(
                float(position[0]) + dx, expected_y + 0.20,
                float(position[2]) + dz,
            )
            hits = sim.cast_ray(
                habitat_sim.geo.Ray(origin, mn.Vector3(0.0, -1.0, 0.0))
            )
            compatible = []
            for hit in getattr(hits, "hits", []):
                if int(hit.object_id) == int(obj.object_id):
                    continue
                if stage_id is not None and int(hit.object_id) != int(stage_id):
                    continue
                normal = getattr(hit, "normal", None)
                if normal is not None:
                    length = math.sqrt(sum(float(normal[i]) ** 2 for i in range(3)))
                    if length <= 1e-8 or float(normal.y) / length < 0.80:
                        continue
                deviation = abs(float(hit.point.y) - expected_y)
                if deviation <= float(height_tolerance):
                    compatible.append(deviation)
            if not compatible:
                return None
            deviations.append(min(compatible))
        return 1.0 - max(deviations) / max(float(height_tolerance), 1e-9)
    finally:
        manager.remove_object_by_id(obj.object_id)


def find_valid_capture_eye(sim, template_handle, position):
    """Return a deterministic navigable camera pose with direct visibility.

    La collision mesh di HM3D contiene talvolta piani nascosti dentro mobili.
    Un ray verticale li scambia per superfici valide; qui istanziamo
    temporaneamente lo stesso oggetto e richiediamo almeno una posa camera
    navigabile con linea di vista diretta sul suo collision object.
    """
    manager = sim.get_rigid_object_manager()
    obj = manager.add_object_by_template_handle(str(template_handle))
    if obj is None:
        return None
    try:
        apply_global_object_scale(obj)
        target = mn.Vector3(position)
        semantic_scene = getattr(sim, "semantic_scene", None)
        region_getter = getattr(semantic_scene, "get_regions_for_point", None)

        def regions_at(point):
            if region_getter is None:
                return set()
            try:
                return {int(region) for region in region_getter(mn.Vector3(point))}
            except (TypeError, ValueError, RuntimeError):
                return set()

        target_regions = regions_at(target)
        obj.motion_type = habitat_sim.physics.MotionType.KINEMATIC
        obj.translation = target
        # Un piano orizzontale puo' appartenere alla collision mesh di un TV
        # o di un mobile. Prima della visibilita' controlliamo che il volume
        # laterale dell'oggetto non intersechi altra geometria a distanza
        # inferiore al suo raggio: il target deve avere spazio reale attorno.
        size = obj.aabb.size()
        # Il controllo deve dipendere dall'ingombro reale del template. Il
        # precedente minimo fisso di 45 cm eliminava quasi tutti i banconi
        # addossati a una parete, anche per una banana larga pochi centimetri.
        # I piani interni restano esclusi dalla successiva linea di vista.
        clearance = max(0.05, 0.55 * max(float(size.x), float(size.z)) + 0.03)
        # Il centro del rigid body puo' stare vicino al fondo della mesh. I
        # raggi laterali dal fondo colpiscono il piano d'appoggio a distanza
        # quasi zero e invalidano ogni candidato; usiamo la meta' superiore
        # del volume per controllare pareti e mobili reali.
        probe_origin = target + mn.Vector3(0.0, max(0.05, 0.5 * float(size.y)), 0.0)
        for axis in (
            mn.Vector3(1.0, 0.0, 0.0), mn.Vector3(-1.0, 0.0, 0.0),
            mn.Vector3(0.0, 0.0, 1.0), mn.Vector3(0.0, 0.0, -1.0),
        ):
            hits = sim.cast_ray(habitat_sim.geo.Ray(probe_origin, axis))
            obstacle = next(
                (hit for hit in hits.hits if hit.object_id != obj.object_id), None
            )
            if obstacle is not None and float(obstacle.ray_distance) < clearance:
                return None
        directions = (
            (1.0, 0.0), (-1.0, 0.0), (0.0, 1.0), (0.0, -1.0),
            (0.707, 0.707), (0.707, -0.707), (-0.707, 0.707), (-0.707, -0.707),
        )
        eye_lifts = (
            max(0.15, 0.5 * float(size.y)),
            max(0.35, 1.0 * float(size.y)),
            max(0.80, 1.5 * float(size.y)),
        )
        for distance in (0.8, 1.25, 2.0, 2.75):
            for dx, dz in directions:
                candidate = target + mn.Vector3(distance * dx, 0.0, distance * dz)
                floor = sim.pathfinder.snap_point(candidate)
                if not all(math.isfinite(float(value)) for value in (floor.x, floor.y, floor.z)):
                    continue
                if math.hypot(float(floor.x - candidate.x), float(floor.z - candidate.z)) > 0.35:
                    continue
                for lift in eye_lifts:
                    eye = mn.Vector3(floor.x, target.y + lift, floor.z)
                    eye_regions = regions_at(eye)
                    if target_regions and eye_regions and not (target_regions & eye_regions):
                        continue
                    direction = target - eye
                    if direction.length() < 0.05:
                        continue
                    hits = sim.cast_ray(habitat_sim.geo.Ray(eye, direction.normalized()))
                    if hits.has_hits() and hits.hits[0].object_id == obj.object_id:
                        return values(eye)
        return None
    finally:
        manager.remove_object_by_id(obj.object_id)


def target_has_visible_view(sim, template_handle, position):
    """Compatibility predicate for callers that only need validity."""
    return find_valid_capture_eye(sim, template_handle, position) is not None


def compile_plan(
    plan, points, sim, objects_dir, margin=0.01, request_text="", template_catalog=None
):
    if not isinstance(plan, dict):
        raise ValueError("il piano deve essere un oggetto JSON")
    if not isinstance(plan.get("steps"), list) or not plan["steps"]:
        raise ValueError("il piano deve contenere almeno uno step")
    if len(plan["steps"]) > MAX_STEPS:
        raise ValueError(f"il piano supera il limite di {MAX_STEPS} step")
    if not math.isfinite(float(margin)) or float(margin) < 0:
        raise ValueError("margin deve essere un numero finito non negativo")

    compiled = {"steps": [], "object_scale": configured_object_scale()}
    compiled["description"] = str(plan.get("description", "")).strip()[:160]
    support_constraints = plan.get("support_constraints", [])
    if not isinstance(support_constraints, list):
        raise ValueError("support_constraints deve essere un array")
    dimensions = {}
    object_templates = {}
    available_templates = available_template_names(objects_dir)
    request_lower = str(request_text).lower()
    request_template = next(
        (item for item in available_templates if item.lower() in request_lower),
        None,
    )
    last_object_name = None
    live_objects = set()
    placement_index = 0
    raw_steps = plan["steps"]
    for index, step in enumerate(raw_steps):
        if not isinstance(step, dict):
            raise ValueError(f"step {index} non valido")
        action = str(step.get("action", "")).strip().lower()
        if action not in VALID_ACTIONS:
            raise ValueError(f"step {index}: azione non supportata: {action or '<vuota>'}")
        out = {"action": action}
        template = step.get("template")
        if template is None and action == "spawn":
            # Recupero automatico per piani LLM che hanno usato solo un nome
            # come banana_1 invece del campo template=banana.
            hint = str(step.get("name") or step.get("object") or "").lower()
            template = next(
                (item for item in available_templates
                 if hint == item.lower() or hint.startswith(item.lower() + "_")),
                None,
            )
            if template is None:
                template = request_template
        if template is not None:
            template = resolve_template_name(template, available_templates)
            if template is None:
                raise ValueError(f"step {index}: template non disponibile")
        if template is None and action == "move":
            reference = step.get("object") or step.get("name")
            template = object_templates.get(reference)
        if template and template not in dimensions:
            handle, support_offset = template_handle_and_support_offset(sim, template, objects_dir)
            dimensions[template] = (handle, support_offset)
        # Un target è utile solo per spawn/move. La LLM può averlo aggiunto
        # erroneamente a remove: in quel caso non va compilato.
        target = step.get("target_point") if action in {"spawn", "move"} else None
        placement_id = step.get("placement_id")
        semantic_support = step.get("support_id")
        semantic_surface = step.get("surface_id")
        if target is None and action in {"spawn", "move"}:
            if isinstance(placement_id, str) and placement_id.strip():
                target = semantic_placement_point(points, placement_id)
                semantic_support, semantic_surface = placement_id.split("::", 1)
            elif isinstance(semantic_support, str) and isinstance(semantic_surface, str):
                target = semantic_target_point(points, semantic_support, semantic_surface)
                placement_id = f"{semantic_support}::{semantic_surface}"
            else:
                raise ValueError(f"step {index}: serve placement_id")
        if target is not None:
            if not isinstance(target, str) or not target.strip():
                raise ValueError(
                    f"step {index}: la LLM ha restituito un target_point vuoto"
                )
            if target not in points:
                raise ValueError(f"target_point non trovato: {target}")
            target_metadata = points[target]
            if placement_id and template_catalog and template in template_catalog:
                allowed = {
                    surface["placement_id"]
                    for support in semantic_surface_catalog(template_catalog[template])
                    for surface in support["surfaces"]
                }
                if placement_id not in allowed:
                    raise ValueError(
                        f"step {index}: placement {placement_id} non valido "
                        f"per il template {template}"
                    )
            expected_constraint = (
                support_constraints[placement_index]
                if placement_index < len(support_constraints)
                else {}
            )
            if not point_matches_support_constraint(target_metadata, expected_constraint):
                requested_label = json.dumps(expected_constraint, ensure_ascii=False)
                raise ValueError(
                    f"step {index}: target_point {target} è categoria "
                    f"'{target_metadata.get('category', '')}', ma la richiesta "
                    f"richiede: {requested_label}"
                )
            if template:
                template_handle, support_offset = dimensions[template]
            elif step.get("object_template") in dimensions:
                template_handle, support_offset = dimensions[step["object_template"]]
            else:
                raise ValueError(f"step {index}: template necessario per calcolare la posa")
            surface = valid_position(target_metadata.get("surface_point"), "surface_point")
            collision_surface = collision_support_point(sim, surface)
            if collision_surface is None:
                raise ValueError(
                    f"step {index}: {target} non corrisponde più a un piano "
                    "orizzontale della collision mesh"
                )
            surface = collision_surface
            out["target_point"] = target
            if semantic_support:
                out["support_id"] = semantic_support
            if semantic_surface:
                out["surface_id"] = semantic_surface
            if placement_id:
                out["placement_id"] = placement_id
            out["position"] = [surface[0], surface[1] + support_offset + margin, surface[2]]
            if footprint_support_quality(
                sim, template_handle, out["position"], surface
            ) is None:
                raise ValueError(
                    f"step {index}: l'impronta completa di {template} non è "
                    f"supportata da {target}"
                )
            capture_eye = find_valid_capture_eye(
                sim, template_handle, out["position"]
            )
            if capture_eye is None:
                raise ValueError(
                    f"step {index}: {target} non ha spazio libero o una vista "
                    "navigabile sul supporto"
                )
            out["capture_eye"] = capture_eye
            out["target_category"] = str(target_metadata.get("category", ""))
            out["target_surface_point"] = surface
            out["target_surface_group"] = str(target_metadata.get("surface_group", target))
            out["visual_surface_validated"] = True
            placement_index += 1
        elif "position" in step and action in {"spawn", "move"}:
            out["position"] = valid_position(step["position"])

        if action == "spawn":
            if not template:
                raise ValueError(f"step {index}: spawn richiede name e template")
            name = str(step.get("name") or f"{template}_1").strip()
            if not name:
                raise ValueError(f"step {index}: name non valido")
            if name in live_objects or name in object_templates:
                raise ValueError(f"step {index}: name duplicato: {name}")
            if "position" not in out:
                raise ValueError(f"step {index}: spawn richiede target_point o position")
            out["name"] = name
            out["template"] = template
            object_templates[name] = template
            live_objects.add(name)
            last_object_name = name
        elif action in {"move", "remove"}:
            reference = step.get("object") or step.get("name") or last_object_name
            if not reference:
                raise ValueError(f"step {index}: {action} richiede object")
            # I nomi logici devono esistere nello script. Un ID numerico e'
            # ammesso per interagire intenzionalmente con un oggetto esterno.
            if isinstance(reference, str) and reference not in live_objects:
                raise ValueError(
                    f"step {index}: oggetto '{reference}' non disponibile "
                    "(spawn mancante, gia' rimosso o ordine errato)"
                )
            if action == "move" and "position" not in out:
                raise ValueError(f"step {index}: move richiede target_point o position")
            out["object"] = reference
            if action == "remove" and isinstance(reference, str):
                live_objects.remove(reference)
                if last_object_name == reference:
                    last_object_name = next(iter(live_objects), None)
        elif action == "wait":
            try:
                seconds = float(step.get("seconds", 0))
            except (TypeError, ValueError) as exc:
                raise ValueError(f"step {index}: seconds non valido") from exc
            if not math.isfinite(seconds) or not 0 <= seconds <= MAX_WAIT_SECONDS:
                raise ValueError(f"step {index}: seconds deve essere tra 0 e {MAX_WAIT_SECONDS}")
            out["seconds"] = seconds
        compiled["steps"].append(out)
    # Il piano descrive azioni fisiche: inseriamo una pausa esplicita tra una
    # posa e l'azione successiva. Questo rende il JSON autoesplicativo e non
    # dipende dal fatto che venga eseguito dal runner con le foto abilitate.
    try:
        settle_seconds = float(os.environ.get("HABITAT_SETTLE_SECONDS", DEFAULT_SETTLE_SECONDS))
    except ValueError:
        settle_seconds = DEFAULT_SETTLE_SECONDS
    settle_seconds = min(max(0.0, settle_seconds), MAX_WAIT_SECONDS)
    with_settling = []
    for index, step in enumerate(compiled["steps"]):
        with_settling.append(step)
        if (
            step["action"] in {"spawn", "move"}
            and index < len(compiled["steps"]) - 1
            and compiled["steps"][index + 1]["action"] != "wait"
        ):
            with_settling.append({
                "action": "wait", "seconds": settle_seconds,
                "reason": "settle",
            })
    compiled["steps"] = with_settling
    if len(compiled["steps"]) > MAX_COMPILED_STEPS:
        raise ValueError(
            f"il piano compilato supera il limite di {MAX_COMPILED_STEPS} step"
        )
    return compiled


def select_candidate_points(points, maximum):
    """Campiona uniformemente il catalogo, evitando il bias dei primi punti."""
    ordered = sorted(points.values(), key=lambda point: str(point.get("id", "")))
    if not ordered:
        raise ValueError("non sono stati trovati punti di appoggio validi nella scena")
    maximum = max(1, min(maximum, len(ordered)))
    # generate_points() ha già confrontato ogni punto con la navmesh locale e
    # ha escluso il pavimento. La Y minima è quindi un supporto valido, non il
    # pavimento, e non deve essere eliminata una seconda volta.
    if maximum >= len(ordered):
        return ordered
    maximum = min(maximum, len(ordered))

    def uniform_sample(items, count):
        if not items:
            return []
        if count >= len(items):
            return items
        if count == 1:
            return [items[len(items) // 2]]
        return [items[round(index * (len(items) - 1) / (count - 1))]
                for index in range(count)]

    return uniform_sample(ordered, maximum)


def template_placeable_points(request, points, sim, objects_dir, margin=0.01):
    """Elimina prima del planning i target non validi per il template richiesto."""
    templates = available_template_names(objects_dir)
    request_lower = str(request).lower()
    matching_templates = [
        name for name in templates
        if re.search(r"\b" + re.escape(name.lower()) + r"\b", request_lower)
    ]
    if len(matching_templates) != 1:
        # Senza un template univoco non conosciamo ingombro e offset. Il
        # controllo resta comunque obbligatorio durante compile_plan().
        return dict(points)

    template = matching_templates[0]
    handle, support_offset = template_handle_and_support_offset(
        sim, template, objects_dir
    )
    object_size = template_object_size(sim, handle)
    valid = {}
    rejected = Counter()
    input_categories = Counter()
    valid_categories = Counter()
    rejected_categories = Counter()
    for point_id, point in points.items():
        category = str(point.get("category", "")).strip().lower() or "<unknown>"
        input_categories[category] += 1
        surface = collision_support_point(sim, point.get("surface_point"))
        if surface is None:
            rejected["collision"] += 1
            rejected_categories[category] += 1
            continue
        position = [
            surface[0], surface[1] + support_offset + float(margin), surface[2]
        ]
        if footprint_support_quality(sim, handle, position, surface) is None:
            rejected["footprint"] += 1
            rejected_categories[category] += 1
            continue
        if not target_has_visible_view(sim, handle, position):
            rejected["view_or_clearance"] += 1
            rejected_categories[category] += 1
            continue
        checked = dict(point)
        checked["surface_point"] = surface
        checked["placement_rank"] = placement_rank(checked, object_size)
        valid[point_id] = checked
        valid_categories[category] += 1
    print(
        "Template target diagnostic: "
        + json.dumps({
            "template": template,
            "input_points": len(points),
            "valid_points": len(valid),
            "rejected": dict(rejected),
            "input_categories": dict(input_categories),
            "valid_categories": dict(valid_categories),
            "rejected_categories": dict(rejected_categories),
        }, ensure_ascii=False),
        flush=True,
    )
    if not valid:
        raise ValueError(
            f"nessuna superficie è utilizzabile con il template '{template}'"
        )
    return valid


def template_placeable_catalog(request, points, sim, objects_dir, margin=0.01):
    """Build a template -> valid placement catalog before asking the LLM."""
    templates = available_template_names(objects_dir)
    request_lower = str(request).lower()
    requested_words = set(re.findall(r"[a-z0-9_-]+", request_lower))
    candidates = [
        template for template in templates
        if any(word in template.lower() for word in requested_words if len(word) > 2)
    ]
    # If the request does not identify a family, retain the complete catalog
    # and let the LLM choose the object type.
    if not candidates:
        return {"__all__": dict(points)}

    catalog = {}
    for template in candidates:
        catalog[template] = template_placeable_points(
            template, points, sim, objects_dir, margin=margin
        )
    return catalog


def union_template_points(template_catalog):
    """Merge per-template points while retaining one canonical point record."""
    merged = {}
    for point_map in template_catalog.values():
        merged.update(point_map)
    if not merged:
        raise ValueError("nessun placement disponibile per i template candidati")
    return merged


def decode_llm_json(content):
    """Accetta JSON puro e l'occasionale blocco Markdown prodotto da un modello."""
    if not isinstance(content, str):
        raise RuntimeError("la risposta del modello non contiene testo JSON")
    text = content.strip()
    if text.startswith("```") and text.endswith("```"):
        text = text.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        preview = text[:500]
        raise RuntimeError(f"Ollama non ha restituito JSON valido: {preview}") from exc


def fallback_plan(request, points, sim, objects_dir, template_catalog=None):
    """Crea un piano minimo valido quando il modello confonde punti e oggetti."""
    templates = available_template_names(objects_dir)
    request_lower = str(request).lower()
    matching = [name for name in templates if name.lower() in request_lower]
    catalog_templates = [
        name for name in (template_catalog or {}) if name != "__all__"
    ]
    if not matching and len(catalog_templates) == 1:
        matching = catalog_templates
    if len(matching) != 1:
        available = ", ".join(templates) or "nessuno"
        raise RuntimeError(
            "Impossibile creare un fallback: indica nella richiesta un solo "
            f"template disponibile. Template: {available}"
        )
    template = matching[0]
    candidate_pool = (
        (template_catalog or {}).get(template, points)
        if template_catalog else points
    )
    first_candidates = select_candidate_points(candidate_pool, 2)
    if not first_candidates:
        raise ValueError("nessun target valido disponibile")
    first = first_candidates[0]
    name = f"{template}_1"
    steps = [{
        "action": "spawn",
        "name": name,
        "template": template,
        "target_point": first["id"],
    }]
    move_words = ("sposta", "muovi", "move", "trasferisci", "porta")
    if any(word in request_lower for word in move_words):
        move_candidates = [
            point for point in select_candidate_points(candidate_pool, len(candidate_pool))
            if point["id"] != first["id"]
        ]
        if not move_candidates:
            raise ValueError("nessun secondo target valido per il supporto richiesto")
        steps.append({
            "action": "move",
            "object": name,
            "target_point": move_candidates[0]["id"],
        })
    remove_words = ("rimuovi", "elimina", "cancella", "remove")
    if any(word in request_lower for word in remove_words):
        steps.append({"action": "remove", "object": name})
    return {
        "description": f"Piano deterministico per {template}.",
        "distinct_destinations": bool(any(word in request_lower for word in move_words)),
        "support_constraints": [
            {"category": "", "room_categories": []}
            for step in steps if step["action"] in {"spawn", "move"}
        ],
        "steps": steps,
    }


def ask_llm_for_plan(request, points, objects_dir, correction="", template_catalog=None):
    """Chiede a Ollama un piano logico usando esclusivamente punti validi."""

    templates = available_template_names(objects_dir)

    # Limitare il catalogo evita prompt enormi e tempi di risposta eccessivi.
    try:
        max_points = int(os.environ.get("OLLAMA_MAX_POINTS", "100"))
    except ValueError:
        max_points = 100
    # La LLM vede solo supporti e superfici semantiche. I surface_* e tutta
    # la geometria restano nella mappa interna usata da compile_plan().
    llm_candidates = semantic_surface_catalog(points)[:max_points]
    available_categories = sorted({str(item["type"]).strip().lower() for item in llm_candidates})
    available_rooms = sorted({
        str(room).strip().lower()
        for item in llm_candidates for room in item.get("room_categories", [])
    })
    expected_constraints = _expected_positioned_action_count(request)
    constraints_schema = {
        "type": "array",
        "description": "One constraint for each spawn/move, in chronological order.",
        "items": {
            "type": "object",
            "properties": {
                "category": {"type": "string", "enum": [""] + available_categories},
                "room_categories": {
                    "type": "array",
                    "items": {"type": "string", "enum": [""] + available_rooms},
                    "maxItems": 3,
                },
            },
            "required": ["category", "room_categories"],
            "additionalProperties": False,
        },
        "maxItems": expected_constraints or MAX_STEPS,
    }
    if expected_constraints is not None:
        constraints_schema["minItems"] = expected_constraints
    schema = {
        "type": "object",
        "properties": {
            "description": {"type": "string", "maxLength": 160},
            "distinct_destinations": {
                "type": "boolean",
                "description": "True when destinations must be different across the plan.",
            },
            "support_constraints": constraints_schema,
            "steps": {
                "type": "array",
                "maxItems": MAX_STEPS,
                # oneOf rende obbligatori i campi dell'azione scelta: con il
                # vecchio schema un remove senza object era formalmente valido.
                "items": {
                    "oneOf": [
                        {
                            "type": "object",
                            "properties": {
                                "action": {"const": "spawn"},
                                "name": {"type": "string", "minLength": 1},
                                "template": {"type": "string", "enum": templates},
                            },
                            "required": ["action", "name", "template"],
                            "additionalProperties": False,
                        },
                        {
                            "type": "object",
                            "properties": {
                                "action": {"const": "move"},
                                "object": {
                                    "type": "string", "minLength": 1,
                                    "description": "Nome esatto di un oggetto creato in uno spawn precedente; mai un placement_id",
                                },
                            },
                            "required": ["action", "object"],
                            "additionalProperties": False,
                        },
                        {
                            "type": "object",
                            "properties": {
                                "action": {"const": "remove"},
                                "object": {
                                    "type": "string", "minLength": 1,
                                    "description": "Nome esatto dell'oggetto da rimuovere, uguale al campo name dello spawn",
                                },
                            },
                            "required": ["action", "object"],
                            "additionalProperties": False,
                        },
                        {
                            "type": "object",
                            "properties": {
                                "action": {"const": "wait"},
                                "seconds": {"type": "number", "minimum": 0, "maximum": 60},
                            },
                            "required": ["action", "seconds"],
                            "additionalProperties": False,
                        },
                    ]
                }
            }
        },
        "required": [
            "description", "distinct_destinations",
            "support_constraints", "steps",
        ],
        "additionalProperties": False
    }
    correction_note = (
        "The previous response was rejected for this reason: "
        + str(correction)[:400] + "\n" if correction else ""
    )
    instructions = (
        "You are a Habitat-Sim task planner. Create a deterministic scene script.\n"
        "Multiple spawns are allowed when the user asks for multiple objects. "
        "Every spawn must have a UNIQUE logical name such as toy_1 or toy_2; "
        "never reuse a name and never use the literal name 'spawn'. For a "
        "singular request, create exactly one spawn.\n"
        "User request: " + request + "\n"
        "Available templates: " + json.dumps(templates, ensure_ascii=False) + "\n"
        "For every spawn, template must be copied EXACTLY from Available templates. "
        "Do not translate, abbreviate, simplify, or invent template names. "
        "For example, if the user says 'tazza', choose the matching available "
        "template such as '025_mug', but output exactly '025_mug'.\n"
        "Do not output placement_id: the compiler assigns geometry-valid placements.\n"
        "Extract semantic support constraints into support_constraints. Use exactly "
        "one item for each spawn or move, in chronological order. Copy category and "
        "room_categories exactly from the semantic support catalog; use empty strings "
        "and an empty room_categories array when the user gives no support constraint. "
        "CRITICAL: do not invent a support, room, or category. For a generic request "
        "such as 'create 10 objects', every support constraint must be {category:'', "
        "room_categories:[]}. Only fill a constraint when the user explicitly names "
        "that support or room.\n"
        "Every spawn needs name and template. Every move needs object. Every remove "
        "needs object.\n"
        "Set distinct_destinations=true only when the user explicitly requires "
        "different destinations. Include move steps only for objects the user asks "
        "to move; do not infer or add extra actions.\n"
        "If the user asks to move ALL/TUTTI spawned objects, include one move for "
        "every spawned logical name. A move must always describe a real relocation; "
        "the compiler will reject the object's current destination. Preserve explicit "
        "minimum waits before the move sequence.\n"
        "ACTION RULE: if the user only says 'metti' or 'posiziona', create exactly "
        "one spawn and do not add move or remove. Add move only when the user "
        "explicitly asks to spostare/muovere/trasferire/portare the object. Add "
        "remove only when the user explicitly asks to rimuoverlo/eliminarlo.\n"
        "The object field is an object name, never a support or surface id.\n"
        "Only remove an object that was spawned earlier in this same script.\n"
        "Do not describe the catalog. Create only the steps required by the user; "
        "do not create one step per point.\n"
        "Keep chronological order: spawn first, then any move/wait steps, and remove last.\n"
        "Return JSON only, conforming exactly to the required schema.\n"
        + correction_note
        + "Semantic support catalog (no coordinates): "
        + json.dumps(llm_candidates, ensure_ascii=False)
    )
    openrouter_config = load_openrouter_config()
    configured_url = os.environ.get(
        "OLLAMA_URL",
        openrouter_config.get("OLLAMA_URL", "http://127.0.0.1:11434/api/chat"),
    )
    configured_url = configured_url.rstrip("/")
    openrouter_key = setting("OPENROUTER_API_KEY", openrouter_config)
    is_openrouter = (
        "openrouter.ai" in configured_url.lower()
        or bool(openrouter_key)
    )
    openai_compatible = configured_url.endswith("/v1") or is_openrouter
    if openai_compatible:
        if is_openrouter:
            endpoint = setting(
                "OPENROUTER_URL", openrouter_config,
                "https://openrouter.ai/api/v1",
            ).rstrip("/") + "/chat/completions"
            model_name = setting(
                "OPENROUTER_MODEL", openrouter_config,
                "qwen/qwen3-30b-a3b-instruct-2507",
            )
        else:
            endpoint = configured_url + "/chat/completions"
            model_name = os.environ.get("OLLAMA_MODEL", "llama3.1:8b")
        payload = {
            "model": model_name,
            "messages": [{"role": "user", "content": instructions}],
            "stream": False,
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "scene_script_plan",
                    "strict": True,
                    "schema": schema,
                },
            },
            "temperature": 0,
            "max_tokens": max(
                1024,
                min(8192, 384 + 180 * (_requested_object_count(request) or 4)),
            ),
        }
        if is_openrouter:
            # Evita che OpenRouter scelga un provider che ignora lo schema JSON.
            payload["provider"] = {"require_parameters": True}
        else:
            # Parametro supportato da alcuni endpoint OpenAI-compatible locali,
            # ma non necessario e non uniformemente accettato da OpenRouter.
            payload["reasoning_effort"] = "none"
    else:
        endpoint = configured_url
        payload = {
        "model": os.environ.get("OLLAMA_MODEL", "qwen3:8b"),
        "messages": [{"role": "user", "content": instructions}],
        "stream": False,
        "think": False,
        "format": schema,
        "options": {"temperature": 0, "num_ctx": 8192}
        }
    headers = {"Content-Type": "application/json"}
    if is_openrouter:
        if not openrouter_key:
            raise RuntimeError(
                "OPENROUTER_API_KEY non impostata. Configurala prima di usare OpenRouter."
            )
        headers["Authorization"] = f"Bearer {openrouter_key}"
        site_url = setting("OPENROUTER_SITE_URL", openrouter_config)
        site_title = setting("OPENROUTER_SITE_NAME", openrouter_config, "lost3dsg")
        if site_url:
            headers["HTTP-Referer"] = site_url
        if site_title:
            headers["X-OpenRouter-Title"] = site_title
    request_obj = Request(
        endpoint,
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    service_name = "OpenRouter" if is_openrouter else "Ollama"
    result = request_llm_json(
        request_obj, service_name, endpoint, retry_transient=is_openrouter
    )
    if openai_compatible:
        content = result.get("choices", [{}])[0].get("message", {}).get("content", "")
    else:
        content = result.get("message", {}).get("content", "")
    return decode_llm_json(content)


def probe_rendered_semantics(sim):
    """Diagnostica il legame fra semantic sensor e categorie della scena.

    Non genera ancora pose: serve a verificare, nella build Habitat attiva,
    che gli ID del render semantico possano essere ricondotti alle categorie
    annotate prima di sostituire il generatore basato su GLB.
    """
    agent = sim.initialize_agent(0)
    point = sim.pathfinder.get_random_navigable_point()
    state = agent.get_state()
    state.position = point
    agent.set_state(state, reset_sensors=True, infer_sensor_states=True)
    observations = sim.get_sensor_observations()
    semantic = observations.get("semantic_sensor")
    depth = observations.get("depth_sensor")
    if semantic is None or depth is None:
        raise RuntimeError("depth_sensor o semantic_sensor non disponibili")
    objects = getattr(getattr(sim, "semantic_scene", None), "objects", []) or []
    categories = {}
    for obj in objects:
        if obj is None:
            continue
        try:
            name = str(obj.category.name())
        except (AttributeError, TypeError):
            name = ""
        categories[str(getattr(obj, "id", ""))] = name
        semantic_id = getattr(obj, "semantic_id", None)
        if semantic_id is not None:
            categories[str(semantic_id)] = name
    ids, counts = np.unique(semantic, return_counts=True)
    observed = [
        {"id": int(item_id), "pixels": int(count), "category": categories.get(str(int(item_id)), "")}
        for item_id, count in zip(ids[:100], counts[:100]) if int(item_id) != 0
    ]
    print(json.dumps({
        "depth_shape": list(depth.shape),
        "semantic_shape": list(semantic.shape),
        "semantic_dtype": str(semantic.dtype),
        "agent_position": values(point),
        "observed_ids": observed,
    }, indent=2, ensure_ascii=False))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("plan", nargs="?", help="JSON con target_point")
    parser.add_argument("--request", help="richiesta da affidare alla LLM")
    parser.add_argument("--list-points", action="store_true")
    parser.add_argument("--probe-rendered-semantics", action="store_true")
    parser.add_argument("--scene", default=os.environ.get("HABITAT_SCENE", DEFAULT_SCENE))
    parser.add_argument(
        "--scene-dataset",
        default=os.environ.get("HABITAT_SCENE_DATASET", DEFAULT_SCENE_DATASET),
    )
    parser.add_argument("--navmesh", default=os.environ.get("HABITAT_NAVMESH", DEFAULT_NAVMESH))
    parser.add_argument("--objects-dir", default=os.environ.get("HABITAT_EXAMPLE_OBJECTS_DIR", DEFAULT_OBJECTS))
    parser.add_argument("--step", type=float, default=0.25)
    parser.add_argument(
        "--object-scale", type=float, default=None,
        help="fattore globale di scala degli oggetti (default: 1.0)",
    )
    parser.add_argument("--output", default=None, help="file JSON finale compilato")
    args = parser.parse_args()

    if args.object_scale is not None:
        if not math.isfinite(args.object_scale) or args.object_scale <= 0:
            parser.error("--object-scale deve essere un numero positivo")
        os.environ["HABITAT_OBJECT_SCALE"] = str(args.object_scale)

    sim_cfg = habitat_sim.SimulatorConfiguration()
    sim_cfg.scene_id = args.scene
    sim_cfg.scene_dataset_config_file = args.scene_dataset
    sim_cfg.enable_physics = True
    sim_cfg.load_semantic_mesh = True
    sensor_specs = []
    for uuid, sensor_type in (
        ("color_sensor", habitat_sim.SensorType.COLOR),
        ("depth_sensor", habitat_sim.SensorType.DEPTH),
        ("semantic_sensor", habitat_sim.SensorType.SEMANTIC),
    ):
        spec = habitat_sim.CameraSensorSpec()
        spec.uuid = uuid
        spec.sensor_type = sensor_type
        spec.sensor_subtype = habitat_sim.SensorSubType.PINHOLE
        spec.resolution = [240, 320]
        spec.position = [0.0, 1.5, 0.0]
        spec.hfov = 90.0
        sensor_specs.append(spec)
    agent_cfg = habitat_sim.agent.AgentConfiguration()
    agent_cfg.sensor_specifications = sensor_specs
    sim = habitat_sim.Simulator(habitat_sim.Configuration(
        sim_cfg, [agent_cfg]
    ))
    try:
        load_navmesh(sim, args.scene, args.navmesh)
        if args.probe_rendered_semantics:
            probe_rendered_semantics(sim)
            return 0
        points = generate_points(sim, args.step, scene=args.scene)
        if args.list_points:
            print(json.dumps(points, indent=2))
            return 0
        if args.request:
            template_catalog = template_placeable_catalog(
                args.request, points, sim, args.objects_dir
            )
            planning_points = union_template_points(template_catalog)
            # Lo schema JSON filtra la forma, ma non puo' verificare riferimenti
            # e ordine temporale. In caso di errore chiediamo correzioni mirate,
            # anziche' salvare uno script apparentemente valido ma rotto.
            correction = ""
            compiled = None
            max_attempts = 3
            for attempt in range(max_attempts):
                plan = ask_llm_for_plan(
                    args.request, planning_points, args.objects_dir,
                    correction=correction, template_catalog=template_catalog,
                )
                if normalize_empty_support_constraints(plan):
                    print(
                        "LLM diagnostic: riallineati automaticamente i vincoli "
                        "di supporto vuoti con gli step spawn/move.",
                        flush=True,
                    )
                print(
                    "LLM diagnostic attempt "
                    f"{attempt + 1}/{max_attempts}: "
                    + json.dumps(plan, ensure_ascii=False),
                    flush=True,
                )
                try:
                    validate_plan_intent(plan, args.request)
                    assign_valid_placements(
                        plan, planning_points, sim, args.objects_dir
                    )
                    compiled = compile_plan(
                        plan, planning_points, sim, args.objects_dir,
                        request_text=args.request, template_catalog=template_catalog,
                    )
                    break
                except ValueError as exc:
                    correction = str(exc)
                    print(
                        "LLM diagnostic rejected: " + correction,
                        flush=True,
                    )
                    if attempt == max_attempts - 1:
                        # I modelli piccoli talvolta inseriscono un target_point
                        # nel campo object. In quel caso non salviamo quel piano:
                        # generiamo una variante minimale, interamente verificata.
                        try:
                            plan = fallback_plan(
                                args.request, planning_points, sim, args.objects_dir,
                                template_catalog=template_catalog,
                            )
                        except (RuntimeError, ValueError) as fallback_exc:
                            raise ValueError(
                                "piano LLM rifiutato dopo "
                                f"{max_attempts} tentativi: {correction}; "
                                f"fallback fallito: {fallback_exc}"
                            ) from fallback_exc
                        assign_valid_placements(
                            plan, planning_points, sim, args.objects_dir
                        )
                        validate_plan_intent(plan, args.request)
                        compiled = compile_plan(
                            plan, planning_points, sim, args.objects_dir,
                            request_text=args.request, template_catalog=template_catalog,
                        )
                        print(
                            "Piano LLM non valido dopo "
                            f"{max_attempts} tentativi ({correction}); "
                            "usato fallback deterministico."
                        )
                        break
            assert compiled is not None
        elif args.plan:
            plan = json.loads(Path(args.plan).read_text(encoding="utf-8"))
            compiled = compile_plan(
                plan, points, sim, args.objects_dir, request_text=""
            )
        else:
            parser.error("indica un piano JSON, --request oppure usa --list-points")
    finally:
        sim.close()

    output = args.output
    if args.request and output is None:
        output = str(Path("scripts") / "compiled_script.json")
    if output:
        Path(output).parent.mkdir(parents=True, exist_ok=True)
        Path(output).write_text(json.dumps(compiled, indent=2), encoding="utf-8")
        print(f"Script completo creato: {output}")

    # scene_script compila soltanto. L'esecuzione viene fatta sempre da
    # run_habitat_script.py/script_runner.py.
    print(json.dumps(compiled, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, RuntimeError, ValueError, json.JSONDecodeError) as exc:
        # Gli errori di pianificazione sono input/servizio non disponibile,
        # non bug Python: mostrarli in modo leggibile evita traceback inutili.
        print(f"Errore: {exc}")
        raise SystemExit(1) from None
