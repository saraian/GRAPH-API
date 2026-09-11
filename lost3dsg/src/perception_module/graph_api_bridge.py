import atexit
import base64
import binascii
import collections
import hashlib
import json
import math
import os
import re
import sqlite3
import struct
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

import cv2
import numpy as np
import rclpy
import uvicorn
from config import CFG  # GA-341: merge-request defaults from the SAME config block the service reads
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image

# Physical TIAGo camera drivers publish sensor data as BEST_EFFORT. A RELIABLE
# subscription does not match that publisher, leaving the bridge apparently live
# while no image ever arrives. Keep only the newest frame for the browser.
_QOS_CAMERA = QoSProfile(
    depth=1,
    reliability=ReliabilityPolicy.BEST_EFFORT,
    durability=DurabilityPolicy.VOLATILE,
    history=HistoryPolicy.KEEP_LAST,
)

from lost3dsg.srv import (
    AddObject,
    DeleteObjects,
    MergeObjects,
    QueryObjects,
    RemoveObject,
    UpdateObject,
)

app = FastAPI(title="Graph API")


class DispatchedTimeout(RuntimeError):
    """GA-183: the ROS request was sent and our wait expired. The work may still land
    (89 merges did, reported as 500, in run 20260901_055513), so it is not a failure."""


@app.exception_handler(DispatchedTimeout)
def _dispatched_timeout(request, exc):
    # 202: accepted, outcome unknown. The client (object_manager_6) treats 2xx as a reply
    # with no counts, logs the message, and re-reads the world model on the next cycle.
    return JSONResponse(status_code=202, content={"success": False, "pending": True, "message": str(exc)})

_MODULE_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = Path(
    _MODULE_DIR.parent.parent
    if "/install/" not in str(_MODULE_DIR)
    else str(_MODULE_DIR).split("/install/", 1)[0]
)
def _active_output_dir():
    """Return the output directory used by the active ROS installation.

    The package can be launched either from ``src`` or from ``install``.
    Those two modes previously made the bridge read different copies of the
    JSON files.  Prefer an explicit directory, otherwise use the candidate
    containing the most recently written map/room data.
    """
    configured = os.environ.get("GRAPH_API_OUTPUT_DIR") or os.environ.get("LOST3DSG_OUTPUT_DIR")
    if configured:
        return Path(configured).expanduser().resolve()

    candidates = [
        Path("/ws/output"),
        Path("/out"),
        Path("/tmp/graphapi_live"),
        _PROJECT_ROOT / "output",
        _MODULE_DIR.parents[2] / "output" if len(_MODULE_DIR.parents) > 2 else _PROJECT_ROOT / "output",
        _MODULE_DIR.parents[1] / "output" if len(_MODULE_DIR.parents) > 1 else _PROJECT_ROOT / "output",
        Path("/ws/install/lost3dsg/output"),
        Path("/tmp"),
    ]
    unique = []
    for candidate in candidates:
        cr = candidate.resolve()
        if cr not in unique:
            unique.append(cr)

    existing = [
        candidate for candidate in unique
        if any((candidate / name).exists() for name in ("room.json", "persistent_perception.json"))
    ]
    if existing:
        return max(
            existing,
            key=lambda candidate: max(
                ((candidate / name).stat().st_mtime for name in ("room.json", "persistent_perception.json")
                 if (candidate / name).exists()),
                default=0.0,
            ),
        )
    return unique[0]
def _find_viewer_dir():
    candidates = [
        Path("/graph_api/lost3dsg/src/perception_module/viewer"),
        _PROJECT_ROOT / "src" / "perception_module" / "viewer",
        _PROJECT_ROOT / "lost3dsg" / "src" / "perception_module" / "viewer",
        _MODULE_DIR / "viewer",
    ]
    for c in candidates:
        if (c / "viewer.html").exists():
            return c
    return _MODULE_DIR / "viewer"


VIEWER_DIR = _find_viewer_dir()

_node = None

app.mount("/viewer", StaticFiles(directory=str(VIEWER_DIR)), name="viewer")


def _viewer_path():
    """Use the robot dashboard only for a physical run."""
    try:
        from config import CFG as runtime_cfg
        physical = not bool(runtime_cfg.get("simulation", True))
    except (ImportError, AttributeError, TypeError):
        physical = os.environ.get("PAL_ROBOT_CONNECTED", "").lower() in (
            "1", "true", "yes", "on")
    tiago = VIEWER_DIR / "tiago_viewer.html"
    return tiago if physical and tiago.exists() else VIEWER_DIR / "viewer.html"

@app.get("/", include_in_schema=False)
def viewer(request: Request = None):
    path = _viewer_path()
    try:
        st = path.stat()
        etag = f'"{int(st.st_mtime)}-{st.st_size}"'
    except OSError:
        return FileResponse(str(path))
    # ~152 KB re-sent on every reload. must-revalidate keeps edits picked up
    # immediately during development while a reload costs a 304, not the file.
    if request is not None and request.headers.get("if-none-match") == etag:
        return Response(status_code=304, headers={"ETag": etag})
    return FileResponse(str(path), headers={"ETag": etag,
                                            "Cache-Control": "no-cache, must-revalidate"})


@app.get("/favicon.ico", include_in_schema=False)
def favicon():
    return Response(status_code=204)


@app.get("/persistent_perception")
def persistent_perception():
    path = _active_output_dir() / "persistent_perception.json"
    if not path.exists():
        return []
    try:
        objects = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return []
    # GA-102. Each object carries its admission grade (admit / hold / decline / no_grounds),
    # so the feed host's grade toggles have something to count and filter. None when no
    # admission row links to the object: absent, not a default.
    if isinstance(objects, list):
        by_oid = _admissions_by_oid()
        for o in objects:
            if isinstance(o, dict):
                dec = by_oid.get(str(o.get("object_id")))
                o["grade"] = (((dec or {}).get("annotation") or {}).get("verdict") or {}).get("grade")
    return objects


@app.get("/rooms")
def rooms():
    # Was also named `persistent_perception`, shadowing the /persistent_perception
    # handler above. Both routes worked (FastAPI binds the function object at
    # decoration time), but the module-level name pointed at this one only.
    path = _active_output_dir() / "room.json"
    if not path.exists():
        return []
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return []


# Frames that could not be decoded. Counted rather than swallowed: see _jpeg_from_msg.
_FRAME_DECODE_FAILURES = {"n": 0, "last": None}


class BridgeNode(Node):
    def __init__(self):
        super().__init__('graph_api_bridge')
        self.cli = {
            'add': self.create_client(AddObject, '/graph/add_object'),
            'remove': self.create_client(RemoveObject, '/graph/remove_object'),
            'update': self.create_client(UpdateObject, '/graph/update_object'),
            'merge': self.create_client(MergeObjects, '/graph/merge_objects'),
            'delete_objects': self.create_client(DeleteObjects, '/graph/delete_objects'),
            'query_objects': self.create_client(QueryObjects, '/graph/query_objects'),
        }
        self.ros_bev = None
        if (_BRIDGE_CFG.get('bev') or {}).get('source') == 'ros':
            from ros_bev import RosBEV
            self.ros_bev = RosBEV(self, _BRIDGE_CFG)
        self.latest_jpeg = None       # /image_with_bb: one frame per perception cycle
        self.last_frame_time = 0.0
        # Arrival times of the last 24 annotated frames, for the MEASURED frame period. The
        # Metrics tab's "total perception cycle" is compute time; frames reach the page less
        # often than that because the agent walks between cycles (6.1 s median vs 3.1 s on
        # 20260906_223701). Two different numbers, so two rows -- and this one is measured.
        self.frame_times = collections.deque(maxlen=24)
        self.raw_jpeg = None          # /camera/rgb: every sim frame, kept separate so a
        self.last_raw_time = 0.0      # stale annotated frame can never masquerade as live
        # the raw feed publishes /camera/rgb (habitat_feed_node.py:126). The old
        # '/camera/rgb/image_raw' had no publisher at all, so this fallback never fired.
        # GA-271. Detected walls, from wall_detector's per-frame depth fit -- NOT from the
        # accumulated map, so this works with no map at all. Each segment carries the height
        # band it was observed over, which is what distinguishes a measured surface from
        # "observed free space stopped here".
        #
        # The bridge is the hub because both views need the same segments: the Habitat window
        # runs on the host and cannot subscribe to ROS, and the dashboard is served from here.
        # Two subscribers would be two versions of the truth.
        from std_msgs.msg import String as _WallStr
        self.latest_walls, self.walls_stamp = [], 0.0
        self.create_subscription(_WallStr, '/detected_wall_segments', self._on_walls, 10)
        self.create_subscription(Image, '/camera/rgb', self._on_raw_image, _QOS_CAMERA)
        self.create_subscription(Image, '/image_with_bb', self._on_annotated_image, 10)

        # TF, for the live overlay's camera pose. Optional on purpose: if tf2_ros is not
        # importable the attribute stays None, _pose_from_tf returns None, and the overlay
        # falls back to the newest recorded pose instead of the node failing to construct.
        # /health reports which source answered, so the fallback is never mistaken for TF.
        self.tf_buffer = None
        try:
            from tf2_ros import TransformListener
            from tf2_ros.buffer import Buffer
            self.tf_buffer = Buffer()
            self.tf_listener = TransformListener(self.tf_buffer, self)
        except ImportError as exc:
            self.get_logger().warn(f"tf2_ros unavailable, overlay will use recorded poses: {exc}")

    def _convert_to_jpeg(self, msg: Image) -> bytes:
        try:
            if not msg.data:
                return None
            img_np, encoding = ros_image_to_array(msg)
            if encoding == 'rgb8':
                img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
            elif encoding == 'rgba8':
                img_np = cv2.cvtColor(img_np, cv2.COLOR_RGBA2BGR)
            elif encoding == 'bgra8':
                img_np = cv2.cvtColor(img_np, cv2.COLOR_BGRA2BGR)
            _, jpeg = cv2.imencode('.jpg', img_np, [cv2.IMWRITE_JPEG_QUALITY, 80])
            return jpeg.tobytes()
        except (ValueError, TypeError, AttributeError) as exc:
            # A malformed frame (wrong buffer length for h*w*3, unexpected encoding) is a
            # real per-frame condition and must not kill the subscription. It is COUNTED,
            # so a feed quietly dropping every frame is visible in /health instead of
            # looking like a feed that simply has nothing to send.
            _FRAME_DECODE_FAILURES["n"] += 1
            _FRAME_DECODE_FAILURES["last"] = f"{type(exc).__name__}: {exc}"
            return None

    def _on_annotated_image(self, msg: Image):
        jpeg = self._convert_to_jpeg(msg)
        if jpeg:
            self.latest_jpeg = jpeg
            self.last_frame_time = time.time()
            self.frame_times.append(self.last_frame_time)

    def _on_walls(self, msg):
        """Keep the newest wall segments. A decode failure is counted, not swallowed."""
        try:
            self.latest_walls = json.loads(msg.data)
            self.walls_stamp = time.time()
        except (ValueError, TypeError) as exc:
            self.get_logger().warn(f"wall segments undecodable: {exc}")

    def _on_raw_image(self, msg: Image):
        jpeg = self._convert_to_jpeg(msg)
        if jpeg:
            self.raw_jpeg = jpeg
            self.last_raw_time = time.time()

    def call(self, key, req):
        client = self.cli[key]

        if not client.wait_for_service(timeout_sec=2.0):
            raise RuntimeError(f"service '{key}' unavailable")

        future = client.call_async(req)

        deadline = time.time() + SERVICE_CALL_TIMEOUT_SEC
        while time.time() < deadline:
            if future.done():
                break
            time.sleep(0.05)

        if not future.done():
            # The request WAS dispatched by call_async above; only our wait expired. Say so,
            # because the caller's next move differs: a retry here re-runs work that is still
            # in flight, and for `merge` that means merging an object twice (GA-183).
            raise DispatchedTimeout(
                f"Timeout dopo {SERVICE_CALL_TIMEOUT_SEC:.0f}s in attesa del servizio '{key}'; "
                f"la richiesta E' STATA INVIATA e puo' ancora completarsi -- non ritentare "
                f"senza verificare lo stato (BRIDGE_SERVICE_TIMEOUT per allungare l'attesa)")

        result = future.result()
        if result is None:
            exc = future.exception()
            if exc is not None:
                raise RuntimeError(f"Errore dal servizio '{key}': {exc}")
            raise RuntimeError(f"no response from service '{key}'")

        return result


def get_node():
    return _node


def require_node():
    node = get_node()
    if node is None:
        raise HTTPException(status_code=503, detail="ROS bridge non pronto")
    return node


def _set_description_embedding(req, body):
    """Fill `description_embedding` AND the flag that says whether it means anything.

    GA-184. JSON can express absence -- `null`, or the key simply missing -- and the ROS
    request cannot: `float32[]` has no null. So the distinction has to be carried in a
    separate boolean, and THIS is the only place that knows both sides. `body.get(k, [])`
    on its own silently turned "absent" into "empty" and the information was gone from
    here on.

    An explicit empty LIST in the JSON is still absence: nothing downstream can use a
    zero-length embedding, and a caller that sends one is saying it has none.
    """
    raw = body.get("description_embedding")
    values = [] if raw is None else [float(x) for x in raw]
    req.description_embedding = values
    req.has_description_embedding = bool(values)
    return req.has_description_embedding


def _set_orientation(req, body: dict):
    """GA-312. Carry the oriented box across the service boundary when the caller sent one.
    `has_orientation` is set from the presence of ALL THREE keys, never from a default, so an
    axis-aligned caller stays axis-aligned and a yaw of 0.0 is a measurement, not an absence."""
    keys = ("yaw", "oriented_center", "oriented_extents")
    if all(body.get(k) is not None for k in keys):
        req.has_orientation = True
        req.yaw = float(body["yaw"])
        req.oriented_center = [float(v) for v in body["oriented_center"]][:3]
        req.oriented_extents = [float(v) for v in body["oriented_extents"]][:3]
    else:
        req.has_orientation = False


@app.post("/objects")
def add_object(body: dict):
    req = AddObject.Request()
    req.label = body["label"]
    req.description = body.get("description", "")
    req.color = body.get("color", "")
    req.material = body.get("material", "")
    req.room_id = body.get("room_id") or ""
    req.x_min = float(body.get("x_min", 0.0))
    req.x_max = float(body.get("x_max", 0.0))
    req.y_min = float(body.get("y_min", 0.0))
    req.y_max = float(body.get("y_max", 0.0))
    req.z_min = float(body.get("z_min", 0.0))
    req.z_max = float(body.get("z_max", 0.0))
    _set_orientation(req, body)
    _set_description_embedding(req, body)
    res = require_node().call('add', req)
    if not res.success:
        raise HTTPException(status_code=400, detail=res.message)

    return {
        "success": res.success,
        "object_id": res.object_id,
        "message": res.message,
    }


@app.delete("/objects/{object_id}")
def remove_object(
    object_id: str,
    remove_from_uncertain_only: bool = False,
    pov_x_min: float = 0.0,
    pov_x_max: float = 0.0,
    pov_y_min: float = 0.0,
    pov_y_max: float = 0.0,
    pov_z_min: float = 0.0,
    pov_z_max: float = 0.0,
):
    req = RemoveObject.Request()
    req.remove_from_uncertain_only = remove_from_uncertain_only
    req.pov_x_min = pov_x_min
    req.pov_x_max = pov_x_max
    req.pov_y_min = pov_y_min
    req.pov_y_max = pov_y_max
    req.pov_z_min = pov_z_min
    req.pov_z_max = pov_z_max
    req.object_id = object_id

    res = require_node().call('remove', req)
    if not res.success:
        raise HTTPException(status_code=404, detail=res.message)

    return {
        "success": res.success,
        "message": res.message,
    }


@app.patch("/objects/{object_id}")
def update_object(object_id: str, body: dict):
    req = UpdateObject.Request()
    req.object_id = object_id
    req.description = body.get("description", "")
    req.color = body.get("color", "")
    req.material = body.get("material", "")

    req.update_bbox = any(
        k in body for k in ["x_min", "x_max", "y_min", "y_max", "z_min", "z_max"]
    )

    if req.update_bbox:
        req.x_min = float(body.get("x_min", 0.0))
        req.x_max = float(body.get("x_max", 0.0))
        req.y_min = float(body.get("y_min", 0.0))
        req.y_max = float(body.get("y_max", 0.0))
        req.z_min = float(body.get("z_min", 0.0))
        req.z_max = float(body.get("z_max", 0.0))
        _set_orientation(req, body)

    _set_description_embedding(req, body)

    res = require_node().call('update', req)
    if not res.success:
        raise HTTPException(status_code=400, detail=res.message)

    return {
        "success": res.success,
        "message": res.message,
        "object_id": res.object_id,
        "distance": res.distance,
        "iou": res.iou,
        "replaced": res.replaced,
    }


# GA-341. The body defaults used to be the literals 0.8 / 0.75 -- and 0.75 sits BELOW
# association.sim_threshold (0.85), the floor object_services asserts at load and now refuses
# per request, so an empty POST /merge would be a guaranteed 400 carrying a dead number. These
# are the same keys, with the same derivation, as object_services.MERGE_MAX_DISTANCE and
# MERGE_MIN_SIMILARITY (that module owns the thresholds; it is not imported here because it
# pulls ROS, the world model and the sentence encoder into the bridge process).
_SIM_THRESHOLD = float(CFG["association"]["sim_threshold"])
MERGE_MAX_DISTANCE_DEFAULT = float(CFG["association"].get("merge_max_distance_m", 0.8))
MERGE_MIN_SIMILARITY_DEFAULT = float(CFG["association"].get(
    "merge_min_similarity", _SIM_THRESHOLD + (1.0 - _SIM_THRESHOLD) / 2.0))


@app.post("/merge")
def merge_objects(body: dict = None):
    body = body or {}
    req = MergeObjects.Request()
    req.max_distance = float(body.get("max_distance", MERGE_MAX_DISTANCE_DEFAULT))
    req.min_similarity = float(body.get("min_similarity", MERGE_MIN_SIMILARITY_DEFAULT))
    req.dry_run = bool(body.get("dry_run", False))

    res = require_node().call('merge', req)
    if not res.success:
        raise HTTPException(status_code=400, detail=res.message)

    merge_log = json.loads(res.merge_log_json) if res.merge_log_json else []

    return {
        "success": res.success,
        "message": res.message,
        "merged_count": res.merged_count,
        "merge_log": merge_log,
    }


@app.post("/delete_objects")
def delete_objects(body: dict):
    req = DeleteObjects.Request()
    req.pov_volume_flat = [float(x) for x in body.get("pov_volume_flat", [])]
    req.current_labels = body.get("current_labels", [])
    req.check_uncertain = bool(body.get("check_uncertain", False))

    res = require_node().call('delete_objects', req)
    if not res.success:
        raise HTTPException(status_code=400, detail=res.message)

    deleted_labels = json.loads(res.deleted_labels_json) if res.deleted_labels_json else []

    return {
        "success": res.success,
        "message": res.message,
        "deleted_count": res.deleted_count,
        "uncertain_removed_count": res.uncertain_removed_count,
        "deleted_labels": deleted_labels,
    }

@app.post("/query_objects")
def query_objects(body: dict = None):
    body = body or {}
    req = QueryObjects.Request()
    req.uncertain_only = bool(body.get("uncertain_only", False))
    req.room_id = body.get("room_id", "")
    req.label_filter = body.get("label_filter", "")
    area_filter = body.get("area_filter")
    if area_filter is not None:
        req.area_filter = [float(x) for x in area_filter]
    else:
        req.area_filter = []

    node = get_node()
    if node is None:
        return {"success": False, "ready": False, "object_ids": [], "objects": []}

    try:
        res = node.call('query_objects', req)
    except RuntimeError:
        return {"success": False, "ready": False, "object_ids": [], "objects": []}

    if not res.success:
        return {"success": False, "ready": True, "object_ids": [], "objects": []}

    parsed = json.loads(res.serialized_json) if res.serialized_json else []

    return {
        "success": res.success,
        "ready": True,
        "object_ids": list(res.object_ids),
        "objects": parsed,
    }


def _graph_version(out):
    """Stat-only fingerprint of everything /graph_data reads. Cheap enough to
    compute before parsing, so an unchanged world model costs four stat calls
    instead of four JSON parses plus a full client-side graph re-sync."""
    parts = []
    for name in ("persistent_perception.json", "room.json",
                 "hook_decisions.jsonl", "uncertain_objects.txt"):
        for base in (out, out.parent):
            path = base / name
            try:
                st = path.stat()
            except OSError:
                continue
            parts.append(f"{name}:{int(st.st_mtime_ns)}:{st.st_size}")
            break
        else:
            parts.append(f"{name}:-")
    return hashlib.sha1("|".join(parts).encode()).hexdigest()[:16]


def _sources_mtime(out: Path):
    """Newest mtime over the files /graph_data is built from, or None when none exists."""
    mts = []
    for name in ("persistent_perception.json", "room.json", "hook_decisions.jsonl", "uncertain_objects.txt"):
        for base in (out, out.parent):
            try:
                mts.append((base / name).stat().st_mtime)
                break
            except OSError:
                continue
    return max(mts) if mts else None


@app.get("/graph_data")
def graph_data(request: Request = None):
    """Format persistent_perception.json + room.json into Cytoscape elements.

    Nodes: rooms + persistent objects. Edges: ``isLocatedIn`` (object -> its
    room) and ``supports`` (object B rests on top of object A, Y-up).
    """
    out = _active_output_dir()
    version = _graph_version(out)
    etag = f'"{version}"'
    if request is not None and request.headers.get("if-none-match") == etag:
        return Response(status_code=304, headers={"ETag": etag})

    def _load(name):
        for path in [out / name, out.parent / name]:
            if not path.exists():
                continue
            try:
                data = json.loads(path.read_text())
            except (OSError, json.JSONDecodeError):
                continue
            if isinstance(data, list) and data:
                return data
            # room_manager writes room.json as a dict, not a list:
            # {current_room_id, building: {rooms: [...]}, rooms: [...]}
            if isinstance(data, dict):
                rooms = data.get("rooms") or (data.get("building") or {}).get("rooms") or []
                if rooms:
                    return rooms
        return []

    objects = _load("persistent_perception.json")
    rooms = _load("room.json")

    # Load ontological hook admission decisions if logged
    decisions = {}
    for rec in _decision_records():
        # Only admission records describe a decision. `link` records share the `object`
        # field (keyed by object_id) and were landing in this map, so a label miss could
        # return a link record as the "decision".
        if rec.get("kind") not in (None, "admission"):
            continue
        obj_name = rec.get("object")
        if obj_name:
            decisions[obj_name] = rec
    # GA-40 (decision-keyed-by-label). The label key above is a per-frame ordinal
    # ('picture frame#2'), so two objects with one label shared one record. The object-id
    # key wins; the label stays as the fallback for logs written before `link` rows.
    decisions.update(_admissions_by_oid())

    # room_manager seeds every new room with these until the VLM names it
    def _room_label(room, rid):
        sem = str(room.get("semantic_label") or "").strip()
        if sem.lower().replace(" ", "_") in ("", "unknownroom", "unknown_room", "unknown", "none"):
            return str(rid).replace("_", " ").title()
        return sem

    def _aligner_iri(entity, evidence):
        """The IRI inside the aligner's own evidence, or None.

        Only returned when the IRI's fragment matches the entity, so a stray URL in an
        evidence string cannot be presented as this object's class.
        """
        if not entity or not evidence:
            return None
        m = re.search(r"\((https?://[^\s()]+)\)", evidence)
        if not m:
            return None
        iri = m.group(1)
        fragment = iri.rsplit("#", 1)[-1].rsplit("/", 1)[-1]
        return iri if fragment.lower() == str(entity).lower() else None

    def _alignment_of(decision):
        """The REAL alignment the ontological filter recorded, or None.

        `annotation.entity` is the class the aligner actually settled on and
        `annotation.alignment` carries its score, status and the evidence string
        (including why it abstained). Nothing here is inferred from the label:
        an object the aligner did not align has no class, and says so.
        """
        ann = (decision or {}).get("annotation") or {}
        al = ann.get("alignment") or {}
        entity = ann.get("entity")
        if not entity and not al:
            return None
        return {
            "entity": entity,
            # The IRI the ALIGNER recorded, never one built from the label. It exists
            # only inside its own evidence string ("... -> Bed (<the aligner's own IRI>)"),
            # so it is extracted from there
            # and then checked: the fragment must match the entity the aligner settled
            # on. Constructing `home#<label>` instead would mint a plausible IRI for
            # every object including ones the aligner declined -- the fake ontology
            # class bug, wearing a URL. No evidence, or a mismatch, means no IRI.
            "iri": _aligner_iri(entity, al.get("evidence") or ""),
            "score": al.get("score"),
            "status": al.get("status") or ("aligned" if entity else "unaligned"),
            "evidence": al.get("evidence") or "",
        }

    def _confidence_of(o, decision):
        """The aligner's recorded score, else whatever the object itself carried.

        Never a default. If neither exists this returns None and the field stays
        absent, which is what the viewer renders as an em dash.
        """
        al = _alignment_of(decision) or {}
        score = al.get("score")
        if isinstance(score, (int, float)):
            return score
        own = o.get("confidence")
        return own if isinstance(own, (int, float)) else None

    def _nid(label):
        # Cytoscape selectors choke on '#' etc. in ids ("sofa#1") — sanitize.
        return "n_" + "".join(c if c.isalnum() else "_" for c in str(label))

    nodes, edges = [], []
    known_room_ids = set()
    for r in rooms:
        rid = r.get("room_id", "room")
        nid = _nid(rid)
        known_room_ids.add(nid)
        sem = str(r.get("semantic_label") or "").strip()
        nodes.append({
            "id": nid,
            "label": _room_label(r, rid),
            "type": "room",
            "room": rid,
            "semantic_label": sem,
            "detected": bool(_room_label(r, rid) == sem and sem),
            "description": r.get("description") or "",
            "objects": r.get("objects") or [],
            "vlm_status": r.get("vlm_status") or {},
        })

    # Node id: object_id when the writer provides it, label for old-format files
    def _oid(o):
        return _nid(o.get("object_id") or o.get("label", "object"))

    for o in objects:
        label = o.get("label", "object")
        bbox = o.get("bbox") or {}
        pos = o.get("position")
        if not pos and bbox and "x_min" in bbox and "x_max" in bbox:
            pos = [
                (bbox["x_min"] + bbox["x_max"]) / 2.0,
                (bbox.get("y_min", 0.0) + bbox.get("y_max", 0.0)) / 2.0,
                (bbox.get("z_min", 0.0) + bbox.get("z_max", 0.0)) / 2.0,
            ]
        decision = decisions.get(o.get("object_id")) or decisions.get(label)
        # Key the crop on the object identity when there is one. The label joined
        # 9 of 11 objects on the 26 Aug run; /crop resolves an object_id back to
        # its label through the `link` records in hook_decisions.jsonl.
        crop_target = o.get("object_id") or str(label).replace("#", "_").replace(" ", "_")
        nodes.append({
            "id": _oid(o),
            "label": label,
            "type": "object",
            "room": o.get("room_id") or "",
            # GA-361: when this object entered the store and when it was last seen (epoch s),
            # so a replay can show the graph AS IT WAS at a frame. Absent stays absent.
            "created_at": o.get("creation_time"),
            "last_seen": o.get("last_perception_timestamp"),
            # No default: a missing confidence rendered as 1.0 showed every object
            # at a confident 100%. Absent stays absent; the viewer renders "—".
            #
            # The column is populated from the ALIGNER'S OWN SCORE when the aligner
            # recorded one (annotation.alignment.score, e.g. "aligner 0.94 (z=7.2)
            # -> Bed"). That is a measured similarity, not a detector confidence and
            # not a probability that the object is real -- `alignment.evidence` carries
            # what it means. Objects the aligner never scored keep an absent
            # confidence and still render as an em dash: a partly filled column is the
            # honest shape here, because only some objects were aligned.
            "confidence": _confidence_of(o, decision),
            "color": o.get("color", ""),
            "material": o.get("material", ""),
            "position": pos,
            "bbox": bbox,
            "status": "permanent" if o.get("object_id") else "temporary",
            "decision": decision,
            # The aligner's own verdict. The viewer used to synthesise a class from
            # the label with a catch-all default, so every object displayed a
            # confident taxonomy the system had never computed.
            "alignment": _alignment_of(decision),
            "crop_url": f"/crop/{crop_target}",
        })
        if o.get("room_id"):
            rid = o["room_id"]
            rnid = _nid(rid)
            if rnid not in known_room_ids:
                nodes.append({
                    "id": rnid,
                    "label": str(rid).replace("_", " ").title(),
                    "type": "room",
                    "room": rid,
                    "semantic_label": "",
                    "detected": False,
                })
                known_room_ids.add(rnid)
            edges.append({
                "id": f"e_{_oid(o)}_loc",
                "source": _oid(o),
                "target": rnid,
                "label": "isLocatedIn",
            })

    # Real spatial relations from the object manager (update_spatial_relations):
    # {pred: [object_id, ...]} with preds isIn/isOn/isNextTo/isAbove/isUnder.
    known_ids = {n["id"] for n in nodes}
    has_relations = any(isinstance(o.get("relations"), dict) for o in objects)
    if has_relations:
        seen_pairs = set()
        for o in objects:
            src = _oid(o)
            for pred, targets in (o.get("relations") or {}).items():
                for t in targets:
                    tgt = _nid(t)
                    if tgt not in known_ids:
                        continue
                    # isNextTo is symmetric — emit one edge per pair
                    if pred == "isNextTo":
                        pair = (pred, *sorted((src, tgt)))
                        if pair in seen_pairs:
                            continue
                        seen_pairs.add(pair)
                    edges.append({
                        "id": f"e_{src}_{pred}_{tgt}",
                        "source": src,
                        "target": tgt,
                        "label": pred,
                    })
    else:
        # ponytail: fallback rests-on heuristic for old-format files without
        # "relations" (z-up world frame, 15cm tolerance, O(n²) — n is tens).
        for a in objects:
            for b in objects:
                if a is b:
                    continue
                ba, bb = a.get("bbox") or {}, b.get("bbox") or {}
                try:
                    rests = abs(bb["z_min"] - ba["z_max"]) < 0.15
                    ox = min(ba["x_max"], bb["x_max"]) - max(ba["x_min"], bb["x_min"])
                    oy = min(ba["y_max"], bb["y_max"]) - max(ba["y_min"], bb["y_min"])
                except KeyError:
                    continue
                if rests and ox > 0 and oy > 0:
                    edges.append({
                        "id": f"e_{_oid(a)}_sup_{_oid(b)}",
                        "source": _oid(a),
                        "target": _oid(b),
                        "label": "supports",
                    })

    # Parse on-hold and rejected objects for audit summary
    on_hold, rejected, abstained = [], [], []
    on_hold_error = None
    u_path = out / "uncertain_objects.txt"
    if u_path.exists():
        try:
            cur = None
            for line in u_path.read_text().splitlines():
                line = line.strip()
                if line and line[0].isdigit() and "." in line:
                    label = line.split(".", 1)[1].strip()
                    cur = {"label": label, "status": "on_hold"}
                    on_hold.append(cur)
                elif cur and "Center position:" in line:
                    cur["position_str"] = line.split(":", 1)[1].strip()
        except (OSError, ValueError, IndexError) as exc:
            # A truncated or half-written file is expected while the writer runs. The
            # on-hold list is then INCOMPLETE, and a partial list must not be presented
            # as the whole one.
            on_hold_error = f"{u_path}: {type(exc).__name__}: {exc}"

    # Third reader of the same file; it shares the one parse now.
    _records, unreadable_records, decisions_error = _decision_log()
    for rec in _records:
        # GA-38: grade on the aligner's own verdict, not the top-level
        # enforcement flag. Measured over 16 archived bundles, 14 diverge and
        # every divergence is 'admit' over a verdict of decline/hold/no_grounds,
        # so grading on `outcome` always over-reports admission. `outcome` stays
        # as the fallback so records written before the writer carried a verdict
        # still classify.
        grade = ((rec.get("annotation") or {}).get("verdict") or {}).get("grade")
        grade = (grade or rec.get("outcome") or "").lower()
        if grade in ("reject", "decline"):
            rejected.append(rec)
        elif grade in ("abstain", "no_grounds", "hold"):
            abstained.append(rec)

    payload = {
        # Bumps only when a file the graph is built from changes, so the viewer can
        # skip an identical re-sync (which re-ran layout and re-fetched every crop).
        "version": version,
        # GA-40 (graphdata-no-timestamp). When this response was built and when the
        # newest source file was last written, so a reader can tell LIVE from "the
        # producer died and this is the last thing it wrote".
        "produced_at": time.time(),
        "sources_mtime": _sources_mtime(out),
        "elements": {
            "nodes": [{"data": n} for n in nodes],
            "edges": [{"data": e} for e in edges],
        },
        "nodes": nodes,
        "objects": [n for n in nodes if n.get("type") == "object"],
        "edges": edges,
        "admission_summary": {
            # An error here is rendered by the viewer as an error, never as zero.
            "error": decisions_error,
            "unreadable_records": unreadable_records,
            # GA-221: rows of other kinds (merge_refused, ...) the byte filter never parsed.
            "skipped_records": _DECISIONS_CACHE["skipped"],
            "admitted_count": len([n for n in nodes if n.get("type") == "object"]),
            "on_hold_error": on_hold_error,
            "on_hold_count": len(on_hold),
            "rejected_count": len(rejected),
            "abstained_count": len(abstained),
            "on_hold": on_hold,
            "rejected": rejected,
            "abstained": abstained,
        }
    }
    # GA-222. THE ETAG WAS COMPUTED AND NEVER SENT. It is built at the top of this function and
    # used only to answer a request that already carries `if-none-match` -- but nothing ever put
    # it on a 200, so no client could learn it, no client sent it back, and the 304 branch above
    # was unreachable. The viewer polls this route every 3 s: on the archived run it re-serialised
    # 270 nodes and 3,231 edges into 19 MB each time, 8.7 s per request against a 3 s poll, and
    # every other endpoint queued behind it. The cache existed in full and was never armed.
    #
    # `request is None` keeps the plain dict, because /admission_audit calls this function
    # directly and does `g.get("admission_summary")` on the result.
    if request is None:
        return payload
    return JSONResponse(content=payload, headers={"ETag": etag, "Cache-Control": "no-cache"})


@app.get("/admission_audit")
def get_admission_audit():
    g = graph_data(request=None)
    return JSONResponse(content=g.get("admission_summary", {}))


# object_id -> label, from the `link` records object_manager_6 writes into
# hook_decisions.jsonl. Rebuilt only when the file changes.
_LINK_CACHE = {"key": None, "map": {}}


def _link_index():
    for base in (_active_output_dir(), _active_output_dir().parent):
        path = base / "hook_decisions.jsonl"
        if not path.exists():
            continue
        try:
            st = path.stat()
        except OSError:
            continue
        key = (str(path), st.st_mtime, st.st_size)
        if _LINK_CACHE["key"] == key:
            return _LINK_CACHE["map"]
        mapping = {}
        for rec in _decision_records():      # shared parse, not a third pass
            if rec.get("kind") != "link":
                continue
            oid, label = rec.get("object"), rec.get("label")
            if oid and label:
                mapping[str(oid)] = str(label)
        _LINK_CACHE.update(key=key, map=mapping)
        return mapping
    return {}


_CROP_DIRS = None


_RUN_REF = {"t": 0.0, "value": None}


def _run_reference_time(ttl: float = 10.0):
    """Earliest mtime among the active output directory's own files, or None.

    Everything in the run's output directory was written during this run, so its
    OLDEST file is at or after the run started. A file found in a fallback directory
    that predates that is from an earlier run.

    Returns None when it cannot be established (empty or unreadable directory). Callers
    treat None as "cannot tell" and accept the candidate, so the default reproduces
    today's behaviour exactly.
    """
    now = time.time()
    if _RUN_REF["t"] and now - _RUN_REF["t"] < ttl:
        return _RUN_REF["value"]
    ref = None
    try:
        mtimes = [f.stat().st_mtime for f in _active_output_dir().iterdir() if f.is_file()]
        ref = min(mtimes) if mtimes else None
    except OSError:
        ref = None
    _RUN_REF.update(t=now, value=ref)
    return ref


def _pick_run_file(dirs, name, errors=None):
    """First `dirs`/`name` that can belong to THIS run.

    GA-103: these lookups used to take the first directory that merely CONTAINED the
    file, walking a list that ends at /tmp. Nothing checked age, so a leftover from an
    earlier run would be served as the current one -- not a lost artefact but a
    MISATTRIBUTED one, which is the harder failure to notice because every number still
    looks plausible.

    A file inside the active output directory is always accepted: it is this run's by
    construction. A file in a fallback directory is accepted only if it is at least as
    new as the run reference; otherwise it is skipped and the skip is RECORDED.
    """
    active = _active_output_dir()
    ref = _run_reference_time()
    for d in dirs:
        candidate = Path(d) / name
        if not candidate.exists():
            continue
        try:
            same_run = Path(d).resolve() == active.resolve()
        except OSError:
            same_run = False
        if same_run or ref is None:
            return candidate
        try:
            age_ok = candidate.stat().st_mtime >= ref
        except OSError:
            continue
        if age_ok:
            return candidate
        if errors is not None:
            errors.append(
                f"{candidate}: predates this run "
                f"({time.strftime('%H:%M:%S', time.localtime(candidate.stat().st_mtime))} "
                f"< {time.strftime('%H:%M:%S', time.localtime(ref))}); skipped, not served "
                f"as current"
            )
    return None


_DECISIONS_CACHE = {"key": None, "records": [], "unreadable": 0, "error": None,
                    "path": None, "offset": 0, "skipped": 0}

# The record kinds anyone reads: `admission` (or a kind-less row from an old log) and `link`.
# Same byte filter as replay_server._install_decision_reader.
_DECISION_KEEP = (b'"kind": "admission"', b'"kind":"admission"', b'"kind": "link"', b'"kind":"link"')


def _decision_log():
    """The cached parse plus what went wrong reading it: (records, unreadable, error).

    The admission summary needs the unreadable count and the read error, so the cache
    carries them rather than each reader re-deriving them from its own pass.
    """
    _decision_records()
    return (_DECISIONS_CACHE["records"], _DECISIONS_CACHE["unreadable"],
            _DECISIONS_CACHE["error"])


def _decision_records():
    """The admission and link records of hook_decisions.jsonl, cached and read incrementally.

    This file was being parsed THREE TIMES per /graph_data: once to build the decisions
    map, once by _link_index for crop labels, and once for the admission summary. Cached
    on (path, mtime, size), the same key _link_index already used.

    GA-221. The whole file was then re-read and re-parsed on every change -- 2.4 s for a
    41 MB / 29,847-row live log, of which the three consumers read 548 rows; on an archived
    1.8 GB log /graph_data never returned. Two changes: a line is filtered AS BYTES before
    json.loads (only `admission`, kind-less and `link` rows are parsed; the rest are counted
    in `skipped`), and the log is read from the offset the last call reached, so a live poll
    costs the bytes appended since, not the file. A new path or a shrunken file starts over.
    A trailing line with no newline is a write in progress and is left for the next call,
    unless the file has been quiet for 5 s.
    """
    c = _DECISIONS_CACHE
    for base in (_active_output_dir(), _active_output_dir().parent):
        path = base / "hook_decisions.jsonl"
        if not path.exists():
            continue
        try:
            st = path.stat()
        except OSError:
            continue
        key = (str(path), st.st_mtime, st.st_size)
        if c["key"] == key:
            return c["records"]
        if c["path"] != str(path) or st.st_size < c["offset"]:
            c.update(records=[], unreadable=0, skipped=0, offset=0, path=str(path))
        records, unreadable, skipped, offset = c["records"], c["unreadable"], c["skipped"], c["offset"]
        quiet = (time.time() - st.st_mtime) > 5.0
        try:
            with path.open("rb") as f:
                f.seek(offset)
                for line in f:
                    if not line.endswith(b"\n") and not quiet:
                        break
                    offset += len(line)
                    if b'"kind"' in line and not any(k in line for k in _DECISION_KEEP):
                        skipped += 1
                        continue
                    if not line.strip():
                        continue
                    try:
                        records.append(json.loads(line))
                    except json.JSONDecodeError:
                        # Counted and surfaced, never swallowed: a truncated log must not read
                        # as a complete one.
                        unreadable += 1
        except OSError as exc:
            # The panel must not render zero counts from a log we could not open: an
            # empty admission panel is indistinguishable from a clean run.
            c.update(key=key, records=[], unreadable=0, skipped=0, offset=0, path=None,
                     error=f"could not read the decision log: {exc}")
            return []
        c.update(key=key, records=records, unreadable=unreadable, skipped=skipped,
                 offset=offset, error=None)
        return records
    c.update(key=None, records=[], unreadable=0, error=None, path=None, offset=0, skipped=0)
    return []


def _admissions_by_oid():
    """Admission records keyed by OBJECT ID, joined through the `link` rows
    (object_id -> decision_id -> annotation.decision_id). GA-40 / GA-102: the admission
    row's own `object` field is the per-frame label, which is not an identity."""
    records = _decision_records()
    by_decision = {}
    for rec in records:
        if rec.get("kind") in (None, "admission"):
            did = (rec.get("annotation") or {}).get("decision_id")
            if did:
                by_decision[did] = rec
    return {str(rec["object"]): by_decision[rec["decision_id"]] for rec in records
            if rec.get("kind") == "link" and rec.get("object")
            and rec.get("decision_id") in by_decision}


def _crop_dirs():
    """Resolve the cropped_images directories for the ACTIVE run.

    Cached, because eight directories were being globbed up to three times per object per
    request and only one of them is ever the run's. KEYED ON THE ACTIVE OUTPUT DIRECTORY,
    because that directory MOVES: the replay dashboard re-points GRAPH_API_OUTPUT_DIR when
    it follows a new run or when a bundle is chosen from the page, and every other reader
    here re-resolves per call.

    Cached once and forever, this returned the FIRST run's crop directory for the whole life
    of the process. Measured 2026-09-04 on the replay dashboard: 146 of 147 nodes carried a
    crop_url and every one of them answered with the 346-byte placeholder SVG, while the same
    route on the live bridge returned real JPEGs -- because the live bridge's output directory
    never moves and the replay server's always does. It reads on screen as "the crops were
    never collected", and the crops were on disk the whole time (414 files in that bundle).
    """
    global _CROP_DIRS
    active = _active_output_dir()
    if _CROP_DIRS is None or _CROP_DIRS[0] != active:
        candidates = [
            active,
            _PROJECT_ROOT / "output",
            Path("/ws/install/lost3dsg/output"),
            Path("/ws/output"),
            Path("/out"),
            Path("/tmp/graphapi_live"),
        ]
        _CROP_DIRS = (active, [c / "cropped_images" for c in candidates])
    return [d for d in _CROP_DIRS[1] if d.exists()]


def _crop_label(target: str) -> str:
    """Label for a crop target, tolerating the graph's node-id form.

    Graph node ids are the object id behind an `n_` prefix (see `_nid`), so a caller
    that passes a node id instead of the node's own `crop_url` missed the link index
    entirely. `_crop_file` then searched for crops named after the raw identifier and
    found none, and the placeholder printed it: N_OBJ_575160D955E149E39A51B7 shown to
    a person where a label belongs.

    Exact match is tried first, so this can only ADD resolutions, never change one that
    already worked.
    """
    index = _link_index()
    if target in index:
        return index[target]
    if target.startswith("n_"):
        stripped = target[2:]
        if stripped in index:
            return index[stripped]
    return target


def _crop_target_forms(target: str):
    """The identifiers a crop file may be named after, most specific first."""
    forms = [target]
    if target.startswith("n_"):
        forms.append(target[2:])
    return forms


def _crop_file(target: str):
    """Newest crop written for `target`, matched EXACTLY.

    input_output.prepare_crops names files crop_<safe_label>_<%Y%m%d>_<%H%M%S>_<idx>.jpg.
    Anchoring on that shape is what makes the match exact: a bare `desk` cannot
    collect `desk#1`'s crops, which a prefix or substring test does. The old
    substring and base-prefix fallbacks are deleted deliberately -- they served a
    wrong image confidently, which is worse than serving none.
    """
    label = _crop_label(target)
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(label))
    pattern = re.compile(r"^crop_" + re.escape(safe) + r"_\d{8}_\d{6}_\d+\.jpg$")
    best = None
    for cdir in _crop_dirs():
        for form in _crop_target_forms(target):
            exact = cdir / f"{form}.jpg"
            if exact.exists():
                return exact
        for p in cdir.glob("crop_*.jpg"):
            if not pattern.match(p.name):
                continue
            try:
                mt = p.stat().st_mtime
            except OSError:
                continue
            if best is None or mt > best[0]:
                best = (mt, p)
    return best[1] if best else None


@app.get("/walls")
def get_walls():
    """Detected wall segments, per-frame from depth. GA-271.

    `available` is explicit because wall_detector is OPT-IN (WALL_DETECTOR=1): it cost 4.4
    cores and starved rtabmap, so it is off by default. An empty list from a node that is not
    running and an empty list from a frame with no walls are different facts, and a viewer
    that cannot tell them apart will report "no walls" for a detector nobody started.
    """
    node = get_node()
    walls = list(getattr(node, "latest_walls", []) or []) if node else []
    stamp = float(getattr(node, "walls_stamp", 0.0) or 0.0) if node else 0.0
    age = (time.time() - stamp) if stamp else None
    return {
        "walls": walls,
        "count": len(walls),
        "age_s": round(age, 2) if age is not None else None,
        "available": stamp > 0.0,
        "why": (None if stamp > 0.0 else
                "no /detected_wall_segments seen; wall_detector is opt-in (WALL_DETECTOR=1)"),
    }


# BOTH spellings, and the singular one matters most: every `crop_url` this bridge emits
# is `/crop/<id>`. When get_walls was added its decorator was inserted BETWEEN
# `@app.get("/crop/{target}")` and this function, so the singular route bound to
# get_walls and every crop request on the dashboard came back as the walls JSON. Nothing
# errored -- the viewer asked for an image, got 136 bytes of JSON, and simply showed no
# thumbnail, which reads as "crops are not being collected" when 133 of them were sitting
# in the bundle. A decorator belongs to the function directly beneath it; anything
# inserted between the two silently steals the route.
@app.get("/crop/{target}")
@app.get("/crops/{target}")
def get_crop_image(target: str, request: Request = None):
    path = _crop_file(target)
    if path is None:
        # GA-40 (crop-substring-mtime). No crop is a 404, not a 200 with a synthesised
        # SVG: the viewer's <img onerror> renders "Crop pending" only when the request
        # fails, so the placeholder made "no evidence" look like a styled card. Not
        # cached: the crop for a real object appears mid-run.
        return Response(status_code=404, headers={"Cache-Control": "no-store"})
    try:
        st = path.stat()
    except OSError:
        return Response(status_code=404)
    etag = f'"{int(st.st_mtime)}-{st.st_size}"'
    if request is not None and request.headers.get("if-none-match") == etag:
        return Response(status_code=304, headers={"ETag": etag})
    # A crop file never changes once written, so it is safe to cache hard. The
    # dashboard re-requests every node's crop on each 3 s graph refresh.
    return FileResponse(str(path), media_type="image/jpeg",
                        headers={"ETag": etag, "Cache-Control": "public, max-age=3600"})


# GA-270. From config; see config.py "services". The bridge probes this address for every
# frame, so a stale literal here degrades the live view silently rather than loudly.
try:
    from config import CFG as _BRIDGE_CFG
except ImportError:
    _BRIDGE_CFG = {}
_SVC_B = (_BRIDGE_CFG.get("services", {}) or {}) if isinstance(_BRIDGE_CFG, dict) else {}
FEED_HOST = os.environ.get("FEED_HOST") or (
    f"http://{_SVC_B.get('feed_host', '127.0.0.1')}:{_SVC_B.get('feed_port', 7790)}")

# Serve composite_*.jpg fallback frames only if this fresh — older ones are
# leftovers from past recordings and masquerade as a live feed.
COMPOSITE_MAX_AGE_SEC = 30.0

# An /image_with_bb frame older than this is last cycle's, not the live view.
# Perception publishes one frame per cycle and none at all while the agent walks,
# so without this the dashboard pins itself to a single annotated frame forever.
ANNOTATED_MAX_AGE_SEC = float(os.environ.get("BRIDGE_ANNOTATED_MAX_AGE", "1.5"))
RAW_MAX_AGE_SEC = float(os.environ.get("BRIDGE_RAW_MAX_AGE", "3.0"))
# GA-183. How long `call()` waits for a ROS service to answer. Was a bare 5.0, and `merge`
# routinely exceeds it: run 20260901_055513 returned 500 on 89 of 140 merge POSTs while the
# merges THEMSELVES SUCCEEDED -- 89 discarded objects left the world model against only 51
# responses that said 200. The request is already dispatched when the wait expires, so a
# timeout here is never evidence that the work did not happen.
SERVICE_CALL_TIMEOUT_SEC = float(os.environ.get("BRIDGE_SERVICE_TIMEOUT", "30.0"))
# resend an unchanged frame at least this often, so a motionless scene still looks live
FEED_HEARTBEAT_SEC = 1.0

# CONFIRMED (dispatch item 8): _best_frame does a synchronous urlopen to the feed
# host with a 0.4 s timeout. With the host down, EVERY /frame.jpg request pays it,
# and /feed's iterfile() loop pays it once per iteration -- the loop drops from
# ~12.5 Hz to ~2.5 Hz and each iteration parks a threadpool worker for 0.4 s.
# After a failure, skip the probe entirely for this long; one request per interval
# re-tests, so recovery costs at most this much latency.
FEED_PROBE_BACKOFF_SEC = float(os.environ.get("BRIDGE_FEED_PROBE_BACKOFF", "3.0"))
_feed_probe_blocked_until = 0.0

# Which of _best_frame's four sources produced the frame the viewer is looking at.
# All four render identically, so "the feed is up" said nothing about whether you
# were seeing the live overlay or a 25 s old composite.
_last_frame_source = None


# ---------------------------------------------------------------------------------------
# THE LIVE OVERLAY. Boxes on the frame the dashboard is actually showing.
#
# The simulator draws its belief overlay only into its own GUI window copy, so the frame it
# SERVES has never carried a box -- measured 2026-09-04, the bridge frame and the host frame
# were byte-identical at 117,117 bytes. Perception's /image_with_bb does carry boxes, but it
# publishes one frame per cycle, so the dashboard alternated between clean frames and an
# occasional annotated one. That is the "an older frame with boxes flashes past" report.
#
# Here the boxes are PROJECTED from the persistent world model onto whichever frame is being
# served, so an object stays outlined for as long as it is in view. The projection lives in
# live_overlay.py and is verified there against the pipeline's own recorded 2D boxes.
try:
    from . import live_overlay as _lo
except ImportError:                                   # launched by file path, not as a package
    import live_overlay as _lo
try:
    from .image_transport import ros_image_to_array
except ImportError:                                   # launched by file path, not as a package
    from image_transport import ros_image_to_array

OVERLAY_ON = os.environ.get("BRIDGE_OVERLAY", "1") == "1"
# The frame the world model is expressed in, and the camera frame to look it up as. Both are
# configurable because a rename in the TF tree must not silently draw boxes in the wrong place.
OVERLAY_MAP_FRAME = os.environ.get(
    "BRIDGE_OVERLAY_MAP_FRAME",
    str((CFG.get("tf") or {}).get("world_frame") or "map"),
)
OVERLAY_CAM_FRAME = os.environ.get(
    "BRIDGE_OVERLAY_CAM_FRAME",
    str((CFG.get("frames") or {}).get("camera") or "habitat_camera_optical"),
)

_OVERLAY_OBJ = {"key": None, "objects": []}
_OVERLAY_INTR = {"key": None, "value": None}
_OVERLAY_OUT = {"key": None, "jpeg": None}
# WHICH source gave the pose. Reported on /health beside frame_source, because a pose from a
# stale detection record and a pose from TF draw the same picture and are not the same claim.
_overlay_pose_source = None


def _overlay_objects():
    """The persistent world model, re-read when the file changes. Same file /graph_data uses."""
    path = _active_output_dir() / "persistent_perception.json"
    try:
        st = path.stat()
    except OSError:
        return []
    key = (str(path), st.st_mtime_ns, st.st_size)
    if _OVERLAY_OBJ["key"] != key:
        try:
            data = json.loads(path.read_text())
        except (OSError, ValueError):
            return _OVERLAY_OBJ["objects"]
        objs = data if isinstance(data, list) else data.get("objects", [])
        _OVERLAY_OBJ.update(key=key, objects=[o for o in objs if isinstance(o, dict)])
    return _OVERLAY_OBJ["objects"]


def _overlay_intrinsics(width, height):
    """Intrinsics for a frame of this size.

    calibration.json is written per run and is authoritative. It is recorded for the sensor's
    own resolution, so a frame served at another size is scaled rather than used as-is -- fx
    and cx are in pixels and do not survive a resize. With no calibration, derive from the
    feed's horizontal field of view, which is what calibration.json itself does.
    """
    path = _active_output_dir() / "calibration.json"
    try:
        st = path.stat()
        key = (str(path), st.st_mtime_ns, width, height)
    except OSError:
        st, key = None, ("hfov", width, height)
    if _OVERLAY_INTR["key"] == key:
        return _OVERLAY_INTR["value"]
    intr = None
    if st is not None:
        try:
            cal = json.loads(path.read_text())
            i, res = cal["intrinsics"], cal.get("resolution") or {}
            sw, sh = res.get("width") or width, res.get("height") or height
            kx, ky = width / float(sw), height / float(sh)
            intr = {"fx": i["fx"] * kx, "fy": i["fy"] * ky,
                    "cx": i["cx"] * kx, "cy": i["cy"] * ky}
        except (OSError, ValueError, KeyError, ZeroDivisionError, TypeError):
            intr = None
    if intr is None:
        hfov = math.radians(float(os.environ.get("FEED_HFOV", "90")))
        f = (width / 2.0) / math.tan(hfov / 2.0)
        intr = {"fx": f, "fy": f, "cx": width / 2.0, "cy": height / 2.0}
    _OVERLAY_INTR.update(key=key, value=intr)
    return intr


def _pose_from_tf():
    """The camera pose from TF, or None. THE CORRECT LIVE SOURCE, and the only per-frame one.

    WRITTEN, NOT RUN. There was no live stack up when this was added, so this path has never
    answered. It returns None on anything unexpected and the caller falls through to the
    recorded pose, which is why a wrong frame name here degrades rather than misdraws -- and
    /health says which source actually answered, so "TF is working" is never assumed.
    """
    node = get_node()
    buf = getattr(node, "tf_buffer", None) if node else None
    if buf is None:
        return None
    try:
        import rclpy.time
        t = buf.lookup_transform(OVERLAY_MAP_FRAME, OVERLAY_CAM_FRAME, rclpy.time.Time())
    except Exception:      # tf2 raises several unrelated exception types; none is fatal here
        return None
    tr, ro = t.transform.translation, t.transform.rotation
    return ([tr.x, tr.y, tr.z], [ro.x, ro.y, ro.z, ro.w], "tf")


def _pose_from_detections(max_age=90.0):
    """The camera pose recorded with the newest detection, or None.

    One perception cycle stale by construction, so boxes lag while the agent walks. It is the
    fallback, not the design, and it is what makes the overlay testable with no ROS at all.
    """
    path = _active_output_dir() / "detections.jsonl"
    try:
        st = path.stat()
        if time.time() - st.st_mtime > max_age:
            return None
        with path.open("rb") as f:
            f.seek(max(0, st.st_size - 262144))
            tail = f.read().splitlines()
    except OSError:
        return None
    for line in reversed(tail):
        try:
            r = json.loads(line)
        except ValueError:
            continue
        if r.get("camera_position") and r.get("camera_quat_xyzw"):
            return (r["camera_position"], r["camera_quat_xyzw"], "newest detection record")
    return None


def _overlay_pose(max_age=90.0):
    """The camera pose to draw with, TF first and the recorded pose second.

    `max_age` is threaded through to the recorded source and defaults to what it always was,
    so every existing caller is unchanged (working rule 6 applied to a signature). Only
    /overlay_pose passes anything else, and only to REPLAY an archived run's own poses -- see
    the route, which refuses to call that a live reading.
    """
    global _overlay_pose_source
    for src in (_pose_from_tf, lambda: _pose_from_detections(max_age=max_age)):
        got = src()
        if got:
            _overlay_pose_source = got[2]
            return got
    _overlay_pose_source = None
    return None


def _with_overlay(jpeg):
    """Draw the world model's boxes onto `jpeg`. Returns the original on any failure.

    Cached on the frame bytes and the pose, because /feed calls this at up to 12 Hz over
    sources that update at ~3 Hz; without it every repeat would pay a decode and an encode.
    """
    if not OVERLAY_ON or not jpeg:
        return jpeg
    objects = _overlay_objects()
    if not objects:
        return jpeg
    pose = _overlay_pose()
    if not pose:
        return jpeg
    pos, quat, _ = pose
    key = (hashlib.sha1(jpeg).hexdigest(), tuple(pos), tuple(quat), len(objects))
    if _OVERLAY_OUT["key"] == key:
        return _OVERLAY_OUT["jpeg"]
    try:
        img = cv2.imdecode(np.frombuffer(jpeg, dtype=np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            return jpeg
        h, w = img.shape[:2]
        boxes = _lo.visible_boxes(objects, pos, quat, _overlay_intrinsics(w, h), w, h)
        if not boxes:
            _OVERLAY_OUT.update(key=key, jpeg=jpeg)
            return jpeg
        _lo.draw(img, boxes, cv2)
        ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if not ok:
            return jpeg
        out = buf.tobytes()
    except (cv2.error, ValueError, TypeError, AttributeError) as exc:
        # COUNTED, not swallowed: an overlay that silently stops drawing looks exactly like a
        # run that has found nothing, which is the confusion this whole file keeps unpicking.
        _OVERLAY_FAILURES["n"] += 1
        _OVERLAY_FAILURES["last"] = f"{type(exc).__name__}: {exc}"
        return jpeg
    _OVERLAY_OUT.update(key=key, jpeg=out)
    return out


_OVERLAY_FAILURES = {"n": 0, "last": None}


# GA-354: how often each source actually answered. _last_frame_source was set on every pick and
# read by nobody but the badge, so the owner's "why does the label alternate" had no number.
# Counted PER SERVED PICK (one per /frame.jpg request or /feed part), not per camera frame;
# written to the run's output dir every 30 s and at exit as feed_source_counts.json (a new
# file, rule 6), and carried on /health.stamp.frame_source_counts.
_FRAME_SOURCE_COUNTS = {"perception overlay": 0, "simulator host": 0, "raw camera": 0,
                        "stale overlay": 0, "stale composite": 0, "none": 0}
# Both of these bytes already contain the perception renderer's boxes.  A stale annotated
# image is still annotated; treating it as a raw frame and applying the persistent world-model
# overlay a second time creates the exact duplicated/misaligned boxes visible in the dashboard.
_ANNOTATED_FRAME_SOURCES = {"perception overlay", "stale overlay"}
_FRAME_SOURCE_WRITE = {"at": 0.0, "lock": threading.Lock()}
_FRAME_PICK_LOCK = threading.Lock()
_FRAME_PICK_LOCAL = threading.local()
atexit.register(lambda: _write_frame_source_counts(force=True))


def _write_frame_source_counts(force=False):
    now = time.time()
    if not force and now - _FRAME_SOURCE_WRITE["at"] < 30.0:
        return
    with _FRAME_SOURCE_WRITE["lock"]:
        _FRAME_SOURCE_WRITE["at"] = now
        try:
            path = _active_output_dir() / "feed_source_counts.json"
            path.write_text(json.dumps({"counts": dict(_FRAME_SOURCE_COUNTS),
                                        "total_picks": sum(_FRAME_SOURCE_COUNTS.values()),
                                        "unit": "served picks (one per /frame.jpg request or /feed part)",
                                        "written_at": now}, indent=1))
        except OSError:
            pass                 # the output dir can vanish at run end; the counts stay in /health


def _best_frame():
    """`_best_frame_pick` plus the GA-354 tally; every caller goes through here."""
    # The source used to be read from the process-global `_last_frame_source` after this
    # function returned. `/feed` and `/frame.jpg` can overlap, so another request could change
    # that value between the pick and the guard, causing an already annotated image to be
    # composited again. Keep the source with this request as well as the health badge's global.
    with _FRAME_PICK_LOCK:
        data = _best_frame_pick()
        source = _last_frame_source
    _FRAME_PICK_LOCAL.source = source
    _FRAME_SOURCE_COUNTS[source or "none"] = _FRAME_SOURCE_COUNTS.get(source or "none", 0) + 1
    _write_frame_source_counts()
    return data


def _frame_needs_world_overlay(source):
    """Whether `source` is an unannotated frame that needs persistent 3D boxes."""
    return source not in _ANNOTATED_FRAME_SOURCES


def _picked_frame_source():
    """Source selected by this request, falling back to the health badge for old callers."""
    return getattr(_FRAME_PICK_LOCAL, "source", _last_frame_source)


def _best_frame_pick():
    """The freshest frame worth showing, newest source first.

    1. the current perception overlay (/image_with_bb) while it is still current
    2. the simulator host's live feed — every rendered frame, belief boxes drawn on
       (habitat_feed_host draws them into CTRL.latest_jpeg), so motion stays smooth
       between cycles instead of freezing on the last cycle's output
    3. the raw ROS camera, if the host is unreachable
    4. a stale annotated frame — worse than nothing only if it pretends to be live,
       and by here every live source has already failed
    """
    global _feed_probe_blocked_until, _last_frame_source
    node = get_node()
    now = time.time()
    if node and node.latest_jpeg and (now - node.last_frame_time) < ANNOTATED_MAX_AGE_SEC:
        _last_frame_source = "perception overlay"
        return node.latest_jpeg
    if _SVC_B.get("feed_enabled", True) and now >= _feed_probe_blocked_until:
        try:
            with urllib.request.urlopen(f"{FEED_HOST}/frame.jpg", timeout=0.4) as r:
                data = r.read()
                _feed_probe_blocked_until = 0.0
                _last_frame_source = "simulator host"
                return data
        except (urllib.error.URLError, OSError, TimeoutError):
            # Unreachable feed host: expected between runs. Back off rather than probing
            # every frame. Anything else here is our bug and must raise.
            _feed_probe_blocked_until = now + FEED_PROBE_BACKOFF_SEC
    if node and node.raw_jpeg and (now - node.last_raw_time) < RAW_MAX_AGE_SEC:
        _last_frame_source = "raw camera"
        return node.raw_jpeg
    if node and node.latest_jpeg:
        _last_frame_source = "stale overlay"
        return node.latest_jpeg
    composite = _latest_fresh_composite()
    _last_frame_source = "stale composite" if composite else None
    return composite


def _latest_fresh_composite():
    for frame_dir in [Path("/out"), Path("/tmp/graphapi_live"), _active_output_dir(), Path("/tmp")]:
        if not frame_dir.exists():
            continue
        composites = sorted(frame_dir.glob("composite_*.jpg"), key=lambda p: p.stat().st_mtime, reverse=True)
        if not composites:
            continue
        newest = composites[0]
        if (time.time() - newest.stat().st_mtime) > COMPOSITE_MAX_AGE_SEC:
            continue
        try:
            return newest.read_bytes()
        except OSError:
            continue
    return None


@app.get("/frame.jpg")
def proxy_frame():
    frame = _best_frame()
    if frame:
        # NOT when perception's own overlay is what answered: that frame already carries the
        # boxes AND the masks, drawn from the detection itself rather than reprojected, and
        # drawing over it would put two rectangles round every object.
        if _frame_needs_world_overlay(_picked_frame_source()):
            frame = _with_overlay(frame)
        return Response(content=frame, media_type="image/jpeg")
    return Response(status_code=503)

@app.get("/last_perception/meta")
def last_perception_meta():
    """Describe the retained cycle image independently from the live feed."""
    node = get_node()
    frame = getattr(node, "latest_jpeg", None) if node else None
    captured_at = float(getattr(node, "last_frame_time", 0.0) or 0.0) if node else 0.0
    if not frame or not captured_at:
        return {
            "available": False,
            "captured_at": None,
            "age_s": None,
            "revision": None,
            "url": None,
        }
    return {
        "available": True,
        "captured_at": captured_at,
        "age_s": round(max(0.0, time.time() - captured_at), 2),
        "revision": str(int(captured_at * 1_000_000)),
        "url": "/last_perception.jpg",
    }


@app.get("/last_perception.jpg")
def last_perception_frame(request: Request = None):
    """Return the newest annotated cycle even after the live feed resumes."""
    node = get_node()
    frame = getattr(node, "latest_jpeg", None) if node else None
    captured_at = float(getattr(node, "last_frame_time", 0.0) or 0.0) if node else 0.0
    if not frame or not captured_at:
        return Response(status_code=404, headers={"Cache-Control": "no-store"})
    etag = f'"perception-{int(captured_at * 1_000_000)}"'
    if request is not None and request.headers.get("if-none-match") == etag:
        return Response(status_code=304, headers={"ETag": etag})
    return Response(
        content=frame,
        media_type="image/jpeg",
        headers={"ETag": etag, "Cache-Control": "no-cache, must-revalidate"},
    )

@app.get("/feed")
@app.get("/feed.mjpg")
def proxy_feed():
    def iterfile():
        last, last_sent = None, 0.0
        while True:
            frame_data = _best_frame()
            if frame_data and _frame_needs_world_overlay(_picked_frame_source()):
                frame_data = _with_overlay(frame_data)   # see proxy_frame for why the guard

            # Don't re-push a frame the browser already has: the sources run at ~3 fps
            # and this loop at 12.5, so most iterations used to resend an identical JPEG
            # — wasted bandwidth, and it made the viewer's FPS readout report the loop
            # rate rather than the real one. A still scene legitimately renders identical
            # frames, so resend anyway on FEED_HEARTBEAT_SEC to stay under the viewer's
            # 2 s "stalled" badge and its 6 s reconnect.
            if (frame_data is not None and frame_data == last
                    and (time.time() - last_sent) < FEED_HEARTBEAT_SEC):
                time.sleep(0.08)
                continue
            last, last_sent = frame_data, time.time()

            if frame_data:
                header = b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: " + str(len(frame_data)).encode() + b"\r\n\r\n"
                yield header + frame_data + b"\r\n"
            else:
                svg = b'--frame\r\nContent-Type: image/svg+xml\r\n\r\n<svg xmlns="http://www.w3.org/2000/svg" width="640" height="480"><rect width="100%" height="100%" fill="#090d16"/><text x="50%" y="50%" fill="#38bdf8" font-size="20" font-weight="bold" text-anchor="middle">CONNECTING TO LIVE ROS STREAM...</text></svg>\r\n'
                yield svg
            time.sleep(0.08)

    return StreamingResponse(iterfile(), media_type="multipart/x-mixed-replace; boundary=frame")

# id -> raw image bytes, for maps lifted out of the /bev_data payload.
_BEV_MAP_BYTES = {}


def _externalise_map(entry):
    if not isinstance(entry, dict) or not entry.get("image"):
        return entry
    uri = entry["image"]
    if not isinstance(uri, str) or not uri.startswith("data:"):
        return entry
    try:
        header, b64 = uri.split(",", 1)
        raw = base64.b64decode(b64)
    except (ValueError, binascii.Error):
        return entry
    mime = header[5:].split(";", 1)[0] or "image/png"
    map_id = hashlib.sha1(raw).hexdigest()[:16]
    _BEV_MAP_BYTES[map_id] = (mime, raw)
    while len(_BEV_MAP_BYTES) > 32:
        _BEV_MAP_BYTES.pop(next(iter(_BEV_MAP_BYTES)))
    out = {k: v for k, v in entry.items() if k != "image"}
    out["url"] = f"/bev_map/{map_id}"
    return out


def _externalise_maps(data):
    if not isinstance(data, dict):
        return data
    if isinstance(data.get("map"), dict):
        data["map"] = _externalise_map(data["map"])
    if isinstance(data.get("maps"), dict):
        data["maps"] = {k: _externalise_map(v) for k, v in data["maps"].items()}
    return data


@app.get("/bev_map/{map_id}")
def get_bev_map(map_id: str, request: Request = None):
    entry = _BEV_MAP_BYTES.get(map_id)
    if entry is None:
        return Response(status_code=404)
    mime, raw = entry
    etag = f'"{map_id}"'          # the id IS the content hash
    if request is not None and request.headers.get("if-none-match") == etag:
        return Response(status_code=304, headers={"ETag": etag})
    return Response(content=raw, media_type=mime,
                    headers={"ETag": etag, "Cache-Control": "public, max-age=86400"})


@app.get("/bev_data")
def proxy_bev_data(floor_y: str = None):
    # floor_y arrives from a query string, so it is untrusted. Removing the old
    # `except Exception: pass` around the slicing block below (GA-91) would otherwise
    # turn `?floor_y=abc` into a 500: the caller sending nonsense is not a server fault.
    # Validate the INPUT here and reject it; leave a genuine internal inconsistency to
    # raise, which is the whole point of removing that handler.
    if floor_y is not None and str(floor_y).lower() != "auto":
        try:
            float(floor_y)
        except (TypeError, ValueError):
            raise HTTPException(
                status_code=400,
                detail=f"floor_y must be a number or 'auto', got {floor_y!r}",
            )

    if (_BRIDGE_CFG.get("bev") or {}).get("source") == "ros":
        if _node is None or _node.ros_bev is None:
            return JSONResponse(content={
                "agent": None, "map": None, "floors": [],
                "source_errors": ["ROS BEV subscriber is not running"],
            })
        data = _externalise_maps(_node.ros_bev.payload())
        latency_file = _active_output_dir() / "perception_latencies.json"
        if latency_file.exists():
            try:
                data["latencies"] = json.loads(latency_file.read_text())
            except (OSError, json.JSONDecodeError) as exc:
                data.setdefault("source_errors", []).append(
                    f"{latency_file}: {type(exc).__name__}: {exc}")
        return JSONResponse(content=data)

    data = {}

    # Rule 14: these three handlers were `except Exception: pass`. The reasons are now
    # kept and travel with the payload, because a BEV that silently fell back looks
    # exactly like a BEV that worked.
    bev_errors = []

    bev_file = _pick_run_file(
        [_active_output_dir(), Path("/out"), Path("/tmp/graphapi_live"), Path("/tmp")],
        "bev_data.json", bev_errors)
    if bev_file is not None:
        # A partially written file is a real, expected condition here: the feed writes
        # this while the viewer polls it. That is a reason to say so, never to be silent.
        try:
            data = json.loads(bev_file.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            bev_errors.append(f"{bev_file}: {type(exc).__name__}: {exc}")
            data = {}

    if not data or not data.get("agent"):
        # An unreachable feed host is normal and transient on a 0.5s timeout, so it does
        # not stop the request -- but the caller is told which source answered.
        url = f"{FEED_HOST}/bev_data"
        if floor_y:
            url += f"?floor_y={floor_y}"
        try:
            with urllib.request.urlopen(url, timeout=0.5) as r:
                data = json.loads(r.read().decode())
        except (urllib.error.URLError, OSError, json.JSONDecodeError, TimeoutError) as exc:
            bev_errors.append(f"{url}: {type(exc).__name__}: {exc}")

    if not data:
        # GA-92: this used to invent `"floors": [-2.5, 0.5]`. Nothing measured them --
        # they were a guess that happens to match one scene, so a reader could not tell a
        # real answer from the invented one. That is the GA-89 shape in a different key:
        # a value that looks measured, produced by the path where nothing answered.
        # An empty list is honest, and the viewer renders it as no floor selector.
        data = {"agent": None, "floors": [], "navmesh": [], "map": None}
        bev_errors.append(
            "no BEV source answered: agent, floors, navmesh and map are unavailable, "
            "not measured as empty"
        )

    # Resolve active map slice by requested floor_y if maps dict is provided.
    #
    # GA-91: this block is why the floor selector was inert. It runs only when the
    # payload carries a `maps` DICT; the feed host ships a single `map`, so it never
    # ran -- and the `except Exception: pass` that used to sit here meant a failure left
    # no trace anywhere. The defect had to be found by comparing payload bytes across
    # three floor_y values, because no log said anything. The handler is gone: a
    # malformed floor key or a non-numeric floor_y is a defect and now raises.
    if data and floor_y and data.get("maps"):
        if not isinstance(data["maps"], dict):
            # The slicing below needs floor -> map. A list carries no floor keys.
            raise TypeError(
                f"bev payload 'maps' must be a dict keyed by floor, got "
                f"{type(data['maps']).__name__}"
            )
        # Not every key is a floor height. The feed host's legacy single-map cache
        # stores the literal key "auto" (habitat_feed_host.py ~line 488), and an
        # unguarded float(k) over the keys raised on it every frame -- that exact fault
        # once took the whole stats/BEV export down, so feed_stats.json and
        # bev_data.json were never written. A non-numeric key is a known upstream
        # reality, not a defect to crash on: skip it, and say that it was skipped.
        numeric_floors = {}
        unparseable = []
        for k in data["maps"]:
            try:
                numeric_floors[float(k)] = k
            except (TypeError, ValueError):
                unparseable.append(k)
        if unparseable:
            bev_errors.append(
                f"bev payload 'maps' has non-numeric floor keys, ignored: {unparseable}"
            )

        if not numeric_floors:
            bev_errors.append(
                "bev payload 'maps' has no numeric floor keys; the map returned is not "
                "floor-selected"
            )
        else:
            # A bad floor_y was already rejected as a 400 above, and agent["z"] is the
            # height in the ROS frame the feed host publishes.
            target = None
            if str(floor_y).lower() != "auto":
                target = float(floor_y)
            elif data.get("agent") and data["agent"].get("z") is not None:
                target = float(data["agent"]["z"])
            if target is not None:
                nearest = min(numeric_floors, key=lambda f: abs(f - target))
                data["map"] = data["maps"][numeric_floors[nearest]]
    elif data and floor_y and str(floor_y).lower() != "auto":
        # The caller asked for a floor and the payload cannot honour it. Say so rather
        # than returning another floor's map as though it were the requested one.
        bev_errors.append(
            f"floor_y={floor_y} requested, but the payload carries no 'maps' dict "
            f"(keys: {sorted(data)}); the map returned is not floor-selected"
        )

    # Rule 6: a new key. Readers that do not look for it are unaffected.
    if bev_errors:
        data["source_errors"] = bev_errors

    # The top-down maps are static per scene per floor, but they were shipped as
    # base64 data URIs inside a payload the viewer polls twice a second -- tens of
    # KB/s to re-send an image that never changes. Replace them with ids and serve
    # the bytes once from /bev_map/{id}, where the browser can cache them properly.
    data = _externalise_maps(data)

    # Attach per-model perception latencies
    # These two loops fed the STAGE/MODEL LATENCY table and the step/distance readout,
    # and both used to swallow their failure with `except Exception: pass`. An
    # unreadable or half-written latency file therefore produced an EMPTY TABLE that
    # looked exactly like a run with no latencies recorded -- the reader could not tell
    # "nothing measured" from "we could not read what was measured". Rule 14.
    _FALLBACKS = [_active_output_dir(), Path("/out"), Path("/tmp/graphapi_live"), Path("/tmp")]
    lat_file = _pick_run_file(_FALLBACKS, "perception_latencies.json", bev_errors)
    if lat_file is not None:
        try:
            data["latencies"] = json.loads(lat_file.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            bev_errors.append(f"{lat_file}: {type(exc).__name__}: {exc}")
    # The measured frame period, median of the gaps between the last annotated frames. None
    # until two frames have arrived; the page prints a dash, never a guess.
    times = list(getattr(_node, "frame_times", []) or [])
    if len(times) >= 2:
        gaps = sorted(b - a for a, b in zip(times, times[1:]))
        lat = data.setdefault("latencies", {})
        lat["frame_period_s"] = round(gaps[len(gaps) // 2], 2)
        lat["frame_period_n"] = len(gaps)

    # Attach step & navigation stats
    if "stats" not in data or not data["stats"]:
        stats_file = _pick_run_file(_FALLBACKS, "feed_stats.json", bev_errors)
        if stats_file is not None:
            try:
                data["stats"] = json.loads(stats_file.read_text())
            except (OSError, json.JSONDecodeError) as exc:
                bev_errors.append(f"{stats_file}: {type(exc).__name__}: {exc}")

    # Re-attach: the reads above may have added reasons after the earlier assignment.
    if bev_errors:
        data["source_errors"] = bev_errors

    return JSONResponse(content=data)

# Incremental line count over the appended latency series: the file only grows,
# so count the newlines in the bytes added since last time rather than re-reading
# it on every 2 s /health poll.
_CYCLE_TALLY = {"path": None, "offset": 0, "count": 0}


def _cycle_seq():
    # GA-103: /tmp is in this list, so an earlier run's series could be counted as this
    # one's cycle number. _pick_run_file rejects a fallback file that predates the run.
    picked = _pick_run_file(
        (_active_output_dir(), _active_output_dir().parent, Path("/tmp")),
        "perception_latencies.jsonl")
    for path in ([picked] if picked is not None else []):
        if not path.exists():
            continue
        try:
            size = path.stat().st_size
        except OSError:
            continue
        if _CYCLE_TALLY["path"] != str(path) or size < _CYCLE_TALLY["offset"]:
            _CYCLE_TALLY.update(path=str(path), offset=0, count=0)  # new run, or truncated
        if size > _CYCLE_TALLY["offset"]:
            try:
                with open(path, "rb") as f:
                    f.seek(_CYCLE_TALLY["offset"])
                    chunk = f.read(size - _CYCLE_TALLY["offset"])
                _CYCLE_TALLY["count"] += chunk.count(b"\n")
                _CYCLE_TALLY["offset"] = size
            except OSError:
                pass
        return _CYCLE_TALLY["count"] or None
    return None


def _stamp():
    """Per-panel provenance: the graph, the metrics and the feed are produced by
    different processes at different rates and could each show a different cycle
    with nothing on screen saying so. These are the three ages the viewer needs to
    say which moment each panel is from."""
    now = time.time()
    out = {"now": now, "cycle": _cycle_seq(),
           "perception_at": None, "graph_at": None, "frame_at": None}
    # GA-103: same guard -- a stale /tmp copy must not date this run's perception.
    lat = _pick_run_file(
        (_active_output_dir(), _active_output_dir().parent, Path("/tmp")),
        "perception_latencies.json")
    if lat is not None:
        try:
            out["perception_at"] = json.loads(lat.read_text()).get("last_updated")
        except (OSError, json.JSONDecodeError):
            pass
    for base in (_active_output_dir(), _active_output_dir().parent):
        pp = base / "persistent_perception.json"
        if pp.exists():
            try:
                out["graph_at"] = pp.stat().st_mtime
            except OSError:
                pass
            break
    node = get_node()
    if node and getattr(node, "last_frame_time", 0):
        out["frame_at"] = node.last_frame_time
    out["frame_source"] = _last_frame_source
    out["frame_source_counts"] = dict(_FRAME_SOURCE_COUNTS)
    out["overlay"] = {"on": OVERLAY_ON, "pose_source": _overlay_pose_source,
                      "objects": len(_OVERLAY_OBJ["objects"]),
                      "failures": _OVERLAY_FAILURES["n"], "last_error": _OVERLAY_FAILURES["last"]}
    return out


# Which of our node scripts are running, from ONE /proc pass, cached briefly.
#
# /health used to do THREE separate full scans of /proc per call, reading every
# process's cmdline: 706 processes x 3 = ~870 ms per call on a poll that fires every
# 2 s -- 43% of a core spent answering a health check. Under run load the call took
# longer than the poll interval, so polls piled up, the browser held all six of its
# connections to the origin waiting on them, and THE WHOLE DASHBOARD STOPPED UPDATING
# while every other endpoint answered in under 0.3 s. Measured, not guessed.
_PROC_CACHE = {"t": 0.0, "names": frozenset()}
_WATCHED_SCRIPTS = ("habitat_feed_node.py", "perception_2.py", "object_manager_6.py")


def _running_scripts(ttl: float = 30.0) -> frozenset:
    """Which watched node scripts are running. Cached, and honest about what counts.

    TTL is 30 s, not 5 s. When none of them are running there is no early exit, so the
    scan reads every process -- measured at 350 ms standalone and over 2 s on a loaded
    host. At a 5 s TTL against a 2 s health poll that was a multi-second stall every few
    seconds; the whole point of caching it was to stop that. Node liveness does not
    change meaningfully faster than 30 s.

    MATCHING: on argv entries, and never our own process. `name in cmd` over the raw
    cmdline counted ANY process merely mentioning the script -- a grep, an editor, a
    health probe's own command line. That is how a stopped pipeline could report itself
    running, which is the exact dishonesty the health panel is supposed to have stopped.
    """
    now = time.time()
    if now - _PROC_CACHE["t"] < ttl and _PROC_CACHE["t"]:
        return _PROC_CACHE["names"]
    own = str(os.getpid())
    found = set()
    for entry in Path("/proc").glob("[0-9]*/cmdline"):
        if entry.parent.name == own:
            continue          # our own command line mentions every watched name
        try:
            raw = entry.read_bytes()
        except OSError:
            continue          # the process exited between the glob and the read
        # cmdline is NUL-separated argv. Only argv[0] and argv[1] are considered: the
        # node is EXECUTED as the script (`python3 .../perception_2.py`), so the name
        # appears there. Scanning every argument instead matched anything that merely
        # passed the name along -- verified with a decoy process whose later arguments
        # named all three scripts, which the looser rule reported as all three running.
        argv = [a for a in raw.decode("utf-8", "ignore").split("\0") if a][:2]
        for arg in argv:
            base = arg.rsplit("/", 1)[-1]
            if base in _WATCHED_SCRIPTS:
                found.add(base)
        if len(found) == len(_WATCHED_SCRIPTS):
            break             # nothing left to learn from the remaining processes
    _PROC_CACHE.update(t=now, names=frozenset(found))
    return _PROC_CACHE["names"]


@app.get("/rtabmap_nodes")
def get_rtabmap_nodes(per_room: bool = False):
    """How much map rtabmap has actually built. READ-ONLY.

    The mapping tour rotates "until coverage is enough", but the rotating process cannot
    see the map, so the stop was approximated with a fixed rate. This is the feedback
    channel: poll it and stop when the node count stops rising. `nodes` is the number
    rtabmap has committed, so a saturating count means the current pose is adding
    nothing new.

    Rule 12: the database is opened `mode=ro` and never written. rtabmap holds it open
    for writing, so a read can legitimately fail -- that is reported, never smoothed into
    a zero. A count of 0 and "could not read the map" are different answers and a caller
    that stops rotating on the wrong one would stop on a failure.
    """
    out = _active_output_dir()
    candidates = [Path(os.environ.get("TIAGO_RTABMAP_DB", str(out / "rtabmap.db"))),
                  out / "rtabmap.db", out.parent / "rtabmap.db", Path("/tmp/rtabmap.db")]
    db = next((c for c in candidates if c.exists()), None)
    if db is None:
        return JSONResponse(
            status_code=503,
            content={"error": "no rtabmap database found",
                     "looked_in": [str(c) for c in candidates]})

    result = {"db": str(db), "nodes": None, "links": None,
              "newest_stamp": None, "age_s": None, "per_room": None, "error": None}
    try:
        conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True, timeout=2.0)
    except sqlite3.Error as exc:
        result["error"] = f"could not open the map: {type(exc).__name__}: {exc}"
        return JSONResponse(status_code=503, content=result)
    try:
        result["nodes"] = conn.execute("SELECT COUNT(*) FROM Node").fetchone()[0]
        result["links"] = conn.execute("SELECT COUNT(*) FROM Link").fetchone()[0]
        newest = conn.execute("SELECT MAX(stamp) FROM Node").fetchone()[0]
        if isinstance(newest, (int, float)):
            result["newest_stamp"] = newest
            result["age_s"] = round(time.time() - newest, 2)
        if per_room:
            result["per_room"] = _nodes_per_room(conn)
    except sqlite3.Error as exc:
        # A locked or half-written database is a real, expected condition while rtabmap
        # is running. Say so; do not return a count that was never read.
        result["error"] = f"could not read the map: {type(exc).__name__}: {exc}"
        conn.close()
        return JSONResponse(status_code=503, content=result)
    conn.close()
    return JSONResponse(content=result)


def _nodes_per_room(conn):
    """Node counts by room, or a reason why not. Cheap: one pass, point-in-polygon."""
    room_file = _active_output_dir() / "room.json"
    if not room_file.exists():
        return {"error": "room.json not available; nodes not attributed to rooms"}
    try:
        rooms = json.loads(room_file.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        return {"error": f"room.json unreadable: {type(exc).__name__}: {exc}"}
    rooms = rooms if isinstance(rooms, list) else (rooms.get("rooms") or [])
    polys = [(r.get("room_id"), r.get("polygon") or []) for r in rooms]
    polys = [(rid, poly) for rid, poly in polys if rid and len(poly) >= 3]
    if not polys:
        return {"error": "no room polygons recorded; nodes not attributed to rooms"}

    def inside(x, y, poly):
        hit = False
        n = len(poly)
        for i in range(n):
            x1, y1 = poly[i][0], poly[i][1]
            x2, y2 = poly[(i + 1) % n][0], poly[(i + 1) % n][1]
            if (y1 > y) != (y2 > y) and x < (x2 - x1) * (y - y1) / ((y2 - y1) or 1e-12) + x1:
                hit = not hit
        return hit

    counts = {rid: 0 for rid, _ in polys}
    counts["unassigned"] = 0
    for (blob,) in conn.execute("SELECT pose FROM Node WHERE pose IS NOT NULL"):
        if not blob or len(blob) != 48:
            continue
        v = struct.unpack("<12f", blob)
        x, y = v[3], v[7]        # translation column of the 3x4 transform
        for rid, poly in polys:
            if inside(x, y, poly):
                counts[rid] += 1
                break
        else:
            counts["unassigned"] += 1
    return counts


@app.get("/overlay_pose")
def get_overlay_pose(width: int = 0, height: int = 0, max_age: float = 90.0):
    """The camera pose the overlay draws with, and the view frustum it implies.

    ONE ROUTE FOR BOTH MODES, which is why it lives here and not in the dashboard. The replay
    server IMPORTS this module, so in replay it answers locally off the bundle's own files; in
    live mode the dashboard's catch-all proxies to it. A second implementation on the
    dashboard would be a second convention, and this file already records what the wrong
    convention costs (live_overlay's header: 1630 px).

    WHICH SOURCE ANSWERED IS PART OF THE ANSWER, never inferred by the caller. `source` is
    "tf" for the real per-frame pose and "newest detection record" for the recorded one, which
    is one perception cycle stale by construction. As of 2026-09-04 `_pose_from_tf` has never
    answered -- its own docstring says WRITTEN, NOT RUN -- so a caller that sees "tf" here is
    seeing something this project has not yet observed, and should say so.

    `max_age` exists so an ARCHIVED run can be replayed through this exact code path. A stale
    pose is still returned with `age_s` and `stale` set, and the caller must not print it as a
    live reading. The default is the live default: 90 s.

    The frustum is `live_overlay.frustum_rays`, in the CAMERA frame, at 1 m. Its shape does
    not depend on the resolution (see that function), so a stale calibration.json moves the
    pixel scale and not the drawing.
    """
    d = _active_output_dir()
    cal = d / "calibration.json"
    res_src = "calibration.json"
    if not width or not height:
        try:
            r = json.loads(cal.read_text())["resolution"]
            width, height = int(r["width"]), int(r["height"])
        except (OSError, ValueError, KeyError, TypeError):
            width, height, res_src = 0, 0, "NOT AVAILABLE (no calibration.json)"
    else:
        res_src = "caller"
    intr = _overlay_intrinsics(width, height) if width and height else None

    det = d / "detections.jsonl"
    try:
        age = time.time() - det.stat().st_mtime
    except OSError:
        age = None

    pose = _overlay_pose(max_age=max_age)
    out = {"output_dir": str(d), "resolution": [width, height], "resolution_source": res_src,
           "intrinsics": intr, "detections_age_s": age, "max_age_s": max_age,
           "tf_map_frame": OVERLAY_MAP_FRAME, "tf_cam_frame": OVERLAY_CAM_FRAME}
    if not pose:
        # NOT A POSE OF ZERO. An absent pose is absent, and the reason is named: a caller that
        # got {0,0,0} back would draw a camera at the origin and it would look like a reading.
        out.update(pose=None, why=(
            "no pose: TF did not answer and no detection record newer than "
            f"{max_age:.0f} s carries camera_position + camera_quat_xyzw"
            + (f" (detections.jsonl is {age:.0f} s old)" if age is not None else
               " (no detections.jsonl in this output directory)")))
        return JSONResponse(out)
    pos, quat, source = pose
    out.update(pose={"position": pos, "quat_xyzw": quat}, source=source,
               stale=bool(source != "tf" and age is not None and age > max_age),
               frustum_rays_cam=(_lo.frustum_rays(intr, width, height, 1.0) if intr else None),
               frustum_note=("corner rays in the CAMERA OPTICAL frame at 1 m; rotate by "
                             "quat_xyzw and add position. Same convention as the projection "
                             "in live_overlay.project_box."))
    return JSONResponse(out)


@app.get("/cycle_series")
def get_cycle_series(limit: int = 400):
    """The per-cycle perception series (GA-334's `perception_latencies.jsonl`), newest last.

    One row per completed cycle: `cycle`, `t`, `frame_id`, `n_detections`, `cycle_ms` (the
    whole cycle, the number to quote), `total_ms` (the detection sub-span) and `stages_ms`.
    `n` is the row count in the file, `rows` the last `limit` of them. Absent series -> rows
    [] and `path` null, never a fabricated series. Same file in live (the run's output dir) and
    in replay (the bundle), so the Metrics graph is one reader for both.
    """
    picked = _pick_run_file(
        (_active_output_dir(), _active_output_dir().parent, Path("/tmp")),
        "perception_latencies.jsonl")
    rows, n = [], 0
    if picked is not None and picked.exists():
        keep = ("cycle", "t", "frame_id", "n_detections", "cycle_ms", "total_ms", "stages_ms")
        with open(picked, "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                try:
                    r = json.loads(line)
                except ValueError:
                    continue
                n += 1
                rows.append({k: r.get(k) for k in keep})
        rows = rows[-max(1, limit):]
    return JSONResponse(content={"rows": rows, "n": n,
                                 "path": str(picked) if picked is not None else None})


@app.get("/health")
def get_pipeline_health():
    components = {}
    
    # 1. Feed Host / Node check (active if node receiving frames or HTTP port open)
    feed_active = False
    node = get_node()
    if node and node.latest_jpeg and (time.time() - node.last_frame_time) < 15.0:
        feed_active = True
    elif node and node.raw_jpeg and (time.time() - node.last_raw_time) < 15.0:
        feed_active = True
    elif _SVC_B.get("feed_enabled", True):
        try:
            with urllib.request.urlopen(f"{FEED_HOST}/bev_data", timeout=0.5):
                feed_active = True
        except (urllib.error.URLError, OSError, TimeoutError):
            feed_active = False      # unreachable is the answer, not an error to hide
    if not feed_active:
        fn_file = Path("/tmp/feed_node.log")
        fn_fresh = fn_file.exists() and (time.time() - fn_file.stat().st_mtime) < 30.0
        fn_running = "habitat_feed_node.py" in _running_scripts()
        if fn_fresh or fn_running:
            feed_active = True

    components["feed"] = {
        "name": "Habitat Feed",
        "active": feed_active,
        "details": "Active & Streaming" if feed_active else "Offline"
    }

    # GA-40. Served any way other than __main__ there is no node and every ROS endpoint
    # 503s; the panel used to show green regardless.
    bridge_up = get_node() is not None
    components["bridge"] = {"name": "ROS2 Bridge", "active": bridge_up,
                            "details": "Bridge Active" if bridge_up else "no ROS node"}

    try:
        p_file = Path("/tmp/perception.log")
        is_fresh = p_file.exists() and (time.time() - p_file.stat().st_mtime) < 30.0
        p_running = "perception_2.py" in _running_scripts()
        if is_fresh or p_running:
            components["perception"] = {"name": "Perception Pipeline", "active": True, "details": "Active"}
        else:
            components["perception"] = {"name": "Perception Pipeline", "active": False, "details": "Node Stopped"}
    except OSError as exc:
        # Only the filesystem probe can fail here; _running_scripts handles its own.
        # "we could not look" is not "it is stopped", so the detail says which it is.
        components["perception"] = {"name": "Perception Pipeline", "active": False,
                              "details": f"Unknown: could not read /tmp/perception.log ({type(exc).__name__})"}

    try:
        om_file = Path("/tmp/om6.log")
        om_fresh = om_file.exists() and (time.time() - om_file.stat().st_mtime) < 30.0
        om_running = "object_manager_6.py" in _running_scripts()
        if om_fresh or om_running:
            components["object_manager"] = {"name": "3D Object Manager", "active": True, "details": "Active"}
        else:
            components["object_manager"] = {"name": "3D Object Manager", "active": False, "details": "Node Stopped"}
    except OSError as exc:
        # Only the filesystem probe can fail here; _running_scripts handles its own.
        # "we could not look" is not "it is stopped", so the detail says which it is.
        components["object_manager"] = {"name": "3D Object Manager", "active": False,
                              "details": f"Unknown: could not read /tmp/om6.log ({type(exc).__name__})"}

    all_active = all(c["active"] for c in components.values())
    return {
        "status": "ok" if all_active else "degraded",
        "all_active": all_active,
        "components": components,
        "stamp": _stamp(),
    }

@app.get("/logs")
def get_logs(lines: int = 200):
    output = []
    log_problems = []
    log_files = [
        ("feed", "/tmp/habitat_feed_host.log"),
        ("feed_node", "/tmp/feed_node.log"),
        ("perception", "/tmp/perception.log"),
        ("object_manager", "/tmp/om6.log"),
        ("bridge", "/tmp/bridge.log"),
        ("rtabmap", "/tmp/rtabmap.log"),
    ]
    for tag, filepath in log_files:
        p = _pick_run_file([_active_output_dir(), Path("/tmp")],
                           Path(filepath).name, log_problems)
        if p is not None:
            try:
                content = p.read_text().splitlines()[-lines:]
                for line in content:
                    if line.strip():
                        output.append(f"[{tag}] {line}")
            except OSError as exc:
                log_problems.append(f"{filepath}: {type(exc).__name__}: {exc}")

    if _SVC_B.get("feed_enabled", True):
        try:
            with urllib.request.urlopen(f"{FEED_HOST}/logs", timeout=1.5) as r:
                data = json.loads(r.read().decode())
                if isinstance(data, dict) and "logs" in data:
                    for line in data["logs"]:
                        if line.strip():
                            output.append(f"[feed] {line}")
        except (urllib.error.URLError, OSError, json.JSONDecodeError, TimeoutError) as exc:
            log_problems.append(f"{FEED_HOST}/logs: {type(exc).__name__}: {exc}")

    if not output:
        # This said "[bridge] System active and listening. Log stream initialized." -- an
        # assertion that the system was healthy, emitted exactly when NOTHING could be
        # read, in the panel a person opens to find out what went wrong. The same line was
        # removed from found/dashboard/server.py; THIS is the copy the 8082 viewer renders.
        output.append("[bridge] no log source could be read")
        output.extend(f"[bridge] {problem}" for problem in log_problems)
        if not log_problems:
            output.append("[bridge] every source was reachable and empty")

    return {"logs": output[-600:], "source_errors": log_problems or None}

@app.get("/auto_mode")
def set_auto_mode(enabled: str = "true"):
    try:
        with urllib.request.urlopen(f"{FEED_HOST}/auto_mode?enabled={enabled}", timeout=2.0) as r:
            return JSONResponse(content=json.loads(r.read().decode()))
    except (urllib.error.URLError, OSError, json.JSONDecodeError, TimeoutError) as e:
        # The feed host being unreachable is a reportable outcome. A TypeError in our own
        # request-building is not -- that must raise rather than be dressed up as a
        # control-surface failure someone will go and investigate at the simulator.
        return {"success": False, "error": f"{type(e).__name__}: {e}"}

@app.get("/action")
@app.get("/control")
def send_action(act: str = "", action: str = "", x: float = 0.0, y: float = 0.0, z: float = 0.0, amount: float = 0.0):
    action_name = act or action
    try:
        url = f"{FEED_HOST}/action?act={action_name}&x={x}&y={y}&z={z}&amount={amount}"
        with urllib.request.urlopen(url, timeout=2.0) as r:
            return JSONResponse(content=json.loads(r.read().decode()))
    except (urllib.error.URLError, OSError, json.JSONDecodeError, TimeoutError) as e:
        # The feed host being unreachable is a reportable outcome. A TypeError in our own
        # request-building is not -- that must raise rather than be dressed up as a
        # control-surface failure someone will go and investigate at the simulator.
        return {"success": False, "error": f"{type(e).__name__}: {e}"}

@app.get("/set_config")
def set_config(request: Request):
    """Forward viewer config toggles (perceive_while_moving, perm/temp/seg/det)
    to the feed host's control server; the host stores and echoes them."""
    try:
        url = f"{FEED_HOST}/set_config?{request.url.query}"
        with urllib.request.urlopen(url, timeout=2.0) as r:
            return JSONResponse(content=json.loads(r.read().decode()))
    except (urllib.error.URLError, OSError, json.JSONDecodeError, TimeoutError) as e:
        # The feed host being unreachable is a reportable outcome. A TypeError in our own
        # request-building is not -- that must raise rather than be dressed up as a
        # control-surface failure someone will go and investigate at the simulator.
        return {"success": False, "error": f"{type(e).__name__}: {e}"}

@app.get("/get_config")
def get_feed_config():
    try:
        with urllib.request.urlopen(f"{FEED_HOST}/get_config", timeout=2.0) as r:
            return JSONResponse(content=json.loads(r.read().decode()))
    except (urllib.error.URLError, OSError, json.JSONDecodeError, TimeoutError) as e:
        # The feed host being unreachable is a reportable outcome. A TypeError in our own
        # request-building is not -- that must raise rather than be dressed up as a
        # control-surface failure someone will go and investigate at the simulator.
        return {"success": False, "error": f"{type(e).__name__}: {e}"}

if __name__ == "__main__":
    rclpy.init()
    _node = BridgeNode()
    threading.Thread(target=rclpy.spin, args=(_node,), daemon=True).start()
    # 8081: host port 8080 belongs to the dashboard server; the object
    # manager's GRAPH_API_BASE_URL default must match this.
    # GA-291. Loopback, not 0.0.0.0. The container runs --network=host, and another program
    # on this host holds <tailnet-ip>:8081; a wildcard bind collides with that (EADDRINUSE,
    # measured 2026-09-03, run 20260903_223859 stored nothing) while 127.0.0.1:8081 binds
    # beside it. Every client (om6, the feed host, the browser on this host) uses 127.0.0.1.
    uvicorn.run(app, host="127.0.0.1", port=int(os.environ.get("BRIDGE_PORT", "8081")))
