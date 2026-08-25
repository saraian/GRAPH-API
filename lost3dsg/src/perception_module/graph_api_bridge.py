import json
import os
import threading
import time
import urllib.parse
import urllib.request
from pathlib import Path

import cv2
import numpy as np
import rclpy
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from rclpy.node import Node
from sensor_msgs.msg import Image

from lost3dsg.srv import (
    AddObject,
    DeleteObjects,
    MergeObjects,
    QueryObjects,
    RemoveObject,
    UpdateObject,
)

app = FastAPI(title="Graph API")

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
        _PROJECT_ROOT / "output",
        _MODULE_DIR.parents[2] / "output" if len(_MODULE_DIR.parents) > 2 else _PROJECT_ROOT / "output",
    ]
    unique = []
    for candidate in candidates:
        candidate = candidate.resolve()
        if candidate not in unique:
            unique.append(candidate)

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

@app.get("/", include_in_schema=False)
def viewer():
    return FileResponse(str(VIEWER_DIR / "viewer.html"))


@app.get("/persistent_perception")
def persistent_perception():
    path = _active_output_dir() / "persistent_perception.json"
    if not path.exists():
        return []
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return []


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
        self.latest_jpeg = None
        self.last_frame_time = 0.0
        self.create_subscription(Image, '/camera/rgb/image_raw', self._on_raw_image, 10)
        self.create_subscription(Image, '/image_with_bb', self._on_annotated_image, 10)

    def _convert_to_jpeg(self, msg: Image) -> bytes:
        try:
            h, w = msg.height, msg.width
            if h <= 0 or w <= 0 or not msg.data:
                return None
            img_np = np.frombuffer(msg.data, dtype=np.uint8).reshape((h, w, 3))
            if msg.encoding in ('rgb8', 'RGB8'):
                img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
            _, jpeg = cv2.imencode('.jpg', img_np, [cv2.IMWRITE_JPEG_QUALITY, 80])
            return jpeg.tobytes()
        except Exception:
            return None

    def _on_annotated_image(self, msg: Image):
        jpeg = self._convert_to_jpeg(msg)
        if jpeg:
            self.latest_jpeg = jpeg
            self.last_frame_time = time.time()

    def _on_raw_image(self, msg: Image):
        if self.latest_jpeg is None or (time.time() - self.last_frame_time) > 2.0:
            jpeg = self._convert_to_jpeg(msg)
            if jpeg:
                self.latest_jpeg = jpeg
                self.last_frame_time = time.time()

    def call(self, key, req):
        client = self.cli[key]

        if not client.wait_for_service(timeout_sec=2.0):
            raise RuntimeError(f"Servizio '{key}' non disponibile")

        future = client.call_async(req)

        deadline = time.time() + 5.0
        while time.time() < deadline:
            if future.done():
                break
            time.sleep(0.05)

        if not future.done():
            raise RuntimeError(f"Timeout in attesa del servizio '{key}'")

        result = future.result()
        if result is None:
            exc = future.exception()
            if exc is not None:
                raise RuntimeError(f"Errore dal servizio '{key}': {exc}")
            raise RuntimeError(f"Nessuna risposta dal servizio '{key}'")

        return result


def get_node():
    return _node


def require_node():
    node = get_node()
    if node is None:
        raise HTTPException(status_code=503, detail="ROS bridge non pronto")
    return node


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
    req.description_embedding = [float(x) for x in body.get("description_embedding", [])]
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

    req.description_embedding = [
        float(x) for x in body.get("description_embedding", [])
    ]

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


@app.post("/merge")
def merge_objects(body: dict = None):
    body = body or {}
    req = MergeObjects.Request()
    req.max_distance = float(body.get("max_distance", 0.8))
    req.min_similarity = float(body.get("min_similarity", 0.75))
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


@app.get("/graph_data")
def graph_data():
    """Format persistent_perception.json + room.json into Cytoscape elements.

    Nodes: rooms + persistent objects. Edges: ``isLocatedIn`` (object -> its
    room) and ``supports`` (object B rests on top of object A, Y-up).
    """
    out = _active_output_dir()

    def _load(name):
        for path in [out / name, out.parent / name]:
            if not path.exists():
                continue
            try:
                data = json.loads(path.read_text())
                if isinstance(data, list) and data:
                    return data
            except (OSError, json.JSONDecodeError):
                pass
        return []

    objects = _load("persistent_perception.json")
    rooms = _load("room.json")

    # Load ontological hook admission decisions if logged
    decisions = {}
    for decisions_path in [out / "hook_decisions.jsonl", out.parent / "hook_decisions.jsonl"]:
        if decisions_path.exists():
            try:
                for line in decisions_path.read_text().splitlines():
                    if not line.strip():
                        continue
                    rec = json.loads(line)
                    obj_name = rec.get("object")
                    if obj_name:
                        decisions[obj_name] = rec
            except Exception:
                pass

    def _nid(label):
        # Cytoscape selectors choke on '#' etc. in ids ("sofa#1") — sanitize.
        return "n_" + "".join(c if c.isalnum() else "_" for c in str(label))

    nodes, edges = [], []
    known_room_ids = set()
    for r in rooms:
        rid = r.get("room_id", "room")
        nid = _nid(rid)
        known_room_ids.add(nid)
        nodes.append({
            "id": nid,
            "label": r.get("semantic_label") or rid,
            "type": "room",
            "room": rid,
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
        decision = decisions.get(label) or decisions.get(o.get("object_id"))
        crop_target = str(label).replace("#", "_").replace(" ", "_")
        nodes.append({
            "id": _oid(o),
            "label": label,
            "type": "object",
            "room": o.get("room_id") or "",
            "confidence": o.get("confidence", 1.0),
            "color": o.get("color", ""),
            "material": o.get("material", ""),
            "position": pos,
            "bbox": bbox,
            "status": "permanent" if o.get("object_id") else "temporary",
            "decision": decision,
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
        except Exception:
            pass

    d_path = out / "hook_decisions.jsonl"
    if d_path.exists():
        try:
            for line in d_path.read_text().splitlines():
                if not line.strip():
                    continue
                rec = json.loads(line)
                outcome = (rec.get("outcome") or "").lower()
                if outcome == "reject":
                    rejected.append(rec)
                elif outcome in ("abstain", "no_grounds"):
                    abstained.append(rec)
        except Exception:
            pass

    return {
        "elements": {
            "nodes": [{"data": n} for n in nodes],
            "edges": [{"data": e} for e in edges],
        },
        "nodes": nodes,
        "objects": [n for n in nodes if n.get("type") == "object"],
        "edges": edges,
        "admission_summary": {
            "admitted_count": len([n for n in nodes if n.get("type") == "object"]),
            "on_hold_count": len(on_hold),
            "rejected_count": len(rejected),
            "abstained_count": len(abstained),
            "on_hold": on_hold,
            "rejected": rejected,
            "abstained": abstained,
        }
    }


@app.get("/admission_audit")
def get_admission_audit():
    g = graph_data()
    return JSONResponse(content=g.get("admission_summary", {}))


@app.get("/crop/{target}")
@app.get("/crops/{target}")
def get_crop_image(target: str):
    clean_target = target.replace("#", "_").replace(" ", "_").lower().replace(".jpg", "")
    base_target = clean_target.split("_")[0]  # e.g., 'trash' from 'trash_can_1'

    for base in [
        _active_output_dir(),
        _PROJECT_ROOT / "output",
        _MODULE_DIR.parents[2] / "output" if len(_MODULE_DIR.parents) > 2 else _PROJECT_ROOT / "output",
        _MODULE_DIR.parents[1] / "output" if len(_MODULE_DIR.parents) > 1 else _PROJECT_ROOT / "output",
        Path("/ws/install/lost3dsg/output"),
        Path("/ws/output"),
        Path("/out"),
        Path("/tmp/graphapi_live"),
    ]:
        cdir = base / "cropped_images"
        if not cdir.exists():
            continue
        # 1. Exact match
        exact = cdir / f"{target}.jpg"
        if exact.exists():
            return FileResponse(str(exact), media_type="image/jpeg")
        # 2. Match with clean target in filename
        matches = sorted(
            [p for p in cdir.glob("*.jpg") if clean_target in p.name.lower()],
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        if matches:
            return FileResponse(str(matches[0]), media_type="image/jpeg")
        # 3. Fallback: match by base label prefix
        if base_target and len(base_target) > 2:
            prefix_matches = sorted(
                [p for p in cdir.glob("*.jpg") if base_target in p.name.lower()],
                key=lambda p: p.stat().st_mtime,
                reverse=True,
            )
            if prefix_matches:
                return FileResponse(str(prefix_matches[0]), media_type="image/jpeg")

    svg = f'<svg xmlns="http://www.w3.org/2000/svg" width="160" height="120"><rect width="100%" height="100%" fill="#0b1329"/><text x="50%" y="45%" fill="#38bdf8" font-size="12" font-weight="bold" text-anchor="middle">{clean_target.upper()}</text><text x="50%" y="65%" fill="#64748b" font-size="9" text-anchor="middle">3D Grounded Object</text></svg>'
    return Response(content=svg, media_type="image/svg+xml")


FEED_HOST = os.environ.get("FEED_HOST", "http://127.0.0.1:7790")

# Serve composite_*.jpg fallback frames only if this fresh — older ones are
# leftovers from past recordings and masquerade as a live feed.
COMPOSITE_MAX_AGE_SEC = 30.0


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
    node = get_node()
    if node and node.latest_jpeg:
        return Response(content=node.latest_jpeg, media_type="image/jpeg")

    try:
        with urllib.request.urlopen(f"{FEED_HOST}/frame.jpg", timeout=1.0) as r:
            return Response(content=r.read(), media_type="image/jpeg")
    except Exception:
        pass

    frame = _latest_fresh_composite()
    if frame:
        return Response(content=frame, media_type="image/jpeg")
    return Response(status_code=503)

@app.get("/feed")
@app.get("/feed.mjpg")
def proxy_feed():
    def iterfile():
        # Try direct HTTP stream first
        try:
            req = urllib.request.urlopen(f"{FEED_HOST}/feed.mjpg", timeout=1.0)
            while True:
                chunk = req.read(4096)
                if not chunk:
                    break
                yield chunk
            return
        except Exception:
            pass

        # Primary: stream live ROS camera/perception frames in real time from node memory
        while True:
            node = get_node()
            frame_data = node.latest_jpeg if (node and node.latest_jpeg) else None

            if not frame_data:
                frame_data = _latest_fresh_composite()

            if frame_data:
                header = b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: " + str(len(frame_data)).encode() + b"\r\n\r\n"
                yield header + frame_data + b"\r\n"
            else:
                svg = b'--frame\r\nContent-Type: image/svg+xml\r\n\r\n<svg xmlns="http://www.w3.org/2000/svg" width="640" height="480"><rect width="100%" height="100%" fill="#090d16"/><text x="50%" y="50%" fill="#38bdf8" font-size="20" font-weight="bold" text-anchor="middle">CONNECTING TO LIVE ROS STREAM...</text></svg>\r\n'
                yield svg
            time.sleep(0.1)

    return StreamingResponse(iterfile(), media_type="multipart/x-mixed-replace; boundary=frame")

@app.get("/bev_data")
def proxy_bev_data(floor_y: str = None):
    data = {}

    for bev_dir in [_active_output_dir(), Path("/out"), Path("/tmp/graphapi_live"), Path("/tmp")]:
        bev_file = bev_dir / "bev_data.json"
        if bev_file.exists():
            try:
                data = json.loads(bev_file.read_text())
                if data and data.get("agent"):
                    break
            except Exception:
                pass

    if not data or not data.get("agent"):
        try:
            url = f"{FEED_HOST}/bev_data"
            if floor_y:
                url += f"?floor_y={floor_y}"
            with urllib.request.urlopen(url, timeout=0.5) as r:
                data = json.loads(r.read().decode())
        except Exception:
            pass

    if not data:
        data = {"agent": None, "floors": [-2.5, 0.5], "navmesh": [], "map": None}

    # Attach per-model perception latencies
    for lat_dir in [_active_output_dir(), Path("/out"), Path("/tmp/graphapi_live"), Path("/tmp")]:
        lat_file = lat_dir / "perception_latencies.json"
        if lat_file.exists():
            try:
                data["latencies"] = json.loads(lat_file.read_text())
                break
            except Exception:
                pass

    # Attach step & navigation stats
    if "stats" not in data or not data["stats"]:
        for stats_dir in [_active_output_dir(), Path("/out"), Path("/tmp/graphapi_live"), Path("/tmp")]:
            stats_file = stats_dir / "feed_stats.json"
            if stats_file.exists():
                try:
                    data["stats"] = json.loads(stats_file.read_text())
                    break
                except Exception:
                    pass

    return JSONResponse(content=data)

@app.get("/health")
def get_pipeline_health():
    components = {}
    
    # 1. Feed Host / Node check (active if node receiving frames or HTTP port open)
    feed_active = False
    node = get_node()
    if node and node.latest_jpeg and (time.time() - node.last_frame_time) < 15.0:
        feed_active = True
    else:
        try:
            with urllib.request.urlopen(f"{FEED_HOST}/bev_data", timeout=0.5):
                feed_active = True
        except Exception:
            pass
    if not feed_active:
        fn_file = Path("/tmp/feed_node.log")
        fn_fresh = fn_file.exists() and (time.time() - fn_file.stat().st_mtime) < 30.0
        fn_running = any("habitat_feed_node.py" in p.read_text(errors="ignore") for p in Path("/proc").glob("[0-9]*/cmdline"))
        if fn_fresh or fn_running:
            feed_active = True

    components["feed"] = {
        "name": "Habitat Feed",
        "active": feed_active,
        "details": "Active & Streaming" if feed_active else "Offline"
    }

    components["bridge"] = {"name": "ROS2 Bridge", "active": True, "details": "Bridge Active"}

    try:
        p_file = Path("/tmp/perception.log")
        is_fresh = p_file.exists() and (time.time() - p_file.stat().st_mtime) < 30.0
        p_running = any("perception_2.py" in p.read_text(errors="ignore") for p in Path("/proc").glob("[0-9]*/cmdline"))
        if is_fresh or p_running:
            components["perception"] = {"name": "Perception Pipeline", "active": True, "details": "Active"}
        else:
            components["perception"] = {"name": "Perception Pipeline", "active": False, "details": "Node Stopped"}
    except Exception:
        components["perception"] = {"name": "Perception Pipeline", "active": False, "details": "Unreachable"}

    try:
        om_file = Path("/tmp/om6.log")
        om_fresh = om_file.exists() and (time.time() - om_file.stat().st_mtime) < 30.0
        om_running = any("object_manager_6.py" in p.read_text(errors="ignore") for p in Path("/proc").glob("[0-9]*/cmdline"))
        if om_fresh or om_running:
            components["object_manager"] = {"name": "3D Object Manager", "active": True, "details": "Active"}
        else:
            components["object_manager"] = {"name": "3D Object Manager", "active": False, "details": "Node Stopped"}
    except Exception:
        components["object_manager"] = {"name": "3D Object Manager", "active": False, "details": "Unreachable"}

    all_active = all(c["active"] for c in components.values())
    return {
        "status": "ok" if all_active else "degraded",
        "all_active": all_active,
        "components": components
    }

@app.get("/logs")
def get_logs(lines: int = 200):
    output = []
    log_files = [
        ("feed", "/tmp/habitat_feed_host.log"),
        ("feed_node", "/tmp/feed_node.log"),
        ("perception", "/tmp/perception.log"),
        ("object_manager", "/tmp/om6.log"),
        ("bridge", "/tmp/bridge.log"),
    ]
    for tag, filepath in log_files:
        p = Path(filepath)
        if p.exists():
            try:
                content = p.read_text().splitlines()[-lines:]
                for line in content:
                    if line.strip():
                        output.append(f"[{tag}] {line}")
            except Exception:
                pass

    try:
        with urllib.request.urlopen(f"{FEED_HOST}/logs", timeout=1.5) as r:
            data = json.loads(r.read().decode())
            if isinstance(data, dict) and "logs" in data:
                for line in data["logs"]:
                    if line.strip():
                        output.append(f"[feed] {line}")
    except Exception:
        pass

    if not output:
        output.append("[bridge] System active and listening. Log stream initialized.")

    return {"logs": output[-600:]}

@app.get("/auto_mode")
def set_auto_mode(enabled: str = "true"):
    try:
        with urllib.request.urlopen(f"{FEED_HOST}/auto_mode?enabled={enabled}", timeout=2.0) as r:
            return JSONResponse(content=json.loads(r.read().decode()))
    except Exception as e:
        return {"success": False, "error": str(e)}

@app.get("/action")
@app.get("/control")
def send_action(act: str = "", action: str = "", x: float = 0.0, y: float = 0.0, z: float = 0.0, amount: float = 0.0):
    action_name = act or action
    try:
        url = f"{FEED_HOST}/action?act={action_name}&x={x}&y={y}&z={z}&amount={amount}"
        with urllib.request.urlopen(url, timeout=2.0) as r:
            return JSONResponse(content=json.loads(r.read().decode()))
    except Exception as e:
        return {"success": False, "error": str(e)}

@app.get("/set_config")
def set_config(request: Request):
    """Forward viewer config toggles (perceive_while_moving, perm/temp/seg/det)
    to the feed host's control server; the host stores and echoes them."""
    try:
        url = f"{FEED_HOST}/set_config?{request.url.query}"
        with urllib.request.urlopen(url, timeout=2.0) as r:
            return JSONResponse(content=json.loads(r.read().decode()))
    except Exception as e:
        return {"success": False, "error": str(e)}

@app.get("/get_config")
def get_feed_config():
    try:
        with urllib.request.urlopen(f"{FEED_HOST}/get_config", timeout=2.0) as r:
            return JSONResponse(content=json.loads(r.read().decode()))
    except Exception as e:
        return {"success": False, "error": str(e)}

if __name__ == "__main__":
    rclpy.init()
    _node = BridgeNode()
    threading.Thread(target=rclpy.spin, args=(_node,), daemon=True).start()
    # 8081: host port 8080 belongs to the FOUND dashboard server; the object
    # manager's GRAPH_API_BASE_URL default must match this.
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("BRIDGE_PORT", "8081")))
