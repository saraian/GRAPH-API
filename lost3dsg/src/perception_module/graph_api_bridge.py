import json
import os
import threading
import time
import rclpy
import uvicorn
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from rclpy.node import Node

from lost3dsg.srv import (
    AddObject,
    RemoveObject,
    UpdateObject,
    MergeObjects,
    DeleteObjects,
    QueryObjects,
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
    configured = os.environ.get("LOST3DSG_OUTPUT_DIR")
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
VIEWER_DIR = _MODULE_DIR / "viewer"

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


if __name__ == "__main__":
    rclpy.init()
    _node = BridgeNode()
    threading.Thread(target=rclpy.spin, args=(_node,), daemon=True).start()
    uvicorn.run(app, host="0.0.0.0", port=8080)
