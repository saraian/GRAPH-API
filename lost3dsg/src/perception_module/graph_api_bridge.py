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

_DEFAULT_OUT = "/root/exchange/output"
PERSISTENT_PATH = Path(os.environ.get("GRAPH_API_OUTPUT_DIR", _DEFAULT_OUT)) / "persistent_perception.json"
ROOM_PATH = Path(os.environ.get("GRAPH_API_OUTPUT_DIR", _DEFAULT_OUT)) / "room.json"

_node = None

# resolve next to this file, not the CWD — the node is launched from anywhere
_VIEWER_DIR = Path(__file__).resolve().parent / "viewer"
app.mount("/viewer", StaticFiles(directory=str(_VIEWER_DIR)), name="viewer")

@app.get("/", include_in_schema=False)
def viewer():
    return FileResponse(str(_VIEWER_DIR / "viewer.html"))


@app.get("/persistent_perception")
def persistent_perception():
    if not PERSISTENT_PATH.exists():
        return []
    try:
        return json.loads(PERSISTENT_PATH.read_text())
    except (OSError, json.JSONDecodeError):
        return []


@app.get("/rooms")
def rooms():
    # Was also named `persistent_perception`, shadowing the /persistent_perception
    # handler above. Both routes worked (FastAPI binds the function object at
    # decoration time), but the module-level name pointed at this one only.
    if not ROOM_PATH.exists():
        return []
    try:
        return json.loads(ROOM_PATH.read_text())
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
    res = get_node().call('add', req)
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

    res = get_node().call('remove', req)
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

    res = get_node().call('update', req)
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

    res = get_node().call('merge', req)
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

    res = get_node().call('delete_objects', req)
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
