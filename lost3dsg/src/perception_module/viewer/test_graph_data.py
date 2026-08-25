"""Self-check: /graph_data formats real output JSONs into Cytoscape elements."""
import json
import os
import sys
import tempfile
import types

# Stub ROS/CV deps so the bridge module imports outside the container
for name in ["cv2", "numpy", "rclpy", "uvicorn", "rclpy.node", "sensor_msgs", "sensor_msgs.msg", "lost3dsg", "lost3dsg.srv"]:
    mod = types.ModuleType(name)
    sys.modules.setdefault(name, mod)
sys.modules["rclpy.node"].Node = object
sys.modules["sensor_msgs.msg"].Image = object
for cls in ["AddObject", "RemoveObject", "UpdateObject", "MergeObjects", "DeleteObjects", "QueryObjects"]:
    setattr(sys.modules["lost3dsg.srv"], cls, type(cls, (), {}))

with tempfile.TemporaryDirectory() as tmp_old:
    old_objs = [
        {"label": "table#1", "room_id": "room_0",
         "bbox": {"x_min": 0.0, "x_max": 1.0, "y_min": 0.0, "y_max": 1.0, "z_min": 0.0, "z_max": 0.8}},
        {"label": "laptop", "room_id": "room_0",
         "bbox": {"x_min": 0.2, "x_max": 0.6, "y_min": 0.2, "y_max": 0.6, "z_min": 0.8, "z_max": 1.0}},
    ]
    old_rooms = [{"room_id": "room_0", "semantic_label": "Office", "objects": ["table#1", "laptop"]}]
    with open(f"{tmp_old}/persistent_perception.json", "w") as f:
        json.dump(old_objs, f)
    with open(f"{tmp_old}/room.json", "w") as f:
        json.dump(old_rooms, f)

    os.environ["GRAPH_API_OUTPUT_DIR"] = tmp_old
    sys.path.insert(0, "/DATA/GRAPH-API/lost3dsg/src/perception_module")
    import graph_api_bridge as bridge  # noqa: E402  (needs sys.path + env set first)

    res = bridge.graph_data()
    nodes, edges = res["nodes"], res["edges"]
    assert res["elements"]["nodes"], "no cytoscape nodes"
    assert any(n["type"] == "room" for n in nodes), "no room node"
    assert any(n["type"] == "object" for n in nodes), "no object nodes"
    loc = [e for e in edges if e["label"] == "isLocatedIn"]
    assert loc, "no isLocatedIn edges"
    node_ids = {n["id"] for n in nodes}
    for e in edges:
        assert e["source"] in node_ids and e["target"] in node_ids, f"dangling edge {e}"
        assert "#" not in e["source"] + e["target"], "unsanitized id in edge"
    sup = [e for e in edges if e["label"] == "supports"]
    assert sup, "no supports edge in heuristic fallback"
    print(f"OK (old format): {len(nodes)} nodes, {len(loc)} isLocatedIn, {len(sup)} supports")

# New-format file: object_id + relations written by save_persistent_perceptions
with tempfile.TemporaryDirectory() as tmp:
    objs = [
        {"object_id": "obj_a", "label": "desk", "room_id": "room_0", "bbox": {},
         "relations": {"isOn": [], "isNextTo": ["obj_b"], "isIn": [], "isAbove": [], "isUnder": []}},
        {"object_id": "obj_b", "label": "lamp#1", "room_id": "room_0", "bbox": {},
         "relations": {"isOn": ["obj_a"], "isNextTo": ["obj_a"], "isIn": [], "isAbove": [], "isUnder": []}},
    ]
    rooms = [{"room_id": "room_0", "semantic_label": "Office", "objects": ["desk", "lamp#1"]}]
    with open(f"{tmp}/persistent_perception.json", "w") as f:
        json.dump(objs, f)
    with open(f"{tmp}/room.json", "w") as f:
        json.dump(rooms, f)
    os.environ["GRAPH_API_OUTPUT_DIR"] = tmp
    res2 = bridge.graph_data()
    labels = sorted(e["label"] for e in res2["edges"])
    assert labels == ["isLocatedIn", "isLocatedIn", "isNextTo", "isOn"], labels  # isNextTo deduped
    ids = {n["id"] for n in res2["nodes"]}
    assert "n_obj_a" in ids and "n_obj_b" in ids, ids
    assert not any(e["label"] == "supports" for e in res2["edges"]), "heuristic ran despite real relations"
    print(f"OK (new format): edges {labels}")
