"""Self-check: /graph_data formats real output JSONs into Cytoscape elements."""
import base64
import json
import os
import pathlib
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
    # Import the bridge NEXT TO THIS TEST, not a hardcoded absolute path. This line
    # used to read /DATA/GRAPH-API/... , so a copy of this file running inside the
    # vendored tree silently tested the other tree and asserted nothing about its own.
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
    import graph_api_bridge as bridge  # noqa: E402  (needs sys.path + env set first)
    assert pathlib.Path(bridge.__file__).resolve().parent == \
        pathlib.Path(__file__).resolve().parent.parent, \
        f"testing the wrong tree: imported {bridge.__file__}"

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

# --- room.json in the real room_manager shape (dict, not list) ---
with tempfile.TemporaryDirectory() as tmp3:
    objs = [{"object_id": "o1", "label": "bed", "room_id": "room_0", "bbox": {}}]
    # room_manager._save_rooms() payload shape
    detected = {"current_room_id": "room_0", "building": {"building_id": "building_0",
                "rooms": [{"room_id": "room_0", "semantic_label": "Bedroom",
                           "description": "d", "objects": ["bed"]}]}}
    with open(f"{tmp3}/persistent_perception.json", "w") as f:
        json.dump(objs, f)
    with open(f"{tmp3}/room.json", "w") as f:
        json.dump(detected, f)
    os.environ["GRAPH_API_OUTPUT_DIR"] = tmp3
    room = [n for n in bridge.graph_data()["nodes"] if n["type"] == "room"][0]
    assert room["label"] == "Bedroom", room          # detected type wins
    assert room["detected"] is True, room

    # placeholder must fall back to a prettified room_id, never "UnknownRoom"
    detected["building"]["rooms"][0]["semantic_label"] = "UnknownRoom"
    with open(f"{tmp3}/room.json", "w") as f:
        json.dump(detected, f)
    room = [n for n in bridge.graph_data()["nodes"] if n["type"] == "room"][0]
    assert room["label"] == "Room 0", room
    assert room["detected"] is False, room

    # vlm_status rides through so the UI can badge reachability
    detected["building"]["rooms"][0]["semantic_label"] = "home office"
    detected["building"]["rooms"][0]["vlm_status"] = {"ok": True, "model": "gemma4-31b"}
    with open(f"{tmp3}/room.json", "w") as f:
        json.dump(detected, f)
    room = [n for n in bridge.graph_data()["nodes"] if n["type"] == "room"][0]
    assert room["label"] == "home office", room
    assert room["vlm_status"]["ok"] is True, room
    print("OK (room.json dict shape): detected label + placeholder fallback + vlm_status")

# --- crop keyed by identity, cycle stamp, and absent confidence ---
with tempfile.TemporaryDirectory() as tmp4:
    os.makedirs(f"{tmp4}/cropped_images")
    objs = [
        {"object_id": "obj_aaa", "label": "desk", "room_id": "room_0", "bbox": {}},
        {"object_id": "obj_bbb", "label": "desk#1", "room_id": "room_0", "bbox": {}},
    ]
    with open(f"{tmp4}/persistent_perception.json", "w") as f:
        json.dump(objs, f)
    # object_manager_6 writes these `link` records to join decision -> object_id
    with open(f"{tmp4}/hook_decisions.jsonl", "w") as f:
        f.write(json.dumps({"kind": "link", "object": "obj_aaa",
                            "decision_id": "d1", "label": "desk"}) + "\n")
        f.write(json.dumps({"kind": "link", "object": "obj_bbb",
                            "decision_id": "d2", "label": "desk#1"}) + "\n")
    for name in ("crop_desk_20260826_090000_0.jpg", "crop_desk_1_20260826_090100_0.jpg"):
        with open(f"{tmp4}/cropped_images/{name}", "wb") as f:
            f.write(b"\xff\xd8\xff")  # minimal JPEG marker; content is never parsed
    os.environ["GRAPH_API_OUTPUT_DIR"] = tmp4
    bridge._CROP_DIRS = None          # directory list is resolved once and cached
    bridge._LINK_CACHE["key"] = None

    # crop_url must key on the object identity, not the label
    nodes = {n["id"]: n for n in bridge.graph_data()["nodes"]}
    assert nodes["n_obj_aaa"]["crop_url"] == "/crop/obj_aaa", nodes["n_obj_aaa"]
    # an absent confidence must stay absent — it used to default to a confident 1.0
    assert nodes["n_obj_aaa"]["confidence"] is None, nodes["n_obj_aaa"]

    # object_id resolves through the link record to that label's own crop...
    assert bridge._crop_file("obj_aaa").name == "crop_desk_20260826_090000_0.jpg"
    assert bridge._crop_file("obj_bbb").name == "crop_desk_1_20260826_090100_0.jpg"
    # ...and `desk` must NEVER collect `desk#1`'s crop, which the deleted prefix
    # fallback did: it served a confidently wrong image.
    assert bridge._crop_file("desk").name == "crop_desk_20260826_090000_0.jpg"
    # unknown identity gets no file (placeholder), not somebody else's photo
    assert bridge._crop_file("obj_zzz") is None
    print("OK (crop identity): object_id -> label -> exact crop, no prefix collision")

    # cycle sequence counts appended lines incrementally, and survives a new run
    with open(f"{tmp4}/perception_latencies.jsonl", "w") as f:
        f.write('{"total_ms": 1}\n{"total_ms": 2}\n')
    bridge._CYCLE_TALLY.update(path=None, offset=0, count=0)
    assert bridge._cycle_seq() == 2, bridge._cycle_seq()
    with open(f"{tmp4}/perception_latencies.jsonl", "a") as f:
        f.write('{"total_ms": 3}\n')
    assert bridge._cycle_seq() == 3, "appended cycle not counted"
    with open(f"{tmp4}/perception_latencies.jsonl", "w") as f:
        f.write('{"total_ms": 1}\n')          # truncated => new run, not a rewind
    assert bridge._cycle_seq() == 1, "truncation must reset the tally"

    with open(f"{tmp4}/perception_latencies.json", "w") as f:
        json.dump({"total_ms": 5, "last_updated": 1000.0}, f)
    st = bridge._stamp()
    assert st["perception_at"] == 1000.0 and st["graph_at"] is not None, st
    assert st["cycle"] == 1 and "frame_source" in st, st
    print("OK (cycle stamp): sequence, per-panel ages, frame source")

# --- BEV maps leave the 2 Hz poll and become cacheable URLs ---
_png = base64.b64encode(b"\x89PNG\r\n\x1a\nFAKE").decode()
payload = {
    "agent": {"x": 0, "y": 0, "z": 1.2},
    "map": {"image": f"data:image/png;base64,{_png}", "bounds_min": [0, 0, 0], "bounds_max": [1, 1, 1]},
    "maps": {"1.2": {"image": f"data:image/png;base64,{_png}", "bounds_min": [0, 0, 0], "bounds_max": [1, 1, 1]}},
}
out = bridge._externalise_maps(payload)
assert "image" not in out["map"], "base64 must not survive into the polled payload"
assert out["map"]["url"].startswith("/bev_map/"), out["map"]
assert out["map"]["bounds_min"] == [0, 0, 0], "bounds must survive"
assert out["maps"]["1.2"]["url"] == out["map"]["url"], "same bytes -> same content-addressed id"
map_id = out["map"]["url"].rsplit("/", 1)[1]
assert bridge._BEV_MAP_BYTES[map_id][1] == b"\x89PNG\r\n\x1a\nFAKE", "bytes must round-trip"
# a payload with no map, or a non-data URL, must pass through untouched
assert bridge._externalise_maps({"agent": None, "map": None})["map"] is None
assert bridge._externalise_maps({"map": {"image": "http://x/y.png"}})["map"]["image"] == "http://x/y.png"
print("OK (bev maps): base64 lifted to /bev_map/<hash>, bounds kept, non-data URLs untouched")

# --- /graph_data version: stat-only, changes exactly when a source file does ---
with tempfile.TemporaryDirectory() as tmp5:
    with open(f"{tmp5}/persistent_perception.json", "w") as f:
        json.dump([{"object_id": "o1", "label": "chair", "room_id": "room_0", "bbox": {}}], f)
    os.environ["GRAPH_API_OUTPUT_DIR"] = tmp5
    bridge._CROP_DIRS = None
    v1 = bridge.graph_data()["version"]
    assert v1 == bridge.graph_data()["version"], "unchanged files must give a stable version"
    with open(f"{tmp5}/persistent_perception.json", "w") as f:
        json.dump([{"object_id": "o1", "label": "chair", "room_id": "room_0", "bbox": {}},
                   {"object_id": "o2", "label": "lamp", "room_id": "room_0", "bbox": {}}], f)
    assert bridge.graph_data()["version"] != v1, "a changed world model must bump the version"
    print("OK (graph version): stable when idle, bumps on change")

# --- alignment comes from the aligner, never from the label ---
with tempfile.TemporaryDirectory() as tmp6:
    objs = [
        {"object_id": "o_al", "label": "chair", "room_id": "room_0", "bbox": {}},
        {"object_id": "o_ab", "label": "picture frame#1", "room_id": "room_0", "bbox": {}},
        {"object_id": "o_no", "label": "wombat", "room_id": "room_0", "bbox": {}},
    ]
    with open(f"{tmp6}/persistent_perception.json", "w") as f:
        json.dump(objs, f)
    with open(f"{tmp6}/hook_decisions.jsonl", "w") as f:
        # aligned
        f.write(json.dumps({"kind": "admission", "object": "chair", "outcome": "admit",
                            "annotation": {"entity": "soma:Chair",
                                           "alignment": {"score": 0.91, "status": "aligned",
                                                         "evidence": "top 0.91 z=4.1"}}}) + "\n")
        # real abstention, exactly as the 26 Aug bundle records it
        f.write(json.dumps({"kind": "admission", "object": "picture frame#1", "outcome": "admit",
                            "annotation": {"entity": None,
                                           "alignment": {"score": 0.821, "status": "unaligned",
                                                         "evidence": "best Wall at 0.82 below rule"}}}) + "\n")
        # a link record must never be mistaken for a decision
        f.write(json.dumps({"kind": "link", "object": "o_no",
                            "decision_id": "d9", "label": "wombat"}) + "\n")
    os.environ["GRAPH_API_OUTPUT_DIR"] = tmp6
    bridge._CROP_DIRS = None
    bridge._LINK_CACHE["key"] = None
    nodes = {n["id"]: n for n in bridge.graph_data()["nodes"]}

    al = nodes["n_o_al"]["alignment"]
    assert al["entity"] == "soma:Chair" and al["score"] == 0.91, al
    assert al["status"] == "aligned", al

    ab = nodes["n_o_ab"]["alignment"]
    assert ab["entity"] is None, "an abstention must not acquire a class"
    assert ab["status"] == "unaligned" and ab["score"] == 0.821, ab
    assert "below rule" in ab["evidence"], ab

    # No decision at all: no alignment, and NOT a fabricated PhysicalArtifact.
    assert nodes["n_o_no"]["alignment"] is None, nodes["n_o_no"]
    # ...and the link record must not have been picked up as its decision.
    assert nodes["n_o_no"]["decision"] is None, nodes["n_o_no"]["decision"]
    print("OK (alignment): real class, abstention stays classless, link records ignored")

# --- GA-38: grade on the aligner's verdict, not the top-level enforcement flag ---
# Measured over 16 archived bundles: 14 diverge, and every divergence is
# outcome="admit" sitting over a verdict of decline/hold/no_grounds. Grading on
# `outcome` therefore always over-reports admission. Bundle 20260826_024423 showed
# 0 rejected / 0 abstained and the message "All admitted" for 61 records of which
# 55 contradicted it.
with tempfile.TemporaryDirectory() as tmp7:
    objs = [{"object_id": f"o{i}", "label": f"thing{i}", "room_id": "room_0", "bbox": {}}
            for i in range(4)]
    with open(f"{tmp7}/persistent_perception.json", "w") as f:
        json.dump(objs, f)
    with open(f"{tmp7}/hook_decisions.jsonl", "w") as f:
        # the divergent shape: admit over a graded refusal
        f.write(json.dumps({"kind": "admission", "object": "thing0", "outcome": "admit",
                            "annotation": {"verdict": {"grade": "decline"}}}) + "\n")
        f.write(json.dumps({"kind": "admission", "object": "thing1", "outcome": "admit",
                            "annotation": {"verdict": {"grade": "no_grounds"}}}) + "\n")
        # 'hold' was classified nowhere before this fix
        f.write(json.dumps({"kind": "admission", "object": "thing2", "outcome": "admit",
                            "annotation": {"verdict": {"grade": "hold"}}}) + "\n")
        # no annotation at all: `outcome` must still classify it
        f.write(json.dumps({"kind": "admission", "object": "thing3",
                            "outcome": "reject"}) + "\n")
        # a truncated final line must be counted, never silently dropped
        f.write('{"kind": "admission", "object": "thing4", "outc')
    os.environ["GRAPH_API_OUTPUT_DIR"] = tmp7
    bridge._CROP_DIRS = None
    summary = bridge.graph_data()["admission_summary"]

    assert summary["rejected_count"] == 2, summary   # decline + the outcome-only reject
    assert summary["abstained_count"] == 2, summary  # no_grounds + hold
    assert summary["unreadable_records"] == 1, summary
    assert summary["error"] is None, summary
    print("OK (GA-38): grades on verdict, falls back to outcome, counts unreadable records")
