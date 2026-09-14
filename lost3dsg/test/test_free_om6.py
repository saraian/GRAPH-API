"""GA-01 / GA-08 / GA-09 / GA-47 / GA-26 in object_manager_6 and object_services:
 - an unreadable 2xx body is a FAILED call and a reply without object_id is REFUSED (GA-01);
 - association.exploration_frame_limit ends exploration, 0 disables it (GA-08);
 - consecutive Graph API failures are counted, logged at ERROR and end the run (GA-09);
 - a merge queues the keeper by id from the REAL merge_log shape, a deletion queues the
   neighbours of the removed object (GA-47);
 - a changed description refreshes the embedding, on the update path and the late path (GA-26)."""
import json
import os
import sys
from types import SimpleNamespace as NS

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src", "perception_module"))
import rosstub  # noqa: E402

rosstub.install()
import object_info  # noqa: E402
import object_manager_6 as om6  # noqa: E402
import object_services as osv  # noqa: E402
from world_model import wm  # noqa: E402

assert om6._room_id_for_topic(None) is None
assert om6._room_id_for_topic("room_1") == "room_1"
assert om6._room_id_for_topic(7) == "7"
assert om6._room_id_for_topic("   ") is None

BOX = dict(x_min=0.0, x_max=1.0, y_min=0.0, y_max=1.0, z_min=0.0, z_max=1.0)


def box(x, s=1.0):
    return dict(x_min=x, x_max=x + s, y_min=0.0, y_max=s, z_min=0.0, z_max=s)


class Log:
    def __init__(self):
        self.errors = []

    def error(self, m):
        self.errors.append(m)

    def warn(self, m):
        pass

    info = warn


def node():
    n = object.__new__(om6.ObjectManagerService)
    n._log = Log()
    n.get_logger = lambda: n._log
    n.object_services = NS(log_both=lambda *a, **k: None)
    n.graph_api_base_url, n.graph_api_timeout = "http://x", 1.0
    n._graph_api_strikes = 0
    n.room_manager = NS(current_room_id="room_1", init_room_node=lambda r: None)
    n.reeval = om6.load_hooks.__globals__["Reevaluation"]()
    n._reeval_last, n._reeval_debounced, n._reeval_fanout_capped = {}, 0, 0
    n.exits = 0
    n._flush_and_exit = lambda: setattr(n, "exits", n.exits + 1)
    return n


def fake_response(status, text):
    def _json():
        return json.loads(text)
    return NS(status_code=status, text=text, json=_json)


# ---------------------------------------------------------------- GA-01
n = node()
om6.requests.request = lambda **k: fake_response(200, "<html>not json</html>")
try:
    om6.ObjectManagerService._call_graph_api(n, "POST", "/objects", json_body={})
    raise AssertionError("an unreadable 2xx body must raise, not return {}")
except RuntimeError as e:
    assert "unreadable body" in str(e), e

om6.requests.request = lambda **k: fake_response(200, '{"success": true, "message": "no id"}')
wm.persistent_perceptions[:] = [NS(label="bed", object_id="bed", bbox=BOX)]
r = om6.ObjectManagerService.add_new_object(n, "bed", dict(BOX), "d", "c", "m")
assert r is None and any("no object_id" in m for m in n._log.errors), "a reply without object_id must be refused"

# ---------------------------------------------------------------- GA-09
n = node()
om6.GRAPH_API_MAX_STRIKES = 3


def boom(**k):
    raise om6.requests.ConnectionError("down")


om6.requests.request = boom
for i in range(2):
    try:
        om6.ObjectManagerService._call_graph_api(n, "POST", "/merge")
    except RuntimeError:
        pass
assert n._graph_api_strikes == 2 and n.exits == 0
assert any("strike 2/3" in m for m in n._log.errors), n._log.errors
om6.requests.request = lambda **k: fake_response(400, '{"detail": "refused"}')   # alive: resets
try:
    om6.ObjectManagerService._call_graph_api(n, "POST", "/merge")
except RuntimeError:
    pass
assert n._graph_api_strikes == 0, "a 4xx is the service answering; it must reset the count"
om6.requests.request = boom
for i in range(3):
    try:
        om6.ObjectManagerService._call_graph_api(n, "POST", "/merge")
    except RuntimeError:
        pass
assert n.exits == 1 and any("ENDING THE RUN" in m for m in n._log.errors), "3 consecutive failures must end the run"
om6.requests.request = lambda **k: fake_response(503, '{"detail": "ROS bridge non pronto"}')
n.exits, n._graph_api_strikes = 0, 0
for i in range(3):
    try:
        om6.ObjectManagerService._call_graph_api(n, "POST", "/merge")
    except RuntimeError:
        pass
assert n.exits == 1, "5xx counts as a failure"
om6.GRAPH_API_MAX_STRIKES = 0
n.exits, n._graph_api_strikes = 0, 0
for i in range(5):
    try:
        om6.ObjectManagerService._call_graph_api(n, "POST", "/merge")
    except RuntimeError:
        pass
assert n.exits == 0 and n._graph_api_strikes == 5, "0 = count only"

# ---------------------------------------------------------------- GA-47 merge: the REAL merge_log shape
n = node()
ATTR = dict(color="", material="", description="")
a = NS(label="chair", object_id="obj_a", bbox=box(0.0), **ATTR)
b = NS(label="chair", object_id="obj_b", bbox=box(0.5), **ATTR)
wm.persistent_perceptions[:] = [a, b]
merge_log = [{"keeper": "chair", "keeper_id": "obj_a", "discarded": "chair#2", "bbox_from_object_id": "obj_a"}]
om6.requests.request = lambda **k: fake_response(200, json.dumps({"success": True, "merged_count": 1, "merge_log": merge_log}))
assert om6.ObjectManagerService.merge_duplicate_objects(n) == 1  # returns the applied COUNT, not a bool
pending = dict(n.reeval.drain())
assert pending.get("obj_a") == "merged" and "obj_b" in pending, pending
# an older log without keeper_id (label under "keeper") must not raise either
n = node()
wm.persistent_perceptions[:] = [a, b]
merge_log = [{"keeper": "chair", "discarded": "chair#2", "bbox_from_object_id": "obj_a"}]
assert om6.ObjectManagerService.merge_duplicate_objects(n) == 1  # returns the applied COUNT, not a bool
assert dict(n.reeval.drain()).get("obj_a") == "merged"

# ---------------------------------------------------------------- GA-47 delete: neighbours of the removed object
osv.save_persistent_perceptions = lambda node: None
osv.publish_pov_volume = lambda *a, **k: None
osv.bbox_centroid_in_volume = lambda bbox, vol: True
osv.MAX_MISSES_BEFORE_DELETE = 1
osv.OPERATIONS_LOG = os.devnull
s = object.__new__(osv.ObjectServices)
s.get_logger = lambda: rosstub.Any()
s.room_manager = NS(scene_graph={})
s.decision_log = NS(write=lambda *a, **k: None)
s.tracking_step_counter, s.considered_volume_pub = 0, None
n = node()
s.on_object_removed = lambda obj: om6.ObjectManagerService._note_removed(n, obj)
gone = NS(label="lamp", object_id="obj_gone", bbox=box(0.0), room_id=None, **ATTR)
near = NS(label="table", object_id="obj_near", bbox=box(1.2), room_id=None, **ATTR)
far = NS(label="sofa", object_id="obj_far", bbox=box(9.0), room_id=None, **ATTR)
wm.persistent_perceptions[:] = [gone, near, far]
req = NS(pov_volume_flat=[0, 10, 0, 10, 0, 10], current_labels=["obj_near", "obj_far"], check_uncertain=False)
resp = osv.ObjectServices._cb_delete_unseen_objects(s, req, NS())
assert resp.deleted_count == 1 and gone not in wm.persistent_perceptions, resp.message
pending = dict(n.reeval.drain())
assert pending == {"obj_near": "neighbour obj_gone deleted"}, pending
# the DELETE route (_cb_remove_object) removes UNCERTAIN objects only: nothing in the map to re-examine
import inspect  # noqa: E402

assert "persistent_perceptions" not in inspect.getsource(osv.ObjectServices._cb_remove_object)

# ---------------------------------------------------------------- GA-26 embedding refresh
calls = []
osv.get_embedding = lambda model, text: calls.append(text) or [1.0]
om6.get_embedding = lambda model, text: calls.append(text) or [2.0]
s.log_both = s.log_operation = lambda *a, **k: None
s.room_manager = NS(current_room_id="room_1", scene_graph={"room_1": {"objects": []}},
                    update_room_geometry=lambda *a: None, assign_room_by_geometry=lambda b: "room_1")
s.db = NS(on_object_moved=lambda *a, **k: None, on_uncertain_added=lambda *a, **k: None)
s.uncertain_objects = []
o = object_info.Object("mug", None, box(0.0, 0.2), description="a mug", color="white", material="ceramic")
o.object_id, o.creation_time, o.embedding = "obj_mug", 0.0, "stale"
wm.persistent_perceptions[:] = [o]
req = NS(object_id="obj_mug", update_bbox=False, has_orientation=False, description_embedding=None,
         description="a mug", color="", material="", **box(0.0, 0.2))
osv.ObjectServices._cb_update_object(s, req, NS())
assert o.embedding == "stale" and calls == [], "same description: no refresh"
req.description = "a blue mug"
osv.ObjectServices._cb_update_object(s, req, NS())
assert o.description == "a blue mug" and o.embedding == [1.0] and calls == ["a blue mug"], (o.embedding, calls)

n = node()
n._n_late_applied = n._n_late_unmatched = 0
n._note_update = lambda *a, **k: None
late = NS(object_id="obj_l", label="lamp#1", description="unknown", color="unknown", material="unknown", shape="unknown",
          observations=[NS(frame_id="1_0", bbox_2d=[0, 0, 10, 10])], bbox={}, embedding=None)
wm.persistent_perceptions[:] = [late]
d = NS(label="lamp#1", origin_frame="1_0", origin_bbox_2d=[0, 0, 10, 10], description="a brass lamp",
       color="unknown", material="unknown", shape="unknown")
om6.ObjectManagerService._late_descriptions_callback(n, NS(descriptions=[d]))
assert late.description == "a brass lamp" and late.embedding == [2.0], (late.description, late.embedding)

# ---------------------------------------------------------------- GA-08 frame limit
def cycle(n, limit, frames):
    om6.EXPLORATION_FRAME_LIMIT = limit
    n.exploration_mode, n.robot_has_moved = True, False
    n.tracking_step_counter = n.exploration_frame_counter = 0
    n.last_room_check_time = 1e12
    n._vlm_status_counts, n.uncertain_objects, n.latest_bboxes = {}, [], {}
    n.room_manager = NS(current_room_id="room_1", assign_room_by_geometry=lambda b: "room_1",
                        update_current_room_semantics=lambda objs: None,
                        update_all_rooms_semantics=lambda objs: None)
    n.tracking_activated_pub = n.kb_add_pub = rosstub.Any()
    n.decision_log = NS(write=lambda *a, **k: None)
    n.filter_hook = NS(name="t", judge=lambda p: NS(admitted=False, outcome="refused", reason="t", annotation={}))
    wm.persistent_perceptions.clear()
    n.check_tracking_transition = lambda *a: (False, None, 0.0)
    n.delete_uncertain_objects = lambda pov: False
    n.merge_duplicate_objects = lambda: False
    n.publish_kb_facts = lambda objs: []
    n.publish_kb_relation_facts = lambda: []
    n.flush_scan_summary = lambda frame_id=None: None
    n._record_sighting = lambda o, t: None
    n.update_spatial_relations = lambda: None
    for f in ("publish_persistent_bboxes", "publish_persistent_centroids", "publish_uncertain_bboxes",
              "publish_uncertain_centroids", "save_uncertain_objects", "save_persistent_perceptions"):
        setattr(om6, f, lambda *a, **k: None)
    bx = NS(label="chair", has_orientation=False, has_bbox_2d=False, has_clip_embedding=False, **BOX)
    out = []
    for i in range(frames):
        req = NS(bboxes=NS(header=NS(stamp=NS(sec=1, nanosec=0)), fov_x_max=0, fov_y_max=0, fov_z_max=0, boxes=[bx]),
                 descriptions=NS(descriptions=[NS(label="chair", color="", material="", description="", crop_path="", status="ok")]))
        out.append(om6.ObjectManagerService.object_tracking_callback(n, req, NS()).tracking_mode_activated)
    return out


n = node()
assert cycle(n, 3, 4) == [False, False, True, False] and n.exploration_mode is False, "limit 3 -> tracking on frame 3"
assert n.exploration_frame_counter == 0 and n.tracking_step_counter == 2
n = node()
assert cycle(n, 0, 12) == [False] * 12 and n.exploration_mode is True, "0 = off"

print("OK GA-01 GA-08 GA-09 GA-47 GA-26: unreadable body refused, frame limit honoured, strikes end the run, "
      "merge keeper + deletion neighbours queued, embeddings refreshed")
