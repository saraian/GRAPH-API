"""Counter-probe: does the hypothesis reset its streak when it SEES the contrary sweep?
Run A: geometry gate active (the change set).  Run B: gate bypassed on sweep 2 only
(LOCALITY_GAP_M=99 m), which is what the baseline 16880b0 evidence arm did (no gate).
Run C: room gate refusing sweep 2 (rooms differ, fused boxes apart)."""
import contextlib
import io
import sys

PM = '/DATA/GRAPH-API/.claude/worktrees/found-merge-algorithm/lost3dsg/src/perception_module'
VERIFY = '/DATA/GRAPH-API/.claude/worktrees/found-merge-algorithm/.handoff/lanes/merge-algorithm/verify'
sys.path.insert(0, PM)
import rosstub  # noqa: E402

rosstub.install()
import object_info  # noqa: E402
import object_services  # noqa: E402
import room_manager  # noqa: E402
from world_model import wm  # noqa: E402

object_services.save_persistent_perceptions = lambda node: None
object_services.publish_persistent_bboxes = lambda *a, **k: None
object_services.publish_persistent_centroids = lambda *a, **k: None
object_services.OPERATIONS_LOG = VERIFY + '/operations_probe.txt'
BOX = {"x_min": 0.0, "x_max": 1.0, "y_min": 0.0, "y_max": 1.0, "z_min": 0.0, "z_max": 1.0}
SHIFT = {**BOX, "x_min": 0.2, "x_max": 1.2}
APART = {**BOX, "x_min": 1.2, "x_max": 2.2}    # gap 0.2 m (passes geometry, fails "overlapping")
FARISH = {**BOX, "x_min": 1.5, "x_max": 2.5}   # gap 0.5 m > LOCALITY_GAP_M 0.3
ANCHOR = {"x_min": 20.0, "x_max": 21.0, "y_min": 0.0, "y_max": 1.0, "z_min": 0.0, "z_max": 1.0}
KEY = ("obj_a", "obj_b")


def make_svc(rows):
    svc = object_services.ObjectServices.__new__(object_services.ObjectServices)
    svc.get_logger = lambda: rosstub.Any()
    svc.log_both = lambda *a, **k: None
    svc.room_manager = room_manager.RoomManager.__new__(room_manager.RoomManager)
    svc.room_manager.scene_graph = {}
    svc.room_manager.current_room_id = "room_1"
    svc.room_manager.room_at_bbox = lambda bbox: None
    svc.room_manager.update_room_geometry = lambda *a, **k: None
    svc._hypotheses, svc._merge_sweep, svc.tracking_step_counter = {}, 0, 0
    svc.persistent_bbox_pub = rosstub.Any()
    svc.persistent_centroids_pub = rosstub.Any()

    class _Log:
        def write(self, kind, oid, **kw):
            rows.append((kind, oid, kw))
    svc.decision_log = _Log()
    return svc


def obj(label, box, desc, color, material, oid, t):
    o = object_info.Object(label, None, dict(box), description=desc, color=color, material=material)
    o.object_id, o.creation_time = oid, t
    return o


def sweep(svc, objects, rows):
    wm.persistent_perceptions.clear()
    wm.persistent_perceptions.extend(objects)
    rows.clear()
    req, resp = rosstub.Any(), rosstub.Any()
    req.max_distance, req.min_similarity, req.dry_run = 0.8, 0.95, True
    with contextlib.redirect_stdout(io.StringIO()):
        object_services.ObjectServices._cb_merge_objects(svc, req, resp)
    assert resp.success is True, resp.message
    h = svc._hypotheses.get(KEY)
    ab = [(kw.get("reason"), kw.get("decision_reason")) for k, _, kw in rows
          if k == "merge_refused" and kw.get("candidate") == "obj_b"]
    merged = [kw.get("merged_from") for k, _, kw in rows if k == "merge"]
    return dict(merged=resp.merged_count, refusal=ab, merge_rows=merged,
                streak=None if h is None else h._streak,
                frames=None if h is None else [f["frame"] for f in h.history],
                totals=None if h is None else [f.get("total") for f in h.history])


def run(mode):
    rows = []
    svc = make_svc(rows)
    a = obj("chair#1", BOX, "dark grey tufted armchair", "grey", "fabric", "obj_a", 1.0)
    b = obj("chair#1", SHIFT, "dark grey armchair", "grey", "fabric", "obj_b", 2.0)
    anchor = obj("lamp", ANCHOR, "a lamp", "black", "metal", "obj_anchor", 3.0)
    out = [("sweep1 SHIFT", sweep(svc, [a, b, anchor], rows))]
    saved = object_services.LOCALITY_GAP_M
    if mode == "room":
        b.bbox = dict(APART)
        svc.room_manager.room_at_bbox = lambda bbox: "r_left" if bbox["x_min"] < 0.1 else "r_right"
    else:
        b.bbox = dict(FARISH)
        if mode == "bypass":
            object_services.LOCALITY_GAP_M = 99.0
    try:
        out.append(("sweep2 contrary", sweep(svc, [a, b, anchor], rows)))
    finally:
        object_services.LOCALITY_GAP_M = saved
        svc.room_manager.room_at_bbox = lambda bbox: None
    b.bbox = dict(SHIFT)
    out.append(("sweep3 SHIFT", sweep(svc, [a, b, anchor], rows)))
    if out[-1][1]["merged"] == 0:
        out.append(("sweep4 SHIFT", sweep(svc, [a, b, anchor], rows)))
    return out


print("engine", object_services.MERGE_ENGINE, "min_consecutive", object_services.MERGE_MIN_CONSECUTIVE,
      "LOCALITY_GAP_M", object_services.LOCALITY_GAP_M)
for name, mode in (("A: change set, geometry gate refuses sweep 2 (gap 0.5)", "gate"),
                   ("B: baseline emulation, gate bypassed on sweep 2 so h.update sees gap 0.5", "bypass"),
                   ("C: change set, room gate refuses sweep 2 (rooms differ, gap 0.2)", "room")):
    print("\n==", name)
    for label, r in run(mode):
        print("  ", label, r)
