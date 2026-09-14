"""Probe: does one stale pair leave merged_count, the `merge` decision rows, merge_log_json and
operations.txt in disagreement?  Same scaffold as test_perception_smoke.merge_lock_covers_writes_only,
but with a RECORDING decision_log (the smoke uses rosstub.Any(), which swallows the rows)."""
import json
import os
import pathlib
import sys
import tempfile

HERE = pathlib.Path("/DATA/GRAPH-API/.claude/worktrees/found-merge-algorithm/lost3dsg/src/perception_module")
sys.path.insert(0, str(HERE))
import rosstub  # noqa: E402

rosstub.install()
import object_info  # noqa: E402
import object_services as osv  # noqa: E402
import room_manager  # noqa: E402
from world_model import wm  # noqa: E402

# The row / merge_log / operations.txt writes sit BELOW the per-pair loop and are engine independent.
osv.MERGE_ENGINE = "legacy"
BOX = {"x_min": 0.0, "x_max": 1.0, "y_min": 0.0, "y_max": 1.0, "z_min": 0.0, "z_max": 1.0}


class RecLog:
    def __init__(self):
        self.rows = []

    def write(self, kind, oid, **kw):
        self.rows.append({"kind": kind, "object": oid, **kw})


def make_svc():
    svc = osv.ObjectServices.__new__(osv.ObjectServices)
    svc.get_logger = lambda: rosstub.Any()
    warns = []
    svc.log_both = lambda level, msg: warns.append((level, msg))
    svc.tracking_step_counter = 0
    svc.room_manager = room_manager.RoomManager.__new__(room_manager.RoomManager)
    svc.room_manager.scene_graph = {}
    svc.room_manager.current_room_id = "room_1"
    svc.room_manager.room_at_bbox = lambda bbox: None
    svc.room_manager.update_room_geometry = lambda *a, **k: None
    svc.decision_log = RecLog()
    svc.persistent_bbox_pub = rosstub.Any()
    svc.persistent_centroids_pub = rosstub.Any()
    return svc, warns


def obj(oid, ct, dx=0.0):
    box = dict(BOX)
    box["x_min"] += dx
    box["x_max"] += dx
    o = object_info.Object("chair", None, box, description="a chair", color="red", material="wood")
    o.object_id, o.creation_time = oid, ct
    return o


def sweep(svc, objects, remove_during_sweep=None):
    wm.persistent_perceptions.clear()
    wm.persistent_perceptions.extend(objects)
    real = osv.lost_similarity_detailed

    def probe(*a, **k):
        if remove_during_sweep is not None and remove_during_sweep in wm.persistent_perceptions:
            wm.persistent_perceptions.remove(remove_during_sweep)  # the race: a delete/update lands mid-sweep
        return real(*a, **k)

    osv.lost_similarity_detailed = probe
    req, resp = rosstub.Any(), rosstub.Any()
    req.max_distance, req.min_similarity, req.dry_run = 0.8, 0.95, False
    with tempfile.TemporaryDirectory() as tmp:
        osv.PROJECT_ROOT = tmp
        osv.OPERATIONS_LOG = os.path.join(tmp, "operations.txt")
        try:
            osv.ObjectServices._cb_merge_objects(svc, req, resp)
        finally:
            osv.lost_similarity_detailed = real
        ops = []
        if os.path.exists(osv.OPERATIONS_LOG):
            ops = [ln for ln in open(osv.OPERATIONS_LOG).read().splitlines() if "MERGE" in ln]
    return resp, ops


def om6_reader(result):
    """object_manager_6.merge_duplicate_objects, reduced to the part that queues keepers."""
    queued = []
    if int(result["merged_count"]):
        for pair in json.loads(result["merge_log_json"]):
            kid = pair.get("keeper_id") or pair.get("bbox_from_object_id")
            if kid:
                queued.append(kid)
    return queued


def report(name, svc, warns, resp, ops):
    rows = [r for r in svc.decision_log.rows if r["kind"] == "merge"]
    ml = json.loads(resp.merge_log_json)
    queued = om6_reader({"merged_count": resp.merged_count, "merge_log_json": resp.merge_log_json})
    print(f"--- {name} ---")
    print(f"merged_count={resp.merged_count}  merge_rows={len(rows)} dry_run={[r['dry_run'] for r in rows]}  "
          f"merge_log_json entries={len(ml)}  operations.txt MERGE lines={len(ops)}  "
          f"stale_warn={[m for _, m in warns if 'skipped' in m]}")
    print(f"rows (keeper, merged_from): {[(r['object'], r['merged_from']) for r in rows]}")
    print(f"merge_log keepers: {[e['keeper_id'] for e in ml]}   om6 would queue as 'merged': {queued}")
    print(f"map after: {[o.object_id for o in wm.persistent_perceptions]}")
    return rows, ml, queued


# A: the exact smoke race -- one pair, the discard leaves the map during the sweep
svc, warns = make_svc()
a, c = obj("obj_keep", 1.0), obj("obj_gone", 3.0)
resp, ops = sweep(svc, [a, c], remove_during_sweep=c)
rows, ml, queued = report("A: single pair, discard leaves mid-sweep", svc, warns, resp, ops)
assert resp.merged_count == 0 and len(rows) == 1 and rows[0]["dry_run"] is False
assert len(ml) == 1 and len(ops) == 1 and queued == []

# B: mixed sweep -- pair (a,b) applies, pair (d,c) goes stale; keepers differ (older wins merge_rank)
svc, warns = make_svc()
a, b = obj("obj_keep", 1.0), obj("obj_drop", 2.0)
d, c = obj("obj_keep2", 4.0, dx=10.0), obj("obj_gone", 5.0, dx=10.0)
resp, ops = sweep(svc, [a, b, d, c], remove_during_sweep=c)
rows, ml, queued = report("B: mixed sweep, one applied one stale", svc, warns, resp, ops)
assert resp.merged_count == 1 and len(rows) == 2 and len(ml) == 2 and len(ops) == 2
assert "obj_keep2" in queued, queued
print("PROBE: reproduced -- one sweep, four records, three of them count the stale pair")
