"""rule15-masking claim: cross-kind containment pair (same_kind False) escapes the GA-328 guard when
attributes + room exceed overlap. Driven through object_services._cb_merge_objects itself (rule 19),
under rosstub (description term dropped; non-identical label strings score 0.0)."""
import math
import sys

PM = '/DATA/GRAPH-API/.claude/worktrees/found-merge-algorithm/lost3dsg/src/perception_module'
VERIFY = '/DATA/GRAPH-API/.claude/worktrees/found-merge-algorithm/.handoff/lanes/merge-algorithm/verify'
sys.path.insert(0, PM)
import rosstub  # noqa: E402

rosstub.install()
import association as A  # noqa: E402
import object_info  # noqa: E402
import object_services  # noqa: E402
import room_manager  # noqa: E402
from world_model import wm  # noqa: E402

print("engine:", object_services.MERGE_ENGINE, "| ontology:", object_services.MERGE_ONTOLOGY_CHANNEL,
      "| attr cap:", object_services.MERGE_ATTRIBUTE_MAX_LOG_ODDS, "| min_consecutive:",
      object_services.MERGE_MIN_CONSECUTIVE, "| commit thr:",
      round(A.commit_threshold(object_services.MERGE_COST_RATIO), 4))
assert object_services.MERGE_ENGINE == "evidence"
object_services.save_persistent_perceptions = lambda node: None
object_services.publish_persistent_bboxes = lambda *a, **k: None
object_services.publish_persistent_centroids = lambda *a, **k: None
object_services.OPERATIONS_LOG = VERIFY + '/operations_probe.txt'


def make_svc(rows):
    svc = object_services.ObjectServices.__new__(object_services.ObjectServices)
    svc.get_logger = lambda: rosstub.Any()
    svc.log_both = lambda *a, **k: None
    svc.room_manager = room_manager.RoomManager.__new__(room_manager.RoomManager)
    svc.room_manager.scene_graph = {}
    svc.room_manager.current_room_id = "room_1"
    svc.room_manager.room_at_bbox = lambda bbox: None
    svc.room_manager.update_room_geometry = lambda *a, **k: None
    svc._hypotheses, svc._merge_sweep = {}, 0
    svc.tracking_step_counter = 0
    svc.persistent_bbox_pub = rosstub.Any()
    svc.persistent_centroids_pub = rosstub.Any()

    class _Log:
        def write(self, kind, oid, **kw):
            rows.append((kind, oid, kw))
    svc.decision_log = _Log()
    return svc


def box(x0, x1, y0, y1, z0, z1):
    return dict(x_min=x0, x_max=x1, y_min=y0, y_max=y1, z_min=z0, z_max=z1)


def obj(label, b, desc, color, material, oid, t, room):
    o = object_info.Object(label, None, dict(b), description=desc, color=color, material=material)
    o.object_id, o.creation_time, o.room_id = oid, t, room
    return o


def sweep(svc, objects, rows):
    wm.persistent_perceptions.clear()
    wm.persistent_perceptions.extend(objects)
    rows.clear()
    req, resp = rosstub.Any(), rosstub.Any()
    req.max_distance, req.min_similarity, req.dry_run = 0.8, 0.95, True
    object_services.ObjectServices._cb_merge_objects(svc, req, resp)
    assert resp.success is True, resp.message
    return resp


def pair_rows(rows):
    out = []
    for k, oid, kw in rows:
        if k == "merge_refused" and kw.get("candidate") in ("pillow", "bed") and oid in ("pillow", "bed"):
            ch = {n: round(c.get("log_odds", 0.0), 3) for n, c in (kw.get("channels") or {}).items()}
            out.append((oid, kw["candidate"], kw["reason"], round(kw.get("total_log_odds") or 0.0, 3), ch,
                        "same_kind=" + str((kw.get("channels") or {}).get("attributes", {}).get("same_kind")),
                        kw.get("decision_reason")))
        elif k == "merge":
            out.append(("MERGE", oid, kw.get("candidate"), kw.get("similarity")))
    return out


# map hull 10 x 10 x 3 = 300 m3 (a house); anchors give the room count
ANCH = [("lamp", box(9, 10, 0, 1, 0, 1), "black", "metal"),
        ("chair", box(0, 1, 9, 10, 0, 1), "brown", "wood"),
        ("light", box(0, 1, 0, 1, 2, 3), "white", "glass"),
        ("sink", box(9, 10, 9, 10, 0, 1), "white", "ceramic"),
        ("door", box(9, 10, 9, 10, 2, 3), "brown", "wood")]


def anchors(rooms):
    return [obj(lab, b, "a " + lab, c, m, f"anc{i}", 3.0, rooms[i]) for i, (lab, b, c, m) in enumerate(ANCH)]


BED = box(0, 2.0, 0, 1.6, 0, 0.6)
# pillow 0.6 x 0.4 x 0.2; the top of the bed box cuts it at 40 % (the guard's own docstring measured pillow#3+bed
# at containment 0.417) or at 85 % (pillow#1+bed 0.850)
P40 = box(0.1, 0.7, 0.1, 0.5, 0.52, 0.72)
P85 = box(0.1, 0.7, 0.1, 0.5, 0.43, 0.63)


def run(title, bed_cm, pillow_cm, rooms, pillow_box):
    rows = []
    svc = make_svc(rows)
    bed = obj("bed", BED, "a bed", *bed_cm, "bed", 1.0, "room_1")
    pillow = obj("pillow", pillow_box, "a pillow", *pillow_cm, "pillow", 2.0, "room_1")
    objs = [bed, pillow] + anchors(rooms)
    print(f"\n--- {title}")
    r = None
    for s in (1, 2):
        r = sweep(svc, objs, rows)
        print(f"  sweep {s}: merged_count={r.merged_count} {pair_rows(rows)}")
    return r.merged_count


R1 = ["room_1"] * 5
R2 = ["room_2"] * 5
R4 = ["room_2", "room_3", "room_4", "room_2", "room_3"]
R6 = ["room_2", "room_3", "room_4", "room_5", "room_6"]
BF = ("blue", "fabric")
for name, pb in (("P40", P40), ("P85", P85)):
    c = A.containment_ratio(A._as_bounds(BED), A._as_bounds(pb))
    print(f"{name}: containment {c:.3f}; overlap channel log(300/1.92)*c = {math.log(300 / 1.92) * c:.3f}")

res = {}
res["agree 4 rooms c0.40"] = run("CROSS-KIND bed/pillow, both blue/fabric, 4 rooms (+1.386), containment 0.40", BF, BF, R4, P40)
res["agree 2 rooms c0.40"] = run("CROSS-KIND bed/pillow, both blue/fabric, 2 rooms (+0.693), containment 0.40", BF, BF, R2, P40)
res["agree 1 room  c0.40"] = run("CROSS-KIND bed/pillow, both blue/fabric, 1 room (room abstains), containment 0.40", BF, BF, R1, P40)
res["agree 4 rooms c0.85"] = run("CROSS-KIND bed/pillow, both blue/fabric, 4 rooms, containment 0.85", BF, BF, R4, P85)
res["agree 6 rooms c0.85"] = run("CROSS-KIND bed/pillow, both blue/fabric, 6 rooms (+1.792), containment 0.85", BF, BF, R6, P85)
res["colour differs 4 rooms c0.40"] = run("CROSS-KIND bed/pillow, blue/fabric vs white/fabric, 4 rooms, containment 0.40",
                                          BF, ("white", "fabric"), R4, P40)
res["attrs unknown 4 rooms c0.40"] = run("CROSS-KIND bed/pillow, attributes UNKNOWN both sides (channel abstains = "
                                         "pre-change arithmetic), 4 rooms, containment 0.40",
                                         ("unknown", "unknown"), ("unknown", "unknown"), R4, P40)

print("\nSUMMARY merged_count on sweep 2 (dry run, through _cb_merge_objects, MERGE_MIN_CONSECUTIVE=2):")
for k, v in res.items():
    print(f"  {k:32s} -> {v}")
