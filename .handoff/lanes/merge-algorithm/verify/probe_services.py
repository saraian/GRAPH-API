"""Probes against object_services._cb_merge_objects under rosstub (as test_perception_smoke does)."""
import json
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

print("engine:", object_services.MERGE_ENGINE, "| ontology channel:", object_services.MERGE_ONTOLOGY_CHANNEL,
      "| attr cap:", object_services.MERGE_ATTRIBUTE_MAX_LOG_ODDS, "| locality gap:", object_services.LOCALITY_GAP_M,
      "| min_consecutive:", object_services.MERGE_MIN_CONSECUTIVE, "| commit thr:",
      round(A.commit_threshold(object_services.MERGE_COST_RATIO), 4))
assert object_services.MERGE_ENGINE == "evidence"

# keep the live path from writing into the tree
object_services.save_persistent_perceptions = lambda node: None
object_services.publish_persistent_bboxes = lambda *a, **k: None
object_services.publish_persistent_centroids = lambda *a, **k: None
object_services.OPERATIONS_LOG = VERIFY + '/operations_probe.txt'

BOX = {"x_min": 0.0, "x_max": 1.0, "y_min": 0.0, "y_max": 1.0, "z_min": 0.0, "z_max": 1.0}
SHIFT = {**BOX, "x_min": 0.2, "x_max": 1.2}
APART = {**BOX, "x_min": 1.2, "x_max": 2.2}       # gap 0.2
FARISH = {**BOX, "x_min": 1.5, "x_max": 2.5}      # gap 0.5
ANCHOR = {"x_min": 20.0, "x_max": 21.0, "y_min": 0.0, "y_max": 1.0, "z_min": 0.0, "z_max": 1.0}


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


def obj(label, box, desc, color, material, oid, t):
    o = object_info.Object(label, None, dict(box), description=desc, color=color, material=material)
    o.object_id, o.creation_time = oid, t
    return o


def sweep(svc, objects, rows, dry_run=True, snapshot=None):
    wm.persistent_perceptions.clear()
    wm.persistent_perceptions.extend(objects)
    rows.clear()
    if snapshot is not None:
        wm.snapshot = lambda: list(snapshot)
    req, resp = rosstub.Any(), rosstub.Any()
    req.max_distance, req.min_similarity, req.dry_run = 0.8, 0.95, dry_run
    try:
        object_services.ObjectServices._cb_merge_objects(svc, req, resp)
    finally:
        if snapshot is not None:
            del wm.snapshot
    assert resp.success is True, resp.message
    return resp


def rows_of(rows, reason=None):
    return [(oid, kw.get("candidate"), kw.get("reason"), kw.get("decision_reason")) for k, oid, kw in rows
            if k == "merge_refused" and (reason is None or kw.get("reason") == reason)]


def fresh_pair():
    a = obj("chair#1", BOX, "dark grey tufted armchair with nailhead trim", "grey", "fabric", "obj_a", 1.0)
    b = obj("chair#1", SHIFT, "dark grey armchair in the foreground", "grey", "fabric", "obj_b", 2.0)
    anchor = obj("lamp", ANCHOR, "a lamp", "black", "metal", "obj_anchor", 3.0)
    return a, b, anchor


print("\n=== A. dry run: merged_count and keeper/discard ===")
rows = []
svc = make_svc(rows)
a, b, anchor = fresh_pair()
r1 = sweep(svc, [a, b, anchor], rows, dry_run=True)
print("  sweep1 merged_count", r1.merged_count, "msg", r1.message, "| refusals", rows_of(rows))
r2 = sweep(svc, [a, b, anchor], rows, dry_run=True)
ml = json.loads(r2.merge_log_json)
print("  sweep2 merged_count", r2.merged_count, "msg", r2.message, "| merge_log keeper/discarded",
      [(m["keeper_id"], m["discarded"]) for m in ml], "| map still has both?",
      a in wm.persistent_perceptions and b in wm.persistent_perceptions)
keeper_id = ml[0]["keeper_id"]
keeper, discard = (a, b) if keeper_id == "obj_a" else (b, a)
print("  keeper", keeper.object_id, "discard", discard.object_id)
print("  hypothesis committed flag after dry-run merge:", [(k, h.committed) for k, h in svc._hypotheses.items()])
r3 = sweep(svc, [a, b, anchor], rows, dry_run=True)
print("  sweep3 (dry, after committed hypothesis) merged_count", r3.merged_count, "| hypotheses now",
      list(svc._hypotheses), "| refusals", rows_of(rows))

print("\n=== B. live path: stale keeper / stale discard / both / control ===")
for case, missing in [("stale keeper", "keeper"), ("stale discard", "discard"), ("both stale", "both"),
                      ("control (both present)", None)]:
    rows = []
    svc = make_svc(rows)
    a, b, anchor = fresh_pair()
    keeper, discard = (a, b) if keeper_id == "obj_a" else (b, a)
    r1 = sweep(svc, [a, b, anchor], rows, dry_run=False)
    assert r1.merged_count == 0, r1.message
    present = {"keeper": [discard, anchor], "discard": [keeper, anchor], "both": [anchor], None: [a, b, anchor]}[missing]
    r2 = sweep(svc, present, rows, dry_run=False, snapshot=[a, b, anchor])
    print(f"  {case:24s} merged_count={r2.merged_count} msg={r2.message!r} merge rows written="
          f"{sum(1 for k, _, _ in rows if k == 'merge')} discard still in map="
          f"{discard in wm.persistent_perceptions} keeper in map={keeper in wm.persistent_perceptions}")
    # what happens on the NEXT sweep after a stale skip: is the hypothesis gone (committed=True at decision time)?
    if missing is not None:
        r3 = sweep(svc, [a, b, anchor], rows, dry_run=False)
        r4 = sweep(svc, [a, b, anchor], rows, dry_run=False)
        print(f"    next two live sweeps with both present: merged_count {r3.merged_count}, {r4.merged_count} "
              f"(hypothesis restarted after the unapplied commit? {r3.merged_count == 0})")

print("\n=== C. room gate with room_at_bbox None on one side (boxes APART, gap 0.2) ===")
rows = []
svc = make_svc(rows)
svc.room_manager.room_at_bbox = lambda bbox: None if bbox["x_min"] < 0.1 else "r_right"
a, _, anchor = fresh_pair()
c = obj("chair#1", APART, "dark grey armchair", "grey", "fabric", "obj_c", 2.0)
sweep(svc, [a, c, anchor], rows)
print("  room refusals:", rows_of(rows, "room"))
print("  all a/c rows:", [(kw.get("reason"), kw.get("room_a"), kw.get("room_b"), kw.get("decision_reason"))
                          for k, oid, kw in rows if k == "merge_refused" and {oid, kw.get("candidate")} == {"obj_a", "obj_c"}])
svc.room_manager.room_at_bbox = lambda bbox: "r_left" if bbox["x_min"] < 0.1 else None
rows.clear()
sweep(svc, [a, c, anchor], rows)
print("  reversed (None on the other side) room refusals:", rows_of(rows, "room"))

print("\n=== D. geometry refusal record keys (gap 0.5) ===")
rows = []
svc = make_svc(rows)
a, _, anchor = fresh_pair()
d = obj("chair#1", FARISH, "dark grey armchair", "grey", "fabric", "obj_d", 2.0)
sweep(svc, [a, d, anchor], rows)
g = [(oid, kw) for k, oid, kw in rows if k == "merge_refused" and kw.get("reason") == "geometry"]
for oid, kw in g:
    print("  object", oid, "keys:", sorted(kw))
    print("  values:", {k: v for k, v in kw.items() if k not in ("a_label", "b_label")})
print("  'evidence_count' present on geometry row?", any("evidence_count" in kw for _, kw in g))
room_rows = [kw for k, oid, kw in rows if k == "merge_refused" and kw.get("reason") == "room"]
print("  offered_by in pair_meta reaches the record?", any("offered_by" in kw for _, kw in g))

print("\n=== E. hypothesis across a sweep the GEOMETRY gate refuses (offered, never updated) ===")
rows = []
svc = make_svc(rows)
a, b, anchor = fresh_pair()
key = tuple(sorted(("obj_a", "obj_b")))
r1 = sweep(svc, [a, b, anchor], rows)
h = svc._hypotheses.get(key)
print("  sweep1 merged", r1.merged_count, "| streak", h._streak, "| frames", [f["frame"] for f in h.history])
b.bbox = dict(FARISH)
r2 = sweep(svc, [a, b, anchor], rows)
h = svc._hypotheses.get(key)
print("  sweep2 (b moved to gap 0.5) merged", r2.merged_count, "| a/b refusal", rows_of(rows)[:2],
      "| hypothesis kept?", h is not None, "| streak", None if h is None else h._streak,
      "| frames", None if h is None else [f["frame"] for f in h.history])
b.bbox = dict(SHIFT)
r3 = sweep(svc, [a, b, anchor], rows)
h = svc._hypotheses.get(key)
print("  sweep3 (b back) merged", r3.merged_count, "| decision rows", rows_of(rows),
      "| merge rows", [(kw.get("merged_from"), kw.get("similarity")) for k, _, kw in rows if k == "merge"])
print("  provenance frames:", None if h is None else [f["frame"] for f in h.history])
# same with the ROOM gate refusing the middle sweep
rows = []
svc = make_svc(rows)
a, b, anchor = fresh_pair()
r1 = sweep(svc, [a, b, anchor], rows)
b.bbox = dict(APART)
svc.room_manager.room_at_bbox = lambda bbox: "r_left" if bbox["x_min"] < 0.1 else "r_right"
r2 = sweep(svc, [a, b, anchor], rows)
print("  room-gate variant sweep2 refusals", rows_of(rows, "room"), "| hypothesis kept?", key in svc._hypotheses)
svc.room_manager.room_at_bbox = lambda bbox: None
b.bbox = dict(SHIFT)
r3 = sweep(svc, [a, b, anchor], rows)
print("  room-gate variant sweep3 merged", r3.merged_count)
# and the NOT-OFFERED variant: b far outside shell and gap -> gc drops it
rows = []
svc = make_svc(rows)
a, b, anchor = fresh_pair()
r1 = sweep(svc, [a, b, anchor], rows)
b.bbox = {**BOX, "x_min": 5.0, "x_max": 6.0}
r2 = sweep(svc, [a, b, anchor], rows)
print("  not-offered variant sweep2: hypothesis kept?", key in svc._hypotheses,
      "| not_offered rows", [kw for k, _, kw in rows if k == "not_offered_summary"])
b.bbox = dict(SHIFT)
r3 = sweep(svc, [a, b, anchor], rows)
print("  not-offered variant sweep3 merged", r3.merged_count, "(restart -> 0 expected)")

print("\n=== F. _hypothesis_gc semantics ===")


class NS:
    pass


ns = NS()
ns._hypotheses = {("a", "b"): A.Hypothesis(("a", "b")), ("c", "d"): A.Hypothesis(("c", "d")),
                  ("e", "f"): A.Hypothesis(("e", "f"))}
ns._hypotheses[("c", "d")].committed = True
dropped = object_services.ObjectServices._hypothesis_gc(ns, {("a", "b"), ("c", "d")})
print("  live={ab, cd}, cd committed, ef not live -> dropped", dropped, "remaining", list(ns._hypotheses))

print("\n=== G. _attribute_channel_input: same_kind on label casing / spacing / empty ===")
for la, lb in [("chair#1", "chair#2"), ("Chair#1", "chair#2"), ("chair #1", "chair#2"), ("", ""), ("chair", "armchair"),
               ("chair#1", "chair")]:
    oa = obj(la, BOX, "x", "grey", "fabric", "oa", 1.0)
    ob = obj(lb, SHIFT, "x", "grey", "fabric", "ob", 2.0)
    aa = A.AssocObject("oa", bbox=BOX, source=oa)
    bb = A.AssocObject("ob", bbox=SHIFT, source=ob)
    score, n_ev, same_kind = object_services._attribute_channel_input(aa, bb)
    print(f"  {la!r:12} vs {lb!r:12} -> score {score:.4f} optional_count {n_ev} same_kind {same_kind}")
print("  no source on one side ->", object_services._attribute_channel_input(A.AssocObject("x", bbox=BOX),
                                                                              A.AssocObject("y", bbox=BOX, source=obj("c", BOX, "x", "g", "f", "y", 1))))
wm.persistent_perceptions.clear()
print("\nprobe_services done")
