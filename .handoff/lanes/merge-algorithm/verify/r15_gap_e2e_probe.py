"""End-to-end probe of the rule15-masking claim through the REAL caller
(ObjectServices._cb_merge_objects -> _merge_candidates -> _assoc_build -> score_pair -> decide),
under rosstub (identical strings score 1.0, description term dropped).

Arm (b): two same-kind boxes 0.25 m apart (gap-compatible, no intersection), identical
         attributes, same room_id, n_rooms = 4, NO observations.
Arm (a): the same pair with ONE sighting each (what _record_sighting produces; the service
         passes neither depth_sparsity nor pose_sigma_m).
Arm (a2): TWO sightings each with a small centroid spread.
"""
import pathlib
import sys

PM = pathlib.Path('/DATA/GRAPH-API/.claude/worktrees/found-merge-algorithm/lost3dsg/src/perception_module')
sys.path.insert(0, str(PM))
import rosstub  # noqa: E402
rosstub.install()
import object_info      # noqa: E402
import object_services  # noqa: E402
import room_manager     # noqa: E402
import association as A  # noqa: E402
from world_model import wm  # noqa: E402

for m in (object_services, room_manager, A):
    assert pathlib.Path(m.__file__).resolve().parent == PM.resolve(), m.__file__
print("component: object_services from", object_services.__file__)
print("MERGE_ENGINE", object_services.MERGE_ENGINE, "ontology", object_services.MERGE_ONTOLOGY_CHANNEL,
      "attr cap", object_services.MERGE_ATTRIBUTE_MAX_LOG_ODDS, "gap", object_services.LOCALITY_GAP_M,
      "min_consecutive", object_services.MERGE_MIN_CONSECUTIVE,
      "threshold", round(A.commit_threshold(object_services.MERGE_COST_RATIO), 3))

svc = object_services.ObjectServices.__new__(object_services.ObjectServices)
svc.get_logger = lambda: rosstub.Any()
svc.log_both = lambda *a, **k: None
svc.room_manager = room_manager.RoomManager.__new__(room_manager.RoomManager)
svc.room_manager.scene_graph = {}
svc.room_manager.current_room_id = "room_0"
svc.room_manager.room_at_bbox = lambda bbox: None      # the room GATE passes; the CHANNEL reads obj.room_id
rows = []


class _Log:
    def write(self, kind, oid, **kw):
        rows.append((kind, oid, kw))


svc.decision_log = _Log()


def box(x0, x1, y0=0.0, y1=1.0, z0=0.0, z1=1.0):
    return dict(x_min=x0, x_max=x1, y_min=y0, y_max=y1, z_min=z0, z_max=z1)


def obj(label, b, oid, room, desc="a grey fabric chair", color="grey", material="fabric", t=1.0):
    o = object_info.Object(label, None, dict(b), description=desc, color=color, material=material)
    o.object_id, o.creation_time, o.room_id = oid, t, room
    o.centroid = object_services._centroid_from_bbox(o.bbox)
    return o


def sweep(objects):
    wm.persistent_perceptions.clear()
    wm.persistent_perceptions.extend(objects)
    rows.clear()
    req, resp = rosstub.Any(), rosstub.Any()
    req.max_distance, req.min_similarity, req.dry_run = 0.8, 0.95, True
    object_services.ObjectServices._cb_merge_objects(svc, req, resp)
    assert resp.success is True, resp.message
    return resp


def pair_rows(pair):
    out = []
    for k, oid, kw in rows:
        if k in ("merge_refused", "merge") and {oid, kw.get("candidate")} == set(pair):
            out.append((k, kw.get("reason"), kw.get("decision_reason"),
                        {c: (round(v.get("log_odds"), 3) if isinstance(v, dict) and v.get("log_odds") is not None else v) for c, v in (kw.get("channels") or {}).items()},
                        round(kw.get("hypothesis_total"), 3) if kw.get("hypothesis_total") is not None else None,
                        sorted((kw.get("abstentions") or {}).keys())))
    return out


def anchors(n_rooms):
    # far objects in OTHER rooms, so _assoc_build's n_rooms = 1 + len(anchors) and the map hull is wide
    return [obj("lamp", box(20.0 + 3 * i, 21.0 + 3 * i, 20.0, 21.0), f"anchor{i}", f"room_{i + 1}",
                desc="a lamp", color="black", material="metal", t=3.0 + i) for i in range(n_rooms - 1)]


def run(name, a, b, n_rooms=4, sweeps=2):
    svc._hypotheses, svc._merge_sweep = {}, 0
    print(f"\n--- {name} (n_rooms={n_rooms})")
    for s in range(1, sweeps + 1):
        r = sweep([a, b] + anchors(n_rooms))
        offered = [kw for k, _, kw in rows if k == "merge_refused" or k == "merge"]
        print(f"  sweep {s}: merged_count={getattr(r, 'merged_count', None)} rows={pair_rows(('obj_a', 'obj_b'))}")
    return r


A_BOX, B_BOX = box(0.0, 1.0), box(1.25, 2.25)      # gap 0.25 m on x, no intersection
ga = A.box_gap(A._as_bounds(A_BOX), A._as_bounds(B_BOX))
print("box gap", ga, "geometry_compatible@0.3", A.geometry_compatible(A._as_bounds(A_BOX), A._as_bounds(B_BOX), 0.3),
      "intersection", A.intersection_volume(A._as_bounds(A_BOX), A._as_bounds(B_BOX)))

# ---- arm (b): no observations ---------------------------------------------------------------
a = obj("chair", A_BOX, "obj_a", "room_0")
b = obj("chair", B_BOX, "obj_b", "room_0", t=2.0)
rb = run("arm (b) no observations, identical attributes, same room", a, b, n_rooms=4)
print("  => arm (b) MERGED" if rb.merged_count == 1 else "  => arm (b) did not merge")
run("arm (b) with n_rooms=2", obj("chair", A_BOX, "obj_a", "room_0"), obj("chair", B_BOX, "obj_b", "room_0", t=2.0), n_rooms=2)
run("arm (b) with n_rooms=3", obj("chair", A_BOX, "obj_a", "room_0"), obj("chair", B_BOX, "obj_b", "room_0", t=2.0), n_rooms=3)

# pre-change engine on the same pair: attribute channel absent (abstains), ontology on
old = object_services._attribute_channel_input
object_services._attribute_channel_input = lambda aa, bb: None
run("arm (b) PRE-CHANGE (attribute channel abstains)", obj("chair", A_BOX, "obj_a", "room_0"),
    obj("chair", B_BOX, "obj_b", "room_0", t=2.0), n_rooms=4)
object_services._attribute_channel_input = old

# ---- arm (a): one sighting each -------------------------------------------------------------
cam = (0.6, -3.0, 0.5)
a1 = obj("chair", A_BOX, "obj_a", "room_0")
b1 = obj("chair", B_BOX, "obj_b", "room_0", t=2.0)
a1.observations = [A.Observation("f1", cam, a1.centroid, bbox_2d=[0, 0, 10, 10])]
b1.observations = [A.Observation("f2", cam, b1.centroid, bbox_2d=[0, 0, 10, 10])]
print("  cov one sighting:", A.position_covariance(a1.observations)[0, 0])
run("arm (a) ONE sighting each", a1, b1, n_rooms=4)

# ---- arm (a2): two sightings each, centroid spread 5 cm -------------------------------------
for spread in (0.0, 0.02, 0.05, 0.10):
    a2 = obj("chair", A_BOX, "obj_a", "room_0")
    b2 = obj("chair", B_BOX, "obj_b", "room_0", t=2.0)
    ca, cb = a2.centroid, b2.centroid
    a2.observations = [A.Observation("f1", cam, ca, bbox_2d=[0, 0, 10, 10]),
                       A.Observation("f3", cam, [ca[0] + spread, ca[1] + spread, ca[2]], bbox_2d=[0, 0, 10, 10])]
    b2.observations = [A.Observation("f2", cam, cb, bbox_2d=[0, 0, 10, 10]),
                       A.Observation("f4", cam, [cb[0] + spread, cb[1] + spread, cb[2]], bbox_2d=[0, 0, 10, 10])]
    print("  cov two sightings spread", spread, ":", round(float(A.position_covariance(a2.observations)[0, 0]), 6))
    run(f"arm (a2) TWO sightings each, spread {spread} m", a2, b2, n_rooms=4)

# ---- was the pair offered by the shell or by the gap? ----------------------------------------
ctx, built = object_services.ObjectServices._assoc_build(svc, [a, b] + anchors(4))
off, exc = A.generate_candidates(list(built.values()), ctx)
for aa, bb, meta in off:
    if {aa.object_id, bb.object_id} == {"obj_a", "obj_b"}:
        print("\noffered_by:", meta["offered_by"], "d", round(meta["distance_m"], 3), "reach", round(meta["reach_m"], 3))
wm.persistent_perceptions.clear()
