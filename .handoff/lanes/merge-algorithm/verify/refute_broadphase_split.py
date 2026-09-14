"""Refutation probe for the rule15-masking claim: locality_ok gates on the FUSED box while
WorldModel.candidates indexes the MEASURED box. Plain python3 + rosstub; the calls are the
ones object_manager_6._association_candidates / locality_ok make.

Part A: the 226 REAL GA-493 detections (capture/consumer/events.jsonl) against the FINAL map
        (bundle/persistent_perception.json, 104 objects, all with fused_bbox). For each
        detection: S_gate = objects locality_ok accepts; S_broad = wm.candidates(det, 0.3).
        A miss is an object in S_gate and not in S_broad. Upper bound: the final fused boxes
        were not all available when each detection arrived.
Part B: the exact miss condition in metres. Broad phase = box_gap(det, measured) <= 0.6
        (query expanded by 2*0.3, closed intervals). Gate = box_gap(det, fused) <= 0.3.
        So a miss needs fused to extend > 0.3 m beyond measured toward the detection --
        NOT > 0.6 m as the reviewer's probe (a cube INSIDE the fused box) requires.
Part C: merge path. The fused-compatible pairs generate_candidates never offers with
        covariance None: how much covariance shell would be needed (d - reach).
"""
import collections
import json
import math
import sys

PM = '/DATA/GRAPH-API/.claude/worktrees/found-merge-algorithm/lost3dsg/src/perception_module'
sys.path.insert(0, PM)
import rosstub  # noqa: E402

rosstub.install()
import association as A  # noqa: E402
import numpy as np  # noqa: E402
import object_info  # noqa: E402
import object_manager_6 as OM6  # noqa: E402
from world_model import wm  # noqa: E402

R = '/DATA/GRAPH-API/.handoff/lanes/perception/ga493_replay_20260914/'
GAP = OM6.ASSOCIATION_MARGIN_M
print("ASSOCIATION_MARGIN_M read by object_manager_6 =", GAP)
print("locality_ok answered by module:", OM6.locality_ok.__module__,
      "| _association_candidates -> wm is", type(wm).__module__ + "." + type(wm).__name__)


def base(label):
    return str(label).split('#')[0].strip().lower()


def box(x0, x1, y0, y1, z0, z1):
    return dict(x_min=x0, x_max=x1, y_min=y0, y_max=y1, z_min=z0, z_max=z1)


final = json.load(open(R + 'bundle/persistent_perception.json'))
wm.persistent_perceptions.clear()
objs = []
for o in final:
    ob = object_info.Object(o['label'], None, dict(o['bbox']))
    ob.object_id = o['object_id']
    ob.fused_bbox = dict(o['fused_bbox'])
    objs.append(ob)
    wm.persistent_perceptions.append(ob)

# ---- Part A
n_det = 0
n_miss_det = 0
n_miss_pairs = 0
n_miss_same = 0
n_miss_det_same = 0
n_gate_only_same_det = 0   # detections whose EVERY same-label gate-accepted object is missed
n_reverse = 0             # candidate offered but gate refuses (harmless direction)
per_obj = collections.Counter()
examples = []
for line in open(R + 'capture/consumer/events.jsonl'):
    e = json.loads(line)
    if e.get('kind') != 'consumer_pair':
        continue
    for d in e['payload']['bboxes']['boxes']:
        n_det += 1
        det = {k: d[k] for k in ('x_min', 'x_max', 'y_min', 'y_max', 'z_min', 'z_max')}
        broad = set(id(c) for c in OM6._association_candidates(det))
        gate = [ob for ob in objs if OM6.locality_ok(det, ob, GAP)]
        n_reverse += sum(1 for c in objs if id(c) in broad and not OM6.locality_ok(det, c, GAP))
        missed = [ob for ob in gate if id(ob) not in broad]
        if missed:
            n_miss_det += 1
            n_miss_pairs += len(missed)
            same = [ob for ob in missed if base(ob.label) == base(d['label'])]
            n_miss_same += len(same)
            if same:
                n_miss_det_same += 1
                gate_same = [ob for ob in gate if base(ob.label) == base(d['label'])]
                if all(id(ob) not in broad for ob in gate_same):
                    n_gate_only_same_det += 1
                for ob in same:
                    m, f = A._as_bounds(ob.bbox), A._as_bounds(ob.fused_bbox)
                    dd = A._as_bounds(det)
                    examples.append((e['cycle_id'][:6], d['label'], ob.label,
                                     round(A.box_gap(dd, f), 2), round(A.box_gap(dd, m), 2)))
            for ob in missed:
                per_obj[ob.label] += 1
print(f"\nA) real detections vs final map: detections {n_det}; "
      f"detections with >=1 gate-accepted object NOT in the broad phase: {n_miss_det} of {n_det} "
      f"({n_miss_pairs} detection-object pairs); same base label: {n_miss_det_same} detections / "
      f"{n_miss_same} pairs; detections whose EVERY same-label gate-accepted object is missed: "
      f"{n_gate_only_same_det}")
print(f"   context: broad-phase candidates the gate then refuses (harmless direction): {n_reverse} pairs")
print("   missed pairs by object:", per_obj.most_common(12))
print("   same-label misses (cycle, det label, obj label, gap to fused, gap to measured):")
for ex in examples[:20]:
    print("    ", ex)

# ---- Part B: the exact miss window with a 0.4 m extension (below the reviewer's 0.6 m cut)
wm.persistent_perceptions.clear()
t = object_info.Object('table', None, box(0, 1, 0, 1, 0, 0.8))
t.object_id = 't1'
t.fused_bbox = box(0, 1.4, 0, 1, 0, 0.8)      # fused extends 0.4 m past measured on x_max
wm.persistent_perceptions.append(t)
det = box(1.65, 2.0, 0, 1, 0, 0.8)             # 0.25 m past fused x_max, 0.65 m past measured
print(f"\nB) extension 0.4 m (< reviewer's 0.6 m cut), detection 0.25 m outside fused box: "
      f"locality_ok={OM6.locality_ok(det, t, GAP)} "
      f"broad={[c.object_id for c in OM6._association_candidates(det)]}")
n_pop = sum(1 for o in final if max(
    o['bbox']['x_min'] - o['fused_bbox']['x_min'], o['fused_bbox']['x_max'] - o['bbox']['x_max'],
    o['bbox']['y_min'] - o['fused_bbox']['y_min'], o['fused_bbox']['y_max'] - o['bbox']['y_max'],
    o['bbox']['z_min'] - o['fused_bbox']['z_min'], o['fused_bbox']['z_max'] - o['bbox']['z_max']) > 0.3)
print(f"   GA-493 final objects whose fused box extends > 0.3 m beyond measured on some side "
      f"(the population where the window exists): {n_pop} of {len(final)}")
wm.persistent_perceptions.clear()

# ---- Part C: merge path, covariance None. Deficit d - reach for fused-compatible pairs never offered.
aobjs = [A.AssocObject(o['object_id'], bbox=o['bbox'], room_id=o.get('room_id'), label=o['label'])
         for o in final]
ctx = A.AssocContext(map_volume_m3=300.0, n_rooms=6, cost_ratio=20.0, use_ontology=False,
                     locality_gap_m=GAP)
off, exc = A.generate_candidates(aobjs, ctx)
offered = {tuple(sorted((x.object_id, y.object_id))) for x, y, _ in off}
deficits = []
for i in range(len(final)):
    for j in range(i + 1, len(final)):
        fa, fb = A._as_bounds(final[i]['fused_bbox']), A._as_bounds(final[j]['fused_bbox'])
        if not A.geometry_compatible(fa, fb, GAP):
            continue
        key = tuple(sorted((final[i]['object_id'], final[j]['object_id'])))
        if key in offered:
            continue
        a, b = aobjs[i], aobjs[j]
        d = float(np.linalg.norm(a.centroid - b.centroid))
        reach = A.search_radius(a, ctx)[0] + A.search_radius(b, ctx)[0]
        deficits.append((round(d - reach, 2), final[i]['label'], final[j]['label']))
deficits.sort()
print(f"\nC) merge path, covariance None: fused-compatible pairs never offered {len(deficits)} of "
      f"{len(final) * (len(final) - 1) // 2}; deficit d-reach (m) min/median/max = "
      f"{deficits[0][0]}/{deficits[len(deficits) // 2][0]}/{deficits[-1][0]}")
sig = [math.sqrt(((x[0] / 2) ** 2) / A.CHI2_99_3DOF) for x in deficits]
print(f"   per-object position sigma (m) the covariance shell would need to close the deficit, "
      f"median {sorted(sig)[len(sig) // 2]:.3f}, max {max(sig):.3f} "
      f"(ESTIMATED: observations are not in the bundle)")
same = [x for x in deficits if base(x[1]) == base(x[2])]
print("   same-base-label among them:", same)
