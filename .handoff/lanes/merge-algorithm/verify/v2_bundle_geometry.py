"""GA-493 bundle, geometry only (plain python3). Denominators stated in every line.

(a) fused_bbox extension beyond bbox per object vs the 0.6 m broad-phase window
    (wm.candidates indexes obj.bbox and expands by 2*association_margin_m).
(b) final-object pairs: gate-compatible on the FUSED box (what _cb_merge_objects tests) vs
    offered by generate_candidates (measured bbox + covariance shell; covariance None here).
(c) same-cycle same-label detection pairs (the same-cycle veto population) and their 2D IoU.
(d) mutation receipts: one object absorbing two detections in one cycle under the old code.
(e) rooms among final objects (n_rooms the room channel uses).
(f) same-label final pairs nested (containment >= 0.8) or adjacent (no intersection, gap <= 0.3).
"""
import collections
import json
import sys

import numpy as np

PM = '/DATA/GRAPH-API/.claude/worktrees/found-merge-algorithm/lost3dsg/src/perception_module'
sys.path.insert(0, PM)
import association as A  # noqa: E402

R = '/DATA/GRAPH-API/.handoff/lanes/perception/ga493_replay_20260914/'
final = json.load(open(R + 'bundle/persistent_perception.json'))
N = len(final)


def base(label):
    return str(label).split('#')[0].strip().lower()


def bnd(b):
    return A._as_bounds(b)


def iou3(a, b):
    i = A.intersection_volume(a, b)
    return i / (A.box_volume(a) + A.box_volume(b) - i) if i > 0 else 0.0


def iou2(p, q):
    return A._iou_2d(p, q)


# (a)
print(f"(a) fused_bbox vs bbox, {N} final objects")
ext = []
for o in final:
    m, f = bnd(o['bbox']), bnd(o['fused_bbox'])
    e = max(m[0] - f[0], f[1] - m[1], m[2] - f[2], f[3] - m[3], m[4] - f[4], f[5] - m[5])
    ext.append((e, o['label'], o['fused_bbox'].get('view_count')))
ext.sort(reverse=True)
print(f"    max one-sided extension of fused beyond measured: >0.6 m in {sum(1 for e in ext if e[0] > 0.6)}, "
      f">0.3 m in {sum(1 for e in ext if e[0] > 0.3)}, median {np.median([e[0] for e in ext]):.3f} m")
print("    top 8:", [(round(e, 2), lab, vc) for e, lab, vc in ext[:8]])
print(f"    fused view_count: {collections.Counter(o['fused_bbox'].get('view_count') for o in final)}")

# (b)
print(f"\n(b) final-object pairs, {N * (N - 1) // 2} all-pairs; covariance None for every object "
      f"(observations are not in the bundle), so the shell = sum of half-diagonals")
objs = []
for o in final:
    objs.append(A.AssocObject(o['object_id'], bbox=o['bbox'], room_id=o.get('room_id'), label=o['label']))
ctx = A.AssocContext(map_volume_m3=300.0, n_rooms=6, cost_ratio=20.0, use_ontology=False, locality_gap_m=0.3)
off, exc = A.generate_candidates(objs, ctx)
offered = {tuple(sorted((x.object_id, y.object_id))) for x, y, _ in off}
by = collections.Counter(m['offered_by'] for _, _, m in off)
print(f"    generate_candidates: offered {len(off)} (shell {by['shell']}, gap {by['gap']}), excluded {len(exc)}")
fused_ok, meas_ok, both, gate_not_offered = 0, 0, 0, []
same_gate_not_offered = 0
for i in range(N):
    for j in range(i + 1, N):
        fa, fb = bnd(final[i]['fused_bbox']), bnd(final[j]['fused_bbox'])
        ma, mb = bnd(final[i]['bbox']), bnd(final[j]['bbox'])
        g_f = A.geometry_compatible(fa, fb, 0.3)
        g_m = A.geometry_compatible(ma, mb, 0.3)
        fused_ok += g_f
        meas_ok += g_m
        both += g_f and g_m
        key = tuple(sorted((final[i]['object_id'], final[j]['object_id'])))
        if g_f and key not in offered:
            gate_not_offered.append((final[i]['label'], final[j]['label'], round(A.box_gap(fa, fb), 2),
                                     round(A.box_gap(ma, mb), 2)))
            same_gate_not_offered += base(final[i]['label']) == base(final[j]['label'])
print(f"    gate-compatible on FUSED: {fused_ok}; on MEASURED: {meas_ok}; both: {both}")
print(f"    fused-compatible but NEVER OFFERED by generate_candidates: {len(gate_not_offered)} "
      f"(same base label: {same_gate_not_offered})")
print("    examples (label_a, label_b, fused gap, measured gap):", gate_not_offered[:12])

# (c)
print("\n(c) same-cycle detection pairs from capture/consumer/events.jsonl")
cycles = 0
pairs_same, pairs_same_gap, pairs_same_iou10 = 0, 0, 0
rows = []
n_det = 0
for line in open(R + 'capture/consumer/events.jsonl'):
    e = json.loads(line)
    if e['kind'] != 'consumer_pair':
        continue
    cycles += 1
    boxes = e['payload']['bboxes']['boxes']
    n_det += len(boxes)
    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            p, q = boxes[i], boxes[j]
            if base(p['label']) != base(q['label']):
                continue
            pairs_same += 1
            bp, bq = bnd(p), bnd(q)
            gap = A.box_gap(bp, bq)
            v = iou3(bp, bq)
            if v >= 0.10:
                pairs_same_iou10 += 1
            if gap <= 0.3:
                pairs_same_gap += 1
                i2 = None
                if p.get('has_bbox_2d') and q.get('has_bbox_2d'):
                    try:
                        i2 = iou2(json.loads(p['bbox_2d'].replace(' ', ',').replace(',,', ',').replace('[,', '[')),
                                  json.loads(q['bbox_2d'].replace(' ', ',').replace(',,', ',').replace('[,', '[')))
                    except Exception:
                        i2 = 'unparsed'
                rows.append((e['cycle_id'][:6], p['label'], q['label'], round(gap, 2), round(v, 2), i2))
print(f"    cycles {cycles}, detections {n_det}, same-base-label pairs in one cycle {pairs_same}; "
      f"of those gap<=0.3 (new locality): {pairs_same_gap}, 3D IoU>=0.10 (old exploration gate): {pairs_same_iou10}")
lo = [r for r in rows if isinstance(r[5], float) and r[5] < 0.30]
print(f"    gap<=0.3 pairs with 2D IoU < 0.30 (co-visibility would VETO their merge): {len(lo)} of {len(rows)}")
for r in rows[:20]:
    print("     ", r)

# (d)
print("\n(d) mutation_receipts: (object_id, cycle) with >= 2 applied add/update receipts")
per = collections.Counter()
for line in open(R + 'bundle/mutation_receipts.jsonl'):
    r = json.loads(line)
    if r['mutation_state'] != 'applied_in_memory' or r['operation'] not in ('add', 'update'):
        continue
    cyc = ((r.get('observation') or {}).get('cycle_id'))
    per[(r['object_id'], cyc)] += 1
multi = {k: v for k, v in per.items() if v >= 2}
print(f"    {len(multi)} of {len(per)} (object, cycle) keys absorbed >= 2 detections in one cycle under the old code")

# (e)
rooms = collections.Counter(o.get('room_id') for o in final)
print(f"\n(e) room_id among {N} final objects: {dict(rooms)}  -> n_rooms (known) = "
      f"{len([r for r in rooms if r not in (None, 'unknown', '')])}, log = "
      f"{np.log(max(len([r for r in rooms if r not in (None, 'unknown', '')]), 1)):.3f}")

# (f)
print("\n(f) same-base-label final pairs, by geometry (measured box | fused box)")
nested_m = nested_f = adj_m = adj_f = inter_m = inter_f = 0
ex_nested, ex_adj = [], []
for i in range(N):
    for j in range(i + 1, N):
        if base(final[i]['label']) != base(final[j]['label']):
            continue
        ma, mb = bnd(final[i]['bbox']), bnd(final[j]['bbox'])
        fa, fb = bnd(final[i]['fused_bbox']), bnd(final[j]['fused_bbox'])
        for tag, x, y in (('m', ma, mb), ('f', fa, fb)):
            inter = A.intersection_volume(x, y) > 0
            cont = A.containment_ratio(x, y) if inter else 0.0
            gap = A.box_gap(x, y)
            if tag == 'm':
                inter_m += inter
                nested_m += cont >= 0.8
                adj_m += (not inter) and gap <= 0.3
            else:
                inter_f += inter
                nested_f += cont >= 0.8
                adj_f += (not inter) and gap <= 0.3
                if cont >= 0.8:
                    ex_nested.append((final[i]['label'], final[j]['label'], round(cont, 2),
                                      final[i]['color'], final[j]['color'], final[i]['material'], final[j]['material']))
                if (not inter) and gap <= 0.3:
                    ex_adj.append((final[i]['label'], final[j]['label'], round(gap, 2),
                                   final[i]['color'], final[j]['color'], final[i]['material'], final[j]['material']))
print(f"    intersecting: measured {inter_m} | fused {inter_f}; nested (containment>=0.8): {nested_m} | {nested_f}; "
      f"adjacent (no intersection, gap<=0.3): {adj_m} | {adj_f}")
print("    nested on fused:", ex_nested[:15])
print("    adjacent on fused:", ex_adj[:15])
