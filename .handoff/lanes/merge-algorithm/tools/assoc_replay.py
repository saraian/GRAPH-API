"""Association counterfactual on the GA-493 bundle, GT-free.

Rebuild the world model in time order from the mutation ledger (add / update), the merge and
disappearance records, and the captured consumer inputs (every detection's 3D box, label,
colour, material, description). At each ADD, ask: did a same-base-label object already exist whose
box overlapped the new detection at or above the locality gate the legacy loop applied at that
moment (0.10 in exploration, 0.30 in tracking)? If yes, geometry passed and only the attribute
score (lost_similarity > 0.85) can have refused the association. `obj.bbox` in the legacy loop is
the last accepted view's AABB, which is what this replay keeps.
"""
import collections
import json

import numpy as np

R = '/DATA/GRAPH-API/.handoff/lanes/perception/ga493_replay_20260914/'
K = ['x_min', 'y_min', 'z_min', 'x_max', 'y_max', 'z_max']
EXPL_IOU, TRACK_IOU = 0.10, 0.30


def arr(b):
    return np.array([float(b[k]) for k in K])


def iou(a, b):
    lo = np.maximum(a[:3], b[:3])
    hi = np.minimum(a[3:], b[3:])
    d = np.clip(hi - lo, 0, None)
    i = d.prod()
    va = (a[3:] - a[:3]).prod()
    vb = (b[3:] - b[:3]).prod()
    return float(i / (va + vb - i)) if va + vb - i > 0 else 0.0


def cen(a):
    return (a[:3] + a[3:]) / 2


def base(label):
    return label.split('#')[0]


# 1. every captured detection, keyed by observation_id
obs = {}
for line in open(R + 'capture/consumer/events.jsonl'):
    e = json.loads(line)
    if e['kind'] != 'consumer_pair':
        continue
    p = e['payload']
    mode_expl = bool(p.get('exploration_mode'))
    moved = bool(p.get('robot_has_moved'))
    descs = {}
    for d in p['descriptions']['descriptions']:
        oid = (d.get('observation') or {}).get('observation_id')
        descs[oid] = d
    for b in p['bboxes']['boxes']:
        oid = (b.get('observation') or {}).get('observation_id')
        d = descs.get(oid, {})
        obs[oid] = {'label': b['label'], 'box': arr(b), 'expl': mode_expl, 'moved': moved,
                    'color': d.get('color', ''), 'material': d.get('material', ''),
                    'description': d.get('description', ''), 'cycle': e['cycle_id']}
print(f"captured detections with 3D box: {len(obs)}")

# 2. events in time order
events = []
for line in open(R + 'bundle/mutation_receipts.jsonl'):
    r = json.loads(line)
    if r['mutation_state'] != 'applied_in_memory':
        continue
    events.append((r['recorded_at'], r['mutation_sequence'], r['operation'], r['object_id'],
                   (r.get('observation') or {}).get('observation_id')))
for line in open(R + 'bundle/hook_decisions.jsonl'):
    r = json.loads(line)
    if r['kind'] == 'merge' and not r.get('dry_run'):
        events.append((r['t'], 10**9, 'merge', r['merged_from'], None))
    elif r['kind'] == 'disappearance_removal':
        events.append((r['t'], 10**9, 'delete', r.get('object'), None))
events.sort(key=lambda e: (e[0], e[1]))
print(f"events: {collections.Counter(e[2] for e in events)}")

# 3. replay
state = {}   # object_id -> dict(label, box, color, material, description)
rows = []
missing_obs = 0
for t, seq, op, oid, obs_id in events:
    if op == 'merge' or op == 'delete':
        state.pop(oid, None)
        continue
    o = obs.get(obs_id)
    if o is None:
        missing_obs += 1
        continue
    if op == 'add':
        thr = EXPL_IOU if o['expl'] else TRACK_IOU
        best_same = (0.0, None)
        best_any = 0.0
        near_same = (float('inf'), None)
        for eid, s in state.items():
            v = iou(o['box'], s['box'])
            best_any = max(best_any, v)
            if base(s['label']) == base(o['label']):
                if v > best_same[0]:
                    best_same = (v, eid)
                dist = float(np.linalg.norm(cen(o['box']) - cen(s['box'])))
                if dist < near_same[0]:
                    near_same = (dist, eid)
        if best_same[0] >= thr:
            cls = 'A_locality_passed_only_attributes_could_refuse'
        elif best_same[0] > 0:
            cls = 'B_same_label_overlap_below_locality_gate'
        elif near_same[0] <= 1.0:
            cls = 'C_same_label_within_1m_no_overlap'
        else:
            cls = 'D_no_same_label_within_1m'
        rows.append({'t': t, 'oid': oid, 'label': o['label'], 'mode': 'expl' if o['expl'] else 'track',
                     'thr': thr, 'cls': cls, 'best_same_iou': best_same[0], 'best_any_iou': best_any,
                     'near_same_m': near_same[0], 'partner': best_same[1] or near_same[1],
                     'obs': o})
    state[oid] = {'label': o['label'], 'box': o['box'], 'color': o['color'],
                  'material': o['material'], 'description': o['description']}
print(f"adds replayed: {len(rows)}; ledger rows without a captured observation: {missing_obs}; "
      f"final objects in replay: {len(state)}")

tally = collections.Counter(r['cls'] for r in rows)
by_mode = collections.Counter((r['mode'], r['cls']) for r in rows)
print("\n=== every ADD (new object created), by what the legacy loop could have refused on ===")
for c in sorted(tally):
    print(f"  {c:48s} {tally[c]:4d}   expl {by_mode[('expl', c)]:3d}  track {by_mode[('track', c)]:3d}")
first = sum(1 for r in rows if r['cls'] == 'D_no_same_label_within_1m')
print(f"\nadds with NO same-label object within 1 m (first sighting of that class here): {first} of {len(rows)}")

print("\n=== class A: geometry passed the locality gate; the attribute score refused (top by IoU) ===")
print(f"{'label':14s} {'mode':5s} {'IoU':>5s} {'partner label':14s}  new colour/material | partner colour/material")
for r in sorted((r for r in rows if r['cls'].startswith('A')), key=lambda r: -r['best_same_iou'])[:25]:
    o = r['obs']
    p = None
    # partner attributes as they were at that moment are not kept after later updates; use the
    # partner's attributes from the add row that created it (state was overwritten since).
    for q in rows:
        if q['oid'] == r['partner']:
            p = q['obs']
            break
    pc = f"{p['color']}/{p['material']}" if p else '?'
    pl = p['label'] if p else '?'
    print(f"{r['label']:14s} {r['mode']:5s} {r['best_same_iou']:5.2f} {pl:14s}  {o['color']}/{o['material']} | {pc}")
    if p:
        print(f"      new: {o['description'][:70]!r}")
        print(f"      old: {p['description'][:70]!r}")

print("\n=== class B: same-label overlap BELOW the locality gate (tracking 0.30) ===")
B = [r for r in rows if r['cls'].startswith('B')]
hist = collections.Counter('0.10-0.30' if r['best_same_iou'] >= 0.10 else '<0.10' for r in B)
print(f"  n={len(B)}; IoU in [0.10,0.30) (would pass the EXPLORATION gate): {hist['0.10-0.30']}; IoU < 0.10: {hist['<0.10']}")
print(f"  {'label':14s} {'IoU':>5s} {'centre m':>8s}  partner colour/material -> new")
for r in sorted(B, key=lambda r: -r['best_same_iou'])[:25]:
    o = r['obs']
    p = next((q['obs'] for q in rows if q['oid'] == r['partner']), None)
    pc = f"{p['color']}/{p['material']}" if p else '?'
    print(f"  {r['label']:14s} {r['best_same_iou']:5.2f} {r['near_same_m']:8.2f}  {pc} -> {o['color']}/{o['material']}")
print("\n=== class C: same-label within 1 m, no overlap ===")
C = [r for r in rows if r['cls'].startswith('C')]
print("  " + ", ".join(f"{r['label']}@{r['near_same_m']:.2f}m" for r in sorted(C, key=lambda r: r['near_same_m'])))
print("\n=== class D by label (first sighting of that class within 1 m) ===")
print("  " + ", ".join(f"{k}x{v}" for k, v in collections.Counter(base(r['label']) for r in rows if r['cls'].startswith('D')).most_common(20)))

# colour / material agreement inside class A
same_c = sum(1 for r in rows if r['cls'].startswith('A') and any(
    q['oid'] == r['partner'] and q['obs']['color'].lower() == r['obs']['color'].lower() for q in rows))
same_m = sum(1 for r in rows if r['cls'].startswith('A') and any(
    q['oid'] == r['partner'] and q['obs']['material'].lower() == r['obs']['material'].lower() for q in rows))
nA = tally['A_locality_passed_only_attributes_could_refuse']
print(f"\nclass A: {nA} adds; partner colour word identical in {same_c}, material word identical in {same_m}")
