"""Follow-ups to v3 with the real MiniLM (mlspaces_310, offline).

1. cross-base-label offered pairs whose FUSED boxes intersect: attribute score, channels, total,
   and whether the GA-328 guard still holds them (overlap >= 50% of total) -- the dilution path.
2. same-base-label final pairs: attribute score under the NEW weights, split by geometry
   (fused-intersecting vs centres > 1.5 m apart). Rule 18: what does the 'second witness' say
   for two DISTINCT objects of one kind?
3. Part-B extras on the exploration detections: the gap from each MOVED detection to the object's
   measured box; detections that become new objects ONLY because of the same-cycle veto; detections
   with >= 2 passing candidates where the first in insertion order is not the best-overlapping.
"""
import collections
import json
import os
import sys
import types

import numpy as np

CSS = {'white': (255, 255, 255), 'blue': (0, 0, 255), 'grey': (128, 128, 128), 'gray': (128, 128, 128),
       'brown': (165, 42, 42), 'green': (0, 128, 0), 'black': (0, 0, 0), 'silver': (192, 192, 192),
       'gold': (255, 215, 0), 'beige': (245, 245, 220), 'teal': (0, 128, 128), 'red': (255, 0, 0)}


class _RGB:
    def __init__(self, r, g, b):
        self.red, self.green, self.blue = r, g, b


def _name_to_rgb(name):
    if name not in CSS:
        raise ValueError(name)
    return _RGB(*CSS[name])


sys.modules['webcolors'] = types.SimpleNamespace(name_to_rgb=_name_to_rgb)
PM = '/DATA/GRAPH-API/.claude/worktrees/found-merge-algorithm/lost3dsg/src/perception_module'
sys.path.insert(0, PM)
os.chdir(PM)
os.environ.setdefault('HF_HOME', '/DATA/huggingface_cache')
import association as A  # noqa: E402
from config import CFG  # noqa: E402
from nlp_utils import get_embedding, lost_similarity_detailed, world2vec  # noqa: E402

assert world2vec.st_model is not None
print("component:", type(world2vec.st_model).__name__, "weights", CFG['similarity'])
R = '/DATA/GRAPH-API/.handoff/lanes/perception/ga493_replay_20260914/'
SIM_T = CFG['association']['sim_threshold']
THR = A.commit_threshold(20.0)
GAP = 0.3


def base(label):
    return str(label).split('#')[0].strip().lower()


def bnd(b):
    return A._as_bounds(b)


def iou3(a, b):
    i = A.intersection_volume(a, b)
    return i / (A.box_volume(a) + A.box_volume(b) - i) if i > 0 else 0.0


def half_diag(b):
    return 0.5 * float(np.linalg.norm([b[1] - b[0], b[3] - b[2], b[5] - b[4]]))


_emb = {}


def emb(text):
    if text not in _emb:
        _emb[text] = get_embedding(world2vec, text)
    return _emb[text]


def attr_score(a, b):
    s, ev = lost_similarity_detailed(world2vec, base(a['label']), base(b['label']), a['color'], b['color'],
                                     a['material'], b['material'], emb(a['description']), emb(b['description']))
    return s, int(ev['optional_count']), (base(a['label']) == base(b['label']) and base(a['label']) != '')


final = json.load(open(R + 'bundle/persistent_perception.json'))
N = len(final)
objs = [A.AssocObject(o['object_id'], bbox=o['bbox'], room_id=o.get('room_id'), label=o['label'], source=o)
        for o in final]
boxes = [bnd(o['bbox']) for o in final]
xs = [b[0] for b in boxes] + [b[1] for b in boxes]
ys = [b[2] for b in boxes] + [b[3] for b in boxes]
zs = [b[4] for b in boxes] + [b[5] for b in boxes]
vol = max((max(xs) - min(xs)) * (max(ys) - min(ys)) * (max(zs) - min(zs)), 1.0)
rooms = {o.get('room_id') for o in final}
rooms.discard(None)
ctx = A.AssocContext(map_volume_m3=vol, n_rooms=max(len(rooms), 1), cost_ratio=20.0, use_ontology=False,
                     attribute_reference=SIM_T, attribute_max_log_odds=2.0, locality_gap_m=GAP,
                     overlap_2d_fn=A.shared_frame_overlap_2d,
                     attribute_score_fn=lambda x, y: attr_score(x.source, y.source))
offered, excluded = A.generate_candidates(objs, ctx)

# 1. cross-label, fused-intersecting, past the gates
print("\n1. cross-base-label offered pairs with intersecting FUSED boxes (past room+geometry gates):")
print("   (label_a, label_b, c/m a, c/m b, score, total, overlap, attrs, guard_holds, decision)")
n_cross, n_guard, n_attr_pos = 0, 0, 0
rows = []
for a, b, meta in offered:
    oa, ob = a.source, b.source
    if base(oa['label']) == base(ob['label']):
        continue
    fa, fb = bnd(oa['fused_bbox']), bnd(ob['fused_bbox'])
    if A.intersection_volume(fa, fb) <= 0:
        continue
    n_cross += 1
    h = A.Hypothesis((a.object_id, b.object_id))
    for f in (1, 2):
        ps = A.score_pair(a, b, ctx)
        h.update(ps, frame_id=f)
        d, why = h.decide(THR, min_evidence=1, min_consecutive=2)
    at = ps.channels.get('attributes', {})
    n_attr_pos += at.get('log_odds', 0.0) > 0
    n_guard += ps.containment_unchecked
    rows.append((oa['label'], ob['label'], f"{oa['color']}/{oa['material']}", f"{ob['color']}/{ob['material']}",
                 at.get('score'), round(ps.total, 2), round(ps.channels.get('overlap', {}).get('log_odds', 0), 2),
                 round(at.get('log_odds', 0.0), 2), ps.containment_unchecked, d))
rows.sort(key=lambda r: -(r[4] or 0))
print(f"   n={n_cross}; attribute channel POSITIVE (score > 0.85) on {n_attr_pos}; guard holds {n_guard}")
for r in rows[:15]:
    print("   ", r)
above = [r for r in rows if r[5] >= THR]
print(f"   pairs with total >= threshold (held ONLY by the guard): {len(above)} -> {[(r[0], r[1], r[4], r[5]) for r in above]}")

# 2. same-label score distribution by geometry
print("\n2. same-base-label final pairs: attribute score under the NEW weights, by geometry")
inter, far = [], []
for i in range(N):
    for j in range(i + 1, N):
        oa, ob = final[i], final[j]
        if base(oa['label']) != base(ob['label']):
            continue
        s, n_ev, _ = attr_score(oa, ob)
        if n_ev < 1:
            continue
        fa, fb = bnd(oa['fused_bbox']), bnd(ob['fused_bbox'])
        d = float(np.linalg.norm(A.box_centroid(fa) - A.box_centroid(fb)))
        if A.intersection_volume(fa, fb) > 0:
            inter.append(s)
        elif d > 1.5:
            far.append(s)
for name, v in (('fused-intersecting (candidate re-observations)', inter), ('centres > 1.5 m apart (distinct objects)', far)):
    v = np.array(v)
    print(f"   {name}: n={len(v)}, median {np.median(v):.3f}, score > 0.85 (attribute channel positive, lift armed): "
          f"{int((v > SIM_T).sum())} ({100 * (v > SIM_T).mean():.0f}%)")

# 3. Part-B extras
obs = {}
cycle_t = {}
for line in open(R + 'capture/consumer/events.jsonl'):
    e = json.loads(line)
    if e['kind'] != 'consumer_pair':
        continue
    p = e['payload']
    cycle_t[e['cycle_id']] = e['recorded_at']
    descs = {(d.get('observation') or {}).get('observation_id'): d for d in p['descriptions']['descriptions']}
    for idx, b in enumerate(p['bboxes']['boxes']):
        oid = (b.get('observation') or {}).get('observation_id')
        d = descs.get(oid, {})
        obs[oid] = {'label': b['label'], 'box': bnd(b), 'expl': bool(p.get('exploration_mode')),
                    'color': d.get('color', ''), 'material': d.get('material', ''),
                    'description': d.get('description', ''), 'cycle': e['cycle_id'], 'idx': idx}
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
state = collections.OrderedDict()   # insertion order = wm.candidates order (DetectionIndex._order)
ev_i = 0
by_cycle = collections.defaultdict(list)
for o in obs.values():
    by_cycle[o['cycle']].append(o)
TRANS_D, STAB_T, FALLBACK_R, EXPL_IOU = 0.35, 3.0, 1.0, 0.10


def score(det, s):
    sc, ev = lost_similarity_detailed(world2vec, base(det['label']), base(s['label']), det['color'], s['color'],
                                      det['material'], s['material'], emb(det['description']), emb(s['description']))
    return sc, ev['optional_count']


moved_gap, veto_only, order_cases = [], [], []
n_det = 0
for cyc in sorted(by_cycle, key=lambda c: cycle_t[c]):
    t = cycle_t[cyc]
    while ev_i < len(events) and events[ev_i][0] < t:
        _, _, op, oid, obs_id = events[ev_i]
        ev_i += 1
        if op in ('merge', 'delete'):
            state.pop(oid, None)
            continue
        o = obs.get(obs_id)
        if o is None:
            continue
        if op == 'add':
            state[oid] = {**{k: o[k] for k in ('label', 'box', 'color', 'material', 'description')},
                          'created': events[ev_i - 1][0]}
        elif oid in state:
            state[oid]['box'] = o['box']
    dets = sorted(by_cycle[cyc], key=lambda o: o['idx'])
    if not dets or not dets[0]['expl']:
        continue
    cycle_objs = set()
    for det in dets:
        n_det += 1
        best, best_s = None, -1.0
        for oid, s in state.items():
            if t - s['created'] < STAB_T:
                continue
            d = float(np.linalg.norm(A.box_centroid(det['box']) - A.box_centroid(s['box'])))
            if d > half_diag(s['box']) + FALLBACK_R + half_diag(det['box']):
                continue
            sc, n_ev = score(det, s)
            if sc > SIM_T and sc > best_s and n_ev >= 1:
                best, best_s, best_d = oid, sc, d
        if best is not None:
            v = iou3(det['box'], state[best]['box'])
            if best_d > TRANS_D and v < EXPL_IOU:
                g = A.box_gap(det['box'], state[best]['box'])
                moved_gap.append((cyc[:6], det['label'], state[best]['label'], round(best_d, 2), round(g, 2),
                                  'gap<=0.3: the exploration loop WOULD have absorbed it' if g <= GAP else 'gap>0.3'))
                continue
        passing = []
        for oid, s in state.items():
            if A.geometry_compatible(det['box'], s['box'], GAP) is not True:
                continue
            sc, n_ev = score(det, s)
            if sc > SIM_T:
                passing.append((oid, sc, iou3(det['box'], s['box'])))
        live = [p for p in passing if p[0] not in cycle_objs]
        if passing and not live:
            veto_only.append((cyc[:6], det['label'], [state[p[0]]['label'] for p in passing]))
        if live:
            cycle_objs.add(live[0][0])
            if len(live) >= 2:
                best_iou = max(live, key=lambda p: p[2])
                order_cases.append((cyc[:6], det['label'], [(state[p[0]]['label'], round(p[1], 3), round(p[2], 2)) for p in live],
                                    'first != best-IoU' if best_iou[0] != live[0][0] else 'first is best-IoU'))
print(f"\n3. exploration detections n={n_det}")
print(f"   MOVED transitions (new weights) {len(moved_gap)}: (cycle, det, obj, centre m, gap m, note)")
for r in moved_gap:
    print("   ", r)
print(f"   detections that become NEW objects only because every passing candidate was absorbed earlier in the cycle: "
      f"{len(veto_only)} -> {veto_only}")
print(f"   detections with >= 2 live passing candidates: {len(order_cases)}; first-in-insertion-order is not the "
      f"best-IoU one in {sum(1 for c in order_cases if c[3].startswith('first !='))}")
for c in order_cases[:10]:
    print("   ", c)
