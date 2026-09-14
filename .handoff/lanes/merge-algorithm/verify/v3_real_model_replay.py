"""GA-493 bundle through the BRANCH code with the REAL MiniLM. Runs under mlspaces_310 offline.

Part A  the evidence merge engine as _cb_merge_objects now wires it (generate_candidates with the
        gap offer, room gate on fused overlap, geometry gate on fused, score_pair with the attribute
        channel, Hypothesis over 2 identical sweeps) over the 104 FINAL objects of the run. Three
        configurations: branch (new weights, attribute channel), baseline weights + attribute
        channel, and the pre-ruling engine (no attribute channel). Observations are not in the
        bundle, so covariance is None (separation abstains) and co-visibility is UNCOLLECTED for
        every pair -- exactly the arm the GA-328 guard exists for.
Part B  per-detection counterfactual in time order against the OLD run's world state (rebuilt from
        the ledger, as tools/assoc_replay.py does): what check_tracking_transition (unchanged code,
        new weights) and the exploration loop (gap locality, same-cycle veto, new weights) would
        have said. Counts under the new weights and under the old weights.
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

assert world2vec.st_model is not None, "real SentenceTransformer did not load"
print("component: nlp_utils.world2vec.st_model =", type(world2vec.st_model).__name__)
print("CFG similarity weights as loaded from the worktree config.yaml:", CFG['similarity'])
print("CFG association merge_engine/ontology/attr cap/margin:", CFG['association'].get('merge_engine'),
      CFG['association'].get('merge_ontology_channel'), CFG['association'].get('merge_attribute_max_log_odds'),
      CFG['association'].get('association_margin_m'))
NEW_W = dict(CFG['similarity'])
OLD_W = {'label': 0.05, 'color': 0.30, 'material': 0.15, 'description': 0.50}
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
    """pair_attribute_score / _attribute_channel_input as object_services defines them."""
    s, ev = lost_similarity_detailed(world2vec, base(a['label']), base(b['label']), a['color'], b['color'],
                                     a['material'], b['material'], emb(a['description']), emb(b['description']))
    return s, int(ev['optional_count']), (base(a['label']) == base(b['label']) and base(a['label']) != '')


# ----------------------------------------------------------------------------------------- Part A
final = json.load(open(R + 'bundle/persistent_perception.json'))
N = len(final)
src = {o['object_id']: o for o in final}


def run_engine(weights, use_attrs, tag):
    CFG['similarity'].clear()
    CFG['similarity'].update(weights)
    objs = [A.AssocObject(o['object_id'], bbox=o['bbox'], room_id=o.get('room_id'), label=o['label'],
                          source=o) for o in final]
    boxes = [bnd(o['bbox']) for o in final]
    xs = [b[0] for b in boxes] + [b[1] for b in boxes]
    ys = [b[2] for b in boxes] + [b[3] for b in boxes]
    zs = [b[4] for b in boxes] + [b[5] for b in boxes]
    vol = max((max(xs) - min(xs)) * (max(ys) - min(ys)) * (max(zs) - min(zs)), 1.0)
    rooms = {o.get('room_id') for o in final}
    rooms.discard(None)
    kw = dict(map_volume_m3=vol, n_rooms=max(len(rooms), 1), cost_ratio=20.0, use_ontology=False,
              attribute_reference=SIM_T, attribute_max_log_odds=2.0, locality_gap_m=GAP,
              overlap_2d_fn=A.shared_frame_overlap_2d)
    if use_attrs:
        kw['attribute_score_fn'] = lambda x, y: attr_score(x.source, y.source)
    ctx = A.AssocContext(**kw)
    offered, excluded = A.generate_candidates(objs, ctx)
    dec = collections.Counter()
    merges, diluted, no_inter, cross = [], [], [], []
    for a, b, meta in offered:
        oa, ob = a.source, b.source
        ba, bb_ = bnd(oa['fused_bbox']), bnd(ob['fused_bbox'])
        overlapping = A.intersection_volume(ba, bb_) > 0
        ra, rb = oa.get('room_id'), ob.get('room_id')
        if ra is not None and rb is not None and ra != rb and not overlapping:
            dec['room'] += 1
            continue
        if A.geometry_compatible(ba, bb_, GAP) is False:
            dec['geometry'] += 1
            continue
        h = A.Hypothesis((a.object_id, b.object_id))
        for f in (1, 2):
            ps = A.score_pair(a, b, ctx)
            h.update(ps, frame_id=f)
            d, why = h.decide(THR, min_evidence=1, min_consecutive=2)
        dec[d] += 1
        if d == 'merge':
            ch = {k: round(v.get('log_odds', 0.0), 2) for k, v in ps.channels.items()}
            ma, mb = bnd(oa['bbox']), bnd(ob['bbox'])
            inter_m = A.intersection_volume(ma, mb) > 0
            attrs = ps.channels.get('attributes', {})
            row = (oa['label'], ob['label'], f"{oa['color']}/{oa['material']}", f"{ob['color']}/{ob['material']}",
                   attrs.get('score'), attrs.get('same_kind'), round(ps.total, 2), ch,
                   'inter' if inter_m else f"gap {A.box_gap(ma, mb):.2f}", round(A.containment_ratio(ma, mb), 2) if inter_m else None,
                   meta.get('offered_by'))
            merges.append(row)
            if base(oa['label']) != base(ob['label']):
                cross.append(row)
            ov = ch.get('overlap', 0.0)
            if ov > 0 and ov < 0.5 * ps.total and attrs.get('same_kind') is not True:
                diluted.append(row)
            if not inter_m:
                no_inter.append(row)
    print(f"\n=== Part A [{tag}] weights {weights} attribute channel {'ON' if use_attrs else 'OFF'} ===")
    print(f"  {N} objects, {N * (N - 1) // 2} all-pairs, offered {len(offered)} "
          f"({collections.Counter(m.get('offered_by') for _, _, m in offered)}), excluded {len(excluded)}")
    print(f"  decisions over offered pairs: {dict(dec)}")
    print(f"  MERGES {len(merges)}: cross-base-label {len(cross)}; overlap<50% of total with same_kind!=True "
          f"(GA-328 guard diluted, not lifted) {len(diluted)}; measured boxes NOT intersecting (room+attributes path) {len(no_inter)}")
    for r in merges:
        print("   ", r)
    return merges


m_new = run_engine(NEW_W, True, 'BRANCH')
m_oldw = run_engine(OLD_W, True, 'old weights + attribute channel')
m_pre = run_engine(NEW_W, False, 'pre-ruling engine, no attribute channel')

# ----------------------------------------------------------------------------------------- Part B
print("\n=== Part B: per-detection counterfactual against the OLD run's state (ledger order) ===")
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
        b2 = None
        if b.get('has_bbox_2d'):
            try:
                b2 = [float(v) for v in str(b['bbox_2d']).strip('[]').split()]
            except ValueError:
                b2 = None
        obs[oid] = {'oid': oid, 'label': b['label'], 'box': bnd(b), 'expl': bool(p.get('exploration_mode')),
                    'moved': bool(p.get('robot_has_moved')), 'color': d.get('color', ''),
                    'material': d.get('material', ''), 'description': d.get('description', ''),
                    'cycle': e['cycle_id'], 't': e['recorded_at'], 'idx': idx, 'bbox_2d': b2}
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

# the world state just before each cycle: replay the ledger up to the cycle's recorded_at
state = {}
ev_i = 0
by_cycle = collections.defaultdict(list)
for o in obs.values():
    by_cycle[o['cycle']].append(o)
cycles = sorted(by_cycle, key=lambda c: cycle_t[c])
TRANS_D, STAB_T, FALLBACK_R, EXPL_IOU = 0.35, 3.0, 1.0, 0.10


def score_w(weights, det, s):
    CFG['similarity'].clear()
    CFG['similarity'].update(weights)
    sc, ev = lost_similarity_detailed(world2vec, base(det['label']), base(s['label']), det['color'], s['color'],
                                      det['material'], s['material'], emb(det['description']), emb(s['description']))
    return sc, ev['optional_count']


tot = collections.Counter()
move_rows, absorb_rows, absorbed_then_moved = [], [], []
for cyc in cycles:
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
            state[oid] = {'label': o['label'], 'box': o['box'], 'color': o['color'], 'material': o['material'],
                          'description': o['description'], 'created': events[ev_i - 1][0]}
        elif oid in state:
            state[oid]['box'] = o['box']
    dets = sorted(by_cycle[cyc], key=lambda o: o['idx'])
    if not dets or not dets[0]['expl']:
        continue
    for W, tag in ((NEW_W, 'new'), (OLD_W, 'old')):
        cycle_objs = set()   # same-cycle veto population (object ids absorbed this cycle)
        for det in dets:
            tot[(tag, 'detections')] += 1
            # 1. check_tracking_transition: stable objects within reach, best score > SIM_T
            best, best_s = None, -1.0
            for oid, s in state.items():
                if t - s['created'] < STAB_T:
                    continue
                d = float(np.linalg.norm(A.box_centroid(det['box']) - A.box_centroid(s['box'])))
                if d > half_diag(s['box']) + FALLBACK_R + half_diag(det['box']):
                    continue
                sc, n_ev = score_w(W, det, s)
                if sc > SIM_T and sc > best_s and n_ev >= 1:
                    best, best_s, best_d = oid, sc, d
            moved = False
            if best is not None:
                v = iou3(det['box'], state[best]['box'])
                if best_d > TRANS_D and v < EXPL_IOU:
                    moved = True
                    tot[(tag, 'transition_moved')] += 1
                    row = (cyc[:6], det['label'], state[best]['label'], round(best_s, 3), round(best_d, 2), round(v, 2),
                           f"{det['color']}/{det['material']}", f"{state[best]['color']}/{state[best]['material']}")
                    move_rows.append((tag, row))
                    if best in cycle_objs:
                        absorbed_then_moved.append((tag, row))
            if moved:
                continue
            # 2. exploration loop: veto, gap locality on the (measured, state) box, score > SIM_T, FIRST match
            hit = None
            for oid, s in state.items():
                if oid in cycle_objs:
                    tot[(tag, 'vetoed_candidate')] += 1
                    continue
                if A.geometry_compatible(det['box'], s['box'], GAP) is not True:
                    continue
                sc, n_ev = score_w(W, det, s)
                if sc > SIM_T:
                    hit = (oid, sc)
                    break
            if hit:
                cycle_objs.add(hit[0])
                s = state[hit[0]]
                v = iou3(det['box'], s['box'])
                kind = 'iou>=0.10' if v >= 0.10 else ('0<iou<0.10' if v > 0 else 'no intersection')
                tot[(tag, 'absorbed', kind)] += 1
                absorb_rows.append((tag, cyc[:6], det['label'], s['label'], round(hit[1], 3), kind,
                                    round(A.box_gap(det['box'], s['box']), 2)))
            else:
                tot[(tag, 'new_object')] += 1

for tag in ('new', 'old'):
    print(f"\n  weights={tag}: " + ", ".join(f"{k[1:] if len(k) > 2 else k[1]}={v}" for k, v in sorted(tot.items(), key=str) if k[0] == tag))
print(f"\n  transition MOVED rows (new weights) {sum(1 for t_, _ in move_rows if t_ == 'new')} / (old) "
      f"{sum(1 for t_, _ in move_rows if t_ == 'old')}: (cycle, det label, object label, score, centre m, IoU, det c/m, obj c/m)")
for t_, r in move_rows:
    if t_ == 'new':
        print("   ", r)
print(f"\n  absorbed by an earlier detection of the SAME cycle, then declared MOVED by a later one: "
      f"{[(t_, r[:3]) for t_, r in absorbed_then_moved]}")
print("\n  absorptions with NO intersection on the measured box (armed by the gap locality), new weights:")
for r in absorb_rows:
    if r[0] == 'new' and r[5] == 'no intersection':
        print("   ", r)
