"""Probe: rule-15 claim on ObjectManagerService.check_tracking_transition (2026-09-14).

Real MiniLM (mlspaces_310, offline), GA-493 bundle, state rebuilt from mutation_receipts + hook_decisions
exactly as v4_real_model_extras.py section 3 does -- with two differences that the claim omits:

Q1. The transition fires ONCE per run: _enter_tracking sets exploration_mode False and nothing sets it
    back (object_manager_6.py: the only `exploration_mode = True` is __init__).  So the walk flips to
    tracking at the first detection that passes the transition gates, and later cycles run the TRACKING
    loop, not the transition scan.  (The reviewer's walk `continue`s and keeps scanning as exploration.)
Q2. Whichever caller hands a detection to modify_existing_object, the SAME handler decides in place vs
    rebuild: object_services._cb_update_object,
    `distance < UPDATE_IN_PLACE_DISTANCE_M (0.5) or iou >= TRACKING_IOU_THRESHOLD (0.3)` -> in place;
    else stable (>= OBJECT_STABILITY_TIMEOUT 3 s) -> rebuild + reset_bbox_fusion + geometry_epoch+1.
Q3. The arming that IS real: in TRACKING cycles the new gap locality routes detections with IoU < 0.3
    into that rebuild branch, which the baseline's tracking gate (locality_ok = IoU >= 0.3) made
    unreachable from the tracking loop.  Counted with its denominator.
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
import nlp_utils  # noqa: E402
from nlp_utils import get_embedding, lost_similarity_detailed, world2vec  # noqa: E402

assert world2vec.st_model is not None
print("component:", type(world2vec.st_model).__name__, "weights", CFG['similarity'])
R = '/DATA/GRAPH-API/.handoff/lanes/perception/ga493_replay_20260914/'
SIM_T = CFG['association']['sim_threshold']
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


OLD_W = {'label': 0.05, 'color': 0.30, 'material': 0.15, 'description': 0.50}
NEW_W = dict(CFG['similarity'])
UPD_IN_PLACE, TRK_IOU, TRANS_D, STAB_T, FALLBACK_R, EXPL_IOU, EXPL_FRAMES = 0.5, 0.3, 0.35, 3.0, 1.0, 0.10, 10


def set_weights(w):
    # nlp_utils may snapshot the weights at import; set both places so either read path sees them.
    CFG['similarity'].clear()
    CFG['similarity'].update(w)
    for name in dir(nlp_utils):
        v = getattr(nlp_utils, name)
        if isinstance(v, dict) and set(v) == {'label', 'color', 'material', 'description'}:
            v.clear()
            v.update(w)
    for key, attr in (('label', 'W_LABEL'), ('color', 'W_COLOR'), ('material', 'W_MATERIAL'),
                      ('description', 'W_DESCRIPTION')):
        if hasattr(nlp_utils, attr):
            setattr(nlp_utils, attr, w[key])


def score(det, s):
    sc, ev = lost_similarity_detailed(world2vec, base(det['label']), base(s['label']), det['color'], s['color'],
                                      det['material'], s['material'], emb(det['description']), emb(s['description']))
    return sc, ev['optional_count']


obs, cycle_t, cycle_expl = {}, {}, {}
for line in open(R + 'capture/consumer/events.jsonl'):
    e = json.loads(line)
    if e['kind'] != 'consumer_pair':
        continue
    p = e['payload']
    cycle_t[e['cycle_id']] = e['recorded_at']
    cycle_expl[e['cycle_id']] = bool(p.get('exploration_mode'))
    descs = {(d.get('observation') or {}).get('observation_id'): d for d in p['descriptions']['descriptions']}
    for idx, b in enumerate(p['bboxes']['boxes']):
        oid = (b.get('observation') or {}).get('observation_id')
        d = descs.get(oid, {})
        obs[oid] = {'label': b['label'], 'box': bnd(b), 'color': d.get('color', ''), 'material': d.get('material', ''),
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
by_cycle = collections.defaultdict(list)
for o in obs.values():
    by_cycle[o['cycle']].append(o)
cycles = sorted(by_cycle, key=lambda c: cycle_t[c])


def run(weights, single_shot=True):
    set_weights(weights)
    state = collections.OrderedDict()
    ev_i = 0
    exploration = True
    first_transition = None
    expl_frames = 0
    n_expl_det = n_trk_det = 0
    would_transition = []
    trk_absorbed, trk_rebuild, trk_inplace = [], [], []
    for ci, cyc in enumerate(cycles):
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
        if not dets:
            continue
        if exploration:
            expl_frames += 1
        cycle_objs = set()
        for det in dets:
            if exploration:
                n_expl_det += 1
                best, best_s, best_d = None, -1.0, None
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
                        upd = ('in_place' if (best_d < UPD_IN_PLACE or v >= TRK_IOU) else 'REBUILD+reset_fusion')
                        would_transition.append((ci, cyc[:6], det['label'], state[best]['label'], round(best_d, 2),
                                                 round(v, 3), round(g, 2), upd))
                        if first_transition is None:
                            first_transition = would_transition[-1]
                        if single_shot:
                            exploration = False   # _enter_tracking: never returns to True
                        cycle_objs.add(best)
                        continue                  # the transition branch `continue`s
                # exploration loop (gap locality): in-place fusion write, never a rebuild
                for oid, s in state.items():
                    if oid in cycle_objs or A.geometry_compatible(det['box'], s['box'], GAP) is not True:
                        continue
                    sc, _ = score(det, s)
                    if sc > SIM_T:
                        cycle_objs.add(oid)
                        break
            else:
                n_trk_det += 1
                best, best_s = None, -1.0
                for oid, s in state.items():
                    if oid in cycle_objs or A.geometry_compatible(det['box'], s['box'], GAP) is not True:
                        continue
                    sc, _ = score(det, s)
                    if sc > SIM_T and sc > best_s:
                        best, best_s = oid, sc
                if best is not None:
                    s = state[best]
                    cycle_objs.add(best)
                    d = float(np.linalg.norm(A.box_centroid(det['box']) - A.box_centroid(s['box'])))
                    v = iou3(det['box'], s['box'])
                    row = (ci, cyc[:6], det['label'], s['label'], round(d, 2), round(v, 3),
                           round(A.box_gap(det['box'], s['box']), 2))
                    trk_absorbed.append(row)
                    if d < UPD_IN_PLACE or v >= TRK_IOU:
                        trk_inplace.append(row)
                    elif t - s['created'] >= STAB_T:
                        trk_rebuild.append(row)
                    else:
                        trk_inplace.append(row + ('unstable->in place',))
        if exploration and expl_frames >= EXPL_FRAMES:
            exploration = False
            if first_transition is None:
                first_transition = ('frame_limit', ci)
    return dict(first_transition=first_transition, would_transition=would_transition, n_expl_det=n_expl_det,
                n_trk_det=n_trk_det, trk_absorbed=trk_absorbed, trk_rebuild=trk_rebuild, trk_inplace=trk_inplace)


print(f"\ncycles with detections: {sum(1 for c in cycles if by_cycle[c])} of {len(cycles)}; detections {len(obs)}; "
      f"old run: exploration cycles {sum(cycle_expl.values())}, mode_transition once ('a stable object moved', sofa#1 0.933 m)")
for name, w in (('OLD weights', OLD_W), ('BRANCH weights', NEW_W)):
    print(f"\n== {name} {w}")
    r0 = run(w, single_shot=False)
    print(f"   reviewer's walk (mode never flips): detections passing the transition gates = {len(r0['would_transition'])}"
          f" of {r0['n_expl_det']} exploration-window detections")
    for row in r0['would_transition']:
        print("      ", row)
    r = run(w, single_shot=True)
    print(f"   single-shot walk (real semantics): exploration-window detections {r['n_expl_det']}, tracking detections {r['n_trk_det']}")
    print(f"   the ONE transition: {r['first_transition']}")
    print(f"   TRACKING loop under gap locality: absorbed {len(r['trk_absorbed'])} of {r['n_trk_det']}; "
          f"in place {len(r['trk_inplace'])}; REBUILD branch (stable, d>=0.5, iou<0.3): {len(r['trk_rebuild'])}")
    for row in r['trk_rebuild']:
        print("      rebuild:", row)
    print("   baseline tracking gate was IoU >= 0.3, and the handler's `iou >= 0.3` is in place: rebuild unreachable from the tracking loop (0 by construction)")
