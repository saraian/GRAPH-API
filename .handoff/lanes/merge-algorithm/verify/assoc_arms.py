"""Association ablation arms on the GA-493 bundle, GT-free, real models offline.

One factor per arm (perception lane's critique 2026-09-14): the 44-of-119 figure bundled the
gate shape, the attribute weights and the same-cycle veto. Arms:
  gate     'iou'  = IoU >= 0.10 (exploration) / 0.30 (tracking) on the measured box (old)
           'gap:g' = boxes intersect or largest per-axis gap <= g metres (new, association.geometry_compatible)
  weights  label/colour/material/description
  label    'mini' = MiniLM on the bare word (production semantic_similarity)
           'owl'  = OWLv2 text tower on "a photo of a {label}" (the detector's query space)
  veto     same-cycle veto on/off
Colour, material and description terms are production's (nlp_utils) with the real MiniLM.
World state is rebuilt from the mutation ledger (adds/updates as recorded), so it follows what
the callback actually did; the candidate's attributes are those at its creation/last update.
"""
import collections
import json
import os
import sys
import types

import numpy as np
import torch

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
os.environ.setdefault('HF_HUB_OFFLINE', '1')
os.environ.setdefault('TRANSFORMERS_OFFLINE', '1')
from association import _as_bounds, geometry_compatible  # noqa: E402
from nlp_utils import (_known, color_similarity_rgb, cosine_similarity, get_embedding,  # noqa: E402
                       semantic_similarity, world2vec)

assert type(world2vec.st_model).__name__ == 'SentenceTransformer'
R = '/DATA/GRAPH-API/.handoff/lanes/perception/ga493_replay_20260914/'
K = ['x_min', 'y_min', 'z_min', 'x_max', 'y_max', 'z_max']


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


def base(label):
    return label.split('#')[0].strip().lower()


# --- label instruments -------------------------------------------------------------------
_owl = {}
from transformers import Owlv2Model, Owlv2Processor  # noqa: E402
_proc = Owlv2Processor.from_pretrained('google/owlv2-base-patch16-ensemble')
_owlm = Owlv2Model.from_pretrained('google/owlv2-base-patch16-ensemble').eval()


def owl_vec(label):
    t = 'a photo of a ' + base(label).replace('_', ' ')
    if t not in _owl:
        with torch.no_grad():
            v = _owlm.get_text_features(**_proc(text=[t], return_tensors='pt'))[0].numpy()
        _owl[t] = v / (np.linalg.norm(v) + 1e-9)
    return _owl[t]


def label_sim(a, b, instrument):
    if base(a) == base(b):
        return 1.0
    if instrument == 'mini':
        return semantic_similarity(world2vec, base(a), base(b))
    return max(0.0, float(owl_vec(a) @ owl_vec(b)))


def score(det, cand, W, instrument):
    """lost_similarity_detailed's formula with a pluggable label term and weights."""
    terms = [(W['label'], label_sim(det['label'], cand['label'], instrument))]
    n = 0
    if _known(det['color']) and _known(cand['color']):
        terms.append((W['color'], color_similarity_rgb(det['color'], cand['color'], world2vec)))
        n += 1
    if _known(det['material']) and _known(cand['material']):
        terms.append((W['material'], semantic_similarity(world2vec, det['material'], cand['material'])))
        n += 1
    ea, eb = det['emb'], cand['emb']
    if ea is not None and eb is not None and len(ea) == len(eb) and len(ea) > 0:
        terms.append((W['description'], float(cosine_similarity(ea, eb))))
        n += 1
    w = sum(x for x, _ in terms)
    return sum(x * v for x, v in terms) / w, n


# --- data ----------------------------------------------------------------------------------
obs = {}
for line in open(R + 'capture/consumer/events.jsonl'):
    e = json.loads(line)
    if e['kind'] != 'consumer_pair':
        continue
    p = e['payload']
    descs = {(d.get('observation') or {}).get('observation_id'): d for d in p['descriptions']['descriptions']}
    for b in p['bboxes']['boxes']:
        oid = (b.get('observation') or {}).get('observation_id')
        d = descs.get(oid, {})
        obs[oid] = {'label': b['label'], 'box': arr(b), 'expl': bool(p.get('exploration_mode')), 'cycle': e['cycle_id'],
                    'color': d.get('color', ''), 'material': d.get('material', ''), 'description': d.get('description', ''),
                    'emb': get_embedding(world2vec, d.get('description', ''))}
events = []
for line in open(R + 'bundle/mutation_receipts.jsonl'):
    r = json.loads(line)
    if r['mutation_state'] == 'applied_in_memory':
        events.append((r['recorded_at'], r['mutation_sequence'], r['operation'], r['object_id'],
                       (r.get('observation') or {}).get('observation_id')))
for line in open(R + 'bundle/hook_decisions.jsonl'):
    r = json.loads(line)
    if r['kind'] == 'merge' and not r.get('dry_run'):
        events.append((r['t'], 10**9, 'merge', r['merged_from'], None))
    elif r['kind'] == 'disappearance_removal':
        events.append((r['t'], 10**9, 'delete', r.get('object'), None))
events.sort(key=lambda e: (e[0], e[1]))

# --- one replay per arm ------------------------------------------------------------------
OLD_W = {'label': 0.05, 'color': 0.30, 'material': 0.15, 'description': 0.50}
NEW_W = {'label': 0.05, 'color': 0.45, 'material': 0.30, 'description': 0.20}


def with_label(w_label):
    """Raise the label weight; scale the other three proportionally from NEW_W."""
    rest = 1.0 - w_label
    tot = NEW_W['color'] + NEW_W['material'] + NEW_W['description']
    return {'label': w_label, 'color': NEW_W['color'] / tot * rest, 'material': NEW_W['material'] / tot * rest,
            'description': NEW_W['description'] / tot * rest}


ARMS = [
    ('A0 old gate, old weights, no veto (control)', 'iou', OLD_W, 'mini', False),
    ('A1 gap 0.3, old weights, no veto (gate only)', 'gap:0.3', OLD_W, 'mini', False),
    ('A2 old gate, new weights, veto (weights only)', 'iou', NEW_W, 'mini', True),
    ('A3 gap 0.3, new weights, veto (the 44)', 'gap:0.3', NEW_W, 'mini', True),
    ('A4 gap 0.2, new weights, veto', 'gap:0.2', NEW_W, 'mini', True),
    ('A5 gap 0.5, new weights, veto', 'gap:0.5', NEW_W, 'mini', True),
    ('A6 gap 0.8, new weights, veto', 'gap:0.8', NEW_W, 'mini', True),
    ('A7 gap 0.3, label .25 (MiniLM), veto', 'gap:0.3', with_label(0.25), 'mini', True),
    ('A8 gap 0.3, label .40 (MiniLM), veto', 'gap:0.3', with_label(0.40), 'mini', True),
    ('A9 gap 0.3, new weights, OWLv2 label, veto', 'gap:0.3', NEW_W, 'owl', True),
    ('A10 gap 0.3, label .25 (OWLv2), veto', 'gap:0.3', with_label(0.25), 'owl', True),
    ('A11 gap 0.3, label .40 (OWLv2), veto', 'gap:0.3', with_label(0.40), 'owl', True),
]


PAIRS = {}   # arm name -> [[new object_id, absorbing object_id, score, new label, target label], ...]


def run(gate, W, instrument, veto_on, arm_name=None):
    state, cur, veto = {}, None, set()
    out = collections.Counter()
    diff = []
    pairs = PAIRS.setdefault(arm_name, [])
    for t, seq, op, oid, obs_id in events:
        if op in ('merge', 'delete'):
            state.pop(oid, None)
            continue
        o = obs[obs_id]
        if o['cycle'] != cur:
            cur, veto = o['cycle'], set()
        if op != 'add':
            s = state.get(oid)
            if s is not None:
                s['box'] = o['box']
            veto.add(oid)
            continue
        best = None
        for eid, s in state.items():
            if veto_on and eid in veto:
                continue
            if gate == 'iou':
                ok = iou(o['box'], s['box']) >= (0.10 if o['expl'] else 0.30)
            else:
                ok = geometry_compatible(_as_bounds(dict(zip(K, o['box']))), _as_bounds(dict(zip(K, s['box']))),
                                         float(gate.split(':')[1])) is True
            if not ok:
                continue
            sc, n = score(o, s, W, instrument)
            if sc > 0.85 and (best is None or sc > best[1]):
                best = (eid, sc)
        if best is None:
            out['new'] += 1
        else:
            same = base(state[best[0]]['label']) == base(o['label'])
            out['assoc_same' if same else 'assoc_diff'] += 1
            if not same:
                diff.append((o['label'], state[best[0]]['label'], round(best[1], 3)))
            # oid is the object the run CREATED for this detection; best[0] is the object the arm
            # would have absorbed it into -- the pair the evaluator can score against GT
            pairs.append([oid, best[0], round(best[1], 4), o['label'], state[best[0]]['label']])
            veto.add(best[0])
        state[oid] = {'label': o['label'], 'box': o['box'], 'color': o['color'], 'material': o['material'],
                      'description': o['description'], 'emb': o['emb']}
        veto.add(oid)
    return out, diff


N = sum(1 for e in events if e[2] == 'add')
print(f"adds: {N}; arms: {len(ARMS)}; label instruments: MiniLM raw word, OWLv2 'a photo of a {{label}}'\n")
print(f"{'arm':50s} {'same':>5s} {'diff':>5s} {'new':>5s}  different-label absorptions")
for name, gate, W, inst, veto_on in ARMS:
    out, diff = run(gate, W, inst, veto_on, arm_name=name)
    print(f"{name:50s} {out['assoc_same']:5d} {out['assoc_diff']:5d} {out['new']:5d}  "
          f"{', '.join(f'{a}->{b}@{s}' for a, b, s in diff)}")
# per-arm association pairs for evaluator-side scoring (GT read only AFTER this file exists)
PAIRS_OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'assoc_arms_pairs.json')
with open(PAIRS_OUT, 'w') as fh:
    json.dump({'schema': 'arm -> [[new_object_id, absorbing_object_id, score, new_label, target_label], ...]',
               'source': 'GA-493 bundle mutation ledger + consumer capture; ids are the run\'s object_ids',
               'arms': PAIRS}, fh, indent=1)
print(f"\npairs written: {PAIRS_OUT}")
