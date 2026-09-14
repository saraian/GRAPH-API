"""Choose the C form from the data. Runs under mlspaces_310 with the REAL MiniLM (offline cache).

Positives (same physical object, GT-free evidence):
  P1  the 15 applied merges (keeper/discard attributes from the ledger's observations)
  P2  same-label pairs refused on similarity whose final fused boxes overlap (31)
  P3  association class A and B (same-label object overlapping the new detection at admission)
Negatives (different physical objects):
  N1  same-label final objects whose boxes do not overlap and whose centres are > 1.5 m apart
  N2  different-label final objects whose boxes overlap (a lamp on a table, a pillow on a sofa)
Forms:
  F0  current: 0.05/0.30/0.15/0.50, raw D
  F1  soft D: D>=0.5 -> 1, D<=0.3 -> 0, linear between; weights unchanged
  F2  weights 0.05/0.45/0.30/0.20, raw D
  F3  soft D with weights F2
"""
import collections
import json
import os
import sys
import types

import numpy as np

# --- webcolors stand-in: CSS3 names present in this run; unknown names raise like the real one
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
from nlp_utils import (_known, color_similarity_rgb, cosine_similarity, get_embedding,  # noqa: E402
                       semantic_similarity, world2vec)

assert world2vec.st_model is not None, "real SentenceTransformer did not load"
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


def cen(a):
    return (a[:3] + a[3:]) / 2


def base(label):
    return label.split('#')[0]


# captured observations (attributes per observation_id)
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
        obs[oid] = {'label': b['label'], 'box': arr(b), 'expl': bool(p.get('exploration_mode')),
                    'color': d.get('color', ''), 'material': d.get('material', ''),
                    'description': d.get('description', '')}
# object -> the observation that created it (attributes at admission)
created = {}
for line in open(R + 'bundle/mutation_receipts.jsonl'):
    r = json.loads(line)
    if r['mutation_state'] == 'applied_in_memory' and r['operation'] == 'add':
        created[r['object_id']] = (r.get('observation') or {}).get('observation_id')
final = {o['object_id']: o for o in json.load(open(R + 'bundle/persistent_perception.json'))}
rows = [json.loads(line) for line in open(R + 'bundle/hook_decisions.jsonl')]


def attrs_final(oid):
    o = final[oid]
    return {'label': o['label'], 'color': o['color'], 'material': o['material'], 'description': o['description']}


def attrs_created(oid):
    o = obs.get(created.get(oid))
    return None if o is None else {'label': o['label'], 'color': o['color'], 'material': o['material'],
                                   'description': o['description']}


def terms(a, b):
    """The four raw terms exactly as lost_similarity_detailed measures them (None = absent)."""
    t = {'label': semantic_similarity(world2vec, base(a['label']), base(b['label'])), 'color': None,
         'material': None, 'description': None}
    if _known(a['color']) and _known(b['color']):
        t['color'] = color_similarity_rgb(a['color'], b['color'], world2vec)
    if _known(a['material']) and _known(b['material']):
        t['material'] = semantic_similarity(world2vec, a['material'], b['material'])
    ea, eb = get_embedding(world2vec, a['description']), get_embedding(world2vec, b['description'])
    if ea is not None and eb is not None and len(ea) == len(eb) and len(ea) > 0:
        t['description'] = float(cosine_similarity(ea, eb))
    return t


def soft(d):
    if d is None:
        return None
    return float(np.clip((d - 0.3) / 0.2, 0.0, 1.0))


FORMS = {
    'F0 current 05/30/15/50 rawD': ((0.05, 0.30, 0.15, 0.50), False),
    'F1 softD  05/30/15/50': ((0.05, 0.30, 0.15, 0.50), True),
    'F2 rawD   05/45/30/20': ((0.05, 0.45, 0.30, 0.20), False),
    'F3 softD  05/45/30/20': ((0.05, 0.45, 0.30, 0.20), True),
}


def score(t, weights, use_soft):
    wl, wc, wm, wd = weights
    parts = [(wl, t['label'])]
    if t['color'] is not None:
        parts.append((wc, t['color']))
    if t['material'] is not None:
        parts.append((wm, t['material']))
    if t['description'] is not None:
        parts.append((wd, soft(t['description']) if use_soft else t['description']))
    w = sum(x for x, _ in parts)
    return sum(x * v for x, v in parts) / w


pos, neg = [], []
# P1 applied merges: keeper (final attrs) vs discard (attrs at creation)
for r in rows:
    if r['kind'] == 'merge' and not r.get('dry_run'):
        a = attrs_final(r['object']) if r['object'] in final else attrs_created(r['object'])
        b = attrs_created(r['merged_from'])
        if a and b:
            pos.append(('P1 merge', a, b))
# P2 same-label overlapping pairs refused on similarity
seen = set()
for r in rows:
    if r['kind'] != 'merge_refused' or r['reason'] != 'similarity':
        continue
    p = tuple(sorted((r['object'], r['candidate'])))
    if p in seen or p[0] not in final or p[1] not in final:
        continue
    seen.add(p)
    oa, ob = final[p[0]], final[p[1]]
    if base(oa['label']) != base(ob['label']):
        continue
    if iou(arr(oa['fused_bbox']), arr(ob['fused_bbox'])) > 0:
        pos.append(('P2 refused-overlap', attrs_final(p[0]), attrs_final(p[1])))
# P3 association class A/B: replay adds against existing same-label overlapping objects
state = {}
events = []
for line in open(R + 'bundle/mutation_receipts.jsonl'):
    r = json.loads(line)
    if r['mutation_state'] == 'applied_in_memory':
        events.append((r['recorded_at'], r['mutation_sequence'], r['operation'], r['object_id'],
                       (r.get('observation') or {}).get('observation_id')))
for r in rows:
    if r['kind'] == 'merge' and not r.get('dry_run'):
        events.append((r['t'], 10**9, 'merge', r['merged_from'], None))
    elif r['kind'] == 'disappearance_removal':
        events.append((r['t'], 10**9, 'delete', r.get('object'), None))
events.sort(key=lambda e: (e[0], e[1]))
for t, seq, op, oid, obs_id in events:
    if op in ('merge', 'delete'):
        state.pop(oid, None)
        continue
    o = obs.get(obs_id)
    if o is None:
        continue
    if op == 'add':
        for eid, s in state.items():
            if base(s['label']) == base(o['label']) and iou(o['box'], s['box']) > 0:
                pos.append(('P3 assoc A/B', {'label': o['label'], 'color': o['color'], 'material': o['material'],
                                             'description': o['description']},
                            {'label': s['label'], 'color': s['color'], 'material': s['material'],
                             'description': s['description']}))
    state[oid] = {'label': o['label'], 'box': o['box'], 'color': o['color'], 'material': o['material'],
                  'description': o['description']}
# N1 same-label, far apart, no overlap  (fused boxes)
ids = list(final)
for i in range(len(ids)):
    for j in range(i + 1, len(ids)):
        oa, ob = final[ids[i]], final[ids[j]]
        fa, fb = arr(oa['fused_bbox']), arr(ob['fused_bbox'])
        same = base(oa['label']) == base(ob['label'])
        dist = float(np.linalg.norm(cen(fa) - cen(fb)))
        if same and iou(fa, fb) == 0 and dist > 1.5:
            neg.append(('N1 same-label far', attrs_final(ids[i]), attrs_final(ids[j])))
        elif not same and iou(fa, fb) > 0:
            neg.append(('N2 diff-label overlap', attrs_final(ids[i]), attrs_final(ids[j])))
print(f"positives: {collections.Counter(k for k, _, _ in pos)}  negatives: {collections.Counter(k for k, _, _ in neg)}")

pos_t = [(k, terms(a, b)) for k, a, b in pos]
neg_t = [(k, terms(a, b)) for k, a, b in neg]
D = [t['description'] for _, t in pos_t if t['description'] is not None]
Dn = [t['description'] for _, t in neg_t if t['description'] is not None]
print(f"raw description cosine: positives n={len(D)} p10 {np.percentile(D, 10):.2f} median {np.median(D):.2f} "
      f"p90 {np.percentile(D, 90):.2f} | negatives n={len(Dn)} p10 {np.percentile(Dn, 10):.2f} "
      f"median {np.median(Dn):.2f} p90 {np.percentile(Dn, 90):.2f}")
print(f"\n{'form':30s} {'pos>=.85':>9s} {'pos>=.925':>9s} {'neg>=.85':>9s} {'neg>=.925':>9s}  best-thr  acc@best")
for name, (w, s) in FORMS.items():
    ps = np.array([score(t, w, s) for _, t in pos_t])
    ns = np.array([score(t, w, s) for _, t in neg_t])
    best = max(((thr, (np.mean(ps >= thr) + np.mean(ns < thr)) / 2) for thr in np.arange(0.5, 1.0, 0.005)),
               key=lambda x: x[1])
    print(f"{name:30s} {np.mean(ps >= 0.85):9.2f} {np.mean(ps >= 0.925):9.2f} {np.mean(ns >= 0.85):9.2f} "
          f"{np.mean(ns >= 0.925):9.2f}   {best[0]:.3f}   {best[1]:.3f}")
    for k in ('P1 merge', 'P2 refused-overlap', 'P3 assoc A/B'):
        sub = np.array([score(t, w, s) for kk, t in pos_t if kk == k])
        if len(sub):
            print(f"      {k:20s} n={len(sub):3d} median {np.median(sub):.3f}  >=.85 {np.mean(sub >= 0.85):.2f}  >=.925 {np.mean(sub >= 0.925):.2f}")
    for k in ('N1 same-label far', 'N2 diff-label overlap'):
        sub = np.array([score(t, w, s) for kk, t in neg_t if kk == k])
        if len(sub):
            print(f"      {k:20s} n={len(sub):3d} median {np.median(sub):.3f}  >=.85 {np.mean(sub >= 0.85):.2f}  >=.925 {np.mean(sub >= 0.925):.2f}")
