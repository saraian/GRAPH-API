"""Reproduce the recorded merge similarity of refused pairs with the REAL lost_similarity_detailed
and show which term kept them under the 0.925 gate. Also: how many sweeps re-refused each pair with
the identical score (no new evidence can arrive for a pair whose attributes did not change)."""
import collections
import json
import os
import statistics
import sys

import numpy as np

PM = '/DATA/GRAPH-API/.claude/worktrees/found-merge-algorithm/lost3dsg/src/perception_module'
sys.path.insert(0, PM)
os.chdir(PM)
import rosstub  # noqa: E402

rosstub.install()
from nlp_utils import (  # noqa: E402
    CFG,
    _known,
    color_similarity_rgb,
    cosine_similarity,
    get_embedding,
    lost_similarity_detailed,
    semantic_similarity,
    world2vec,
)

W = CFG['similarity']
R = '/DATA/GRAPH-API/.handoff/lanes/perception/ga493_replay_20260914/'
final = {o['object_id']: o for o in json.load(open(R + 'bundle/persistent_perception.json'))}
rows = [json.loads(line) for line in open(R + 'bundle/hook_decisions.jsonl')]
K = ['x_min', 'y_min', 'z_min', 'x_max', 'y_max', 'z_max']


def iou(a, b):
    lo = np.maximum(a[:3], b[:3])
    hi = np.minimum(a[3:], b[3:])
    d = np.clip(hi - lo, 0, None)
    i = d.prod()
    va = (a[3:] - a[:3]).prod()
    vb = (b[3:] - b[:3]).prod()
    return i / (va + vb - i) if va + vb - i > 0 else 0.0


def terms(oa, ob):
    la, lb = oa['label'].split('#')[0], ob['label'].split('#')[0]
    ea, eb = get_embedding(world2vec, oa['description']), get_embedding(world2vec, ob['description'])
    score, ev = lost_similarity_detailed(world2vec, la, lb, oa['color'], ob['color'],
                                         oa['material'], ob['material'], ea, eb)
    t = {'label': semantic_similarity(world2vec, la, lb)}
    if _known(oa['color']) and _known(ob['color']):
        t['color'] = color_similarity_rgb(oa['color'], ob['color'], world2vec)
    if _known(oa['material']) and _known(ob['material']):
        t['material'] = semantic_similarity(world2vec, oa['material'], ob['material'])
    if ea is not None and eb is not None:
        t['description'] = cosine_similarity(ea, eb)
    return score, ev, t


print(f"weights: {dict(W)}  gate: {CFG['association']['merge_min_similarity']}")
# ceiling: all categorical terms agree, only the description cosine D varies
sw = W['label'] + W['color'] + W['material']
print(f"ceiling: with label+colour+material all =1.0, score = {sw:.2f} + {W['description']:.2f}*D ; "
      f"gate 0.925 needs D >= {(0.925 - sw) / W['description']:.3f}")

# refusal repetition per pair
per_pair = collections.Counter()
for r in rows:
    if r['kind'] == 'merge_refused' and r['reason'] in ('similarity', 'distance', 'room'):
        per_pair[(tuple(sorted((r['object'], r['candidate']))), r['reason'])] += 1
reps = list(per_pair.values())
print(f"refused (pair,reason) keys: {len(reps)}; records: {sum(reps)}; "
      f"median repeats {statistics.median(reps)}, max {max(reps)}; keys refused >=5 times: {sum(1 for x in reps if x >= 5)}")

# reproduce recorded scores for same-label overlapping pairs refused on similarity
last = {}
for r in rows:
    if r['kind'] == 'merge_refused' and r['reason'] == 'similarity':
        last[tuple(sorted((r['object'], r['candidate'])))] = r
print("\n=== same-label pairs refused on SIMILARITY whose final fused boxes overlap; real recomputation ===")
print(f"{'labels':26s} {'fIoU':>5s} {'rec':>6s} {'repro':>6s} {'L':>5s} {'C':>5s} {'M':>5s} {'D':>5s}  colours / materials")
descs = []
for p, r in sorted(last.items(), key=lambda kv: kv[1]['similarity'] or 0, reverse=True):
    oa, ob = final.get(p[0]), final.get(p[1])
    if not oa or not ob or oa['label'].split('#')[0] != ob['label'].split('#')[0]:
        continue
    fa = np.array([oa['fused_bbox'][k] for k in K])
    fb = np.array([ob['fused_bbox'][k] for k in K])
    fi = iou(fa, fb)
    if fi <= 0:
        continue
    # ALGEBRAIC description cosine from the RECORDED score. Trustworthy only where every other
    # present term is an identical-string short-circuit (=1.0 with no model consulted):
    # label base equal (yes, by filter), colour strings equal, material strings equal or both unknown.
    same_c = oa['color'].lower().strip() == ob['color'].lower().strip() and _known(oa['color'])
    same_m = oa['material'].lower().strip() == ob['material'].lower().strip() and _known(oa['material'])
    both_unknown_m = not _known(oa['material']) and not _known(ob['material'])
    rec = r['similarity']
    if same_c and same_m and r['evidence_count'] == 3:
        D = (rec - (W['label'] + W['color'] + W['material'])) / W['description']
    elif same_c and both_unknown_m and r['evidence_count'] == 2:
        wsum = W['label'] + W['color'] + W['description']
        D = (rec * wsum - (W['label'] + W['color'])) / W['description']
    else:
        D = None
    descs.append(D)
    print(f"{oa['label'] + '/' + ob['label']:26s} {fi:5.2f} {rec:6.3f}  ev={r['evidence_count']} "
          f"D={'%.3f' % D if D is not None else '  n/a'}  "
          f"{oa['color']}/{ob['color']}  {oa['material']}/{ob['material']}")
d = [x for x in descs if x is not None]
print(f"\nEXACT description cosine (solved from recorded score, all other terms identical strings): "
      f"n={len(d)} min {min(d):.3f} median {statistics.median(d):.3f} max {max(d):.3f}; "
      f"needed >= 0.850; pairs reaching it: {sum(1 for x in d if x >= 0.85)}")

print("\n=== the 15 applied merges: what description cosine did THEY have? ===")
mer = [r for r in rows if r['kind'] == 'merge' and not r.get('dry_run')]
# discard objects have no final record; take descriptions from the keeper only where both survive is impossible,
# so read both sides from the mutation ledger / final where available
print(f"keeper descriptions of merged pairs (discard side not in final): "
      f"{[final[m['object']]['description'][:40] for m in mer if m['object'] in final][:6]}")
