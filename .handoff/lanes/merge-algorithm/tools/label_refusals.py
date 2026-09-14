"""Label each refused/merged pair in the GA-493 bundle by whether both sides sit on the
same GT object. GT read only here, after the fact. Boxes come from persistent_perception.json,
so a pair whose side was later merged away has no box and is counted as 'no_box'."""
import collections
import json
import statistics

import numpy as np

R = '/DATA/GRAPH-API/.handoff/lanes/perception/ga493_replay_20260914/'
gt = json.load(open(R + 'evaluator/hm3d_00824_run_gt.json'))
truth = []
for o in gt['ground_truth_objects']:
    lo, hi = np.array(o['aabb_min_m']), np.array(o['aabb_max_m'])
    box = np.array([-hi[2], -hi[0], lo[1], -lo[2], -lo[0], hi[1]])
    truth.append((o['object_id'], o['category_name'], box))
final = {o['object_id']: o for o in json.load(open(R + 'bundle/persistent_perception.json'))}
K = ['x_min', 'y_min', 'z_min', 'x_max', 'y_max', 'z_max']


def cen(b):
    return (b[:3] + b[3:]) / 2


def iou(a, b):
    lo = np.maximum(a[:3], b[:3])
    hi = np.minimum(a[3:], b[3:])
    d = np.clip(hi - lo, 0, None)
    i = d.prod()
    va = (a[3:] - a[:3]).prod()
    vb = (b[3:] - b[:3]).prod()
    return i / (va + vb - i) if va + vb - i > 0 else 0.0


def best_gt(oid, which):
    o = final.get(oid)
    if o is None:
        return None
    b = np.array([o[which][k] for k in K])
    c = cen(b)
    base = o['label'].split('#')[0]
    same = [(gid, cat, tb) for gid, cat, tb in truth if cat == base]
    pool = same if same else truth
    # nearest centre among same-category GT (falls back to all GT)
    gid, cat, tb = min(pool, key=lambda t: np.linalg.norm(cen(t[2]) - c))
    return gid, float(np.linalg.norm(cen(tb) - c)), float(iou(b, tb)), bool(same)


rows = [json.loads(line) for line in open(R + 'bundle/hook_decisions.jsonl')]
ref = [r for r in rows if r['kind'] == 'merge_refused']
mer = [r for r in rows if r['kind'] == 'merge' and not r.get('dry_run')]
# distinct pairs, keep LAST record per (pair, reason)
last = {}
for r in ref:
    key = (tuple(sorted((r['object'], r['candidate']))), r['reason'])
    last[key] = r
print(f"refusal records {len(ref)} -> distinct (pair,reason) {len(last)}; "
      f"distinct pairs {len(set(k[0] for k in last))}")
for which in ('fused_bbox', 'bbox'):
    print(f"\n=== side boxes from final {which} ; GT = nearest same-category centre ===")
    tab = collections.defaultdict(collections.Counter)
    dist_same, dist_diff, sim_same, sim_diff = [], [], [], []
    for (pair, reason), r in last.items():
        ga, gb = best_gt(pair[0], which), best_gt(pair[1], which)
        base_same = r['a_label'].split('#')[0] == r['b_label'].split('#')[0]
        if ga is None or gb is None:
            tab[reason]['no_box'] += 1
            continue
        cls = 'same_gt' if ga[0] == gb[0] else 'diff_gt'
        tab[reason][cls] += 1
        if base_same:
            tab[reason][cls + '_samelabel'] += 1
        if reason == 'distance' and base_same:
            (dist_same if cls == 'same_gt' else dist_diff).append(r['distance'])
        if reason == 'similarity' and base_same:
            (sim_same if cls == 'same_gt' else sim_diff).append(r['similarity'])
    for reason in ('distance', 'similarity', 'room', 'already_condemned'):
        print(f"  {reason:18s} {dict(tab[reason])}")
    if dist_same:
        print(f"  same-label DISTANCE refusals, same GT: n={len(dist_same)} "
              f"median {statistics.median(dist_same):.2f} m, max {max(dist_same):.2f}")
    if dist_diff:
        print(f"  same-label DISTANCE refusals, diff GT: n={len(dist_diff)} "
              f"median {statistics.median(dist_diff):.2f} m")
    if sim_same:
        print(f"  same-label SIMILARITY refusals, same GT: n={len(sim_same)} median "
              f"{statistics.median(sim_same):.3f} min {min(sim_same):.3f} max {max(sim_same):.3f}")
    if sim_diff:
        print(f"  same-label SIMILARITY refusals, diff GT: n={len(sim_diff)} median "
              f"{statistics.median(sim_diff):.3f} min {min(sim_diff):.3f} max {max(sim_diff):.3f}")
    print("  applied merges (keeper -> nearest GT; discard has no final box):")
    for m in mer:
        g = best_gt(m['object'], which)
        if g:
            print(f"    {m['keeper_label']:14s} <- {m['discarded_label']:14s} "
                  f"sim={m['similarity']:.3f} keeper->GT {g[0]} d={g[1]:.2f} iou={g[2]:.3f}")
        else:
            print(f"    {m['keeper_label']} keeper not in final")
# STRICT: GT-free mutual geometry of the refused pair, plus strict same-GT (both within 1.0 m)
print("\n=== STRICT: refused pairs by mutual geometry of the two FINAL fused boxes ===")
strict = collections.defaultdict(collections.Counter)
examples = collections.defaultdict(list)
for (pair, reason), r in last.items():
    oa, ob = final.get(pair[0]), final.get(pair[1])
    if oa is None or ob is None:
        continue
    ba = np.array([oa['fused_bbox'][k] for k in K])
    bb = np.array([ob['fused_bbox'][k] for k in K])
    mi = iou(ba, bb)
    md = float(np.linalg.norm(cen(ba) - cen(bb)))
    base_same = r['a_label'].split('#')[0] == r['b_label'].split('#')[0]
    ga, gb = best_gt(pair[0], 'fused_bbox'), best_gt(pair[1], 'fused_bbox')
    strict_same = ga[0] == gb[0] and ga[1] <= 1.0 and gb[1] <= 1.0
    strict[reason]['pairs'] += 1
    if base_same:
        strict[reason]['samelabel'] += 1
        if mi > 0:
            strict[reason]['samelabel_overlap'] += 1
            examples[reason].append((r['a_label'], r['b_label'], round(mi, 3), round(md, 2),
                                     round(r.get('similarity') or -1, 3), round(r.get('distance') or -1, 2),
                                     'GT=' + ga[0] if strict_same else 'GT?'))
        if strict_same:
            strict[reason]['samelabel_strictGT'] += 1
    if mi > 0:
        strict[reason]['any_overlap'] += 1
for reason in ('distance', 'similarity', 'room'):
    print(f"  {reason:12s} {dict(strict[reason])}")
    for e in sorted(examples[reason], key=lambda e: -e[2])[:12]:
        print(f"      {e}")
print("\n=== GT categories that look like cabinets/counters ===")
cats = collections.Counter(c for _, c, _ in truth)
print({c: n for c, n in cats.items() if any(w in c for w in ('cabinet', 'cupboard', 'counter', 'drawer', 'shelf', 'chair', 'table'))})
# how many final objects share a GT (duplicates surviving)
print("\n=== final objects per nearest same-category GT (fused_bbox) ===")
byg = collections.defaultdict(list)
for oid in final:
    g = best_gt(oid, 'fused_bbox')
    byg[g[0]].append((final[oid]['label'], round(g[1], 2), round(g[2], 3)))
dups = {g: v for g, v in byg.items() if len(v) > 1}
print(f"GT ids claimed by >1 final object: {len(dups)} ; objects in those groups: "
      f"{sum(len(v) for v in dups.values())} of {len(final)}")
for g, v in sorted(dups.items(), key=lambda kv: -len(kv[1]))[:15]:
    print(f"  {g:22s} x{len(v)}: {v}")
