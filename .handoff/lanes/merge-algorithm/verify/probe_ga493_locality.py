"""MEASURED on the GA-493 final objects: does the locality test read the same box everywhere?

generate_candidates (merge broad phase) and WorldModel.candidates (association broad phase) read the
MEASURED bbox; the merge gate (_cb_merge_objects evidence arm) and locality_ok read the FUSED box when
one exists. This counts how far apart those two boxes are on the 104 final objects and how many
final-object pairs the fused test accepts that the measured-box candidate generation never offers.
"""
import itertools
import json
import sys

import numpy as np

PM = '/DATA/GRAPH-API/.claude/worktrees/found-merge-algorithm/lost3dsg/src/perception_module'
sys.path.insert(0, PM)
import association as A  # noqa: E402

R = '/DATA/GRAPH-API/.handoff/lanes/perception/ga493_replay_20260914/bundle/'
objs = json.load(open(R + 'persistent_perception.json'))
GAP = 0.3
n = len(objs)
print(f"objects: {n}; with fused_bbox: {sum(1 for o in objs if o.get('fused_bbox'))}")

# 1. fused overhang beyond the measured box, per object
overs, inside = [], 0
for o in objs:
    m, f = A._as_bounds(o['bbox']), A._as_bounds(o['fused_bbox'])
    over = max(m[0] - f[0], f[1] - m[1], m[2] - f[2], f[3] - m[3], m[4] - f[4], f[5] - m[5])
    overs.append(over)
    if not (f[0] <= m[0] and f[1] >= m[1] and f[2] <= m[2] and f[3] >= m[3] and f[4] <= m[4] and f[5] >= m[5]):
        inside += 1
overs = np.array(overs)
print(f"fused overhang beyond measured (max over 6 faces): median {np.median(overs):.3f} m, "
      f"p90 {np.percentile(overs, 90):.3f}, max {overs.max():.3f}; > {GAP} m: {(overs > GAP).sum()}/{n}; "
      f"> {2 * GAP} m (WorldModel.candidates expands by 2*margin): {(overs > 2 * GAP).sum()}/{n}; "
      f"fused NOT a superset of measured: {inside}/{n}")

# 2. pairs of final objects: fused-gap test vs measured-box candidate generation
built = [A.AssocObject(o['object_id'], bbox=o['bbox'], label=o['label'], source=o) for o in objs]
ctx = A.AssocContext(map_volume_m3=300.0, n_rooms=6, cost_ratio=20.0, use_ontology=False, locality_gap_m=GAP)
offered, excluded = A.generate_candidates(built, ctx)
offered_keys = {frozenset((a.object_id, b.object_id)) for a, b, _ in offered}
by = {}
for _, _, meta in offered:
    by[meta['offered_by']] = by.get(meta['offered_by'], 0) + 1
pairs = n * (n - 1) // 2
print(f"generate_candidates (measured bbox, extent-only shell, gap {GAP}): offered {len(offered)}/{pairs} "
      f"({by}), excluded {len(excluded)}")

fused_ok = meas_ok = fused_ok_not_offered = offered_fused_refused = 0
same_label_missed = []
for oa, ob in itertools.combinations(objs, 2):
    fa, fb = A._as_bounds(oa['fused_bbox']), A._as_bounds(ob['fused_bbox'])
    ma, mb = A._as_bounds(oa['bbox']), A._as_bounds(ob['bbox'])
    gf, gm = A.box_gap(fa, fb), A.box_gap(ma, mb)
    key = frozenset((oa['object_id'], ob['object_id']))
    if gf <= GAP:
        fused_ok += 1
        if key not in offered_keys:
            fused_ok_not_offered += 1
            if oa['label'].split('#')[0] == ob['label'].split('#')[0]:
                same_label_missed.append((oa['label'], ob['label'], round(gf, 3), round(gm, 3)))
    if gm <= GAP:
        meas_ok += 1
    if key in offered_keys and gf > GAP:
        offered_fused_refused += 1
print(f"pairs passing the FUSED gap test (what the merge gate applies): {fused_ok}/{pairs}")
print(f"pairs passing the MEASURED gap test: {meas_ok}/{pairs}")
print(f"pairs the merge gate would accept but generate_candidates never offers: {fused_ok_not_offered}/{fused_ok}"
      f" (same base label among them: {len(same_label_missed)})")
for row in same_label_missed:
    print("   ", row)
print(f"offered pairs the fused geometry gate then refuses (reason 'geometry'): {offered_fused_refused}/{len(offered)}")
print("probe_ga493_locality done")
