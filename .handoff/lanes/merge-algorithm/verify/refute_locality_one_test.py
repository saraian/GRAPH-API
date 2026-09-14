"""Refutation attempt for the 'locality is not ONE test' finding.

Part A (merge path, exact where the shell is extent-only): of the fused-accepted pairs that
generate_candidates never offers on the MEASURED box, how many involve only single-view objects?
In production `_assoc_build` passes neither depth_sparsity nor pose_sigma_m, so an object with ONE
observation has covariance 1e-9 and its shell is the extent alone -- the extent-only assumption is
then exact, not an upper bound. Pairs with a >=2-view object could still be reached by the spread
term; they stay an upper bound.

Part B (association path, ESTIMATED on the final state): for every captured detection box, count
final objects that locality_ok would accept on the fused box (gap <= 0.3) but WorldModel.candidates
never returns because the measured box is more than 2*0.3 m away. The final boxes are not the boxes
that existed when the detection arrived, so this is an estimate of the population, not a replay.
"""
import itertools
import json
import sys

PM = '/DATA/GRAPH-API/.claude/worktrees/found-merge-algorithm/lost3dsg/src/perception_module'
sys.path.insert(0, PM)
import association as A  # noqa: E402

R = '/DATA/GRAPH-API/.handoff/lanes/perception/ga493_replay_20260914/'
GAP = 0.3
objs = json.load(open(R + 'bundle/persistent_perception.json'))
n = len(objs)
views = {o['object_id']: int((o['fused_bbox'] or {}).get('view_count') or 1) for o in objs}

built = [A.AssocObject(o['object_id'], bbox=o['bbox'], label=o['label']) for o in objs]
ctx = A.AssocContext(map_volume_m3=300.0, n_rooms=6, cost_ratio=20.0, use_ontology=False, locality_gap_m=GAP)
offered, _ = A.generate_candidates(built, ctx)
offered_keys = {frozenset((a.object_id, b.object_id)) for a, b, _ in offered}

# what generate_candidates WOULD offer if it read locality_bounds (fused first) instead of .bbox
built_f = [A.AssocObject(o['object_id'], bbox=o['fused_bbox'], label=o['label']) for o in objs]
offered_f, _ = A.generate_candidates(built_f, ctx)
offered_f_keys = {frozenset((a.object_id, b.object_id)) for a, b, _ in offered_f}

fused_ok = 0
missed = []
missed_f = 0
for oa, ob in itertools.combinations(objs, 2):
    fa, fb = A._as_bounds(oa['fused_bbox']), A._as_bounds(ob['fused_bbox'])
    if A.box_gap(fa, fb) <= GAP:
        fused_ok += 1
        key = frozenset((oa['object_id'], ob['object_id']))
        if key not in offered_keys:
            missed.append((oa, ob))
        if key not in offered_f_keys:
            missed_f += 1
single = [(a, b) for a, b in missed if views[a['object_id']] == 1 and views[b['object_id']] == 1]
same = [(a, b) for a, b in single if a['label'].split('#')[0] == b['label'].split('#')[0]]
print(f"A. final objects {n}; single-view {sum(1 for v in views.values() if v == 1)}/{n}")
print(f"A. fused-gap-accepted pairs {fused_ok}/{n*(n-1)//2}; not offered on measured bbox {len(missed)}/{fused_ok}")
print(f"A. of those, BOTH sides single-view (shell = extent exactly, so EXACT miss): {len(single)}/{len(missed)}; "
      f"same base label among them: {len(same)}")
for a, b in same:
    print("     ", a['label'], b['label'], "fused gap",
          round(A.box_gap(A._as_bounds(a['fused_bbox']), A._as_bounds(b['fused_bbox'])), 3),
          "measured gap", round(A.box_gap(A._as_bounds(a['bbox']), A._as_bounds(b['bbox'])), 3))
print(f"A. counterfactual: generate_candidates reading the FUSED box offers {len(offered_f)} pairs; "
      f"fused-accepted pairs it misses: {missed_f}/{fused_ok}")

# Part B
dets = []
for line in open(R + 'capture/consumer/events.jsonl'):
    e = json.loads(line)
    if e['kind'] != 'consumer_pair':
        continue
    for b in e['payload']['bboxes']['boxes']:
        bd = A._as_bounds(b)
        if bd is not None:
            dets.append((b['label'], bd))
print(f"B. captured detections with a 3D box: {len(dets)}")
acc = unreachable = unreachable_same = 0
dets_with_any_unreachable = 0
dets_only_unreachable = 0
for lab, d in dets:
    any_unreach = False
    any_reach = False
    for o in objs:
        f, m = A._as_bounds(o['fused_bbox']), A._as_bounds(o['bbox'])
        if A.box_gap(d, f) <= GAP:
            acc += 1
            if A.box_gap(d, m) > 2 * GAP:      # WorldModel.candidates(bbox, 0.3) expands by 0.6
                unreachable += 1
                any_unreach = True
                if o['label'].split('#')[0] == lab.split('#')[0]:
                    unreachable_same += 1
            else:
                any_reach = True
    dets_with_any_unreachable += any_unreach
    dets_only_unreachable += (any_unreach and not any_reach)
print(f"B. (detection, final object) pairs locality_ok accepts on the fused box: {acc}")
print(f"B. of those, NOT returned by WorldModel.candidates (measured gap > 0.6): {unreachable}/{acc}; "
      f"same base label: {unreachable_same}")
print(f"B. detections with at least one unreachable fused-accepted object: {dets_with_any_unreachable}/{len(dets)}; "
      f"with ONLY unreachable ones (every locality partner hidden): {dets_only_unreachable}/{len(dets)}")
print("refute_locality_one_test done")
