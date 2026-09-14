"""Refine the 76 'fused-accepted but never offered' pairs with the shell production would use.

`_assoc_build` passes neither depth_sparsity nor pose_sigma_m, so `position_covariance` is
eye*1e-9/n for a single sighting (shell ~1e-4 m beyond the extent) and the SAMPLE spread of the
sighting centroids divided by n for >=2 sightings. Sightings are appended once per cycle in which
the object was current (add or update). This rebuilds each final object's sighting centroids from
the mutation ledger (applied add/update receipts, joined to the captured detection box by
observation_id) and re-runs generate_candidates with that covariance. The centroid of the
detection box stands in for `obj.centroid` at sighting time, so the widened shell is ESTIMATED;
the single-sighting case needs no estimate.
"""
import collections
import itertools
import json
import sys

import numpy as np

PM = '/DATA/GRAPH-API/.claude/worktrees/found-merge-algorithm/lost3dsg/src/perception_module'
sys.path.insert(0, PM)
import association as A  # noqa: E402

R = '/DATA/GRAPH-API/.handoff/lanes/perception/ga493_replay_20260914/'
GAP = 0.3
objs = json.load(open(R + 'bundle/persistent_perception.json'))
n = len(objs)

box_by_obs = {}
for line in open(R + 'capture/consumer/events.jsonl'):
    e = json.loads(line)
    if e['kind'] != 'consumer_pair':
        continue
    for b in e['payload']['bboxes']['boxes']:
        oid = (b.get('observation') or {}).get('observation_id')
        bd = A._as_bounds(b)
        if oid and bd is not None:
            box_by_obs[oid] = bd

sightings = collections.defaultdict(list)
ops = collections.Counter()
for line in open(R + 'bundle/mutation_receipts.jsonl'):
    r = json.loads(line)
    if r['mutation_state'] != 'applied_in_memory' or r['operation'] not in ('add', 'update'):
        continue
    ops[(r['object_id'], r['operation'])] += 1
    bd = box_by_obs.get((r.get('observation') or {}).get('observation_id'))
    if bd is not None:
        sightings[r['object_id']].append(A.box_centroid(bd))

final_ids = {o['object_id'] for o in objs}
n_sight = {i: len(sightings.get(i, [])) for i in final_ids}
print(f"final objects {n}; sightings per object: "
      f"{sorted(collections.Counter(n_sight.values()).items())} (0 = ledger row without a captured box)")


def build(o):
    obs = [A.Observation(frame_id=k, camera_position=c + np.array([1.0, 0.0, 0.0]), centroid=c)
           for k, c in enumerate(sightings.get(o['object_id'], []))]
    return A.AssocObject(o['object_id'], bbox=o['bbox'], label=o['label'], observations=obs)


built = [build(o) for o in objs]
ctx = A.AssocContext(map_volume_m3=300.0, n_rooms=6, cost_ratio=20.0, use_ontology=False, locality_gap_m=GAP)
offered, _ = A.generate_candidates(built, ctx)
offered_keys = {frozenset((a.object_id, b.object_id)) for a, b, _ in offered}
radii = {b.object_id: A.search_radius(b, ctx) for b in built}
widen = [radii[b.object_id][0] - A.search_radius(A.AssocObject(b.object_id, bbox=b.bbox), ctx)[0]
         for b in built]
print(f"shell widening beyond the extent with the rebuilt sightings: max {max(widen):.3f} m, "
      f"objects widened by > 0.01 m: {sum(1 for w in widen if w > 0.01)}/{n}")

fused_ok = 0
missed = []
for oa, ob in itertools.combinations(objs, 2):
    fa, fb = A._as_bounds(oa['fused_bbox']), A._as_bounds(ob['fused_bbox'])
    if A.box_gap(fa, fb) <= GAP:
        fused_ok += 1
        if frozenset((oa['object_id'], ob['object_id'])) not in offered_keys:
            missed.append((oa, ob))
same = [(a, b) for a, b in missed if a['label'].split('#')[0] == b['label'].split('#')[0]]
print(f"with the production shell: fused-accepted pairs never offered {len(missed)}/{fused_ok}; "
      f"same base label {len(same)}")
for a, b in same:
    print("     ", a['label'], b['label'],
          "sightings", n_sight[a['object_id']], n_sight[b['object_id']],
          "fused gap", round(A.box_gap(A._as_bounds(a['fused_bbox']), A._as_bounds(b['fused_bbox'])), 3),
          "measured gap", round(A.box_gap(A._as_bounds(a['bbox']), A._as_bounds(b['bbox'])), 3))
print("refute_locality_shell_real done")
