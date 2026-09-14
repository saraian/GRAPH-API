#!/usr/bin/env python3
"""Score each merge-decision option against GROUND TRUTH. Owner standing instruction
2026-09-14: always evaluate with GT.

GT IS EVALUATOR-ONLY. It is opened here, after the run's decisions exist on disk, and it never
touched candidate generation or any gate. The decisions come from the bundle's
hook_decisions.jsonl; every row carries each channel's log-odds, so a different attribute rule
is an exact arithmetic substitution, not a re-run.

THE TEST. Two predicted objects OUGHT to merge exactly when they are two views of ONE true
object. So each predicted object is assigned to a GT object, and a candidate pair is
  true_merge   both sides assigned to the SAME GT object   -> merging is correct
  false_merge  assigned to DIFFERENT GT objects            -> merging destroys an identity
  unknown      either side has no GT assignment            -> not counted either way
Assignment is class-agnostic and MANY-TO-ONE: each prediction takes its own best-IoU GT object,
independently of the others (see the note at the assignment for why one-to-one is the wrong
instrument here). Same fixed feed transform, no fitted alignment. Predictions whose best IoU is
0 get a centre-distance fallback, reported separately.

    python3 attr_options_gt.py [<run_dir>] [<gt_json>]
"""
import collections
import json
import os
import sys

import numpy as np

RUN = sys.argv[1] if len(sys.argv) > 1 else '/home/xps/graphapi_ws/results/20260914_174342_hm3d_00824'
GT = (sys.argv[2] if len(sys.argv) > 2 else
      '/DATA/GRAPH-API/.handoff/lanes/perception/ga493_replay_20260914/evaluator/hm3d_00824_run_gt.json')
THRESHOLD = 2.9957
REF, CAP, FLOOR = 0.85, 2.0, 8.0
K = ['x_min', 'y_min', 'z_min', 'x_max', 'y_max', 'z_max']
STRUCTURAL = {'wall', 'floor', 'ceiling', 'door', 'doorway', 'door frame', 'doorframe', 'unknown', ''}
FALLBACK_M = 0.75   # centre-distance fallback when a prediction overlaps no GT box


def box(d):
    return np.array([float(d[k]) for k in K])


def iou(a, b):
    lo = np.maximum(a[:3], b[:3])
    hi = np.minimum(a[3:], b[3:])
    d = np.clip(hi - lo, 0, None)
    i = d.prod()
    va = (a[3:] - a[:3]).prod()
    vb = (b[3:] - b[:3]).prod()
    return float(i / (va + vb - i)) if va + vb - i > 0 else 0.0


def centre(a):
    return (a[:3] + a[3:]) / 2.0


gt = json.load(open(GT))
assert str(gt.get('scene')) == '00824', gt.get('scene')
truth = []
for o in gt['ground_truth_objects']:
    if str(o.get('category_name', '')).strip().lower() in STRUCTURAL:
        continue
    lo, hi = np.array(o['aabb_min_m']), np.array(o['aabb_max_m'])
    # habitat_pose_to_ros: ROS = (-h.z, -h.x, h.y). No fitted correction.
    truth.append((o['object_id'], o['category_name'],
                  np.array([-hi[2], -hi[0], lo[1], -lo[2], -lo[0], hi[1]])))

pred = json.load(open(os.path.join(RUN, 'persistent_perception.json')))
pred = [o for o in pred if str(o['label']).split('#')[0].strip().lower() not in STRUCTURAL]
pids = [o['object_id'] for o in pred]
pboxes = [box(o['bbox']) for o in pred]
plabels = {o['object_id']: o['label'] for o in pred}

# MANY-TO-ONE, and that is the whole point. evaluate.py assigns ONE-TO-ONE (Hungarian) because
# it scores how many GT objects were found; used here it would be the wrong instrument (rule 18).
# The question here is "are these two predictions the same true object?", and a one-to-one
# constraint makes that unanswerable BY CONSTRUCTION: two predictions of one object cannot both
# map to it, so the second is pushed onto some other GT object and the pair is then labelled a
# false merge. MEASURED on 20260914_180343: two `sofa` detections of couch_8 -- exactly the
# duplicate a merge should fix -- were scored FALSE because the Hungarian step pushed one of
# them onto pillow_58. Each prediction therefore takes its OWN best GT object, independently.
M = np.zeros((len(pboxes), len(truth)))
for i, pb in enumerate(pboxes):
    for j, (_, _, tb) in enumerate(truth):
        M[i, j] = iou(pb, tb)
assign, how = {}, {}
for i, pb in enumerate(pboxes):
    j = int(np.argmax(M[i]))
    if M[i, j] > 0:
        assign[pids[i]] = truth[j][0]
        how[pids[i]] = f'IoU {M[i, j]:.3f} -> {truth[j][1]}'
        continue
    # overlaps nothing: nearest GT centre, within a stated radius
    dist, gid, cat = min((float(np.linalg.norm(centre(pb) - centre(tb))), gid, cat)
                         for gid, cat, tb in truth)
    if dist <= FALLBACK_M:
        assign[pids[i]] = gid
        how[pids[i]] = f'centre {dist:.2f} m -> {cat} (fallback)'
print(f"run: {RUN}\nGT: {GT}  ({len(truth)} movable GT objects; {len(pred)} predicted)")
n_iou = sum(1 for v in how.values() if v.startswith('IoU'))
print(f"GT assignment: {n_iou} by IoU, {len(assign) - n_iou} by centre fallback (<= {FALLBACK_M} m), "
      f"{len(pred) - len(assign)} unassigned\n")
dup = collections.Counter(assign.values())
print("GT objects covered by MORE THAN ONE prediction (the duplicates a merge should fix):")
for gid, n in sorted(dup.items(), key=lambda kv: -kv[1]):
    if n > 1:
        members = [f"{plabels[p]}" for p, g in assign.items() if g == gid]
        print(f"    {gid:22s} x{n}: {members}")
truth_pairs = {tuple(sorted((a, b)))
               for a in assign for b in assign if a < b and assign[a] == assign[b]}
print(f"  -> {len(truth_pairs)} GT-true duplicate pair(s) exist among the predictions\n")

rows = [json.loads(line) for line in open(os.path.join(RUN, 'hook_decisions.jsonl'))
        if '"merge_refused"' in line]


def attr_llr(score, same_kind, option):
    if score is None or option == 'off':
        return 0.0
    s = float(score)
    ref = 0.75 if option == 'ref075' else REF
    if s >= ref:
        return min(CAP, CAP * (s - ref) / (1.0 - ref))
    neg = max(-FLOOR, -FLOOR * (ref - s) / max(ref - 0.5, 1e-9))
    if same_kind and option == 'floor1':
        neg = max(neg, -1.0)
    if same_kind and option == 'floor0':
        neg = 0.0
    return neg


OPTIONS = ['current', 'floor1', 'floor0', 'ref075', 'off']
# and the co-visibility rule, both ways
RULES = [('veto 2D only (before)', False), ('veto needs 3D too (ruling)', True)]
print(f"threshold {THRESHOLD:.4f}; {len(rows)} recorded decisions\n")
print(f"{'co-visibility rule':28s} {'attributes':10s} {'merges':>7s} {'TRUE':>5s} {'FALSE':>6s} {'unk':>4s} "
      f"{'recall':>7s} {'precision':>10s}")
detail = {}
for rule_name, needs_3d in RULES:
    for opt in OPTIONS:
        over = []
        for r in rows:
            ch = r.get('channels') or {}
            at = ch.get('attributes') or {}
            cov = ch.get('covisibility') or {}
            ov = ch.get('overlap') or {}
            if r.get('reason') == 'geometry':
                continue
            vetoed = bool(cov.get('veto'))
            if vetoed and needs_3d and float(ov.get('intersection_m3') or 0.0) > 0.0:
                vetoed = False          # the ruling spares this pair
            if vetoed:
                continue
            total = r.get('hypothesis_total')
            if total is None:
                rest = sum(v.get('log_odds', 0.0) for k, v in ch.items()
                           if k != 'attributes' and isinstance(v, dict) and 'log_odds' in v)
            else:
                rest = float(total) - float(at.get('log_odds') or 0.0)
            if rest + attr_llr(at.get('score'), bool(at.get('same_kind')), opt) >= THRESHOLD:
                over.append(r)
        tp = fp = unk = 0
        rowsd = []
        for r in over:
            a, b = r.get('object'), r.get('candidate')
            ga, gb = assign.get(a), assign.get(b)
            if ga is None or gb is None:
                unk += 1
                verdict = 'unknown'
            elif ga == gb:
                tp += 1
                verdict = f'TRUE ({ga})'
            else:
                fp += 1
                verdict = f'FALSE ({ga} vs {gb})'
            rowsd.append((f"{r.get('a_label')} <-> {r.get('b_label')}", verdict))
        recall = tp / len(truth_pairs) if truth_pairs else float('nan')
        prec = tp / (tp + fp) if (tp + fp) else float('nan')
        print(f"{rule_name:28s} {opt:10s} {len(over):7d} {tp:5d} {fp:6d} {unk:4d} "
              f"{recall:7.2f} {prec:10.2f}")
        detail[(rule_name, opt)] = rowsd

print("\nPAIRS EACH OPTION WOULD MERGE, with the GT verdict (co-visibility ruling applied):")
for opt in OPTIONS:
    d = detail[('veto needs 3D too (ruling)', opt)]
    print(f"  {opt}: {len(d)} pair(s)")
    for labels, verdict in d:
        print(f"      {labels:34s} {verdict}")
