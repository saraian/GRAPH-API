#!/usr/bin/env python3
"""WHY the GT-true duplicate pairs do not merge. GT is evaluator-only, opened after the run.

For every pair of predictions that GT says are the SAME object, say what the engine did with it:
never offered, refused on room, refused on geometry (with the gap), vetoed by co-visibility,
held below the threshold (with the total and the per-channel breakdown), or merged.

    python3 recall_gap_gt.py [<run_dir>]
"""
import collections
import json
import os
import sys

import numpy as np

RUN = sys.argv[1] if len(sys.argv) > 1 else '/home/xps/graphapi_ws/results/20260914_180343_hm3d_00824'
GT = '/DATA/GRAPH-API/.handoff/lanes/perception/ga493_replay_20260914/evaluator/hm3d_00824_run_gt.json'
THRESHOLD, K = 2.9957, ['x_min', 'y_min', 'z_min', 'x_max', 'y_max', 'z_max']
STRUCTURAL = {'wall', 'floor', 'ceiling', 'door', 'doorway', 'door frame', 'doorframe', 'unknown', ''}


def box(d):
    return np.array([float(d[k]) for k in K])


def iou(a, b):
    lo, hi = np.maximum(a[:3], b[:3]), np.minimum(a[3:], b[3:])
    i = np.clip(hi - lo, 0, None).prod()
    va, vb = (a[3:] - a[:3]).prod(), (b[3:] - b[:3]).prod()
    return float(i / (va + vb - i)) if va + vb - i > 0 else 0.0


def gap(a, b):
    return max(0.0, max(a[0] - b[3], b[0] - a[3]), max(a[1] - b[4], b[1] - a[4]),
               max(a[2] - b[5], b[2] - a[5]))


truth = []
for o in json.load(open(GT))['ground_truth_objects']:
    if str(o.get('category_name', '')).strip().lower() in STRUCTURAL:
        continue
    lo, hi = np.array(o['aabb_min_m']), np.array(o['aabb_max_m'])
    truth.append((o['object_id'], o['category_name'],
                  np.array([-hi[2], -hi[0], lo[1], -lo[2], -lo[0], hi[1]])))

pred = [o for o in json.load(open(os.path.join(RUN, 'persistent_perception.json')))
        if str(o['label']).split('#')[0].strip().lower() not in STRUCTURAL]
P = {o['object_id']: o for o in pred}
assign = {}
for o in pred:
    b = box(o['bbox'])
    best = max(((iou(b, tb), gid) for gid, _, tb in truth), key=lambda t: t[0])
    if best[0] > 0:
        assign[o['object_id']] = best[1]
    else:
        d = min((float(np.linalg.norm((b[:3] + b[3:]) / 2 - (tb[:3] + tb[3:]) / 2)), gid)
                for gid, _, tb in truth)
        if d[0] <= 0.75:
            assign[o['object_id']] = d[1]

rows = [json.loads(line) for line in open(os.path.join(RUN, 'hook_decisions.jsonl'))]
bykey = {}
for r in rows:
    if r.get('kind') in ('merge_refused', 'merge'):
        bykey.setdefault(tuple(sorted((r.get('object'), r.get('candidate') or r.get('merged_from')))), []).append(r)

pairs = [(a, b) for a in assign for b in assign if a < b and assign[a] == assign[b]]
print(f"run: {RUN}\n{len(pred)} predictions, {len(pairs)} GT-true duplicate pairs, "
      f"{len(rows)} decision rows\n")
verdicts = collections.Counter()
for a, b in sorted(pairs, key=lambda p: assign[p[0]]):
    la, lb = P[a]['label'], P[b]['label']
    ba, bb = box(P[a]['bbox']), box(P[b]['bbox'])
    fa = box(P[a].get('fused_bbox') or P[a]['bbox'])
    fb = box(P[b].get('fused_bbox') or P[b]['bbox'])
    seen = bykey.get((a, b), [])
    if not seen:
        v = 'NEVER OFFERED'
    else:
        last = seen[-1]
        if last['kind'] == 'merge':
            v = 'MERGED'
        elif last.get('reason') == 'geometry':
            v = f"refused GEOMETRY (gap {last.get('gap_m')} m)"
        elif last.get('reason') == 'room':
            v = 'refused ROOM'
        elif (last.get('channels') or {}).get('covisibility', {}).get('veto'):
            cov = last['channels']['covisibility']
            ov = last['channels'].get('overlap') or {}
            v = (f"VETOED covis (2D IoU {cov.get('overlap_2d')}, "
                 f"3D inter {ov.get('intersection_m3')})")
        else:
            ch = last.get('channels') or {}
            parts = {k: round(vv.get('log_odds'), 2) for k, vv in ch.items()
                     if isinstance(vv, dict) and vv.get('log_odds') is not None}
            v = (f"HELD total {last.get('hypothesis_total')} < {THRESHOLD:.2f}  {parts}")
    verdicts[v.split('(')[0].split('total')[0].strip()] += 1
    print(f"  {assign[a]:22s} {la:12s} <-> {lb:12s} measIoU {iou(ba, bb):.2f} fusedIoU {iou(fa, fb):.2f} "
          f"gap {gap(ba, bb):.2f} m | {v}")
print("\nsummary of what stops the GT-true pairs:")
for k, n in verdicts.most_common():
    print(f"  {n:3d}  {k}")
