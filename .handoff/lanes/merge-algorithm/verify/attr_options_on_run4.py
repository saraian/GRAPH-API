#!/usr/bin/env python3
"""Owner ruling 2026-09-14 ("measure first"): replay run 4's RECORDED merge decisions under each
attribute-channel option. Exact, no model: every merge_refused row carries each channel's
log_odds, so a new total is the recorded total with the attributes term swapped.

    python3 attr_options_on_run4.py [<run_dir>]

Options compared (the attribute channel's contribution for a pair scoring s, reference r = 0.85,
positive cap c = 2.0):
  current   s >= r: +c*(s-r)/(1-r)          s < r: -8*(r-s)/(r-0.5)
  floor1    same as current, but a SAME-KIND pair's negative arm is clamped at -1.0
  floor0    same as current, but a SAME-KIND pair never contributes below 0
  ref075    reference 0.75 for every pair (positive above it, negative below)
  off       the attribute channel contributes nothing (still recorded)
Also reported: the effect of the 2026-09-14 co-visibility change (veto needs 3D separation too),
since a vetoed pair has no total at all.
"""
import collections
import json
import os
import sys

RUN = sys.argv[1] if len(sys.argv) > 1 else '/home/xps/graphapi_ws/results/20260914_174342_hm3d_00824'
THRESHOLD = 2.9957  # log(20)
REF, CAP, FLOOR = 0.85, 2.0, 8.0

rows = [json.loads(line) for line in open(os.path.join(RUN, 'hook_decisions.jsonl'))
        if '"merge_refused"' in line]
print(f"run: {RUN}\nmerge_refused rows: {len(rows)}; threshold {THRESHOLD:.4f}\n")


def attr_llr(score, same_kind, option):
    if score is None:
        return 0.0
    s = float(score)
    if option == 'off':
        return 0.0
    ref = 0.75 if option == 'ref075' else REF
    if s >= ref:
        return min(CAP, CAP * (s - ref) / (1.0 - ref))
    neg = -FLOOR * (ref - s) / max(ref - 0.5, 1e-9)
    neg = max(-FLOOR, neg)
    if same_kind and option == 'floor1':
        neg = max(neg, -1.0)
    if same_kind and option == 'floor0':
        neg = 0.0
    return neg


OPTIONS = ['current', 'floor1', 'floor0', 'ref075', 'off']
result = {o: collections.Counter() for o in OPTIONS}
flips = collections.defaultdict(list)
covis_spared = []

for r in rows:
    ch = r.get('channels') or {}
    at = ch.get('attributes') or {}
    cov = ch.get('covisibility') or {}
    ov = ch.get('overlap') or {}
    labels = f"{r.get('a_label')} <-> {r.get('b_label')}"
    same_kind = bool(at.get('same_kind'))
    # the 2026-09-14 co-visibility rule: a veto needs the 3D boxes to be disjoint as well
    vetoed_now = bool(cov.get('veto'))
    intersects_3d = float(ov.get('intersection_m3') or 0.0) > 0.0
    if vetoed_now and intersects_3d:
        covis_spared.append((labels, same_kind, at.get('score'), ov.get('containment'),
                             cov.get('overlap_2d')))
    if vetoed_now and not intersects_3d:
        for o in OPTIONS:
            result[o]['vetoed'] += 1
        continue
    if r.get('reason') == 'geometry':
        for o in OPTIONS:
            result[o]['geometry'] += 1
        continue
    # recorded total minus the recorded attributes term = the rest of the evidence
    recorded_total = r.get('hypothesis_total')
    recorded_attr = at.get('log_odds')
    if recorded_total is None:
        # a vetoed pair now spared: rebuild from the channels that did run
        rest = sum(v.get('log_odds', 0.0) for k, v in ch.items()
                   if k != 'attributes' and isinstance(v, dict) and 'log_odds' in v)
    else:
        rest = float(recorded_total) - float(recorded_attr or 0.0)
    for o in OPTIONS:
        total = rest + attr_llr(at.get('score'), same_kind, o)
        over = total >= THRESHOLD
        result[o]['over_threshold' if over else 'below'] += 1
        if over:
            flips[o].append((labels, same_kind, round(total, 2), round(rest, 2),
                             at.get('score'), ov.get('containment')))

print("PAIRS OVER THE COMMIT THRESHOLD (one sweep; a merge still needs two consecutive)\n")
print(f"{'option':10s} {'over':>5s} {'below':>6s} {'vetoed':>7s} {'geometry':>9s}   same-kind / cross-kind among those over")
for o in OPTIONS:
    c = result[o]
    sk = sum(1 for f in flips[o] if f[1])
    xk = len(flips[o]) - sk
    print(f"{o:10s} {c['over_threshold']:5d} {c['below']:6d} {c['vetoed']:7d} {c['geometry']:9d}   {sk} same-kind, {xk} CROSS-KIND")

for o in OPTIONS:
    if not flips[o]:
        continue
    print(f"\n--- {o}: pairs that clear the threshold")
    for labels, sk, total, rest, score, contain in sorted(flips[o], key=lambda f: -f[2]):
        kind = 'same-kind' if sk else 'CROSS-KIND'
        print(f"    {labels:34s} {kind:10s} total {total:6.2f} (geometry etc {rest:6.2f}, "
              f"attr score {score}, containment {contain})")

print(f"\n--- co-visibility rule change: pairs whose veto is lifted (2D-disjoint but 3D-intersecting): {len(covis_spared)}")
for labels, sk, score, contain, ov2d in covis_spared:
    print(f"    {labels:34s} {'same-kind' if sk else 'CROSS-KIND':10s} attr {score} containment {contain} 2D IoU {ov2d}")
