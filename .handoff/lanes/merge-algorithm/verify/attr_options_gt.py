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
import copy
import json
import os
import sys

import numpy as np

ARGS = [a for a in sys.argv[1:] if not a.startswith('--')]
FLAGS = {a for a in sys.argv[1:] if a.startswith('--')}
SELFCHECK = '--selfcheck' in FLAGS
RUN = ARGS[0] if ARGS else '/home/xps/graphapi_ws/results/20260914_174342_hm3d_00824'
GT = (ARGS[1] if len(ARGS) > 1 else
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
assert str(gt.get("scene", "")).startswith("00824"), gt.get("scene")   # GT files name it "00824" or "00824-Dd4bFSTQ8gi"
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

def _attr_llr_WITHDRAWN(score, same_kind, option):
    """WITHDRAWN 2026-09-14, kept only so the record says why.

    This re-implemented `association.channel_attributes` in the tool. MEASURED against the
    97 recorded attribute rows of the two 2026-09-14 runs, it disagreed with the engine on
    3 of 50 rows of 20260914_180343 -- every one a same-kind pair where the engine returns
    0.0 (the owner's same-kind floor of that morning) and this function returned about -3.8,
    a 3.88-nat error on the arm the whole table is read from -- and drifted by up to 0.01
    nats on 54 of 97 rows because it recomputes from `score`, which the record rounds to 4
    decimal places. It is the same defect as the one this repair exists to fix, one level
    down: a copy of a rule that moved. The options now call the engine function.
    """
    raise NotImplementedError("use attr_log_odds(); see the docstring")



# ---------------------------------------------------------------------------------------
# THE ENGINE ITSELF, not a copy of it.
#
# WHAT THIS REPLACED, AND WHY (repair 2026-09-14, plan step 0).
# The block that used to stand here decided a merge with ONE arithmetic test per LOG ROW:
# "recorded total, minus the recorded attribute term, plus the option's term, >= 2.9957".
# The shipped node does not decide that way. `assoc.Hypothesis` requires the SAME PAIR to
# hold above the threshold on `merge_min_consecutive` SUCCESSIVE scored sweeps, breaks that
# streak on a room or geometry refusal, applies the unwitnessed-overlap rule, and answers
# once per PAIR rather than once per row. So the old block reported merges the node was
# structurally unable to make: 15 of 16 on 20260914_180343 and 13 of 13 on 20260914_174342.
#
# The repair deletes the copy and calls the original. Nothing here re-implements a rule.
# Constants come from the file the node reads and from the row, never from a literal here.
# ---------------------------------------------------------------------------------------
SRC = '/DATA/GRAPH-API/.claude/worktrees/found-merge-algorithm/lost3dsg/src/perception_module'
sys.path.insert(0, SRC)
import association as assoc  # noqa: E402, I001 -- Hypothesis/PairScore: what the node runs; needs sys.path above

_cfg_text = open(os.path.join(SRC, 'config.yaml')).read()


def _cfg_int(name, default):
    """Read one association constant out of config.yaml without importing the node's config.

    Deliberately a plain scan: importing config.py pulls in ROS. The key is matched at the
    start of a line so a mention inside a comment cannot answer for the setting.
    """
    for line in _cfg_text.split('\n'):
        s = line.strip()
        if s.startswith(name + ':'):
            return int(s.split(':', 1)[1].split('#')[0].strip())
    return default


MIN_CONSECUTIVE = _cfg_int('merge_min_consecutive', 2)
MIN_EVIDENCE = _cfg_int('merge_min_evidence', 1)

# `_measured_abstentions` is NOT written to any bundle (that is the additive telemetry key
# this repair asks for separately). It is reconstructible exactly for the two channels that
# can set it, because each sets it on ONE reason string that the channel itself formats:
#   association.channel_covisibility -> "co-visible but overlapping (2D IoU ...)"
#   association.channel_attributes   -> "N optional term(s) comparable < M"
# Every other Abstain in association.py leaves measured False. Matching those two strings is
# therefore a read of the producer's own text, not an inference about it.
def _cfg_float(name, default):
    for line in _cfg_text.split('\n'):
        t = line.strip()
        if t.startswith(name + ':'):
            return float(t.split(':', 1)[1].split('#')[0].strip())
    return default


REFERENCE = _cfg_float('sim_threshold', 0.85)
MAX_LOG_ODDS = _cfg_float('merge_attribute_max_log_odds', 2.0)
ATTR_MIN_EVIDENCE = _cfg_float('merge_attribute_min_evidence', 1)

# THE ATTRIBUTE OPTIONS, AS ARGUMENTS TO THE ENGINE'S OWN FUNCTION.
# 'shipped' is not recomputed at all: it is the number the run recorded, so the baseline
# column is an exact replay by construction and cannot drift from the node again.
# `floor1` is the one option the engine cannot express, so it is applied as a stated
# post-adjustment and labelled as a counterfactual rather than a setting.
OPTION_ARGS = {
    'shipped':  None,                                              # read, never recomputed
    'no_floor': dict(reference=REFERENCE, same_kind_floor_zero=False),
    'floor1':   dict(reference=REFERENCE, same_kind_floor_zero=False, clamp_same_kind=-1.0),
    'ref075':   dict(reference=0.75, same_kind_floor_zero=True),
    'off':      dict(zero=True),
}


def attr_log_odds(at, option):
    """-> the attribute channel's log-odds under one option, from association.channel_attributes.

    `at` is the recorded attributes channel. Nothing here re-derives the rule.
    """
    if option == 'shipped':
        return at.get('log_odds')
    args = OPTION_ARGS[option]
    if args.get('zero'):
        return 0.0
    r = assoc.channel_attributes(
        at.get('score'), int(at.get('evidence_count') or 0),
        args.get('reference', REFERENCE), MAX_LOG_ODDS,
        min_evidence=int(ATTR_MIN_EVIDENCE), same_kind=bool(at.get('same_kind')),
        same_kind_floor_zero=args.get('same_kind_floor_zero', True))
    if isinstance(r, assoc.Abstain):
        return None                       # the channel abstains: it leaves `channels` entirely
    llr = float(r[0])
    clamp = args.get('clamp_same_kind')
    if clamp is not None and at.get('same_kind'):
        llr = max(llr, clamp)
    return llr


MEASURED_ABSTAIN = {
    'covisibility': 'co-visible but overlapping (2D IoU',
    'attributes': 'optional term(s) comparable <',
}
# One reason in the 2026-09-14 bundles comes from a code epoch that no longer exists: the
# 3D-intersect relaxation of the co-visibility veto, withdrawn on ground truth the same day
# (association.channel_covisibility, the WITHDRAWN comment). Whether that build marked it
# measured cannot be read off the bundle, so rows carrying it are COUNTED AND REPORTED, never
# assumed either way.
STALE_EPOCH_ABSTAIN = 'co-visible and 2D-disjoint, but the 3D boxes intersect'

iou_of = {pids[i]: float(M[i].max()) for i in range(len(pids))}

rows = [json.loads(line) for line in open(os.path.join(RUN, 'hook_decisions.jsonl'))]
merge_rows = [r for r in rows if r.get('kind') in ('merge_refused', 'merge')]
applied_rows = [r for r in rows if r.get('kind') == 'merge']
by_pair = collections.defaultdict(list)
for r in merge_rows:
    by_pair[tuple(sorted((r['object'], r['candidate'])))].append(r)
for v in by_pair.values():
    # No bundle carries a sweep index; `t` is the only ordering the rows supply.
    v.sort(key=lambda r: r['t'])

stale_epoch_rows = sum(1 for r in merge_rows
                       if STALE_EPOCH_ABSTAIN in str((r.get('abstentions') or {}).get('covisibility', '')))


def build_pair_score(r, opt, needs_3d):
    """-> PairScore or None. The recorded channels, with ONE substitution: the attribute rule.

    Returns None for a row that carries no channels at all (a legacy-engine row), which is
    not a scored sweep and must not reach `update`.
    """
    ch = copy.deepcopy(r.get('channels') or {})
    if not ch:
        return None
    if needs_3d:
        cov, ov = ch.get('covisibility') or {}, ch.get('overlap') or {}
        if cov.get('veto') and float(ov.get('intersection_m3') or 0.0) > 0.0:
            ch.pop('covisibility')          # the counterfactual ruling spares this pair
    at = ch.get('attributes')
    if at is not None and 'log_odds' in at:
        sub = attr_log_odds(at, opt)
        if sub is None:
            ch.pop('attributes')          # the engine abstains: the channel is not in `channels`
        else:
            at['log_odds'] = sub
    ps = assoc.PairScore()
    ps.channels = ch
    ps.abstentions = dict(r.get('abstentions') or {})
    ps.vetoed_by = [n for n, c in ch.items() if c.get('veto')]
    ps._measured_abstentions = {
        name for name, needle in MEASURED_ABSTAIN.items()
        if needle in str(ps.abstentions.get(name, ''))}
    ps.total = sum(c['log_odds'] for c in ch.values() if 'log_odds' in c)
    return ps


def replay(key, rs, opt, needs_3d):
    """-> (merged, reason, n_scored). ONE verdict per PAIR, from the engine's own class."""
    h = assoc.Hypothesis(key)
    why, n_scored = 'never scored', 0
    for r in rs:
        if r.get('kind') == 'merge':
            # The applied-merge row carries no channels, so it cannot be replayed. It is
            # still the node's own answer: honour it and say where it came from.
            return True, 'recorded as applied by the run', n_scored
        if r.get('reason') in ('room', 'geometry'):
            h.interrupt(r['reason'], frame_id=r['t'])
            continue
        ps = build_pair_score(r, opt, needs_3d)
        if ps is None:
            continue
        n_scored += 1
        h.update(ps, frame_id=r['t'])
        decision, why = h.decide(float(r.get('threshold_log_odds') or THRESHOLD),
                                 min_evidence=MIN_EVIDENCE, min_consecutive=MIN_CONSECUTIVE)
        if decision == 'merge':
            return True, why, n_scored
    return False, why, n_scored


# ---------------------------------------------------------------------------------------
# Self-check: does the replay reproduce what the run recorded?
# ---------------------------------------------------------------------------------------
if SELFCHECK:
    print(f"SELF-CHECK on {RUN}")
    print(f"  merge_min_consecutive={MIN_CONSECUTIVE}  merge_min_evidence={MIN_EVIDENCE} "
          f"(read from {os.path.join(SRC, 'config.yaml')})")
    checked = failed = skipped = checked_u = 0
    for key, rs in sorted(by_pair.items()):
        h = assoc.Hypothesis(key)
        for r in rs:
            if r.get('kind') == 'merge':
                continue
            if r.get('reason') in ('room', 'geometry'):
                h.interrupt(r['reason'], frame_id=r['t'])
                continue
            ps = build_pair_score(r, 'shipped', False)   # the recorded numbers, unaltered
            if ps is None:
                continue
            h.update(ps, frame_id=r['t'])
            _d, why = h.decide(float(r.get('threshold_log_odds') or THRESHOLD),
                               min_evidence=MIN_EVIDENCE, min_consecutive=MIN_CONSECUTIVE)
            # ASSERTION 1, and the one with teeth. `updates` is len(h.history) at the moment
            # the node wrote the row (object_services.py, the merge_refused emit). History
            # grows on every update AND on every interrupt, so this single number disagrees
            # if the pair grouping is wrong, if the rows are replayed out of order, or if a
            # room/geometry refusal stops breaking the streak. It does not need a pair to
            # reach the threshold, which is why it works on bundles where none does.
            rec_updates = r.get('updates')
            if rec_updates is not None:
                checked_u += 1
                if len(h.history) != int(rec_updates):
                    failed += 1
                    print(f"  HISTORY MISMATCH  {key[0][:14]}/{key[1][:14]}  t={r['t']:.3f}  "
                          f"recorded updates={rec_updates}  replayed={len(h.history)}")
            # ASSERTION 2: the decision text itself, on the branches the bundle can support.
            recorded = str(r.get('decision_reason') or '')
            if not recorded.startswith('log-odds'):
                skipped += 1
                continue
            checked += 1
            if why != recorded:
                failed += 1
                print(f"  MISMATCH  {key[0][:14]}/{key[1][:14]}  t={r['t']:.3f}\n"
                      f"      recorded: {recorded}\n      replayed: {why}")
    # WHAT THIS CHECK CANNOT SEE ON THIS BUNDLE. Stated, not hidden: the streak-breaking
    # interrupt is only exercised by a pair that receives BOTH a room/geometry refusal and a
    # scored sweep. Mutation-tested 2026-09-14: deleting the interrupt call is NOT caught by
    # either 2026-09-14 bundle, because that count is 0 in both.
    interrupt_pairs = sum(
        1 for rs in by_pair.values()
        if any(r.get('reason') in ('room', 'geometry') for r in rs)
        and any(r.get('channels') for r in rs))
    merged_pairs = sum(1 for key, rs in by_pair.items() if replay(key, rs, 'shipped', False)[0])
    print(f"  update counts: {checked_u} asserted against the recorded `updates`")
    print(f"  log-odds reasons: {checked} checked, {skipped} not asserted "
          f"(veto/abstain branches; they need the measured_abstentions key)")
    print(f"  total mismatches: {failed}")
    print(f"  pairs exercising the streak interrupt: {interrupt_pairs}"
          + ("  <- ZERO: the interrupt is UNTESTED by this bundle" if not interrupt_pairs else ""))
    print(f"  pairs the replay merges: {merged_pairs}; kind=='merge' rows in the bundle: "
          f"{len(applied_rows)}")
    if stale_epoch_rows:
        print(f"  NOTE {stale_epoch_rows} row(s) carry the withdrawn 3D-intersect abstention, "
              f"whose measured flag this bundle cannot supply")
    ok = (failed == 0 and merged_pairs == len(applied_rows))
    print("  SELF-CHECK " + ("PASS" if ok else "FAIL"))
    sys.exit(0 if ok else 1)

# ---------------------------------------------------------------------------------------
# The option table. One row per (co-visibility rule, attribute option); one verdict per PAIR.
# ---------------------------------------------------------------------------------------
OPTIONS = ['shipped', 'no_floor', 'floor1', 'ref075', 'off']
RULES = [('veto 2D only (shipped)', False), ('veto needs 3D too (withdrawn)', True)]

scored_pairs = {k for k, rs in by_pair.items() if any(r.get('channels') for r in rs)}
twice = {k for k in scored_pairs if sum(1 for r in by_pair[k] if r.get('channels')) >= 2}
print(f"\nmerge decisions: {len(merge_rows)} row(s) over {len(by_pair)} distinct pair(s); "
      f"{len(scored_pairs)} pair(s) scored at least once, {len(twice)} scored at least twice")
print(f"merge_min_consecutive = {MIN_CONSECUTIVE}, so ONLY those {len(twice)} pair(s) can "
      f"commit at all. A table built on fewer is a sample of zero for persistence, not of one.")
print(f"threshold: per row, from threshold_log_odds; min_evidence {MIN_EVIDENCE}\n")

print(f"{'co-visibility rule':30s} {'attributes':10s} {'merges':>7s} {'TRUE':>5s} {'FALSE':>6s} "
      f"{'unk':>4s} {'recall':>7s} {'precision':>10s}")
detail = {}
for rule_name, needs_3d in RULES:
    for opt in OPTIONS:
        merged = [k for k, rs in by_pair.items() if replay(k, rs, opt, needs_3d)[0]]
        tp = fp = unk = 0
        rowsd = []
        for a, b in merged:
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
            rowsd.append((f"{plabels.get(a, a)} <-> {plabels.get(b, b)}", verdict))
        recall = tp / len(truth_pairs) if truth_pairs else float('nan')
        prec = tp / (tp + fp) if (tp + fp) else float('nan')
        print(f"{rule_name:30s} {opt:10s} {len(merged):7d} {tp:5d} {fp:6d} {unk:4d} "
              f"{recall:7.2f} {prec:10.2f}")
        detail[(rule_name, opt)] = rowsd

# ---------------------------------------------------------------------------------------
# The denominator, swept. NO floor is adopted: every line carries its own n.
# ---------------------------------------------------------------------------------------
print("\nRECALL AGAINST EACH DENOMINATOR (shipped co-visibility rule).")
print("The GT assignment is class-agnostic, best-IoU and MANY-TO-ONE, so a pair can be called")
print("'true' on an overlap of a few percent. The floor is the owner's to choose; it is printed,")
print("not picked.")
same_label_pairs = {p for p in truth_pairs
                    if str(plabels.get(p[0], '')).split('#')[0].strip().lower()
                    == str(plabels.get(p[1], '')).split('#')[0].strip().lower()}
merged_by_opt = {opt: {k for k, rs in by_pair.items() if replay(k, rs, opt, False)[0]}
                 for opt in OPTIONS}
print(f"\n{'denominator':34s} {'n':>4s} " + ' '.join(f'{o:>8s}' for o in OPTIONS))
for floor in (0.0, 0.05, 0.10, 0.25, 0.50):
    pairs = {p for p in truth_pairs
             if min(iou_of.get(p[0], 0.0), iou_of.get(p[1], 0.0)) >= floor}
    cells = []
    for opt in OPTIONS:
        tp = len(pairs & merged_by_opt[opt])
        cells.append(f"{(tp / len(pairs)) if pairs else float('nan'):8.2f}")
    print(f"{'min assignment IoU >= %.2f' % floor:34s} {len(pairs):4d} " + ' '.join(cells))
cells = []
for opt in OPTIONS:
    tp = len(same_label_pairs & merged_by_opt[opt])
    cells.append(f"{(tp / len(same_label_pairs)) if same_label_pairs else float('nan'):8.2f}")
print(f"{'same base label (any IoU)':34s} {len(same_label_pairs):4d} " + ' '.join(cells))

print("\nPAIRS EACH OPTION WOULD MERGE, with the GT verdict (shipped co-visibility rule):")
for opt in OPTIONS:
    d = detail[('veto 2D only (shipped)', opt)]
    print(f"  {opt}: {len(d)} pair(s)")
    for labels, verdict in d:
        print(f"      {labels:34s} {verdict}")
if stale_epoch_rows:
    print(f"\nNOTE: {stale_epoch_rows} row(s) carry the withdrawn 3D-intersect co-visibility "
          f"abstention, from a code epoch that no longer exists.")
