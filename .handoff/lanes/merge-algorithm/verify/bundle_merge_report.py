#!/usr/bin/env python3
"""Merge/association telemetry of ONE run bundle, GT-free, in the fields the DGX lane used so the
numbers are comparable.  python3 bundle_merge_report.py <run_dir>"""
import collections
import json
import os
import sys

run = sys.argv[1]
P = lambda *a: os.path.join(run, *a)  # noqa: E731


def rows(path):
    if not os.path.exists(path):
        return []
    return [json.loads(line) for line in open(path) if line.strip()]


meta = json.load(open(P('run_metadata.json'))) if os.path.exists(P('run_metadata.json')) else {}
ec = ((meta.get('resolved_config') or {}).get('effective_config') or {})
print(f"run: {run}")
print(f"engine stamped: {ec.get('association.merge_engine')} effective: {ec.get('association.merge_engine_effective')} "
      f"| ontology {ec.get('association.merge_ontology_channel')} | attr cap {ec.get('association.merge_attribute_max_log_odds')} "
      f"| margin {ec.get('association.association_margin_m')} | weights L/C/M/D "
      f"{ec.get('similarity.label')}/{ec.get('similarity.color')}/{ec.get('similarity.material')}/{ec.get('similarity.description')} "
      f"| vlm {ec.get('vlm.model')} thinking={ec.get('vlm.enable_thinking')}")
cap = meta.get('cap') or {}
print(f"cap: {cap.get('cap_minutes')} min, fired={cap.get('capped')}, elapsed {cap.get('elapsed_seconds')} s")

hd = rows(P('hook_decisions.jsonl'))
kinds = collections.Counter(r.get('kind') for r in hd)
print(f"hook_decisions rows: {len(hd)} kinds: {dict(kinds)}")
lat = rows(P('perception_latencies.jsonl'))
if lat:
    import statistics
    print(f"perception cycles: {len(lat)}; median cycle {statistics.median(r['cycle_ms'] for r in lat if 'cycle_ms' in r):.0f} ms")
led = [r for r in rows(P('mutation_receipts.jsonl')) if r.get('mutation_state') == 'applied_in_memory']
ops = collections.Counter(r.get('operation') for r in led)
print(f"mutation ledger (in-memory): {dict(ops)}")
pp = P('persistent_perception.json')
if os.path.exists(pp):
    objs = json.load(open(pp))
    vc = collections.Counter((o.get('bbox_fusion') or {}).get('view_count') for o in objs)
    print(f"final objects: {len(objs)}; view-count histogram {dict(sorted((k or 0, v) for k, v in vc.items()))}")
mp = P('merge_pending.json')
if os.path.exists(mp):
    m = json.load(open(mp))
    print(f"merge_pending: sweep {m.get('sweep')} needs_max {m.get('needs_max')} pending {m.get('pending')} engine {m.get('engine')}")

merges = [r for r in hd if r.get('kind') == 'merge']
ref = [r for r in hd if r.get('kind') == 'merge_refused']
print(f"\nMERGES: {len(merges)} (dry_run split {dict(collections.Counter(bool(r.get('dry_run')) for r in merges))})")
for r in merges:
    print(f"  {r.get('keeper_label')} <- {r.get('discarded_label')} sim/total={r.get('similarity')} rooms {r.get('keeper_room')}/{r.get('discard_room')}")
print(f"merge_refused rows: {len(ref)} by reason {dict(collections.Counter(r.get('reason') for r in ref))}")


def hold_kind(t):
    t = t or ''
    if 'vetoed' in t:
        return 'vetoed'
    if 'consecutive' in t:
        return 'held for persistence (streak)'
    if 'containment carries' in t:
        return 'containment carries (cross-kind / unmeasured)'
    if 'no positive overlap' in t:
        return 'no positive overlap'
    if '<' in t:
        return 'below threshold'
    return t[:40]


print("decision_reason split:", dict(collections.Counter(hold_kind(r.get('decision_reason')) for r in ref if r.get('reason') in ('hold', 'reject', 'abstain'))))
print("offered_by split (rows):", dict(collections.Counter(r.get('offered_by') for r in ref if r.get('offered_by'))))
ns = [r for r in hd if r.get('kind') == 'not_offered_summary']
print(f"not_offered_summary rows: {len(ns)}; last: {ns[-1].get('by_reason') if ns else None}")

# distinct pairs with detail
pairs = {}
for r in ref:
    key = tuple(sorted((r.get('object'), r.get('candidate'))))
    pairs.setdefault(key, []).append(r)
print(f"\ndistinct refused pairs: {len(pairs)}")
for key, rs in sorted(pairs.items(), key=lambda kv: -len(kv[1]))[:30]:
    r = rs[-1]
    ch = r.get('channels') or {}
    cov = ch.get('covisibility') or {}
    ov = ch.get('overlap') or {}
    at = ch.get('attributes') or {}
    print(f"  {r.get('a_label')} <-> {r.get('b_label')} x{len(rs)} reason={r.get('reason')} {hold_kind(r.get('decision_reason'))} "
          f"offered_by={r.get('offered_by')} total={r.get('hypothesis_total')} | cov n={cov.get('n_covisible')} ov2d={cov.get('overlap_2d')} "
          f"veto={cov.get('veto')} | overlap inter={ov.get('intersection_m3')} contain={ov.get('containment')} llr={ov.get('log_odds')} "
          f"| attr s={at.get('score')} same_kind={at.get('same_kind')} llr={at.get('log_odds')} | rooms {r.get('room_a')}/{r.get('room_b')} "
          f"gap={r.get('gap_m')}")
