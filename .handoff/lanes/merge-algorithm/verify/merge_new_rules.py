"""Measure the merge stage under the 2026-09-14 rules on the GA-493 bundle, GT-free.

Runs under mlspaces_310 with the REAL MiniLM (offline cache) and the REAL merge callback:
  ObjectServices._cb_merge_objects (dry run) -> _merge_candidates -> _assoc_build ->
  association.generate_candidates / score_pair / Hypothesis, with object_services'
  _attribute_channel_input -> pair_attribute_score -> nlp_utils.lost_similarity_detailed.
Nothing in the decision path is reimplemented here. Two identical dry-run sweeps simulate
merge_min_consecutive = 2; sweep 2's rows are the decisions.

Objects: the 104 final objects of persistent_perception.json, no observations (co-visibility
abstains, as it does for never-co-observed pairs). Two box wirings are measured:
  A  "as instructed": Object.bbox = fused_bbox, so every reader sees the fused box.
  B  "as production wires it": Object.bbox = measured bbox, Object.fused_bbox = fused box.
     The gates in _cb_merge_objects read locality_bounds (fused) but _assoc_build passes
     bbox=o.bbox (measured) to AssocObject, so generate_candidates and channel_overlap read the
     measured box.
Old-merge counterfactual: each discarded object is rebuilt from its creation observation (label,
colour, material, description, captured 3D box) and swept beside the 104 finals.
"""
import collections
import contextlib
import io
import json
import os
import sys
import types

# --- webcolors stand-in: CSS3 names present in this run; unknown names raise like the real one
CSS = {'white': (255, 255, 255), 'blue': (0, 0, 255), 'grey': (128, 128, 128), 'gray': (128, 128, 128),
       'brown': (165, 42, 42), 'green': (0, 128, 0), 'black': (0, 0, 0), 'silver': (192, 192, 192),
       'gold': (255, 215, 0), 'beige': (245, 245, 220), 'teal': (0, 128, 128), 'red': (255, 0, 0)}


class _RGB:
    def __init__(self, r, g, b):
        self.red, self.green, self.blue = r, g, b


def _name_to_rgb(name):
    if name not in CSS:
        raise ValueError(name)
    return _RGB(*CSS[name])


sys.modules['webcolors'] = types.SimpleNamespace(name_to_rgb=_name_to_rgb)
PM = '/DATA/GRAPH-API/.claude/worktrees/found-merge-algorithm/lost3dsg/src/perception_module'
sys.path.insert(0, PM)
os.chdir(PM)
os.environ.setdefault('HF_HOME', '/DATA/huggingface_cache')
os.environ.pop('GRAPH_API_OUTPUT_DIR', None)   # _publish_merge_pending writes nothing

import nlp_utils  # noqa: E402

assert nlp_utils.world2vec.st_model is not None, "real SentenceTransformer did not load"
import rosstub  # noqa: E402

# packages that are REAL in mlspaces_310 stay real; only ROS and the absent ones are stubbed
REAL_HERE = ('cv2', 'torch', 'sentence_transformers', 'matplotlib', 'PIL', 'shapely', 'scipy',
             'sklearn', 'onnxruntime', 'transformers', 'torchvision', 'webcolors')
rosstub.STUBBED = tuple(n for n in rosstub.STUBBED if n not in REAL_HERE)
rosstub.install()

import association as assoc  # noqa: E402
import object_info  # noqa: E402
import object_services as osv  # noqa: E402
import room_manager  # noqa: E402
from world_model import wm  # noqa: E402

OUT = '/DATA/GRAPH-API/.claude/worktrees/found-merge-algorithm/.handoff/lanes/merge-algorithm/verify/merge_new_rules.txt'
R = '/DATA/GRAPH-API/.handoff/lanes/perception/ga493_replay_20260914/'
K = ('x_min', 'x_max', 'y_min', 'y_max', 'z_min', 'z_max')
THR = assoc.commit_threshold(osv.MERGE_COST_RATIO)
out = []


def say(s=''):
    out.append(s)
    print(s)


say("COMPONENTS (rule 2)")
say(f"  interpreter          {sys.executable}")
say(f"  embedding model      {type(nlp_utils.world2vec.st_model).__name__} from "
    f"{sys.modules['sentence_transformers'].__file__}")
say(f"  association          {assoc.__file__}")
say(f"  object_services      {osv.__file__}")
say("  decision path        ObjectServices._cb_merge_objects (dry_run) -> _merge_candidates -> _assoc_build -> "
    "assoc.generate_candidates / score_pair / Hypothesis; attribute_score_fn = object_services._attribute_channel_input")
say(f"  config in force      MERGE_ENGINE={osv.MERGE_ENGINE} MERGE_ONTOLOGY_CHANNEL={osv.MERGE_ONTOLOGY_CHANNEL} "
    f"MERGE_ATTRIBUTE_MAX_LOG_ODDS={osv.MERGE_ATTRIBUTE_MAX_LOG_ODDS} LOCALITY_GAP_M={osv.LOCALITY_GAP_M} "
    f"SIM_THRESHOLD={osv.SIM_THRESHOLD} MERGE_COST_RATIO={osv.MERGE_COST_RATIO} threshold=log(20)={THR:.4f} "
    f"MERGE_MIN_EVIDENCE={osv.MERGE_MIN_EVIDENCE} MERGE_MIN_CONSECUTIVE={osv.MERGE_MIN_CONSECUTIVE} MERGE_KNN_K={osv.MERGE_KNN_K}")
say(f"  similarity weights   {[osv.CFG['similarity'][k] for k in ('label', 'color', 'material', 'description')]}")
assert osv.MERGE_ENGINE == 'evidence' and osv.MERGE_ONTOLOGY_CHANNEL is False
assert osv.MERGE_MIN_CONSECUTIVE == 2 and osv.MERGE_MIN_EVIDENCE == 1
assert [osv.CFG['similarity'][k] for k in ('label', 'color', 'material', 'description')] == [0.05, 0.45, 0.30, 0.20]

# --- bundle ---------------------------------------------------------------------------------
final_rows = json.load(open(R + 'bundle/persistent_perception.json'))
final = {o['object_id']: o for o in final_rows}
rows_old = [json.loads(line) for line in open(R + 'bundle/hook_decisions.jsonl')]
old_merges = [r for r in rows_old if r['kind'] == 'merge' and not r.get('dry_run')]
old_refused = collections.defaultdict(set)
for r in rows_old:
    if r['kind'] == 'merge_refused':
        old_refused[tuple(sorted((r['object'], r['candidate'])))].add(r['reason'])
obs = {}
for line in open(R + 'capture/consumer/events.jsonl'):
    e = json.loads(line)
    if e['kind'] != 'consumer_pair':
        continue
    p = e['payload']
    descs = {(d.get('observation') or {}).get('observation_id'): d for d in p['descriptions']['descriptions']}
    for b in p['bboxes']['boxes']:
        oid = (b.get('observation') or {}).get('observation_id')
        d = descs.get(oid, {})
        obs[oid] = {'label': b['label'], 'box': {k: float(b[k]) for k in K}, 'color': d.get('color', ''),
                    'material': d.get('material', ''), 'description': d.get('description', '')}
created = {}
for line in open(R + 'bundle/mutation_receipts.jsonl'):
    r = json.loads(line)
    if r['mutation_state'] == 'applied_in_memory' and r['operation'] == 'add':
        created[r['object_id']] = (r.get('observation') or {}).get('observation_id')


def base(label):
    return str(label).split('#')[0].strip().lower()


def bounds(box):
    return assoc._as_bounds(box)


def iou(a, b):
    ba, bb = bounds(a), bounds(b)
    inter = assoc.intersection_volume(ba, bb)
    union = assoc.box_volume(ba) + assoc.box_volume(bb) - inter
    return inter / union if union > 0 else 0.0


def make_object(o, mode):
    box = o.get('fused_bbox') or o['bbox']
    live = box if mode == 'A' else o['bbox']
    obj = object_info.Object(o['label'], None, {k: float(live[k]) for k in K}, description=o['description'],
                             color=o['color'], material=o['material'], object_id=o['object_id'])
    obj.fused_bbox = {k: float(box[k]) for k in K}
    obj.room_id = o.get('room_id')
    obj.creation_time = o.get('creation_time')
    obj.admission_grade = o.get('admission_grade')
    obj.admission_filled = o.get('admission_filled')
    return obj


def make_discard(oid):
    """The discarded side of an old merge, as it was at creation (no final record exists)."""
    ob = obs.get(created.get(oid))
    if ob is None:
        return None
    obj = object_info.Object(ob['label'], None, dict(ob['box']), description=ob['description'],
                             color=ob['color'], material=ob['material'], object_id=oid)
    obj.fused_bbox = None
    obj.room_id = None
    return obj


def make_svc(objects):
    svc = osv.ObjectServices.__new__(osv.ObjectServices)
    svc.get_logger = lambda: rosstub.Any()
    svc.log_both = lambda *a, **k: None
    svc.room_manager = room_manager.RoomManager.__new__(room_manager.RoomManager)
    svc.room_manager.scene_graph = {}
    svc.room_manager.current_room_id = None
    room_of = {id(o.bbox): o.room_id for o in objects}
    svc.room_manager.room_at_bbox = lambda bbox: room_of.get(id(bbox))
    svc._hypotheses, svc._merge_sweep = {}, 0
    # 2026-09-14 review fix: dry runs accumulate in their own store; this counterfactual runs
    # dry sweeps only, so alias the two stores and keep reading svc._hypotheses below.
    svc._dry_hypotheses = svc._hypotheses
    svc.rows = []

    class _Log:
        def write(self, kind, oid, **kw):
            svc.rows.append((kind, oid, kw))

    svc.decision_log = _Log()
    return svc


def sweep(svc, objects):
    wm.persistent_perceptions.clear()
    wm.persistent_perceptions.extend(objects)
    svc.rows = []
    req, resp = rosstub.Any(), rosstub.Any()
    req.max_distance, req.min_similarity, req.dry_run = 0.8, 0.95, True
    with contextlib.redirect_stdout(io.StringIO()):
        osv.ObjectServices._cb_merge_objects(svc, req, resp)
    assert resp.success is True, resp.message
    return list(svc.rows), resp.merged_count


def hold_kind(reason_text):
    t = reason_text or ''
    if 'containment carries' in t:
        return 'containment_unchecked'
    if 'consecutive' in t:
        return 'streak'
    if '<' in t:
        return 'below_threshold'
    return 'other:' + t[:40]


def pair_state(h):
    """Pair-level verdict from the hypothesis state alone (no cascade, no streak)."""
    if h.vetoed_by:
        return 'reject'
    last = h.history[-1] if h.history else {}
    if h.total >= THR:
        return 'hold:containment_unchecked' if last.get('containment_unchecked') else 'merge_eligible'
    return 'hold:below_threshold'


def run_pass(mode, extra=()):
    objects = [make_object(o, mode) for o in final_rows] + list(extra)
    by_id = {o.object_id: o for o in objects}
    svc = make_svc(objects)
    rows1, n1 = sweep(svc, objects)
    rows2, n2 = sweep(svc, objects)
    # the same real candidate generator, called directly for the offered_by breakdown
    ctx, built = osv.ObjectServices._assoc_build(svc, objects)
    offered, excluded = assoc.generate_candidates([built[id(o)] for o in objects if id(o) in built], ctx,
                                                  k=osv.MERGE_KNN_K)
    return dict(mode=mode, objects=objects, by_id=by_id, svc=svc, rows1=rows1, rows2=rows2, n1=n1, n2=n2,
                ctx=ctx, offered=offered, excluded=excluded)


def report_pass(P):
    objects, rows1, rows2, svc = P['objects'], P['rows1'], P['rows2'], P['svc']
    n = len(objects)
    all_pairs = n * (n - 1) // 2
    by_kind = collections.Counter(k for k, _, _ in rows2)
    reasons2 = collections.Counter(kw.get('reason') for k, _, kw in rows2 if k == 'merge_refused')
    reasons1 = collections.Counter(kw.get('reason') for k, _, kw in rows1 if k == 'merge_refused')
    not_off = [kw for k, _, kw in rows2 if k == 'not_offered_summary']
    n_not_off = sum(kw['n_pairs'] for kw in not_off)
    n_offered = len(P['offered'])
    by_basis = collections.Counter(m['offered_by'] for _, _, m in P['offered'])
    say(f"\n=== PASS {P['mode']}: {'Object.bbox = fused_bbox (as instructed)' if P['mode'] == 'A' else 'Object.bbox = measured bbox, fused_bbox beside it (as _assoc_build wires it)'} ===")
    say(f"  objects {n}; all pairs {all_pairs}; map_volume_m3 {P['ctx'].map_volume_m3:.1f}; n_rooms {P['ctx'].n_rooms}")
    say(f"  generate_candidates: offered {n_offered}/{all_pairs} (shell {by_basis.get('shell', 0)}, gap-only {by_basis.get('gap', 0)}); "
        f"excluded {len(P['excluded'])}/{all_pairs}; callback's not_offered_summary {n_not_off} "
        f"(agree: {n_offered + n_not_off == all_pairs})")
    say(f"  sweep 1 (streak 1/2): merged_count {P['n1']}; refusals by reason {dict(reasons1)}")
    say(f"  sweep 2 (streak 2/2): merged_count {P['n2']}; rows {dict(by_kind)}; refusals by reason {dict(reasons2)}")
    holds2 = collections.Counter(hold_kind(kw.get('decision_reason')) for k, _, kw in rows2
                                 if k == 'merge_refused' and kw.get('reason') == 'hold')
    say(f"  sweep-2 HOLD by reason {dict(holds2)}  (denominator: {reasons2.get('hold', 0)} holds)")
    logged2 = sum(1 for k, _, _ in rows2 if k in ('merge', 'merge_refused'))
    say(f"  sweep-2 offered pairs with a row {logged2}/{n_offered}; silent skips (a already condemned) {n_offered - logged2}")
    # pair-level state, independent of the in-sweep cascade
    states = collections.Counter(pair_state(h) for h in svc._hypotheses.values())
    say(f"  hypotheses after sweep 2: {len(svc._hypotheses)} (= pairs that reached the evidence decision); "
        f"pair-level state {dict(states)}")
    merges = [(oid, kw) for k, oid, kw in rows2 if k == 'merge']
    say(f"\n  MERGES decided in sweep 2: {len(merges)}")
    cross = 0
    merge_keys = set()
    for oid, kw in merges:
        a, b = P['by_id'][oid], P['by_id'][kw['merged_from']]
        key = tuple(sorted((a.object_id, b.object_id)))
        merge_keys.add(key)
        h = svc._hypotheses[key]
        score, ev = osv.pair_attribute_score(a, b)
        same = base(a.label) == base(b.label)
        cross += not same
        st = {k: round(v, 2) for k, v in h.state.items()}
        say(f"    {a.label:>14s} + {b.label:<14s} same_kind={same!s:5s} IoU={iou(a.bbox, b.bbox):.3f} "
            f"attr={score:.3f} (n_opt {ev['optional_count']}) log-odds={h.total:.2f} {st} "
            f"rooms {a.room_id}/{b.room_id}")
    say(f"  cross-kind merges (base labels differ): {cross}/{len(merges)}")
    no_inter = [k for k in merge_keys if svc._hypotheses[k].state.get('overlap', 0.0) <= 0.0]
    cross_room = [k for k in merge_keys if P['by_id'][k[0]].room_id != P['by_id'][k[1]].room_id]
    say(f"  merges with NO box intersection (carried by attributes + room): {len(no_inter)}/{len(merges)}; "
        f"merges across rooms (boxes overlap, room gate passed): {len(cross_room)}/{len(merges)}")
    elig = {k for k, h in svc._hypotheses.items() if pair_state(h) == 'merge_eligible'}
    lost = elig - merge_keys
    say(f"  merge-eligible pairs by state {len(elig)}; not turned into a merge row by the in-sweep cascade "
        f"(a side already condemned) {len(lost)}/{len(elig)}")
    for key in sorted(lost):
        a, b = P['by_id'][key[0]], P['by_id'][key[1]]
        say(f"    cascade-lost: {a.label} + {b.label} same_kind={base(a.label) == base(b.label)} "
            f"IoU={iou(a.bbox, b.bbox):.3f} log-odds={svc._hypotheses[key].total:.2f}")
    # holds on containment_unchecked: what kind of pairs are held
    cu = [(k, h) for k, h in svc._hypotheses.items() if pair_state(h) == 'hold:containment_unchecked']
    cu_same = sum(1 for k, h in cu if base(P['by_id'][k[0]].label) == base(P['by_id'][k[1]].label))
    say(f"  held on containment_unchecked: {len(cu)}; same base label {cu_same}, different {len(cu) - cu_same}")
    for k, h in sorted(cu, key=lambda kh: -kh[1].total)[:12]:
        a, b = P['by_id'][k[0]], P['by_id'][k[1]]
        attrs = h.state.get('attributes')
        say(f"    {a.label:>14s} + {b.label:<14s} log-odds={h.total:.2f} attr_llr={attrs if attrs is None else round(attrs, 2)} "
            f"IoU={iou(a.bbox, b.bbox):.3f}")
    say(f"  the {cu_same} SAME-kind pairs held on containment_unchecked (attribute channel not positive, so no second witness):")
    for k, h in sorted(cu, key=lambda kh: -kh[1].total):
        a, b = P['by_id'][k[0]], P['by_id'][k[1]]
        if base(a.label) != base(b.label):
            continue
        score, ev = osv.pair_attribute_score(a, b)
        say(f"    {a.label:>14s} + {b.label:<14s} IoU={iou(a.bbox, b.bbox):.3f} attr={score:.3f} (n_opt {ev['optional_count']}) "
            f"colour {a.color}/{b.color} material {a.material}/{b.material} "
            f"attr_llr={round(h.state.get('attributes', 0.0), 2)}")
    return merge_keys


def old_comparison(P, merge_keys):
    svc, by_id = P['svc'], P['by_id']
    verdict = {}
    for k, oid, kw in P['rows2']:
        if k == 'merge':
            verdict[tuple(sorted((oid, kw['merged_from'])))] = 'merge'
        elif k == 'merge_refused' and kw.get('reason') in ('room', 'geometry', 'already_condemned'):
            verdict.setdefault(tuple(sorted((oid, kw['candidate']))), kw['reason'])
    for key, h in svc._hypotheses.items():
        verdict[key] = 'merge' if key in merge_keys else pair_state(h)
    say(f"\n  OLD ENGINE (legacy, hook_decisions.jsonl): {len(old_merges)} merges, {len(old_refused)} distinct refused pairs")
    both = {k: v for k, v in old_refused.items() if k[0] in by_id and k[1] in by_id}
    say(f"  old refused pairs with both sides among the 104 finals: {len(both)}/{len(old_refused)}")

    def first_reason(rs):
        for r in ('similarity', 'room', 'distance', 'already_condemned'):
            if r in rs:
                return r
        return sorted(rs)[0]

    tab = collections.defaultdict(collections.Counter)
    for k, rs in both.items():
        tab[first_reason(rs)][verdict.get(k, 'not_offered')] += 1
    for reason, c in sorted(tab.items()):
        say(f"    old reason {reason:18s} n={sum(c.values()):4d} -> new: {dict(c)}")
    p2 = [k for k, rs in both.items() if 'similarity' in rs
          and base(by_id[k[0]].label) == base(by_id[k[1]].label)
          and iou(by_id[k[0]].fused_bbox, by_id[k[1]].fused_bbox) > 0]
    c2 = collections.Counter(verdict.get(k, 'not_offered') for k in p2)
    say(f"  old similarity-refused, same base label, fused boxes overlap: {len(p2)} -> new: {dict(c2)}")
    for k in p2:
        a, b = by_id[k[0]], by_id[k[1]]
        h = svc._hypotheses.get(k)
        say(f"    {a.label:>14s} + {b.label:<14s} fusedIoU={iou(a.fused_bbox, b.fused_bbox):.3f} "
            f"new={verdict.get(k, 'not_offered'):26s} log-odds={'-' if h is None else round(h.total, 2)}")


def old_merge_counterfactual(mode):
    discards = {r['merged_from']: make_discard(r['merged_from']) for r in old_merges}
    missing = [oid for oid, o in discards.items() if o is None]
    extra = [o for o in discards.values() if o is not None]
    P = run_pass(mode, extra=extra)
    svc, by_id = P['svc'], P['by_id']
    verdict = {}
    for k, oid, kw in P['rows2']:
        if k == 'merge':
            verdict[tuple(sorted((oid, kw['merged_from'])))] = 'merge'
        elif k == 'merge_refused' and kw.get('reason') in ('room', 'geometry', 'already_condemned'):
            verdict.setdefault(tuple(sorted((oid, kw['candidate']))), kw['reason'])
    for key, h in svc._hypotheses.items():
        if verdict.get(key) != 'merge':
            verdict[key] = pair_state(h)
    say(f"\n  OLD MERGES re-decided (pass {mode}; discard rebuilt from its creation observation, keeper = final object): "
        f"{len(old_merges)} pairs, {len(missing)} discards not reconstructable")
    c = collections.Counter()
    for r in old_merges:
        key = tuple(sorted((r['object'], r['merged_from'])))
        v = verdict.get(key, 'not_offered' if r['merged_from'] not in missing else 'unreconstructable')
        c[v] += 1
        a, b = by_id.get(key[0]), by_id.get(key[1])
        h = svc._hypotheses.get(key)
        extra_s = ''
        if a is not None and b is not None:
            score, ev = osv.pair_attribute_score(a, b)
            extra_s = (f"IoU={iou(a.bbox, b.bbox):.3f} gap={assoc.box_gap(bounds(a.bbox), bounds(b.bbox)):.2f}m "
                       f"attr_new={score:.3f} (n_opt {ev['optional_count']}) log-odds={'-' if h is None else round(h.total, 2)}")
        say(f"    {r['keeper_label']:>10s} + {r['discarded_label']:<10s} old_sim={r['similarity']:.3f} new={v:26s} {extra_s}")
    say(f"  summary: {dict(c)} of {len(old_merges)}")
    # pair-level state for merge_eligible means: two consecutive sweeps over threshold, no
    # containment_unchecked hold, no veto -- i.e. the callback merges it unless the cascade
    # gives one side to an earlier pair in the same sweep.


PA = run_pass('A')
mk_a = report_pass(PA)
old_comparison(PA, mk_a)
old_merge_counterfactual('A')
PB = run_pass('B')
mk_b = report_pass(PB)
old_comparison(PB, mk_b)
old_merge_counterfactual('B')

say("\nCAVEATS")
say("  - Final fused boxes are not the boxes at sweep time: the old engine decided on single-view measured boxes")
say("    while objects were being created, and the keeper's final fused box already contains the views merged into it.")
say("  - room gate reads the stored room_id of each final object; live it reads room_manager.room_at_bbox(measured bbox).")
say("  - no observations -> co-visibility abstains (uncollected), separation abstains (no covariance), appearance abstains.")
say("  - the discarded sides of the 15 old merges carry their creation-time attributes and captured box, no fused box.")
open(OUT, 'w').write('\n'.join(out) + '\n')
print(f"\nwritten {OUT}")
