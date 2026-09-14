"""Association stage on the GA-493 bundle under the NEW rules (owner rulings 2026-09-14) versus the
OLD rules (baseline 16880b0, the code that produced the bundle). GT-free.

Runs under mlspaces_310 with the REAL MiniLM (offline cache):
  HF_HOME=/DATA/huggingface_cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  /home/xps/miniconda3/envs/mlspaces_310/bin/python assoc_new_rules.py

WHAT ANSWERS (rule 2)
  locality NEW  : association.geometry_compatible / association._as_bounds, imported from the worktree
                  perception_module (the real function object_manager_6.locality_ok calls).
  locality OLD  : a local reimplementation of utils.compute_iou_3d (utils.py imports rclpy/cv_bridge and
                  cannot load here); same formula, same 0.01 m minimum-size expansion.
  attributes    : the REAL nlp_utils.lost_similarity_detailed with the REAL SentenceTransformer
                  all-MiniLM-L6-v2 (asserted below), weights switched through CFG["similarity"]
                  (that function reads CFG on every call). webcolors is a stand-in (CSS3 values for
                  every colour word in this run; a name CSS3 does not know raises, as the real one does).

WHAT THE REPLAY HAS (and has not)
  The world model is rebuilt in time order from the mutation ledger (add / update /
  direct_exploration_update) and the merge and disappearance records; each object's box is the box
  of its LAST ACCEPTED VIEW (what obj.bbox holds in the live loop) -- the replay has NO fused box, so
  locality_bounds' fused-box branch cannot be exercised here and the measured box stands in for it.
  Each object's colour / material / description are those of the view that CREATED it:
  object_manager_6.modify_existing_object keeps the stored attributes on an update, and obj.embedding
  is computed once from obj.description.
  The same-cycle veto is replayed as: an object the ledger added or updated earlier in the same
  cycle_id, or one the counterfactual associated earlier in the same cycle, is skipped.
  The replay keeps the ledger's world (the add still happens even when the counterfactual would have
  associated), so a counterfactual association never changes later candidates.
"""
import collections
import json
import os
import sys
import types

import numpy as np

# --- webcolors stand-in (CSS3 values; unknown names raise like the real one) --------------------
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
from association import _as_bounds, geometry_compatible  # noqa: E402
from config import CFG  # noqa: E402
from nlp_utils import get_embedding, lost_similarity_detailed, world2vec  # noqa: E402

assert type(world2vec.st_model).__name__ == 'SentenceTransformer', "real MiniLM did not load"
assert world2vec.w2v_model is None
NEW_W = dict(CFG['similarity'])
assert NEW_W == {'label': 0.05, 'color': 0.45, 'material': 0.3, 'description': 0.2}, NEW_W
OLD_W = {'label': 0.05, 'color': 0.30, 'material': 0.15, 'description': 0.50}
SIM_THRESHOLD = float(CFG['association']['sim_threshold'])
assert SIM_THRESHOLD == 0.85
GAP_M = float(CFG['association']['association_margin_m'])
assert GAP_M == 0.3
EXPL_IOU = float(CFG['association']['exploration_iou_threshold'])
TRACK_IOU = float(CFG['association']['tracking_iou_threshold'])
assert (EXPL_IOU, TRACK_IOU) == (0.10, 0.30)

R = '/DATA/GRAPH-API/.handoff/lanes/perception/ga493_replay_20260914/'
OUT = '/DATA/GRAPH-API/.claude/worktrees/found-merge-algorithm/.handoff/lanes/merge-algorithm/verify/assoc_new_rules.txt'
K = ['x_min', 'y_min', 'z_min', 'x_max', 'y_max', 'z_max']
_out = open(OUT, 'w')


def say(*a):
    s = ' '.join(str(x) for x in a)
    print(s)
    _out.write(s + '\n')


def box(b):
    return {k: float(b[k]) for k in K}


def iou_legacy(b1, b2, min_size=0.01):
    """Reimplementation of utils.compute_iou_3d (same formula, same min-size expansion)."""
    def expand(b):
        e = dict(b)
        for ax in 'xyz':
            if b[f'{ax}_max'] - b[f'{ax}_min'] < min_size:
                c = (b[f'{ax}_min'] + b[f'{ax}_max']) / 2
                e[f'{ax}_min'], e[f'{ax}_max'] = c - min_size / 2, c + min_size / 2
        return e
    a, b = expand(b1), expand(b2)
    d = [min(a[f'{ax}_max'], b[f'{ax}_max']) - max(a[f'{ax}_min'], b[f'{ax}_min']) for ax in 'xyz']
    if any(x < 0 for x in d):
        return 0.0
    inter = d[0] * d[1] * d[2]
    v = [(e['x_max'] - e['x_min']) * (e['y_max'] - e['y_min']) * (e['z_max'] - e['z_min']) for e in (a, b)]
    union = v[0] + v[1] - inter
    return inter / union if union > 0 else 0.0


def base(label):
    return label.split('#')[0]


def score(det, cand, weights):
    """The REAL scorer, the weights it reads swapped in CFG for this call."""
    CFG['similarity'] = weights
    s, ev = lost_similarity_detailed(world2vec, base(det['label']), base(cand['label']), det['color'], cand['color'],
                                     det['material'], cand['material'], det['emb'], cand['emb'])
    return float(s), ev['optional_count']


# 1. every captured detection, keyed by observation_id
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
        obs[oid] = {'label': b['label'], 'box': box(b), 'expl': bool(p.get('exploration_mode')),
                    'color': d.get('color', ''), 'material': d.get('material', ''),
                    'description': d.get('description', ''), 'cycle': e['cycle_id'], 'obs_id': oid}
for o in obs.values():
    o['emb'] = get_embedding(world2vec, o['description'])
say(f"captured detections with 3D box: {len(obs)}; with a description embedding: "
    f"{sum(1 for o in obs.values() if o['emb'] is not None)}")

# 2. events in time order
events = []
for line in open(R + 'bundle/mutation_receipts.jsonl'):
    r = json.loads(line)
    if r['mutation_state'] != 'applied_in_memory':
        continue
    events.append((r['recorded_at'], r['mutation_sequence'], r['operation'], r['object_id'],
                   (r.get('observation') or {}).get('observation_id')))
for line in open(R + 'bundle/hook_decisions.jsonl'):
    r = json.loads(line)
    if r['kind'] == 'merge' and not r.get('dry_run'):
        events.append((r['t'], 10**9, 'merge', r['merged_from'], None))
    elif r['kind'] == 'disappearance_removal':
        events.append((r['t'], 10**9, 'delete', r.get('object'), None))
events.sort(key=lambda e: (e[0], e[1]))
say(f"ledger events: {dict(collections.Counter(e[2] for e in events))}")


def decide(det, state, veto, mode_expl, rules):
    """One association decision. Returns dict(geo=[ids passing locality], scored=[(id, score)],
    best=(id, score) or None, best_unvetoed=(id, score) or None)."""
    geo, scored = [], []
    for eid, s in state.items():
        if rules == 'new':
            ok = geometry_compatible(_as_bounds(det['box']), _as_bounds(s['box']), GAP_M)
        else:
            ok = iou_legacy(det['box'], s['box']) >= (EXPL_IOU if mode_expl else TRACK_IOU)
        if not ok:
            continue
        geo.append(eid)
        sc, _n = score(det, s, NEW_W if rules == 'new' else OLD_W)
        scored.append((eid, sc))
    passing = [(eid, sc) for eid, sc in scored if sc > SIM_THRESHOLD]
    best = max(passing, key=lambda x: x[1]) if passing else None
    unv = [(eid, sc) for eid, sc in passing if eid not in veto]
    best_unv = max(unv, key=lambda x: x[1]) if unv else None
    return {'geo': geo, 'scored': scored, 'passing': passing, 'best': best, 'best_unvetoed': best_unv}


# 3. replay
state = {}            # object_id -> {label, box, color, material, description, emb, created_obs}
rows = []             # one per ADD
upd_rows = []         # one per ledger association (update / direct_exploration_update)
cur_cycle, veto = None, {}    # veto: object_id -> what touched it earlier in this cycle
for t, seq, op, oid, obs_id in events:
    if op in ('merge', 'delete'):
        state.pop(oid, None)
        continue
    o = obs[obs_id]
    if o['cycle'] != cur_cycle:
        cur_cycle, veto = o['cycle'], {}
    if op != 'add':
        # a ledger association: the OLD rules accepted it. Ask what the NEW rules say about the SAME target.
        s = state.get(oid)
        if s is not None:
            geo = geometry_compatible(_as_bounds(o['box']), _as_bounds(s['box']), GAP_M)
            sc_new, n_new = score(o, s, NEW_W)
            sc_old, _ = score(o, s, OLD_W)
            upd_rows.append({'op': op, 'oid': oid, 'label': o['label'], 'geo_new': geo, 'iou': iou_legacy(o['box'], s['box']),
                             'sc_new': sc_new, 'sc_old': sc_old, 'vetoed': veto.get(oid), 'optional': n_new,
                             'expl': o['expl'], 'det': o, 'cand': dict(s), 'cycle': o['cycle']})
            s['box'] = o['box']
        veto.setdefault(oid, f'ledger {op} of {o["label"]}')
        continue
    new = decide(o, state, veto, o['expl'], 'new')
    old = decide(o, state, set(), o['expl'], 'old')   # OLD rules had no same-cycle veto
    rows.append({'t': t, 'oid': oid, 'obs': o, 'cycle': o['cycle'], 'expl': o['expl'], 'new': new, 'old': old,
                 'state_n': len(state), 'veto_n': len(veto), 'veto_why': dict(veto),
                 'cands': {eid: dict(state[eid]) for eid in set(new['geo']) | set(old['geo'])}})
    state[oid] = {'label': o['label'], 'box': o['box'], 'color': o['color'], 'material': o['material'],
                  'description': o['description'], 'emb': o['emb'], 'created_obs': obs_id}
    veto.setdefault(oid, f'ledger add of {o["label"]}')
    if new['best_unvetoed']:
        veto.setdefault(new['best_unvetoed'][0], f'counterfactual association of {o["label"]}')
say(f"adds replayed: {len(rows)}; ledger associations replayed: {len(upd_rows)}; final objects in replay: {len(state)}")
N = len(rows)


def outcome(r, rules):
    d = r[rules]
    if d['best_unvetoed']:
        eid, sc = d['best_unvetoed']
        return 'ASSOC_same_label' if base(r['cands'][eid]['label']) == base(r['obs']['label']) else 'ASSOC_diff_label'
    if not d['geo']:
        return 'NEW_no_geometry_compatible_candidate'
    if d['best']:
        return 'NEW_same_cycle_veto'
    return 'NEW_candidate_but_score_le_threshold'


for rules, title in (('new', 'NEW rules: geometry_compatible(gap<=0.3 m, measured box), weights .05/.45/.30/.20, '
                             'best score > 0.85, same-cycle veto'),
                     ('old', 'OLD rules (control): IoU >= 0.10 expl / 0.30 track on measured box, weights .05/.30/.15/.50, '
                             'score > 0.85, no veto')):
    say(f"\n=== {title} ===")
    tally = collections.Counter(outcome(r, rules) for r in rows)
    by_mode = collections.Counter((r['expl'], outcome(r, rules)) for r in rows)
    for c in ('ASSOC_same_label', 'ASSOC_diff_label', 'NEW_no_geometry_compatible_candidate',
              'NEW_candidate_but_score_le_threshold', 'NEW_same_cycle_veto'):
        say(f"  {c:40s} {tally[c]:4d} of {N}   (expl {by_mode[(True, c)]:3d} of {sum(1 for r in rows if r['expl'])}, "
            f"track {by_mode[(False, c)]:3d} of {sum(1 for r in rows if not r['expl'])})")
    assoc = tally['ASSOC_same_label'] + tally['ASSOC_diff_label']
    say(f"  would ASSOCIATE instead of add: {assoc} of {N}; still a new object: {N - assoc} of {N}")
    geo_any = sum(1 for r in rows if r[rules]['geo'])
    say(f"  adds with >= 1 locality-passing candidate: {geo_any} of {N}; "
        f"with >= 2 candidates above the score gate (exploration loop takes the FIRST in index order, "
        f"tracking the best): {sum(1 for r in rows if len(r[rules]['passing']) >= 2)} of {N}")
    acc = [r[rules]['best_unvetoed'][1] for r in rows if r[rules]['best_unvetoed']]
    ref = [max(sc for _, sc in r[rules]['scored']) for r in rows if r[rules]['scored'] and not r[rules]['best_unvetoed']]
    allsc = [sc for r in rows for _, sc in r[rules]['scored']]

    def dist(name, xs):
        if not xs:
            say(f"  {name}: n=0")
            return
        xs = np.array(xs)
        say(f"  {name}: n={len(xs)} min {xs.min():.3f} p10 {np.percentile(xs, 10):.3f} median {np.median(xs):.3f} "
            f"p90 {np.percentile(xs, 90):.3f} max {xs.max():.3f}")
    dist("score of the ACCEPTED candidate (best, unvetoed) per associated add", acc)
    dist("best score among REFUSED locality-passing candidates per still-new add", ref)
    dist("every (add, locality-passing candidate) pair score", allsc)
    hist = collections.Counter()
    for r in rows:
        for _, sc in r[rules]['scored']:
            hist['<0.50' if sc < 0.5 else '0.50-0.70' if sc < 0.7 else '0.70-0.85' if sc <= 0.85 else '0.85-0.925' if sc < 0.925 else '>=0.925'] += 1
    say(f"  pair-score histogram (n={sum(hist.values())} pairs): " + ", ".join(f"{k} {hist[k]}" for k in ('<0.50', '0.50-0.70', '0.70-0.85', '0.85-0.925', '>=0.925')))

# 4. what the NEW-accepted pairs look like on the OLD gate's instrument (IoU on the measured box)
say("\n=== NEW-rule associations: IoU on the measured box (what the OLD gate read) ===")
bins = collections.Counter()
for r in rows:
    b = r['new']['best_unvetoed']
    if not b:
        continue
    v = iou_legacy(r['obs']['box'], r['cands'][b[0]]['box'])
    thr = EXPL_IOU if r['expl'] else TRACK_IOU
    bins['IoU=0 (gap only)' if v == 0 else f'0<IoU<{thr} (below the old gate)' if v < thr else f'IoU>={thr} (old gate passed; old score refused)'] += 1
nA = sum(1 for r in rows if r['new']['best_unvetoed'])
for k, v in sorted(bins.items()):
    say(f"  {k:48s} {v:3d} of {nA}")

# 5. every different-label absorption (RISK): both descriptions
say("\n=== RISK: NEW-rule associations to a DIFFERENT base label (every one) ===")
n_diff = 0
for r in rows:
    b = r['new']['best_unvetoed']
    if not b:
        continue
    c = r['cands'][b[0]]
    if base(c['label']) == base(r['obs']['label']):
        continue
    n_diff += 1
    o = r['obs']
    sc_old, _ = score(o, c, OLD_W)
    say(f"  [{n_diff}] add {o['label']} (obj {r['oid'][:8]}) -> absorbed by {c['label']} (obj {b[0][:8]})  "
        f"score NEW {b[1]:.3f} OLD {sc_old:.3f}  IoU {iou_legacy(o['box'], c['box']):.2f}  "
        f"same-label passing alternatives: {sum(1 for eid, _ in r['new']['passing'] if base(r['cands'][eid]['label']) == base(o['label']))}")
    say(f"       new: {o['color']}/{o['material']}: {o['description']!r}")
    say(f"       old: {c['color']}/{c['material']}: {c['description']!r}")
say(f"  different-label absorptions: {n_diff} of {nA} NEW-rule associations (of {N} adds)")

# 6. every same-label association, one line each (for the reader who wants to eyeball them)
say("\n=== NEW-rule associations to the SAME base label (every one) ===")
for r in rows:
    b = r['new']['best_unvetoed']
    if not b:
        continue
    c = r['cands'][b[0]]
    if base(c['label']) != base(r['obs']['label']):
        continue
    o = r['obs']
    say(f"  {o['label']:14s} {'expl' if r['expl'] else 'track'} score {b[1]:.3f} IoU {iou_legacy(o['box'], c['box']):.2f}  "
        f"{o['color']}/{o['material']} vs {c['color']}/{c['material']}")
    say(f"       new: {o['description'][:90]!r}")
    say(f"       old: {c['description'][:90]!r}")

# 7. vetoed: which adds the same-cycle veto turned from ASSOC into NEW
say("\n=== NEW rules: adds where the same-cycle veto changed the outcome ===")
for r in rows:
    d = r['new']
    if d['best'] and not d['best_unvetoed']:
        c = r['cands'][d['best'][0]]
        say(f"  {r['obs']['label']:14s} would take {c['label']} at {d['best'][1]:.3f}, vetoed: "
            f"{r['veto_why'].get(d['best'][0])} earlier in cycle {r['cycle'][:8]}")

# 8. the ledger's own associations (OLD rules accepted them): what the NEW rules say about the same target
say(f"\n=== the {len(upd_rows)} ledger associations (19 update + 1 direct_exploration_update): same target under NEW rules ===")
say(f"  geometry_compatible True: {sum(1 for u in upd_rows if u['geo_new'])} of {len(upd_rows)}; "
    f"NEW score > 0.85: {sum(1 for u in upd_rows if u['sc_new'] > SIM_THRESHOLD)} of {len(upd_rows)}; "
    f"OLD score > 0.85 (should be all): {sum(1 for u in upd_rows if u['sc_old'] > SIM_THRESHOLD)} of {len(upd_rows)}; "
    f"target vetoed by same-cycle rule: {sum(1 for u in upd_rows if u['vetoed'])} of {len(upd_rows)}")
say(f"  control on the replay's boxes: OLD IoU gate (0.10 expl / 0.30 track) passed by "
    f"{sum(1 for u in upd_rows if u['iou'] >= (EXPL_IOU if u['expl'] else TRACK_IOU))} of {len(upd_rows)} "
    f"(the exploration-mode 'update' rows came through check_tracking_transition, which has no IoU gate)")
for u in upd_rows:
    flag = '' if u['sc_new'] > SIM_THRESHOLD and u['geo_new'] and not u['vetoed'] else \
        f"   <-- NEW rules would NOT associate{' (veto: ' + u['vetoed'] + ')' if u['vetoed'] else ''}"
    say(f"  {u['op']:26s} {u['label']:14s} {'expl' if u['expl'] else 'track'} IoU {u['iou']:.2f} old {u['sc_old']:.3f} new {u['sc_new']:.3f} "
        f"optional_terms {u['optional']}{flag}")

# 9. OLD-rule control: it must reproduce ~0 associations (these WERE adds)
say("\n=== control: OLD rules on the 119 adds (the ledger says every one was an add) ===")
for r in rows:
    b = r['old']['best_unvetoed']
    if b:
        c = r['cands'][b[0]]
        o = r['obs']
        say(f"  OLD would have associated {o['label']} -> {c['label']} at {b[1]:.3f} IoU {iou_legacy(o['box'], c['box']):.2f} "
            f"({'expl' if r['expl'] else 'track'}, cycle {r['cycle'][:8]})")
        say(f"       new: {o['color']}/{o['material']}: {o['description'][:90]!r}")
        say(f"       old: {c['color']}/{c['material']}: {c['description'][:90]!r}")
say(f"  OLD-rule associations among the adds: {sum(1 for r in rows if r['old']['best_unvetoed'])} of {N}")

# 10. the ledger associations the NEW rules would refuse: the four raw terms, from the real functions
from nlp_utils import _known, color_similarity_rgb, cosine_similarity, semantic_similarity  # noqa: E402

say("\n=== ledger associations the NEW rules would refuse: raw terms (real nlp_utils functions) ===")
for u in upd_rows:
    if u['sc_new'] > SIM_THRESHOLD and u['geo_new'] and not u['vetoed']:
        continue
    d, c = u['det'], u['cand']
    terms = {'label': semantic_similarity(world2vec, base(d['label']), base(c['label']))}
    terms['color'] = color_similarity_rgb(d['color'], c['color'], world2vec) if _known(d['color']) and _known(c['color']) else None
    terms['material'] = semantic_similarity(world2vec, d['material'], c['material']) if _known(d['material']) and _known(c['material']) else None
    terms['description'] = float(cosine_similarity(d['emb'], c['emb'])) if d['emb'] is not None and c['emb'] is not None else None
    say(f"  {u['op']} {u['label']} (cycle {u['cycle'][:8]}): terms " + ", ".join(f"{k} {v if v is None else round(v, 3)}" for k, v in terms.items())
        + f" -> old {u['sc_old']:.3f} new {u['sc_new']:.3f}; veto: {u['vetoed']}")
    say(f"       det: {d['label']} {d['color']}/{d['material']}: {d['description']!r}")
    say(f"       obj: {c['label']} {c['color']}/{c['material']}: {c['description']!r}")

# 11. gap of the NEW-accepted pairs (what geometry_compatible read)
from association import box_gap  # noqa: E402

say("\n=== NEW-rule associations: largest per-axis gap between the detection box and the candidate's measured box ===")
gh = collections.Counter()
for r in rows:
    b = r['new']['best_unvetoed']
    if not b:
        continue
    g = box_gap(_as_bounds(r['obs']['box']), _as_bounds(r['cands'][b[0]]['box']))
    gh['0 (boxes intersect)' if g == 0 else '(0, 0.1] m' if g <= 0.1 else '(0.1, 0.2] m' if g <= 0.2 else '(0.2, 0.3] m'] += 1
for k in ('0 (boxes intersect)', '(0, 0.1] m', '(0.1, 0.2] m', '(0.2, 0.3] m'):
    say(f"  {k:22s} {gh[k]:3d} of {nA}")

# 12. rule 19 check on the live path: the broad phase (world_model.candidates -> detection_index.query) expands the
# detection box by 2*margin = 0.6 m per side and tests overlap against the STORED MEASURED obj.bbox; locality_ok then
# reads the FUSED box. A candidate whose fused box is within 0.3 m of the detection but whose measured box is more
# than 0.6 m away never reaches locality_ok. How far does the fused box extend past the measured one on this bundle?
# (Final fused boxes, end of run: an upper bound on any moment's extension.)
final = json.load(open(R + 'bundle/persistent_perception.json'))
ext = []
for f in final:
    m, fb = box(f['bbox']), box(f['fused_bbox'])
    ext.append(max(max(m[f'{ax}_min'] - fb[f'{ax}_min'], fb[f'{ax}_max'] - m[f'{ax}_max']) for ax in 'xyz'))
ext = np.array(ext)
say(f"\n=== fused box extension past the measured box, {len(final)} final objects (end-of-run boxes) ===")
say(f"  max per-axis extension: median {np.median(ext):.2f} m, p90 {np.percentile(ext, 90):.2f} m, max {ext.max():.2f} m; "
    f"> 0.3 m: {int((ext > 0.3).sum())} of {len(final)}; > 0.6 m: {int((ext > 0.6).sum())} of {len(final)}")
say("  (a detection within 0.3 m of a fused box that extends > 0.3 m past the measured box CAN be further than 0.6 m from the "
    "measured box, and then the broad phase drops it before locality_ok runs; the replay cannot count how often, it has no fused boxes)")
_out.close()
