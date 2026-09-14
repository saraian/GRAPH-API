"""Unit probes against association.py as it is on the branch (plain python3, no model).

Q1  same-kind lift of PairScore.containment_unchecked: two DISTINCT same-label instances nested
    (containment 1.0), never co-visible, attributes agree.
Q2  gap-offered pairs: what score_pair says about a non-intersecting pair generate_candidates now
    offers, with and without a covariance.
Q2b room + attributes with NO positive geometry: can a non-intersecting pair commit?
"""
import math
import sys

PM = '/DATA/GRAPH-API/.claude/worktrees/found-merge-algorithm/lost3dsg/src/perception_module'
sys.path.insert(0, PM)
import association as A  # noqa: E402


def box(x0, x1, y0, y1, z0, z1):
    return dict(x_min=x0, x_max=x1, y_min=y0, y_max=y1, z_min=z0, z_max=z1)


THR = A.commit_threshold(20.0)


def run(a, b, ctx, sweeps=2):
    h = A.Hypothesis(('a', 'b'))
    for f in range(1, sweeps + 1):
        ps = A.score_pair(a, b, ctx)
        h.update(ps, frame_id=f)
        d = h.decide(THR, min_consecutive=2)
    ch = {k: round(v.get('log_odds', float('nan')), 3) for k, v in ps.channels.items()}
    return d[0], round(ps.total, 3), ch, ps.containment_unchecked, sorted(ps.abstentions)


def ctx(score, same_kind, n_rooms=6, attrs=True, vol=300.0):
    kw = dict(map_volume_m3=vol, n_rooms=n_rooms, cost_ratio=20.0, use_ontology=False,
              locality_gap_m=0.3)
    if attrs:
        kw['attribute_score_fn'] = lambda x, y: (score, 3, same_kind)
    return A.AssocContext(**kw)


print(f"commit threshold log(20) = {THR:.3f}\n")

# ---- Q1: nested distinct same-kind instances -------------------------------------------------
print("Q1  nested same-label pair (containment 1.0, e.g. pillow inside a pillow box, cabinet run "
      "containing a drawer labelled 'cabinet'); no observations -> co-visibility UNCOLLECTED")
big = A.AssocObject('cab_run', bbox=box(0, 2.0, 0, 0.6, 0, 0.9), room_id='r1')
drawer = A.AssocObject('cab_drawer', bbox=box(0.5, 1.0, 0.05, 0.55, 0.3, 0.6), room_id='r1')
for name, c in [('pre-change engine (no attribute channel)', ctx(None, None, attrs=False)),
                ('same_kind=True, score 0.95', ctx(0.95, True)),
                ('same_kind=True, score 0.86 (just above the 0.85 reference)', ctx(0.86, True)),
                ('same_kind=False, score 0.95 (synonym / cross-kind)', ctx(0.95, False)),
                ('same_kind=True, score 0.95, ONE room (room channel abstains)', ctx(0.95, True, n_rooms=1))]:
    print(f"  {name:62s} -> {run(big, drawer, c)}")

# the ruling's own stated protected case: cross-kind pillow-in-bed at containment 0.85
bed = A.AssocObject('bed', bbox=box(0, 2.0, 0, 1.6, 0, 0.6), room_id='r1')
pillow = A.AssocObject('pillow', bbox=box(0.1, 0.7, 0.1, 0.5, 0.5, 0.7), room_id='r1')
print(f"  {'pillow-in-bed, same_kind=False, score 0.95':62s} -> {run(bed, pillow, ctx(0.95, False))}")
print(f"  {'pillow-in-bed, same_kind=False, score 0.44 (measured lamp/table)':62s} -> {run(bed, pillow, ctx(0.44, False))}")

# ---- Q2: gap-offered non-intersecting pair ---------------------------------------------------
print("\nQ2  two 1 m cubes 0.25 m apart (gap-compatible at 0.3, no intersection)")
a = A.AssocObject('a', bbox=box(0, 1, 0, 1, 0, 1), room_id='r1')
b = A.AssocObject('b', bbox=box(1.25, 2.25, 0, 1, 0, 1), room_id='r1')
for nr in (1, 3, 4, 6):
    c = ctx(1.0, True, n_rooms=nr)
    off, exc = A.generate_candidates([a, b], c)
    meta = (off or exc)[0][2]
    print(f"  no observations, n_rooms={nr}: offered_by={meta.get('offered_by')} d={meta['distance_m']:.3f} "
          f"reach={meta['reach_m']:.3f} -> {run(a, b, c)}")
c = ctx(0.86, True, n_rooms=6)
print(f"  no observations, n_rooms=6, score 0.86 -> {run(a, b, c)}")
c = ctx(1.0, True, n_rooms=6, attrs=False)
print(f"  no observations, n_rooms=6, PRE-CHANGE engine -> {run(a, b, c)}")

# same pair, each side with ONE sighting at 3 m range (what _record_sighting produces; the
# service passes depth_sparsity=pose_sigma_m=0, so the model covariance is 1e-9)
cam = (0.5, -3.0, 0.5)
a1 = A.AssocObject('a', bbox=box(0, 1, 0, 1, 0, 1), room_id='r1',
                   observations=[A.Observation('f1', cam, (0.5, 0.5, 0.5), bbox_2d=[0, 0, 10, 10])])
b1 = A.AssocObject('b', bbox=box(1.25, 2.25, 0, 1, 0, 1), room_id='r1',
                   observations=[A.Observation('f2', cam, (1.75, 0.5, 0.5), bbox_2d=[0, 0, 10, 10])])
c = ctx(1.0, True, n_rooms=6)
off, exc = A.generate_candidates([a1, b1], c)
meta = (off or exc)[0][2]
print(f"  one sighting each (cov {a1.covariance[0,0]:.1e}): offered_by={meta.get('offered_by')} "
      f"d={meta['distance_m']:.3f} reach={meta['reach_m']:.3f} -> {run(a1, b1, c)}")
# what the shell alone would have done with these covariances (pre-change candidate generation)
c0 = A.AssocContext(map_volume_m3=300.0, n_rooms=6, cost_ratio=20.0, use_ontology=False)
off0, exc0 = A.generate_candidates([a1, b1], c0)
print(f"  same pair, locality_gap_m=None (pre-change): offered={len(off0)} excluded={len(exc0)}")

# ---- Q2b: what total is reachable with overlap <= 0 ------------------------------------------
print("\nQ2b room + attributes ceiling with no positive geometry: attrs cap 2.0 + log(n_rooms) - |giou|")
for nr in (2, 3, 4, 5, 6, 8):
    print(f"  n_rooms={nr}: 2.0 + log({nr}) = {2.0 + math.log(nr):.3f}  {'>= thr' if 2.0 + math.log(nr) >= THR else '< thr'}")

# ---- Q1b: same-kind pair intersecting at a small containment --------------------------------
print("\nQ1b two 1 m cubes of one kind intersecting by 0.2 m on x (containment 0.2), n_rooms=1")
b2 = A.AssocObject('b', bbox=box(0.8, 1.8, 0, 1, 0, 1), room_id='r1')
for sc in (None, 0.86, 0.95, 1.0):
    c = ctx(sc, True, n_rooms=1, attrs=sc is not None)
    print(f"  score {sc}: -> {run(a, b2, c)}")
