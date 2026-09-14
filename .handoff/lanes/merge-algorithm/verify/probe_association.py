"""Probes against the REAL association module (plain python3, numpy only)."""
import sys

import numpy as np

PM = '/DATA/GRAPH-API/.claude/worktrees/found-merge-algorithm/lost3dsg/src/perception_module'
sys.path.insert(0, PM)
import association as A  # noqa: E402


def B(x0, x1, y0, y1, z0, z1):
    return (float(x0), float(x1), float(y0), float(y1), float(z0), float(z1))


print("=== 1. box_gap / geometry_compatible ===")
a = B(0, 1, 0, 1, 0, 1)
cases = {
    'identical': a,
    'touching_face_x': B(1, 2, 0, 1, 0, 1),
    'nested_inside': B(0.2, 0.8, 0.2, 0.8, 0.2, 0.8),
    'nested_outside(a inside b)': B(-1, 2, -1, 2, -1, 2),
    'diag_offset_0.2x_0.2y': B(1.2, 2.2, 1.2, 2.2, 0, 1),
    'diag_offset_0.2x_0.5y': B(1.2, 2.2, 1.5, 2.5, 0, 1),
    'corner_touch_xyz': B(1, 2, 1, 2, 1, 2),
    'degenerate_point_inside': B(0.5, 0.5, 0.5, 0.5, 0.5, 0.5),
    'degenerate_point_0.25_off_x': B(1.25, 1.25, 0.5, 0.5, 0.5, 0.5),
    'degenerate_plane_on_face': B(1, 1, 0, 1, 0, 1),
    'degenerate_plane_0.3_off': B(1.3, 1.3, 0, 1, 0, 1),
    'far_5m': B(5, 6, 0, 1, 0, 1),
    'inverted_min>max_touching': B(2, 1, 0, 1, 0, 1),
}
for n, b in cases.items():
    g = A.box_gap(a, b)
    print(f"  {n:32s} gap={g:.4f} compat@0.3={A.geometry_compatible(a, b, 0.3)} "
          f"compat@0.0={A.geometry_compatible(a, b, 0.0)} symmetric={A.box_gap(b, a) == g}")
print("  None sides:", A.geometry_compatible(None, a, 0.3), A.geometry_compatible(a, None, 0.3),
      A.geometry_compatible(None, None, 0.3))
try:
    A.box_gap(None, a)
except Exception as e:
    print("  box_gap(None, a) raises", type(e).__name__)
nanb = B(float('nan'), 1, 0, 1, 0, 1)
print("  NaN x_min box vs a: gap", A.box_gap(a, nanb), "compat@0.3", A.geometry_compatible(a, nanb, 0.3),
      "| reversed gap", A.box_gap(nanb, a), A.geometry_compatible(nanb, a, 0.3))
nan_far = B(float('nan'), 1, 5, 6, 0, 1)
print("  NaN x_min, y 5 m away: gap", A.box_gap(a, nan_far), "compat", A.geometry_compatible(a, nan_far, 0.3))
print("  gap_m as string '0.3':", A.geometry_compatible(a, cases['diag_offset_0.2x_0.2y'], '0.3'))

print("\n=== 2. locality_bounds ===")


class O:
    pass


BOX = {'x_min': 0, 'x_max': 1, 'y_min': 0, 'y_max': 1, 'z_min': 0, 'z_max': 1, 'has_orientation': False}
FUSED = {'x_min': -0.1, 'x_max': 1.1, 'y_min': 0, 'y_max': 1, 'z_min': 0, 'z_max': 1,
         'source': 'multi_observation_voxel_agreement', 'voxel_size_m': 0.03, 'view_count': 2,
         'required_views': 1, 'voxel_count': 744, 'agreement_fallback': False}
for name, fused, bbox in [
        ('fused with extra keys (GA-493 shape)', FUSED, BOX),
        ('fused = {}', {}, BOX),
        ('fused = None', None, BOX),
        ('fused malformed {x_min only}', {'x_min': 0.0}, BOX),
        ('fused with string values', {k: str(v) for k, v in FUSED.items()}, BOX),
        ('fused with None value', {**FUSED, 'z_max': None}, BOX),
        ('no fused, bbox None', None, None),
        ('fused ok, bbox None', FUSED, None),
        ('fused = 0 (falsy non-dict)', 0, BOX),
        ('fused = list of 6', [0, 1, 0, 1, 0, 1], BOX)]:
    o = O()
    o.bbox = bbox
    o.fused_bbox = fused
    print(f"  {name:38s} -> {A.locality_bounds(o)}")
o = O()
o.bbox = BOX
print("  no fused_bbox attribute at all       ->", A.locality_bounds(o))
print("  obj None                             ->", A.locality_bounds(None))

print("\n=== 3. channel_attributes (reference 0.85, cap 2.0) ===")
for s in (0.85, 1.0, 0.5, 0.0, 0.925, 0.9, 0.7, 0.44, 1.2, -0.3, 0.85000001, 0.84999999):
    r = A.channel_attributes(s, 3, 0.85, 2.0, same_kind=True)
    print(f"  score {s:<11} -> {r}")
print("  evidence_count 0        ->", A.channel_attributes(1.0, 0, 0.85, 2.0),
      ".measured", A.channel_attributes(1.0, 0, 0.85, 2.0).measured)
print("  score None              ->", A.channel_attributes(None, 3, 0.85, 2.0),
      ".measured", A.channel_attributes(None, 3, 0.85, 2.0).measured)
print("  same_kind False detail  ->", A.channel_attributes(0.9, 3, 0.85, 2.0, same_kind=False)[1])
print("  same_kind None detail   ->", A.channel_attributes(0.9, 3, 0.85, 2.0)[1])
print("  numpy inputs            ->", A.channel_attributes(np.float32(0.9), np.int64(2), 0.85, 2.0,
                                                          same_kind=np.bool_(True)))
for label, args in [('evidence_count None', (0.9, None)), ('evidence_count "3"', (0.9, "3")),
                    ('score "0.9"', ("0.9", 3))]:
    try:
        print(f"  {label:22s} ->", A.channel_attributes(args[0], args[1], 0.85, 2.0))
    except Exception as e:
        print(f"  {label:22s} -> RAISES {type(e).__name__}: {e}")
cap5 = A.channel_attributes(1.0, 3, 0.85, 5.0)[0]
print("  cap 5.0 (above log(20)=3.0):", cap5, "-> attributes alone clear commit_threshold?",
      cap5 >= A.commit_threshold(20))
print("  reference 0.5 (division guard):", A.channel_attributes(0.4, 3, 0.5, 2.0))
print("  reference 1.0 (division guard):", A.channel_attributes(1.0, 3, 1.0, 2.0),
      A.channel_attributes(0.9, 3, 1.0, 2.0))

print("\n=== 4. score_pair with attribute_score_fn variants ===")
v1 = A.AssocObject("v1", bbox=A._box(0, 0, 0.5, 1.0, 1.0, 1.0), room_id="r")
v2 = A.AssocObject("v2", bbox=A._box(0.2, 0, 0.5, 1.0, 1.0, 1.0), room_id="r")


def ctx(fn, **kw):
    return A.AssocContext(map_volume_m3=300.0, n_rooms=1, cost_ratio=20.0, use_ontology=False,
                          attribute_score_fn=fn, **kw)


def raiser(x, y):
    raise RuntimeError("scorer failed")


variants = [
    ('2-tuple', lambda x, y: (0.95, 3)),
    ('3-tuple', lambda x, y: (0.95, 3, True)),
    ('3-tuple same_kind False', lambda x, y: (0.95, 3, False)),
    ('None', lambda x, y: None),
    ('1-tuple', lambda x, y: (0.95,)),
    ('scalar', lambda x, y: 0.95),
    ('4-tuple', lambda x, y: (0.95, 3, True, 'extra')),
    ('list', lambda x, y: [0.95, 3, True]),
    ('numpy types', lambda x, y: (np.float32(0.95), np.int64(3), np.bool_(True))),
    ('empty tuple', lambda x, y: ()),
    ('raises', raiser),
]
for name, fn in variants:
    try:
        s = A.score_pair(v1, v2, ctx(fn))
        print(f"  {name:24s} channel={s.channels.get('attributes')} abstention={s.abstentions.get('attributes')} "
              f"total={s.total:.3f} cu={s.containment_unchecked}")
    except Exception as e:
        print(f"  {name:24s} RAISES {type(e).__name__}: {e}")
s = A.score_pair(v1, v2, A.AssocContext(map_volume_m3=300.0, n_rooms=1, cost_ratio=20.0))
print("  default ctx (no fn, ontology on): attributes in channels?", 'attributes' in s.channels,
      "| ontology in channels?", 'ontology' in s.channels, "| abstentions", s.abstentions)
s = A.score_pair(v1, v2, ctx(None))
print("  use_ontology False, fn None: abstentions", s.abstentions)

print("\n=== 5. PairScore.containment_unchecked truth table (overlap +3.0 carries) ===")


def ps(cov, attr, same_kind):
    s = A.PairScore()
    if cov == 'channel':
        s.add('covisibility', (0.5, {}))
    elif cov == 'measured-abstain':
        s.add('covisibility', A.Abstain('dup shape', measured=True))
    elif cov == 'uncollected-abstain':
        s.add('covisibility', A.Abstain('no obs', measured=False))
    # 'absent': never added
    s.add('overlap', (3.0, {}))
    if attr == 'pos':
        s.add('attributes', (1.5, {'same_kind': same_kind}))
    elif attr == 'neg':
        s.add('attributes', (-8.0, {'same_kind': same_kind}))
    elif attr == 'zero':
        s.add('attributes', (0.0, {'same_kind': same_kind}))
    elif attr == 'abstain':
        s.add('attributes', A.Abstain('x'))
    return s


print(f"  {'covisibility':22s} {'attributes':10s} {'same_kind':10s} -> containment_unchecked")
for cov in ('channel', 'measured-abstain', 'uncollected-abstain', 'absent'):
    for attr in ('pos', 'neg', 'zero', 'abstain', 'absent'):
        for sk in (True, False, None):
            if attr in ('abstain', 'absent') and sk is not True:
                continue
            print(f"  {cov:22s} {attr:10s} {str(sk):10s} -> {ps(cov, attr, sk).containment_unchecked}")
s = A.PairScore()
s.add('covisibility', A.Abstain('no obs'))
s.add('overlap', (3.0, {}))
s.add('attributes', (1.5, {'same_kind': np.bool_(True)}))
print("  uncollected / pos / same_kind=np.bool_(True) ->", s.containment_unchecked,
      "(np.bool_(True) is True?", np.bool_(True) is True, ")")
s = A.PairScore()
s.add('covisibility', A.Abstain('no obs'))
s.add('overlap', (1.0, {}))
s.add('separation', (4.0, {}))
print("  uncollected / no attrs / overlap 1.0 of total 5.0 ->", s.containment_unchecked)

print("\n=== 6. Hypothesis: streak across a SKIPPED sweep, and stale channel state ===")
thr = A.commit_threshold(20.0)
same = ctx(lambda x, y: (1.0, 3, True))
sc = A.score_pair(v1, v2, same)
print(f"  pair score total {sc.total:.3f} vs threshold {thr:.3f}, containment_unchecked={sc.containment_unchecked}")
h = A.Hypothesis(('v1', 'v2'))
h.update(sc, frame_id=1)
print("  sweep 1:", h.decide(thr, min_consecutive=2), "streak", h._streak)
print("  sweep 2: SKIPPED (pair offered but refused by a gate before update)")
h.update(sc, frame_id=3)
print("  sweep 3:", h.decide(thr, min_consecutive=2), "streak", h._streak,
      "frames", [f['frame'] for f in h.history])
h2 = A.Hypothesis(('v1', 'v2'))
h2.update(A.score_pair(v1, v2, same), frame_id=1)
t1 = h2.total
h2.update(A.score_pair(v1, v2, ctx(lambda x, y: None)), frame_id=2)
print(f"  attributes +{h2.state.get('attributes')} in sweep 1, ABSTAINS in sweep 2 -> state keeps it? "
      f"{'attributes' in h2.state}; total sweep1 {t1:.3f} sweep2 {h2.total:.3f}; "
      f"history[1] abstentions {h2.history[1]['abstentions'].get('attributes')!r}")
cross = ctx(lambda x, y: (1.0, 3, False))
h3 = A.Hypothesis(('v1', 'v2'))
for f in (1, 2, 3):
    h3.update(A.score_pair(v1, v2, cross), frame_id=f)
    print("  cross-kind sweep", f, h3.decide(thr, min_consecutive=2)[0], "streak", h3._streak)
h4 = A.Hypothesis(('v1', 'v2'))
h4.update(A.score_pair(v1, v2, same), frame_id=1)
d1 = h4.decide(thr, min_consecutive=2)
h4.update(A.score_pair(v1, v2, cross), frame_id=2)
d2 = h4.decide(thr, min_consecutive=2)
h4.update(A.score_pair(v1, v2, same), frame_id=3)
d3 = h4.decide(thr, min_consecutive=2)
print("  same/cross/same ->", d1[0], d2[0], d3[0], "streak", h4._streak)
print("\nprobe_association done")
