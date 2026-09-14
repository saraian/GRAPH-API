"""rule15-masking claim: the same-kind lift of PairScore.containment_unchecked.

Nested same-base-label pair (containment 1.0), attributes agree at 0.86 (above the 0.85
reference, BELOW the legacy merge bar merge_min_similarity 0.925). Plain python3, no model:
the attribute score is injected through AssocContext.attribute_score_fn exactly as
object_services._attribute_channel_input returns it: (score, optional_count, same_kind).
"""
import sys

PM = '/DATA/GRAPH-API/.claude/worktrees/found-merge-algorithm/lost3dsg/src/perception_module'
sys.path.insert(0, PM)
import association as A  # noqa: E402

THR = A.commit_threshold(20.0)


def box(x0, x1, y0, y1, z0, z1):
    return dict(x_min=x0, x_max=x1, y_min=y0, y_max=y1, z_min=z0, z_max=z1)


def run(a, b, ctx, sweeps=2):
    """One update AND one decision per sweep, as _cb_merge_objects does (min_consecutive=2)."""
    h = A.Hypothesis(('a', 'b'))
    for f in range(1, sweeps + 1):
        ps = A.score_pair(a, b, ctx)
        h.update(ps, frame_id=f)
        d = h.decide(THR, min_consecutive=2)
    ch = {k: round(v['log_odds'], 2) for k, v in ps.channels.items() if 'log_odds' in v}
    cov = ps.channels.get('covisibility', {}).get('veto') and 'VETO' or ps.abstentions.get('covisibility', 'measured')[:38]
    return d[0], round(ps.total, 2), ch, 'guard=' + str(ps.containment_unchecked), 'covis=' + str(cov)


def ctx(score=None, same_kind=None):
    kw = dict(map_volume_m3=300.0, n_rooms=6, cost_ratio=20.0, use_ontology=False,
              locality_gap_m=0.3, overlap_2d_fn=A.shared_frame_overlap_2d)
    if score is not None:
        kw['attribute_score_fn'] = lambda x, y: (score, 3, same_kind)
    return A.AssocContext(**kw)


def obs(frame, bbox_2d=None):
    return A.Observation(frame_id=frame, camera_position=(3.0, 3.0, 1.5), centroid=(0.3, 0.3, 0.6),
                         stamp=float(frame), bbox_2d=bbox_2d)


BIG = box(0, 0.6, 0, 0.6, 0.5, 0.7)          # pillow
SMALL = box(0.1, 0.5, 0.1, 0.5, 0.55, 0.65)  # second pillow, box fully inside: containment 1.0


def pair(obs_a=None, obs_b=None):
    return (A.AssocObject('pillow', bbox=BIG, room_id='r1', observations=obs_a),
            A.AssocObject('pillow#1', bbox=SMALL, room_id='r1', observations=obs_b))


print(f"commit threshold log(20) = {THR:.3f}; legacy merge bar merge_min_similarity = 0.925\n")
cases = [
    ("A pre-ruling evidence engine (no attribute channel), no sightings", pair(), ctx()),
    ("B branch: same_kind=True score 0.86, no sightings", pair(), ctx(0.86, True)),
    ("C branch: same_kind=True 0.86, SAME frame, no 2D box (GA-493 capture shape)",
     pair([obs(1)], [obs(1)]), ctx(0.86, True)),
    ("D branch: same_kind=True 0.86, SAME frame, 2D boxes DISJOINT",
     pair([obs(1, [0, 0, 10, 10])], [obs(1, [100, 100, 110, 110])]), ctx(0.86, True)),
    ("E branch: same_kind=True 0.86, SAME frame, 2D boxes identical",
     pair([obs(1, [0, 0, 10, 10])], [obs(1, [0, 0, 10, 10])]), ctx(0.86, True)),
    ("F branch: same_kind=True 0.86, sightings in DIFFERENT frames",
     pair([obs(1)], [obs(2)]), ctx(0.86, True)),
    ("G branch: same_kind=False 0.86 (synonym), no sightings", pair(), ctx(0.86, False)),
    ("H branch: same_kind=True, score 0.85 exactly (channel 0.0), no sightings", pair(), ctx(0.85, True)),
]
for name, (a, b), c in cases:
    print(f"  {name:74s} -> {run(a, b, c)}")
