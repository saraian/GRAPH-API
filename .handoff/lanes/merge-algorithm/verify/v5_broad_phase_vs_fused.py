"""Rule 19: locality_ok reads the FUSED box, but the caller's broad phase (wm.candidates) indexes the
MEASURED box and expands the query by 2 * association_margin_m = 0.6 m. A detection that is
geometry-compatible with the fused box but > 0.6 m from the measured box never reaches locality_ok.
Plain python3 with rosstub; uses world_model.WorldModel and association.* exactly as
object_manager_6.locality_ok / _association_candidates do (verified by reading both).
"""
import json
import sys

PM = '/DATA/GRAPH-API/.claude/worktrees/found-merge-algorithm/lost3dsg/src/perception_module'
sys.path.insert(0, PM)
import rosstub  # noqa: E402

rosstub.install()
import association as A  # noqa: E402
import object_info  # noqa: E402
from world_model import wm  # noqa: E402

ASSOCIATION_MARGIN_M = 0.3


def locality_ok(bbox, obj, gap_m):   # object_manager_6.locality_ok, verbatim logic
    return bool(A.geometry_compatible(A._as_bounds(bbox), A.locality_bounds(obj), gap_m))


def box(x0, x1, y0, y1, z0, z1):
    return dict(x_min=x0, x_max=x1, y_min=y0, y_max=y1, z_min=z0, z_max=z1)


# synthetic: a sofa whose measured (last view) box is 1 m long and whose fused box spans 3 m
sofa = object_info.Object('sofa', (0.5, 0.5, 0.3), box(0, 1, 0, 1, 0, 0.6))
sofa.object_id = 'sofa_1'
sofa.fused_bbox = box(0, 3, 0, 1, 0, 0.6)
wm.persistent_perceptions.clear()
wm.persistent_perceptions.append(sofa)
det = box(3.1, 3.6, 0, 1, 0, 0.6)     # 0.1 m past the fused box's far end, 2.1 m past the measured box
cands = wm.candidates(det, ASSOCIATION_MARGIN_M)
print("synthetic sofa: locality_ok(det, sofa) =", locality_ok(det, sofa, ASSOCIATION_MARGIN_M),
      "| wm.candidates(det, 0.3) =", [getattr(o, 'object_id', None) for o in cands],
      "-> the loop never calls locality_ok for it" if not cands else "")

# the GA-493 objects whose fused box extends > 0.6 m beyond the measured box: a detection touching
# the fused box's far end is compatible on the fused box and pruned by the broad phase
R = '/DATA/GRAPH-API/.handoff/lanes/perception/ga493_replay_20260914/'
final = json.load(open(R + 'bundle/persistent_perception.json'))
wm.persistent_perceptions.clear()
objs = []
for o in final:
    ob = object_info.Object(o['label'], None, dict(o['bbox']))
    ob.object_id = o['object_id']
    ob.fused_bbox = dict(o['fused_bbox'])
    objs.append(ob)
    wm.persistent_perceptions.append(ob)
n_unreachable = 0
rows = []
for ob in objs:
    m, f = A._as_bounds(ob.bbox), A._as_bounds(ob.fused_bbox)
    # probe: a 0.3 m cube just inside the fused box at the side where it extends furthest
    sides = [(m[0] - f[0], 'x_min'), (f[1] - m[1], 'x_max'), (m[2] - f[2], 'y_min'), (f[3] - m[3], 'y_max')]
    e, side = max(sides)
    if e <= 0.6:
        continue
    if side == 'x_min':
        probe = box(f[0], f[0] + 0.3, f[2], f[3], f[4], f[5])
    elif side == 'x_max':
        probe = box(f[1] - 0.3, f[1], f[2], f[3], f[4], f[5])
    elif side == 'y_min':
        probe = box(f[0], f[1], f[2], f[2] + 0.3, f[4], f[5])
    else:
        probe = box(f[0], f[1], f[3] - 0.3, f[3], f[4], f[5])
    ok = locality_ok(probe, ob, ASSOCIATION_MARGIN_M)
    reached = any(c is ob for c in wm.candidates(probe, ASSOCIATION_MARGIN_M))
    n_unreachable += ok and not reached
    rows.append((ob.label, side, round(e, 2), ok, reached))
print(f"GA-493: objects whose fused box extends > 0.6 m beyond the measured box on one side: {len(rows)} of {len(final)}; "
      f"a detection inside the fused box at that end passes locality_ok but is NOT a broad-phase candidate for {n_unreachable} of them")
for r in rows:
    print("  (label, side, extension m, locality_ok, reached by wm.candidates) =", r)
wm.persistent_perceptions.clear()
