"""For the same-label pairs whose FUSED boxes pass the 0.3 m gap test while their MEASURED boxes
fail it (box_choice_gap.py, section b), ask whether the evidence engine's candidate generator
would offer them at all. generate_candidates offers a pair when `d <= reach` (covariance shell,
search_radius on the MEASURED bbox + observations) OR the gap test on the MEASURED bbox passes.
The service's geometry gate then reads the FUSED box -- but only for pairs that were offered.
"""
import itertools
import json
import sys

sys.path.insert(0, "/DATA/GRAPH-API/.claude/worktrees/found-merge-algorithm/lost3dsg/src/perception_module")
import numpy as np  # noqa: E402

import association as assoc  # noqa: E402

P = "/DATA/GRAPH-API/.handoff/lanes/perception/ga493_replay_20260914/bundle/persistent_perception.json"
GAP = 0.3

d = json.load(open(P))
objs = d if isinstance(d, list) else (d.get("objects") or d.get("persistent_perceptions") or d)
if isinstance(objs, dict):
    objs = list(objs.values())

print("keys on one object:", sorted(objs[0].keys()))
print("objects with an 'observations' list:", sum(1 for o in objs if isinstance(o.get("observations"), list)),
      "of", len(objs))


def base(o):
    return str(o.get("label", "")).split("#")[0].strip().lower()


def build(o):
    obs = []
    for rec in o.get("observations") or []:
        if not isinstance(rec, dict):
            continue
        try:
            obs.append(assoc.Observation(rec.get("frame_id"), rec.get("camera_position"),
                                         rec.get("centroid") or rec.get("centroid_world"),
                                         bbox_2d=rec.get("bbox_2d")))
        except Exception as exc:  # measurement script only; report, never hide
            print("   observation not rebuilt:", exc)
    return assoc.AssocObject(object_id=o.get("object_id") or o.get("label"), label=o.get("label"),
                             bbox=o.get("bbox"), centroid=o.get("centroid"), observations=obs)


ctx = assoc.AssocContext(map_volume_m3=300.0, n_rooms=6, cost_ratio=20.0, locality_gap_m=GAP)
built = [(o, build(o)) for o in objs if assoc._as_bounds(o.get("bbox"))]
rows = []
for (oa, a), (ob, b) in itertools.combinations(built, 2):
    fa = assoc.locality_bounds(type("S", (), {"fused_bbox": oa.get("fused_bbox"), "bbox": oa.get("bbox")})())
    fb = assoc.locality_bounds(type("S", (), {"fused_bbox": ob.get("fused_bbox"), "bbox": ob.get("bbox")})())
    gap_fused = assoc.geometry_compatible(fa, fb, GAP)
    gap_meas = assoc.geometry_compatible(assoc._as_bounds(a.bbox), assoc._as_bounds(b.bbox), GAP)
    if not (gap_fused and not gap_meas):
        continue
    r_a, basis_a = assoc.search_radius(a, ctx)
    r_b, basis_b = assoc.search_radius(b, ctx)
    dist = float(np.linalg.norm(np.asarray(a.centroid) - np.asarray(b.centroid)))
    offered = dist <= r_a + r_b
    rows.append((base(oa) == base(ob), oa.get("label"), ob.get("label"), round(dist, 3),
                 round(r_a + r_b, 3), offered, basis_a))

same = [r for r in rows if r[0]]
print(f"fused-only pairs: {len(rows)}; offered by the covariance shell anyway: "
      f"{sum(1 for r in rows if r[5])} of {len(rows)}; NOT offered (never reach the fused-box gate): "
      f"{sum(1 for r in rows if not r[5])} of {len(rows)}")
print(f"same base label among them: {len(same)}; offered by shell: {sum(1 for r in same if r[5])} of {len(same)}")
for r in same:
    print("   ", r[1:])
