"""Independent re-check of the 'measured box at the broad phase, fused box at the gate' claim.

Runs the REAL association.generate_candidates (gap arm + covariance shell) on AssocObjects built
the way ObjectServices._assoc_build builds them (bbox=o.bbox, measured), then applies the REAL
service gate (assoc.locality_bounds -> fused box, assoc.geometry_compatible, 0.3 m) to every pair,
and counts pairs the gate would ACCEPT that the generator never OFFERED. Then the counterfactual:
the same run with AssocObject.bbox = locality_bounds(o) (the proposed fix).

Shell basis: the bundle carries no observations, so search_radius answers 'extent only
(no covariance)' for every object. That is a LOWER bound on what the live shell offers.
"""
import itertools
import json
import sys

PM = "/DATA/GRAPH-API/.claude/worktrees/found-merge-algorithm/lost3dsg/src/perception_module"
sys.path.insert(0, PM)
import association as assoc  # noqa: E402

P = "/DATA/GRAPH-API/.handoff/lanes/perception/ga493_replay_20260914/bundle/persistent_perception.json"
GAP = 0.3
objs = json.load(open(P))


class Src:  # stands in for the world-model Object the service passes to locality_bounds
    def __init__(self, o):
        self.label = o["label"]
        self.object_id = o["object_id"]
        self.bbox = o["bbox"]
        self.fused_bbox = o.get("fused_bbox")


def base(lab):
    return str(lab).split("#")[0].strip().lower()


def run(box_choice):
    srcs = [Src(o) for o in objs if assoc._as_bounds(o.get("bbox"))]
    ctx = assoc.AssocContext(map_volume_m3=300.0, n_rooms=6, cost_ratio=20.0, locality_gap_m=GAP)
    built = []
    for s in srcs:
        b = s.bbox if box_choice == "measured" else dict(zip(assoc._KEYS, assoc.locality_bounds(s)))
        built.append(assoc.AssocObject(object_id=s.object_id, label=s.label, bbox=b, source=s))
    offered, excluded = assoc.generate_candidates(built, ctx, k=None)
    offered_keys = {frozenset((id(a.source), id(b.source))) for a, b, _m in offered}
    basis = {m["basis_a"] for _a, _b, m in offered} | {m["basis_a"] for _a, _b, m in excluded}
    gate_ok = []
    for sa, sb in itertools.combinations(srcs, 2):
        ba, bb = assoc.locality_bounds(sa), assoc.locality_bounds(sb)
        if assoc.geometry_compatible(ba, bb, GAP) is True:
            gate_ok.append((sa, sb))
    hidden = [(sa, sb) for sa, sb in gate_ok if frozenset((id(sa), id(sb))) not in offered_keys]
    hidden_same = [(sa.label, sb.label,
                    round(assoc.box_gap(assoc._as_bounds(sa.bbox), assoc._as_bounds(sb.bbox)), 3),
                    round(assoc.box_gap(assoc.locality_bounds(sa), assoc.locality_bounds(sb)), 3))
                   for sa, sb in hidden if base(sa.label) == base(sb.label)]
    n_pairs = len(srcs) * (len(srcs) - 1) // 2
    print(f"[AssocObject.bbox = {box_choice}] objects {len(srcs)}; all pairs {n_pairs}; "
          f"offered by generate_candidates {len(offered)}; shell basis {sorted(basis)}")
    print(f"   pairs the service gate (fused box, gap<=0.3) ACCEPTS: {len(gate_ok)} of {n_pairs}; "
          f"of those NOT offered by the generator: {len(hidden)} "
          f"(same base label: {len(hidden_same)})")
    for r in hidden_same:
        print("      (a, b, measured gap m, fused gap m) =", r)


run("measured")   # what _assoc_build does today
run("fused")      # the proposed fix
