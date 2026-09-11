#!/usr/bin/env python3
"""Record WHICH FLOOR a published rtabmap map covers, measured from its own node poses.

Called by run.sh's publish step. Usage: stamp_floor.py <db> [scene_floors_json]

WHY THIS EXISTS. hm3d_00861 has four floors and only ONE has ever been mapped, by any run. The
map did not say which. A robot loading it could not tell whether it was the right storey, and
every coverage claim about the scene was silently a claim about one floor.

WHY IT IS A MEASUREMENT AND NOT A DECLARATION. The height is read from the Node pose transforms
already in the database, not from the config that was requested or the floor the tour intended.
A run that was asked for floor 0.43 and spawned on 1.35 produces a map of 1.35, and this says so.

The spread is reported beside the height because it is the evidence that the map IS single-floor:
hm3d_00861's canonical map spans 3 cm across 1096 nodes. A map that had stacked storeys would
show metres here, and the field would be the thing that revealed it.
"""
import json
import pathlib
import sqlite3
import struct
import sys


def node_poses(db):
    """(tx, ty, tz) of every node. rtabmap stores a 3x4 row-major float32 transform."""
    c = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    out = []
    for (b,) in c.execute("SELECT pose FROM Node WHERE pose IS NOT NULL"):
        if b and len(b) >= 48:
            m = struct.unpack("12f", b[:48])
            out.append((m[3], m[7], m[11]))
    c.close()
    return out


def node_heights(db):
    return [p[2] for p in node_poses(db)]


def main():
    db = pathlib.Path(sys.argv[1])
    floors = json.loads(sys.argv[2]) if len(sys.argv) > 2 else []
    poses = node_poses(db)
    zs = [p[2] for p in poses]
    if not zs:
        print(f"[map] {db.name}: NO node poses — floor NOT stamped. An unstamped map is honest; "
              "a guessed one is not.")
        return 1
    zs.sort()
    med = zs[len(zs) // 2]
    lo, hi, spread = zs[0], zs[-1], zs[-1] - zs[0]
    nearest = min(floors, key=lambda f: abs(f - med)) if floors else None

    # THE QUESTION IS "are any nodes on a DIFFERENT storey", not "how much vertical variation is
    # there". Answer it per node against the scene's own floor set: relief, ramps and pose-height
    # variation stay on one floor however large they are; a node whose nearest floor differs is
    # the only thing that means two storeys.
    if floors:
        assign = [min(floors, key=lambda f: abs(f - z)) for z in zs]
        per_floor = {f"{f:+.2f}": assign.count(f) for f in sorted(set(assign))}
        max_dev = max(abs(z - a) for z, a in zip(zs, assign))
        single = len(per_floor) == 1
        basis = "nearest-scene-floor per node"
    else:
        per_floor, max_dev = {}, spread
        single = bool(spread < 0.5)
        basis = "spread only — NO SCENE FLOOR SET SUPPLIED, so this is the weaker test"
    out = {
        "floor_height_m": round(med, 3),
        "floor_height_source": "median of Node pose tz in this database — MEASURED, not the "
                               "floor the run was asked for",
        "node_z_min": round(lo, 3), "node_z_max": round(hi, 3),
        "node_z_spread_m": round(spread, 3),
        "nodes": len(zs),
        "single_floor": single,
        "single_floor_basis": basis,
        "single_floor_note": "TRUE when every node's NEAREST SCENE FLOOR is the same floor — not "
                             "when the vertical spread is small. The first version used raw "
                             "spread < 0.5 m and was WRONG in both directions: floor 0.43's map "
                             "has 0.658 m of within-floor relief and was called multi-floor, "
                             "while a scene whose storeys are 0.4 m apart would pass while "
                             "genuinely straddling two. Found by the testing lane on the first "
                             "map the field was applied to. With no scene floors to compare "
                             "against, this falls back to spread and says so in "
                             "single_floor_basis.",
        "nodes_per_nearest_floor": per_floor,
        "max_distance_from_own_floor_m": round(max_dev, 3),
        "scene_floors": floors,
        "nearest_scene_floor": nearest,
        "floors_unmapped": [f for f in floors if f != nearest],
        "coverage_note": "this map covers ONE storey. Any coverage claim about the scene that "
                         "cites it is a claim about that storey alone.",
    }
    # COVERAGE. A map can be one storey, open cleanly, and still be a map of nowhere.
    #
    # Measured across every map this project has published:
    #     hm3d 1.35 (canonical)  1096 nodes, 269 distinct poses, 8.15 x 12.23 m
    #     hm3d -1.59 (rejected)   368 nodes, 160 distinct,       7.02 x  8.68 m
    #     mp3d_17DRP              304 nodes,  73 distinct,       2.91 x  4.73 m
    #     hm3d 0.43               471 nodes,  17 distinct,       1.04 x  0.42 m
    #     hm3d 2.21               260 nodes,  10 distinct,       0.79 x  0.67 m
    #
    # The last two are not tours that failed — they are the whole of a stair landing and one
    # fragment of a gallery, neither of which is a storey. NODE COUNT DOES NOT SEPARATE THEM: 471
    # nodes bought 17 viewpoints in a box a metre across. Distinct poses does.
    #
    # THE THRESHOLD IS UNMEASURED. Real maps here run 73-269 distinct and the non-maps 10-17, so
    # anything between 18 and 72 separates them on this evidence — which is a gap on five samples,
    # not a rule. 30 is a floor that refuses the clearly-empty and will not refuse a small real
    # room; the measured values are recorded above so the first map that argues with it can be
    # judged against them rather than against my preference.
    distinct = len(set(poses))
    xs = [q[0] for q in poses]
    ys = [q[1] for q in poses]
    min_distinct = 30
    out["distinct_poses"] = distinct
    out["footprint_m"] = [round(max(xs) - min(xs), 2), round(max(ys) - min(ys), 2)]
    out["covers_a_storey"] = distinct >= min_distinct
    out["min_distinct_poses"] = min_distinct
    out["coverage_note"] = ("UNMEASURED threshold. distinct_poses counts unique rtabmap node "
                            "positions: 471 nodes of 17 viewpoints is a map of one spot, and node "
                            "count alone cannot say so. See the source for the five measured maps "
                            "this was set against.")
    pathlib.Path(str(db) + ".floor.json").write_text(json.dumps(out, indent=2))
    print(f"[map] floor stamped: z={out['floor_height_m']:+.2f} spread={spread:.3f}m "
          f"distinct={distinct} covers={out['covers_a_storey']} "
          f"nodes={len(zs)} single_floor={out['single_floor']}"
          + (f" nearest_of {floors}" if floors else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
