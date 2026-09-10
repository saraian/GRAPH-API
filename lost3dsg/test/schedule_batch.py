#!/usr/bin/env python3
"""One exploration schedule per SCENE: every storey, its roadmap, its DFS trajectory, its laps.

WHY PER SCENE AND PRECOMPUTED. The roadmap depends only on the navmesh, which does not change when
objects are spawned, moved or removed -- the dynamic dataset alters furniture, not the building. So
the schedule can be computed once per scene and reused by every run on it. Two consequences worth
having: two runs of the same scene visit the SAME places in the SAME order, which is what makes
them comparable; and the storey detection stops being a per-run draw that can land on a stair
landing.

THE STOREY RULE IS THE FEED HOST'S, NOT A NEW ONE (habitat_feed_host.py:1659-1700): sample navigable
points, cluster their heights with a 0.8 m gap, take each cluster's MEDIAN as the storey height, and
accept a cluster as a storey only if it holds at least `min_floor_share` of the samples. Clusters
below the bar are LEVELS -- stair landings and fragmented galleries -- and are recorded, never
toured. Reproducing the rule here rather than inventing one keeps the schedule and the run talking
about the same storeys.
"""
import argparse
import hashlib
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import voronoi_roadmap as V  # noqa: E402


def scene_storeys(navmesh, samples=20000, min_share=0.10, seed=7,
                  bin_m=0.05, min_gap_m=1.5, band_m=0.75):
    """-> (storeys, levels, unassigned share). Storeys are PEAKS in the height histogram.

    THE OLD RULE CUT THE HEIGHT AXIS INTO FIXED BANDS. It walked the sorted heights and started a
    new cluster whenever a sample sat more than 0.8 m above the FIRST member of the current one, so
    every cluster was a 0.8 m band whose edges fell wherever the first sample happened to land. Two
    ways that goes wrong: a storey whose samples spread over more than 0.8 m is split in two, and
    each half's share is halved, which can drop a real storey below the acceptance bar; and the
    staircase, which is a thin continuum of samples all the way between storeys, is chopped into
    bands that are reported as levels. Scene 00337 came out as 3 storeys and EIGHT levels.

    MEASURED, WHICH IS WHY PEAKS ARE THE RIGHT SHAPE. At 10 cm resolution hm3d_00861 puts 9251 and
    9062 samples in two bins and 10 to 20 in every bin between them; 00337 puts 5013, 5149 and 4175
    in three bins against the same thin floor. A storey is a spike. A staircase is the noise between
    spikes, and it belongs to no storey.

    So: histogram the heights, find the peaks at least `min_gap_m` apart -- a storey is metres from
    the next, not centimetres -- and give every sample to the nearest peak within `band_m`. Samples
    near no peak are stairs and ramps, and they are counted and reported rather than being made into
    levels nobody asked for.
    """
    import habitat_sim
    from scipy.signal import find_peaks
    pf = habitat_sim.nav.PathFinder()
    pf.load_nav_mesh(navmesh)
    if not pf.is_loaded:
        raise SystemExit(f"navmesh did not load: {navmesh}")
    pf.seed(seed)
    h = np.array(sorted(float(pf.get_random_navigable_point()[1]) for _ in range(samples)))

    lo, hi = h.min() - bin_m, h.max() + bin_m
    edges = np.arange(lo, hi + bin_m, bin_m)
    cnt, _ = np.histogram(h, bins=edges)
    centres = (edges[:-1] + edges[1:]) / 2
    # A storey's samples spread over a few centimetres, so smooth by three bins before looking for
    # peaks. More than that would merge a mezzanine into the storey below it.
    smooth = np.convolve(cnt, np.ones(3) / 3.0, mode="same")
    idx, _ = find_peaks(smooth, distance=max(1, int(round(min_gap_m / bin_m))))
    if len(idx) == 0:
        idx = np.array([int(np.argmax(smooth))])
    peaks = centres[idx]

    # Nearest peak within the band; everything else is stairs.
    d = np.abs(h[:, None] - peaks[None, :])
    nearest = d.argmin(axis=1)
    within = d[np.arange(len(h)), nearest] <= band_m
    storeys, levels = [], []
    for k, pk in enumerate(peaks):
        member = h[(nearest == k) & within]
        if len(member) == 0:
            continue
        entry = {"z": round(float(np.median(member)), 2),
                 "share": round(len(member) / len(h), 4),
                 "samples": int(len(member)),
                 "peak_z": round(float(pk), 2)}
        (storeys if entry["share"] >= min_share else levels).append(entry)
    unassigned = round(float((~within).sum()) / len(h), 4)
    return storeys, levels, unassigned


def schedule_for(navmesh, height, args):
    """-> the schedule dict for one storey, or None when the storey yields no roadmap."""
    mpp = args.mpp
    grid, (x0, z0), _ = V.topdown(navmesh, height, mpp)
    free = V.largest_component(np.asarray(grid, dtype=bool))
    if free.sum() * mpp * mpp < args.min_area:
        return {"skipped": f"only {free.sum() * mpp * mpp:.1f} m2 navigable, "
                           f"below --min-area {args.min_area}"}
    ridge, dist = V.voronoi_ridge(free, args.robot_radius / mpp)
    ridge = V.largest_component(V.thin(V.largest_component(ridge)))
    g = V.prune_spurs(V.build_graph(ridge), min_len_px=1.0 / mpp)
    if not g:
        return {"skipped": "no roadmap survived pruning"}

    raw = V.pick_waypoints(g, args.spacing / mpp)
    wps, absorbed, n_merged = V.merge_close(raw, args.merge_radius / mpp, g)
    wps, recovered = V.cover_junctions(g, wps, args.merge_radius / mpp)
    edges = V.waypoint_edges(g, wps)
    junc, missed = V.junctions_uncovered(g, wps, args.merge_radius / mpp)

    deg = {w: len(nb) for w, nb in edges.items()}
    max_deg = max(deg.values())
    roots = sorted([w for w, d in deg.items() if d == max_deg])
    root = roots[np.random.default_rng(args.seed).integers(len(roots))]
    order, walk = V.dfs_route(edges, root)

    def world(p):
        return [round(x0 + p[1] * mpp, 3), round(height, 3), round(z0 + p[0] * mpp, 3)]

    def clear_world(wx, wz):
        gx, gy = int(round((wx - x0) / mpp)), int(round((wz - z0) / mpp))
        return 0 <= gy < free.shape[0] and 0 <= gx < free.shape[1] and bool(free[gy, gx])

    one, lap_budget = V.build_trajectory(edges, order, walk, world, args.simplify, args.step,
                                         V.SCAN_STEP_DEG, 360.0, clear_world)
    budget = {k: (round(v * args.laps, 2) if isinstance(v, float) else v * args.laps)
              for k, v in lap_budget.items()}
    budget["per_lap"] = lap_budget
    budget["laps"] = args.laps
    budget["trajectory_points"] = len(one) * args.laps

    # COVERAGE, BECAUSE THE ROADMAP CAN MISS ROOMS. largest_component keeps only the biggest
    # connected piece of the ridge, and a room whose ridge does not join the main one is dropped
    # with it -- visible on 00337 y=-0.00, where 73 m2 is navigable and the tour walks a central
    # cross. Measured as the share of free space within `coverage_radius` of a stop, so the gap is
    # a number in the schedule rather than something a reader has to notice in a picture.
    stop_px = np.zeros_like(free, dtype=bool)
    for t in one:
        if t["scan_deg"]:
            gx = int(round((t["xyz"][0] - x0) / mpp))
            gy = int(round((t["xyz"][2] - z0) / mpp))
            if 0 <= gy < free.shape[0] and 0 <= gx < free.shape[1]:
                stop_px[gy, gx] = True
    from scipy import ndimage as _nd
    reach = _nd.distance_transform_edt(~stop_px) * mpp
    covered = float((free & (reach <= args.coverage_radius)).sum()) / max(1, int(free.sum()))
    return {
        "coverage_share": round(covered, 3),
        "coverage_radius_m": args.coverage_radius,
        "height": round(height, 3),
        "navigable_m2": round(float(free.sum()) * mpp * mpp, 1),
        "waypoints": len(wps),
        "waypoints_before_merge": len(raw),
        "junctions": len(junc),
        "junctions_without_a_stop": len(missed),
        "stranded_junctions_recovered": recovered,
        "max_degree": max_deg,
        "root_candidates": [world(r) for r in roots],
        "root": world(root),
        "unreached": [world(w) for w in wps if w not in set(order)],
        "budget": budget,
        "visit_order": [world(p) for p in order],
        "trajectory": one,          # ONE lap; the run repeats it `laps` times
    }


def settings_of(a):
    return {k: getattr(a, k) for k in
            ("mpp", "robot_radius", "spacing", "merge_radius", "simplify", "step", "laps", "seed",
             "min_area", "coverage_radius")}


def settings_sha(settings):
    """A short digest of everything that changes the roadmap.

    THE CACHE KEY IS THE SETTINGS, NOT THE SCENE NAME. A schedule built at merge_radius 0.75 is not
    the schedule for 1.5, and reusing it because the file happens to exist would run one geometry
    while the config describes another. The digest is stored in the file and compared on every run.
    """
    return hashlib.sha256(json.dumps(settings, sort_keys=True).encode()).hexdigest()[:16]


def build_scene(navmesh, scene_id, a):
    """-> the schedule document for one scene, every storey."""
    storeys, levels, stairs = scene_storeys(navmesh)
    doc = {"scene_id": scene_id, "navmesh": navmesh,
           "settings": settings_of(a), "settings_sha256_16": settings_sha(settings_of(a)),
           "storeys_detected": storeys, "levels_not_toured": levels,
           "stairs_share": stairs, "schedule": []}
    print(f"\n{scene_id}: {len(storeys)} storey(s), {len(levels)} level(s) not toured, "
          f"{100 * stairs:.1f}% of samples on stairs or ramps")
    for st in storeys:
        s_ = schedule_for(navmesh, st["z"], a)
        s_["share"] = st["share"]
        doc["schedule"].append(s_)
        if "skipped" in s_:
            print(f"  y={st['z']:+6.2f}  SKIPPED: {s_['skipped']}")
        else:
            b = s_["budget"]
            print(f"  y={st['z']:+6.2f}  {s_['navigable_m2']:6.1f} m2  "
                  f"{s_['waypoints']:3d} stops  root deg {s_['max_degree']}  "
                  f"cover {100 * s_['coverage_share']:3.0f}%  "
                  f"{b['total_frames']:6d} frames = {b['total_frames'] / a.fps / 60:5.1f} min"
                  + (f"  !! {s_['junctions_without_a_stop']} junction(s) with no stop"
                     if s_["junctions_without_a_stop"] else ""))
    return doc


def ensure(navmesh, scene_id, out_dir, a):
    """Cache, or build. -> (path, why). Called by live_run.sh before every run.

    A) CACHED when a file exists for this scene AND its recorded settings match this run's.
    B) BUILT when it does not exist, when the settings differ, or when --regenerate says so
       (habitat.regenerate_schedule in config.yaml).
    A stale schedule is never silently reused: the digest decides, not the file name.
    """
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{scene_id}.schedule.json")
    want = settings_sha(settings_of(a))
    if os.path.isfile(path) and not a.regenerate:
        try:
            have = json.load(open(path))
        except (OSError, ValueError) as exc:
            print(f"[schedule] {path} unreadable ({exc.__class__.__name__}); rebuilding")
        else:
            if have.get("settings_sha256_16") == want:
                n = sum(1 for e in have.get("schedule", []) if "skipped" not in e)
                print(f"[schedule] CACHED {path} ({n} storey(s), settings {want})")
                return path, "cached"
            print(f"[schedule] settings changed ({have.get('settings_sha256_16')} -> {want}); "
                  f"rebuilding {path}")
    elif os.path.isfile(path):
        print(f"[schedule] regenerate_schedule is set; rebuilding {path}")
    else:
        print(f"[schedule] no schedule for {scene_id}; building {path}")
    json.dump(build_scene(navmesh, scene_id, a), open(path, "w"), indent=1)
    print(f"[schedule] BUILT {path} (settings {want})")
    return path, "built"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene-root", default="/DATA/habitat_matterport/hm3d_example",
                    help="directory holding one subdirectory per scene")
    ap.add_argument("--navmesh", default="", help="one scene instead of a whole root")
    ap.add_argument("--scene-id", default="", help="cache key; defaults to the navmesh's directory")
    ap.add_argument("--ensure", action="store_true",
                    help="build only when missing, stale or --regenerate; then print the path")
    ap.add_argument("--regenerate", action="store_true",
                    help="rebuild even when a matching schedule is cached")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--mpp", type=float, default=0.05)
    ap.add_argument("--robot-radius", type=float, default=0.25)
    ap.add_argument("--spacing", type=float, default=2.0)
    ap.add_argument("--merge-radius", type=float, default=0.75)
    ap.add_argument("--simplify", type=float, default=0.20)
    ap.add_argument("--step", type=float, default=0.15)
    ap.add_argument("--laps", type=int, default=3)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--min-area", type=float, default=5.0,
                    help="m2: a storey smaller than this yields no schedule")
    ap.add_argument("--fps", type=float, default=3.0)
    ap.add_argument("--coverage-radius", type=float, default=3.0,
                    help="m: free space within this distance of a stop counts as covered")
    a = ap.parse_args()

    if a.navmesh:
        if not os.path.isfile(a.navmesh):
            raise SystemExit(f"[schedule] no navmesh at {a.navmesh}")
        sid = a.scene_id or os.path.basename(os.path.dirname(a.navmesh))
        path, why = ensure(a.navmesh, sid, a.out_dir, a)
        print(f"SCHEDULE_FILE={path}")
        return 0

    os.makedirs(a.out_dir, exist_ok=True)
    navmeshes = []
    for root, _dirs, files in os.walk(a.scene_root):
        for f in files:
            if f.endswith(".navmesh"):
                navmeshes.append(os.path.join(root, f))
    navmeshes.sort()
    if not navmeshes:
        raise SystemExit(f"no .navmesh under {a.scene_root}")

    index = []
    for nm in navmeshes:
        scene_id = os.path.basename(os.path.dirname(nm))
        entry = build_scene(nm, scene_id, a)
        path = os.path.join(a.out_dir, f"{scene_id}.schedule.json")
        json.dump(entry, open(path, "w"), indent=1)
        toured = [s_ for s_ in entry["schedule"] if "skipped" not in s_]
        index.append({"scene_id": scene_id, "file": os.path.basename(path),
                      "storeys": len(entry["storeys_detected"]),
                      "storeys_with_a_schedule": len(toured),
                      "stops": sum(s_["waypoints"] for s_ in toured),
                      "frames": sum(s_["budget"]["total_frames"] for s_ in toured)})
    json.dump({"scene_root": a.scene_root, "scenes": index},
              open(os.path.join(a.out_dir, "index.json"), "w"), indent=1)
    print(f"\n{len(index)} scene(s) -> {a.out_dir}/index.json")


if __name__ == "__main__":
    sys.exit(main())
