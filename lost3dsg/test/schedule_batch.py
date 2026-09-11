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
import math
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


def _in_poly(poly, x, z):
    """Ray casting on a region's XZ polyloop. HM3D regions are polygons, not boxes."""
    n, ins = len(poly), False
    for i in range(n):
        x1, z1 = poly[i]
        x2, z2 = poly[(i + 1) % n]
        if (z1 > z) != (z2 > z) and x < (x2 - x1) * (z - z1) / (z2 - z1 + 1e-12) + x1:
            ins = not ins
    return ins


def room_seeds(args, ridge, free, dist, height, x0, z0, mpp):
    """-> one ridge pixel to add for each GT room that holds none. Empty without a manifest."""
    rooms = _rooms_at(args, height, free, x0, z0, mpp)
    if not rooms:
        return []

    def world(p):
        return (x0 + p[1] * mpp, z0 + p[0] * mpp)

    ry, rx = np.nonzero(ridge)
    have = [(int(a), int(b)) for a, b in zip(ry, rx)]
    seeds = []
    for r in rooms:
        poly = r.get("polygon_xz_m") or []
        if len(poly) < 3 or any(_in_poly(poly, *world(p)) for p in have):
            continue
        a, b = r["aabb_min_m"], r["aabb_max_m"]
        ys = range(max(0, int((a[2] - z0) / mpp)), min(free.shape[0], int((b[2] - z0) / mpp) + 1))
        xs = range(max(0, int((a[0] - x0) / mpp)), min(free.shape[1], int((b[0] - x0) / mpp) + 1))
        cand = [(y, x) for y in ys for x in xs if free[y, x] and _in_poly(poly, *world((y, x)))]
        if cand:
            # the most open point in the room: the best place in it to stand and turn
            seeds.append(max(cand, key=lambda p: dist[p]))
    return seeds


def _rooms_at(args, height, free=None, x0=0.0, z0=0.0, mpp=0.05):
    """-> the ground-truth rooms that have WALKABLE FLOOR on this storey.

    GA-472. THE MANIFEST'S floor_index IS NOT WHERE THE ROOM IS WALKABLE, and trusting it hid a
    room from the tour entirely. MEASURED on hm3d_00861: region 8 is filed under floor 0, the
    -1.59 storey, and every navigable point inside its polygon sits between +0.74 and +1.24 -- the
    UPPER storey. Selected by floor_index it was searched for on a storey where it has no floor,
    found empty, and reported unreachable; on the storey where it is actually walkable it was never
    looked for at all.

    So membership is decided by the storey's own navigable raster: a room belongs here when the
    robot can stand somewhere inside its polygon at this height. That is the same question the tour
    has to answer anyway, and it cannot disagree with itself.

    Without a `free` grid this falls back to floor_index, which is what the batch reporting path
    uses before a storey is rasterised.
    """
    if not args.gt_manifest:
        return []
    gt = json.load(open(args.gt_manifest))
    regions = gt.get("ground_truth_regions", [])
    if free is None:
        floors = gt.get("ground_truth_floors_m") or []
        if not floors:
            return []
        fi = min(range(len(floors)), key=lambda i: abs(floors[i] - height))
        return [r for r in regions if r.get("floor_index") == fi]

    here = []
    for r in regions:
        poly = r.get("polygon_xz_m") or []
        if len(poly) < 3:
            continue
        a, b = r["aabb_min_m"], r["aabb_max_m"]
        ys = range(max(0, int((a[2] - z0) / mpp)), min(free.shape[0], int((b[2] - z0) / mpp) + 1))
        xs = range(max(0, int((a[0] - x0) / mpp)), min(free.shape[1], int((b[0] - x0) / mpp) + 1))
        if any(free[y, x] and _in_poly(poly, x0 + x * mpp, z0 + y * mpp)
               for y in ys for x in xs):
            here.append(r)
    return here


def cover_rooms(args, g, waypoints, free, dist, height, x0, z0, mpp):
    """Put a stop inside every ground-truth room that has none. -> (added, still unreachable).

    OWNER REQUEST, 2026-09-10: at least one 360 scan in every room. The tour follows the Voronoi
    ridge, which runs where the space is widest, so a small room off a corridor -- a bathroom, a
    closet -- can hold no ridge at all, or hold a stub the 2 m spacing never lands a stop on.
    MEASURED against the HM3D region polyloops on hm3d_00861: 9 of 12 rooms per storey, the six
    missed ones between 1.0 and 2.7 m across.

    THE STOP GOES ON THE ROOM'S OWN RIDGE WHERE THERE IS ONE -- already connected to the roadmap,
    already holding the robot's clearance, and its widest point is the best place in the room to
    scan from. Only a room with no ridge falls back to the free pixel with the most clearance, and
    that case is REPORTED rather than folded in: such a stop may not join the roadmap, and a
    schedule that silently claims a stop the agent cannot drive to is worse than one that says so.
    """
    rooms = _rooms_at(args, height, free, x0, z0, mpp)
    if not rooms:
        return [], None

    def world(p):
        return (x0 + p[1] * mpp, z0 + p[0] * mpp)

    ridge_px = list(g.keys())
    added, unreachable = [], []
    for r in rooms:
        poly = r.get("polygon_xz_m") or []
        if len(poly) < 3:
            continue
        if any(_in_poly(poly, *world(w)) for w in waypoints):
            continue
        inside_ridge = [p for p in ridge_px if _in_poly(poly, *world(p))]
        if inside_ridge:
            added.append(max(inside_ridge, key=lambda p: dist[p]))
            continue
        # No ridge inside even after seeding and bridging: the room could not be joined to the
        # roadmap. Reported, never faked -- a stop the agent cannot drive to is worse than a gap
        # the schedule admits to.
        unreachable.append({"region_id": r.get("region_id"),
                            "why": "no ridge inside the room after seeding and bridging; either it "
                                   "holds no navigable pixel at this storey, or the bridge to it "
                                   "would have left the walkable surface"})
    return added, unreachable


def schedule_for(navmesh, height, args):
    """-> the schedule dict for one storey, or None when the storey yields no roadmap."""
    mpp = args.mpp
    grid, (x0, z0), _ = V.topdown(navmesh, height, mpp)
    free = V.largest_component(np.asarray(grid, dtype=bool))
    if free.sum() * mpp * mpp < args.min_area:
        return {"skipped": f"only {free.sum() * mpp * mpp:.1f} m2 navigable, "
                           f"below --min-area {args.min_area}"}
    ridge, dist = V.voronoi_ridge(free, args.robot_radius / mpp)
    ridge = V.thin(ridge)
    # GA-471. SEED THE ROOMS THAT HOLD NO RIDGE, BEFORE bridging, not after. Adding a free-space
    # point to the WAYPOINT list later does not work and the first version did exactly that: the
    # roadmap graph is built on ridge pixels, so a waypoint that is not one has no edges, never
    # enters the depth-first walk, and the room reads as uncovered while the schedule claims a stop
    # in it. Seeded here, the point becomes ridge, the bridging step connects it like any other
    # piece, and the tour reaches it.
    ridge, bridges = V.connect_ridge_components(ridge, free, args.max_bridge / mpp)
    ridge = V.largest_component(ridge)
    g = V.prune_spurs(V.build_graph(ridge), min_len_px=1.0 / mpp)
    if not g:
        return {"skipped": "no roadmap survived pruning"}

    # GA-471. THE ROOMS ARE SEEDED AFTER PRUNING, and the order is the whole fix. prune_spurs drops
    # every dead-end branch under a metre, which is exactly the stub reaching into a 0.5 m2 alcove:
    # seeding before it ran put a pixel in the room and pruning took it straight back out, so the
    # stop count never moved and the room stayed uncovered. Seeded here, against the graph that
    # survives, and joined through free space -- then the graph is rebuilt WITHOUT pruning so the
    # new stubs live.
    kept = np.zeros_like(ridge)
    for y, x in g:
        kept[y, x] = True
    seeds = room_seeds(args, kept, free, dist, height, x0, z0, mpp)
    seeds_joined = 0
    for sy, sx in seeds:
        kept[sy, sx] = True
        # Through free space, not in a straight line: the path from a small room to the corridor
        # goes through a doorway, and a straight segment clips its frame.
        seeds_joined += bool(V.connect_through_free(kept, free, (sy, sx),
                                                    int(args.max_room_path / mpp)))
    if seeds:
        g = V.build_graph(V.largest_component(kept))
        ridge = kept

    raw = V.pick_waypoints(g, args.spacing / mpp)
    wps, absorbed, n_merged = V.merge_close(raw, args.merge_radius / mpp, g)
    wps, recovered = V.cover_junctions(g, wps, args.merge_radius / mpp)
    rooms_added, rooms_missed = cover_rooms(args, g, wps, free, dist, height, x0, z0, mpp)
    wps = sorted(set(wps) | set(rooms_added))
    edges = V.waypoint_edges(g, wps)
    junc, missed = V.junctions_uncovered(g, wps, args.merge_radius / mpp)

    deg = {w: len(nb) for w, nb in edges.items()}
    max_deg = max(deg.values())
    roots = sorted([w for w, d in deg.items() if d == max_deg])
    root = roots[np.random.default_rng(args.seed).integers(len(roots))]
    order, walk = V.dfs_route(edges, root)
    # ROUTE ORDER. The DFS leaves a branch the way it came in, so its walk crosses itself; 2-opt
    # reverses any segment that shortens the tour, over roadmap distances rather than straight
    # lines, so nothing it proposes goes through a wall. `dfs` is kept because every schedule
    # before 2026-09-11 used it and a comparison needs the old arm.
    if args.route_order == "2opt":
        dmat = _roadmap_distances(edges)

        def _d(a, b):
            return dmat.get((a, b), float("inf"))

        order = V.two_opt(order, _d)
        walk = _walk_through(edges, order, dmat)

    def world(p):
        return [round(x0 + p[1] * mpp, 3), round(height, 3), round(z0 + p[0] * mpp, 3)]

    def clear_world(wx, wz):
        gx, gy = int(round((wx - x0) / mpp)), int(round((wz - z0) / mpp))
        return 0 <= gy < free.shape[0] and 0 <= gx < free.shape[1] and bool(free[gy, gx])

    # PER-STOP SCAN ANGLES. A full circle is 360/turn_step frames and 45% of a run went on them.
    # Each stop is asked what its turn actually reveals: the bearings from it to free cells that no
    # EARLIER stop already sees. A stop in a fresh room still turns the full circle; one in a
    # corridor the route has already walked turns through the arc that holds the new ground and
    # stops there. Rounded UP to a whole number of turn actions, and never below one.
    scan_for = None
    if args.adaptive_scan:
        scan_for = _scan_angles(free, order, x0, z0, mpp, args)

    one, lap_budget = V.build_trajectory(edges, order, walk, world, args.simplify, args.step,
                                         args.turn_step_deg, 360.0, clear_world,
                                         scan_for=scan_for, smooth=args.smooth_path)
    # RE-PRICE THE SCANS UNDER THE PLAN. build_trajectory charges one frame per turn action, which
    # is the continuous scan. A stepped scan holds each heading for hold_frames and repeats the
    # rotation once per tilt, and the floor lifts any stop the adaptive angle cut too short. The
    # trajectory is unchanged; only what each stop COSTS changes, and the bundle must say so.
    plan = scan_plan(args)
    scan_frames = sum(scan_cost_frames(t["scan_deg"], args) for t in one if t["scan_deg"])
    lap_budget["scan_frames"] = scan_frames
    lap_budget["total_frames"] = (lap_budget["drive_frames"] + lap_budget["corner_turn_frames"]
                                  + scan_frames)
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
    stops_px = []
    for t in one:
        if t["scan_deg"]:
            gx = int(round((t["xyz"][0] - x0) / mpp))
            gy = int(round((t["xyz"][2] - z0) / mpp))
            if 0 <= gy < free.shape[0] and 0 <= gx < free.shape[1]:
                stops_px.append((gy, gx))
    rad_px, rng_px = args.coverage_radius / mpp, args.coverage_range / mpp
    cov = V.covered_mask(free, stops_px, args.coverage_model, rad_px, rng_px)
    covered = float(cov.sum()) / max(1, int(free.sum()))

    # THE COVERING VARIANT. `--covering` keeps adding stops until every navigable cell is covered,
    # so the file is a promise rather than a measurement. It is a SEPARATE variant, not the
    # default, because the extra stops cost scan frames and the owner decides that trade per run.
    topped_up = []
    covered_before_topup = covered
    if args.covering and covered < args.coverage_target:
        topped_up, covered_before_topup, covered = V.top_up_stops(
            free, stops_px, args.coverage_model, rad_px, rng_px,
            target=args.coverage_target, max_add=args.max_extra_stops)
        for gy, gx in topped_up:
            one.append({"xyz": world((gy, gx)), "scan_deg": 360.0,
                        "stop": len(order) + len(topped_up) - 1, "leg": None,
                        "added_for": "coverage"})
        # The budget has to grow with them or the file understates its own run.
        extra = len(topped_up) * scan_cost_frames(360.0, args)
        lap_budget["scan_stops"] += len(topped_up)
        lap_budget["scan_frames"] += extra
        lap_budget["total_frames"] += extra
        budget = {k: (round(v * args.laps, 2) if isinstance(v, float) else v * args.laps)
                  for k, v in lap_budget.items()}
        budget["per_lap"] = lap_budget
        budget["laps"] = args.laps
        budget["trajectory_points"] = len(one) * args.laps
    return {
        "gt_rooms_without_a_stop": rooms_missed,
        "gt_room_stops_added": len(rooms_added),
        "ridge_bridges": bridges,
        "room_seeds": len(seeds),
        "room_seeds_joined": seeds_joined,
        "coverage_share": round(covered, 3),
        # WHAT THAT SHARE MEANS. Three models give three different numbers for the same schedule,
        # so the number is useless without the model beside it and a reader must never compare
        # across them. coverage_share_before_topup says what the route alone achieved; the
        # difference is what --covering had to buy.
        "coverage_model": args.coverage_model,
        "coverage_share_before_topup": covered_before_topup,
        "coverage_stops_added": len(topped_up),
        "coverage_target": args.coverage_target if args.covering else None,
        "coverage_radius_m": args.coverage_radius if args.coverage_model == "radius" else None,
        "coverage_range_m": args.coverage_range if args.coverage_model == "los_range" else None,
        "route_order": args.route_order,
        "turn_step_deg": args.turn_step_deg,
        # HOW A STOP SCANS, as data the feed host executes rather than a number it has to infer.
        "scan_plan": plan,
        "scan_frames_min": min((scan_cost_frames(t["scan_deg"], args)
                                for t in one if t["scan_deg"]), default=0),
        "smooth_path": args.smooth_path,
        "adaptive_scan": args.adaptive_scan,
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
             "min_area", "coverage_radius", "coverage_model", "coverage_range", "coverage_target",
             "covering", "turn_step_deg", "route_order", "smooth_path", "adaptive_scan",
             "min_scan_cycles", "cycle_seconds", "fps_budget", "stepped_scan",
             "scan_hold_frames", "scan_tilts")}


def settings_sha(settings):
    """A short digest of everything that changes the roadmap.

    THE CACHE KEY IS THE SETTINGS, NOT THE SCENE NAME. A schedule built at merge_radius 0.75 is not
    the schedule for 1.5, and reusing it because the file happens to exist would run one geometry
    while the config describes another. The digest is stored in the file and compared on every run.
    """
    return hashlib.sha256(json.dumps(settings, sort_keys=True).encode()).hexdigest()[:16]


def scan_floor_frames(a):
    """-> the fewest turn actions any stop may spend scanning.

    A merge commits only after `merge_min_consecutive` consecutive sweeps see the same pair, and a
    detection cycle takes --cycle-seconds. Below that many frames a stop cannot produce a merge,
    whatever its geometry says it needs to see.
    """
    return max(1, int(math.ceil(a.min_scan_cycles * a.cycle_seconds * a.fps_budget)))


def scan_plan(a):
    """-> how a stop performs its scan, as data the feed host executes.

    continuous  one turn action per frame. A heading is held for ONE frame, 0.33 s at 3 f/s, which
                is why a completed tour recorded a longest still stretch of 1 second in 24 minutes:
                the scan is a drive-through wearing the name of a scan.
    stepped     each heading held still for `hold_frames`, so a detection cycle can finish on a
                fixed view. `tilts_deg` repeats the whole rotation once per camera tilt.

    OFF BY DEFAULT and the cost is why: stepped is 150 h a lap over 71 storeys against 22, and
    287 h at the owner's two tilts. Owner 2026-09-11: build it, leave it off.
    """
    tilts = [float(t) for t in str(a.scan_tilts).split(",") if t.strip() != ""] or [0.0]
    hold = int(a.scan_hold_frames) or max(1, int(math.ceil(a.cycle_seconds * a.fps_budget)))
    return {
        "mode": "stepped" if a.stepped_scan else "continuous",
        "tilts_deg": tilts,
        "hold_frames": hold if a.stepped_scan else 1,
        "turn_step_deg": a.turn_step_deg,
        "min_scan_frames": scan_floor_frames(a),
        "note": ("each heading is held still for hold_frames, once per tilt"
                 if a.stepped_scan else
                 "one turn action per frame; a heading is held for one frame"),
    }


def scan_cost_frames(deg, a):
    """-> frames one stop spends scanning, under the plan in force."""
    per_rotation = max(scan_floor_frames(a), int(round(float(deg) / a.turn_step_deg)))
    plan = scan_plan(a)
    return per_rotation * plan["hold_frames"] * len(plan["tilts_deg"])


def _roadmap_distances(edges):
    """-> {(a, b): metres} between every pair of waypoints, along the roadmap.

    Dijkstra from each waypoint over the waypoint graph. 2-opt needs a distance it can trust: a
    straight line between two waypoints may cross a wall, and a tour optimised on straight lines is
    shorter only on paper.
    """
    import heapq
    out = {}
    for src in edges:
        seen = {src: 0.0}
        q = [(0.0, src)]
        while q:
            d, u = heapq.heappop(q)
            if d > seen.get(u, float("inf")):
                continue
            for v, (w, _path) in edges[u].items():
                nd = d + w
                if nd < seen.get(v, float("inf")):
                    seen[v] = nd
                    heapq.heappush(q, (nd, v))
        for v, d in seen.items():
            out[(src, v)] = d
    return out


def _walk_through(edges, order, dmat):
    """-> the waypoints the robot actually drives, for a visiting order that may jump.

    A 2-opt tour names stops that are not neighbours on the roadmap, so between two of them the
    agent drives a shortest path. This expands each hop into that path; consecutive duplicates are
    dropped so a hop of length one does not repeat a waypoint.
    """
    import heapq
    walk = [order[0]]
    for a, b in zip(order, order[1:]):
        if b in edges[a]:
            walk.append(b)
            continue
        prev, seen, q = {}, {a: 0.0}, [(0.0, a)]
        while q:
            d, u = heapq.heappop(q)
            if u == b:
                break
            if d > seen.get(u, float("inf")):
                continue
            for v, (w, _p) in edges[u].items():
                nd = d + w
                if nd < seen.get(v, float("inf")):
                    seen[v] = nd
                    prev[v] = u
                    heapq.heappush(q, (nd, v))
        if b not in seen:
            walk.append(b)          # disconnected: the mover will report the leg as unreachable
            continue
        chain, cur = [b], b
        while cur != a:
            cur = prev[cur]
            chain.append(cur)
        walk.extend(reversed(chain[:-1]))
    return walk


def _scan_angles(free, order, x0, z0, mpp, args):
    """-> a function waypoint -> degrees to turn there.

    WHAT A TURN IS FOR is seeing ground the run has not seen. Each stop, in visiting order, is given
    the bearings to the free cells that are visible FROM IT and were not already visible from an
    earlier stop; the angle returned is the smallest arc holding those bearings, rounded up to a
    whole turn action. The first stop always turns the full circle, and so does any stop whose new
    ground wraps around it.

    ORDER MATTERS AND THAT IS DELIBERATE. A stop early in the route pays for the ground; a later one
    passing the same corridor does not pay twice.
    """
    cap = (args.coverage_range / mpp) if args.coverage_model == "los_range" else None
    step = float(args.turn_step_deg)
    # THE FLOOR. A stop that turns through fewer frames than a merge needs cannot produce one, so
    # the arc a stop "needs" geometrically is not the arc it may be given. 649 of 1900 stops fell
    # under it when the adaptive angle was free to choose (measured 2026-09-11), the shortest a
    # single frame, while the MEAN read a comfortable 3.5 cycles.
    floor_deg = min(360.0, scan_floor_frames(args) * step)
    seen_any = np.zeros_like(free, dtype=bool)
    angles = {}
    for i, w in enumerate(order):
        gy, gx = int(w[0]), int(w[1])
        if not (0 <= gy < free.shape[0] and 0 <= gx < free.shape[1]) or not free[gy, gx]:
            angles[w] = 360.0
            continue
        vis = V.visible_from(free, gy, gx, cap, n_rays=360) & free
        new = vis & ~seen_any
        seen_any |= vis
        cnt = int(new.sum())
        if i == 0 or cnt == 0:
            angles[w] = 360.0 if i == 0 else floor_deg
            continue
        ys, xs = np.nonzero(new)
        bear = np.degrees(np.arctan2(ys - gy, xs - gx)) % 360.0
        # The smallest arc covering every bearing is 360 minus the widest empty gap between two
        # consecutive bearings. A stop whose new ground surrounds it has no gap, so it turns fully.
        b = np.sort(bear)
        gaps = np.diff(np.concatenate([b, b[:1] + 360.0]))
        span = 360.0 - float(gaps.max())
        angles[w] = min(360.0, max(floor_deg, math.ceil(span / step) * step))
    return lambda w: angles.get(w, 360.0)


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


def variant_suffix(a):
    """-> "_covering" for a topped-up schedule, "" for the plain one."""
    return "_covering" if getattr(a, "covering", False) else ""


def ensure(navmesh, scene_id, out_dir, a):
    """Cache, or build. -> (path, why). Called by run.sh before every run.

    A) CACHED when a file exists for this scene AND its recorded settings match this run's.
    B) BUILT when it does not exist, when the settings differ, or when --regenerate says so
       (habitat.regenerate_schedule in config.yaml).
    A stale schedule is never silently reused: the digest decides, not the file name.
    """
    os.makedirs(out_dir, exist_ok=True)
    # THE VARIANT IS IN THE NAME. A covering schedule and a plain one are different runs of the
    # same scene -- different stop counts, different lengths, different coverage guarantees -- so
    # they must not overwrite each other, and a bundle naming its schedule file must say which it
    # drove. The digest still decides whether a file is stale; the suffix only keeps the two apart.
    path = os.path.join(out_dir, f"{scene_id}{variant_suffix(a)}.schedule.json")
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
    ap.add_argument("--max-room-path", type=float, default=12.0,
                    help="m: longest free-space path used to join a room's seed to the roadmap")
    ap.add_argument("--max-bridge", type=float, default=2.0,
                    help="m: longest straight bridge allowed between two pieces of ridge")
    ap.add_argument("--gt-manifest", default="",
                    help="hm3d_ground_truth_manifest.py output; guarantees a stop in every GT room")
    ap.add_argument("--coverage-radius", type=float, default=3.0,
                    help="m: radius model only -- free space within this of a stop counts as "
                         "covered, straight-line THROUGH WALLS")
    # --- how coverage is measured (owner 2026-09-11: every model a parameter, los the default) ---
    ap.add_argument("--coverage-model", choices=V.COVERAGE_MODELS, default="los",
                    help="los: a stop covers what it can SEE, no distance limit, which is what the "
                         "simulator's depth sensor does. los_range: the same, stopped at "
                         "--coverage-range. radius: the pre-2026-09-11 measure, kept so old numbers "
                         "reproduce; it counts a room behind a wall as covered")
    ap.add_argument("--coverage-range", type=float, default=8.0,
                    help="m: los_range only -- the useful depth of the camera. CHOSEN, not "
                         "measured; no artefact records the real range yet")
    ap.add_argument("--coverage-target", type=float, default=1.0,
                    help="the share of navigable area --covering tops up to")
    ap.add_argument("--covering", action="store_true",
                    help="add stops until coverage reaches --coverage-target. Writes the "
                         "_covering variant: a promise rather than a measurement, at the price of "
                         "a full scan per added stop")
    ap.add_argument("--max-extra-stops", type=int, default=60,
                    help="a ceiling on what --covering may add, so one bad storey cannot make a "
                         "schedule nobody can run")
    # --- what a lap costs (all four default ON, owner 2026-09-11) ---
    ap.add_argument("--turn-step-deg", type=float, default=V.SCAN_STEP_DEG,
                    help="degrees per turn action. habitat_feed_host.py must use the same number "
                         "or the budget describes a run that did not happen")
    ap.add_argument("--route-order", choices=("2opt", "dfs"), default="2opt",
                    help="2opt: nearest neighbour then 2-opt over roadmap distances. dfs: the "
                         "pre-2026-09-11 depth-first order, which drives every backtrack")
    ap.add_argument("--smooth-path", dest="smooth_path", action="store_true", default=True,
                    help="drop any path point its neighbours can see past (default)")
    ap.add_argument("--no-smooth-path", dest="smooth_path", action="store_false",
                    help="keep the simplified ridge path corner for corner")
    # ADAPTIVE SCAN IS OPT-IN, and that is a measured decision rather than caution. It shortens a
    # stop's turn to the arc holding ground no earlier stop has seen, which saved 10 percentage
    # points of run time -- and left 649 of 1900 stops (34.2%) below the frames a merge needs, the
    # shortest of them a single frame. The MEAN was a comfortable 3.5 cycles per visit; the tail
    # was not, and the mean is what hid it. Owner 2026-09-11: a full circle at every stop.
    ap.add_argument("--adaptive-scan", dest="adaptive_scan", action="store_true", default=False,
                    help="turn only through the arc holding ground no earlier stop has seen. "
                         "FASTER AND UNSAFE FOR MERGES unless --min-scan-cycles covers the tail")
    ap.add_argument("--full-scan", dest="adaptive_scan", action="store_false",
                    help="a full circle at every stop, whatever it reveals (default)")
    ap.add_argument("--min-scan-cycles", type=float, default=2.0,
                    help="no stop turns through fewer frames than this many detection cycles. "
                         "Matches merge_min_consecutive: a merge cannot commit on fewer sweeps")
    ap.add_argument("--cycle-seconds", type=float, default=4.3,
                    help="a detection cycle in seconds. MEASURED with time_metrics.py on the two "
                         "complete-tour bundles 20260911_133641 and _140421: total_ms median "
                         "4097-4306, cycle_ms median 4295-4398, of which vlm_ms 3871-4222 -- the "
                         "labelling call IS the cycle. 3.2 was an estimate and 28% optimistic; a "
                         "floor derived from it fell short of merge_min_consecutive. Re-measure "
                         "and pass the new number rather than editing any other value")
    ap.add_argument("--fps-for-budget", dest="fps_budget", type=float, default=3.0,
                    help="frames per second the run publishes, used only to turn the frame floor "
                         "into seconds. It does not change what the agent does")
    # --- the stepped scan (owner's ask 2026-09-11), OFF by default: it costs 13x ---
    ap.add_argument("--stepped-scan", action="store_true",
                    help="hold each heading still for --scan-hold-frames instead of turning every "
                         "frame. A continuous turn holds a heading for ONE frame, so a scan is a "
                         "drive-through; stepped makes it a scan. Costs 150 h a lap over 71 storeys "
                         "against 22, and 287 h with two tilts")
    ap.add_argument("--scan-hold-frames", type=int, default=0,
                    help="frames to hold each heading under --stepped-scan. 0 derives it from "
                         "--cycle-seconds so each heading gets one whole detection cycle")
    ap.add_argument("--scan-tilts", default="0",
                    help="comma-separated camera tilts in degrees, one full rotation each. "
                         "\"30,0\" is the owner's two-rotation ask and doubles the scan bill")
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
        path = os.path.join(a.out_dir, f"{scene_id}{variant_suffix(a)}.schedule.json")
        json.dump(entry, open(path, "w"), indent=1)
        toured = [s_ for s_ in entry["schedule"] if "skipped" not in s_]
        index.append({"scene_id": scene_id, "file": os.path.basename(path),
                      "storeys": len(entry["storeys_detected"]),
                      "storeys_with_a_schedule": len(toured),
                      "stops": sum(s_["waypoints"] for s_ in toured),
                      "frames": sum(s_["budget"]["total_frames"] for s_ in toured)})
    json.dump({"scene_root": a.scene_root, "settings": settings_of(a),
               "settings_sha256_16": settings_sha(settings_of(a)), "scenes": index},
              open(os.path.join(a.out_dir, f"index{variant_suffix(a)}.json"), "w"), indent=1)
    print(f"\n{len(index)} scene(s) -> {a.out_dir}/index.json")


if __name__ == "__main__":
    sys.exit(main())
