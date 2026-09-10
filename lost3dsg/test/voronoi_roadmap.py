#!/usr/bin/env python3
"""A waypoint roadmap and a DFS exploration route for one storey, from the navmesh.

WHY THIS AND NOT RANDOM GOALS. The current policy samples navigable points at random. A random point
sits anywhere -- often against a wall, often behind furniture -- and the walk between two of them is
whatever the path follower produces. The generalized Voronoi diagram is the set of points equidistant
from two or more obstacles, so it runs down the MIDDLE of every corridor and through the centre of
every room. A route along it keeps the camera clear of walls and reaches every region.

THE PIPELINE
  1. rasterise the navmesh at the storey height
  2. distance transform of the free space; mark where neighbouring pixels have DIFFERENT nearest
     obstacles -- that boundary is the Voronoi ridge
  3. thin to one pixel, drop spurs, keep only pixels with room for the robot
  4. waypoints = junctions, dead ends, and a point every `spacing` metres along the corridors
  5. MERGE waypoints closer than `merge_radius` into their centroid, snapped back onto the ridge
  6. build the roadmap: waypoints joined by the ridge paths between them
  7. root set = the waypoints of MAXIMUM DEGREE; pick one at random (seeded)
  8. depth-first search from that root; the robot walks the tree edges, backtracking as DFS does
  9. a 360 degree scan at every stop
"""
import argparse
import heapq
import json
import math
import os
import sys

import numpy as np
from scipy import ndimage

SCAN_STEP_DEG = 10.0      # habitat's turn action, so a full scan is 36 actions


def topdown(navmesh, height, mpp):
    """-> (bool grid, world origin, pathfinder). True where the robot can stand at this storey."""
    import habitat_sim
    pf = habitat_sim.nav.PathFinder()
    pf.load_nav_mesh(navmesh)
    if not pf.is_loaded:
        raise SystemExit(f"navmesh did not load: {navmesh}")
    grid = np.asarray(pf.get_topdown_view(mpp, height))
    lo, _ = pf.get_bounds()
    return grid, (float(lo[0]), float(lo[2])), pf


def largest_component(mask):
    lab, n = ndimage.label(mask, structure=np.ones((3, 3)))
    if n == 0:
        return mask
    sizes = ndimage.sum(mask, lab, range(1, n + 1))
    return lab == (int(np.argmax(sizes)) + 1)


def voronoi_ridge(free, min_clear_px, source_gap_px=3.0):
    """-> (ridge mask, clearance px). The ridge is the generalized Voronoi diagram of free space.

    distance_transform_edt with return_indices gives, for every free pixel, the nearest obstacle
    pixel. Two neighbouring free pixels whose nearest obstacles are far apart sit on the boundary
    between two Voronoi cells, and that boundary is the medial axis.
    """
    dist, ind = ndimage.distance_transform_edt(free, return_indices=True)
    sy, sx = ind[0], ind[1]
    ridge = np.zeros_like(free, dtype=bool)
    for axis, shift in ((0, 1), (1, 1)):
        ay, ax = np.roll(sy, -shift, axis=axis), np.roll(sx, -shift, axis=axis)
        both = free & np.roll(free, -shift, axis=axis)
        ridge |= both & (np.hypot(ay - sy, ax - sx) > source_gap_px)
    # A ridge pixel the robot cannot stand on is not a waypoint.
    return ridge & (dist >= min_clear_px), dist


def thin(mask):
    """Zhang-Suen thinning: reduce a two-pixel-wide ridge to a one-pixel line.

    Without this almost every pixel is a junction, because a boundary two pixels wide gives most
    pixels three or more neighbours.
    """
    img = mask.astype(np.uint8).copy()
    while True:
        removed = False
        for step in (0, 1):
            p = np.pad(img, 1)
            P2, P3, P4 = p[:-2, 1:-1], p[:-2, 2:], p[1:-1, 2:]
            P5, P6, P7 = p[2:, 2:], p[2:, 1:-1], p[2:, :-2]
            P8, P9 = p[1:-1, :-2], p[:-2, :-2]
            seq = [P2, P3, P4, P5, P6, P7, P8, P9, P2]
            B = sum(seq[:-1])
            A = sum(((seq[i] == 0) & (seq[i + 1] == 1)).astype(np.uint8) for i in range(8))
            c1, c2 = ((P2 * P4 * P6, P4 * P6 * P8) if step == 0 else (P2 * P4 * P8, P2 * P6 * P8))
            kill = (img == 1) & (B >= 2) & (B <= 6) & (A == 1) & (c1 == 0) & (c2 == 0)
            if kill.any():
                img[kill] = 0
                removed = True
        if not removed:
            return img.astype(bool)


NEIGH = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]


def build_graph(ridge):
    """-> {pixel: {pixel: weight}} over the thinned ridge, without redundant diagonals.

    A staircase of pixels is 8-connected to its own neighbours' neighbours, so a plain 8-neighbour
    graph turns every bend into a triangle and every pixel into a junction: the skeleton measured
    510 nodes of degree 3 and 302 of degree 4 before this. If either orthogonal corner is present,
    the diagonal adds nothing the two steps around it do not already give.
    """
    pts = {(int(y), int(x)) for y, x in zip(*np.nonzero(ridge))}
    g = {p: {} for p in pts}
    for (y, x) in pts:
        for dy, dx in NEIGH:
            q = (y + dy, x + dx)
            if q not in pts:
                continue
            if dy and dx and ((y + dy, x) in pts or (y, x + dx) in pts):
                continue
            g[(y, x)][q] = math.hypot(dy, dx)
    return g


def prune_spurs(g, min_len_px):
    """Drop dead-end branches shorter than min_len_px. A short spur is a rasterising artefact."""
    changed = True
    while changed:
        changed = False
        for e in [p for p, nb in g.items() if len(nb) == 1]:
            if e not in g:
                continue
            path, cur, prev, length = [e], e, None, 0.0
            while True:
                nb = [q for q in g.get(cur, {}) if q != prev]
                if len(nb) != 1:
                    break
                length += g[cur][nb[0]]
                prev, cur = cur, nb[0]
                path.append(cur)
                if len(g.get(cur, {})) != 2:
                    break
            if length < min_len_px and len(path) > 1:
                for p in path[:-1]:
                    for q in list(g.get(p, {})):
                        g[q].pop(p, None)
                    g.pop(p, None)
                changed = True
    return {p: nb for p, nb in g.items() if nb}


def pick_waypoints(g, spacing_px):
    """Junctions and dead ends, plus a point every spacing_px along the corridors between them."""
    key = [p for p, nb in g.items() if len(nb) != 2]
    chosen, seen = list(key), set(key)
    for start in key:
        for first in g[start]:
            cur, prev, run = first, start, g[start][first]
            while cur not in seen and len(g.get(cur, {})) == 2:
                if run >= spacing_px:
                    chosen.append(cur)
                    seen.add(cur)
                    run = 0.0
                nb = [q for q in g[cur] if q != prev]
                if not nb:
                    break
                run += g[cur][nb[0]]
                prev, cur = cur, nb[0]
    return chosen


def merge_close(waypoints, radius_px, g):
    """Collapse every cluster of waypoints closer than radius_px to ONE of its own members.

    OWNER REQUEST, 2026-09-10, and its follow-up: the merge applies to crossings too. A T-junction
    rasterises into two or three ridge nodes a few centimetres apart, and each would earn its own
    stop and its own 360 degree scan.

    THE REPRESENTATIVE IS A MEMBER, NOT A FREE CENTROID. The first version averaged the positions
    and snapped the average to the nearest ridge pixel. That pixel need not be a junction, so the
    junction stopped being a waypoint while the corridors still met there: two roadmap edges ran
    through a point with no stop on it, and the search could not turn there. Picking the member of
    HIGHEST RIDGE DEGREE -- ties broken by nearness to the centroid -- keeps the stop on the busiest
    junction of the clump, which is also the best place in the clump to scan from.

    Single-link clustering: two waypoints join the same cluster when they are within the radius, so
    a chain of near neighbours collapses to one point. A clump is a clump however it is shaped.
    """
    pts = list(waypoints)
    parent = list(range(len(pts)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(len(pts)):
        for j in range(i + 1, len(pts)):
            if math.dist(pts[i], pts[j]) <= radius_px:
                a, b = find(i), find(j)
                if a != b:
                    parent[a] = b
    clusters = {}
    for i, p in enumerate(pts):
        clusters.setdefault(find(i), []).append(p)

    merged, absorbed, n_merged = [], 0, 0
    for members in clusters.values():
        if len(members) > 1:
            absorbed += len(members) - 1
            n_merged += 1
        cy = sum(m[0] for m in members) / len(members)
        cx = sum(m[1] for m in members) / len(members)
        merged.append(max(members, key=lambda m: (len(g.get(m, {})),
                                                  -((m[0] - cy) ** 2 + (m[1] - cx) ** 2))))
    return sorted(set(merged)), absorbed, n_merged


def waypoint_edges(g, waypoints):
    """-> {waypoint: {waypoint: (length_px, ridge path)}}, the roadmap proper.

    A Dijkstra from each waypoint that STOPS at any other waypoint, so an edge is a stretch of ridge
    with no third waypoint on it. Walking the pixels by hand fails here: after merging, a junction
    pixel may no longer be a waypoint, and a hand walk cannot decide which branch to follow.
    """
    wset = set(waypoints)
    edges = {w: {} for w in waypoints}
    for w in waypoints:
        dist, prev, pq = {w: 0.0}, {}, [(0.0, w)]
        while pq:
            d, u = heapq.heappop(pq)
            if d > dist.get(u, math.inf):
                continue
            if u != w and u in wset:
                path, c = [u], u
                while c in prev:
                    c = prev[c]
                    path.append(c)
                if u not in edges[w] or d < edges[w][u][0]:
                    edges[w][u] = (d, path[::-1])
                continue                      # do not expand past a waypoint
            for v, wt in g.get(u, {}).items():
                nd = d + wt
                if nd < dist.get(v, math.inf):
                    dist[v], prev[v] = nd, u
                    heapq.heappush(pq, (nd, v))
    return edges


def cover_junctions(g, waypoints, radius_px):
    """Give a stop back to any junction the merge left stranded. -> (waypoints, how many).

    SINGLE-LINK CLUSTERING CHAINS. A joins B and B joins C when each pair is within the radius, but
    A and C can then be twice the radius apart, and the representative sits near only one end. On
    scene 00770 that stranded 7 junctions and on 00337 it stranded 2 -- the invariant check found
    them, which is the whole reason it exists.

    Adding a stranded junction back cannot recreate a clump: it is further than the radius from
    every waypoint, which is exactly why it was reported. One pass is enough, because a junction
    that becomes a waypoint covers itself.
    """
    _junc, missed = junctions_uncovered(g, waypoints, radius_px)
    if not missed:
        return waypoints, 0
    return sorted(set(waypoints) | set(missed)), len(missed)


def junctions_uncovered(g, waypoints, radius_px):
    """-> (all ridge junctions, the ones no waypoint stands within radius_px of).

    THE INVARIANT AFTER CONTRACTION. "No roadmap edge may pass through a junction" is too strict
    once clumps are deliberately merged: the merged stop inherits every branch of its cluster, so a
    junction half a metre away is represented, and an edge drawn through it loses nothing. What must
    hold is that every place corridors meet has a stop WITHIN THE MERGE RADIUS. Checked, not
    assumed -- a junction further away than that is a crossing the robot passes without scanning.
    """
    junc = [p for p in g if len(g[p]) >= 3]
    if not waypoints:
        return junc, junc
    missed = [p for p in junc
              if min(math.dist(p, w) for w in waypoints) > radius_px]
    return junc, missed


def dfs_route(edges, root):
    """-> (visit order, full walk of waypoints). Depth-first, nearest branch first.

    THE ROBOT MUST TRAVEL THE BACKTRACK. A DFS visit order names each waypoint once, but between two
    consecutive names the agent may have to retrace several edges. The walk is what it actually
    drives; the order is what it scans.
    """
    order, walk, seen = [], [root], {root}
    stack = [(root, iter(sorted(edges[root], key=lambda v: edges[root][v][0])))]
    order.append(root)
    while stack:
        node, it = stack[-1]
        for nxt in it:
            if nxt in seen:
                continue
            seen.add(nxt)
            order.append(nxt)
            walk.append(nxt)
            stack.append((nxt, iter(sorted(edges[nxt], key=lambda v: edges[nxt][v][0]))))
            break
        else:
            stack.pop()
            if stack:
                walk.append(stack[-1][0])     # backtrack, and the robot drives it
    return order, walk


def resolve_laps(cli, cfg_path):
    """-> (laps, where it came from). Environment beats the file, a flag beats both.

    THE SOURCE IS REPORTED, not just the value. Every other setting in this stack is resolved the
    same way and the bundle records which one won, because a run whose log says "3 laps" and whose
    config says 1 is a run nobody can reproduce.
    """
    if cli is not None:
        return max(1, cli), "--laps"
    env = os.environ.get("FEED_EXPLORATION_LAPS")
    if env:
        try:
            return max(1, int(env)), "FEED_EXPLORATION_LAPS"
        except ValueError:
            raise SystemExit(f"FEED_EXPLORATION_LAPS={env!r} is not a whole number")
    if cfg_path and os.path.isfile(cfg_path):
        import yaml
        cfg = yaml.safe_load(open(cfg_path)) or {}
        v = ((cfg.get("habitat") or {}).get("exploration_laps"))
        if v is not None:
            return max(1, int(v)), f"{cfg_path}: habitat.exploration_laps"
    return 3, "default"


def rdp(pts, eps):
    """Ramer-Douglas-Peucker: drop the points a straight line already represents.

    The ridge is a pixel chain, so a 30 m corridor arrives as 600 points that all sit on one line.
    The robot needs the corners, not the raster. eps is in the same units as pts.
    """
    if len(pts) < 3:
        return list(pts)
    a, b = pts[0], pts[-1]
    ax, ay = a
    bx, by = b
    dx, dy = bx - ax, by - ay
    n = math.hypot(dx, dy)
    worst, wi = -1.0, 0
    for i in range(1, len(pts) - 1):
        px, py = pts[i]
        d = (abs(dy * px - dx * py + bx * ay - by * ax) / n) if n else math.dist(pts[i], a)
        if d > worst:
            worst, wi = d, i
    if worst <= eps:
        return [a, b]
    return rdp(pts[:wi + 1], eps)[:-1] + rdp(pts[wi:], eps)


def build_trajectory(edges, order, walk, to_world, eps_m, step_m, turn_deg, scan_deg,
                     clear_world=None):
    """-> (trajectory, budget). The path the robot drives, and what it costs in frames.

    A STOP IS SCANNED ONCE, ON FIRST ARRIVAL. Depth-first search comes back through a waypoint every
    time it backtracks out of a branch; scanning again would spend 36 frames looking at a place the
    run has already seen. `order` is where it scans, `walk` is what it drives.
    """
    first_visit = {w: i for i, w in enumerate(order)}
    unsafe = [0]
    traj, seen = [], set()
    start = walk[0]
    traj.append({"xyz": to_world(start), "scan_deg": scan_deg, "stop": 0, "leg": None})
    seen.add(start)
    drive_m = 0.0
    for leg, (u, v) in enumerate(zip(walk, walk[1:])):
        _, path = edges[u][v]
        pts = [to_world(q) for q in path]
        xz = [(q[0], q[2]) for q in pts]
        keep = rdp(xz, eps_m)
        # SIMPLIFYING CUTS CORNERS, and a cut corner can cross a wall. The ridge keeps at least the
        # robot's clearance from every obstacle; the straight line between two ridge points does
        # not. Checked by sampling, not assumed: if any shortcut in this leg leaves the free space,
        # the leg keeps its full ridge path.
        if clear_world is not None and len(keep) < len(xz):
            ok = True
            for k in range(1, len(keep)):
                (ax_, az_), (bx_, bz_) = keep[k - 1], keep[k]
                n = max(2, int(math.dist(keep[k - 1], keep[k]) / (step_m / 2)))
                for t in range(n + 1):
                    f = t / n
                    if not clear_world(ax_ + f * (bx_ - ax_), az_ + f * (bz_ - az_)):
                        ok = False
                        break
                if not ok:
                    break
            if not ok:
                keep = xz
                unsafe[0] += 1
        for k in range(1, len(keep)):
            drive_m += math.dist(keep[k - 1], keep[k])
        for k, (x, z) in enumerate(keep[1:], start=1):
            last = (k == len(keep) - 1)
            entry = {"xyz": [round(x, 3), pts[0][1], round(z, 3)], "scan_deg": 0,
                     "stop": None, "leg": leg}
            if last and v not in seen:
                entry["scan_deg"] = scan_deg
                entry["stop"] = first_visit[v]
                seen.add(v)
            traj.append(entry)

    # Turning to face each segment is a real cost, not a rounding error: at 10 degrees per action a
    # right-angle corner is 9 frames.
    turn_total = 0.0
    prev_h = None
    for a_, b_ in zip(traj, traj[1:]):
        h = math.atan2(b_["xyz"][2] - a_["xyz"][2], b_["xyz"][0] - a_["xyz"][0])
        if prev_h is not None:
            d = abs(h - prev_h) % (2 * math.pi)
            turn_total += math.degrees(min(d, 2 * math.pi - d))
        prev_h = h
    budget = {
        "drive_m": round(drive_m, 2),
        "drive_frames": int(round(drive_m / step_m)),
        "corner_turn_deg": int(round(turn_total)),
        "corner_turn_frames": int(round(turn_total / turn_deg)),
        "scan_stops": len(order),
        "scan_frames": len(order) * int(round(scan_deg / turn_deg)),
        "trajectory_points": len(traj),
        "legs_kept_unsimplified": unsafe[0],
    }
    budget["total_frames"] = (budget["drive_frames"] + budget["corner_turn_frames"]
                              + budget["scan_frames"])
    return traj, budget


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--navmesh", required=True)
    ap.add_argument("--height", type=float, required=True, help="storey height, habitat y")
    ap.add_argument("--mpp", type=float, default=0.05, help="metres per pixel")
    ap.add_argument("--robot-radius", type=float, default=0.25, help="metres of clearance required")
    ap.add_argument("--spacing", type=float, default=2.0, help="metres between corridor waypoints")
    ap.add_argument("--merge-radius", type=float, default=0.75,
                    help="metres: waypoints closer than this collapse to their centroid")
    ap.add_argument("--seed", type=int, default=7, help="chooses the root among the max-degree set")
    ap.add_argument("--laps", type=int, default=None,
                    help="complete passes of the storey; overrides FEED_EXPLORATION_LAPS and "
                         "habitat.exploration_laps in the config")
    ap.add_argument("--config", default="/DATA/GRAPH-API/lost3dsg/test/regolo_config.yaml",
                    help="where habitat.exploration_laps is read from")
    ap.add_argument("--fps", type=float, default=3.0, help="feed rate, for the time estimate")
    ap.add_argument("--step", type=float, default=0.15, help="metres per move_forward action")
    ap.add_argument("--simplify", type=float, default=0.20,
                    help="metres: drop trajectory points a straight line already represents")
    ap.add_argument("--out", required=True)
    ap.add_argument("--json", default="")
    a = ap.parse_args()

    grid, (x0, z0), _ = topdown(a.navmesh, a.height, a.mpp)
    mpp = a.mpp
    free = largest_component(np.asarray(grid, dtype=bool))
    ridge, dist = voronoi_ridge(free, a.robot_radius / mpp)
    ridge = largest_component(thin(largest_component(ridge)))

    g = prune_spurs(build_graph(ridge), min_len_px=1.0 / mpp)
    if not g:
        raise SystemExit("no roadmap survived pruning; lower --robot-radius or --mpp")

    raw = pick_waypoints(g, a.spacing / mpp)
    wps, absorbed, n_merged = merge_close(raw, a.merge_radius / mpp, g)
    wps, recovered = cover_junctions(g, wps, a.merge_radius / mpp)
    edges = waypoint_edges(g, wps)
    junc, missed = junctions_uncovered(g, wps, a.merge_radius / mpp)

    # OWNER'S POLICY: the roots are the waypoints of MAXIMUM degree, and one is drawn at random.
    # A high-degree waypoint is where the most corridors meet, so a search from there reaches the
    # branches early instead of driving a long corridor before the house opens up.
    deg = {w: len(nb) for w, nb in edges.items()}
    max_deg = max(deg.values())
    roots = sorted([w for w, d in deg.items() if d == max_deg])
    laps, laps_from = resolve_laps(a.laps, a.config)

    # ONE ROOT, ONE PATH, REPEATED. Owner instruction, 2026-09-10: a lap is the same trajectory
    # driven again, not a new one. A LAP ENDS WHERE IT STARTED -- a depth-first walk backtracks out
    # of every branch and the last backtrack returns to the root -- so the laps chain with no gap
    # and no travel between them. Verified on this roadmap: walk[-1] == root.
    root = roots[np.random.default_rng(a.seed).integers(len(roots))]
    order, walk = dfs_route(edges, root)

    full = [walk[0]]
    walk_px = 0.0
    for u, v in zip(walk, walk[1:]):
        length, path = edges[u][v]
        walk_px += length
        full.extend(path[1:])
    walk_m = walk_px * mpp

    unreached = [w for w in wps if w not in set(order)]

    def world(p):
        return [round(x0 + p[1] * mpp, 3), round(a.height, 3), round(z0 + p[0] * mpp, 3)]

    def clear_world(wx, wz):
        gx = int(round((wx - x0) / mpp))
        gy = int(round((wz - z0) / mpp))
        return 0 <= gy < free.shape[0] and 0 <= gx < free.shape[1] and bool(free[gy, gx])

    one, lap_budget = build_trajectory(edges, order, walk, world, a.simplify, a.step,
                                       SCAN_STEP_DEG, 360.0, clear_world)
    # The lap's first point IS the root, and a later lap starts standing on it, so repeating that
    # entry costs no travel and earns the root its scan on every lap.
    traj = [dict(e, lap=lap) for lap in range(laps) for e in one]
    budget = {k: (round(v * laps, 2) if isinstance(v, float) else v * laps)
              for k, v in lap_budget.items()}
    budget["per_lap"] = lap_budget
    budget["trajectory_points"] = len(traj)
    budget["laps"] = laps

    print(f"free area        : {free.sum() * mpp * mpp:8.1f} m2")
    print(f"ridge length     : {len(g) * mpp:8.1f} m")
    print(f"waypoints        : {len(raw):4d} -> {len(wps):4d}   ({absorbed} absorbed into "
          f"{n_merged} clusters at <= {a.merge_radius} m, "
          f"{recovered} stranded junction(s) given a stop back)")
    print(f"crossings        : {len(junc):4d} ridge junctions, "
          f"{len(junc) - len(missed)} with a stop within {a.merge_radius} m")
    if missed:
        print(f"!! {len(missed)} junction(s) have NO stop within the merge radius: "
              + ", ".join(str(m) for m in missed[:6]))
    print(f"degree           : max {max_deg}, {len(roots)} root candidate(s) at that degree")
    print(f"laps             : {laps:4d}   (from {laps_from}) — the SAME trajectory each time")
    print(f"root chosen      : {root} (seed {a.seed})")
    print(f"DFS visits       : {len(order):4d} of {len(wps)} waypoints"
          + (f"   UNREACHED {len(unreached)}" if unreached else ""))
    print(f"walk length      : {walk_m:8.1f} m  (DFS retraces edges; {len(walk) - 1} legs)")
    print(f"trajectory       : {budget['trajectory_points']} points after "
          f"simplifying at {a.simplify} m"
          + (f"  ({budget['legs_kept_unsimplified']} leg(s) kept full: a shortcut left free space)"
             if budget["legs_kept_unsimplified"] else ""))
    print(f"  drive          : {budget['drive_m']:8.1f} m = {budget['drive_frames']:5d} frames "
          f"at {a.step} m/step")
    print(f"  corner turns   : {budget['corner_turn_deg']:8d} deg = "
          f"{budget['corner_turn_frames']:5d} frames")
    print(f"  360 scans      : {budget['scan_stops']:8d} stops = {budget['scan_frames']:5d} frames")
    print(f"  TOTAL          : {budget['total_frames']:5d} frames = "
          f"{budget['total_frames'] / a.fps / 60:.1f} min at {a.fps:g} f/s")

    if a.json:
        json.dump({"navmesh": a.navmesh, "height": a.height, "meters_per_pixel": mpp,
                   "robot_radius_m": a.robot_radius, "spacing_m": a.spacing,
                   "merge_radius_m": a.merge_radius, "seed": a.seed,
                   "max_degree": max_deg, "root_candidates": [world(r) for r in roots],
                   "root": world(root), "walk_length_m": round(walk_m, 2),
                   "scan_step_deg": SCAN_STEP_DEG,
                   "step_m": a.step, "simplify_m": a.simplify, "budget": budget,
                   "laps": laps, "laps_source": laps_from,
                   "visit_order": [world(p) for p in order],
                   "drive_walk": [world(p) for p in walk],
                   "trajectory": traj},
                  open(a.json, "w"), indent=1)
        print(f"waypoints json   : {a.json}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(21, 9.5))
    ys, xs = np.nonzero(free)
    pad = 10
    xlim = (xs.min() - pad, xs.max() + pad)
    ylim = (ys.min() - pad, ys.max() + pad)
    for ax, t in zip(axes, ("roadmap: Voronoi ridge, merged waypoints, degree",
                            f"DFS from a max-degree root, 360 scan at each of {len(order)} stops"
                            + (f", x{laps} laps" if laps > 1 else ""))):
        ax.imshow(np.where(free, 1.0, 0.0), cmap="Greys", vmin=0, vmax=2.4)
        ax.set_xlim(*xlim)
        ax.set_ylim(ylim[1], ylim[0])
        ax.set_title(t, fontsize=14)
        ax.set_xticks([])
        ax.set_yticks([])
        bx, by = xlim[0] + 8, ylim[1] - 8
        ax.plot([bx, bx + 2.0 / mpp], [by, by], "-", c="#111", lw=3)
        ax.annotate("2 m", (bx + 1.0 / mpp, by - 4), ha="center", fontsize=10)

    sy = np.array([p[0] for p in g])
    sx = np.array([p[1] for p in g])
    axes[0].scatter(sx, sy, s=0.7, c="#cfd8e3", label="Voronoi ridge")
    drawn = set()
    for u, nbs in edges.items():
        for v, (_, path) in nbs.items():
            if (v, u) in drawn:
                continue
            drawn.add((u, v))
            axes[0].plot([p[1] for p in path], [p[0] for p in path], "-", c="#1f77b4", lw=1.4)
    axes[0].plot([], [], "-", c="#1f77b4", lw=1.4, label=f"{len(drawn)} roadmap edges")
    wy = [p[0] for p in wps]
    wx = [p[1] for p in wps]
    sizes = [18 + 26 * deg[w] for w in wps]
    axes[0].scatter(wx, wy, s=sizes, c="#d62728", zorder=5,
                    label=f"{len(wps)} waypoints (size = degree)")
    axes[0].scatter([r[1] for r in roots], [r[0] for r in roots], s=190, marker="o",
                    facecolors="none", edgecolors="goldenrod", linewidths=2.2, zorder=6,
                    label=f"{len(roots)} root candidate(s), degree {max_deg}")
    axes[0].legend(loc="lower right", fontsize=10.5, framealpha=0.92)

    axes[1].scatter(sx, sy, s=0.5, c="#e6ebf2")
    lap_colours = ["#ff7f0e", "#1f77b4", "#2ca02c", "#9467bd", "#8c564b", "#e377c2"]
    for lap in range(laps):
        pts_l = [t for t in traj if t["lap"] == lap]
        if not pts_l:
            continue
        lx = [(t["xyz"][0] - x0) / mpp for t in pts_l]
        ly = [(t["xyz"][2] - z0) / mpp for t in pts_l]
        # DRAWING ONLY: laps run over the same roadmap, so without a sideways offset the last one
        # painted hides every earlier one and three laps look like one. The exported trajectory is
        # untouched -- this shifts pixels in the figure, not waypoints in the route.
        if laps > 1:
            off = (lap - (laps - 1) / 2) * 2.6
            ox, oy = [], []
            for k in range(len(lx)):
                j = min(k + 1, len(lx) - 1)
                i = max(k - 1, 0)
                dx, dy = lx[j] - lx[i], ly[j] - ly[i]
                n = math.hypot(dx, dy) or 1.0
                ox.append(lx[k] - dy / n * off)
                oy.append(ly[k] + dx / n * off)
            lx, ly = ox, oy
        c = lap_colours[lap % len(lap_colours)]
        axes[1].plot(lx, ly, "-", c=c, lw=2.0, zorder=4, alpha=0.85 if laps > 1 else 1.0,
                     label=(f"lap {lap + 1}, {lap_budget['drive_m']:.0f} m"
                            if laps > 1 else
                            f"trajectory, {budget['drive_m']:.0f} m in "
                            f"{budget['trajectory_points']} points"))
    axes[1].scatter(wx, wy, s=34, c="#d62728", zorder=5)
    if laps == 1:
        for i, p in enumerate(order):
            axes[1].annotate(str(i), (p[1], p[0]), fontsize=7.5, color="#111",
                             xytext=(3, 3), textcoords="offset points", zorder=6)
    axes[1].scatter([root[1]], [root[0]], s=260, marker="*", c="#2ca02c", zorder=7,
                    label=f"root, degree {max_deg}")
    axes[1].plot([], [], " ", label=f"{budget['total_frames']} frames = "
                                   f"{budget['total_frames'] / a.fps / 60:.0f} min at {a.fps:g} f/s")
    if unreached:
        axes[1].scatter([p[1] for p in unreached], [p[0] for p in unreached], s=60, marker="x",
                        c="#7f0000", zorder=7, label=f"{len(unreached)} unreached")
    axes[1].legend(loc="lower right", fontsize=10.5, framealpha=0.92)

    fig.suptitle(f"{os.path.basename(a.navmesh)}   storey y={a.height:+.2f} m   "
                 f"{free.sum() * mpp * mpp:.0f} m2 navigable   {mpp * 100:.0f} cm/px   "
                 f"merge {a.merge_radius} m   seed {a.seed}", fontsize=13)
    fig.tight_layout()
    fig.savefig(a.out, dpi=110, bbox_inches="tight")
    print(f"png              : {a.out}")


if __name__ == "__main__":
    sys.exit(main())
