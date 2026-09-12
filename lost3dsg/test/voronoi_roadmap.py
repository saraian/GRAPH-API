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

# THE TURN ACTION, in degrees, and it is a MERGE parameter as much as a speed one.
#
# A merge commits only after `merge_min_consecutive` consecutive sweeps see the same pair -- 2 in
# the arm running now -- and a detection cycle takes about 3.2 s, so a stop needs roughly 19 frames
# at 3 f/s before a merge can commit there. A full 360 scan costs 360 / SCAN_STEP_DEG frames:
#
#     10 deg -> 36 frames = 12.0 s = 3.75 cycles     clears the threshold
#     20 deg -> 18 frames =  6.0 s = 1.87 cycles     DOES NOT, at any stop
#
# 20 was tried on 2026-09-11 for the 36% of run time scans cost and reverted the same day, because
# the saving was taken out of the one thing the scan exists to produce. habitat_feed_host.py MUST
# agree with this number -- it is the amount on the turn_left action spec and the divisor in
# ScheduledTour._scan_frames -- or the budget in the schedule describes a run that did not happen.
# habitat.turn_step_deg carries it to both.
SCAN_STEP_DEG = 10.0

# How coverage is measured. The choice changes what "100%" means, so it is recorded in every
# schedule beside the number.
#   los        a stop covers what it can SEE: the ray to the point is unobstructed. No distance
#              limit, which is right for the simulator -- its depth sensor has none.
#   los_range  the same ray test, stopped at a measured useful depth.
#   radius     free space within a fixed radius, straight-line, THROUGH WALLS. What every schedule
#              before 2026-09-11 used. Kept so old numbers stay reproducible, not because it is
#              right: it counts the bathroom behind a wall as covered from the hall.
COVERAGE_MODELS = ("los", "los_range", "radius")


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


def connect_through_free(ridge, free, seed, max_len_px):
    """Join one seed pixel to the ridge along a path through free space. -> True if joined.

    GA-471. A STRAIGHT BRIDGE IS NOT ENOUGH FOR A DOORWAY. connect_ridge_components refuses a
    segment that leaves the walkable surface, which is right, but the line from a 0.5 m alcove to
    the corridor ridge clips the door frame and is refused -- so the alcove was dropped and its
    ground-truth room read as unreachable while the agent could plainly walk there. MEASURED on
    hm3d_00861 regions 0 and 21: 0.5 and 1.1 m2 of navigable floor, both inside the main free
    component, both discarded.

    A breadth-first walk over the free mask finds the shortest pixel path to any ridge pixel, so it
    succeeds whenever the free space connects at all -- which is the honest test of whether the
    robot can get there. The path is drawn into the ridge and becomes roadmap.
    """
    from collections import deque
    h, w = free.shape
    sy, sx = seed
    if not free[sy, sx]:
        return False
    prev = {(sy, sx): None}
    q = deque([(sy, sx, 0)])
    while q:
        y, x, d = q.popleft()
        if ridge[y, x] and (y, x) != (sy, sx):
            while (y, x) is not None:
                ridge[y, x] = True
                nxt = prev[(y, x)]
                if nxt is None:
                    break
                y, x = nxt
            return True
        if d >= max_len_px:
            continue
        for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            ny, nx = y + dy, x + dx
            if 0 <= ny < h and 0 <= nx < w and free[ny, nx] and (ny, nx) not in prev:
                prev[(ny, nx)] = (y, x)
                q.append((ny, nx, d + 1))
    return False


def connect_ridge_components(ridge, free, max_gap_px):
    """Join the ridge's separate pieces through free space. -> (ridge, bridges drawn).

    GA-469 (owner 2026-09-10). largest_component threw away every piece of ridge that did not touch
    the biggest one, and with it every room whose corridor pinches shut in the raster -- 6 of 24
    ground-truth rooms on hm3d_00861 had no stop, and 00337's middle storey covered 65% of its own
    navigable area. The pieces are not unreachable: they are separated by a doorway narrower than
    twice the robot's clearance, which the ridge cannot run through even though the agent can.

    Each piece is bridged to the main one by the SHORTEST straight segment that stays inside free
    space. The distance transform from the main component gives, for every pixel, the distance and
    the index of the nearest main pixel, so the bridge point is the piece's minimum of that -- exact
    and O(N), not a nearest-pair search. A segment that leaves free space, or is longer than
    max_gap_px, is refused: two rooms that only look close on the raster are not joined.
    """
    lab, n = ndimage.label(ridge, structure=np.ones((3, 3)))
    if n <= 1:
        return ridge, 0
    sizes = ndimage.sum(ridge, lab, range(1, n + 1))
    main = int(np.argmax(sizes)) + 1
    out = ridge.copy()
    bridges = 0
    for _round in range(n):
        lab, n2 = ndimage.label(out, structure=np.ones((3, 3)))
        if n2 <= 1:
            break
        sizes = ndimage.sum(out, lab, range(1, n2 + 1))
        main = int(np.argmax(sizes)) + 1
        dist, ind = ndimage.distance_transform_edt(lab != main, return_indices=True)
        best = None
        for comp in range(1, n2 + 1):
            if comp == main:
                continue
            m = lab == comp
            # distance FROM the main component, evaluated on this piece
            d2, ind2 = ndimage.distance_transform_edt(lab != main, return_indices=True)
            cand = np.where(m)
            if not len(cand[0]):
                continue
            k = int(np.argmin(d2[cand]))
            py, px_ = int(cand[0][k]), int(cand[1][k])
            gap = float(d2[py, px_])
            if best is None or gap < best[0]:
                best = (gap, (py, px_), (int(ind2[0][py, px_]), int(ind2[1][py, px_])))
        if best is None or best[0] > max_gap_px:
            break
        gap, a_, b_ = best
        pts = _segment(a_, b_)
        if not all(free[y, x] for y, x in pts):
            # The straight line leaves the walkable surface: refuse rather than draw a bridge the
            # robot cannot walk. The piece stays separate and its rooms are reported uncovered.
            out[a_] = True
            break
        for y, x in pts:
            out[y, x] = True
        bridges += 1
    return out, bridges


def _segment(a, b):
    """Integer points along a to b, inclusive. Bresenham without the branches."""
    (y0, x0), (y1, x1) = a, b
    n = max(abs(y1 - y0), abs(x1 - x0)) or 1
    return [(int(round(y0 + (y1 - y0) * i / n)), int(round(x0 + (x1 - x0) * i / n)))
            for i in range(n + 1)]


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


def visible_from(free, sy, sx, max_px=None, n_rays=720):
    """-> bool grid of the cells a sensor at (sy, sx) can SEE.

    Ray marching, vectorised: every ray advances one step per iteration, all of them at once, and a
    ray stops the first time it leaves free space. That is what makes this affordable -- the naive
    loop is 720 rays x 800 steps per stop in Python, and there are thousands of stops.

    WHY LINE OF SIGHT AND NOT A RADIUS. The old measure was a distance transform over free space,
    which is straight-line distance ignoring walls, so a stop in the hall counted the bathroom
    behind it as covered. A schedule could read 100% while never entering a room. The ray test
    cannot say that: a wall between the stop and the cell ends the ray.
    """
    h, w = free.shape
    if max_px is None:
        max_px = float(math.hypot(h, w))
    seen = np.zeros_like(free, dtype=bool)
    if not free[sy, sx]:
        return seen
    seen[sy, sx] = True
    ang = np.linspace(0.0, 2.0 * math.pi, int(n_rays), endpoint=False)
    dy, dx = np.sin(ang), np.cos(ang)
    y = np.full(ang.shape, float(sy))
    x = np.full(ang.shape, float(sx))
    alive = np.ones(ang.shape, dtype=bool)
    for _ in range(int(max_px)):
        y[alive] += dy[alive]
        x[alive] += dx[alive]
        iy = np.rint(y).astype(np.int32)
        ix = np.rint(x).astype(np.int32)
        inside = alive & (iy >= 0) & (iy < h) & (ix >= 0) & (ix < w)
        alive &= inside
        if not alive.any():
            break
        yy, xx = iy[alive], ix[alive]
        open_ = free[yy, xx]
        seen[yy[open_], xx[open_]] = True
        # A ray dies ON the obstacle, not before it: the wall itself is visible, the room behind
        # it is not.
        idx = np.flatnonzero(alive)
        alive[idx[~open_]] = False
    return seen


def covered_mask(free, stops_px, model, radius_px, range_px, n_rays=720):
    """-> bool grid of the free cells some stop covers, under the named model."""
    if model not in COVERAGE_MODELS:
        raise ValueError(f"coverage model {model!r} is not one of {COVERAGE_MODELS}")
    if model == "radius":
        mark = np.zeros_like(free, dtype=bool)
        for sy, sx in stops_px:
            if 0 <= sy < free.shape[0] and 0 <= sx < free.shape[1]:
                mark[sy, sx] = True
        return free & (ndimage.distance_transform_edt(~mark) <= radius_px)
    cap = range_px if model == "los_range" else None
    out = np.zeros_like(free, dtype=bool)
    for sy, sx in stops_px:
        if 0 <= sy < free.shape[0] and 0 <= sx < free.shape[1]:
            out |= visible_from(free, sy, sx, cap, n_rays)
    return out & free


def top_up_stops(free, stops_px, model, radius_px, range_px, target=1.0,
                 candidates=240, max_add=60, n_rays=180, rng=None):
    """-> (extra stops, coverage before, coverage after). Add stops until coverage reaches target.

    GREEDY SET COVER. Each round samples free cells that are not yet covered, measures what each
    would see, and keeps the best one. Greedy is within ln(n) of optimal for set cover and there is
    no cheaper guarantee; more to the point, it stops when the target is met rather than adding a
    fixed number.

    IT SAMPLES RATHER THAN SCORING EVERY CELL. A storey has tens of thousands of free cells and
    each score is a full visibility pass; 240 candidates drawn from the uncovered region find a
    good stop without pricing all of them. The candidates come from the UNCOVERED cells, so every
    one of them is worth something.
    """
    rng = rng or np.random.default_rng(7)
    stops = list(stops_px)
    total = int(free.sum())
    if total == 0:
        return [], 1.0, 1.0
    cov = covered_mask(free, stops, model, radius_px, range_px, n_rays=720)
    before = float(cov.sum()) / total
    added = []
    while float(cov.sum()) / total < target and len(added) < max_add:
        gap = free & ~cov
        ys, xs = np.nonzero(gap)
        if ys.size == 0:
            break
        pick = rng.choice(ys.size, size=min(int(candidates), ys.size), replace=False)
        best, best_gain, best_seen = None, 0, None
        for i in pick:
            sy, sx = int(ys[i]), int(xs[i])
            seen = (visible_from(free, sy, sx, range_px if model == "los_range" else None, n_rays)
                    if model != "radius" else
                    covered_mask(free, [(sy, sx)], "radius", radius_px, range_px))
            gain = int((seen & gap).sum())
            if gain > best_gain:
                best, best_gain, best_seen = (sy, sx), gain, seen
        if best is None or best_gain == 0:
            break
        stops.append(best)
        added.append(best)
        cov |= best_seen
    return added, round(before, 4), round(float(cov.sum()) / total, 4)


def shortcut(walk_xz, clear_world, step_m, rounds=3):
    """-> a shorter, straighter version of the path, every shortcut checked against free space.

    CORNER TURNING COSTS AS MUCH AS DRIVING: 1.30 million degrees a lap over the 209 storeys, at
    27.3% of the frames. Total turning along a polyline is a property of its SHAPE, not of how
    finely it is sampled, so nothing is saved by re-spacing the points -- the zig-zags themselves
    have to go. This drops any middle point whose two neighbours can see each other through free
    space, which cuts distance and turning together.
    """
    pts = list(walk_xz)
    for _ in range(int(rounds)):
        out, i, dropped = [pts[0]], 1, False
        while i < len(pts) - 1:
            a, c = out[-1], pts[i + 1]
            n = max(2, int(math.dist(a, c) / (step_m / 2)))
            if all(clear_world(a[0] + (c[0] - a[0]) * t / n, a[1] + (c[1] - a[1]) * t / n)
                   for t in range(n + 1)):
                i += 1          # the middle point is redundant, skip it
                dropped = True
            else:
                out.append(pts[i])
                i += 1
        out.append(pts[-1])
        pts = out
        if not dropped:
            break
    return pts


def two_opt(order, dist, rounds=40):
    """-> a shorter visiting order. Nearest neighbour, then 2-opt until it stops improving.

    The DFS order drives every backtrack: it leaves a branch the way it came. 2-opt reverses a
    segment whenever that shortens the tour, which removes the crossings a depth-first walk leaves
    behind. `dist` is the roadmap distance, not the straight line, so a shortcut through a wall is
    never proposed.

    THE ROOT STAYS FIRST. It is where the run starts, and a schedule that begins somewhere else
    would need the agent teleported there.
    """
    if len(order) < 4:
        return list(order)
    root, rest = order[0], list(order[1:])
    tour, pool = [root], set(rest)
    while pool:
        nxt = min(pool, key=lambda v: dist(tour[-1], v))
        tour.append(nxt)
        pool.discard(nxt)
    for _ in range(int(rounds)):
        improved = False
        for i in range(1, len(tour) - 2):
            for k in range(i + 1, len(tour) - 1):
                a, b, c, d = tour[i - 1], tour[i], tour[k], tour[k + 1]
                if dist(a, b) + dist(c, d) > dist(a, c) + dist(b, d) + 1e-9:
                    tour[i:k + 1] = reversed(tour[i:k + 1])
                    improved = True
        if not improved:
            break
    return tour


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
                     clear_world=None, scan_for=None, smooth=True):
    """-> (trajectory, budget). The path the robot drives, and what it costs in frames.

    A STOP IS SCANNED ONCE, ON FIRST ARRIVAL. Depth-first search comes back through a waypoint every
    time it backtracks out of a branch; scanning again would spend 36 frames looking at a place the
    run has already seen. `order` is where it scans, `walk` is what it drives.
    """
    # scan_for(waypoint) -> the degrees to turn at that stop, or None to use scan_deg everywhere.
    # A full circle is what a stop in a room needs; a stop in a corridor whose walls the run has
    # already seen needs less, and the frames saved are real. The caller decides, because only it
    # knows what is already covered.
    def _scan(w):
        return float(scan_deg if scan_for is None else scan_for(w))

    first_visit = {w: i for i, w in enumerate(order)}
    unsafe = [0]
    traj, seen = [], set()
    start = walk[0]
    traj.append({"xyz": to_world(start), "scan_deg": _scan(start), "stop": 0, "leg": None})
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
        # STRAIGHTEN WHAT IS LEFT. RDP only drops a point that is close to the chord between its
        # neighbours; it keeps a wide zig-zag whose corners are far from that chord, and those
        # corners are where the turning bill is. The shortcut pass drops any point its neighbours
        # can see past, so distance and turning fall together. Every shortcut is checked against
        # free space, exactly as the RDP result above is.
        if smooth and clear_world is not None and len(keep) > 2:
            keep = shortcut(keep, clear_world, step_m)
        for k in range(1, len(keep)):
            drive_m += math.dist(keep[k - 1], keep[k])
        for k, (x, z) in enumerate(keep[1:], start=1):
            last = (k == len(keep) - 1)
            entry = {"xyz": [round(x, 3), pts[0][1], round(z, 3)], "scan_deg": 0,
                     "stop": None, "leg": leg}
            if last and v not in seen:
                entry["scan_deg"] = _scan(v)
                entry["stop"] = first_visit[v]
                seen.add(v)
            traj.append(entry)

    # Turning to face each segment is a real cost, not a rounding error: at 20 degrees per action a
    # right-angle corner is 5 frames, and the corners across one lap of 209 storeys came to 1.30
    # million degrees before this pass existed.
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
        "scan_deg_total": int(round(sum(_scan(w) for w in order))),
        "scan_frames": sum(max(1, int(round(_scan(w) / turn_deg))) for w in order),
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
