"""A stylized 3D view of a run: the object boxes the pipeline built, over the floor it mapped.

WHAT IT SHOWS, and why each layer is separable. Three things are drawn in one world frame:

  the FLOOR      the run's own occupancy map, laid flat as the ground plane
  the BOXES      the 269 oriented boxes in persistent_perception.json -- what the belief
  the WALLS      ground truth, when a GT file exists for the scene -- what is actually there

Keeping belief and truth as separate toggles is the point. A view that draws them in one
colour lets a wall the pipeline never saw pass for one it mapped, which is exactly the
confusion this project keeps having to unpick.

  the MESH        the scene's own Matterport GLB, textured -- what the room actually looks like

A WALL IS A SURFACE, NOT A HEAP OF BOXES. `merge_walls` folds the ground truth's 202 wall
SEGMENTS into 91 surfaces, and `cut_openings` cuts the doors and windows back through them,
because the merge's own bounding box closes the gaps the segments left. The page extrudes one
slab per surface with the apertures as holes in its outline -- so a doorway is a doorway.

TRANSLUCENT WALLS **AND** THE CUTAWAY, which is the owner's settled combination as of
2026-09-04. The walls were briefly made opaque on the reading that "Sims-like" meant solid;
the transparency at 0.28 was the version that was wanted, and the cutaway is wanted on top of
it. They are not redundant: transparency lets you see THROUGH a near wall, the cutaway TAKES
IT AWAY, and a translucent wall still tints and clutters everything behind it. Together you
get an open-plan view with the far walls still drawn.

BELIEF AND TRUTH STAY APART, and the opacity was never what carried that. Ground truth is
FLAT UNTEXTURED GREY with a pale outline; the pipeline's own boxes are SATURATED per-label
hues, translucent, with a bright edge and a label. Grey solid against coloured translucent
separates at a glance, which is the point of the view: a wall the pipeline never saw must not
pass for one it mapped.

ONE RUN MAPS ONE STOREY, and the page says which. hm3d_00861 has four navmesh height
clusters and only two are storeys; a run is held to one of them by `stats.floor_guard`. The
storey picker filters the walls, openings and boxes by their own z extent and CLIPS the mesh
to the storey band, and the bar under it names the mapped storey with the guard's evidence --
because a viewer who picks the other storey sees an empty shell, and the honest reading of an
empty shell, without that line, is "the pipeline found nothing".

THREE.JS, SERVED FROM DISK. The page was a 2D-canvas axonometric projection with a painter's
sort until 2026-09-04. A 2D canvas cannot texture-map 408,722 triangles, so the drawing surface
is now WebGL. NOTHING is fetched from a CDN: r128 is vendored under found/dashboard/vendor_three
and served by replay_server at /vendor, which is the same no-network rule the rest of this
dashboard follows.

THE WORLD FRAME IS THE MAP FRAME, z up (GA-255). The object boxes, the walls, the openings, the
agent poses and the floor bounds are already in it and are NOT swizzled anywhere. The only thing
transformed is the GLB, by a rotation of -90 degrees about z; `page` asserts the rotated mesh's
Box3 against the ground truth's own AABB and paints the status line red when it disagrees.
"""
import importlib.util
import json
import re
from pathlib import Path

try:
    from found.dashboard import dash_env
except ImportError:
    import dash_env

RUNS_DIR = dash_env.runs_dir()
GT_DIR = dash_env.gt_dir()

# Labels in the ground truth that describe the SHELL of the building rather than its
# contents. Read from the GT's own vocabulary rather than guessed: this scene uses "wall",
# "recessed wall", "compound wall", "wall panel" and "shower wall", and a substring test on
# "wall" alone would also be right here but would silently take "wall clock" in another scene.
WALL_LABELS = {"wall", "recessed wall", "compound wall", "wall panel", "shower wall"}
OPENING_LABELS = {"door", "door frame", "window", "window frame", "shower door frame"}


def merge_walls(walls, plane_tol=0.25, gap_tol=0.12):
    """Collapse the ground truth's wall SEGMENTS into the surfaces they belong to.

    HM3D annotates one wall as many short axis-aligned boxes -- 202 of them in hm3d_00861 --
    and drawn one at a time they read as a heap of sections rather than a building. Two
    segments are one surface when they share a thin axis and their boxes touch on all three
    axes: within `plane_tol` across the wall's thickness, within `gap_tol` along its length
    and its height. Measured on hm3d_00861, the pieces of one wall are a full-height slab, a
    skirting strip 5 cm tall and a band over a window, at plane offsets up to 20 cm apart; a
    key on a rounded height band kept every one of them separate (202 -> 190).

    THIS IS THE DRAWING'S GEOMETRY, NOT A DATA CHANGE. `walls` still carries every segment with
    its `region`, because the room segmentation is wanted later; this adds a second view of the
    same boxes. Nothing that reads `walls` sees a difference.

    CEILINGS, stated:
      * AXIS-ALIGNED ONLY. A wall at 30 degrees has no thin axis in x or y, so it stays its own
        surface. Correct, merely not merged.
      * TRANSITIVE. A stair of segments each within tolerance of the next joins into one slab
        as thick as the whole stair. That draws a wall thicker than any one annotation, never
        a wall that is not there.
    """
    segs = []
    for w in walls:
        ex, ey, ez = (list(w["ext"]) + [0, 0, 0])[:3]
        x, y, z = (list(w["pos"]) + [0, 0, 0])[:3]
        axis = 0 if abs(ex) <= abs(ey) else 1          # the THIN axis; the wall runs along 1-axis
        lo = [x - abs(ex) / 2, y - abs(ey) / 2, z - abs(ez) / 2]
        hi = [x + abs(ex) / 2, y + abs(ey) / 2, z + abs(ez) / 2]
        segs.append((axis, lo, hi, w))

    # ponytail: O(n^2) pair scan with union-find; 202 segments is 20k checks. A sweep if a
    # scene ever has thousands.
    parent = list(range(len(segs)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def touch(a, b, k, tol):
        return a[1][k] <= b[2][k] + tol and b[1][k] <= a[2][k] + tol

    for i, a in enumerate(segs):
        for j in range(i):
            b = segs[j]
            if a[0] != b[0]:
                continue
            if (touch(a, b, a[0], plane_tol) and touch(a, b, 1 - a[0], gap_tol)
                    and touch(a, b, 2, gap_tol)):
                parent[find(i)] = find(j)

    groups = {}
    for i, sg in enumerate(segs):
        groups.setdefault(find(i), []).append(sg)
    out = []
    for items in groups.values():
        out.append({"lo": [min(sg[1][k] for sg in items) for k in range(3)],
                    "hi": [max(sg[2][k] for sg in items) for k in range(3)],
                    "n": len(items),
                    "regions": sorted({sg[3].get("region") for sg in items} - {None})})
    return out


def _union_rects(rects, tol):
    """Fold overlapping or nearly touching (u, v) rectangles into their bounding boxes.

    THIS IS NOT COSMETIC. A door and its own door frame are two ground-truth instances over
    one aperture, and `THREE.ExtrudeGeometry` triangulates a shape whose holes OVERLAP into
    garbage. Two holes go in, one goes out.

    ponytail: bounding box, not a true polygon union, and a repeat-until-stable O(n^3) scan.
    A merged pair is drawn as the box that contains both, so an L of two apertures would open
    slightly wider than annotated. Measured on hm3d_00861: at most 6 rectangles reach one
    surface, so the scan is nothing. A rectangle-decomposition union if a scene ever puts two
    unrelated apertures in one bounding box.
    """
    rects = [list(r) for r in rects]
    changed = True
    while changed:
        changed = False
        for i in range(len(rects)):
            for j in range(i + 1, len(rects)):
                a, b = rects[i], rects[j]
                if (a[0] <= b[2] + tol and b[0] <= a[2] + tol
                        and a[1] <= b[3] + tol and b[1] <= a[3] + tol):
                    rects[i] = [min(a[0], b[0]), min(a[1], b[1]),
                                max(a[2], b[2]), max(a[3], b[3])]
                    rects.pop(j)
                    changed = True
                    break
            if changed:
                break
    return rects


def _plane_axes(axis):
    """The surface's two IN-PLANE axes, ascending -- (1,2), (0,2) or (0,1).

    For a wall (axis 0 or 1) this is exactly the old `u = 1 - axis` with z vertical, so
    the page's rectangles keep their meaning. For a slab (axis 2) it is (x, y), which the
    old convention had no way to name. One definition, used by the cut, the page and the
    real-scene assertion, so the three cannot drift apart.
    """
    return tuple(k for k in (0, 1, 2) if k != axis)


def cut_openings(surfaces, openings, lip=0.01, join_tol=0.02):
    """Assign every door and window to the merged wall surface it pierces, as a HOLE.

    `merge_walls` takes the axis-aligned box over a group of segments, and that box CLOSES
    the very gaps the segments left -- the doorway between two sections, the strip under a
    window. So the merged surface is a solid rectangle that has to have its apertures cut
    back out of it, or the building has no doors.

    Each surface gains two keys and nothing is changed or removed (working rule 6):
      `axis`   0, 1 or 2 -- the surface's thin axis, so the page does not re-derive it
      `holes`  a list of [u0, v0, u1, v1] in WORLD units, where (u, v) are the surface's
               two IN-PLANE axes, ascending: (1, 2) for axis 0, (0, 2) for axis 1, (0, 1)
               for axis 2. For a wall that is the old (length, z) pair, unchanged.
    Each opening gains `cut`: True when it became a hole, False when no surface took it. An
    opening that no wall claims is NOT dropped -- the page still draws it as a box, because
    an aperture that vanishes from the picture looks exactly like one that is not in the
    ground truth.

    CEILINGS, stated:
      * ONE SURFACE PER OPENING, the one whose face it overlaps most. A door in a doorway
        between two parallel walls is cut through the nearer one only.
      * A `lip` of 1 cm is kept between a hole and the edge of its wall. Doors reach the
        floor, so without it the hole would touch the outline and the triangulator would
        have to cut a notch instead of a hole. The lip is the width of a pencil line at any
        zoom this page is used at, and it is why a doorway reads as a doorway.
      * `axis` IS THE THINNEST OF THE THREE, z included (owner decision 2026-09-06;
        the rule was x-or-y until then). On hm3d_00861 three of the 91 surfaces are
        genuinely thinnest in z -- extents [0.24, 0.92, 0.189], [0.729, 0.43, 0.287] and
        [0.434, 0.316, 0.18] -- i.e. horizontal slabs, not walls, and the old rule gave
        each of them an axis that was not its thinnest. A slab is now its own case rather
        than a vertical pane's degenerate one: its in-plane pair is (x, y), so an aperture
        through a floor -- a stairwell, a hatch -- is expressible, which `u = 1 - axis`
        with z always vertical could not say at all.
      * WHAT THIS DOES NOT CHANGE, measured on hm3d_00861 rather than assumed: the
        extrusion spans the full lo..hi box on all three axes whichever axis is picked, so
        a hole-free surface renders identically either way (worst placed-AABB error over
        all 91 surfaces: 4.6e-7 m), and the three slabs carry no hole before or after.
        The aperture count is unchanged at 34 over 54 openings.
      * THE REMAINING CEILING IS ONE LEVEL UP, in `merge_walls`: it still groups segments
        by an x-or-y thin axis, so which segments become one surface is decided by the old
        rule even though how that surface is drawn is now decided by the new one. Two
        slabs are grouped as though they were panes. It is named, not fixed, because
        changing it changes the 91-surface count, which is a different decision.
    """
    for s in surfaces:
        ex = [s["hi"][k] - s["lo"][k] for k in range(3)]
        # THE THINNEST AXIS, z included. index(min(...)) takes the first on a tie, which
        # keeps the old x-over-y preference for a square-section wall.
        s["axis"] = ex.index(min(ex))
        s["holes"] = []
    for o in openings:
        lo = [o["pos"][k] - abs(o["ext"][k]) / 2 for k in range(3)]
        hi = [o["pos"][k] + abs(o["ext"][k]) / 2 for k in range(3)]
        best = None
        for s in surfaces:
            a = s["axis"]
            u, v = _plane_axes(a)
            if min(hi[a], s["hi"][a]) - max(lo[a], s["lo"][a]) <= 0:
                continue          # not in this surface's slab at all
            r = [max(lo[u], s["lo"][u] + lip), max(lo[v], s["lo"][v] + lip),
                 min(hi[u], s["hi"][u] - lip), min(hi[v], s["hi"][v] - lip)]
            area = (r[2] - r[0]) * (r[3] - r[1])
            if r[2] - r[0] <= 2 * lip or r[3] - r[1] <= 2 * lip:
                continue          # touches the face edge-on; there is no hole to cut
            if best is None or area > best[0]:
                best = (area, s, r)
        o["cut"] = best is not None
        if best:
            best[1]["holes"].append(best[2])
    for s in surfaces:
        s["holes"] = _union_rects(s["holes"], join_tol)
    return surfaces


def read_floors(bev):
    """-> (storeys, levels, source) for a bundle, from its OWN bev_data.json.

    THE FLOOR SET IS NOT A CONSTANT. `habitat_feed_host.py` derives it from 300 random
    navmesh samples per run: heights chained at 0.8 m, cluster centre = median, kept as a
    storey when its share reaches `habitat.min_floor_share` (0.10, and the source itself
    records that threshold as UNMEASURED). Measured over 43 run directories: 21 carry no
    bev_data.json at all, 3 carry the hard-coded [-2.5, 0.5] of GA-93, 3 carry four raw
    clusters with no `floor_detail`, and 16 carry [-1.59, 1.21] with `floor_detail`. The
    same scene therefore reports different floors in different bundles, so this reads the
    bundle and never a table.

    THE THREE ABSENT CASES RETURN AN EMPTY LIST AND A REASON, never a default. A picker
    that silently offers one floor because it found no data looks exactly like a
    single-storey building, which is the failure this scene would show first: hm3d_00861
    has two storeys and one run maps one of them.
    """
    if not isinstance(bev, dict):
        return [], [], "NOT MEASURED -- this bundle has no readable bev_data.json"
    det = bev.get("floor_detail")
    if isinstance(det, dict) and det.get("floors"):
        return ([dict(f) for f in det["floors"]],
                [dict(f) for f in (det.get("levels") or [])],
                "bev_data.json floor_detail, 300 navmesh samples; a cluster is a storey at "
                "min_floor_share " + str(det.get("min_floor_share")) + " (UNMEASURED)")
    flat = bev.get("floors")
    if isinstance(flat, list) and flat:
        if [round(float(z), 2) for z in flat] == [-2.5, 0.5]:
            return [], [], ("NOT MEASURED -- bev_data.json carries the hard-coded "
                            "[-2.5, 0.5] of GA-93, which is not a measurement of this scene")
        return ([{"z": float(z)} for z in flat], [],
                "bev_data.json `floors` list only -- no floor_detail, so the shares and the "
                "storey/level split are NOT AVAILABLE in this bundle")
    return [], [], "NOT MEASURED -- bev_data.json names no floors"


def assign_floor(z_lo, z_hi, floors, snap=0.4):
    """-> the storey indices an item stands on. An item can be on two.

    THE RULE IS THE ITEM'S OWN BOTTOM, snapped to the highest floor at or below
    `z_lo + snap`. `pos[2]` is already the map frame with z vertical (GA-255), so index 2
    IS the height and nothing is swizzled. Measured on data/gt/hm3d_00861.json against
    floors [-1.59, 1.21]: 202 wall segments split 107 / 95 and 54 openings split 27 / 27,
    and a per-REGION rule agrees on 200 of 202 walls and 54 of 54 openings. Region is
    therefore shown as a label and is NOT the authority: region ids carry no level prefix
    and two regions are genuinely mixed (`_2` at 0.75 purity, `_11`, the stairwell, at
    0.63).

    SPANNING ITEMS ARE REAL and are returned on EVERY storey they cross -- 7 instances in
    this scene reach from below -1.0 to above 1.21: 4 walls, a window, a window frame and
    a stairs railing. Dropping one leaves a hole in a wall that has no opening.

    ponytail: floors are treated as an ordered list of heights and an item as its z
    interval. No navmesh footprint is consulted, so a mezzanine over a double-height room
    is assigned by height alone. Ceiling named; the upgrade path is a per-floor navmesh
    polygon test, and `bev_data.json` carries the 300 sample points to build one from.
    """
    if not floors:
        return []
    zs = [f["z"] for f in floors]
    below = [i for i, z in enumerate(zs) if z <= z_lo + snap]
    out = {max(below) if below else 0}
    out.update(i for i, z in enumerate(zs) if z_lo < z < z_hi)
    return sorted(out)


def _floor_deck(maps, z):
    """-> the rendered top-down PNG bev_data.json holds FOR THAT STOREY, or None.

    `maps` is keyed by the floor height AS A STRING, so the key is matched numerically to
    within 1 cm rather than by formatting -- "-1.59" and "-1.590" are the same storey and
    a string compare would call one of them missing.
    """
    if not isinstance(maps, dict):
        return None
    best = None
    for k, v in maps.items():
        try:
            d = abs(float(k) - z)
        except (TypeError, ValueError):
            continue
        if d < 0.01 and (best is None or d < best[0]) and isinstance(v, dict):
            best = (d, v)
    if not best:
        return None
    v = best[1]
    if not (v.get("image") and v.get("bounds_min") and v.get("bounds_max")):
        return None
    return {"image": v["image"], "x0": v["bounds_min"][0], "y0": v["bounds_min"][2],
            "x1": v["bounds_max"][0], "y1": v["bounds_max"][2], "z": v["bounds_min"][1]}


def _viewer_kg_style():
    """The dashboard's OWN cytoscape stylesheet, read from viewer.html.

    The owner's report: the knowledge graph on this page should look like the one on the main
    dashboard. It did not -- viewer.html styles nodes at 26 px with a labelled background, an
    arrowed bezier edge and a colour per relation, while this page had a 12 px node, a hairline
    haystack edge and no relation colours at all. Two stylesheets, one of them a poor relation.

    READ, NOT COPIED. A copy is what produced the divergence in the first place: ~130 lines of
    selectors that have to be edited in two files to stay one design, and nothing to say when
    they stop matching. viewer.html stays the single source and this function lifts the literal
    out of it, so a change there reaches this page with no second edit.

    NOT A SHARED ROUTE, deliberately. `/vendor` is mounted by the dashboard, but viewer.html is
    ALSO served by the bridge on :8081 in live mode, where a new dashboard route would not
    exist -- the graph would then break in live mode to fix its looks in replay.

    Returns None when the literal cannot be found, and the caller SAYS so on the page rather
    than quietly falling back: a graph that silently reverts to the old look is the same
    divergence again, just harder to notice.
    """
    v = _viewer_html_path()
    if v is None:
        return None
    try:
        h = v.read_text()
    except OSError:
        return None
    i = h.find("cytoscape({")
    if i == -1:
        return None
    j = h.find("style: [", i)
    if j == -1:
        return None
    # LINE BY LINE, WITH `//` COMMENTS STRIPPED, not a character scanner. The first version
    # balanced brackets over the raw text and ran 43 KB past the end of the array, stopping
    # inside the regex literal `/[&<>"']/` further down the file -- a bracket in a regex is not
    # a bracket in the data. The style array is pure data with line comments, so stripping the
    # comment and counting brackets per line terminates exactly where the array does.
    lines = h[j:].splitlines(keepends=True)
    out, depth = [], 0
    for ln in lines:
        code = ln.split("//", 1)[0] if "//" in ln and not re.search(r"https?://", ln) else ln
        out.append(ln)
        depth += code.count("[") - code.count("]")
        if depth == 0 and len(out) > 1:
            body = "".join(out)
            k = body.index("[")
            return body[k:body.rindex("]") + 1]
    return None


def _viewer_html_path():
    """Where viewer.html is, in either tree layout. Same search as the dashboard's own."""
    here = Path(__file__).resolve()
    for base in list(here.parents)[:6]:
        for rel in ("lost3dsg/src/perception_module/viewer/viewer.html",
                    "src/perception_module/viewer/viewer.html"):
            cand = base / rel
            if cand.is_file():
                return cand
    return None


def scene_payload(bundle, gt_scene=None):
    """-> everything the page draws, in one world frame.

    `measured` and `truth` are kept in separate keys all the way to the browser so the page
    cannot accidentally render one as the other.
    """
    d = RUNS_DIR / bundle
    out = {"bundle": bundle, "objects": [], "walls": [], "openings": [],
           "agent": None, "floor": None, "gt_scene": None, "gt_regions": 0,
           "gt_aabb": None,
           # SEPARATE KEYS, for the reason the docstring above already gives about measured
           # vs truth: `walls` is GROUND TRUTH from the scene file, `detected_walls` is what
           # the wall detector measured, and `schedule` is what the run was told to walk. Three
           # different claims; merging any two would let the page render one as another.
           "detected_walls": [], "schedule": None, "ceiling_z": None}
    _gt_items = []

    # The run's own room segmentation and the plan it followed. Both are optional: every
    # bundle recorded so far has an empty `detected_walls` because these runs launch with
    # `use_wall_detector:=false` (measured: 0 of 126), and `schedule` is null unless a
    # schedule drove the run.
    try:
        _room = json.loads((d / "room.json").read_text())
        if isinstance(_room, dict):
            out["detected_walls"] = [w for w in (_room.get("detected_walls") or [])
                                     if isinstance(w, (list, dict))]
    except (OSError, ValueError):
        pass
    try:
        _bev = json.loads((d / "bev_data.json").read_text())
        if isinstance(_bev, dict) and isinstance(_bev.get("schedule"), dict):
            out["schedule"] = _bev["schedule"]
    except (OSError, ValueError):
        pass

    pp = d / "persistent_perception.json"
    if pp.is_file():
        try:
            data = json.loads(pp.read_text())
        except (OSError, ValueError):
            data = []
        objs = data if isinstance(data, list) else data.get("objects", [])
        for o in objs:
            b = o.get("bbox") or {}
            if not all(k in b for k in ("x_min", "x_max", "y_min", "y_max", "z_min", "z_max")):
                continue
            out["objects"].append({
                "label": o.get("label"),
                "id": o.get("object_id"),
                "room": o.get("room_id"),
                "colour": o.get("color"),
                "box": [b["x_min"], b["y_min"], b["z_min"], b["x_max"], b["y_max"], b["z_max"]],
            })

    # GA-259. The run's own occupancy map, as the ground plane. The boxes floated over
    # nothing before; the map is the thing that says WHERE they are.
    bev = d / "bev_data.json"
    bevdoc = None
    if bev.is_file():
        try:
            b = json.loads(bev.read_text())
            bevdoc = b
            m = b.get("map") or {}
            if m.get("image") and m.get("bounds_min") and m.get("bounds_max"):
                # bounds are (x, floor_height, y) in the same map frame as everything else.
                out["floor"] = {"image": m["image"],
                                "x0": m["bounds_min"][0], "y0": m["bounds_min"][2],
                                "x1": m["bounds_max"][0], "y1": m["bounds_max"][2],
                                "z": m["bounds_min"][1]}
        except (OSError, ValueError, IndexError, TypeError):
            pass

    poses = d / "agent_poses.json"
    if poses.is_file():
        try:
            p = json.loads(poses.read_text())
            p = p if isinstance(p, list) else p.get("poses", [])
            if p:
                out["agent"] = {"path": [[q.get("x"), q.get("y"), q.get("z")] for q in p
                                         if q.get("x") is not None]}
        except (OSError, ValueError):
            pass

    # Ground truth is OPTIONAL and its absence is reported, not defaulted: a page that draws
    # no walls because it found no file looks identical to one drawn for a scene with none.
    scene = gt_scene or _scene_of(d)
    if scene:
        gt = GT_DIR / f"{scene}.json"
        if gt.is_file():
            try:
                g = json.loads(gt.read_text())
            except (OSError, ValueError):
                g = []
            items = g if isinstance(g, list) else g.get("objects", g.get("instances", []))
            _gt_items = items
            regions = set()
            lo = [float("inf")] * 3
            hi = [float("-inf")] * 3
            for it in items:
                lab = str(it.get("label") or "").strip().lower()
                pos, ext = it.get("pos"), it.get("extents")
                regions.add(it.get("region"))
                if not (pos and ext and len(pos) >= 3 and len(ext) >= 3):
                    continue
                # THE MESH TRANSFORM'S ONLY RUNNABLE CHECK, and it has to be built here
                # because the page cannot compute it. The axis-aligned box over EVERY ground
                # truth instance is the same box as the scene mesh's own vertex AABB -- both
                # come from the same simulator, in the same frame. Measured on hm3d_00861
                # over all 870 instances: min (-9.4556, -2.3623, -1.7929), max (1.3000,
                # 11.7858, 4.8024), which is the GLB's POSITION AABB under map = (y, -x, z)
                # to four decimals. Every wrong rotation misses the first number by metres.
                for k in range(3):
                    lo[k] = min(lo[k], pos[k] - abs(ext[k]) / 2)
                    hi[k] = max(hi[k], pos[k] + abs(ext[k]) / 2)
                rec = {"label": lab, "pos": pos[:3], "ext": ext[:3], "region": it.get("region")}
                if lab in WALL_LABELS:
                    out["walls"].append(rec)
                elif lab in OPENING_LABELS:
                    out["openings"].append(rec)
            out["gt_scene"] = scene
            out["gt_regions"] = len([r for r in regions if r not in (None, "_-1")])
            if lo[0] != float("inf"):
                out["gt_aabb"] = {"lo": lo, "hi": hi, "n": len(items)}
    out["wall_mass"] = cut_openings(merge_walls(out["walls"]), out["openings"])
    out["openings_cut"] = sum(1 for o in out["openings"] if o["cut"])

    # ---- WHICH STOREY EACH THING IS ON, and which one this run actually mapped -------
    # Keys only; nothing above is changed or removed (working rule 6). A reader that does
    # not look for `floor` sees the same payload it always did.
    floors, levels, src = read_floors(bevdoc)
    out["floors"] = floors
    out["levels"] = levels
    out["floor_source"] = src
    out["floor_decks"] = [_floor_deck((bevdoc or {}).get("maps"), f["z"]) for f in floors]
    # `stats.floor_guard.floor_y` is the floor the run WAS HELD TO, with the evidence
    # beside it. `run_metadata.feed.spawn_floor_requested` is what was ASKED FOR and its
    # own note says so -- it is intent, not outcome, and is deliberately not read here.
    guard = ((bevdoc or {}).get("stats") or {}).get("floor_guard")
    out["mapped_floor"] = dict(guard) if isinstance(guard, dict) and "floor_y" in guard else None
    # COMPUTED HERE, AFTER mapped_floor EXISTS. The first version sat above, beside the
    # wall merge, where `out["mapped_floor"]` and `out["floors"]` were both still unset --
    # so every fallback found nothing and ceiling_z was None on every bundle, which reads
    # exactly like a scene with no annotated ceiling.
    # THE CEILING OVER THE STOREY THIS RUN MAPPED, so the dollhouse cut has a height that
    # comes from the building rather than from a guess. HM3D annotates ceilings (24 instances
    # in hm3d_00861), and they sit at TWO heights here: z 0.70-1.05 is the lower storey's
    # ceiling, under the upper floor at 1.21, and the 12 above it start at 3.32. Taking the
    # LOWEST ceiling that begins above the mapped floor picks the right one of those without
    # knowing how many storeys there are.
    #
    # None when the scene annotates no ceiling above the storey -- the page then leaves the
    # roof alone rather than cutting at an invented height.
    _mf = (out.get("mapped_floor") or {}).get("floor_y")
    if _mf is None and out.get("floors"):
        _mf = out["floors"][-1].get("z")
    if _mf is not None:
        _tops = []
        for it in _gt_items:
            if str(it.get("label") or "").strip().lower() != "ceiling":
                continue
            pos, ext = it.get("pos"), it.get("extents")
            if not pos or not ext:
                continue
            lo_z = float(pos[2]) - abs(float(ext[2])) / 2
            if lo_z > float(_mf) + 0.3:      # above the floor, not its own slab
                _tops.append(lo_z)
        out["ceiling_z"] = min(_tops) if _tops else None
    for o in out["objects"]:
        o["floor"] = assign_floor(o["box"][2], o["box"][5], floors)
    for rec in out["walls"] + out["openings"]:
        z, e = rec["pos"][2], abs(rec["ext"][2]) / 2
        rec["floor"] = assign_floor(z - e, z + e, floors)
    for w in out["wall_mass"]:
        # THE SURFACE'S OWN z EXTENT, not its `regions` list. Measured: 4 of the 91 merged
        # surfaces span both storeys, and 2 of the 28 multi-region surfaces draw segments
        # from regions on different floors -- so a region-keyed filter would put a wall on
        # the wrong storey twice over.
        w["floor"] = assign_floor(w["lo"][2], w["hi"][2], floors)
    out["floor_counts"] = [{"objects": sum(1 for o in out["objects"] if i in o["floor"]),
                            "walls": sum(1 for w in out["walls"] if i in w["floor"]),
                            "surfaces": sum(1 for w in out["wall_mass"] if i in w["floor"]),
                            "openings": sum(1 for o in out["openings"] if i in o["floor"])}
                           for i in range(len(floors))]
    return out


def _scene_of(d):
    """The scene id this bundle ran against, from its own metadata. Never from the name."""
    md = d / "run_metadata.json"
    if not md.is_file():
        return None
    try:
        return json.loads(md.read_text()).get("scene")
    except (OSError, ValueError):
        return None


_CSS = """
  :root {
    --bg:#080d18; --card:#0e1626; --rule:#1e2b44; --ink:#e2e8f0; --dim:#8296b4;
    --faint:#4a5c78; --accent:#38bdf8;
  }
  body { background:var(--bg); color:var(--ink); margin:0;
         font:13px/1.55 ui-monospace,SFMono-Regular,Menlo,monospace; }
  header { padding:12px 18px; border-bottom:1px solid var(--rule);
            display:flex; align-items:baseline; gap:14px; flex-wrap:wrap; }
  h1 { font-size:14px; margin:0; letter-spacing:.06em; text-transform:uppercase; }
  .sub { color:var(--dim); font-size:11px; }
  .bar { display:flex; gap:6px; align-items:center; padding:9px 18px; flex-wrap:wrap;
          border-bottom:1px solid var(--rule); }
  button { background:var(--card); color:var(--ink); border:1px solid var(--rule);
            border-radius:5px; padding:5px 11px; cursor:pointer;
            font:600 11px ui-monospace,monospace; }
  button:hover { border-color:var(--accent); color:var(--accent); }
  button.on { background:rgba(56,189,248,.16); border-color:var(--accent); color:var(--accent); }
  /* Scene left, knowledge graph top right, table bottom right. All three visible at
     once, so a selection can be SEEN travelling between them -- which is the point of
     linking them at all. */
  #split { display:flex; height:calc(100vh - 122px); }
  /* GA-397. EMBEDDED: this page is a panel in the dashboard, not a page of its own. The header,
     the button bars, the storey paragraph and the two status lines are ~200 px of chrome that made
     the 3D view scroll inside its quadrant instead of filling it. In embedded mode they are moved
     into HUD overlays over the canvas -- MOVED, not duplicated, so every existing handler and id
     keeps working and there is one control per function, not two. */
  body.embedded > header, body.embedded > .bar,
  body.embedded > #meshStat, body.embedded > #liveStat { display:none; }
  body.embedded #split { height:100vh; }
  /* THE EMBEDDED PANEL NEVER SCROLLS AT THE DOCUMENT LEVEL. It is exactly the size of the
     iframe the dashboard gives it, and anything that genuinely scrolls -- the objects table --
     carries its own scroller.
     Without this the panel had a scrollbar that APPEARED AND DISAPPEARED, which is a feedback
     loop rather than a stray element: `resize()` sizes the canvas to `wrap.clientWidth`, so a
     scrollbar taking its gutter narrows the wrap, which reflows, which clears the scrollbar,
     which widens it again. MEASURED while maximized: documentElement.scrollWidth 1245 against
     clientWidth 1240 -- five pixels, the width of a gutter, and gone on the next reading.
     overflow:hidden on the container removes the only thing the loop can toggle. Scoped to
     `.embedded` so the standalone page, which is a normal scrolling document, is untouched. */
  body.embedded { overflow:hidden; }
  /* The knowledge graph and the objects table appear ONLY when the panel is maximized: in a
     quadrant they leave the scene too little room to be worth anything. */
  body.embedded #side { display:none; }
  body.embedded.maxi #side { display:flex; }
  .hud { position:absolute; z-index:6; background:rgba(8,13,24,.86); border:1px solid var(--rule);
         border-radius:7px; backdrop-filter:blur(6px); }
  #hudLayers { left:10px; top:10px; display:flex; flex-wrap:wrap; gap:4px; padding:5px; max-width:52%; }
  #hudLayers button { padding:3px 7px; font:600 10px ui-monospace,monospace; }
  /* BOTTOM-right, not top-right: the parent page paints its MAXIMIZE button over the frame's top
     right corner, and clicks at the right end of this header were hitting it (review 2026-09-08). */
  #hudHelp { right:10px; bottom:10px; max-width:min(420px, 46%); }
  #hudHelp .head { display:flex; align-items:center; gap:6px; padding:4px 7px; cursor:pointer;
                   font:700 10px ui-monospace,monospace; color:var(--accent); letter-spacing:.05em; }
  #hudHelp .body { padding:0 9px 8px; font-size:10px; line-height:1.5; color:var(--dim); }
  #hudHelp.closed .body { display:none; }
  /* ONE CLIPPED LINE ALONG THE BOTTOM, between the two corner widgets, and CLICK-THROUGH.
     Three placements were wrong before this one and each was measured: at bottom-left it sat ON the
     storey picker and swallowed the storey-down button (the reviewer's two-storey bundle needs it,
     so the normal case); at top-right it grew to 180 px tall, because it holds two wrapping
     sentences, and collided with the layers strip. Bounded HERE by construction rather than by
     hoping the text stays short: it starts right of the storey picker, ends left of the help panel,
     and cannot wrap. pointer-events:none as well, so it can never take a click from anything under
     it even if it is moved again. */
  #hudStat { left:156px; right:140px; bottom:10px; top:auto; padding:3px 8px; font-size:10px;
             color:var(--dim); pointer-events:none; white-space:nowrap; overflow:hidden;
             text-overflow:ellipsis; }
  #hudStat > div { display:inline; margin-right:10px; }
  body:not(.embedded) .hud { display:none; }
  #wrap { position:relative; flex:1 1 62%; min-width:0; overflow:hidden; }
  #side { flex:0 0 38%; display:flex; flex-direction:column; min-width:0;
           border-left:1px solid var(--rule); }
  #kgPane { flex:1 1 50%; min-height:0; position:relative; display:flex; flex-direction:column; }
  #kg { flex:1; min-height:0; }
  #tablePane { flex:1 1 50%; min-height:0; overflow:auto; border-top:1px solid var(--rule); }
  .paneHead { padding:6px 10px; font-size:10px; letter-spacing:.06em; color:var(--dim);
               text-transform:uppercase; border-bottom:1px solid var(--rule);
               background:var(--card); }
  table { width:100%; border-collapse:collapse; font-size:11px; }
  /* sticky: a selection in the 3D view or the KG auto-scrolls the table, and without this
     the header scrolls out of view and the columns go unlabelled. #tablePane is the
     scroll container. */
  th { text-align:left; color:var(--dim); font-weight:600; padding:5px 8px;
        border-bottom:1px solid var(--rule); background:var(--card);
        position:sticky; top:0; z-index:1; }
  td { padding:4px 8px; border-bottom:1px solid rgba(30,43,68,.55); }
  tr.sel td { background:rgba(56,189,248,.20); }
  tbody tr:hover td { background:rgba(56,189,248,.09); cursor:pointer; }
  canvas#cv { display:block; width:100%; height:100%; cursor:grab; }
  canvas#cv:active { cursor:grabbing; }
  #labels { position:absolute; inset:0; pointer-events:none; overflow:hidden; z-index:3; }
  /* THE STOREY PICKER, over the bottom-left of the scene rather than as a row of prose in
     the header. It was a text bar naming each storey's navmesh share, and it read as a
     caption -- it was asked for again while it was on screen and working. */
  #floorPick { position:absolute; left:12px; bottom:12px; z-index:4; display:flex;
               flex-direction:column; align-items:stretch; gap:3px; width:136px; }
  #floorPick button { padding:2px 0; font:600 12px ui-monospace,monospace; line-height:1.1; }
  #floorPick button[disabled] { opacity:.32; cursor:default; }
  #floorPick .who { background:rgba(8,13,24,.92); border:1px solid var(--rule);
                    border-radius:4px; padding:3px 6px; text-align:center;
                    font:600 10px ui-monospace,monospace; color:var(--ink); }
  #floorPick .who.mapped { border-color:#4ade80; color:#4ade80; }
  #floorPick .who small { display:block; color:var(--dim); font-weight:400; font-size:9px; }
  #floorPick .who.warn { color:#eab308; border-color:#eab308; }
  #labels span { position:absolute; transform:translate(4px,-14px); white-space:nowrap;
                 background:rgba(8,13,24,.72); padding:0 3px; border-radius:2px;
                 font:600 10px ui-monospace,monospace; }
  #tip { position:absolute; pointer-events:none; background:rgba(8,13,24,.95);
          border:1px solid var(--accent); border-radius:4px; padding:4px 8px;
          font:11px ui-monospace,monospace; color:var(--ink); display:none; z-index:5; }
  .legend { color:var(--faint); font-size:10px; margin-left:auto; }
  .legend i { width:9px; height:9px; display:inline-block; border-radius:2px; margin:0 3px 0 10px; }
  /* THE MESH STATUS LINE. It reports WHICH renderer answered, WHICH revision, how many
     triangles actually entered the scene and how many materials actually got a texture --
     because "the mesh loaded" is not a check (working rule 2). Red is a real failure and
     the page must look failed when it is. */
  #meshStat, #liveStat { padding:6px 18px; border-bottom:1px solid var(--rule); font-size:11px;
              color:var(--dim); background:var(--card); }
  #meshStat.ok { color:#4ade80; }
  #meshStat.warn { color:#eab308; }
  #liveStat .warn { color:#eab308; }
  #meshStat.bad { color:#f87171; background:rgba(248,113,113,.12); font-weight:700; }
  /* THE STOREY BAR. `mapped` marks the ONE storey this run was held to; `lvl` is a
     navmesh cluster the source refused as a storey and it is deliberately not a button.
     `warn` is the NOT MEASURED state -- yellow, because a disabled picker must not read
     as a single-storey building. */
  tr.off td { opacity:.32; }
"""

# The whole drawing surface, as one ES module. Kept out of the f-string on purpose: the
# previous version doubled every brace in 400 lines of JavaScript, which is unreadable and
# is how a stray brace gets lost. Two placeholders are substituted, both JSON-encoded.
_JS = r"""
import * as THREE from 'three';
// GA-380: RELATIVE, for the same reason as the import map above. A module's import specifier is
// not something the path-prefix proxy can rewrite (it sees literal src="/ and fetch('/ only), so
// an absolute /vendor/... asked the SITE ROOT for these three and got 404 behind a prefix --
// measured through a stand-in proxy, and the scene then renders nothing at all.
import { GLTFLoader } from './vendor/GLTFLoader.js';
import { BasisTextureLoader } from './vendor/BasisTextureLoader.js';
import { OrbitControls } from './vendor/OrbitControls.js';

const P = __PAYLOAD__;
// GA-380: the app can be served under a path prefix (a deployment may mount it under one). The
// proxy's rewrite only sees literal `src="/` and `fetch('/`, so every URL the SCRIPT builds is
// invisible to it and resolves against the site root -- measured as a 404 on the replay frame.
// One helper, computed once, used at every such site: three inline copies is how a fourth site
// gets missed. Trailing slashes are stripped: at the start page the path is `/` or `/<prefix>/`,
// and without that the result was `//dash`, which a browser reads as a HOST, not a path.
// ponytail: drop this when the server grows real root_path support -- that set is next.
const PFX = location.pathname
.replace(new RegExp('/(dash|replay|scene3d|bundles|arch|blockers)/?$'), '')
.replace(new RegExp('/+$'), '');
// The mesh URL is a bare path in a JS string, so the proxy's rewrite cannot see it either.
const MESH_URL = PFX + __MESHURL__;

const cv = document.getElementById('cv');
const wrap = document.getElementById('wrap');
const tip = document.getElementById('tip');
const labelHost = document.getElementById('labels');
const stat = document.getElementById('meshStat');
const parts = {renderer: 'renderer: not created yet', mesh: 'mesh: not requested yet'};
function say(k, text, cls) {
  parts[k] = text;
  stat.textContent = parts.renderer + '  |  ' + parts.mesh;
  if (cls) stat.className = cls;
}

// WebGL is a property of the VIEWING browser and nothing here can measure it in advance.
// If the context cannot be created, say so in words instead of leaving a black rectangle
// that reads as an empty scene.
let renderer;
try {
  renderer = new THREE.WebGLRenderer({canvas: cv, antialias: true});
} catch (err) {
  say('renderer', 'WebGL unavailable in this browser: ' + err.message, 'bad');
  throw err;
}
renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
const gl = renderer.getContext();
say('renderer', 'three r' + THREE.REVISION + ' / ' +
    (renderer.capabilities.isWebGL2 ? 'WebGL2' : 'WebGL1') + ' / ' +
    gl.getParameter(gl.RENDERER));

const scene = new THREE.Scene();
scene.background = new THREE.Color(0x080d18);
// THE MAP FRAME IS THE WORLD FRAME. z is up (GA-255): the object boxes, the walls, the
// openings, the agent poses and the floor bounds are all already in it, so NOTHING below
// is swizzled. Only the GLB is transformed, and only by a rotation about z.
const camera = new THREE.PerspectiveCamera(50, 1, 0.05, 500);
camera.up.set(0, 0, 1);
const controls = new OrbitControls(camera, renderer.domElement);
controls.screenSpacePanning = true;
// THE MIDDLE BUTTON IS TAKEN OFF OrbitControls, deliberately and before anything else.
// r128 binds MIDDLE to DOLLY in its own `mouseButtons` default, so left alone it would
// zoom while the centre is being dragged. A null action reaches OrbitControls'
// `default: state = STATE.NONE`, which registers no move listener at all, so the handler
// further down owns the button outright instead of fighting one.
controls.mouseButtons.MIDDLE = null;

// YAW IS NORMALISED BY THE PANEL'S WIDTH, not its height, and this is set when a drag
// BEGINS so it is always the size the drag is actually happening in.
//
// OrbitControls r128 rotates by `2*PI * deltaX / element.clientHeight` -- it divides a
// HORIZONTAL gesture by a VERTICAL measurement. In a tall window that is merely odd; in the
// dashboard's 3D tab, which is short and wide, it makes the view uncontrollable. MEASURED in
// the tab at 606x418: a 100 px drag turned the camera 86.1 degrees and a full turn took
// 418 px, so there was no such thing as a small adjustment.
//
// It bites HERE and not on the standalone page because the panel is shorter, and it bites at
// all only because the pitch is pinned (`lockPitch`, owner decision 2026-09-04): with the
// vertical drag doing nothing by design, yaw is the WHOLE control, and it was the one axis
// scaled by the wrong dimension.
//
// rotateSpeed multiplies that angle, so h/w cancels the height and substitutes the width:
// one full turn per panel width, whatever the shape. ON 'start' RATHER THAN ON RESIZE: the
// first version set it in resize(), which runs on the maximise event before the panel has
// its new size, so it kept the small-panel value at full size (0.69 measured where 0.545 was
// due). A drag cannot begin before the panel exists, so this reading is never early.
controls.addEventListener('start', () => {
  const el = renderer.domElement;
  controls.rotateSpeed = el.clientHeight / Math.max(1, el.clientWidth);
});

// YAW AND ZOOM ONLY, by owner decision 2026-09-04. Left-drag turns the model about the
// vertical; it does not tip it.
//
// The pitch is pinned by clamping OrbitControls' own polar limits to whatever the fitted
// view arrived at, rather than by intercepting the drag. Its update() clamps the spherical
// polar between minPolarAngle and maxPolarAngle every frame, so with the two equal the
// vertical component of a drag has nowhere to go and the azimuth is unaffected. Wheel zoom
// is a radius change and is untouched, as is the middle-button centre drag, which moves the
// pivot rather than orbiting round it.
//
// Read the angle back from the controls rather than computing it: the camera is Z-UP
// (camera.up is (0,0,1) above) and OrbitControls carries its own quaternion to that frame,
// so a Spherical built here from camera.position - target would be in the WRONG frame and
// would pin the pitch to a plausible but different angle.
function lockPitch() {
  controls.minPolarAngle = 0;
  controls.maxPolarAngle = Math.PI;
  controls.update();
  const a = controls.getPolarAngle();
  controls.minPolarAngle = controls.maxPolarAngle = a;
}

// ---- THE PIVOT IS FREE OF THE AIM ----------------------------------------------------
// OrbitControls ends every update() with `camera.lookAt(target)`, so the orbit centre is
// also the aim point. That is the whole difficulty of "drag the centre": under a
// look-at camera, moving the centre sideways and keeping the picture still and keeping
// the camera facing the centre are three demands and only two can hold at once.
//   * translate camera AND centre together -> the aim holds and the PICTURE SLIDES. That
//     is round 2's first attempt, and it is arithmetically the same thing as right-drag
//     pan, which the page already had.
//   * move the centre alone -> the picture holds, and the next update swings the camera
//     round to face the new centre.
// AIMOFF drops the third demand instead of the second. It is the rotation from "looking
// at the centre" to "the orientation we are holding", expressed in the CAMERA's own
// frame, and it is put back after every update(). Constant in the local frame, so an
// orbit -- which only moves the frame -- carries it along: the camera revolves about the
// crosshair without ever facing it, and nothing snaps, then or later.
// ponytail: the wheel still dollies along camera->centre, so once the centre is off the
// view axis the zoom pulls a little sideways. The ceiling is named rather than fixed;
// fixing it means dollying along the view direction, which is a different control from
// the one round 1 verified.
let AIMOFF = null;
const _ctlUpdate = controls.update.bind(controls);
controls.update = function () {
  const moved = _ctlUpdate();
  if (AIMOFF) { camera.quaternion.multiply(AIMOFF); camera.updateMatrixWorld(); }
  return moved;
};
// Records the offset and leaves the camera EXACTLY where and how it was, so every fixed
// world point projects to the pixel it already occupied. `camera.lookAt(controls.target)`
// is the same call OrbitControls makes, used here only to read the aim it would impose.
function holdAim() {
  const q0 = camera.quaternion.clone();
  camera.lookAt(controls.target);
  AIMOFF = camera.quaternion.clone().invert().multiply(q0);
  camera.quaternion.copy(q0);
  camera.updateMatrixWorld();
}

const G = {};
// `dwall` and `sched` are their own groups, NOT extra children of `wall` and `path`: the
// GT walls and the traveled path are a different claim from the detected walls and the
// planned route, and one toggle over both would make them one thing on screen.
for (const k of ['floor', 'mesh', 'wall', 'dwall', 'open', 'obj', 'path', 'sched', 'robot']) {
  G[k] = new THREE.Group();
  scene.add(G[k]);
}
const show = {obj: true, wall: true, open: true, path: true, label: false, floor: true,
              mesh: true, robot: true, dwall: true, sched: true};

// THE ORBIT CENTRE, DRAWN. There was no marker here before this round -- the centre was an
// invisible point you could only locate by orbiting and watching what stood still. Three
// crossed segments, not a ring: a ring has a plane and this thing does not.
// depthTest is off because the centre is normally INSIDE the mesh, and a marker you cannot
// see while dragging it is not a marker.
const XHAIR = new THREE.LineSegments(
  new THREE.BufferGeometry().setFromPoints([
    new THREE.Vector3(-1, 0, 0), new THREE.Vector3(1, 0, 0),
    new THREE.Vector3(0, -1, 0), new THREE.Vector3(0, 1, 0),
    new THREE.Vector3(0, 0, -1), new THREE.Vector3(0, 0, 1)]),
  new THREE.LineBasicMaterial({color: 0x38bdf8, transparent: true, opacity: 0.9,
                               depthTest: false}));
XHAIR.renderOrder = 998;
scene.add(XHAIR);

function hueOf(name) {
  let h = 0;
  const s = String(name);
  for (let k = 0; k < s.length; k++) h = (h * 31 + s.charCodeAt(k)) % 360;
  return h;
}

// A translucent box plus its own edges. depthWrite is off so a stack of translucent boxes
// blends instead of the nearest one erasing the ones behind it -- which is what the old
// painter's-algorithm sort could not do at all.
function addBox(group, lo, hi, colour, opacity, edgeColour, edgeOpacity, userData) {
  const sx = Math.max(1e-4, hi[0] - lo[0]);
  const sy = Math.max(1e-4, hi[1] - lo[1]);
  const sz = Math.max(1e-4, hi[2] - lo[2]);
  const g = new THREE.BoxGeometry(sx, sy, sz);
  const m = new THREE.Mesh(g, new THREE.MeshBasicMaterial({
    color: colour, transparent: true, opacity: opacity, depthWrite: false,
    side: THREE.DoubleSide}));
  m.position.set((lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2, (lo[2] + hi[2]) / 2);
  const e = new THREE.LineSegments(new THREE.EdgesGeometry(g),
    new THREE.LineBasicMaterial({color: edgeColour, transparent: true, opacity: edgeOpacity}));
  m.add(e);
  m.userData = Object.assign({edge: e, baseColour: colour}, userData || {});
  group.add(m);
  return m;
}

// ---- measured object boxes ----------------------------------------------------------
// MESHMATS is filled by the GLB load callback and is EMPTY until it returns. Selection
// dimming reads it, so a click before the mesh arrives dims nothing and throws nothing.
const OBJMESH = [];
let MESHMATS = [];
// ONE BOX, MADE ONCE, so the same call serves the first paint and a box that appears
// mid-run. Everything downstream -- the table, the storey picker, picking, selection --
// keys on `objKey`, so the factory sets it rather than each caller re-deriving it.
// ponytail: the key is `id || label`, which is what the table and FLOOROF already used
// before this round. Ceiling: two objects with NO object_id and the same label collide
// into one. Measured on 20260904_082146: 75 of 75 objects carry an object_id, so nothing
// collides in this bundle. Upgrade path is a server-side index in `scene_payload`.
function objKey(o) { return o.id || o.label; }
const OBJBY = new Map();
function addObject(o) {
  const b = o.box;
  const h = hueOf(o.label);
  const c = new THREE.Color().setHSL(h / 360, 0.80, 0.55);
  const m = addBox(G.obj, [b[0], b[1], b[2]], [b[3], b[4], b[5]], c, 0.30, c, 1.0,
                   {o: o, hue: h, floor: o.floor, key: objKey(o)});
  OBJMESH.push(m);
  OBJBY.set(objKey(o), m);
  return m;
}
function dropObject(key) {
  const m = OBJBY.get(key);
  if (!m) return;
  G.obj.remove(m);
  m.geometry.dispose();
  m.material.dispose();
  OBJMESH.splice(OBJMESH.indexOf(m), 1);
  OBJBY.delete(key);
}
for (const o of P.objects) addObject(o);

// A box drawn as its 12 edges only, with no face. Used for an aperture reveal and for an
// opening no wall claimed -- both are outlines round empty space, not solids.
function addFrame(group, lo, hi, colour, opacity) {
  const g = new THREE.BoxGeometry(Math.max(1e-4, hi[0] - lo[0]),
                                  Math.max(1e-4, hi[1] - lo[1]),
                                  Math.max(1e-4, hi[2] - lo[2]));
  const e = new THREE.LineSegments(new THREE.EdgesGeometry(g),
    new THREE.LineBasicMaterial({color: colour, transparent: true, opacity: opacity}));
  e.position.set((lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2, (lo[2] + hi[2]) / 2);
  group.add(e);
  return e;
}

// ---- ground truth: SOLID WALL SURFACES, WITH THE OPENINGS CUT THROUGH THEM ------------
// GA-255 again: `lo`/`hi` and `pos`/`ext` are ALREADY the map frame with z vertical.
//
// ONE EXTRUDED SLAB PER MERGED SURFACE, not a box per segment and not a yellow box laid
// over the doorway. `cut_openings` gives each surface its thin `axis` and its apertures as
// [u0, v0, u1, v1] rectangles in world units, (u, v) being its two in-plane axes ascending
// -- (length, z) for a wall, (x, y) for a horizontal slab; the outline goes in as a THREE.Shape and the
// apertures as its holes, so the doorway is a hole in the geometry and stays a hole when
// the OPENINGS toggle is off.
//
// THE WALLS ARE SOLID -- no `transparent`, no `opacity`, so three's defaults apply and
// the material is opaque with depth writing on. Round 1 left them at 0.28 and flagged it.
// A translucent wall is not a wall: it is a diagram of one, and the interior it reveals is
// smeared with everything standing behind the building.
//
// THE BELIEF/TRUTH SEPARATION SURVIVES THE CHANGE and is still the point of the view. It
// was never carried by the opacity alone. Ground truth is FLAT UNTEXTURED GREY (0x93a6c0,
// desaturated, unlit) with a pale outline; the pipeline's own boxes are SATURATED per-label
// hues at 0.30 with a bright edge and a label. Grey solid against coloured translucent
// reads at a glance, and a wall the pipeline never saw still cannot pass for one it mapped.
//
// TRANSLUCENT, by owner decision 2026-09-04: "previous wall transparency was ok".
//
// The solid-wall version needed a camera-facing cutaway to see inside at all, and the
// cutaway is a second mechanism whose only job is to undo the first. A translucent wall
// shows the interior with no mechanism, which is why it was right the first time. The
// cutaway code survives behind CUT, defaulting OFF, so nothing that referenced it broke.
const WALL_MAT = new THREE.MeshBasicMaterial({
  color: 0x93a6c0, transparent: true, opacity: 0.28, depthWrite: false,
  side: THREE.DoubleSide});
const WALL_EDGE = new THREE.LineBasicMaterial({
  color: 0xc3d2e6, transparent: true, opacity: 0.55});

for (const w of P.wall_mass) {
  // a: the surface's thin axis, 0, 1 or 2. u, v: its two in-plane axes, ascending. For a
  // wall that is (length, z), exactly what `u = 1 - a` gave; for a slab it is (x, y),
  // which the old rule could not name -- see the axis note in cut_openings.
  const a = w.axis, u = a === 0 ? 1 : 0, v = a === 2 ? 1 : 2;
  const shape = new THREE.Shape();
  shape.moveTo(w.lo[u], w.lo[v]);
  shape.lineTo(w.hi[u], w.lo[v]);
  shape.lineTo(w.hi[u], w.hi[v]);
  shape.lineTo(w.lo[u], w.hi[v]);
  shape.closePath();
  for (const h of w.holes) {
    const q = new THREE.Path();
    q.moveTo(h[0], h[1]);
    q.lineTo(h[2], h[1]);
    q.lineTo(h[2], h[3]);
    q.lineTo(h[0], h[3]);
    q.closePath();
    shape.holes.push(q);
  }
  const g = new THREE.ExtrudeGeometry(shape, {
    depth: Math.max(1e-3, w.hi[a] - w.lo[a]), bevelEnabled: false, curveSegments: 1});
  // The shape was drawn in (u, v) and extruded along its own +z. Stand it up by naming
  // where its three local axes point in the world: local x is world u, local y is world v,
  // local z is the surface's thickness. For a wall thin in y that basis is LEFT-handed,
  // which reverses the face winding -- harmless here and only here, because the material is
  // DoubleSide and unlit, so no normal is ever consulted. A slab's basis is (x, y, z),
  // right-handed, so the third case neither gains nor loses by it.
  const e1 = new THREE.Vector3(), e2 = new THREE.Vector3(), e3 = new THREE.Vector3();
  const t = new THREE.Vector3();
  e1.setComponent(u, 1);
  e2.setComponent(v, 1);
  e3.setComponent(a, 1);
  t.setComponent(a, w.lo[a]);
  const m = new THREE.Mesh(g, WALL_MAT);
  m.matrixAutoUpdate = false;
  m.matrix.makeBasis(e1, e2, e3);
  m.matrix.setPosition(t);
  // EdgesGeometry keeps an edge only where two faces meet at an angle, so the outline and
  // every aperture survive and the triangulator's cuts across the flat face do not. That is
  // what stops 91 slabs reading as 91 boxes.
  m.add(new THREE.LineSegments(new THREE.EdgesGeometry(g), WALL_EDGE));
  // matrixAutoUpdate is off and the matrix is a basis, so m.position is NOT the centre.
  // The cutaway and the floor picker both need it, so it is carried explicitly.
  m.userData = {floor: w.floor, axis: a, onFloor: true,
                centre: new THREE.Vector3((w.lo[0] + w.hi[0]) / 2,
                                          (w.lo[1] + w.hi[1]) / 2,
                                          (w.lo[2] + w.hi[2]) / 2)};
  G.wall.add(m);

  // The reveal: a yellow outline round the aperture, through the wall's thickness. The
  // OPENINGS toggle still names something now that an opening is a hole.
  for (const h of w.holes) {
    const lo = [0, 0, 0], hi = [0, 0, 0];
    lo[u] = h[0]; hi[u] = h[2];
    lo[v] = h[1]; hi[v] = h[3];
    lo[a] = w.lo[a]; hi[a] = w.hi[a];
    addFrame(G.open, lo, hi, 0xeab308, 0.95).userData = {floor: w.floor};
  }
}

// An opening NO surface claimed is still drawn, as the box it always was. Dropping it would
// make an aperture the ground truth does have look like one it does not.
// ---- DETECTED WALLS AND THE PLANNED ROUTE ---------------------------------------------
// Both are drawn at the storey height rather than guessed: every schedule point carries its
// own y, and a detected wall is a ground-plane segment lifted to the floor it belongs to.
// AMBER for detected, against the grey of ground truth, and VIOLET dashes for the plan
// against the solid line of the path actually walked.
const DWALL_MAT = new THREE.LineBasicMaterial({color: 0xf59e0b, linewidth: 2});
const SCHED_MAT = new THREE.LineDashedMaterial({color: 0xa78bfa, dashSize: 0.25, gapSize: 0.18});

function wallSegments(w) {
  // THE PRODUCER'S ACTUAL SHAPE IS {start:{x,y}, end:{x,y}}. room_manager.py:2708 reads
  // `wall["start"]["x"]` and object_manager_6.py's walls_callback says so in as many words,
  // and room.json's top-level `detected_walls` is that same `_detected_wall_map`
  // (room_manager.py:3006). The first version of this function accepted only [[x,y],[x,y]]
  // and {points: [...]} -- two shapes I had INVENTED -- so with the detector switched on this
  // layer would have drawn nothing and looked like a detector that found no walls. Verifying
  // a reader against your own guess at the format proves only that the guess is self-consistent.
  //
  // The two array forms are kept as tolerated alternatives, cheap and harmless, but `start`
  // and `end` are the shape that actually arrives.
  let pts = null;
  if (w && w.start && w.end &&
      typeof w.start.x === 'number' && typeof w.end.x === 'number') {
    pts = [[w.start.x, w.start.y], [w.end.x, w.end.y]];
  } else {
    pts = Array.isArray(w) ? w : (w && w.points);
  }
  if (!Array.isArray(pts) || pts.length < 2) return null;
  const z = (w && typeof w.z === 'number') ? w.z : (P.floor && P.floor.z) || 0;
  const out = [];
  for (const q of pts) {
    if (!q || q.length < 2) continue;
    // A 3-component point states its own height; a 2-component one sits on the storey.
    out.push(new THREE.Vector3(q[0], q[1], q.length > 2 ? q[2] : z));
  }
  return out.length >= 2 ? out : null;
}

for (const w of (P.detected_walls || [])) {
  const pts = wallSegments(w);
  if (!pts) continue;
  const line = new THREE.Line(new THREE.BufferGeometry().setFromPoints(pts), DWALL_MAT);
  line.userData = {floor: (w && w.floor) != null ? w.floor : null};
  G.dwall.add(line);
}

if (P.schedule && Array.isArray(P.schedule.path) && P.schedule.path.length > 1) {
  const sy = typeof P.schedule.storey_y === 'number' ? P.schedule.storey_y : 0;
  const at = q => new THREE.Vector3(q[0], q[1], q.length > 2 ? q[2] : sy);
  const line = new THREE.Line(
    new THREE.BufferGeometry().setFromPoints(P.schedule.path.map(at)), SCHED_MAT);
  // computeLineDistances or the dashes never appear -- a dashed material on a Line draws
  // solid without it, which would make the plan indistinguishable from the walked path.
  line.computeLineDistances();
  G.sched.add(line);
  // Each 360-degree scan point as a small ring lying in the floor plane, so it reads as a
  // place to stand rather than as another detected object.
  const ringGeo = new THREE.RingGeometry(0.12, 0.2, 20);
  const ringMat = new THREE.MeshBasicMaterial({color: 0xa78bfa, side: THREE.DoubleSide,
                                               transparent: true, opacity: 0.9});
  for (const st of (P.schedule.stops || [])) {
    const q = st && (st.xyz || st);
    if (!q || q.length < 2) continue;
    const r = new THREE.Mesh(ringGeo, ringMat);
    r.position.copy(at(q));
    r.userData = {order: st && st.order};
    G.sched.add(r);
  }
  if (Array.isArray(P.schedule.root) && P.schedule.root.length >= 2) {
    const rootRing = new THREE.Mesh(new THREE.RingGeometry(0.3, 0.42, 24), ringMat);
    rootRing.position.copy(at(P.schedule.root));
    G.sched.add(rootRing);
  }
}

for (const w of P.openings) {
  if (w.cut) continue;
  const [x, y, z] = w.pos, [ex, ey, ez] = w.ext;
  addFrame(G.open, [x - ex / 2, y - ey / 2, z - ez / 2],
                   [x + ex / 2, y + ey / 2, z + ez / 2], 0xeab308, 0.55).userData = {floor: w.floor};
}

// ---- agent path ----------------------------------------------------------------------
if (P.agent && P.agent.path && P.agent.path.length) {
  const pts = P.agent.path.map(q => new THREE.Vector3(q[0], q[1], q[2]));
  G.path.add(new THREE.Line(new THREE.BufferGeometry().setFromPoints(pts),
    new THREE.LineBasicMaterial({color: 0xf472b6})));
}

// ---- the run's own occupancy map, as a textured ground plane -------------------------
// flipY is FALSE on purpose. The bounds say image row 0 sits at y0; three's default
// flipY would put the image's LAST row there, i.e. the floor plan upside down.
//
// THE PLAN IS A DECK, NOT A TINT, and this is what FLOOR MAP was missing. It drew
// nothing readable whenever GT WALLS was on. Measured on hm3d_00861: the plane sits at
// z = 1.21 -- which is right, the agent walked at z 1.190 .. 1.217 and every object box
// starts at z >= 1.152 -- while the ground truth's walls span z -1.64 .. 4.78, because
// the GT covers the WHOLE building and the run explored ONE storey of it. So the plane
// cuts through the wall slabs at mid height. Everything was transparent with depthWrite
// off, and three sorts the transparent queue by each object's centre depth, so roughly
// half the slabs sorted in front of the plane and half behind it. The plan was
// composited OVER the walls instead of under them, and what reached the screen was a
// uniform darkening of every wall and no plan anywhere.
//
// Three properties, each doing one job:
//   renderOrder -1  draws the plane FIRST in the transparent queue, so the walls of the
//                   storey above it blend over the plan instead of under it.
//   depthWrite      makes the deck OPAQUE TO WHAT IS BENEATH IT. Without it the plan is
//                   still legible only where nothing else covers it, because it competes
//                   with the lower storey's slabs seen through the floor. A floor you can
//                   see the cellar through is not a floor.
//   opacity 0.85    the deck sits under the storey's own walls at 0.28 each.
// Measured cost, stated rather than hidden: a fragment strictly BELOW the deck and inside
// its footprint is now depth-rejected. That is 11 of 302 agent-path pixels, the poses that
// dip to z = 1.190 under a deck at z = 1.21. The path stays continuous and readable. The
// alternative was to drop the plane a few centimetres, which buys those 11 pixels by
// putting the floor somewhere the map does not say it is.
// ONE DECK PER STOREY, not one deck. `bev_data.json` renders a top-down PNG for every
// floor it found and keys them by the floor height as a string; `scene_payload` resolves
// those to `floor_decks`, parallel to `floors`. Exactly one is visible at a time and
// `applyFloor` chooses which -- the picked storey's, or under ALL the storey the run
// actually mapped, because that is the one the boxes belong to.
function addDeck(f, idx) {
  if (!f || !f.image) return;
  new THREE.TextureLoader().load(f.image, tex => {
    tex.flipY = false;
    const g = new THREE.PlaneGeometry(Math.abs(f.x1 - f.x0), Math.abs(f.y1 - f.y0));
    const m = new THREE.Mesh(g, new THREE.MeshBasicMaterial({
      map: tex, transparent: true, opacity: 0.85, depthWrite: true, side: THREE.DoubleSide}));
    m.position.set((f.x0 + f.x1) / 2, (f.y0 + f.y1) / 2, f.z);
    m.renderOrder = -1;
    m.userData = {deck: idx};
    G.floor.add(m);
    applyFloor();
    render();
  });
}
if (P.floor_decks && P.floor_decks.some(d => d)) {
  P.floor_decks.forEach(addDeck);
} else if (P.floor) {
  // No per-floor renders in this bundle. The run's single map is still drawn, and it is
  // tagged -1 so applyFloor knows it belongs to no particular storey and always shows it.
  addDeck(P.floor, -1);
}

// ---- THE STOREY PICKER ---------------------------------------------------------------
// hm3d_00861 has FOUR navmesh height clusters and only TWO are storeys. ONE RUN MAPS ONE
// STOREY, so a viewer who orbits down to the other floor sees an empty shell and would
// otherwise conclude the pipeline found nothing. The bar therefore states which storey
// this run was held to, with the guard's own evidence beside it, and marks that button.
//
// Every drawn thing already carries `floor`, a LIST of storey indices computed server-side
// from its own z extent (see assign_floor). A list, not an index, because 7 ground-truth
// instances in this scene genuinely span both storeys -- 4 walls, a window, a window frame
// and a stairs railing -- and dropping one leaves a hole in a wall that has no opening.
//
// THE MESH IS THE ONE THING THAT CANNOT BE FILTERED. It is a single continuous building,
// so it is CLIPPED instead, by two horizontal planes through the storey band, using
// three's own per-material clippingPlanes. Nothing is re-uploaded and nothing is rebuilt.
renderer.localClippingEnabled = true;
// THE SIMS CUTOUT, as a clipping plane through the orbit centre facing the camera. The
// wall-mesh cutaway below it can only hide GT wall objects, and HM3D ships none -- its
// semantic annotation is a flat instance list with no wall geometry -- so on every scene here
// the cutaway had nothing to act on and the building stayed sealed. Clipping the MESH is what
// opens it, and the machinery was already present for storey banding.
const CUTPLANE = new THREE.Plane(new THREE.Vector3(0, 0, 1), 0);
// Keeps z BELOW its constant: normal (0,0,-1) makes the signed distance (constant - z), so
// the half-space kept is everything under the ceiling. Same convention as CLIP[1].
const CEILPLANE = new THREE.Plane(new THREE.Vector3(0, 0, -1), 0);
const CLIP = [new THREE.Plane(new THREE.Vector3(0, 0, 1), 0),
              new THREE.Plane(new THREE.Vector3(0, 0, -1), 0)];
let FLOOR = null;                 // null = ALL storeys
let CUT = true;                   // the Sims cutaway, on by default

// The storey this run mapped, as an INDEX into P.floors, or -1. Matched on the guard's
// own floor_y to within 1 cm rather than on a formatted string.
const MAPPEDI = (P.mapped_floor && P.floors.length)
  ? P.floors.findIndex(f => Math.abs(f.z - P.mapped_floor.floor_y) < 0.01) : -1;

function applyFloor() {
  const all = FLOOR === null;
  const on = m => all || (m.userData.floor && m.userData.floor.indexOf(FLOOR) >= 0);
  // Walls get a flag, not a visibility: the cutaway decides the rest of it every frame.
  for (const m of G.wall.children) m.userData.onFloor = on(m);
  for (const m of G.open.children) m.visible = on(m);
  for (const m of OBJMESH) m.visible = on(m);
  for (const m of G.floor.children) {
    const d = m.userData.deck;
    m.visible = (d === -1) || (all ? d === MAPPEDI : d === FLOOR);
  }
  // THE AGENT PATH BELONGS TO THE STOREY THE RUN MAPPED, and to no other. Left on, it
  // drew the robot's trajectory across a storey the robot never entered -- which is the
  // exact misreading this picker exists to prevent, arriving through the one layer that
  // is not ground truth and not a box.
  // ponytail: the whole polyline is shown or hidden. It is one THREE.Line over the run's
  // poses, and this run's z spans 1.190 .. 1.217 -- a single storey -- so there is nothing
  // to split. Ceiling: a run that legitimately changed storey mid-way would vanish from
  // both. Upgrade path is one line per contiguous same-storey stretch of poses.
  G.path.children.forEach(m => {
    m.visible = all || MAPPEDI < 0 || FLOOR === MAPPEDI;
  });
  // The table is the same selection surface as the scene, so it says the same thing: a row
  // for a box on another storey is dimmed, not deleted -- deleting it would make the run
  // look as though it had found fewer objects than it did.
  for (const tr of document.querySelectorAll('#objBody tr'))
    tr.classList.toggle('off', !(all || (FLOOROF[tr.dataset.id] || []).indexOf(FLOOR) >= 0));
  // THE MESH, CLIPPED. keep z > z0 is normal (0,0,1) with constant -z0; keep z < z1 is
  // normal (0,0,-1) with constant z1. The band is lowered by 0.3 m so the storey's own
  // floor slab, which sits just under the navmesh height, stays in the picture.
  let planes = [];
  if (!all && P.floors.length) {
    CLIP[0].constant = -(P.floors[FLOOR].z - 0.3);
    planes = [CLIP[0]];
    if (FLOOR + 1 < P.floors.length) {
      CLIP[1].constant = P.floors[FLOOR + 1].z - 0.3;
      planes.push(CLIP[1]);
    }
  }
  // THE CUTOUT RIDES ALONG WITH THE STOREY BAND. Membership is set here; the plane's own
  // normal and offset are refreshed every frame in `cutaway()`, and three reads the plane
  // object at draw time, so following the camera costs no material rebuild.
  //
  // THE ROOF COMES OFF TOO, which is what makes it a dollhouse rather than a sliced loaf.
  // `ceiling_z` is the lowest annotated ceiling that begins above the storey this run mapped
  // (computed server-side from the GT ceilings). On hm3d_00861 the mapped storey is the UPPER
  // one at z 1.21 and its ceiling is at 3.32, while the lower storey's ceiling sits at
  // 0.70-1.05 -- so a fixed offset above the floor would have cut the wrong one. Null when
  // the scene annotates no ceiling above the storey, and then the roof is left alone rather
  // than cut at an invented height.
  if (CUT && typeof P.ceiling_z === 'number') {
    CEILPLANE.constant = P.ceiling_z;
    planes = planes.concat([CEILPLANE]);
  }
  if (CUT) planes = planes.concat([CUTPLANE]);
  for (const m of MESHMATS) { m.clippingPlanes = planes; m.needsUpdate = true; }
}

const FLOOROF = {};
function indexFloors() {
  for (const k of Object.keys(FLOOROF)) delete FLOOROF[k];
  for (const o of P.objects) FLOOROF[objKey(o)] = o.floor;
}
indexFloors();

function setFloor(i) {
  FLOOR = i;
  // FLOOR is null for ALL and an index otherwise. Kept as null rather than -1 because
  // applyFloor already tests `FLOOR === null` in four places.
  paintFloorPick();
  applyFloor();
  render();
}

// THE ORDER THE ARROWS WALK. Index -1 is ALL; 0..n-1 are the storeys, and P.floors is
// already sorted by z ascending, so UP really is up. ALL sits below the lowest storey
// rather than beside them, because a list with a mode in the middle of it cannot be walked
// with two arrows without the arrows meaning two different things.
function floorSeq() { return [null].concat(P.floors.map((_, i) => i)); }

function stepFloor(d) {
  const seq = floorSeq();
  const at = seq.indexOf(FLOOR);
  const to = at + d;
  if (to < 0 || to >= seq.length) return;
  setFloor(seq[to]);
}

function buildFloorPick() {
  const host = document.getElementById('floorPick');
  if (!P.floors.length) {
    // NOT MEASURED, said in words. A picker that silently offers one floor because it found
    // no data looks identical to a single-storey building.
    host.innerHTML = '<div class="who warn">STOREY<small>' + P.floor_source +
                     ' &mdash; picker disabled</small></div>';
    return;
  }
  host.innerHTML = '<button id="fUp" title="storey up">&#9650;</button>' +
                   '<div class="who" id="fWho"></div>' +
                   '<button id="fDn" title="storey down">&#9660;</button>';
  document.getElementById('fUp').onclick = () => stepFloor(1);
  document.getElementById('fDn').onclick = () => stepFloor(-1);
  paintFloorPick();
}

function paintFloorPick() {
  const who = document.getElementById('fWho');
  if (!who) return;
  const seq = floorSeq(), at = seq.indexOf(FLOOR);
  document.getElementById('fUp').disabled = at >= seq.length - 1;
  document.getElementById('fDn').disabled = at <= 0;
  if (FLOOR === null) {
    who.className = 'who';
    who.innerHTML = 'ALL STOREYS<small>' + P.floors.length + ' of ' +
                    (P.floors.length + P.levels.length) + ' levels are storeys</small>';
    return;
  }
  const f = P.floors[FLOOR], c = P.floor_counts[FLOOR] || {};
  const share = (f.share === undefined) ? 'share n/a' : (f.share * 100).toFixed(1) + '% navmesh';
  who.className = 'who' + (FLOOR === MAPPEDI ? ' mapped' : '');
  // The count and the MAPPED flag stay ON the control. An empty storey is the honest result
  // of picking one the run never went to, and without this it reads as a pipeline failure.
  who.innerHTML = 'z ' + f.z.toFixed(2) + (FLOOR === MAPPEDI ? ' &middot; MAPPED' : '') +
                  '<small>' + (c.objects || 0) + ' of ' + P.objects.length + ' boxes &middot; ' +
                  share + '</small>';
}
buildFloorPick();

// ---- THE SIMS CUTAWAY ------------------------------------------------------------------
// In The Sims the walls are solid and the interior is visible because the surfaces standing
// between the camera and the room are taken away as the camera orbits. Same here, and it is
// recomputed every frame from the camera, so it follows an orbit continuously.
//
// TWO CONDITIONS, both necessary:
//   1. THE SURFACE IS IN FRONT OF THE CONTENT. `(centre - target) . viewdir < 0` puts it in
//      the half space between the camera and the orbit centre. The orbit centre is what the
//      viewer is looking INTO, so it is the right anchor -- and it is the one the middle
//      drag already lets them place.
//   2. THE SURFACE ACTUALLY BLOCKS. A merged surface is a slab with a thin axis, so its
//      normal is that axis. |viewdir . normal| > 0.35 keeps the walls the camera is looking
//      ALONG -- roughly 20 degrees off edge-on -- which is what stops the cutaway stripping
//      the side walls of the room it just opened.
//
// ponytail: a half-space test, not an occlusion test. It does not ask whether a particular
// object is behind a particular wall, so a wall in the near half that happens to block
// nothing is still removed -- which is exactly what The Sims does. The named ceiling is
// that with the orbit centre parked in a far corner the near half is most of the building.
// Upgrade path: one raycast per object box against G.wall, at 75 x 91 tests per camera
// change; it was not written because the half-space rule is five lines and this page has a
// render-on-demand loop to keep cheap.
const CAMDIR = new THREE.Vector3(), WOFF = new THREE.Vector3();
function cutaway() {
  const use = CUT && show.wall;
  CAMDIR.subVectors(controls.target, camera.position).normalize();
  // The cutout plane keeps the FAR side of the orbit centre, so everything between you and
  // what you are looking at is removed -- the near wall and the roof over it. It is aimed
  // afresh every frame, which is what makes it follow an orbit instead of cutting one fixed
  // face. `setFromNormalAndCoplanarPoint` keeps the half-space the normal points into.
  CUTPLANE.setFromNormalAndCoplanarPoint(CAMDIR, controls.target);
  for (const m of G.wall.children) {
    let vis = m.userData.onFloor !== false;
    if (vis && use) {
      WOFF.subVectors(m.userData.centre, controls.target);
      if (Math.abs(CAMDIR.getComponent(m.userData.axis)) > 0.35 && WOFF.dot(CAMDIR) < 0)
        vis = false;
    }
    m.visible = vis;
  }
}

// ---- view fitting --------------------------------------------------------------------
function contentBox() {
  const b = new THREE.Box3();
  for (const m of OBJMESH) b.expandByObject(m);
  for (const m of G.wall.children) b.expandByObject(m);
  // THE ROUTE AND THE DETECTED WALLS COUNT AS CONTENT. They did not, and on a bundle whose
  // only content IS the route the view framed nothing: measured on 20260910_185402_hm3d_00861,
  // which has 0 objects and 0 GT walls -- the schedule group held 36 drawables (a 179-point
  // path plus 34 stop rings and the root) and the camera pointed away from all of them, so a
  // layer that WAS drawing looked like a layer that was broken.
  //
  // `isEmpty()` in fitView is the tell: an empty box means "nothing to look at", and that
  // claim has to be made over everything the scene can show, not over two of its groups.
  for (const m of G.sched.children) b.expandByObject(m);
  for (const m of G.dwall.children) b.expandByObject(m);
  return b;
}

function fitView() {
  const b = contentBox();
  if (b.isEmpty()) return;
  const c = b.getCenter(new THREE.Vector3()), s = b.getSize(new THREE.Vector3());
  const r = Math.max(s.x, s.y, s.z, 1) * 0.5;
  const d = r / Math.tan((camera.fov / 2) * Math.PI / 180) * 1.5;
  controls.target.copy(c);
  // THE ELEVATION IS 45 DEGREES ABOVE THE HORIZON (owner decision 2026-09-06; it was
  // ~30, the triple 0.62 / -0.62 / 0.50). Written as a UNIT direction times d so the angle
  // is the ONLY thing that changed: the old triple has length 1.009 d, and simply raising
  // its z to 0.88 would have length 1.242 d -- a 23 % zoom-out smuggled in with the angle.
  // The azimuth is untouched, +x and -y in equal parts. EL is the knob: one number, in
  // degrees, if the storey plan wants more or less of the frame.
  const EL = 45 * Math.PI / 180, HZ = Math.cos(EL) / Math.SQRT2;
  camera.position.set(c.x + d * HZ, c.y - d * HZ, c.z + d * Math.sin(EL));
  camera.near = Math.max(0.02, d / 400);
  camera.far = d * 40;
  camera.updateProjectionMatrix();
  AIMOFF = null;      // RESET VIEW resets the aim too, or the fitted view arrives crooked
  controls.update();
}

function resize() {
  const w = wrap.clientWidth || window.innerWidth;
  const h = wrap.clientHeight || (window.innerHeight - 122);
  // devicePixelRatio changes when the window moves between monitors; the load-time value
  // goes stale and the backing store stops matching the CSS box (measured: factors 1.971
  // vs 2.000 after a 2 -> 1.6 move). setPixelRatio and setSize go together.
  renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
  renderer.setSize(w, h, false);
  camera.aspect = w / Math.max(1, h);
  camera.updateProjectionMatrix();
  // YAW IS NORMALISED BY THE PANEL'S WIDTH, not its height. OrbitControls r128 rotates by
  // `2*PI * deltaX / element.clientHeight` -- it divides a HORIZONTAL gesture by a VERTICAL
  // measurement. In a tall window that is merely odd; in the dashboard's 3D tab, which is
  // short and wide, it makes the view uncontrollable. MEASURED in the tab at 606x418: a
  // 100 px drag turned the camera 86.1 degrees and a full turn took 418 px, so there was no
  // such thing as a small adjustment.
  //
  // It bites HERE and not on the standalone page because the panel is shorter, and it bites
  // at all only because the pitch is pinned (owner decision 2026-09-04, `lockPitch`): with
  // the vertical drag doing nothing by design, yaw is the whole control, and it was the one
  // axis scaled by the wrong dimension.
  //
  // (the yaw normalisation that belongs with this is set at DRAG START, not here -- see
  //  the 'start' handler by the OrbitControls construction. Setting it on resize read the
  //  panel before the maximise transition had settled: MEASURED 0.69 still in force at
  //  1255x684, where it should have been 0.545.)
  // The cytoscape canvas does not follow its container on its own: without this the
  // viewport keeps its old size and the graph drifts out of the pane (measured: 77 of 77
  // nodes inside at load, 39 of 77 after one window resize, 77 again after a reload).
  // Padding 12 matches the cose layout's own, so the fit is the one it arrived with.
  if (CYK) { CYK.resize(); CYK.fit(CYK.nodes(), 12); }
  render();
}

// ---- labels, as an HTML overlay ------------------------------------------------------
// Projected per frame rather than baked into sprites: 84 spans is cheaper than 84 canvas
// textures, and the text stays crisp at every zoom.
const V = new THREE.Vector3();
function drawLabels() {
  if (!show.obj || !show.label) { labelHost.innerHTML = ''; return; }
  const w = renderer.domElement.clientWidth, h = renderer.domElement.clientHeight;
  const out = [];
  for (const m of OBJMESH) {
    if (!m.visible) continue;             // no label for a box on another storey
    V.copy(m.position).project(camera);
    if (V.z > 1) continue;
    const x = (V.x * 0.5 + 0.5) * w, y = (-V.y * 0.5 + 0.5) * h;
    out.push('<span style="left:' + x.toFixed(0) + 'px;top:' + y.toFixed(0) +
             'px;color:hsl(' + m.userData.hue + ',90%,72%)">' +
             String(m.userData.o.label).replace(/[<&]/g, '') + '</span>');
  }
  labelHost.innerHTML = out.join('');
}

// RENDER ON DEMAND. An always-on animation loop spins the GPU for a static diagram; the
// only things that move the picture are the controls, a toggle, a selection and a load.
let queued = false;
function render() {
  if (queued) return;
  queued = true;
  requestAnimationFrame(() => {
    queued = false;
    // EVERY GROUP, from G itself rather than from a list repeated here. The list was
    // hardcoded and did not include the two groups added later (`dwall`, `sched`), so their
    // toggles set a flag that nothing read: the buttons appeared to work and changed nothing.
    // Deriving the loop from G means a group cannot be added and forgotten.
    for (const k of Object.keys(G))
      if (k in show) G[k].visible = show[k];
    // FLOOR MAP IS WHAT YOU SEE WITH THE MESH OFF (owner decision 2026-09-06). Both
    // defaulted on and the GLB is opaque with its own floor at the deck's height -- the
    // deck sits at z 1.21 on hm3d_00861 and the agent walked 1.190 .. 1.217 -- so the plan
    // was built and immediately buried, and FLOOR MAP looked like a button that did
    // nothing. Linked rather than fought with the depth buffer: unchecking MESH reveals
    // the plan, checking it hides the plan again.
    // ponytail: one condition at the single place visibility is already decided, not a
    // second mechanism. Ceiling: with MESH on the FLOOR MAP button still looks dead.
    // Upgrade path is to grey the button out while show.mesh is true.
    G.floor.visible = show.floor && !show.mesh;
    // The crosshair follows the centre and is scaled by the viewing distance, so it keeps
    // roughly one apparent size from across the building and from inside a cupboard.
    XHAIR.position.copy(controls.target);
    XHAIR.scale.setScalar(Math.max(0.05, camera.position.distanceTo(controls.target) * 0.02));
    cutaway();
    renderer.render(scene, camera);
    drawLabels();
  });
}
controls.addEventListener('change', render);
window.addEventListener('resize', resize);

// ---- THE MESH ------------------------------------------------------------------------
// GLTFLoader has never supported GOOGLE_texture_basis -- it knows KHR_texture_basisu only.
// This plugin is GLTFTextureBasisUExtension.loadTexture with the extension name changed
// and a BasisTextureLoader in place of options.ktx2Loader. `_invokeOne` walks the
// registered plugins and takes the first truthy result, so it is reached whether or not
// the asset declares the extension in extensionsUsed (this one does not; it is
// non-conformant and it does not matter).
const basis = new BasisTextureLoader();
// GA-380: the loader fetches the transcoder at RUNTIME, so the prefix proxy never sees this
// path. Measured behind a stand-in proxy: basis_transcoder.js and .wasm both 404ed under
// a path prefix, which leaves every texture untranscoded.
basis.setTranscoderPath(PFX + '/vendor/basis/');
basis.detectSupport(renderer);
// BPTC IS REFUSED, AND ASTC WITH IT, because these are ETC1S textures. BasisTextureLoader
// picks a transcode target by GPU support in the order ASTC, BPTC, DXT, ETC, PVRTC
// (vendor_three/BasisTextureLoader.js:630-677), and BPTC means BC7_M5 -- a target meant for
// UASTC sources. Feeding it ETC1S is what produced the scrambled checkerboard the owner
// photographed twice: whole surfaces rebuilt from small tiles of unrelated texture. The
// canonical ETC1S targets are BC1 and BC3, which is the DXT branch immediately below.
//
// WHY IT WAS NOT SEEN FROM HERE. `detectSupport` asks the GPU, so a headless Chrome on
// swiftshader and a real GPU choose DIFFERENT branches: the mesh looked right in every
// screenshot I took and wrong on the owner's machine, from the same code. A defect that
// depends on the renderer cannot be closed by looking at one renderer.
//
// Two flags, not one: without also refusing ASTC a machine that supports it would take the
// first branch and never reach DXT, which is the same bug wearing a different format. The
// chosen format is printed on the status line, so what actually happened is readable rather
// than assumed.
if (basis.workerConfig) {
  basis.workerConfig.bptcSupported = false;
  basis.workerConfig.astcSupported = false;
}
basis.setWorkerLimit(4);

const loader = new GLTFLoader();
loader.register(parser => ({
  name: 'GOOGLE_texture_basis',
  loadTexture(i) {
    const d = parser.json.textures[i];
    const e = d.extensions && d.extensions.GOOGLE_texture_basis;
    return e ? parser.loadTextureImage(i, parser.json.images[e.source], basis) : null;
  },
}));

const T0 = performance.now();
say('mesh', 'mesh: requesting ' + MESH_URL);

// ponytail: the fallback is a per-vertex height ramp over LOCAL z, decided before the
// loader was written. Ceiling: it is local z, not world z, so a GLB whose nodes carry
// their own translations would band wrongly. hm3d chunk nodes carry a mesh and nothing
// else, so local z is world z here. Upgrade path is one applyMatrix4 per node.
function heightRamp(root, zlo, zhi) {
  const c = new THREE.Color();
  const span = Math.max(1e-6, zhi - zlo);
  root.traverse(n => {
    if (!n.isMesh) return;
    const pos = n.geometry.attributes.position;
    const col = new Float32Array(pos.count * 3);
    for (let i = 0; i < pos.count; i++) {
      const t = Math.min(1, Math.max(0, (pos.getZ(i) - zlo) / span));
      c.setHSL(0.62 - 0.62 * t, 0.70, 0.24 + 0.42 * t);
      col[i * 3] = c.r; col[i * 3 + 1] = c.g; col[i * 3 + 2] = c.b;
    }
    n.geometry.setAttribute('color', new THREE.BufferAttribute(col, 3));
    n.material = new THREE.MeshBasicMaterial({vertexColors: true});
  });
}

loader.load(MESH_URL, g => {
  const root = g.scene;
  // THE TRANSFORM. map = (glb.y, -glb.x, glb.z), i.e. a rotation of -90 degrees about the
  // world z axis and nothing else -- no scale, no translation, no Y-up swizzle. The GLB is
  // Z-UP: hm3d_annotated_basis.scene_dataset_config.json declares up [0,0,1] for every
  // stage. Habitat rotates it to its own Y-up at load; tools/extract_gt.py::hab_to_ros
  // rotates back. The two compose to a pure rotation about the file's own z.
  root.rotation.set(0, 0, -Math.PI / 2);
  root.updateMatrixWorld(true);

  let tris = 0, textured = 0, fmt = '';
  const mats = new Set();
  let raggedMips = 0;   // compressed textures whose mip chain is not 4x4-block aligned
  root.traverse(n => {
    if (!n.isMesh) return;
    const a = n.geometry.attributes.position;
    tris += (n.geometry.index ? n.geometry.index.count : (a ? a.count : 0)) / 3;
    for (const m of [].concat(n.material)) {
      if (mats.has(m)) continue;
      mats.add(m);
      if (!m.map) continue;
      textured++;
      fmt = fmt || (m.map.isCompressedTexture ? ('compressed 0x' + m.map.format.toString(16))
                                              : 'uncompressed RGBA32');
      // A compressed texture with ONE mip level under a mipmapping minFilter samples as
      // black in WebGL. GLTFLoader overwrites minFilter from the glTF sampler (9986 here,
      // NearestMipmapLinear), so the guard has to come after it, not inside the loader.
      //
      // AND A MIP CHAIN THAT IS NOT BLOCK-ALIGNED SAMPLES AS GARBAGE. These are BC7
      // (COMPRESSED_RGBA_BPTC_UNORM, 0x8e8c), which stores 4x4 blocks: a level whose width
      // or height is not a multiple of 4 has no whole-block representation, and sampling it
      // returns neighbouring blocks in the wrong order -- the scrambled checkerboard the
      // owner photographed on a distant wall, which is exactly where the small mips are
      // used. The single-mip case above was already handled; this is the same fault one
      // level further in, and it only shows at a distance, which is why it read as "the
      // textures look weird" rather than as a broken texture.
      //
      // Dropping to LinearFilter uses level 0 alone: slightly more aliasing far away, and
      // no garbage. Kept per-texture rather than applied to all of them, so the textures
      // whose chains ARE aligned keep their mipmaps.
      const mips = m.map.mipmaps || [];
      const ragged = mips.length > 1 && mips.some(
        lv => lv && ((lv.width % 4) !== 0 || (lv.height % 4) !== 0));
      if (mips.length < 2 || (m.map.isCompressedTexture && ragged)) {
        m.map.minFilter = THREE.LinearFilter;
        m.map.needsUpdate = true;
        if (ragged) raggedMips++;
      }
    }
  });

  const box = new THREE.Box3().setFromObject(root);
  const got = [box.min.x, box.min.y, box.min.z, box.max.x, box.max.y, box.max.z];
  const fx = v => v.map(q => q.toFixed(3)).join(', ');
  let verdict, cls;
  if (!P.gt_aabb) {
    // SKIPPED is not PASSED (working rule 2). There is no ground truth for this scene, so
    // nothing here has checked the transform and the line must not imply that it did.
    verdict = 'transform NOT CHECKED (no ground truth for this scene) box [' + fx(got) + ']';
    cls = 'warn';
  } else {
    const exp = P.gt_aabb.lo.concat(P.gt_aabb.hi);
    let worst = 0;
    for (let i = 0; i < 6; i++) worst = Math.max(worst, Math.abs(got[i] - exp[i]));
    if (worst <= 0.01) {
      verdict = 'transform VERIFIED vs GT AABB, worst axis error ' + worst.toFixed(4) + ' m';
      cls = 'ok';
    } else {
      verdict = 'MESH TRANSFORM WRONG -- worst axis error ' + worst.toFixed(3) +
                ' m. mesh [' + fx(got) + '] vs ground truth [' + fx(exp) + ']';
      cls = 'bad';
    }
  }

  if (textured === 0 && mats.size) {
    // Every material here is KHR_materials_unlit with baseColorFactor [1,1,1,1], so a
    // texture-less mesh is a flat white blob with no depth cue whatever. Colour by height
    // instead, and SAY that the textures are missing -- a fallback that looks like a
    // success is the defect working rule 14 names.
    heightRamp(root, box.min.z, box.max.z);
    cls = 'bad';
  }

  // THE MATERIALS, KEPT. Selection dims the mesh, and re-traversing 408,722 triangles on
  // every click to find 57 materials is work already done. Collected HERE and not from the
  // `mats` set above, because the height-ramp fallback replaces every material after that
  // set was built -- so `mats` can hold materials no longer attached to anything.
  const keep = new Set();
  root.traverse(n => { if (n.isMesh) for (const m of [].concat(n.material)) keep.add(m); });
  MESHMATS = [...keep];

  G.mesh.add(root);
  const secs = ((performance.now() - T0) / 1000).toFixed(1);
  say('mesh', 'mesh: ' + Math.round(tris).toLocaleString() + ' triangles, ' +
      textured + ' of ' + mats.size + ' materials textured' +
      // SAID OUT LOUD, because a silent downgrade is how the original fault survived: the
      // page reported "57 of 57 materials textured" while some of them were sampling
      // garbage from a ragged mip chain. If this number is non-zero the mesh is sharp at
      // level 0 and slightly aliased far away, which is the trade being made.
      (raggedMips ? ' (' + raggedMips + ' mip chains not block-aligned, using level 0)' : '') +
      (textured ? ' (' + fmt + ')' : ' -- HEIGHT-RAMP FALLBACK, textures did NOT transcode') +
      ', ' + secs + ' s. ' + verdict, cls);
  // A selection made during the ~1.2 s load ran dimMesh over an EMPTY MESHMATS, so the
  // mesh would arrive fully lit with a box still selected. Repaint from the live SEL now
  // that there are materials to paint.
  paintSelection();
  // The clipping planes are per MATERIAL, and MESHMATS was empty until this line above.
  // A storey picked during the ~1.2 s load would otherwise leave the mesh unclipped.
  applyFloor();
  render();
}, xhr => {
  if (xhr.total) say('mesh', 'mesh: ' + (xhr.loaded / 1048576).toFixed(1) + ' / ' +
                     (xhr.total / 1048576).toFixed(1) + ' MB downloaded');
}, err => {
  say('mesh', 'MESH FAILED: ' + (err && err.message ? err.message : String(err)) +
      ' (' + MESH_URL + ')', 'bad');
});

// ---- picking -------------------------------------------------------------------------
const ray = new THREE.Raycaster();
const ndc = new THREE.Vector2();
function hitsAt(ev) {
  if (!show.obj) return [];
  const r = cv.getBoundingClientRect();
  ndc.set(((ev.clientX - r.left) / r.width) * 2 - 1, -((ev.clientY - r.top) / r.height) * 2 + 1);
  ray.setFromCamera(ndc, camera);
  const seen = new Set(), out = [];
  for (const h of ray.intersectObjects(OBJMESH, false)) {
    if (!h.object.visible) continue;      // a box the storey picker hid is not pickable
    if (seen.has(h.object.id)) continue;
    seen.add(h.object.id);
    out.push(h);
  }
  return out;
}

function rayToPlane(ev, plane, out) {
  const r = cv.getBoundingClientRect();
  ndc.set(((ev.clientX - r.left) / r.width) * 2 - 1, -((ev.clientY - r.top) / r.height) * 2 + 1);
  ray.setFromCamera(ndc, camera);
  return ray.ray.intersectPlane(plane, out) !== null;
}

// ---- MIDDLE-DRAG DRAGS THE MAP -------------------------------------------------------
// Owner, 2026-09-08: "dragging with center mouse should not reset the cursor but rather drag the
// map (therefore when center-dragging we are following the cursor pivot moving)".
//
// So the grabbed world point stays UNDER THE CURSOR for the whole drag, and the camera and the
// orbit centre travel together by the same vector. The scene slides with the hand, which is the
// gesture every map has; the crosshair keeps its place in the scene rather than jumping to the
// pointer, so nothing is "reset" underneath the drag.
//
// It moved the ORBIT CENTRE ALONE before, holding the camera still: that reads as the cross
// running away from a stationary picture, which is the behaviour this replaces. Right-drag pan
// remains OrbitControls' own screen-space pan; this one is a drag across the floor plane, which is
// what makes it track the cursor exactly rather than approximately.
const DRAGPLANE = new THREE.Plane(new THREE.Vector3(0, 0, 1), 0);
const dragFrom = new THREE.Vector3(), dragAt = new THREE.Vector3();
let dragging = false;
cv.addEventListener('pointerdown', e => {
  if (e.button !== 1) return;
  e.preventDefault();
  DRAGPLANE.constant = -controls.target.z;
  // Edge on to the plane there is no grab point, so refuse rather than jump to infinity.
  if (!rayToPlane(e, DRAGPLANE, dragFrom)) return;
  dragging = true;
});
// ON WINDOW, not on the canvas, and not via setPointerCapture. The drag has to survive the
// cursor leaving the canvas -- capture would do that too, but it THROWS on a pointerId that
// is not active, which is exactly the case a synthetic-event test produces, and a handler
// that only works under a real mouse cannot be exercised.
const dragShift = new THREE.Vector3();
window.addEventListener('pointermove', e => {
  if (!dragging || !rayToPlane(e, DRAGPLANE, dragAt)) return;
  // The vector that puts the grabbed point back under the cursor. Applied to the CAMERA AND the
  // target together, so the view direction and the distance are untouched and only the world
  // slides: no re-aim, no zoom, no drift.
  dragShift.subVectors(dragFrom, dragAt);
  camera.position.add(dragShift);
  controls.target.add(dragShift);
  holdAim();
  // NOT controls.update(): nothing angular changed, and update() would re-aim the camera.
  render();
});
window.addEventListener('pointerup', () => { dragging = false; });
window.addEventListener('pointercancel', () => { dragging = false; });
// Middle-click on some browsers opens autoscroll or pastes; the drag has consumed it.
cv.addEventListener('auxclick', e => { if (e.button === 1) e.preventDefault(); });

// ---- DOUBLE-CLICK FLIES TO THE BOX ---------------------------------------------------
// Animated, not teleported. A teleport leaves the viewer working out where the camera went;
// half a second of travel shows them.
let flight = null;
const QNONE = new THREE.Quaternion();
function flyStep() {
  if (!flight) return;
  const k = Math.min(1, (performance.now() - flight.t0) / flight.ms);
  const e = k < 0.5 ? 2 * k * k : 1 - Math.pow(-2 * k + 2, 2) / 2;   // ease in, ease out
  // A flight ENDS on the box, so it also ends with the camera facing the centre again --
  // the aim offset a centre drag left behind is eased out over the same half second
  // instead of being dropped in one frame, which would be the snap this all avoids.
  if (flight.off0) AIMOFF = k < 1 ? flight.off0.clone().slerp(QNONE, e) : null;
  camera.position.lerpVectors(flight.p0, flight.p1, e);
  controls.target.lerpVectors(flight.c0, flight.c1, e);
  controls.update();                       // its 'change' event calls render()
  if (k < 1) requestAnimationFrame(flyStep); else flight = null;
}
function flyTo(centre, radius) {
  // MARGIN. radius / tan(fov/2) is the distance at which the box's bounding sphere exactly
  // fills the vertical field of view; 1.6x of that leaves the box framed with room round it.
  const d = Math.max(0.5, radius / Math.tan((camera.fov / 2) * Math.PI / 180) * 1.6);
  // Approach along the CURRENT view direction, so the flight is a move in and not a spin
  // round to a canned viewpoint. The viewer keeps their bearings because the bearing is
  // theirs. Read off the camera, not from camera.position - controls.target: with the
  // centre dragged off the view axis those two are no longer the same direction, and it
  // is the one the viewer is LOOKING along that has to be preserved. With no offset they
  // are identical, so this changes nothing about the flight round 2 measured.
  const dir = camera.getWorldDirection(new THREE.Vector3()).negate();
  flight = {t0: performance.now(), ms: 500, off0: AIMOFF ? AIMOFF.clone() : null,
            p0: camera.position.clone(), c0: controls.target.clone(),
            p1: centre.clone().addScaledVector(dir, d), c1: centre.clone()};
  flyStep();
}

// ---- ONE selection, three views ------------------------------------------------------
// SEL holds an object id. select() is the only thing that writes it, and it updates all
// three views from that single value.
let SEL = null;
let CYK = null;

function kgIdFor(id) { return 'n_' + id; }

// ponytail: SELECTION DIMS THE WHOLE MESH, not the part of it around the box. The request
// was "for that box only"; the mesh is 57 SHARED materials over chunk nodes that do not
// follow room boundaries, so there is no per-box slice of it to address at all. The
// CEILING, named: a genuinely local effect needs a clipping volume or a shader keyed on
// distance to the selected box. The stated purpose -- read the box without hiding the room
// -- is met by dropping the whole mesh back, and the room stays on screen.
//
// The originals are read off the material the first time it is dimmed, so restoring is
// restoring rather than guessing. `transparent` and `depthWrite` both have to move: an
// opacity below 1 on a material still marked opaque does nothing at all.
function dimMesh(on) {
  for (const m of MESHMATS) {
    if (m.userData.o0 === undefined) {
      m.userData.o0 = m.opacity;
      m.userData.t0 = m.transparent;
      m.userData.d0 = m.depthWrite;
    }
    m.transparent = on ? true : m.userData.t0;
    // 0.55, not 0.18. At 0.18 the room dissolved and the box floated in the dark, which is
    // the opposite of the request -- the point is to read the box WITHOUT losing the room.
    m.opacity = on ? 0.55 : m.userData.o0;
    m.depthWrite = on ? false : m.userData.d0;
    m.needsUpdate = true;
  }
}

function paintSelection() {
  const any = !!SEL;
  for (const m of OBJMESH) {
    const o = m.userData.o;
    const is = any && (o.id || o.label) === SEL;
    m.material.opacity = is ? 0.78 : (any ? 0.10 : 0.30);
    m.userData.edge.material.color.set(is ? 0xffffff : m.userData.baseColour);
    m.userData.edge.material.opacity = is ? 1.0 : (any ? 0.30 : 1.0);
    // THE SELECTED BOX'S READABILITY IS NOT LEFT TO THE TRANSPARENT SORT. three orders that
    // queue by each object's centre depth, and dimming the mesh adds 57 more objects to it
    // -- the same sort that already had to be worked around for the floor deck. So the one
    // selected box stops depth-testing and draws last instead: it is legible through the
    // wall in front of it whatever the sort decides.
    m.material.depthTest = !is;
    m.userData.edge.material.depthTest = !is;
    m.renderOrder = is ? 997 : 0;
    m.userData.edge.renderOrder = is ? 997 : 0;
  }
  dimMesh(any);
}

function select(id, source) {
  SEL = (SEL === id) ? null : id;

  for (const tr of document.querySelectorAll('#objBody tr'))
    tr.classList.toggle('sel', tr.dataset.id === SEL);
  const row = SEL && document.querySelector('#objBody tr[data-id="' + CSS.escape(SEL) + '"]');
  if (row && source !== 'table') row.scrollIntoView({block: 'nearest'});

  if (CYK) {
    let hit = null;
    CYK.batch(() => {
      CYK.elements().removeClass('sel dim');
      if (SEL) {
        const n = CYK.getElementById(kgIdFor(SEL));
        if (n && n.length) {
          CYK.elements().addClass('dim');
          n.removeClass('dim').addClass('sel');
          n.connectedEdges().removeClass('dim');
          n.neighborhood().removeClass('dim');
          hit = n;
        }
      }
    });
    // The KG pane needs the same treatment the table row already gets, for the same
    // reason. The cose layout puts 16 of the 77 nodes below the pane's 347 px bottom
    // edge, so selecting one of those lit the 3D box and the table row and left the
    // graph looking like it had failed -- the node carried `sel` the whole time and was
    // off-screen. Recentre ONLY when the node is outside the viewport, so a selection
    // already in view does not jump the pane. Outside the batch on purpose: `center`
    // is a viewport change, and a batch defers rendering.
    if (hit && source !== 'graph') {
      const rp = hit.renderedPosition();
      if (rp.x < 12 || rp.y < 12 || rp.x > CYK.width() - 12 || rp.y > CYK.height() - 12)
        CYK.center(hit);
    }
  }
  // ZOOM ON EVERY SELECTION THAT IS NOT ALREADY IN THE 3D VIEW. Owner decision 2026-09-04:
  // clicking a knowledge-graph node or a table row must move the camera onto the object, the
  // way a double-click in the scene already does. A click in the scene is left alone -- you
  // are already looking at what you clicked, and flying on every scene click would make the
  // view lurch while picking through a pile of boxes.
  if (SEL && source !== 'scene') {
    const m = OBJMESH.find(x => (x.userData.o.id || x.userData.o.label) === SEL);
    if (m) {
      const b = m.userData.o.box;
      flyTo(new THREE.Vector3((b[0] + b[3]) / 2, (b[1] + b[4]) / 2, (b[2] + b[5]) / 2),
            0.5 * Math.hypot(b[3] - b[0], b[4] - b[1], b[5] - b[2]));
    }
  }
  paintSelection();
  render();
}

function buildTable() {
  const tb = document.getElementById('objBody');
  tb.innerHTML = P.objects.map(o => {
    const b = o.box;
    const sz = [(b[3] - b[0]), (b[4] - b[1]), (b[5] - b[2])].map(v => v.toFixed(2)).join(' x ');
    const id = o.id || o.label;
    return '<tr data-id="' + id + '"><td>' + o.label + '</td><td>' +
           (o.room || '-') + '</td><td>' + sz + '</td></tr>';
  }).join('');
  tb.querySelectorAll('tr').forEach(tr => { tr.onclick = () => select(tr.dataset.id, 'table'); });
}

function initKG() {
  const host = document.getElementById('kg');
  if (typeof cytoscape !== 'function') {
    host.innerHTML = '<div style="padding:10px;color:#f87171;font-size:11px">' +
      'knowledge graph unavailable: cytoscape did not load from /viewer/cytoscape.min.js</div>';
    return;
  }
  fetch('/graph_data').then(r => r.json()).then(d => {
    const els = (d.elements && d.elements.nodes) ? d.elements : null;
    if (!els) {
      host.innerHTML = '<div style="padding:10px;color:#eab308;font-size:11px">' +
        'no graph in this bundle</div>';
      return;
    }
    // THE DASHBOARD'S OWN STYLESHEET, lifted from viewer.html at page build (see
    // `_viewer_kg_style`). This page used to carry a second, thinner one -- 12 px nodes,
    // hairline haystack edges, no colour per relation -- so the same graph looked like a
    // different product depending on which panel you opened it in (owner report). The two
    // selectors below are appended, not merged into that file: `.dim` and `.sel` are this
    // page's own selection mechanics and mean nothing on the dashboard.
    CYK = cytoscape({
      container: host, elements: els,
      style: __KGSTYLE__.concat([
        {selector: '.dim', style: {'opacity': 0.12}},
        {selector: '.sel', style: {'background-color': '#38bdf8', 'width': 20, 'height': 20,
          'border-width': 2, 'border-color': '#e2e8f0', 'font-size': '10px',
          'color': '#ffffff', 'z-index': 99}},
      ]),
      layout: {name: 'cose', animate: false, numIter: 250, nodeRepulsion: 9000,
               idealEdgeLength: 40, padding: 12},
    });
    CYK.on('tap', 'node', ev => {
      const id = String(ev.target.id()).replace(/^n_/, '');
      select(id, 'graph');
    });
    if (__KGWARN__) {
      const w = document.createElement('div');
      w.style.cssText = 'padding:4px 8px;color:#eab308;font-size:10px';
      w.textContent = __KGWARN__;
      host.parentElement.insertBefore(w, host);
    }
  }).catch(e => {
    host.innerHTML = '<div style="padding:10px;color:#f87171;font-size:11px">' +
      'knowledge graph unavailable: ' + e.message + '</div>';
  });
}

// A click selects; a drag does not. OrbitControls also fires click at the end of an orbit,
// so the pointer travel is measured and a click that moved is discarded.
let downAt = null;
cv.addEventListener('pointerdown', e => { downAt = [e.clientX, e.clientY]; });
cv.addEventListener('click', e => {
  if (downAt && Math.hypot(e.clientX - downAt[0], e.clientY - downAt[1]) > 4) return;
  const h = hitsAt(e);
  select(h.length ? (h[0].object.userData.o.id || h[0].object.userData.o.label) : null, 'scene');
});

// Double-click moves AND zooms the camera onto the box under the cursor, over half a
// second. It still re-centres the orbit, because the centre is the flight's destination.
cv.addEventListener('dblclick', e => {
  const h = hitsAt(e);
  if (!h.length) return;
  const b = h[0].object.userData.o.box;
  flyTo(new THREE.Vector3((b[0] + b[3]) / 2, (b[1] + b[4]) / 2, (b[2] + b[5]) / 2),
        0.5 * Math.hypot(b[3] - b[0], b[4] - b[1], b[5] - b[2]));
  select(h[0].object.userData.o.id || h[0].object.userData.o.label, 'scene');
});

// Hover names the nearest box AND everything else the ray passes through, because a pile
// of boxes at one spot is what a duplicate detection looks like and this is where you
// would notice it.
cv.addEventListener('pointermove', e => {
  const r = cv.getBoundingClientRect();
  if (dragging) { tip.style.display = 'none'; return; }
  const h = hitsAt(e);
  if (!h.length) { tip.style.display = 'none'; return; }
  const o = h[0].object.userData.o;
  const sz = [(o.box[3] - o.box[0]), (o.box[4] - o.box[1]), (o.box[5] - o.box[2])]
    .map(v => v.toFixed(2)).join(' x ');
  let html = o.label + '<br>' + sz + ' m' + (o.room ? '<br>room ' + o.room : '');
  if (h.length > 1) {
    const others = h.slice(1, 6).map(q => q.object.userData.o.label);
    const more = h.length - 1 - others.length;
    html += '<br><span style="color:#eab308">+' + (h.length - 1) + ' behind:</span> ' +
            others.join(', ') + (more > 0 ? ', +' + more : '');
  }
  tip.style.display = 'block';
  tip.style.left = (e.clientX - r.left + 12) + 'px';
  tip.style.top = (e.clientY - r.top + 12) + 'px';
  tip.innerHTML = html;
});
cv.addEventListener('pointerleave', () => { tip.style.display = 'none'; });

function tog(k, el) { show[k] = !show[k]; el.classList.toggle('on', show[k]); render(); }
document.getElementById('bObj').onclick = e => tog('obj', e.target);
document.getElementById('bWall').onclick = e => tog('wall', e.target);
document.getElementById('bDWall').onclick = e => tog('dwall', e.target);
document.getElementById('bSched').onclick = e => tog('sched', e.target);
document.getElementById('bOpen').onclick = e => tog('open', e.target);
document.getElementById('bPath').onclick = e => tog('path', e.target);
document.getElementById('bFloor').onclick = e => tog('floor', e.target);
document.getElementById('bMesh').onclick = e => tog('mesh', e.target);
document.getElementById('bCut').onclick = e => { CUT = !CUT;
  // applyFloor, not just render: CUT now decides whether the cutout plane is IN the
  // materials' clippingPlanes at all, and that membership is set there. Repainting alone
  // left the mesh clipped after the button said the cutout was off.
  e.target.classList.toggle('on', CUT); applyFloor(); render(); };
document.getElementById('bCam').onclick = e => tog('robot', e.target);
document.getElementById('bLabel').onclick = e => { show.label = !show.label;
  e.target.classList.toggle('on', show.label); render(); };
document.getElementById('bReset').onclick = () => { fitView(); lockPitch(); render(); };


// ======================================================================================
// REAL TIME: the object list grows, and the camera and the body are drawn in the mesh
// ======================================================================================
// WHY THIS EXISTS. Until 2026-09-04 this page was an ARCHIVE VIEWER wearing a live name.
// The payload was substituted into the script at build time, so the scene was a snapshot
// for the life of the tab; and /scene3d was registered only on the REPLAY app, so the one
// mode with a camera to draw could not open the page at all. Both are fixed on the server
// (replay_server._install_launcher). This block is the browser half.
//
// EVERY FETCH IS SAME-ORIGIN. The live bridge's port is NOT fixed -- it was 8085 on
// 2026-09-04 because 8081 is held by an unrelated project -- so nothing here builds a
// bridge URL. /mode_info names the mode; the live app's catch-all forwards the rest.
const BUNDLE = __BUNDLE__;
const liveStat = document.getElementById('liveStat');
const LIVE = {mode: null, polls: 0, changes: 0, added: 0, gone: 0, pose: 0,
              source: null, why: 'not asked yet', age: null, stale: null, res: null};

function liveLine() {
  const m = LIVE.mode ? LIVE.mode.toUpperCase() : 'asking /mode_info';
  const pose = LIVE.source
    ? ('pose from ' + LIVE.source + (LIVE.stale ? ' -- STALE, an ARCHIVED reading ' +
        (LIVE.age === null ? '' : 'recorded ' + (LIVE.age / 3600).toFixed(1) + ' h ago') +
        ', NOT a live one' : '') +
       (LIVE.age === null ? '' : ' (' + LIVE.age.toFixed(0) + ' s old)') +
       (LIVE.res ? ' -- frustum for ' + LIVE.res[0] + 'x' + LIVE.res[1] : ''))
    : ('NO CAMERA POSE: ' + LIVE.why);
  liveStat.innerHTML =
    '<b>' + m + '</b> &middot; objects polled ' + LIVE.polls + 'x, ' + LIVE.changes +
    ' change' + (LIVE.changes === 1 ? '' : 's') + ' (+' + LIVE.added + ' / -' + LIVE.gone +
    ') &middot; ' + pose +
    ' &middot; <b class="warn">CAMERA</b> = frustum from the recorded pose and ' +
    'calibration.json. <b class="warn">BODY</b> = a 1.5 m placeholder cylinder &mdash; ' +
    'NO ROBOT MODEL EXISTS ON THIS MACHINE. This is not a render of the robot.';
}
liveLine();

// ---- THE OBJECT LIST GROWS -----------------------------------------------------------
// Reconcile by key. A run in progress rewrites persistent_perception.json periodically, so
// boxes APPEAR, and a merge can move or remove one. Nothing is rebuilt when nothing
// changed: the common case walks 75 keys, compares 75 box arrays and returns false, which
// is why this can run every 3 s without fighting the renderer.
function reconcile(q) {
  let added = 0, moved = 0, gone = 0;
  const keys = new Set();
  for (const o of q.objects) {
    const k = objKey(o);
    keys.add(k);
    const m = OBJBY.get(k);
    if (!m) { addObject(o); added++; continue; }
    if (String(m.userData.o.box) !== String(o.box)) { dropObject(k); addObject(o); moved++; }
    else { m.userData.o = o; m.userData.floor = o.floor; }
  }
  for (const k of [...OBJBY.keys()]) if (!keys.has(k)) { dropObject(k); gone++; }
  LIVE.added += added; LIVE.gone += gone;
  if (!(added || moved || gone)) return false;
  P.objects = q.objects;
  // ponytail: the STOREY SET is not re-derived from a poll, only its counts. It comes from
  // 300 random navmesh samples and moves between runs, so re-deriving it mid-session would
  // move the picker under the viewer's hand and could renumber the storey they are looking
  // at. Ceiling: a run that discovers a second storey after the page opened keeps the
  // storey list it loaded with, and the line below says when the two stop agreeing.
  if (q.floors && q.floors.length === P.floors.length) P.floor_counts = q.floor_counts;
  else LIVE.why = 'the storey set changed since this page loaded -- reload to re-pick';
  indexFloors();
  buildTable();
  buildFloorBar();
  applyFloor();
  paintSelection();
  return true;
}

// ---- THE CAMERA, AS A FRUSTUM, AND THE BODY, AS A LABELLED PLACEHOLDER ----------------
// THE CONVENTION IS NOT RE-DERIVED HERE. `camera_quat_xyzw` is the OPTICAL frame -- the
// camera looks down +Z with +Y DOWN in image space, which is NOT three's own -Z-forward
// default -- and live_overlay.py owns that fact, having measured the two alternatives at
// 1630 px and at nothing-in-front-of-the-camera. The four corner rays come from
// live_overlay.frustum_rays over the /overlay_pose route, IN THE CAMERA FRAME, and the
// only thing done here is the rotation. `camToWorld` is quat_to_R transliterated, entry
// for entry, rather than handed to THREE.Quaternion: a frustum pointing the wrong way
// looks entirely plausible, so the page must be reading the same nine numbers the
// projection does and not three's internal convention.
const CAMV = new THREE.Vector3();
function camToWorld(q, v, out) {
  const x = q[0], y = q[1], z = q[2], w = q[3];
  const r00 = 1 - 2 * (y * y + z * z), r01 = 2 * (x * y - z * w), r02 = 2 * (x * z + y * w);
  const r10 = 2 * (x * y + z * w), r11 = 1 - 2 * (x * x + z * z), r12 = 2 * (y * z - x * w);
  const r20 = 2 * (x * z - y * w), r21 = 2 * (y * z + x * w), r22 = 1 - 2 * (x * x + y * y);
  return out.set(r00 * v[0] + r01 * v[1] + r02 * v[2],
                 r10 * v[0] + r11 * v[1] + r12 * v[2],
                 r20 * v[0] + r21 * v[1] + r22 * v[2]);
}

// SENSOR_HEIGHT, not a guess: habitat_feed_host.py sets SENSOR_HEIGHT = 1.5 and puts the
// sensor at [0, 1.5, 0] on the agent, so the camera sits 1.5 m above the agent's foot.
// The RADIUS is habitat_sim AgentConfiguration's DEFAULT of 0.1 m -- a default, not a
// measurement; habitat_feed_host does not set it.
const SENSOR_H = 1.5, BODY_R = 0.1;
let CAMOBJ = null;
function drawRobot(pos, quat, rays) {
  if (CAMOBJ) {
    G.robot.remove(CAMOBJ);
    CAMOBJ.traverse(n => { if (n.geometry) n.geometry.dispose(); });
    CAMOBJ = null;
  }
  if (!pos) { render(); return; }
  const g = new THREE.Group();

  // THE FRUSTUM. Apex at the camera, four corner rays out to 2 m, and the rim joining
  // them, so it reads as a pyramid of view rather than as a stick. Drawn only when the
  // rays arrived: with no calibration there is no frustum and a made-up field of view
  // would look exactly like a measured one.
  if (rays && rays.length === 4) {
    const apex = new THREE.Vector3(0, 0, 0);
    const c = rays.map(r => camToWorld(quat, [r[0] * 2, r[1] * 2, r[2] * 2],
                                       new THREE.Vector3()));
    const pts = [];
    for (let i = 0; i < 4; i++) {
      pts.push(apex.clone(), c[i].clone());
      pts.push(c[i].clone(), c[(i + 1) % 4].clone());
    }
    // The TOP edge of the image, marked, so the frustum's roll is readable. Corner 0 is
    // pixel (0,0) and corner 1 is (width,0) -- the top of the image, because +Y is DOWN.
    const top = new THREE.LineSegments(
      new THREE.BufferGeometry().setFromPoints([c[0].clone(), c[1].clone()]),
      new THREE.LineBasicMaterial({color: 0xfacc15}));
    g.add(new THREE.LineSegments(new THREE.BufferGeometry().setFromPoints(pts),
      new THREE.LineBasicMaterial({color: 0x38bdf8})));
    g.add(top);
  }

  // THE BODY. A wireframe cylinder, one flat colour, no head, no wheels, no articulation,
  // because a shaded torso reads as a model whatever the caption says. It hangs from the
  // camera down by SENSOR_H, which is where the agent's foot is.
  const cyl = new THREE.CylinderGeometry(BODY_R, BODY_R, SENSOR_H, 12, 1, true);
  const body = new THREE.LineSegments(new THREE.EdgesGeometry(cyl),
    new THREE.LineBasicMaterial({color: 0x94a3b8, transparent: true, opacity: 0.75}));
  cyl.dispose();
  body.rotation.x = Math.PI / 2;                 // three builds a cylinder up +Y; here z is up
  body.position.set(0, 0, -SENSOR_H / 2);
  g.add(body);

  g.position.set(pos[0], pos[1], pos[2]);
  G.robot.add(g);
  CAMOBJ = g;
  render();
}

// ---- THE TWO POLLS -------------------------------------------------------------------
// 3 s for the object list and 1 s for the pose. Both are render-on-demand: neither calls
// render() unless something actually changed, so the idle page still draws no frames.
// The pose source is one perception cycle stale by construction even when live, because
// _pose_from_tf has never answered on this project -- see /overlay_pose.
function pollPose() {
  // LIVE asks for a FRESH pose and gets nothing if there is none. REPLAY deliberately asks
  // for the ARCHIVED one, and the status line calls it archived. This is an honest test of
  // the DRAWING through the live code path; it is NOT a test of the live transport.
  // GA-380: built at runtime, so the proxy cannot rewrite it -- 404 under a prefix, and the
  // live camera overlay then silently has no pose to draw with.
  const url = PFX + '/overlay_pose' + (LIVE.mode === 'live' ? '' : '?max_age=1000000000');
  return fetch(url).then(r => r.json()).then(d => {
    LIVE.res = d.resolution && d.resolution[0] ? d.resolution : null;
    if (!d.pose) {
      LIVE.source = null; LIVE.why = d.why || 'the route returned no pose and no reason';
      drawRobot(null); liveLine(); return;
    }
    const pos = d.pose.position, quat = d.pose.quat_xyzw;
    const key = String(pos) + '|' + String(quat);
    LIVE.source = d.source; LIVE.age = d.detections_age_s; LIVE.stale = d.stale;
    liveLine();
    if (key === LIVE._poseKey) return;           // nothing moved; do not redraw
    LIVE._poseKey = key;
    LIVE.pose++;
    drawRobot(pos, quat, d.frustum_rays_cam);
  }).catch(e => {
    LIVE.source = null; LIVE.why = 'fetch /overlay_pose failed: ' + e.message;
    liveLine();
  });
}

function pollObjects() {
  return fetch('/scene3d_data?bundle=' + encodeURIComponent(BUNDLE))
    .then(r => r.json()).then(d => {
      LIVE.polls++;
      if (d.objects && reconcile(d)) { LIVE.changes++; render(); }
      liveLine();
    }).catch(e => { LIVE.why = 'fetch /scene3d_data failed: ' + e.message; liveLine(); });
}

fetch('/mode_info').then(r => r.json()).then(d => { LIVE.mode = d.mode; liveLine(); })
  .catch(() => { LIVE.mode = 'unknown'; liveLine(); })
  .then(() => {
    pollPose();
    pollObjects();
    setInterval(pollPose, 1000);
    setInterval(pollObjects, 3000);
  });

// THE HANDLE THE RENDER VERIFIER NEEDS. This is an ES module, so its scope is private and
// nothing outside it can state where the camera or the orbit centre actually ARE -- only
// that the page looks plausible. MESHMATS and SEL are reassigned, so they are getters.
// GA-345/set 7: the dashboard embeds this page as its 3D tab and can MAXIMIZE that tab into an
// overlay. Its Escape handler is on the PARENT document, and key events do not cross the frame
// boundary, so once focus is in here Escape stopped closing the overlay. Forwarded from inside.
// MEASURED, because the obvious mechanism is the wrong one: an orbit drag does NOT move focus
// (OrbitControls cancels the default action on pointerdown -- after one, the parent's active
// element is still its own button and this document does not have focus). What moves focus is a
// click on a CONTROL in this page's side panel, e.g. a layer toggle, which is an ordinary action.
// Same origin, so this is a direct call; a failure is SAID rather than swallowed, because a dead
// Escape is invisible otherwise.
window.addEventListener('keydown', e => {
  if (e.key !== 'Escape' || window.parent === window) return;
  try {
    if (typeof window.parent.toggleScene3dMax === 'function') window.parent.toggleScene3dMax(false);
  } catch (err) {
    console.warn('[scene3d] could not close the parent overlay: ' + err);
  }
});

// ---- GA-397: EMBEDDED LAYOUT ----------------------------------------------------------------
// Run when this page is a panel inside the dashboard. The controls are MOVED into the overlays,
// never copied: two buttons for one layer is how they drift apart, and every handler already bound
// to these elements keeps working because they are the same elements.
(function embedIfFramed() {
  if (window.parent === window) return;          // opened on its own: the full page is right
  document.body.classList.add('embedded');
  const bars = Array.from(document.querySelectorAll('body > .bar'));
  const layers = document.getElementById('hudLayers');
  const help = document.querySelector('#hudHelp .body');
  const stat = document.getElementById('hudStat');
  bars.forEach(bar => {
    Array.from(bar.children).forEach(el => {
      if (el.tagName === 'BUTTON') layers.appendChild(el);
      else help.appendChild(el);                 // the two legends and the instructions line
    });
    if (bar.textContent.trim()) help.appendChild(bar);   // the storey paragraph, kept but folded away
  });
  // The two status lines report what the renderer and /mode_info are doing. They belong on screen,
  // but as one small line rather than two full-width rows.
  ['meshStat', 'liveStat'].forEach(id => {
    const el = document.getElementById(id);
    if (el) { el.style.padding = '0'; el.style.border = '0'; stat.appendChild(el); }
  });
})();

// The dashboard tells this page when its panel is maximized; the graph and the table are shown
// only then. An explicit call, not a guess from the viewport: a size threshold would be a second
// rule that can disagree with the first.
window.SCENE3D_SET_MAXIMIZED = function (on) {
  document.body.classList.toggle('maxi', !!on);
  // The renderer measures its own box; the panel just changed shape.
  window.dispatchEvent(new Event('resize'));
};

window.SCENE3D = {camera, controls, XHAIR, OBJMESH, show, G, setFloor,
                  meshMats: () => MESHMATS, sel: () => SEL, flying: () => !!flight,
                  aimOff: () => AIMOFF, floor: () => FLOOR, cut: () => CUT,
                  cyk: () => CYK,  // for the browser verification only; read-only use
                  mappedFloor: () => MAPPEDI,
                  cutHidden: () => G.wall.children.filter(m => !m.visible).length,
                  live: () => LIVE, camObj: () => CAMOBJ, objBy: OBJBY,
                  camToWorld, reconcile, pollPose, pollObjects, addObject, dropObject};

resize();
fitView();
lockPitch();          // after the fit, so the pinned pitch is the fitted one
buildTable();
applyFloor();
initKG();
render();
"""


def _scenes_table():
    """The scene id -> GLB path table, read from tools/extract_gt.py.

    ONE table, not a copy: extract_gt.py is what produced the ground truth in data/gt, so a
    second list here could name a different mesh from the one the boxes were measured
    against and nothing would say so.
    """
    p = Path(__file__).resolve().parents[2] / "tools" / "extract_gt.py"
    if not p.is_file():
        # SAY SO RATHER THAN 500. This raised FileNotFoundError deep inside /scene_mesh, so the
        # 3D tab showed a bare framework error page and nothing named the missing file. The mesh
        # is optional -- the 3D view still draws the measured boxes without it -- so an absent
        # extractor must degrade to "no mesh", not to a stack trace.
        raise FileNotFoundError(
            f"the scene table lives in {p}, which is not present. The 3D view draws its boxes "
            f"without a mesh; restore that file to get the ground-truth mesh back.")
    spec = importlib.util.spec_from_file_location("_scene_table", p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.SCENES


def mesh_path(bundle):
    """-> (Path | None, scene_id | None) for this bundle's own scene mesh.

    The scene comes from the bundle's `run_metadata.json`, never from the directory name,
    and an unknown scene returns None rather than a default mesh: a page that silently drew
    a different building's mesh under these boxes would look exactly like a working one.
    """
    scene = _scene_of(RUNS_DIR / bundle)
    if not scene:
        return None, None
    try:
        entry = _scenes_table().get(scene)
    except FileNotFoundError as exc:
        # No scene table in this checkout: the mesh is unavailable, the boxes are not. Returning
        # "no mesh" is the same answer as an unknown scene, which the caller already renders.
        print(f"[scene3d] no ground-truth mesh: {exc}", flush=True)
        return None, scene
    if not entry:
        return None, scene
    p = Path(entry[0])
    return (p if p.is_file() else None), scene


def page(bundle):
    p = scene_payload(bundle)
    n_gt = len(p["walls"])
    n_holes = sum(len(w["holes"]) for w in p["wall_mass"])
    gt_line = (f'{n_gt} wall segments merged into {len(p["wall_mass"])} surfaces, '
               f'{p["openings_cut"]} of {len(p["openings"])} openings cut through them '
               f'as {n_holes} apertures, '
               f'{p["gt_regions"]} regions &mdash; scene {p["gt_scene"]}'
               if n_gt else
               'no ground-truth file for this scene &mdash; walls cannot be drawn')
    # WHICH STOREY THIS RUN MAPPED, with the guard's own evidence, in the page and not in
    # a tooltip. A viewer who picks the other storey sees an empty shell, and without this
    # line the honest reading of that is "the pipeline found nothing".
    g = p["mapped_floor"]
    if g and p["floors"]:
        i = next((k for k, f in enumerate(p["floors"])
                  if abs(f["z"] - g["floor_y"]) < 0.01), None)
        c = p["floor_counts"][i] if i is not None else {}
        floor_line = (
            f'THIS RUN MAPPED ONE STOREY: z = {g["floor_y"]} m. '
            f'floor guard &mdash; tolerance {g.get("tolerance_m")} m, mode {g.get("mode")}, '
            f'{g.get("corrections")} corrections, max drift {g.get("max_drift_m")} m. '
            f'{c.get("objects", 0)} of {len(p["objects"])} measured boxes and '
            f'{c.get("surfaces", 0)} of {len(p["wall_mass"])} ground-truth wall surfaces sit '
            f'on it. The other storeys are ground truth only &mdash; an empty storey here '
            f'means this run never went there, NOT that the pipeline found nothing. '
            f'Floor set: {p["floor_source"]}.')
    elif p["floors"]:
        floor_line = ('WHICH STOREY THIS RUN MAPPED: NOT MEASURED &mdash; this bundle has no '
                      f'stats.floor_guard. Floor set: {p["floor_source"]}.')
    else:
        floor_line = f'STOREYS: {p["floor_source"]}'
    try:
        from found.dashboard import dash_ext as _dx
    except ImportError:
        import dash_ext as _dx
    _brand = _dx.brand()
    # The dashboard's cytoscape stylesheet, so the graph here is the graph there. When it
    # cannot be read the page SAYS so on screen rather than reverting to a second look that
    # nobody would notice was the wrong one -- an empty array styles nothing, which is
    # unmistakable, and `kgStyleWarn` prints the reason beside the graph.
    _kg = _viewer_kg_style()
    _kg_warn = "" if _kg else (
        "the dashboard stylesheet could not be read from viewer.html; "
        "this graph is UNSTYLED rather than silently different")
    js = (_JS.replace("__PAYLOAD__", json.dumps(p))
             .replace("__KGSTYLE__", _kg or "[]")
             .replace("__KGWARN__", json.dumps(_kg_warn))
             .replace("__BUNDLE__", json.dumps(bundle))
             .replace("__MESHURL__", json.dumps("/scene_mesh?bundle=" + bundle)))
    return f"""<title>{_brand} scene &middot; {bundle}</title>
<style>{_CSS}</style>
<header>
  <h1>{_brand} &middot; scene</h1>
  <span class="sub">{bundle} &middot; {len(p["objects"])} measured boxes &middot; {gt_line}</span>
</header>
<div class="bar">
  <button id="bObj" class="on">OBJECTS</button>
  <button id="bWall" class="on">GT WALLS</button>
  <button id="bDWall" class="on">DET WALLS</button>
  <button id="bSched" class="on">SCHEDULE</button>
  <button id="bOpen" class="on">GT OPENINGS</button>
  <button id="bPath" class="on">AGENT PATH</button>
  <button id="bFloor" class="on">FLOOR MAP</button>
  <button id="bMesh" class="on">MESH</button>
  <button id="bCut" class="on">CUTAWAY</button>
  <button id="bCam" class="on">CAMERA + BODY</button>
  <button id="bLabel">LABELS</button>
  <button id="bReset">RESET VIEW</button>
  <span class="legend" style="margin-left:14px">
    left-drag orbit &middot; right-drag pan &middot; wheel zoom &middot;
    <b style="color:#38bdf8">middle-drag drags the map</b> (the point you grab stays under the cursor) &middot;
    click a box dims the mesh &middot; double-click flies to it &middot;
    a graph node or a table row flies to it too &middot;
    <b style="color:#38bdf8">CUTAWAY</b> removes the walls between you and the orbit centre
  </span>
  <span class="legend">
    <i style="background:linear-gradient(90deg,#22c55e,#38bdf8,#a855f7,#f97316)"></i>measured object (hue per label, translucent)
    <i style="background:#93a6c0"></i>ground-truth wall (grey, translucent)
    <i style="background:#eab308"></i>opening
    <i style="background:#38bdf8"></i>camera frustum (measured pose + calibration.json)
    <i style="background:#94a3b8"></i>body &mdash; A PLACEHOLDER, NOT A ROBOT MODEL
  </span>
</div>
<div class="bar" style="color:var(--faint);font-size:10px;display:block">{floor_line}</div>
<div id="meshStat">starting the WebGL renderer&hellip;</div>
<div id="liveStat">asking /mode_info&hellip;</div>
<div id="split">
  <div id="wrap"><canvas id="cv"></canvas><div id="labels"></div><div id="floorPick"></div><div id="tip"></div>
    <div id="hudLayers" class="hud"></div>
    <div id="hudHelp" class="hud closed"><div class="head" onclick="this.parentNode.classList.toggle('closed')">&#9432; HELP &amp; LEGEND</div><div class="body"></div></div>
    <div id="hudStat" class="hud"></div>
  </div>
  <div id="side">
    <div id="kgPane">
      <div class="paneHead">knowledge graph &mdash; click a node to select</div>
      <div id="kg"></div>
    </div>
    <div id="tablePane">
      <div class="paneHead">objects &mdash; click a row to select</div>
      <table id="objTable"><thead><tr><th>label</th><th>room</th><th>size (m)</th></tr></thead>
      <tbody id="objBody"></tbody></table>
    </div>
  </div>
</div>
<!-- GA-380: RELATIVE on purpose. An import map is JSON, so the proxy that adapts this app to a
     path prefix cannot see it (it rewrites literal src="/ and fetch('/ only), and an absolute
     /vendor/... asked the SITE ROOT for three.js under a prefix -- the module then fails to
     resolve and the scene never loads. "./vendor/..." resolves against the document's own
     directory: / at the root, /<prefix>/ behind one. No JS and no server knowledge needed. -->
<script type="importmap">{{"imports": {{"three": "./vendor/three.module.js"}}}}</script>
<script src="/viewer/cytoscape.min.js"></script>
<script type="module">
{js}
</script>
"""


def _check_merge_walls():
    """The merge has to join what is one surface and refuse what is not. Both directions,
    because a merge that joined everything would also 'fix' the picture."""
    def seg(x, y, z, ex, ey, ez, region="r"):
        return {"label": "wall", "pos": [x, y, z], "ext": [ex, ey, ez], "region": region}

    # three touching sections of ONE wall running along y, thin in x
    run = [seg(0.0, 0.0, 1.25, 0.1, 2.0, 2.5),
           seg(0.0, 2.0, 1.25, 0.1, 2.0, 2.5),
           seg(0.0, 4.0, 1.25, 0.1, 2.0, 2.5)]
    m = merge_walls(run)
    assert len(m) == 1, f"one wall in three sections merged to {len(m)}"
    assert abs(m[0]["lo"][1] - (-1.0)) < 1e-6 and abs(m[0]["hi"][1] - 5.0) < 1e-6, m[0]
    assert m[0]["n"] == 3

    # NEGATIVE CONTROLS -- these must NOT merge
    assert len(merge_walls(run + [seg(3.0, 0.0, 1.25, 0.1, 2.0, 2.5)])) == 2, "another plane merged in"
    assert len(merge_walls(run + [seg(0.0, 0.0, 4.25, 0.1, 2.0, 2.5)])) == 2, "another storey merged in"
    assert len(merge_walls(run + [seg(0.0, 9.0, 1.25, 0.1, 2.0, 2.5)])) == 2, "a gap was closed"
    assert len(merge_walls(run + [seg(0.0, 0.0, 1.25, 2.0, 0.1, 2.5)])) == 2, "the other axis merged in"
    print("  merge_walls: 3 sections -> 1 surface; plane, storey, gap and axis all held apart")


def _check_cut_openings():
    """The cut has to open what is an aperture and refuse what is not, and it has to hand
    THREE.ExtrudeGeometry a shape it can triangulate: holes strictly inside the outline and
    never overlapping each other. Both are checked, because a cut that opened everything
    would also 'fix' the picture."""
    wall = [{"label": "wall", "pos": [0.0, 0.0, 1.25], "ext": [0.2, 6.0, 2.5], "region": "r"}]

    def op(x, y, z, ex, ey, ez, label="door"):
        return {"label": label, "pos": [x, y, z], "ext": [ex, ey, ez], "region": "r"}

    # a door in the middle of the wall, reaching the floor
    doors = [op(0.0, 0.0, 1.05, 0.3, 0.9, 2.1)]
    m = cut_openings(merge_walls(wall), doors)
    assert m[0]["axis"] == 0, m[0]["axis"]
    assert len(m[0]["holes"]) == 1, m[0]["holes"]
    h = m[0]["holes"][0]
    assert abs(h[0] - (-0.45)) < 1e-9 and abs(h[2] - 0.45) < 1e-9, h
    assert doors[0]["cut"] is True
    # THE LIP. The door reaches the floor at z=0 and the wall starts at z=0, so an uncut
    # hole would touch the outline. It must be lifted clear of it, and by the lip exactly.
    assert abs(h[1] - 0.01) < 1e-9, h
    assert m[0]["lo"][2] < h[1] < h[3] < m[0]["hi"][2], (m[0]["lo"], h, m[0]["hi"])

    # A DOOR AND ITS OWN DOOR FRAME are two instances over ONE aperture. They must come out
    # as one hole: ExtrudeGeometry triangulates overlapping holes into garbage.
    both = [op(0.0, 0.0, 1.05, 0.3, 0.9, 2.1),
            op(0.0, 0.0, 1.08, 0.25, 1.0, 2.16, "door frame")]
    m = cut_openings(merge_walls(wall), both)
    assert len(m[0]["holes"]) == 1, f"door + frame gave {len(m[0]['holes'])} overlapping holes"
    assert both[0]["cut"] and both[1]["cut"]

    # NEGATIVE CONTROLS -- these must NOT be cut out of this wall
    far = [op(4.0, 0.0, 1.05, 0.3, 0.9, 2.1)]          # a different plane entirely
    assert not cut_openings(merge_walls(wall), far)[0]["holes"], "another plane was cut through"
    assert far[0]["cut"] is False, "an unmatched opening was reported as cut"
    edge = [op(0.0, 3.0, 1.05, 0.3, 0.02, 2.1)]        # 2 cm wide, at the very end
    assert not cut_openings(merge_walls(wall), edge)[0]["holes"], "an edge-on sliver was cut"
    apart = [op(0.0, -2.0, 1.05, 0.3, 0.9, 2.1), op(0.0, 2.0, 1.05, 0.3, 0.9, 2.1)]
    assert len(cut_openings(merge_walls(wall), apart)[0]["holes"]) == 2, "two doorways were joined"

    # A HORIZONTAL SLAB IS ITS OWN CASE (owner decision 2026-09-06), not a vertical pane's
    # degenerate one. The slab is thin in z, so `axis` is 2 and its aperture is cut in
    # (x, y) -- a stairwell hole in a floor. The old x-or-y rule could not express this:
    # it called the slab thin in y and would have cut the hole in (x, z), through 20 cm of
    # thickness, which is not where the hole is.
    slab = [{"label": "floor", "pos": [0.0, 0.0, 2.5], "ext": [6.0, 4.0, 0.2], "region": "r"}]
    hatch = [op(1.0, 0.0, 2.5, 1.2, 1.0, 0.3, "stairs opening")]
    m = cut_openings(merge_walls(slab), hatch)
    assert m[0]["axis"] == 2, m[0]["axis"]
    assert _plane_axes(2) == (0, 1), _plane_axes(2)
    assert len(m[0]["holes"]) == 1, m[0]["holes"]
    h = m[0]["holes"][0]
    assert abs(h[0] - 0.4) < 1e-9 and abs(h[2] - 1.6) < 1e-9, h      # the hatch, in x
    assert abs(h[1] + 0.5) < 1e-9 and abs(h[3] - 0.5) < 1e-9, h      # the hatch, in y
    assert hatch[0]["cut"] is True
    assert m[0]["lo"][0] < h[0] < h[2] < m[0]["hi"][0], (m[0]["lo"], h, m[0]["hi"])
    assert m[0]["lo"][1] < h[1] < h[3] < m[0]["hi"][1], (m[0]["lo"], h, m[0]["hi"])
    # NEGATIVE CONTROL, and it is the one that matters now: a wall-height door standing
    # under the slab must NOT become a hole in the floor. Nothing but the slab-band
    # overlap separates a doorway from a stairwell once both axes can be cut.
    under = [op(1.0, 0.0, 1.05, 0.3, 0.9, 2.1)]
    assert not cut_openings(merge_walls(slab), under)[0]["holes"], "a doorway cut the floor open"
    assert under[0]["cut"] is False, "a doorway under the slab was reported as cut"
    # A WALL IS STILL A WALL: the new rule must not have moved a vertical surface's axis.
    assert cut_openings(merge_walls(wall), doors)[0]["axis"] == 0, "the wall lost its own axis"

    print("  cut_openings: 1 doorway cut, door+frame unified, lip held; plane, sliver "
          "and separate doorways all refused; slab cut in (x, y), door under it refused")


def _check_floors():
    """The picker must separate what is on different storeys, keep what spans both, and
    say NOT MEASURED rather than invent a floor set. All three, because a rule that put
    everything on every storey would also 'fix' the picture."""
    fl = [{"z": -1.59}, {"z": 1.21}]
    assert assign_floor(-1.60, 0.90, fl) == [0], "a lower-storey wall left its storey"
    assert assign_floor(1.15, 3.20, fl) == [1], "an upper-storey wall left its storey"
    # A door sitting 0.35 m above its floor still belongs to it -- the snap is 0.4 m.
    assert assign_floor(1.56, 2.10, fl) == [1], "the snap dropped an item onto the storey below"
    # SPANNING. A stair railing from the lower floor to above the upper one is on both.
    assert assign_floor(-1.64, 4.78, fl) == [0, 1], "a spanning wall was drawn on one storey"
    # NEGATIVE CONTROLS
    assert assign_floor(0.0, 1.0, []) == [], "a floor was assigned with no floor set"
    assert assign_floor(-1.60, 0.90, fl) != [0, 1], "everything landed on every storey"

    assert read_floors(None)[0] == [] and "NOT MEASURED" in read_floors(None)[2]
    bad = read_floors({"floors": [-2.5, 0.5]})
    assert bad[0] == [] and "GA-93" in bad[2], "the hard-coded [-2.5, 0.5] was taken as a floor set"
    flat = read_floors({"floors": [-1.59, 1.21]})
    assert [f["z"] for f in flat[0]] == [-1.59, 1.21] and "NOT AVAILABLE" in flat[2]
    det = read_floors({"floor_detail": {"floors": [{"z": 1.21, "share": 0.49}],
                                        "levels": [{"z": 1.81, "share": 0.02}],
                                        "min_floor_share": 0.1}})
    assert [f["z"] for f in det[0]] == [1.21] and [f["z"] for f in det[1]] == [1.81]
    assert "UNMEASURED" in det[2], "the storey threshold stopped being reported as unmeasured"
    print("  floors: storeys held apart, a spanning item kept on both, [-2.5, 0.5] and a "
          "missing bev_data both refused as NOT MEASURED")


def _check_layers():
    """Every scene group is toggleable, and the visibility loop is derived from the groups.

    WHERE THIS CHECK USED TO SIT, and why that was worthless: in the `__main__` block, AFTER
    `assert a, "no gt_aabb ..."`. On any bundle without ground truth that assertion aborts the
    run first, so these two lines never executed -- they passed a mutation that hardcoded the
    loop again and a mutation that added a group with no flag. A check below the thing that can
    stop the runner is not a check. It needs only the page source, so it runs first now.
    """
    h = page("__layers__")
    # THE VISIBILITY LOOP MUST BE DERIVED FROM G, not from a list repeated beside it. The
    # hardcoded version silently skipped the two groups added after it was written (`dwall`,
    # `sched`), so their buttons toggled a flag nothing read: they looked wired and did nothing.
    assert "for (const k of Object.keys(G))" in h, \
        "render() iterates a hardcoded group list again; a new group will be skipped"
    groups = re.search(r"for \(const k of \[([^\]]+)\]\) \{\n  G\[k\]", h)
    assert groups, "the group list moved; this check is looking at the wrong place"
    names = [x.strip().strip("'\"") for x in groups.group(1).split(",") if x.strip()]
    assert {"wall", "dwall", "sched"} <= set(names), names
    show = re.search(r"const show = \{([^}]+)\}", h)
    assert show, "the show table moved"
    missing = [n for n in names if f"{n}:" not in show.group(1)]
    assert not missing, f"groups with no show flag, so never toggleable: {missing}"
    for btn in ("bObj", "bWall", "bDWall", "bSched", "bOpen", "bPath", "bFloor", "bLabel",
                "bReset", "bMesh"):
        assert f'id="{btn}' in h and f"getElementById('{btn}')" in h, f"{btn} lost its handler"
    # ONE KNOWLEDGE GRAPH, TWO PANELS. The style is read out of viewer.html, so the page must
    # carry viewer.html's own selectors -- not a second stylesheet that merely looks similar.
    # Asserted on selectors this file has never defined itself, so a local copy cannot satisfy
    # it: a relation colour and the concept node are viewer.html's vocabulary alone.
    kg = _viewer_kg_style()
    assert kg, "the dashboard stylesheet could not be read; the graph would render unstyled"
    for sel in ('node[type="concept"]', 'edge[label="supports"]', ':selected'):
        assert sel in kg and sel in h, f"the graph lost the dashboard's {sel} styling"
    assert "'width': 26" in h, "node size no longer matches the dashboard's"
    print("  graph: styled from viewer.html's own stylesheet, not a second copy")
    print(f"  layers: {len(names)} groups, each with a show flag and the loop derived from G")


if __name__ == "__main__":
    import sys
    _check_layers()
    _check_merge_walls()
    _check_cut_openings()
    _check_floors()
    b = sys.argv[1] if len(sys.argv) > 1 else "20260901_174810_hm3d_00861"
    p = scene_payload(b)
    n_holes = sum(len(w["holes"]) for w in p["wall_mass"])
    n_slab = sum(1 for w in p["wall_mass"] if w["axis"] == 2)
    print(f"  objects {len(p['objects'])}  walls {len(p['walls'])} segments "
          f"-> {len(p['wall_mass'])} surfaces  openings {len(p['openings'])} "
          f"-> {p['openings_cut']} cut as {n_holes} apertures ({n_slab} surfaces thin in z)  "
          f"regions {p['gt_regions']}  scene {p['gt_scene']}")
    # WHAT THE PAGE'S TRIANGULATOR CANNOT SURVIVE, asserted on the real scene rather than
    # only on the toy wall: a hole touching the outline, or two holes overlapping.
    for w in p["wall_mass"]:
        u, v = _plane_axes(w["axis"])
        for i, h in enumerate(w["holes"]):
            assert w["lo"][u] < h[0] < h[2] < w["hi"][u], (w["lo"], h, w["hi"])
            assert w["lo"][v] < h[1] < h[3] < w["hi"][v], (w["lo"], h, w["hi"])
            for k in w["holes"][i + 1:]:
                assert not (h[0] < k[2] and k[0] < h[2] and h[1] < k[3] and k[1] < h[3]), (h, k)
    print(f"  every one of the {n_holes} apertures is strictly inside its surface and "
          f"overlaps no other")
    assert p["objects"], "no measured boxes -- the bundle has no persistent_perception.json"
    a = p["gt_aabb"]
    assert a, "no gt_aabb -- the page cannot check the mesh transform against anything"
    print(f"  gt_aabb over {a['n']} instances  lo {[round(v, 4) for v in a['lo']]}  "
          f"hi {[round(v, 4) for v in a['hi']]}")
    if p["gt_scene"] == "hm3d_00861":
        # The instrument, anchored. These six numbers were measured on 2026-09-04 by
        # decoding the GLB's POSITION accessors and applying map = (glb.y, -glb.x, glb.z);
        # they are also the bundle's own bev_data.json map bounds. If this drifts, the
        # page's Box3 assertion is checking the mesh against something else.
        want_lo, want_hi = [-9.4556, -2.3623, -1.7929], [1.3000, 11.7858, 4.8024]
        for k in range(3):
            assert abs(a["lo"][k] - want_lo[k]) < 0.001, (k, a["lo"], want_lo)
            assert abs(a["hi"][k] - want_hi[k]) < 0.001, (k, a["hi"], want_hi)
        print("  gt_aabb matches the measured GLB AABB under Rz(-90) to 0.001 m")

    mp, sc = mesh_path(b)
    assert sc, "the bundle did not name a scene"
    print(f"  mesh {sc} -> {mp} "
          f"({mp.stat().st_size:,} bytes)" if mp else f"  mesh {sc} -> NOT ON DISK")

    h = page(b)
    # The three facts the page cannot work without: the module loads three from the
    # importmap, the mesh is rotated, and the rotation is asserted rather than assumed.
    assert "<title>" in h
    assert '"three": "./vendor/three.module.js"' in h, \
        "no importmap, or it is absolute again -- GLTFLoader cannot resolve three behind a path prefix"
    assert "root.rotation.set(0, 0, -Math.PI / 2)" in h, "the mesh transform is gone"
    assert "MESH TRANSFORM WRONG" in h, "the Box3 assertion is gone"
    assert "GOOGLE_texture_basis" in h, "the basis texture plugin is gone"
    assert "THREE.ExtrudeGeometry" in h, "the walls went back to being boxes"
    assert "shape.holes.push" in h, "the openings are no longer cut through the walls"
    # FLOOR MAP drew nothing visible under GT WALLS without this: the plane sorted into
    # the middle of the wall slabs and composited over them.
    assert "m.renderOrder = -1" in h, "the floor plane lost its render order and will hide behind the walls"
    assert h.count("depthWrite: true") == 1, "the floor plane stopped being a deck and went back to a tint"
    # And the KG pane only lit up for a node the cose layout happened to place in view.
    assert "CYK.center(hit)" in h, "the knowledge graph no longer scrolls to the selected node"
    # ROUND 2, the three interactions. Each assertion names the ONE line that makes its
    # feature different from the thing it would otherwise degrade into.
    assert "controls.mouseButtons.MIDDLE = null" in h, \
        "MIDDLE is bound to DOLLY again and will zoom while the centre is dragged"
    # Owner 2026-09-08: middle-drag DRAGS THE MAP -- the grabbed point stays under the cursor, so
    # the camera and the orbit centre move by the SAME vector. Asserted as that pair, because
    # translating one without the other is the defect: target alone leaves the cross running away
    # from a still picture, camera alone re-aims the view.
    assert "camera.position.add(dragShift)" in h and "controls.target.add(dragShift)" in h \
        and "dragShift.subVectors(dragFrom, dragAt)" in h, \
        "middle-drag no longer translates camera and centre together; the map will not follow the cursor"
    # Comment lines are stripped first: the drag's own comment SAYS "NOT controls.update()", and a
    # plain text search matched that sentence and failed on working code -- a check reading prose
    # rather than code, which is the shape this file keeps catching elsewhere.
    _drag = "\n".join(ln for ln in h.split("pointermove")[1][:900].splitlines()
                       if not ln.strip().startswith("//"))
    assert "controls.update()" not in _drag, \
        "the map drag calls controls.update(), which re-aims the camera mid-drag"
    assert "camera.quaternion.multiply(AIMOFF)" in h, \
        "the aim offset is not put back after update() -- the next orbit will snap the view"
    assert "scene.add(XHAIR)" in h, "the orbit centre lost its crosshair"
    assert "function dimMesh(" in h and "MESHMATS = [...keep]" in h, \
        "selection no longer dims the mesh, or no longer has the materials to dim"
    assert "function flyTo(" in h and "requestAnimationFrame(flyStep)" in h, \
        "double-click no longer flies to the box, or teleports instead of animating"
    # ROUND 2c, OWNER CORRECTIONS 2026-09-04. The walls are TRANSLUCENT again and the
    # cutaway is off by default: "previous wall transparency was ok". The storey picker
    # filters what it can and clips the one thing it cannot.
    assert "opacity: 0.28, depthWrite: false" in h, \
        "the walls are not translucent -- the owner asked for the earlier transparency"
    assert "let CUT = true;" in h, "the cutaway is off; the owner asked for it back"
    assert 'id="bCut"' in h and "getElementById('bCut')" in h, \
        "the CUTAWAY button or its handler is gone"
    assert "function stepFloor(" in h and 'id="floorPick"' in h, \
        "the storey picker is not the two-arrow control"
    assert 'id="floorBar"' not in h, "the old storey text bar came back"
    assert "controls.minPolarAngle = controls.maxPolarAngle = a" in h, \
        "the orbit pitch is not pinned -- the camera is not yaw-and-zoom only"
    assert "m.opacity = on ? 0.55" in h, \
        "the selected-state mesh is too transparent again -- the room disappears"
    assert "if (SEL && source !== 'scene')" in h, \
        "selecting from the graph or the table no longer flies to the box"
    assert "renderer.localClippingEnabled = true" in h and "m.clippingPlanes = planes" in h, \
        "the mesh is no longer clipped to the picked storey"
    assert "function applyFloor(" in h and "function setFloor(" in h, "the storey picker is gone"
    assert "MAPPED" in h, "the page no longer says which storey the run mapped"
    for btn in ("bObj", "bWall", "bDWall", "bSched", "bOpen", "bPath", "bFloor", "bLabel", "bReset", "bMesh"):
        assert f'id="{btn}' in h and f"getElementById('{btn}')" in h, f"{btn} lost its handler"
    # BROWSER FINDINGS, 2026-09-04: the KG viewport did not follow a container resize
    # (39 of 77 nodes rendered inside after one window resize vs 77 of 77 at load), the
    # objects-table header scrolled away under an auto-scrolled selection, and the backing
    # store kept its load-time pixel ratio across a monitor-scale change.
    assert "CYK.resize()" in h and "CYK.fit(CYK.nodes(), 12)" in h, \
        "the KG pane no longer refits after a window resize"
    assert h.count("renderer.setPixelRatio") == 2, \
        "the pixel ratio is not recomputed in resize() -- it goes stale on a monitor move"
    assert "position:sticky; top:0; z-index:1" in h, \
        "the objects-table header is not sticky and scrolls away with the rows"
    # OWNER TASTE, 2026-09-06. Three decisions, one line each, and each named line is the
    # whole difference between the decision and what the page did before it.
    assert "G.floor.visible = show.floor && !show.mesh" in h, \
        "FLOOR MAP draws under the opaque mesh again -- the plan is buried and the button looks dead"
    assert "const EL = 45 * Math.PI / 180" in h and "Math.sin(EL)" in h, \
        "the fitted camera is not the 45-degree unit direction -- the elevation or the distance drifted"
    assert "const a = w.axis, u = a === 0 ? 1 : 0, v = a === 2 ? 1 : 2;" in h, \
        "the page went back to u = 1 - axis, so a horizontal slab cannot carry an aperture"
    print(f"  page {len(h):,} bytes -- importmap, Rz(-90), Box3 assertion, basis plugin, "
          f"floor render order, KG recentre and 8 controls all present")
