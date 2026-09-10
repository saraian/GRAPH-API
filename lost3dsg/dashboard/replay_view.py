"""Frame-by-frame replay of an archived run: the frames, with the boxes the detector drew.

WHY THIS IS POSSIBLE AT ALL. A bundle recorded with `archive.per_detection: true` holds
`frames/<frame_id>.jpg` and a `detections.jsonl` whose rows carry `frame_id`, `bbox_2d`,
`label`, `score` and the ground-truth instance id. That is everything needed to play the run
back exactly as the detector saw it — so this reads artefacts and invents nothing. A bundle
recorded WITHOUT that flag has no frames, and this page says so rather than rendering empty.

WHAT IT IS NOT. It is not the live view. The live camera is served by the bridge inside the
container on :8081 and exists only while a run is up. This is the same distinction the rest
of the replay dashboard already makes, and the reason it matters is that "no boxes appearing"
looks identical on a page that has no boxes to show and on a run that is producing none.

No external asset, like every other page here: it has to render on a machine with no network.
"""
import json
import math
import os
import re
import struct

try:
    from found.dashboard import dash_env
except ImportError:
    import dash_env

RUNS_DIR = dash_env.runs_dir()
# The simulator's horizontal field of view. Config default; FEED_HFOV overrides it
# at run time, and habitat_feed_host records the value it used into run_metadata.
HFOV_DEG = float(os.environ.get("FEED_HFOV", "90.0"))


def list_bundles(limit=40):
    """-> [(name, n_frames, n_detections, replayable, reason)], newest first.

    A bundle is listed whether or not it can be replayed; a run that recorded no frames is a
    fact worth showing, not an entry to hide. `reason` says why when it cannot.
    """
    out = []
    if not RUNS_DIR.is_dir():
        return out
    dirs = [d for d in RUNS_DIR.iterdir() if d.is_dir() and not d.is_symlink()]
    for d in sorted(dirs, key=lambda p: p.name, reverse=True)[:limit]:
        det = d / "detections.jsonl"
        frames = d / "frames"
        n_fr = len(list(frames.glob("*.jpg"))) if frames.is_dir() else 0
        n_det = 0
        if det.is_file():
            try:
                with det.open() as f:
                    n_det = sum(1 for _ in f)
            except OSError:
                n_det = 0
        if n_fr and n_det:
            out.append((d.name, n_fr, n_det, True, ""))
        elif not det.is_file():
            out.append((d.name, n_fr, 0, False, "no detections.jsonl — per-detection recording was off"))
        elif not n_fr:
            out.append((d.name, 0, n_det, False, "detections recorded but no frames/ directory"))
        else:
            out.append((d.name, n_fr, n_det, False, "empty detections.jsonl"))
    return out


_SIZE_CACHE = {}


def _jpeg_size(path):
    """(width, height) from the JPEG's SOF marker, or None; cached per (path, mtime, size).

    GA-345: a live page re-indexes the bundle every 5 s, and without the cache every tick opened
    every frame file (139 files; ~15 ms of a 105-321 ms call, the detections.jsonl pass being the rest). A frame that is rewritten (new mtime or size)
    is re-read; a bundle of thousands of frames still costs one stat per frame per tick.
    """
    try:
        st = os.stat(path)
    except OSError:
        return None
    key = (str(path), st.st_mtime_ns, st.st_size)
    if key in _SIZE_CACHE:
        return _SIZE_CACHE[key]
    size = _jpeg_size_read(path)
    if size is not None:                 # a failed read (a frame mid-write) is retried next time, never cached
        if len(_SIZE_CACHE) > 20000:
            _SIZE_CACHE.clear()          # ponytail: crude bound; an LRU if it ever matters
        _SIZE_CACHE[key] = size
    return size


def _jpeg_size_read(path):
    """(width, height) from the JPEG's SOF marker, or None. No image library needed.

    Read rather than assumed: the box coordinates are in pixels of THIS frame, and a page
    that scales them against a guessed sensor size draws boxes in the wrong place while
    looking entirely plausible.
    """
    try:
        with open(path, "rb") as f:
            if f.read(2) != b"\xff\xd8":
                return None
            while True:
                b = f.read(1)
                while b and b != b"\xff":
                    b = f.read(1)
                marker = f.read(1)
                while marker == b"\xff":
                    marker = f.read(1)
                if not marker:
                    return None
                m = marker[0]
                if m in (0xD8, 0xD9) or 0xD0 <= m <= 0xD7:
                    continue
                seg = f.read(2)
                if len(seg) < 2:
                    return None
                length = struct.unpack(">H", seg)[0]
                if 0xC0 <= m <= 0xCF and m not in (0xC4, 0xC8, 0xCC):
                    data = f.read(5)
                    if len(data) < 5:
                        return None
                    h, w = struct.unpack(">HH", data[1:5])
                    return int(w), int(h)
                f.seek(length - 2, 1)
    except OSError:
        return None


def frame_index(bundle):
    """-> {"frames": [{id, w, h, detections: [...]}, ...]} in capture order."""
    d = RUNS_DIR / bundle
    det_path, frames_dir = d / "detections.jsonl", d / "frames"
    by_frame = {}
    if det_path.is_file():
        with det_path.open() as f:
            for line in f:
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                fid = r.get("frame_id")
                if fid is None:
                    continue
                by_frame.setdefault(str(fid), []).append({
                    "label": r.get("instance_label") or r.get("label"),
                    "score": r.get("score"),
                    "bbox": r.get("bbox_2d"),
                    "gt": r.get("habitat_gt_instance_id"),
                    "gt_absent": r.get("habitat_gt_absent_reason"),
                    "room": r.get("room_id"),
                })
    frames = []
    if frames_dir.is_dir():
        # Sorted by the frame id, which is a stamp -- so this is capture order, not
        # directory order. They coincide today and there is no reason to depend on it.
        for p in sorted(frames_dir.glob("*.jpg"), key=lambda q: q.stem):
            size = _jpeg_size(p) or (0, 0)
            frames.append({"id": p.stem, "w": size[0], "h": size[1],
                           "detections": by_frame.get(p.stem, [])})
    orphans = sorted(set(by_frame) - {f["id"] for f in frames})
    return {"bundle": bundle, "frames": frames,
            # Detections whose frame was never written. Reported rather than dropped: a
            # silent mismatch between the two artefacts is the kind of gap this project
            # keeps finding, and it belongs on the page.
            "detections_without_frame": len(orphans),
            "frames_without_detections": sum(1 for f in frames if not f["detections"])}


def cycle_rows(bundle):
    """The per-cycle series `perception_latencies.jsonl` (GA-334, writer since vendor 7b98837),
    one dict per completed perception cycle, in file order. [] when the bundle has none --
    every bundle before 2026-09-07 -- and the reader says so rather than showing the snapshot."""
    p = RUNS_DIR / bundle / "perception_latencies.jsonl"
    rows = []
    if p.is_file():
        with p.open() as f:
            for line in f:
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(r, dict):
                    rows.append(r)
    return rows


def cycle_for_frame(bundle, frame_id):
    """-> {"row": dict|None, "match": "frame"|"preceding"|None, "n_rows": int}.

    The row whose `frame_id` is this frame; else the LAST row completed before the frame's
    stamp, labelled "preceding" so the page never passes another cycle's numbers off as this
    frame's; else None. A frame that never entered a cycle (motion-gated, GA-337) has no row.
    """
    rows = cycle_rows(bundle)
    for r in rows:
        if str(r.get("frame_id")) == str(frame_id):
            return {"row": r, "match": "frame", "n_rows": len(rows)}
    try:
        sec, nsec = str(frame_id).split("_")
        at = int(sec) + int(nsec) / 1e9
    except ValueError:
        return {"row": None, "match": None, "n_rows": len(rows)}
    before = [r for r in rows if isinstance(r.get("t"), (int, float)) and r["t"] <= at]
    if before:
        return {"row": max(before, key=lambda r: r["t"]), "match": "preceding", "n_rows": len(rows)}
    return {"row": None, "match": None, "n_rows": len(rows)}


def camera_poses(bundle):
    """-> {frame_id: {"R": [9 floats], "C": [3], "rms_deg": float, "n": int}}. GA-234.

    THE ROTATION IS SOLVED, NOT READ. `detections.jsonl` stores `bbox_3d`, `centroid` and
    `camera_position` and NO orientation, so I said earlier that 3D boxes could not be drawn
    back onto an archived frame. That was wrong: it is only true if you insist on reading the
    pose. Every detection is a CORRESPONDENCE -- a known 3D point in the map frame and the
    pixel it landed on -- and a frame with several of them determines the rotation.

    This is Wahba's problem: find R minimising the angle between each pixel's back-projected
    ray and the direction from the camera to that object. Closed form via SVD (Kabsch), with
    the determinant correction that keeps R a rotation rather than a reflection.

    MEASURED on 20260901_174810_hm3d_00861: 71 of 84 frames have the >=4 detections this
    needs, and across all 71 the median angular residual is 0.75 deg, p90 1.01, max 2.08.
    The residual is RETURNED, per frame, so the page can refuse to draw a frame that did not
    fit rather than drawing a confident wrong box -- which is the whole reason to prefer a
    solve with a measurable error over a reconstruction with none.

    INTRINSICS ARE DERIVED FROM THE ARCHIVED FRAME, not from calibration.json. That file says
    640x480 with fx=320 while this run recorded 1280x960 -- it was not updated when the
    sensor was (GA-233), so trusting it puts every ray at half the right angle.
    """
    import math

    import numpy as np

    d = RUNS_DIR / bundle / "detections.jsonl"
    if not d.is_file():
        return {}
    by_frame = {}
    with d.open() as f:
        for line in f:
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if r.get("bbox_2d") and r.get("centroid") and r.get("camera_position"):
                by_frame.setdefault(str(r["frame_id"]), []).append(r)

    out = {}
    for fid, ds in by_frame.items():
        size = _jpeg_size(RUNS_DIR / bundle / "frames" / f"{fid}.jpg")
        if not size:
            continue
        w, h = size
        fx = fy = (w / 2) / math.tan(math.radians(HFOV_DEG) / 2)
        cx, cy = w / 2, h / 2
        C = np.array(ds[0]["camera_position"], dtype=float)
        # GA-360. READ THE ROTATION WHEN THE ROW CARRIES IT. Every detections row since the
        # optical-frame fix records `camera_quat_xyzw` (770 of 770 on 20260907_152446); the
        # solve below was written for bundles that did not. Solving with 3 degrees of freedom
        # from four near-coplanar points leaves the ROLL nearly free: measured on 152446, the
        # solved rotation tilted map-vertical edges by a median 1.04 deg, p95 4.08, max 7.24
        # (cycle 53: 2.3-5.4 deg, the owner's screenshot), while the row's quaternion keeps them
        # within 0.57 median / 1.72 max. The residual is still computed against the
        # correspondences, so the page's "refuse above 5 deg" rule applies to both sources, and
        # `source` names which one drew.
        q = ds[0].get("camera_quat_xyzw")
        if q and len(q) == 4:
            R = _quat_xyzw_to_map_to_cam(q)
        else:
            R = None
        dcam, dworld = [], []
        for r in ds:
            b = r["bbox_2d"]
            u, v = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
            a = np.array([(u - cx) / fx, (v - cy) / fy, 1.0])
            a /= np.linalg.norm(a)
            wv = np.array(r["centroid"], dtype=float) - C
            n = float(np.linalg.norm(wv))
            if n < 1e-6:
                continue
            dcam.append(a)
            dworld.append(wv / n)
        # Three correspondences determine a rotation; four is the smallest number that can
        # DISAGREE, which is what makes the residual meaningful rather than always zero.
        if R is None:
            if len(dcam) < 4:
                continue
            A = np.array(dcam).T @ np.array(dworld)
            U, _, Vt = np.linalg.svd(A)
            R = U @ np.diag([1.0, 1.0, float(np.sign(np.linalg.det(U @ Vt)))]) @ Vt
            source = "solved"
        else:
            source = "quat"
        errs = [math.degrees(math.acos(max(-1.0, min(1.0, float(a @ (R @ wv))))))
                for a, wv in zip(dcam, dworld, strict=True)]   # same loop built both; prove it
        errs.sort()
        out[fid] = {"R": [float(x) for x in R.flatten()],
                    "C": [float(x) for x in C],
                    "fx": fx, "fy": fy, "cx": cx, "cy": cy,
                    "rms_deg": round(errs[len(errs) // 2], 3) if errs else 0.0,
                    "n": len(dcam), "source": source}
    return out


def _quat_xyzw_to_map_to_cam(q):
    """The row's `camera_quat_xyzw` (tf2 order, camera-optical -> map) as the 3x3 that maps
    MAP-frame offsets into the CAMERA-optical frame, i.e. the transpose of the tf rotation.
    Same convention as live_overlay.quat_to_R + _world_to_cam, which the live overlay uses."""
    import numpy as np
    x, y, z, w = (float(v) for v in q)
    Rc2m = np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])
    return Rc2m.T


def _self_check_level_camera():
    """GA-360: a LEVEL camera must keep map-vertical edges image-vertical. Optical frame: x right,
    y down, z forward. Looking along map +x with z up: cam x = map -y, cam y = map -z, cam z = map +x.
    That rotation as a quaternion (xyzw) is (0.5, -0.5, 0.5, -0.5) up to sign; assert the projected
    vertical segment is vertical to 1e-6 deg, and that a rolled quaternion is NOT."""
    import math

    import numpy as np
    q_level = (0.5, -0.5, 0.5, -0.5)
    R = _quat_xyzw_to_map_to_cam(q_level)
    C = np.array([0.0, 0.0, 1.5])
    def proj(p):
        d = R @ (np.array(p) - C)
        return (d[0] / d[2], d[1] / d[2])
    a, b = proj([4.0, 0.7, 1.0]), proj([4.0, 0.7, 1.6])
    ang = abs(math.degrees(math.atan2(b[0] - a[0], -(b[1] - a[1]))))
    assert ang < 1e-6, f"level camera drew a vertical edge at {ang} deg"
    assert a[1] > b[1], "the higher point must land higher on the image (smaller v)"
    # roll the camera 10 deg about its optical axis: the edge must tilt by 10 deg
    s, c = math.sin(math.radians(5)), math.cos(math.radians(5))
    # quaternion product: q_level * roll(10 deg about z) -- roll about the optical axis
    x1, y1, z1, w1 = q_level
    x2, y2, z2, w2 = 0.0, 0.0, s, c
    q_roll = (w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2, w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
              w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2, w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2)
    R2 = _quat_xyzw_to_map_to_cam(q_roll)
    def proj2(p):
        d = R2 @ (np.array(p) - C)
        return (d[0] / d[2], d[1] / d[2])
    a2, b2 = proj2([4.0, 0.7, 1.0]), proj2([4.0, 0.7, 1.6])
    ang2 = abs(math.degrees(math.atan2(b2[0] - a2[0], -(b2[1] - a2[1]))))
    assert abs(ang2 - 10.0) < 1e-6, f"a 10 deg roll drew {ang2} deg"
    return ang, ang2


DEGENERATE_M = 0.01     # an extent under a centimetre is a sliver mask, not an object


def _box8(b):
    """One `bbox_3d` dict -> (corners8, aabb8, oriented, degenerate), or None.

    THE ORIENTED BOX, when there is one. `bbox_3d` carries `oriented_center`,
    `oriented_extents` and `yaw` beside the axis-aligned min/max, and the oriented one is what
    the pipeline computed (_add_pca_orientation) and what the size envelopes are checked
    against. Drawing the AABB instead shows a box the system never reasoned about -- wider than
    the object whenever it is not axis-aligned.

    DEGENERATE is stated, not hidden. On 20260906_223701 the air conditioner seen as a sliver
    at the right edge of frame 1788727565 came out with oriented extents [0.39, 0.00, 0.50]:
    a zero-width box. Drawn, it is a line pretending to be a box; the page must be able to
    refuse it by name.
    """
    if not isinstance(b, dict):
        return None
    try:
        x0, x1 = b["x_min"], b["x_max"]
        y0, y1 = b["y_min"], b["y_max"]
        z0, z1 = b["z_min"], b["z_max"]
    except KeyError:
        return None
    aabb = [[x0, y0, z0], [x1, y0, z0], [x1, y1, z0], [x0, y1, z0],
            [x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1]]
    oc, oe = b.get("oriented_center"), b.get("oriented_extents")
    yaw = b.get("yaw")
    if oc and oe and yaw is not None and len(oc) >= 3 and len(oe) >= 3:
        cy_, sy_ = math.cos(float(yaw)), math.sin(float(yaw))
        hx, hy, hz = oe[0] / 2.0, oe[1] / 2.0, oe[2] / 2.0
        corners = []
        for sx, sy2, sz in ((-1, -1, -1), (1, -1, -1), (1, 1, -1), (-1, 1, -1),
                            (-1, -1, 1), (1, -1, 1), (1, 1, 1), (-1, 1, 1)):
            lx, ly, lz = sx * hx, sy2 * hy, sz * hz
            corners.append([oc[0] + lx * cy_ - ly * sy_,
                            oc[1] + lx * sy_ + ly * cy_,
                            oc[2] + lz])
        degenerate = min(abs(float(v)) for v in oe[:3]) < DEGENERATE_M
        return corners, aabb, True, degenerate
    degenerate = min(x1 - x0, y1 - y0, z1 - z0) < DEGENERATE_M
    return aabb, aabb, False, degenerate


def _touches_border(bbox, w, h, tol=1.5):
    """Does a 2D detection box touch the frame edge? Then the object was NOT seen whole, and
    its 3D box is an estimate from a partial view -- the shape of GA-315."""
    if not bbox or len(bbox) < 4 or not w or not h:
        return False
    x0, y0, x1, y1 = bbox[:4]
    return x0 <= tol or y0 <= tol or x1 >= w - tol or y1 >= h - tol


def frame_boxes3d(bundle):
    """-> {frame_id: [{label, c, aabb, oriented, partial, degenerate}]} in the MAP frame.

    These are the DETECTIONS of each frame -- what perception produced from that one view.
    They vary frame to frame and are small or degenerate when the object is clipped by the
    frame edge; that is a property of the detection and the page says so with `partial`. The
    settled box per object is `belief_boxes3d`.
    """
    d = RUNS_DIR / bundle / "detections.jsonl"
    if not d.is_file():
        return {}
    # Frame size, read once from the first archived frame -- every frame of a run is one
    # size, and the border test needs it in pixels of THIS run, not a guess.
    frames = sorted((RUNS_DIR / bundle / "frames").glob("*.jpg"))
    w, h = (_jpeg_size(frames[0]) or (0, 0)) if frames else (0, 0)
    out = {}
    with d.open() as f:
        for line in f:
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            box = _box8(r.get("bbox_3d"))
            if box is None:
                continue
            corners, aabb, oriented, degenerate = box
            out.setdefault(str(r["frame_id"]), []).append({
                "label": r.get("instance_label") or r.get("label"),
                "c": corners,
                # The axis-aligned hull, always, so the viewer can show what the PCA step
                # actually bought.
                "aabb": aabb,
                "oriented": oriented,
                "partial": _touches_border(r.get("bbox_2d"), w, h),
                "degenerate": degenerate,
            })
    return out


def belief_boxes3d(bundle):
    """-> [{label, object_id, c, aabb, oriented, degenerate}]: the world model's SETTLED box per
    object, from persistent_perception.json. One set for the whole replay -- the store's end
    state, because no per-frame belief history is archived. Said on the page as such.

    Why this layer exists: a viewer stepping through frames saw the air conditioner's box
    "perfect" in one frame and "small and unrotated" in the next, and read it as a rendering
    bug. It was the per-frame DETECTION changing with the view. The belief box is the one the
    system actually holds about the object, and it is what a reader expects to stay put.
    Where a bad partial-view estimate has leaked INTO the store (GA-315), this layer shows
    that too -- 7 "air conditioner" objects on 20260906_223701, one of them zero-width.
    """
    p = RUNS_DIR / bundle / "persistent_perception.json"
    if not p.is_file():
        return []
    try:
        pp = json.loads(p.read_text())
    except (OSError, json.JSONDecodeError):
        return []
    if isinstance(pp, dict):
        objs = pp.get("objects") or pp.get("persistent_objects")
        if objs is None:
            objs = [v for v in pp.values() if isinstance(v, dict) and "bbox" in v]
    else:
        objs = pp
    out = []
    for o in objs or []:
        if not isinstance(o, dict):
            continue
        box = _box8(o.get("bbox") or o.get("bbox_3d"))
        if box is None:
            continue
        corners, aabb, oriented, degenerate = box
        out.append({"label": o.get("label") or o.get("instance_label"),
                    "object_id": o.get("object_id"), "c": corners, "aabb": aabb,
                    "oriented": oriented, "degenerate": degenerate})
    return out


def object_verdicts(bundle):
    """-> {"by_object": {obj_id: grade}, "by_label": {label: [grades]}, ...}. GA-239.

    THE LABEL IS NOT AN IDENTITY, and the first version of this function assumed it was.

    `_assign_instance_labels` runs once per FRAME with a fresh counter, so `dining chair#13`
    means "the 13th dining chair in this frame, in detector output order". It is a within-frame
    ordinal. MEASURED on 20260901_174810_hm3d_00861: 269 persistent objects carry only 117
    distinct labels; `dining chair#13` belongs to TEN different objects up to 7.5 m apart, and
    `doorway#1` to nine up to 7.7 m apart.

    The admission log keys its `object` field on that ordinal, so 24 of 143 labels received
    more than one distinct grade -- `dining table#1` is recorded admit AND decline AND hold,
    because those are three different pieces of furniture. Keying a verdict map on the label,
    as I did first, silently keeps whichever came last and throws the others away.

    THE REAL JOIN IS RECORDED and was there all along: an admission carries
    `annotation.decision_id`, and a `link` record carries that same `decision_id` beside the
    true `object` id (`obj_<uuid>`). That pair is the only thing in the bundle that ties a
    verdict to a thing in the world.

    Both views are returned, and the caller must choose knowingly: `by_object` is sound,
    `by_label` is ambiguous and reports EVERY grade a label received rather than pretending
    there is one.
    """
    d = RUNS_DIR / bundle / "hook_decisions.jsonl"
    if not d.is_file():
        return {"by_object": {}, "by_label": {}, "unlinked": 0, "ambiguous_labels": 0}

    grades, label_of, links = {}, {}, {}
    with d.open("rb") as f:
        for line in f:
            is_adm = b'"kind": "admission"' in line or b'"kind":"admission"' in line
            is_link = b'"kind": "link"' in line or b'"kind":"link"' in line
            if not (is_adm or is_link):
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if is_adm:
                ann = r.get("annotation") or {}
                did = ann.get("decision_id")
                if not did:
                    continue
                g = (ann.get("verdict") or {}).get("grade") or r.get("outcome")
                grades[did] = g
                label_of[did] = r.get("object")
            else:
                # link: decision_id -> the object the Graph API actually minted
                if r.get("decision_id") and r.get("object"):
                    links[r["decision_id"]] = r["object"]

    by_object, by_label = {}, {}
    unlinked = 0
    for did, g in grades.items():
        oid = links.get(did)
        if oid is None:
            # No link record. Expected for a refusal -- nothing was minted to link TO -- so
            # this is counted, not treated as an error.
            unlinked += 1
        else:
            by_object[oid] = g
        lab = label_of.get(did)
        if lab:
            by_label.setdefault(lab, []).append(g)

    return {
        "by_object": by_object,
        "by_label": {k: sorted(set(v)) for k, v in by_label.items()},
        "unlinked": unlinked,
        # The headline number a caller needs to see before trusting any label-keyed view.
        "ambiguous_labels": sum(1 for v in by_label.values() if len(set(v)) > 1),
    }


def frame_masks(bundle, frame_id):
    """-> [{label, score, bbox, rle}] for ONE frame. GA-227.

    Kept out of the frame index on purpose. `detections.jsonl` carries a full `mask_rle` per
    detection -- 960x1280 run-lengths -- and this bundle holds 1,530 of them. Putting them in
    the index would turn a 300 KB payload into tens of megabytes to draw one frame's worth.
    Asked for per frame, only when the segmentation layer is actually switched on.
    """
    d = RUNS_DIR / bundle / "detections.jsonl"
    if not d.is_file():
        return []
    want, out = str(frame_id), []
    with d.open() as f:
        for line in f:
            # Cheap reject before the parse: 1,530 json.loads to answer for ~18 rows is the
            # same waste that made /graph_data unusable.
            if want not in line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if str(r.get("frame_id")) != want:
                continue
            rle = r.get("mask_rle")
            if rle:
                out.append({"label": r.get("instance_label") or r.get("label"),
                            "score": r.get("score"), "bbox": r.get("bbox_2d"), "rle": rle})
    return out


def frame_path(bundle, frame_id):
    """Resolved and CONTAINED. `frame_id` comes off a URL, so it is checked to be a real
    child of this bundle's frames/ rather than trusted to be a bare stem."""
    base = (RUNS_DIR / bundle / "frames").resolve()
    p = (base / f"{frame_id}.jpg").resolve()
    if base not in p.parents or not p.is_file():
        return None
    return p


def _human_bytes(n):
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.0f} {unit}" if unit == "B" else f"{n:.1f} {unit}"
        n /= 1024.0
    return f"{n:.1f} GB"


def list_logs(bundle):
    """-> [{name, size, bytes}] for the bundle's captured terminal output. GA-220.

    Every node's stdout is already archived -- om6.log alone reached 39 MB on the 61-minute
    run -- and nothing surfaced it beside the frames until now.
    """
    d = RUNS_DIR / bundle / "logs"
    if not d.is_dir():
        return []
    out = []
    for p in sorted(d.glob("*.log")):
        try:
            n = p.stat().st_size
        except OSError:
            continue
        out.append({"name": p.name, "bytes": n, "size": _human_bytes(n)})
    return out


def log_path(bundle, name):
    """Resolved and CONTAINED, same rule as frame_path: the name arrives off a URL."""
    base = (RUNS_DIR / bundle / "logs").resolve()
    p = (base / name).resolve()
    if base not in p.parents or not p.is_file() or p.suffix != ".log":
        return None
    return p


_STAMP = re.compile(rb"\[(\d{10})\.")


def _stamp_at(f, offset, limit=64):
    """-> (epoch, line_offset) for the first stamped line at or after `offset`, or None."""
    f.seek(offset)
    if offset:
        f.readline()  # the seek landed mid-line; that fragment is not a line
    for _ in range(limit):
        here = f.tell()
        line = f.readline()
        if not line:
            return None
        m = _STAMP.search(line)
        if m:
            return int(m.group(1)), here
    return None


def log_window(bundle, name, at=None, span=(120_000, 280_000)):
    """A window of one log, around the wall-clock time `at`. GA-220.

    WHY NOT THE TAIL. om6.log is 38.6 MB and its last 8 MB begins at 1788280821, while this
    run's frames begin at 1788277804. Serving the tail and letting the browser search inside
    it puts 78 of 84 frames before the first byte the browser holds -- where a nearest-line
    search clamps to line 1 and reports it as the line nearest that frame. That is a wrong
    answer wearing the shape of a right one, which is the exact failure this project keeps
    finding; a window that cannot contain the frame must say so rather than point somewhere.

    BINARY SEARCH, because a linear scan of 38.6 MB per frame step is not affordable and an
    in-memory index of every log is not either. ROS writes these append-only with a
    monotonic stamp, so bisecting on byte offset converges in ~20 seeks. Where the file is
    NOT monotonic the window lands slightly off, which costs a few lines of context -- not a
    wrong claim, because the returned range is reported and the caller can see the miss.
    """
    p = log_path(bundle, name)
    if p is None:
        return None
    size = p.stat().st_size
    with p.open("rb") as f:
        if at is None:
            start = max(0, size - sum(span))
        else:
            lo, hi = 0, size
            while hi - lo > 65536:
                mid = (lo + hi) // 2
                got = _stamp_at(f, mid)
                if got is None:
                    hi = mid
                elif got[0] < at:
                    lo = mid
                else:
                    hi = mid
            start = max(0, lo - span[0])
        f.seek(start)
        if start:
            f.readline()
        body = f.read(sum(span)).decode("utf-8", errors="replace")
    lines = body.splitlines()
    stamps = [int(m.group(1)) for m in (_STAMP.search(ln.encode()) for ln in lines) if m]
    return {
        "text": body,
        "bytes": size,
        "window": [start, min(size, start + sum(span))],
        "t_lo": stamps[0] if stamps else None,
        "t_hi": stamps[-1] if stamps else None,
        # Stated so the page can say "this frame is outside the loaded window" instead of
        # silently showing the nearest line it happens to hold.
        "covers": (bool(stamps) and at is not None and stamps[0] <= at <= stamps[-1]),
    }


def page(bundle=None):
    bundles = list_bundles()
    replayable = [b for b in bundles if b[3]]
    if bundle is None and replayable:
        bundle = replayable[0][0]
    options = "".join(
        f'<option value="{n}"{" selected" if n == bundle else ""}'
        f'{" disabled" if not ok else ""}>{n} — {nf} frames, {nd} detections'
        f'{"" if ok else " (" + why + ")"}</option>'
        for n, nf, nd, ok, why in bundles)
    if not replayable:
        body = ('<div class="warn"><b>No bundle can be replayed.</b> Replay needs frames and '
                'detections, which a run only writes with <code>archive.per_detection: true</code>. '
                'Every run below recorded one or neither.</div>')
    else:
        body = ""
    return f"""<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Replay a bundle</title>
<style>
:root{{--ground:#f7f6f3;--ink:#15161a;--dim:#5a5f68;--faint:#8d939c;--rule:#dfdcd6;--card:#fffefc;
  --box:#1d7a52;--box-gt:#4a4396;--warn:#b03a2b}}
@media (prefers-color-scheme:dark){{:root{{--ground:#111318;--ink:#e8e8ea;--dim:#9aa0aa;
  --faint:#6d747e;--rule:#282c34;--card:#181b21;--box:#57c48d;--box-gt:#a29bf0;--warn:#f0857a}}}}
*{{box-sizing:border-box}}
body{{margin:0;background:var(--ground);color:var(--ink);
  font:14px/1.6 ui-sans-serif,system-ui,-apple-system,sans-serif}}
.wrap{{max-width:1100px;margin:0 auto;padding:22px 20px 60px}}
a.back{{font:600 11px ui-monospace,monospace;color:var(--dim);text-decoration:none;
  border:1px solid var(--rule);border-radius:3px;padding:5px 10px}}
h1{{font-size:25px;margin:15px 0 5px;letter-spacing:-.015em}}
p.sub{{color:var(--dim);margin:0 0 14px;max-width:64ch}}
.bar{{display:flex;gap:10px;align-items:center;flex-wrap:wrap;background:var(--card);
  border:1px solid var(--rule);border-radius:3px;padding:10px 12px;margin-bottom:12px}}
select,button{{font:600 12px ui-monospace,monospace;background:var(--ground);color:var(--ink);
  border:1px solid var(--rule);border-radius:3px;padding:6px 10px;cursor:pointer}}
button:disabled{{opacity:.45;cursor:default}}
input[type=range]{{flex:1;min-width:220px}}
#stage{{position:relative;display:inline-block;background:#000;border:1px solid var(--rule);
  border-radius:3px;line-height:0;max-width:100%}}
#stage img{{display:block;max-width:100%;height:auto}}
#ov{{position:absolute;inset:0;width:100%;height:100%}}
.meta{{font:11px ui-monospace,monospace;color:var(--dim);margin-top:8px;line-height:1.8}}
.warn{{background:var(--card);border:1px solid var(--rule);border-left:3px solid var(--warn);
  border-radius:3px;padding:12px 15px;margin:12px 0}}
table{{border-collapse:collapse;width:100%;font:12px ui-monospace,monospace;margin-top:12px}}
th{{text-align:left;color:var(--faint);font-size:10px;text-transform:uppercase;letter-spacing:.08em;
  padding:7px 9px;border-bottom:1px solid var(--rule)}}
td{{padding:6px 9px;border-bottom:1px solid var(--rule)}}
.tw{{border:1px solid var(--rule);border-radius:3px;background:var(--card);overflow-x:auto}}
</style></head><body><div class="wrap">
<!-- GA-380: relative. An absolute href goes to the SITE ROOT behind a path prefix. -->
<a class="back" href="./">&larr; DASHBOARD</a>
<h1>Replay a bundle</h1>
<p class="sub">The archived frames with the boxes the detector actually drew. This reads a run
directory — it is not the live view, which the bridge serves on :8081 while a run is up.</p>
{body}
<div class="bar">
  <select id="bundle" onchange="location.search='?bundle='+encodeURIComponent(this.value)">{options}</select>
  <button id="play">PLAY</button>
  <button id="stop">STOP</button>
  <button id="prev">&larr;</button>
  <button id="next">&rarr;</button>
  <input type="range" id="seek" min="0" value="0">
  <select id="speed"><option value="500">2 fps</option><option value="250" selected>4 fps</option>
    <option value="100">10 fps</option><option value="1000">1 fps</option></select>
  <label style="font:11px ui-monospace,monospace;color:var(--dim)">
    <input type="checkbox" id="gtonly"> GT-joined only</label>
</div>
<div id="stage"><img id="img" alt="archived frame"><svg id="ov" preserveAspectRatio="none"></svg></div>
<div class="meta" id="meta">loading…</div>
<div class="tw"><table id="dets"><tr><th>label</th><th>score</th><th>box (px)</th><th>gt id</th><th>room</th></tr></table></div>

<h2 style="font:600 11px ui-monospace,monospace;text-transform:uppercase;letter-spacing:.1em;
  color:var(--dim);margin:26px 0 10px;padding-bottom:6px;border-bottom:1px solid var(--rule)">
  Terminal log at this frame</h2>
<div class="bar">
  <select id="logfile"></select>
  <label style="font:11px ui-monospace,monospace;color:var(--dim)">
    <input type="checkbox" id="logfollow" checked> follow the frame</label>
  <span id="logmeta" style="font:11px ui-monospace,monospace;color:var(--faint)"></span>
</div>
<pre id="logbox" style="background:var(--card);border:1px solid var(--rule);border-radius:3px;
  padding:10px 12px;max-height:320px;overflow:auto;font:11px/1.5 ui-monospace,monospace;
  white-space:pre-wrap;margin:0">choose a log above</pre>
</div>
<script>
const BUNDLE = {json.dumps(bundle)};
let F = [], i = 0, timer = null;
const $ = id => document.getElementById(id);

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
function draw() {{
  if (!F.length) return;
  const f = F[i];
  $('img').src = PFX + '/replay/frame/' + encodeURIComponent(BUNDLE) + '/' + encodeURIComponent(f.id) + '.jpg';
  const ov = $('ov');
  ov.setAttribute('viewBox', '0 0 ' + (f.w || 640) + ' ' + (f.h || 480));
  const only = $('gtonly').checked;
  const dets = f.detections.filter(d => !only || d.gt != null);
  ov.innerHTML = dets.map(d => {{
    if (!d.bbox || d.bbox.length !== 4) return '';
    const [x1, y1, x2, y2] = d.bbox;
    const w = Math.abs(x2 - x1), h = Math.abs(y2 - y1);
    const c = d.gt != null ? 'var(--box-gt)' : 'var(--box)';
    const lbl = (d.label || '?') + (d.score != null ? ' ' + Number(d.score).toFixed(2) : '');
    return '<rect x="' + Math.min(x1, x2) + '" y="' + Math.min(y1, y2) + '" width="' + w +
           '" height="' + h + '" fill="none" stroke="' + c + '" stroke-width="2"/>' +
           '<text x="' + (Math.min(x1, x2) + 3) + '" y="' + (Math.min(y1, y2) - 4) +
           '" font-family="ui-monospace,monospace" font-size="13" fill="' + c + '">' +
           lbl.replace(/[&<>]/g, s => ({{'&':'&amp;','<':'&lt;','>':'&gt;'}})[s]) + '</text>';
  }}).join('');
  $('seek').value = i;
  $('meta').textContent = 'frame ' + (i + 1) + ' / ' + F.length + '  ·  id ' + f.id +
    '  ·  ' + f.w + '×' + f.h + '  ·  ' + f.detections.length + ' detections' +
    (only ? ' (' + dets.length + ' GT-joined shown)' : '');
  $('dets').innerHTML = '<tr><th>label</th><th>score</th><th>box (px)</th><th>gt id</th><th>room</th></tr>' +
    dets.map(d => '<tr><td>' + (d.label || '?') + '</td><td>' +
      (d.score != null ? Number(d.score).toFixed(3) : '—') + '</td><td>' +
      (d.bbox ? d.bbox.map(v => Math.round(v)).join(', ') : '—') + '</td><td>' +
      (d.gt != null ? d.gt : '<span style="color:var(--faint)">' + (d.gt_absent || '—') + '</span>') +
      '</td><td>' + (d.room || '—') + '</td></tr>').join('');
}}
function step(n) {{ if (F.length) {{ i = (i + n + F.length) % F.length; draw(); }} }}
const _draw0 = draw;
draw = function () {{ _draw0(); renderLog(); }};
$('prev').onclick = () => step(-1);
$('next').onclick = () => step(1);
$('seek').oninput = e => {{ i = +e.target.value; draw(); }};
$('gtonly').onchange = draw;
// STOP is not PAUSE: it halts AND rewinds to frame 0. Pause leaves you where you were,
// which is what you want mid-inspection; conflating the two makes one behaviour unreachable.
$('stop').onclick = () => {{
  if (timer) {{ clearInterval(timer); timer = null; $('play').textContent = 'PLAY'; }}
  i = 0; draw();
}};
$('play').onclick = () => {{
  if (timer) {{ clearInterval(timer); timer = null; $('play').textContent = 'PLAY'; }}
  else {{ timer = setInterval(() => step(1), +$('speed').value); $('play').textContent = 'PAUSE'; }}
}};
$('speed').onchange = () => {{ if (timer) {{ clearInterval(timer);
  timer = setInterval(() => step(1), +$('speed').value); }} }};
document.addEventListener('keydown', e => {{
  if (e.key === 'ArrowLeft') step(-1);
  if (e.key === 'ArrowRight') step(1);
  if (e.key === ' ') {{ e.preventDefault(); $('play').click(); }}
}});
// Frame ids ARE stamps (seconds_nanoseconds), so the log line nearest a frame is findable by
// time. That is the point of this pane: "what did the pipeline SAY while it was looking at
// this?" cannot be answered from the picture alone.
let logName = null, logWin = null, logBusy = false;
function frameEpoch(id) {{
  const m = String(id).match(/^(\\d+)/);
  return m ? Number(m[1]) : null;
}}
// A WINDOW around the frame's own time, fetched from the server, not the whole log searched
// in the browser. om6.log is 38.6 MB; its tail begins after 78 of this run's 84 frames, so a
// tail plus an in-browser search would clamp those frames to line 1 and call it the nearest
// line. Refetched only when the frame leaves the window that is already loaded.
function loadWindow(at) {{
  if (!logName || logBusy) return;
  logBusy = true;
  // GA-380: the helper is defined a few lines above and was applied to the frame image only.
  // Concatenated into a variable, this one read as ordinary code and survived the sweep.
  const q = PFX + '/replay/log/' + encodeURIComponent(BUNDLE) + '/' + encodeURIComponent(logName)
          + (at == null ? '' : '?at=' + at);
  fetch(q).then(r => r.json()).then(w => {{ logWin = w; logBusy = false; renderLog(); }})
    .catch(e => {{ logBusy = false; $('logbox').textContent = 'could not load: ' + e; }});
}}
function renderLog() {{
  const box = $('logbox'), follow = $('logfollow').checked;
  if (!logName) {{ box.textContent = 'choose a log above'; $('logmeta').textContent = ''; return; }}
  const at = (follow && F.length) ? frameEpoch(F[i].id) : null;
  if (!logWin || (at != null && !(logWin.t_lo <= at && at <= logWin.t_hi))) {{
    if (!logBusy) {{ box.textContent = 'loading…'; loadWindow(at); }}
    return;
  }}
  const lines = logWin.text.split('\\n');
  if (at == null) {{
    box.textContent = lines.slice(-400).join('\\n');
    $('logmeta').textContent = 'tail of ' + (logWin.bytes / 1048576).toFixed(1) + ' MB';
    return;
  }}
  let best = 0;
  for (let n = 0; n < lines.length; n++) {{
    const m = lines[n].match(/\\[(\\d{{10}})\\./);
    if (m && Number(m[1]) <= at) best = n;
  }}
  box.textContent = lines.slice(Math.max(0, best - 12), best + 28).join('\\n');
  // The window is stated, so a frame the log does not cover is visible as that, not as a
  // confident pointer at whatever line happened to be loaded.
  $('logmeta').textContent = logWin.covers
    ? 'line ' + (best + 1) + ' of ' + lines.length + ' in this window — at the frame'
    : 'this frame is outside the log — nothing was written at ' + at;
}}
$('logfollow').onchange = () => {{ logWin = null; renderLog(); }};
$('logfile').onchange = e => {{
  logName = e.target.value || null; logWin = null; renderLog();
}};
if (BUNDLE) fetch('/replay/logs/' + encodeURIComponent(BUNDLE))
  .then(r => r.json()).then(d => {{
    $('logfile').innerHTML = '<option value="">— choose a log —</option>' +
      (d.logs || []).map(l => '<option value="' + l.name + '">' + l.name + '  (' + l.size + ')</option>').join('');
  }}).catch(() => {{}});

if (BUNDLE) fetch('/replay/index/' + encodeURIComponent(BUNDLE))
  .then(r => r.json()).then(d => {{
    F = d.frames || [];
    $('seek').max = Math.max(0, F.length - 1);
    if (!F.length) {{ $('meta').textContent = 'this bundle has no frames to replay'; return; }}
    draw();
    if (d.detections_without_frame || d.frames_without_detections)
      $('meta').textContent += '  ·  ' + d.detections_without_frame +
        ' detections with no frame, ' + d.frames_without_detections + ' frames with no detection';
  }})
  .catch(e => {{ $('meta').textContent = 'could not load the index: ' + e; }});
else $('meta').textContent = 'no replayable bundle';
</script></body></html>"""


if __name__ == "__main__":
    bs = list_bundles()
    print(f"bundles found: {len(bs)}, replayable: {sum(1 for b in bs if b[3])}")
    for n, nf, nd, ok, why in bs[:6]:
        print(f"  {'OK ' if ok else '-- '} {n}  frames={nf:<5} dets={nd:<6} {why}")
    rep = [b for b in bs if b[3]]
    if rep:
        idx = frame_index(rep[0][0])
        f = idx["frames"][0]
        print(f"\n  first frame of {rep[0][0]}: id={f['id']} {f['w']}x{f['h']} "
              f"dets={len(f['detections'])}")
        assert f["w"] > 0 and f["h"] > 0, "JPEG size must be read, never assumed"
        boxed = sum(1 for fr in idx["frames"] for d in fr["detections"]
                    if d["bbox"] and len(d["bbox"]) == 4)
        print(f"  boxes across the run: {boxed}")
        print(f"  detections with no frame: {idx['detections_without_frame']}")
        assert frame_path(rep[0][0], "../../etc/passwd") is None, "path traversal must be refused"
        # GA-334 reader: exact frame -> its row; a later stamp -> the preceding row, said so; an
        # earlier stamp -> nothing. Against the first bundle that carries a series, if any.
        with_series = next((b[0] for b in rep if cycle_rows(b[0])), None)
        if with_series:
            rows = cycle_rows(with_series)
            r0 = rows[0]
            hit = cycle_for_frame(with_series, r0["frame_id"])
            assert hit["match"] == "frame" and hit["row"]["cycle"] == r0["cycle"], hit
            sec = int(str(r0["frame_id"]).split("_")[0])
            later = cycle_for_frame(with_series, f"{sec + 3600}_0")
            assert later["match"] == "preceding" and later["row"] is not None, later
            earlier = cycle_for_frame(with_series, f"{sec - 3600}_0")
            assert earlier["match"] is None and earlier["row"] is None, earlier
            print(f"  per-cycle series: {with_series} has {len(rows)} rows; reader OK")
        else:
            print("  per-cycle series: no bundle carries one (reader untested here)")
        assert cycle_for_frame(rep[0][0], "not_a_stamp")["row"] is None
        # GA-360: the pose reader keeps a level camera level, and sees a roll when there is one
        lv, rl = _self_check_level_camera()
        print(f"  level-camera check: vertical edge at {lv:.2e} deg; a 10 deg roll reads {rl:.4f} deg")
        cp = camera_poses(rep[0][0])
        src = {}
        for v in cp.values():
            src[v.get("source")] = src.get(v.get("source"), 0) + 1
        print(f"  camera_poses on {rep[0][0]}: {len(cp)} frames, by source {src}")
        print("\nreplay_view self-check OK")
