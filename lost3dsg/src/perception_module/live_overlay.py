"""Draw what the pipeline detected onto the frame the dashboard is showing.

WHY THIS EXISTS. The dashboard's live frames come from the simulator host, which draws its
belief overlay only into its own GUI window copy (`if SHOW and OVERLAY:` in
habitat_feed_host.py) -- so the frame served to the browser has never carried a box. The ROS
topic `/image_with_bb` does carry boxes and masks, but perception publishes one frame per
cycle and none at all while the agent walks, so the dashboard alternated between clean host
frames and an occasional annotated one. On screen that reads as "the boxes flicker in from
somewhere older", which is exactly what it is.

Here the boxes are PROJECTED from the persistent 3D world model instead, so an object stays
outlined for as long as it is in view -- "once found, always drawn" -- rather than only on the
cycle it was detected.

THE PROJECTION IS VERIFIED, not assumed. `detections.jsonl` records, per detection, the
camera pose that produced it AND the 2D box that resulted. Projecting the recorded 3D box with
the recorded pose must reproduce the recorded 2D box. Measured on 414 detections of
20260903_230232_hm3d_00861:

    convention                in front   median centre error
    camera_quat as OPTICAL     414/414             9.3 px      <- correct
    camera_quat as ROS body    192/414          1630.3 px
    camera_quat as Y-down        0/414                  -

9.3 px on a 1280 x 960 image is 0.7 % of the width, and the residual is the 3D box being a
fitted cuboid rather than the detector's pixel-tight rectangle. `--self-check` re-runs this
against a real bundle, so a change to the convention fails loudly instead of drawing plausible
boxes in the wrong place.

MASKS, stated plainly. A mask is a 2D region belonging to the frame it was segmented from. It
cannot be honestly redrawn on a later frame taken from a different viewpoint, so masks are
drawn only while the frame they belong to is the one on screen. The persistent thing here is
the box, because a box has a 3D extent and a mask does not.
"""
import json
import math
import os
from pathlib import Path

try:
    from .box_view import BOX_EDGES, box_corners_map
except ImportError:  # launched as a top-level script by the bridge
    from box_view import BOX_EDGES, box_corners_map

# Objects further than this are drawn thinner: past a few metres a projected cuboid is a
# handful of pixels and a full-strength outline is noise rather than information.
FAR_M = float(os.environ.get("BRIDGE_OVERLAY_FAR", "6.0"))
# And past THIS they are not drawn at all.
#
# THERE IS NO OCCLUSION TEST HERE, and that is the reason this cap exists. The projection asks
# only "is it in the frustum", so an object two rooms away, behind two walls, projects exactly
# like one in front of you -- measured on a real frame, a doorway view drew the whole next
# bathroom over a blank wall. A depth test needs the scene mesh, which this process does not
# have; /scene3d has it and does occlude properly.
# ponytail: distance cap, not occlusion. Depth-test against the GLB if the clutter still bites.
MAX_M = float(os.environ.get("BRIDGE_OVERLAY_MAX", "4.0"))
# Reject a projection that fills the frame: a box straddling the camera plane projects to
# something enormous, and drawing it hides everything else.
MAX_FRAC = 0.9


def quat_to_R(x, y, z, w):
    """Rotation matrix of the quaternion, rows first. Same convention as tf2: xyzw in, and
    the matrix maps CAMERA-frame vectors into the WORLD frame."""
    return (
        (1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)),
        (2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)),
        (2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)),
    )


def _world_to_cam(R, rel):
    """R transposed times rel -- the inverse rotation, without building a second matrix."""
    return tuple(sum(R[k][i] * rel[k] for k in range(3)) for i in range(3))


def box_corners(b):
    """Return the same oriented corners used by the ROS and Habitat renderers."""
    corners, _ = box_corners_map(b)
    return [tuple(float(v) for v in corner) for corner in corners]


def _project_box(bbox, cam_pos, cam_quat_xyzw, intr, width, height):
    """-> envelope, projected corners and orientation, or None when not drawable.

    None means one of: any corner is behind or on the image plane, the projection lands
    entirely off-screen, or it covers more than MAX_FRAC of the frame. Each of those draws
    something misleading rather than something wrong-looking, which is worse.
    """
    corners = box_corners(bbox)
    if len(corners) != 8:
        return None
    R = quat_to_R(*cam_quat_xyzw)
    us, vs, zs = [], [], []
    for p in corners:
        rel = [p[i] - cam_pos[i] for i in range(3)]
        X, Y, Z = _world_to_cam(R, rel)
        if Z <= 0.05:
            return None                       # behind the camera, or on the plane
        us.append(intr["fx"] * X / Z + intr["cx"])
        vs.append(intr["fy"] * Y / Z + intr["cy"])
        zs.append(Z)
    u0, u1, v0, v1 = min(us), max(us), min(vs), max(vs)
    if u1 < 0 or v1 < 0 or u0 > width or v0 > height:
        return None                           # entirely off screen
    if (u1 - u0) > MAX_FRAC * width and (v1 - v0) > MAX_FRAC * height:
        return None                           # straddling the camera; would cover everything
    _, oriented = box_corners_map(bbox)
    return (u0, v0, u1, v1, sum(zs) / len(zs),
            list(zip(us, vs)), oriented)


def project_box(bbox, cam_pos, cam_quat_xyzw, intr, width, height):
    """-> (u0, v0, u1, v1, depth_m) in pixels, or None when the box is not drawable.

    The public five-value return shape is preserved for callers and archived self-checks. The
    live overlay internally uses `_project_box` so it can draw the actual projected cuboid edges
    instead of an axis-aligned 2D rectangle.
    """
    result = _project_box(bbox, cam_pos, cam_quat_xyzw, intr, width, height)
    return None if result is None else result[:5]


def frustum_rays(intr, width, height, depth=2.0):
    """-> the four image-corner rays of this camera, IN ITS OWN OPTICAL FRAME, at `depth` m.

    THE EXACT INVERSE OF `project_box`, and that is the whole point of putting it here.
    `project_box` maps a camera-frame point to `u = fx*X/Z + cx`, `v = fy*Y/Z + cy`; the
    corner pixel `(u, v)` at range Z is therefore `X = (u - cx)/fx * Z`, `Y = (v - cy)/fy * Z`.
    Same fx, same cx, same sign of Y. A drawing that built its own corner rays would be a
    SECOND convention, and the header above records what the other two conventions cost:
    1630 px and a frame with nothing in front of it.

    Returned in the CAMERA frame, not the world frame. The caller rotates them by the same
    quaternion `quat_to_R` takes -- `camera_quat_xyzw`, which is the OPTICAL frame -- and adds
    the camera position. So the world transform is done once, by whoever draws.

    THE FRUSTUM SHAPE DOES NOT DEPEND ON THE RESOLUTION. Both `u - cx` and `fx` scale with the
    image width, so `(u - cx)/fx` at a corner is invariant. GA-233 recorded calibration.json
    going stale (640x480 with fx=320 while the frames were 1280x960); that staleness moves the
    pixel scale and CANNOT move this frustum, provided the aspect ratio is unchanged.
    """
    fx, fy, cx, cy = intr["fx"], intr["fy"], intr["cx"], intr["cy"]
    return [[(u - cx) / fx * depth, (v - cy) / fy * depth, depth]
            for u, v in ((0.0, 0.0), (width, 0.0), (width, height), (0.0, height))]


def visible_boxes(objects, cam_pos, cam_quat_xyzw, intr, width, height):
    """Project every object that has a 3D box. Far ones last, so near labels stay readable."""
    out = []
    for o in objects:
        b = o.get("bbox") or o.get("bbox_3d")
        if not b or not all(k in b for k in ("x_min", "x_max", "y_min", "y_max", "z_min", "z_max")):
            continue
        r = _project_box(b, cam_pos, cam_quat_xyzw, intr, width, height)
        if r is None:
            continue
        if r[4] > MAX_M:
            continue
        out.append({"label": o.get("label") or o.get("object_id") or "?",
                    "u0": r[0], "v0": r[1], "u1": r[2], "v1": r[3], "depth": r[4],
                    "points": r[5], "oriented": r[6],
                    "verdict": (o.get("annotation") or {}).get("verdict", {}).get("grade")
                    if isinstance(o.get("annotation"), dict) else o.get("verdict")})
    out.sort(key=lambda d: -d["depth"])
    return out


# Verdict -> BGR. Same four grades the rest of the dashboard uses, so a box on the video and a
# row in the table cannot disagree about what a colour means.
COLOURS = {"admit": (129, 185, 16), "hold": (21, 191, 234),
           "decline": (113, 113, 248), "no_grounds": (150, 130, 100)}
DEFAULT_COLOUR = (232, 189, 56)


def draw(frame_bgr, boxes, cv2, mask_layer=None):
    """Draw the projected boxes onto a BGR image IN PLACE. cv2 is passed in so this module
    imports on a host with no OpenCV and its geometry stays unit-testable."""
    h, w = frame_bgr.shape[:2]
    if mask_layer is not None:
        cv2.addWeighted(mask_layer, 0.35, frame_bgr, 0.65, 0, frame_bgr)
    for b in boxes:
        colour = COLOURS.get(b.get("verdict"), DEFAULT_COLOUR)
        thick = 2 if b["depth"] < FAR_M else 1
        x0, y0 = int(max(0, b["u0"])), int(max(0, b["v0"]))
        x1, y1 = int(min(w - 1, b["u1"])), int(min(h - 1, b["v1"]))
        if x1 <= x0 or y1 <= y0:
            continue
        points = b.get("points")
        if points and len(points) == 8:
            points = [(int(round(x)), int(round(y))) for x, y in points]
            for i, j in BOX_EDGES:
                cv2.line(frame_bgr, points[i], points[j], colour, thick, cv2.LINE_AA)
        else:
            cv2.rectangle(frame_bgr, (x0, y0), (x1, y1), colour, thick)
        text = f"{b['label']} {b['depth']:.1f}m"
        scale = 0.45 if thick == 2 else 0.38
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
        ty = y0 - 4 if y0 - th - 6 >= 0 else y1 + th + 4      # flip inside when at the top edge
        cv2.rectangle(frame_bgr, (x0, ty - th - 3), (x0 + tw + 4, ty + 2), colour, -1)
        cv2.putText(frame_bgr, text, (x0 + 2, ty), cv2.FONT_HERSHEY_SIMPLEX, scale,
                    (16, 20, 32), 1, cv2.LINE_AA)
    return frame_bgr


def _self_check(bundle):
    """Re-derive the convention from a real bundle. Fails loudly if the projection moves.

    Uses detections.jsonl, which records the camera pose AND the 2D box that resulted, so this
    is a comparison against the pipeline's own output rather than against my arithmetic.
    """
    b = Path(bundle)
    intr = json.loads((b / "calibration.json").read_text())["intrinsics"]
    res = json.loads((b / "calibration.json").read_text())["resolution"]
    W, H = res["width"], res["height"]
    dets = []
    with (b / "detections.jsonl").open() as f:
        for line in f:
            try:
                d = json.loads(line)
            except ValueError:
                continue
            if d.get("camera_position") and d.get("camera_quat_xyzw") and d.get("bbox_2d") \
                    and d.get("bbox_3d"):
                dets.append(d)
    assert dets, f"no usable detections in {b}"
    errs, drawn = [], 0
    for d in dets:
        r = project_box(d["bbox_3d"], d["camera_position"], d["camera_quat_xyzw"], intr, W, H)
        if r is None:
            continue
        drawn += 1
        gu0, gv0, gu1, gv1 = d["bbox_2d"]
        errs.append(math.hypot((r[0] + r[2]) / 2 - (gu0 + gu1) / 2,
                               (r[1] + r[3]) / 2 - (gv0 + gv1) / 2))
    errs.sort()
    p50 = errs[len(errs) // 2]
    print(f"  {drawn}/{len(dets)} detections project in front of their own camera")
    print(f"  centre error vs the recorded 2D box: p50 {p50:.1f} px on {W}x{H}")
    assert drawn >= 0.95 * len(dets), "the recorded pose should put its own detection in view"
    assert p50 < 40, f"median centre error {p50:.1f} px -- the convention has moved"

    # NEGATIVE CONTROL. A projection that never refuses anything would post the same median
    # error and be useless, so put a box BEHIND the camera and require a refusal. Behind is
    # derived from the pose itself: the optical frame looks down +Z, so the world-frame
    # forward direction is R * (0, 0, 1) and stepping the opposite way is behind.
    d = dets[0]
    cp = d["camera_position"]
    R = quat_to_R(*d["camera_quat_xyzw"])
    fwd = [R[i][2] for i in range(3)]
    c = [cp[i] - 5.0 * fwd[i] for i in range(3)]
    behind = {"x_min": c[0] - 0.4, "x_max": c[0] + 0.4, "y_min": c[1] - 0.4,
              "y_max": c[1] + 0.4, "z_min": c[2] - 0.4, "z_max": c[2] + 0.4}
    assert project_box(behind, cp, d["camera_quat_xyzw"], intr, W, H) is None, \
        "a box five metres behind the camera was projected"
    ahead = {}
    for i, k in enumerate(("x", "y", "z")):
        ahead[f"{k}_min"] = cp[i] + 3.0 * fwd[i] - 0.4
        ahead[f"{k}_max"] = cp[i] + 3.0 * fwd[i] + 0.4
    assert project_box(ahead, cp, d["camera_quat_xyzw"], intr, W, H) is not None, \
        "a box three metres straight ahead was refused"
    print("  refuses a box 5 m behind the camera, accepts one 3 m ahead")

    # THE FRUSTUM IS THE SAME CONVENTION AS THE PROJECTION, checked rather than asserted in
    # prose. Take each corner ray, treat it as a world point seen by a camera at the origin
    # with the identity rotation, and project it with THIS module's own arithmetic. It must
    # land back on the image corner it came from. If anyone flips the sign of Y or makes the
    # camera look down -Z, this fails; a frustum drawn the wrong way round otherwise looks
    # entirely plausible on screen.
    rays = frustum_rays(intr, W, H, depth=2.0)
    corners = ((0.0, 0.0), (W, 0.0), (W, H), (0.0, H))
    for (X, Y, Z), (gu, gv) in zip(rays, corners):
        u = intr["fx"] * X / Z + intr["cx"]
        v = intr["fy"] * Y / Z + intr["cy"]
        assert abs(u - gu) < 1e-6 and abs(v - gv) < 1e-6, \
            f"frustum corner ({X:.3f},{Y:.3f},{Z:.3f}) projects to ({u:.2f},{v:.2f}), not ({gu},{gv})"
    assert all(r[2] > 0 for r in rays), "the optical frame looks down +Z; a corner ray went backwards"
    # NEGATIVE CONTROL on that check: a ray built with Y up instead of Y down must FAIL it,
    # or the check above would pass for a frustum drawn upside down.
    flipped = [[X, -Y, Z] for X, Y, Z in rays]
    bad = sum(1 for (X, Y, Z), (gu, gv) in zip(flipped, corners)
              if abs(intr["fy"] * Y / Z + intr["cy"] - gv) > 1.0)
    assert bad >= 2, "a Y-flipped frustum passed the corner check -- the check measures nothing"
    print(f"  frustum: 4 corner rays re-project to their own image corners on {W}x{H}; "
          f"{bad}/4 fail when Y is flipped")
    print("  self-check OK")


if __name__ == "__main__":
    import sys
    target = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("RUN_DIR", "")
    _self_check(target)
