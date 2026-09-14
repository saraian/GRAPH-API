"""object_manager_6.locality_ok under rosstub, with fused-box variants; and the same-cycle veto's equality."""
import sys

PM = '/DATA/GRAPH-API/.claude/worktrees/found-merge-algorithm/lost3dsg/src/perception_module'
sys.path.insert(0, PM)
import rosstub  # noqa: E402

rosstub.install()
import object_info  # noqa: E402
import object_manager_6 as om6  # noqa: E402
import object_services  # noqa: E402

print("ASSOCIATION_MARGIN_M", om6.ASSOCIATION_MARGIN_M, "| object_services.LOCALITY_GAP_M", object_services.LOCALITY_GAP_M,
      "| TRACKING_IOU_THRESHOLD still defined:", hasattr(om6, "TRACKING_IOU_THRESHOLD"),
      "| EXPLORATION_IOU_THRESHOLD still defined:", hasattr(om6, "EXPLORATION_IOU_THRESHOLD"))
BOX = {"x_min": 0.0, "x_max": 1.0, "y_min": 0.0, "y_max": 1.0, "z_min": 0.0, "z_max": 1.0}
DET_02 = {**BOX, "x_min": 1.2, "x_max": 2.2, "bbox_2d": [0, 0, 10, 10]}   # gap 0.2 to BOX
DET_05 = {**BOX, "x_min": 1.5, "x_max": 2.5}                             # gap 0.5 to BOX
DET_FUSED_ONLY = {**BOX, "x_min": 1.35, "x_max": 2.35}                   # gap 0.35 to BOX, 0.25 to FUSED
FUSED = {"x_min": -0.1, "x_max": 1.1, "y_min": 0, "y_max": 1, "z_min": 0, "z_max": 1,
         "source": "multi_observation_voxel_agreement", "voxel_size_m": 0.03, "view_count": 2,
         "required_views": 1, "voxel_count": 744, "agreement_fallback": False}
M = om6.ASSOCIATION_MARGIN_M


def mk(bbox, fused="unset"):
    o = object_info.Object("chair", None, bbox)
    if fused != "unset":
        o.fused_bbox = fused
    return o


for name, det, o in [
        ("det gap 0.2, no fused", DET_02, mk(BOX)),
        ("det gap 0.5, no fused", DET_05, mk(BOX)),
        ("det gap 0.35 measured / 0.25 fused", DET_FUSED_ONLY, mk(BOX, FUSED)),
        ("det gap 0.35 measured, fused None", DET_FUSED_ONLY, mk(BOX, None)),
        ("fused = {} (falls back?)", DET_02, mk(BOX, {})),
        ("fused malformed {x_min only}, det overlaps bbox", BOX, mk(BOX, {"x_min": 0.0})),
        ("fused ok, obj.bbox None", DET_02, mk(None, FUSED)),
        ("det None", None, mk(BOX)),
        ("obj None", DET_02, None),
        ("obj.bbox None, no fused", DET_02, mk(None)),
        ("det identical to bbox", BOX, mk(BOX)),
        ("det with extra keys only", {**DET_02, "has_orientation": True, "yaw": 0.3}, mk(BOX))]:
    try:
        print(f"  {name:48s} -> {om6.locality_ok(det, o, M)!r}")
    except Exception as e:
        print(f"  {name:48s} -> RAISES {type(e).__name__}: {e}")

print("\nsame-cycle veto: Object.__eq__ is object.__eq__ (identity)?", object_info.Object.__eq__ is object.__eq__,
      "| __hash__ default?", object_info.Object.__hash__ is object.__hash__)
o1, o2 = mk(BOX), mk(BOX)
o2.object_id = o1.object_id = "same_id"
print("  two distinct Objects with equal fields/ids: o1 in [o2] ->", o1 in [o2], "| o1 in [o1] ->", o1 in [o1])
print("probe_om6_locality done")
