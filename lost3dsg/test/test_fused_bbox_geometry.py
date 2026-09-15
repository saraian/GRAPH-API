"""Two ways the fused box lied about the object, both MEASURED on 20260915_014149.

1. CONSENSUS COLLAPSE. The view-agreement rule keeps only voxels that enough views agree on.
   After a merge the keeper holds views of two tracks, and views of two distinct objects agree
   on almost nothing: a merged sofa kept 43 of 5,849 voxels (0.7 %) and fitted a
   0.60 x 0.42 x 0.03 m box, IoU 0.015 against ground truth. Genuine consensus on the same
   runs kept 13-52 %. Below a retained-fraction floor the box now falls back to every observed
   voxel, exactly as it already did when nothing agreed, and says so.

2. A VOXEL HAS EXTENT. The box was fitted to voxel CENTRES, so one layer of voxels -- every
   flat object -- fitted a box of zero thickness, zero volume and IoU 0.0 however well it was
   placed: rug 156/156 voxels, table#1 832/832, book 74/74, all volume 0.0 on that run.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src", "perception_module"))
import rosstub  # noqa: E402

rosstub.install()
import bbox_fusion as bf  # noqa: E402

V = 0.03


def state(voxels, views):
    """voxels: {(ix, iy, iz): set(view_ids)}"""
    return {"voxel_m": V, "views": set(views), "voxels": dict(voxels)}


def dims(b):
    return (b["x_max"] - b["x_min"], b["y_max"] - b["y_min"], b["z_max"] - b["z_min"])


def test_one_layer_of_voxels_has_the_thickness_of_a_voxel():
    # a 10 x 10 plane of voxels, one layer thick, one view
    vox = {(x, y, 0): {"v1"} for x in range(10) for y in range(10)}
    b = bf.fused_bbox_from_state(state(vox, {"v1"}))
    dx, dy, dz = dims(b)
    assert abs(dz - V) < 1e-9, dz                       # was 0.0
    assert abs(dx - 10 * V) < 1e-9 and abs(dy - 10 * V) < 1e-9, (dx, dy)
    assert dx * dy * dz > 0.0


def test_collapsed_consensus_falls_back_to_every_observed_voxel_and_says_so():
    # views v1..v4 saw a 10x10x10 block; only ONE voxel is seen by two views
    views = {"v1", "v2", "v3", "v4"}
    vox = {(x, y, z): {"v1"} for x in range(10) for y in range(10) for z in range(10)}
    vox[(5, 5, 5)] = {"v1", "v2"}
    assert bf.required_agreement(len(views)) == bf.MIN_AGREEING_VIEWS == 2
    b = bf.fused_bbox_from_state(state(vox, views))
    assert b["agreement_fallback"] is True, b
    assert b["retained_fraction"] < bf.MIN_RETAINED_FRACTION, b["retained_fraction"]
    assert b["voxel_count"] == 1000, b["voxel_count"]     # every observed voxel, not the one
    dx, dy, dz = dims(b)
    assert abs(dx - 10 * V) < 1e-9, dx                    # was one voxel wide


def test_genuine_consensus_is_still_trimmed():
    # 4 views; the central 6x6x6 core is seen by all, a 10x10x10 shell by one
    views = {"v1", "v2", "v3", "v4"}
    vox = {(x, y, z): {"v1"} for x in range(10) for y in range(10) for z in range(10)}
    for x in range(2, 8):
        for y in range(2, 8):
            for z in range(2, 8):
                vox[(x, y, z)] = set(views)
    b = bf.fused_bbox_from_state(state(vox, views))
    assert b["agreement_fallback"] is False, b
    assert b["voxel_count"] == 216, b["voxel_count"]
    assert abs(b["retained_fraction"] - 0.216) < 1e-9, b["retained_fraction"]
    dx, dy, dz = dims(b)
    assert abs(dx - 6 * V) < 1e-9, dx                     # the core, plus half a voxel each side


if __name__ == "__main__":
    n = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            n += 1
            print(f"  ok  {name}")
    print(f"{n} checks passed")
