#!/usr/bin/env python3
"""Focused offline checks for multi-observation bbox fusion."""
from types import SimpleNamespace

import numpy as np
from bbox_fusion import (
    VOXEL_SIZE_M,
    add_fusion_view,
    apply_request_fusion,
    decode_voxel_keys,
    flatten_voxel_keys,
    fusion_eligible_label,
    fusion_http_fields,
    fusion_payload_from_points,
    fusion_summary,
    merge_bbox_fusion,
    reset_bbox_fusion,
    voxel_keys_from_points,
)


def flat(*keys):
    return [value for key in keys for value in key]


def test_voxel_transport_round_trip():
    points = np.array([
        [0.001, 0.001, 0.001],
        [0.029, 0.020, 0.010],
        [0.031, 0.001, 0.001],
        [np.nan, 1.0, 2.0],
    ])
    keys = voxel_keys_from_points(points)
    assert keys.tolist() == [[0, 0, 0], [1, 0, 0]]
    assert decode_voxel_keys(flatten_voxel_keys(keys)) == {(0, 0, 0), (1, 0, 0)}
    assert fusion_payload_from_points(points, "chair") == flatten_voxel_keys(keys)
    assert fusion_payload_from_points(points, "wall") == []


def test_agreement_starts_at_four_views():
    obj = SimpleNamespace()
    common = ((0, 0, 0), (1, 0, 0))
    for index in range(1, 4):
        assert add_fusion_view(
            obj, "chair", f"view-{index}", VOXEL_SIZE_M,
            flat(*common, (100 + index, 0, 0)),
        )
    assert obj.fused_bbox["required_views"] == 1
    assert obj.fused_bbox["voxel_count"] == 5

    assert add_fusion_view(
        obj, "chair", "view-4", VOXEL_SIZE_M,
        flat(*common, (104, 0, 0)),
    )
    assert obj.fused_bbox["required_views"] == 2
    assert obj.fused_bbox["voxel_count"] == 2
    assert obj.fused_bbox["x_max"] < 0.1


def test_repeated_delivery_and_merge_are_idempotent():
    keeper = SimpleNamespace()
    discard = SimpleNamespace()
    payload = flat((0, 0, 0), (1, 0, 0))
    assert add_fusion_view(keeper, "chair", "same-cycle", VOXEL_SIZE_M, payload)
    assert not add_fusion_view(keeper, "chair", "same-cycle", VOXEL_SIZE_M, payload)
    assert add_fusion_view(discard, "chair", "same-cycle", VOXEL_SIZE_M, payload)
    assert add_fusion_view(
        discard, "chair", "discard-only-cycle", VOXEL_SIZE_M, payload
    )
    assert merge_bbox_fusion(keeper, discard)
    assert fusion_summary(keeper)["view_count"] == 2
    assert not merge_bbox_fusion(keeper, discard)


def test_move_reset_drops_old_position():
    obj = SimpleNamespace()
    add_fusion_view(obj, "chair", "old", VOXEL_SIZE_M, flat((0, 0, 0)))
    reset_bbox_fusion(obj)
    add_fusion_view(obj, "chair", "new", VOXEL_SIZE_M, flat((100, 0, 0)))
    assert fusion_summary(obj)["view_count"] == 1
    assert obj.fused_bbox["x_min"] > 2.9


def test_structural_and_unknown_labels_are_not_fused():
    for label in ("wall", "door#2", "unknown", ""):
        obj = SimpleNamespace()
        assert not fusion_eligible_label(label)
        assert not add_fusion_view(
            obj, label, "view", VOXEL_SIZE_M, flat((0, 0, 0)))
        assert not hasattr(obj, "fused_bbox")


def test_malformed_transport_is_rejected():
    try:
        decode_voxel_keys([1, 2])
    except ValueError as exc:
        assert "multiple of 3" in str(exc)
    else:
        raise AssertionError("malformed voxel payload was accepted")


def test_http_and_service_transport_preserve_view_identity():
    fusion = {
        "view_id": "cycle-17",
        "voxel_m": VOXEL_SIZE_M,
        "flat_keys": flat((0, 0, 0), (1, 2, 3)),
    }
    http = fusion_http_fields(fusion)
    assert http == {
        "fusion_view_id": "cycle-17",
        "fusion_voxel_size_m": VOXEL_SIZE_M,
        "fusion_voxel_keys": flat((0, 0, 0), (1, 2, 3)),
    }

    request = SimpleNamespace(has_fusion_voxels=True, **http)
    obj = SimpleNamespace()
    assert apply_request_fusion(obj, request, label="chair")
    assert fusion_summary(obj)["view_count"] == 1
    assert not apply_request_fusion(obj, request, label="chair")


if __name__ == "__main__":
    tests = [value for name, value in sorted(globals().items())
             if name.startswith("test_") and callable(value)]
    for test in tests:
        test()
    print(f"bbox fusion self-check OK ({len(tests)} checks)")
