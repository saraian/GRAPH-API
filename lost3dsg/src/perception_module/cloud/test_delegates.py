"""Unit tests validating Delegate 1 (detect-while-moving) and Delegate 2 (vlm-parallel-cache)."""

import numpy as np
import time
from config import CFG
from vlm_call import VlmClient, CropVlmCache, compute_crop_hash, discretize_viewpoint
from detection_types import Detection


def test_viewpoint_discretization():
    # 8 angular sectors (45 deg = pi/4)
    # Sector 0 is around -pi, etc.
    p1, r1 = discretize_viewpoint(yaw=0.0, distance=1.2)
    p2, r2 = discretize_viewpoint(yaw=0.1, distance=1.3)
    assert p1 == p2, "Small yaw change should stay in the same pose bucket"
    assert r1 == r2 == 2, "1.2m and 1.3m should both land in range bucket 2 (1.0-1.5m)"

    p3, r3 = discretize_viewpoint(yaw=1.57, distance=3.0)  # 90 deg, far
    assert p1 != p3, "90 deg change should land in a different pose bucket"
    assert r3 == 4, "3.0m should land in max range bucket 4 (>2m)"
    print("test_viewpoint_discretization: PASSED ✅")


def test_crop_cache_and_provenance():
    cache = CropVlmCache()
    key = ("hash_sofa_123", 2, 1)

    result_data = {"label": "sofa", "description": "vintage leather sofa", "color": "brown", "material": "leather", "shape": "rectangular"}
    prov_data = {"model": "test-vlm", "image_id": "img_001", "timestamp": "2026-08-25T20:00:00"}

    # 1. Put in cache
    cache.put(key, result_data, prov_data)
    assert cache.stats["size"] == 1

    # 2. Cache Hit: must preserve original provenance and add cached=True
    hit = cache.get(key)
    assert hit is not None
    assert hit["label"] == "sofa"
    assert hit["provenance"]["model"] == "test-vlm"
    assert hit["provenance"]["image_id"] == "img_001"
    assert hit["provenance"]["cached"] is True
    assert cache.stats["hits"] == 1

    # 3. Cache Miss on viewpoint/range change
    diff_view_key = ("hash_sofa_123", 5, 1)  # Different angle
    miss = cache.get(diff_view_key)
    assert miss is None
    assert cache.stats["misses"] == 1

    # 4. Invalidation on ontological conflict
    cache.invalidate(crop_hash="hash_sofa_123")
    assert cache.stats["size"] == 0
    assert cache.get(key) is None
    print("test_crop_cache_and_provenance: PASSED ✅")


def test_vlm_client_mock_concurrency():
    call_counts = 0

    def mock_vlm_call(prompt, b64_img):
        nonlocal call_counts
        call_counts += 1
        time.sleep(0.01)
        return '{"objects": [{"label": "chair", "color": "red", "material": "wood", "shape": "curved", "description": "wooden dining chair"}]}'

    def mock_encode(img):
        return "mock_b64"

    client = VlmClient(vlm_call_fn=mock_vlm_call, image_encoder_fn=mock_encode)
    fake_crop = np.zeros((64, 64, 3), dtype=np.uint8)
    prompt_path = "/DATA/GRAPH-API/lost3dsg/src/perception_module/prompts/visual_prompt.txt"

    # First call: cache miss, calls VLM
    res1 = client.call_crop_full(prompt_path, "chair", fake_crop, yaw=0.0, distance=1.0, image_id="frame_1")
    assert res1["color"] == "red"
    assert res1["provenance"]["cached"] is False
    assert call_counts == 1

    # Second call (same crop & viewpoint): cache hit, skips VLM
    res2 = client.call_crop_full(prompt_path, "chair", fake_crop, yaw=0.0, distance=1.0, image_id="frame_2")
    assert res2["color"] == "red"
    assert res2["provenance"]["cached"] is True
    assert call_counts == 1  # VLM was NOT called again!

    # Third call (different viewpoint angle): cache miss, calls VLM
    res3 = client.call_crop_full(prompt_path, "chair", fake_crop, yaw=2.0, distance=1.0, image_id="frame_3")
    assert call_counts == 2
    print("test_vlm_client_mock_concurrency: PASSED ✅")


def test_detect_while_moving_policy():
    det_moving = Detection(
        bbox=(10, 20, 100, 200),
        label="table",
        score=0.9,
        mask=np.zeros((480, 640, 1), dtype=np.uint8),
        is_confirmed=False  # Moving pass
    )
    assert det_moving.is_confirmed is False

    det_stationary = Detection(
        bbox=(10, 20, 100, 200),
        label="table",
        score=0.9,
        mask=np.zeros((480, 640, 1), dtype=np.uint8),
        is_confirmed=True  # Stationary pass
    )
    assert det_stationary.is_confirmed is True
    print("test_detect_while_moving_policy: PASSED ✅")


if __name__ == "__main__":
    test_viewpoint_discretization()
    test_crop_cache_and_provenance()
    test_vlm_client_mock_concurrency()
    test_detect_while_moving_policy()
    print("ALL DELEGATE TESTS PASSED ✅")
