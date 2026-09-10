"""Test harness for Cloud Perception Backend and RLE codec."""

import os

import numpy as np
from client import ModalPerceptionBackend, rle_decode, rle_encode


def test_rle_codec():
    # Create synthetic binary mask
    h, w = 120, 160
    mask = np.zeros((h, w), dtype=np.uint8)
    mask[20:60, 30:90] = 1
    mask[80:100, 100:140] = 1

    encoded = rle_encode(mask)
    assert "counts" in encoded
    assert encoded["size"] == [h, w]

    decoded = rle_decode(encoded)
    assert decoded.shape == (h, w)
    assert np.array_equal(mask, decoded)
    print("test_rle_codec: PASSED")


def test_modal_client_mock():
    # Test client instantiation and offline health check handling
    client = ModalPerceptionBackend(endpoint_url="http://127.0.0.1:9999")
    h = client.health()
    assert h["reachable"] is False  # Expected when no server running on port 9999
    print("test_modal_client_mock: PASSED")


def test_crop_regions():
    """GA-342 / GA-17: a sliver under MIN_CROP_PX is skipped and the surviving crops keep
    their BOX index, so a skipped crop cannot shift another object's embedding onto it.
    Imports the service module with a stub `modal` so this runs on any host."""
    import sys
    import types

    if "modal" not in sys.modules:
        class _Chain:
            def __getattr__(self, _name):
                return lambda *a, **k: self

        stub = types.ModuleType("modal")
        stub.Image = _Chain()
        stub.App = lambda *a, **k: _Chain()
        stub.enter = lambda *a, **k: (lambda f: f)
        stub.fastapi_endpoint = lambda *a, **k: (lambda f: f)
        sys.modules["modal"] = stub
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import modal_perception as mp

    rgb = np.zeros((100, 100, 3), dtype=np.uint8)
    boxes = np.array([[0, 0, 50, 50], [10, 10, 12, 13], [90, 90, 120, 120]], dtype=float)
    regions = mp.crop_regions(boxes, rgb, 100, 100)
    assert [i for i, _ in regions] == [0, 2], regions          # the 2x3 sliver is skipped
    assert regions[1][1].shape == (10, 10, 3), regions[1][1].shape   # clipped to the image
    assert mp.MIN_CROP_PX == 4
    print("test_crop_regions: PASSED")


def test_modal_client_live():
    # Test client calling live deployed Modal endpoint
    endpoint = os.environ.get("MODAL_PERCEPTION_URL", "")
    if not endpoint:
        # The URL is a credential (the endpoint has no proxy auth), so it is not
        # written down here. Export MODAL_PERCEPTION_URL to run this against the
        # real service.
        print("SKIP: MODAL_PERCEPTION_URL is not set")
        raise SystemExit(0)
    client = ModalPerceptionBackend(endpoint_url=endpoint)
    h = client.health()
    assert h["reachable"] is True, f"Health check failed: {h}"
    print(f"test_modal_client_live (health): PASSED ({h['details']})")

    # Synthetic RGB image with a blue square (chair proxy)
    rgb = np.zeros((480, 640, 3), dtype=np.uint8)
    rgb[100:300, 150:400] = [200, 50, 50]  # red box

    detections, timings = client.detect_and_segment(rgb, ["box", "chair", "table"])
    print(f"test_modal_client_live (predict): PASSED (returned {len(detections)} detections, timings={timings})")


if __name__ == "__main__":
    test_rle_codec()
    test_modal_client_mock()
    test_crop_regions()
    test_modal_client_live()
    print("ALL CLOUD PERCEPTION TESTS PASSED.")
