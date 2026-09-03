"""Test harness for Cloud Perception Backend and RLE codec."""

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


def test_modal_client_live():
    # Test client calling live deployed Modal endpoint
    endpoint = "https://emanuelemusumeci--lost3dsg-perception-perceptionservice-predict.modal.run"
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
    test_modal_client_live()
    print("ALL CLOUD PERCEPTION TESTS PASSED.")
