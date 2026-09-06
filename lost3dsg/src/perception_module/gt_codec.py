"""Lossless wire encoding for habitat's per-pixel instance-id frame (GA-330 follow-up).

The semantic sensor renders uint32 ids at the colour sensor's resolution: 4.9 MB per frame at
1280x960. Sent raw inside the feed pickle AND republished raw as a reliable ROS Image, it cut the
feed from 0.56 to 0.10 frames/s and dropped the socket twice in run 20260906_234050. PNG cannot
hold 32-bit samples, so the id is split into two 16-bit planes (low, high) plus a zero plane and
encoded as one 16-bit RGB PNG: exact for every uint32, and ~1-3% of the raw size on a real scene
(hundreds of distinct ids, large flat regions). Both ends of both hops use these two functions.
"""
import numpy as np


def encode(sem) -> bytes:
    import cv2
    a = np.ascontiguousarray(sem, dtype=np.uint32)
    if a.ndim != 2:
        raise ValueError(f"semantic frame must be HxW, got shape {a.shape}")
    planes = np.dstack([(a & 0xFFFF).astype(np.uint16), (a >> 16).astype(np.uint16),
                        np.zeros(a.shape, np.uint16)])
    ok, buf = cv2.imencode(".png", planes, [cv2.IMWRITE_PNG_COMPRESSION, 3])
    if not ok:
        raise RuntimeError("PNG encode of the semantic frame failed")
    return buf.tobytes()


def decode(data: bytes):
    """-> int32 HxW (the ROS-side dtype the archive join already reads), or None on a bad blob."""
    import cv2
    buf = np.frombuffer(data, dtype=np.uint8)
    planes = cv2.imdecode(buf, cv2.IMREAD_UNCHANGED)
    if planes is None or planes.ndim != 3 or planes.shape[2] < 2 or planes.dtype != np.uint16:
        return None
    ids = planes[..., 0].astype(np.uint32) | (planes[..., 1].astype(np.uint32) << 16)
    return ids.astype(np.int32)


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    a = rng.integers(0, 1 << 20, size=(96, 128), dtype=np.uint32)
    a[10:40, 20:90] = 907          # a flat region, like a wall
    a[0, 0] = 0xFFFFFFFF           # the extreme value must survive
    blob = encode(a)
    back = decode(blob)
    assert back is not None and back.shape == a.shape
    assert np.array_equal(back.astype(np.uint32), a), "round trip is not exact"
    real = np.full((960, 1280), 7, np.uint32)
    real[300:600, 200:900] = 412
    real[::7, ::5] = 901
    blob2 = encode(real)
    print(f"gt_codec self-check OK: random {a.nbytes} B -> {len(blob)} B; scene-like "
          f"{real.nbytes} B -> {len(blob2)} B ({100.0 * len(blob2) / real.nbytes:.2f}%)")
    assert len(blob2) < 0.05 * real.nbytes
