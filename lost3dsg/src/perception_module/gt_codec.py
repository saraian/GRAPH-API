"""Lossless wire encoding for habitat's per-pixel instance-id frame (GA-330 follow-up).

The semantic sensor renders uint32 ids at the colour sensor's resolution: 4.9 MB per frame at
1280x960. Sent raw inside the feed pickle AND republished raw as a reliable ROS Image, it cut the
feed from 0.56 to 0.10 frames/s and dropped the socket twice in run 20260906_234050. The first
replacement, a 16-bit PNG, was exact and 1% of the size but cost 245 ms per frame on the host's
single feed thread (run 20260907_001120: 23 frames at 120 s against run C's 74). An instance-id
frame is flat regions, so a RUN-LENGTH encoding is exact, tiny, and a few milliseconds each way
with nothing but numpy. Both ends of both hops use these two functions.

Format: magic b"GTRL", uint32 height, uint32 width, uint32 n_runs, then n_runs uint32 values and
n_runs uint32 run lengths, all little-endian.
"""
import struct

import numpy as np

_MAGIC = b"GTRL"


def encode(sem) -> bytes:
    a = np.ascontiguousarray(sem, dtype=np.uint32)
    if a.ndim != 2:
        raise ValueError(f"semantic frame must be HxW, got shape {a.shape}")
    flat = a.ravel()
    if flat.size == 0:
        return _MAGIC + struct.pack("<III", a.shape[0], a.shape[1], 0)
    starts = np.flatnonzero(np.diff(flat)) + 1
    starts = np.concatenate(([0], starts))
    vals = flat[starts].astype("<u4")
    lens = np.diff(np.concatenate((starts, [flat.size]))).astype("<u4")
    return _MAGIC + struct.pack("<III", a.shape[0], a.shape[1], vals.size) + vals.tobytes() + lens.tobytes()


def decode(data: bytes):
    """-> int32 HxW (the ROS-side dtype the archive join already reads), or None on a bad blob."""
    if data is None or len(data) < 16 or bytes(data[:4]) != _MAGIC:
        return None
    h, w, n = struct.unpack("<III", bytes(data[4:16]))
    if (len(data) - 16) % 4:
        return None                      # a truncated blob is refused, not raised on
    body = np.frombuffer(data, dtype="<u4", offset=16)
    if body.size != 2 * n:
        return None
    vals, lens = body[:n], body[n:]
    if n == 0 or int(lens.sum()) != h * w:
        return None if h * w else np.zeros((h, w), np.int32)
    return np.repeat(vals, lens).reshape(h, w).astype(np.int32)


if __name__ == "__main__":
    import time
    rng = np.random.default_rng(0)
    a = rng.integers(0, 1 << 20, size=(96, 128), dtype=np.uint32)
    a[10:40, 20:90] = 907          # a flat region, like a wall
    a[0, 0] = 0xFFFFFFFF           # the extreme value must survive
    back = decode(encode(a))
    assert back is not None and back.shape == a.shape
    assert np.array_equal(back.astype(np.uint32), a), "round trip is not exact"
    for cut in (1, 2, 3, 5, 8):
        assert decode(encode(a)[:-cut]) is None, f"a blob cut by {cut} bytes must be refused"
    assert decode(b"junk") is None
    real = np.full((960, 1280), 7, np.uint32)
    for _ in range(200):           # ~200 instance regions, like a real scene
        y, x = rng.integers(0, 900), rng.integers(0, 1200)
        real[y:y + rng.integers(10, 120), x:x + rng.integers(10, 160)] = rng.integers(1, 908)
    t = time.perf_counter()
    blob = encode(real)
    te = (time.perf_counter() - t) * 1000
    t = time.perf_counter()
    back = decode(blob)
    td = (time.perf_counter() - t) * 1000
    assert np.array_equal(back.astype(np.uint32), real)
    print(f"gt_codec self-check OK: scene-like {real.nbytes} B -> {len(blob)} B "
          f"({100.0 * len(blob) / real.nbytes:.2f}%), encode {te:.1f} ms, decode {td:.1f} ms")
    assert te < 500 and td < 500, "the point of this codec is speed (74/64 ms measured on a 400 MHz-throttled host)"
