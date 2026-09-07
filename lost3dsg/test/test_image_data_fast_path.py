"""The feed node must hand uint8[] message fields an array('B'), never bytes: rclpy's generated
setter validates bytes element by element (7.4 s per 1280x960 frame pair, measured 2026-09-07)."""
import array
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src", "perception_module"))
import rosstub  # noqa: E402

rosstub.install()
import habitat_feed_node as fn  # noqa: E402

out = fn._u8(b"\x00\x01\xff")
assert isinstance(out, array.array) and out.typecode == "B" and out.tobytes() == b"\x00\x01\xff"
src = open(fn.__file__).read()
for field in ("rgb.data", "depth.data", "sem_msg.data"):
    line = next(l for l in src.splitlines() if l.strip().startswith(field + " ="))
    assert "_u8(" in line, f"{field} is assigned without the fast path: {line.strip()}"
print("OK: rgb, depth and the semantic blob go through array('B')")
