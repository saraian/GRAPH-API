"""GA-334: every completed cycle appends one JSON line to perception_latencies.jsonl (the
snapshot's keys plus t, cycle, frame_id, n_detections), the bridge's _cycle_seq counts those
lines as the cycle number, and a node start truncates the series."""
import json
import os
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace as NS

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src", "perception_module"))
import rosstub  # noqa: E402

rosstub.install()
import graph_api_bridge as gb  # noqa: E402
import perception_2 as p2  # noqa: E402

td = tempfile.mkdtemp()
p2.LATENCY_JSON_PATHS = (os.path.join(td, "perception_latencies.json"),)
p2.LATENCY_JSONL_PATHS = (os.path.join(td, "perception_latencies.jsonl"),)
assert p2.LATENCY_JSONL_PATHS[0].endswith(".jsonl")

node = NS(_io_executor=NS(submit=lambda fn, *a: fn(*a)), latest_latencies={"total_ms": 1.5, "nms_ms": 0.0})
p2._truncate_cycle_series()
p2.DetectObjectsNode._record_cycle_ms(node, 0.5, stages={"crops": 2.0}, frame_id="000123", n_detections=3)
p2.DetectObjectsNode._record_cycle_ms(node, 0.7, frame_id="000124", n_detections=0)

rows = [json.loads(line) for line in open(p2.LATENCY_JSONL_PATHS[0])]
assert [r["cycle"] for r in rows] == [1, 2], rows
assert rows[0]["frame_id"] == "000123" and rows[0]["n_detections"] == 3
assert rows[0]["cycle_ms"] == 500.0 and rows[0]["total_ms"] == 1.5 and rows[0]["nms_ms"] == 0.0
assert rows[0]["stages_ms"] == {"crops": 2.0} and rows[0]["t"] == rows[0]["last_updated"]
assert rows[1]["n_detections"] == 0 and rows[1]["cycle_ms"] == 700.0
snapshot = json.load(open(p2.LATENCY_JSON_PATHS[0]))
assert snapshot["cycle_ms"] == 700.0 and "cycle" not in snapshot, "the snapshot keeps its shape"

# the bridge reads the same file from the active output directory
gb._active_output_dir = lambda: Path(td)
gb._CYCLE_TALLY.update(path=None, offset=0, count=0)
assert gb._cycle_seq() == 2, gb._cycle_seq()
p2._truncate_cycle_series()
assert gb._cycle_seq() is None, "a fresh series must not carry the old count"
print("test_ga334_cycle_series: ok")
