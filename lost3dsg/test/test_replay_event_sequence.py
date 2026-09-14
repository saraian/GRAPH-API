#!/usr/bin/env python3
"""ReplayCapture.event() must give every row a distinct, contiguous sequence number under
concurrent callers. Before the lock (2026-09-14) it read `self._sequence`, did an fsync'd
append, then incremented, so two ROS callback groups could take the same number and
`finalize()` raised "GA-493 replay event sequence is not contiguous" at teardown.

Negative control: monkeypatch the lock away and the same load produces duplicates.
"""
import contextlib
import json
import os
import sys
import tempfile
import threading
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src" / "perception_module"))
from ga493_replay_capture import ReplayCapture  # noqa: E402

THREADS, PER_THREAD = 8, 40


def run(root, with_lock=True):
    cap = ReplayCapture(Path(root), "consumer", max_cycles=1, max_bytes=50 * 1024 * 1024)
    if not with_lock:
        cap._event_lock = contextlib.nullcontext()
    start = threading.Barrier(THREADS)

    def worker(n):
        start.wait()
        for i in range(PER_THREAD):
            cap.event(f"kind_{n}", {"i": i})

    threads = [threading.Thread(target=worker, args=(n,)) for n in range(THREADS)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    rows = [json.loads(line) for line in cap.events_path.read_text().splitlines() if line.strip()]
    return rows, cap._sequence


with tempfile.TemporaryDirectory() as tmp:
    rows, seq = run(os.path.join(tmp, "locked"))
    seqs = sorted(r["sequence"] for r in rows)
    assert len(rows) == THREADS * PER_THREAD, len(rows)
    assert seqs == list(range(len(rows))), "sequences must be distinct and contiguous"
    assert seq == len(rows), (seq, len(rows))
    # finalize()'s own check: len(event_rows) == self._sequence
    print(f"OK locked: {len(rows)} rows, sequences 0..{len(rows) - 1}, no duplicates")

with tempfile.TemporaryDirectory() as tmp:
    rows, seq = run(os.path.join(tmp, "unlocked"), with_lock=False)
    dupes = len(rows) - len({r["sequence"] for r in rows})
    print(f"negative control (no lock): {len(rows)} rows, {dupes} duplicate sequence number(s)"
          + ("" if dupes else "  -- not reproduced this time; the race is timing-dependent"))
