#!/usr/bin/env python3
"""Collector lanciato da habitat_launch.py; salva le metriche alla chiusura."""
from __future__ import annotations

import argparse
import signal
import threading
import time
from pathlib import Path

from habitat_run_metrics import _state, collect, write


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    state = _state(args.run_dir)
    stopped = threading.Event()
    reason = {"value": "launch terminato"}

    def stop(signum, _frame):
        reason["value"] = signal.Signals(signum).name
        stopped.set()

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)
    print(f"[metriche] raccolta automatica attiva -> {args.output}", flush=True)
    stopped.wait()
    # Gli altri nodi ricevono lo stesso segnale. Un piccolo margine consente
    # alle loro scritture finali atomiche di completarsi.
    time.sleep(1.0)
    write(collect(state, ended_at=time.time(), stop_reason=reason["value"]), args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
