"""Keep PyTorch's import-time clock calibration on one CPU, then restore affinity."""
import importlib
import os
from pathlib import Path


def _threads():
    return {int(path.name) for path in Path('/proc/self/task').iterdir()}


def import_torch():
    if not hasattr(os, 'sched_getaffinity') or not hasattr(os, 'sched_setaffinity'):
        return importlib.import_module('torch')
    allowed = os.sched_getaffinity(0)
    existing = _threads()
    os.sched_setaffinity(0, {min(allowed)})
    try:
        return importlib.import_module('torch')
    finally:
        os.sched_setaffinity(0, allowed)
        # Native threads created during import inherit the temporary mask.
        # Restore their full mask too; inference must retain its normal CPU pool.
        for tid in _threads() - existing:
            try:
                os.sched_setaffinity(tid, allowed)
            except ProcessLookupError:
                pass
