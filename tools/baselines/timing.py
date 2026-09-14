"""External wall-clock timing of native entry points, never inference estimates."""
import json
import time
from contextlib import contextmanager
from pathlib import Path


@contextmanager
def phase(rows, name):
    start = time.perf_counter()
    yield
    rows[name] = time.perf_counter() - start


def save(output, phases, scope):
    doc = {'schema': 'graphapi.baseline_timing.v1', 'clock': 'time.perf_counter',
           'phases_s': phases, 'scope': scope,
           'limitations': ['Wall time includes waits and I/O within the named phase.',
                           'These measurements are not per-frame inference or input-to-result latency.']}
    with (Path(output) / 'execution_timing.json').open('x') as stream:
        json.dump(doc, stream, indent=2, allow_nan=False)
