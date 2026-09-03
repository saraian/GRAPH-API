"""Which perception backend the factory hands back, and when it refuses.

The factory warned that `modal_endpoint` was unset — saying it would use a "local
placeholder" — and then returned a Modal backend pointed at an empty URL. Two defects
in three lines: the run continues with a backend that cannot answer, and the message
describes something the code does not do. Working rule 14: a missing component must
stop the run.

Run: python3 test_backend_selection.py   (from this directory; no pytest needed)
"""
import os
import sys
from datetime import datetime

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.dirname(_HERE))   # client.py imports detection_types flat

from client import (  # noqa: E402
    LocalPerceptionBackend,
    ModalPerceptionBackend,
    get_perception_backend,
)


def _without_env(fn):
    previous = os.environ.pop("MODAL_PERCEPTION_URL", None)
    try:
        return fn()
    finally:
        if previous is not None:
            os.environ["MODAL_PERCEPTION_URL"] = previous


def test_modal_without_an_endpoint_refuses_to_build_a_backend():
    def attempt():
        try:
            get_perception_backend({"perception": {"backend": "modal"}})
        except RuntimeError as exc:
            assert "modal_endpoint" in str(exc), exc
            return True
        return False

    assert _without_env(attempt), \
        "the factory returned a Modal backend with an empty endpoint URL"


def test_modal_with_an_endpoint_builds_the_modal_backend():
    """Guards the test above against a repair that refuses every Modal request."""
    backend = get_perception_backend({
        "perception": {"backend": "modal", "modal_endpoint": "https://example.invalid/predict"},
    })
    assert isinstance(backend, ModalPerceptionBackend)


def test_modal_endpoint_may_come_from_the_environment():
    """The env var is a documented source, not a fallback: it is read before the
    decision, and an absent value still stops the run."""
    os.environ["MODAL_PERCEPTION_URL"] = "https://example.invalid/from-env"
    try:
        backend = get_perception_backend({"perception": {"backend": "modal"}})
        assert isinstance(backend, ModalPerceptionBackend)
    finally:
        os.environ.pop("MODAL_PERCEPTION_URL", None)


def test_the_default_backend_is_local():
    assert isinstance(get_perception_backend({}), LocalPerceptionBackend)


# ── every network timeout comes from the configured field ─────────────────────
#
# Run 10 died here. `ModalPerceptionBackend.__init__` takes `timeout_seconds`,
# `detect_and_segment` uses it, and `health()` passed a hardcoded 25.0 — the field
# existed and one call site did not reach it.
#
# Measured cold-start latency, n=7, one before each launch on 2026-08-30:
#
#     25.3  42.3  45.8  41.1  26.6  47.7  45.6      min 25.3   max 47.7
#
# Every sample exceeds 25.0, so the first call after the Modal container goes cold
# could not succeed — not usually; on this evidence never. And the 35.0 default was
# itself below five of the seven, so pointing health() at the field would have fixed
# the wrong half alone. The default must clear the measured MAXIMUM with headroom.

MEASURED_COLD_START_MAX = 47.7      # n=7, 2026-08-30


def test_no_network_timeout_is_hardcoded():
    """Counts code, not occurrences: the comment recording the measurements is not a
    call site. Mutation-tested by restoring the 25.0 and watching this fail."""
    here = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(here, "client.py")
    assert os.path.exists(path), path
    with open(path) as f:
        body = f.read()
    code = "\n".join(line.split("#", 1)[0] for line in body.splitlines()
                     if not line.lstrip().startswith("#"))
    calls = [seg.split(")")[0] for seg in code.split("timeout=")[1:]]
    assert calls, "no timeout= call site found; has the file moved?"
    for call in calls:
        assert call.strip().startswith("self.timeout_seconds"), \
            f"a network call passes a literal timeout: timeout={call.strip()!r}"


def test_the_default_timeout_clears_the_measured_cold_start():
    """A replacement comes from a measured distribution, not from one sample. The
    default must beat the maximum with room, not merely beat the number it replaces."""
    import inspect
    default = inspect.signature(ModalPerceptionBackend.__init__).parameters["timeout_seconds"].default
    assert default > MEASURED_COLD_START_MAX, (default, MEASURED_COLD_START_MAX)
    assert default >= MEASURED_COLD_START_MAX * 1.2, (
        f"default {default} clears the measured max {MEASURED_COLD_START_MAX} by less "
        "than 20%; a cold start slower than any yet seen would fail again")


# ── a failed perception request must say how long it waited ───────────────────
#
# Three runs died on client.py's urlopen and not one recorded the duration, so "it
# timed out" was all anyone had — and a timeout with no elapsed time cannot be told
# from a hang, a slow cold start, or a request the server never began. Measured in
# isolation this call is 43.6 s cold and 1.2-1.4 s warm, inside the 60 s timeout; in-run
# it exceeded 60 s three times. Only the failing run can say by how much.


def _failing_backend(exc):
    import urllib.request

    from client import ModalPerceptionBackend
    backend = ModalPerceptionBackend(endpoint_url="https://example.invalid/predict")

    def boom(*a, **k):
        raise exc
    urllib.request.urlopen = boom            # restored by the caller
    return backend


def _run_and_capture(exc):
    import urllib.request

    import numpy as np
    original = urllib.request.urlopen
    backend = _failing_backend(exc)
    try:
        backend.detect_and_segment(np.zeros((8, 8, 3), dtype=np.uint8), ["chair", "table"])
    except Exception as raised:
        return raised
    finally:
        urllib.request.urlopen = original
    raise AssertionError("the failing request did not raise")


def test_a_timeout_carries_the_elapsed_time_and_keeps_its_type():
    raised = _run_and_capture(TimeoutError("timed out"))
    assert isinstance(raised, TimeoutError), type(raised)
    assert "waited" in str(raised), raised
    assert "60.0s timeout" in str(raised), raised
    assert "2 labels" in str(raised), raised
    assert raised.__cause__ is not None, "the original exception must remain chained"


def test_an_exception_that_takes_no_message_still_reports_the_wait():
    """`HTTPError` wants five arguments. An unguarded `type(exc)(msg)` raises TypeError
    inside the error handler and replaces a timeout with a confusing traceback — losing
    the type is bad, losing the failure is worse."""
    import urllib.error
    raised = _run_and_capture(urllib.error.HTTPError("u", 500, "boom", {}, None))
    assert isinstance(raised, RuntimeError), type(raised)
    assert "waited" in str(raised), raised
    assert isinstance(raised.__cause__, urllib.error.HTTPError)


if __name__ == "__main__":
    failures = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"  ok    {name}")
            except Exception as exc:
                failures += 1
                print(f"  FAIL  {name}: {exc}")
    # The time the result was taken, for the same reason as test_config.py.
    print("backend selection:", "all passed" if not failures else f"{failures} failed",
          "|", datetime.now().astimezone().isoformat(timespec="seconds"))
    sys.exit(1 if failures else 0)
