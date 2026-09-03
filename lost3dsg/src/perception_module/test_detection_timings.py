#!/usr/bin/env python3
"""GA-85 follow-on: zero detections must not be read as a broken timing contract.

Run 19 died on its fourth cycle with

    RuntimeError: perception backend returned no timing for ['detector', 'sam2']
    (reported keys: ['total', 'yolo_world'])

Its three earlier cycles reported detector/sam2 and produced 4, 3 and 1 detections. The
fourth found nothing: the server returns early when the detector finds no boxes, SAM never
runs, and there is no sam2 stage to report. The check was demanding a measurement of work
that was not done, so a legitimate empty frame killed a measured run.

The two cases that must not be confused, and this file is what keeps them apart:

  zero detections, no stage keys  -> a RESULT. Return empty, record no timing, do not raise.
  detections > 0, no stage keys   -> the CONTRACT VIOLATION GA-14 is for. Still raises.

The empty path reads no timing key by name, so it is correct under both the deployed Modal
build (which calls that timing `yolo_world`) and the built one (which calls it `detector`).
Case 4 is the guard against "fixing" this by accepting `yolo_world` as a detector alias:
under such a fix it would stop raising.

Run: python3 test_detection_timings.py
"""
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import rosstub  # noqa: E402

rosstub.install()

import detection_pipeline as dp  # noqa: E402

assert pathlib.Path(dp.__file__).resolve().parent == HERE, \
    f"testing the wrong tree: imported {dp.__file__}"

RUN19_KEYS = {"total": 812.0, "yolo_world": 780.4}     # verbatim from run 19's traceback
REDEPLOYED = {"total": 812.0, "detector": 780.4}       # modal_perception.py:188 after rename


class StubBackend:
    def __init__(self, detections, timings):
        self._d, self._t = detections, timings

    def detect_and_segment(self, rgb, labels, **kw):
        return self._d, self._t


class Host(dp.DetectionPipelineMixin):
    """The five members run_detection reaches before the timing block."""

    def __init__(self, backend):
        self.perception_backend = backend
        self.logged = []

    def log_both(self, level, msg):
        self.logged.append(msg)

    def _refresh_room_geometry_if_available(self):
        pass

    def _abort_if_moving(self, _where):
        return False

    def _extract_detection_labels(self, _rgb):
        return ["doorway"]           # run 19's fourth cycle returned exactly this


def run(detections, timings):
    dp.CFG = {"perception": {"backend": "modal"}}
    host = Host(StubBackend(detections, timings))
    return host, host.run_detection({"rgb": object()})


checks = []


def check(name, fn):
    try:
        fn()
        checks.append((name, None))
    except Exception as exc:
        checks.append((name, f"{type(exc).__name__}: {exc}"))


def empty_deployed_build():
    host, out = run([], RUN19_KEYS)
    assert out == [], f"expected no detections, got {out!r}"
    assert any("found no objects" in m for m in host.logged), host.logged
    # the reported keys are the discriminant that named the deploy skew; they must be logged
    assert any("yolo_world" in m for m in host.logged), \
        f"reported timing keys are not in the log: {host.logged}"


def empty_redeployed_build():
    _host, out = run([], REDEPLOYED)
    assert out == [], f"expected no detections, got {out!r}"


def empty_no_keys_at_all():
    _host, out = run([], {})
    assert out == [], f"expected no detections, got {out!r}"


def detections_without_stage_keys_still_raise():
    try:
        run([{"label": "doorway"}], RUN19_KEYS)
    except RuntimeError as exc:
        assert "detector" in str(exc) and "sam2" in str(exc), exc
        assert "yolo_world" in str(exc), f"the error must name what DID arrive: {exc}"
        return
    raise AssertionError("detections with no stage timing did not raise; GA-14 is gone")


def full_response_is_untouched():
    ok = {"total": 900.0, "detector": 700.0, "sam2": 150.0}
    try:
        run([{"label": "doorway"}], ok)
    except RuntimeError as exc:
        raise AssertionError(f"a complete timing dict raised: {exc}")
    except Exception:
        pass    # anything further downstream is not this file's subject


for name, fn in [("empty + deployed build keys ['total','yolo_world']", empty_deployed_build),
                 ("empty + built keys ['total','detector']", empty_redeployed_build),
                 ("empty + no timings at all", empty_no_keys_at_all),
                 ("detections + no stage keys STILL raises", detections_without_stage_keys_still_raise),
                 ("complete timing dict does not raise", full_response_is_untouched)]:
    check(name, fn)

width = max(len(n) for n, _ in checks)
failed = 0
for name, err in checks:
    if err:
        failed += 1
        print(f"  FAIL  {name:<{width}}  {err}")
    else:
        print(f"  ok    {name}")
print(f"\n{len(checks) - failed}/{len(checks)} checks passed")
sys.exit(1 if failed else 0)
