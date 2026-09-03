"""Print the worst frame age the previous run REJECTED, in seconds. GA-283.

Read HOST-SIDE, because preflight_gate.py runs INSIDE the container where /ws/output is the
current bundle and the host's runs/ directory does not exist.

WHY THIS QUANTITY AND NOT THE CYCLE TIME. The first version of this printed the previous run's
`total_ms`, and probe a10 compared the frame-age guard against it. That was wrong in
principle: total_ms sums work that does not gate the loop, so it could fail a healthy run or
pass a deadlocked one. Run 20260903_110622 exhibited both at once -- total_ms 17290.7 ms
beside a 15.0 s guard and ZERO frames rejected.

What the guard is ACTUALLY compared against at runtime is the age of a cached frame, and the
ages at which frames were REJECTED are logged verbatim by utils.py:203:

    Cached frame too old (6.60s), discarding

So this reports the worst such age from the newest run that logged any. If the last run
rejected a frame at 8.52 s and the guard is still 5.0 s, it will reject them again.

Prints an EMPTY LINE when the previous run rejected nothing, or when there is no previous run.
Both mean "no evidence of a problem", and a10 then passes while recording that it asserted
nothing. Neither is an error and neither must read as one.

    python3 last_frame_age_rejected.py /DATA/FOUND/runs      # -> "8.52" or ""
"""
import os
import re
import sys

PAT = re.compile(r"Cached frame too old \(([\d.]+)s\)")


def worst_rejected_age(runs_dir):
    """-> "8.52" or "". Newest bundle first; the first run that logged a rejection wins.

    A run that rejected NOTHING is not skipped over -- it ends the search and returns "".
    Skipping it to find an older run that did reject would compare today's guard against a
    problem that has already been fixed.
    """
    try:
        entries = sorted(os.listdir(runs_dir), reverse=True)
    except OSError:
        return ""
    for d in entries:
        if d == "latest":
            continue                    # a symlink to one of the others, not a second sample
        log = os.path.join(runs_dir, d, "logs", "perception.log")
        if not os.path.isfile(log):
            continue                    # no perception log: this run says nothing either way
        if _bridge_never_bound(runs_dir, d):
            continue                    # see the docstring below: not a sample of a working loop
        worst = 0.0
        try:
            with open(log, errors="replace") as fh:
                for line in fh:
                    m = PAT.search(line)
                    if m:
                        try:
                            worst = max(worst, float(m.group(1)))
                        except ValueError:
                            pass
        except OSError:
            continue
        return f"{worst:.2f}" if worst > 0 else ""
    return ""


def _bridge_never_bound(runs_dir, bundle):
    """True when this run's bridge never got its port. Such a run is NOT a sample.

    GA-292. Run 20260903_223859 collided with an unrelated process on 8081, so the bridge
    never bound and every object_manager POST failed. It rejected 4 frames, worst 18.22 s,
    and a10 then refused the next launch against a 15.0 s guard.

    THE GUARD IS NOT THE THING THAT WAS WRONG. Four runs on 2026-09-03 (110622, 123748,
    135823, 144312) rejected ZERO frames under that same 15.0 s guard, and every rejection in
    223859 came AFTER the first failed POST (unreachable from epoch 1788468265; the first
    rejection over 15 s at 1788468306). Association, not proof of mechanism -- but a run whose
    storage path was dead is not evidence about frame ages when it is alive, and raising the
    guard to clear it would tune the safety limit to fit a fault that is already fixed.

    This is the SAME rule the search already applies to a newer clean run: do not refuse a run
    for a solved fault. Narrow on purpose -- one condition, read from the bundle's own
    bridge.log, and it skips ONLY a bundle that recorded the bind failure.
    """
    blog = os.path.join(runs_dir, bundle, "logs", "bridge.log")
    try:
        with open(blog, errors="replace") as fh:
            return any("address already in use" in ln for ln in fh)
    except OSError:
        return False


def _selfcheck():
    import tempfile
    root = tempfile.mkdtemp()

    def bundle(name, lines):
        d = os.path.join(root, name, "logs")
        os.makedirs(d, exist_ok=True)
        if lines is not None:
            with open(os.path.join(d, "perception.log"), "w") as fh:
                fh.write(lines)
        return d

    assert worst_rejected_age(root) == "", "an empty runs dir must print nothing"
    assert worst_rejected_age(os.path.join(root, "nope")) == "", "a missing dir must not raise"

    bundle("20260101_000000_x", "Cached frame too old (6.60s), discarding\n"
                                "Cached frame too old (8.52s), discarding\n"
                                "Cached frame too old (5.57s), discarding\n")
    assert worst_rejected_age(root) == "8.52", worst_rejected_age(root)

    # A BUNDLE WHOSE BRIDGE NEVER BOUND IS SKIPPED, and the search continues past it.
    b = bundle("20260303_000000_x", "Cached frame too old (18.22s), discarding\n")
    with open(os.path.join(b, "bridge.log"), "w") as fh:
        fh.write("ERROR: [Errno 98] error while attempting to bind on address "
                 "('0.0.0.0', 8081): address already in use\n")
    assert worst_rejected_age(root) == "8.52", worst_rejected_age(root)
    os.remove(os.path.join(b, "bridge.log"))
    assert worst_rejected_age(root) == "18.22", worst_rejected_age(root)
    import shutil
    shutil.rmtree(os.path.join(root, "20260303_000000_x"))

    # THE NEWEST RUN WINS EVEN WHEN IT REJECTED NOTHING. A newer clean run means the problem
    # is fixed; reaching past it to an older bad run would refuse a run for a solved fault.
    bundle("20260202_000000_x", "everything fine here\n")
    assert worst_rejected_age(root) == "", worst_rejected_age(root)

    # A run with no perception log says nothing and is passed over.
    bundle("20260303_000000_x", None)
    assert worst_rejected_age(root) == "", worst_rejected_age(root)

    bundle("20260404_000000_x", "Cached frame too old (12.00s), discarding\n")
    assert worst_rejected_age(root) == "12.00", worst_rejected_age(root)

    # A malformed age must not raise or become a reading.
    bundle("20260505_000000_x", "Cached frame too old (abcs), discarding\n")
    assert worst_rejected_age(root) == "", worst_rejected_age(root)

    os.symlink(os.path.join(root, "20260404_000000_x"), os.path.join(root, "latest"))
    assert worst_rejected_age(root) == "", "`latest` is a name, not a sample"
    print("  last_frame_age_rejected selfcheck OK (empty/missing dir, worst-of-many, newest "
          "clean run wins, dead-bridge bundle skipped, no-log skipped, malformed age ignored, `latest` ignored)")
    return 0


if __name__ == "__main__":
    if "--selfcheck" in sys.argv:
        raise SystemExit(_selfcheck())
    print(worst_rejected_age(sys.argv[1] if len(sys.argv) > 1 else ""))
