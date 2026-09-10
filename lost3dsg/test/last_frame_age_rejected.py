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

ONLY LOCALIZATION RUNS ARE SAMPLES (owner ruling, 4 Sep ~15:55: "we will not use slam"; GA-290
register, SLAM ban). a10 guards a LOCALIZATION launch, and the SLAM run 20260904_082146 starved
frames to 24.30 s at ~0.14 Hz because the graph it was optimising grew under it -- a mode
artifact, not a modal-run reading, and a10 then refused the authorized modal run for it. A
bundle whose run_metadata.json cannot certify localization (`resolved_config.localize_db`)
is skipped like the dead-bridge and empty-log bundles: a bundle is evidence only about a
pipeline that executed, in the mode that is about to be launched.

Prints an EMPTY LINE when the previous run rejected nothing, or when there is no previous run.
Both mean "no evidence of a problem", and a10 then passes while recording that it asserted
nothing. Neither is an error and neither must read as one.

    python3 last_frame_age_rejected.py <RUNS_DIR>            # -> "8.52" or ""
"""
import json
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
        try:
            if os.path.getsize(log) == 0:
                continue                # GA-301: an EMPTY log is not evidence of a clean run
        except OSError:
            continue
        if _not_localization(runs_dir, d):
            continue                    # SLAM ban: not a sample of a localization run
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

    GA-294. (Renumbered from GA-292, which the orchestrator had already assigned to the
    owner-ordered view MIN_SUPPORT fix.) Run 20260903_223859 collided with an unrelated process on 8081, so the bridge
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


def _not_localization(runs_dir, bundle):
    """True when this bundle was not a LOCALIZATION run, or cannot prove it was.

    Owner ruling, 4 Sep ~15:55: "we will not use slam" (GA-290 register, SLAM ban). SLAM and
    mapping-only runs are not samples of a localization run: the SLAM run 20260904_082146
    starved frames to 24.30 s at ~0.14 Hz -- the graph it was optimising grew under it -- and
    a10 then refused the authorized modal launch on that reading, a mode artifact read as a
    modal-run reading.

    The mode is certified by run_metadata.json -> resolved_config.localize_db: a localization
    run carries the published map's path there; SLAM and mapping-only carry null. A bundle
    whose metadata is missing, malformed, or lacks resolved_config cannot certify its mode
    either way, so it is skipped too -- the same rule as the empty log (GA-301) and the dead
    bridge (GA-294): a bundle is evidence only about a pipeline that executed, in the mode
    that is about to be launched.
    """
    meta = os.path.join(runs_dir, bundle, "run_metadata.json")
    try:
        with open(meta, errors="replace") as fh:
            resolved = json.load(fh).get("resolved_config")
    except (OSError, ValueError):
        return True
    return not (isinstance(resolved, dict) and resolved.get("localize_db"))


def _selfcheck():
    import tempfile
    root = tempfile.mkdtemp()

    # EVERY CASE IS A LOCALIZATION RUN unless localize_db says otherwise -- the mode is part of
    # the sample. localize_db=None writes a SLAM/mapping-only bundle (resolved_config.localize_db
    # null); "NOMETA" writes no run_metadata.json at all.
    def bundle(name, lines, localize_db="/ext/maps/hm3d_00861/rtabmap.db"):
        d = os.path.join(root, name, "logs")
        os.makedirs(d, exist_ok=True)
        if lines is not None:
            with open(os.path.join(d, "perception.log"), "w") as fh:
                fh.write(lines)
        if localize_db != "NOMETA":
            with open(os.path.join(root, name, "run_metadata.json"), "w") as fh:
                json.dump({"resolved_config": {"localize_db": localize_db}}, fh)
        return d

    assert worst_rejected_age(root) == "", "an empty runs dir must print nothing"
    assert worst_rejected_age(os.path.join(root, "nope")) == "", "a missing dir must not raise"

    bundle("20260101_000000_x", "Cached frame too old (6.60s), discarding\n"
                                "Cached frame too old (8.52s), discarding\n"
                                "Cached frame too old (5.57s), discarding\n")
    assert worst_rejected_age(root) == "8.52", worst_rejected_age(root)

    # AN EMPTY perception.log IS NOT A CLEAN RUN. GA-301: run 20260904_143558 was REFUSED by the
    # gate, so perception never started and its log was 0 bytes. "No rejection lines" then read as
    # "the newest run was fine" and the very next launch passed a10 -- so a refused run silenced the
    # probe, and the gate could be defeated by launching twice. An empty log is no evidence either
    # way, and the search must continue past it.
    bundle("20260404_000000_x", "")
    assert worst_rejected_age(root) == "8.52", worst_rejected_age(root)
    import shutil as _sh
    _sh.rmtree(os.path.join(root, "20260404_000000_x"))

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

    # ONLY LOCALIZATION RUNS ARE SAMPLES (owner ruling 4 Sep, "we will not use slam"; GA-290
    # register, SLAM ban). The exact case that ruled it: a SLAM bundle with a 24.30 s rejection
    # (run 082146) must be skipped in favour of an OLDER localization bundle -- a10 was refusing
    # the authorized modal launch on the SLAM run's mode artifact.
    bundle("20260606_000000_x", "Cached frame too old (24.30s), discarding\n", localize_db=None)
    assert worst_rejected_age(root) == "", worst_rejected_age(root)
    bundle("20260707_000000_x", "Cached frame too old (9.99s), discarding\n")
    assert worst_rejected_age(root) == "9.99", worst_rejected_age(root)
    bundle("20260808_000000_x", "Cached frame too old (24.30s), discarding\n", localize_db=None)
    assert worst_rejected_age(root) == "9.99", worst_rejected_age(root)
    # A bundle with NO run_metadata.json cannot certify its mode either way: skipped, not read.
    bundle("20260909_000000_x", "Cached frame too old (30.00s), discarding\n", localize_db="NOMETA")
    assert worst_rejected_age(root) == "9.99", worst_rejected_age(root)
    # So is one whose metadata is malformed.
    b = bundle("20261010_000000_x", "Cached frame too old (31.00s), discarding\n")
    with open(os.path.join(root, "20261010_000000_x", "run_metadata.json"), "w") as fh:
        fh.write("{not json")
    assert worst_rejected_age(root) == "9.99", worst_rejected_age(root)

    os.symlink(os.path.join(root, "20260707_000000_x"), os.path.join(root, "latest"))
    assert worst_rejected_age(root) == "9.99", "`latest` is a name, not a sample"
    print("  last_frame_age_rejected selfcheck OK (empty/missing dir, worst-of-many, newest "
          "clean localization run wins, dead-bridge, empty-log and non-localization bundles skipped, "
          "no-log skipped, malformed age and metadata ignored, `latest` ignored)")
    return 0


if __name__ == "__main__":
    if "--selfcheck" in sys.argv:
        raise SystemExit(_selfcheck())
    print(worst_rejected_age(sys.argv[1] if len(sys.argv) > 1 else ""))
