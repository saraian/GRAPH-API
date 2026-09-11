#!/usr/bin/env python3
"""The sampling policy is gone, and every way of asking for it REFUSES.

Owner, 2026-09-11: "Completely remove the old sampling policy from the repo." The policy was a
greedy nearest-unvisited coverage tour over navmesh samples (class Tour), a mapping phase that
selected it (mapping_seconds), and walk/dwell bursts with an adaptive hold. The schedule is the
only motion policy now.

WHY A TEST AND NOT A GREP. Removal has two failure modes and neither shows up in a diff:

  1. A knob survives as a name that is silently IGNORED. A config still asking for walk_frames
     would then run scheduled motion while its own config.yaml describes bursts, and the bundle
     would carry both stories. Every retired name must refuse.
  2. The schedule becomes optional again. With no fallback, a run without one publishes frames
     from a robot that never moves -- for the whole cap, producing a bundle that looks complete.
     The launcher and the feed host must each refuse before anything starts.

Run: python3 test_ga480_sampling_policy_removed.py   (or under pytest)
"""
import ast
import io
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
HOST = os.path.join(HERE, "habitat_feed_host.py")
RUNNER = os.path.join(HERE, "..", "..", "run.sh")
CONTAINER = os.path.join(HERE, "live_stack_container.sh")
DEFAULTS = os.path.join(HERE, "..", "src", "perception_module", "config.py")
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))

RETIRED_CFG = ("walk_frames", "dwell_frames", "dwell_mode", "dwell_min_frames", "dwell_max_frames",
               "dwell_dynamic", "tour_waypoints", "tour_scan_frames", "mapping_seconds")
RETIRED_ENV = ("FEED_WALK", "FEED_DWELL", "FEED_DWELL_MODE", "FEED_DWELL_MIN", "FEED_DWELL_MAX",
               "FEED_TEST_TOUR", "FEED_TEST_TOUR_SCAN", "FEED_MAPPING_SECONDS")


def _read(p):
    return io.open(p, encoding="utf-8").read()


def test_the_tour_class_is_gone():
    tree = ast.parse(_read(HOST))
    names = {n.name for n in ast.walk(tree) if isinstance(n, ast.ClassDef)}
    assert "Tour" not in names, "class Tour is back in habitat_feed_host.py"
    assert "ScheduledTour" in names, "ScheduledTour is missing; nothing would drive a run"


def test_the_adaptive_hold_module_is_gone():
    assert not os.path.exists(os.path.join(HERE, "adaptive_hold.py")), \
        "adaptive_hold.py is back; it exists only for the walk/dwell burst loop"
    assert "AdaptiveHold" not in _read(HOST), "habitat_feed_host.py imports AdaptiveHold again"


def test_every_retired_name_is_refused_by_the_feed_host():
    """Each name must appear in a refusal, not merely be absent."""
    src = _read(HOST)
    assert "_TOUR_RETIRED" in src, "the feed host has no list of retired config keys to refuse"
    for k in RETIRED_CFG:
        assert re.search(rf'"{k}"', src), \
            f"the feed host does not name {k!r}, so a config still setting it is silently ignored"
    for e in ("FEED_WALK", "FEED_DWELL", "FEED_TEST_TOUR"):
        assert e in src, f"the feed host does not refuse the environment variable {e}"
    assert "FEED_MAPPING_SECONDS" in src, "the feed host does not refuse FEED_MAPPING_SECONDS"


def test_no_config_in_the_repo_still_declares_a_retired_key():
    """Bundle copies under results/ are RECORDS of past runs and are deliberately excluded."""
    import yaml
    offenders = []
    for root, dirs, files in os.walk(REPO):
        dirs[:] = [d for d in dirs
                   if d not in (".git", "results", "runs", "maps", "ws", "node_modules", "__pycache__")]
        for f in files:
            if not f.endswith(".yaml"):
                continue
            p = os.path.join(root, f)
            try:
                hab = (yaml.safe_load(io.open(p, encoding="utf-8")) or {}).get("habitat") or {}
            except Exception:
                continue
            bad = [k for k in RETIRED_CFG + ("tour_all_floors",) if hab.get(k) is not None]
            if bad:
                offenders.append(f"{os.path.relpath(p, REPO)}: {', '.join(bad)}")
    assert not offenders, "configs still declare retired keys, and the feed host refuses them:\n  " \
        + "\n  ".join(offenders)


def test_the_defaults_no_longer_carry_the_policy():
    import importlib.util
    spec = importlib.util.spec_from_file_location("cfg_defaults_under_test", DEFAULTS)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    hab = mod.CFG["habitat"]
    left = [k for k in RETIRED_CFG + ("tour_all_floors",) if k in hab]
    assert not left, f"config.py still defaults {left}; a default is a setting nobody had to type"
    assert hab.get("exploration_laps"), "the defaults do not state exploration_laps"
    assert hab.get("navigation_mode"), "the defaults do not state navigation_mode"


def test_the_launcher_assigns_no_retired_name():
    """Comments may name them -- that is the explanation. Assignments may not."""
    for line in _read(RUNNER).split("\n"):
        if line.lstrip().startswith("#"):
            continue
        m = re.match(r"\s*(?:export\s+)?([A-Z_][A-Z0-9_]*)=", line)
        if m and m.group(1) in RETIRED_ENV:
            raise AssertionError(f"run.sh still assigns {m.group(1)}: {line.strip()}")


def test_the_launcher_refuses_a_run_with_no_schedule():
    src = _read(RUNNER)
    block = src[src.index("THE SCHEDULE IS MANDATORY"):src.index("# NUMERIC positions")]
    # EVERY refusal must EXIT, checked one by one. A count of `exit 1` in the block is not the same
    # claim: dropping the exit from one branch leaves the count high enough and the branch falls
    # through to a run with no motion, which is the defect this test exists for. (Measured: a
    # count-based version of this assertion passed with that exact mutation applied.)
    lines = block.split("\n")
    for i, line in enumerate(lines):
        if 'echo "!!' not in line:
            continue
        tail = "\n".join(lines[i:i + 8])
        assert "exit 1" in tail, \
            f"this refusal does not exit, so the run continues with no motion: {line.strip()}"
    assert "export FEED_SCHEDULE" in block, "the launcher never exports the schedule it built"
    for why in ("no navmesh", "does not exist", "set and empty"):
        assert why in block, f"the launcher does not refuse the case: {why}"


def test_the_schedule_is_decided_before_the_bundle_is_stamped():
    """A refusal must come before run_metadata.json, or the bundle records an empty schedule."""
    src = _read(RUNNER)
    assert src.index("THE SCHEDULE IS MANDATORY") < src.index('cat <<EOF > "$RUN_DIR/run_metadata.json"'), \
        "the schedule block sits below the metadata heredoc, so the bundle stamps an empty schedule"


def test_the_feed_host_refuses_a_run_with_no_schedule():
    fn = next(n for n in ast.parse(_read(HOST)).body
              if isinstance(n, ast.FunctionDef) and n.name == "main")
    src = ast.unparse(fn)
    assert "FEED_SCHEDULE is not set" in src, \
        "main() does not refuse a missing schedule; it would publish a stationary robot"
    assert "Tour(" not in src.replace("ScheduledTour(", ""), "main() still builds the sampling Tour"


def test_the_bundle_still_says_which_policy_drove_it():
    """Archived bundles say "sampled". A new one must say "schedule", or the two pool silently."""
    assert '"motion_policy": "schedule"' in _read(RUNNER), \
        "run_metadata.json no longer stamps motion_policy"
    assert '"motion_policy": "schedule"' in _read(HOST), \
        "feed_stats.json no longer stamps motion_policy"


def test_the_mapping_only_shape_is_refused_not_ignored():
    runner = _read(RUNNER)
    # The refusal, not the first mention: MAPPING_ONLY is also tested inside a helper further up.
    i = runner.index('echo "!! MAPPING_ONLY=1:')
    assert "exit 1" in runner[i:i + 600], "MAPPING_ONLY=1 does not exit; it would run an ordinary run"
    assert "MAPPING_ONLY:-0" not in _read(CONTAINER).replace(
        "# MAPPING_ONLY", "#"), "the container still branches on MAPPING_ONLY"


if __name__ == "__main__":
    fails = 0
    for name, fn in sorted(list(globals().items())):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASS {name}")
            except AssertionError as exc:
                fails += 1
                print(f"FAIL {name}: {exc}")
    raise SystemExit(1 if fails else 0)
