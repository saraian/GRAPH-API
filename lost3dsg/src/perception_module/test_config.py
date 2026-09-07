"""Which config file the module actually loaded.

GA-52: `_load` computed the path as a local and discarded it. There was no
module-level name for it, the feed host never logged it, and so no artefact
recorded which yaml a run had loaded or what value was in force. That is what made
the floor ablation undiagnosable after the fact: both arms would run identically and
the bundle would record neither.

A path that merely EXISTS does not prove it was the one loaded. So `CFG_PATH` is the
file that was read, and it is `None` when no file was found and the defaults are in
force — the two cases a reader must never confuse.

Run: python3 test_config.py      (from this directory; no pytest needed)
"""
import importlib
import os
import sys
import tempfile
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def _reload_with(path):
    """Re-import config with GRAPH_API_CONFIG pointed at `path` (None = unset)."""
    previous = os.environ.get("GRAPH_API_CONFIG")
    if path is None:
        os.environ.pop("GRAPH_API_CONFIG", None)
    else:
        os.environ["GRAPH_API_CONFIG"] = path
    try:
        import config
        return importlib.reload(config)
    finally:
        if previous is None:
            os.environ.pop("GRAPH_API_CONFIG", None)
        else:
            os.environ["GRAPH_API_CONFIG"] = previous


def test_the_loaded_path_is_recorded():
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
        f.write("perception:\n  score_threshold: 0.99\n")
        path = f.name
    try:
        cfg = _reload_with(path)
        assert cfg.CFG_PATH == path, cfg.CFG_PATH
        assert cfg.CFG["perception"]["score_threshold"] == 0.99, cfg.CFG["perception"]
    finally:
        os.unlink(path)


def test_a_missing_file_records_no_path_rather_than_the_one_it_looked_for():
    """The distinction the finding turns on. A path written down when nothing was
    read is worse than no path: the next reader takes it for the config in force."""
    cfg = _reload_with("/nonexistent/graph_api_config_that_is_not_there.yaml")
    assert cfg.CFG_PATH is None, cfg.CFG_PATH
    assert cfg.CFG["perception"]["score_threshold"] == \
        cfg._DEFAULTS["perception"]["score_threshold"]


def test_the_defaults_are_not_mutated_by_a_load():
    """`_merge` must not write into `_DEFAULTS`, or the next reload inherits the
    previous run's overrides and `CFG_PATH is None` stops meaning "defaults"."""
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
        f.write("perception:\n  score_threshold: 0.01\n")
        path = f.name
    try:
        cfg = _reload_with(path)
        assert cfg._DEFAULTS["perception"]["score_threshold"] == 0.15, cfg._DEFAULTS
    finally:
        os.unlink(path)


def test_there_is_no_fallback_label_seam_left_to_arm():
    """GA-53: `vlm.fallback_labels` armed a static open-vocabulary list that replaced
    the VLM on the live detection path when it was unreachable — the third shape in
    working rule 14, a run reporting success on behalf of something that never ran.
    WORKING_RULES.md lists a fallback label list among the substitutions already
    FOUND AND REMOVED; that removal was in the other checkout only, and this is the
    tree that runs.

    Both halves are asserted: the key is gone from the defaults, and no reader of it
    survives in the detection path. The source check names a file whose existence is
    asserted first — TS-01 is a hygiene gate that greps a path that does not exist and
    therefore cannot fail.
    """
    cfg = _reload_with("/nonexistent/graph_api_config_that_is_not_there.yaml")
    assert "fallback_labels" not in cfg._DEFAULTS["vlm"], cfg._DEFAULTS["vlm"]

    here = os.path.dirname(os.path.abspath(__file__))
    pipeline = os.path.join(here, "detection_pipeline.py")
    assert os.path.exists(pipeline), pipeline
    with open(pipeline) as f:
        source = f.read()
    # Count the CODE, not the occurrences. The finding itself was first written as
    # "the reader appears twice", which counted one comment and one read and would
    # have sent the next person hunting for a second seam. The comment recording why
    # the seam was removed must not make this test fail.
    code = "\n".join(line for line in source.splitlines()
                     if not line.lstrip().startswith("#"))
    assert "fallback_labels" not in code, "the seam is back in detection_pipeline.py"


def test_the_vlm_failure_path_still_records_what_failed():
    """Removing the substitution must not remove the telemetry: the handler records
    the failure and re-raises unconditionally. A handler that re-raises is not a mute;
    one that returns a substitute is."""
    here = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(here, "detection_pipeline.py")) as f:
        source = f.read()
    body = source.split("def _extract_detection_labels", 1)[1].split("\n    def ", 1)[0]
    assert "unreachable" in body
    assert "raise" in body


def test_the_config_declares_the_key_the_code_reads():
    """GA-49: `max_match_distance_m` was documented as the tracking match gate, with
    `0 = off`. No code reads it any more — GA-04 replaced it with
    `reevaluation_radius_m` — so the yaml was a false statement about the system, which
    is the same defect as a comment that outlives its code. The replacement was read
    through a `.get` default and declared nowhere, so the value in force could not be
    seen in any config file.

    The default here must equal the `.get` fallback in `object_manager_6.py`, or
    declaring the key changes behaviour instead of documenting it.
    """
    cfg = _reload_with("/nonexistent/graph_api_config_that_is_not_there.yaml")
    assoc = cfg.CFG["association"]
    assert "max_match_distance_m" not in assoc, assoc
    assert assoc["reevaluation_radius_m"] == 2.0, assoc


def test_no_code_declares_the_removed_gate_anywhere_in_the_package():
    """GA-49's guard, in the form that survives the next config file.

    It globs every `*.yaml` and `*.py` under the package rather than naming two files,
    so a config added later cannot be added outside its reach — a hardcoded tuple
    closes today's instance and leaves the identical hole open.

    Two deliberate details:

    * It counts CODE, not occurrences: comment lines are dropped and trailing comments
      are cut. The comments recording why the key was removed are not readers of it,
      and a mention-counting sweep under-reports dead keys — the direction that hides
      defects rather than inventing them.
    * The forbidden literal is split across a concatenation, and this file is excluded
      by name, because a guard that must contain the string it forbids would otherwise
      be self-exempting twice over. The path exclusion is the only hole.

    Demonstrated, not assumed. Run against the real pre-removal files on 2026-08-30,
    before the simulator lane cleaned them, it FIRED on `regolo_config.yaml` and
    `smoke_config.yaml`; against the cleaned copies it passed. That evidence could only
    be taken while the dirty state existed: a guard that has only ever seen a clean tree
    has not fired.
    """
    dead_key = "max_match" + "_distance_m"
    package = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    assert os.path.isdir(package), package

    offenders = []
    for root, _, files in os.walk(package):
        if "__pycache__" in root:
            continue
        for name in files:
            if not name.endswith((".yaml", ".py")) or name == os.path.basename(__file__):
                continue
            path = os.path.join(root, name)
            with open(path, errors="replace") as f:
                body = f.read()
            code = "\n".join(line.split("#", 1)[0] for line in body.splitlines()
                             if not line.lstrip().startswith("#"))
            if dead_key in code:
                offenders.append(os.path.relpath(path, package))
    assert not offenders, f"{dead_key} is still declared in: {offenders}"



def test_no_dead_perception_keys_are_declared():
    """GA-19: `reachability_strict` and `detect_while_moving` were declared in the
    defaults and read by nothing; a declared key that no code reads is a false statement
    about the system. Absent from the defaults, and the quoted key is read nowhere."""
    cfg = _reload_with("/nonexistent/graph_api_config_that_is_not_there.yaml")
    dead = ("reachability" + "_strict", "detect_while" + "_moving")
    for key in dead:
        assert key not in cfg.CFG["perception"], key
    here = os.path.dirname(os.path.abspath(__file__))
    offenders = []
    for root, _, files in os.walk(here):
        for name in files:
            if not name.endswith(".py") or "__pycache__" in root or name == os.path.basename(__file__):
                continue
            with open(os.path.join(root, name), errors="replace") as f:
                code = "\n".join(line.split("#", 1)[0] for line in f.read().splitlines())
            if any(f'"{k}"' in code or f"'{k}'" in code for k in dead):
                offenders.append(os.path.relpath(os.path.join(root, name), here))
    assert not offenders, offenders


if __name__ == "__main__":
    tests = [(n, f) for n, f in sorted(globals().items())
             if n.startswith("test_") and callable(f)]
    failures = 0
    for name, fn in tests:
        try:
            fn()
            print(f"  ok    {name}")
        except Exception as exc:
            failures += 1
            print(f"  FAIL  {name}: {exc}")
    # The count is printed because this runner reads globals() at the moment it runs:
    # a test appended BELOW this block is defined too late and is silently skipped.
    # That happened once. A runner that cannot report what it did not run is the same
    # defect as a gate that cannot fail.
    # A result carries the time it was taken. Two correct measurements of one path
    # disagree when someone fixes it in between, and neither reader is wrong — a
    # finding is true at a time, not simply true. Frozen-root digests carry a time;
    # test results did not.
    print(f"config tests: {len(tests)} run,",
          "all passed" if not failures else f"{failures} failed",
          "|", datetime.now().astimezone().isoformat(timespec="seconds"))
    sys.exit(1 if failures else 0)
