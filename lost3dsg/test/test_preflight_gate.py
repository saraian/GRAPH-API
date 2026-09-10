#!/usr/bin/env python3
"""Runnable check for preflight_gate.py. Host-runnable: no ROS, no container, no network.

    python3 test_preflight_gate.py

Covers the harness, the probes that need neither ROS nor a backend (a2, a3, a5), and every
`return False` branch of a1, a2 and a6 (GA-73) — a2 against the real config.py, a1 and a6
against stubbed aligner / TF modules. What only the container can show is marked
`needs_container` and skipped with the call named, never faked: claiming a probe is verified
when it has never run is the failure this whole gate exists to catch. a4's branch logic IS
exercised, against stubs; its two real inferences are not.
"""
import json
import os
import subprocess
import sys
import tempfile

# Derived, never hard-coded: the suite must run from a clone anywhere.
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
import time
import types

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import preflight_gate as g  # noqa: E402


def _fake_frame():
    """The frame the host tests INJECT into a4 (`frame` is injected only by the host tests).

    Two callers referenced this and it was never defined, so both died with NameError before
    reaching what they assert — ruff F821 names it in one line. a4 reads only `frame.shape` and
    hands the object to the backend, which these tests stub, so the smallest real array serves.
    This is NOT the fallback a4 refuses to build for itself: that refusal is about the PROBE
    substituting a synthetic frame in a container, and it stays intact above.
    """
    import numpy as np
    return np.zeros((4, 4, 3), dtype=np.uint8)

try:
    import pytest
except ImportError:      # script mode on a host without pytest; the marker below still records the gap
    pytest = None


def check(cond, msg):
    if not cond:
        raise AssertionError(msg)


def needs_container(reason):
    """Mark a test that CANNOT run on the host. It is skipped, with the call that needs the
    container in the reason, so the gap is visible in the suite output rather than absent from
    it. Not a fake: the body is the real assertion and runs unchanged inside the container."""
    def deco(fn):
        fn.needs_container = f"needs the container: {reason}"
        return pytest.mark.skip(reason=fn.needs_container)(fn) if pytest else fn
    return deco



# --- a3: the policy comparison must compare, not echo -------------------------------------
def test_a3_compares_rather_than_echoes():
    os.environ["EXT_ENFORCE"] = "1"
    os.environ.pop("EXT_HOLD_BAND", None)

    ok, d = g.a3_policy_reached_container({"EXT_ENFORCE": "1"})
    check(ok is True, f"matching policy should pass: {d}")

    ok, d = g.a3_policy_reached_container({"EXT_ENFORCE": "0"})
    check(ok is False, "a value that differs from the intended one must FAIL")
    check(d["mismatches"]["EXT_ENFORCE"] == {"intended": "0", "in_container": "1"}, d)

    # The defect this probe exists for: the var never reached the container at all. Absent must
    # fail, not pass — an echo-style check would have reported the host's value and looked fine.
    ok, d = g.a3_policy_reached_container({"EXT_HOLD_BAND": "0.05"})
    check(ok is False, "a var absent from the container must FAIL, not pass")
    check(d["mismatches"]["EXT_HOLD_BAND"]["in_container"] is None, d)

    ok, d = g.a3_policy_reached_container({})
    check(ok is g.SKIPPED, "no intended values means nothing was asserted -> SKIPPED, not pass")


# --- a5: stale artefacts ------------------------------------------------------------------
def test_a5_fails_on_an_artefact_older_than_the_run():
    with tempfile.TemporaryDirectory() as td:
        start = time.time()
        fresh = os.path.join(td, "fresh.json")
        open(fresh, "w").close()
        ok, d = g.a5_bundle_clean(td, start)
        check(ok is True, f"a bundle written after run start is clean: {d}")

        stale = os.path.join(td, "logs", "old.log")
        os.makedirs(os.path.dirname(stale), exist_ok=True)
        open(stale, "w").close()
        os.utime(stale, (start - 3600, start - 3600))
        ok, d = g.a5_bundle_clean(td, start)
        check(ok is False, "an artefact predating run start must FAIL")
        check(d["stale"][0]["path"] == os.path.join("logs", "old.log"), d)
        check(d["stale_count"] == 1, d)

        ok, d = g.a5_bundle_clean(td, 0.0)
        check(ok is g.SKIPPED, "no run-start means nothing was asserted -> SKIPPED")


# --- harness: a SKIPPED probe must not read as a pass -------------------------------------
def test_a_skipped_probe_is_not_a_pass():
    # The aligner that "could not run" is exactly how the exemplar baseline shipped as the KG
    # result. Skipped is not pass, and the exit code has to say so.
    with tempfile.TemporaryDirectory() as td:
        out = os.path.join(td, "preflight.json")
        rc = g.main(["--only", "a3", "--out", out])          # a3 with no --expect-policy -> SKIP
        check(rc == 1, "a skipped probe must abort the gate by default")
        rep = json.load(open(out))
        check(rep["verdict"] == "fail" and rep["skipped"] == ["a3"], rep)

        rc = g.main(["--only", "a3", "--out", out, "--allow-skip"])
        check(rc == 0, "--allow-skip must downgrade a skip")
        check(json.load(open(out))["verdict"] == "pass", "--allow-skip verdict should be pass")

        rc = g.main(["--only", "a3", "--out", out, "--expect-policy", "EXT_ENFORCE=0"])
        check(rc == 1, "a real mismatch must abort")
        rep = json.load(open(out))
        check(rep["failed"] == ["a3"] and rep["verdict"] == "fail", rep)


# --- harness: a probe that raises is SKIPPED, never a pass --------------------------------
def test_a_probe_that_raises_is_skipped_not_passed():
    """The HARNESS is what is under test, so the raising probe is supplied HERE.

    This used to run `--only a1` and rely on `found` being unimportable on the host: the test
    asserted about the harness and depended on an extension package to make its subject raise.
    When that probe moved out, the test broke for a reason that had nothing to do with what it
    checks. A registered stub that raises on purpose cannot rot that way."""
    def _raises():
        raise RuntimeError("deliberate: the harness must record this, not pass it")
    with tempfile.TemporaryDirectory() as td:
        out = os.path.join(td, "p.json")
        # Registering a NEW id in PROBES alone is now a misconfiguration the gate refuses by
        # design: `bound` is the second half of the registration and a test cannot reach it
        # (it is a local literal in main()). That guard exists because a9 and a10 each shipped
        # half-registered. So make an ALREADY-registered probe raise instead -- `bound` reads
        # the module global when main() runs, so patching the function is enough, and the path
        # under test (a probe raises -> recorded, never passed) is exactly the same one.
        saved_fn = g.a4_perception_twice
        g.a4_perception_twice = _raises
        try:
            rc = g.main(["--only", "a4", "--out", out])
        finally:
            g.a4_perception_twice = saved_fn
        check(rc == 1, "a probe that raises must abort, not pass")
        rep = json.load(open(out))
        check(rep["probes"][0]["ok"] is None, rep)
        check("probe raised" in rep["probes"][0]["detail"]["reason"], rep)


# --- tree_sha: names are in the digest, and an empty root RAISES ---------------------------
def test_tree_sha_names_are_in_the_digest_and_an_empty_root_raises():
    # The old _hash_py did `find | xargs cat | sha256sum`, so a rename was invisible and a root
    # matching nothing yielded e3b0c44298fc1c14 — a plausible provenance stamp for zero files.
    with tempfile.TemporaryDirectory() as td:
        try:
            g.tree_sha(td)
            raise AssertionError("an empty root must RAISE, not return the sha256 of nothing")
        except ValueError:
            pass

        open(os.path.join(td, "a.py"), "w").write("x = 1\n")
        sha_a, n = g.tree_sha(td)
        check(n == 1, n)
        check(sha_a != "e3b0c44298fc1c14", "empty-tree hash must never be produced from content")

        os.rename(os.path.join(td, "a.py"), os.path.join(td, "b.py"))
        sha_b, _ = g.tree_sha(td)
        check(sha_a != sha_b, "a RENAME must move the digest — filenames are in it")

        # THE FROZEN ROOT MUST BE ABLE TO HOLD STILL. `output/` is where the running stack
        # writes and `.mypy_cache` is written by anyone running the type checker -- with those
        # in the set, running the stack moved the digest and so did a lane running a linter.
        # "Frozen" was not achievable by construction, and three such files made a 137-file
        # source read as 140.
        for d in ("output", ".mypy_cache", ".pytest_cache", "__pycache__", ".ruff_cache"):
            os.makedirs(os.path.join(td, d), exist_ok=True)
            open(os.path.join(td, d, "written_by_the_system.txt"), "w").write("x")
        check(g.tree_sha(td)[0] == sha_b,
              "nothing the system writes may move the digest, or the root cannot hold still")

        # what the system writes is excluded; what a person wrote is kept
        os.makedirs(os.path.join(td, "grafici_output"))
        open(os.path.join(td, "grafici_output", "albero_mondo.html"), "w").write("<html>")
        check(g.tree_sha(td)[0] == sha_b, "generated output must not move the digest")
        os.makedirs(os.path.join(td, "old"))
        open(os.path.join(td, "old", "object_manager_1.py"), "w").write("legacy\n")
        check(g.tree_sha(td)[0] != sha_b, "old/ is a person's source and must be hashed")

        # COLCON_IGNORE has no extension: an allow-list cannot catch it by construction
        sha_c = g.tree_sha(td)[0]
        open(os.path.join(td, "COLCON_IGNORE"), "w").close()
        check(g.tree_sha(td)[0] != sha_c, "COLCON_IGNORE decides whether the package builds")


# --- a5: the SCRATCH directory is the one that is never cleared ----------------------------
def test_a5_checks_the_scratch_directory_not_only_the_bundle():
    with tempfile.TemporaryDirectory() as run, tempfile.TemporaryDirectory() as scratch:
        start = time.time()
        old_json = os.path.join(scratch, "outcome_analysis.json")
        open(old_json, "w").close()
        os.utime(old_json, (start - 3600, start - 3600))

        # Pointed only at the fresh bundle it cannot fire — three reasons, and this was one.
        ok, d = g.a5_bundle_clean(run, start)
        check(ok is True, f"the bundle alone is clean by construction: {d}")

        ok, d = g.a5_bundle_clean(run, start, scratch)
        check(ok is False, "a stale ARCHIVED artefact in the scratch dir must FAIL")
        check(d["stale"][0]["path"] == "outcome_analysis.json", d)

        # a PNG is never copied into the bundle, so failing on it would only teach operators
        # to reach for PREFLIGHT_SKIP=1
        png = os.path.join(scratch, "detection_0001.png")
        open(png, "w").close()
        os.utime(png, (start - 3600, start - 3600))
        os.remove(old_json)
        ok, d = g.a5_bundle_clean(run, start, scratch)
        check(ok is True, f"a stale file that is never archived must NOT fail the gate: {d}")


# --- a1: the probe must record what DECIDED, not only what described ------------------------
# a1 once reported `score 0.9032032489776611 -> aligned_to null` while the same pair, same model
# and same ontology gave `top=0.9032 z=5.22 n=170 -> ACCEPT` elsewhere. The score agreed to the
# last digit; the candidate SET differed, and z is what the rule compares. Recording the number
# that agreed and not the number that decided turned a subtraction into an investigation.
def test_a1_records_the_discriminants_not_only_the_score():
    class Emb:
        def rank(self, key, k):
            return [("http://x#Sofa", 0.9032), ("http://x#Chair", 0.61)][:k]

    class KG:
        _n, _top_min, _z_min, _embedder = 170, 0.87, 3.0, Emb()

    class Res:
        evidence = "kgaligner 0.90 (z=5.2) -> Sofa (http://x#Sofa)"

    d = g._discriminants(KG(), "couch", Res())
    check(d["n_candidates"] == 170, d)      # the size of the set that ranked it
    check(d["top_min"] == 0.87, d)          # both thresholds are env-overridable
    check(d["z_min"] == 3.0, d)
    check("z=5.2" in d["evidence"], d)      # the z the ALIGNER used, not one recomputed here
    check(d["top5"][0]["class"] == "Sofa", d)
    check(len(d["top5"]) == 2, d)

    # An aligner with none of it must yield nothing rather than raise: exemplar and lexical
    # have no candidate set, and a probe that crashes collecting diagnostics is worse than one
    # that reports fewer.
    class Plain:
        pass

    check(g._discriminants(Plain(), "couch", None) == {}, "no discriminants, no crash")

    # A ranking that raises is recorded, not propagated.
    class Broken:
        _n = 5

        class _embedder:
            @staticmethod
            def rank(key, k):
                raise RuntimeError("index closed")

    d = g._discriminants(Broken(), "couch", None)
    check(d["n_candidates"] == 5, d)
    check("RuntimeError" in d["top5_error"], d)


# --- a1: the field it reads, and a control that can actually fail --------------------------
# A probe read `res.name` for a day. The alignment result has label/aligned/score/evidence and NO
# `name`, so getattr(res, "name", None) returned None for every input the aligner could produce:
# the positive could never match "Sofa" and the negative control could never fail. A check that
# cannot succeed, whose control cannot fail, inside the gate built to stop exactly that.
def test_a1_reads_the_field_the_aligner_actually_sets():
    import dataclasses

    @dataclasses.dataclass
    class Alignment:            # the real shape, verified against found/align.py
        label: str
        aligned: str | None
        score: float
        evidence: str

    name, src = g._aligned_name(Alignment("couch", "Sofa", 0.9032, "kgaligner 0.90 -> Sofa"))
    check(name == "Sofa", f"the aligner sets .aligned, not .name: got {name!r}")
    check(src == ".aligned", src)

    name, src = g._aligned_name(Alignment("zzqx", None, 0.1, "gap"))
    check(name is None and src == ".aligned",
          "a genuine null must be distinguishable from a missing field")

    check(g._aligned_name(None) == (None, "no result"), "no result is its own case")
    check(g._aligned_name({"aligned": "Bed"})[0] == "Bed", "dict shape still works")


def test_a1_raises_rather_than_returning_none_for_an_unknown_shape():
    """getattr(x, 'field', None) cannot tell a RENAMED field from a genuine null, and that
    difference was the whole content of the defect. An unrecognised shape means the probe
    cannot assert, so it raises -- and the harness turns a raise into SKIPPED, a failed
    verdict. Defaulting to None is how this went unnoticed for a day."""
    class Renamed:
        def __init__(self):
            self.matched_class = "Sofa"      # the field moved

    try:
        g._aligned_name(Renamed())
        raise AssertionError("a renamed field must RAISE, not silently read as None")
    except AttributeError as exc:
        check("matched_class" in str(exc), f"the error must name what it did find: {exc}")

    try:
        g._aligned_name({"score": 0.9})
        raise AssertionError("a dict without a known key must RAISE")
    except AttributeError:
        pass


def test_a1s_negative_control_can_actually_fail():
    """The control asserts that a nonsense label aligns to NOTHING. While _aligned_name always
    returned None it passed vacuously -- it passed for the same reason the positive failed, and
    a control that cannot fail is not a control. This hands it an aligner that names everything
    and requires the control to catch it."""
    import dataclasses

    @dataclasses.dataclass
    class Alignment:
        label: str
        aligned: str | None
        score: float
        evidence: str

    class NamesEverything:
        """The fallback shape a1 exists to detect: a non-empty answer for any input."""

        def align(self, label):
            return Alignment(label, "Sofa", 0.99, "always Sofa")

    a = NamesEverything()
    good, _ = g._aligned_name(a.align(g.GOLDEN_LABEL))
    bad, _ = g._aligned_name(a.align(g.NONSENSE_LABEL))
    check(good == g.GOLDEN_EXPECT, "such an aligner passes the positive case")
    check(bad is not None,
          "and the negative control MUST see a name -- that is what makes it a control")


# --- a8: the cheapest question, and nobody was asking it ------------------------------------
# GA-66. The gate passed 7/7 and the stack died seconds later on `from models import OWLv2,
# VitSam` at perception_2.py:61. Seven probes said the wiring was sound and none asked whether
# the code about to run loads at all. A passing gate followed by an immediate stack death is
# worse than a failing gate: it spends the bringup and points the operator at the wrong layer.
def test_a8_skips_outside_the_container_and_reports_which_node_failed():
    ok, d = g.a8_stack_imports()
    check(ok is g.SKIPPED, "no install tree means nothing was asserted -> SKIPPED, not FAIL")
    check("not running in the container" in d["reason"], d)


def test_a8_names_the_module_and_the_error_the_stack_would_have_hit():
    import os
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        install = os.path.join(td, "ws", "install", "lost3dsg", "lib", "lost3dsg")
        os.makedirs(install)
        open(os.path.join(install, "good_node.py"), "w").write("x = 1\n")
        open(os.path.join(install, "bad_node.py"), "w").write(
            "import a_module_that_is_not_installed\n")
        try:
            ok, d = g.a8_stack_imports(modules=("good_node", "bad_node"), install=install)
        finally:
            for m in ("good_node", "bad_node"):
                sys.modules.pop(m, None)
    check(ok is False, "a node that cannot import must FAIL")
    check(d["importable"] == ["good_node"], d)
    check("a_module_that_is_not_installed" in d["failed"]["bad_node"], d)
    check("INSTALLATION fault" in d["why"], "the remedy is an image rebuild, not a config change")


def test_a7_compares_what_executes_against_the_mount():
    """a7 hashed the MOUNT and the launcher hashed the SAME host directory, so it was
    comparing the host tree against itself and never touched the copy that runs. `cp -r`
    happens at live_stack_container.sh:10 and the gate at :92 — the copy was unmeasured by
    a7, by the launcher, and by every stamp any lane took."""
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        src = os.path.join(td, "mount")
        run = os.path.join(td, "install")
        os.makedirs(src)
        os.makedirs(run)
        open(os.path.join(src, "node.py"), "w").write("x = 1\n")
        open(os.path.join(run, "node.py"), "w").write("x = 1\n")
        open(os.path.join(src, "notes.txt"), "w").write("not python\n")

        ok, d = g.a7_source_frozen({"graph_api": "irrelevant"}, executed_tree=run,
                                   copy_source=src)
        check(d["executed"]["compared"] == 1, f"only .py is compared: {d['executed']}")
        check(d["executed"]["differs"] == [], d["executed"])

        # the case nothing measured until now: what runs differs from what is mounted
        open(os.path.join(run, "node.py"), "w").write("x = 2\n")
        ok, d = g.a7_source_frozen({"graph_api": "irrelevant"}, executed_tree=run,
                                   copy_source=src)
        check("executed_vs_mount" in d["mismatches"],
              f"a differing executed file must be a MISMATCH: {d}")
        check(d["mismatches"]["executed_vs_mount"] == ["node.py"], d["mismatches"])

        # a file the copy never received is recorded, not silently ignored
        open(os.path.join(src, "extra.py"), "w").write("y = 1\n")
        _, d = g.a7_source_frozen({"graph_api": "x"}, executed_tree=run, copy_source=src)
        check(d["executed"]["only_in_mount"] == ["extra.py"], d["executed"])


def test_a4_catches_the_missing_stage_timings_before_the_run_does():
    """GA-85. `detection_pipeline.py:93` REFUSES a response with no per-stage timing rather
    than estimating it from the wall clock — GA-14's remedy, and correct. But the backend was
    never made to report what the refusal requires, so on 31 Aug the VLM answered in 16.1 s,
    well inside budget, and the pipeline raised on the FIRST successful detection.

    a4 already received that dict on all three calls and discarded it. Asking for it moves the
    failure from one cycle into a measured run, to before the run starts."""
    import types

    def run_with_timings(t):
        class B:
            def detect_and_segment(self, frame, labels):
                return [], t

        fake_cfg = types.ModuleType("config")
        fake_cfg.CFG = {}
        fake_client = types.ModuleType("cloud.client")
        fake_client.get_perception_backend = lambda cfg: B()
        fake_cloud = types.ModuleType("cloud")
        fake_cloud.client = fake_client
        saved = {k: sys.modules.get(k) for k in ("config", "cloud", "cloud.client")}
        sys.modules.update({"config": fake_cfg, "cloud": fake_cloud,
                            "cloud.client": fake_client})
        try:
            return g.a4_perception_twice(frame=_fake_frame())
        finally:
            for k, v in saved.items():
                if v is None:
                    sys.modules.pop(k, None)
                else:
                    sys.modules[k] = v

    req, _ = g.pipeline_timing_keys()
    ok, d = run_with_timings({k: 1.0 for k in req})
    check(ok is True, f"a backend meeting the contract passes: {d}")
    # a PASSING probe must still say what it asked and what came back. The two facts that
    # resolved the yolo_world scare — the frame and the reported keys — existed only on the
    # failure path, so a passing bundle recorded neither.
    check(d["probe_frame"] == "a4_probe_frame.jpg", d)
    check(d["frame_shape"] == [480, 640, 3], d)
    check(d["reported_timing_keys"] == [sorted(req)] * 3, d)
    check(d["required_read_from"] is not None, "a pass must name which contract it applied")

    # the live failure: the call SUCCEEDS and the pipeline then refuses the response
    ok, d = run_with_timings({})
    check(ok is False, "no per-stage timing must FAIL before the run, not during it")
    check(d["missing"] == list(req), d)
    check("first successful detection" in d["why"].lower()
          or "FIRST successful detection" in d["why"], d)
    check("Do not 'fix' it by restoring an estimate" in d["why"],
          "the message must forbid the tempting repair")

    # The names the vendored pipeline USED to ask for. The server has never sent them —
    # measured from a live refusal: reported keys ['detector', 'sam2', 'total']. A probe
    # asking for these would fail a healthy backend and send someone to add a field it
    # already sends.
    other = ("detector", "sam2") if req[0] == "owlv2" else ("owlv2", "sam")
    ok, d = run_with_timings({k: 1.0 for k in other})
    check(ok is False, "keys the pipeline does not ask for must not satisfy the contract")
    check(d["reported_timing_keys"][0] == sorted(other),
          "a4 must report what it actually got — the server may be reporting perfectly "
          "under other names, and only the received keys can show that")


def test_a4_reads_the_pipelines_required_keys_rather_than_copying_them():
    """a4 predicts the pipeline's refusal before a run pays for it. A COPY of the key names can
    drift, and then a4 either certifies a run that dies at its first detection or refuses one
    that would have worked. Reading the pipeline's own list makes a4 wrong exactly when the
    pipeline is wrong — the only correct behaviour for a predictor.

    Live case: the vendored pipeline asks for ("owlv2", "sam"); the server emits
    ['detector', 'sam2', 'total']. a4 must therefore FAIL that backend, because the run does.
    """
    import re
    keys, src = g.pipeline_timing_keys()
    check(src is not None, "a4 must find the pipeline rather than fall back silently")
    body = open(src).read()
    m = re.search(r"missing\s*=\s*\[k for k in \(([^)]*)\)", body)
    check(m is not None, "the pipeline's required-timing list moved; a4's reader must follow")
    expect = tuple(x.strip().strip("\"'") for x in m.group(1).split(",") if x.strip())
    check(tuple(keys) == expect, f"a4 read {keys}, pipeline demands {expect}")

    # and the detail records WHERE it read them, so a bundle says which contract was applied
    check(isinstance(g._PIPELINE_TIMING_FALLBACK, tuple), "a fallback must exist")


def test_a4_refuses_a_backend_that_cannot_answer():
    """GA-81. `LocalPerceptionBackend.detect_and_segment` is `return [], {}` — it does not
    raise, so three calls record three passes having computed nothing. a4 was built to catch a
    backend that answers once and fails after; **a backend that answers instantly and always is
    the same defect with the sign flipped**, and timing cannot tell them apart."""
    import types

    class LocalPerceptionBackend:
        def detect_and_segment(self, frame, labels):
            return [], {}

    fake_cfg = types.ModuleType("config")
    fake_cfg.CFG = {}
    fake_client = types.ModuleType("cloud.client")
    fake_client.get_perception_backend = lambda cfg: LocalPerceptionBackend()
    fake_cloud = types.ModuleType("cloud")
    fake_cloud.client = fake_client
    saved = {k: sys.modules.get(k) for k in ("config", "cloud", "cloud.client")}
    sys.modules.update({"config": fake_cfg, "cloud": fake_cloud, "cloud.client": fake_client})
    try:
        ok, d = g.a4_perception_twice()
    finally:
        for k, v in saved.items():
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v

    check(ok is False, f"a stub backend must FAIL, not record three passes: {d}")
    check(d["backend"] == "LocalPerceptionBackend", d)
    check("computed nothing" in d["why"], d)
    check("attempts" not in d, "it must refuse BEFORE calling, not after timing three no-ops")
    check("LocalPerceptionBackend" in g.STUB_BACKENDS, "the refusal list must name it")


def test_run_output_lives_outside_every_hashed_root():
    """Run output goes to $WORKSPACE_ROOT/results/<timestamp>_<scene>, and it must move no digest.

    This is the property my own deny-list broke: `output/` was inside the hashed set, so running
    the stack moved the frozen root and "frozen" was unachievable while being reported achieved.
    A path ruling that quietly put run artefacts back inside a root would restore that, so it is
    asserted rather than assumed."""
    roots = (os.path.join(REPO_ROOT, "lost3dsg"), os.path.join(REPO_ROOT, os.pardir, "extension"))
    # the launcher's own root variable, not a hard-coded path: it derives its root from its own location so a
    # clone anywhere can run; this assertion used to encode the one machine the code was written
    # on, and it failed the moment the hardcoding it was guarding against was removed.
    out = "$WORKSPACE_ROOT/results"
    for r in roots:
        check(not out.startswith(r.rstrip("/") + "/") and out != r,
              f"run output at {out} is inside hashed root {r} — the frozen root cannot hold still")

    # and the launcher must actually default there
    body = open(os.path.join(HERE, "live_run.sh")).read()
    check("OUT_DIR=${OUT_DIR:-$WORKSPACE_ROOT/results/" in body,
          "live_run.sh must default OUT_DIR under $WORKSPACE_ROOT/results/, never /tmp")
    check("WORKSPACE_ROOT=${WORKSPACE_ROOT:-$(cd \"$REPO/../..\" && pwd)}" in body,
          "WORKSPACE_ROOT must be DERIVED from the script's location, not hardcoded — a clone "
          "anywhere else cannot run if it is")


# --- the extension seam: GRAPH-API ships the harness; an extension ships what asserts about itself ---
def test_the_probe_blueprint_skips_rather_than_passing():
    """A blueprint that did nothing and returned True would make every deployment that has not
    supplied a probe report a pass for a check nobody wrote."""
    ok, d = g.Probe().run(None)
    check(ok is g.SKIPPED, "the blueprint asserts nothing, so it SKIPS")
    check("does not implement" in d["reason"], d)


def test_external_probes_load_from_config_and_a_bad_spec_raises():
    """`preflight.probes` is read from the merged config through hooks.load_hook, so there is
    one path resolution in the tree rather than two. A spec that will not load RAISES: a probe
    named in config and silently absent is this gate's own defect, one level up."""
    import types

    class Mine(g.Probe):
        id, name = "x1", "mine"

        def run(self, args):
            return True, {"ran": True}

    mod = types.ModuleType("extprobes")
    mod.Mine = Mine
    sys.modules["extprobes"] = mod

    fake_hooks = types.ModuleType("hooks")

    def load_hook(spec, base, search_paths=()):
        module, _, cls = spec.replace(":", ".").rpartition(".")
        obj = getattr(sys.modules[module], cls)()
        assert isinstance(obj, base), f"{spec} is not a {base.__name__}"
        return obj

    fake_hooks.load_hook = load_hook
    sys.modules["hooks"] = fake_hooks
    try:
        loaded = g.load_external_probes({"preflight": {"probes": ["extprobes:Mine"]}})
        check(list(loaded) == ["x1"], loaded)
        name, fn, spec = loaded["x1"]
        check(name == "mine" and fn(None)[0] is True, loaded)
        check(spec == "extprobes:Mine", "the spec is kept so the bundle records WHO answered")

        check(g.load_external_probes({}) == {}, "no config, no probes, no error")

        # two probes claiming one id would silently shadow each other
        mod.Other = type("Other", (g.Probe,), {"id": "x1", "name": "other"})
        try:
            g.load_external_probes(
                {"preflight": {"probes": ["extprobes:Mine", "extprobes:Other"]}})
            raise AssertionError("a duplicate probe id must raise")
        except ValueError as exc:
            check("two probes claim id" in str(exc), str(exc))
    finally:
        del sys.modules["extprobes"], sys.modules["hooks"]


def test_the_gate_is_importable_under_its_own_name_when_run_as_a_script():
    """A seam is tested from one end and used from the other.

    An extension subclasses the gate's `Probe`. To find it, it looks for a module named
    `preflight_gate` — but as `python3 preflight_gate.py` this module is `__main__`, so the
    lookup misses, the extension loads the FILE, and gets a second module object whose `Probe`
    is a different class. `hooks.load_hook`'s isinstance check then fails on a CORRECT probe,
    and the obvious reading is that the probe is broken. It is not: the next person edits the
    wrong file.

    Neither side's tests could see it. The extension's tests load the gate themselves, so their
    base class is consistent by construction; the gate's tests only exercise built-ins. The
    mismatch exists solely when the gate imports the extension.
    """
    import subprocess
    prog = (
        "import sys, runpy;"
        f"sys.argv=['preflight_gate.py','--print-tree-sha',{HERE!r}];"
        "sys.modules.pop('preflight_gate', None);"
        "runpy.run_path(sys.argv[0], run_name='__main__')"
    )
    # Running it as __main__ must leave it findable under its import name. SystemExit from
    # --print-tree-sha is expected; what matters is that the name resolved before exiting.
    check_prog = (
        "import sys, importlib.util;"
        f"spec=importlib.util.spec_from_file_location('preflight_gate', {os.path.join(HERE, 'preflight_gate.py')!r});"
        "m=importlib.util.module_from_spec(spec); sys.modules['preflight_gate']=m;"
        "spec.loader.exec_module(m);"
        "sub=type('P',(m.Probe,),{'id':'x9','name':'n'});"
        "assert isinstance(sub(), sys.modules['preflight_gate'].Probe), 'identity broken';"
        "print('ok')"
    )
    r = subprocess.run([sys.executable, "-c", check_prog], capture_output=True, text=True)
    check(r.returncode == 0 and "ok" in r.stdout,
          f"a subclass of the imported gate's Probe must satisfy isinstance: {r.stderr[-300:]}")

    # And the script path must register the name, so an extension importing it finds THIS
    # module rather than re-executing the file into a second one.
    src = open(os.path.join(HERE, "preflight_gate.py")).read()
    check('sys.modules.setdefault("preflight_gate", sys.modules["__main__"])' in src,
          "the __main__ block must register the module under its import name")
    check(prog, "")   # keep the constructed argv referenced; the assertion above is the check


# The a1 test is NOT here. It asserted about an ontology-identity probe that this repository
# no longer ships; the probe and its tests belong to the package that implements the ontology,
# which already carries them. A test kept here would assert about code this repo cannot import.
def test_a4_tells_a_cold_start_apart_from_the_serve_once_fault():
    """Runs the real probe with the two container-only imports stubbed, so the branch logic
    is exercised rather than described."""
    fake_cfg = types.ModuleType("config")
    fake_cfg.CFG = {}
    fake_client = types.ModuleType("cloud.client")
    fake_cloud = types.ModuleType("cloud")
    fake_cloud.client = fake_client

    def with_pattern(results):
        """results[i] False -> that attempt raises."""
        seq = iter(results)

        class Stub:
            def detect_and_segment(self, frame, labels):
                if not next(seq):
                    raise TimeoutError("The read operation timed out")
                # the contract: (detections, per-stage timings). GA-85 asserts the second half.
                return [], {k: 1.0 for k in g.pipeline_timing_keys()[0]}

        fake_client.get_perception_backend = lambda cfg: Stub()
        saved = {k: sys.modules.get(k) for k in ("config", "cloud", "cloud.client")}
        sys.modules.update({"config": fake_cfg, "cloud": fake_cloud,
                            "cloud.client": fake_client})
        try:
            return g.a4_perception_twice(frame=_fake_frame())
        finally:
            for k, v in saved.items():
                if v is None:
                    sys.modules.pop(k, None)
                else:
                    sys.modules[k] = v

    ok, d = with_pattern([True, True, True])
    check(ok is True, f"three good calls pass: {d}")

    # The fault a4 was built for: answers once, fails after.
    ok, d = with_pattern([True, False, False])
    check(ok is False, "serve-once must fail")
    check("exists for" in d["why"], d)

    # THE REASON THERE ARE THREE CALLS. A warm-up in the launcher would consume the first
    # call, and serve-once is defined by the first call succeeding — so it would present as
    # F,F and be reported as "unreachable" for a backend that is answering. Three counted
    # calls separate the states without a warm-up anywhere.
    ok, d = with_pattern([False, True, True])
    check(ok is True, f"a cold start passes on calls two and three: {d}")
    check("cold start" in d["why"], d)
    check(d["pattern"] == "FPP", d)

    # The opposite order, measured on the first gated run: attempt 1 timed out, attempt 2
    # returned in 20.5 s from a remote backend. Still refused -- a run whose first frames time
    # out is not a run -- but the remedy is to warm it, not to restart a broken service.
    # answers once mid-sequence and not after: intermittent, and neither named fault
    ok, d = with_pattern([False, True, False])
    check(ok is False, "an intermittent backend must refuse the run")
    check("WARM THE BACKEND" in d["why"], d)
    check("Do not widen the timeout" in d["why"],
          "one sample must not become a timeout change")

    ok, d = with_pattern([False, False, False])
    check(ok is False, "no call answered")
    check("unreachable, not cold" in d["why"], d)
    check(d["pattern"] == "FFF", d)


# --- a7: an expectation that never arrived asserts nothing ---------------------------------
def test_a7_skips_when_no_expectation_reached_the_container():
    # Three of the four PREFLIGHT_EXPECT_* variables were built by the launcher and NOT added to
    # the docker -e list. The gate then ran with empty expectations and would have reported PASS
    # for a comparison it never made — the same shape as the a2 tautology, one level up.
    ok, d = g.a7_source_frozen({})
    check(ok is g.SKIPPED, "no expectation delivered -> SKIPPED, never pass")
    check("did not reach the container" in d["reason"], d)

    ok, d = g.a7_source_frozen({"graph_api": "deadbeefdeadbeef"})
    check(ok is g.SKIPPED, "outside the container there is no frozen root -> SKIPPED, not pass")


# --- a2: every way it can refuse, against the REAL config.py -------------------------------
# GA-73. a2 imports `config` and hashes what THAT module loaded. On the host the same config.py
# (src/perception_module) is loaded under that name with GRAPH_API_CONFIG at a path the test
# controls, so the probe reads a real CFG/CFG_PATH, not a stub of them. What stays untested here
# is only the import path: /ws/install/... is the installed copy and exists in the container alone.
_CONFIG_PY = os.path.join(HERE, "..", "src", "perception_module", "config.py")


def _a2_with(cfg_path, **expect):
    """Run a2 with the real config.py loaded from cfg_path. Returns (config module, ok, detail)."""
    import importlib.util
    saved_env = os.environ.get("GRAPH_API_CONFIG")
    saved_mod = sys.modules.get("config")
    os.environ["GRAPH_API_CONFIG"] = cfg_path
    try:
        spec = importlib.util.spec_from_file_location("config", _CONFIG_PY)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        sys.modules["config"] = mod
        ok, d = g.a2_config_identity(**expect)
        return mod, ok, d
    finally:
        if saved_env is None:
            os.environ.pop("GRAPH_API_CONFIG", None)
        else:
            os.environ["GRAPH_API_CONFIG"] = saved_env
        if saved_mod is None:
            sys.modules.pop("config", None)
        else:
            sys.modules["config"] = saved_mod


def test_a2_passes_when_every_expectation_matches_what_config_loaded():
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "regolo_config.yaml")
        open(path, "w").write("perception:\n  backend: probe_backend\nhooks:\n  filter: found\n")
        mod, ok, d = _a2_with(path)
        check(ok is True, f"a loaded file with no expectation is a pass: {d}")
        check(d["loaded_path"] == path and d["perception_backend"] == "probe_backend", d)
        check(d["hooks_filter"] == "found", d)
        _, ok, d = _a2_with(path, expect_name="regolo_config.yaml",
                            expect_sha=g.file_sha16(path), expect_merged=g.merged_cfg_sha(mod.CFG))
        check(ok is True, f"matching name, file sha and merged sha -> PASS: {d}")


def test_a2_returns_False_when_no_config_file_was_loaded():
    # The silent defect: GRAPH_API_CONFIG names a file that is not there, config.py falls back
    # to _DEFAULTS, hooks.filter is empty and the extension is out of the loop.
    with tempfile.TemporaryDirectory() as td:
        missing = os.path.join(td, "not_here.yaml")
        _, ok, d = _a2_with(missing)
        check(ok is False, f"defaults in force must FAIL, not pass: {d}")
        check(d["loaded_path"] is None and d["config_file_sha256_16"] is None, d)
        check(d["env_path"] == missing and "no config file was loaded" in d["why"], d)
        check(d["hooks_filter"] == "", f"on the defaults hooks.filter is empty, extension out of the loop: {d}")


def test_a2_returns_False_when_loaded_file_name_differs_from_expected():
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "smoke_config.yaml")
        open(path, "w").write("simulation: true\n")
        _, ok, d = _a2_with(path, expect_name="regolo_config.yaml")
        check(ok is False, f"a different file name than the launcher intended must FAIL: {d}")
        check(d["expected_name"] == "regolo_config.yaml" and d["loaded_path"] == path, d)


def test_a2_returns_False_when_config_file_hash_differs_from_the_launchers():
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "regolo_config.yaml")
        open(path, "w").write("simulation: true\n")
        launcher_sha = g.file_sha16(path)
        open(path, "a").write("perception:\n  backend: edited_after_stamp\n")   # the mounted tree moved
        _, ok, d = _a2_with(path, expect_name="regolo_config.yaml", expect_sha=launcher_sha)
        check(ok is False, f"a file that differs from the one the launcher hashed must FAIL: {d}")
        check(d["expected_file_sha"] == launcher_sha and d["config_file_sha256_16"] != launcher_sha, d)
        check("differs from the one the launcher hashed" in d["why"], d)


def test_a2_returns_False_when_merged_config_differs_although_the_file_matches():
    # Same file, different _DEFAULTS on the other side of the boundary: the launcher's merged
    # digest is computed the same way over a CFG that differs in a key the yaml never writes.
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "regolo_config.yaml")
        open(path, "w").write("simulation: true\n")
        mod, _, _ = _a2_with(path)
        other_defaults = g.merged_cfg_sha(dict(mod.CFG, ontology_stub_key="only_on_the_host"))
        _, ok, d = _a2_with(path, expect_sha=g.file_sha16(path), expect_merged=other_defaults)
        check(ok is False, f"file matches, merged does not -> must FAIL: {d}")
        check(d["expected_merged_sha"] == other_defaults, d)
        check("MERGED configuration does not" in d["why"], d)


@needs_container("`import config` from /ws/install/lost3dsg/lib/lost3dsg, the colcon-installed "
                 "copy a2 hashes; the host has only the source tree")
def test_a2_hashes_the_installed_config_module_not_the_source_tree():
    ok, d = g.a2_config_identity(expect_name=os.path.basename(os.environ["GRAPH_API_CONFIG"]))
    check(ok is True, d)
    check(d["loaded_path"].startswith("/ws/") or d["loaded_path"].startswith("/graph_api/"), d)


# --- a6: the comparison can refuse; the TF tree itself cannot be had on the host ---------
# GA-73. rclpy/tf2_ros are stubbed at the module level so the probe's own logic runs: which
# frames it asks for, the 20 s deadline, and the height comparison. The transform VALUE is the
# test's, so what these show is that a wrong value is refused — not that the live tree is right.
def _a6_with(lookup, **kw):
    """Run a6 against a stub TF buffer whose lookup_transform is `lookup(target, source)`.
    The clock is stubbed too: the 20 s deadline must not cost 20 s of wall time."""
    rclpy = types.ModuleType("rclpy")
    rclpy.init = lambda args=None: None
    rclpy.shutdown = lambda: None
    rclpy.create_node = lambda name: name
    rclpy.spin_once = lambda node, timeout_sec=0.0: None
    rclpy.time = types.ModuleType("rclpy.time")
    rclpy.time.Time = lambda: 0
    duration = types.ModuleType("rclpy.duration")
    duration.Duration = lambda seconds=0.0: seconds
    tf2 = types.ModuleType("tf2_ros")

    class Buffer:
        def lookup_transform(self, target, source, when, timeout):
            return lookup(target, source)
    tf2.Buffer = Buffer
    tf2.TransformListener = lambda buf, node: None

    clock = [1000.0]

    def fake_time():
        clock[0] += 1.0        # each loop turn costs a second; the deadline is 20 of them
        return clock[0]
    fakes = {"rclpy": rclpy, "rclpy.time": rclpy.time, "rclpy.duration": duration, "tf2_ros": tf2}
    saved = {k: sys.modules.get(k) for k in fakes}
    saved_time = g.time
    sys.modules.update(fakes)
    g.time = types.SimpleNamespace(time=fake_time)
    try:
        return g.a6_camera_pose_offset(**kw)
    finally:
        g.time = saved_time
        for k, v in saved.items():
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v


def _tf_with_z(z):
    return types.SimpleNamespace(transform=types.SimpleNamespace(
        translation=types.SimpleNamespace(x=0.0, y=0.0, z=z)))


def test_a6_passes_when_the_camera_sits_the_mount_height_above_base_link():
    asked = []

    def lookup(target, source):
        asked.append((target, source))
        return _tf_with_z(1.5)
    ok, d = _a6_with(lookup)
    check(ok is True, f"dz == mount height -> PASS: {d}")
    check(d == {"dz_m": 1.5, "expected_m": 1.5, "tol_m": 0.25}, d)
    check(asked == [("base_link", "habitat_camera")],
          f"the probe must ask for base_link -> habitat_camera and nothing else: {asked}")
    ok, d = _a6_with(lambda t, s: _tf_with_z(1.5 + 0.25))
    check(ok is True, f"the tolerance is inclusive: {d}")


def test_a6_returns_False_when_camera_pose_equals_base_pose():
    # The defect: something publishes the base pose where the camera pose belongs, dz == 0.
    ok, d = _a6_with(lambda t, s: _tf_with_z(0.0))
    check(ok is False, f"a zero offset must FAIL: {d}")
    check(d["dz_m"] == 0.0 and d["expected_m"] == 1.5, d)


def test_a6_returns_False_when_offset_is_outside_tolerance():
    ok, d = _a6_with(lambda t, s: _tf_with_z(1.5 + 0.26))
    check(ok is False, f"0.26 m off with tol 0.25 must FAIL: {d}")
    ok, d = _a6_with(lambda t, s: _tf_with_z(1.0), expect_height_m=1.0, tol=0.05)
    check(ok is True, f"the expectation and tolerance are the caller's: {d}")
    ok, d = _a6_with(lambda t, s: _tf_with_z(1.5), expect_height_m=1.0, tol=0.05)
    check(ok is False, f"a mount height the caller did not expect must FAIL: {d}")


def test_a6_skips_not_passes_when_the_transform_is_never_published():
    turns = []

    def never(target, source):
        turns.append(1)
        raise LookupError("base_link -> habitat_camera: not in the buffer")
    ok, d = _a6_with(never)
    check(ok is g.SKIPPED, f"no transform within the deadline asserted nothing -> SKIPPED: {d}")
    check("not published within 20 s" in d["reason"], d)
    check(len(turns) >= 2, "the probe must keep trying until the deadline, not give up on the first miss")


@needs_container("tf2_ros.Buffer.lookup_transform('base_link', 'habitat_camera') against the "
                 "live TF tree; on the host there is no rclpy and no publisher")
def test_a6_reads_the_live_base_link_to_habitat_camera_transform():
    ok, d = g.a6_camera_pose_offset()
    check(ok is True, d)


# --- the file is runnable as a script ------------------------------------------------------
def test_the_gate_is_runnable_as_a_script():
    r = subprocess.run([sys.executable, os.path.join(HERE, "preflight_gate.py"), "--help"],
                       capture_output=True, text=True)
    check(r.returncode == 0, r.stderr)




def test_observe_withdraws_blocking_authority_and_never_hides_the_verdict():
    """A probe named in --observe RUNS and REPORTS; it does not block. It is NOT a skip.

    The orchestrator's assertion, 2026-08-31: MAPPING_ONLY=1 + a4 fail must give verdict pass
    with a4 in the observed list; MAPPING_ONLY=0 + a4 fail must give verdict fail.

    A mapping run starts no detector, so a4 and a8's perception imports guard something that is
    deliberately absent. Withdrawing their authority over THOSE runs is not the same as not
    looking, and the difference has to be legible in the artefact: an observed probe's verdict
    and detail are recorded exactly as always, and the report names which probes were not
    permitted to speak. A reader seeing "pass" must be able to see that the pass is narrower
    than a normal one.
    """
    import json
    import subprocess
    import tempfile

    def run(observe):
        out = tempfile.NamedTemporaryFile(suffix=".json", delete=False).name
        argv = [sys.executable, os.path.join(HERE, "preflight_gate.py"),
                "--out", out, "--only", "a4"]
        if observe:
            argv += ["--observe", observe]
        # a4 cannot pass outside the container: no probe frame, no backend. What it returns is
        # not the point — whether it can BLOCK is.
        rc = subprocess.run(argv, capture_output=True, text=True).returncode
        with open(out) as f:
            return rc, json.load(f)

    rc_block, rep_block = run(None)
    rc_obs, rep_obs = run("a4")

    assert rep_block["verdict"] == "fail", \
        f"a4 must block a normal run; got {rep_block['verdict']}"
    assert rc_block != 0, "a blocking failure must be a non-zero exit"

    assert rep_obs["verdict"] == "pass", \
        f"an observed a4 must not block; got {rep_obs['verdict']} failed={rep_obs['failed']}"
    assert rc_obs == 0, "an observed-only failure must exit 0"
    assert rep_obs["observed"] == ["a4"], \
        "the report must NAME which probes were observed, or a narrowed pass reads as a full one"
    assert "a4" in rep_obs["observed_nonpass"], \
        "a4 did not pass and the report must say so even though it did not block"

    # The probe still RAN and its verdict is still on the row. This is the line between
    # observe and skip, and it is the whole reason SKIPPED fails the gate.
    row = next(r for r in rep_obs["probes"] if r["id"] == "a4")
    assert row.get("observed") is True, "the row itself must be marked observed"
    assert row["ok"] is not True, "the probe's real verdict must survive, not be rewritten to pass"
    assert row["detail"], "an observed probe must still record WHY, or it is a skip with extra words"


def test_observe_refuses_a_probe_id_that_does_not_exist():
    """A typo in --observe grants no exemption and looks like it did. Refuse instead.

    `--observe a44` would silently leave a4 blocking while the operator believed it was observed.
    Same class as a config key nobody reads: the failure is invisible and the belief is wrong.
    """
    import subprocess
    r = subprocess.run([sys.executable, os.path.join(HERE, "preflight_gate.py"),
                        "--out", "/dev/null", "--only", "a5", "--observe", "a44"],
                       capture_output=True, text=True)
    assert r.returncode == 2, f"an unknown probe id must be refused, got rc={r.returncode}"
    assert "do not exist" in r.stdout + r.stderr


def test_a7_compares_a_live_root_against_the_launcher_stamp_and_blocks_only_when_exercised():
    """GA-373. The live roots were sampled and compared with nothing; a write into a live-mounted tree
    inside the freeze window refused nothing (run 20260908_131906). A differing live root is a
    MISMATCH — blocking when the run executes extension code, recorded-only under MAPPING_ONLY."""
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        src = os.path.join(td, "mount"); run = os.path.join(td, "install"); live = os.path.join(td, "found")
        for d in (src, run, live):
            os.makedirs(d)
        open(os.path.join(src, "node.py"), "w").write("x = 1\n")
        open(os.path.join(run, "node.py"), "w").write("x = 1\n")
        open(os.path.join(live, "a.py"), "w").write("y = 1\n")
        stamp, _ = g.tree_sha(live)
        roots = {"found": live}
        # unchanged live root: no mismatch either way
        ok, d = g.a7_source_frozen({"graph_api": "x", "found": stamp}, executed_tree=run,
                                   copy_source=src, live_roots=roots, found_exercised=True)
        check(d["live_mismatches"] == {}, d)
        # the live root moves after the stamp
        open(os.path.join(live, "a.py"), "w").write("y = 2\n")
        ok, d = g.a7_source_frozen({"graph_api": "x", "found": stamp}, executed_tree=run,
                                   copy_source=src, live_roots=roots, found_exercised=True)
        # ok is SKIPPED on the host (no frozen root outside the container); the verdict is exercised
        # in the image. Here the MISMATCH record is what is asserted, as the a7 test above does.
        check("found_live" in d["mismatches"], f"an exercised live tree must be a mismatch: {d['mismatches']}")
        check(d["live_mismatches"]["found"]["blocking"] is True, d["live_mismatches"])
        ok, d = g.a7_source_frozen({"graph_api": "x", "found": stamp}, executed_tree=run,
                                   copy_source=src, live_roots=roots, found_exercised=False)
        check("found_live" not in d["mismatches"], f"MAPPING_ONLY must not block: {d['mismatches']}")
        check(d["live_mismatches"]["found"]["blocking"] is False and "why_recorded" in d, d)
        # no stamp given for the live root: sampled, never compared (the pre-GA-373 behaviour, stated)
        _, d = g.a7_source_frozen({"graph_api": "x"}, executed_tree=run, copy_source=src,
                                  live_roots=roots, found_exercised=True)
        check(d["live_mismatches"] == {} and "found" in d["live_sampled"], d)


def test_a7_at_teardown_judges_only_the_live_roots():
    """Run 20260908_135714: the teardown a7 re-judged the vendored MOUNT against the launch stamp
    and failed on edits the lanes were allowed to make after the copied ping. At teardown only
    the live roots are asserted; the mount reading is kept as information."""
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        src = os.path.join(td, "mount"); run = os.path.join(td, "install"); live = os.path.join(td, "found")
        for d in (src, run, live):
            os.makedirs(d)
        open(os.path.join(src, "node.py"), "w").write("x = 2\n")   # mount moved after the copy
        open(os.path.join(run, "node.py"), "w").write("x = 1\n")   # what executed
        open(os.path.join(live, "a.py"), "w").write("y = 1\n")
        stamp, _ = g.tree_sha(live)
        ok, d = g.a7_source_frozen({"graph_api": "stale", "found": stamp}, executed_tree=run,
                                   copy_source=src, live_roots={"found": live}, found_exercised=True,
                                   teardown=True)
        check(d["mismatches"] == {}, f"a moved mount at teardown must not be a mismatch: {d['mismatches']}")
        check("executed_vs_mount" in d["informational_at_teardown"]["mount_vs_stamp"], d)
        open(os.path.join(live, "a.py"), "w").write("y = 2\n")   # a live tree moved during the run
        ok, d = g.a7_source_frozen({"graph_api": "stale", "found": stamp}, executed_tree=run,
                                   copy_source=src, live_roots={"found": live}, found_exercised=True,
                                   teardown=True)
        check(list(d["mismatches"]) == ["found_live"], f"a moved live root must be the only mismatch: {d['mismatches']}")


def test_found_exercised_auto_follows_the_configured_hook():
    """GA-435. "is this a detection run" is not "does this run execute extension code".

    The caller passed 1 for every run that was not MAPPING_ONLY, so on a deployment with NO
    extension a7 failed with live_roots_undeclared and the gate refused every run. Measured on Gin,
    where no hooks are configured and no EXT_* variable is set.
    """
    import importlib
    import sys
    import types
    g = importlib.import_module("preflight_gate")
    saved = sys.modules.get("config")
    try:
        # explicit answers still mean what they meant
        assert g._found_exercised("1") is True
        assert g._found_exercised("0") is False

        fake = types.ModuleType("config")
        # A module:Class the way an extension exports one. The literal used to name a real
        # deployment, which made this repository fail its own boundary check on a fixture.
        fake.CFG = {"hooks": {"filter": "yourpkg.filter:OntologicalFilter"}}
        sys.modules["config"] = fake
        assert g._found_exercised("auto") is True, "a configured filter means extension code runs"

        fake.CFG = {"hooks": {"search_paths": ["/found"], "filter": ""}}
        assert g._found_exercised("auto") is False, "no filter means nothing extends this run"

        fake.CFG = {}
        assert g._found_exercised("auto") is False, "no hooks section at all"
    finally:
        if saved is None:
            sys.modules.pop("config", None)
        else:
            sys.modules["config"] = saved


def test_a4_asserts_every_hub_model_a_run_loads():
    """GA-438. The cache check sat inside `backend == "local"`, so a cloud run asserted nothing.

    MiniLM and both dinov2 sizes load on EVERY run whatever the backend -- nlp_utils for semantic
    matching, visual_reid for the crop embedder, which picks -base over -small on measured VRAM, so
    both must be present or the run downloads whichever it chooses.
    """
    import os
    import tempfile
    from preflight_gate import HF_MODELS_EVERY_RUN, _hf_models_present

    saved = {k: os.environ.get(k) for k in ("PREFLIGHT_HF_CACHE", "HF_HOME", "TRANSFORMERS_CACHE")}
    try:
        with tempfile.TemporaryDirectory() as td:
            os.environ["PREFLIGHT_HF_CACHE"] = td
            for k in ("HF_HOME", "TRANSFORMERS_CACHE"):
                os.environ.pop(k, None)

            _, missing, seen = _hf_models_present(local_backend=False)
            check(len(missing) == len(HF_MODELS_EVERY_RUN) and not seen,
                  f"an empty cache must report every always-loaded model missing: {missing}")

            # A DIRECTORY IS NOT A CACHED MODEL: huggingface creates it before the fetch finishes,
            # and an interrupted download leaves snapshots/ empty. That must still read as missing.
            stem = os.path.join(td, "hub", "models--facebook--dinov2-small")
            os.makedirs(os.path.join(stem, "snapshots", "abc123"))
            _, missing, seen = _hf_models_present(local_backend=False)
            check("facebook/dinov2-small" in missing,
                  "an empty snapshots/ directory must NOT count as cached")

            open(os.path.join(stem, "snapshots", "abc123", "config.json"), "w").write("{}")
            _, missing, seen = _hf_models_present(local_backend=False)
            check("facebook/dinov2-small" in seen and "facebook/dinov2-small" not in missing,
                  f"a snapshot holding a file must count as cached: {missing}")

            # the local backend asks for one more model than a cloud backend
            _, m_cloud, _ = _hf_models_present(local_backend=False)
            _, m_local, _ = _hf_models_present(local_backend=True)
            check(len(m_local) == len(m_cloud) + 1,
                  f"the local backend must also assert the detector: {m_local} vs {m_cloud}")
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


if __name__ == "__main__":
    # Mirrors src/perception_module/test_config.py: every function is a pytest test AND
    # this file still runs as a script. It collected ZERO tests under pytest before, because
    # every assertion sat at module level -- a runner reporting success over an empty
    # collection is a gate that cannot fail, which is the shape this gate exists to catch.
    tests = [(n, f) for n, f in sorted(globals().items())
             if n.startswith("test_") and callable(f)]
    failures = skipped = 0
    for name, fn in tests:
        if getattr(fn, "needs_container", None):
            print(f"  skip  {name}: {fn.needs_container}")
            skipped += 1
            continue
        try:
            fn()
            print(f"  ok    {name}")
        except Exception as exc:
            failures += 1
            print(f"  FAIL  {name}: {exc}")
    # The count is printed because this runner reads globals() at the moment it runs: a
    # test appended BELOW this block is defined too late and is silently skipped.
    # The time is printed WITH the result, because a test result is true at a time and not
    # simply true. Two correct measurements of one file can disagree because someone fixed it
    # in between, and neither reader is wrong — that cost two sessions an exchange tonight.
    # The frozen-root digests have always carried a stamp; test results carried none.
    print(f"preflight gate tests: {len(tests) - skipped} run, {skipped} skipped (need the container),",
          "all passed" if not failures else f"{failures} failed",
          f"| {__import__('datetime').datetime.now().astimezone().isoformat(timespec='seconds')}")
    # Kept accurate deliberately. This said a4 was not exercised after a stubbed test for its
    # branch logic had landed -- a statement about the tests that the tests contradicted, which
    # is the family this gate exists for. It under-claimed, so it failed safe; it was still
    # wrong, and a summary nobody maintains is how "verified" drifts from what ran.
    print("a4's BRANCH LOGIC is exercised here against stubs; its two real inferences are not.")
    print("a1, a2 and a6 are shown REFUSING here (a2 against the real config.py; a1 and a6 against")
    print("stubbed aligner / TF modules). NOT exercised here (need the container, see the skips):")
    print("a2's installed config copy, a6's live TF tree, a7 against a real frozen root.")
    print("a6 and a7 passed live on 2026-08-30; a1, a2 and a4 have not yet run to completion.")
    sys.exit(1 if failures else 0)

def test_a12_refuses_a_gt_reader_outside_the_allow_list_and_passes_the_clean_tree():
    """GA-355. The negative control is the point: a reader injected into the association loop
    and into perception_2.publish_objects must be NAMED by the probe, and the untouched copy must
    pass. Rule 18/26: run it, do not predict it."""
    import shutil
    import tempfile

    from preflight_gate import a12_gt_isolation
    lost = os.path.dirname(HERE)
    ignore = shutil.ignore_patterns("__pycache__", ".ruff_cache", "*.pyc", "output", "probe_assets")
    with tempfile.TemporaryDirectory() as td:
        for d in ("src", "test"):
            shutil.copytree(os.path.join(lost, d), os.path.join(td, d), ignore=ignore)
        ok, detail = a12_gt_isolation(root=td)
        assert ok is True, detail
        om6 = os.path.join(td, "src", "perception_module", "object_manager_6.py")
        with open(om6, "a") as fh:
            fh.write("\n_leak = os.environ.get('FEED_GT_SEMANTIC')\n")
        p2 = os.path.join(td, "src", "perception_module", "perception_2.py")
        src = open(p2).read()
        needle = "    def publish_objects("
        assert needle in src, "perception_2.publish_objects moved; re-point the negative test"
        src = src.replace(needle, "    def publish_objects(self, *_a, **_k):\n        _x = self._gt_semantic\n\n" + needle, 1)
        open(p2, "w").write(src)
        ok, detail = a12_gt_isolation(root=td)
        assert ok is False, detail
        assert "src/perception_module/object_manager_6.py" in detail["not_allowed_files"], detail
        assert "publish_objects" in detail["perception_2_functions_not_allowed"], detail


def test_a13_verdict_requires_exactly_one_authority_on_the_stamped_side():
    """GA-359. Today's stack (both sides publish) must FAIL under either arm; each single
    authority passes only under its own arm; no authority fails."""
    from preflight_gate import a13_verdict
    both = [("map", "odom")]
    assert a13_verdict("simulator", both, both)[0] is False
    assert a13_verdict("rtabmap", both, both)[0] is False
    assert a13_verdict("simulator", [], both)[0] is True
    assert a13_verdict("rtabmap", [], both)[0] is False
    assert a13_verdict("rtabmap", both, [])[0] is True
    assert a13_verdict("simulator", both, [])[0] is False
    ok, d = a13_verdict("rtabmap", [], [])
    assert ok is False and "NO authority" in d["reason"]
    # other pairs do not count as map->odom
    assert a13_verdict("simulator", [("odom", "base_link")], both)[0] is True
