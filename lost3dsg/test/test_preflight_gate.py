#!/usr/bin/env python3
"""Runnable check for preflight_gate.py. Host-runnable: no ROS, no container, no network.

    python3 test_preflight_gate.py

Covers the harness and the three probes that need neither ROS nor a backend (a2, a3, a5).
a1/a4/a6 need the container and are NOT exercised here — see the note at the bottom, which
is deliberate: claiming a probe is verified when it has never run is the failure this whole
gate exists to catch.
"""
import json
import os
import subprocess
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import preflight_gate as g  # noqa: E402


def check(cond, msg):
    if not cond:
        raise AssertionError(msg)



# --- a3: the policy comparison must compare, not echo -------------------------------------
def test_a3_compares_rather_than_echoes():
    os.environ["FOUND_ENFORCE"] = "1"
    os.environ.pop("FOUND_HOLD_BAND", None)

    ok, d = g.a3_policy_reached_container({"FOUND_ENFORCE": "1"})
    check(ok is True, f"matching policy should pass: {d}")

    ok, d = g.a3_policy_reached_container({"FOUND_ENFORCE": "0"})
    check(ok is False, "a value that differs from the intended one must FAIL")
    check(d["mismatches"]["FOUND_ENFORCE"] == {"intended": "0", "in_container": "1"}, d)

    # The defect this probe exists for: the var never reached the container at all. Absent must
    # fail, not pass — an echo-style check would have reported the host's value and looked fine.
    ok, d = g.a3_policy_reached_container({"FOUND_HOLD_BAND": "0.05"})
    check(ok is False, "a var absent from the container must FAIL, not pass")
    check(d["mismatches"]["FOUND_HOLD_BAND"]["in_container"] is None, d)

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

        rc = g.main(["--only", "a3", "--out", out, "--expect-policy", "FOUND_ENFORCE=0"])
        check(rc == 1, "a real mismatch must abort")
        rep = json.load(open(out))
        check(rep["failed"] == ["a3"] and rep["verdict"] == "fail", rep)


# --- harness: a probe that raises is SKIPPED, never a pass --------------------------------
def test_a_probe_that_raises_is_skipped_not_passed():
    with tempfile.TemporaryDirectory() as td:
        out = os.path.join(td, "p.json")
        rc = g.main(["--only", "a1", "--out", out])   # `found` is not importable on the host
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


# --- the file is runnable as a script ------------------------------------------------------
def test_the_gate_is_runnable_as_a_script():
    r = subprocess.run([sys.executable, os.path.join(HERE, "preflight_gate.py"), "--help"],
                       capture_output=True, text=True)
    check(r.returncode == 0, r.stderr)



if __name__ == "__main__":
    # Mirrors src/perception_module/test_config.py: every function is a pytest test AND
    # this file still runs as a script. It collected ZERO tests under pytest before, because
    # every assertion sat at module level -- a runner reporting success over an empty
    # collection is a gate that cannot fail, which is the shape this gate exists to catch.
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
    # The count is printed because this runner reads globals() at the moment it runs: a
    # test appended BELOW this block is defined too late and is silently skipped.
    print(f"preflight gate tests: {len(tests)} run,",
          "all passed" if not failures else f"{failures} failed")
    print("NOT exercised here (need the container): a1 aligner identity, a2 config identity,")
    print("a4 perception twice, a6 camera pose offset, a7 against a real frozen root.")
    print("They are WRITTEN, not TESTED, until the first gated run.")
    sys.exit(1 if failures else 0)