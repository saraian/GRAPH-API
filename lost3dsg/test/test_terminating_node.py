#!/usr/bin/env python3
"""GA-430: how a launch ENDED must be a field, and 'absent' must not read as 'unknown'.

Rule 73 removed the cap, so a launch now ends in exactly two ways: the tour completes, or something
dies. That makes "how did this end" the load-bearing fact in a bundle, and the testing lane's
early-death condition still reads a LOG LINE for it -- a sentence in the stack log, which is prose a
future edit breaks silently.

live_stack_container.sh writes terminating_node.json from the loop that watched the nodes, at the
moment it knows; live_run.sh folds it into run_metadata.json. THE BRANCH NO RUN EXERCISES is the
absent one: a container killed from outside, or one that died before its watch loop, leaves no
terminating_node.json at all, and the fold must say that in words rather than leaving a null a
reader will interpret as "nobody looked".

This runs the launcher's OWN stamping block, extracted between its TEST-EXTRACT markers, so it
tests the code that ships rather than a copy of it. Same convention as test_env_stamp.sh.

Run: python3 test_terminating_node.py   (or under pytest)
"""
import json
import os
import re
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
LAUNCHER = os.path.join(HERE, "live_run.sh")
BEGIN = "# >>> TEST-EXTRACT stamp_block"
END = "# <<< TEST-EXTRACT stamp_block"


def stamp_block():
    """-> the launcher's stamping python, verbatim."""
    text = open(LAUNCHER).read()
    i, j = text.index(BEGIN), text.index(END)
    body = text[i:j]
    body = body[body.index("\n") + 1:]
    # THE MARKERS MUST STILL BOUND REAL CODE. An extraction that silently returns a comment block
    # would make every assertion below pass over nothing.
    assert re.search(r"^import json", body, re.M), f"extracted no code:\n{body[:200]}"
    assert "terminating_node" in body, "the extracted block does not stamp terminating_node"
    return body


def run_stamp(bundle, term=None):
    """Write a metadata file, optionally a terminating_node.json, run the block, return the JSON."""
    meta = os.path.join(bundle, "run_metadata.json")
    json.dump({"run_id": "test"}, open(meta, "w"))
    if term is not None:
        open(os.path.join(bundle, "terminating_node.json"), "w").write(term)
    script = os.path.join(bundle, "_stamp.py")
    open(script, "w").write(stamp_block())
    r = subprocess.run([sys.executable, script, meta, "", "1234", "false", "null", "no cap", "0"],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    return json.load(open(meta))


def test_a_completed_tour_is_named_as_a_field():
    with tempfile.TemporaryDirectory() as td:
        d = run_stamp(td, json.dumps({"node": "FEED_ENDED", "exit_status": 0,
                                      "reason": "the feed host completed the house tour"}))
        tn = d["terminating_node"]
        assert tn["node"] == "FEED_ENDED", tn
        # A STATUS OF 0 IS STILL AN ENDING. Reading the status alone cannot tell a finished tour
        # from a watched node exiting 0 unexpectedly; the NODE is what discriminates.
        assert tn["exit_status"] == 0, tn


def test_a_death_is_named_with_its_signal():
    with tempfile.TemporaryDirectory() as td:
        d = run_stamp(td, json.dumps({"node": "RTABMAP", "exit_status": -6,
                                      "reason": "SIGABRT — an assertion or uncaught exception"}))
        tn = d["terminating_node"]
        assert tn["node"] == "RTABMAP" and tn["exit_status"] == -6, tn


def test_absent_says_so_rather_than_reading_as_unknown():
    """The branch a run cannot produce on demand: killed before the watch loop wrote anything."""
    with tempfile.TemporaryDirectory() as td:
        d = run_stamp(td, term=None)
        tn = d["terminating_node"]
        assert tn["node"] is None, tn
        assert "note" in tn and tn["note"], "an absent file must carry its own explanation"
        assert "killed" in tn["note"], tn["note"]


def test_a_torn_file_takes_the_absent_branch_and_names_the_error():
    """A container killed mid-write leaves half a JSON object. That must not abort the stamp."""
    with tempfile.TemporaryDirectory() as td:
        d = run_stamp(td, term='{"node": "RTAB')
        tn = d["terminating_node"]
        assert tn["node"] is None, tn
        assert "ValueError" in tn["note"] or "JSONDecodeError" in tn["note"], tn["note"]


def test_ended_is_a_closed_vocabulary_a_reader_can_test():
    """The testing lane's early-death condition should read a VALUE, not grep a sentence."""
    cases = [(json.dumps({"node": "FEED_ENDED", "exit_status": 0}), "tour_complete"),
             (json.dumps({"node": "MAPPING_TIME", "exit_status": 0}), "mapping_time"),
             (json.dumps({"node": "RTABMAP", "exit_status": -6}), "node_death"),
             # A WATCHED NODE EXITING 0 IS STILL A DEATH, which is why the node and not the status
             # decides. A reader keying on exit_status would call this one a clean finish.
             (json.dumps({"node": "PERCEPTION", "exit_status": 0}), "node_death"),
             (None, "unrecorded")]
    for term, want in cases:
        with tempfile.TemporaryDirectory() as td:
            d = run_stamp(td, term)
            got = d["terminating_node"]["ended"]
            assert got == want, f"{term} -> {got}, wanted {want}"


def test_the_cap_block_still_lands_beside_it():
    """GA-395's keys must survive: one stamp writes both and a regression here loses the cap."""
    with tempfile.TemporaryDirectory() as td:
        d = run_stamp(td, json.dumps({"node": "FEED_ENDED", "exit_status": 0}))
        assert d["cap"]["capped"] is False and d["cap"]["elapsed_seconds"] == 1234, d["cap"]
        assert d["run_id"] == "test", "the stamp must not drop the metadata it was handed"


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"ok  {name}")
    print("all checks passed")
