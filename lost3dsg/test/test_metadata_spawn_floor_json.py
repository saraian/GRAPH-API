#!/usr/bin/env python3
"""A signed positive floor makes run_metadata.json invalid and aborts the launch.

GA-507 (2026-09-14, simulator lane). The run_metadata heredoc in `run_sim.sh` puts
`$FEED_SPAWN_FLOOR` into two BARE JSON slots -- `spawn_floor_requested` and `spawn_floor` -- with no
quotes, because the floor is a number. JSON forbids a leading `+`, so `+0.07` writes

    "spawn_floor_requested": +0.07,

and the validator at the end of the heredoc aborts the run before the container starts.

MEASURED 2026-09-14, with the launcher's own expression: `+0.07` INVALID, `-1.59` VALID, `0.00`
VALID, empty VALID. The failure is loud and cheap -- no invalid bundle ever ships -- but the run
does not start.

WHY IT REACHES EVERY STOREY AT z >= 0, not just one edge case. `run_sim.sh` formats every published
map directory with `:+.2f`, which ALWAYS writes a sign. Of the four published floors on this machine
-- floor_+0.00, floor_+0.11, floor_+1.21, floor_-1.59 -- three carry `+`. Both paths that choose a
floor hand the signed text straight through: the auto-draw path, and the house driver, which strips
only the `floor_` prefix. So the base run policy (working rule 73: full house tour, every storey)
aborts on any storey at z >= 0, and only the one negative floor launches.

THE TEST EVALUATES THE LAUNCHER'S OWN LINES, not a copy of them (working rule 75): it reads each
slot out of `run_sim.sh` as it stands today, runs it through bash with a floor value, and parses the
result with the same `json` module the launcher's validator uses. A rewrite of those lines is
therefore tested, not just the string that is there now.

Run: python3 test_metadata_spawn_floor_json.py   (or under pytest)
"""
import json
import os
import re
import subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPT = os.path.join(HERE, "..", "..", "run_sim.sh")

# The published floors on this machine, plus the unsigned and empty cases. A sign is what the
# formatter writes, so the positive ones are the whole point of the test.
FLOORS = ["+0.00", "+0.11", "+1.21", "-1.59", "0.00", ""]

# A JSON key whose value is produced from FEED_SPAWN_FLOOR without surrounding quotes.
SLOT_RE = re.compile(r'^\s*"(?P<key>[a-z_]+)":\s*(?P<expr>\$\(.*FEED_SPAWN_FLOOR.*\)),?\s*$')


def spawn_floor_slots(text):
    """-> [(line number, key, shell expression)] for every bare JSON slot fed by the floor."""
    out = []
    for n, ln in enumerate(text.split("\n"), start=1):
        m = SLOT_RE.match(ln)
        if m:
            out.append((n, m.group("key"), m.group("expr")))
    return out


def render(expr, floor):
    """Run one slot expression under bash, exactly as the unquoted heredoc would."""
    r = subprocess.run(
        ["bash", "-c", 'printf "%s" "' + expr + '"'],
        env={**os.environ, "FEED_SPAWN_FLOOR": floor},
        capture_output=True, text=True, check=True)
    return r.stdout


def test_every_spawn_floor_slot_survives_a_signed_floor():
    slots = spawn_floor_slots(open(SCRIPT).read())
    # Assert the instrument found its subject. A regex that matches nothing passes vacuously
    # (working rule 78), and the slots have moved line numbers twice already.
    assert len(slots) >= 2, f"expected the two spawn-floor slots in {SCRIPT}, found {slots}"
    bad = []
    for n, key, expr in slots:
        for floor in FLOORS:
            doc = '{"%s": %s}' % (key, render(expr, floor))
            try:
                json.loads(doc)
            except ValueError as e:
                bad.append(f"  {SCRIPT}:{n}  FEED_SPAWN_FLOOR={floor!r} -> {doc}  ({e})")
    assert not bad, (
        "these run_metadata.json slots are not valid JSON, so the launcher's validator aborts "
        "the run before the container starts:\n" + "\n".join(bad)
        + "\nStrip the sign at the slot: ${FEED_SPAWN_FLOOR#+}. JSON allows a leading minus and "
          "forbids a leading plus.")


def test_the_check_fails_on_the_defect_it_was_written_for():
    """Exercise the instrument where the answer is PRESENT (working rules 60, 70, 77).

    This is the exact text the two slots carried before GA-507 was fixed.
    """
    unfixed = '    "spawn_floor": $([ -n "${FEED_SPAWN_FLOOR:-}" ] && echo "$FEED_SPAWN_FLOOR" || echo null),'
    slots = spawn_floor_slots(unfixed)
    assert len(slots) == 1, slots
    _, key, expr = slots[0]
    assert json.loads('{"%s": %s}' % (key, render(expr, "-1.59"))) == {"spawn_floor": -1.59}
    assert json.loads('{"%s": %s}' % (key, render(expr, ""))) == {"spawn_floor": None}
    try:
        json.loads('{"%s": %s}' % (key, render(expr, "+0.11")))
    except ValueError:
        return
    raise AssertionError("the unfixed slot accepted +0.11; this check cannot see GA-507")


def test_the_fix_keeps_the_value_it_reports():
    """Stripping the sign must not change the number, and must leave a minus alone."""
    slots = spawn_floor_slots(open(SCRIPT).read())
    for n, key, expr in slots:
        assert json.loads('{"%s": %s}' % (key, render(expr, "+1.21")))[key] == 1.21, n
        assert json.loads('{"%s": %s}' % (key, render(expr, "-1.59")))[key] == -1.59, n
        assert json.loads('{"%s": %s}' % (key, render(expr, "")))[key] is None, n


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"ok  {name}")
    print("all checks passed")
