"""The viewer's staleness helpers, exercised rather than eyeballed.

The graph fades an object that has stopped being perceived, and the objects table shows how
long ago each one was last seen. Both read `last_seen`, which the bridge publishes per object,
and both measure it against NOW -- which on a replay is the frame on screen, not today's clock.
Get that wrong and every object on a recorded run reads as hours stale.

The functions are inline in viewer.html, so they are lifted out by name and run under node. If
a rename makes the lift fail this test FAILS rather than passing on an empty extraction, which
is the way a check like this usually goes quiet.
"""
import json
import shutil
import subprocess
import sys
from pathlib import Path

HTML = (Path(__file__).resolve().parent.parent /
        "src" / "perception_module" / "viewer" / "viewer.html")

node = shutil.which("node") or shutil.which("nodejs")
if node is None:
    print("SKIP: no node on this host")
    sys.exit(0)

src = HTML.read_text()


def lift(name, kind="const"):
    """Pull one `const name = ...;` or `function name(...) {...}` out of the page."""
    if kind == "function":
        i = src.index(f"function {name}(")
    else:
        i = src.index(f"const {name} = ")
    depth, j, started = 0, i, False
    while j < len(src):
        if src[j] == "{":
            depth += 1
            started = True
        elif src[j] == "}":
            depth -= 1
            if started and depth == 0:
                return src[i:j + 1]
        elif src[j] == ";" and not started:
            return src[i:j + 1]
        j += 1
    raise AssertionError(f"could not lift {name}")


parts = [lift("graphNowSec", "function"), lift("nowSec"), lift("ageText"),
         lift("staleColour"), lift("STALE_AGING_S")]
for p in parts:
    assert len(p) > 20, f"lifted something empty: {p!r}"
assert "REPLAY_NOW" in parts[0], "graphNowSec no longer consults the replay clock"
# One definition of "now", shared by the graph and the table. Two copies of a rule
# about time is how they end up disagreeing on a scrubbed replay.
assert src.count("window.REPLAY_NOW > 0)") == 1, \
    "the replay-clock rule has been duplicated again"

NOW = 1_000_000.0
script = """
const window = {};
%s
const out = {
  // absent stays absent, at every call site
  ageNull: ageText(null), ageNaN: ageText(NaN), ageUndef: ageText(undefined),
  colourNull: staleColour(null),
  // and the bands
  fresh: ageText(NOW - 5),      freshC: staleColour(NOW - 5),
  aging: ageText(NOW - 60),     agingC: staleColour(NOW - 60),
  stale: ageText(NOW - 600),    staleC: staleColour(NOW - 600),
  hours: ageText(NOW - 7200),
  // a stamp in the future must not read as a negative age
  future: ageText(NOW + 90),
  // the replay clock wins over the wall clock when it is set
  replay: (function () { window.REPLAY_NOW = NOW; return ageText(NOW - 10); })(),
  thresholds: [STALE_AGING_S, STALE_GONE_S],
};
console.log(JSON.stringify(out));
""" % ("const NOW = %r;\n" % NOW + "\n".join(parts))

# `nowSec` falls back to Date.now(); pin it so the bands are deterministic.
script = script.replace("(Date.now() / 1000)", "NOW")

r = subprocess.run([node, "-e", script], capture_output=True, text=True)
assert r.returncode == 0, f"node refused the lifted helpers:\n{r.stderr}"
out = json.loads(r.stdout)

assert out["ageNull"] == "—" and out["ageNaN"] == "—" and out["ageUndef"] == "—", out
assert out["colourNull"] == "var(--muted)", out
assert out["fresh"] == "5s" and out["aging"] == "60s", out
assert out["stale"] == "10m", out
assert out["hours"] == "2.0h", out
assert out["future"] == "0s", f"a future stamp read as {out['future']}"
assert out["replay"] == "10s", "the replay clock is not being used"
assert out["freshC"] != out["agingC"] != out["staleC"], "the three bands look identical"
assert out["thresholds"] == [30, 180], out["thresholds"]

# And the table really has the column, or the helpers above are dead code.
assert "<th title=\"How long ago this object was last perceived" in src, \
    "the objects table lost its Last seen column"
assert "node.aging" in src and "node.stale" in src, "the graph lost its staleness styles"
assert "applyNodeStaleness()" in src, "nothing re-applies staleness after a sync"

print("OK: age bands, null handling, future stamps and the replay clock all behave; "
      "column and graph styles present")
