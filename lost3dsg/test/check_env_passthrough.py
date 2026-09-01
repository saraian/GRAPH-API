#!/usr/bin/env python3
"""Every environment variable the CONTAINER reads must actually reach the container.

Owner ruling 22, 2026-09-01: "sixteen settings that look adjustable and aren't — fix the class."

WHY A SECOND CHECKER RATHER THAN A BIGGER GREP. test_env_stamp.sh §10 already checks this, and it
caught RTABMAP_CLOSE_TIMEOUT four minutes after it was written — but it greps `${VAR}` out of
live_stack_container.sh, WHICH IS A SHELL SCRIPT. graph_api_bridge.py reads BRIDGE_PORT through
os.environ, and no shell grep can see that. The four variables §10 did catch were all read by the
shell script itself, so they were in scope by construction; the Python side has never been visible.

Measured when this was written: 19 environment variables are read by container-side Python, 3 are
set inside the container, and 16 were neither set nor passed. Most have working defaults and were
not defects — they were sixteen levers that LOOK settable from the host and are not. BRIDGE_PORT
was the one that cost something: it was the only mitigation for the 8081 port conflict and setting
it reached nothing.

The failure shape is the same one that runs through this whole tree: an unset variable takes the
default branch silently, so "the setting is off" and "the setting never arrived" are one
observation.
"""
import ast
import pathlib
import re
import sys

# Set inside live_stack_container.sh, so the host never needs to pass them.
SET_IN_CONTAINER = {"GRAPH_API_CONFIG", "GRAPH_API_OUTPUT_DIR", "HF_HOME", "PYTHONPATH",
                    "RTABMAP_GRID_ARGS", "PREFLIGHT_OBSERVE", "LOG_DIR", "ROS_DOMAIN_ID"}
# Read by host-side processes only; passing them into the container would mean nothing.
HOST_ONLY = {"OUT_DIR", "FEED_SHOW", "FEED_OVERLAY", "FEED_SEED", "FEED_WALK", "FEED_DWELL",
             "FEED_FPS", "FEED_SPAWN_FLOOR", "FEED_BRIDGE", "FEED_CTRL_PORT", "FEED_SEND_TIMEOUT",
             "FEED_MAX_SEND_BACKLOG", "HABITAT_SCENE", "HABITAT_DATASET", "HOME"}


def env_reads(py: pathlib.Path):
    """Every os.environ / os.getenv key read in this module."""
    try:
        tree = ast.parse(py.read_text())
    except (SyntaxError, UnicodeDecodeError):
        return set()
    out = set()
    for n in ast.walk(tree):
        name = None
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr in ("get", "getenv"):
            if n.args and isinstance(n.args[0], ast.Constant) and isinstance(n.args[0].value, str):
                if "environ" in ast.unparse(n.func.value) or n.func.attr == "getenv":
                    name = n.args[0].value
        elif isinstance(n, ast.Subscript) and isinstance(n.value, ast.Attribute) and n.value.attr == "environ":
            if isinstance(n.slice, ast.Constant) and isinstance(n.slice.value, str):
                name = n.slice.value
        if name and name.isupper():
            out.add(name)
    return out


def check(pkg: pathlib.Path):
    launcher = (pkg / "test" / "live_run.sh").read_text()
    # Only the docker run -e list counts. A variable exported host-side but not listed here does
    # not cross into the container, which is the entire defect this checks for.
    passed = set(re.findall(r"-e ([A-Z_][A-Z0-9_]*)", launcher))
    # BOTH TREES. Scanning only perception_module is what let FOUND_CORPUS_ORDER and
    # FOUND_KG_ALIASES be recorded in the bundle and never passed: they are read in
    # /DATA/FOUND/found, one tree over. That is the SAME blind-spot shape this checker was written
    # for — it found the shell/python boundary and had a tree boundary of its own. The FOUND tree
    # is mounted at /found and its modules run inside the container exactly like these do.
    roots = [pkg / "src" / "perception_module"]
    found_tree = pathlib.Path("/DATA/FOUND/found")
    if found_tree.is_dir():
        roots.append(found_tree)
    reads = {}
    for root in roots:
        for py in sorted(root.glob("*.py")):
            if py.name.startswith(("test_", "check_")) or "rosstub" in py.name:
                continue
            for v in env_reads(py):
                reads.setdefault(v, set()).add(f"{root.name}/{py.name}")
    missing = {v: s for v, s in reads.items()
               if v not in passed and v not in SET_IN_CONTAINER and v not in HOST_ONLY}
    return reads, passed, missing


def main():
    pkg = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else ".")
    reads, passed, missing = check(pkg)
    if missing:
        print("!! read by container-side PYTHON but never passed by docker run -e:")
        for v in sorted(missing):
            print(f"   {v:28s} <- {', '.join(sorted(missing[v]))}")
        print("   Setting these host-side reaches nothing. Add them to the -e list, set them in")
        print("   live_stack_container.sh, or list them in SET_IN_CONTAINER / HOST_ONLY here with")
        print("   a reason. An unlisted one is a lever that looks connected and is not.")
        return 1
    print(f"env passthrough OK — {len(reads)} vars read by container-side python, "
          f"all passed, set in-container, or declared host-only")
    return 0


if __name__ == "__main__":
    sys.exit(main())
