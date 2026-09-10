#!/usr/bin/env python3
"""Serve the GRAPH-API bridge dashboard from a run directory on disk.

WHAT THIS IS, PLAINLY
---------------------
This is NOT the live ROS bridge. It is the same `graph_api_bridge` FastAPI app and the
same `viewer/viewer.html`, run on the host with the ROS layer stubbed out, reading a run
directory's JSON artefacts straight off disk. Everything file-backed works: the scene
graph, the objects table, the admission panel, room and BEV data, the latency table,
concepts with their IRIs.

Two things do NOT work, by construction, and the page says so:

  * the camera feed -- there is no camera behind an archived bundle, so `/feed`
    answers 503 immediately rather than holding a connection open forever;
  * the ROS-only health components -- perception and the object manager are reported
    from process liveness, so they read as stopped unless a real run is up.

WHAT IT READS
-------------
One directory, given by --bundle (or GRAPH_API_OUTPUT_DIR). Point it at:

  * an ARCHIVED run  -> <runs>/<stamp>_<scene>   -> a fixed picture of that run
  * the ACTIVE run   -> <runs>/latest            -> updates as the run writes

Both are honest uses. The page banner names the directory and the age of its newest
artefact, so a screenshot can never be mistaken for live data -- which matters, because
"official dashboard" and "replay of a past run" are very different claims and nobody
should have to infer which one they are looking at.

WHY IT EXISTS
-------------
Port 8081 (the live bridge) has been held by an unrelated project, and the bridge needs
a running container besides. This serves the same dashboard from artefacts alone, with
no container, no ROS and no simulator.

HOW TO START IT
---------------
    python3 found/dashboard/replay_server.py                       # runs/latest, port 8082
    python3 replay_server.py --bundle <runs>/20260831_184822_hm3d_00861
    python3 found/dashboard/replay_server.py --port 8090 --host 0.0.0.0

BINDING: the default is 127.0.0.1, deliberately. This dashboard exposes run artefacts,
file paths and a `/set_config` control surface, and it has no authentication of any kind.
Binding it to every interface is a decision to publish that, so it is opt-in via
--host 0.0.0.0 rather than the default. Nothing here needs to change for local use.
"""

import argparse
import importlib.util
import json
import os
import re
import shutil
import signal
import socket as _socket
import subprocess
import sys
import tempfile
import threading
import time
import types
from pathlib import Path

try:
    from found.dashboard import dash_env, dash_ext
except ImportError:
    import dash_env
    import dash_ext

# The graph-api tree sits at the root of this repository, and under vendor/ in an extension's
# checkout (owner 2026-09-09: this server runs here with no extension required). Deriving it from
# this file's own depth pinned it to one layout, so a copy anywhere else refused to start before it
# served a single route. Search upward for the bridge instead, in both layouts; GRAPH_API_ROOT
# overrides, the way the runs directory does below.
_BRIDGE_REL = Path("lost3dsg/src/perception_module/graph_api_bridge.py")


def _graph_api_root() -> Path:
    env = os.environ.get("GRAPH_API_ROOT")
    if env:
        return Path(env)
    # Owner ruling 2026-09-10: an extension depends on the CONSOLIDATED checkout, and this
    # resolver agrees with found/hooks.py and found/probes.py rather than preferring a vendored
    # copy that happens to sit nearer. A vendored tree is still accepted, second, so a checkout
    # that still carries the submodule keeps working while it is retired.
    consolidated = Path("/DATA/GRAPH-API")
    if (consolidated / _BRIDGE_REL).exists():
        return consolidated
    here = Path(__file__).resolve()
    for up in here.parents:
        for cand in (up / "vendor/graph-api", up):
            if (cand / _BRIDGE_REL).exists():
                return cand
    return here.parents[2] / "vendor/graph-api"      # for the error message only


GRAPH_API_ROOT = _graph_api_root()
BRIDGE = GRAPH_API_ROOT / _BRIDGE_REL
# dash_env, the same convention bundle_index.py, replay_view.py and scene3d.py already
# use with the same default. Hardcoding the laptop path left the bundle picker EMPTY and
# /load_bundle refusing every name in any deployment but this laptop (owner, 2026-09-08).
RUNS_ROOT = dash_env.runs_dir()
DEFAULT_BUNDLE = "latest"
DEFAULT_PORT = 8082
VIEW_RVIZ = GRAPH_API_ROOT / "lost3dsg/test/view_rviz.sh"
RVIZ_LOG = Path("/tmp/graphapi_live/rviz.log")
RVIZ_CONTAINER = "graphapi_rviz"   # the name view_rviz.sh gives its container


def resolve_bundle(spec):
    """Turn a --bundle value into a directory. "latest" means the NEWEST RUN.

    Deliberately not the runs/latest symlink: that symlink is updated when a run ends,
    not when one starts, so during a live run it points at the PREVIOUS run. Following
    it would serve last night's archive during tonight's run and label it green -- the
    exact misattribution the banner exists to prevent. Sort the run directories by name
    instead; the names are timestamps, so lexicographic order is chronological and does
    not depend on anyone maintaining a link.
    """
    if str(spec) != "latest":
        return Path(spec).resolve()
    if not RUNS_ROOT.is_dir():
        # A FRESH CHECKOUT HAS NO RUNS, and that is not an error: it is the normal state of the
        # repository this dashboard now ships in. Refusing to start meant a reader could not open
        # the dashboard at all until somebody had recorded a run, so the start page -- which exists
        # to say "no runs in <dir>" and let one be launched -- could never be reached.
        return None
    runs = sorted((d for d in RUNS_ROOT.iterdir() if d.is_dir() and not d.is_symlink()),
                  key=lambda d: d.name)
    if not runs:
        return None
    ready = [d for d in runs if _has_graph(d)]
    SKIPPED["newer"] = (runs[-1].name if ready and runs[-1] != ready[-1] else None)
    return (ready[-1] if ready else runs[-1]).resolve()


# The newest run directory, when it exists but is NOT the one being served because it has no
# graph yet. Reported by /mode_info so the page can SAY a run is in progress rather than leaving
# the reader to wonder why the newest run is not on screen.
SKIPPED = {"newer": None}


def _has_graph(d: Path) -> bool:
    """Does this run yet hold the file the graph is built from.

    A run directory is created when the run STARTS and fills up over the next twenty minutes;
    persistent_perception.json is written near the end. The follower used to adopt the newest
    directory the moment it appeared, so starting a run emptied the replay dashboard mid-session
    -- 0 nodes, 0 edges, no crops -- and it read as "the crops are missing again" rather than
    "you are looking at a run that has not produced a graph yet".

    Same predicate the bridge's own _active_output_dir uses, so the follower and the reader
    cannot disagree about which directories count.
    """
    return any((d / name).exists() for name in ("persistent_perception.json", "room.json"))


# Set when a bundle is chosen from the dashboard. The follower stands down while it is
# set, so a deliberately chosen bundle is not dragged back to the newest run 15 s later
# with nothing on screen explaining the jump.
BUNDLE_PIN = {"pinned": False}

# One definition of where the live bridge is, so a probe and a proxy cannot disagree
# about it -- they already did: /walls_view probed 8091 while the bridge served 8081.
BRIDGE_URL = os.environ.get("ROS_BRIDGE_HOST",
                            f"http://127.0.0.1:{os.environ.get('BRIDGE_PORT', '8081')}")

# Which dashboard this is. Read by the page through /mode_info, and by the injections,
# so every gate asks the same question once instead of each panel guessing.
# `by` says WHO last moved the mode -- "start" (the flag or the startup probe), "probe" (the
# follower) or "pick" (a person choosing a bundle in one tab). A live page ignores a "pick"
# reading: picking the RUNNING bundle leaves it unpinned, and without this every other live tab
# reloaded itself into replay mid-run (review 2026-09-08). Never absent: a reader that cannot
# tell "nobody has decided yet" from "the key is missing" is the rule-18 shape.
MODE = {"mode": "replay", "why": "not yet determined", "by": "default"}
# The replay-started app loads the bridge IN-PROCESS (build_app); that module probes FEED_HOST
# for frames, logs, auto_mode and actions. In REPLAY it must reach nothing live: measured
# 2026-09-07 (bughunt V-1) a replay of run H served the LIVE simulator's frame and said the feed
# was streaming while run 152446 walked. So its FEED_HOST is unroutable while MODE is replay and
# the real host only while MODE is live (the follower swaps it). Rule 11: a replay that cannot
# reach a live host fails visibly; one that can, controls a run nobody meant to touch.
_INPROC = {"bridge": None, "feed_host": None}
_UNROUTABLE_FEED_HOST = "http://127.0.0.1:1"


def _sync_inproc_feed_host():
    m = _INPROC["bridge"]
    if m is not None and _INPROC["feed_host"] is not None:
        m.FEED_HOST = _INPROC["feed_host"] if MODE["mode"] == "live" else _UNROUTABLE_FEED_HOST


def _probe_bridge(url: str = None, timeout: float = 2.0):
    """(reachable_and_ours, why). A 200 is NOT enough.

    Port 8081 has been held by an unrelated project's dashboard that answers 200 with
    {"ok": true}. Detecting "live" from reachability alone would point this dashboard at
    a foreign server and render its JSON as ours -- and that is not hypothetical, it
    already reported `bridge: active` for the squatter once. Require the payload to look
    like OUR bridge's health.
    """
    import urllib.error
    import urllib.request
    url = url or BRIDGE_URL
    try:
        with urllib.request.urlopen(f"{url}/health", timeout=timeout) as r:
            if r.status != 200:
                return False, f"{url}/health returned HTTP {r.status}"
            payload = json.loads(r.read().decode())
    except (urllib.error.URLError, OSError, json.JSONDecodeError, TimeoutError) as exc:
        return False, f"{url} unreachable: {type(exc).__name__}: {exc}"
    comps = (payload or {}).get("components") or {}
    if isinstance(comps, dict) and {"feed", "bridge"} <= set(comps):
        return True, f"{url} answered with a GRAPH-API health payload"
    return False, (f"{url} answered, but it is not the GRAPH-API bridge "
                   f"(keys: {sorted(comps) if isinstance(comps, dict) else type(comps).__name__})")



_UVICORN_BIND = re.compile(r"Uvicorn running on https?://[\w.]+:(\d+)")


def discovered_bridge_port():
    """The port the ACTIVE run's bridge actually bound, read from the bridge's own first log
    lines -- or None.

    2026-09-06, run 20260906_223701: the bridge came up on :8085 (BRIDGE_PORT set to dodge an
    unrelated dashboard squatting :8081) and served the whole run. This dashboard probed
    :8081 only, was refused, and reported replay mode over a live stack. The run records
    where its bridge listens in logs/bridge.log ("Uvicorn running on http://0.0.0.0:8085");
    that line is evidence, not a guess, so it is read here. run_metadata.json does not carry
    the port (asked of the simulator lane); when it does, prefer it.
    """
    try:
        run = resolve_bundle("latest")
    except OSError:
        return None
    if run is None:
        # No runs dir, or an empty one. Either way there is no bridge.log to read. (It used to
        # RAISE SystemExit here, which `except Exception` never caught, so it went through
        # _follow_mode's handler and killed the follower thread.)
        return None
    for log in (run / "logs" / "bridge.log", run / "bridge.log"):
        try:
            with open(log, "r", encoding="utf-8", errors="replace") as f:
                head = f.read(4096)
        except OSError:
            continue
        hit = _UVICORN_BIND.search(head)
        if hit:
            return int(hit.group(1))
    return None


def bridge_identified(url: str = None, timeout: float = 2.0):
    """(reachable_and_ours, why). The configured URL first; if that is refused, the port the
    active run's bridge says it bound. On success the discovered URL becomes BRIDGE_URL for
    every proxy in this process, and `why` names the evidence."""
    global BRIDGE_URL
    url = url or BRIDGE_URL
    ok, why = _probe_bridge(url, timeout)
    if ok:
        return ok, why
    port = discovered_bridge_port()
    if port is None or f":{port}" in url:
        return ok, why
    alt = f"http://127.0.0.1:{port}"
    ok2, why2 = _probe_bridge(alt, timeout)
    if ok2:
        BRIDGE_URL = alt
        return True, (f"{url} refused, but the run's bridge.log says the bridge bound :{port} "
                      f"and {alt} identifies as ours -- using it")
    return False, f"{why}; bridge.log says the bridge bound :{port} but {alt} also failed: {why2}"

def _follow_mode(interval: float = 15.0):
    """Re-probe the bridge every `interval` s and move MODE with it.

    MODE was decided once, at startup: a dashboard started before a run stayed in replay for the
    whole run ("live dashboard not served", owner 2026-09-06), and one started while a bridge was
    answering stayed live after the stack went down. The per-request paths (/dash injection,
    /mode_info, the tools menu) already read MODE each time, so moving it is enough for them.
    ponytail: the replay-only routes build_app installs (/feed stand-in, /load_bundle, the
    replay/* readers) are still installed only when the process STARTED in replay; a live-started
    process that drops to replay serves the page but not those. Named ceiling; the upgrade is to
    install every route in both modes and decide per request, as /scene3d and now /arch do.
    """
    import threading

    def loop():
        misses = 0
        while True:
            time.sleep(interval)
            if BUNDLE_PIN.get("pinned"):
                continue                      # a chosen bundle IS replay; do not fight the choice
            try:
                ok, why = bridge_identified(timeout=5.0)
            except Exception as exc:          # noqa: BLE001 - a probe must never kill the follower
                ok, why = False, f"probe raised {type(exc).__name__}: {exc}"
            # Measured 2026-09-07 on run 152446: with a 1.5 s probe the bridge timed out under a
            # 23 s perception cycle and MODE flapped live <-> replay every 15 s; a page served
            # during a replay flap carries the poller blocker and never refreshes again (the
            # owner's "Metrics has no live refresh"). One slow answer is not a dead bridge:
            # drop to replay only after two consecutive misses. Rule 11: a stale "live" shows a
            # visible "bridge unreachable" on the page; a false "replay" freezes it silently.
            misses = 0 if ok else misses + 1
            want = "live" if ok else ("replay" if misses >= 2 else MODE["mode"])
            if want != MODE["mode"]:
                print(f"[dash] mode {MODE['mode']} -> {want} because {why}", flush=True)
                MODE.update(mode=want, why=why, by="probe")
                _sync_inproc_feed_host()
            else:
                # STAMPED ON EVERY PROBE, not only on a transition. Written only on a change, the
                # reason went STICKY and inverted the fault it was added for: after someone picked
                # a bundle the reason stayed "pick" forever, so when the run genuinely ended the
                # probes failed, the mode was ALREADY replay, nothing was written, and every live
                # tab ignored every reading and sat showing LIVE over a dead stack (review
                # 2026-09-08, third read). The mode is unchanged here; only who last looked, and
                # what they saw, is refreshed.
                MODE.update(why=why, by="probe")

    threading.Thread(target=loop, name="mode-follower", daemon=True).start()


def _follow_latest(interval: float = 15.0):
    """Keep GRAPH_API_OUTPUT_DIR on the newest run.

    The bridge reads that variable on every call, so re-pointing it is enough for every
    endpoint to follow -- no restart, and the banner re-reads the directory on each page
    load, so it renames itself when a new run starts.
    """
    current = {"dir": None}
    skipped_said = {"name": None}

    def loop():
        while True:
            try:
                if BUNDLE_PIN["pinned"]:
                    time.sleep(interval)
                    continue
                newest = resolve_bundle("latest")
                if newest is not None and newest != current["dir"]:
                    current["dir"] = newest
                    os.environ["GRAPH_API_OUTPUT_DIR"] = str(newest)
                    print(f"[replay] following {newest}", flush=True)
                if SKIPPED["newer"] and SKIPPED["newer"] != skipped_said["name"]:
                    skipped_said["name"] = SKIPPED["newer"]
                    print(f"[replay] {SKIPPED['newer']} is newer but has no graph yet; "
                          f"staying on {newest.name}", flush=True)
            except Exception as exc:            # never let the follower kill the server
                print(f"[replay] follow failed: {type(exc).__name__}: {exc}", flush=True)
            time.sleep(interval)

    threading.Thread(target=loop, daemon=True).start()


def _install_ros_stubs():
    """Let `graph_api_bridge` import on a host with no ROS.

    Only the imports are replaced. Every endpoint this dashboard serves reads JSON,
    SQLite and image files; none of them calls into ROS or OpenCV. The one part that
    genuinely needs ROS -- the live camera subscription -- is the part disabled below.
    """
    def mod(name, **attrs):
        m = types.ModuleType(name)
        for k, v in attrs.items():
            setattr(m, k, v)
        sys.modules[name] = m
        return m

    class _Stub:
        def __init__(self, *a, **k):
            pass

        def __getattr__(self, _):
            return lambda *a, **k: None

    mod("rclpy", init=lambda *a, **k: None, spin=lambda *a, **k: None,
        shutdown=lambda *a, **k: None, ok=lambda: True)
    mod("rclpy.node", Node=_Stub)
    sys.modules["rclpy"].node = sys.modules["rclpy.node"]
    mod("cv2", imencode=lambda *a, **k: (False, None), imdecode=lambda *a, **k: None,
        cvtColor=lambda *a, **k: None, COLOR_BGR2RGB=4)
    mod("cv_bridge", CvBridge=_Stub)
    for pkg, names in (("sensor_msgs", ("Image", "CompressedImage", "CameraInfo")),
                       ("std_msgs", ("String", "Bool", "Float32")),
                       ("geometry_msgs", ("Twist", "PoseStamped", "Point"))):
        mod(pkg, msg=None)
        mod(f"{pkg}.msg", **{n: object for n in names})
        sys.modules[pkg].msg = sys.modules[f"{pkg}.msg"]
    mod("lost3dsg", srv=None, msg=None)
    mod("lost3dsg.srv", **{n: object for n in (
        "AddObject", "DeleteObjects", "MergeObjects", "QueryObjects",
        "RemoveObject", "UpdateObject")})
    mod("lost3dsg.msg", Bbox3d=object, ObjectInfo=object)
    sys.modules["lost3dsg"].srv = sys.modules["lost3dsg.srv"]
    sys.modules["lost3dsg"].msg = sys.modules["lost3dsg.msg"]


def _load_scene3d():
    """The 3D scene module. Imported lazily and NOT cached here: `page()` reads the bundle
    off disk on every call, and this indirection is what lets the module be reloaded."""
    try:
        from . import scene3d
    except ImportError:
        import scene3d
    return scene3d


def _load_bundle_index():
    try:
        from found.dashboard import bundle_index
    except ModuleNotFoundError:
        import bundle_index
    return bundle_index


def _load_bridge():
    spec = importlib.util.spec_from_file_location("graph_api_bridge_replay", BRIDGE)
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(BRIDGE.parent))
    spec.loader.exec_module(module)
    return module


def _replay_head_html() -> str:
    """Installed BEFORE the viewer's own script. GA-228.

    WHY THE ORDER MATTERS, and the first attempt got it wrong. The viewer registers its
    pollers as `setInterval(updateHealthStatus, 2000)` -- it passes the FUNCTION OBJECT.
    Reassigning `window.updateHealthStatus` afterwards changes nothing, because the timer
    holds the original reference; and sweeping `clearInterval` from a script at the end of
    the body misses `setInterval(refreshGraph, 3000)` at viewer.html:3237, which had not been
    registered yet. Both mistakes were made and both left the page still polling.

    So the registration itself is intercepted, before any of it runs. A poller that never
    starts cannot be stopped incorrectly.

    WHAT IS BLOCKED, and why each one:
      updateHealthStatus   reds the page because feed and perception are offline. In an
                           archive they are supposed to be. An always-red badge says nothing.
      checkFeedHeartbeat   ages the newest frame against the wall clock. Every archived frame
                           is old, so this reports STALLED by construction.
      refreshGraph         re-fetches /graph_data every 3 s. The graph cannot change: the run
                           ended. This is the flicker.
      updateBEV            polls a static file twice a second. Called ONCE below instead.
      updateLogs           tails the log. Replaced by a frame-synchronised view, so the log
                           follows the frame you are looking at rather than the clock.
    """
    return """
<script>
(function () {
  window.REPLAY_MODE = true;
  const BLOCK = new Set(['updateHealthStatus', 'checkFeedHeartbeat', 'refreshGraph',
                         'updateBEV', 'updateLogs']);
  // NOT blocked: refreshCropBackgrounds. A replay page is exactly where it is needed -- the graph
  // version never changes there, so applyCropBackgrounds runs once and every crop that failed on
  // first load stays missing for the life of the page (its own comment records that defect).
  // Blocking it in rev3 re-introduced the bug the ticker was written to fix; the detached guard
  // inside it already covers the live-scrub case, which was the real need.
  // The blocked pollers cannot be restarted in place (the timers were never registered), so
  // a page served in replay that the server has since moved to live is reloaded, once, and
  // says so. Without this a viewer opened before a run stayed frozen for the whole run.
  window.setInterval(() => {
    fetch('/mode_info').then(r => r.json()).then(m => {
      if (m && m.mode === 'live' && !m.pinned) {
        console.warn('[replay] server is live now; reloading the page into live mode');
        location.reload();
      }
    }).catch(() => {});
  }, 15000);
  const realSetInterval = window.setInterval.bind(window);
  window.__replayBlocked = [];
  window.setInterval = function (fn, ms, ...rest) {
    const name = (typeof fn === 'function' && fn.name) || '';
    if (BLOCK.has(name)) {
      // Recorded rather than silently dropped: a page that quietly disables things is how
      // you end up debugging a feature that was never running.
      window.__replayBlocked.push(name + '@' + ms + 'ms');
      return 0;
    }
    return realSetInterval(fn, ms, ...rest);
  };
})();
</script>
"""


def _replay_mode_html(bundle: Path) -> str:
    """The replay UI, injected at the END of the body. GA-224.

    Replaces the agent D-pad -- which posts to /action to drive an agent whose run ended --
    with a transport bar under the feed, and draws the detector's own output over the frame.

    THE OVERLAY IS DRAWN CLIENT-SIDE, and it has to be. In a live run `/image_with_bb` arrives
    already annotated by the perception node. An archive stores the RAW frame plus
    detections.jsonl, so the boxes must be re-drawn here from the recorded coordinates. That
    is why the LAYERS bar did nothing in replay: it toggles a server-rendered overlay that
    does not exist. Each layer is now wired to something this page can actually draw.
    """
    import json as _json
    name = bundle.name
    return f"""
<style>
  /* GA-345: ONE bar in BOTH modes. `body.dash-scrub` = the page shows a RECORDED frame (all of
     replay mode, or a live page whose reader scrubbed away from the head). Only then are the
     live-only controls hidden; while the page follows the live head they stay. */
  body.dash-scrub #feedOfflineOverlay, body.dash-scrub #feedStatusBadge,
  body.dash-scrub .teleop-overlay {{ display: none !important; }}
  /* Liveness, hidden in replay. These age the newest artefact against the WALL CLOCK, so an
     archive reads "PERCEPTION 99+s / GRAPH 99+s" forever and the phase badge freezes on
     whatever the run was doing when it stopped. A permanently-red indicator carries no
     information; worse, it invites the reader to diagnose a fault that is just the date. */
  body.dash-scrub #perceptionStatusBadge, body.dash-scrub #cycleStamp {{ display: none !important; }}
  /* Following the head, the bridge's stream already carries its own overlay: the recorded-box
     canvas is hidden so the two cannot stack, and clicks reach the d-pad under it. */
  body:not(.dash-scrub) #replayCanvas {{ display: none !important; }}
  #rLive {{ color:#fca5a5; border-color:rgba(239,68,68,.6) !important; }}
  #rLive.on {{ background:rgba(127,29,29,.85) !important; color:#fff; }}
  #rLive.on::before {{ content:'\\25cf '; color:#ef4444; }}
  /* The D-pad reserved 324px on the right of the feed. With it gone that is dead space --
     visible in the screenshot as a black column beside the image. Give it to the frame. */
  #feedWrapper {{ padding-right: 8px !important; padding-bottom: 44px !important; }}
  #replayBar {{
    position:absolute; left:8px; right:8px; bottom:4px; z-index:35;
    display:flex; align-items:center; gap:8px; padding:5px 9px; border-radius:7px;
    background:rgba(15,23,42,.92); border:1px solid rgba(56,189,248,.30);
    font:700 10px ui-monospace,monospace; color:#cbd5e1;
  }}
  #replayBar button {{
    padding:4px 10px; cursor:pointer; border-radius:5px; white-space:nowrap;
    background:rgba(30,41,59,.9); color:#e2e8f0; border:1px solid rgba(255,255,255,.16);
    font:700 10px ui-monospace,monospace;
  }}
  #replayBar button:hover {{ background:rgba(56,189,248,.25); border-color:#38bdf8; }}
  /* Icon buttons. The glyph is an inline SVG painted with currentColor, so recolouring the
     button recolours the icon -- which is how the arrows take the armed event's colour. */
  #replayBar button.icn {{ padding:4px 7px; line-height:0; }}
  #replayBar button.icn svg {{ display:block; }}
  #replayBar .track {{ flex:1; min-width:160px; position:relative; height:30px; }}
  #replayBar input[type=range] {{
    width:100%; position:absolute; left:0; bottom:0; margin:0; accent-color:#38bdf8;
  }}
  #replayEvents {{ position:absolute; left:0; top:0; width:100%; height:15px; display:block; }}
  #replayBar .pos {{ color:#38bdf8; min-width:190px; text-align:right; }}
  #replayLegend {{ display:flex; gap:9px; align-items:center; color:#64748b; font-size:9px; }}
  #replayLegend i {{ width:8px; height:8px; border-radius:2px; display:inline-block; margin-right:3px; }}
  /* The left column held a 298x223 minimap with dead space under it. The BEV is the only
     spatial view of the run; give it the column. */
  .bev-hud-overlay, #bevHudCard {{ width: 316px !important; }}
  #bevCanvasHUD {{ width: 300px !important; height: 460px !important; }}
  /* pointer-events:auto, deliberately: the boxes are the only handle the feed has on an
     object, so they must be clickable. Nothing underneath the canvas wants the click --
     the frame is an <img> and the teleop overlay is hidden above. */
  #replayCanvas {{ position:absolute; pointer-events:auto; z-index:20; }}
  #replayBadge {{
    position:absolute; top:6px; right:14px; z-index:40; padding:3px 8px; border-radius:5px;
    background:rgba(15,23,42,.88); border:1px solid rgba(56,189,248,.45); color:#38bdf8;
    font:700 9px ui-monospace,monospace; letter-spacing:.08em;
  }}
</style>
<script>
(function () {{
  const BUNDLE = {_json.dumps(name)};
  const $ = id => document.getElementById(id);
  const feed = $('habitatFeedImg'), wrap = $('feedWrapper');
  // GA-345. LIVE_MODE: this page was served while a bridge answers (window.DASH_MODE), so the
  // bundle on disk is GROWING and the bridge streams its head. `following` = the page shows the
  // live head (stream + the viewer's pollers); false = it shows a RECORDED frame (a scrub on a
  // live page, or all of replay mode). Rule 4: the two states carry two names on the badge and
  // on the LIVE button, and a recorded frame is never shown under the LIVE label.
  // The body class is set BEFORE the early return below: the CSS that hides the live-only
  // badges on an archive must not depend on the feed wrapper existing.
  // GA-380: the app can be served under a path prefix (a deployment may mount it under one). The
  // proxy's rewrite only sees literal `src="/` and `fetch('/`, so every URL the SCRIPT builds is
  // invisible to it and resolves against the site root -- measured as a 404 on the replay frame.
  // One helper, computed once, used at every such site: three inline copies is how a fourth site
  // gets missed. Trailing slashes are stripped: at the start page the path is `/` or `/<prefix>/`,
  // and without that the result was `//dash`, which a browser reads as a HOST, not a path.
  // ponytail: drop this when the server grows real root_path support -- that set is next.
  const PFX = location.pathname
    .replace(new RegExp('/(dash|replay|scene3d|bundles|arch|blockers)/?$'), '')
    .replace(new RegExp('/+$'), '');
  const BRAND = window.DASH_BRAND || 'Scene graph';
  const LIVE_MODE = window.DASH_MODE === 'live';
  let following = LIVE_MODE;
  document.body.classList.toggle('dash-scrub', !following);
  if (!wrap) return;

  let F = [], i = 0, timer = null, speed = 1;
  // `perm` now means ADMITTED and `temp` means NOT YET JUDGED -- see the verdict block below.
  // The three grade layers are off by default: 128 of this run's 143 graded objects are
  // decline or no_grounds, and drawing them all is the frame under a mesh of red.
  // GA-241. EVERY drawable thing is a layer. `box3d` was drawn unconditionally with no
  // button, so switching every visible layer off still left a screen full of cages -- the
  // owner's screenshot. A control surface that cannot reach a drawn thing is worse than none.
  // `pca` selects the ORIENTED box the pipeline reasoned about vs the axis-aligned hull.
  // `belief` is the world model's settled box per object (end of run); `box3d` is the
  // per-frame DETECTION, drawn as a thin wireframe so the two cannot be confused. Owner,
  // 2026-09-06: the air conditioner "perfect in one frame, small and unrotated in the next"
  // was the detection changing with a clipped view, not a rendering bug -- the belief box is
  // the one a reader expects to stay put, so it is the primary layer.
  // Owner, second look 2026-09-06: with both layers on, the oriented belief box and the AABB
  // detection stacked on every object ("PCA over AABB, not a toggle"), and the belief layer drew
  // the WHOLE store, so boxes showed through walls. DETECTION is therefore off by default, and a
  // belief box is drawn only where THIS frame's detections put something (see beliefFor).
  // `box2d` = the per-detection 2D rectangles (dashed unless admitted); `det` = the label text.
  // They were one thing: the owner had no way to hide the dotted rectangles and keep the labels.
  const layers = {{ perm: true, temp: true, hold: false, decline: false, no_grounds: false, walls: false,
                    seg: false, det: true, belief: true, box3d: false, pca: true, box2d: true,
                    time: true }};   // GA-361: graph + table show only objects that existed at the viewed frame
  let masks = null, masksFor = null;
  let POSE = {{}}, BOX3 = {{}}, BELIEF = [];
  let VERDICT = {{}};                      // instance_label -> admit|hold|decline|no_grounds
  let selected = null;                     // the object highlighted across all three panels
  let evFilter = null;                     // which event colour the arrows step through

  // window.cy is assigned by the VIEWER's own script at viewer.html:2504, which runs AFTER
  // this block -- it is injected at the end of the body but the viewer's script tag is
  // earlier in the document and its assignment happens later, inside its init. So every
  // `if (window.cy)` guard written here tested `undefined` and silently no-opped: that is
  // the single reason node selection, the graph framing and the highlight all did nothing.
  // Queue the work and run it when cytoscape actually exists.
  const cyQueue = [];
  function whenCy(fn) {{
    if (window.cy) {{ try {{ fn(window.cy); }} catch (e) {{}} return; }}
    cyQueue.push(fn);
  }}
  const cyWait = setInterval(() => {{
    if (!window.cy) return;
    clearInterval(cyWait);
    while (cyQueue.length) {{ try {{ cyQueue.shift()(window.cy); }} catch (e) {{}} }}
  }}, 120);

  // Monochrome inline SVG, sized in the element and painted with currentColor. Characters
  // were used before (&larr;, PLAY); they inherit the page font, so they rendered at a
  // different weight per platform and could not be recoloured independently of the label.
  const ICON = {{
    play: '<svg viewBox="0 0 16 16" width="11" height="11"><path fill="currentColor" d="M4 2.5v11l9-5.5z"/></svg>',
    pause: '<svg viewBox="0 0 16 16" width="11" height="11"><path fill="currentColor" d="M4 2.5h3v11H4zm5 0h3v11H9z"/></svg>',
    stop: '<svg viewBox="0 0 16 16" width="11" height="11"><path fill="currentColor" d="M3.5 3.5h9v9h-9z"/></svg>',
    prev: '<svg viewBox="0 0 16 16" width="11" height="11"><path fill="currentColor" d="M4 2.5h2v11H4zm9 0v11L6 8z"/></svg>',
    next: '<svg viewBox="0 0 16 16" width="11" height="11"><path fill="currentColor" d="M10 2.5h2v11h-2zM3 2.5v11L10 8z"/></svg>'
  }};

  // ---- badge + canvas ----------------------------------------------------------------
  const badge = document.createElement('div');
  badge.id = 'replayBadge';
  badge.textContent = (LIVE_MODE ? 'LIVE' : 'REPLAY') + ' \u00b7 ' + BUNDLE;
  wrap.appendChild(badge);

  const cv = document.createElement('canvas');
  cv.id = 'replayCanvas';
  wrap.appendChild(cv);
  const ctx = cv.getContext('2d');

  // ---- transport ---------------------------------------------------------------------
  const bar = document.createElement('div');
  bar.id = 'replayBar';
  bar.innerHTML =
    '<button class="icn" id="rPlay" title="play / pause">' + ICON.play + '</button>' +
    '<button class="icn" id="rStop" title="stop and rewind">' + ICON.stop + '</button>' +
    '<button class="icn" id="rPrev" title="previous frame">' + ICON.prev + '</button>' +
    '<button class="icn" id="rNext" title="next frame">' + ICON.next + '</button>' +
    '<button id="rSpeed">1x</button>' +
    '<div class="track"><canvas id="replayEvents"></canvas>' +
    '<input type="range" id="rSeek" min="0" max="0" value="0"></div>' +
    '<span id="replayLegend">' +
    '<span data-kind="admit"><i style="background:#22c55e"></i>admit</span>' +
    '<span data-kind="hold"><i style="background:#eab308"></i>hold</span>' +
    '<span data-kind="decline"><i style="background:#ef4444"></i>decline</span>' +
    '<span data-kind="no_grounds"><i style="background:#64748b"></i>no grounds</span>' +
    '<span data-kind="merge"><i style="background:#38bdf8"></i>merge</span></span>' +
    '<button id="rLive" title="follow the live head (YouTube-livestream style); any scrub detaches">LIVE</button>' +
    '<span class="pos" id="rPos">loading frames</span>';
  wrap.parentNode.appendChild(bar);
  if (!LIVE_MODE) $('rLive').style.display = 'none';   // an archive has no head to follow

  // ---- overlay ------------------------------------------------------------------------
  // object-fit: contain letterboxes the frame inside the <img>, so the drawn rectangle is
  // NOT the element rectangle. Computing it is the difference between boxes that sit on
  // their objects and boxes that sit near them -- and "near" looks like a calibration bug.
  function fitCanvas(nat_w, nat_h) {{
    const r = feed.getBoundingClientRect(), p = wrap.getBoundingClientRect();
    const scale = Math.min(r.width / nat_w, r.height / nat_h);
    const w = nat_w * scale, h = nat_h * scale;
    cv.width = Math.round(w); cv.height = Math.round(h);
    cv.style.left = Math.round(r.left - p.left + (r.width - w) / 2) + 'px';
    cv.style.top = Math.round(r.top - p.top + (r.height - h) / 2) + 'px';
    cv.style.width = Math.round(w) + 'px'; cv.style.height = Math.round(h) + 'px';
    return scale;
  }}

  function colourFor(label) {{
    let h = 0;
    for (let k = 0; k < String(label).length; k++) h = (h * 31 + String(label).charCodeAt(k)) % 360;
    return h;
  }}

  function drawOverlay() {{
    if (following || !F.length || !feed.naturalWidth) return;   // the stream carries its own overlay
    const f = F[i];
    const s = fitCanvas(feed.naturalWidth, feed.naturalHeight);
    ctx.clearRect(0, 0, cv.width, cv.height);

    if (layers.seg && masks && masksFor === f.id) {{
      for (const m of masks) {{
        const hue = colourFor(m.label);
        paintRLE(m.rle, s, 'hsla(' + hue + ',85%,55%,0.34)');
      }}
    }}
    // 3D boxes. The rotation is SOLVED per frame (Wahba/SVD) from this frame's own
    // detections, and the solve reports its angular residual -- a frame that did not fit is
    // NOT drawn, because a confident wrong box over a photo is worse than no box.
    if (layers.box3d || layers.belief) {{
      const P = POSE[f.id], B = BOX3[f.id] || [];
      if (P && P.rms_deg <= 5.0) {{
        const R = P.R, C = P.C;
        const proj = q => {{
          const d = [q[0]-C[0], q[1]-C[1], q[2]-C[2]];
          const z = R[6]*d[0] + R[7]*d[1] + R[8]*d[2];
          if (z <= 0.05) return null;      // behind the camera; a pinhole would wrap it
          return [(R[0]*d[0]+R[1]*d[1]+R[2]*d[2]) / z * P.fx + P.cx,
                  (R[3]*d[0]+R[4]*d[1]+R[5]*d[2]) / z * P.fy + P.cy];
        }};
        // SHADED FACES, PAINTER'S ALGORITHM. Forty wireframe cages are a mesh of crossing
        // lines with no depth cue -- nothing says which cage a line belongs to, which is the
        // owner's "difficult to see with many intersecting boxes". Translucent faces drawn
        // far-to-near give occlusion, the cue the eye actually uses. Same approach /scene3d
        // already uses for the world view.
        const camZ = q => {{
          const d = [q[0]-C[0], q[1]-C[1], q[2]-C[2]];
          return R[6]*d[0] + R[7]*d[1] + R[8]*d[2];
        }};
        // WALLS. Each segment is (p0, p1, z_min, z_max) in the MAP frame -- a vertical
        // quad, drawn under the belief boxes so it reads as the surface they sit against.
        if (layers.walls && WALLS.length) {{
          ctx.save();
          for (const w of WALLS) {{
            const c = [[w[0][0], w[0][1], w[2]], [w[1][0], w[1][1], w[2]],
                       [w[1][0], w[1][1], w[3]], [w[0][0], w[0][1], w[3]]].map(proj);
            if (c.some(q => q === null)) continue;
            ctx.beginPath();
            ctx.moveTo(c[0][0], c[0][1]);
            for (let i = 1; i < 4; i++) ctx.lineTo(c[i][0], c[i][1]);
            ctx.closePath();
            ctx.fillStyle = 'rgba(56,189,248,0.16)';
            ctx.strokeStyle = 'rgba(56,189,248,0.75)';
            ctx.lineWidth = 1.5;
            ctx.fill(); ctx.stroke();
          }}
          ctx.restore();
        }}
        const F6 = [[0,1,2,3],[4,5,6,7],[0,1,5,4],[1,2,6,5],[2,3,7,6],[3,0,4,7]];
        const E12 = [[0,1],[1,2],[2,3],[3,0],[4,5],[5,6],[6,7],[7,4],[0,4],[1,5],[2,6],[3,7]];
        // `pca` off falls back to the axis-aligned hull, so the two can be compared. The
        // oriented box is what the envelopes were checked against; the hull is wider
        // wherever the object is not axis-aligned, and seeing that gap is the point.
        const cornersOf = b => (layers.pca || !b.aabb) ? b.c : b.aabb;
        // BELIEF, BUT ONLY WHERE THIS FRAME SAW SOMETHING. The store holds every object of the
        // run; projected wholesale it drew boxes through walls, because a pinhole has no
        // occlusion. There is no depth image in a bundle to test against, so visibility is
        // taken from the one thing the frame does record: its detections. A belief box is drawn
        // when its centre lies within MATCH_M of a detection centre in this frame -- the same
        // spatial join the association channel makes -- and that belief box then REPLACES the
        // detection's own (clipped, partial) box, which is what the owner asked to see. A
        // detection with no belief within reach keeps its wireframe, so nothing seen vanishes.
        // ponytail: nearest-centre within 0.6 m, no box overlap test; a small object next to a
        // large one can borrow the wrong box. Ceiling named; upgrade path is IoU of the AABBs.
        const MATCH_M = 0.6;
        const centre = c => [0,1,2].map(k => c.reduce((a, p) => a + p[k], 0) / c.length);
        const detC = B.filter(b => !b.degenerate).map(b => ({{b, c: centre(b.c)}}));
        // ONE belief box per detection, the nearest, each belief used once. Matching the
        // other way round drew 33 boxes for 11 detections on frame 1788727450: the store holds
        // near-duplicates (7 "air conditioner" objects within reach of one detection) and every
        // one of them claimed the same detection. The duplicates are a backend fact (GA-314's
        // merge chain); the picture shows one box where one thing was seen, and the panel's
        // count says how many the store holds.
        const used = new Set(), taken = new Set();
        const beliefFor = [];
        if (layers.belief) {{
          const cands = BELIEF.filter(bb => !bb.degenerate).map(bb => ({{bb, c: centre(bb.c)}}));
          for (const d of detC) {{
            let best = null, bd = MATCH_M;
            for (const k of cands) {{ if (taken.has(k.bb)) continue;
              const dd = Math.hypot(k.c[0]-d.c[0], k.c[1]-d.c[1], k.c[2]-d.c[2]); if (dd < bd) {{ bd = dd; best = k; }} }}
            if (best) {{ beliefFor.push(best.bb); taken.add(best.bb); used.add(d.b); }}
          }}
        }}
        window.REPLAY_BELIEF_DRAWN = beliefFor.length;
        // A degenerate box (an extent under a centimetre) is never drawn as a box; it is
        // counted, and the count is on the page.
        const quads = []; let degenerateSkipped = BELIEF.filter(b => b.degenerate).length;
        for (const b of beliefFor) {{
          const corners = cornersOf(b), pts = corners.map(proj);
          if (pts.some(q => q === null)) continue;
          const hue = colourFor(b.label);
          const dim = selected && b.label !== selected;
          for (const f of F6) {{
            quads.push({{pts: f.map(k => pts[k]), hue: hue, dim: dim,
                         depth: (camZ(corners[f[0]]) + camZ(corners[f[1]])
                               + camZ(corners[f[2]]) + camZ(corners[f[3]])) / 4}});
          }}
        }}
        // DETECTIONS: a thin wireframe per box, dashed when the detection touched the frame
        // edge -- an estimate from a partial view, the shape of GA-315 -- and skipped when
        // the box is degenerate. With the layer OFF, only detections that found no belief box
        // are drawn, so what the frame saw is never silently dropped; with it ON, all of them.
        const wires = [];
        for (const b of B) {{
          if (b.degenerate) {{ degenerateSkipped++; continue; }}
          if (!layers.box3d && (used.has(b) || !layers.belief)) continue;
          const corners = cornersOf(b), pts = corners.map(proj);
          if (pts.some(q => q === null)) continue;
          wires.push({{pts, hue: colourFor(b.label), dim: selected && b.label !== selected,
                       partial: !!b.partial}});
        }}
        window.REPLAY_DEGENERATE_SKIPPED = degenerateSkipped;
        quads.sort((u, v) => v.depth - u.depth);   // far first, so near boxes occlude
        ctx.setLineDash([]);
        for (const q of quads) {{
          // A selection pushes the rest far back rather than hiding them, so the chosen box
          // keeps a context to sit in.
          const fa = q.dim ? 0.04 : 0.15, ea = q.dim ? 0.10 : 0.92;
          ctx.beginPath();
          ctx.moveTo(q.pts[0][0]*s, q.pts[0][1]*s);
          for (let k = 1; k < 4; k++) ctx.lineTo(q.pts[k][0]*s, q.pts[k][1]*s);
          ctx.closePath();
          ctx.fillStyle = 'hsla(' + q.hue + ',85%,58%,' + fa + ')';
          ctx.fill();
          ctx.strokeStyle = 'hsla(' + q.hue + ',95%,72%,' + ea + ')';
          ctx.lineWidth = q.dim ? 0.7 : 1.3;
          ctx.stroke();
        }}
        for (const w of wires) {{
          ctx.strokeStyle = 'hsla(' + w.hue + ',95%,72%,' + (w.dim ? 0.12 : 0.85) + ')';
          ctx.lineWidth = w.dim ? 0.6 : 1.0;
          ctx.setLineDash(w.partial ? [4, 3] : []);
          ctx.beginPath();
          for (const [i, j] of E12) {{
            ctx.moveTo(w.pts[i][0]*s, w.pts[i][1]*s); ctx.lineTo(w.pts[j][0]*s, w.pts[j][1]*s);
          }}
          ctx.stroke();
        }}
        ctx.setLineDash([]);
      }}
    }}
    // THE LAYERS ARE THE ADMISSION VERDICT, not ground truth. The previous test was
    // `d.gt != null` -- `habitat_gt_instance_id`, which is null on EVERY row of a run
    // captured without a semantic frame, so PERM and TEMP selected the same 1,530 boxes and
    // the two buttons looked identical because they were. The distinction the page means is
    // whether the Bookkeeper ADMITTED the object, and that is recorded per object.
    // MEASURED on this bundle: 143 graded objects -- 15 admit, 15 hold, 25 decline,
    // 88 no_grounds -- joining to 143 of the 169 distinct detection labels. The other 26
    // were never put to a verdict, and those are the honest TEMP.
    if (drawnLayers()) {{
      ctx.lineWidth = 2; ctx.font = '600 12px ui-monospace,monospace';
      for (const d of f.detections) {{
        if (!d.bbox || d.bbox.length < 4) continue;
        const grade = VERDICT[d.label] || null;
        if (!layers[LAYER_OF[grade] || 'temp']) continue;
        // A grade gets the scrubber's colour so the box, the tick and the legend agree.
        // An UNJUDGED box keeps its per-label hue: "not yet decided" is not one of the four
        // grades and must not borrow one of their colours.
        const stroke = grade ? GRADE_COLOUR[grade] : 'hsl(' + colourFor(d.label) + ',85%,60%)';
        const [x0, y0, x1, y1] = d.bbox.map(v => v * s);
        const dim = selected && d.label !== selected;
        ctx.globalAlpha = dim ? 0.18 : 1;
        ctx.strokeStyle = stroke;
        if (layers.box2d) {{
          ctx.setLineDash(grade === 'admit' ? [] : [5, 3]);   // solid == admitted
          ctx.strokeRect(x0, y0, x1 - x0, y1 - y0);
        }}
        if (layers.det) {{
          const txt = d.label + (d.score != null ? '  ' + d.score.toFixed(2) : '') +
                      (grade ? '  ' + grade : '');
          const tw = ctx.measureText(txt).width + 8;
          ctx.setLineDash([]);
          ctx.fillStyle = grade ? GRADE_FILL[grade]
                                : 'hsla(' + colourFor(d.label) + ',85%,25%,0.92)';
          ctx.fillRect(x0, Math.max(0, y0 - 15), tw, 15);
          ctx.fillStyle = '#f8fafc';
          ctx.fillText(txt, x0 + 4, Math.max(11, y0 - 4));
        }}
      }}
      ctx.setLineDash([]); ctx.globalAlpha = 1;
    }}
  }}

  // grade -> which toggle governs it. `perm` and `temp` keep their keys because viewer.html
  // hard-codes onclick="toggleVizLayer('perm')" and this file does not own that file.
  const LAYER_OF = {{admit: 'perm', hold: 'hold', decline: 'decline', no_grounds: 'no_grounds'}};

  // Wall segments are PER-FRAME and are not archived in a bundle, so in a replay there is
  // nothing to show and the button must say so rather than sit empty looking broken.
  let WALLS = [], WALLS_WHY = '';
  function pollWalls() {{
    if (!layers.walls || window.REPLAY_DETACHED) return;   // live segments must not draw over a recorded frame
    fetch('/walls_view').then(r => r.json()).then(d => {{
      WALLS = (d && d.walls) || [];
      WALLS_WHY = d && d.available === false ? (d.why || 'no wall detector') : '';
      const el = $(BTN.walls);
      if (el) el.title = WALLS_WHY || (WALLS.length + ' wall segments');
      show();
    }}).catch(() => {{}});
  }}
  setInterval(pollWalls, 2000);
  const GRADE_COLOUR = {{admit: '#22c55e', hold: '#eab308', decline: '#ef4444',
                        no_grounds: '#94a3b8'}};
  const GRADE_FILL = {{admit: 'rgba(21,94,52,.92)', hold: 'rgba(113,84,7,.92)',
                      decline: 'rgba(127,29,29,.92)', no_grounds: 'rgba(51,65,85,.92)'}};
  function drawnLayers() {{
    return layers.perm || layers.temp || layers.hold || layers.decline || layers.no_grounds;
  }}

  // ROW-MAJOR RLE. I wrote this column-major first, on the assumption that a field named
  // mask_rle follows COCO, which is Fortran order. It does not, and the assumption produced a
  // mask smeared diagonally across the frame. MEASURED on doorway#1 of this bundle:
  //   row-major     x 940-998  y 408-612   <- matches
  //   column-major  x 545-817  y   0-959
  //   its bbox_2d   x 940-1000 y 406-606
  // The bbox was sitting right there to check against, which is the lesson.
  //
  // Runs are therefore HORIZONTAL, so each one draws as a span rather than pixel by pixel --
  // 38 masks at 1.2 M pixels was ~46 M fillRect calls a frame; this is a few thousand.
  function paintRLE(rle, s, fill) {{
    const [h, w] = rle.size;
    let idx = 0, val = rle.first_val;
    ctx.fillStyle = fill;
    for (const run of rle.counts) {{
      if (val) {{
        let p = idx, left = run;
        while (left > 0) {{
          const y = (p / w) | 0, x = p % w;
          const n = Math.min(left, w - x);   // clipped to the end of this row
          ctx.fillRect(x * s, y * s, n * s, Math.max(1, s));
          p += n; left -= n;
        }}
      }}
      idx += run; val = val ? 0 : 1;
    }}
  }}

  // ---- the event track -----------------------------------------------------------------
  // Ticks over the scrubber, on the SAME time axis as the frames, so you can scrub to the
  // moment something was decided instead of hunting for it. Placed by wall-clock time, not
  // by frame index: decisions and frames do not occur at the same rate, and spacing the
  // ticks evenly would put them next to the wrong frames while looking plausible.
  const EV_COLOUR = {{admit: '#22c55e', hold: '#eab308', decline: '#ef4444',
                     no_grounds: '#64748b', reject: '#ef4444', abstain: '#64748b',
                     merge: '#38bdf8'}};
  let EVENTS = [];
  function drawEvents() {{
    const c = $('replayEvents');
    if (!c || !F.length) return;
    const rect = c.getBoundingClientRect();
    c.width = Math.max(1, Math.round(rect.width)); c.height = 15;
    const g = c.getContext('2d');
    g.clearRect(0, 0, c.width, c.height);
    const t0 = Number(String(F[0].id).split('_')[0]);
    const t1 = Number(String(F[F.length - 1].id).split('_')[0]);
    const span = Math.max(1, t1 - t0);
    // Merges last, so the rarest events are not painted over by the commonest.
    const order = ['no_grounds', 'abstain', 'decline', 'reject', 'hold', 'admit', 'merge'];
    for (const kind of order) {{
      g.fillStyle = EV_COLOUR[kind] || '#475569';
      for (const e of EVENTS) {{
        if (e.kind !== kind) continue;
        const x = ((e.t - t0) / span) * c.width;
        if (x < -2 || x > c.width + 2) continue;
        g.fillRect(Math.round(x), kind === 'merge' ? 0 : 4, 2, kind === 'merge' ? 15 : 11);
      }}
    }}
    // Where you are now.
    const tc = Number(String(F[i].id).split('_')[0]);
    g.fillStyle = '#f8fafc';
    g.fillRect(Math.round(((tc - t0) / span) * c.width) - 1, 0, 2, 15);
  }}
  fetch('/replay/pose3d/' + encodeURIComponent(BUNDLE))
    .then(r => r.json()).then(d => {{ POSE = d.poses || {{}}; BOX3 = d.boxes || {{}}; BELIEF = d.belief || []; drawOverlay(); }})
    .then(() => {{ window.REPLAY_READY = true; }})
    .catch(() => {{}});
  // Read-only debug hook, as /scene3d exposes SCENE3D: the page's state is `let`-scoped and a
  // browser check cannot otherwise ask what was drawn. `goto` steps to a frame by id.
  window.REPLAY = {{ state: () => ({{ frames: F.length, i, layers, belief: BELIEF.length,
                                     boxes: (BOX3[F[i] && F[i].id] || []).length,
                                     pose: POSE[F[i] && F[i].id] ? {{rms_deg: POSE[F[i].id].rms_deg}} : null,
                                     skipped: window.REPLAY_DEGENERATE_SKIPPED,
                                     belief_drawn: window.REPLAY_BELIEF_DRAWN, cycle: window.REPLAY_CYCLE }}),
                    redraw: () => drawOverlay(),
                    goto: id => {{ const k = F.findIndex(f => f.id === id); if (k >= 0) {{ detach(); i = k; show(); }} return k; }},
                    live: on => setLive(on !== false), following: () => following }};

  fetch('/replay/events/' + encodeURIComponent(BUNDLE))
    .then(r => r.json()).then(d => {{
      EVENTS = d.events || [];
      for (const k in EVF) delete EVF[k];      // any table built before this is stale
      drawEvents();
    }})
    .catch(() => {{}});

  fetch('/replay/verdicts/' + encodeURIComponent(BUNDLE))
    .then(r => r.json()).then(d => {{
      VERDICT = d.verdicts || {{}};
      const amb = (d.ambiguous || []).length;
      const n = {{}};
      for (const g of Object.values(VERDICT)) n[g] = (n[g] || 0) + 1;
      // Put the counts on the toggles. A layer that turns out to be empty in this run should
      // say so on its face rather than looking like a control that does not work.
      const cnt = k => (n[k] || 0);
      setBtn('btnVizPerm', 'ADMITTED ' + cnt('admit'), 'Objects the Bookkeeper admitted');
      setBtn('btnVizTemp', 'UNJUDGED' + (amb ? ' +' + amb + ' MIXED' : ''),
             'Detections with no admission verdict recorded' + (amb ? '; plus ' + amb +
             ' labels that received more than one verdict (a label is a per-frame ordinal, ' +
             'not an identity), drawn here rather than under a grade picked at random' : ''));
      setBtn('btnViz_hold', 'HELD ' + cnt('hold'), 'Objects held for more evidence');
      setBtn('btnViz_decline', 'DECLINED ' + cnt('decline'), 'Objects the Bookkeeper declined');
      setBtn('btnViz_no_grounds', 'NO-GROUNDS ' + cnt('no_grounds'),
             'Objects judged without grounds to decide');
      drawOverlay();
    }})
    .catch(() => {{}});

  // ---- the log panel, following the frame ---------------------------------------------
  const logEl = $('logConsole');
  let logBusy = false, logWin = null;
  function updateReplayLog() {{
    if (!logEl || !F.length || logBusy) return;
    const at = Number(String(F[i].id).split('_')[0]);
    if (logWin && logWin.t_lo <= at && at <= logWin.t_hi) return renderReplayLog(at);
    logBusy = true;
    fetch('/replay/log/' + encodeURIComponent(BUNDLE) + '/perception.log?at=' + at)
      .then(r => r.json()).then(w => {{ logWin = w; logBusy = false; renderReplayLog(at); }})
      .catch(() => {{ logBusy = false; }});
  }}
  function renderReplayLog(at) {{
    if (!logWin || !logEl) return;
    const lines = logWin.text.split('\\n');
    let best = 0;
    for (let n = 0; n < lines.length; n++) {{
      const mm = lines[n].match(/\\[(\\d{{10}})\\./);
      if (mm && Number(mm[1]) <= at) best = n;
    }}
    logEl.textContent = lines.slice(Math.max(0, best - 40), best + 6).join('\\n');
    logEl.scrollTop = logEl.scrollHeight;
  }}

  // ---- transport wiring ----------------------------------------------------------------
  function show() {{
    if (!F.length || following) return;      // following the head, the <img> IS the stream
    const f = F[i];
    feed.src = PFX + '/replay/frame/' + encodeURIComponent(BUNDLE) + '/' + f.id + '.jpg';
    $('rSeek').value = i;
    const t = new Date(Number(String(f.id).split('_')[0]) * 1000);
    $('rPos').textContent = (i + 1) + ' / ' + F.length + ' \u00b7 ' +
      f.detections.length + ' det \u00b7 ' + t.toLocaleTimeString();
    drawEvents();
    if (layers.seg) {{
      masksFor = f.id;
      fetch('/replay/masks/' + encodeURIComponent(BUNDLE) + '/' + f.id)
        .then(r => r.json()).then(d => {{ masks = d.masks; drawOverlay(); }}).catch(() => {{}});
    }}
    updateReplayLog();
    updateCycleMetrics(f.id);
    applyTimeFilter(f.id);
    liveBadge();
  }}
  // ---- GA-361: the knowledge graph and the object set evolve with the scrubber -------------
  // The bundle holds the END STATE of the store; each object carries created_at (its
  // creation_time). At frame t the page shows the objects with created_at <= t and hides the
  // rest ('future'); the table follows through window.REPLAY_TIME_FILTER (viewer.verdictRows).
  // DENOMINATOR, said on the bar: "k of N end-state objects". Objects merged away or deleted
  // before the end are NOT in the store and cannot be shown; that is a decisions-log join, not
  // done here. Frames stamps and creation_time are both wall-clock seconds of the same host.
  let timeStyled = false, T0 = null;
  function frameT(id) {{ const [a, b] = String(id).split('_'); return Number(a) + (Number(b) || 0) / 1e9; }}
  function applyTimeFilter(fid) {{
    const t = following ? Infinity : frameT(fid);    // the live head IS the store's present
    if (T0 === null && F.length) {{ T0 = frameT(F[0].id); window.REPLAY_T0 = T0; }}
    window.REPLAY_TIME_FILTER = layers.time
      ? (n => !(typeof n.created_at === 'number') || n.created_at <= t) : null;
    whenCy(cy => {{
      if (!timeStyled) {{ cy.style().selector('.future').style({{ display: 'none' }}).update(); timeStyled = true; }}
      let shown = 0, total = 0;
      cy.batch(() => cy.nodes().forEach(n => {{
        const c = n.data('created_at');
        if (n.data('type') !== 'object') return;
        total++;
        const future = layers.time && typeof c === 'number' && c > t;
        n.toggleClass('future', future);
        if (!future) shown++;
      }}));
      const pos = $('rPos');
      if (pos) pos.textContent += ' \u00b7 objects so far ' + (layers.time ? shown : total) + ' of ' + total + (LIVE_MODE ? ' (store now)' : ' (end state)');
    }});
    if (typeof window.renderObjectsTable === 'function') {{ try {{ window.renderObjectsTable(); }} catch (e) {{}} }}
  }}
  // ---- the Metrics tab follows the frame (GA-334) ------------------------------------------
  // updateBEV painted the run's LAST snapshot into these cells once, for every frame; that read
  // as "averages". Each frame now shows its own cycle's row, or says it has none. Only the
  // cycle cells are touched: FRAME PERIOD is a run-level measurement and stays.
  let cycleReq = 0;
  const CYCLE_CELLS = {{ latVlm: 'vlm_ms', latOwlv2: 'owlv2_ms', latNms: 'nms_ms', latSam: 'sam_ms',
                        latProj: 'projection_ms', latTotal: 'total_ms', latCycle: 'cycle_ms',
                        latVlmPill: 'vlm_ms', latOwlPill: 'owlv2_ms', latSamPill: 'sam_ms', latProjPill: 'projection_ms' }};
  let lastCycle = null;
  function paintCycleCells(c) {{
    const row = c && c.row;
    for (const id in CYCLE_CELLS) {{
      const el = $(id); if (!el) continue;
      const v = row ? row[CYCLE_CELLS[id]] : null;
      el.textContent = (v !== undefined && v !== null) ? v + ' ms' : '\u2014 ms';
      el.style.opacity = (row && c.match === 'preceding') ? '0.55' : '';
    }}
  }}
  // THE LOAD-ORDER RACE, observed 2026-09-07 on run H (no series): the one-shot updateBEV()
  // below resolves /bev_data AFTER the first per-frame paint and writes the run's END SNAPSHOT
  // into the same cells -- the "averages" the owner reported, back for one page load in two.
  // The cells are re-asserted from the last per-frame answer whenever they disagree with it.
  setInterval(() => {{
    if (!lastCycle) return;
    const el = $('latCycle'), row = lastCycle.row;
    const want = (row && row.cycle_ms != null) ? row.cycle_ms + ' ms' : '\u2014 ms';
    if (el && el.textContent !== want) paintCycleCells(lastCycle);
  }}, 1000);
  function paintCycle(c) {{
    const row = c && c.row;
    lastCycle = c || {{row: null, match: null, n_rows: 0}};
    paintCycleCells(c);
    const note = !c || !c.n_rows ? 'no per-cycle series in this bundle'
      : c.match === 'frame' ? 'cycle ' + row.cycle + ' (this frame)'
      : c.match === 'preceding' ? 'cycle ' + row.cycle + ' (preceding; this frame has no row)'
      : 'no cycle row for this frame';
    window.REPLAY_CYCLE = {{ match: c ? c.match : null, cycle: row ? row.cycle : null, n_rows: c ? c.n_rows : 0, note }};
    const pos = $('rPos'); if (pos) pos.textContent += ' \u00b7 ' + note;
    const lc = $('latCycle'); if (lc) lc.title = note;
  }}
  function updateCycleMetrics(fid) {{
    const my = ++cycleReq;
    fetch('/replay/cycle/' + encodeURIComponent(BUNDLE) + '/' + encodeURIComponent(fid))
      .then(r => r.json()).then(c => {{ if (my === cycleReq) paintCycle(c); }})
      .catch(() => {{ if (my === cycleReq) paintCycle(null); }});
  }}
  feed.addEventListener('load', drawOverlay);
  window.addEventListener('resize', drawOverlay);

  // ---- click a box in the feed to select the object --------------------------------------
  // The boxes were the one view of an object with no way to point at it: the table row and
  // the graph node were both clickable and the thing you were actually looking at was not.
  // Only boxes that are CURRENTLY DRAWN are hit-testable -- a hidden layer must not answer a
  // click, or a decline you switched off would still select and there would be no way to see
  // why. The SMALLEST containing box wins: a small object almost always sits inside a larger
  // one (a knob on a cabinet), and the outer box is never the one that was aimed at.
  // A click on empty frame clears the selection and restores every box. Without it the only
  // way back to the full picture was to find and re-click the same box, which is a trap: the
  // boxes you can still see are the ones already selected.
  function clearSelectionIfEmpty(hit) {{
    if (!hit && selected) {{ selectObject(selected); return true; }}   // toggles it off
    return false;
  }}

  function hitTest(ev) {{
    if (!F.length || !feed.naturalWidth || !cv.width) return null;
    const r = cv.getBoundingClientRect();
    const s = r.width / feed.naturalWidth;          // canvas px per frame px; fitCanvas set it
    const x = (ev.clientX - r.left) / s, y = (ev.clientY - r.top) / s;
    let best = null, bestArea = Infinity;
    for (const d of F[i].detections) {{
      if (!d.bbox || d.bbox.length < 4) continue;
      if (!layers[LAYER_OF[VERDICT[d.label] || null] || 'temp']) continue;
      const [x0, y0, x1, y1] = d.bbox;
      if (x < x0 || x > x1 || y < y0 || y > y1) continue;
      const a = Math.abs((x1 - x0) * (y1 - y0));
      if (a < bestArea) {{ bestArea = a; best = d.label; }}
    }}
    return best;
  }}
  cv.addEventListener('click', ev => {{
    const h = hitTest(ev);
    if (h) selectObject(h); else clearSelectionIfEmpty(h);
  }});
  cv.addEventListener('mousemove', ev => {{
    const h = hitTest(ev);
    cv.style.cursor = h ? 'pointer' : 'default';
    cv.title = h || '';
  }});

  function stopTimer() {{ if (timer) {{ clearInterval(timer); timer = null; }} $('rPlay').innerHTML = ICON.play; }}
  // GA-345: every transport action that MOVES the frame (seek, prev, stop, goto, play from a
  // scrub) leaves the live head first, and the recorded frame it lands on is shown under SCRUB,
  // never under LIVE. detach() changes the STATE only; the caller then sets `i` and calls show(),
  // so the seek value is read before anything rewrites the slider (the first version detached
  // via setLive(false) -> show(), which snapped the slider back to the head before the value was
  // read: the first click on the track was ignored). At the head, PLAY and NEXT do nothing;
  // stepping or playing past the LAST recorded frame on a live page re-engages the head instead
  // of wrapping to frame 1 (an archive still wraps).
  function detach() {{ if (following) setState(false); }}
  function atEnd() {{ return LIVE_MODE && i >= F.length - 1; }}
  $('rPlay').onclick = () => {{
    if (timer) return stopTimer();
    if (following) return;                 // nothing to play at the head; scrub back first
    $('rPlay').innerHTML = ICON.pause;
    timer = setInterval(() => {{
      if (atEnd()) {{ stopTimer(); setLive(true); return; }}
      i = (i + 1) % F.length; show();
    }}, 700 / speed);
  }};
  $('rStop').onclick = () => {{ stopTimer(); detach(); i = 0; show(); }};
  $('rPrev').onclick = () => {{ stopTimer(); detach(); if (!jumpEvent(-1)) {{ i = (i - 1 + F.length) % F.length; show(); }} }};
  $('rNext').onclick = () => {{
    stopTimer();
    if (atEnd()) {{ if (!following) setLive(true); return; }}
    detach();
    if (!jumpEvent(+1)) {{ i = (i + 1) % F.length; show(); }}
  }};
  $('rLive').onclick = () => {{ stopTimer(); setLive(true); }};
  $('rSpeed').onclick = () => {{
    speed = speed >= 8 ? 0.5 : speed * 2;
    $('rSpeed').textContent = speed + 'x';
    if (timer) {{ stopTimer(); $('rPlay').click(); }}
  }};
  $('rSeek').oninput = e => {{ const v = Number(e.target.value); stopTimer(); detach(); i = v; show(); }};

  // The LAYERS bar toggled a server-rendered overlay that does not exist in a replay, so it
  // did nothing. Point it at the overlay this page draws.
  const BTN = {{perm: 'btnVizPerm', temp: 'btnVizTemp', seg: 'btnVizSeg', det: 'btnVizDet',
                hold: 'btnViz_hold', decline: 'btnViz_decline',
                no_grounds: 'btnViz_no_grounds', walls: 'btnViz_walls'}};
  window.toggleVizLayer = function (which) {{
    layers[which] = !layers[which];
    const el = $(BTN[which]); if (el) el.classList.toggle('active', layers[which]);
    if (which === 'seg' && layers.seg) {{ masks = null; masksFor = null; show(); }}
    if (which === 'walls' && layers.walls) pollWalls();
    drawOverlay();
  }};

  // Relabel a LAYERS button in place. The <i> icon is preserved so the bar keeps its look;
  // only the trailing text changes, which is why this walks the child nodes rather than
  // rewriting innerHTML (that would drop a lucide icon already rendered to inline SVG).
  function setBtn(id, text, title) {{
    const el = $(id);
    if (!el) return;
    let last = el.lastChild;
    while (last && last.nodeType !== 3) last = last.previousSibling;
    if (last) last.nodeValue = ' ' + text; else el.appendChild(document.createTextNode(' ' + text));
    if (title) el.title = title;
  }}

  // The three grade layers have no buttons in viewer.html and this file does not own that
  // file, so they are added here, next to the ones that are there.
  (function addGradeButtons() {{
    const host = $('btnVizDet') && $('btnVizDet').parentNode;
    if (!host) return;
    [['hold', 'HELD', '#eab308'], ['decline', 'DECLINED', '#ef4444'],
     ['no_grounds', 'NO-GROUNDS', '#94a3b8'],
     // GA-271. Detected wall segments, the SAME source the Habitat window's `w` key draws:
     // both read the bridge's /walls, so the two views cannot disagree about what was found.
     ['walls', 'WALLS', '#38bdf8']].forEach(([k, txt, col]) => {{
      const b = document.createElement('button');
      b.id = BTN[k];
      b.className = 'viz-btn';
      b.innerHTML = '<i style="display:inline-block;width:8px;height:8px;border-radius:2px;' +
                    'background:' + col + ';"></i>';
      b.appendChild(document.createTextNode(' ' + txt));
      b.onclick = () => window.toggleVizLayer(k);
      host.appendChild(b);
    }});
  }})();
  ['btnVizSeg'].forEach(id => {{ const e = $(id); if (e) e.classList.remove('active'); }});

  // STAGES named a SERVER-RENDERED per-stage composite from /image_with_bb. No archive has
  // one, so the name promised something replay cannot show. The button was not dead -- this
  // file had already rewired it to `layers.det`, the per-box label and score -- it was
  // MISNAMED, which is the same defect wearing a better disguise. crop_meta was the other
  // candidate for a real per-stage overlay and it is not one: MEASURED on this bundle it
  // reads status="ok", construction="contour" on all 1,530 rows, so drawing it would add an
  // identical string to every box. Name the button after what it does.
  setBtn('btnVizDet', 'SCORES', 'Draw the label, detector score and verdict on each box');

  // The header read "GRAPH API | AUTONOMOUS SYSTEM CONTROL". Name the thing after what it
  // shows -- the run being replayed -- not after an aspiration.
  document.querySelectorAll('h1, .hud-title, .app-title').forEach(el => {{
    if (/AUTONOMOUS SYSTEM CONTROL/i.test(el.textContent)) {{
      el.textContent = BRAND + ' \u00b7 semantic mapping \u00b7 replay';
    }}
  }});
  document.title = BRAND + ' replay \u00b7 ' + BUNDLE;

  // Finer zoom on the graph. The default wheel step jumps roughly 2x per notch, which at 270
  // nodes overshoots past the cluster you were aiming at in one movement.
  whenCy(cy => {{
    cy.minZoom(0.02); cy.maxZoom(8);
    const box = cy.container();
    // GA-253. COALESCE INTO ONE ZOOM PER FRAME. A trackpad emits wheel events faster than
    // the graph can redraw, and this applied cy.zoom() to EVERY one -- several full
    // re-renders inside a single frame, resolving out of step with the display. That is the
    // jitter the owner saw: not slowness, a queue of conflicting zooms.
    let wheelAccum = 0, wheelX = 0, wheelY = 0, wheelPending = false;
    box.addEventListener('wheel', ev => {{
      ev.preventDefault(); ev.stopPropagation();
      const r = box.getBoundingClientRect();
      wheelX = ev.clientX - r.left; wheelY = ev.clientY - r.top;
      // deltaMode 1 is lines and 2 is pages; without normalising, a mouse notch and a
      // trackpad swipe differ by two orders of magnitude.
      const unit = ev.deltaMode === 1 ? 16 : (ev.deltaMode === 2 ? 400 : 1);
      wheelAccum += ev.deltaY * unit;
      if (wheelPending) return;
      wheelPending = true;
      requestAnimationFrame(() => {{
        wheelPending = false;
        const dd = wheelAccum; wheelAccum = 0;
        if (!dd) return;
        // Continuous in the accumulated delta, so a fast swipe travels further than a slow
        // one instead of quantising to one fixed step per event.
        const f = Math.exp(-dd * 0.0016);
        cy.zoom({{level: Math.max(cy.minZoom(), Math.min(cy.maxZoom(), cy.zoom() * f)),
                 renderedPosition: {{x: wheelX, y: wheelY}}}});
      }});
    }}, {{passive: false, capture: true}});
  }});

  // ---- one selection, three panels -----------------------------------------------------
  // Clicking an object anywhere must show it everywhere, because the three views answer
  // different questions about the SAME thing: the table says what it is, the graph says what
  // it relates to, the BEV says where it is. Selecting in one and hunting in the others is
  // how you end up comparing two different objects and not noticing.
  // Reads `selected` when it runs, not when it is queued -- so a click that arrives before
  // cytoscape is up still paints the CURRENT selection once cytoscape appears.
  function paintGraphSelection(cy) {{
    cy.elements().removeClass('faded');
    if (!selected) return;
    const n = cy.nodes().filter(e => (e.data('label') || '') === selected);
    if (!n.length) return;
    cy.elements().difference(n.closedNeighborhood()).addClass('faded');
    // The old behaviour zoomed to the node alone, which fills the pane with one circle and
    // loses every edge that made it worth clicking. Fit the node AND its neighbours, then
    // cap the zoom so a lone node cannot fill the pane either.
    cy.animate({{fit: {{eles: n.closedNeighborhood(), padding: 90}}, duration: 320}},
               {{complete: () => {{ if (cy.zoom() > 1.6) cy.zoom({{
                  level: 1.6, renderedPosition: {{x: cy.width()/2, y: cy.height()/2}} }}); }}}});
  }}

  function selectObject(label) {{
    selected = (selected === label) ? null : label;

    // 1. the graph -- fade the rest, then FRAME the node rather than diving into it.
    whenCy(paintGraphSelection);

    // 2. the objects table. THE SELECTORS WERE WRONG and that is why the row never lit up:
    // there is no #objectsList and no .object-row anywhere in this page. The rows are
    // `#tableWrap tbody tr`, carrying data-node-id (269 of 294 rows) and data-label (all
    // 294). Matching on the row's TEXT was wrong too -- the label is not unique, 269 objects
    // share 117 labels, so a text match lights up every dining chair at once.
    document.querySelectorAll('#tableWrap tbody tr').forEach(tr => {{
      const hit = selected && (tr.dataset.label || '') === selected;
      tr.style.opacity = (!selected || hit) ? '1' : '0.3';
      tr.style.background = hit ? 'rgba(2,132,199,0.35)' : '';
      tr.style.borderLeft = hit ? '3px solid #38bdf8' : '';
      if (hit && tr.scrollIntoView) tr.scrollIntoView({{block: 'nearest'}});
    }});
    // Agent B exposed this for exactly this purpose and it was never called.
    if (selected && typeof window.highlightObjectRow === 'function') {{
      const row = [...document.querySelectorAll('#tableWrap tbody tr')]
        .find(r => (r.dataset.label || '') === selected);
      if (row) {{ try {{ window.highlightObjectRow(row.dataset.nodeId, selected); }} catch (e) {{}} }}
    }}

    // 3. the BEV, and the feed overlay
    window.__bevSelected = selected;
    if (typeof window.updateBEV === 'function') {{ try {{ window.updateBEV(); }} catch (e) {{}} }}
    drawOverlay();
  }}
  window.selectObject = selectObject;

  whenCy(cy => {{
    cy.on('tap', 'node', ev => selectObject(ev.target.data('label') || ''));
    cy.on('tap', e => {{ if (e.target === cy && selected) selectObject(selected); }});
  }});
  document.addEventListener('click', ev => {{
    const row = ev.target.closest && ev.target.closest('#objectsList tr, .object-row');
    if (!row) return;
    const cell = row.querySelector('td, .obj-label');
    if (cell) selectObject(cell.textContent.trim());
  }}, true);

  // ---- BEV: the map had 300x460 CSS pixels and a 298x223 backing store --------------------
  // viewer.html fixes the canvas attributes at 298x223 and nothing resizes them, so the CSS
  // above was stretching that bitmap 2.06x vertically: a blurred map, a squashed north-south
  // axis, and half the label room the panel appears to have. Match the backing store to the
  // box. This also repairs the waypoint click maths in viewer.html, which divides by
  // canvas.width/height and was therefore off by the same 2.06x in y.
  (function sizeBEV() {{
    const c = $('bevCanvasHUD');
    if (c && (c.width !== 300 || c.height !== 460)) {{ c.width = 300; c.height = 460; }}
  }})();

  // ---- BEV: keep one readable label per cluster ------------------------------------------
  // TWO failures, one after the other, both unreadable. First every one of the 269 objects
  // wrote its name at its own position, so 16 dining chairs wrote 16 overlapping strings.
  // Then a declutter pass dropped any label that collided and hid every unselected label
  // outright -- which emptied the map: the owner's screenshot has four labels on it.
  //
  // What is drawn instead:
  //   * a cluster of the SAME name is merged, not deleted -- "chair x6" is the fact the
  //     map is there to report, and it is lost either way round if the labels just vanish;
  //   * bigger objects, and the selected one, claim their slot first, so the survivor is
  //     the landmark rather than whichever object the bridge serialised first;
  //   * a selection DIMS the others to 0.38 rather than removing them: a label you can
  //     still read is what tells you where the selected thing is;
  //   * the label PILL is held back with its text. The previous pass suppressed only the
  //     fillText and left the pill, so a dropped label became an empty box -- which reads
  //     as a rendering fault rather than as a decision.
  //
  // The interception is safe to key on shape because renderCanvasBEV makes exactly ONE
  // fillRect (viewer.html:1733, the pill) and exactly ONE fillText (:1738, its text); its
  // other strokeRect (:1715) is the object footprint and is matched out by coordinates.
  (function declutterBEV() {{
    const draw = window.renderCanvasBEV;
    if (typeof draw !== 'function') return;
    const area = o => {{
      const b = o && o.bbox;
      if (!b || b.x_min === undefined) return 0;
      return Math.abs((b.x_max - b.x_min) * (b.y_max - b.y_min));
    }};

    // THE OBJECT LIST HAS TO COME FROM HERE. /bev_data carries no `objects` in this bundle,
    // so viewer.html:1794 falls back to `lastGraphData` -- a block-scoped `let` in ANOTHER
    // script tag, which this file cannot see and therefore cannot reorder. Without this the
    // priority sort below would silently never run: the draw order would stay whatever the
    // bridge serialised, and the label that won a crowded spot would be arbitrary.
    let GOBJ = null;
    fetch('/graph_data').then(r => r.json()).then(d => {{
      GOBJ = (d.objects || d.nodes || []).filter(o => o.type !== 'room' && o.type !== 'concept');
      if (typeof window.updateBEV === 'function') {{ try {{ window.updateBEV(); }} catch (e) {{}} }}
    }}).catch(() => {{}});

    window.renderCanvasBEV = function (canvasId, data, zoom) {{
      const c = document.getElementById(canvasId);
      const g = c && c.getContext ? c.getContext('2d') : null;
      if (!g) return draw.call(this, canvasId, data, zoom);
      const sel = window.__bevSelected ? String(window.__bevSelected).split('#')[0] : null;

      // Draw order IS priority: the first label at a spot keeps it. Biggest first so the
      // survivor of a crowd is the landmark, and the selected object before anything.
      const src = (data && Array.isArray(data.objects)) ? data.objects : GOBJ;
      if (Array.isArray(src)) {{
        const objs = src.slice().sort((a, b) => {{
          const nm = o => String(o.name || o.label || '').split('#')[0];
          if (sel) {{
            const as = nm(a) === sel, bs = nm(b) === sel;
            if (as !== bs) return as ? -1 : 1;
          }}
          return area(b) - area(a);
        }});
        data = Object.assign({{}}, data, {{objects: objs}});
      }}

      const realFillRect = g.fillRect.bind(g);
      const realStrokeRect = g.strokeRect.bind(g);
      const realFillText = g.fillText.bind(g);
      const placed = [];
      let pend = null, merged = 0, dropped = 0, seen = 0;

      function paint(rec) {{
        const txt = rec.n > 1 ? rec.text + ' x' + rec.n : rec.text;
        const sf = g.fillStyle, ss = g.strokeStyle, sw = g.lineWidth,
              sa = g.globalAlpha, sfo = g.font;
        g.font = rec.font;
        g.globalAlpha = (sel && !rec.sel) ? 0.38 : 1;
        const w = g.measureText(txt).width + 6;
        if (rec.pill) {{
          g.fillStyle = rec.pill.fill;
          realFillRect(rec.pill.x, rec.pill.y, w, rec.pill.h);
          if (rec.pill.stroke) {{
            g.strokeStyle = rec.pill.stroke.s; g.lineWidth = rec.pill.stroke.w;
            realStrokeRect(rec.pill.x, rec.pill.y, w, rec.pill.h);
          }}
        }}
        g.fillStyle = rec.fill;
        realFillText(txt, rec.tx, rec.ty);
        g.fillStyle = sf; g.strokeStyle = ss; g.lineWidth = sw;
        g.globalAlpha = sa; g.font = sfo;
        rec.w = Math.max(rec.w, w);
      }}

      g.fillRect = function (x, y, w, h) {{
        if (h === 12) {{ pend = {{x, y, w, h, fill: g.fillStyle, stroke: null}}; return; }}
        return realFillRect(x, y, w, h);
      }};
      g.strokeRect = function (x, y, w, h) {{
        if (pend && x === pend.x && y === pend.y && w === pend.w && h === pend.h) {{
          pend.stroke = {{s: g.strokeStyle, w: g.lineWidth}};
          return;
        }}
        return realStrokeRect(x, y, w, h);
      }};
      g.fillText = function (txt, x, y) {{
        const s = String(txt);
        seen++;
        const isSel = !!(sel && s === sel);
        const box = pend ? {{x: pend.x, y: pend.y, w: pend.w, h: pend.h}}
                         : {{x: x - 3, y: y - 9, w: g.measureText(s).width + 6, h: 12}};
        // Pill-against-pill, not glyph-against-glyph: two pills that overlap are unreadable
        // even where the letters happen not to touch.
        const hit = placed.find(p => !(box.x > p.x + p.w || box.x + box.w < p.x ||
                                       box.y > p.y + p.h || box.y + box.h < p.y));
        // A collision is resolved the same way for the selected object as for anything
        // else. Letting the selection ignore the test looked right and was not: `sel` is a
        // base name, so "dining chair" matched all 91 of them and every one drew -- 118
        // labels on a 300x460 canvas, the original illegible pile back again for whatever
        // you had just clicked. The selection gets its advantage from the SORT above, which
        // lets it claim its slot first, and from alpha, which dims everything else.
        if (hit) {{
          if (hit.text === s) {{ hit.n++; merged++; }} else {{ dropped++; }}
          pend = null;
          return;
        }}
        const rec = {{x: box.x, y: box.y, w: box.w, h: box.h, text: s, n: 1, tx: x, ty: y,
                     pill: pend, fill: g.fillStyle, font: g.font, sel: isSel}};
        placed.push(rec);
        paint(rec);
        pend = null;
      }};

      try {{
        return draw.call(this, canvasId, data, zoom);
      }} finally {{
        g.fillRect = realFillRect; g.strokeRect = realStrokeRect; g.fillText = realFillText;
        // A label that absorbed neighbours was drawn before it knew its count, so the ones
        // that grew are repainted over their own pill. Only those -- repainting all of them
        // would put every label on top of the vision cone for no reason.
        for (const r of placed) if (r.n > 1) paint(r);
        window.__bevLabelStats = {{canvas: canvasId, candidates: seen, drawn: placed.length,
                                  clusters: placed.filter(r => r.n > 1).length,
                                  merged: merged, dropped: dropped}};
      }}
    }};
  }})();

  // ---- jump between events ---------------------------------------------------------------
  // Clicking a colour in the legend arms it; the arrows then step between events of that
  // kind instead of one frame at a time. Scrubbing 84 frames to find 53 holds is the kind of
  // search a timeline should do for you.
  function nearestFrameTo(t) {{
    let best = 0, bd = Infinity;
    for (let n = 0; n < F.length; n++) {{
      const d = Math.abs(Number(String(F[n].id).split('_')[0]) - t);
      if (d < bd) {{ bd = d; best = n; }}
    }}
    return best;
  }}
  // WHY THIS WAS DEAD, measured rather than guessed. The old version picked the next EVENT
  // after the current frame's stamp, then mapped that event to its nearest frame. But this
  // bundle records 84 frames across 3,543 s -- p50 47 s apart, max 96 -- while events land
  // within seconds of each other. So the frame nearest the next event is very often the
  // frame you are ALREADY on: it assigned `i = i`, called show() on an identical src (which
  // does not even re-fire `load`), and returned TRUE -- which also suppressed the caller's
  // fallback single-frame step. Pressing again recomputed from the same stamp and chose the
  // same event. MEASURED from frame 0, where the page opens: admit, hold and no_grounds
  // stuck on the first press; decline moved two frames then stuck; merge moved one then
  // stuck. Across all 84 start frames, 17-33 per kind were dead on arrival.
  //
  // Step between the DISTINCT FRAMES that carry an event of the armed kind. Then every press
  // moves, and the arrow runs out at the end of the run instead of at an arbitrary frame.
  const EVF = {{}};
  function eventFrames(kind) {{
    if (!EVF[kind]) {{
      const s = new Set();
      for (const e of EVENTS) if (e.kind === kind) s.add(nearestFrameTo(e.t));
      EVF[kind] = Array.from(s).sort((a, b) => a - b);
    }}
    return EVF[kind];
  }}
  function jumpEvent(dir) {{
    if (!evFilter || !EVENTS.length || !F.length) return false;
    const fr = eventFrames(evFilter);
    let nxt = -1;
    if (dir > 0) {{
      for (const n of fr) {{ if (n > i) {{ nxt = n; break; }} }}
    }} else {{
      for (const n of fr) {{ if (n < i) nxt = n; else break; }}
    }}
    if (nxt < 0) return false;          // no further event of this kind; fall back to 1 frame
    i = nxt;
    show();
    return true;
  }}
  document.querySelectorAll('#replayLegend span[data-kind]').forEach(el => {{
    el.style.cursor = 'pointer';
    el.onclick = () => {{
      evFilter = (evFilter === el.dataset.kind) ? null : el.dataset.kind;
      document.querySelectorAll('#replayLegend span[data-kind]').forEach(o => {{
        o.style.opacity = (!evFilter || o.dataset.kind === evFilter) ? '1' : '0.35';
        o.style.textDecoration = (o.dataset.kind === evFilter) ? 'underline' : 'none';
      }});
      // The arrows wear the armed colour. The icons are drawn in currentColor precisely so
      // that setting the button's colour recolours the glyph: which mode the transport is in
      // is then visible on the control you are about to press, not only in the legend you
      // pressed a moment ago and may have scrolled past.
      const col = evFilter ? (EV_COLOUR[evFilter] || '') : '';
      // Guarded: eventFrames maps events onto FRAMES, so calling it before the frame index
      // lands would cache a table built against an empty F and never recompute it.
      const n = (evFilter && EVENTS.length && F.length) ? eventFrames(evFilter).length : 0;
      ['rPrev', 'rNext'].forEach(id => {{
        const b = $(id);
        if (!b) return;
        b.style.color = col;
        b.style.borderColor = col || '';
        b.title = evFilter ? ((id === 'rNext' ? 'next' : 'previous') + ' frame with a ' +
                              evFilter + ' event (' + n + ' such frames)')
                           : ((id === 'rNext' ? 'next' : 'previous') + ' frame');
      }});
      $('rPos').title = evFilter ? ('arrows now step between ' + evFilter + ' events')
                                 : 'arrows step one frame';
    }};
  }});

  // GA-241. Two drawable things had no control: the 3D cage itself, and whether it uses the
  // PCA-oriented box or the axis-aligned hull. Added to the LAYERS bar beside the rest, so
  // "all layers off" now genuinely clears the frame.
  (function addBoxLayerButtons() {{
    const bar = document.querySelector('.viz-layer-bar-outside') ||
                (document.getElementById('btnVizSeg') || {{}}).parentNode;
    if (!bar) return;
    const mk = (id, text, key, on) => {{
      const b = document.createElement('button');
      b.id = id; b.className = 'viz-btn' + (on ? ' active' : '');
      b.textContent = text;
      b.onclick = () => {{
        layers[key] = !layers[key];
        b.classList.toggle('active', layers[key]);
        if (key === 'time') show(); else drawOverlay();
      }};
      bar.appendChild(b);
      return b;
    }};
    mk('btnVizTime', 'FOLLOW TIME', 'time', true).title =
      'Show only the objects that existed at the viewed frame (created_at <= frame time); off = the end state';
    mk('btnVizBox2d', '2D BOXES', 'box2d', true);
    mk('btnVizBelief', 'BELIEF BOXES', 'belief', true);
    mk('btnVizBox3d', 'DETECTION BOXES', 'box3d', false);
    mk('btnVizPca', 'PCA ROTATION', 'pca', true);
  }})();

  // GA-265. The HABITAT WINDOW's layers, mirrored here. The window is the only view of the
  // simulator while a run walks, and its keys and these buttons write the same dict on the
  // feed host -- so a keypress in the window moves these, and a click here moves the window.
  (function feedLayerPanel() {{
    const bar = document.querySelector('.viz-layer-bar-outside');
    if (!bar) return;
    const wrap = document.createElement('span');
    wrap.id = 'feedLayers';
    wrap.style.cssText = 'margin-left:14px;padding-left:12px;border-left:1px solid ' +
      'rgba(255,255,255,.16);display:inline-flex;gap:4px;align-items:center';
    bar.appendChild(wrap);
    let known = null;
    function render(d) {{
      if (!d || !d.layers) {{ wrap.style.display = 'none'; return; }}
      wrap.style.display = 'inline-flex';
      const keyOf = {{}};
      for (const [k, v] of Object.entries(d.keys || {{}})) keyOf[v] = k;
      wrap.innerHTML = '<span style="font:700 9px ui-monospace,monospace;color:#64748b">' +
        (d.live ? 'HABITAT WINDOW' : 'HABITAT (recorded)') + '</span>' +
        Object.entries(d.layers).map(([n, on]) =>
          '<button class="viz-btn' + (on ? ' active' : '') + '" data-fl="' + n + '"' +
          (d.live ? '' : ' disabled') + ' title="' +
          (keyOf[n] ? 'key: ' + keyOf[n] : '') + '">' + n + '</button>').join('');
      wrap.querySelectorAll('button[data-fl]').forEach(b => {{
        b.onclick = () => {{
          if (window.REPLAY_DETACHED) return;   // GA-345: a scrubbed page must not drive the running window
          const now = b.classList.contains('active');
          fetch('/feed_layers?set=' + encodeURIComponent(b.dataset.fl + ':' + (now ? '0' : '1')))
            .then(r => r.json()).then(render).catch(() => {{}});
        }};
      }});
      known = JSON.stringify(d.layers);
    }}
    function poll() {{
      if (window.REPLAY_DETACHED) return;       // GA-345: running state must not paint over a recorded frame
      fetch('/feed_layers').then(r => r.json()).then(d => {{
        // Only re-render on a CHANGE, so a keypress in the window updates these buttons
        // without the DOM being rebuilt twice a second for nothing.
        if (d && d.layers && JSON.stringify(d.layers) !== known) render(d);
        else if (!d || !d.layers) render(d);
      }}).catch(() => {{}});
    }}
    poll();
    setInterval(poll, 2000);
  }})();

  // GA-258. A live readout of what association is trying to confirm. The owner asked to see
  // which pairs are being compared, and this is the population that matters: over the
  // evidence bar, short of the confirmation streak -- the merges one more look would commit.
  (function mergePendingPanel() {{
    const host = document.getElementById('panelLogs') || document.body;
    const el = document.createElement('div');
    el.id = 'mergePending';
    el.style.cssText = 'position:absolute;left:8px;bottom:8px;z-index:45;max-width:340px;' +
      'background:rgba(11,19,41,.94);border:1px solid rgba(56,189,248,.3);border-radius:6px;' +
      'padding:6px 9px;font:10px ui-monospace,monospace;color:#cbd5e1;display:none';
    host.appendChild(el);
    function refresh() {{
      if (window.REPLAY_DETACHED) return;       // GA-345: a run-level readout, frozen while a recorded frame is shown
      fetch('/merge_pending').then(r => r.json()).then(d => {{
        if (!d.available) {{ el.style.display = 'none'; return; }}
        el.style.display = 'block';
        const rows = (d.pairs || []).slice(0, 8).map(p =>
          '<div style="display:flex;gap:6px"><span style="color:#eab308">' +
          p.streak + '/' + p.needs + '</span><span style="color:#64748b">' +
          p.log_odds.toFixed(1) + '</span><span style="overflow:hidden;text-overflow:ellipsis;' +
          'white-space:nowrap">' + p.a.slice(-8) + ' \u2194 ' + p.b.slice(-8) + '</span></div>'
        ).join('');
        el.innerHTML =
          '<div style="color:#38bdf8;font-weight:700;letter-spacing:.06em">MERGES PENDING ' +
          d.pending + '</div><div style="color:#64748b;margin-bottom:3px">sweep ' + d.sweep +
          ' \u00b7 threshold ' + d.threshold + ' \u00b7 needs ' + d.min_consecutive +
          ' consecutive</div>' + (rows ||
          '<div style="color:#64748b">nothing over the bar awaiting confirmation</div>');
      }}).catch(() => {{}});
    }}
    refresh();
    // Slow poll: this is a run-level statistic, not a per-frame one.
    setInterval(refresh, 4000);
  }})();

  // updateBEV was blocked as a poller; the map is static, so call it once.
  if (typeof window.updateBEV === 'function') {{ try {{ window.updateBEV(); }} catch (e) {{}} }}

  function loadIndex() {{
    return fetch('/replay/index/' + encodeURIComponent(BUNDLE)).then(r => r.json()).then(d => {{
      const grew = (d.frames || []).length !== F.length;
      F = d.frames || [];
      $('rSeek').max = Math.max(0, F.length - 1);
      armTransport();
      return grew;
    }});
  }}
  // GA-396. A bundle with no frames (a MAPPING-ONLY run writes none) left every transport control
  // enabled and doing nothing: pressing step at the end looked like a player that would not loop,
  // when there was nothing to step through. Owner report 2026-09-08 on 20260908_150019_hm3d_00770,
  // which has 0 frames. The controls now say so instead of failing silently -- rule 4, an absent
  // thing must not look like a broken one.
  function armTransport() {{
    const usable = F.length > 1;
    const why = !F.length ? 'this bundle recorded no frames -- a mapping-only run writes none'
                          : (F.length === 1 ? 'this bundle recorded a single frame; there is nothing to step through' : '');
    ['rPlay', 'rStop', 'rPrev', 'rNext', 'rSpeed', 'rSeek'].forEach(id => {{
      const el = $(id);
      if (!el) return;
      el.disabled = !usable;
      el.style.opacity = usable ? '' : '.4';
      // CLEARED when it no longer applies, not only set when it does. A live run writes its first
      // frames a minute in, and the old version left "a mapping-only run writes none" sitting on
      // controls that had just become usable -- a stale explanation is worse than none, because it
      // is read as current (review 2026-09-08).
      el.title = why;
    }});
    if (why) $('rPos').textContent = why;
  }}
  // ---- GA-345: one bar, two states ---------------------------------------------------------
  // Following: the <img> is the bridge's stream (its own overlay), the viewer's pollers own the
  // logs, the heartbeat and the metrics cells, the slider sits on the newest recorded frame and
  // the graph shows the store as it is now. Scrubbed: a recorded frame with the recorded boxes,
  // the frame's own cycle row and log window, the graph as it was at that frame; the viewer's
  // three pollers that would overwrite those read window.REPLAY_DETACHED and stand back.
  function setState(on) {{
    following = LIVE_MODE && on;
    window.REPLAY_DETACHED = LIVE_MODE && !following;
    document.body.classList.toggle('dash-scrub', !following);
    const b = $('rLive'); if (b) b.classList.toggle('on', following);
    paintBadge();
  }}
  // ONE painter for the badge, so every state (and every recovery from an error text) draws the
  // same thing: LIVE / SCRUB · behind live by N / REPLAY.
  function paintBadge() {{
    if (following) {{ badge.textContent = 'LIVE · ' + BUNDLE; return; }}
    if (!LIVE_MODE) {{ badge.textContent = 'REPLAY · ' + BUNDLE; return; }}
    const behind = Math.max(0, F.length - 1 - i);
    badge.textContent = 'SCRUB · ' + BUNDLE + ' · behind live by ' + behind + ' frame' + (behind === 1 ? '' : 's');
  }}
  function setLive(on) {{
    setState(on);
    if (following) {{
      ++cycleReq;            // a cycle fetch still in flight from the scrub would repaint the live cells for good
      i = Math.max(0, F.length - 1);
      $('rSeek').value = i;
      ctx.clearRect(0, 0, cv.width, cv.height);
      lastCycle = null;                                  // the metrics cells belong to updateBEV again
      if (!String(feed.getAttribute('src') || '').startsWith(PFX + '/feed')) feed.src = PFX + '/feed?t=' + Date.now();
      $('rPos').textContent = 'LIVE · ' + F.length + ' frames so far' +
        (F.length ? ' · last ' + new Date(frameT(F[i].id) * 1000).toLocaleTimeString() : '');
      drawEvents();
      applyTimeFilter(null);
    }} else {{
      show();
    }}
  }}
  function liveBadge() {{ paintBadge(); }}
  loadIndex().then(() => {{
      if (LIVE_MODE) setLive(true);
      else if (F.length) show(); else $('rPos').textContent = 'this bundle recorded no frames';
    }})
    .catch(e => {{ $('rPos').textContent = 'could not load frames: ' + e; }});
  if (LIVE_MODE) {{
    // The bundle grows one frame per perception cycle (6 s apart on run 152446, measured) and is
    // re-indexed every 5 s: following, the slider tracks the head; scrubbed, the badge says how
    // far behind the head the shown frame is. Poses, belief boxes and events are re-read only
    // when the index grew. ponytail: the whole index is re-read on every tick by every open tab
    // -- one detections.jsonl pass plus one JPEG header per frame, the header cached per
    // (path, mtime, size) in replay_view (139 frames: 105 ms here, 187-321 ms on the reviewer's read, the detections.jsonl pass being ~90% of it; the per-frame header opens are the small part).
    // Ceiling: linear in frames x tabs; a `?since=` cursor when runs reach thousands of frames.
    // A failed re-read is SAID on the bar, not swallowed: a frozen timeline that looks healthy
    // is the failure this page exists to avoid.
    let barError = false, badgeError = false;
    setInterval(() => {{
      loadIndex().then(grew => {{
        if (barError) {{ barError = false; if (following) setLive(true); else show(); }}   // recovery repaints the bar
        if (!grew) return;
        if (following) setLive(true); else {{ liveBadge(); drawEvents(); }}
        fetch('/replay/pose3d/' + encodeURIComponent(BUNDLE)).then(r => r.json())
          .then(d => {{ POSE = d.poses || {{}}; BOX3 = d.boxes || {{}}; BELIEF = d.belief || []; drawOverlay(); }})
          .catch(e => console.warn('[dash] pose3d re-read failed: ' + e));
        fetch('/replay/events/' + encodeURIComponent(BUNDLE)).then(r => r.json())
          .then(d => {{ EVENTS = d.events || []; for (const k in EVF) delete EVF[k]; drawEvents(); }})
          .catch(e => console.warn('[dash] events re-read failed: ' + e));
      }}).catch(e => {{
        barError = true;
        $('rPos').textContent = 'could not re-read the frame index: ' + e;
        console.warn('[dash] frame index re-read failed: ' + e);
      }});
    }}, 5000);
    // The run ends: the follower drops the server to replay after two missed probes and this
    // page reloads ONCE into the replay of the same bundle -- the mirror of the replay->live
    // reload in _replay_head_html. ponytail: the scrub position is lost on that reload.
    // Owner ruling 2026-09-08: the follower's two misses (~30 s) can also be a bridge that is
    // merely busy (152446 flapped 9x under the old 1.5 s probe), so this page waits for TWO
    // consecutive replay readings (~60 s after the last good probe) before it reloads -- rule
    // 11, a page that stays live shows a red badge, one that drops early hides it. A reading
    // with pinned=true came from a person picking a bundle; loadPickedBundle reloads itself.
    let replaySeen = 0;
    setInterval(() => {{
      fetch('/mode_info').then(r => r.json()).then(m => {{
        if (badgeError) {{ badgeError = false; paintBadge(); }}   // the server answered again
        // A bundle PICKED from a menu (`by: pick`, pinned or not -- picking the running bundle
        // leaves it unpinned) is a person's move to replay in that tab, which reloads itself;
        // it must not reload every other live tab.
        if (!m || m.pinned || m.by === 'pick') {{ replaySeen = 0; return; }}
        replaySeen = (m.mode === 'replay') ? replaySeen + 1 : 0;
        if (replaySeen === 1) console.warn('[dash] server reports replay (' + m.why + '); waiting for a second reading');
        if (replaySeen >= 2) {{
          console.warn('[dash] server is in replay for two readings (run ended); reloading the page into replay mode');
          location.reload();
        }}
      }}).catch(e => {{
        // The dashboard server itself did not answer: there is nothing to reload into, so the
        // page says so on the badge instead of showing LIVE over a server that is gone. The two
        // readings must be CONSECUTIVE: a failed reading resets the count.
        replaySeen = 0; badgeError = true;
        badge.textContent = (following ? 'LIVE?' : 'SCRUB?') + ' · ' + BUNDLE + ' · dashboard server unreachable';
        console.warn('[dash] /mode_info failed: ' + e);
      }});
    }}, 15000);
  }}
}})();
</script>
"""


def _tools_menu_html(bundles=None) -> str:
    """One collapsible menu, bottom right, replacing two loose bars of fixed buttons.

    They were two `position: fixed` groups stacked in the same corner -- links at
    bottom:52px, RViz at bottom:12px. Both floated over the objects table, they collided
    at shorter window heights, and they were reported three times as "the bottom right
    buttons disappeared" when in fact they were overlapping each other and the panel
    beneath them. Collapsed by default, so the corner is clear until it is asked for.

    RVIZ VISIBILITY: loopback only, enforced rather than merely documented. RViz opens a
    window on the SERVER's display, so from another machine the button would do nothing
    the person clicking it can see.
    """
    import html as _h
    # Owner ruling 2026-09-08: the ARIA infra link names a tailnet-only host. It is printed only
    # when this instance is NOT a public deployment; the droplet container sets
    # DASH_PUBLIC=1 and never prints it (dead there, and an internal hostname leak).
    internal = "" if dash_env.flag("DASH_PUBLIC") else (
        '      <a href="https://orchestrator-droplet.tailbd3bab.ts.net:7443/" target="_blank" rel="noopener"\n'
        '         title="ARIA &amp; Personal Infra Orchestration Dashboard (tailnet only)"\n'
        '         style="background:#1e293b;color:#e2e8f0;border:1px solid #334155;border-radius:5px;padding:6px 10px;text-decoration:none;">ARIA INFRA &nearr;</a>\n')
    opts = "".join(f'<option value="{_h.escape(b)}" title="{_h.escape(_bundle_tag(b)[1])}">'
                   f'{_h.escape(b)} \u2014 {_h.escape(_bundle_tag(b)[0])}</option>'
                   for b in (bundles or []))
    # An extension's menu rows, built from the SAME Page objects that install the routes, so a link
    # and its page cannot drift apart -- the failure that used to leave the menu offering a 404.
    _style = ("background:#1e293b;color:#e2e8f0;border:1px solid #334155;"
              "border-radius:5px;padding:6px 10px;text-decoration:none;")
    ext_links = "".join(
        f'      <a href="{_h.escape(pg.route.lstrip("/"))}" style="{_style}">'
        f'{_h.escape(pg.menu_label)} &rarr;</a>\n'
        for pg in dash_ext.pages() if pg.menu_label)
    menu = (TOOLS_MENU_TEMPLATE.replace("__OPTS__", opts)
            .replace("__INTERNAL_LINKS__", internal)
            .replace("__EXT_LINKS__", ext_links))
    if dash_env.flag("DASH_PUBLIC"):
        # Owner ruling 2026-09-08 (typed to ARIA, applied to the droplet snapshot first): a public
        # copy keeps only what means something without a live stack -- OPEN WORK, ARCHITECTURE,
        # BUNDLES and the bundle picker. START, DASHBOARD and REPLAY are dropped from the MENU; the
        # routes still answer by URL. (3D SCENE is not listed: this set removed that link outright.)
        # A line filter rather than a template rewrite, so a new link cannot silently inherit
        # public visibility -- it has to be added here to be hidden, and the test names the set.
        # These match the RELATIVE hrefs above. If a link's spelling changes this list must change
        # in the SAME edit: a stale entry silently re-exposes a link the owner ordered removed and
        # nothing errors. test_routes asserts both halves.
        dead = ('href="./"', 'href="dash"', 'href="replay"')
        menu = "\n".join(ln for ln in menu.splitlines()
                          if not any(d in ln for d in dead))
    return menu


TOOLS_MENU_TEMPLATE = """
<div id="toolsMenu" style="position:fixed;right:12px;bottom:12px;z-index:99999;
     font:600 11px ui-monospace,monospace;text-align:right;">
  <div id="toolsPanel" hidden style="margin-bottom:6px;background:#0b1220;border:1px solid #334155;
       border-radius:8px;padding:8px;min-width:250px;box-shadow:0 6px 24px rgba(0,0,0,.5);">
    <div style="color:#64748b;margin-bottom:6px;letter-spacing:.05em;">LOAD BUNDLE &mdash; switches to replay</div>
    <div style="display:flex;gap:5px;margin-bottom:9px;">
      <select id="bundlePick" style="flex:1;min-width:0;background:#0f172a;color:#e2e8f0;
              border:1px solid #334155;border-radius:5px;padding:4px;font:inherit;">__OPTS__</select>
      <button onclick="loadPickedBundle()" style="background:#0e2537;color:#38bdf8;
              border:1px solid #38bdf8;border-radius:5px;padding:4px 9px;cursor:pointer;font:inherit;">LOAD</button>
    </div>
    <div id="bundleMsg" style="color:#94a3b8;margin-bottom:9px;white-space:normal;"></div>
    <div style="display:flex;flex-direction:column;gap:5px;">
      <a href="./"        style="background:#1e293b;color:#e2e8f0;border:1px solid #334155;border-radius:5px;padding:6px 10px;text-decoration:none;">START &rarr;</a>
      <a href="dash"     style="background:#1e293b;color:#e2e8f0;border:1px solid #334155;border-radius:5px;padding:6px 10px;text-decoration:none;">DASHBOARD &rarr;</a>
__EXT_LINKS__      <a href="replay"   style="background:#1e293b;color:#e2e8f0;border:1px solid #334155;border-radius:5px;padding:6px 10px;text-decoration:none;">REPLAY &rarr;</a>
      <a href="bundles"  style="background:#1e293b;color:#e2e8f0;border:1px solid #334155;border-radius:5px;padding:6px 10px;text-decoration:none;">BUNDLES &rarr;</a>
__INTERNAL_LINKS__      <button id="rvizLaunchBtn" onclick="startRviz()" hidden
              style="background:#1e293b;color:#e2e8f0;border:1px solid #334155;border-radius:5px;
                     padding:6px 10px;cursor:pointer;font:inherit;">OPEN RVIZ</button>
      <div id="rvizLaunchMsg" style="color:#94a3b8;white-space:normal;"></div>
    </div>
  </div>
  <button id="toolsToggle" onclick="toggleTools()"
          style="background:#1e293b;color:#e2e8f0;border:1px solid #334155;border-radius:6px;
                 padding:7px 12px;cursor:pointer;font:inherit;">&#9776; TOOLS</button>
</div>
<script>
  function toggleTools() {
    var p = document.getElementById('toolsPanel');
    p.hidden = !p.hidden;
    try { localStorage.setItem('toolsOpen', p.hidden ? '0' : '1'); } catch (e) {}
  }
  (function () {
    try { if (localStorage.getItem('toolsOpen') === '1') document.getElementById('toolsPanel').hidden = false; } catch (e) {}
    var local = ['localhost', '127.0.0.1', '::1', ''];
    if (local.indexOf(location.hostname) !== -1) document.getElementById('rvizLaunchBtn').hidden = false;
  })();
  async function loadPickedBundle() {
    var sel = document.getElementById('bundlePick'), msg = document.getElementById('bundleMsg');
    if (!sel || !sel.value) { msg.textContent = 'no bundle selected'; return; }
    msg.style.color = '#94a3b8'; msg.textContent = 'loading ' + sel.value + '...';
    try {
      var r = await fetch('/load_bundle?name=' + encodeURIComponent(sel.value), { method: 'POST' });
      var d = await r.json();
      if (d.ok) { msg.style.color = '#10b981'; msg.textContent = 'now serving ' + d.bundle + ' -- reloading';
        // GA-380: the twin of the start page's navigation, one function away, and missed the first
        // time -- which is the copy warning this file already carries. Relative, so it needs no helper.
        setTimeout(function () { location.href = 'dash'; }, 700); }
      else { msg.style.color = '#f87171'; msg.textContent = 'failed: ' + (d.why || ('HTTP ' + r.status)); }
    } catch (e) { msg.style.color = '#f87171'; msg.textContent = 'failed: ' + e.message; }
  }
  async function startRviz() {
    var btn = document.getElementById('rvizLaunchBtn'), msg = document.getElementById('rvizLaunchMsg');
    btn.disabled = true; msg.style.color = '#94a3b8'; msg.textContent = 'starting RViz...';
    try {
      var r = await fetch('/start_rviz', { method: 'POST' });
      var d = await r.json();
      if (d.started) { msg.style.color = '#10b981'; msg.textContent = 'RViz starting (pid ' + d.pid + ') on the SERVER display. Log: ' + d.log; }
      else if (d.already_running) { msg.style.color = '#eab308'; msg.textContent = d.reason; }
      else { msg.style.color = '#f87171'; msg.textContent = 'failed: ' + (d.reason || ('HTTP ' + r.status)); }
    } catch (e) { msg.style.color = '#f87171'; msg.textContent = 'failed: ' + e.message; }
    btn.disabled = false;
  }
</script>
"""


def _bundle_names(limit: int = 60):
    """Run directory names, newest first, for the dashboard's bundle picker."""
    try:
        runs = [d for d in RUNS_ROOT.iterdir() if d.is_dir() and not d.is_symlink()]
    except OSError:
        return []
    return [d.name for d in sorted(runs, key=lambda d: d.name, reverse=True)[:limit]]


def with_tools_menu(html: str) -> str:
    """Append the TOOLS menu to any page this dashboard serves.

    It was injected into `/` alone, so every link INSIDE the menu -- 3D scene, open work,
    replay, bundles -- led to a page with no menu and no way back but the browser's Back
    button. That is the "bottom right buttons disappeared" report: on those four pages
    they had. Appended rather than inserted after <body>, because the sub-pages are
    fragments with no <body> tag, and `position: fixed` does not care where in the
    document its element sits.
    """
    if 'id="toolsMenu"' in html:
        return html
    return html + _tools_menu_html(_bundle_names())


# ---------------------------------------------------------------------------------------
# THE START PAGE AND THE RUN LAUNCHER
#
# ONE dashboard, one process, three states -- START (nothing chosen yet), LIVE (a bridge is
# answering) and REPLAY (a bundle is pinned). The chooser is NOT a second application: it is
# served by this file, it reuses this file's bundle list and the same TOOLS menu, and it hands
# off to the SAME dashboard page. `/dash` is that page; `/` is where you land.
#
# Splitting it into its own service was the alternative and it was rejected: a second port and
# a second process to keep alive, to do what this always-running one already does -- it already
# lists bundles and already switches which one it serves.
# ---------------------------------------------------------------------------------------

# The checkout the runs directory sits in, used only to DISPLAY default paths in the
# launcher form. Derived, never a literal, so it names no deployment.
RUNS_PARENT = RUNS_ROOT.parent
LIVE_RUN = GRAPH_API_ROOT / "lost3dsg/test/live_run.sh"
LAUNCH_LOG_DIR = Path(tempfile.gettempdir()) / "found-launcher"

# The four the script's own `case` statement accepts. Anything else exits 1 before it starts,
# so the form offers exactly these rather than a free-text box that fails a minute later.
SCENES = ["hm3d_00861", "hm3d_00337", "hm3d_00770", "mp3d_17DRP"]

# EVERY variable live_run.sh reads, grouped, each with THE SCRIPT'S OWN DEFAULT as its
# placeholder. Read out of the script rather than remembered: a form that offers a stale
# default is worse than one that offers none, because it looks authoritative.
#
# The provenance variables (SRC_SHA, KB_SHA ...) are deliberately absent. They look
# like settings -- `${SRC_SHA:-}` -- but the script COMPUTES them from the source trees and
# asserts them before writing run_metadata.json. A form that let you type one would let you
# label a run with a commit it was not built from.
#
#   kind: "bool"   0/1 in the script, so 0/1 on the wire -- rendered as a checkbox
#         "secret" never pre-filled, never echoed back, never written to the launch log
#         "choice:a|b"
#         anything else is a text field
RUN_SETTINGS = [
    ("SCENE", "which scene, and the paths it resolves to", [
        ("HABITAT_SCENE", "", "text", "full .glb path; overrides the scene picked above"),
        ("HABITAT_DATASET", "", "text", "scene_dataset_config.json; overrides the picker"),
        ("HM3D_ROOT", "/DATA/habitat_matterport/hm3d_example", "text", "where the HM3D scenes live"),
        ("MP3D_ROOT", "/DATA/habitat_matterport/versioned_data/mp3d_example_scene_1.1",
         "text", "where the MP3D scenes live"),
    ]),
    ("FEED", "the habitat feed: camera, motion, and what the window draws", [
        ("FEED_WIDTH", "1280", "text", "render width"),
        ("FEED_HEIGHT", "960", "text", "render height"),
        ("FEED_HFOV", "90", "text", "horizontal field of view, degrees"),
        ("FEED_FPS", "3", "text", "frames per second published to the stack"),
        ("FEED_SEED", "7", "text", "walk seed -- same seed, same route"),
        ("FEED_WALK", "6", "text", "walk speed"),
        ("FEED_DWELL", "0", "text", "seconds to stand still at each waypoint"),
        ("FEED_SHOW", "1", "bool", "open the habitat window on DISPLAY"),
        ("FEED_OVERLAY", "1", "bool", "draw the detection overlay in that window"),
        ("FEED_GT_SEMANTIC", "0", "bool", "publish ground-truth semantics instead of detections"),
        ("FEED_TEST_TOUR", "0", "bool", "scripted tour instead of the random walk"),
        ("FEED_TEST_TOUR_SCAN", "12", "text", "scan steps per tour stop"),
        ("DISPLAY", ":1", "text", "X display the habitat window opens on"),
    ]),
    ("MAP", "rtabmap: whether this run maps, for how long, and from where", [
        ("MAPPING_ONLY", "0", "bool", "map and publish, no perception"),
        ("FEED_MAPPING_SECONDS", "150", "text", "how long to map (900 when MAPPING_ONLY=1)"),
        ("FEED_SPAWN_FLOOR", "", "text", "spawn height; blank lets the navmesh choose"),
        ("RTABMAP_LOCALIZE_DB", "", "text", "localise against this .db instead of mapping"),
    ]),
    ("KEYS", "left blank they are inherited from this server's environment", [
        ("REGOLO_API_KEY", "", "secret", "a key here selects regolo_config.yaml; none selects the offline smoke config"),
        ("OPENAI_API_KEY", "", "secret", "falls back to REGOLO_API_KEY when unset"),
    ]),
    ("RUNTIME", "image, caches and the paths the container mounts", [
        ("CFG_NAME", "regolo_config.yaml", "choice:regolo_config.yaml|smoke_config.yaml",
         "chosen by the key above unless set here"),
        ("GRAPH_API_CONFIG", "", "text", "a config path outside the test directory"),
        ("IMAGE_TAG", "graphapi-run:humble", "text", "the container image to run"),
        ("SAM_MODEL_DIR", "/DATA/models/efficientvit_sam", "text", "SAM weights"),
        ("HF_CACHE", "<runs parent>/.hf_cache", "text", "per-checkout hugging-face cache"),
        ("HF_SHARED_CACHE", "/DATA/huggingface_cache", "text", "shared hugging-face cache"),
        ("GRAPH_API_RUNS_DIR", "<runs parent>/runs", "text", "where the bundle is archived"),
    ]),
]

# An extension's own policy vocabulary is ITS declaration, not a group this file carries: the
# repository would otherwise name the package that extends it. Appended after the generic groups so
# the launcher form reads stack-first, extension-last, and an absent extension simply adds nothing.
RUN_SETTINGS = RUN_SETTINGS + dash_ext.env_groups()

_SETTING_NAMES = {f[0] for _, _, fields in RUN_SETTINGS for f in fields}
_SECRET_NAMES = {f[0] for _, _, fields in RUN_SETTINGS for f in fields if f[2] == "secret"}
_ENV_NAME = re.compile(r"^[A-Z][A-Z0-9_]{1,60}$")

# The launched run, as this process knows it. `pid` is the process GROUP leader, because
# live_run.sh is started in its own session -- see _start_run for why that matters to stopping.
RUN_PROC = {"pid": None, "scene": None, "log": None, "started": None}


def _pid_alive(pid) -> bool:
    if not pid:
        return False
    try:
        os.kill(int(pid), 0)
    except OSError:
        return False
    return True


def _feed_pid():
    """The pid of a habitat feed already running on this machine, or None.

    Checked BEFORE launching, and it is not paranoia: two feeds fight over the ROS graph, the
    control port and the DISPLAY, and the second one's bundle is the one that looks broken.
    Matched on argv[0:3] so a shell that merely mentions the script in an argument is not it.
    """
    try:
        entries = list(Path("/proc").iterdir())
    except OSError:
        return None
    for d in entries:
        if not d.name.isdigit():
            continue
        try:
            argv = (d / "cmdline").read_bytes().split(b"\0")
        except OSError:
            continue
        if any(b"habitat_feed_host.py" in a for a in argv[:3]):
            return int(d.name)
    return None


CONTAINER = "graphapi_live"


def _container_up() -> bool:
    """Is the run's container up. live_run.sh refuses to start over one (fixed name, fixed
    ports), and its EXIT trap does not stop it: the trap kills the feed and archives the
    bundle, and leaves `docker run` to the signal. So the container can outlive the run."""
    try:
        out = subprocess.run(["docker", "ps", "-q", "-f", f"name=^{CONTAINER}$"],
                             capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.SubprocessError):
        return False
    return bool(out.stdout.strip())


def _stop_container_after(pid, grace=20):
    """Give the SIGINT `grace` seconds to bring the stack down on its own, then stop the
    container if it is still up. Measured 2026-09-03: after SIGINT to the group the container
    was still up 8 s later and was stopped by hand; the next launch would have been refused."""
    for _ in range(grace):
        if not _pid_alive(pid) and not _container_up():
            return
        time.sleep(1)
    if _container_up():
        subprocess.run(["docker", "stop", "-t", "15", CONTAINER],
                       capture_output=True, timeout=60)


def _clean_env(raw):
    """Validate the submitted settings. Returns (env, rejected).

    Names must look like environment variables and values must be one line. The values reach
    `Popen` through `env=`, never through a shell string, so this is not the only thing between
    a value and a shell -- but a value carrying a newline would still corrupt the launch log it
    is echoed into, and a name that is not a name can only be a mistake.
    """
    env, rejected = {}, []
    for name, value in (raw or {}).items():
        name = str(name).strip()
        value = "" if value is None else str(value)
        if not _ENV_NAME.match(name):
            rejected.append(f"{name!r}: not an environment variable name")
            continue
        if "\n" in value or "\0" in value:
            rejected.append(f"{name}: value spans more than one line")
            continue
        if len(value) > 4096:
            rejected.append(f"{name}: value is {len(value)} characters")
            continue
        if value == "":
            continue          # blank means "leave it to the script", not "set it to empty"
        env[name] = value
    return env, rejected


def _launch_log_tail(path, lines=400):
    try:
        raw = Path(path).read_bytes()
    except OSError as exc:
        return [f"[launcher] cannot read {path}: {exc}"]
    return raw.decode("utf-8", "replace").splitlines()[-lines:]


START_PAGE_CSS = """
<style>
  :root { --bg:#080d18; --card:#0e1626; --rule:#1e2b44; --ink:#e2e8f0; --dim:#8296b4;
          --faint:#4a5c78; --accent:#38bdf8; --ok:#10b981; --bad:#f87171; --warn:#eab308; }
  body { background:var(--bg); color:var(--ink); margin:0;
         font:13px/1.55 ui-monospace,SFMono-Regular,Menlo,monospace; }
  header { padding:14px 20px; border-bottom:1px solid var(--rule);
           display:flex; align-items:baseline; gap:14px; flex-wrap:wrap; }
  h1 { font-size:15px; margin:0; letter-spacing:.08em; text-transform:uppercase; }
  .sub { color:var(--dim); font-size:11px; }
  .badge { font-size:10px; letter-spacing:.08em; padding:2px 8px; border-radius:10px;
           border:1px solid var(--rule); color:var(--dim); text-transform:uppercase; }
  .badge.live { color:var(--ok); border-color:var(--ok); }
  .badge.replay { color:var(--accent); border-color:var(--accent); }
  main { display:grid; grid-template-columns:minmax(320px,1fr) minmax(420px,1.35fr);
         gap:18px; padding:18px 20px 60px; align-items:start; }
  @media (max-width:1000px) { main { grid-template-columns:1fr; } }
  section { background:var(--card); border:1px solid var(--rule); border-radius:10px;
            padding:14px 16px; }
  h2 { font-size:11px; margin:0 0 10px; letter-spacing:.1em; color:var(--dim);
       text-transform:uppercase; font-weight:600; }
  .runs { max-height:56vh; overflow:auto; margin:-4px -6px 0; }
  .run { display:flex; justify-content:space-between; align-items:center; gap:10px;
         padding:7px 10px; border-radius:6px; cursor:pointer; border:1px solid transparent; }
  .run:hover { background:#132038; border-color:var(--rule); }
  .run .n { color:var(--ink); }
  .run .m { color:var(--faint); font-size:11px; white-space:nowrap; }
  /* GA-398: a run that cannot be replayed is marked on the row rather than discovered by
     clicking it and finding a player that does nothing. AMBER, not red: an empty bundle is not
     an error, and one with no frames may still hold the evidence a figure is quoted from. */
  .run.empty { border-style:dashed; }
  .run.empty .n { color:var(--dim); }
  .run.empty .m { color:#eab308; }
  .run.cur { border-color:var(--accent); }
  .run.cur .n { color:var(--accent); }
  label { display:block; color:var(--dim); font-size:11px; margin:0 0 3px; }
  .hint { color:var(--faint); font-size:10px; margin:2px 0 0; }
  input[type=text], input[type=password], select {
    width:100%; box-sizing:border-box; background:#0b1220; color:var(--ink);
    border:1px solid var(--rule); border-radius:5px; padding:5px 7px; font:inherit; }
  input::placeholder { color:var(--faint); }
  .grid { display:grid; grid-template-columns:repeat(auto-fill,minmax(230px,1fr)); gap:10px 14px; }
  details { border-top:1px solid var(--rule); padding:10px 0 4px; }
  details > summary { cursor:pointer; color:var(--ink); font-size:11px; letter-spacing:.08em;
                      text-transform:uppercase; list-style:none; display:flex;
                      justify-content:space-between; align-items:center; }
  details > summary::-webkit-details-marker { display:none; }
  details > summary .g { color:var(--faint); font-size:10px; text-transform:none;
                         letter-spacing:0; font-weight:400; }
  details[open] > summary { margin-bottom:10px; }
  .chk { display:flex; align-items:center; gap:7px; color:var(--ink); }
  .chk input { accent-color:var(--accent); }
  button { background:#0e2537; color:var(--accent); border:1px solid var(--accent);
           border-radius:6px; padding:7px 14px; cursor:pointer; font:inherit; font-weight:600; }
  button.plain { background:#1e293b; color:var(--ink); border-color:var(--rule); }
  button.stop { background:#2a1114; color:var(--bad); border-color:var(--bad); }
  button:disabled { opacity:.45; cursor:default; }
  .bar { display:flex; gap:9px; align-items:center; margin-top:14px; flex-wrap:wrap; }
  pre.log { background:#060a12; border:1px solid var(--rule); border-radius:6px;
            padding:10px; margin:12px 0 0; max-height:34vh; overflow:auto;
            font-size:11px; color:#b6c6de; white-space:pre-wrap; }
  a.link { color:var(--accent); text-decoration:none; }
</style>
"""


def _bundle_tag(name):
    """(short tag, long title) for a bundle -- what it holds, in the reader's words.

    GA-398, owner 2026-09-08: a run with no frames offered a player that did nothing, and the
    picker gave no way to tell an empty launch from the archive that carries the citable evidence.
    The counts come from bundle_index, which CACHES them outside the bundle (rule 12), so this
    costs one cached read per row rather than a directory walk.

    "No frames" is NOT "empty", and the tag must not say so: the 26 August archives have no
    frames because that era wrote none, and one of them is the only citable run in the project.
    """
    try:
        c = _load_bundle_index().describe(name)
    except Exception as exc:                       # noqa: BLE001 - a listing must not die on one bad row
        return "unreadable", f"{type(exc).__name__}: {exc}"
    # WHOSE RUN IS THIS. Bundles from two machines can sit in one picker only if each says which
    # machine made it; before 2026-09-10 none did, so an older bundle reads "machine not recorded"
    # rather than being assumed local. Prefixed, so it is the first thing in the row rather than a
    # detail at the end of a title nobody hovers.
    _m = c.get("machine")
    _host = _socket.gethostname()
    if _m and _m != _host:
        _mach_short, _mach_long = f"[{_m}] ", f"recorded on {_m}, not this machine ({_host}). "
    elif _m:
        _mach_short, _mach_long = "", f"recorded on this machine ({_m}). "
    else:
        _mach_short, _mach_long = "", "machine not recorded (bundle predates the field). "
    frames, dets = c.get("frames") or 0, c.get("detections") or 0
    kinds = c.get("decisions") or {}
    decisions = sum(v for v in kinds.values() if isinstance(v, int))
    if frames:
        # `replayable` is the INDEX's own verdict (frames AND detections), not a second rule here:
        # a run with frames but no detections can be stepped through and shows no boxes, and saying
        # "replayable" of it would be this file disagreeing with the page that lists it.
        ok = c.get("replayable")
        return (_mach_short + f"{frames} frames \u00b7 {dets} detections",
                _mach_long + f"{frames} frames, {dets} detections, {decisions} decisions, {c.get('size')} -- "
                + ("replayable" if ok else (c.get("why_not") or "not replayable")))
    if decisions:
        return (_mach_short + "NO FRAMES \u00b7 %d decisions" % decisions,
                _mach_long + f"no frames -- this run recorded none, but it holds {decisions} decisions and "
                f"{c.get('objects', 0)} objects. The player cannot step through it; its evidence is intact.")
    return (_mach_short + "EMPTY \u00b7 nothing recorded",
            _mach_long + "no frames, no detections and no decisions: an aborted launch or a mapping-only run, "
            "which writes no perception output by design. Nothing here can be replayed or quoted.")


def _start_page_html(bundles, current, mode, why):
    """The initial window: pick a recorded run, or configure and launch a new one.

    EVERY variable live_run.sh reads is on the form, grouped and collapsed. The alternative --
    a short list of "the ones that matter" -- was rejected by the owner, and the objection is
    sound: which ones matter is a property of the experiment, not of the launcher, so a
    launcher that decides for you is a launcher you have to leave to change one field.

    Nothing here is submitted as a shell string. The scene is checked against the script's own
    four, and the settings go to `Popen(env=)` as a dict.
    """
    import html as _h

    rows = []
    for b in bundles:
        cur = " cur" if current and b == current else ""
        tag, title = _bundle_tag(b)
        empty = " empty" if tag.startswith(("EMPTY", "NO FRAMES")) else ""
        rows.append(f'<div class="run{cur}{empty}" onclick="openBundle(this.dataset.b)" '
                    f'data-b="{_h.escape(b)}" title="{_h.escape(title)}">'
                    f'<span class="n">{_h.escape(b)}</span>'
                    f'<span class="m">{_h.escape(tag)}</span></div>')

    groups = []
    for name, blurb, fields in RUN_SETTINGS:
        cells = []
        for var, default, kind, hint in fields:
            lab = f'<label for="f_{var}">{var}</label>'
            if kind == "bool":
                on = "checked" if default == "1" else ""
                cells.append(f'<div><div class="chk"><input type="checkbox" id="f_{var}" '
                             f'data-var="{var}" data-bool="1" {on}><span>{var}</span></div>'
                             f'<div class="hint">{_h.escape(hint)}</div></div>')
            elif kind.startswith("choice:"):
                opts = "".join(f'<option{" selected" if o == default else ""}>{_h.escape(o)}</option>'
                               for o in kind.split(":", 1)[1].split("|"))
                cells.append(f'<div>{lab}<select id="f_{var}" data-var="{var}">{opts}</select>'
                             f'<div class="hint">{_h.escape(hint)}</div></div>')
            else:
                # SECRETS ARE NEVER PRE-FILLED, not even from this server's own environment.
                # A key rendered into a page is a key in the browser cache, in the devtools
                # network log and in any screenshot of the launcher.
                typ = "password" if kind == "secret" else "text"
                inherited = (" - inherited" if kind == "secret" and os.environ.get(var) else "")
                ph = "" if kind == "secret" else _h.escape(default)
                cells.append(f'<div>{lab}<input type="{typ}" id="f_{var}" data-var="{var}" '
                             f'placeholder="{ph}" autocomplete="off">'
                             f'<div class="hint">{_h.escape(hint)}{inherited}</div></div>')
        groups.append(f'<details><summary>{name} <span class="g">{_h.escape(blurb)}</span></summary>'
                      f'<div class="grid">{"".join(cells)}</div></details>')

    scene_opts = "".join(f'<option>{s}</option>' for s in SCENES)
    badge = f'<span class="badge {mode}">{mode}</span>'
    # GA-380: relative, like the menu's links -- an absolute href navigates to the SITE ROOT behind
    # a path prefix. This one sits in a different function from the menu and survived two sweeps.
    resume = (f'<a class="link" href="dash">resume {_h.escape(current)} &rarr;</a>'
              if current else '<span class="sub">no bundle selected</span>')

    _brand = dash_ext.brand()
    return (f"<title>{_h.escape(_brand)} &middot; start</title>{START_PAGE_CSS}"
            f'<header><h1>{_h.escape(_brand)}</h1>{badge}<span class="sub">{_h.escape(why or "")}</span>'
            f'<span style="flex:1"></span>{resume}</header>'
            '<main>'
            '<section><h2>Open a recorded run</h2>'
            f'<div class="runs">{"".join(rows) or "<div class=sub>no runs in " + str(RUNS_ROOT) + "</div>"}</div>'
            '<div id="openMsg" class="sub" style="margin-top:10px"></div></section>'
            # Owner 2026-09-08 (applied first to the droplet snapshot by ARIA, carried here): a
            # PUBLIC deployment cannot launch anything, so the launch section is not offered there.
            # Gated on DASH_PUBLIC, not on the mode: a dashboard started BEFORE a run is in
            # replay mode and is exactly where LAUNCH RUN is needed (review 2026-09-08).
            + ("" if dash_env.flag("DASH_PUBLIC") else
               '<section><h2>Start a new run</h2>'
               f'<div><label for="f_scene">scene</label><select id="f_scene">{scene_opts}</select>'
               '<div class="hint">the four live_run.sh accepts; override the paths under SCENE</div></div>'
               f'{"".join(groups)}'
               '<div class="bar"><button id="go" onclick="startRun()">LAUNCH RUN</button>'
               '<button class="plain stop" id="stopBtn" onclick="stopRun()" hidden>STOP (SIGINT)</button>'
               '<span id="runMsg" class="sub"></span></div>'
               '<pre class="log" id="runLog" hidden></pre></section>')
            + '</main>'
            + START_PAGE_JS)


START_PAGE_JS = """
<script>
async function openBundle(name) {
  var msg = document.getElementById('openMsg');
  msg.textContent = 'loading ' + name + '...';
  try {
    var r = await fetch('/load_bundle?name=' + encodeURIComponent(name), {method:'POST'});
    var d = await r.json();
    if (d.ok) { msg.textContent = 'opening ' + d.bundle;
      // GA-380: same prefix problem, at a site outside the injected script (see PFX there).
      location.href = location.pathname.replace(new RegExp('/(dash|replay|scene3d|bundles|arch|blockers)/?$'), '').replace(new RegExp('/+$'), '') + '/dash'; }
    else { msg.style.color = '#f87171'; msg.textContent = 'failed: ' + (d.why || ('HTTP ' + r.status)); }
  } catch (e) { msg.style.color = '#f87171'; msg.textContent = 'failed: ' + e.message; }
}
function collectEnv() {
  var env = {};
  document.querySelectorAll('[data-var]').forEach(function (el) {
    if (el.dataset.bool) {
      // A checkbox always has a value, so sending it always would override the script's
      // default for every box the person never touched. Sent only when it DIFFERS.
      var now = el.checked ? '1' : '0';
      if (now !== (el.defaultChecked ? '1' : '0')) env[el.dataset.var] = now;
    } else if (el.tagName === 'SELECT') {
      // Same rule as the checkbox: the script's own default is a BRANCH (CFG_NAME depends
      // on MAPPING_ONLY), so an untouched select must send nothing, not its first option.
      if (!el.options[el.selectedIndex].defaultSelected) env[el.dataset.var] = el.value;
    } else if (el.value !== '' && el.value !== el.placeholder) {
      env[el.dataset.var] = el.value;
    }
  });
  return env;
}
async function startRun() {
  var msg = document.getElementById('runMsg'), go = document.getElementById('go');
  go.disabled = true; msg.style.color = '#8296b4'; msg.textContent = 'starting...';
  try {
    var r = await fetch('/start_run', {method:'POST', headers:{'Content-Type':'application/json'},
      body: JSON.stringify({scene: document.getElementById('f_scene').value, env: collectEnv()})});
    var d = await r.json();
    if (d.started) { msg.style.color = '#10b981';
      msg.textContent = 'run ' + d.scene + ' started (pid ' + d.pid + ')'; pollRun(); }
    else { msg.style.color = '#f87171';
      msg.textContent = 'not started: ' + (d.why || ('HTTP ' + r.status));
      (d.rejected || []).forEach(function (x) { msg.textContent += ' | ' + x; }); }
  } catch (e) { msg.style.color = '#f87171'; msg.textContent = 'failed: ' + e.message; }
  go.disabled = false;
}
async function stopRun() {
  var msg = document.getElementById('runMsg');
  msg.style.color = '#eab308'; msg.textContent = 'sending SIGINT (the map publishes on the trap)...';
  try { var d = await (await fetch('/stop_run', {method:'POST'})).json();
    msg.textContent = d.stopped ? 'SIGINT sent to process group ' + d.pgid : ('not stopped: ' + d.why);
  } catch (e) { msg.textContent = 'failed: ' + e.message; }
}
async function pollRun() {
  var log = document.getElementById('runLog'), stop = document.getElementById('stopBtn');
  var msg = document.getElementById('runMsg');
  if (!log) return;              // no launch section on this page (public deployment): nothing to poll
  try {
    var s = await (await fetch('/run_status')).json();
    stop.hidden = !s.alive;
    if (s.log) {
      var t = await (await fetch('/run_log')).json();
      log.hidden = false; log.textContent = (t.lines || []).join('\\n');
      log.scrollTop = log.scrollHeight;
    }
    if (!s.alive && s.pid) { msg.style.color = '#eab308'; msg.textContent = 'run ended -- log below'; }
  } catch (e) {}
  setTimeout(pollRun, 3000);
}
pollRun();
</script>
"""


def _install_launcher(app, index_fn, bundles_fn, current_fn):
    """Register the start page, the dashboard and the run-launch routes on `app`.

    Called by BOTH app builders, so live mode and replay mode land on the same page with the
    same components. In live mode it MUST be called before the catch-all proxy: the first
    matching route wins, and a catch-all registered first sends `/dash` to the bridge.

    `/` is the start page and `/dash` is the dashboard. The dashboard did not move for the
    sake of moving: with `/` serving it, there was no state in which the dashboard was not
    already committed to a bundle, so "which run do you want" had nowhere to be asked.
    """
    from fastapi import Request
    from fastapi.responses import FileResponse, HTMLResponse, JSONResponse

    def _local(request) -> bool:
        return (request.client.host if request and request.client else "") in (
            "127.0.0.1", "::1", "localhost")

    @app.get("/", response_class=HTMLResponse)
    def _start_page():
        try:
            current = current_fn()
        except OSError:
            current = None
        return HTMLResponse(with_tools_menu(
            _start_page_html(bundles_fn(), current, MODE["mode"], MODE["why"])))

    @app.get("/dash", response_class=HTMLResponse)
    def _dash():
        return index_fn()

    # ---- THE 3D SCENE, IN BOTH MODES ---------------------------------------------------
    # These three lived in `build_app` until 2026-09-04, which meant /scene3d, /scene_mesh
    # and /vendor EXISTED ONLY IN REPLAY. In live mode they fell through to the catch-all
    # proxy and the bridge answered 404 for all three -- so the one mode that has a camera
    # to draw was the one mode that could not open the page. Registering them here puts
    # them on both apps from ONE definition, and in live mode `_install_launcher` runs
    # before the catch-all, which is the order that decides who answers.
    #
    # ponytail: `bundle` is resolved per request through `resolve_bundle`, so
    # `/scene3d?bundle=latest` follows the newest run with a graph. Ceiling: with no
    # `?bundle=` the live app answers for the directory it was STARTED on, because
    # `current_fn` in build_live_app captures it once and live mode does not run the
    # follower. Open the live page as `/scene3d?bundle=latest`. Upgrade path is one call
    # to `_follow_latest()` in `build_live_app` and a re-resolving `current_fn`.
    def _bundle_name(spec):
        if not spec:
            return current_fn()
        try:
            got = resolve_bundle(spec)
        except OSError:
            return spec
        return got.name if got is not None else spec

    # /blockers AND /arch LIVED IN build_app UNTIL 2026-09-07, which meant they existed only in
    # replay mode. The dashboard came up in live mode on 2026-09-06 23:45 (a bridge was answering
    # on :8085) and both pages answered 503 -- the exact shape /scene3d had until 2026-09-04. The
    # register is not a property of a run; these two are served whatever the mode.
    # PAGES AN EXTENSION SUPPLIES. /blockers and /arch were registered here by name, which meant
    # this file named the package that extends it. They are now declared by the extension and
    # installed in a loop; with no extension the dashboard simply has fewer pages, which is the
    # copy this repository ships. Registered whatever the mode, for the reason recorded above.
    for _page in dash_ext.pages():
        _path = "/" + _page.route.lstrip("/")
        _name = f"ext_{_page.route.strip('/').replace('/', '_')}"
        if _page.raw_response:
            # registered as it stands: the framework reads ITS signature, so a page with query
            # parameters and its own status codes keeps both
            app.get(_path, name=_name)(_page.render)
            continue

        def _make(pg):
            def _render():
                return HTMLResponse(with_tools_menu(pg.render(f" · viewing {current_fn()}")))
            return _render
        app.get(_path, response_class=HTMLResponse, name=_name)(_make(_page))

    # /bundles AND /start_rviz likewise lived in build_app until 2026-09-07: in the live app the
    # tools menu linked /bundles and posted /start_rviz, both fell through the catch-all, and
    # the bridge answered 404 for each (rule 19: the control looked wired). File-backed and
    # host-local, so they belong to both modes.
    @app.get("/bundles", response_class=HTMLResponse)
    def _bundles():
        """Every archived run with its configuration, statistics and map, so a bundle is
        chosen by what it recorded rather than by its timestamp. Cached outside the bundle
        (rule 12) and keyed on the decision file's size, because that file reaches 475 MB."""
        return HTMLResponse(with_tools_menu(_load_bundle_index().page()))

    @app.post("/start_rviz")
    def _start_rviz():
        """Launch RViz via the existing helper. Reports what actually happened.

        One fixed command, no arguments from the request: nothing a caller sends
        reaches a shell. `view_rviz.sh` blocks (it runs docker in the foreground and
        tees to a log), so it is started detached and its output goes to the log.
        """
        if not VIEW_RVIZ.exists():
            return JSONResponse(status_code=500, content={
                "started": False, "reason": f"helper not found at {VIEW_RVIZ}"})
        if shutil.which("docker") is None:
            return JSONResponse(status_code=500, content={
                "started": False, "reason": "docker is not on PATH for this server"})
        try:
            running = subprocess.run(
                ["docker", "ps", "--filter", f"name={RVIZ_CONTAINER}", "--format", "{{.Names}}"],
                capture_output=True, text=True, timeout=10)
            if RVIZ_CONTAINER in running.stdout:
                return {"started": False, "already_running": True,
                        "reason": f"container {RVIZ_CONTAINER} is already up",
                        "log": str(RVIZ_LOG)}
        except (subprocess.SubprocessError, OSError) as exc:
            # Could not tell -- say so rather than starting a second one blindly.
            return JSONResponse(status_code=503, content={
                "started": False,
                "reason": f"could not check for a running RViz: {type(exc).__name__}: {exc}"})
        try:
            RVIZ_LOG.parent.mkdir(parents=True, exist_ok=True)
            handle = open(RVIZ_LOG, "ab")
            proc = subprocess.Popen(
                ["bash", str(VIEW_RVIZ)],
                stdout=handle, stderr=subprocess.STDOUT,
                start_new_session=True, cwd=str(VIEW_RVIZ.parent))
            handle.close()          # the child holds its own copy of the fd
            # Reap it. Without this the wrapper sits as <defunct> for the life of the
            # server -- verified: a real press left `Zs [bash] <defunct>` behind, one
            # per press. A long-lived server that leaks a process per button click is
            # the same shape of defect as a handler that swallows its error: nothing
            # visibly breaks, and the cost only shows up much later.
            threading.Thread(target=proc.wait, daemon=True).start()
        except (OSError, subprocess.SubprocessError) as exc:
            return JSONResponse(status_code=500, content={
                "started": False, "reason": f"{type(exc).__name__}: {exc}"})
        return {"started": True, "pid": proc.pid, "log": str(RVIZ_LOG),
                "note": "RViz opens on the SERVER's display, not the viewer's",
                "display": os.environ.get("DISPLAY", ":1 (helper default)")}

    @app.get("/scene3d", response_class=HTMLResponse)
    def _scene3d(bundle: str = None):
        """A stylized 3D view: measured object boxes over ground-truth walls. GA-230."""
        # Owner 2026-09-08: the scene has no window of its own any more -- it is the dashboard's
        # 3D tab (an iframe of this route, GA-347) with a maximize overlay. Served WITHOUT the
        # tools menu, which used to nest a second TOOLS button inside the tab.
        return HTMLResponse(_load_scene3d().page(_bundle_name(bundle)))

    @app.get("/scene3d_data")
    def _scene3d_data(bundle: str = None):
        """The same payload /scene3d inlines, as JSON, so the page can POLL it.

        A run in progress rewrites persistent_perception.json periodically, so the object
        list GROWS while the page is open. Before this route the page substituted one
        `__PAYLOAD__` at build time and the scene was a snapshot for the life of the tab.
        Same function, `scene3d.scene_payload` -- not a second reader, so the polled scene
        and the first paint cannot disagree about what a box is.
        """
        return JSONResponse(_load_scene3d().scene_payload(_bundle_name(bundle)))

    @app.get("/scene_mesh")
    def _scene_mesh(bundle: str = None):
        """The scene mesh (GLB) for a bundle, streamed from disk.

        FileResponse streams the file in chunks and answers Range requests; the 38 MB GLB is
        never read into this process. The scene id comes from the bundle's own metadata via
        scene3d.mesh_path -- an unknown scene is a 404 that NAMES the scene, not a fallback
        to some other building's mesh.
        """
        p, scene = _load_scene3d().mesh_path(_bundle_name(bundle))
        if p is None:
            return JSONResponse(status_code=404, content={
                "error": "no mesh on disk for this bundle",
                "scene": scene,
                "note": "scene id read from run_metadata.json; table is tools/extract_gt.py::SCENES"})
        return FileResponse(str(p), media_type="model/gltf-binary",
                            headers={"Cache-Control": "public, max-age=3600"})

    # /scene3d IS A WebGL PAGE and it needs three.js off this server, because the machine it
    # runs on has no network. r128 is vendored under found/dashboard/vendor_three.
    from fastapi.staticfiles import StaticFiles
    _vendor_dir = Path(__file__).resolve().parent / "vendor_three"
    if (_vendor_dir / "three.module.js").is_file():
        app.mount("/vendor", StaticFiles(directory=str(_vendor_dir)), name="vendor")
    else:
        # SAY SO. Without this the mount is simply absent, every /vendor/* request 404s, the
        # module script never runs, and /scene3d answers 200 with an empty canvas, no status
        # line and nothing in the console that names the cause. A missing dependency that
        # renders as "the scene has nothing in it" is the failure shape this project keeps
        # removing. The files are 1.2 MB of vendored three.js r128 and are UNTRACKED as of
        # 2026-09-04, so a fresh clone hits exactly this.
        print(f"[dash] WARNING: {_vendor_dir}/three.module.js is missing -- /scene3d will not "
              f"render. Restore found/dashboard/vendor_three/ (three.module.js, GLTFLoader.js, "
              f"OrbitControls.js, BasisTextureLoader.js, basis/).", flush=True)

        @app.get("/vendor/{rest:path}")
        def _vendor_missing(rest: str):
            return JSONResponse(status_code=503, content={
                "error": "three.js is not vendored on this machine",
                "missing": str(_vendor_dir / "three.module.js"),
                "why": "/scene3d is a WebGL page and loads three.js from this server, "
                       "because the machine it runs on has no network"})


    # /arch LAYS ITSELF OUT WITH ELK (elkjs 0.12.0, vendored under found/dashboard/vendor_elk)
    # for the same reason three.js is vendored: the machine has no network, and a diagram
    # library fetched from a CDN would make the page that says "here is the machine" the one
    # page that breaks without a route out. Same shape as the three.js block above, same
    # loud 503 when the file is not there.
    _elk_dir = Path(__file__).resolve().parent / "vendor_elk"
    if (_elk_dir / "elk.bundled.js").is_file():
        app.mount("/vendor_elk", StaticFiles(directory=str(_elk_dir)), name="vendor_elk")
    else:
        print(f"[dash] WARNING: {_elk_dir}/elk.bundled.js is missing -- /arch will not lay out. "
              f"Restore found/dashboard/vendor_elk/ (elk.bundled.js from elkjs 0.12.0).", flush=True)

        @app.get("/vendor_elk/{rest:path}")
        def _elk_missing(rest: str):
            return JSONResponse(status_code=503, content={
                "error": "elkjs is not vendored on this machine",
                "missing": str(_elk_dir / "elk.bundled.js"),
                "why": "/arch lays itself out in the browser with ELK and loads it from this "
                       "server, because the machine it runs on has no network"})

    @app.get("/run_status")
    def _run_status():
        """What this process knows about the launched run. NEVER the settings it was given:
        the KEYS group holds API keys, and a status endpoint that echoes them back would put
        them in every poll of a page anyone can open."""
        feed = _feed_pid()
        return JSONResponse({
            "pid": RUN_PROC["pid"], "alive": _pid_alive(RUN_PROC["pid"]),
            "scene": RUN_PROC["scene"], "log": RUN_PROC["log"],
            "started": RUN_PROC["started"], "feed_pid": feed,
            "container_up": _container_up(),
            "script": str(LIVE_RUN), "script_present": LIVE_RUN.is_file()})

    @app.get("/run_log")
    def _run_log(lines: int = 400):
        if not RUN_PROC["log"]:
            return JSONResponse({"lines": [], "why": "no run has been launched from here"})
        return JSONResponse({"lines": _launch_log_tail(RUN_PROC["log"], max(1, min(lines, 5000))),
                             "log": RUN_PROC["log"]})

    @app.post("/start_run")
    async def _start_run(request: Request):
        """Launch live_run.sh with the submitted settings.

        LOOPBACK ONLY. The run opens a habitat window on the SERVER's display and takes the
        server's GPU, ROS graph and control ports; from another machine the person pressing
        the button could not see it, could not stop it and would not know it was theirs.

        A NEW SESSION, deliberately (`start_new_session=True`). Two things follow, both
        wanted: the run survives a restart of this dashboard, and it gets its own process
        group, which is the only way `/stop_run` can deliver SIGINT to the whole stack.
        live_run.sh publishes the map from an EXIT trap, so the documented way to stop it is
        the interrupt -- kill the pid alone and the trap runs while its children keep the
        ports.
        """
        if not _local(request):
            return JSONResponse(status_code=403, content={
                "started": False,
                "why": "runs can only be launched from the machine that would render them"})
        if not LIVE_RUN.is_file():
            return JSONResponse(status_code=503, content={
                "started": False, "why": f"{LIVE_RUN} does not exist"})
        if _pid_alive(RUN_PROC["pid"]):
            return JSONResponse(status_code=409, content={
                "started": False,
                "why": f"a run launched from here is still going (pid {RUN_PROC['pid']})"})
        feed = _feed_pid()
        if feed:
            return JSONResponse(status_code=409, content={
                "started": False,
                "why": (f"a habitat feed is already running (pid {feed}); two feeds fight over "
                        "the ROS graph, the control port and the display")})
        if _container_up():
            return JSONResponse(status_code=409, content={
                "started": False,
                "why": (f"container {CONTAINER} is still up; live_run.sh refuses to start over "
                        f"it. Press STOP, or: docker stop {CONTAINER}")})
        try:
            body = await request.json()
        except ValueError as exc:
            return JSONResponse(status_code=400,
                                content={"started": False, "why": f"bad request body: {exc}"})
        scene = str(body.get("scene") or SCENES[0])
        if scene not in SCENES:
            return JSONResponse(status_code=400, content={
                "started": False, "why": f"unknown scene {scene!r}", "known": SCENES})
        env_in, rejected = _clean_env(body.get("env"))
        if rejected:
            return JSONResponse(status_code=400, content={
                "started": False, "why": "some settings were refused", "rejected": rejected})

        env = dict(os.environ)
        env.update(env_in)
        env.setdefault("DISPLAY", ":1")
        LAUNCH_LOG_DIR.mkdir(parents=True, exist_ok=True)
        stamp = time.strftime("%Y%m%d_%H%M%S")
        log = LAUNCH_LOG_DIR / f"{stamp}_{scene}.log"
        # The settings are recorded at the head of the log, MINUS the secrets: the run needs
        # to be reproducible from what is on disk, and a launcher whose only record of its
        # own arguments is the browser tab that sent them is not.
        shown = {k: ("<set>" if k in _SECRET_NAMES else v) for k, v in sorted(env_in.items())}
        header = (f"[launcher] {stamp} scene={scene}\n"
                  f"[launcher] settings: {json.dumps(shown)}\n"
                  f"[launcher] {LIVE_RUN} {scene}\n")
        try:
            handle = open(log, "wb")
            handle.write(header.encode())
            handle.flush()
            proc = subprocess.Popen(
                ["bash", str(LIVE_RUN), scene], cwd=str(LIVE_RUN.parent), env=env,
                stdout=handle, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                start_new_session=True)
        except OSError as exc:
            return JSONResponse(status_code=500,
                                content={"started": False, "why": f"could not launch: {exc}"})
        # Reaped, or every launch leaves a zombie for as long as this dashboard lives.
        threading.Thread(target=proc.wait, daemon=True).start()
        RUN_PROC.update(pid=proc.pid, scene=scene, log=str(log), started=time.time())
        return JSONResponse({"started": True, "pid": proc.pid, "scene": scene, "log": str(log),
                             "settings": shown})

    @app.post("/stop_run")
    def _stop_run(request: Request = None):
        if not _local(request):
            return JSONResponse(status_code=403,
                                content={"stopped": False, "why": "loopback only"})
        pid = RUN_PROC["pid"]
        if not _pid_alive(pid):
            return JSONResponse(status_code=409, content={
                "stopped": False, "why": "no run launched from here is running"})
        # SIGINT, not SIGTERM, and to the GROUP. live_run.sh publishes the map from its EXIT
        # trap and documents Ctrl-C as the way to stop it; SIGTERM to the leader alone leaves
        # the container and the feed holding their ports.
        try:
            pgid = os.getpgid(pid)
            os.killpg(pgid, signal.SIGINT)
        except OSError as exc:
            return JSONResponse(status_code=500,
                                content={"stopped": False, "why": f"could not signal: {exc}"})
        threading.Thread(target=_stop_container_after, args=(pid,), daemon=True).start()
        return JSONResponse({"stopped": True, "pgid": pgid, "signal": "SIGINT",
                             "then": f"docker stop {CONTAINER} if still up after 20 s"})


_LINE_STAMP = re.compile(r"\[(\d{10})\.(\d+)\]")
_last_stamp = [0.0]


def _line_stamp(line):
    """The ROS stamp on a log line, or the previous line's stamp for a continuation."""
    mo = _LINE_STAMP.search(line)
    if mo:
        _last_stamp[0] = float(f"{mo.group(1)}.{mo.group(2)}")
    return _last_stamp[0]


def _install_decision_reader(m, bundle: Path):
    """Replace the bridge's decision-log reader with a streaming, filtered one. GA-221.

    THE BRIDGE'S VERSION DOES `path.read_text()` ON hook_decisions.jsonl, then `.splitlines()`,
    then `json.loads` PER LINE. On a live run that file is small and the approach is right. On
    an ARCHIVED run it is not: this bundle's log is 1.83 GB and 2.6 M lines, so the call asks
    for ~2 GB as one string, a list of 2.6 M more, and 2.6 M json.loads. /graph_data did not
    return in 120 s, and because the viewer polls it every 3 s, the whole dashboard sat empty.

    NOTHING NEEDS THOSE LINES. The bridge tests exactly two kinds -- `admission` (or a record
    with no kind, from older logs) and `link`. Everything else is parsed and discarded. In this
    bundle that is 533 admissions and ~2,000 links inside 2.6 M records: 99.9 % of the parse
    produced nothing anyone reads.

    So the line is FILTERED AS BYTES before it is ever parsed, and the survivors are cached to a
    sidecar. The count of what was skipped is kept and reported, because a filter that silently
    drops records is the shape of bug this project keeps finding -- the panel must be able to
    say 2.6 M records were seen, not imply the run made 533 decisions.

    The sidecar lives OUTSIDE the bundle (rule 12: no tool writes into the artefact it reads)
    and is keyed on the log's size and mtime, so a file that changes is re-read.

    RESOLVED PER CALL, not once at startup. The first version captured the startup bundle's
    records into a closure, so after the dashboard followed a new run or a bundle was chosen
    from the page, every reader still got the FIRST run's decisions. Measured 2026-09-04:
    after switching to 20260903_230232, 146 of 146 objects had a crop file on disk and all 146
    ids were covered by that bundle's own `link` records, yet every thumbnail came back as the
    346-byte placeholder -- because `_link_index` was reading the previous run's records, could
    not map an object id to its label, and asked for a file named after the raw id. The crops
    were never missing. The reader was pointed at the wrong run.
    """
    import hashlib
    
    # Kept as bytes: this test runs 2.6 M times and decoding each line first would put the
    # cost straight back where it was taken from.
    # `merge` joins admission and link: ~120 of them in this bundle, against 2.35 M
    # `merge_refused`. Keeping merges costs nothing and lets the scrubber tag them.
    KEEP = (b'"kind": "admission"', b'"kind":"admission"', b'"kind": "link"', b'"kind":"link"',
            b'"kind": "merge"', b'"kind":"merge"')
    HAS_KIND = b'"kind"'

    cache_dir = Path(dash_env.env(
        "DASH_REPLAY_CACHE",
        str(Path(tempfile.gettempdir()) / "dash-replay-cache")))

    # ponytail: three bundles' records held in memory at once. Switching between two runs is
    # the case that matters and it never re-reads; a longer history would only add memory.
    memo = {}

    def _blob_for(log: Path, st):
        import json as _json
        key = hashlib.sha256(f"{log}|{st.st_size}|{st.st_mtime}".encode()).hexdigest()[:16]
        if key in memo:
            return memo[key]
        cache_dir.mkdir(parents=True, exist_ok=True)
        sidecar = cache_dir / f"decisions-{key}.json"
        blob = None
        if sidecar.exists():
            try:
                blob = _json.loads(sidecar.read_text())
            except (OSError, ValueError):
                blob = None
        if blob is None:
            records, seen, skipped, unreadable = [], 0, 0, 0
            with log.open("rb") as f:
                for line in f:
                    seen += 1
                    if not any(k in line for k in KEEP):
                        # A record with no `kind` at all is an OLD admission record and is
                        # kept: the bridge's own test is `kind not in (None, "admission")`.
                        if HAS_KIND in line:
                            skipped += 1
                            continue
                    try:
                        records.append(_json.loads(line))
                    except ValueError:
                        unreadable += 1
            blob = {"records": records, "seen": seen, "skipped": skipped,
                    "unreadable": unreadable, "log_bytes": st.st_size}
            try:
                sidecar.write_text(_json.dumps(blob))
            except OSError:
                pass  # a cache that cannot be written is slow, not wrong
            print(f"  decisions: {len(blob['records'])} kept, {blob['skipped']:,} skipped "
                  f"of {blob['seen']:,} ({blob['log_bytes'] / (1 << 30):.2f} GB) "
                  f"from {log.parent.name}", flush=True)
        memo[key] = blob
        for stale in list(memo)[:-3]:
            del memo[stale]
        return blob

    def _current_log():
        """The log of the run being served RIGHT NOW -- the same directory every other reader
        resolves per call, not the one this process started on."""
        configured = os.environ.get("GRAPH_API_OUTPUT_DIR")
        return (Path(configured) if configured else bundle) / "hook_decisions.jsonl"

    def _cached_records():
        log = _current_log()
        try:
            st = log.stat()
        except OSError:
            m._DECISIONS_CACHE.update(key=None, records=[], unreadable=0,
                                      error=f"no hook_decisions.jsonl in {log.parent.name}")
            return []
        blob = _blob_for(log, st)
        m._DECISIONS_CACHE.update(
            key=(str(log), st.st_mtime, st.st_size),
            records=blob["records"], unreadable=blob["unreadable"],
            # Surfaced wherever the bridge shows its decision-log error line, so the
            # filtering is visible on the page rather than only in this docstring.
            error=(None if not blob["skipped"] else
                   f"replay: read {blob['seen']:,} records from a "
                   f"{blob['log_bytes'] / (1 << 30):.2f} GB log; "
                   f"{blob['skipped']:,} merge/update records were skipped as unread "
                   f"by the viewer"))
        return blob["records"]

    m._decision_records = _cached_records
    _cached_records()          # warm the startup bundle, and print its counts as before


def build_app(bundle: Path):
    os.environ["GRAPH_API_OUTPUT_DIR"] = str(bundle)
    _install_ros_stubs()
    m = _load_bridge()
    _INPROC.update(bridge=m, feed_host=m.FEED_HOST)
    _sync_inproc_feed_host()          # replay at start -> unroutable; the follower swaps it back when live

    from fastapi import Request
    from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response

    _install_decision_reader(m, bundle)

    # THE MEASURED FRAME PERIOD, IN REPLAY. The bridge computes it from frame arrivals; here
    # there are no arrivals, but the frames carry their own stamps, so the same number comes
    # from the bundle. Same key, same row on the page, honest in both modes.
    _bev_inner = m.proxy_bev_data
    m.app.router.routes = [r for r in m.app.router.routes
                           if getattr(r, "path", None) != "/bev_data"]

    def _frame_period(b):
        stems = sorted(p.stem for p in (b / "frames").glob("*.jpg")) if (b / "frames").is_dir() else []
        ts = []
        for st in stems:
            try:
                sec, nsec = st.split("_")
                ts.append(int(sec) + int(nsec) / 1e9)
            except ValueError:
                continue
        gaps = sorted(y - x for x, y in zip(ts, ts[1:]))
        return (round(gaps[len(gaps) // 2], 2), len(gaps)) if gaps else (None, 0)

    @m.app.get("/bev_data")
    def _bev_data_replay():
        resp = _bev_inner()
        try:
            body = json.loads(bytes(resp.body))
        except (ValueError, TypeError, AttributeError):
            return resp
        period, n = _frame_period(current_bundle())
        if period is not None:
            lat = body.setdefault("latencies", {})
            lat["frame_period_s"], lat["frame_period_n"] = period, n
        return JSONResponse(content=body)

    # Drop the live camera routes. Their MJPEG generator waits on frames that never
    # arrive here, and it holds a worker while it waits -- with no camera it starved
    # every other request and the whole page hung. Answer immediately instead.
    keep = [r for r in m.app.router.routes
            if getattr(r, "path", None) not in ("/feed", "/feed.mjpg", "/")]
    m.app.router.routes = keep

    # A 503 here made the browser draw its own broken-image icon, which reads as a
    # fault rather than as "there is no camera in a replay". Serve a real image that
    # says what is actually true, so the feed panel looks deliberate.
    _NO_FEED_SVG = (
        '<svg xmlns="http://www.w3.org/2000/svg" width="640" height="480">'
        '<rect width="100%" height="100%" fill="#0b1220"/>'
        '<text x="50%" y="46%" fill="#64748b" font-family="ui-monospace,monospace" '
        'font-size="22" font-weight="bold" text-anchor="middle">NO CAMERA IN REPLAY</text>'
        '<text x="50%" y="55%" fill="#475569" font-family="ui-monospace,monospace" '
        'font-size="13" text-anchor="middle">this dashboard reads a run directory, '
        'not a live simulator</text></svg>'
    ).encode()

    def _relay_live(path: str, query: str = ""):
        """Proxy one GET to the bridge when MODE is live; None when it is not or it failed.
        The MJPEG feed is relayed chunk by chunk (GA-343). Used by the camera routes only:
        /bev_data and /graph_data read the run's files in-process and already follow the run."""
        import urllib.error
        import urllib.parse
        import urllib.request

        from fastapi.responses import StreamingResponse
        if MODE["mode"] != "live":
            return None
        # FastAPI hands over the DECODED path; a label with a space ("/crop/framed picture")
        # raised http.client.InvalidURL upstream and answered 500. Re-quote before relaying.
        url = f"{BRIDGE_URL}/{urllib.parse.quote(path, safe='/')}" + (f"?{query}" if query else "")
        try:
            r = urllib.request.urlopen(urllib.request.Request(url), timeout=8.0)
        except (urllib.error.URLError, OSError, TimeoutError):
            return None
        ctype = r.headers.get("Content-Type", "application/octet-stream")
        if ctype.startswith("multipart/x-mixed-replace"):
            def _gen(src=r):
                with src:
                    while True:
                        chunk = src.read1(65536)   # whatever is buffered now; read() would hold a frame's tail until the next frame
                        if not chunk:
                            break
                        yield chunk
            return StreamingResponse(_gen(), status_code=r.status, media_type=ctype,
                                     headers={"Cache-Control": "no-store"})
        with r:
            return Response(content=r.read(), status_code=r.status, media_type=ctype,
                            headers={"Cache-Control": "no-store"})

    # GA-379. THE BRIDGE DEFINES THESE TWO ITSELF, so registering ours on the same app would
    # register a SECOND route for each path and FastAPI serves the FIRST match -- the bridge's,
    # answering from its stub node. Measured: /last_perception/meta returned `available: false`
    # through the dashboard while the bridge beside it returned the real cycle. Same shape as
    # /feed, /frame.jpg and /health above, and as the /crop decorator incident. Strip, then define.
    m.app.router.routes = [r for r in m.app.router.routes
                           if getattr(r, "path", None) not in ("/last_perception/meta",
                                                               "/last_perception.jpg")]

    @m.app.get("/last_perception/meta")
    @m.app.get("/last_perception.jpg")
    def _last_perception(request: Request):
        # The retained annotated cycle lives on the RUNNING node, so this is a live-only route and
        # the in-process bridge cannot answer it (its node is a stub, and a stub's answer -- "no
        # cycle yet" -- is indistinguishable from "this is a replay"). Relayed when live; refused
        # with the reason when not, so the page can say why rather than draw an empty pane.
        live = _relay_live(request.url.path.lstrip("/"), request.url.query)
        if live is not None:
            return live
        # TWO DIFFERENT FAILURES, and they were reported with one sentence. `_relay_live` returns
        # None both when this is a replay (there is no second feed at all) and when the bridge did
        # not answer (there is one, and it is broken). Saying "no second feed in a replay" to a
        # live viewer whose bridge just died is a transport failure wearing a mode's clothes.
        if MODE["mode"] == "live":
            return JSONResponse(status_code=503, content={
                "available": False, "mode": "live",
                "error": f"the bridge at {BRIDGE_URL} did not answer; the second feed is live-only "
                         f"and comes straight from the running perception node"})
        return JSONResponse(status_code=503, content={
            "available": False, "mode": "replay",
            "error": "no second feed in a replay: the annotated frame comes from the running "
                     "perception node and a bundle does not record it"})

    @m.app.get("/feed")
    @m.app.get("/feed.mjpg")
    def _no_feed(request: Request):
        # A replay-started process that the mode follower moved to LIVE served this placeholder
        # for the whole run ("FEED OFFLINE", broken image, owner 2026-09-07 15:38): the feed
        # routes were decided at startup. Decide per request instead, like /arch and /scene3d.
        live = _relay_live(request.url.path.lstrip("/"), request.url.query)
        if live is not None:
            return live
        return Response(content=_NO_FEED_SVG, media_type="image/svg+xml",
                        headers={"Cache-Control": "no-store"})

    _frame_inner = [r for r in m.app.router.routes if getattr(r, "path", None) == "/frame.jpg"]
    m.app.router.routes = [r for r in m.app.router.routes if getattr(r, "path", None) != "/frame.jpg"]

    @m.app.get("/frame.jpg")
    def _frame_follow(request: Request):
        # In-process there is no node and FEED_HOST is unroutable in replay (V-1), so this is a
        # 503/500 here; the bridge has the frame when live.
        live = _relay_live("frame.jpg", request.url.query)
        if live is not None:
            return live
        if _frame_inner:
            return _frame_inner[0].endpoint()
        return JSONResponse(status_code=503, content={"error": "no frame: replay has no camera"})

    _health_inner = [r for r in m.app.router.routes if getattr(r, "path", None) == "/health"]
    m.app.router.routes = [r for r in m.app.router.routes if getattr(r, "path", None) != "/health"]

    @m.app.get("/health")
    def _health_follow():
        # The in-process bridge module has a stub node: frame_at is None and the page's feed
        # heartbeat reads that as silence ("Feed silent 206 s"). Live: the bridge's own /health.
        live = _relay_live("health")
        if live is not None:
            return live
        return _health_inner[0].endpoint() if _health_inner else JSONResponse(status_code=503, content={"error": "no health reader"})

    def current_bundle() -> Path:
        """The directory being served RIGHT NOW.

        Not the one captured at startup. The follower re-points
        GRAPH_API_OUTPUT_DIR, and the bridge reads that per call -- so the DATA moved to
        the new run while the banner and /replay_info still named the old one. The
        banner exists to say which run you are looking at, so a stale value there is
        worse than no banner: it was reading "PINNED TO AN EARLIER RUN" and naming the
        previous bundle while serving the current one.
        """
        configured = os.environ.get("GRAPH_API_OUTPUT_DIR")
        return Path(configured) if configured else bundle

    def _load_replay_view():
        # Both launch styles, same reason as /blockers: this server is started BY FILE PATH,
        # so sys.path[0] is this directory and the package import fails; under -m the bare
        # import fails instead.
        try:
            from found.dashboard import replay_view
        except ModuleNotFoundError:
            import replay_view
        return replay_view

    @m.app.get("/replay", response_class=HTMLResponse)
    def _replay(bundle: str = None):
        """Frame-by-frame replay of an ARCHIVED run. Not the live view -- the bridge serves
        that on :8081 while a run is up, and the two are easy to confuse precisely when
        nothing is appearing."""
        return HTMLResponse(with_tools_menu(_load_replay_view().page(bundle)))

    @m.app.get("/replay/index/{bundle}")
    def _replay_index(bundle: str):
        return JSONResponse(_load_replay_view().frame_index(bundle))

    @m.app.get("/replay/frame/{bundle}/{frame_id}.jpg")
    def _replay_frame(bundle: str, frame_id: str):
        p = _load_replay_view().frame_path(bundle, frame_id)
        if p is None:
            # Refused, not 404-with-a-guess: `frame_id` comes off a URL and frame_path
            # resolves it INSIDE the bundle or returns None.
            return JSONResponse(status_code=404, content={"error": "no such frame in that bundle"})
        # GA-345: on a live page the newest frame may still be being written; a partial JPEG
        # cached for an hour would stay partial. A frame younger than 3 s is served no-store.
        fresh = (time.time() - p.stat().st_mtime) < 3.0
        return FileResponse(str(p), media_type="image/jpeg",
                            headers={"Cache-Control": "no-store" if fresh else "public, max-age=3600"})

    @m.app.get("/replay/cycle/{bundle}/{frame_id}")
    def _replay_cycle(bundle: str, frame_id: str):
        """GA-334: the perception metrics OF THE VIEWED FRAME'S CYCLE, not the run's last
        snapshot. `match` says whether the row is this frame's or the preceding cycle's."""
        return JSONResponse(_load_replay_view().cycle_for_frame(bundle, frame_id))

    @m.app.get("/replay/events/{bundle}")
    def _replay_events(bundle: str):
        """Timestamped events for the scrubber. GA-229.

        Read from the SAME filtered records the graph uses, so the ticks and the panels can
        never disagree about what happened. `merge_refused` is deliberately absent: 2.35 M
        ticks is not a timeline, it is a solid bar.
        """
        out = []
        for r in m._DECISIONS_CACHE["records"]:
            t = r.get("t")
            if not isinstance(t, (int, float)):
                continue
            k = r.get("kind")
            if k == "admission":
                grade = ((r.get("annotation") or {}).get("verdict") or {}).get("grade")
                # The FOUR-VALUED grade, not the three-way `outcome` projection: `hold` is a
                # verdict of its own and collapsing it into abstain hides it entirely.
                out.append({"t": t, "kind": grade or r.get("outcome"), "label": r.get("object")})
            elif k == "merge":
                out.append({"t": t, "kind": "merge", "label": r.get("object")})
        out.sort(key=lambda e: e["t"])
        return JSONResponse({"events": out})

    @m.app.get("/replay/verdicts/{bundle}")
    def _replay_verdicts(bundle: str):
        """{label: grade} per object, for the verdict-driven bbox layers. GA-237.

        The overlay used to split PERM from TEMP on `habitat_gt_instance_id`, which is null
        on every row of a run captured with no semantic frame -- so the two buttons selected
        the same population and looked broken because they were the same button twice. The
        real distinction is the admission verdict. Measured on 20260901_174810_hm3d_00861:
        143 graded objects, 15 admit / 15 hold / 25 decline / 88 no_grounds, joining by
        `instance_label` to 143 of the 169 distinct detection labels.
        """
        # GA-239 changed object_verdicts to return {by_object, by_label, unlinked,
        # ambiguous_labels} and this route kept passing the whole thing through as `verdicts`.
        # The page keys `VERDICT[d.label]` and counts Object.values(VERDICT): it saw FOUR keys,
        # every box read UNJUDGED and the grade buttons said 0/0/0/0 -- on every bundle since
        # cffb278 (measured 2026-09-07 on runs H and 152446: 213 graded admissions, 4 "verdicts").
        # A detection carries no object id, so the label IS the page's only join key: serve one
        # grade per label where the label received exactly one, and NAME the ambiguous ones
        # (a per-frame ordinal that got several verdicts) instead of picking one.
        v = _load_replay_view().object_verdicts(bundle)
        one = {lab: gs[0] for lab, gs in v["by_label"].items() if len(gs) == 1}
        return JSONResponse({"verdicts": one,
                             "ambiguous": sorted(lab for lab, gs in v["by_label"].items() if len(gs) > 1),
                             "by_object": v["by_object"], "unlinked": v["unlinked"]})

    # GA-252. TRIM THE POLL. /graph_data is 19.2 MB and the viewer re-fetches it every 3 s;
    # measured, JSON.parse alone blocks the main thread for 438 ms each time, which lands in
    # the middle of whatever gesture is in progress.
    #
    # 8.9 MB of it is DUPLICATION: `nodes`, `edges` and `objects` repeat what `elements`
    # already carries, and the viewer reads only `data.elements.*` -- verified, zero
    # references to the top-level copies anywhere in viewer.html or the tools.
    #
    # `admission_summary` (6.4 MB) is KEPT: agent B wired the objects table's verdict column
    # to it, so dropping it would silently empty a column rather than make a page faster.
    _graph_data_orig = m.app.router.routes
    m.app.router.routes = [r for r in m.app.router.routes
                           if getattr(r, "path", None) != "/graph_data"]

    @m.app.get("/graph_data")
    def _graph_data_trimmed(request: Request = None):
        g = m.graph_data(request=None)
        if not isinstance(g, dict):
            return g
        # `nodes` and `edges` ONLY. Both have an explicit fallback in the viewer
        # (viewer.html:3982-3983 read data.elements.nodes/edges when the top-level copy is
        # absent), so dropping them is invisible.
        #
        # `objects` is NOT dropped, though it is the same 4.1 MB again. viewer.html:1858
        # feeds the BEV's object list from `lastGraphData.objects`, because /bev_data carries
        # no objects of its own -- dropping it would empty the BEV silently, which is a worse
        # outcome than a larger payload. Removing it needs that fallback repointed at
        # elements.nodes first, and that is a change to BEV behaviour, not to a payload.
        slim = {k: v for k, v in g.items() if k not in ("nodes", "edges")}
        # GA-260. `objects[].decision` is a SECOND copy of `elements.nodes[].data.decision`
        # -- 3.9 MB of the payload, twice over. Verified before removing it:
        #   the BEV reads objects[] for .bbox, .position and .status only (viewer.html:1858)
        #   the verdict table reads rawNodes, which falls back to elements.nodes[].data
        #     now that the top-level `nodes` copy is gone (viewer.html:3982, 4020)
        #   the details panel reads a cytoscape node's own data, also elements
        # So nothing reads the copy in objects[], and the copy in elements is untouched.
        if isinstance(slim.get("objects"), list):
            slim["objects"] = [
                {k: v for k, v in o.items() if k != "decision"} if isinstance(o, dict) else o
                for o in slim["objects"]
            ]
        etag = f'"{g.get("version")}"'
        if request is not None and request.headers.get("if-none-match") == etag:
            return Response(status_code=304, headers={"ETag": etag})
        return JSONResponse(content=slim,
                            headers={"ETag": etag, "Cache-Control": "no-cache"})

    @m.app.get("/walls_view")
    def _walls_view():
        """Detected wall segments for the dashboard overlay. GA-271.

        Proxies the bridge live; in a replay there is no bridge, and there is no recorded
        wall artefact either, so it says so rather than returning an empty list a reader
        would take for "no walls were found".
        """
        import json as _j
        import urllib.request as _u
        # The bridge's own default is 8081 (graph_api_bridge.py, last line). This probed
        # 8091, so in live mode it always failed to reach a bridge that was running, and
        # the panel reported "no bridge reachable" -- which on screen reads as "no walls
        # were detected". One digit, and the two answers are indistinguishable to a reader.
        host = BRIDGE_URL
        # Only in LIVE mode. Measured 2026-09-07 on a replay instance pinned to run H while run
        # 152446's bridge was up on :8085: this answered {"live": true, ...} with the LIVE run's
        # walls on run H's page. A replay reads its bundle and nothing else.
        if MODE["mode"] == "live":
            try:
                with _u.urlopen(f"{host}/walls", timeout=1.5) as r:
                    d = _j.loads(r.read().decode())
                d["live"] = True
                return JSONResponse(d)
            except (OSError, ValueError, TimeoutError):
                pass
        return JSONResponse({"live": False, "walls": [], "available": False,
                             "why": ("no bridge reachable" if MODE["mode"] == "live" else
                                     "replay mode") + "; wall segments are per-frame "
                                    "and are not archived in the bundle"})

    @m.app.get("/feed_layers")
    def _feed_layers(set: str = None):
        """Read or set what the HABITAT WINDOW draws. GA-265.

        One shared dict, two front ends. The window toggles it by keypress and republishes;
        the dashboard reads it here and can set it through the feed host's control server.
        Neither owns it, so a toggle made in either place is visible in the other.

        Live, this proxies to the feed host on :7790. In a replay there is no feed host, so
        it falls back to the `feed_layers.json` the run left behind -- which shows what the
        layers WERE, and says so, rather than pretending they can still be changed.
        """
        import json as _j
        import urllib.request as _u
        host = os.environ.get("FEED_HOST", "http://127.0.0.1:7790")
        # Only in LIVE mode. Measured 2026-09-07 on a replay instance pinned to run H while run
        # 152446 walked: this reached the LIVE feed host, rendered "HABITAT WINDOW" with enabled
        # buttons on run H's replay page, and a click would have toggled the live run's window.
        if MODE["mode"] == "live":
            try:
                q = f"?set={set}" if set else ""
                with _u.urlopen(f"{host}/layers{q}", timeout=1.5) as r:
                    d = _j.loads(r.read().decode())
                d["live"] = True
                return JSONResponse(d)
            except (OSError, ValueError, TimeoutError):
                pass
        f = current_bundle() / "feed_layers.json"
        if f.is_file():
            try:
                d = _j.loads(f.read_text())
                d["live"] = False
                d["note"] = ("recorded state from the bundle; the feed host is not running, "
                             "so these cannot be changed")
                return JSONResponse(d)
            except (OSError, ValueError):
                pass
        return JSONResponse({"live": False, "layers": None,
                             "why": "no feed host on :7790 and no feed_layers.json in this "
                                    "bundle (written from GA-265 onward)"})

    @m.app.get("/merge_pending")
    def _merge_pending():
        """What the association layer is currently trying to confirm. GA-258.

        The pairs whose evidence is over the threshold but whose confirmation streak is not
        yet complete -- i.e. exactly the merges another look would commit. Served from the
        sidecar the object manager writes each sweep, so it works live and reads a finished
        bundle afterwards without a second code path.
        """
        import json as _j
        f = current_bundle() / "merge_pending.json"
        if not f.is_file():
            return JSONResponse({"available": False,
                                 "why": "no merge_pending.json in this bundle; it is written "
                                        "per sweep by object_services from GA-258 onward, so "
                                        "runs before that do not have one"})
        try:
            d = _j.loads(f.read_text())
        except (OSError, ValueError) as exc:
            return JSONResponse({"available": False, "why": f"unreadable: {exc}"})
        d["available"] = True
        return JSONResponse(d)

    @m.app.get("/replay/pose3d/{bundle}")
    def _replay_pose3d(bundle: str):
        """Solved camera rotations + the 3D boxes to project with them. GA-234."""
        rv = _load_replay_view()
        return JSONResponse({"poses": rv.camera_poses(bundle), "boxes": rv.frame_boxes3d(bundle),
                             "belief": rv.belief_boxes3d(bundle)})

    @m.app.get("/replay/masks/{bundle}/{frame_id}")
    def _replay_masks(bundle: str, frame_id: str):
        """Segmentation masks for one frame, fetched only when that layer is on. GA-227."""
        return JSONResponse({"masks": _load_replay_view().frame_masks(bundle, frame_id)})

    @m.app.get("/replay/logs/{bundle}")
    def _replay_logs(bundle: str):
        """What terminal output this bundle captured. GA-220."""
        return JSONResponse({"logs": _load_replay_view().list_logs(bundle)})

    @m.app.get("/replay/log/{bundle}/{name}")
    def _replay_log(bundle: str, name: str, at: int = None):
        """A window of one log around wall-clock `at` (the frame's own stamp). GA-220.

        Not the tail: om6.log's last 8 MB begins after 78 of this run's 84 frames, and a
        browser searching inside a tail reports line 1 as the line nearest every earlier
        frame. The window is found by bisecting the file and its true time range is returned,
        so a frame the log does not cover reads as uncovered rather than as a wrong line.
        """
        w = _load_replay_view().log_window(bundle, name, at)
        if w is None:
            return JSONResponse(status_code=404, content={"error": "no such log in that bundle"})
        return JSONResponse(w)

    # The bridge's /logs proxies to the feed host on :7790, which does not exist in a replay --
    # the panel showed "Connection refused" where the run's own output was sitting on disk the
    # whole time. Serve the archived logs instead, newest last, tagged by the node that wrote
    # each line so a merged view stays attributable.
    _LOG_ORDER = ("perception.log", "om6.log", "rtabmap.log", "bridge.log",
                  "feed_host.log", "feed_node.log", "system_health.log")

    m.app.router.routes = [r for r in m.app.router.routes
                           if getattr(r, "path", None) != "/logs"]

    @m.app.get("/logs")
    def _bundle_logs(lines: int = 400):
        rv = _load_replay_view()
        out, missing = [], []
        for name in _LOG_ORDER:
            path = rv.log_path(bundle.name, name)
            if path is None:
                missing.append(name)
                continue
            tag = f"[{name[:-4]}]"
            # Tail only: om6.log is 38.6 MB and the panel wants the end of the run.
            with path.open("rb") as f:
                size = path.stat().st_size
                f.seek(max(0, size - 262144))
                if size > 262144:
                    f.readline()
                tail = f.read().decode("utf-8", errors="replace").splitlines()
            for ln in tail[-lines:]:
                out.append((_line_stamp(ln), f"{tag} {ln}"))
        # Sorted by the ROS stamp so the merge is chronological rather than file-ordered.
        # Lines without a stamp keep the stamp of the line above them, so a traceback stays
        # attached to the message that introduced it instead of migrating to the top.
        out.sort(key=lambda x: x[0])
        return JSONResponse({
            "logs": [t for _, t in out][-lines:],
            "source_errors": ([f"not in this bundle: {', '.join(missing)}"] if missing else []),
        })

    def _list_bundle_names(limit: int = 60):
        return _bundle_names(limit)

    @m.app.post("/load_bundle")
    def _load_bundle(name: str = ""):
        """Point the dashboard at another run, from the dashboard.

        PINS the view: the follower would otherwise drag it back to the newest run within
        15 s, and the person who just chose a bundle would watch it change under them
        with nothing saying why. Choosing the newest run again releases the pin.

        The name is resolved against RUNS_ROOT and the result must still be inside it, so
        a crafted name cannot walk out of the runs directory.
        """
        if not name:
            return JSONResponse(status_code=400, content={"ok": False, "why": "no bundle named"})
        target = (RUNS_ROOT / name).resolve()
        try:
            inside = target.is_relative_to(RUNS_ROOT.resolve())
        except AttributeError:                       # Python < 3.9
            inside = str(target).startswith(str(RUNS_ROOT.resolve()))
        if not inside:
            return JSONResponse(status_code=400,
                                content={"ok": False, "why": f"{name} is outside {RUNS_ROOT}"})
        if not target.is_dir():
            return JSONResponse(status_code=404,
                                content={"ok": False, "why": f"{target} is not a directory"})
        os.environ["GRAPH_API_OUTPUT_DIR"] = str(target)
        BUNDLE_PIN["pinned"] = (target != resolve_bundle("latest"))
        # Choosing a recorded bundle IS a move into replay mode. Saying "mode": "replay"
        # while still serving the live assembly would be a label with nothing behind it.
        was = MODE["mode"]
        MODE.update(mode="replay", why=f"bundle {target.name} chosen from the dashboard", by="pick")
        print(f"[dash] bundle switched to {target} (pinned={BUNDLE_PIN['pinned']}, "
              f"mode {was} -> replay)", flush=True)
        return {"ok": True, "bundle": str(target), "pinned": BUNDLE_PIN["pinned"],
                "mode": "replay", "was": was}

    @m.app.get("/mode_info")
    def _mode_info():
        """What this dashboard is showing, and on what evidence.

        `/replay_info` said mode=replay unconditionally. This reports the mode actually in
        force plus the bundle and whether it is pinned, so a page can gate its controls on
        one answer instead of each panel guessing.
        """
        replay = MODE["mode"] == "replay"
        return {"mode": MODE["mode"], "why": MODE["why"], "by": MODE.get("by"),
                "bundle": str(current_bundle()), "pinned": BUNDLE_PIN["pinned"],
                "live_camera": not replay, "ros": not replay,
                "bridge": None if replay else BRIDGE_URL}

    @m.app.get("/decisions")
    def _decisions():
        """Every admission record for this bundle.

        Ported from found/dashboard/server.py so retiring that server loses nothing. Its
        version swallowed read errors with `except Exception: pass` and returned a short
        list that looked complete; this one says what went wrong instead.
        """
        path = current_bundle() / "hook_decisions.jsonl"
        if not path.exists():
            return JSONResponse(status_code=404,
                                content={"error": f"no hook_decisions.jsonl in {current_bundle()}"})
        records, unreadable = [], 0
        try:
            raw = path.read_text()
        except OSError as exc:
            return JSONResponse(status_code=503,
                                content={"error": f"could not read {path}: {exc}"})
        for line in raw.splitlines():
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                unreadable += 1
        return {"records": records, "count": len(records), "unreadable_records": unreadable}

    @m.app.get("/replay_info")
    def _replay_info():
        return {"mode": MODE["mode"], "bundle": str(current_bundle()),
                "serves": "graph_api_bridge", "live_camera": False, "ros": False}

    def _index():
        """One page, two modes.

        BOTH modes get the transport bar and the frame overlays (GA-345): on a live page the
        bar follows the growing bundle's head until the reader scrubs. Only REPLAY gets the
        poller blocker: _replay_head_html blocks the five pollers by name, so injecting it in
        live mode would leave the page frozen on its first frame with nothing erroring --
        health permanently stale, the graph never refreshing, and no clue on screen as to why.

        The TOOLS menu is injected in BOTH: its bundle picker is how a live viewer moves
        to a recorded run, and the RViz button is loopback-gated independently of mode.
        """
        page = (m.VIEWER_DIR / "viewer.html").read_text()
        replay = MODE["mode"] == "replay"

        head = ('<script>window.DASH_MODE=%s;window.REPLAY_MODE=%s;window.DASH_BRAND=%s;</script>'
                % (repr(MODE["mode"]), "true" if replay else "false",
                   json.dumps(dash_ext.brand())))
        lower = page.lower()
        i = lower.find("<body")
        if i != -1:
            i = page.find(">", i) + 1
            page = (page[:i] + head
                    + (_replay_head_html() if replay else "")
                    + _tools_menu_html(_list_bundle_names()) + page[i:])
        # GA-345: the transport bar is injected in BOTH modes (it reads window.DASH_MODE and
        # follows the live head until the reader scrubs). Only the poller blocker above is
        # replay-only. The rewiring must run AFTER the viewer's own script, not before it.
        j = page.lower().rfind("</body>")
        inject = _replay_mode_html(current_bundle())
        page = (page[:j] + inject + page[j:]) if j != -1 else (page + inject)
        return HTMLResponse(page)

    _install_launcher(m.app, _index, _list_bundle_names, lambda: current_bundle().name)
    return m


def build_live_app(bundle: Path):
    """LIVE mode: serve the page locally, proxy everything else to the running bridge.

    The bridge cannot be imported in-process on the host -- it needs real ROS -- so live
    mode does what found/dashboard/server.py proved works: a thin host-side front over the
    containerised bridge. That server hand-wrote thirteen proxies; this is one catch-all.

    ROUTE ORDER IS LOad-BEARING. The catch-all matches everything, so every local route
    must be registered BEFORE it. FastAPI serves the first match, and a catch-all placed
    first would swallow /mode_info and /bundles and forward them to a bridge that answers
    404. That is the same shape as the /crop/{target} decorator incident: the route that
    answered was not the route anyone meant.
    """
    import urllib.error
    import urllib.parse
    import urllib.request

    from fastapi import FastAPI, Request
    from fastapi.responses import HTMLResponse, JSONResponse, Response, StreamingResponse
    from fastapi.staticfiles import StaticFiles

    app = FastAPI(title=f"{dash_ext.brand()} dashboard (live)")
    viewer_dir = BRIDGE.parent / "viewer"
    if (viewer_dir / "viewer.html").is_file():
        # The page hard-fails to "GRAPH OFFLINE" without its vendored cytoscape/lucide.
        app.mount("/viewer", StaticFiles(directory=str(viewer_dir)), name="viewer")

    def _bundles():
        try:
            runs = [d for d in RUNS_ROOT.iterdir() if d.is_dir() and not d.is_symlink()]
        except OSError:
            return []
        return [d.name for d in sorted(runs, key=lambda d: d.name, reverse=True)[:60]]

    @app.get("/mode_info")
    def _mode_info():
        ok, why = bridge_identified()
        return {"mode": MODE["mode"], "why": MODE["why"], "bridge": BRIDGE_URL,
                "bridge_reachable_now": ok, "bridge_probe": why,
                "bundle": str(bundle), "pinned": BUNDLE_PIN["pinned"],
                "live_camera": True, "ros": True}

    @app.post("/load_bundle")
    def _load_bundle(name: str = ""):
        """Switching to a recorded bundle from LIVE mode.

        This process is the live assembly and cannot become the replay one without a
        restart, so it says exactly that instead of pretending to switch. A control that
        reports success for something that did not happen is the defect this project has
        spent days removing.
        """
        return JSONResponse(status_code=409, content={
            "ok": False,
            "why": ("this dashboard is running in LIVE mode; restart it with "
                    f"--mode replay --bundle {name or '<bundle>'} to view a recorded run"),
            "mode": "live"})

    @app.get("/replay")
    def _replay_live(bundle: str = None):
        """The frame-by-frame readers (/replay/*) exist only in the replay assembly. Say so;
        before this the tools menu's REPLAY link fell through to the bridge's 404."""
        return JSONResponse(status_code=409, content={
            "ok": False, "mode": "live",
            "why": ("this dashboard is running in LIVE mode; restart it with "
                    f"--mode replay --bundle {bundle or '<bundle>'} for the frame-by-frame replay")})

    def _index():
        page = (viewer_dir / "viewer.html").read_text()
        head = ("<script>window.DASH_MODE='live';window.REPLAY_MODE=false;"
                "window.DASH_BRAND=%s;</script>" % json.dumps(dash_ext.brand()))
        lower = page.lower()
        i = lower.find("<body")
        if i != -1:
            i = page.find(">", i) + 1
            page = page[:i] + head + _tools_menu_html(_bundles()) + page[i:]
        return HTMLResponse(page)

    _install_launcher(app, _index, _bundles, lambda: bundle.name)

    # --- the catch-all, registered LAST on purpose (see the docstring) ---------------
    @app.api_route("/{path:path}",
                   methods=["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS"])
    def _proxy(path: str, request: Request):
        url = f"{BRIDGE_URL}/{urllib.parse.quote(path, safe='/')}"   # decoded path -> re-quoted (GA-346)
        if request.url.query:
            url += "?" + request.url.query
        req = urllib.request.Request(url, method=request.method)
        try:
            r = urllib.request.urlopen(req, timeout=8.0)
            ctype = r.headers.get("Content-Type", "application/octet-stream")
            if ctype.startswith("multipart/x-mixed-replace"):
                # The camera feed is an MJPEG stream that never ends. r.read() on it never
                # returned, so /feed hung, the page's load event never fired, and every
                # browser retry pinned one more worker thread (measured 2026-09-07 on run
                # 152446: /feed 0 bytes after 6 s, /frame.jpg fine beside it). Relay it
                # chunk by chunk instead. ponytail: one thread per open stream; fine for a
                # handful of viewers, revisit if the dashboard ever has many.
                def _relay(src=r):
                    with src:
                        while True:
                            chunk = src.read1(65536)   # whatever is buffered now; read() would hold a frame's tail until the next frame
                            if not chunk:
                                break
                            yield chunk
                return StreamingResponse(_relay(), status_code=r.status, media_type=ctype,
                                         headers={"Cache-Control": "no-store"})
            with r:
                return Response(content=r.read(), status_code=r.status, media_type=ctype)
        except urllib.error.HTTPError as exc:
            # Pass the bridge's own status through rather than turning it into a 200.
            return Response(content=exc.read(), status_code=exc.code,
                            media_type=exc.headers.get("Content-Type", "application/json"))
        except (urllib.error.URLError, OSError, TimeoutError) as exc:
            # Never an empty success: a viewer that draws zero nodes from a failed fetch
            # is indistinguishable from a scene with no objects.
            return JSONResponse(status_code=503, content={
                "error": f"bridge unreachable at {BRIDGE_URL}: {type(exc).__name__}: {exc}",
                "path": path})

    return app


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--bundle", default=os.environ.get("GRAPH_API_OUTPUT_DIR", DEFAULT_BUNDLE),
                    help='run directory to serve, or "latest" to follow the newest run '
                         '(default: latest)')
    ap.add_argument("--no-follow", action="store_true",
                    help='with --bundle latest, resolve once at startup instead of tracking')
    ap.add_argument("--port", type=int, default=int(os.environ.get("REPLAY_PORT", DEFAULT_PORT)))
    ap.add_argument("--host", default="127.0.0.1",
                    help="default 127.0.0.1; pass 0.0.0.0 to publish it deliberately")
    ap.add_argument("--mode", choices=("live", "replay", "auto"), default="auto",
                    help='live = proxy a running bridge; replay = read a run directory; '
                         'auto (default) = live only if a GRAPH-API bridge actually answers')
    args = ap.parse_args()

    # MODE. An explicit flag always wins; auto probes, and the probe checks WHAT answered,
    # not merely that something did.
    if args.mode == "auto":
        ok, why = bridge_identified()
        MODE.update(mode="live" if ok else "replay", why=why, by="start")
    elif args.mode == "replay":
        # Explicit replay: no bridge probe at all. The droplet deployment runs this way and its
        # start line read "probe said: ... Connection refused", which the owner read as an error
        # (2026-09-08; applied first to the droplet snapshot by ARIA, carried here).
        MODE.update(mode="replay", why="--mode replay", by="start")
    else:
        ok, why = bridge_identified()
        MODE.update(mode=args.mode,
                    why=(f"--mode {args.mode} (probe said: {why})"))
        if args.mode == "live" and not ok:
            print(f"[dash] WARNING: --mode live but the bridge did not identify itself: {why}",
                  flush=True)
    print(f"[dash] mode={MODE['mode']} because {MODE['why']}", flush=True)
    # The mode is re-probed every 15 s from here on, in BOTH modes: a dashboard started before the
    # stack must be able to go live, and one started during a run must be able to fall back.
    # Only under --mode auto: the follower used to override an explicit flag 15 s after start
    # (measured 2026-09-07: `--mode replay` pinned to run H went live because a bridge answered
    # on :8085), and the docstring above promised the flag wins.
    if args.mode == "auto":
        _follow_mode()
    else:
        print(f"[dash] --mode {args.mode} is explicit: the mode follower is off", flush=True)

    bundle = resolve_bundle(args.bundle)
    if bundle is None:
        # NO RUNS YET. Serve anyway: the start page exists to say "no runs in <dir>" and to let
        # one be launched, and refusing here meant a fresh checkout could never reach it. A named
        # bundle that is missing is still an error -- that is a typo, not an empty archive.
        print(f"[dash] no runs under {RUNS_ROOT} yet -- serving the start page", flush=True)
        bundle = RUNS_ROOT / "(no runs yet)"
    elif not bundle.is_dir():
        # Refuse rather than serve an empty dashboard that looks like a run with no data.
        raise SystemExit(f"replay_server: {bundle} is not a directory")
    if not BRIDGE.exists():
        raise SystemExit(f"replay_server: bridge not found at {BRIDGE}")

    # The catch-all live app has no replay routes: started during a run it answered 503 for
    # EVERYTHING once the stack went down (measured 13:50Z 2026-09-07, the owner's page went
    # dark at run end). The replay-started app now proxies the camera routes per request when
    # the follower says live, so it is the standing shape in BOTH cases; the pure live app is
    # only built when asked for by name.
    if args.mode == "live":
        app = build_live_app(bundle)
    else:
        module = build_app(bundle)
        app = module.app
        if str(args.bundle) == "latest" and not args.no_follow:
            _follow_latest()
    import uvicorn
    print(f"[dash] serving {bundle} on http://{args.host}:{args.port}")
    if MODE["mode"] == "replay":
        print("[dash] replay: file-backed, no camera feed, no ROS.")
    else:
        print(f"[dash] live: bridge at {BRIDGE_URL}")
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
