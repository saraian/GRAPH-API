#!/usr/bin/env python3
"""
Host Dashboard Server for Graph API.
Runs independently on host (port 8080), separating the Web Dashboard UI
from the ROS2 / Perception / KG container stack.
"""
import json
import os
import tempfile
from pathlib import Path

try:
    from found.dashboard import dash_env
except ImportError:
    import dash_env

import requests
import uvicorn
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

app = FastAPI(title="Graph API Host Dashboard")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Dynamic path resolution using environment variables with relative fallback
HERE = Path(__file__).resolve().parent

def _resolve_dir(env_var: str, default_path: Path) -> Path:
    env_val = os.environ.get(env_var, "").strip()
    if env_val:
        p = Path(env_val)
        if p.exists():
            return p
    return default_path

DASHBOARD_DIR = _resolve_dir("DASH_DIR", HERE)

# FD-31. The old default was parents[3]/"GRAPH-API"/... which resolves to
# /GRAPH-API/... (HERE.parents[3] is "/"), and the hardcoded fallback below it was
# /DATA/GRAPH-API/lost3dsg/output. Neither exists, so OUTPUT_DIR silently became the
# dashboard source directory and the mkdir below created cropped_images inside it.
# /DATA/GRAPH-API is also a read-only remedy source now, so the running dashboard must
# not read its data from there. Order: explicit env var, then the repaired tree, then
# the most recent run bundle.
# GA-380, second attempt. The first put the environment-aware entry SECOND, behind a hardcoded
# path that exists on this machine -- so the variable could never change the answer here, and the
# fix was written but untestable: a remedy placed behind the very thing it was meant to replace.
# An EXPLICITLY SET runs directory outranks the default, because setting it is a statement about
# where this deployment's data lives.
# 2026-09-10: the vendored graph-api tree is retired, and with it the
# `vendor/graph-api/lost3dsg/output` candidate that used to sit here. It was NOT repointed at the
# consolidated checkout: that would be the very thing the check below forbids -- run data read out
# of a SOURCE tree instead of a run bundle.
# The runs directory now comes from `dash_env`, which owns the dashboard's environment contract and
# names no deployment, so this file can ship upstream. `runs_dir()` already prefers an explicit
# setting and falls back beside the graph-api tree, so ONE candidate says what two literals did.
_OUTPUT_CANDIDATES = [dash_env.runs_dir() / "latest"]
OUTPUT_DIR = _resolve_dir("GRAPH_API_OUTPUT_DIR", HERE)
if OUTPUT_DIR == HERE:
    for _cand in _OUTPUT_CANDIDATES:
        if _cand.exists():
            OUTPUT_DIR = _cand
            break
if OUTPUT_DIR == HERE:
    # NO CANDIDATE EXISTS, and falling back to this file's own directory is the FD-31 defect
    # itself: OUTPUT_DIR silently became the dashboard SOURCE directory and the crops mkdir wrote
    # into it. It went unnoticed while the dashboard lived beside a runs directory that always
    # existed; shipped in a repository with no runs, it is the normal case rather than the edge.
    # A neutral empty directory reads as "no run yet" and cannot be written into by mistake.
    OUTPUT_DIR = Path(tempfile.gettempdir()) / "dash-no-run"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Run bundles name this "crops"; the older layout used "cropped_images". Resolve to
# whichever is actually there and NEVER create it inside OUTPUT_DIR: when OUTPUT_DIR is
# a run bundle, creating a directory in it modifies the artefact this server only reads.
CROPS_DIR = next(
    (d for d in (OUTPUT_DIR / "crops", OUTPUT_DIR / "cropped_images") if d.is_dir()),
    None,
)
if CROPS_DIR is None:
    # Nothing to serve. Use a scratch directory we own rather than writing into
    # OUTPUT_DIR, so /crops mounts and returns 404s instead of failing at import.
    CROPS_DIR = Path(dash_env.env("DASH_CROPS_SCRATCH", "/tmp/dash_crops"))
    CROPS_DIR.mkdir(parents=True, exist_ok=True)

FEED_HOST = os.environ.get("FEED_HOST", "http://localhost:7790")
ROS_BRIDGE_HOST = os.environ.get("ROS_BRIDGE_HOST", "http://localhost:8081")

app.mount("/crops", StaticFiles(directory=str(CROPS_DIR)), name="crops")
app.mount("/static", StaticFiles(directory=str(DASHBOARD_DIR)), name="static")

# THE VIEWER ROUTE IS GONE, 2026-09-10, and with it this module's copy of viewer.html.
#
# There were two viewers. This file served a copy last touched on 31 August -- 1,457 lines against
# the canonical viewer's 4,898, differing by 3,947 -- while replay_server.py served the canonical
# one. The pair here was self-consistent and jointly dead: the old copy's fetch set was covered
# EXACTLY by this module's twelve routes, gap of zero, while the canonical viewer needs a
# thirteenth (/cycle_series) that this module does not define. Nothing started this server: the
# systemd unit execs replay_server.py, and the only importer is test_routes.py.
#
# WHAT IS DELIBERATELY KEPT is everything above: the OUTPUT_DIR and CROPS_DIR resolution. A route
# surface nobody serves is dead code; a path-resolution rule with four checks and a past defect
# (FD-31) behind it is a specification. Deleting the module would have taken the second with the
# first. The retired viewer is preserved at
# .handoff/plan/10-dashboard-usability/RETIRED_viewer_2026-08-31_shape_deleted_2026-09-10.html.
#
# Do NOT point this at the canonical viewer instead: it would then serve a page fetching a route
# this module does not define, which trades a dead pair for a live 404.

@app.get("/feed")
def get_feed():
    def generate():
        try:
            r = requests.get(f"{FEED_HOST}/feed.mjpg", stream=True, timeout=2.0)
            if r.status_code == 200:
                for chunk in r.iter_content(chunk_size=4096):
                    yield chunk
                return
        except Exception:
            pass
        black_svg = b'<svg xmlns="http://www.w3.org/2000/svg" width="640" height="480"><rect width="100%" height="100%" fill="#090d16"/><text x="50%" y="50%" fill="#38bdf8" font-size="20" text-anchor="middle">Connecting to Live Feed...</text></svg>'
        yield black_svg

    return StreamingResponse(generate(), media_type="multipart/x-mixed-replace; boundary=frame")

@app.get("/bev_data")
def get_bev_data(floor_y: str = None):
    # floor_y was accepted by the viewer and the feed host but dropped here, so the
    # floor selector silently showed the default floor.
    params = {"floor_y": floor_y} if floor_y else None
    try:
        r = requests.get(f"{FEED_HOST}/bev_data", params=params, timeout=1.0)
        return r.json()
    except Exception as live_exc:
        bev_file = OUTPUT_DIR / "bev_data.json"
        if bev_file.exists():
            try:
                return json.loads(bev_file.read_text())
            except (OSError, json.JSONDecodeError):
                pass
        # No fabricated pose. A zeroed agent is indistinguishable from a real one
        # standing at the origin. The viewer already guards on `data.agent`, so
        # omitting it makes the failure visible instead of plausible.
        return JSONResponse(
            content={"error": f"no BEV telemetry: {live_exc}"}, status_code=503
        )

@app.get("/action")
def send_action(act: str, x: float = None, y: float = None, z: float = None, amount: float = None):
    try:
        params = {"act": act}
        if x is not None:
            params["x"] = x
        if y is not None:
            params["y"] = y
        if z is not None:
            params["z"] = z
        if amount is not None:
            params["amount"] = amount
        r = requests.get(f"{FEED_HOST}/action", params=params, timeout=2.0)
        return r.json()
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.get("/auto_mode")
def set_auto_mode(enabled: str = "1"):
    try:
        r = requests.get(f"{FEED_HOST}/auto_mode?enabled={enabled}", timeout=2.0)
        return r.json()
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.get("/health")
def get_pipeline_health():
    components = {}

    # 1. Sim Feed
    try:
        r = requests.get(f"{FEED_HOST}/bev_data", timeout=1.5)
        components["feed"] = {"name": "Habitat Feed", "active": r.status_code == 200, "details": "Port 7790 Active" if r.status_code == 200 else f"HTTP {r.status_code}"}
    except Exception:
        components["feed"] = {"name": "Habitat Feed", "active": False, "details": "Offline"}

    # 2. ROS Bridge. This used to be hardcoded active:True with the details string
    # "Port 8080 Active" -- it reported the health of THIS process, which is
    # necessarily up if it is answering, under a key the viewer shows as the bridge.
    # So the panel showed the bridge green whether or not the bridge was running,
    # while /graph_data and /set_config were failing 503 against that same bridge.
    # Probe the thing the key names.
    # The timeout was 1.5s. Measured during run B the bridge answers HTTP 200 in
    # 2.7-4.9s while the container sits at ~619% CPU, so a healthy but loaded bridge
    # read as Offline -- and since perception and object_manager are now derived from
    # this one response, a single slow reply took THREE badges down at once. That is
    # exactly when the owner is watching. Allow for a loaded stack.
    bridge_components = None
    try:
        r = requests.get(f"{ROS_BRIDGE_HOST}/health", timeout=8.0)
        components["bridge"] = {
            "name": "ROS Bridge",
            "active": r.status_code == 200,
            "details": "Active" if r.status_code == 200 else f"HTTP {r.status_code}",
        }
        if r.status_code == 200:
            # Rule 2: assert WHAT answered, not that something did. Port 8081 has been
            # taken by an unrelated project's dashboard, which returns 200 to this probe
            # -- so "the bridge is up" was being reported on the strength of somebody
            # else's web server. Require the payload to look like OUR bridge's health.
            payload = r.json() if r.headers.get("content-type", "").startswith("application/json") else {}
            comps = (payload or {}).get("components") or {}
            ours = isinstance(comps, dict) and {"feed", "bridge"} <= set(comps)
            if ours:
                bridge_components = comps
            else:
                components["bridge"] = {
                    "name": "ROS Bridge",
                    "active": False,
                    "details": (f"{ROS_BRIDGE_HOST} answered, but it is not the GRAPH-API "
                                f"bridge (no recognisable health payload)"),
                }
    except requests.exceptions.Timeout:
        # Slow is not down. A loaded bridge that eventually answers is a different
        # state from a socket that refuses, and the two must not print the same.
        components["bridge"] = {"name": "ROS Bridge", "active": False,
                                "details": "No answer within 8s (stack loaded?)"}
    except requests.exceptions.ConnectionError:
        components["bridge"] = {"name": "ROS Bridge", "active": False,
                                "details": "Offline: connection refused"}
    except Exception as exc:
        components["bridge"] = {"name": "ROS Bridge", "active": False,
                                "details": f"Offline: {type(exc).__name__}"}

    # 3 and 4. Perception and the object manager.
    #
    # These used to be judged by the mtime of /tmp/perception.log and /tmp/om6.log. This
    # dashboard runs on the HOST and those processes run INSIDE the container, so the
    # files do not exist here at all -- `p_file.exists()` was False on every call, and
    # both badges read "Inactive / Log Stale" in every run whatever the pipeline was
    # doing. The owner saw OFFLINE beside a graph growing to 73 objects. Inside the
    # container the same files exist but are written once at startup, so mounting them
    # would not have helped either: at a 15 s threshold they are stale within seconds.
    #
    # The bridge already reports both, from the ROS side that can actually see them, and
    # its /health was fetched once above. Probe the thing the key names.
    for key, label in (("perception", "Perception Pipeline"),
                       ("object_manager", "3D Object Manager")):
        if bridge_components is None:
            # Unknown is not the same as inactive. Say which one this is, so a reader
            # cannot mistake "we could not ask" for "it is not running".
            components[key] = {"name": label, "active": False,
                               "details": "Unknown: no reading from the bridge"}
        elif key not in bridge_components:
            components[key] = {"name": label, "active": False,
                               "details": "Unknown: bridge reports no such component"}
        else:
            reported = bridge_components[key]
            components[key] = {
                "name": label,
                "active": bool(reported.get("active")),
                "details": reported.get("details") or "reported by bridge",
                "source": f"{ROS_BRIDGE_HOST}/health",
            }

    all_active = all(c["active"] for c in components.values())
    return {
        "status": "ok" if all_active else "degraded",
        "all_active": all_active,
        "components": components
    }

@app.get("/logs")
def get_logs(lines: int = 200):
    output = []
    log_files = [
        ("feed", "/tmp/habitat_feed_host.log"),
        ("feed_node", "/tmp/feed_node.log"),
        ("perception", "/tmp/perception.log"),
        ("object_manager", "/tmp/om6.log"),
        ("bridge", "/tmp/bridge.log"),
    ]
    # These paths are the CONTAINER's. This dashboard runs on the host, so they are
    # normally absent here -- the same host/container mismatch that made the health
    # check report OFFLINE forever. They are still read, because some deployments do
    # mount them, but the real source is the bridge, which lives where the logs do.
    problems = []
    for tag, filepath in log_files:
        f = Path(filepath)
        if not f.exists():
            continue
        try:
            for line in f.read_text().splitlines()[-lines:]:
                if line.strip():
                    output.append(f"[{tag}] {line}")
        except OSError as exc:
            problems.append(f"{filepath}: {type(exc).__name__}: {exc}")

    for label, base in (("bridge", ROS_BRIDGE_HOST), ("feed", FEED_HOST)):
        try:
            r = requests.get(f"{base}/logs", timeout=3.0)
            data = r.json()
            entries = data.get("logs") if isinstance(data, dict) else None
            for line in (entries or []):
                if str(line).strip():
                    output.append(f"[{label}] {line}")
        except Exception as exc:
            problems.append(f"{base}/logs: {type(exc).__name__}: {exc}")

    if not output:
        # This used to append "[bridge] System active and listening. Log stream
        # initialized." -- a line asserting the system was healthy, emitted precisely
        # when NOTHING could be read. A fabricated reassurance in a log panel is the
        # worst place for one: the panel is where a person looks to find out what went
        # wrong. Say what was tried and what failed.
        output.append("[dashboard] no log source could be read")
        for problem in problems:
            output.append(f"[dashboard] {problem}")
        if not problems:
            output.append("[dashboard] every source was reachable and empty")

    return {"logs": output[-600:], "source_errors": problems or None}

@app.get("/graph_data")
def get_graph_data():
    """viewer.html fetches this; it was never defined here, so the graph canvas
    had nothing to draw. The bridge owns the graph, so proxy to it."""
    try:
        r = requests.get(f"{ROS_BRIDGE_HOST}/graph_data", timeout=3.0)
        return JSONResponse(content=r.json(), status_code=r.status_code)
    except Exception as e:
        # An explicit error, never an empty graph: a viewer that draws zero nodes
        # from a failed fetch is indistinguishable from a scene with no objects.
        return JSONResponse(
            content={"error": f"graph unavailable from bridge: {e}"},
            status_code=503,
        )

@app.get("/set_config")
def set_config(request: Request):
    """viewer.html calls this for the perceive-while-moving and viz-layer toggles.
    It was never defined here, so every toggle 404'd and the viewer's .catch()
    swallowed it, leaving the control looking as though it worked. The bridge
    forwards config to the feed host, which stores it, so proxy there."""
    try:
        r = requests.get(
            f"{ROS_BRIDGE_HOST}/set_config", params=dict(request.query_params), timeout=2.0
        )
        return JSONResponse(content=r.json(), status_code=r.status_code)
    except Exception as e:
        return JSONResponse(
            content={"success": False, "error": f"config not applied: {e}"},
            status_code=503,
        )

@app.api_route("/query_objects", methods=["GET", "POST"])
def query_objects():
    try:
        r = requests.get(f"{ROS_BRIDGE_HOST}/query_objects", timeout=1.0)
        return r.json()
    except Exception:
        path = OUTPUT_DIR / "persistent_perception.json"
        if path.exists():
            try:
                data = json.loads(path.read_text())
                return {"objects": data, "ready": True}
            except Exception:
                pass
        return {"objects": [], "ready": True}

@app.get("/persistent_perception")
def persistent_perception():
    path = OUTPUT_DIR / "persistent_perception.json"
    if path.exists():
        try:
            return json.loads(path.read_text())
        except Exception:
            pass
    return []

@app.get("/rooms")
def get_rooms():
    try:
        r = requests.get(f"{ROS_BRIDGE_HOST}/rooms", timeout=1.0)
        return r.json()
    except Exception:
        path = OUTPUT_DIR / "room.json"
        if path.exists():
            try:
                return json.loads(path.read_text())
            except Exception:
                pass
        return []

@app.get("/decisions")
def get_decisions():
    decisions_file = OUTPUT_DIR / "hook_decisions.jsonl"
    results = []
    if decisions_file.exists():
        try:
            for line in decisions_file.read_text().splitlines():
                line = line.strip()
                if line:
                    results.append(json.loads(line))
        except Exception:
            pass
    return results

if __name__ == "__main__":
    print("[dashboard] Independent Host Dashboard running on http://0.0.0.0:8080")
    uvicorn.run(app, host="0.0.0.0", port=8080, log_level="warning")
