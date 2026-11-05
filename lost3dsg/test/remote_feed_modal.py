"""Runs habitat_feed_host.py on a Modal GPU for a stack host that has no GPU.

The stack host starts this through run_sim.sh (FEED_REMOTE_CMD=lost3dsg/test/remote_feed.sh). One SSH
connection from the Modal container back to the stack host carries everything the two sides share:

  -R 7799, -R 7790   the frame and control ports, so the container still connects to 127.0.0.1
  -L 8081            the bridge, which the feed polls
  rsync every 2 s    feed outputs into RUN_DIR (feed_ended.json ends the run), merge_pending.json back

Paths are mirrored: the scene volume mounts at /DATA/habitat_matterport and the code, config and
schedule land at their stack-host paths, so habitat_feed_host.py runs unmodified.

Needs: Modal volume `hm3d-scenes` (hm3d_example/...), Modal secret `graphapi-feed-ssh` with
FEED_SSH_KEY_B64 (base64 of the private key), FEED_SSH_TARGET (user@host) and FEED_SSH_KNOWN_HOSTS (one line).
"""
import base64
import json
import os
import subprocess
import sys
import threading
import time

import modal

REPO = os.environ.get("GRAPH_API_REPO", "/DATA/GRAPH-API")
image = (
    modal.Image.micromamba(python_version="3.9")
    # the laptop habitat_env's exact conda packages (conda list --explicit, defaults channel removed)
    .micromamba_install(spec_file=f"{REPO}/lost3dsg/test/remote_feed_conda.lock")
    .apt_install("libegl1", "libgl1", "libopengl0", "openssh-client", "rsync")
    .pip_install_from_requirements(f"{REPO}/lost3dsg/test/remote_feed_requirements.txt")
    .add_local_dir(f"{REPO}/lost3dsg/test", f"{REPO}/lost3dsg/test", ignore=["output", "__pycache__", "env.local.sh"])
    .add_local_dir(f"{REPO}/lost3dsg/src/perception_module", f"{REPO}/lost3dsg/src/perception_module",
                   ignore=["__pycache__"])
)
app = modal.App("graphapi-remote-feed", image=image)
# Region of the feed container: near the stack host, because every frame crosses the tunnel.
REGION = os.environ.get("REMOTE_FEED_REGION") or None
scenes = modal.Volume.from_name("hm3d-scenes")

# Feed outputs the stack side reads, and the one file it writes for the feed.
PUSH = ["feed_ended.json", "feed_stats.json", "bev_data.json", "feed_layers.json", "frame_poses.jsonl",
        "scan_events.jsonl"]
PULL = "merge_pending.json"
SYNC_S = 2.0


@app.function(gpu="T4", volumes={"/DATA/habitat_matterport": scenes}, secrets=[modal.Secret.from_name("graphapi-feed-ssh")],
              timeout=4 * 3600, region=REGION)
def feed(env: dict, files: dict):
    for path, text in files.items():               # config and schedule, at their stack-host paths
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as fh:
            fh.write(text)
    run_dir = env["RUN_DIR"]
    os.makedirs(run_dir, exist_ok=True)
    os.makedirs(os.path.expanduser("~/.ssh"), mode=0o700, exist_ok=True)
    key = os.path.expanduser("~/.ssh/feed_key")
    with open(key, "w") as fh:
        fh.write(base64.b64decode(os.environ["FEED_SSH_KEY_B64"]).decode().strip() + "\n")
    os.chmod(key, 0o600)
    with open(os.path.expanduser("~/.ssh/known_hosts"), "w") as fh:
        fh.write(os.environ["FEED_SSH_KNOWN_HOSTS"].strip() + "\n")
    target = os.environ["FEED_SSH_TARGET"]
    ssh = ["ssh", "-i", key, "-o", "BatchMode=yes", "-o", "ServerAliveInterval=15"]
    port, ctrl, bridge = env.get("FEED_PORT", "7799"), env.get("FEED_CTRL_PORT", "7790"), "8081"
    tunnel = subprocess.Popen(ssh + ["-N", "-o", "ExitOnForwardFailure=yes",
                                     "-R", f"127.0.0.1:{port}:127.0.0.1:{port}",
                                     "-R", f"127.0.0.1:{ctrl}:127.0.0.1:{ctrl}",
                                     "-L", f"127.0.0.1:{bridge}:127.0.0.1:{bridge}", target])
    time.sleep(3)
    if tunnel.poll() is not None:
        raise SystemExit(f"[remote-feed] tunnel to {target} failed (exit {tunnel.returncode})")
    print(f"[remote-feed] tunnel up to {target}", flush=True)

    # rrsync on the stack host roots paths at its results directory: send RUN_DIR's basename.
    remote = f"{target}:{os.path.basename(run_dir.rstrip('/'))}/"
    rsh = " ".join(ssh)
    stop = threading.Event()

    def sync():
        push = [os.path.join(run_dir, f) for f in PUSH if os.path.exists(os.path.join(run_dir, f))]
        if push:
            subprocess.run(["rsync", "-a", "-e", rsh] + push + [remote], capture_output=True)
        subprocess.run(["rsync", "-a", "-e", rsh, remote + PULL, run_dir + "/"], capture_output=True)

    def loop():
        while not stop.is_set():
            sync()
            stop.wait(SYNC_S)

    threading.Thread(target=loop, daemon=True).start()
    proc = subprocess.Popen([sys.executable, f"{REPO}/lost3dsg/test/habitat_feed_host.py"],
                            env={**os.environ, **env}, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    for line in proc.stdout:                        # streamed back: run_sim.sh waits for "listening"
        print(line, end="", flush=True)
    rc = proc.wait()
    stop.set()
    sync()                                          # the last outputs, feed_ended.json among them
    tunnel.terminate()
    print(f"[remote-feed] feed host exited {rc}", flush=True)
    return rc


@app.function(volumes={"/DATA/habitat_matterport": scenes}, secrets=[modal.Secret.from_name("graphapi-feed-ssh")],
              timeout=600, region=REGION)
def tunnel_check(hold_s: int = 90):
    """The tunnel without a run: GET the stack host's 127.0.0.1:8081 through -L, serve 127.0.0.1:7799 for
    hold_s seconds through -R (probe it from the stack host), and rsync a file both ways."""
    import http.server
    import urllib.request

    import cv2  # noqa: F401
    import habitat_sim  # noqa: F401  -- the image's conda lock really imports
    # the feed's config module, which needs yaml at import time (run 24a died on it). The modules that
    # carry ground-truth names are not imported here: pre-flight a12 refuses such names outside its list.
    sys.path.insert(0, f"{REPO}/lost3dsg/src/perception_module")
    import box_view  # noqa: F401
    import config  # noqa: F401
    os.makedirs(os.path.expanduser("~/.ssh"), mode=0o700, exist_ok=True)
    key = os.path.expanduser("~/.ssh/feed_key")
    with open(key, "w") as fh:
        fh.write(base64.b64decode(os.environ["FEED_SSH_KEY_B64"]).decode().strip() + "\n")
    os.chmod(key, 0o600)
    with open(os.path.expanduser("~/.ssh/known_hosts"), "w") as fh:
        fh.write(os.environ["FEED_SSH_KNOWN_HOSTS"].strip() + "\n")
    target = os.environ["FEED_SSH_TARGET"]
    ssh = ["ssh", "-i", key, "-o", "BatchMode=yes"]
    os.chdir("/tmp")
    with open("/tmp/big.bin", "wb") as fh:          # 100 MB for the stack host to pull through -R
        fh.write(os.urandom(100_000_000))
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 7799), http.server.SimpleHTTPRequestHandler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    tunnel = subprocess.Popen(ssh + ["-N", "-o", "ExitOnForwardFailure=yes", "-R", "127.0.0.1:7799:127.0.0.1:7799",
                                     "-L", "127.0.0.1:8081:127.0.0.1:8081", target])
    time.sleep(4)
    out = {"tunnel_alive": tunnel.poll() is None}
    t0 = time.time()
    try:
        out["bridge_get"] = urllib.request.urlopen("http://127.0.0.1:8081/", timeout=10).status
    except Exception as e:
        out["bridge_get"] = repr(e)[:120]
    out["bridge_ms"] = round(1000 * (time.time() - t0))
    os.makedirs("/tmp/rc", exist_ok=True)
    open("/tmp/rc/ping.txt", "w").write("ping")
    r1 = subprocess.run(["rsync", "-a", "-e", " ".join(ssh), "/tmp/rc/ping.txt", f"{target}:_tunnel_check/"],
                        capture_output=True, text=True)
    r2 = subprocess.run(["rsync", "-a", "-e", " ".join(ssh), f"{target}:_tunnel_check/ping.txt", "/tmp/rc/back.txt"],
                        capture_output=True, text=True)
    out["rsync_push"], out["rsync_pull"] = r1.returncode, r2.returncode
    out["rsync_err"] = (r1.stderr + r2.stderr)[-200:]
    print("[tunnel-check] holding 127.0.0.1:7799 open for", hold_s, "s", flush=True)
    time.sleep(hold_s)
    tunnel.terminate()
    return out


@app.local_entrypoint()
def check():
    print(tunnel_check.remote())


@app.local_entrypoint()
def main(spec: str):
    """spec: path of a JSON file {"env": {...}, "files": [paths to send]} written by remote_feed.sh."""
    s = json.load(open(spec))
    files = {p: open(p).read() for p in s["files"]}
    rc = feed.remote(s["env"], files)
    raise SystemExit(rc)
