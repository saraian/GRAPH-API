#!/usr/bin/env python3
"""Per-process resource usage + per-model location inventory for live runs.

One JSON snapshot per invocation (--once) or a sampling loop (--interval S).
Writes to stdout; live_run.sh redirects it to $OUT_DIR/model_resources.json.

Uses only what already exists on this machine (no new deps):
  - nvidia-smi --query-compute-apps / --query-gpu   (per-PID VRAM, device totals)
  - docker top                                      (host-PID -> container cmdline)
  - psutil                                          (per-process CPU/RSS)
"""

import argparse
import json
import os
import re
import subprocess
import time

try:
    import psutil
except ImportError:
    psutil = None

OUT = {
    "timestamp": None,
    "gpu": [],
    "processes": [],
    "models": [],
}

# role -> (regex over cmdline, human label)
ROLES = [
    ("sim_render", r"habitat_feed_host\.py", "Habitat sim render + control server (host GPU/EGL)"),
    ("ros_feed_node", r"habitat_feed_node", "ROS frame/TF relay from TCP feed"),
    ("rtabmap", r"rtabmap_slam/rtabmap", "RTAB-Map SLAM"),
    ("perception", r"perception_2\.py", "OWLv2+CLIP & SAM inference, VLM client"),
    ("object_manager", r"object_manager_6", "World model / object lifecycle / graph writer"),
    ("bridge", r"graph_api_bridge", "Dashboard HTTP bridge"),
]


def sh(cmd):
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=20).stdout
    except Exception:
        return ""


def gpu_snapshot():
    rows = []
    q = sh(["nvidia-smi", "--query-gpu=name,memory.used,memory.total,utilization.gpu",
            "--format=csv,noheader,nounits"])
    for line in q.strip().splitlines():
        p = [x.strip() for x in line.split(",")]
        if len(p) == 4:
            rows.append({"name": p[0], "mem_used_mib": int(p[1]),
                         "mem_total_mib": int(p[2]), "util_pct": int(p[3])})
    return rows


def compute_apps():
    """pid -> VRAM MiB for processes holding GPU memory."""
    out = {}
    q = sh(["nvidia-smi", "--query-compute-apps=pid,used_memory",
            "--format=csv,noheader,nounits"])
    for line in q.strip().splitlines():
        pid, _, mem = line.partition(",")
        try:
            out[int(pid)] = int(mem.strip())
        except ValueError:
            continue
    return out


def container_cmds():
    """host-PID -> cmdline for every process inside graphapi_live."""
    cmds = {}
    top = sh(["docker", "top", "graphapi_live", "-eo", "pid,args"])
    for line in top.splitlines()[1:]:
        pid, _, args = line.partition(" ")
        if pid.isdigit():
            cmds[int(pid)] = args.strip()
    return cmds


def classify(args):
    for role, pat, _ in ROLES:
        if re.search(pat, args):
            return role
    return None


def build_inventory():
    """Per-model location breakdown from the active GRAPH_API_CONFIG yaml."""
    cfg_path = os.environ.get("GRAPH_API_CONFIG", "")
    models = []
    cfg = {}
    cfg_read = False
    if cfg_path and os.path.isfile(cfg_path):
        try:
            import yaml
            with open(cfg_path) as f:
                cfg = yaml.safe_load(f) or {}
            cfg_read = True
        except Exception:
            cfg = {}
    if not cfg_read:
        # GA-36. "No endpoint configured" is a MEASUREMENT of the config; a config that could not
        # be read yields no measurement at all, and writing the fallback row here made an unread
        # file indistinguishable from a deliberately endpoint-less run in every bundle.
        return [{"model": "unknown", "role": "open-vocab labels", "location": "unknown",
                 "reason": ("GRAPH_API_CONFIG not set" if not cfg_path else
                            f"config not readable: {cfg_path}"),
                 "runs_in": "perception"}]
    vlm = cfg.get("vlm", {}) or {}
    if vlm.get("base_url"):
        models.append({"model": vlm.get("model", "?"), "role": "open-vocab labels",
                       "location": "endpoint", "endpoint": vlm["base_url"],
                       "runs_in": "perception"})
    else:
        models.append({"model": "static fallback labels", "role": "open-vocab labels",
                       "location": "none (no endpoint configured)",
                       "labels": len(vlm.get("fallback_labels", [])),
                       "runs_in": "perception"})
    paths = cfg.get("paths", {}) or {}
    for key, name, role in (("vitsam_encoder", "EfficientViT-SAM encoder", "segmentation"),
                            ("vitsam_decoder", "EfficientViT-SAM decoder", "segmentation")):
        p = paths.get(key)
        if p:
            models.append({"model": name, "role": role, "location": "local-onnx",
                           "path": p, "runs_in": "perception"})
    w2v = (cfg.get("embedding", {}) or {}).get("word2vec_path")
    if w2v:
        models.append({"model": "word2vec", "role": "semantic embeddings",
                       "location": "local", "path": w2v, "runs_in": "perception"})
    # OWLv2 CLIP backbone is loaded from the HF cache mount, not the config
    hf = "/DATA/huggingface_cache"
    models.append({"model": "OWLv2 (CLIP backbone)", "role": "detection + crop embeddings",
                   "location": "local-hf-cache", "path": hf if os.path.isdir(hf) else "?",
                   "runs_in": "perception"})
    return models


def sample():
    snap = dict(OUT)
    snap["timestamp"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    snap["gpu"] = gpu_snapshot()
    vram = compute_apps()
    ccmds = container_cmds()

    procs = []
    seen = set()
    candidates = []
    for pid, args in ccmds.items():
        candidates.append((pid, args, "container"))
    if psutil:
        me_host_cmds = {}
        for p in psutil.process_iter(["pid", "cmdline"]):
            try:
                cl = p.info["cmdline"]
                if cl:
                    me_host_cmds[p.info["pid"]] = " ".join(cl)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        for pid, args in me_host_cmds.items():
            if pid not in ccmds:
                candidates.append((pid, args, "host"))

    now = time.time()
    for pid, args, where in candidates:
        if re.search(r"(?:^|/)bin/ros2 (?:run|launch)\b|^ros2 (?:run|launch)\b", args):
            continue  # launcher stub; the real node process is a separate PID
        role = classify(args)
        if not role or pid in seen:
            continue
        seen.add(pid)
        entry = {"role": role, "pid": pid, "where": where,
                 "cmd": args[:120], "gpu_mem_mib": vram.get(pid)}
        if psutil:
            try:
                pr = psutil.Process(pid)
                entry["cpu_percent"] = pr.cpu_percent(interval=None)
                # cpu_percent needs two samples; prime it and fall back to /proc stat
                if entry["cpu_percent"] == 0.0:
                    with open(f"/proc/{pid}/stat") as f:
                        fields = f.read().split()
                    utime, stime = int(fields[13]), int(fields[14])
                    hz = os.sysconf("SC_CLK_TCK")
                    uptime_s = now - psutil.boot_time()
                    start_s = float(fields[21]) / hz
                    entry["cpu_percent"] = round(
                        (utime + stime) / hz / max(uptime_s - start_s, 1e-6) * 100, 1)
                entry["rss_mb"] = round(pr.memory_info().rss / 1e6, 1)
                entry["num_threads"] = pr.num_threads()
            except (psutil.NoSuchProcess, psutil.AccessDenied, Exception):
                pass
        procs.append(entry)

    procs.sort(key=lambda e: (e.get("cpu_percent", 0) or 0), reverse=True)
    snap["processes"] = procs
    snap["models"] = build_inventory()
    return snap


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--interval", type=float, default=0,
                    help="sample forever every N seconds (default: one shot)")
    ap.add_argument("--out", default="-", help="output file (default stdout)")
    args = ap.parse_args()

    def emit(snap):
        data = json.dumps(snap, indent=2)
        if args.out == "-":
            print(data)
        else:
            tmp = args.out + ".tmp"
            with open(tmp, "w") as f:
                f.write(data + "\n")
            os.replace(tmp, args.out)

    if args.interval <= 0:
        emit(sample())
        return
    while True:
        emit(sample())
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
