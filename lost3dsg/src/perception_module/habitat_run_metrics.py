#!/usr/bin/env python3
"""Metriche operative di un run Habitat, senza ground truth o manifest HOV-SG.

Modalita separata:
  habitat_run_metrics.py start --run-dir /path/output --state /tmp/habitat-run.json
  # avvia e ferma ros2 launch normalmente
  habitat_run_metrics.py report --state /tmp/habitat-run.json --output metrics.json
"""
from __future__ import annotations
import argparse, json, math, os, re, signal, subprocess, sys, time
from pathlib import Path

SERIES = ("frame_poses.jsonl", "perception_latencies.jsonl", "hook_decisions.jsonl")
LOG_GLOBS = ("*.log", "logs/*.log")

def _state(run_dir: Path):
    files = {}
    for name in SERIES:
        p=run_dir/name; files[name]=p.stat().st_size if p.exists() else 0
    for pattern in LOG_GLOBS:
        for p in run_dir.glob(pattern): files[str(p.relative_to(run_dir))]=p.stat().st_size
    return {"run_dir":str(run_dir.resolve()),"started_at":time.time(),"offsets":files}

def _json(path, default):
    try: return json.loads(path.read_text(encoding="utf-8"))
    except (OSError,ValueError): return default

def _new_jsonl(path, offset=0):
    if not path.exists(): return []
    rows=[]
    with path.open("rb") as f:
        f.seek(min(offset,path.stat().st_size))
        if offset: f.readline()  # discard an old partial line
        for raw in f:
            try: rows.append(json.loads(raw))
            except (UnicodeDecodeError,ValueError): pass
    return rows

def _during_run(rows, started_at):
    """Select this run even when a producer truncates and recreates its JSONL."""
    result=[]
    for row in rows:
        stamp=row.get("t",row.get("stamp",row.get("last_updated")))
        try:
            if float(stamp) >= float(started_at): result.append(row)
        except (TypeError,ValueError): pass
    return result

def _wait_for_final_artifacts(root, started_at, timeout=60):
    """ROS launch may return before its nodes complete their final JSON writes."""
    required=("frame_poses.jsonl","persistent_perception.json","room.json","bev_data.json")
    deadline=time.monotonic()+timeout; stable_since=None; previous=None
    last_notice=0
    while time.monotonic()<deadline:
        signature=[]; all_fresh=True
        for name in required:
            p=root/name
            if not p.exists() or p.stat().st_mtime < started_at or p.stat().st_size==0:
                all_fresh=False; signature.append((name,None,None))
            else: signature.append((name,p.stat().st_size,p.stat().st_mtime_ns))
        signature=tuple(signature)
        if all_fresh and signature==previous:
            stable_since=stable_since or time.monotonic()
            if time.monotonic()-stable_since>=3: return True
        else: stable_since=None
        now=time.monotonic()
        if now-last_notice>=5:
            waiting=[name for name,size,_mtime in signature if size is None]
            print("[metriche] attendo il flush finale di Habitat: "+
                  (", ".join(waiting) if waiting else "file ancora in aggiornamento"),
                  file=sys.stderr,flush=True)
            last_notice=now
        previous=signature; time.sleep(.5)
    print("[metriche] timeout: alcuni artefatti finali non sono arrivati",file=sys.stderr,flush=True)
    return False

def _run_process_group(command, shutdown_timeout=90):
    """Run ROS in its own group and do not return while descendant nodes are alive."""
    process=subprocess.Popen(command,start_new_session=True)
    try:
        return process.wait(), "terminato"
    except KeyboardInterrupt:
        # The child is in another session, therefore the terminal Ctrl-C reaches us only.
        # Forward it to every process spawned by ros2 launch, not just its parent CLI.
        try: os.killpg(process.pid,signal.SIGINT)
        except ProcessLookupError: pass
        try:
            process.wait(timeout=shutdown_timeout)
            return 130, "Ctrl-C; gruppo ROS chiuso correttamente"
        except subprocess.TimeoutExpired:
            try: os.killpg(process.pid,signal.SIGTERM)
            except ProcessLookupError: pass
            try: process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                try: os.killpg(process.pid,signal.SIGKILL)
                except ProcessLookupError: pass
                process.wait()
            return 130, "Ctrl-C; timeout di chiusura ROS, gruppo terminato"

def _distance(rows):
    points=[]
    for r in rows:
        try: points.append((float(r["x"]),float(r["y"]),float(r["z"])))
        except (KeyError,TypeError,ValueError): pass
    return sum(math.dist(a,b) for a,b in zip(points,points[1:]))

def _percentile(values, q):
    if not values: return None
    values=sorted(values); k=(len(values)-1)*q; lo=int(k); hi=min(lo+1,len(values)-1)
    return values[lo]+(values[hi]-values[lo])*(k-lo)

def collect(state, ended_at=None, exit_code=None, stop_reason=None):
    root=Path(state["run_dir"]); offsets=state.get("offsets",{})
    # Byte offsets are unsafe here: launch can truncate a series and regrow it past its
    # previous size. Timestamps unambiguously select rows belonging to this run.
    poses=_during_run(_new_jsonl(root/"frame_poses.jsonl"),state["started_at"])
    lat=_during_run(_new_jsonl(root/"perception_latencies.jsonl"),state["started_at"])
    decisions=_during_run(_new_jsonl(root/"hook_decisions.jsonl"),state["started_at"])
    if not lat:
        latest=_json(root/"perception_latencies.json",{})
        if latest and float(latest.get("last_updated",0)) >= state["started_at"]: lat=[latest]
    persistent=_json(root/"persistent_perception.json",[])
    actual=_json(root/"actual_perceptions.json",[])
    room=_json(root/"room.json",{}); bev=_json(root/"bev_data.json",{})
    timestamps=[float(r["stamp"]) for r in poses if r.get("stamp") is not None]
    span=max(timestamps)-min(timestamps) if len(timestamps)>1 else 0
    cycle=[float(r["cycle_ms"]) for r in lat if r.get("cycle_ms") is not None]
    detections=[int(r.get("n_detections",0)) for r in lat if r.get("n_detections") is not None]
    if not detections:
        detections=[int(r.get("detections",0)) for r in decisions if r.get("kind")=="tracking_scan_summary"]
    collisions=0; collision_source=False
    for pattern in LOG_GLOBS:
        for p in root.glob(pattern):
            rel=str(p.relative_to(root)); start=offsets.get(rel,0)
            try:
                with p.open("rb") as f: f.seek(min(start,p.stat().st_size)); text=f.read().decode(errors="replace")
                # Counts positive collision events, not messages publishing false.
                collisions += len(re.findall(r"(?:collision|collided)[^\n]{0,40}(?:true|yes|1)\b",text,re.I)); collision_source=True
            except OSError: pass
    representation_names=("persistent_perception.json","room.json","clip_embeddings.json",
                          "knowledge_graph.ttl","map.db","rtabmap.db","tiago_temporal_map_5.db")
    rep={};
    for name in representation_names:
        p=root/name
        if p.is_file(): rep[name]=p.stat().st_size
    rooms=room.get("rooms",[]) if isinstance(room,dict) else []
    active_rooms=[r for r in rooms if not isinstance(r,dict) or r.get("active",True) is not False]
    confirmed_rooms=[r for r in rooms if isinstance(r,dict) and bool(r.get("confirmed"))]
    segmentation=room.get("segmentation",{}) if isinstance(room,dict) else {}
    end=ended_at or time.time()
    return {
      "run":{"directory":str(root),"started_at_unix":state["started_at"],"ended_at_unix":end,
             "duration_s":round(end-state["started_at"],3),"exit_code":exit_code,"stop_reason":stop_reason},
      "frames":{"rendered":len(poses),"processed_perception_cycles":len(lat),
                "rendered_fps":round(len(poses)/span,3) if span>0 else None},
      "perception":{"detections_total":sum(detections) if detections else None,
                    "detections_last_cycle":len(actual) if isinstance(actual,list) else None,
                    "persistent_objects":len(persistent) if isinstance(persistent,list) else None,
                    "cycle_latency_ms_mean":round(sum(cycle)/len(cycle),3) if cycle else None,
                    "cycle_latency_ms_p50":round(_percentile(cycle,.5),3) if cycle else None,
                    "cycle_latency_ms_p95":round(_percentile(cycle,.95),3) if cycle else None},
      "scene":{"rooms":len(active_rooms),"rooms_active":len(active_rooms),
               "rooms_total_records":len(rooms),"rooms_confirmed_records":len(confirmed_rooms),
               "regions":segmentation.get("regions_after_small_merge"),
               "floors":len(bev.get("floors",[])) if isinstance(bev,dict) and "floors" in bev else None,
               "floor_heights_m":bev.get("floors") if isinstance(bev,dict) else None},
      "trajectory":{"samples":len(poses),"distance_m":round(_distance(poses),3) if poses else None},
      "collisions":{"count":collisions if collision_source else None,
                    "note":None if collision_source else "nessun log collisioni disponibile"},
      "representation":{"size_mb":round(sum(rep.values())/1e6,6),
                        "files_mb":{k:round(v/1e6,6) for k,v in rep.items()}},
      "missing_sources":[name for name in SERIES if not (root/name).exists()]}

def write(report,path):
    text=json.dumps(report,indent=2,ensure_ascii=False,allow_nan=False); print(text)
    if path: path.parent.mkdir(parents=True,exist_ok=True); path.write_text(text+"\n",encoding="utf-8")

def main():
    ap=argparse.ArgumentParser(description=__doc__); sub=ap.add_subparsers(dest="action",required=True)
    start=sub.add_parser("start"); start.add_argument("--run-dir",type=Path,required=True); start.add_argument("--state",type=Path,required=True)
    report=sub.add_parser("report"); report.add_argument("--state",type=Path,required=True); report.add_argument("--output",type=Path,required=True)
    run=sub.add_parser("run"); run.add_argument("--run-dir",type=Path,required=True); run.add_argument("--output",type=Path,required=True); run.add_argument("command",nargs=argparse.REMAINDER)
    a=ap.parse_args()
    if a.action=="start":
        a.state.parent.mkdir(parents=True,exist_ok=True); a.state.write_text(json.dumps(_state(a.run_dir),indent=2)+"\n"); print(a.state); return 0
    state=_json(a.state,{}) if a.action=="report" else _state(a.run_dir)
    if not state: ap.error("state non valido")
    if a.action=="report":
        write(collect(state,stop_reason="launch eseguito separatamente"),a.output); return 0
    cmd=a.command[1:] if a.command[:1]==["--"] else a.command
    if not cmd: ap.error("run richiede un comando dopo --")
    code,reason=_run_process_group(cmd)
    # At this point the complete ROS process tree is gone. This short stability check is
    # only for filesystem flushes; it no longer masks nodes still writing in background.
    ready=_wait_for_final_artifacts(Path(state["run_dir"]),state["started_at"],timeout=60)
    if not ready:
        reason += "; timeout artefatti Habitat"
    result=collect(state,exit_code=code,stop_reason=reason)
    if result["frames"]["rendered"] == 0 and not result["representation"]["files_mb"]:
        print("[metriche] ERRORE: nessun artefatto del run; il file di output non viene sovrascritto.",
              file=sys.stderr,flush=True)
        return 2
    write(result,a.output); return code
if __name__=="__main__": raise SystemExit(main())
