"""Launch service used by CLI, dashboard and batch. Algorithms stay in existing runtimes."""
from __future__ import annotations

import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import yaml

from .configuration import environment, resolve_config, workflow_name, tiago_map_source, node_profile
from .docker import ensure_image
from .registry import Registry, alive, container_alive, owned_containers, process_identity, write_json
from .paths import output_directory
from .stack_health import pane_exit
from .credentials import prepare_vlm_credentials, uses_vlm


def runtime(root, name):
    return root / "graphapi_cli/runtime" / (name + ".sh")


def stop_simulation(state, close_timeout="120"):
    """Stop the owned workload before its launcher, preserving application cleanup."""
    actor = state.get("actor")
    launcher = state["container"] + "-launcher"
    names = [name for name in owned_containers(state["operation_id"])
             if name not in (actor, launcher)]
    if names:
        subprocess.run(["docker", "stop", "-t", str(close_timeout), *names],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                       timeout=int(close_timeout) + 15)
    # The launcher normally exits when the application closes. Bound its wait
    # as well if it is still attached to another Docker client or a log tail.
    if launcher != actor and launcher in owned_containers(state["operation_id"]):
        subprocess.run(["docker", "stop", "-t", "15", launcher],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=30)
    remaining = [name for name in owned_containers(state["operation_id"]) if name != actor]
    if remaining:
        raise ValueError("simulation containers did not stop: " + ", ".join(remaining))


def prepare_job(root, request, inherited=None):
    env, _ = environment(root, request.get("local"), inherited)
    workspace = Path(env["WORKSPACE_ROOT"])
    mode = request["mode"]
    map_source = tiago_map_source(mode, request.get("map_source"), env)
    if map_source is not None:
        request["map_source"] = map_source
        env["FOUND_START_RTABMAP"] = "1" if map_source == "slam" else "0"
        # The explicit input kind takes precedence over leftover shell settings.
        if mode == "tiago":
            env.pop("TIAGO_BAG_PATH", None)
            env.pop("TIAGO_ROSBAG", None)
            env["TIAGO_BAG_MODE"] = "0"
        else:
            env["TIAGO_BAG_FILTER_CONFLICTING"] = env["FOUND_START_RTABMAP"]
    if mode == "bag":
        bag = Path(request["bag"]).resolve()
        if not (bag / "metadata.yaml").is_file():
            raise ValueError(f"rosbag directory or metadata.yaml not found: {bag}")
        request["bag"] = str(bag)
        # The detached controller needs this input mounted before execute()
        # starts; setting the bag directory only in execute() is too late.
        env["TIAGO_BAG_DIR"] = str(bag.parent)
    if mode == "sim" and (not env.get("HM3D_ROOT") or not Path(env["HM3D_ROOT"]).is_dir()):
        raise ValueError("register the scene dataset first: graphapi setup sim --dataset /path/to/hm3d")
    selected = request.get("config")
    profile = request.get("profile")
    if not selected and profile and profile not in ("sim", "sim-gt", "tiago", "bag"):
        selected = "config/" + (profile if profile.endswith(".yaml") else profile + ".yaml")
    if not selected and mode in ("tiago", "bag"):
        selected = node_profile(mode, map_source)
    overrides = {}
    if request.get("pose") or profile == "sim-gt":
        pose = request.get("pose") or "simulator"
        overrides = {"habitat": {"localization_mode": "ground_truth" if pose == "simulator" else "rtabmap"},
                     "run": {"pose_source": pose}}
    source, cfg = resolve_config(root, selected, request.get("local"), overrides)
    if uses_vlm(request, env):
        prepare_vlm_credentials(root, cfg, env)
    if cfg.get("vlm", {}).get("api_key"):
        cfg["vlm"]["api_key"] = ""
    if cfg.get("perception", {}).get("modal_endpoint"):
        env.setdefault("MODAL_PERCEPTION_URL", cfg["perception"]["modal_endpoint"])
        cfg["perception"]["modal_endpoint"] = ""
    # Presentation defaults are explicit; experiment/tour values remain untouched.
    if mode == "sim":
        env.setdefault("RVIZ", "1" if request.get("gui") else "0")
        env.setdefault("FEED_SHOW", "1" if request.get("gui") else "0")
        if request.get("gui"):
            env["RVIZ"] = env["FEED_SHOW"] = "1"
    elif request.get("gui"):
        env["FOUND_START_RVIZ"] = "1"
    env["GRAPHAPI_RECORD"] = "0" if request.get("no_record") else "1"
    if request.get("no_perception"):
        env["FOUND_START_PERCEPTION"] = "0"
        if mode == "sim":
            env["MAPPING_ONLY"] = "1"
    if mode in ("tiago", "bag") and env.get("FOUND_START_PERCEPTION") == "0":
        request["no_perception"] = True
    for flag, key in [("floor", "FEED_SPAWN_FLOOR"), ("floors", "HOUSE_FLOORS"),
                      ("visits", "MULTI_FLOOR_SEQUENCE"), ("transforms", "MULTI_FLOOR_TRANSFORMS"),
                      ("tour_schedule", "FEED_SCHEDULE"), ("map", "RTABMAP_LOCALIZE_DB")]:
        if request.get(flag) is not None:
            env[key] = str(request[flag])
    if request.get("mapping_only"):
        env["MAPPING_ONLY"] = "1"
    gpu = str(request.get("gpu") or env.get("BASELINE_GPU") or env.get("GRAPH_API_GPUS", "device=0").removeprefix("device="))
    if not gpu.isdecimal():
        raise ValueError("--gpu must identify one GPU index")
    env["GRAPH_API_GPUS"] = "device=" + gpu
    env["CUDA_VISIBLE_DEVICES"] = gpu
    for directory in ("results", "maps", "ws", "schedules"):
        output_directory(workspace / directory)
    reg = Registry(workspace)
    domain = int(env.get("TIAGO_ROS_DOMAIN_ID", "1")) if mode == "tiago" else (int(env["ROS_DOMAIN_ID"]) if env.get("ROS_DOMAIN_ID") else None)
    bridge_key = "TIAGO_BRIDGE_PORT" if mode != "sim" else "BRIDGE_PORT"
    ports = {bridge_key: int(env[bridge_key]) if env.get(bridge_key) else None}
    if mode == "sim":
        ports.update({key: int(env[key]) if env.get(key) else None for key in ("FEED_PORT", "FEED_CTRL_PORT")})
    private_container = env.get("GRAPHAPI_PAL_CONTAINER", env["TIAGO_DOCKER_TARGET"] + "-dev")
    op, allocated, domain = reg.reserve(mode, gpu=gpu, robot=env["TIAGO_DOCKER_TARGET"] if mode == "tiago" else None,
                                        ports=ports, domain=domain,
                                        extra=("pal-container:" + private_container,) if mode != "sim" else ())
    try:
        env.update({k: str(v) for k, v in allocated.items()})
        env["ROS_DOMAIN_ID"] = str(domain)
        if mode == "bag":
            env["TIAGO_BAG_DOMAIN_ID"] = str(domain)
        env["GRAPHAPI_OPERATION_ID"] = op
        env["GRAPH_API_RUN_ID"] = op
        env["GRAPH_API_CONTAINER_NAME"] = "graphapi-" + op.lower()
        env["GRAPH_API_AUTOSTART"] = "0"
        bridge = allocated[bridge_key]
        env["GRAPH_API_BASE_URL"] = f"http://127.0.0.1:{bridge}"
        env["GRAPHAPI_RESOLVED_CONFIG"] = "1"
        env["WS_DIR"] = str(workspace / "ws" / op)
        output = reg.operations / op
        output.mkdir(parents=True)
        resolved = root / "config/.runtime" / (op + ".yaml")
        resolved.parent.mkdir(parents=True, exist_ok=True)
        resolved.write_text(yaml.safe_dump(cfg, sort_keys=False))
        resolved.chmod(0o600)
        env["GRAPH_API_CONFIG"] = str(resolved)
        env["GRAPHAPI_RESULT_FILE"] = str(output / "bundles.txt")
        env["GRAPH_API_RUNS_DIR"] = str(workspace / "results")
        if mode == "sim":
            cfg.setdefault("services", {}).update(feed_port=allocated["FEED_CTRL_PORT"], bridge_port=bridge)
            # These are allocated endpoints, not experiment parameters.
            resolved.write_text(yaml.safe_dump(cfg, sort_keys=False))
            env["HF_CACHE"] = env["HF_SHARED_CACHE"]
        else:
            env.setdefault("FOUND_TMUX_SESSION", "found_tiago")
        state = {"operation_id": op, "mode": mode, "state": "PREPARING", "pid": os.getpid(),
                 "process_identity": process_identity(os.getpid()), "project": str(root),
                 "actor": env.get("GRAPHAPI_RESOURCE_ACTOR"),
                 "workspace": str(workspace), "config_source": str(source), "config": str(resolved),
                 "log": str(output / "launch.log"), "bundles": [], "ports": allocated,
                 "bridge_url": env["GRAPH_API_BASE_URL"], "domain": domain, "gpu": gpu,
                 "container": env.get("GRAPHAPI_PAL_CONTAINER", env["TIAGO_DOCKER_TARGET"] + "-dev") if mode != "sim" else env["GRAPH_API_CONTAINER_NAME"],
                 "session": env.get("FOUND_TMUX_SESSION"), "scene": request.get("scene"), "created": time.time(), "recording": env["GRAPHAPI_RECORD"] == "1"}
        from .rviz_config import settings as viewer_settings
        state["viewer"] = viewer_settings(mode, cfg, env)
        if mode in ("tiago", "bag"):
            state.update(workflow=workflow_name(mode), platform="tiago", runtime="private-pal",
                         map_source=map_source,
                         perception=env.get("FOUND_START_PERCEPTION", "1") != "0",
                         map_frame=cfg.get("tf", {}).get("world_frame"),
                         map_topic=cfg.get("bev", {}).get("map_topic"))
            if mode == "bag":
                state.update(bag=request["bag"], rate=request.get("rate", 1.0), loop=bool(request.get("loop")))
            else:
                state["robot_address"] = env.get("TIAGO_ROBOT_IP", "10.68.0.1")
        write_json(output / "status.json", state)
        return env, state
    except BaseException:
        reg.release(op)
        raise


def start(root, request, *, detach=False, inherited=None):
    env, state = prepare_job(root, request, inherited)
    if not detach:
        return execute(root, request, env, state)
    payload = Path(state["log"]).parent / "job.json"
    write_json(payload, {"request": request, "state": state})
    try:
        env["GRAPHAPI_DETACHED"] = "1"
        actor = env.get("GRAPHAPI_RESOURCE_ACTOR")
        reg = Registry(state["workspace"])
        if actor:
            from .bootstrap import controller_command
            supervisor = "graphapi-" + state["operation_id"].lower() + "-supervisor"
            state.update(actor=supervisor, pid=None, process_identity=None)
            write_json(payload, {"request": request, "state": state})
            write_json(Path(state["log"]).parent / "status.json", state)
            args = controller_command(root, env,
                                      ["-c", 'exec python3 -m graphapi_cli.cli _job "$1" >>"$2" 2>&1',
                                       "graphapi", str(payload), state["log"]],
                                      name=supervisor, operation=state["operation_id"], executable="bash")
            args[2:2] = ["-d"]
            # Wait until Docker accepts the independent supervisor. A background
            # Docker client would die with this short-lived CLI container.
            subprocess.run(args, cwd=root, env=env, check=True, capture_output=True, text=True)
            # The worker may already have published its PID. Never overwrite it.
            reg.db.execute("UPDATE leases SET actor=? WHERE operation=?", (supervisor, state["operation_id"]))
            reg.db.commit()
            return json.loads((Path(state["log"]).parent / "status.json").read_text())
        with open(state["log"], "ab", buffering=0) as log:
            args = [sys.executable, "-m", "graphapi_cli.cli", "_job", str(payload)]
            child = subprocess.Popen(args,
                                     cwd=root, env=env, stdin=subprocess.DEVNULL,
                                     stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        reg.db.execute("UPDATE leases SET pid=?,identity=? WHERE operation=?",
                       (child.pid, process_identity(child.pid), state["operation_id"]))
        reg.db.commit()
        state.update(pid=child.pid, process_identity=process_identity(child.pid))
        # The child writes its own status; do not overwrite a newer RUNNING/FAILED state.
        return state
    except BaseException as exc:
        state.update(state="FAILED", error=str(exc), returncode=1, finished=time.time())
        write_json(Path(state["log"]).parent / "status.json", state)
        if not owned_containers(state["operation_id"]):
            Registry(state["workspace"]).release(state["operation_id"])
        raise


def execute(root, request, env, state):
    status = Path(state["log"]).parent / "status.json"
    reg = Registry(state["workspace"])
    state.update(pid=os.getpid(), process_identity=process_identity(os.getpid()))
    write_json(status, state)
    reg.db.execute("UPDATE leases SET pid=?,identity=?,actor=? WHERE operation=?", (os.getpid(), state["process_identity"], state.get("actor"), state["operation_id"]))
    reg.db.commit()
    stopping = False
    process = None
    private_owned = False
    monitor_done = None

    def interrupt(signum, _frame):
        nonlocal stopping
        if stopping:
            return
        stopping = True
        state["state"] = "DRAINING"
        write_json(status, state)
        if request["mode"] == "sim":
            # SIGINT to the Docker client is insufficient for a detached shell.
            stop_simulation(state, env.get("RTABMAP_CLOSE_TIMEOUT", "120"))
        if process and process.poll() is None:
            os.killpg(process.pid, signal.SIGINT)

    handlers = {sig: signal.signal(sig, interrupt) for sig in (signal.SIGINT, signal.SIGTERM)}
    rc = 1
    try:
        mode = request["mode"]
        if mode == "sim":
            ensure_image(root, env)
            from .docker import run_python
            models = Path(env["SAM_MODEL_DIR"])
            if not all((models / name).is_file() for name in ("l2_encoder.onnx", "l2_decoder.onnx")):
                models.mkdir(parents=True, exist_ok=True)
                Path(env["HF_SHARED_CACHE"]).mkdir(parents=True, exist_ok=True)
                prepare_env = dict(env, GRAPHAPI_PREPARE="1", HF_HOME=env["HF_SHARED_CACHE"])
                if run_python(root, prepare_env, [str(root / "graphapi_cli/prepare_models.py"), str(models)]):
                    raise ValueError("model preparation failed")
            args = ["bash", str(runtime(root, "run_sim")), "--config", env["GRAPH_API_CONFIG"]]
            if request.get("scene"):
                args.append(request["scene"])
            if request.get("one_storey") or request.get("floor") is not None:
                args.append("--one-storey")
            if request.get("visits"):
                args.append("--multi-floor")
            args += request.get("legacy_args", [])
            from .docker import command
            args = command(root, env, args[1:], executable="bash", control=True, gpu=True,
                           name=env["GRAPH_API_CONTAINER_NAME"] + "-launcher")
        else:
            action = "physical" if mode == "tiago" else "bag"
            args = ["bash", str(runtime(root, "run_tiago")), action]
            if mode == "bag":
                bag = Path(request["bag"])
                env["TIAGO_BAG_DIR"] = str(bag.parent)
                args.append(bag.name)
            if request.get("map_source"):
                args.append("--rtabmap" if request["map_source"] == "slam" else "--no-rtabmap")
            if request.get("rate"):
                args += ["--rate", str(request["rate"])]
            if request.get("loop"):
                args.append("--loop")
            container = env.get("GRAPHAPI_PAL_CONTAINER")
            if container:
                args += ["--container", container]
            args += request.get("legacy_args", [])
        state["state"] = "PREPARING"
        write_json(status, state)
        print(f"operation: {state['operation_id']}\nbridge: {state['bridge_url']}", flush=True)
        if mode in ("tiago", "bag"):
            print(f"workflow: {state.get('workflow', workflow_name(mode))}\n"
                  f"runtime: private PAL Docker — {state['container']}\n"
                  f"input: {state.get('bag') or state.get('robot_address', 'physical TIAGO')}\n"
                  f"map source: {state.get('map_source', request.get('map_source'))}\n"
                  f"perception: {'enabled' if state.get('perception', True) else 'disabled'}\n"
                  f"recording: {'enabled' if state['recording'] else 'disabled'}\n"
                  f"config: {state['config']}", flush=True)
        with open(state["log"], "ab", buffering=0) as log:
            process = subprocess.Popen(args, cwd=root, env=env, stdout=subprocess.PIPE,
                                       stderr=subprocess.STDOUT, start_new_session=True)
            import threading
            import urllib.request
            monitor_done = threading.Event()

            def readiness():
                while not monitor_done.wait(0.5):
                    if state["state"] != "PREPARING":
                        return
                    try:
                        with urllib.request.urlopen(state["bridge_url"] + "/health", timeout=1) as response:
                            health = json.load(response)
                            if health.get("stamp", {}).get("operation_id") != state["operation_id"] and not (request.get("resume") or "--resume" in request.get("legacy_args", [])):
                                continue
                            components = health.get("components", {})
                        ready = components.get("bridge", {}).get("active") and components.get("feed", {}).get("active")
                        if not request.get("no_perception") and not request.get("mapping_only"):
                            ready = ready and components.get("perception", {}).get("active") and components.get("object_manager", {}).get("active")
                        if ready and state["state"] == "PREPARING":
                            state["state"] = "RUNNING"
                            state["ready"] = time.time()
                            write_json(status, state)
                            return
                    except (OSError, ValueError):
                        pass

            monitor = threading.Thread(target=readiness, daemon=True)
            monitor.start()
            for line in iter(process.stdout.readline, b""):
                if env.get("GRAPHAPI_DETACHED") != "1":
                    log.write(line)
                sys.stdout.buffer.write(line)
                sys.stdout.flush()
            rc = process.wait()
            if mode == "sim" or rc != 0:
                monitor_done.set()
                monitor.join(timeout=2)
        if mode != "sim" and rc == 0:
            identity = subprocess.run(["docker", "exec", state["container"], "cat", "/ws/output/operation_id"],
                                      capture_output=True, text=True)
            private_owned = identity.returncode == 0 and identity.stdout.strip() == state["operation_id"]
            if not private_owned and (request.get("resume") or "--resume" in request.get("legacy_args", [])):
                state["resumed_from"] = identity.stdout.strip()
                state["configuration_applied"] = False
                adopted = subprocess.run(["docker", "exec", "-u", "user", state["container"], "python3", "-c",
                                          "from pathlib import Path; import sys; Path('/ws/output/operation_id').write_text(sys.argv[1]+'\\n')", state["operation_id"]])
                private_owned = adopted.returncode == 0
            if not private_owned:
                raise ValueError("private session did not acknowledge this operation; leaving the existing session intact")
            reg.db.execute("UPDATE leases SET actor=? WHERE operation=?", ("pal:" + state["container"], state["operation_id"]))
            reg.db.commit()
            tmux = ["docker", "exec", "-u", "user", state["container"], "tmux", "has-session", "-t", state["session"]]
            pending_exits = {}
            while not stopping and subprocess.run(tmux, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode == 0:
                panes = subprocess.run(["docker", "exec", "-u", "user", state["container"], "tmux", "list-panes", "-s", "-t", state["session"],
                                        "-F", "#{window_name}|#{pane_dead}|#{pane_dead_status}"], capture_output=True, text=True)
                ended = False
                for line in panes.stdout.splitlines():
                    bag_code = None
                    if line.startswith("bag|1|"):
                        run_id = state.get("resumed_from") or state["operation_id"]
                        result = subprocess.run(["docker", "exec", state["container"], "cat", "/ws/output/bag_exit_" + run_id],
                                                capture_output=True, text=True)
                        if result.returncode == 0:
                            bag_code = result.stdout
                    event = pane_exit(line, pending_exits, time.monotonic(), bag_code)
                    if event is None:
                        continue
                    name, code = event
                    if name == "bag" and not request.get("loop") and code == "0":
                        ended = True
                    else:
                        state["error"] = f"required private component {name} exited ({code or 'status unavailable; possibly terminated by a signal'})"
                        rc = int(code) if code.isdecimal() and int(code) else 1
                        ended = True
                if ended:
                    break
                time.sleep(1)
            monitor_done.set()
            monitor.join(timeout=2)
            closed = stop_tiago(root, env, state)
            if closed:
                raise ValueError("private stack is still closing; output was not copied and its resource lease remains held")
            destination = status.parent / "bundle"
            copied = subprocess.run(["docker", "cp", state["container"] + ":/ws/output", str(destination)]).returncode
            if copied == 0:
                state["bundles"] = [str(destination)]
                needs_map = mode == "tiago" and request.get("map_source") != "robot" or request.get("map_source") == "slam"
                if needs_map:
                    import sqlite3
                    database = destination / "rtabmap.db"
                    try:
                        with sqlite3.connect(database.as_uri() + "?mode=ro", uri=True) as connection:
                            if connection.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
                                raise ValueError("private output database failed integrity_check")
                    except sqlite3.Error as exc:
                        raise ValueError(f"private output database could not be validated: {exc}") from exc
                if state["recording"] and not (destination / "recording/metadata.yaml").exists() and not state.get("resumed_from"):
                    rc = 1
                    state["error"] = "topic recording did not produce closed metadata"
            else:
                rc = copied
        else:
            result = status.parent / "bundles.txt"
            if result.exists():
                state["bundles"] = list(dict.fromkeys(result.read_text().splitlines()))
        state.update(returncode=rc, finished=time.time())
        state["state"] = "INTERRUPTED" if stopping or rc in (130, 143, -2, -15) else ("COMPLETED" if rc == 0 else "FAILED")
        if mode == "sim" and state["state"] == "COMPLETED":
            complete = bool(state["bundles"])
            for bundle in state["bundles"]:
                try:
                    terminal = json.loads((Path(bundle) / "terminating_node.json").read_text())["node"]
                    complete &= terminal in ("FEED_ENDED", "MAPPING_TIME", "CAP")
                    if state["recording"]:
                        closed = json.loads((Path(bundle) / "recording_status.json").read_text())["closed"]
                        complete &= closed and (Path(bundle) / "recording/metadata.yaml").exists()
                except (OSError, KeyError, json.JSONDecodeError):
                    complete = False
            if not complete:
                state["state"] = "INTERRUPTED" if any((Path(b) / "feed_ended.json").exists() for b in state["bundles"]) else "FAILED"
        if state["state"] == "FAILED" and rc == 0:
            state["returncode"] = 1
        elif state["state"] == "INTERRUPTED":
            state["returncode"] = 130
        if state["state"] == "COMPLETED" and state["bundles"]:
            latest = Path(state["workspace"]) / "results/latest"
            temporary = latest.with_name("latest." + state["operation_id"])
            temporary.symlink_to(state["bundles"][-1])
            os.replace(temporary, latest)
        write_json(status, state)
        return state
    except BaseException as exc:
        state.update(state="FAILED", error=str(exc), returncode=1, finished=time.time())
        write_json(status, state)
        raise
    finally:
        if monitor_done:
            monitor_done.set()
        if process and process.poll() is None:
            os.killpg(process.pid, signal.SIGINT)
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
        # Container labels/names identify only this operation, never broad process patterns.
        if request["mode"] == "sim":
            stop_simulation(state, env.get("RTABMAP_CLOSE_TIMEOUT", "120"))
        elif private_owned and state.get("state") == "FAILED":
            stop_tiago(root, env, state)
        remaining = [name for name in owned_containers(state["operation_id"]) if name != state.get("actor")]
        private_running = False
        if private_owned:
            private_running = subprocess.run(["docker", "exec", "-u", "user", state["container"], "tmux", "has-session", "-t", state["session"]],
                                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode == 0
        if not remaining and not private_running:
            reg.release(state["operation_id"])
        for sig, handler in handlers.items():
            signal.signal(sig, handler)


def stop_tiago(root, env, state):
    command = ["bash", str(runtime(root, "run_tiago")), "stop", state["container"]]
    rc = subprocess.run(command, cwd=root, env=env).returncode
    if rc:
        # A busy pane can finish at the edge of the helper's shutdown wait.
        exists = subprocess.run(["docker", "exec", "-u", "user", state["container"],
                                 "tmux", "has-session", "-t", state["session"]],
                                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        if exists.returncode:
            return 0
        rc = subprocess.run(command, cwd=root, env=env).returncode
    return rc


def stop_operation(workspace, selector):
    reg = Registry(workspace)
    state = reg.select(selector)
    if state["state"] in ("PREPARING", "RUNNING", "DRAINING"):
        if state.get("actor") and container_alive(state["actor"]):
            if state["state"] == "DRAINING" and state.get("mode") == "sim":
                # Recover an older supervisor that only interrupted its Docker client.
                stop_simulation(state)
                return reg.select(state["operation_id"])
            if state.get("pid") is None:
                subprocess.run(["docker", "kill", "--signal=SIGINT", state["actor"]], check=True,
                               stdout=subprocess.DEVNULL)
                return state
            subprocess.run(["docker", "exec", state["actor"], "python3", "-c",
                            "import os,signal; os.kill(int(__import__('sys').argv[1]), signal.SIGINT)", str(state["pid"])], check=True)
            return state
        if alive(state["pid"], state.get("process_identity")):
            os.kill(state["pid"], signal.SIGINT)
            return state
    # Recover only containers carrying this exact operation label / private acknowledgement.
    names = owned_containers(state["operation_id"])
    if names:
        subprocess.run(["docker", "stop", "-t", "30", *names], check=True)
    if state.get("mode") in ("tiago", "bag"):
        identity = subprocess.run(["docker", "exec", state["container"], "cat", "/ws/output/operation_id"], capture_output=True, text=True)
        if identity.returncode == 0 and identity.stdout.strip() == state["operation_id"]:
            env, _ = environment(Path(state["project"]))
            env["FOUND_TMUX_SESSION"] = state["session"]
            if stop_tiago(Path(state["project"]), env, state):
                raise ValueError("private components are still closing; resource ownership retained")
    if not owned_containers(state["operation_id"]):
        reg.release(state["operation_id"])
    return state
