"""Public GRAPH-API command line. Help does not import ROS, Habitat or CUDA."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

from .configuration import ConfigurationError, environment, initialize, path_from, project_root, read_yaml, resolve_config, workflow_name, tiago_map_source, node_profile
from .help_text import configure_help, selected_help


def parser():
    ap = argparse.ArgumentParser(prog="graphapi", description="Setup once; launch GRAPH-API workflows through one CLI.")
    ap.add_argument("--project", metavar="DIRECTORY", help="repository folder; normally found automatically")
    ap.add_argument("--local", metavar="FILE", help="local paths and Docker settings (default: config/local.yaml)")
    ap.add_argument("--config", metavar="FILE", help="pipeline settings; otherwise use the default for the chosen run")
    ap.add_argument("--json", action="store_true", help="print results as JSON; send progress messages to stderr")
    commands = ap.add_subparsers(dest="command", required=True)
    init = commands.add_parser("init", help="create config/local.yaml without overwriting it")
    init.add_argument("--workspace", metavar="DIRECTORY", help="folder for results, maps, schedules and build files")
    setup = commands.add_parser("setup", help="prepare existing workflows; public dependencies may be installed")
    setup.add_argument("mode", choices=("sim", "gazebo", "tiago", "baselines", "cloud", "all"), nargs="?", default="sim",
                       help="what to prepare (default: sim); all prepares every listed mode")
    setup_options = {
        "dataset": "HM3D folder containing scene folders and the scene dataset config JSON",
        "models": "folder for the VitSAM encoder and decoder model files",
        "cache": "folder for downloaded Hugging Face models",
        "workspace": "folder for results, maps, schedules and build files",
        "pal-bundle": "private PAL build folder; see the TIAGO section in README.md",
        "container": "name of your private PAL container (for example: tiago-127-dev)",
        "baseline-repos": "folder containing the comparison-method repositories",
    }
    for flag, help_text in setup_options.items():
        setup.add_argument("--" + flag, help=help_text)
    setup.add_argument("--rebuild", action="store_true", help="rebuild public Docker images even if they already exist")
    run = commands.add_parser("run", help="fresh simulation, physical robot or bag operation")
    modes = run.add_subparsers(dest="mode", required=True)
    simulation = modes.add_parser("sim", help="Habitat scene simulation")
    tiago_run = modes.add_parser("tiago", help="TIAGO workflows using private PAL Docker")
    tiago_modes = tiago_run.add_subparsers(dest="tiago_workflow", required=True)
    physical = tiago_modes.add_parser("physical", help="connect to the physical TIAGO robot",
                                     description="Physical TIAGO in private PAL Docker. Starts application RTAB-Map by default.")
    physical.set_defaults(mode="tiago")
    bag = tiago_modes.add_parser("bag", help="replay a recorded TIAGO RGB-D bag",
                                description="Offline TIAGO bag replay in private PAL Docker. Uses recorded map/TF by default; no robot connection.")
    bag.set_defaults(mode="bag")
    old_bag = modes.add_parser("bag", help="TIAGO bag compatibility alias; use run tiago bag")
    old_bag.set_defaults(tiago_workflow="bag")
    for name, sub in (("sim", simulation), ("tiago", physical), ("bag", bag), ("bag", old_bag)):
        if name == "bag":
            sub.add_argument("bag", metavar="BAG_DIRECTORY", help="TIAGO bag folder containing metadata.yaml")
        sub.add_argument("--profile", metavar="NAME", help="use a named settings file from config/; sim-gt is already the default" if name == "sim" else "use a named settings file from config/")
        sub.add_argument("--config", dest="run_config", metavar="FILE", help="use this YAML settings file for the run")
        sub.add_argument("--gpu", metavar="INDEX", help="GPU number to use, for example 0")
        sub.add_argument("--gui", action="store_true", help="open visualization windows on this computer")
        sub.add_argument("--no-record", action="store_true", help="do not save another ROS topic recording; keep normal results and logs")
        sub.add_argument("--no-perception", action="store_true", help="run mapping without object detection; no annotations will be produced" if name == "sim" else "disable object detection and the object manager; no annotations will be produced")
        sub.add_argument("--detach", action="store_true", help="run in the background and print an ID for status, logs and stop")
        if name == "sim":
            sub.add_argument("--scene", metavar="NAME", help="scene to load, for example hm3d_00861")
            sub.add_argument("--one-storey", action="store_true", help="run one floor instead of separate runs for each saved floor")
            floors = sub.add_mutually_exclusive_group()
            floors.add_argument("--floor", metavar="HEIGHT", help="run one floor at this height in metres; for negative values use --floor=-1.59")
            floors.add_argument("--floors", metavar="HEIGHTS", help='run these floor heights, for example --floors="-1.59 1.21"')
            floors.add_argument("--visits", metavar="HEIGHTS", help='visit floors in this order, including repeats; for example --visits="0 2.8 0"; needs --transforms')
            sub.add_argument("--transforms", metavar="FILE", help="JSON file describing how each floor map fits into the building")
            sub.add_argument("--tour-schedule", metavar="FILE", help="JSON movement schedule for the scene or floor visits")
            sub.add_argument("--pose", choices=("simulator", "rtabmap"), help="where the camera position comes from (default: simulator)")
            sub.add_argument("--map", metavar="FILE", help="use an existing RTAB-Map database; work on a copy, keeping the original")
            sub.add_argument("--mapping-only", action="store_true", help="build a map without object detection")
        else:
            sub.add_argument("--container", metavar="NAME", help="private PAL container name; otherwise use local settings")
            sub.add_argument("--resume", action="store_true", help="reuse the current TIAGO session and output; does not restart an old run ID")
            sub.add_argument("--map-source", choices=("slam", "robot") if name == "tiago" else ("slam", "recorded"),
                             help="slam: build a map with RTAB-Map (default); robot: use the robot's map and position" if name == "tiago" else
                                  "recorded: use the bag's map and transforms (default); slam: build a new map with RTAB-Map")
        if name == "bag":
            sub.add_argument("--rate", type=float, default=1.0, help="playback speed: 1 is normal, 0.5 is half speed (default: 1)")
            sub.add_argument("--loop", action="store_true", help="restart the bag when it ends; keep playing until you stop the run")
    doctor = commands.add_parser("doctor", help="check software/configuration; no hardware requirements checks")
    doctor.add_argument("--mode", choices=("sim", "tiago-physical", "tiago-bag", "tiago", "bag"), default="sim",
                        help="what to check (default: sim); use tiago-physical for a robot or tiago-bag for a recording")
    doctor.add_argument("--live", action="store_true", help="also check Python and model loading in the private TIAGO container")
    batch = commands.add_parser("batch", help="run an existing experiment schedule with exact child identities")
    batch.add_argument("schedule", metavar="FILE", help="YAML file listing the experiments to run")
    batch.add_argument("--continue-on-failure", action="store_true", help="run the next experiment even if one fails")
    batch.add_argument("--force", action="store_true", help="run completed experiments again instead of skipping them")
    batch.add_argument("--dry-run", action="store_true", help="show the planned experiments without starting them")
    batch.add_argument("--gpus", default="0", help="GPU numbers to assign in turn, for example 0,1 (default: 0)")
    for name in ("status", "logs", "stop", "attach"):
        sub = commands.add_parser(name, help={
            "status": "Show active runs and dashboards.",
            "logs": "Read a run's startup log.",
            "stop": "Stop a run or dashboard.",
            "attach": "Watch a run log.",
        }[name])
        sub.add_argument("operation", metavar="RUN_ID", nargs="?", default=None if name == "status" else "active",
                         help="ID from status; omit to list active runs" if name == "status" else
                              "ID from status, or active if exactly one run or dashboard is active (default: active)")
        if name == "status":
            sub.description = "Show active operations; use --all for operation history."
            sub.add_argument("--all", action="store_true", help="include completed, failed and interrupted operations")
        if name in ("status", "stop"):
            sub.add_argument("--json", action="store_true", default=argparse.SUPPRESS, help="print results as JSON")
        if name in ("logs", "attach"):
            sub.add_argument("--follow", action="store_true", help="keep watching new log lines; attach already does this")
    dashboard = commands.add_parser("dashboard", help="existing combined live/replay dashboard")
    dashboard.add_argument("bundle", metavar="RUN_ID_OR_FOLDER", nargs="?", default="latest", help="saved run ID or results folder (default: latest completed run)")
    dashboard.add_argument("--port", type=int, default=8082, help="web address port (default: 8082)")
    dashboard.add_argument("--host", default="127.0.0.1", help="address to listen on (default: 127.0.0.1, this computer only)")
    dashboard.add_argument("--mode", choices=("auto", "live", "replay"), default="auto", help="live: running pipeline; replay: saved results; auto: choose at startup (default: auto)")
    viewer = commands.add_parser("view", help="open RViz on an active operation's DDS domain")
    viewer.add_argument("operation", metavar="RUN_ID", nargs="?", default="active", help="run ID; omit when only one pipeline is running")
    evaluation = commands.add_parser("eval", help="existing evaluation pipeline, entirely in Docker")
    evaluation.add_argument("bundle", metavar="RUN_ID_OR_FOLDER", nargs="?", default="latest", help="run ID or results folder (default: latest completed run)")
    evaluation.add_argument("--force", action="store_true", help="rebuild ground-truth data even if it already exists")
    tools = commands.add_parser("tools", help="all existing auxiliary entrypoints, registered by environment")
    tool_sub = tools.add_subparsers(dest="action", required=True)
    listing = tool_sub.add_parser("list")
    listing.add_argument("--all", action="store_true", help="include tests and tools used internally by the pipeline")
    tool_run = tool_sub.add_parser("run")
    tool_run.add_argument("--dry-run", action="store_true", help="print the command without starting the tool")
    tool_run.add_argument("tool", metavar="NAME", help="tool name from ./graphapi tools list")
    tool_run.add_argument("arguments", metavar="TOOL_OPTIONS", nargs=argparse.REMAINDER, help="options for the tool; put -- before them")
    launches = commands.add_parser("launch", help="existing TIAGO Gazebo/navigation/bag ROS launch variants")
    launches.add_argument("name", help="tiago-gazebo, tiago-navigation or tiago-bag-slam; older launch names also work")
    launches.add_argument("arguments", metavar="LAUNCH_OPTIONS", nargs=argparse.REMAINDER, help="ROS arguments after --, written as NAME:=VALUE")
    baseline = commands.add_parser("baseline", help="native baselines and acquisition workflows")
    base_sub = baseline.add_subparsers(dest="action", required=True)
    native = base_sub.add_parser("run")
    native.add_argument("name", choices=("clio", "hovsg", "dynamicgsg"), help="comparison method to run")
    native.add_argument("recording", metavar="RECORDING", help="input recording folder")
    native.add_argument("output", metavar="OUTPUT", help="folder for the comparison results")
    native.add_argument("arguments", metavar="METHOD_OPTIONS", nargs=argparse.REMAINDER, help="extra options after --")
    for name in ("acquire", "pair", "remote"):
        sub = base_sub.add_parser(name)
        sub.add_argument("arguments", metavar="TOOL_OPTIONS", nargs=argparse.REMAINDER, help="options for the existing tool; put -- before them")
    cloud = commands.add_parser("cloud", help="existing Modal deployment/run interface")
    cloud.add_argument("action", choices=("deploy", "run", "serve"), help="Modal action to perform")
    cloud.add_argument("arguments", metavar="OPTIONS", nargs=argparse.REMAINDER, help="extra Modal options after --")
    tiago = commands.add_parser("tiago", help="private Docker lifecycle and physical network compatibility")
    tiago.add_argument("action", choices=("network", "check", "build", "shell", "start", "new", "stop", "attach"), help="container action, or network to check/apply robot network settings")
    tiago.add_argument("arguments", metavar="OPTIONS", nargs=argparse.REMAINDER, help="container name or action options; use network status to inspect the network")
    maps = commands.add_parser("maps", help="inspect the existing per-scene/per-floor map library")
    maps.add_argument("action", choices=("list",), help="list saved map databases")
    job = commands.add_parser("_job", help=argparse.SUPPRESS)
    job.add_argument("payload")
    commands._choices_actions[:] = [a for a in commands._choices_actions if a.dest != "_job"]
    commands.metavar = "{" + ",".join(a.dest for a in commands._choices_actions) + "}"
    configure_help(ap)
    return ap


def _without_separator(arguments):
    return arguments[1:] if arguments[:1] == ["--"] else arguments


def _emit(value, structured):
    def explicit_workflow(row):
        if isinstance(row, dict) and row.get("operation_id") and workflow_name(row.get("mode")) != row.get("mode"):
            return dict(row, mode=workflow_name(row["mode"]), workflow=workflow_name(row["mode"]))
        return row
    value = [explicit_workflow(row) for row in value] if isinstance(value, list) else explicit_workflow(value)
    if structured:
        print(json.dumps(value, indent=2))
    elif isinstance(value, dict):
        print(json.dumps(value, indent=2))
    else:
        print(value)


def _show_status(value, *, history=False):
    """Keep IDs copyable and put lifecycle state ahead of diagnostic metadata."""
    rows = [value] if isinstance(value, dict) else value
    if not rows:
        print("No recorded operations." if history else
              "No active operations. Use './graphapi status --all' to view history.")
        return
    columns = [("OPERATION", "MODE", "STATE", "INPUT", "ENDPOINT")]
    for row in rows:
        dashboard = row.get("ports", {}).get("dashboard")
        endpoint = row.get("bridge_url") or (f"http://127.0.0.1:{dashboard}" if dashboard else "—")
        columns.append(tuple(str(part).replace("\n", " ") for part in
                             (row["operation_id"], row.get("workflow") or workflow_name(row.get("mode", "—")), row["state"],
                              row.get("scene") or (Path(row["bag"]).name if row.get("bag") else row.get("robot_address")) or "—", endpoint)))
    widths = [max(len(row[index]) for row in columns) for index in range(len(columns[0]))]
    for row in columns:
        print("  ".join(part.ljust(width) for part, width in zip(row, widths)).rstrip())
    if isinstance(value, dict):
        from datetime import datetime, timezone
        for key, label in (("created", "Started"), ("finished", "Finished")):
            if value.get(key) is not None:
                stamp = datetime.fromtimestamp(value[key], timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
                print(f"{label}: {stamp}")
        for key, label in (("bag", "TIAGO bag"), ("robot_address", "TIAGO robot address"),
                           ("map_source", "Map source"), ("map_frame", "Map frame"), ("map_topic", "Map topic"),
                           ("runtime", "Runtime"), ("container", "Container"),
                           ("perception", "Perception enabled"), ("recording", "Recording enabled"),
                           ("gpu", "GPU"), ("domain", "ROS domain"), ("returncode", "Exit code"),
                           ("error", "Error"), ("log", "Log"), ("config", "Config")):
            if value.get(key) is not None:
                print(f"{label}: {value[key]}")
        for bundle in value.get("bundles", []):
            print(f"Bundle: {bundle}")


def _bundles(root, env, selector):
    from .registry import Registry
    candidate = path_from(selector, root)
    if candidate.is_dir() and (candidate / "run_metadata.json").exists():
        return [candidate]
    rows = Registry(env["WORKSPACE_ROOT"])
    # Preserve explicit legacy folder names without newest-directory attribution.
    legacy = Path(env["WORKSPACE_ROOT"]) / "results" / selector
    if selector not in ("latest", "latest-started", "active") and legacy.is_dir():
        return [legacy]
    row = rows.select(selector)
    if not row.get("bundles"):
        result = Path(row["log"]).parent / "bundles.txt"
        row["bundles"] = result.read_text().splitlines() if result.exists() else []
    if not row["bundles"]:
        raise ConfigurationError(f"{selector}: no recorded bundle yet")
    return [Path(p) for p in row["bundles"]]


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        root = project_root(args.project)
        env, _ = environment(root, args.local)
        command = args.command
        remainder = getattr(args, "arguments", [])
        boundary = remainder.index("--") if "--" in remainder else len(remainder)
        if any(flag in remainder[:boundary] for flag in ("--help", "-h")):
            # Help without an explicit payload separator never launches a workflow.
            if command == "launch":
                from .catalogue import catalogue, launch_name
                entry = catalogue(root).get(launch_name(args.name))
                if entry is None:
                    raise ConfigurationError(f"unknown launch: {args.name}")
                import ast
                tree = ast.parse((root / entry["path"]).read_text())
                runtime_name = 'private PAL Docker' if entry['environment'] == 'pal' else 'Docker'
                print(f"Run the {args.name} ROS launch file.\n"
                      f"Usage: ./graphapi launch {args.name} -- NAME:=VALUE ...\n"
                      f"Environment: {entry['environment']} ({runtime_name})\n\n"
                      "Put launch options after --. Write each option as NAME:=VALUE.\n\n"
                      "Launch options:")
                option_names = []
                explanations = {'bag_path': 'folder containing the recorded TIAGO bag',
                                'map_yaml': 'YAML file for a saved map',
                                'rviz_config': 'RViz settings file',
                                'slam_params': 'SLAM settings file'}
                for node in ast.walk(tree):
                    if isinstance(node, ast.Call) and getattr(node.func, "id", None) == "DeclareLaunchArgument" and node.args and isinstance(node.args[0], ast.Constant):
                        name = str(node.args[0].value)
                        option_names.append(name)
                        detail = explanations.get(name)
                        if not detail:
                            detail = next((str(k.value.value) for k in node.keywords
                                           if k.arg == 'description' and isinstance(k.value, ast.Constant)), '')
                        print("  " + name + (": " + detail if detail else ''))
                example = ('bag_path:=/bags/example' if 'bag_path' in option_names else
                           'map_yaml:=/data/maps/office.yaml' if 'map_yaml' in option_names else '')
                print(f"\nExamples:\n  Run with the default launch settings:\n"
                      f"    ./graphapi launch {args.name}")
                if example:
                    print(f"\n  Choose your input path:\n    ./graphapi launch {args.name} -- {example}")
            else:
                print(selected_help(parser(), args))
            return 0
        if command == "_job":
            from .launch import execute
            payload = json.loads(Path(args.payload).read_text())
            row = execute(root, payload["request"], dict(os.environ), payload["state"])
            return row["returncode"]
        if command == "init":
            target = initialize(root, args.local)
            if args.workspace:
                import yaml
                settings = read_yaml(target)
                settings["paths"]["workspace"] = str(path_from(args.workspace, root))
                target.write_text(yaml.safe_dump(settings, sort_keys=False))
            _emit({"local_config": str(target)}, args.json)
            return 0
        if command == "setup":
            from .setup import setup
            setup(root, env, args)
            return 0
        if command == "run":
            from .launch import start
            request = vars(args).copy()
            request["config"] = args.run_config or args.config
            for name in ("bag", "transforms", "tour_schedule", "map"):
                if request.get(name):
                    request[name] = str(path_from(request[name], root))
            if request.get("visits") and not request.get("transforms"):
                raise ConfigurationError("--visits requires --transforms and a compatible tour schedule")
            if request.get("rate", 1) <= 0:
                raise ConfigurationError("--rate must be positive")
            if getattr(args, "container", None):
                env["GRAPHAPI_PAL_CONTAINER"] = args.container
            if getattr(args, "resume", False):
                env.setdefault("FOUND_TMUX_SESSION", "found_tiago")
                request["legacy_args"] = ["--resume"]
            if args.json and not args.detach:
                # Workload progress cannot corrupt structured stdout.
                import contextlib
                with contextlib.redirect_stdout(sys.stderr):
                    row = start(root, request, inherited=env)
            else:
                row = start(root, request, detach=args.detach, inherited=env)
            _emit(row, args.json)
            return 0 if args.detach else row["returncode"]
        if command == "doctor":
            mode = {"tiago-physical": "tiago", "tiago-bag": "bag"}.get(args.mode, args.mode)
            map_source = tiago_map_source(mode, None, env)
            source, cfg = resolve_config(root, args.config or node_profile(mode, map_source), args.local)
            software = subprocess.run(["docker", "info"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode == 0
            checks = {"docker_available": software, "config_valid": True, "workflow": workflow_name(mode), "config": str(source)}
            if mode == "sim":
                checks["default_pose"] = cfg["habitat"]["localization_mode"]
                checks["dataset_registered"] = bool(env.get("HM3D_ROOT") and Path(env["HM3D_ROOT"]).is_dir())
                checks["image_available"] = subprocess.run(["docker", "image", "inspect", env["IMAGE_TAG"]], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode == 0
            else:
                container = env.get("GRAPHAPI_PAL_CONTAINER", env["TIAGO_DOCKER_TARGET"] + "-dev")
                checks.update(runtime="private-pal", container=container, map_source=map_source,
                              map_frame=cfg["tf"]["world_frame"], map_topic=cfg["bev"]["map_topic"])
                checks["private_container_available"] = subprocess.run(["docker", "inspect", container], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode == 0
                if args.live:
                    from .launch import runtime
                    checks["existing_startup_check"] = subprocess.run(["bash", str(runtime(root, "run_tiago")), "check", container], env=env).returncode == 0
            _emit(checks, args.json)
            return 0 if all(v for v in checks.values() if isinstance(v, bool)) else 1
        if command == "batch":
            from .batch import run_batch
            return run_batch(root, env, args)
        if command in ("status", "stop", "logs", "attach"):
            from .registry import Registry
            registry = Registry(env["WORKSPACE_ROOT"])
            if command == "status":
                if args.operation:
                    result = registry.select(args.operation)
                else:
                    result = [row for row in registry.rows() if args.all or row["state"] in ("PREPARING", "RUNNING", "DRAINING")]
                if args.json:
                    _emit(result, True)
                else:
                    _show_status(result, history=args.all)
                return 0
            if command == "stop":
                from .launch import stop_operation
                result = stop_operation(env["WORKSPACE_ROOT"], args.operation)
                if args.json:
                    _emit(result, True)
                elif result["state"] in ("PREPARING", "RUNNING", "DRAINING"):
                    print(f"Stop requested for {result['operation_id']} ({workflow_name(result['mode'])}).")
                    print(f"Check shutdown with: ./graphapi status {result['operation_id']}")
                else:
                    print(f"Already stopped: {result['operation_id']} ({workflow_name(result['mode'])}, {result['state']}).")
                return 0
            row = registry.select(args.operation)
            with open(row["log"]) as stream:
                while True:
                    line = stream.readline()
                    if line:
                        print(line, end="", flush=True)
                    elif (args.follow or command == "attach") and registry.select(row["operation_id"])["state"] in ("PREPARING", "RUNNING", "DRAINING"):
                        import time
                        time.sleep(0.2)
                    else:
                        break
            return 0
        if command in ("tools", "launch", "baseline"):
            from .catalogue import catalogue, run_tool
            if args.config:
                env["GRAPH_API_CONFIG"] = str(path_from(args.config, root))
            if command == "tools" and args.action == "list":
                entries = catalogue(root)
                entries = {k: v for k, v in entries.items() if args.all or
                           (v.get("visibility") == "public" and v.get("kind") != "internal")}
                _emit(entries if args.json else "\n".join(f"{k:45s} {v['environment']:13s} {v['path']}" for k, v in entries.items()), args.json)
                return 0
            if command == "launch":
                from .catalogue import launch_name
                name = launch_name(args.name)
                values = args.arguments
            elif command == "baseline":
                name = {"run": "baseline-native", "acquire": "baseline-entrypoint", "pair": "baseline-run-pair", "remote": "baseline-night-remote"}[args.action]
                values = [args.name, str(path_from(args.recording, root)), str(path_from(args.output, root)), *_without_separator(args.arguments)] if args.action == "run" else args.arguments
            else:
                name, values = args.tool, args.arguments
            if command == "tools":
                boundary = values.index("--") if "--" in values else len(values)
                if "--dry-run" in values[:boundary]:
                    values.remove("--dry-run")
                    args.dry_run = True
            return run_tool(root, env, name, _without_separator(values), getattr(args, "dry_run", False))
        if command == "cloud":
            from .docker import command as docker_command, ensure_image
            ensure_image(root, env)
            cmd = docker_command(root, env, ["-m", "modal", args.action, "/graph_api/lost3dsg/src/perception_module/cloud/modal_perception.py", *_without_separator(args.arguments)])
            return subprocess.run(cmd, env=env).returncode
        if command == "tiago":
            from .launch import runtime
            action = "firewall" if args.action == "network" else args.action
            values = _without_separator(args.arguments)
            if args.action in ("start", "new"):
                from .launch import start
                if values and not values[0].startswith("-"):
                    env["GRAPHAPI_PAL_CONTAINER"] = values.pop(0)
                resume = args.action == "start"
                mode = "bag" if env.get("TIAGO_BAG_PATH") else "tiago"
                row = start(root, {"mode": mode, "bag": env.get("TIAGO_BAG_PATH"), "local": args.local,
                                   "config": args.config, "resume": resume,
                                   "legacy_args": (["--resume"] if resume else []) + values}, inherited=env)
                _emit(row, args.json)
                return row["returncode"]
            if args.action == "network" and values[:1] == ["status"]:
                return subprocess.run(["bash", str(root / "tiago/tiago-host-dds.sh"), "status"], env=env).returncode
            if args.action == "network":
                values = values[1:] if values[:1] == ["apply"] else values
            return subprocess.run(["bash", str(runtime(root, "run_tiago")), action, *values], cwd=root, env=env).returncode
        if command == "view":
            from .viewer import acquisition, start
            result = start(root, env, acquisition(env["WORKSPACE_ROOT"], args.operation))
            return result if isinstance(result, int) else 0
        if command == "maps":
            _emit([str(p) for p in sorted((Path(env["WORKSPACE_ROOT"]) / "maps").rglob("rtabmap.db"))], args.json)
            return 0
        if command in ("eval", "dashboard"):
            from .docker import command as docker_command, ensure_image
            ensure_image(root, env)
            try:
                bundles = _bundles(root, env, args.bundle)
            except (ConfigurationError, ValueError):
                if command != "dashboard" or args.bundle != "latest":
                    raise
                bundles = [Path(env["WORKSPACE_ROOT"]) / "results"]
            if command == "dashboard":
                if len(bundles) != 1:
                    raise ConfigurationError("dashboard requires one bundle; pass a session bundle directory")
                env["GRAPH_API_RUNS_DIR"] = str(Path(env["WORKSPACE_ROOT"]) / "results")
                cmd = docker_command(root, env, ["/graph_api/lost3dsg/dashboard/replay_server.py", "--bundle", "latest" if args.bundle == "latest" else str(bundles[0]), "--port", str(args.port), "--host", args.host, "--mode", args.mode], control=True)
                from .auxiliary import supervise
                return supervise(root, env, "dashboard", lambda managed: docker_command(root, managed, ["/graph_api/lost3dsg/dashboard/replay_server.py", "--bundle", "latest" if args.bundle == "latest" else str(bundles[0]), "--port", str(args.port), "--host", args.host, "--mode", args.mode], control=True, name=managed["GRAPH_API_CONTAINER_NAME"]), ports={"dashboard": args.port}, need_domain=False)
            env.update(EVAL_CONDA_PY="/opt/habitat/bin/python", EVAL_PY="python3", EVAL_REPORT_PY="python3")
            from .launch import runtime
            for bundle in bundles:
                cmd = docker_command(root, env, [str(runtime(root, "eval")), str(bundle), *(["--force"] if args.force else [])], executable="bash", gpu=True)
                result = subprocess.run(cmd, env=env).returncode
                if result:
                    return result
            return 0
        raise ConfigurationError(f"unhandled command: {command}")
    except (ConfigurationError, ValueError, OSError, subprocess.CalledProcessError) as exc:
        if args.json:
            print(json.dumps({"error": str(exc)}))
        else:
            print(f"graphapi: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
