"""Finite registry of existing launch files and auxiliary commands."""
import json
from pathlib import Path
import subprocess

from .docker import command, ensure_image, ensure_gazebo_image


def launch_name(name):
    """Explicit TIAGO spellings for the existing ROS launch variants."""
    aliases = {"tiago-gazebo": "simulation", "tiago-navigation": "simulation2",
               "tiago-bag-slam": "bag-slam", "tiago-bag-slam-1": "bag-slam-1",
               "tiago-bag-slam-2": "bag-slam-2"}
    return "launch-" + aliases.get(name, name)


def catalogue(root):
    return json.loads((root / "config/entrypoints.json").read_text())


def run_tool(root, env, name, arguments, dry_run=False, _managed=False):
    import shlex
    entries = catalogue(root)
    if name not in entries:
        raise ValueError(f"unknown tool {name!r}; use graphapi tools list")
    entry = entries[name]
    if entry["kind"] == "internal":
        raise ValueError(entry["use"])
    if not dry_run and not (entry["path"] == "graphapi_cli/runtime/run_pipelines.sh" and arguments[:1] in (["--help"], ["-h"])):
        from .credentials import prepare_tool_credentials
        cfg = prepare_tool_credentials(root, entry, env)
        if _managed and cfg is not None:
            import yaml
            cfg["vlm"]["api_key"] = ""
            resolved = root / "config/.runtime" / (env["GRAPHAPI_OPERATION_ID"] + ".yaml")
            resolved.parent.mkdir(parents=True, exist_ok=True)
            resolved.write_text(yaml.safe_dump(cfg, sort_keys=False))
            resolved.chmod(0o600)
            env["GRAPH_API_CONFIG"] = str(resolved)
            env["GRAPHAPI_RESOLVED_CONFIG"] = "1"
    if not dry_run and not _managed:
        from .auxiliary import supervise
        return supervise(root, env, name, lambda managed: run_tool(root, managed, name, arguments, _managed=True),
                         gpu=entry["gpu"], need_domain=entry["environment"] not in ("host", "orchestrator"),
                         extra=("pal-container:" + env.get("GRAPHAPI_PAL_CONTAINER", env["TIAGO_DOCKER_TARGET"] + "-dev"),) if entry["environment"] == "pal" else ())
    path = root / entry["path"]
    if entry["kind"] == "internal":
        raise ValueError(entry["use"])
    if entry["kind"] == "launch" and _managed:
        env["GRAPHAPI_LAUNCH_SOURCE"] = str(path) if entry["environment"] != "pal" else "/graph_api/" + entry["path"]
        if entry["environment"] == "pal":
            env["GRAPHAPI_COMPONENT_EVENTS"] = "/ws/output/launch_events_" + env["GRAPHAPI_OPERATION_ID"] + ".jsonl"
            path = Path("/graph_api/graphapi_cli/ros_launch.py")
        else:
            env["GRAPHAPI_COMPONENT_EVENTS"] = str(Path(env["WORKSPACE_ROOT"]) / "results/operations" / env["GRAPHAPI_OPERATION_ID"] / "component_events.jsonl")
            path = root / "graphapi_cli/ros_launch.py"
    if entry["kind"] == "launch" and entry["environment"] == "pal":
        container = env.get("GRAPHAPI_PAL_CONTAINER", env["TIAGO_DOCKER_TARGET"] + "-dev")
        cmd = ["docker", "exec", "-u", "user", "-e", "ROS_DOMAIN_ID=" + env.get("ROS_DOMAIN_ID", "1"),
               container,
               "bash", "-lc", 'source /etc/profile.d/99-found.sh; ros2 launch "$@"; rc=$?; python3 -c '
               + shlex.quote("import json,os; from pathlib import Path; p=Path(os.environ.get('GRAPHAPI_COMPONENT_EVENTS','/nonexistent')); events=[json.loads(s) for s in p.read_text().splitlines()] if p.exists() else []; raise SystemExit(int(any(e.get('required') and not e.get('expected') and not e.get('during_shutdown') for e in events)))")
               + '; check=$?; if [ "$rc" != 0 ]; then exit "$rc"; fi; exit "$check"' , "graphapi",
               str(path) if _managed else "/graph_api/" + entry["path"], *arguments]
        if _managed:
            position = cmd.index(container)
            cmd[position:position] = [value for key in ("GRAPHAPI_LAUNCH_SOURCE", "GRAPHAPI_COMPONENT_EVENTS") for value in ("-e", key + "=" + env[key])]
    elif entry["kind"] == "launch":
        cmd = command(root, env, [], gpu=True)
        cmd[-1:] = ["ros2", "launch", str(path), *arguments]
    elif entry["environment"] == "host":
        cmd = ["bash", str(path), *arguments]
    elif entry["environment"] == "pal":
        container = env.get("GRAPHAPI_PAL_CONTAINER", env["TIAGO_DOCKER_TARGET"] + "-dev")
        executable = "bash" if path.suffix == ".sh" else "python3"
        cmd = ["docker", "exec", "-u", "user", container, "bash", "-lc",
               'source /etc/profile.d/99-found.sh; exec "$@"', "graphapi", executable,
               "/graph_api/" + entry["path"], *arguments]
    else:
        if path.suffix == ".sh":
            cmd = command(root, env, [str(path), *arguments], executable="bash",
                          control=entry["environment"] == "orchestrator", gpu=entry["gpu"])
        elif entry.get("module"):
            cmd = command(root, env, ["-m", entry["module"], *arguments],
                          habitat=entry["environment"] == "habitat",
                          control=entry["environment"] == "orchestrator", gpu=entry["gpu"])
        else:
            cmd = command(root, env, [str(path), *arguments],
                          habitat=entry["environment"] == "habitat",
                          control=entry["environment"] == "orchestrator", gpu=entry["gpu"])
    if entry["environment"] not in ("pal", "host"):
        if not dry_run:
            if name in ("launch-simulation", "launch-simulation2"):
                selected = ensure_gazebo_image(root, env)
                cmd[cmd.index(env["IMAGE_TAG"])] = selected
            else:
                ensure_image(root, env)
        if env.get("GRAPH_API_CONTAINER_NAME"):
            cmd[2:2] = ["--name", env["GRAPH_API_CONTAINER_NAME"]]
    if _managed:
        if entry["environment"] == "pal":
            from .docker import RUNTIME_ENV
            position = cmd.index(container)
            cmd[position:position] = [value for key in sorted(env)
                                     if key in RUNTIME_ENV or key.startswith("GRAPHAPI_")
                                     for value in ("-e", key)]
        return cmd
    if dry_run:
        print(shlex.join(cmd))
        return 0
    return subprocess.run(cmd, cwd=root, env=env).returncode
