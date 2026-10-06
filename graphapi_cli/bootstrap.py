"""Resolve host mounts inside Docker; output NUL-delimited arguments, never code."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

from .configuration import PATH_ENV, ConfigurationError, read_yaml
from .docker import RUNTIME_ENV
from .paths import host_path


def forwarded(key):
    return key in RUNTIME_ENV or key in {
        "HOME", "USER", "WAYLAND_DISPLAY", "DOCKER_CONFIG", "DOCKER_HOST", "DOCKER_CONTEXT", "SSH_AUTH_SOCK",
        "SKIP_TIAGO_HOST_DDS", "GIN_PROJECT_ROOT", "CLIO_IMAGE", "HOV_IMAGE",
        "CUDA_VISIBLE_DEVICES",
    } or key.startswith(("FEED_", "HABITAT_", "MULTI_FLOOR_", "GRAPHAPI_", "GRAPH_API_",
                         "DASH_", "BASELINE_", "EVAL_", "RTABMAP_", "PREFLIGHT_", "VITSAM_",
                         "PERCEPTION_", "EXT_", "TIAGO_", "FOUND_", "GT_", "MODAL_", "OLLAMA_",
                         "OPENAI_", "OPENROUTER_", "REGOLO_", "HF_", "CLIO_", "HOV_", "GIN_", "GA"))


def absolute(value, base, home):
    value = str(value)
    if value == "~" or value.startswith("~/"):
        value = home + value[1:]
    return Path(os.path.abspath(os.path.join(str(base), value)))


def options(arguments):
    """Inspect filesystem options without changing the public command parser."""
    result = {}
    names = {"project", "local", "config", "dataset", "workspace", "models", "cache",
             "pal-bundle", "baseline-repos", "transforms", "tour-schedule", "map"}
    boundary = arguments.index("--") if "--" in arguments else len(arguments)
    for i, value in enumerate(arguments[:boundary]):
        name, separator, inline = value.removeprefix("--").partition("=")
        if value.startswith("--") and name in names:
            if separator:
                result[name] = inline
            elif i + 1 < boundary:
                result[name] = arguments[i + 1]
    return result


def mount_plan(root, arguments, inherited, *, host_view=False, host_root=Path("/host")):
    env = dict(inherited)
    home = env.get("HOME", "/root")
    selected = options(arguments)
    root = absolute(selected.get("project", root), env.get("GRAPHAPI_HOST_CWD", Path.cwd()), home)

    def visible(path):
        resolved = host_path(path, host_root if host_view else None)
        return Path(host_root) / str(resolved).lstrip("/") if host_view else resolved

    local = absolute(selected.get("local", "config/local.yaml"), root, home)
    settings = {} if any(value in ("--help", "-h") for value in arguments) else read_yaml(visible(local), optional=True)
    if any(not isinstance(settings.get(section, {}), dict) for section in ("paths", "environment")):
        raise ConfigurationError(f"{local}: paths and environment must be mappings")
    for key, value in settings.get("environment", {}).items():
        if not isinstance(value, (dict, list)):
            env.setdefault(key, str(value))
    env["HOME"] = home
    env["GRAPHAPI_ROOT"] = str(root)
    env["GRAPHAPI_CLI_CONTAINER"] = "1"
    env["PYTHONPATH"] = str(root) + ":/opt/graphapi-cli"
    env.setdefault("XDG_CACHE_HOME", home + "/.cache")
    mounts = {root: True}

    def add(path, write=False, required=False, exact=False):
        path = absolute(path, root, home)
        if not exact and visible(path).is_file():
            # File bind mounts pin an inode and prevent atomic replacement.
            # Mount the parent so editable configuration and status keep working.
            path = path.parent
            if not write and any(enabled and path.is_relative_to(existing) for existing, enabled in mounts.items()):
                return
        if not visible(path).exists():
            if not write:
                if required:
                    add(path.parent, False)
                return
            path = host_path(path, host_root if host_view else None)
            while not visible(path).exists() and path != path.parent:
                path = path.parent
            if path == Path("/"):
                raise ConfigurationError("create a parent directory for the selected output path on the Docker host")
        # Keep narrower read-only inputs even inside a writable workspace mount.
        resolved = host_path(path, host_root if host_view else None)
        mounts[resolved] = mounts.get(resolved, False) or write
        # An ancestor mount already preserves this alias and its symlink. For
        # external aliases, retain an identity mount as well as their target.
        if resolved == path or not any(path.is_relative_to(existing) for existing in mounts if existing != resolved):
            mounts[path] = mounts.get(path, False) or write

    add(local.parent, True)
    defaults = {"workspace": root, "models": root / "models/efficientvit_sam",
                "cache": root / "models/huggingface", "bags": root / "bags",
                "pal_bundle": root / "TIAGO_ISO"}
    for key, env_key in PATH_ENV.items():
        value = env.get(env_key)
        if value:
            path = absolute(value, root, home)
        else:
            value = settings.get("paths", {}).get(key, defaults.get(key))
            if value is None:
                continue
            path = absolute(value, local.parent, home)
        flag = key.replace("_", "-")
        if flag in selected:
            path = absolute(selected[flag], root, home)
        # Baseline setup installs compatibility entrypoints in supplied checkouts.
        add(path, key in {"workspace", "models", "cache", "pal_bundle", "baseline_repos"})
        if key == "workspace":
            for name in ("results", "maps", "schedules", "ws"):
                add(path / name, True)
    add(Path(env["XDG_CACHE_HOME"]) / "graphapi", True)
    for key in ("config", "transforms", "tour-schedule", "map"):
        if key in selected:
            add(selected[key], False, required=True)
    for key in ("FEED_SCHEDULE", "MULTI_FLOOR_TRANSFORMS", "RTABMAP_LOCALIZE_DB",
                "EVAL_SCENE", "EVAL_DATASET_CONFIG", "RUN_DIR", "TIAGO_BAG_PATH", "MP3D_ROOT", "XAUTHORITY"):
        if env.get(key):
            add(env[key], False, required=True)
    # Positional paths and forwarded ROS arguments can name external inputs.
    for value in arguments:
        candidate = value.split(":=", 1)[-1].split("=", 1)[-1]
        if candidate.startswith(("/", "~/")) or "/" in candidate and not value.startswith("--"):
            add(candidate, False)
    # Native outputs and controller output flags may not exist yet.
    remaining = public_arguments(arguments)
    if remaining[:2] == ["baseline", "run"] and len(remaining) > 4:
        add(remaining[4], True)
    for i, value in enumerate(arguments[:-1]):
        if value in ("--output", "--out-dir", "--results-root", "--output-root"):
            add(arguments[i + 1], True)
    for path in (Path(home) / ".ssh", Path(env.get("DOCKER_CONFIG", home + "/.docker"))):
        add(path)
    for path in (Path("/etc/passwd"), Path("/etc/group"), Path(home) / ".modal.toml"):
        add(path, exact=True)
    if env.get("SSH_AUTH_SOCK"):
        add(env["SSH_AUTH_SOCK"], exact=True)
    socket = env.get("GRAPHAPI_DOCKER_SOCKET", "/var/run/docker.sock")
    add(socket)
    env["DOCKER_HOST"] = "unix://" + socket
    env.pop("DOCKER_CONTEXT", None)  # The host context has resolved to this socket.
    args = ["--network=host", "-w", str(root)]
    for path, write in sorted(mounts.items(), key=lambda item: len(item[0].parts)):
        args += ["-v", f"{path}:{path}" + ("" if write else ":ro")]
    for key in sorted(env):
        if forwarded(key) or key == "PYTHONPATH" or key in settings.get("environment", {}):
            args += ["-e", key + "=" + env[key]]
    return root, env, args


def public_arguments(arguments):
    remaining = list(arguments)
    while remaining and remaining[0].startswith("--"):
        value = remaining.pop(0)
        if value in ("--project", "--local", "--config") and remaining:
            remaining.pop(0)
    return remaining


def host_action(root, arguments, env):
    """Retain host DDS effects rather than modifying a container firewall."""
    if "--help" in arguments or "-h" in arguments:
        return "none", [], []
    remaining = public_arguments(arguments)
    action = None
    instead = False
    if remaining[:2] == ["tiago", "network"]:
        action = remaining[2:] or ["apply"]
        instead = True
    elif remaining[:2] == ["tools", "run"] and remaining[2:3] == ["tiago-host-dds"]:
        action = remaining[3:]
        instead = True
    elif remaining[:3] == ["run", "tiago", "physical"] or remaining[:2] in (["tiago", "start"], ["tiago", "new"]):
        action = ["apply"]
    explicit_physical = remaining[:3] == ["run", "tiago", "physical"]
    if action is None or not instead and (env.get("SKIP_TIAGO_HOST_DDS") == "1" or
                                          env.get("TIAGO_BAG_PATH") and not explicit_physical):
        return "none", [], []
    if "--dry-run" in action:
        return "none", [], []
    action = action[1:] if action[:1] == ["--"] else action
    helper = env.get("TIAGO_HOST_DDS", str(root / "tiago/tiago-host-dds.sh"))
    host_env = [key + "=" + env[key] for key in ("TIAGO_ROBOT_IP", "TIAGO_ROBOT_WLAN_IP", "TIAGO_NET_IFACE") if env.get(key)]
    if not instead:
        env["SKIP_TIAGO_HOST_DDS"] = "1"
    return "instead" if instead else "before", host_env, [helper, *action]


def controller_command(root, env, arguments, *, name, operation, executable="python3"):
    """Detached CLI supervisors use the small image and the same identity mounts."""
    _, resolved, args = mount_plan(root, arguments, env)
    socket = resolved.get("GRAPHAPI_DOCKER_SOCKET", "/var/run/docker.sock")
    return ["docker", "run", "--rm", "--init", "--sig-proxy=false", "--name", name,
            "--label", "graphapi.operation=" + operation, "--user", f"{os.getuid()}:{os.getgid()}",
            "--group-add", str(Path(socket).stat().st_gid), *args,
            "-e", "GRAPHAPI_RESOURCE_ACTOR=" + name, "--entrypoint", executable,
            env.get("GRAPHAPI_CLI_IMAGE", "graphapi-cli:latest"), *arguments]


def write_arguments(path, values):
    path.write_bytes(b"".join(str(value).encode() + b"\0" for value in values))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("arguments", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    arguments = args.arguments[1:] if args.arguments[:1] == ["--"] else args.arguments
    try:
        if not any(value in ("--help", "-h") for value in arguments):
            from .cli import parser as cli_parser
            cli_parser().parse_args(arguments)
        root, env, docker_args = mount_plan(Path(args.root), arguments, os.environ, host_view=True)
        mode, host_env, host_args = host_action(root, arguments, env)
        if mode == "before":
            docker_args += ["-e", "SKIP_TIAGO_HOST_DDS=1"]
        write_arguments(args.output / "docker.args", docker_args)
        write_arguments(args.output / "host.env", host_env)
        write_arguments(args.output / "host.args", host_args)
        (args.output / "host.mode").write_text(mode)
        return 0
    except (ConfigurationError, OSError, ValueError) as exc:
        print(f"graphapi: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
