"""Docker command construction shared by setup, Habitat, tools and evaluation."""
from __future__ import annotations

import os
import hashlib
from pathlib import Path
import shlex
import subprocess
import sys

from graphapi_cli.paths import output_directory


RUNTIME_ENV = {
    "OUT_DIR", "WORKSPACE_ROOT", "RESULTS_DIR", "SCHEDULE_DIR", "HM3D_ROOT",
    "SAM_MODEL_DIR", "HF_SHARED_CACHE", "HF_HOME", "DISPLAY", "XAUTHORITY", "QT_X11_NO_MITSHM",
    "REGOLO_API_KEY", "OPENAI_API_KEY", "OPENAI_BASE_URL", "OPENROUTER_API_KEY",
    "GEMINI_API_KEY", "GOOGLE_API_KEY", "MODAL_PERCEPTION_URL", "MODAL_TOKEN_ID",
    "FASTRTPS_DEFAULT_PROFILES_FILE", "FASTDDS_DEFAULT_PROFILES_FILE",
    "MODAL_TOKEN_SECRET", "ARCHIVE_DEPTH", "PYTHONUNBUFFERED", "ROS_DOMAIN_ID",
    "RVIZ", "USE_RVIZ", "HOUSE_FLOORS", "MAPPING_ONLY", "MAPPING_SECONDS", "WALL_DETECTOR",
    "CAP_MIN", "MERGE_ENGINE", "FRAME_QUEUE_MAX", "ROOM_FRAME_MAX", "ROOM_FRAME_STRIDE_M",
    "SCAN_COMPLETE_TOPIC", "SCAN_MERGE_SETTLE_S", "MOTION_POSITION_THRESHOLD", "MERGE_MIN_CONSECUTIVE",
    "HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "OPENAI_MODEL", "RUN_DIR", "PAL_ROBOT_CONNECTED", "RMW_IMPLEMENTATION", "ROS_LOCALHOST_ONLY", "CYCLONEDDS_URI", "MP3D_ROOT", "HF_CACHE",
    "IMAGE_TAG", "XDG_CACHE_HOME", "BRIDGE_PORT", "ROOM_VLM_MODEL", "EVAL_SCENE", "EVAL_DATASET_CONFIG",
    "DOCKER_HOST", "DOCKER_CONFIG", "HOME", "USER", "SKIP_TIAGO_HOST_DDS", "SSH_AUTH_SOCK",
}


def mounted_paths(root, env, writable=True):
    """Identity mounts preserve existing file references across all Python environments."""
    mounts = {str(root): False}
    workspace = Path(env["WORKSPACE_ROOT"])
    for name in ("results", "maps", "schedules", "ws"):
        path = workspace / name
        if writable:
            output_directory(path)
        if path.exists():
            resolved = path.resolve()
            mounts[str(resolved)] = writable
            if resolved == path or not path.is_relative_to(root):
                mounts[str(path)] = writable
    for key in ("HM3D_ROOT", "SAM_MODEL_DIR", "HF_SHARED_CACHE", "TIAGO_BAG_DIR", "BASELINE_REPOS_ROOT"):
        value = env.get(key)
        if value and Path(value).exists():
            mounts[value] = writable if key == "HF_SHARED_CACHE" or (key == "SAM_MODEL_DIR" and env.get("GRAPHAPI_PREPARE") == "1") else False
    for key in ("FEED_SCHEDULE", "MULTI_FLOOR_TRANSFORMS", "EVAL_SCENE", "EVAL_DATASET_CONFIG"):
        if env.get(key) and Path(env[key]).exists():
            parent = str(Path(env[key]).parent)
            if not any(Path(parent).is_relative_to(Path(existing)) for existing in mounts):
                mounts[parent] = False
    if env.get("XAUTHORITY") and Path(env["XAUTHORITY"]).is_file():
        mounts[env["XAUTHORITY"]] = False
    if env.get("SSH_AUTH_SOCK") and Path(env["SSH_AUTH_SOCK"]).exists():
        mounts[env["SSH_AUTH_SOCK"]] = False
    modal_settings = Path.home() / ".modal.toml"
    if modal_settings.is_file():
        mounts[str(modal_settings)] = False
    if Path("/tmp/.X11-unix").exists():
        mounts["/tmp/.X11-unix"] = False
    # A bind of the checkout preserves its symlinks, but their external targets
    # must also be visible to sibling application containers.
    resolved_mounts = {}
    for source, write in mounts.items():
        path = Path(source)
        resolved = path.resolve()
        resolved_mounts[str(resolved)] = write
        if resolved != path and not any(path.is_relative_to(Path(parent)) for parent in mounts
                                        if Path(parent) != path):
            resolved_mounts[source] = write
    return resolved_mounts


def command(root, env, arguments, *, habitat=False, name=None, gpu=False, control=False, executable=None):
    args = ["docker", "run", "--rm", "--init", "-i", "--network=host", "--entrypoint", "bash"]
    if name:
        args += ["--name", name]
    if env.get("GRAPHAPI_OPERATION_ID"):
        args += ["--label", "graphapi.operation=" + env["GRAPHAPI_OPERATION_ID"]]
    if gpu:
        args += ["--gpus", env.get("GRAPH_API_GPUS", "device=0"),
                 "-e", "NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics,display"]
    for key in sorted(env):
        if key in RUNTIME_ENV or key in env.get("GRAPHAPI_ENV_KEYS", "").split(",") or key.startswith(("FEED_", "HABITAT_", "MULTI_FLOOR_", "GRAPHAPI_", "GRAPH_API_", "DASH_", "BASELINE_", "EVAL_", "RTABMAP_", "PREFLIGHT_", "VITSAM_", "PERCEPTION_", "EXT_", "TIAGO_", "FOUND_", "GT_", "MODAL_", "FRAME_QUEUE_", "MOTION_", "SCAN_MERGE_", "OLLAMA_", "OPENROUTER_", "GA")):
            args += ["-e", key]
    for source, write in mounted_paths(root, env).items():
        args += ["-v", f"{source}:{source}" + ("" if write else ":ro")]
    for key, target, write in (("SAM_MODEL_DIR", "/models/vitsam", False), ("HF_SHARED_CACHE", "/models/hf", True)):
        if env.get(key) and Path(env[key]).is_dir():
            args += ["-v", env[key] + ":" + target + ("" if write else ":ro")]
    args += ["-v", f"{root}:/graph_api:ro", "-w", str(root)]
    if control:
        # Controllers share the host-user registry; actor identity survives PID namespaces.
        cache = Path(env.get("XDG_CACHE_HOME", str(Path.home() / ".cache")))
        (cache / "graphapi").mkdir(parents=True, exist_ok=True)
        args += ["-e", "XDG_CACHE_HOME=" + str(cache), "-v", f"{cache / 'graphapi'}:{cache / 'graphapi'}"]
        settings = root / "config/.runtime"
        settings.mkdir(parents=True, exist_ok=True)
        args += ["-v", f"{settings}:{settings}"]
        if name:
            args += ["-e", "GRAPHAPI_RESOURCE_ACTOR=" + name]
        # Existing native/remote orchestration tools launch sibling Docker containers.
        socket = env.get("GRAPHAPI_DOCKER_SOCKET", "/var/run/docker.sock")
        args += ["-v", f"{socket}:{socket}"]
        ssh = Path.home() / ".ssh"
        if ssh.is_dir():
            args += ["-v", f"{ssh}:{ssh}:ro"]
    python = executable or ("/opt/habitat/bin/python" if habitat else "python3")
    prefix = "source /opt/ros/humble/setup.bash; if [ -f /opt/graphapi_ws/install/setup.bash ]; then source /opt/graphapi_ws/install/setup.bash; fi; "
    prefix += 'for overlay in /root/tiago_public_ws/install/setup.bash /root/franka_ws/install/setup.bash; do if [ -f "$overlay" ]; then source "$overlay"; fi; done; '
    prefix += 'export PYTHONPATH="/graph_api:/graph_api/lost3dsg/src/perception_module:/graph_api/lost3dsg/test:${PYTHONPATH:-}"; '
    args += [env["IMAGE_TAG"], "-c", prefix + 'exec "$@"', "graphapi", python, *arguments]
    return args


def ensure_image(root, env, rebuild=False):
    dockerfile = root / "docker/sim/Dockerfile"
    revision = hashlib.sha256(dockerfile.read_bytes()).hexdigest()
    if not rebuild:
        installed = subprocess.run(["docker", "image", "inspect", "-f", '{{index .Config.Labels "graphapi.runtime"}}|{{index .Config.Labels "graphapi.runtime.build"}}', env["IMAGE_TAG"]],
                                   capture_output=True, text=True)
        if installed.returncode == 0:
            label, _, built = installed.stdout.strip().partition("|")
            if label == "2" and built == revision:
                return
            if label in ("", "<no value>"):
                # Unlabelled custom images must also provide the perception ROS dependency.
                check = subprocess.run(["docker", "run", "--rm", "--entrypoint", "bash", env["IMAGE_TAG"], "-lc",
                                        "test -x /opt/habitat/bin/python && source /opt/ros/humble/setup.bash && "
                                        "python3 -c 'import numpy as np; np.float = float; import tf_transformations'"],
                                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                if check.returncode == 0:
                    return
    subprocess.run(["docker", "build", "--build-arg", "RUNTIME_BUILD_REVISION=" + revision, "-f", str(dockerfile),
                    "-t", env["IMAGE_TAG"], str(root)], check=True)


def ensure_gazebo_image(root, env):
    image = env.get("GRAPHAPI_GAZEBO_IMAGE", "graphapi-public:latest")
    ensure_image(root, env)
    if subprocess.run(["docker", "image", "inspect", image], stdout=subprocess.DEVNULL,
                      stderr=subprocess.DEVNULL).returncode:
        subprocess.run(["docker", "build", "--build-arg", "BASE_IMAGE=" + env["IMAGE_TAG"], "-t", image, str(root)], check=True)
    return image


def run_python(root, env, arguments, **kwargs):
    return subprocess.run(command(root, env, arguments, **kwargs), env=env).returncode


def main():
    """Internal shell adapter: all Habitat imports happen in Docker, never on the host."""
    root = Path(os.environ["GRAPHAPI_ROOT"])
    env = dict(os.environ)
    role = sys.argv[1]
    script = {"feed": "habitat_feed_host.py", "persistent": "persistent_habitat_feed.py",
              "schedule": "schedule_batch.py"}[role]
    name = env.get("GRAPH_API_CONTAINER_NAME", "graphapi_live") + "-feed" if role != "schedule" else None
    if name and env.get("MULTI_FLOOR_COORD_DIR"):
        (Path(env["MULTI_FLOOR_COORD_DIR"]) / "feed.container").write_text(name + "\n")
    args = command(root, env, [str(root / "lost3dsg/test" / script), *sys.argv[2:]],
                   habitat=True, name=name, gpu=True)
    os.execvpe(args[0], args, env)


if __name__ == "__main__":
    main()
