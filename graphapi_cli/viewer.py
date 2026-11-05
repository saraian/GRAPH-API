"""RViz launch shared by the CLI and dashboard, tied to an acquisition."""
from pathlib import Path
import json
import subprocess
import time

import yaml

from .configuration import ConfigurationError, read_yaml, resolve_config, node_profile
from .docker import command, ensure_image
from .registry import Registry
from .rviz_config import settings, configure


def prepare_config(root, env, row):
    mode = row["mode"]
    cfg = (read_yaml(Path(row["config"])) if row.get("config") else
           resolve_config(root, node_profile(mode, row.get("map_source")), env.get("GRAPHAPI_LOCAL_CONFIG"))[1])
    view = row.get("viewer") or settings(mode, cfg, env)
    source = root / "config" / ("live.rviz" if mode == "sim" else "tiago_live.rviz")
    resolved = Path(row["log"]).with_name("view.rviz")
    resolved.parent.mkdir(parents=True, exist_ok=True)
    resolved.write_text(yaml.safe_dump(configure(read_yaml(source), cfg, mode, view), sort_keys=False))
    return resolved, view


def acquisition(workspace, selector="active"):
    registry = Registry(workspace)
    if selector == "active":
        rows = [row for row in registry.rows()
                if row.get("mode") in ("sim", "tiago", "bag")
                and row["state"] in ("PREPARING", "RUNNING")]
        if len(rows) != 1:
            raise ConfigurationError(f"RViz requires exactly one active acquisition; found {len(rows)}")
        return rows[0]
    row = registry.select(selector)
    if row.get("mode") not in ("sim", "tiago", "bag") or row["state"] not in ("PREPARING", "RUNNING"):
        raise ConfigurationError("RViz requires an active acquisition")
    return row


def start(root, env, row, *, detach=False):
    """Use the acquisition's domain/GPU; report early GUI failures to the caller."""
    if row.get("mode") not in ("sim", "tiago", "bag") or row["state"] not in ("PREPARING", "RUNNING"):
        raise ConfigurationError("RViz requires an active acquisition")
    name = "graphapi-" + row["operation_id"].lower() + "-rviz"
    running = subprocess.run(["docker", "inspect", "-f", "{{.State.Running}}", name],
                             capture_output=True, text=True, timeout=10)
    if running.returncode == 0 and running.stdout.strip() == "true":
        return {"started": False, "already_running": True, "container": name,
                "operation_id": row["operation_id"]}
    managed = dict(env)
    if not managed.get("DISPLAY"):
        raise ConfigurationError("RViz needs a desktop display; launch the dashboard from a terminal with DISPLAY set")
    managed.update(ROS_DOMAIN_ID=str(row["domain"]), GRAPHAPI_OPERATION_ID=row["operation_id"],
                   GRAPH_API_GPUS="device=" + row["gpu"], QT_X11_NO_MITSHM="1")
    config, view = prepare_config(root, managed, row)
    managed.update({key: view[key] for key in ("RMW_IMPLEMENTATION", "ROS_LOCALHOST_ONLY", "CYCLONEDDS_URI")})
    private = row["mode"] in ("tiago", "bag")
    if private:
        info = subprocess.run(["docker", "inspect", row["container"]], capture_output=True, text=True, timeout=10)
        if info.returncode:
            raise ConfigurationError("RViz needs the active run's private PAL container and robot model assets")
        managed["IMAGE_TAG"] = json.loads(info.stdout)[0]["Image"]
    else:
        ensure_image(root, managed)
    if managed["RMW_IMPLEMENTATION"] in ("rmw_fastrtps_cpp", "rmw_fastrtps_dynamic_cpp"):
        managed["FASTRTPS_DEFAULT_PROFILES_FILE"] = "/graph_api/config/rviz_fastdds.xml"
        managed["FASTDDS_DEFAULT_PROFILES_FILE"] = managed["FASTRTPS_DEFAULT_PROFILES_FILE"]
    if detach:
        log = Path(row["log"]).with_name("rviz.log")
        log.parent.mkdir(parents=True, exist_ok=True)
        managed["GRAPHAPI_RVIZ_LOG"] = str(log)
    arguments = ["-d", str(config), "--ros-args", "-p", "use_sim_time:=" + str(view["use_sim_time"]).lower()]
    cmd = command(root, managed, arguments,
                  executable="rviz2", gpu=True, name=name)
    if private:
        cmd[2:2] = ["--volumes-from", row["container"] + ":ro"]
        position = cmd.index(managed["IMAGE_TAG"]) + 2
        cmd[position] = cmd[position].replace('exec "$@"',
            'source /opt/pal/alum/setup.bash; if [ -f /ws/install/setup.bash ]; then source /ws/install/setup.bash; fi; exec "$@"')
    if not detach:
        return subprocess.run(cmd, env=managed).returncode
    cmd[2:2] = ["--detach"]
    cmd[cmd.index(managed["IMAGE_TAG"]) + 2] += ' >> "$GRAPHAPI_RVIZ_LOG" 2>&1'
    result = subprocess.run(cmd, env=managed, capture_output=True, text=True, timeout=30)
    if result.returncode:
        raise ConfigurationError(result.stderr.strip() or "Docker could not start RViz")
    time.sleep(1)
    running = subprocess.run(["docker", "inspect", "-f", "{{.State.Running}}", name],
                             capture_output=True, text=True, timeout=10)
    if running.returncode or running.stdout.strip() != "true":
        detail = "\n".join(log.read_text(errors="replace").splitlines()[-12:]) if log.exists() else "no RViz log was produced"
        raise ConfigurationError("RViz exited during startup: " + detail)
    return {"started": True, "container": name, "operation_id": row["operation_id"],
            "log": str(log), "display": managed["DISPLAY"]}
