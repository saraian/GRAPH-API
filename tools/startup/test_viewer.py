"""RViz ownership, GUI startup failures and dashboard operation selection."""
from pathlib import Path
import json
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import yaml

from graphapi_cli.configuration import ConfigurationError, environment
from graphapi_cli import viewer


ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def launch(tmp_path):
    env, _ = environment(ROOT, inherited={"WORKSPACE_ROOT": str(tmp_path), "DISPLAY": ":0"})
    row = {"operation_id": "20261005T000000Z_test", "mode": "sim", "state": "RUNNING",
           "gpu": "2", "domain": 27, "log": str(tmp_path / "results/operations/test/launch.log")}
    return env, row


def test_active_view_selects_acquisition_when_dashboard_is_running(launch):
    env, row = launch
    with patch.object(viewer, "Registry") as registry:
        registry.return_value.rows.return_value = [dict(row, mode="dashboard"), row]
        assert viewer.acquisition(env["WORKSPACE_ROOT"]) == row


@pytest.mark.parametrize("rows", [[], [{"mode": "dashboard", "state": "RUNNING"}],
                                 [{"mode": "sim", "state": "RUNNING"}] * 2])
def test_active_view_refuses_missing_or_ambiguous_acquisition(rows):
    with patch.object(viewer, "Registry") as registry:
        registry.return_value.rows.return_value = rows
        with pytest.raises(ConfigurationError, match="exactly one active acquisition"):
            viewer.acquisition("/unused")


@pytest.mark.parametrize("mode", ["sim", "tiago", "bag"])
def test_detached_view_uses_run_domain_gpu_config_and_ownership(launch, mode):
    env, row = launch
    row["mode"] = mode
    row["container"] = "private-pal" if mode != "sim" else "simulation"
    responses = [SimpleNamespace(returncode=1, stdout="")]
    if mode != "sim":
        responses.append(SimpleNamespace(returncode=0, stdout=json.dumps([{"Image": "sha256:private-pal"}])))
    responses.extend([SimpleNamespace(returncode=0, stderr=""), SimpleNamespace(returncode=0, stdout="true\n")])
    with patch.object(viewer, "ensure_image"), patch.object(viewer.time, "sleep"), \
            patch.object(viewer.subprocess, "run", side_effect=responses) as run:
        result = viewer.start(ROOT, env, row, detach=True)
    call = next(call for call in run.call_args_list if call.args[0][:2] == ["docker", "run"])
    cmd = call.args[0]
    managed = call.kwargs["env"]
    assert "--detach" in cmd
    assert str(Path(row["log"]).with_name("view.rviz")) in cmd
    assert managed["ROS_DOMAIN_ID"] == "27"
    assert cmd[cmd.index("--gpus") + 1] == "device=2"
    assert "graphapi.operation=" + row["operation_id"] in cmd
    assert '>> "$GRAPHAPI_RVIZ_LOG" 2>&1' in cmd[cmd.index(managed["IMAGE_TAG"]) + 2]
    if mode == "sim":
        assert managed["FASTRTPS_DEFAULT_PROFILES_FILE"] == "/graph_api/config/rviz_fastdds.xml"
        assert "FASTRTPS_DEFAULT_PROFILES_FILE" in cmd
        assert "use_sim_time:=false" in cmd
    else:
        assert managed["IMAGE_TAG"] == "sha256:private-pal"
        assert "private-pal:ro" in cmd
        assert "source /opt/pal/alum/setup.bash" in cmd[cmd.index(managed["IMAGE_TAG"]) + 2]
        assert managed["ROS_LOCALHOST_ONLY"] == ("1" if mode == "bag" else "0")
        assert "use_sim_time:=" + ("true" if mode == "bag" else "false") in cmd
    assert result["started"]
    assert result["operation_id"] == row["operation_id"]


def test_view_reports_early_exit_with_gui_diagnostic(launch):
    env, row = launch
    log = Path(row["log"]).with_name("rviz.log")
    log.parent.mkdir(parents=True)
    log.write_text("qt.qpa.xcb: could not connect to display :0\n")
    with patch.object(viewer, "ensure_image"), patch.object(viewer.time, "sleep"), \
            patch.object(viewer.subprocess, "run", side_effect=[
                SimpleNamespace(returncode=1, stdout=""), SimpleNamespace(returncode=0, stderr=""),
                SimpleNamespace(returncode=1, stdout="")]):
        with pytest.raises(ConfigurationError, match="could not connect to display"):
            viewer.start(ROOT, env, row, detach=True)


def test_view_reuses_existing_window(launch):
    env, row = launch
    with patch.object(viewer.subprocess, "run", return_value=SimpleNamespace(returncode=0, stdout="true\n")) as run:
        result = viewer.start(ROOT, env, row, detach=True)
    assert result["already_running"]
    assert run.call_count == 1


def test_view_requires_display_before_creating_gui_container(launch):
    env, row = launch
    env.pop("DISPLAY", None)
    with patch.object(viewer.subprocess, "run", return_value=SimpleNamespace(returncode=1, stdout="")) as run:
        with pytest.raises(ConfigurationError, match="desktop display"):
            viewer.start(ROOT, env, row, detach=True)
    assert run.call_count == 1


@pytest.mark.parametrize("mode,frame,map_topic,cloud", [
    ("sim", "map", "/rtabmap/map", True),
    ("tiago", "found_map", "/rtabmap/map", True),
    ("bag", "map", "/map", False),
])
def test_generated_config_matches_map_camera_and_robot(launch, mode, frame, map_topic, cloud):
    env, row = launch
    row["mode"] = mode
    path, view = viewer.prepare_config(ROOT, env, row)
    layout = yaml.safe_load(path.read_text())["Visualization Manager"]
    displays = layout["Displays"]
    assert layout["Global Options"]["Fixed Frame"] == frame
    maps = next(d for d in displays if d["Class"] == "rviz_default_plugins/Map")
    assert maps["Topic"]["Value"] == map_topic
    cloud_display = next(d for d in displays if d.get("Name") == "rtabmap cloud_map")
    assert cloud_display["Enabled"] == cloud
    image = next(d for d in displays if d["Name"] == "camera rgb")
    assert image["Topic"]["Value"] == ("/camera/rgb" if mode == "sim" else "/head_front_camera/rgb/image_raw")
    assert image["Topic"]["Reliability Policy"] == "Best Effort"
    axes = next(d for d in displays if d["Class"] == "rviz_default_plugins/Axes")
    assert axes["Reference Frame"] == ("base_link" if mode == "sim" else "base_footprint")
    assert layout["Views"]["Current"]["Target Frame"] == axes["Reference Frame"]
    if mode != "sim":
        robot = next(d for d in displays if d["Class"] == "rviz_default_plugins/RobotModel")
        assert robot["Description Topic"]["Value"] == "/robot_description"
        assert robot["Description Topic"]["Durability Policy"] == "Transient Local"


def test_view_uses_run_snapshot_instead_of_changed_dashboard_settings(launch, tmp_path):
    from graphapi_cli.rviz_config import settings
    env, row = launch
    cfg = {"tf": {"world_frame": "office"}, "bev": {"map_topic": "/office/map", "cloud_map_topic": ""},
           "frames": {"camera": "custom_camera"}}
    selected = tmp_path / "run.yaml"
    selected.write_text(yaml.safe_dump(cfg))
    row.update(mode="bag", config=str(selected), viewer=settings("bag", cfg, {"TIAGO_RGB_TOPIC": "/selected/rgb"}))
    env.update(TIAGO_RGB_TOPIC="/wrong/rgb", CYCLONEDDS_URI="robot-only.xml")
    path, view = viewer.prepare_config(ROOT, env, row)
    manager = yaml.safe_load(path.read_text())["Visualization Manager"]
    assert manager["Global Options"]["Fixed Frame"] == "office"
    assert next(d for d in manager["Displays"] if d["Name"] == "camera rgb")["Topic"]["Value"] == "/selected/rgb"
    assert view["CYCLONEDDS_URI"] == ""


def test_bag_replays_a_latched_robot_description():
    source = (ROOT / "tiago/found-docker/found-robot-stack.sh").read_text()
    bag_topics = source.split("BAG_TOPICS=(", 1)[1].split(")", 1)[0]
    assert "/robot_description" in bag_topics
    qos = yaml.safe_load((ROOT / "config/tiago_bag_qos.yaml").read_text())
    assert qos["/robot_description"]["durability"] == "transient_local"
