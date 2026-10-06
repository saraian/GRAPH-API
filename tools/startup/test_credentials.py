"""Missing credentials must fail before any workload or resource allocation."""
from pathlib import Path
import ast
import os
import urllib.parse
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import yaml

from graphapi_cli import cli, launch
from graphapi_cli.batch import run_batch
from graphapi_cli.catalogue import run_tool
from graphapi_cli.configuration import ConfigurationError, environment
from graphapi_cli.credentials import prepare_vlm_credentials, uses_vlm
from graphapi_cli.registry import Registry


@pytest.fixture
def project(tmp_path, monkeypatch):
    root = tmp_path / "repo"
    (root / "config").mkdir(parents=True)
    module = root / "lost3dsg/src/perception_module"
    module.mkdir(parents=True)
    (module / "config.py").write_text("_DEFAULTS = {}\n")
    config = {"vlm": {"base_url": "https://api.regolo.ai/v1", "api_key": ""},
              "habitat": {"localization_mode": "ground_truth"},
              "tf": {"world_frame": "map"}, "bev": {"map_topic": "/map"}}
    for name in ("graphapi", "tiago_robot", "tiago_bag"):
        (root / f"config/{name}.yaml").write_text(yaml.safe_dump(config))
    workspace = tmp_path / "work"
    workspace.mkdir()
    bag = tmp_path / "bag"
    bag.mkdir()
    (bag / "metadata.yaml").write_text("{}\n")
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    env = {"WORKSPACE_ROOT": str(workspace), "HM3D_ROOT": str(tmp_path),
           "XDG_CACHE_HOME": str(tmp_path / "cache")}
    return root, env, bag, config


@pytest.mark.parametrize("mode", ["sim", "tiago", "bag"])
def test_missing_key_fails_before_reservation_or_workload(project, monkeypatch, mode):
    root, env, bag, _ = project
    reserve = Mock(side_effect=AssertionError("must not reserve resources"))
    execute = Mock(side_effect=AssertionError("must not start workload"))
    monkeypatch.setattr(Registry, "reserve", reserve)
    monkeypatch.setattr(launch, "execute", execute)
    with pytest.raises(ConfigurationError, match="export REGOLO_API_KEY"):
        launch.start(root, {"mode": mode, "bag": str(bag)}, inherited=env, detach=True)
    reserve.assert_not_called()
    execute.assert_not_called()
    assert not (Path(env["WORKSPACE_ROOT"]) / "results").exists()
    assert not (root / "config/.runtime").exists()


def test_cli_reports_missing_key_with_nonzero_exit(project, monkeypatch, capsys):
    root, env, _, _ = project
    for name in ("OPENAI_API_KEY", "REGOLO_API_KEY", "OPENROUTER_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    for name, value in env.items():
        monkeypatch.setenv(name, value)
    assert cli.main(["--project", str(root), "run", "sim", "--detach"]) == 2
    assert "VLM API key is missing" in capsys.readouterr().err
    assert not (Path(env["WORKSPACE_ROOT"]) / "results").exists()


@pytest.mark.parametrize("name", ["OPENAI_API_KEY", "REGOLO_API_KEY", "OPENROUTER_API_KEY"])
def test_supported_environment_keys_reach_inference(project, name):
    root, env, _, cfg = project
    env[name] = "test-key"
    prepare_vlm_credentials(root, cfg, env)
    assert env["OPENAI_API_KEY"] == "test-key"


@pytest.mark.parametrize("base_url", ["https://generativelanguage.googleapis.com/v1",
                                      "https://aiplatform.googleapis.com/v1"])
@pytest.mark.parametrize("name", ["GEMINI_API_KEY", "GOOGLE_API_KEY"])
def test_google_keys_are_accepted_for_gemini(project, base_url, name):
    root, env, _, cfg = project
    cfg["vlm"]["base_url"] = base_url
    env[name] = "google-test-key"
    prepare_vlm_credentials(root, cfg, env)
    assert env["GEMINI_API_KEY"] == "google-test-key"


def test_google_key_does_not_satisfy_regolo_authentication(project):
    root, env, _, cfg = project
    env["GEMINI_API_KEY"] = "google-test-key"
    with pytest.raises(ConfigurationError, match="REGOLO_API_KEY"):
        prepare_vlm_credentials(root, cfg, env)


@pytest.mark.parametrize("vlm,keys", [
    ({"base_url": "https://api.regolo.ai/v1"}, {"OPENAI_API_KEY": "first", "REGOLO_API_KEY": "second"}),
    ({"base_url": "https://api.regolo.ai/v1", "api_key": "configured"}, {"OPENAI_API_KEY": "exported"}),
    ({"base_url": "https://api.regolo.ai/v1"}, {"OPENROUTER_API_KEY": "router"}),
    ({"base_url": "https://aiplatform.googleapis.com/v1"}, {"GOOGLE_API_KEY": "google"}),
    ({"base_url": "https://generativelanguage.googleapis.com/v1"}, {"GEMINI_API_KEY": "gemini", "GOOGLE_API_KEY": "google"}),
    ({"base_url": "https://api.regolo.ai/v1", "provider": "gemini"}, {"GEMINI_API_KEY": "gemini"}),
    ({"base_url": "https://aiplatform.googleapis.com/v1", "provider": "openai"}, {"OPENAI_API_KEY": "openai"}),
    ({"base_url": "http://localhost:11434/v1"}, {}),
])
def test_startup_resolution_matches_actual_inference_client(project, monkeypatch, vlm, keys):
    root, env, _, cfg = project
    # Load only the existing auth functions, avoiding ROS/CUDA imports. This
    # compares independent startup validation with the actual inference rules.
    source = Path(__file__).resolve().parents[2] / "lost3dsg/src/perception_module/cv_utils.py"
    tree = ast.parse(source.read_text())
    wanted = {"_endpoint_is_local", "_is_gemini_vlm", "_resolve_api_key"}
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in wanted]
    namespace = {"CFG": {"vlm": vlm}, "os": os, "urllib": urllib,
                 "__file__": str(root / "lost3dsg/src/perception_module/cv_utils.py"),
                 "_LOCAL_HOSTS": {"localhost", "127.0.0.1", "::1", "0.0.0.0", "host.docker.internal"}}
    for name in ("GEMINI_API_KEY", "GOOGLE_API_KEY", "OPENAI_API_KEY", "REGOLO_API_KEY", "OPENROUTER_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    for name, value in keys.items():
        monkeypatch.setenv(name, value)
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(source), "exec"), namespace)
    before = namespace["_resolve_api_key"]()
    cfg["vlm"] = dict(vlm)
    env.update(keys)
    prepare_vlm_credentials(root, cfg, env)
    # The launcher removes config secrets before handing the config to ROS.
    namespace["CFG"]["vlm"]["api_key"] = ""
    for name, value in env.items():
        if name.endswith("API_KEY"):
            monkeypatch.setenv(name, value)
    assert namespace["_resolve_api_key"]() == before


def test_config_key_wins_and_is_not_saved_in_runtime_yaml(project):
    root, env, _, _ = project
    (root / "config/local.yaml").write_text("overrides:\n  vlm:\n    api_key: private-test-key\n")
    env["OPENAI_API_KEY"] = "older-test-key"
    prepared, state = launch.prepare_job(root, {"mode": "sim"}, env)
    try:
        assert prepared["OPENAI_API_KEY"] == "private-test-key"
        assert "private-test-key" not in Path(state["config"]).read_text()
        assert "private-test-key" not in Path(state["log"]).with_name("status.json").read_text()
    finally:
        Registry(env["WORKSPACE_ROOT"]).release(state["operation_id"])


def test_legacy_key_reaches_container_environment(project):
    root, env, _, cfg = project
    (root / "lost3dsg/src/perception_module/api.txt").write_text("legacy-test-key\n")
    prepare_vlm_credentials(root, cfg, env)
    assert env["OPENAI_API_KEY"] == "legacy-test-key"


@pytest.mark.parametrize("hostname", ["localhost", "127.0.0.1", "[::1]", "host.docker.internal"])
def test_local_vlm_needs_no_key(project, hostname):
    root, env, _, cfg = project
    cfg["vlm"]["base_url"] = f"http://{hostname}:11434/v1"
    prepare_vlm_credentials(root, cfg, env)
    assert "OPENAI_API_KEY" not in env


@pytest.mark.parametrize("options", [{"no_perception": True}, {"mapping_only": True}])
def test_disabled_perception_skips_credential_check(project, options):
    root, env, _, _ = project
    prepared, state = launch.prepare_job(root, {"mode": "sim", **options}, env)
    Registry(env["WORKSPACE_ROOT"]).release(state["operation_id"])
    assert "GRAPHAPI_VLM_CHECKED" not in prepared


def test_tiago_disabled_perception_and_legacy_overrides():
    assert not uses_vlm({"mode": "bag"}, {"FOUND_START_PERCEPTION": "0"})
    assert not uses_vlm({"mode": "tiago", "legacy_args": ["--no-perception"]}, {})
    assert uses_vlm({"mode": "bag", "legacy_args": ["--no-perception", "--perception"]}, {})


def test_whitespace_is_a_missing_key_and_error_hides_other_secrets(project):
    root, env, _, cfg = project
    cfg["vlm"]["base_url"] = "https://secret-user:secret-password@api.regolo.ai/v1?token=secret-token"
    env["REGOLO_API_KEY"] = " \n "
    with pytest.raises(ConfigurationError) as error:
        prepare_vlm_credentials(root, cfg, env)
    assert "secret" not in str(error.value)


def test_batch_checks_all_pending_arms_before_first_launch(project, monkeypatch):
    root, env, _, _ = project
    schedule = root / "config/batch.yaml"
    schedule.write_text("arms:\n- name: local\n  config: {vlm.base_url: 'http://localhost:11434/v1'}\n"
                        "- name: remote\n")
    start = Mock(side_effect=AssertionError("must not launch first arm"))
    monkeypatch.setattr("graphapi_cli.batch.start", start)
    args = SimpleNamespace(schedule=str(schedule), config=None, local=None, gpus="0",
                           force=False, dry_run=False, continue_on_failure=False)
    with pytest.raises(ConfigurationError, match="REGOLO_API_KEY"):
        run_batch(root, env, args)
    start.assert_not_called()
    assert not (root / "config/.runtime").exists()


def test_direct_vlm_tool_checks_key_before_supervision(project, monkeypatch):
    root, env, _, _ = project
    monkeypatch.setattr("graphapi_cli.catalogue.catalogue", lambda _root: {
        "perception": {"kind": "script", "path": "lost3dsg/src/perception_module/perception_2.py",
                       "environment": "ros", "gpu": True}})
    supervise = Mock(side_effect=AssertionError("must not allocate tool operation"))
    monkeypatch.setattr("graphapi_cli.auxiliary.supervise", supervise)
    with pytest.raises(ConfigurationError, match="REGOLO_API_KEY"):
        run_tool(root, env, "perception", [])
    supervise.assert_not_called()


def test_tool_passes_key_to_docker_and_saves_only_redacted_config(project, monkeypatch):
    root, env, _, _ = project
    env, _ = environment(root, inherited=env)
    env.update(REGOLO_API_KEY="tool-test-key", GRAPHAPI_OPERATION_ID="test-tool")
    monkeypatch.setattr("graphapi_cli.catalogue.catalogue", lambda _root: {
        "perception": {"kind": "script", "path": "lost3dsg/src/perception_module/perception_2.py",
                       "environment": "ros", "gpu": True}})
    monkeypatch.setattr("graphapi_cli.catalogue.ensure_image", lambda *_args: None)
    command = run_tool(root, env, "perception", [], _managed=True)
    assert "OPENAI_API_KEY" in command
    assert "tool-test-key" not in command
    assert env["OPENAI_API_KEY"] == "tool-test-key"
    assert "tool-test-key" not in Path(env["GRAPH_API_CONFIG"]).read_text()


def test_parallel_tool_rejects_missing_key_before_spawning(project, monkeypatch):
    root, env, _, _ = project
    spawn = Mock(side_effect=AssertionError("must not spawn pipelines"))
    monkeypatch.setattr("graphapi_cli.cli.subprocess.run", spawn)
    monkeypatch.setattr("graphapi_cli.catalogue.catalogue", lambda _root: {
        "run-pipelines": {"kind": "script", "path": "graphapi_cli/runtime/run_pipelines.sh",
                          "environment": "orchestrator", "gpu": False}})
    with pytest.raises(ConfigurationError, match="REGOLO_API_KEY"):
        run_tool(root, env, "run-pipelines", ["0", "hm3d_00861"])
    spawn.assert_not_called()


def test_help_and_tool_dry_run_do_not_require_credentials(project, monkeypatch, capsys):
    root, env, _, _ = project
    with pytest.raises(SystemExit) as exited:
        cli.main(["--project", str(root), "run", "sim", "--help"])
    assert exited.value.code == 0
    monkeypatch.setattr("graphapi_cli.catalogue.catalogue", lambda _root: {
        "perception": {"kind": "script", "path": "lost3dsg/src/perception_module/perception_2.py",
                       "environment": "ros", "gpu": True}})
    env, _ = environment(root, inherited=env)
    assert run_tool(root, env, "perception", [], dry_run=True) == 0
    assert "docker run" in capsys.readouterr().out
