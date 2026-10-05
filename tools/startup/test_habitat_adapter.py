"""Exercise the adapter process, including the former direct-script invocation."""
import os
from pathlib import Path
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize("invocation", ["module", "script"])
@pytest.mark.parametrize("role,script", [("schedule", "schedule_batch.py"),
                                        ("feed", "habitat_feed_host.py"),
                                        ("persistent", "persistent_habitat_feed.py")])
def test_adapter_reaches_docker_for_each_role_from_another_directory(tmp_path, invocation, role, script):
    binary = tmp_path / "bin/docker"
    binary.parent.mkdir()
    binary.write_text('#!/bin/sh\nprintf "%s\\n" "$@" > "$GRAPHAPI_DOCKER_ARGS_FILE"\nexit 42\n')
    binary.chmod(0o755)
    captured = tmp_path / "arguments"
    env = dict(os.environ, PATH=str(binary.parent) + os.pathsep + os.environ["PATH"],
               PYTHONPATH=str(ROOT), GRAPHAPI_ROOT=str(ROOT), WORKSPACE_ROOT=str(tmp_path / "workspace"),
               IMAGE_TAG="runtime:test", GRAPHAPI_DOCKER_ARGS_FILE=str(captured))
    command = [sys.executable, "-m", "graphapi_cli.docker"] if invocation == "module" else [sys.executable, str(ROOT / "graphapi_cli/docker.py")]
    result = subprocess.run([*command, role, "--help"], cwd=tmp_path, env=env,
                            capture_output=True, text=True)
    assert result.returncode == 42, result.stderr
    args = captured.read_text().splitlines()
    assert str(ROOT / "lost3dsg/test" / script) in args
    assert "/opt/habitat/bin/python" in args
    assert "--help" in args
