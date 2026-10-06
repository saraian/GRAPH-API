"""Status visibility and history access through the public CLI."""
import json
import os

import pytest

from graphapi_cli import cli
from graphapi_cli.registry import Registry, write_json


@pytest.fixture
def records(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    monkeypatch.setattr(cli, "environment", lambda *args: ({"WORKSPACE_ROOT": str(tmp_path)}, {}))
    registry = Registry(tmp_path)
    for state in ("PREPARING", "RUNNING", "DRAINING", "COMPLETED", "FAILED", "INTERRUPTED"):
        row = dict(operation_id="op-" + state.lower(), state=state, mode="sim", pid=os.getpid(),
                   created=1, log=str(tmp_path / "launch.log"), scene="hm3d_00861",
                   bridge_url="http://127.0.0.1:18000", bundles=[])
        write_json(registry.operations / row["operation_id"] / "status.json", row)
    return registry


def test_default_status_shows_only_active_operations(records, capsys):
    assert cli.main(["status"]) == 0
    out = capsys.readouterr().out
    for state in ("PREPARING", "RUNNING", "DRAINING"):
        assert "op-" + state.lower() in out
    for state in ("COMPLETED", "FAILED", "INTERRUPTED"):
        assert "op-" + state.lower() not in out
    assert "OPERATION" in out and "STATE" in out and "ENDPOINT" in out
    assert "http://127.0.0.1:18000" in out
    assert "'operation_id':" not in out
    # Filtering the display must preserve the complete operation history.
    assert len(records.rows()) == 6


def test_all_status_includes_terminal_history(records, capsys):
    assert cli.main(["status", "--all", "--json"]) == 0
    rows = json.loads(capsys.readouterr().out)
    assert len(rows) == 6


@pytest.mark.parametrize("state", ["completed", "failed", "interrupted"])
def test_terminal_operation_remains_available_by_exact_id(records, capsys, state):
    assert cli.main(["status", "op-" + state]) == 0
    out = capsys.readouterr().out
    assert "op-" + state in out
    assert "Log: " in out


@pytest.mark.parametrize("arguments", [["--json", "status"], ["status", "--json"]])
def test_json_status_obeys_default_visibility_and_accepts_both_flag_positions(records, capsys, arguments):
    assert cli.main(arguments) == 0
    rows = json.loads(capsys.readouterr().out)
    assert {row["state"] for row in rows} == {"PREPARING", "RUNNING", "DRAINING"}


def test_no_active_operations_has_clear_message_and_empty_json(records, capsys):
    for row in records.rows():
        if row["state"] in ("PREPARING", "RUNNING", "DRAINING"):
            (records.operations / row["operation_id"] / "status.json").unlink()
    assert cli.main(["status"]) == 0
    assert "No active operations" in capsys.readouterr().out
    assert cli.main(["status", "--json"]) == 0
    assert json.loads(capsys.readouterr().out) == []
    assert len(records.rows()) == 3
    for row in records.rows():
        (records.operations / row["operation_id"] / "status.json").unlink()
    assert cli.main(["status", "--all"]) == 0
    assert "No recorded operations." in capsys.readouterr().out


def test_stopping_terminal_operation_explains_it_is_already_stopped(records, capsys, monkeypatch):
    monkeypatch.setattr("graphapi_cli.launch.stop_operation", lambda workspace, selector: records.select(selector))
    assert cli.main(["stop", "op-failed"]) == 0
    assert "Already stopped: op-failed (sim, FAILED)." in capsys.readouterr().out
    assert cli.main(["stop", "op-failed", "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["state"] == "FAILED"


def test_stopping_active_operation_reports_request_and_exact_check_command(records, capsys, monkeypatch):
    monkeypatch.setattr("graphapi_cli.launch.stop_operation", lambda workspace, selector: records.select(selector))
    assert cli.main(["stop", "op-running"]) == 0
    out = capsys.readouterr().out
    assert "Stop requested for op-running" in out
    assert "./graphapi status op-running" in out
