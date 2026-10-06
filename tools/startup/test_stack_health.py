"""Required child exits must terminate startup waits, including clean exits."""
import json
import os
from pathlib import Path
import subprocess

import pytest

from graphapi_cli.stack_health import stack_failure, pane_exit
from graphapi_cli import torch_startup


ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize('line', ['bridge|0|', 'rviz|1|1', 'monitor|1|0'])
def test_live_or_optional_private_panes_do_not_end_run(line):
    assert pane_exit(line, {}, 10) is None


def test_private_pane_waits_for_exit_status_then_accepts_zero():
    pending = {}
    assert pane_exit('bag|1|', pending, 10) is None
    assert pane_exit('bag|1|', pending, 11) is None
    assert pane_exit('bag|1|0', pending, 11.5) == ('bag', '0')


def test_private_pane_unknown_exit_cannot_wait_forever():
    pending = {}
    assert pane_exit('bag|1|', pending, 10) is None
    assert 'status unavailable' in pane_exit('bag|1|', pending, 12)[1]


def test_native_player_signal_exit_is_preserved():
    assert pane_exit('bag|1|139', {}, 10) == ('bag', '139')


@pytest.mark.parametrize('pane_code,player_code', [('', '0\n'), ('139', '0\n'), ('0', '137\n')])
def test_exact_player_exit_overrides_wrapper_status(pane_code, player_code):
    assert pane_exit('bag|1|' + pane_code, {}, 10, player_code) == ('bag', player_code.strip())


@pytest.mark.parametrize('code', [-6, 0, 1])
def test_required_exit_is_reported_while_launcher_is_alive(tmp_path, code):
    events = tmp_path / 'events'
    event = dict(component='perception_2', returncode=code, required=True, during_shutdown=False)
    events.write_text(json.dumps(event) + '\n')
    assert stack_failure(events, os.getpid()) == event


def test_optional_and_shutdown_exits_do_not_end_the_run(tmp_path):
    events = tmp_path / 'events'
    events.write_text('\n'.join(json.dumps(event) for event in
        [dict(required=False, during_shutdown=False), dict(required=True, during_shutdown=True)]))
    assert stack_failure(events, os.getpid()) is None


def test_zombie_launcher_is_detected(tmp_path, monkeypatch):
    monkeypatch.setattr(Path, 'read_text', lambda *args: '123 (launcher) Z 1 2 3')
    assert stack_failure(tmp_path / 'missing', os.getpid())['component'] == 'ROS2_LAUNCH'


@pytest.mark.parametrize('code', [-6, 0])
def test_actual_startup_guard_exits_and_records_failure_before_warmup(tmp_path, code):
    source = (ROOT / 'lost3dsg/test/live_stack_container.sh').read_text()
    block = source.split('_check_ros_stack() {', 1)[1].split('\n}\n', 1)[0]
    block = block.replace('/ws/output', str(tmp_path))
    (tmp_path / 'component_events.jsonl').write_text(json.dumps(
        dict(component='perception_2', returncode=code, required=True, during_shutdown=False)) + '\n')
    script = 'sleep 30 &\nLAUNCH_PID=$!\ntrap \'kill "$LAUNCH_PID" 2>/dev/null || true\' EXIT\n'
    script += '_check_ros_stack() {' + block + '\n}\n_check_ros_stack\necho INCORRECTLY_CONTINUED\n'
    result = subprocess.run(['bash', '-c', script], env=dict(os.environ, PYTHONPATH=str(ROOT)),
                            capture_output=True, text=True, timeout=10)
    assert result.returncode == 1, result.stderr
    assert 'required component perception_2 exited' in result.stdout
    assert 'INCORRECTLY_CONTINUED' not in result.stdout
    terminal = json.loads((tmp_path / 'terminating_node.json').read_text())
    assert terminal['exit_status'] == code


def test_torch_import_restores_main_and_new_native_thread_affinity(monkeypatch):
    calls = []
    monkeypatch.setattr(torch_startup.os, 'sched_getaffinity', lambda pid: {3, 8})
    monkeypatch.setattr(torch_startup.os, 'sched_setaffinity', lambda pid, mask: calls.append((pid, mask)))
    snapshots = iter([{100}, {100, 101}])
    monkeypatch.setattr(torch_startup, '_threads', lambda: next(snapshots))
    result = object()

    def load(name):
        assert calls == [(0, {3})]
        return result

    monkeypatch.setattr(torch_startup.importlib, 'import_module', load)
    assert torch_startup.import_torch() is result
    assert calls == [(0, {3}), (0, {3, 8}), (101, {3, 8})]


def test_torch_import_failure_still_restores_affinity(monkeypatch):
    calls = []
    monkeypatch.setattr(torch_startup.os, 'sched_getaffinity', lambda pid: {3, 8})
    monkeypatch.setattr(torch_startup.os, 'sched_setaffinity', lambda pid, mask: calls.append((pid, mask)))
    monkeypatch.setattr(torch_startup, '_threads', lambda: set())

    def fail(name):
        raise ImportError('test import failure')

    monkeypatch.setattr(torch_startup.importlib, 'import_module', fail)
    with pytest.raises(ImportError):
        torch_startup.import_torch()
    assert calls == [(0, {3}), (0, {3, 8})]
