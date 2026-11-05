"""Stopping a Docker client must also close its detached simulation workload."""
import os
import json
import signal
from types import SimpleNamespace

import pytest

from graphapi_cli import launch
from graphapi_cli.registry import Registry, write_json


def test_stop_closes_owned_workload_before_launcher_and_keeps_supervisor(monkeypatch):
    active = ['application', 'feed', 'application-launcher', 'supervisor']
    calls = []
    monkeypatch.setattr(launch, 'owned_containers', lambda _operation: list(active))

    def stop(command, **kwargs):
        calls.append((command, kwargs['timeout']))
        for name in command[4:]:
            active.remove(name)
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(launch.subprocess, 'run', stop)
    launch.stop_simulation(dict(operation_id='mine', container='application', actor='supervisor'))
    assert calls == [(['docker', 'stop', '-t', '120', 'application', 'feed'], 135),
                     (['docker', 'stop', '-t', '15', 'application-launcher'], 30)]
    assert active == ['supervisor']


def test_stop_does_not_guess_names_or_stop_another_operations_containers(monkeypatch):
    monkeypatch.setattr(launch, 'owned_containers', lambda _operation: ['supervisor'])
    monkeypatch.setattr(launch.subprocess, 'run', lambda *_args, **_kwargs: pytest.fail('no owned workload'))
    launch.stop_simulation(dict(operation_id='mine', container='reused-name', actor='supervisor'))


def test_stop_retains_failure_when_docker_does_not_close_owned_workload(monkeypatch):
    monkeypatch.setattr(launch, 'owned_containers', lambda _operation: ['application', 'supervisor'])
    monkeypatch.setattr(launch.subprocess, 'run', lambda *_args, **_kwargs: SimpleNamespace(returncode=1))
    with pytest.raises(ValueError, match='simulation containers did not stop: application'):
        launch.stop_simulation(dict(operation_id='mine', container='application', actor='supervisor'))


def test_repeated_stop_recovers_an_old_draining_supervisor(monkeypatch, tmp_path):
    monkeypatch.setenv('XDG_CACHE_HOME', str(tmp_path / 'cache'))
    state = dict(operation_id='mine', mode='sim', state='DRAINING', actor='supervisor',
                 container='application', pid=os.getpid(), created=1)
    registry = Registry(tmp_path)
    write_json(registry.operations / 'mine/status.json', state)
    monkeypatch.setattr('graphapi_cli.registry.actor_alive', lambda *_args: True)
    monkeypatch.setattr(launch, 'container_alive', lambda _actor: True)

    def close_workload(row, *_args):
        assert row['operation_id'] == 'mine'
        row.update(state='INTERRUPTED', returncode=130)
        write_json(registry.operations / 'mine/status.json', row)

    monkeypatch.setattr(launch, 'stop_simulation', close_workload)
    monkeypatch.setattr(launch.subprocess, 'run', lambda *_args, **_kwargs: pytest.fail('signal would only repeat the stall'))
    assert launch.stop_operation(tmp_path, 'mine')['state'] == 'INTERRUPTED'


def test_first_stop_unblocks_supervision_and_records_interruption(monkeypatch, tmp_path):
    monkeypatch.setenv('XDG_CACHE_HOME', str(tmp_path / 'cache'))
    models = tmp_path / 'models'
    models.mkdir()
    for name in ('l2_encoder.onnx', 'l2_decoder.onnx'):
        (models / name).touch()
    state = dict(operation_id='mine', mode='sim', state='PREPARING', actor='supervisor',
                 container='application', workspace=str(tmp_path), created=1, recording=False,
                 log=str(tmp_path / 'results/operations/mine/launch.log'), bundles=[],
                 bridge_url='http://127.0.0.1:18000')
    env = dict(WORKSPACE_ROOT=str(tmp_path), SAM_MODEL_DIR=str(models),
               GRAPH_API_CONFIG='test.yaml', GRAPH_API_CONTAINER_NAME='application')
    handlers = {}
    active = ['application', 'supervisor']
    monkeypatch.setattr(launch.signal, 'signal', lambda sig, callback: handlers.setdefault(sig, callback))
    monkeypatch.setattr(launch, 'ensure_image', lambda *_args: None)
    monkeypatch.setattr('graphapi_cli.docker.command', lambda *_args, **_kwargs: ['test launcher'])
    monkeypatch.setattr(launch, 'owned_containers', lambda _operation: list(active))

    class Output:
        def readline(self):
            handlers[signal.SIGINT](signal.SIGINT, None)
            handlers[signal.SIGINT](signal.SIGINT, None)  # Repeated signals are idempotent.
            assert active == ['supervisor']
            return b''

    process = SimpleNamespace(pid=999, stdout=Output(), poll=lambda: 143 if len(active) == 1 else None,
                              wait=lambda: 143)
    monkeypatch.setattr(launch.subprocess, 'Popen', lambda *_args, **_kwargs: process)

    def stop(command, **_kwargs):
        assert command == ['docker', 'stop', '-t', '120', 'application']
        active.remove('application')
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(launch.subprocess, 'run', stop)
    result = launch.execute(tmp_path, {'mode': 'sim'}, env, state)
    assert result['state'] == 'INTERRUPTED'
    assert result['returncode'] == 130
    saved = json.loads((tmp_path / 'results/operations/mine/status.json').read_text())
    assert saved['state'] == 'INTERRUPTED'
