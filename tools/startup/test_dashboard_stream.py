"""A bridge disappearing mid-stream must close the relay without an ASGI error."""
import http.client
import signal
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from graphapi_cli.http_stream import relay_response
from graphapi_cli.auxiliary import supervise
from graphapi_cli.launch import stop_tiago
from graphapi_cli.stack_health import apply_component_settings


class Source:
    def __init__(self, items):
        self.items = iter(items)
        self.closed = False

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.closed = True

    def read1(self, _size):
        item = next(self.items, b'')
        if isinstance(item, Exception):
            raise item
        return item


def test_relay_delivers_frames_and_closes_at_eof():
    source = Source([b'first frame', b'second frame', b''])
    assert list(relay_response(source)) == [b'first frame', b'second frame']
    assert source.closed


@pytest.mark.parametrize('error', [http.client.IncompleteRead(b'partial'),
                                  http.client.RemoteDisconnected(), ConnectionResetError(), TimeoutError()])
def test_upstream_disconnect_during_stream_is_a_clean_end(error):
    source = Source([b'complete frame', error])
    assert list(relay_response(source)) == [b'complete frame']
    assert source.closed


def test_client_disconnect_closes_upstream():
    source = Source([b'frame'])
    stream = relay_response(source)
    assert next(stream) == b'frame'
    stream.close()
    assert source.closed


def test_unexpected_programming_error_is_not_hidden():
    source = Source([ValueError('bug')])
    with pytest.raises(ValueError):
        list(relay_response(source))
    assert source.closed


def components():
    return {key: dict(name=key, active=key in ('feed', 'bridge'), details='Node Stopped')
            for key in ('feed', 'bridge', 'perception', 'object_manager')}


def test_disabled_perception_explains_missing_annotations_and_keeps_sensor_health():
    health = components()
    assert apply_component_settings(health, {'FOUND_START_PERCEPTION': '0', 'TIAGO_BAG_MODE': '1'})
    assert health['feed']['name'] == 'TIAGO Bag Replay'
    for key in ('perception', 'object_manager'):
        assert health[key]['enabled'] is False
        assert 'Disabled' in health[key]['details']
        assert health[key]['active'] is False


def test_enabled_but_stopped_perception_stays_a_failure():
    health = components()
    assert not apply_component_settings(health, {'FOUND_START_PERCEPTION': '1', 'PAL_ROBOT_CONNECTED': '1'})
    assert health['feed']['name'] == 'TIAGO Robot Cameras'
    assert health['perception']['details'] == 'Node Stopped'


@pytest.mark.parametrize('codes,expected', [([0], 0), ([1, 1], 0), ([1, 0, 0], 0), ([1, 0, 1], 1)])
def test_private_shutdown_retries_busy_panes_but_preserves_real_failure(codes, expected, tmp_path):
    results = [SimpleNamespace(returncode=code) for code in codes]
    with patch('graphapi_cli.launch.subprocess.run', side_effect=results) as run:
        assert stop_tiago(tmp_path, {}, {'container': 'private', 'session': 'mine'}) == expected
    assert run.call_count == len(codes)


def test_dashboard_stop_closes_owned_server_without_signalling_parent(monkeypatch, tmp_path):
    handlers = {}
    owned = ['dashboard-child', 'controller-parent']
    calls = []
    monkeypatch.setenv('XDG_CACHE_HOME', str(tmp_path / 'cache'))
    monkeypatch.setattr('graphapi_cli.auxiliary.owned_containers', lambda _op: list(owned))
    monkeypatch.setattr('graphapi_cli.auxiliary.signal.signal', lambda sig, handler: handlers.setdefault(sig, handler))

    class Output:
        def readline(self):
            handlers[signal.SIGINT](signal.SIGINT, None)
            assert owned == ['controller-parent']
            return b''

    process = SimpleNamespace(stdout=Output(), pid=123, wait=lambda: 0, poll=lambda: 0)
    monkeypatch.setattr('graphapi_cli.auxiliary.subprocess.Popen', lambda *_a, **_kw: process)

    def stop(command, **_kwargs):
        calls.append(command)
        if 'dashboard-child' in command:
            owned.remove('dashboard-child')
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr('graphapi_cli.auxiliary.subprocess.run', stop)
    # The parent carries a different operation label in normal launches.
    # Remove it after the handler to model the final operation-only lookup.
    def finished():
        owned.clear()
        return 0
    process.wait = finished
    env = {'WORKSPACE_ROOT': str(tmp_path), 'GRAPHAPI_RESOURCE_ACTOR': 'controller-parent'}
    assert supervise(tmp_path, env, 'dashboard', lambda _env: ['docker', 'run'], need_domain=False) == 130
    assert calls == [['docker', 'stop', '-t', '10', 'dashboard-child']]
