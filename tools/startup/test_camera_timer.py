"""Execute the real timer without loading its ROS/model imports."""
import ast
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[2]
tree = ast.parse((ROOT / 'lost3dsg/src/perception_module/perception_2.py').read_text())
method = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == '_perception_timer_callback')
namespace = {}
exec(compile(ast.Module(body=[method], type_ignores=[]), '<real perception timer>', 'exec'), namespace)


@pytest.mark.parametrize('manual', [False, True])
@pytest.mark.parametrize('ready', [False, True])
def test_first_and_manual_detection_receive_the_frame_checked_for_readiness(manual, ready):
    frame = object()
    captured = []

    class Camera:
        available = ready

        def get_synced_data(self, *, consume=True):
            if not self.available:
                return None
            if consume:
                self.available = False
            return frame

    camera = Camera()
    node = SimpleNamespace(log_both=lambda *_: None, manual_trigger_requested=manual,
                           first_detection_done=False, is_stationary=True,
                           time_stationary_start=None, frame_queue=None, camera_data=camera,
                           _run_perception_cycle=lambda *_: captured.append(camera.get_synced_data()))
    namespace['_perception_timer_callback'](node)
    assert captured == ([frame] if ready else [])
    assert node.manual_trigger_requested is (manual and not ready)
