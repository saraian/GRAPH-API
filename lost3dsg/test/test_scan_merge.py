from pathlib import Path
import sys


MODULE_DIR = Path(__file__).resolve().parents[1] / "src" / "perception_module"
sys.path.insert(0, str(MODULE_DIR))

from scan_hook import FullTurnDetector  # noqa: E402


def test_full_turn_emits_once_and_resets():
    detector = FullTurnDetector(full_turn_degrees=360.0)

    assert [detector.update("turn_left", 30.0) for _ in range(11)] == [None] * 11
    assert detector.update("turn_left", 30.0) == 1
    assert detector.degrees == 0.0
    assert detector.update("turn_left", 30.0) is None


def test_full_turn_direction_change_discards_partial_rotation():
    detector = FullTurnDetector(full_turn_degrees=360.0)

    detector.update("turn_left", 180.0)
    assert detector.update("turn_right", 180.0) is None
    assert detector.degrees == 180.0
    assert detector.update("turn_right", 180.0) == 1
