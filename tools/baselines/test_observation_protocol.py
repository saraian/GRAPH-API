import json
import tempfile
import unittest
from pathlib import Path

from tools.baselines.observation_protocol import (
    prepare_shared_recording,
    scan_windows,
    select_shared_frames,
    validate_recording_visibility,
    validate_script_contract,
)
from tools.baselines.preflight_action_visibility import sampled_heading_indices


def scan_frame(index, position, semantic_ids=(), event=None):
    return {'index': index, 'stem': f'{index:06d}', 'time_s': float(index + 1),
            'reason': 'tour', 'event': [event] if event else None,
            'base_position': list(position), 'visible_semantic_ids': list(semantic_ids),
            'dynamic_ground_truth': []}


class ObservationProtocolTests(unittest.TestCase):
    def test_preflight_uses_same_eight_evenly_spaced_scan_headings(self):
        self.assertEqual(sampled_heading_indices(36, 8), [0, 5, 10, 15, 20, 25, 30, 35])

    def fixture(self, root):
        recording = root / 'recording'
        for folder in ('rgb', 'depth', 'depth_m', 'pose', 'semantic'):
            (recording / folder).mkdir(parents=True)
        event = lambda lap, stop, position: {
            'event': 'scan_complete', 'lap': lap, 'stop': stop,
            'xyz': list(position), 'observed_from': list(position)}
        frames = [
            scan_frame(0, [0, 0, 0]),
            scan_frame(1, [0, 0, 0], event=event(0, 0, [0, 0, 0])),
            scan_frame(2, [1, 0, 0], [1001]),
            scan_frame(3, [1, 0, 0], [1001], event(0, 1, [1, 0, 0])),
            scan_frame(4, [0, 0, 0]),
            scan_frame(5, [0, 0, 0], event=event(1, 0, [0, 0, 0])),
            scan_frame(6, [1, 0, 0]),
            scan_frame(7, [1, 0, 0], event=event(1, 1, [1, 0, 0])),
        ]
        (recording / 'frames.jsonl').write_text(
            ''.join(json.dumps(row) + '\n' for row in frames))
        (recording / 'acquisition.json').write_text(json.dumps({
            'complete': True, 'frames': len(frames), 'ground_truth_recorded': True}))
        for row in frames:
            for folder, suffix in (('rgb', '.png'), ('depth', '.png'),
                                   ('depth_m', '.npz'), ('pose', '.txt'),
                                   ('semantic', '.npz')):
                (recording / folder / (row['stem'] + suffix)).write_bytes(folder.encode())
        script = {'steps': [
            {'action': 'spawn', 'name': 'object', 'at_waypoint': {'stop': 0, 'lap': 0},
             'expected_observation': {'stop': 1, 'lap': 0, 'state': 'present',
                                      'timing': 'next_waypoint_scan'}},
            {'action': 'remove', 'object': 'object',
             'at_waypoint': {'stop': 1, 'lap': 0},
             'expected_observation': {'stop': 1, 'lap': 1, 'state': 'absent',
                                      'timing': 'same_waypoint_following_lap'}},
        ]}
        script_path = root / 'script.json'
        script_path.write_text(json.dumps(script))
        actions = [
            {'action': 'spawn', 'after_frame': 1,
             'result': {'success': True, 'object_id': 1,
                        'evaluation_object_id': 'dynamic-000001',
                        'gt_semantic_id': 1001}},
            {'action': 'remove', 'after_frame': 3,
             'result': {'success': True, 'object_id': 1,
                        'evaluation_object_id': 'dynamic-000001'}},
        ]
        (recording / 'object_actions.jsonl').write_text(
            ''.join(json.dumps(row) + '\n' for row in actions))
        schedule = {'trajectory': [
            {'stop': 0, 'scan_deg': 360, 'xyz': [0, 0, 0]},
            {'stop': 1, 'scan_deg': 360, 'xyz': [1, 0, 0]},
        ]}
        return recording, script_path, script, schedule, frames

    def test_next_waypoint_and_following_lap_contract_passes(self):
        with tempfile.TemporaryDirectory() as directory:
            recording, script_path, script, schedule, _ = self.fixture(Path(directory))
            plan = validate_script_contract(script, schedule, 2)
            gate = validate_recording_visibility(recording, script_path)
            self.assertTrue(plan['complete'])
            self.assertTrue(gate['complete'])
            self.assertEqual(gate['actions'][0]['visible_frame_indices'], [2, 3])
            self.assertEqual(gate['actions'][1]['visible_frame_indices'], [])

    def test_shared_selection_is_identical_when_gt_visibility_changes(self):
        with tempfile.TemporaryDirectory() as directory:
            _, _, script, _, frames = self.fixture(Path(directory))
            first = select_shared_frames(frames, script, stride=5, scan_samples=2)
            for row in frames:
                row['visible_semantic_ids'] = [999999]
            second = select_shared_frames(frames, script, stride=5, scan_samples=2)
            self.assertEqual(first, second)

    def test_shared_input_contains_no_gt_and_preserves_source_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            recording, script_path, _, _, _ = self.fixture(root)
            output, indices, receipt = prepare_shared_recording(
                recording, script_path, root / 'shared', stride=5, scan_samples=2)
            rows = [json.loads(line) for line in
                    (output / 'frames.jsonl').read_text().splitlines()]
            self.assertEqual([row['source_frame_index'] for row in rows], indices)
            self.assertFalse(any('visible_semantic_ids' in row or
                                 'dynamic_ground_truth' in row for row in rows))
            self.assertFalse((output / 'semantic').exists())
            self.assertFalse(json.loads((output / 'acquisition.json').read_text())[
                'ground_truth_recorded'])
            self.assertFalse(receipt['ground_truth_in_baseline_input'])

    def measured_scan(self, executed, scheduled, first=0, length=8, lap=0, stop=0):
        """One 360-degree scan as the runtime actually records it.

        The robot stands where the navmesh put it, which is not the scheduled waypoint,
        and only the last frame of the stationary run carries the completion event.
        """
        event = {'event': 'scan_complete', 'lap': lap, 'stop': stop,
                 'xyz': list(scheduled), 'observed_from': list(scheduled)}
        return [scan_frame(first + offset, executed,
                           event=event if offset == length - 1 else None)
                for offset in range(length)]

    def test_window_follows_the_executed_pose_not_the_scheduled_waypoint(self):
        # Measured on the live 00813 acquisition: the executed pose stood 0.2974 m from
        # the scheduled waypoint, so anchoring on observed_from left every window with
        # the single carrier frame and the gate refused six of eight actions.
        frames = self.measured_scan(executed=[0.30, 0, 0], scheduled=[0, 0, 0], length=8)
        self.assertEqual(scan_windows(frames)[(0, 0)], list(range(8)))

    def test_stale_action_copy_of_a_scan_event_is_not_a_second_scan(self):
        # The runtime attaches the trigger scan's event to the frame it captures right
        # after an object action. Counting that copy raised on the live recording with
        # "Duplicate completed scan event (0, 0)" before any native baseline ran.
        frames = self.measured_scan(executed=[0.30, 0, 0], scheduled=[0, 0, 0], length=8)
        carried = dict(frames[-1])
        carried.update(index=8, stem='000008', reason='object:spawn',
                       event=dict(frames[-1]['event'][0]))
        frames.append(carried)
        self.assertEqual(scan_windows(frames)[(0, 0)], list(range(8)))

    def test_shared_input_holds_real_files_a_container_can_open(self):
        # The native baselines mount the shared input tree and nothing else. Symlinks
        # carrying the source recording's absolute host path dangled inside the
        # containers and both baselines died on frame 000000 with FileNotFoundError.
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            recording, script_path, _, _, _ = self.fixture(root)
            output, _, _ = prepare_shared_recording(
                recording, script_path, root / 'shared', stride=5, scan_samples=2)
            linked = [path for folder in ('rgb', 'depth', 'depth_m', 'pose')
                      for path in (output / folder).iterdir()]
            self.assertTrue(linked)
            self.assertEqual([p for p in linked if p.is_symlink()], [])
            for path in linked:
                self.assertEqual(path.read_bytes(), path.parent.name.encode())

    def test_old_same_waypoint_contract_is_rejected_before_acquisition(self):
        with tempfile.TemporaryDirectory() as directory:
            _, _, script, schedule, _ = self.fixture(Path(directory))
            script['steps'][0]['expected_observation']['stop'] = 0
            with self.assertRaisesRegex(ValueError, 'expected'):
                validate_script_contract(script, schedule, 2)


if __name__ == '__main__':
    unittest.main()
