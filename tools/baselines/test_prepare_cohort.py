import unittest

from tools.baselines.prepare_cohort import (next_waypoint_target_maps,
                                            select_medium_object_template,
                                            spread_valid_lifecycles,
                                            distribute_lifecycles_across_laps,
                                            waypoint_lifecycles)


class WaypointLifecycleTests(unittest.TestCase):
    def test_every_stop_gets_one_balanced_action_and_no_object_is_left_live(self):
        scans = [{'stop': stop, 'xyz': [float(stop), 0.0, 0.0]} for stop in range(5)]
        positions = {'left': [0.0, 0.0, 0.0], 'right': [2.0, 0.0, 0.0]}
        actions = waypoint_lifecycles(scans, [['left', 'right']] * len(scans), positions)
        self.assertEqual([action['action'] for action in actions],
                         ['spawn', 'move', 'move', 'move', 'remove'])
        self.assertEqual([action['scan_index'] for action in actions], list(range(5)))
        self.assertEqual(actions[0]['name'], actions[-1]['object'])
        self.assertEqual(sum(action['action'] == 'spawn' for action in actions), 1)
        self.assertEqual(sum(action['action'] == 'move' for action in actions), 3)
        self.assertEqual(sum(action['action'] == 'remove' for action in actions), 1)

    def test_rejects_a_waypoint_without_a_valid_local_support(self):
        scans = [{'stop': stop, 'xyz': [float(stop), 0.0, 0.0]} for stop in range(3)]
        positions = {'left': [0.0, 0.0, 0.0], 'right': [2.0, 0.0, 0.0]}
        with self.assertRaisesRegex(ValueError, 'scan stop 1'):
            waypoint_lifecycles(scans, [['left'], [], ['right']], positions)

    def test_final_object_persists(self):
        scans = [{'stop': stop, 'xyz': [float(stop), 0.0, 0.0]} for stop in range(6)]
        positions = {'left': [0.0, 0.0, 0.0], 'right': [2.0, 0.0, 0.0]}
        actions = waypoint_lifecycles(scans, [['left', 'right']] * len(scans), positions)
        self.assertEqual([action['action'] for action in actions],
                         ['spawn', 'move', 'remove', 'spawn', 'move', 'move'])

    def test_spawn_and_move_targets_are_visible_from_next_waypoint(self):
        scans = [{'stop': stop, 'xyz': [float(stop), 0.0, 0.0]}
                 for stop in range(3)]
        positions = {'at_zero': [0.0, 0.0, 0.0],
                     'at_one': [1.0, 0.0, 0.0],
                     'at_two': [2.0, 0.0, 0.0]}
        visible = {(0, 'at_zero'), (1, 'at_one'), (2, 'at_two')}
        placement, removal = next_waypoint_target_maps(
            scans, positions, lambda index, key: (index, key) in visible)
        self.assertEqual(placement[0], ['at_one'])
        self.assertEqual(placement[1], ['at_two'])
        self.assertEqual(placement[2], [])
        self.assertEqual(removal[0], ['at_zero'])
        self.assertEqual(removal[1], ['at_one'])
        self.assertEqual(removal[2], ['at_two'])

    def test_next_waypoint_target_must_also_be_within_trigger_range(self):
        scans = [{'stop': 0, 'xyz': [0.0, 0.0, 0.0]},
                 {'stop': 1, 'xyz': [10.0, 0.0, 0.0]}]
        positions = {'next': [10.0, 0.0, 0.0]}
        placement, _ = next_waypoint_target_maps(
            scans, positions, lambda index, key: index == 1,
            maximum_distance=4.0)
        self.assertEqual(placement[0], [])

    def test_medium_template_is_selected_from_measured_scaled_extent(self):
        class Size:
            x, y, z = .1, .2, .4
        class Aabb:
            def size(self):
                return Size()
        class Object:
            object_id = 7
            aabb = Aabb()
        class Manager:
            def __init__(self):
                self.removed = []
            def add_object_by_template_handle(self, handle):
                self.handle = handle
                return Object()
            def remove_object_by_id(self, object_id):
                self.removed.append(object_id)
        manager = Manager()
        class Sim:
            def get_rigid_object_manager(self):
                return manager
        class Compiler:
            @staticmethod
            def available_template_names(_):
                return ['061_foam_brick']
            @staticmethod
            def template_handle_and_support_offset(_, template, __):
                return template + '.handle', 0
            @staticmethod
            def apply_global_object_scale(_):
                return None
        template, extent = select_medium_object_template(Sim(), Compiler(), '/objects')
        self.assertEqual((template, extent), ('061_foam_brick', .4))
        self.assertEqual(manager.removed, [7])

    def test_lifecycles_are_compatible_and_spread_before_selection(self):
        scans = [{'stop': i, 'xyz': [float(i), 0, 0]} for i in range(18)]
        positions = {'a': [0, 0, 0], 'b': [2, 0, 0]}
        targets = {i: (['a'] if i % 3 == 0 else ['b']) for i in range(18)}
        actions = spread_valid_lifecycles(scans, targets, positions, 'medium', windows=3)
        self.assertEqual([a['action'] for a in actions],
                         ['spawn', 'move', 'remove', 'spawn', 'move', 'remove',
                          'spawn', 'move'])
        self.assertLess(actions[2]['scan_index'], actions[3]['scan_index'])
        self.assertEqual(actions[-1]['action'], 'move')

    def test_sparse_cross_window_pair_falls_back_and_keeps_deletion(self):
        scans = [{'stop': i, 'xyz': [float(i), 0, 0]} for i in range(9)]
        positions = {'a': [0, 0, 0], 'b': [2, 0, 0]}
        targets = {i: [] for i in range(9)}
        targets[1] = ['a']
        targets[7] = ['b']
        targets[8] = ['b']
        actions = spread_valid_lifecycles(scans, targets, positions, 'medium', windows=3)
        self.assertEqual([a['action'] for a in actions], ['spawn', 'move', 'remove'])
        self.assertEqual([a['scan_index'] for a in actions], [1, 7, 8])

    def test_remove_uses_a_later_scan_that_is_near_the_moved_object(self):
        scans = [{'stop': i, 'xyz': [float(i), 0, 0]} for i in range(6)]
        positions = {'a': [0, 0, 0], 'b': [2, 0, 0]}
        placement_targets = {i: [] for i in range(6)}
        placement_targets[0] = ['a']
        placement_targets[1] = ['b']
        removal_targets = {i: [] for i in range(6)}
        removal_targets[3] = ['b']
        actions = spread_valid_lifecycles(
            scans, placement_targets, positions, 'medium', windows=1,
            removal_targets_by_scan=removal_targets,
        )
        self.assertEqual([a['action'] for a in actions], ['spawn', 'move', 'remove'])
        self.assertEqual([a['scan_index'] for a in actions], [0, 1, 3])

    def test_lifecycle_is_refused_without_a_local_later_removal(self):
        scans = [{'stop': i, 'xyz': [float(i), 0, 0]} for i in range(4)]
        positions = {'a': [0, 0, 0], 'b': [2, 0, 0]}
        placement_targets = {0: ['a'], 1: ['b'], 2: [], 3: []}
        removal_targets = {i: [] for i in range(4)}
        with self.assertRaisesRegex(ValueError, 'No action-compatible lifecycle'):
            spread_valid_lifecycles(
                scans, placement_targets, positions, 'medium', windows=1,
                removal_targets_by_scan=removal_targets,
            )

    def test_lifecycles_are_spread_across_three_complete_laps(self):
        actions = []
        for object_index in range(1, 6):
            name = f'dynamic_object_{object_index:03d}'
            base = (object_index - 1) * 3
            actions.extend([
                {'action': 'spawn', 'name': name, 'scan_index': base},
                {'action': 'move', 'object': name, 'scan_index': base + 1},
            ])
            if object_index < 5:
                actions.append({'action': 'remove', 'object': name,
                                'scan_index': base + 2})
        distributed = distribute_lifecycles_across_laps(actions, 3)
        self.assertEqual({action['lap'] for action in distributed}, {0, 1, 2})
        self.assertEqual(
            [(action.get('name') or action.get('object'), action['lap'])
             for action in distributed if action['action'] == 'spawn'],
            [('dynamic_object_001', 0), ('dynamic_object_004', 0),
             ('dynamic_object_002', 1), ('dynamic_object_003', 2),
             ('dynamic_object_005', 2)],
        )
        self.assertEqual(distributed[-1]['object'], 'dynamic_object_005')
        self.assertEqual(distributed[-1]['action'], 'move')

    def test_lap_distribution_refuses_too_few_lifecycles(self):
        actions = [
            {'action': 'spawn', 'name': 'dynamic_object_001', 'scan_index': 0},
            {'action': 'move', 'object': 'dynamic_object_001', 'scan_index': 1},
            {'action': 'remove', 'object': 'dynamic_object_001', 'scan_index': 2},
        ]
        with self.assertRaisesRegex(ValueError, 'across 2 laps'):
            distribute_lifecycles_across_laps(actions, 2)


if __name__ == '__main__':
    unittest.main()
