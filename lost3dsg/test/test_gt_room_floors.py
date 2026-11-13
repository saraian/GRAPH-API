"""Floor metadata and evaluation regressions; no Habitat runtime required."""
import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src' / 'perception_module'))
from hm3d_ground_truth_manifest import region_floor_height, repair_cached_region_floors
from build_hm3d_eval_manifest import build
from metrics_eval import filtered_scene, floor_regions


def legacy_gt():
    rows = []
    for name, base in [('down', -1.6), ('up', 1.2)]:
        rows.append({'region_id': name, 'floor_index': 0, 'floor_height_m': 0.,
                     'polygon_xz_m': [[0, 0], [4, 0], [4, 4], [0, 4]],
                     'aabb_min_m': [0, base, 0], 'aabb_max_m': [4, base+2.4, 4]})
    return {'scene': 'test', 'geometry_space': {'coordinate_frame': 'Habitat: Y up'},
            'ground_truth_regions': rows, 'ground_truth_objects': [],
            'ground_truth_floors_m': [0.],
            'ground_truth_source': {'selected_floor_index': None, 'all_floor_heights_m': [0.]}}


class FloorTests(unittest.TestCase):
    def test_zero_is_valid_only_near_geometry(self):
        self.assertEqual(region_floor_height(0, -.05), 0.)
        self.assertEqual(region_floor_height(0, -1.6), -1.6)
        self.assertEqual(region_floor_height(0, 1.2), 1.2)
        self.assertEqual(region_floor_height(1.21, 1.15), 1.21)
        for missing in (None, float('nan'), float('inf')):
            self.assertEqual(region_floor_height(missing, 1.2), 1.2)

    def test_cached_repair_is_idempotent_and_preserves_geometry(self):
        gt = legacy_gt()
        before = copy.deepcopy(gt)
        fixed = repair_cached_region_floors(gt)
        self.assertEqual(gt, before)
        self.assertEqual(fixed['ground_truth_floors_m'], [-1.6, 1.2])
        self.assertEqual([r['floor_index'] for r in fixed['ground_truth_regions']], [0, 1])
        for a, b in zip(gt['ground_truth_regions'], fixed['ground_truth_regions']):
            self.assertEqual(a['polygon_xz_m'], b['polygon_xz_m'])
            self.assertEqual(a['aabb_min_m'], b['aabb_min_m'])
        self.assertEqual(repair_cached_region_floors(fixed), fixed)

    def test_ros_coordinates_are_not_reinterpreted_as_habitat(self):
        gt = legacy_gt()
        gt['geometry_space']['coordinate_frame'] = 'ROS: Z up'
        self.assertIs(repair_cached_region_floors(gt), gt)

    def test_selected_floor_cannot_silently_expand_to_two_storeys(self):
        gt = legacy_gt()
        gt['ground_truth_source']['selected_floor_index'] = 0
        with self.assertRaises(ValueError):
            repair_cached_region_floors(gt)

    def test_selected_upper_floor_keeps_local_index(self):
        gt = legacy_gt()
        gt['ground_truth_regions'] = gt['ground_truth_regions'][1:]
        gt['ground_truth_source']['selected_floor_index'] = 1
        fixed = repair_cached_region_floors(gt)
        self.assertEqual(fixed['ground_truth_regions'][0]['floor_index'], 0)
        self.assertEqual(fixed['ground_truth_floors_m'], [1.2])

    def test_build_and_metrics_exclude_stacked_room_on_other_floor(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            # Habitat XZ converts to ROS (-Z,-X).
            room = {'room_id': 'room_0', 'active': True,
                    'polygon': [[0, 0], [0, -4], [-4, -4], [-4, 0]]}
            for name, value in {
                'room.json': {'rooms': [room]},
                'bev_data.json': {'agent': {'z': 1.21}, 'floors': [-1.59, 1.21]},
                'persistent_perception.json': [],
            }.items():
                (root / name).write_text(json.dumps(value))
            manifest = build(legacy_gt(), root)
            self.assertEqual(manifest['adapter_notes']['active_floor_index'], 1)
            selected = filtered_scene(manifest, include_regions=True)
            self.assertEqual([r['region_id'] for r in selected['ground_truth_regions']], ['up'])
            metrics = floor_regions([manifest], .5)
            self.assertEqual(metrics['ground_truth_regions'], 1)
            self.assertEqual(metrics['region_matches'], 1)
            self.assertEqual(metrics['region_recall_pct'], 100.)


if __name__ == '__main__':
    unittest.main()
