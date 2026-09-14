import unittest

from tools.baselines.select_scenes import select_floor_ground_truth


def obj(identifier, category, region, low_y, high_y):
    return {'object_id': str(identifier), 'category_name': category, 'region_id': region,
            'aabb_min_m': [0, low_y, 0], 'aabb_max_m': [1, high_y, 1]}


class FloorGroundTruthTests(unittest.TestCase):
    def test_semantic_regions_select_the_requested_floor(self):
        objects = [
            obj(1, 'floor', 'lower', 0.0, 0.0),
            obj(2, 'chair', 'lower', 0.0, 1.0),
            obj(3, 'floor', 'upper', 3.0, 3.0),
            obj(4, 'table', 'upper', 3.0, 4.0),
        ]
        selected, audit = select_floor_ground_truth(objects, [0.0, 3.0], 1)
        self.assertEqual([row['object_id'] for row in selected], ['3', '4'])
        self.assertEqual(audit['selected_region_ids'], ['upper'])
        self.assertEqual(audit['full_scene_native_object_count'], 4)

    def test_regionless_object_uses_explicit_lower_face_fallback(self):
        objects = [
            obj(1, 'floor', 'lower', 0.0, 0.0),
            obj(2, 'floor', 'upper', 3.0, 3.0),
            obj(3, 'lamp', None, 3.1, 4.0),
        ]
        selected, audit = select_floor_ground_truth(objects, [0.0, 3.0], 1)
        self.assertEqual([row['object_id'] for row in selected], ['2', '3'])
        self.assertEqual(audit['fallback_object_ids'], ['3'])


if __name__ == '__main__':
    unittest.main()
