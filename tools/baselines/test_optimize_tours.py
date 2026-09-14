import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from tools.baselines.optimize_tours import greedy_cover, insert_points


class OptimizeTests(unittest.TestCase):
    def test_coverage_union_does_not_hide_inaccessible_cells(self):
        views = np.array([[1, 1, 0, 0, 0], [0, 1, 1, 0, 0], [0, 0, 0, 1, 0]], dtype=bool)
        selected, covered, ceiling = greedy_cover(views, np.zeros(5, dtype=bool), 1.)
        self.assertEqual(covered.tolist(), [True, True, True, True, False])
        self.assertEqual(ceiling.tolist(), covered.tolist())
        self.assertEqual(len(selected), 3)
        self.assertEqual(covered.mean(), .8)

    def test_tilt_alone_needs_no_extra_stops(self):
        selected, covered, ceiling = greedy_cover(np.array([[1, 0]], dtype=bool), np.array([True, True]))
        self.assertEqual(selected, [])
        self.assertTrue(covered.all())
        self.assertTrue(ceiling.all())

    def test_insertion_preserves_original_order_and_transfer_endpoints(self):
        points = [{'xyz': [x, 0, 0], 'stop': i} for i, x in enumerate([0, 10, 20])]
        extra = {'xyz': [4, 0, 1], 'stop': 3, 'coverage_added': True}
        def distance(pf, a, b):
            return SimpleNamespace(geodesic_distance=float(np.linalg.norm(np.array(a)-b)))
        with patch('tools.baselines.optimize_tours.geodesic', side_effect=distance):
            result = insert_points(None, points, [extra])
        self.assertEqual([p for p in result if not p.get('coverage_added')], points)
        self.assertEqual([p['stop'] for p in result], [0, 3, 1, 2])
        self.assertEqual(points[0]['xyz'], [0, 0, 0])
        with patch('tools.baselines.optimize_tours.geodesic', return_value=None):
            with self.assertRaisesRegex(ValueError, 'No connected insertion'):
                insert_points(None, points, [extra])


if __name__ == '__main__':
    unittest.main()
