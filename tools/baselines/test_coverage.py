import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from tools.baselines.tour_coverage import coverage, in_view
from tools.baselines.validate_cohort import validate


class CoverageTests(unittest.TestCase):
    def test_range_and_vertical_camera_blind_spot(self):
        points = np.array([[0, .02, 1], [0, .02, 2.5], [0, .02, 4]])
        self.assertEqual(in_view([0, 1.5, 0], points, 3, 74, [0]).tolist(), [False, True, False])
        self.assertEqual(in_view([0, 1.5, 0], points, 3, 74, [-45]).tolist(), [True, True, False])

    def test_free_area_includes_space_outside_navmesh_and_walls_occlude(self):
        sim = SimpleNamespace(pathfinder=SimpleNamespace(is_navigable=lambda p: False))
        level = {'height': 0., 'coverage_radius_m': 3., 'trajectory': [
            {'xyz': [.5, 0., .5], 'scan_deg': 360., 'stop': 0}]}
        def ray(wall):
            def run(sim, origin, direction, maximum):
                origin, direction = np.array(origin), np.array(direction)
                hits = []
                if direction[1] < 0:
                    t = -origin[1]/direction[1]
                    if 0 <= t <= maximum:
                        p = origin+t*direction
                        hits.append(SimpleNamespace(point=SimpleNamespace(y=p[1]), normal=SimpleNamespace(y=1.), ray_distance=t))
                if wall and direction[0] > 0:
                    t = (2-origin[0])/direction[0]
                    if 0 <= t <= maximum:
                        hits.append(SimpleNamespace(point=SimpleNamespace(y=origin[1]+t*direction[1]), normal=SimpleNamespace(y=0.), ray_distance=t))
                return hits
            return run
        with tempfile.TemporaryDirectory() as tmp:
            records = []
            for wall in (False, True):
                with patch('tools.baselines.tour_coverage.trace', side_effect=ray(wall)):
                    records.append(coverage(sim, level, ([0, 0, 0], [4, 2, 4]),
                        Path(tmp)/f'{wall}.npz', resolution=1., floor_gt=[([0, -.01, 0], [4, .01, 4])]))
            self.assertEqual(records[0]['free_area_m2'], 16)
            self.assertEqual(records[0]['free_area_outside_navmesh_m2'], 16)
            self.assertGreater(records[0]['covered_area_m2'], records[1]['covered_area_m2'])
            self.assertGreater(records[1]['covered_area_m2'], 0)
            self.assertLess(records[0]['coverage_pct'], 100)
            self.assertEqual(records[0]['per_stop'][0]['cumulative_area_m2'], records[0]['covered_area_m2'])

    def test_incomplete_cohort_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'cohort.json'
            path.write_text('{"scenes": []}')
            with self.assertRaisesRegex(ValueError, 'Incomplete cohort'):
                validate(path)


if __name__ == '__main__':
    unittest.main()
