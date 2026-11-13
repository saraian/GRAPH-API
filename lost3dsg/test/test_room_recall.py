"""Room recall regressions, using real OpenCV and only stubbing ROS interfaces.

Run in a separate process: python3 lost3dsg/test/test_room_recall.py
"""
import math
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import Mock, patch

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src' / 'perception_module'))
import rosstub
rosstub.install()
from room_manager import Region, RoomManager


def grid():
    return NS(header=NS(frame_id='map'),
              info=NS(width=100, height=100, resolution=.05,
                      origin=NS(position=NS(x=0., y=0., z=0.),
                                orientation=NS(x=0., y=0., z=0., w=1.))),
              data=np.zeros(10000, dtype=np.int16))


def candidate(x0, x1):
    polygon = [(x0, 0), (x1, 0), (x1, 4), (x0, 4)]
    return polygon, (x1-x0)*4, ((x0+x1)/2, 2), False


class RoomRecallTests(unittest.TestCase):
    def setUp(self):
        self.rm = RoomManager(node=None)
        self.addCleanup(self.rm._room_vlm_executor.shutdown, wait=True)
        self.rm._params.update(gvd_cut_margin_px=1, min_room_area_m2=1.5)

    def map_pipeline(self):
        rm = self.rm
        for method in ('_schedule_window_vlm', '_assign_detected_walls_to_rooms',
                       '_publish_geometry', '_save_rooms', '_update_doorway_room_graph'):
            setattr(rm, method, Mock())
        rm._segment_regions_gvd = Mock(return_value=[candidate(0, 8)])
        rm._update_regions = Mock()
        rm._params.update(room_resegment_local_change_max_cells=2,
                          room_resegment_local_change_max_bbox_fraction=.01)
        return grid()

    def cut(self, free, points):
        return self.rm._cut_free_space(
            free, cv2.distanceTransform(free, cv2.DIST_L2, 5), points, .1)

    def test_two_entrances_are_cut_together(self):
        free = np.zeros((60, 100), np.uint8)
        free[2:58, 2:98] = 255
        free[2:58, 49:52] = 0
        free[12:20, 49:52] = 255
        free[38:46, 49:52] = 255
        points = [(16, 50, 0., 4., .8), (42, 50, 0., 4., .8)]
        cut = self.cut(free, points)
        self.assertEqual(cv2.connectedComponents(cut, 8)[0]-1, 2)
        self.assertEqual(self.rm._last_cut_stats['accepted_collective'], 2)
        self.assertEqual(self.rm._last_cut_stats['rejected_no_split'], 0)
        self.assertTrue(np.all(cut <= free))

    def test_collective_cuts_cannot_create_small_room(self):
        free = np.zeros((60, 100), np.uint8)
        free[2:58, 2:98] = 255
        free[2:58, 6:9] = 0
        free[12:20, 6:9] = 255
        free[38:46, 6:9] = 255
        self.rm._params['min_room_area_m2'] = 3.
        cut = self.cut(free, [(16, 7, 0., 4., .8), (42, 7, 0., 4., .8)])
        self.assertEqual(cv2.connectedComponents(cut, 8)[0]-1, 1)
        self.assertEqual(self.rm._last_cut_stats['accepted_collective'], 0)

    def test_three_entrances_and_duplicate_proposal(self):
        free = np.zeros((80, 100), np.uint8)
        free[2:78, 2:98] = 255
        free[2:78, 49:52] = 0
        points = []
        for y in (16, 42, 66):
            free[y-4:y+4, 49:52] = 255
            points.append((y, 50, 0., 4., .8))
        cut = self.cut(free, points + [points[0]])
        self.assertEqual(cv2.connectedComponents(cut, 8)[0]-1, 2)
        self.assertEqual(self.rm._last_cut_stats['accepted_collective'], 3)

    def test_open_room_does_not_accept_short_strokes(self):
        free = np.zeros((60, 100), np.uint8)
        free[2:58, 2:98] = 255
        # Metric endpoints supply two short strokes that do not reach walls.
        g = grid()
        points = [(16, 50, 0., 4., .8, (2.5, .6), (2.5, 1.0)),
                  (42, 50, 0., 4., .8, (2.5, 1.9), (2.5, 2.3))]
        cut = self.rm._cut_free_space(free, np.ones_like(free), points, .1, grid=g)
        np.testing.assert_array_equal(cut, free)

    def test_incremental_changes_accumulate_and_unchanged_map_is_cached(self):
        g = self.map_pipeline()
        self.rm.process_grid(g)
        self.rm.process_grid(g)
        for i in range(2):
            g.data[i] = 100
            self.rm.process_grid(g)
        self.assertEqual(self.rm._segment_regions_gvd.call_count, 1)
        g.data[2] = 100
        self.rm.process_grid(g)
        self.assertEqual(self.rm._segment_regions_gvd.call_count, 2)

    def test_pending_split_finishes_on_unchanged_maps(self):
        g = self.map_pipeline()
        poly, area, center, _ = candidate(0, 8)
        self.rm.regions = {'old': Region('old', poly, area, center, room_id='old')}
        self.rm._segment_regions_gvd.return_value = [candidate(0, 4), candidate(4, 8)]
        for _ in range(2):
            self.rm.process_grid(g)
        self.rm._update_regions.assert_not_called()
        self.rm.process_grid(g)
        self.rm._update_regions.assert_called_once()
        self.assertEqual(self.rm.last_segmentation_stats['stability']['confirmations'], 3)

    def test_origin_change_is_not_filtered_as_local_noise(self):
        g = self.map_pipeline()
        self.rm.process_grid(g)
        g.info.origin.position.x = 1.
        g.data[0] = 100
        self.rm.process_grid(g)
        g.info.origin.orientation.z = math.sin(.1)
        g.info.origin.orientation.w = math.cos(.1)
        self.rm.process_grid(g)
        self.assertEqual(self.rm._segment_regions_gvd.call_count, 3)

    def test_failed_segmentation_is_retried(self):
        g = self.map_pipeline()
        self.rm._segment_regions_gvd.side_effect = [RuntimeError('transient'), [candidate(0, 8)]]
        with self.assertRaises(RuntimeError):
            self.rm.process_grid(g)
        self.rm.process_grid(g)
        self.assertEqual(self.rm._segment_regions_gvd.call_count, 2)

    def test_nonsegmenting_call_does_not_consume_grid(self):
        g = self.map_pipeline()
        self.rm.process_grid(g, full_resegment=False)
        self.rm.process_grid(g)
        self.rm._segment_regions_gvd.assert_called_once()

    def test_failed_forced_pass_is_retried_on_cached_grid(self):
        g = self.map_pipeline()
        self.rm.process_grid(g)
        self.rm._segment_regions_gvd.side_effect = [RuntimeError('transient'), [candidate(0, 8)]]
        with self.assertRaises(RuntimeError):
            self.rm.process_grid(g, force_resegment=True)
        self.rm.process_grid(g)
        self.assertEqual(self.rm._segment_regions_gvd.call_count, 3)

    def test_wall_update_invalidates_grid_cache(self):
        g = self.map_pipeline()
        self.rm.process_grid(g)
        self.rm._wall_geometry = Mock()
        self.rm._consolidate_detected_walls = Mock()
        self.rm._publish_detected_wall_markers = Mock()
        self.rm.ingest_detected_walls([{'start': [0, 0], 'end': [0, 3]}])
        self.rm.process_grid(g)
        self.assertEqual(self.rm._segment_regions_gvd.call_count, 2)

    def test_cloud_update_invalidates_grid_cache(self):
        g = self.map_pipeline()
        self.rm.process_grid(g)
        msg = NS(header=NS(frame_id='map', stamp=None))
        with patch('room_manager.point_cloud2.read_points', return_value=[(1., 2., 3.)]):
            self.rm._cloud_callback(msg)
        self.rm.process_grid(g)
        self.assertEqual(self.rm._segment_regions_gvd.call_count, 2)

    def test_door_nms_uses_metres_at_multiple_resolutions(self):
        for res in (.05, .1):
            # Branches expose two minima 0.4 m apart, within the 0.8 m NMS radius.
            x0, x1 = int(1/res), int(1.4/res)
            paths = [[(y, x) for y in range(2, 22)] for x in (x0, x1)]
            dist = np.ones((30, 40), np.float32) * (.8/res)
            for path in paths:
                y, x = path[10]
                dist[y, x] = .3/res
            self.rm._params['gvd_door_nms_m'] = .8
            self.rm._trace_branches = Mock(return_value=paths)
            self.rm._door_wall_support_score = Mock(return_value=0.)
            points = self.rm._critical_points(np.ones_like(dist), dist, res)
            self.assertEqual(len(points), 1, res)


if __name__ == '__main__':
    unittest.main()
