"""General topology regressions, with real OpenCV and synthetic metric geometry."""
import sys
import unittest
from pathlib import Path
import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tools.analysis.probe_room_segmentation import manager, two_rooms, transformed, grid


class TopologyRepairs(unittest.TestCase):
    def setUp(self):
        self.rm = manager()
        self.addCleanup(self.rm._room_vlm_executor.shutdown)

    def test_small_residue_does_not_veto_two_rooms(self):
        for res in (.025, .05, .1):
            for turns in (0, 1, 2, 3):
                with self.subTest(res=res, turns=turns):
                    a,p = two_rooms(res, noise=True)
                    a,p = transformed(a,p,turns)
                    out = self.rm._cut_free_space(a, cv2.distanceTransform(a,cv2.DIST_L2,5), [p],res)
                    self.assertEqual(cv2.connectedComponents(out,8)[0]-1,2)
                    self.assertGreater(self.rm._last_cut_stats['discarded_noise_area_m2'],0)
                    self.assertTrue(np.all(out <= a))

    def test_noise_budget_requires_two_large_regions_and_total_limit(self):
        # Pixel areas at 10 cm resolution: threshold 150 px = 1.5 m².
        for areas in ([1200, 5], [1200, 1200, 11], [1200,1200]+[1]*11, [150,150,5]):
            with self.subTest(areas=areas):
                self.assertFalse(self.rm._cut_partition_valid(areas,150,.1))
        self.assertTrue(self.rm._cut_partition_valid([1200,1200,5],150,.1))
        self.assertTrue(self.rm._cut_partition_valid([150,150],150,.1))
        self.rm._params['room_cut_noise_max_area_m2']=0
        self.assertFalse(self.rm._cut_partition_valid([1200,1200,1],150,.1))

    def test_gap_recovery_through_full_pipeline(self):
        for res in (.025,.05,.1):
            for turns in (0,1,2,3):
                with self.subTest(res=res,turns=turns):
                    a,p=two_rooms(res,gap=.2);a,p=transformed(a,p,turns)
                    candidates=self.rm._segment_regions_gvd(grid(a,res))
                    self.assertEqual(len(candidates),2)
                    self.assertGreater(self.rm.last_segmentation_stats['wall_gap_repair']['repaired_cells'],0)

    def test_real_doorways_stay_open(self):
        for res in (.025,.05,.1):
            for gap in (.55,.8,1.2,2.):
                with self.subTest(res=res,gap=gap):
                    a,_=two_rooms(res,gap=gap);g=grid(a,res)
                    _,occ=self.rm._binary_free_obstacle(g)
                    structural=self.rm._structural_obstacles(occ,res,cloud_support=None)
                    repaired,added=self.rm._repair_wall_gaps(occ,structural,g)
                    y=int(round(.5/res))+int(round(gap/res))//2
                    x=int(round(4.5/res))
                    self.assertEqual(repaired[y,x],0)
                    self.assertFalse(added[y,x])

    def test_open_space_stays_one_room(self):
        a,_=two_rooms(.05)
        a[4:-4,4:-4]=255
        self.assertEqual(len(self.rm._segment_regions_gvd(grid(a,.05))),1)

    def test_unknown_gap_is_not_invented_wall(self):
        a,_=two_rooms(.05,gap=.2);g=grid(a,.05)
        raw=g.data.reshape(a.shape)
        raw[10:14,84:96]=-1
        _,occ=self.rm._binary_free_obstacle(g)
        repaired,added=self.rm._repair_wall_gaps(occ,occ,g)
        self.assertFalse(added[10:14,84:96].any())

    def test_confirmed_opening_is_protected(self):
        a,_=two_rooms(.05,gap=.2);g=grid(a,.05)
        _,occ=self.rm._binary_free_obstacle(g)
        self.rm._confirmed_door_cuts={'door':{'world':[4.5,.6]}}
        _,added=self.rm._repair_wall_gaps(occ,occ,g)
        self.assertFalse(added[10:14,84:96].any())

    def test_isolated_furniture_is_not_connected(self):
        occ=np.zeros((80,80),np.uint8)
        occ[20:30,30:36]=255;occ[34:44,30:36]=255
        g=grid(np.where(occ>0,0,255).astype(np.uint8),.05)
        _,added=self.rm._repair_wall_gaps(occ,np.zeros_like(occ),g)
        self.assertFalse(added.any())

    def test_diagonal_collinear_wall_gap(self):
        occ=np.zeros((60,60),np.uint8)
        for i in list(range(5,25))+list(range(29,50)):occ[i,i]=255
        g=grid(np.where(occ>0,0,255).astype(np.uint8),.025)
        repaired,added=self.rm._repair_wall_gaps(occ,occ,g)
        self.assertTrue(all(repaired[i,i] for i in range(25,29)))
        self.assertEqual(np.count_nonzero(added),4)

    def test_gap_repair_can_be_disabled(self):
        a,_=two_rooms(.05,gap=.2);g=grid(a,.05)
        _,occ=self.rm._binary_free_obstacle(g)
        self.rm._params['gvd_wall_gap_max_m']=0
        repaired,added=self.rm._repair_wall_gaps(occ,occ,g)
        np.testing.assert_array_equal(repaired,occ)
        self.assertFalse(added.any())


if __name__=='__main__':unittest.main()
