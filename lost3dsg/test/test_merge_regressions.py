"""Execute merge methods without loading ROS, camera or cloud SDK dependencies."""
import ast
import math
import re
from pathlib import Path
from types import SimpleNamespace as NS
import unittest
from unittest.mock import Mock

import numpy as np


SOURCE = Path(__file__).resolve().parents[1] / "src/perception_module/object_services.py"


def load_merge_methods():
    tree = ast.parse(SOURCE.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "ObjectServices")
    methods = [n for n in cls.body if isinstance(n, ast.FunctionDef)
               and ("merge" in n.name or n.name == "_freeze_merge_value")]
    cls.body, cls.bases, cls.decorator_list = methods, [], []
    helpers = [n for n in tree.body if isinstance(n, ast.FunctionDef)
               and n.name in ("_aabb_surface_distance", "_aabb_overlap_evidence", "_merge_base_label")]
    utils = ast.parse(SOURCE.with_name("utils.py").read_text())
    helpers += [n for n in utils.body if isinstance(n, ast.FunctionDef) and n.name == "compute_iou_3d"]
    module = ast.Module(body=helpers + [cls], type_ignores=[])
    env = dict(math=math, re=re, np=np, SimpleNamespace=NS,
               CFG={"association": {"merge_full_sweep_interval": 10}},
               SIM_THRESHOLD=.85, MERGE_MAX_DISTANCE=.8, MERGE_MIN_SIMILARITY=.9,
               MERGE_MAX_SURFACE_GAP=.02, MERGE_NEAR_DISTANCE_M=.02,
               MERGE_NEAR_MAX_VOLUME_RATIO=6, MERGE_NEAR_IOU_THRESHOLD=.25,
               MERGE_NEAR_CONTAINMENT_THRESHOLD=.8, MERGE_MIN_EVIDENCE=1,
               assoc=NS(covisibility_records=lambda obj: []),
               compute_iou_3d=lambda a, b: 0.)
    exec(compile(module, str(SOURCE), "exec"), env)
    return env


def box(x=0.):
    return dict(x_min=x, x_max=x + 1, y_min=0., y_max=1., z_min=0., z_max=1.)


class MergeRegressions(unittest.TestCase):
    def setUp(self):
        self.env = load_merge_methods()
        self.svc = self.env["ObjectServices"]()
        self.svc._merge_request_count = 0
        self.svc._merge_object_signatures = {}
        self.svc._merge_room_signature = None
        self.svc._merge_request_signature = None
        self.svc._merge_force_next_full = False
        self.svc._merge_refusal_cache = {}
        self.svc._merge_candidate_neighbors = {}
        self.svc.log_both = Mock()
        self.svc.decision_log = Mock()
        self.svc.get_logger = Mock()
        self.svc.room_manager = NS(regions={}, scene_graph={}, room_at_bbox=Mock(return_value=None))
        self.objects = [NS(label="chair", bbox=box()), NS(label="chair#2", bbox=box(4))]
        self.env["wm"] = NS(snapshot=lambda: list(self.objects))
        self.req = NS(max_distance=.8, min_similarity=.9, dry_run=False, full_audit=False)

    def prepare(self):
        return self.svc._prepare_merge_sweep(
            self.req, NS(), self.req.max_distance, self.req.min_similarity)

    def finish(self, sweep):
        self.svc._finish_merge_sweep(sweep, [], sweep.object_ids, set(), [])

    def test_unchanged_request_skips_but_changed_limits_reevaluate(self):
        for field, value in (("max_distance", 5.), ("min_similarity", .99)):
            with self.subTest(field=field):
                self.setUp()
                self.finish(self.prepare())
                self.assertIsNone(self.prepare())
                setattr(self.req, field, value)
                sweep = self.prepare()
                self.assertIsNotNone(sweep, "new request limits must invalidate the sweep cache")
                self.assertTrue(sweep.full_audit)
                self.finish(sweep)
                self.assertIsNone(self.prepare())

    def test_invalid_box_does_not_abort_other_pairs(self):
        for bad in ({}, {**box(), "x_min": "invalid"}, {**box(), "x_min": float("nan")},
                    {**box(), "x_min": 2.}):
            with self.subTest(bad=bad):
                self.setUp()
                malformed = NS(label="bad", bbox=bad)
                self.objects.append(malformed)
                sweep = self.prepare()
                refused = Mock()
                self.svc._merge_pair_similarity = Mock(side_effect=AssertionError("not needed"))
                proposals = self.svc._evaluate_merge_candidates(
                    [(malformed, self.objects[0]), tuple(self.objects[:2])], sweep, set(), refused)
                self.assertEqual(proposals, [])
                self.assertEqual([call.args[2] for call in refused.call_args_list],
                                 ["bbox_invalid", "distance"])
                self.assertEqual(self.svc.room_manager.room_at_bbox.call_count, 2)

    def test_nonfinite_thresholds_are_rejected(self):
        for field in ("max_distance", "min_similarity"):
            for value in (float("nan"), float("inf"), -float("inf")):
                with self.subTest(field=field, value=value):
                    req = NS(max_distance=.8, min_similarity=.9)
                    setattr(req, field, value)
                    response = NS()
                    self.assertIsNone(self.svc._validate_merge_request(req, response))
                    self.assertFalse(response.success)

    def test_attribute_drift_does_not_veto_same_class_overlap(self):
        # Reproduces the basket/bed failure shape observed in the 20261109_195427
        # hook log: low semantic score due to viewpoint-dependent material/color.
        for color, material in (("beige", "woven fiber"), ("brown", "woven_fiber")):
            obj = self.objects[0] if color == "beige" else self.objects[1]
            obj.label = "basket"
            obj.bbox = box(0. if color == "beige" else .1)
            obj.color, obj.material, obj.description = color, material, "basket on counter"
        self.env.update(_known=bool, merge_rank=lambda o: id(o),
                        _combine_object_geometry=lambda a, b: (a.bbox, None))
        self.svc._merge_pair_similarity = lambda *args: (.778, {"optional_count": 3})
        refused = Mock()
        sweep = self.prepare()
        proposals = self.svc._evaluate_merge_candidates(
            [tuple(self.objects)], sweep, set(), refused)
        self.assertEqual(len(proposals), 1)
        self.assertTrue(proposals[0]["geometry_override"])
        self.assertTrue(proposals[0]["color_conflict"])
        self.assertTrue(proposals[0]["material_conflict"])
        refused.assert_not_called()

        # Same attributes must not make a separated pair or another class eligible.
        self.objects[1].bbox = box(.9)
        self.assertEqual(self.svc._evaluate_merge_candidates(
            [tuple(self.objects)], self.prepare(), set(), refused), [])
        self.objects[1].bbox = box(.1)
        self.objects[1].label = "sink"
        self.assertEqual(self.svc._evaluate_merge_candidates(
            [tuple(self.objects)], self.prepare(), set(), refused), [])

    def test_disjoint_2d_detections_do_not_veto_eligible_3d_merge(self):
        # One physical object can be segmented into separate regions of an image.
        for obj, x, image_x in zip(self.objects, (0., .1), (0, 200)):
            obj.bbox = box(x)
            obj.color, obj.material, obj.description = "brown", "wood", "chair"
            obj.observations = [NS(capture_id="same-capture", frame_id="same-frame",
                                   bbox_2d=[image_x, 0, image_x + 100, 100])]
            obj.covisibility_observations = obj.observations
        self.env.update(_known=bool, merge_rank=lambda o: id(o),
                        _combine_object_geometry=lambda a, b: (a.bbox, None))
        self.svc._merge_pair_similarity = Mock(return_value=(.95, {"optional_count": 3}))
        refused = Mock()
        proposals = self.svc._evaluate_merge_candidates(
            [tuple(self.objects)], self.prepare(), set(), refused)
        self.assertEqual(len(proposals), 1)
        refused.assert_not_called()

        # Removing the 2D veto must not bypass the remaining evidence/semantic gates.
        self.svc._merge_pair_similarity.return_value = (.95, {"optional_count": 0})
        self.assertEqual(self.svc._evaluate_merge_candidates(
            [tuple(self.objects)], self.prepare(), set(), refused), [])
        self.assertEqual(refused.call_args.args[2], "evidence_absent")
        self.objects[1].label = "table"
        self.svc._merge_pair_similarity.return_value = (.5, {"optional_count": 3})
        self.assertEqual(self.svc._evaluate_merge_candidates(
            [tuple(self.objects)], self.prepare(), set(), refused), [])
        self.assertEqual(refused.call_args.args[2], "similarity")

    def test_final_audit_rechecks_competing_candidates_after_each_merge(self):
        for full_audit, dry_run, expected in ((True, False, 3), (False, False, 1), (True, True, 1)):
            with self.subTest(full_audit=full_audit, dry_run=dry_run):
                self.req.full_audit, self.req.dry_run = full_audit, dry_run
                sweep = NS(objects=[1, 2, 3], dry_run=dry_run, max_distance=.8)
                self.svc._prepare_merge_sweep = Mock(return_value=sweep)
                decisions = NS(accepted=[], pair_iter=[], scope_ids=set(), current_candidate_pairs=set())
                self.svc._evaluate_merge_sweep = Mock(return_value=decisions)
                self.svc._apply_merge_proposals = Mock(side_effect=[
                    [{"merge_record": {"keeper_id": "a"}}],
                    [{"merge_record": {"keeper_id": "b"}}], []])
                self.svc._finish_merge_sweep = Mock()
                response = self.svc._cb_merge_objects(self.req, NS())
                self.assertTrue(response.success, response.message)
                self.assertEqual(self.svc._evaluate_merge_sweep.call_count, expected)
                self.assertEqual(response.merged_count, 2 if expected == 3 else 1)


if __name__ == "__main__":
    unittest.main()
