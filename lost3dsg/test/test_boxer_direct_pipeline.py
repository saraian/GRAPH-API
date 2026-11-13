"""Exercise native Boxer -> GRAPH_API boundaries without ROS or model checkpoints."""
import ast
from concurrent.futures import Future
from dataclasses import dataclass, field
import os
from pathlib import Path
import time
from types import SimpleNamespace as NS
from typing import Any, Dict, List, Optional, Tuple
import unittest
from unittest.mock import Mock, mock_open, patch

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src/perception_module"


def load_pipeline():
    env = dict(np=np, time=time, os=os, SimpleNamespace=NS,
               dataclass=dataclass, field=field, Optional=Optional, List=List,
               Dict=Dict, Any=Any, Tuple=Tuple, ObservationRefData=object,
               CFG={"perception": {"detector": "boxer", "backend": "local"}})
    types = ast.parse((SRC / "detection_types.py").read_text())
    detection = next(n for n in types.body if isinstance(n, ast.ClassDef) and n.name == "Detection")
    pipeline = ast.parse((SRC / "detection_pipeline.py").read_text())
    nodes = [detection] + [n for n in pipeline.body
                           if isinstance(n, (ast.FunctionDef, ast.ClassDef))]
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(SRC / "detection_pipeline.py"), "exec"), env)
    node = ast.parse((SRC / "perception_2.py").read_text())
    cls = next(n for n in node.body if isinstance(n, ast.ClassDef) and n.name == "DetectObjectsNode")
    cls.bases, cls.decorator_list = [ast.Name(id="DetectionPipelineMixin", ctx=ast.Load())], []
    cls.body = [n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name in
                ("_compute_3d_geometry", "_publish_bbox_array")]
    exec(compile(ast.fix_missing_locations(ast.Module(body=[cls], type_ignores=[])),
                 str(SRC / "perception_2.py"), "exec"), env)
    return env


def native_box(x):
    return dict(center=[x, 2., 3.], extents=[2., 4., 6.], yaw=0.,
                aabb_min=[x - 1., 0., 0.], aabb_max=[x + 1., 4., 6.], confidence=.9)


class BoxerDirectPipeline(unittest.TestCase):
    def setUp(self):
        self.env = load_pipeline()
        self.node = self.env["DetectObjectsNode"]()
        self.node.log_both = Mock()
        self.node.get_logger = Mock()
        self.node._abort_if_moving = Mock(return_value=False)
        self.node._refresh_room_geometry_if_available = Mock()
        self.node.clear_accumulated_markers = Mock()
        self.node._attach_local_clip_embeddings = Mock()
        self.node._segment_scene_objects = Mock(side_effect=AssertionError("SAM must not run"))
        self.node._publish_detection_pointclouds = Mock(side_effect=AssertionError("no measured masks"))
        self.node._vlm_executor = NS(submit=self.submit)
        self.semantics = {0: {"label": "chair", "color": "red", "material": "wood"},
                          2: {"label": "table", "description": "round table"}}
        self.records = [dict(proposal_index=i, bbox=[1, 2, 5, 8], label="owl hint",
                             score=.9, box_3d=native_box(x)) for i, x in [(2, 10.), (0, 20.)]]
        self.node.boxer_runtime = NS(infer=self.infer)

    def submit(self, *args):
        future = Future()
        future.set_result(self.semantics)
        return future

    def infer(self, *args, on_boxes_2d):
        on_boxes_2d([dict(proposal_index=i) for i in range(3)])
        return dict(detections=self.records, timing_ms=dict(owl=10., boxernet=20.))

    def detect(self):
        camera = dict(rgb=np.zeros((10, 10, 3), dtype=np.uint8), depth=None,
                      camera_info=None, transform=None)
        with patch("builtins.open", mock_open()), patch("os.makedirs"):
            return self.node.run_detection(camera)

    def test_semantics_follow_proposal_ids_and_geometry_needs_no_masks(self):
        detections = self.detect()
        self.assertEqual([d.label for d in detections], ["table", "chair"])
        self.assertEqual(detections[1].color, "red")
        self.assertTrue(all(d.mask is None for d in detections))
        centers, boxes = self.node._compute_3d_geometry(detections, None, None, None)
        self.assertEqual(centers, [[10., 2., 3.], [20., 2., 3.]])
        self.assertEqual([b["oriented_center"] for b in boxes], centers)
        self.assertTrue(all(d.fusion_voxel_keys is None for d in detections))
        self.node._add_pca_orientation(detections, boxes, None,
                                       NS(k=[1.] * 9), None)
        self.node._segment_scene_objects.assert_not_called()
        self.node._publish_detection_pointclouds.assert_not_called()
        self.assertFalse(self.node.latest_latencies["sam_enabled"])
        self.assertEqual(self.node.latest_latencies["sam_ms"], 0.)

    def test_missing_semantics_does_not_invent_labels_or_drop_geometry(self):
        self.semantics = {}
        detections = self.detect()
        self.assertEqual(len(detections), 2)
        self.assertTrue(all(d.label == "unknown" for d in detections))
        self.assertTrue(all(d.material == "unknown" for d in detections))

    def test_vlm_mode_still_segments_and_publishes_measured_points(self):
        self.env["CFG"]["perception"]["detector"] = "vlm"
        self.node._extract_scene_objects = Mock(return_value=[NS(label="chair")])
        segmented = NS(label="chair", mask=np.ones((10, 10, 1), dtype=np.uint8))
        self.node._segment_scene_objects = Mock(return_value=[segmented])
        self.node._publish_detection_pointclouds = Mock()
        self.assertEqual(self.detect(), [segmented])
        self.node._segment_scene_objects.assert_called_once()
        self.node._publish_detection_pointclouds.assert_called_once()
        self.assertTrue(self.node.latest_latencies["sam_enabled"])

    def test_maskless_crop_uses_original_rgb_box(self):
        env = {}
        exec(compile((SRC / "crop_context.py").read_text(), "crop_context.py", "exec"), env)
        rgb = np.arange(10 * 10 * 3, dtype=np.uint8).reshape(10, 10, 3)
        crop, meta = env["build_context_crop"](
            rgb, None, detector_box=(1, 2, 5, 8), size=0, construction="tight")
        np.testing.assert_array_equal(crop, rgb[2:8, 1:5])
        self.assertFalse(meta["used_mask_bbox"])

    def test_invalid_native_box_does_not_shift_attributes_to_another_object(self):
        for field, value in [("center", [float("nan"), 0., 0.]),
                             ("extents", [1., 0., 1.]), ("aabb_min", [1., 2.])]:
            with self.subTest(field=field):
                self.records[0]["box_3d"] = dict(native_box(10.), **{field: value})
                detections = self.detect()
                self.assertEqual(len(detections), 1)
                self.assertEqual(detections[0].label, "chair")
                self.assertEqual(detections[0].boxer_bbox_3d["center"][0], 20.)

    def test_ros_publication_preserves_native_box_without_voxel_evidence(self):
        detections = self.detect()
        centers, boxes = self.node._compute_3d_geometry(detections, None, None, None)
        # Use the actual ROS schema's field names; no model-only keys can cross it.
        fields = {}
        for line in (ROOT / "msg/Bbox3d.msg").read_text().splitlines():
            if not line.strip() or line.startswith("#"):
                continue
            kind, name = line.split()[:2]
            fields[name] = [] if "[" in kind else False if kind == "bool" else 0
        self.env.update(Bbox3d=lambda: NS(**fields), Bbox3dArray=object,
                        world_frame=lambda: "map", write_observation_msg=Mock())
        self.node.make_header_msg = Mock(return_value=NS(boxes=[]))
        self.node.bbox_pub = Mock()
        for det in detections:
            det.instance_label = det.label
        self.node._publish_bbox_array(detections, boxes, None, None, "cycle-1")
        message = self.node.bbox_pub.publish.call_args.args[0]
        self.assertEqual(len(message.boxes), 2)
        for msg, center, box in zip(message.boxes, centers, boxes):
            self.assertEqual(msg.oriented_center, center)
            self.assertEqual(msg.x_min, box["x_min"])
            self.assertFalse(msg.has_fusion_voxels)
            self.assertTrue(msg.has_orientation)
            self.assertTrue(msg.has_bbox_2d)
            self.assertTrue(msg.has_boxer_confidence)
            self.assertAlmostEqual(msg.boxer_confidence, .9)


if __name__ == "__main__":
    unittest.main()
