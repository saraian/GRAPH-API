"""Focused regression coverage for the VLM -> SAM -> overlay path.

These checks stay ROS-free where possible so they can run in a checkout before a
container is started. Source-contract checks cover the process boundaries that
cannot be constructed without a running ROS graph: cycle timestamp propagation
and the installed bridge's image subscriber.
"""

import ast
import json
import math
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

MODULE_DIR = Path(__file__).resolve().parents[1] / "src" / "perception_module"
sys.path.insert(0, str(MODULE_DIR))

from box_view import pca_oriented_box  # noqa: E402
from build_hm3d_eval_manifest import _category_embedding_map, _hovsg_vector  # noqa: E402
from clip_embedder import pixel_crop_rgb  # noqa: E402
from image_transport import ros_image_to_array  # noqa: E402
from live_overlay import _project_box, visible_boxes  # noqa: E402
from scene_analysis import clip_pixel_bbox, parse_scene_analysis  # noqa: E402


PERCEPTION = MODULE_DIR / "perception_2.py"
MODELS = MODULE_DIR / "models.py"
BRIDGE = MODULE_DIR / "graph_api_bridge.py"


def _scene_object(bbox):
    return {
        "label": "chair",
        "description": "red seat",
        "color": "red",
        "material": "fabric",
        "shape": "irregular",
        "bbox": bbox,
    }


class BoundingBoxPathTests(unittest.TestCase):
    def test_normalized_vlm_box_is_pixel_space_and_half_open(self):
        result = parse_scene_analysis(
            json.dumps({"objects": [_scene_object({
                "x_min": 0, "y_min": 0, "x_max": 1000, "y_max": 1000,
            })]}),
            image_width=640,
            image_height=360,
        )
        self.assertEqual(result[0].bbox, (0.0, 0.0, 640.0, 360.0))
        self.assertEqual(clip_pixel_bbox((-20, 10, 700, 400), 640, 360),
                         (0.0, 10.0, 640.0, 360.0))
        self.assertIsNone(clip_pixel_bbox((700, 10, 720, 100), 640, 360))
        self.assertIsNone(clip_pixel_bbox((10, 400, 100, 420), 640, 360))
        self.assertIsNone(clip_pixel_bbox((10, 10, 10, 20), 640, 360))
        self.assertIsNone(clip_pixel_bbox((0, 0, math.nan, 20), 640, 360))

    def test_image_metadata_preserves_padded_rows(self):
        msg = SimpleNamespace(
            height=2, width=2, encoding="bgr8", step=8,
            data=bytes([1, 2, 3, 4, 5, 6, 99, 99,
                        7, 8, 9, 10, 11, 12, 88, 88]),
        )
        pixels, encoding = ros_image_to_array(msg)
        self.assertEqual(encoding, "bgr8")
        np.testing.assert_array_equal(
            pixels,
            np.array([[[1, 2, 3], [4, 5, 6]],
                      [[7, 8, 9], [10, 11, 12]]], dtype=np.uint8),
        )
        msg.data = msg.data[:-1]
        with self.assertRaises(ValueError):
            ros_image_to_array(msg)

    def test_clip_crop_clips_half_open_box_and_converts_bgr_to_rgb(self):
        image = np.zeros((4, 5, 3), dtype=np.uint8)
        image[0, 0] = [10, 20, 30]  # BGR; PIL must expose RGB (30, 20, 10)
        crop = pixel_crop_rgb(image, (-2.0, -1.0, 4.1, 3.2), min_crop_px=2)
        self.assertIsNotNone(crop)
        # The cloud and local crop paths both truncate floating endpoints before
        # NumPy/PIL slicing: clipped y_max=3.2 therefore selects rows [0, 3).
        self.assertEqual(crop.size, (4, 3))
        self.assertEqual(crop.getpixel((0, 0)), (30, 20, 10))
        self.assertIsNone(pixel_crop_rgb(image, (4.0, 3.0, 4.5, 3.5), min_crop_px=2))

    def test_runtime_clip_and_offline_hovsg_channels_do_not_collide(self):
        runtime = [0.0] * 512
        hovsg = [0.0] * 1024
        self.assertIsNone(_hovsg_vector({}, {"clip_embedding": runtime}))
        self.assertEqual(_hovsg_vector({}, {"clip_embedding": hovsg}), hovsg)
        self.assertEqual(_hovsg_vector({"hovsg_embedding": hovsg}, {}), hovsg)
        self.assertEqual(_category_embedding_map({
            "schema": "lost3dsg.runtime_clip_embeddings.v1",
            "detections": {"d-1": {"embedding": runtime}},
        }), {})

    def test_empty_or_malformed_3d_boxes_are_not_projected(self):
        camera = ([0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0],
                  {"fx": 100.0, "fy": 100.0, "cx": 80.0, "cy": 60.0}, 160, 120)
        self.assertIsNone(_project_box({}, *camera))
        self.assertEqual(visible_boxes([{"label": "bad", "bbox": {}}], *camera), [])

    def test_known_box_projects_and_pca_encloses_points(self):
        camera = ([0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0],
                  {"fx": 100.0, "fy": 100.0, "cx": 80.0, "cy": 60.0}, 160, 120)
        box = {"x_min": -0.2, "x_max": 0.2, "y_min": -0.2, "y_max": 0.2,
               "z_min": 2.0, "z_max": 2.5}
        projected = visible_boxes([{"label": "known", "bbox": box}], *camera)
        self.assertEqual(len(projected), 1)
        self.assertEqual(len(projected[0]["points"]), 8)
        self.assertLess(projected[0]["u0"], projected[0]["u1"])
        self.assertLess(projected[0]["v0"], projected[0]["v1"])

        theta = 0.45
        local = np.array([[x, y, z] for x in (-1.0, 1.0)
                          for y in (-0.25, 0.25) for z in (0.0, 0.5)])
        c, s = np.cos(theta), np.sin(theta)
        points = local.copy()
        points[:, :2] = points[:, :2] @ np.array([[c, s], [-s, c]])
        fitted = pca_oriented_box(points)
        self.assertIsNotNone(fitted)
        extents = np.asarray(fitted["oriented_extents"])
        yaw = fitted["yaw"]
        u = points[:, 0] * np.cos(yaw) + points[:, 1] * np.sin(yaw)
        v = -points[:, 0] * np.sin(yaw) + points[:, 1] * np.cos(yaw)
        self.assertLessEqual(np.max(u) - np.min(u), extents[0] + 1e-6)
        self.assertLessEqual(np.max(v) - np.min(v), extents[1] + 1e-6)

    def test_vitsam_encoder_and_decoder_share_prompt_target(self):
        tree = ast.parse(MODELS.read_text(encoding="utf-8"))
        constants = {
            node.targets[0].id: node.value.value
            for node in tree.body
            if isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and isinstance(node.value, ast.Constant)
        }
        self.assertEqual(constants.get("VITSAM_IMAGE_SIZE"), 512)
        self.assertEqual(constants.get("VITSAM_PROMPT_IMAGE_SIZE"), 1024)
        decoder_calls = [
            node for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "SamDecoder"
        ]
        self.assertTrue(decoder_calls)
        for call in decoder_calls:
            target = next((kw.value for kw in call.keywords if kw.arg == "target_size"), None)
            self.assertIsInstance(target, ast.Name)
            self.assertEqual(target.id, "VITSAM_PROMPT_IMAGE_SIZE")
        self.assertIn("img_size=VITSAM_IMAGE_SIZE", MODELS.read_text(encoding="utf-8"))

    def test_cycle_uses_one_stamp_for_overlay_and_graph_messages(self):
        tree = ast.parse(PERCEPTION.read_text(encoding="utf-8"))
        publish = next(
            node for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef) and node.name == "publish_objects"
        )
        calls = [
            node for node in ast.walk(publish)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
        ]
        names = {node.func.attr: node for node in calls}
        for name in ("_publish_image_with_bb", "_publish_bbox_array",
                     "_publish_description_array", "publish_empty_state"):
            self.assertIn(name, names)
            self.assertTrue(any(isinstance(arg, ast.Name) and arg.id == "cycle_stamp"
                                for arg in names[name].args), name)

        bridge_source = BRIDGE.read_text(encoding="utf-8")
        self.assertIn("ros_image_to_array(msg)", bridge_source)
        self.assertIn("step", (MODULE_DIR / "image_transport.py").read_text(encoding="utf-8"))
        self.assertIn("boxes = list(bboxes_3d or [])", PERCEPTION.read_text(encoding="utf-8"))
        self.assertIn("camera_frame=camera_data.get(\"camera_frame\")",
                      PERCEPTION.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
