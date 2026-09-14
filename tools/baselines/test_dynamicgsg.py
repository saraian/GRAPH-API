"""Tests for the frozen-source DynamicGSG integration boundary."""
import gzip
import hashlib
import json
import pickle
import subprocess
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from tools.baselines.dynamicgsg_export import export
from tools.baselines.dynamicgsg_run import (
    MODEL_FIELDS,
    NATIVE_RESOURCE_FIELDS,
    _native_resource,
    _source_identity,
    export_graph,
    prepare_config,
)
from tools.baselines.replay_model import dynamicgsg_graph
from tools.baselines.replay_scene import export_scene


class DynamicGSGTests(unittest.TestCase):
    def make_recording(self, root: Path, count=3):
        recording = root / "recording"
        for name in ("rgb", "depth", "pose"):
            (recording / name).mkdir(parents=True, exist_ok=True)
        rows = []
        for index in range(count):
            stem = f"{index:06d}"
            Image.fromarray(np.full((3, 4, 3), index * 20, dtype=np.uint8)).save(recording / "rgb" / f"{stem}.png")
            Image.fromarray(np.full((3, 4), 1000 + index, dtype=np.uint16)).save(recording / "depth" / f"{stem}.png")
            pose = np.eye(4)
            pose[:3, 3] = [10 + index, 1, -3]
            np.savetxt(recording / "pose" / f"{stem}.txt", pose.reshape(1, -1))
            rows.append({"index": index, "stem": stem, "time_s": index + 0.5,
                         "width": 4, "height": 3, "hfov_deg": 90})
        (recording / "frames.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
        (recording / "acquisition.json").write_text(json.dumps({
            "complete": True, "frames": count, "depth_png_units": "millimetres",
            "pose_convention": "Habitat/OpenGL camera-to-world",
        }))
        return recording

    def test_lossless_native_export_and_generated_config(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            recording = self.make_recording(root)
            dataset = root / "dataset"
            manifest = export(recording, dataset, max_frames=2)
            native_rgb = dataset / "scene/results/frame000000.jpg"
            self.assertEqual(native_rgb.stat().st_ino, (recording / "rgb/000000.png").stat().st_ino)
            with Image.open(native_rgb) as image:
                self.assertEqual(image.format, "PNG")
            self.assertEqual(manifest["source_frames_total"], 3)
            self.assertEqual(manifest["exported_frames"], 2)
            self.assertEqual(len((dataset / "scene/traj.txt").read_text().splitlines()), 2)
            first_native_pose = np.loadtxt(dataset / "scene/traj.txt", max_rows=1).reshape(4, 4)
            np.testing.assert_allclose(first_native_pose, np.array([
                [1, 0, 0, 10], [0, -1, 0, 1], [0, 0, -1, -3], [0, 0, 0, 1],
            ]))
            self.assertEqual(manifest["source_pose_convention"],
                             "Habitat/OpenGL camera-to-world")
            self.assertIn("OpenCV camera", manifest["native_pose_convention"])

            native = root / "native"
            profile = native / "configs/found/profile.py"
            profile.parent.mkdir(parents=True)
            resources = {}
            for index, field in enumerate(NATIVE_RESOURCE_FIELDS):
                resource = native / f"resource-{index}.txt"
                resource.write_text(field)
                resources[field] = str(resource.relative_to(native))
            profile.write_text("config={'data': {'stride': 4}, 'lang': " + repr(resources) + ", 'viz': {}, "
                               "'tracking': {}, 'mapping': {}, 'whether_to_update': True}\n")
            models = root / "models"
            models.mkdir()
            for filename in MODEL_FIELDS.values():
                (models / filename).write_bytes(b"model")
            output = root / "result"
            output.mkdir()
            config_path, config = prepare_config(
                native, dataset, output, models, profile,
                dynamic_start_frame=2, variant="upstream-execfix",
            )
            self.assertEqual(config["data"]["stride"], 4)
            self.assertEqual(config["data"]["desired_image_width"], 4)
            self.assertFalse(config["lang"]["use_dam"])
            self.assertFalse(config["live_viewer"])
            self.assertTrue(all(Path(config["lang"][field]).is_absolute()
                                for field in NATIVE_RESOURCE_FIELDS))
            self.assertIn("png_depth_scale: 1000.0", (output / "data_config.yaml").read_text())
            self.assertTrue(config_path.is_file())
            provenance = json.loads((output / "config_provenance.json").read_text())
            by_key = {row["key"]: row["source"] for row in provenance["rows"]}
            self.assertEqual(by_key["data.sequence"], "dataset_adapter")
            self.assertEqual(by_key["data.stride"], "upstream_profile")
            self.assertEqual(by_key["data.frame_begin_update"], "dataset_scheduler")
            self.assertEqual(provenance["variant"], "upstream-execfix")
            self.assertEqual(provenance["experimental_rows"], [])

            legacy_manifest = json.loads((dataset / "export_manifest.json").read_text())
            legacy_manifest["schema"] = "graphapi.dynamicgsg_input.v1"
            (dataset / "export_manifest.json").write_text(json.dumps(legacy_manifest))
            with self.assertRaises(ValueError):
                prepare_config(native, dataset, output, models, profile)

    def test_source_patch_requires_exact_hash(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            subprocess.run(["git", "init", "-q", str(root)], check=True)
            source = root / "source.py"
            source.write_text("answer = 1\n")
            subprocess.run(["git", "-C", str(root), "add", "source.py"], check=True)
            subprocess.run([
                "git", "-C", str(root), "-c", "user.name=test",
                "-c", "user.email=test@example.invalid", "commit", "-qm", "reference",
            ], check=True)
            self.assertIsNone(_source_identity(root)["patch_sha256"])
            source.write_text("answer = 2\n")
            with self.assertRaises(RuntimeError):
                _source_identity(root)
            patch = subprocess.run(
                ["git", "-C", str(root), "diff", "--binary", "HEAD", "--"],
                check=True, capture_output=True,
            ).stdout
            expected = hashlib.sha256(patch).hexdigest()
            identity = _source_identity(root, expected)
            self.assertEqual(identity["patch_sha256"], expected)
            with self.assertRaises(RuntimeError):
                _source_identity(root, "0" * 64)

    def test_official_grounding_config_relocation_is_path_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            expected = (root / "submodules/GroundingDINO/groundingdino/config/"
                        "GroundingDINO_SwinT_OGC.py")
            expected.parent.mkdir(parents=True)
            expected.write_text("# pinned upstream config\n")
            resolved, relocated = _native_resource(
                root, "grounding_dino_config_path",
                "./dgsg/navigation/groundingdino/config/GroundingDINO_SwinT_OGC.py",
            )
            self.assertTrue(relocated)
            self.assertEqual(resolved, expected.resolve())

    def test_native_gaussians_become_habitat_graph_and_scene(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            recording = self.make_recording(root, count=1)
            dataset = root / "dataset"
            export(recording, dataset)
            result = root / "result"
            result.mkdir()
            np.savez(result / "params_with_idx.npz",
                     means3D=np.array([[0, 0, 0], [2, 2, 2], [9, 9, 9]], dtype=np.float32),
                     object_idx=np.array([[7], [7], [0]], dtype=np.int32))
            with gzip.open(result / "objects.pkl.gz", "wb") as stream:
                pickle.dump([{"idx": 7, "category": "chair", "num_detections": 2},
                             {"idx": 8, "category": "pruned"}], stream)
            projected = export_graph(result, dataset)
            first_native_pose = np.loadtxt(dataset / "scene/traj.txt", max_rows=1).reshape(4, 4)
            self.assertEqual(projected["nodes"][0]["position"], [11.0, 0.0, -4.0])
            self.assertEqual(projected["pipeline_to_habitat"], first_native_pose.tolist())
            self.assertEqual(projected["input_first_camera_to_world"][0][3], 10.0)
            self.assertEqual(projected["excluded_objects_without_gaussians"], [8])
            self.assertTrue(projected["semantic_output"]["semantic_accuracy_eligible"])
            self.assertEqual(projected["semantic_output"]["evaluation_scope"],
                             "official_semantic_output")
            self.assertEqual(projected["nodes"][0]["semantic_label_source"],
                             "official_dam_qwen_category")
            graph = dynamicgsg_graph(result)
            self.assertEqual(graph["nodes"][0]["id"], "object:7")

            model = root / "dynamicgsg.replay.json"
            model.write_text(json.dumps({"baseline": "dynamicgsg", "graph": graph}))
            scene_path = root / "dynamicgsg.replay.scene.json"
            summary = export_scene(model, result, scene_path)
            self.assertEqual(summary["native_vertices"], 2)
            scene = json.loads(scene_path.read_text())
            self.assertEqual(scene["objects"][0]["positions"][:3], [10.0, 1.0, -3.0])

    def test_raw_no_dam_objects_are_explicitly_class_agnostic(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            recording = self.make_recording(root, count=1)
            dataset = root / "dataset"
            export(recording, dataset)
            result = root / "result"
            result.mkdir()
            np.savez(result / "params_with_idx.npz",
                     means3D=np.array([[0, 0, 0]], dtype=np.float32),
                     object_idx=np.array([[257]], dtype=np.int32))
            with gzip.open(result / "objects.pkl.gz", "wb") as stream:
                pickle.dump([{"idx": 257, "class_id": [3, 8], "num_detections": 2}], stream)
            projected = export_graph(result, dataset)
            node = projected["nodes"][0]
            self.assertEqual(node["label"], "object 257")
            self.assertFalse(node["semantic_label_available"])
            self.assertIsNone(node["semantic_label_source"])
            self.assertEqual(projected["semantic_output"], {
                "official_postprocessor": "DAM-3B + qwen2.5-vl-72b-instruct",
                "official_category_nodes": 0,
                "included_nodes": 1,
                "semantic_accuracy_eligible": False,
                "evaluation_scope": "class_agnostic_geometry_tracking",
            })


if __name__ == "__main__":
    unittest.main()
