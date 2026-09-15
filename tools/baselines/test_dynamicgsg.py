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

from tools.baselines.dynamicgsg_color_book import extend
from tools.baselines.dynamicgsg_export import export
from tools.baselines.dynamicgsg_observer import (
    ObjectStreamObserver,
    _centroids,
    tolerate_empty_detections,
    write_checkpoint,
)
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

    def make_native_tree(self, root: Path):
        native = root / "native"
        profile = native / "configs/found/profile.py"
        profile.parent.mkdir(parents=True, exist_ok=True)
        resources = {}
        for index, field in enumerate(NATIVE_RESOURCE_FIELDS):
            resource = native / f"resource-{index}.txt"
            resource.write_text(field)
            resources[field] = str(resource.relative_to(native))
        profile.write_text(
            "config={'data': {'stride': 4}, 'lang': " + repr(resources) + ", 'viz': {}, "
            "'tracking': {'use_gt_poses': False, 'num_iters': 200}, "
            "'mapping': {'num_iters': 80}, "
            "'whether_to_update': True}\n")
        models = root / "models"
        models.mkdir(exist_ok=True)
        for filename in MODEL_FIELDS.values():
            (models / filename).write_bytes(b"model")
        return native, profile, models

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
            # Sampling is declared by the export, so the profile's own stride never wins.
            self.assertEqual(config["data"]["stride"], 1)
            with self.assertRaises(ValueError):
                prepare_config(native, dataset, root / "result-strided", models, profile,
                               stride=4, variant="upstream-execfix")
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
            self.assertEqual(by_key["data.stride"], "input_contract_sampling")
            sampling = json.loads((output / "input_sampling.json").read_text())
            self.assertEqual(sampling["uniform_stride"], 1)
            self.assertEqual(sampling["sampled_source_frame_indices"], [0, 1])
            self.assertEqual((output / "frame_mapping.jsonl").read_text(),
                             (dataset / "frame_mapping.jsonl").read_text())
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

    def test_observer_centroids_are_per_object_means_in_habitat(self):
        # Two objects, one Gaussian apart, under a camera-to-world that shifts by +10 in x
        # and flips y and z -- the same basis the adapter applies to the final graph.
        params = {"means3D": np.array([[0., 0., 0.], [2., 0., 0.], [0., 4., 0.]]),
                  "object_idx": np.array([[5], [5], [9]])}
        transform = np.array([[1., 0, 0, 10], [0, -1., 0, 0], [0, 0, -1., 0], [0, 0, 0, 1.]])
        centroids = _centroids(params, transform)
        self.assertEqual(sorted(centroids), [5, 9])
        np.testing.assert_allclose(centroids[5][0], [11., 0., 0.])
        self.assertEqual(centroids[5][1], 2)
        np.testing.assert_allclose(centroids[9][0], [10., -4., 0.])
        self.assertEqual(centroids[9][1], 1)
        self.assertEqual(_centroids({}, transform), {})

    def test_observer_stream_names_removed_objects_by_difference(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)
            observer = ObjectStreamObserver(output, np.eye(4))
            params = {"means3D": np.zeros((2, 3)), "object_idx": np.array([[1], [2]])}
            observer.record(0, [{"idx": 1, "num_detections": 3}, {"idx": 2}], params)
            observer.record(1, [{"idx": 1, "num_detections": 4}], params)
            rows = [json.loads(line) for line in
                    (output / "graph_stream.jsonl").read_text().splitlines()]
            self.assertEqual(rows[0]["removed"], [])
            self.assertEqual(rows[1]["removed"], [2])
            self.assertEqual(rows[0]["objects"][0]["detections"], 3)
            self.assertFalse(observer.receipt()["native_algorithm_modified"])
            self.assertEqual(observer.receipt()["recorded_frames"], 2)

    def test_gt_pose_override_is_recorded_as_an_explicit_experiment(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            recording = self.make_recording(root)
            dataset = root / "dataset"
            export(recording, dataset)
            native, profile, models = self.make_native_tree(root)
            (root / "gt-poses").mkdir()
            (root / "as-published").mkdir()
            _, config = prepare_config(native, dataset, root / "gt-poses", models, profile,
                                       use_gt_poses=True, mapping_num_iters=40,
                                       variant="algorithm-experiment")
            self.assertTrue(config["tracking"]["use_gt_poses"])
            self.assertEqual(config["mapping"]["num_iters"], 40)
            provenance = json.loads((root / "gt-poses/config_provenance.json").read_text())
            by_key = {row["key"]: row["source"] for row in provenance["rows"]}
            self.assertEqual(by_key["tracking.use_gt_poses"], "explicit_test_override")
            self.assertEqual(by_key["mapping.num_iters"], "explicit_test_override")
            self.assertEqual(sorted(provenance["experimental_rows"]),
                             ["mapping.num_iters", "tracking.use_gt_poses"])
            # Left alone, the profile's own value must survive untouched.
            _, untouched = prepare_config(native, dataset, root / "as-published", models, profile)
            self.assertFalse(untouched["tracking"]["use_gt_poses"])
            self.assertEqual(untouched["mapping"]["num_iters"], 80)
            other = json.loads((root / "as-published/config_provenance.json").read_text())
            self.assertEqual(other["experimental_rows"], [])

    def test_checkpoint_graph_matches_what_the_final_export_would_produce(self):
        class FakeObjects(list):
            def to_serializable(self):
                return list(self)

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset = root / "dataset"
            (dataset / "scene").mkdir(parents=True)
            np.savetxt(dataset / "scene/traj.txt", np.eye(4).reshape(1, -1))
            output = root / "run"
            output.mkdir()
            params = {"means3D": np.array([[0., 0., 0.], [1., 1., 1.], [5., 5., 5.]]),
                      "object_idx": np.array([[3], [3], [8]])}
            objects = FakeObjects([{"idx": 3, "class_name": "chair", "num_detections": 4},
                                   {"idx": 8, "class_name": "table", "num_detections": 2}])
            graph = write_checkpoint(output, dataset, 120, objects, params)

            checkpoint = output / "checkpoint"
            self.assertTrue((checkpoint / "dynamicgsg_graph.json").is_file())
            marker = json.loads((checkpoint / "checkpoint.json").read_text())
            self.assertEqual(marker["native_frame_index"], 120)
            self.assertEqual(marker["objects"], 2)
            # The checkpoint must be the SAME artefact a completed run produces: running
            # the final export over the checkpoint's own files must reproduce the graph.
            saved = json.loads((checkpoint / "dynamicgsg_graph.json").read_text())
            self.assertEqual(export_graph(checkpoint, dataset)["nodes"], saved["nodes"])
            self.assertEqual(saved["schema"], graph["schema"])
            self.assertEqual([n["id"] for n in saved["nodes"]], ["object:3", "object:8"])
            self.assertEqual(saved["native_gaussians"], 3)
            # A second checkpoint replaces the first and leaves no staging directory.
            write_checkpoint(output, dataset, 240, objects, params)
            self.assertEqual(json.loads((output / "checkpoint/checkpoint.json").read_text())
                             ["native_frame_index"], 240)
            self.assertEqual(sorted(p.name for p in output.glob(".checkpoint*")), [])

    def test_extended_colour_book_keeps_the_upstream_prefix_verbatim(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            upstream = root / "scannet200.txt"
            original = ["1.040000000000000000e+02 2.040000000000000000e+02 2.550000000000000000e+02",
                        "1.880000000000000000e+02 1.890000000000000000e+02 3.400000000000000000e+01",
                        "0.000000000000000000e+00 0.000000000000000000e+00 0.000000000000000000e+00"]
            upstream.write_text("R G B\n" + "\n".join(original) + "\n")
            report = extend(upstream, root / "extended.txt", entries=64)
            self.assertEqual(report["upstream_entries"], 3)
            self.assertEqual(report["total_entries"], 64)
            written = (root / "extended.txt").read_text().splitlines()
            # Every index the upstream book could serve must resolve to the same colour.
            self.assertEqual(written[0], "R G B")
            self.assertEqual(written[1:4], original)
            self.assertEqual(upstream.read_text(), "R G B\n" + "\n".join(original) + "\n")
            triplets = [tuple(round(float(v)) for v in row.split()) for row in written[1:]]
            self.assertEqual(len(set(triplets)), 64)
            # Refuse to shrink: that would silently drop colours already in use.
            with self.assertRaises(ValueError):
                extend(upstream, root / "smaller.txt", entries=2)

    def test_empty_detection_guard_returns_upstreams_own_empty_value(self):
        class Sentinel(list):
            pass

        class FakeNative:
            DetectionList = Sentinel

            @staticmethod
            def process_this_frame_detection(flag):
                if flag == "empty":
                    raise UnboundLocalError(
                        "local variable 'xyxy_tensor' referenced before assignment")
                if flag == "other":
                    raise UnboundLocalError("local variable 'masks_np' referenced before assignment")
                return ["a detection"]

        native, counter = FakeNative(), []
        tolerate_empty_detections(native, counter)
        self.assertEqual(native.process_this_frame_detection("ok"), ["a detection"])
        self.assertEqual(counter, [])
        self.assertEqual(native.process_this_frame_detection("empty"), Sentinel())
        self.assertEqual(counter, [1])
        # Any other unbound variable must still raise; the guard is not a blanket catch.
        with self.assertRaises(UnboundLocalError):
            native.process_this_frame_detection("other")

    def test_export_stride_samples_the_recording_uniformly(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            recording = self.make_recording(root, count=5)
            dataset = root / "dataset"
            manifest = export(recording, dataset, stride=2)
            self.assertEqual(manifest["sampling_stride"], 2)
            self.assertEqual(manifest["exported_frames"], 3)
            self.assertEqual(manifest["sampled_source_frame_indices"], [0, 2, 4])
            self.assertEqual(manifest["native_stride_required"], 1)
            mapping = [json.loads(line) for line in
                       (dataset / "frame_mapping.jsonl").read_text().splitlines()]
            self.assertEqual([row["source_index"] for row in mapping], [0, 2, 4])
            self.assertEqual([row["exported_index"] for row in mapping], [0, 1, 2])
            # The native reader indexes frame000000.. contiguously; a strided export
            # must still hand it the source bytes of the sampled frame.
            self.assertEqual((dataset / "scene/results/frame000001.jpg").stat().st_ino,
                             (recording / "rgb/000002.png").stat().st_ino)

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
