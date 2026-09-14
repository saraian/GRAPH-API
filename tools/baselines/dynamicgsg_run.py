"""Run frozen DynamicGSG source on an externally exported shared acquisition."""
from __future__ import annotations

import argparse
import copy
import gzip
import hashlib
import importlib.util
import json
import os
import pickle
import pprint
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import yaml

from .dynamicgsg_export import SCHEMA as INPUT_SCHEMA
from .replay_model import bounds_corners, file_stamp
from .timing import phase, save

MODEL_FIELDS = {
    "grounding_dino_checkpoint_path": "groundingdino_swint_ogc.pth",
    "sam_model_path": "sam_l.pt",
    "ram_model_path": "ram_plus_swin_large_14m.pth",
    "clip_model_path": "open_clip_pytorch_model.bin",
    "yolo_model_path": "yolov8l-world.pt",
}
NATIVE_RESOURCE_FIELDS = (
    "color_book_path",
    "grounding_dino_config_path",
    "sys_prompt_file",
    "obj_prompt_file",
    "obj_caption_file",
    "classes_file",
)
RESOURCE_FALLBACKS = {
    "grounding_dino_config_path":
        "submodules/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py",
    "sys_prompt_file": "configs/prompts/parsing_query.txt",
    "obj_prompt_file": "configs/prompts/parsing_objects.txt",
    "obj_caption_file": "configs/prompts/parsing_objects_caption.txt",
}


def _load_profile(path: Path) -> dict:
    spec = importlib.util.spec_from_file_location("dynamicgsg_external_profile", path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return copy.deepcopy(module.config)


def _source_identity(root: Path, expected_patch_sha256=None) -> dict:
    git = ["git", "-c", f"safe.directory={root}", "-C", str(root)]
    status = subprocess.run(
        [*git, "status", "--porcelain", "--untracked-files=no", "--ignore-submodules=none"],
        check=True, capture_output=True, text=True,
    ).stdout
    revision = subprocess.run([*git, "rev-parse", "HEAD"], check=True,
                              capture_output=True, text=True).stdout.strip()
    patch = subprocess.run([*git, "diff", "--binary", "HEAD", "--"], check=True,
                           capture_output=True).stdout
    patch_sha256 = hashlib.sha256(patch).hexdigest() if patch else None
    if status and expected_patch_sha256 is None:
        raise RuntimeError("DynamicGSG tracked source or submodules are dirty")
    if expected_patch_sha256 is not None and patch_sha256 != expected_patch_sha256:
        raise RuntimeError(
            "DynamicGSG source patch hash mismatch: "
            f"expected {expected_patch_sha256}, got {patch_sha256}"
        )
    return {"revision": revision, "patch_sha256": patch_sha256,
            "tracked_status": status.splitlines()}


def _source_revision(root: Path) -> str:
    return _source_identity(root)["revision"]


def _flatten_config(value, prefix=""):
    if isinstance(value, dict):
        for key, item in value.items():
            name = f"{prefix}.{key}" if prefix else key
            yield from _flatten_config(item, name)
    else:
        yield prefix, value


def _native_resource(root: Path, field: str, configured) -> tuple[Path, bool]:
    path = Path(configured)
    if not path.is_absolute():
        path = root / path
    if path.is_file():
        return path.resolve(), False
    fallback = RESOURCE_FALLBACKS.get(field)
    if fallback is not None:
        candidate = root / fallback
        if candidate.is_file():
            return candidate.resolve(), True
    raise FileNotFoundError(f"Missing DynamicGSG native resource {field}: {path}")


def prepare_config(root: Path, dataset: Path, output: Path, models: Path, profile: Path,
                   stride=None, enable_dam=False, dynamic_start_frame=None,
                   variant="delivered-found-fork") -> tuple[Path, dict]:
    manifest = json.loads((dataset / "export_manifest.json").read_text())
    if manifest.get("schema") != INPUT_SCHEMA or not manifest.get("complete"):
        raise ValueError("DynamicGSG input export is incomplete or unknown")
    mapping = dataset / "frame_mapping.jsonl"
    scene = dataset / manifest["sequence"]
    expected = manifest["exported_frames"]
    basis = np.asarray(manifest.get("pose_basis_transform"), dtype=float)
    expected_basis = np.diag([1.0, -1.0, -1.0, 1.0])
    if (manifest.get("source_pose_convention") != "Habitat/OpenGL camera-to-world"
            or not str(manifest.get("native_pose_convention", "")).startswith(
                "Habitat world from OpenCV camera"
            )
            or basis.shape != (4, 4) or not np.array_equal(basis, expected_basis)):
        raise ValueError("DynamicGSG input pose-basis contract is missing or invalid")
    if file_stamp(mapping)["sha256"] != manifest["mapping_sha256"]:
        raise ValueError("DynamicGSG input mapping hash mismatch")
    if file_stamp(scene / "traj.txt")["sha256"] != manifest["trajectory_sha256"]:
        raise ValueError("DynamicGSG input trajectory hash mismatch")
    if (len(list((scene / "results").glob("frame*.jpg"))) != expected or
            len(list((scene / "results").glob("depth*.png"))) != expected or
            len((scene / "traj.txt").read_text().splitlines()) != expected):
        raise ValueError("DynamicGSG exported file counts disagree with manifest")
    config = _load_profile(profile)
    sources = {}
    def sourced(key, source):
        sources[key] = source

    config["workdir"] = str(output.parent)
    sourced("workdir", "runtime_adapter")
    config["run_name"] = output.name
    sourced("run_name", "runtime_adapter")
    config["primary_device"] = "cuda:0"
    sourced("primary_device", "runtime_adapter")
    config["use_wandb"] = False
    sourced("use_wandb", "runtime_adapter")
    config["save_checkpoints"] = False
    sourced("save_checkpoints", "runtime_adapter")
    config["live_viewer"] = False
    sourced("live_viewer", "runtime_adapter")
    data = config["data"]
    dataset_values = dict(basedir=str(dataset), sequence=manifest["sequence"], start=0, end=-1,
                          num_frames=-1, desired_image_height=manifest["height"],
                          desired_image_width=manifest["width"])
    data.update(dataset_values)
    for key in dataset_values:
        sourced(f"data.{key}", "dataset_adapter")
    if dynamic_start_frame is not None:
        if dynamic_start_frame < 0:
            raise ValueError("dynamic start frame must be non-negative")
        data["frame_begin_update"] = dynamic_start_frame
        sourced("data.frame_begin_update", "dataset_scheduler")
    if stride is not None:
        if stride < 1:
            raise ValueError("stride must be positive")
        data["stride"] = stride
        sourced("data.stride", "explicit_test_override")
    camera_path = output / "data_config.yaml"
    camera_doc = {"dataset_name": "replica", "camera_params": {
        "image_height": manifest["height"], "image_width": manifest["width"],
        **manifest["camera"],
    }}
    camera_path.write_text(yaml.safe_dump(camera_doc, sort_keys=False))
    data["gradslam_data_cfg"] = str(camera_path)
    sourced("data.gradslam_data_cfg", "dataset_adapter")
    for field, filename in MODEL_FIELDS.items():
        path = models / filename
        if not path.is_file():
            raise FileNotFoundError(f"Missing cached DynamicGSG model: {path}")
        config["lang"][field] = str(path)
        sourced(f"lang.{field}", "pinned_model_path")
    resource_stamps = []
    for field in NATIVE_RESOURCE_FIELDS:
        original = config["lang"][field]
        path, relocated = _native_resource(root, field, original)
        config["lang"][field] = str(path)
        sourced(f"lang.{field}", "upstream_resource_relocation" if relocated
                else "upstream_resource_path")
        resource_stamps.append({"field": field, "configured": original,
                                "relocated": relocated, **file_stamp(path)})
    config["lang"]["use_dam"] = bool(enable_dam)
    sourced("lang.use_dam", "output_adapter")
    if enable_dam:
        raise ValueError("DAM requires an explicitly prepared local LLM service; offline runner refuses it")
    config["viz"]["clip_model_path"] = str(models / MODEL_FIELDS["clip_model_path"])
    sourced("viz.clip_model_path", "pinned_model_path")
    config_path = output / "adapter_config.py"
    config_path.write_text("# Generated outside frozen DynamicGSG source.\nconfig = " +
                           pprint.pformat(config, sort_dicts=False, width=100) + "\n")
    rows = [{"key": key, "value": value, "source": sources.get(key, "upstream_profile")}
            for key, value in _flatten_config(config)]
    provenance = {
        "schema": "graphapi.dynamicgsg_config_provenance.v1",
        "variant": variant,
        "profile": file_stamp(profile),
        "rows": rows,
        "native_resources": resource_stamps,
        "experimental_rows": [row["key"] for row in rows
                              if row["source"] == "explicit_test_override"],
    }
    (output / "config_provenance.json").write_text(
        json.dumps(provenance, indent=2, allow_nan=False) + "\n"
    )
    return config_path, config


def export_graph(output: Path, dataset: Path) -> dict:
    params_path = output / "params_with_idx.npz"
    objects_path = output / "objects.pkl.gz"
    with np.load(params_path, allow_pickle=False) as params:
        means = np.asarray(params["means3D"], dtype=float)
        object_ids = np.asarray(params["object_idx"]).reshape(-1)
    if means.ndim != 2 or means.shape[1] != 3 or len(means) != len(object_ids) or not np.isfinite(means).all():
        raise ValueError("Invalid saved DynamicGSG Gaussian/object arrays")
    with gzip.open(objects_path, "rb") as stream:
        objects = pickle.load(stream)
    if not isinstance(objects, list):
        raise ValueError("Native objects.pkl.gz is not a serialized object list")
    first_pose = np.loadtxt(dataset / "scene" / "traj.txt", max_rows=1).reshape(4, 4)
    nodes = []
    excluded = []
    official_semantic_labels = 0
    for obj in objects:
        idx = int(obj["idx"])
        selected = means[object_ids == idx]
        if not len(selected):
            excluded.append(idx)
            continue
        # The native dataset normalizes every camera pose to the first camera.
        # Saved means3D is therefore in that first OpenCV-camera frame. The
        # exported first c2w maps it back into the source Habitat world frame.
        world = np.column_stack((selected, np.ones(len(selected)))) @ first_pose.T
        world = world[:, :3]
        lo, hi = world.min(0), world.max(0)
        category = obj.get("category")
        class_name = obj.get("class_name")
        if category:
            label = category
            label_source = "official_dam_qwen_category"
            official_semantic_labels += 1
        elif class_name:
            label = class_name
            label_source = "native_class_name_fallback"
        else:
            label = f"object {idx}"
            label_source = None
        nodes.append({"id": f"object:{idx}", "label": str(label), "type": "object",
                      "position": ((lo + hi) / 2).tolist(),
                      "corners": bounds_corners((lo + hi) / 2, hi - lo).tolist(),
                      "native_object_idx": idx, "native_gaussians": len(selected),
                      "native_detections": int(obj.get("num_detections", 0)),
                      "semantic_label_available": label_source is not None,
                      "semantic_label_source": label_source})
    if not nodes:
        raise RuntimeError("DynamicGSG produced no fused 3-D objects")
    semantic_eligible = official_semantic_labels == len(nodes)
    graph = {"schema": "graphapi.dynamicgsg_graph.v1", "coordinate_frame": "Habitat Y-up",
             "pipeline_to_habitat": first_pose.tolist(),
             "input_first_camera_to_world": first_pose.tolist(),
             "native_geometry_frame": "first exported OpenCV-camera frame",
             "nodes": nodes, "edges": [],
             "native_gaussians": len(means),
             "excluded_objects_without_gaussians": excluded,
             "semantic_output": {
                 "official_postprocessor": "DAM-3B + qwen2.5-vl-72b-instruct",
                 "official_category_nodes": official_semantic_labels,
                 "included_nodes": len(nodes),
                 "semantic_accuracy_eligible": semantic_eligible,
                 "evaluation_scope": ("official_semantic_output" if semantic_eligible
                                      else "class_agnostic_geometry_tracking"),
             },
             "sources": [file_stamp(params_path), file_stamp(objects_path)]}
    graph_path = output / "dynamicgsg_graph.json"
    graph_path.write_text(json.dumps(graph, indent=2, allow_nan=False) + "\n")
    return graph


def run(args):
    root = Path(args.baseline_root).resolve(strict=True)
    dataset = Path(args.dataset_export).resolve(strict=True)
    output = Path(args.output).resolve()
    models = Path(args.models).resolve(strict=True)
    profile = (root / args.profile).resolve(strict=True)
    if root not in profile.parents:
        raise ValueError("Profile must be inside the frozen DynamicGSG checkout")
    source = _source_identity(root, args.source_patch_sha256)
    if args.variant == "upstream-pristine" and source["patch_sha256"] is not None:
        raise ValueError("upstream-pristine requires an unmodified checkout")
    if args.variant in ("upstream-pristine-io", "upstream-execfix") and source["patch_sha256"] is None:
        raise ValueError(f"{args.variant} requires an exact non-empty source patch")
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    phases = {}
    started = time.perf_counter()
    with phase(phases, "adapter_configuration"):
        config_path, config = prepare_config(
            root, dataset, output, models, profile, args.stride,
            dynamic_start_frame=args.dynamic_start_frame,
            variant=args.variant,
        )
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", HF_HUB_OFFLINE="1",
               TRANSFORMERS_OFFLINE="1", WANDB_MODE="disabled", MPLBACKEND="Agg")
    with phase(phases, "native_pipeline"):
        with (output / "native.log").open("w") as log:
            subprocess.run(["xvfb-run", "-a", sys.executable,
                            str(root / "scripts/dynamic_gsg_real_ssim.py"), str(config_path)],
                           cwd=output, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
    with phase(phases, "adapter_verification"):
        graph = export_graph(output, dataset)
    phases["run_wall_time"] = time.perf_counter() - started
    save(output, phases, "External configuration, complete native DynamicGSG process, and output verification; "
         "native_pipeline is the measured native subprocess wall time")
    input_manifest = json.loads((dataset / "export_manifest.json").read_text())
    result = {
        "complete": True, "baseline": "DynamicGSG", "variant": args.variant,
        "container_image_id": os.environ.get("BASELINE_IMAGE_ID"),
        "source": str(root), "source_revision": source["revision"],
        "source_patch_sha256": source["patch_sha256"], "profile": str(profile),
        "recording": input_manifest["recording"], "dataset_export": str(dataset),
        "input_frames": input_manifest["exported_frames"], "stride": config["data"]["stride"],
        "effective_frames": len(range(0, input_manifest["exported_frames"], config["data"]["stride"])),
        "objects": len(graph["nodes"]), "gaussians": graph["native_gaussians"],
        "excluded_objects_without_gaussians": graph["excluded_objects_without_gaussians"],
        "dynamic_update_enabled": bool(config["whether_to_update"]),
        "dynamic_update_start_frame": config["data"]["frame_begin_update"],
        "dam_descriptions_enabled": False,
        "evaluation_scope": graph["semantic_output"]["evaluation_scope"],
        "semantic_accuracy_eligible": graph["semantic_output"]["semantic_accuracy_eligible"],
        "timing_scope": "native_mapping_without_official_dam_qwen_postprocessing",
        "live_viewer_enabled": False,
        "input_manifest": file_stamp(dataset / "export_manifest.json"),
        "resolved_config": file_stamp(config_path),
        "config_provenance": file_stamp(output / "config_provenance.json"),
    }
    (output / "baseline_result.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(result, indent=2))
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-root", required=True)
    parser.add_argument("--dataset-export", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--models", required=True)
    parser.add_argument("--profile", default="configs/found/dgsg_dynamic_fast.py")
    parser.add_argument(
        "--variant",
        choices=("upstream-pristine", "upstream-pristine-io", "upstream-execfix",
                 "delivered-found-fork", "algorithm-experiment"),
        default="delivered-found-fork",
        help="Provenance class; upstream-execfix requires an exact source patch hash",
    )
    parser.add_argument("--stride", type=int, help="Override the native profile stride")
    parser.add_argument("--dynamic-start-frame", type=int,
                        help="Dataset-scheduler effective frame where native updating begins")
    parser.add_argument("--source-patch-sha256",
                        help="Required exact git diff hash when native tracked source is patched")
    args = parser.parse_args(argv)
    run(args)


if __name__ == "__main__":
    main()
