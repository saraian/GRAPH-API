"""Export a shared acquisition into DynamicGSG's immutable Replica input contract."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path

import numpy as np
from PIL import Image

SCHEMA = "graphapi.dynamicgsg_input.v3"
OPENGL_TO_OPENCV_CAMERA = np.diag([1.0, -1.0, -1.0, 1.0])


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _link(source: Path, target: Path) -> None:
    if not source.is_file() or source.is_symlink():
        raise FileNotFoundError(f"Missing regular source file: {source}")
    os.link(source, target)
    if source.stat().st_ino != target.stat().st_ino or source.stat().st_dev != target.stat().st_dev:
        raise RuntimeError(f"Export is not a hard link: {target}")


def export(recording, output, max_frames=None, stride=1):
    recording, output = Path(recording).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError(output)
    acquisition_path = recording / "acquisition.json"
    frames_path = recording / "frames.jsonl"
    acquisition = json.loads(acquisition_path.read_text())
    frames = _read_rows(frames_path)
    if not acquisition.get("complete") or acquisition.get("frames") != len(frames) or not frames:
        raise ValueError("Acquisition is incomplete or its frame count disagrees")
    if acquisition.get("depth_png_units") != "millimetres":
        raise ValueError("DynamicGSG export requires millimetre uint16 depth PNGs")
    if acquisition.get("pose_convention") != "Habitat/OpenGL camera-to-world":
        raise ValueError("Unknown source pose convention")
    if max_frames is not None:
        if max_frames < 1:
            raise ValueError("max_frames must be positive")
        frames = frames[:max_frames]
    if stride < 1:
        raise ValueError("stride must be positive")
    source_frames_considered = len(frames)
    # Same uniform, ground-truth-independent rule HOV-SG applies in
    # tools/baselines/hovsg_run.py:select_hov_frames. Sampling is declared here
    # once; the native profile must then consume every exported frame.
    selected_source_indices = list(range(0, source_frames_considered, stride))
    frames = [frames[index] for index in selected_source_indices]

    sequence = output / "scene"
    results = sequence / "results"
    results.mkdir(parents=True)
    mappings = []
    pose_lines = []
    aggregate = hashlib.sha256()
    for exported_index, row in enumerate(frames):
        if (row.get("index") != selected_source_indices[exported_index]
                or Path(row["stem"]).name != row["stem"]):
            raise ValueError("Export only supports the acquisition's ordered uniform sample")
        stem = row["stem"]
        rgb = recording / "rgb" / f"{stem}.png"
        depth = recording / "depth" / f"{stem}.png"
        pose_path = recording / "pose" / f"{stem}.txt"
        pose = np.loadtxt(pose_path).reshape(4, 4)
        if not np.isfinite(pose).all() or not np.allclose(pose[3], [0, 0, 0, 1], atol=1e-6):
            raise ValueError(f"Invalid camera-to-world pose: {pose_path}")
        target_rgb = results / f"frame{exported_index:06d}.jpg"
        target_depth = results / f"depth{exported_index:06d}.png"
        _link(rgb, target_rgb)
        _link(depth, target_depth)
        stamps = {"rgb_sha256": _sha256(rgb), "depth_sha256": _sha256(depth),
                  "pose_sha256": _sha256(pose_path)}
        for key in ("rgb_sha256", "depth_sha256", "pose_sha256"):
            aggregate.update(f"{exported_index}:{key}:{stamps[key]}\n".encode())
        mappings.append({"exported_index": exported_index, "source_index": row["index"],
                         "source_stem": stem, "time_s": row["time_s"], **stamps})
        native_pose = pose @ OPENGL_TO_OPENCV_CAMERA
        pose_lines.append(" ".join(format(float(value), ".17g")
                                   for value in native_pose.reshape(-1)))

    trajectory_path = sequence / "traj.txt"
    trajectory_path.write_text("\n".join(pose_lines) + "\n")
    mapping_path = output / "frame_mapping.jsonl"
    mapping_path.write_text("".join(json.dumps(row, separators=(",", ":")) + "\n" for row in mappings))

    # The native reader selects frame*.jpg but imageio dispatches by signature. Hard-linking
    # the original PNG bytes avoids a lossy JPEG transcode while satisfying that selector.
    for index in sorted({0, len(frames) - 1}):
        source = recording / "rgb" / f"{frames[index]['stem']}.png"
        target = results / f"frame{index:06d}.jpg"
        with Image.open(source) as left, Image.open(target) as right:
            if left.format != "PNG" or right.format != "PNG" or not np.array_equal(np.asarray(left), np.asarray(right)):
                raise ValueError("Lossless RGB signature/decode validation failed")
        with Image.open(results / f"depth{index:06d}.png") as depth_image:
            depth_mode = depth_image.mode
            depth = np.asarray(depth_image)
        # Pillow may expose a 16-bit grayscale PNG as mode I/int32 after decoding.
        if (depth_mode not in {"I;16", "I"} or not np.issubdtype(depth.dtype, np.integer)
                or depth.min() < 0 or depth.max() > np.iinfo(np.uint16).max
                or not np.any(depth > 0)):
            raise ValueError("Depth must be finite uint16 millimetres with positive samples")

    first = frames[0]
    focal = first["width"] / (2 * math.tan(math.radians(first["hfov_deg"]) / 2))
    manifest = {
        "schema": SCHEMA,
        "complete": True,
        "recording": str(recording),
        "acquisition_sha256": _sha256(acquisition_path),
        "frames_sha256": _sha256(frames_path),
        "source_frames_total": acquisition["frames"],
        "source_frames_considered": source_frames_considered,
        "exported_frames": len(frames),
        "sampling_stride": stride,
        "sampling_strategy": "uniform GT-independent frame stride",
        "sampling_method": ("Uniform temporal coverage only. Ground-truth visibility, object "
                            "actions and evaluation outcomes do not select DynamicGSG input "
                            "frames. This matches the HOV-SG sampling rule so the two "
                            "baselines observe the same frames of the same tour."),
        "sampled_source_frame_indices": selected_source_indices,
        "native_stride_required": 1,
        "sequence": "scene",
        "rgb_storage": "original PNG bytes hard-linked under native frame*.jpg names",
        "depth_storage": "original uint16 millimetre PNG bytes hard-linked",
        "pose_storage": "source camera-to-world matrices converted to the native camera basis",
        "source_pose_convention": acquisition["pose_convention"],
        "native_pose_convention": "Habitat world from OpenCV camera (X right, Y down, Z forward)",
        "pose_basis_transform": OPENGL_TO_OPENCV_CAMERA.tolist(),
        "trajectory_sha256": _sha256(trajectory_path),
        "width": first["width"], "height": first["height"], "hfov_deg": first["hfov_deg"],
        "camera": {"fx": focal, "fy": focal, "cx": first["width"] / 2,
                   "cy": first["height"] / 2, "png_depth_scale": 1000.0},
        "content_aggregate_sha256": aggregate.hexdigest(),
        "mapping_sha256": _sha256(mapping_path),
    }
    (output / "export_manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recording", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-frames", type=int, help="Ordered prefix for a bounded smoke export")
    parser.add_argument("--stride", type=int, default=1,
                        help="Uniform sampling stride over the recording, as HOV-SG applies it")
    args = parser.parse_args(argv)
    print(json.dumps(export(args.recording, args.output, args.max_frames, args.stride), indent=2))


if __name__ == "__main__":
    main()
