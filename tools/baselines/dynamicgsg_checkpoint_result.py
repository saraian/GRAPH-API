"""Turn a mid-run DynamicGSG checkpoint into a directory the evaluator can read.

A checkpoint holds the map (``dynamicgsg_graph.json`` and the native files it was
exported from).  The evaluator also needs the run's streaming records, and the two
receipts the native pipeline only writes when it finishes.  This assembles all of
that into one directory without touching the live run: the streaming records are
symlinked, and the two receipts are written with ``complete: false`` and the frame
the checkpoint actually reached, so nothing here can be mistaken for a full run.

Run it after the fact.  It never writes into the run directory.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

STREAMED = ("graph_stream.jsonl", "frame_mapping.jsonl", "native.log",
            "input_sampling.json", "native_observer.json", "config_provenance.json")
FRAME_LINE = re.compile(r'^([0-9.]+)\tframe (\d+) num of objects')


def native_progress(log: Path):
    """Frames completed and native seconds elapsed, from the stamped log."""
    frames, elapsed = 0, 0.0
    for line in log.open(errors="replace"):
        match = FRAME_LINE.match(line)
        if match:
            frames += 1
            elapsed = float(match.group(1))
    return frames, elapsed


def assemble(run, output):
    run, output = Path(run).resolve(strict=True), Path(output)
    checkpoint = run / "checkpoint"
    marker = json.loads((checkpoint / "checkpoint.json").read_text())
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    for name in ("dynamicgsg_graph.json", "objects.pkl.gz", "params_with_idx.npz"):
        source = checkpoint / name
        if source.is_file():
            (output / name).symlink_to(source)
    missing = [name for name in STREAMED if not (run / name).is_file()]
    for name in STREAMED:
        if (run / name).is_file():
            (output / name).symlink_to(run / name)
    frames, elapsed = native_progress(run / "native.log")
    sampling = json.loads((run / "input_sampling.json").read_text())
    (output / "execution_timing.json").write_text(json.dumps({
        "schema": "graphapi.baseline_timing.v1",
        "clock": "time.perf_counter",
        "phases_s": {"native_pipeline": elapsed, "run_wall_time": elapsed},
        "scope": ("PARTIAL RUN. Native wall time up to the checkpoint frame, not a "
                  "complete tour; it must not be compared with a finished run's time."),
        "limitations": ["The run did not reach the end of its sampled input."],
    }, indent=2) + "\n")
    result = {
        "complete": False,
        "partial_checkpoint": True,
        "baseline": "DynamicGSG",
        "source_run": str(run),
        "checkpoint_native_frame_index": marker["native_frame_index"],
        "frames_completed": frames,
        "sampled_frames": sampling["sampled_frames"],
        "coverage_fraction": round(frames / sampling["sampled_frames"], 6),
        "objects": marker["objects"],
        "gaussians": marker["native_gaussians"],
        "recording": sampling["source_recording"],
        "input_sampling_stride": sampling["uniform_stride"],
        "missing_streamed_files": missing,
        "warning": ("Partial map. Every metric computed from it describes the tour only up "
                    "to the checkpoint frame."),
    }
    if (run / "baseline_result.json").is_file():
        original = json.loads((run / "baseline_result.json").read_text())
        for key in ("variant", "source_revision", "source_patch_sha256",
                    "container_image_id", "tracking_use_gt_poses", "mapping_num_iters",
                    "color_book_entries"):
            if key in original:
                result[key] = original[key]
    (output / "baseline_result.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, help="Run directory holding checkpoint/")
    parser.add_argument("--output", required=True, help="New directory to assemble into")
    args = parser.parse_args(argv)
    print(json.dumps(assemble(args.run, args.output), indent=2))


if __name__ == "__main__":
    main()
