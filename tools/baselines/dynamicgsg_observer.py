"""Read-only observer that records native DynamicGSG object state once per frame.

No native file is edited and no native value is changed.  The native function
``compute_similarities_and_merge`` is wrapped in its own module before the
frozen script imports it, so the script runs exactly as delivered and the
wrapper only reads what the native call returns.

Object indices here are the native persistent ``idx`` values, not GRAPH-API
object identities.
"""
from __future__ import annotations

import argparse
import gzip
import json
import pickle
import runpy
import shutil
import sys
import time
from importlib.machinery import SourceFileLoader
from pathlib import Path

import numpy as np

STREAM_NAME = "graph_stream.jsonl"


def _centroids(params, transform):
    """One Habitat-frame centroid per native object index, in a single pass.

    The native means live in the first exported camera frame, so ``transform``
    is that camera-to-world matrix; this is the same convention the adapter
    applies to the final graph in dynamicgsg_run.export_graph.
    """
    means = params.get("means3D")
    indices = params.get("object_idx")
    if means is None or indices is None:
        return {}
    if hasattr(means, "detach"):
        means = means.detach().cpu().numpy()
    if hasattr(indices, "detach"):
        indices = indices.detach().cpu().numpy()
    means = np.asarray(means, dtype=float)
    indices = np.asarray(indices).reshape(-1).astype(np.int64)
    if means.ndim != 2 or means.shape[1] != 3 or len(means) != len(indices) or not len(means):
        return {}
    if indices.min() < 0:
        keep = indices >= 0
        means, indices = means[keep], indices[keep]
        if not len(means):
            return {}
    counts = np.bincount(indices)
    sums = np.stack([np.bincount(indices, weights=means[:, axis], minlength=len(counts))
                     for axis in range(3)], axis=1)
    present = np.flatnonzero(counts)
    centers = sums[present] / counts[present][:, None]
    world = np.column_stack((centers, np.ones(len(centers)))) @ np.asarray(transform, dtype=float).T
    return {int(index): (world[row, :3].tolist(), int(counts[index]))
            for row, index in enumerate(present)}


class ObjectStreamObserver:
    def __init__(self, output, transform):
        self.path = Path(output) / STREAM_NAME
        self.transform = transform
        self.previous = set()
        self.origin = time.perf_counter()
        self.overhead_s = 0.0
        self.frames = 0

    def record(self, frame_index, objects, params):
        started = time.perf_counter()
        centroids = _centroids(params, self.transform)
        rows = []
        current = set()
        for item in objects:
            index = int(item["idx"])
            current.add(index)
            centroid, gaussians = centroids.get(index, (None, 0))
            rows.append({"idx": index,
                         "category": str(item.get("category") or item.get("class_name") or ""),
                         "detections": int(item.get("num_detections", 0)),
                         "gaussians": gaussians,
                         "centroid": centroid})
        entry = {"frame": int(frame_index), "num_objects": len(rows),
                 "num_gaussians": int(len(params["object_idx"])) if "object_idx" in params else None,
                 "objects": rows,
                 "removed": sorted(self.previous - current),
                 "native_elapsed_s": round(started - self.origin, 6)}
        self.previous = current
        with self.path.open("a") as stream:
            stream.write(json.dumps(entry) + "\n")
        self.frames += 1
        self.overhead_s += time.perf_counter() - started

    def receipt(self):
        return {"schema": "graphapi.dynamicgsg_native_observer.v1",
                "observation_unit": "native DynamicGSG object instance",
                "native_algorithm_modified": False,
                "hook": "utils.map_objects_utils_up_with_groupv3.compute_similarities_and_merge",
                "hook_kind": "read-only wrapper around the unchanged native call",
                "recorded_frames": self.frames,
                "observer_overhead_s": round(self.overhead_s, 6),
                "stream": STREAM_NAME}


def write_checkpoint(output, dataset, frame_index, objects, params):
    """Export an evaluable graph from the live map, using the final export path.

    The native pipeline writes params_with_idx.npz and objects.pkl.gz only when it
    finishes, so a run stopped early yields no graph at all.  This lays down those
    two files from in-memory state in the same format and then calls the SAME
    dynamicgsg_run.export_graph the completed run uses, so a checkpoint graph and a
    final graph are produced by one code path, not two.

    Written to a temporary directory and renamed, so a checkpoint interrupted
    mid-write never replaces a good one.  Only the newest is kept.
    """
    from .dynamicgsg_run import export_graph

    output, dataset = Path(output), Path(dataset)
    staging = output / f".checkpoint-{frame_index:06d}"
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    means, indices = params.get("means3D"), params.get("object_idx")
    if hasattr(means, "detach"):
        means = means.detach().cpu().numpy()
    if hasattr(indices, "detach"):
        indices = indices.detach().cpu().numpy()
    np.savez(staging / "params_with_idx.npz",
             means3D=np.asarray(means), object_idx=np.asarray(indices))
    with gzip.open(staging / "objects.pkl.gz", "wb") as stream:
        pickle.dump(objects.to_serializable(), stream)
    graph = export_graph(staging, dataset)
    (staging / "checkpoint.json").write_text(json.dumps({
        "schema": "graphapi.dynamicgsg_checkpoint.v1",
        "native_frame_index": int(frame_index),
        "objects": len(graph["nodes"]),
        "native_gaussians": graph["native_gaussians"],
        "note": ("Partial map at this frame, exported by dynamicgsg_run.export_graph, "
                 "the same function the completed run uses."),
    }, indent=2) + "\n")
    final = output / "checkpoint"
    previous = output / ".checkpoint-previous"
    if final.exists():
        final.rename(previous)
    staging.rename(final)
    if previous.exists():
        shutil.rmtree(previous)
    return graph


def tolerate_empty_detections(native, counter):
    """Reach the zero-detection branch upstream already wrote, from outside.

    ``process_this_frame_detection`` binds ``xyxy_tensor`` only inside
    ``if len(detections_gd.class_id) > 0:``.  When GroundingDINO finds nothing on
    a frame the function raises UnboundLocalError one line before its own
    ``if xyxy_tensor.numel() != 0: ... else: return detections`` handler.  At that
    point ``detections`` is still the empty ``DetectionList()`` bound at the top of
    the function and nothing else has been mutated, so returning an empty
    DetectionList is the same value on the same path -- not a new behaviour.

    The catch names the variable, so any other UnboundLocalError still raises.
    """
    original = native.process_this_frame_detection

    def guarded(*args, **kwargs):
        try:
            return original(*args, **kwargs)
        except UnboundLocalError as error:
            if "xyxy_tensor" not in str(error):
                raise
            counter.append(1)
            return native.DetectionList()

    native.process_this_frame_detection = guarded
    return original


def install(script, config_path, output, checkpoint_every=0):
    """Wrap the native call, then run the frozen script unchanged."""
    module = SourceFileLoader("dynamicgsg_observer_config",
                              str(config_path)).load_module()
    data = module.config["data"]
    transform = np.loadtxt(Path(data["basedir"]) / data["sequence"] / "traj.txt",
                           max_rows=1).reshape(4, 4)
    import utils.map_objects_utils_up_with_groupv3 as native

    observer = ObjectStreamObserver(output, transform)
    empty_detection_frames = []
    original_detection = tolerate_empty_detections(native, empty_detection_frames)
    original = native.compute_similarities_and_merge

    def observed(*args, **kwargs):
        result = original(*args, **kwargs)
        caller = sys._getframe(1).f_locals
        params = caller.get("params")
        frame_index = caller.get("time_idx")
        if params is not None and frame_index is not None and len(result) > 2:
            observer.record(frame_index, result[2], params)
            if checkpoint_every and frame_index and frame_index % checkpoint_every == 0:
                started = time.perf_counter()
                try:
                    write_checkpoint(output, data["basedir"], frame_index, result[2], params)
                    checkpoints.append(int(frame_index))
                except Exception as error:  # a checkpoint must never end the run
                    print(f"[checkpoint] frame {frame_index} failed: {error}", flush=True)
                observer.overhead_s += time.perf_counter() - started
        return result

    checkpoints = []
    native.compute_similarities_and_merge = observed
    sys.argv = [str(script), str(config_path)]
    try:
        runpy.run_path(str(script), run_name="__main__")
    finally:
        native.compute_similarities_and_merge = original
        native.process_this_frame_detection = original_detection
        receipt = observer.receipt()
        receipt["frames_with_no_detections"] = len(empty_detection_frames)
        receipt["checkpoint_every"] = checkpoint_every
        receipt["checkpoints_written"] = checkpoints
        receipt["empty_detection_handling"] = (
            "Upstream raises UnboundLocalError on xyxy_tensor before reaching its own "
            "'else: return detections' branch. The wrapper returns the same empty "
            "DetectionList that branch returns. No native file is edited.")
        (Path(output) / "native_observer.json").write_text(
            json.dumps(receipt, indent=2) + "\n")
    return observer


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--script", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--checkpoint-every", type=int, default=0,
                        help="Export an evaluable partial graph every N native frames")
    args = parser.parse_args(argv)
    install(args.script, args.config, args.output, args.checkpoint_every)


if __name__ == "__main__":
    main()
