#!/usr/bin/env python3
"""Single-entry HM3D/HOV-SG evaluation pipeline."""
from __future__ import annotations
import argparse, json, re
from pathlib import Path
import numpy as np
import build_hm3d_eval_manifest as adapter
import extract_hovsg_object_embeddings as embedding_stage
import metrics_eval
import metrics_eval_visualize
import object_metrics

SCENES = {"808":"00808-y9hTuugGdiq","810":"00810-CrMo8WxCyVb","813":"00813-svBbv1Pavdk","814":"00814-p53SfW6mjZe","815":"00815-h1zeeAwLh9Z","820":"00820-mL8ThkuaVTM","821":"00821-eF36g7L6Z9M","824":"00824-Dd4bFSTQ8gi","827":"00827-BAbdmeyTvMZ","829":"00829-QaLdnwvtxbs","862":"00862-LT9Jq6dN3Ea","873":"00873-bxsVRursffK","877":"00877-4ok3usBNeis"}
HERE = Path(__file__).resolve().parent

def _gt(directory, key):
    canonical = SCENES[key].split("-", 1)[0]
    paths = [directory / f"manifest_gt_{canonical}.json", directory / f"manifest_gt_{key}.json", HERE / f"manifest_gt_{canonical}.json", HERE / f"manifest_gt_{key}.json"]
    return next((p for p in paths if p.is_file()), paths[0])

_ground_truth_path = _gt


def _scene_number(value):
    """Extract the canonical five-digit HM3D scene number from metadata."""
    match = re.search(r"(?<!\d)(\d{5})(?!\d)", str(value or ""))
    return match.group(1) if match else None


def _validate_scene_provenance(expected, ground_truth, bev):
    """Reject crossed GT/run artifacts before producing plausible-looking scores."""
    expected = str(expected).zfill(5)
    gt_scene = _scene_number(ground_truth.get("scene"))
    gt_source_scene = _scene_number(
        (ground_truth.get("ground_truth_source") or {}).get("scene"))
    for label, actual in (("manifest GT", gt_scene),
                          ("sorgente del manifest GT", gt_source_scene)):
        if actual is not None and actual != expected:
            raise ValueError(
                f"{label} della scena {actual}, ma è stata richiesta {expected}")

    # New bundles record the source scene in bev_data.json.  Keep accepting old
    # bundles without provenance, but never accept an explicit mismatch.
    run_scene = _scene_number(bev.get("scene") or bev.get("scene_id"))
    if run_scene is not None and run_scene != expected:
        raise ValueError(
            f"run-dir della scena {run_scene}, ma è stata richiesta {expected}")

def _write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    def json_default(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.generic):
            return obj.item()
        raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False,
                               allow_nan=False, default=json_default) + "\n",
                    encoding="utf-8")


def _rooms_from_rviz_snapshot(run_dir, active_floor_index):
    """Read exactly the geometry published by RoomManager's RViz topic.

    ``room.json`` is the persisted snapshot of ``/room_areas_array``.  The
    publisher sends each active ``room['polygon']`` point as ROS ``x,y``;
    therefore this function deliberately performs no axis swap, sign change,
    rotation, or rescaling.
    """
    document = json.loads((run_dir / "room.json").read_text(encoding="utf-8"))
    if isinstance(document, list) and not document:
        document = {"rooms": []}
    if not isinstance(document, dict):
        raise ValueError("room.json deve contenere un oggetto JSON")
    rows = []
    # RoomManager serializes the building used by the MarkerArray under this
    # exact path.  Do not reconstruct regions from GT or from segmentation.
    room_source = document.get("building", {}).get("rooms")
    if room_source is None:
        raise ValueError("room.json non contiene building.rooms, cioè room areas")
    for room in room_source:
        if not isinstance(room, dict) or room.get("active", True) is False:
            continue
        polygon = room.get("polygon", [])
        if not isinstance(polygon, list) or len(polygon) < 3:
            continue
        try:
            points = [[float(point[0]), float(point[1])] for point in polygon]
        except (IndexError, TypeError, ValueError):
            continue
        rows.append({"polygon_xz_m": points, "room_id": room.get("room_id"),
                     "floor_index": active_floor_index,
                     "predicted_label": room.get("semantic_label", "")})
    return rows

def _result_row(report, task):
    summary = report.get("summary", {})
    columns = ("Method", "Task", "top5", "top10", "top25", "top100",
               "top250", "top500", "AUC_top_k", "Time [s]", "AP",
               "Acc_F [%]", "Precision regions [%]", "Recall regions [%]")
    # The JSON summary uses the HOV-SG column names with explicit units for
    # AUC/time/region fields.  Previously the console looked empty even when
    # the JSON contained the values, which made debugging the region result
    # misleading.
    keys = ("top5", "top10", "top25", "top100", "top250", "top500",
            "AUC_top_k_pct", "Time_s", "AP", "Acc_F_pct",
            "Precision_regions_pct", "Recall_regions_pct")
    values = ["HOV-SG", task] + [summary.get(k) for k in keys]
    def show(value):
        return "-" if value is None else str(value)
    return "\t".join(columns) + "\n" + "\t".join(show(v) for v in values)


def _polygon_area(row):
    points = np.asarray(row.get("polygon_xz_m", []), dtype=float)
    if points.ndim != 2 or points.shape[0] < 3 or points.shape[1] != 2:
        return 0.0
    return abs(float(np.sum(
        points[:, 0] * np.roll(points[:, 1], -1)
        - points[:, 1] * np.roll(points[:, 0], -1)) / 2.0))


def _print_region_diagnostics(scene):
    """Print the exact directional region quantities used by HOV-SG."""
    predicted = scene.get("predicted_regions", [])
    ground_truth = scene.get("ground_truth_regions", [])
    print("\nREGION DIAGNOSTICS (HOV-SG, raster 5 cm)")
    if not predicted or not ground_truth:
        print(f"  prediction regions: {len(predicted)} | GT regions: {len(ground_truth)}")
        return
    precision, recall = metrics_eval.hovsg_region_overlap_matrices(
        predicted, ground_truth, resolution=0.05)
    print(f"  prediction regions: {len(predicted)} | GT regions: {len(ground_truth)}")
    print("  GT region | best pred | area GT [m2] | area pred [m2] | recall | precision")
    for gi, gt_row in enumerate(ground_truth):
        pi = int(np.argmax(recall[:, gi]))
        print(
            f"  {gt_row.get('region_id', gi)!s:9} | "
            f"{predicted[pi].get('room_id', pi)!s:9} | "
            f"{_polygon_area(gt_row):12.3f} | "
            f"{_polygon_area(predicted[pi]):15.3f} | "
            f"{recall[pi, gi]:.4f} | {precision[pi, gi]:.4f}"
        )
    print("  prediction region | best GT | precision")
    for pi, pred_row in enumerate(predicted):
        gi = int(np.argmax(precision[pi, :]))
        print(
            f"  {pred_row.get('room_id', pi)!s:17} | "
            f"{ground_truth[gi].get('region_id', gi)!s:8} | "
            f"{precision[pi, gi]:.4f}"
        )

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("scene"); ap.add_argument("--run-dir", type=Path, required=True); ap.add_argument("--output-dir", type=Path, default=HERE)
    ap.add_argument("--ground-truth", type=Path); ap.add_argument("--persistent-perception", type=Path)
    ap.add_argument("--checkpoint", default=str(HERE / "checkpoints" / "laion2b_s32b_b79k.bin")); ap.add_argument("--device", default="cuda"); ap.add_argument("--batch-size", type=int, default=8); ap.add_argument("--seed", type=int, default=7, help="seed per Python/NumPy/PyTorch (default: 7)"); ap.add_argument("--max-crop-delta-s", type=float, default=120.0)
    ap.add_argument("--region-iou", type=float, default=.5); ap.add_argument("--object-distance", type=float, default=.5)
    args = ap.parse_args(); key = str(args.scene).strip().lower().removeprefix("scene_").lstrip("0") or "0"
    if key not in SCENES: ap.error("scena non supportata")
    run, out = args.run_dir.resolve(), args.output_dir.resolve(); gt = (args.ground_truth or _gt(out, key)).resolve(); persistent = args.persistent_perception.resolve() if args.persistent_perception else None
    if not run.is_dir(): ap.error(f"run directory inesistente: {run}")
    if not gt.is_file(): ap.error(f"manifest GT inesistente: {gt}")
    out.mkdir(parents=True, exist_ok=True); canonical = SCENES[key].split("-",1)[0]
    # Always build from the two explicit current inputs. No yaw option and no
    # previous evaluation artifact can influence this manifest.
    gt_document = json.loads(gt.read_text(encoding="utf-8"))
    bev_document = json.loads((run / "bev_data.json").read_text(encoding="utf-8"))
    try:
        _validate_scene_provenance(canonical, gt_document, bev_document)
    except ValueError as exc:
        ap.error(str(exc))
    manifest = adapter.build(gt_document, run,
                             persistent or run / "persistent_perception.json", 0.0)
    # Keep the browser and the region metric tied to the same RViz snapshot,
    # even if the adapter implementation changes later.
    manifest["predicted_regions"] = _rooms_from_rviz_snapshot(
        run, manifest.get("adapter_notes", {}).get("active_floor_index", 0))
    crops = run / "cropped_images"
    if crops.is_dir():
        objects = json.loads((persistent or run / "persistent_perception.json").read_text())
        mapping = embedding_stage.build_mapping(objects, crops, args.max_crop_delta_s)
        ids = {str(x.get("object_id")) for x in manifest.get("predicted_objects", [])}; usable = [x for x in mapping if x["status"] == "matched" and str(x.get("object_id")) in ids]
        if usable:
            vectors = embedding_stage.encode_crops(
                usable, args.checkpoint, args.device, args.batch_size, args.seed)
            manifest, _ = embedding_stage.enrich_manifest(
                manifest, vectors, args.seed); manifest["embedding_crop_map"] = mapping
            manifest["embedding_seed"] = args.seed
    manifest["evaluation_protocol"] = (
        "HOV-SG top-k object association + HOV-SG 5 cm directional room overlap "
        "+ object_metrics center3d")
    manifest["region_association_metric"] = "HOV-SG max directional overlap"
    manifest["region_overlap_threshold"] = args.region_iou
    manifest["object_distance_threshold_m"] = args.object_distance
    manifest_path = out / f"manifest_eval_{canonical}.json"; metrics_path = out / f"risultati_eval_{canonical}.json"; visual_path = out / f"boxes_{canonical}.html"
    _write(manifest_path, manifest)
    evaluated = metrics_eval.filtered_scene(manifest)
    geometry, associations = object_metrics.evaluate_geometry([evaluated], args.object_distance)
    labels = object_metrics.evaluate_labels([evaluated], associations)
    report = metrics_eval.evaluate([manifest], out, args.region_iou, args.object_distance, True, False)
    # The old report fields remain available, while the authoritative object
    # table follows the HOV-SG-compatible v2 protocol in object_metrics.py.
    #
    # TWO PROTOCOLS LAND IN ONE TABLE, so say which key came from which. metrics_eval's own
    # objects() is IoU-based and TIME-BLIND -- it has no reference to object_metrics and never sees
    # a scripted window -- while object_metrics' v2 is centre-distance and time-gated. Merging them
    # key by key leaves a reader unable to tell a gated number from an ungated one, which matters
    # exactly when a scene script ran: the v1 keys it did not overwrite still describe a world with
    # no timeline in it.
    v1_only = sorted(set(report["table_iv_objects"]) - set(geometry) - set(labels))
    report["table_iv_objects"].update(geometry)
    report["table_iv_objects"].update(labels)
    scripted = (geometry.get("scripted") or {})
    report["table_iv_objects"]["protocol_provenance"] = {
        "v2_time_gated": sorted(set(geometry) | set(labels)),
        "v1_iou_time_blind": v1_only,
        "note": ("v2 keys come from object_metrics (centre-distance, scripted windows honoured); "
                 "v1 keys come from metrics_eval.objects (IoU, no notion of time). When a scene "
                 "script ran, only the v2 keys describe the world as it was."),
        "scripted_run": bool(scripted.get("present")),
        "v1_keys_not_meaningful_here": v1_only if scripted.get("present") else [],
    }
    report["summary"]["AP"] = geometry.get("ap")
    _print_region_diagnostics(evaluated)
    _write(metrics_path, report)
    metrics_eval_visualize.render([manifest], visual_path, args.object_distance, args.region_iou)
    print("\n" + _result_row(report, canonical))
    print(f"GT: {gt}\nManifest: {manifest_path}\nMetrics: {metrics_path}\nVisual: {visual_path}")

if __name__ == "__main__": raise SystemExit(main())
