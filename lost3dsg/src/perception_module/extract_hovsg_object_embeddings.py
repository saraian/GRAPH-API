#!/usr/bin/env python3
"""Calcola offline embedding oggetto HOV-SG senza modificare il ramo OWLv2.

Il crop associato a un oggetto persistente e' quello con la stessa instance
label e data di scrittura piu' vicina alla creation_time dell'oggetto.  Ogni
scelta viene salvata in un report, incluso lo scarto temporale, cosi' che la
provenienza dell'embedding resti verificabile.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import tempfile
from pathlib import Path
from typing import Any

import numpy as np


MODEL_NAME = "ViT-H-14"
PRETRAINED = "laion2b_s32b_b79k"


def _load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ValueError(f"JSON non valido o illeggibile: {path}: {exc}") from exc


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.",
        suffix=".tmp", delete=False
    ) as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")
        temporary = Path(stream.name)
    os.replace(temporary, path)


def _safe_label(label: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", label)


def select_crop(obj: dict[str, Any], crops_dir: Path,
                max_delta_s: float) -> dict[str, Any]:
    """Restituisce una selezione auditabile; non usa mai un label diverso."""
    label = str(obj.get("label", ""))
    object_id = obj.get("object_id")
    timestamp = obj.get("creation_time")
    timestamp_source = "creation_time"
    if not isinstance(timestamp, (int, float)) or not math.isfinite(timestamp):
        timestamp = obj.get("last_perception_timestamp")
        timestamp_source = "last_perception_timestamp"

    result = {
        "object_id": object_id,
        "label": label,
        "status": "unmatched",
        "timestamp_source": timestamp_source,
        "object_timestamp": timestamp,
    }
    if not label or not isinstance(timestamp, (int, float)) or not math.isfinite(timestamp):
        result["reason"] = "label o timestamp dell'oggetto mancante"
        return result

    candidates = []
    pattern = f"crop_{_safe_label(label)}_*.jpg"
    for crop_path in crops_dir.glob(pattern):
        try:
            delta = abs(crop_path.stat().st_mtime - float(timestamp))
        except OSError:
            continue
        candidates.append((delta, crop_path))
    candidates.sort(key=lambda item: (item[0], str(item[1])))
    result["candidate_count"] = len(candidates)
    if not candidates:
        result["reason"] = f"nessun crop con pattern {pattern}"
        return result

    delta, crop_path = candidates[0]
    result["crop_path"] = str(crop_path.resolve())
    result["delta_s"] = delta
    if len(candidates) > 1:
        result["runner_up_delta_s"] = candidates[1][0]
    if delta > max_delta_s:
        result["reason"] = f"crop piu' vicino oltre la soglia di {max_delta_s:g} s"
        return result
    result["status"] = "matched"
    return result


def build_mapping(objects: list[dict[str, Any]], crops_dir: Path,
                  max_delta_s: float) -> list[dict[str, Any]]:
    return [select_crop(obj, crops_dir, max_delta_s) for obj in objects]


def encode_crops(rows: list[dict[str, Any]], checkpoint: str,
                 device: str, batch_size: int) -> dict[str, list[float]]:
    """Carica OpenCLIP solo quando si esegue davvero l'estrazione."""
    try:
        import open_clip
        import torch
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError(
            "Servono open_clip, torch e Pillow; usa l'ambiente semantic_perception"
        ) from exc

    selected = [row for row in rows if row["status"] == "matched"]
    pretrained = checkpoint or PRETRAINED
    model, _, preprocess = open_clip.create_model_and_transforms(
        MODEL_NAME, pretrained=pretrained, device=device)
    model.eval()
    result: dict[str, list[float]] = {}
    with torch.inference_mode():
        for start in range(0, len(selected), batch_size):
            batch_rows = selected[start:start + batch_size]
            tensors = []
            for row in batch_rows:
                with Image.open(row["crop_path"]) as image:
                    tensors.append(preprocess(image.convert("RGB")))
            images = torch.stack(tensors).to(device)
            features = model.encode_image(images)
            features = features / features.norm(dim=-1, keepdim=True)
            vectors = features.detach().cpu().float().numpy()
            if vectors.ndim != 2 or vectors.shape[1] != 1024:
                raise RuntimeError(
                    f"{MODEL_NAME} ha prodotto forma inattesa {vectors.shape}, attesa (*, 1024)")
            if not np.all(np.isfinite(vectors)):
                raise RuntimeError("OpenCLIP ha prodotto valori non finiti")
            for row, vector in zip(batch_rows, vectors):
                result[str(row["object_id"])] = vector.tolist()
    return result


def enrich_manifest(manifest: dict[str, Any],
                    vectors: dict[str, list[float]]) -> tuple[dict[str, Any], int]:
    enriched = dict(manifest)
    predicted = []
    attached = 0
    for original in manifest.get("predicted_objects", []):
        row = dict(original)
        # Il manifest di base puo' contenere embedding OWLv2 a 512 dimensioni.
        # Questo output e' volutamente un canale separato e omogeneo: prima li
        # rimuoviamo tutti, poi inseriamo soltanto i ViT-H/14 calcolati qui.
        row.pop("embedding", None)
        vector = vectors.get(str(row.get("object_id")))
        if vector is not None:
            row["embedding"] = vector
            attached += 1
        predicted.append(row)
    enriched["predicted_objects"] = predicted
    enriched["object_embedding_model"] = {
        "architecture": MODEL_NAME,
        "pretrained": PRETRAINED,
        "library": "open_clip",
        "normalized": True,
        "dimension": 1024,
        "embedded_objects": attached,
    }
    return enriched, attached


def enrich_persistent(objects: list[dict[str, Any]],
                      vectors: dict[str, list[float]]) -> tuple[list[dict[str, Any]], int]:
    """Attach each HOV-SG image embedding to its persistent object's bbox."""
    enriched = []
    attached = 0
    for original in objects:
        row = dict(original)
        bbox = row.get("bbox")
        if isinstance(bbox, dict):
            bbox = dict(bbox)
            bbox.pop("clip_embedding", None)
            vector = vectors.get(str(row.get("object_id")))
            if vector is not None:
                bbox["clip_embedding"] = vector
                attached += 1
            row["bbox"] = bbox
        enriched.append(row)
    return enriched, attached


def _summary(mapping: list[dict[str, Any]]) -> str:
    matched = [row for row in mapping if row["status"] == "matched"]
    deltas = [float(row["delta_s"]) for row in matched]
    maximum = max(deltas) if deltas else None
    return (f"{len(matched)}/{len(mapping)} oggetti associati ai crop"
            + (f"; scarto massimo {maximum:.3f} s" if maximum is not None else ""))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True,
                        help="directory con persistent_perception.json e cropped_images/")
    parser.add_argument("--manifest", type=Path,
                        help="manifest da arricchire (obbligatorio senza --dry-run)")
    parser.add_argument("--output", type=Path,
                        help="nuovo manifest; il sorgente non viene mai sovrascritto")
    parser.add_argument("--persistent-output", type=Path,
                        help="copia di persistent_perception.json con clip_embedding nel bbox")
    parser.add_argument("--report", type=Path,
                        help="report object_id -> crop (default: <output>.crop-map.json)")
    parser.add_argument("--checkpoint", default="",
                        help="checkpoint HOV-SG locale; senza, OpenCLIP scarica il pretrained")
    parser.add_argument("--device", default="cuda",
                        help="device Torch (default: cuda; usare cpu solo per prove lente)")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-delta-s", type=float, default=120.0)
    parser.add_argument("--dry-run", action="store_true",
                        help="verifica solo l'associazione oggetto-crop, senza caricare OpenCLIP")
    args = parser.parse_args()

    persistent_path = args.run_dir / "persistent_perception.json"
    crops_dir = args.run_dir / "cropped_images"
    objects = _load_json(persistent_path)
    if not isinstance(objects, list):
        parser.error(f"attesa una lista in {persistent_path}")
    if not crops_dir.is_dir():
        parser.error(f"directory crop mancante: {crops_dir}")
    if args.batch_size < 1 or args.max_delta_s < 0:
        parser.error("batch-size deve essere positivo e max-delta-s non negativo")

    mapping = build_mapping(objects, crops_dir, args.max_delta_s)
    print(_summary(mapping))
    if args.dry_run:
        if args.report:
            _atomic_json(args.report, mapping)
            print(f"report: {args.report}")
        return 0 if any(row["status"] == "matched" for row in mapping) else 2

    if args.manifest is None or args.output is None:
        parser.error("--manifest e --output sono obbligatori senza --dry-run")
    try:
        if args.output.resolve() == args.manifest.resolve():
            parser.error("--output deve essere diverso da --manifest")
    except OSError:
        pass
    manifest = _load_json(args.manifest)
    if not isinstance(manifest, dict):
        parser.error(f"atteso un oggetto JSON in {args.manifest}")

    manifest_ids = {str(row.get("object_id"))
                    for row in manifest.get("predicted_objects", [])}
    usable = [row for row in mapping
              if row["status"] == "matched" and str(row["object_id"]) in manifest_ids]
    if not usable:
        parser.error("nessun object_id associato compare nel manifest; run e manifest non coincidono")
    vectors = encode_crops(usable, args.checkpoint, args.device, args.batch_size)
    enriched, attached = enrich_manifest(manifest, vectors)
    _atomic_json(args.output, enriched)
    if args.persistent_output:
        enriched_persistent, persistent_attached = enrich_persistent(objects, vectors)
        _atomic_json(args.persistent_output, enriched_persistent)
        print(f"{args.persistent_output}: {persistent_attached} embedding aggiunti ai bbox persistenti")
    report = args.report or args.output.with_suffix(args.output.suffix + ".crop-map.json")
    _atomic_json(report, mapping)
    print(f"{args.output}: {attached} embedding HOV-SG aggiunti; report: {report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
