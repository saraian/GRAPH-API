"""Standalone process that imports and runs the original facebookresearch/boxer code."""
from __future__ import annotations

import math
import os
from pathlib import Path
import sys
import tempfile
import time
import traceback
from contextlib import nullcontext

import numpy as np


def _rotation_from_quaternion_xyzw(q):
    x, y, z, w = q
    norm = math.sqrt(x * x + y * y + z * z + w * w)
    if norm < 1e-9:
        raise ValueError("camera transform has an invalid zero quaternion")
    x, y, z, w = (v / norm for v in (x, y, z, w))
    return np.asarray([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ], dtype=np.float32)


def _obb_record(obb):
    import torch

    center = obb.T_world_object.t.reshape(-1, 3)[0].detach().float().cpu().numpy()
    angles = obb.T_world_object.to_euler().reshape(-1, 3)[0].detach().float().cpu().numpy()
    bounds = obb.bb3_object.reshape(-1, 6)[0].detach().float().cpu().numpy()
    extents = np.asarray([bounds[1] - bounds[0], bounds[3] - bounds[2],
                          bounds[5] - bounds[4]], dtype=np.float32)
    yaw = float(angles[2])
    c, s = abs(math.cos(yaw)), abs(math.sin(yaw))
    aabb_half = np.asarray([
        c * extents[0] / 2 + s * extents[1] / 2,
        s * extents[0] / 2 + c * extents[1] / 2,
        extents[2] / 2,
    ], dtype=np.float32)
    result = {
        "center": center.tolist(), "extents": extents.tolist(), "yaw": yaw,
        "aabb_min": (center - aabb_half).tolist(),
        "aabb_max": (center + aabb_half).tolist(),
        "confidence": float(obb.prob.reshape(-1)[0].detach().float().cpu()),
    }
    return result


def _initialize(config):
    import torch

    root = Path(os.environ.get("BOXER_ROOT", "/opt/boxer")).resolve()
    if not (root / "run_boxer.py").is_file():
        raise FileNotFoundError(f"upstream Boxer checkout is missing: {root}")
    checkpoints = root / "ckpts"
    required = (
        "boxernet_hw960in2x6d768-c88128f8.ckpt",
        "dinov3_vits16plus_pretrain_lvd1689m-4057cbaa.pth",
        "owlv2-base-patch16-ensemble.pt",
    )
    missing = [name for name in required if not (checkpoints / name).is_file()]
    if missing:
        runtime_checkpoints = Path(os.environ.get(
            "GRAPHAPI_BOXER_CHECKPOINTS_RUNTIME", "/models/boxer/ckpts"))
        runtime_missing = [name for name in required if not (runtime_checkpoints / name).is_file()]
        if runtime_checkpoints.is_dir() and not runtime_missing:
            # PAL mounts /opt/boxer read-only and places the large checkpoints in a
            # separate read-only model mount. Use that directory directly instead
            # of trying to create a symlink inside the source checkout.
            checkpoints = runtime_checkpoints
            missing = [name for name in required if not (checkpoints / name).is_file()]
    if missing:
        raise FileNotFoundError(
            f"Boxer checkpoints missing under {checkpoints}: {', '.join(missing)}; "
            "run tools/download_boxer_checkpoints.sh")
    sys.path.insert(0, str(root))
    from boxernet import dinov3_wrapper
    from boxernet.boxernet import BoxerNet
    from owl import owl_wrapper
    from utils.taxonomy import load_text_labels
    dinov3_wrapper.CKPT_PATH = str(checkpoints)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    label_spec = config.get("labels", "lvisplus")
    if isinstance(label_spec, str):
        labels = load_text_labels(label_spec)
    else:
        labels = [str(value) for value in label_spec]
    if not labels:
        raise ValueError("Boxer requires at least one OWLv2 text prompt")
    precision = config.get("precision", "auto")
    # Upstream OWLv2 uses a module-level checkpoint path and writes its text
    # embedding cache beside that checkpoint. Point it at writable temporary
    # storage with links to the mounted weights and any existing caches.
    with tempfile.TemporaryDirectory(prefix="graphapi-owl-") as owl_storage:
        owl_checkpoint = Path(owl_storage) / required[2]
        owl_checkpoint.symlink_to((checkpoints / required[2]).resolve())
        for cache in checkpoints.glob("owlv2-base-patch16-ensemble_textemb_*.pt"):
            (Path(owl_storage) / cache.name).symlink_to(cache.resolve())
        owl_wrapper._CKPT_PATH = str(owl_checkpoint)
        owl = owl_wrapper.OwlWrapper(
            device=device,
            text_prompts=labels,
            min_confidence=float(config.get("threshold_2d", 0.20)),
            precision=None if precision == "auto" else precision,
            nms_iou_threshold=float(config.get("nms_iou_threshold", 0.5)),
        )
    model = BoxerNet.load_from_checkpoint(
        str(checkpoints / required[0]), device=device)
    return torch, labels, owl, model, device


def _infer(request, state, config, on_boxes_2d=None):
    import torch
    from utils.tw.camera import CameraTW
    from utils.tw.pose import PoseTW, IdentityPose

    torch, labels, owl, model, device = state
    owl_t0 = time.perf_counter()
    bgr = np.asarray(request["rgb_bgr"])
    rgb = np.ascontiguousarray(bgr[:, :, ::-1])
    depth = np.asarray(request["depth_m"], dtype=np.float32)
    info = request["intrinsics"]
    height, width = rgb.shape[:2]
    model_hw = model.hw
    if isinstance(model_hw, (int, np.integer)):
        out_h = out_w = int(model_hw)
    else:
        out_h, out_w = (int(x) for x in model_hw)
    resized = __import__("cv2").resize(rgb, (out_w, out_h), interpolation=__import__("cv2").INTER_LINEAR)
    image = torch.from_numpy(resized.copy()).permute(2, 0, 1).unsqueeze(0).float() / 255.0

    s_x, s_y = out_w / width, out_h / height
    fx, fy = float(info["fx"]) * s_x, float(info["fy"]) * s_y
    cx, cy = float(info["cx"]) * s_x, float(info["cy"]) * s_y
    camera = CameraTW.from_surreal(
        width=out_w, height=out_h,
        type_str="Pinhole",
        params=torch.tensor([fx, fy, cx, cy]),
        T_camera_rig=PoseTW(IdentityPose.clone()),
    )

    pose = request["T_world_camera"]
    rotation = _rotation_from_quaternion_xyzw(pose["rotation_xyzw"])
    translation = np.asarray(pose["translation"], dtype=np.float32)
    stride = max(1, int(config.get("depth_stride", 8)))
    sampled = depth[::stride, ::stride]
    rows, cols = np.indices(sampled.shape, dtype=np.float32)
    rows = rows * stride
    cols = cols * stride
    z = sampled
    valid = np.isfinite(z) & (z > 0.05) & (z < float(config.get("max_depth_m", 10.0)))
    if np.any(valid):
        x = (cols[valid] - float(info["cx"])) * z[valid] / float(info["fx"])
        y = (rows[valid] - float(info["cy"])) * z[valid] / float(info["fy"])
        points_camera = np.stack([x, y, z[valid]], axis=-1)
        points_world = points_camera @ rotation.T + translation
        cap = int(config.get("max_depth_points", 12000))
        if len(points_world) > cap:
            select = np.linspace(0, len(points_world) - 1, cap).astype(np.int64)
            points_world = points_world[select]
    else:
        points_world = np.zeros((0, 3), dtype=np.float32)

    world_camera = PoseTW.from_Rt(torch.from_numpy(rotation), torch.from_numpy(translation))
    with torch.inference_mode():
        boxes_boxer, scores_2d, label_ids, _ = owl.forward(
            torch.from_numpy(resized.copy()).permute(2, 0, 1).unsqueeze(0).float(),
            resize_to_HW=(int(owl.native_size[0]), int(owl.native_size[1])),
        )
        excluded = {str(value).strip().lower() for value in config.get("excluded_labels", [])}
        if excluded and len(boxes_boxer):
            keep_2d = torch.tensor(
                [labels[int(label_id)].strip().lower() not in excluded
                 for label_id in label_ids], dtype=torch.bool)
            boxes_boxer = boxes_boxer[keep_2d]
            scores_2d = scores_2d[keep_2d]
            label_ids = label_ids[keep_2d]
        if len(boxes_boxer) == 0:
            return {"detections": [], "timing_ms": {
                "owl": round((time.perf_counter() - owl_t0) * 1000.0, 1),
                "boxernet": 0.0}}

        # Hand OWLv2's original-image 2D proposals to the host before BoxerNet runs.
        # The host can classify these exact crops with the original VLM while this
        # worker continues to predict the metric 3D boxes in parallel.
        if on_boxes_2d is not None:
            proposals = []
            for proposal_index, (box, score, label_id) in enumerate(
                    zip(boxes_boxer, scores_2d, label_ids)):
                x1, x2, y1, y2 = (float(v) for v in box.float().cpu().numpy())
                proposals.append({
                    "proposal_index": proposal_index,
                    "bbox": [x1 / s_x, y1 / s_y, x2 / s_x, y2 / s_y],
                    "label": labels[int(label_id)],
                    "score": float(score.float().cpu()),
                })
            on_boxes_2d(proposals)

        # OWLv2 boxes are in the resized input image, matching the data loader
        # contract used by run_boxer.py. BoxerNet consumes its own x1,x2,y1,y2 order.
        boxes_for_model = boxes_boxer

        # OWLv2 returns the original Boxer's (x1,x2,y1,y2) convention.
        boxernet_input = {
            "img0": image,
            "cam0": camera,
            "T_world_rig0": world_camera,
            "sdp_w": torch.from_numpy(points_world),
            "bb2d": boxes_for_model.float(),
        }
        owl_ms = (time.perf_counter() - owl_t0) * 1000.0
        boxer_t0 = time.perf_counter()
        precision = config.get("precision", "auto")
        use_bfloat16 = (
            device == "cuda"
            and (precision == "bfloat16"
                 or (precision == "auto" and torch.cuda.is_bf16_supported()))
        )
        autocast = (torch.autocast(device_type="cuda", dtype=torch.bfloat16)
                    if use_bfloat16 else nullcontext())
        with autocast:
            output = model(boxernet_input)
        obbs = output["obbs_pr_w"].cpu()[0]
        boxer_ms = (time.perf_counter() - boxer_t0) * 1000.0
        labels_for_boxes = [labels[int(i)] for i in label_ids]
        scores_2d = scores_2d.float()
        scores_3d = obbs.prob.reshape(-1).float()
        threshold = float(config.get("threshold_3d", 0.5))
        keep = scores_3d >= threshold
        indices = torch.nonzero(keep, as_tuple=False).reshape(-1).tolist()
        obbs_kept = obbs[keep].clone()

    # Preserve the native prediction/proposal pairing; GRAPH_API owns tracking.
    records = []
    for kept_pos, original_index in enumerate(indices):
        box = boxes_boxer[original_index].float().cpu().numpy()
        x1, x2, y1, y2 = (float(v) for v in box)
        box_xyxy = [x1 / s_x, y1 / s_y, x2 / s_x, y2 / s_y]
        label = labels_for_boxes[original_index]
        prediction = _obb_record(obbs_kept[kept_pos])
        records.append({
            "proposal_index": original_index,
            "bbox": box_xyxy,
            "label": label,
            "score": float((scores_2d[original_index] + scores_3d[original_index]) / 2),
            "box_3d": prediction,
        })
    return {"detections": records, "timing_ms": {
        "owl": round(owl_ms, 1), "boxernet": round(boxer_ms, 1)}}


def worker_main(address, authkey_hex):
    from multiprocessing.connection import Client

    connection = Client(address, family="AF_UNIX", authkey=bytes.fromhex(authkey_hex))
    state = None
    config = None
    while True:
        request = connection.recv()
        if request.get("command") == "stop":
            break
        if request.get("command") == "init":
            try:
                config = request["config"]
                state = _initialize(config)
                connection.send({"ok": True})
            except Exception:
                connection.send({"ok": False, "error": traceback.format_exc()[-6000:]})
                return
            continue
        try:
            response = _infer(
                request, state, config,
                on_boxes_2d=lambda detections: connection.send({
                    "ok": True, "phase": "boxes_2d", "detections": detections,
                }),
            )
            connection.send({"ok": True, **response})
        except Exception:
            connection.send({"ok": False, "error": traceback.format_exc()[-6000:]})


if __name__ == "__main__" and len(sys.argv) == 4 and sys.argv[1] == "--worker":
    worker_main(sys.argv[2], sys.argv[3])
