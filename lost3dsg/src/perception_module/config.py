#!/usr/bin/env python3
"""Central runtime configuration for the perception module.

Values come from config.yaml next to this file (override the location with the
GRAPH_API_CONFIG env var). Anything missing from the yaml falls back to the
defaults below, which reproduce the historical hardcoded behaviour — so a
checkout with no config.yaml at all runs exactly as before.
"""
import os

_DEFAULTS = {
    # pre-existing flag (this module used to contain only this line)
    "simulation": False,
    "vlm": {
        "base_url": "http://localhost:11434/v1",
        "model": "gemma4:e2b",
        # empty -> use OPENAI_API_KEY env if set, else the legacy api.txt next
        # to cv_utils.py if present, else "ollama" (local server ignores it)
        "api_key": "",
        "timeout": 30.0,
        "retries": 2,
        "crop_concurrency": 4,
        "crop_timeout": 15.0,
        # non-empty -> detection falls back to this static label list (with a
        # warning) when the VLM is unreachable, instead of failing the cycle
        "fallback_labels": [],
    },
    "embedding": {
        "word2vec_path": "/root/gensim-data/word2vec-google-news-300/word2vec-google-news-300.gz",
        "word2vec_limit": 200000,
    },
    # lost_similarity weights; must sum to 1.0
    "similarity": {"label": 0.05, "color": 0.30, "material": 0.15, "description": 0.50},
    "association": {
        "exploration_iou_threshold": 0.10,
        "sim_threshold": 0.85,
        "tracking_iou_threshold": 0.3,
        "volume_expansion_ratio": 0.01,
        "exploration_frame_limit": 10,
        "object_stability_timeout": 3.0,
        "pov_scale_factor": 1.0,
        "max_volume_threshold": 0.5,
        "bbox_reduction_ratio": 0.30,
        # tracking-mode match gate: candidates farther than this (bbox centres)
        # are never merged. 0 = disabled (historical behaviour)
        "max_match_distance_m": 0.0,
        # moving-pass proposals cannot mutate or fuse into established boxes
        "confirm_stationary": True,
    },
    "frames": {
        # The frame the perception back-projects into. Must be an OPTICAL frame
        # (x right, y down, z forward). Publishing a body pose under this name
        # puts depth into the height axis — see habitat_camera_node.py, which
        # publishes habitat_camera (body) -> habitat_camera_optical.
        "camera": "habitat_camera_optical",
    },
    "paths": {
        "operations_log": "/root/exchange/output/operations.txt",
        # empty -> <this package>/utils/l2_{encoder,decoder}.onnx
        "vitsam_encoder": "",
        "vitsam_decoder": "",
        # empty -> <this package>/prompts/<name>.txt
        "identification_prompt": "",
        "visual_prompt": "",
    },
    "tf": {
        # seconds to wait for a transform lookup before giving up on the frame
        "lookup_timeout": 0.1,
    },
    "rooms": {
        # non-empty -> objects detected before any room polygon exists are
        # assigned to this room instead of being rejected. Empty (default)
        # keeps the strict behaviour: no room known -> AddObject refuses.
        "default_room_id": "",
    },
    "habitat": {  # HM3D (Matterport) scenes; defaults = previous hardcoded values
        "scene": "/root/exchange/lost3dsg/habitat/hm3d-val-habitat-v0.2/00801-HaxA7YrQdEC/HaxA7YrQdEC.basis.glb",
        "scene_dataset": "/root/exchange/lost3dsg/habitat/hm3d-val-habitat-v0.2/hm3d_annotated_basis.scene_dataset_config.json",
        "nav_scene": "/root/exchange/lost3dsg/habitat/hm3d-val-habitat-v0.2/00802-wcojb4TFT35/wcojb4TFT35.basis.glb",
        "nav_scene_dataset": "/root/exchange/lost3dsg/habitat/hm3d-val-semantic-configs-v0.2/hm3d_annotated_basis.scene_dataset_config.json",
        "nav_navmesh": "/root/exchange/lost3dsg/habitat/hm3d-val-habitat-v0.2/00802-wcojb4TFT35/wcojb4TFT35.basis.navmesh",
        "width": 640,
        "height": 480,
        # keep the agent on the floor it starts on: a 2D grid cannot separate
        # storeys, and HM3D navmeshes join them through the stairs
        "single_floor": True,
        "floor_tolerance_m": 0.5,
        "mapping_seconds": 150.0,
        "walk_frames": 6,
        "dwell_frames": 60,
        "fps": 3.0,
    },
    # extension seam (see hooks.py): empty = the pass-through blueprints
    "hooks": {
        "search_paths": [],
        "filter": "",
        "refiner": "",
        "store": "",           # subclass of hooks.Store; empty -> SQLite temporal map
        "decisions_log": "",   # empty -> <package>/output/hook_decisions.jsonl
    },
    "perception": {
        "backend": "local",  # "modal", "managed", "local"
        "modal_endpoint": "",  # e.g. "https://<user>--lost3dsg-perception-predict.modal.run"
        "score_threshold": 0.15,
        "nms_threshold": 0.50,
        "reachability_strict": False,
        # allow detection passes while moving, proposals marked as unconfirmed
        "detect_while_moving": False,
    },
}


def _merge(base, override):
    out = dict(base)
    for k, v in (override or {}).items():
        out[k] = _merge(base[k], v) if isinstance(base.get(k), dict) and isinstance(v, dict) else v
    return out


def _load():
    path = os.environ.get(
        "GRAPH_API_CONFIG",
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.yaml"),
    )
    if not os.path.exists(path):
        return dict(_DEFAULTS)
    import yaml
    with open(path) as f:
        return _merge(_DEFAULTS, yaml.safe_load(f) or {})


CFG = _load()

# Backward compatibility: utils.py does `import config` / `config.simulation`.
simulation = CFG["simulation"]


if __name__ == "__main__":
    assert CFG["vlm"]["model"], CFG
    assert abs(sum(CFG["similarity"].values()) - 1.0) < 1e-6, CFG["similarity"]
    assert _merge({"a": {"b": 1, "c": 2}}, {"a": {"b": 9}}) == {"a": {"b": 9, "c": 2}}
    print("config OK:", {k: (list(v) if isinstance(v, dict) else v) for k, v in CFG.items()})
