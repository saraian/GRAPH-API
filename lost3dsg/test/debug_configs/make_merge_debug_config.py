#!/usr/bin/env python3
"""Build the merge-lane debug config FROM the tracked config.yaml plus only the GA-493 debug
overrides. A config passed as GRAPH_API_CONFIG REPLACES config.yaml, so the old
ga493_bbox_replay.yaml (a full copy taken before 2026-09-14) silently carried the legacy merge
engine and the old similarity weights; this generator keeps every merge/association/similarity
key exactly as the branch's config.yaml states them and prints the parsed difference.

    python3 lost3dsg/test/debug_configs/make_merge_debug_config.py [--engine legacy]
"""
import argparse
import copy
import os
import sys

import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
TRACKED = os.path.join(REPO, "lost3dsg", "src", "perception_module", "config.yaml")

OVERRIDES = {
    "simulation": False,
    "run": {"cap_min": 5, "gt_semantic": False, "rviz": False, "show": False, "overlay": False,
            "map_draw": False, "wall_detector": True, "pose_source": "simulator"},
    "perception": {"backend": "modal", "modal_endpoint": "", "cloud_timeout_s": 60.0,
                   "provider": "fal", "containment_threshold": 0.85, "fov_max_depth_m": 1.8,
                   "vlm_strikes_max": 3},
    "perception_parallel": {"enabled": True, "bbox_backend": "cuda", "bbox_cuda_devices": [0]},
    "archive": {"per_detection": False},
    "hooks": {"filter": ""},
    # qwen3.8-27b on Regolo, SERVED THE WAY THE DGX LANE SERVES IT (their message 00002, owner
    # ruled). MEASURED on this workstation over four runs: the VLM round trip is a median 17-23 s
    # and reaches 31 s, against 5.9 s for gemma4-31b in the GA-493 reference. 60 s left no margin
    # above the measured maximum; the DGX raised it to 120 s.
    "vlm": {"timeout": 120.0, "crop_concurrency": 4, "crop_timeout": 15.0, "grid_cells": 0},
    # WITHOUT THESE THE RUN DIES AT HALF A TOUR. object_manager_6's producer-silence watchdog
    # calls a live producer dead after input_silence_max_strikes consecutive silent checks of
    # input_silence_timeout_s each. The defaults are 60 s x 3 = 180 s, and a qwen cycle takes
    # 19 s median with a 31 s tail, so a slow stretch reads as a dead producer and the node
    # calls os._exit(1). The DGX lane hit exactly this at scan 31 of 60 and raised the three
    # keys to 180.0 / 3 / 6 (their message 00002, item 5). Made EXPLICIT here, not left to the
    # module default, so the bundle records what was in force.
    "association": {"input_silence_timeout_s": 180.0, "input_silence_max_strikes": 3,
                    "input_silence_min_stops": 6},
    # THE MERGE SWEEP RUNS ON A TIMER, NOT ONLY ON A SCAN STOP. MEASURED on the 15-minute
    # baseline 20260915_000120: TWO sweeps in 974 s, the first seeing 0 objects and the second
    # 34; 0 of 82 offered pairs was ever scored a second time, so merge_min_consecutive = 2
    # was unpayable BY CONSTRUCTION and the run applied 0 merges. The machinery already
    # existed -- object_manager_6's TIAGO_MERGE_INTERVAL_S reads merge.periodic_interval_s --
    # and shipped at 0.0, i.e. off, with only the Tiago YAML ever setting it.
    #
    # 5.0 s IS THE TIAGO VALUE, the only precedent in the tree. It is NOT measured for this
    # scene and must be swept before it is defended; the sweep now records its own duration
    # so the cost is a number rather than a belief.
    #
    # SAFE ONLY BECAUSE OF THE 2026-09-15 STREAK CHANGE. Before it, `decide` advanced the
    # streak on every update, so a 5 s timer would have let two IDENTICAL measurements satisfy
    # merge_min_consecutive = 2 -- frequency buying persistence, which is the repetition
    # defect `Hypothesis` already fixed for the total. The streak now advances only when the
    # evidence changes, so raising the frequency cannot manufacture certainty.
    "merge": {"periodic_interval_s": 5.0},
}


def merge(base, over):
    out = copy.deepcopy(base)
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = merge(out[k], v)
        else:
            out[k] = v
    return out


def flat(d, p=""):
    out = {}
    for k, v in d.items():
        if isinstance(v, dict):
            out.update(flat(v, p + k + "."))
        else:
            out[p + k] = v
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", choices=["evidence", "legacy"], default=None,
                    help="override association.merge_engine (default: as config.yaml)")
    ap.add_argument("--cap-min", type=int, default=5)
    a = ap.parse_args()
    base = yaml.safe_load(open(TRACKED))
    over = copy.deepcopy(OVERRIDES)
    over["run"]["cap_min"] = a.cap_min
    if a.engine:
        # UPDATE, never assign. Assigning replaced the whole `association` override block and
        # silently dropped the input_silence keys above -- which are the ones that keep the run
        # alive. A generator that loses a key it was given is the same defect class as a config
        # snapshot that goes stale.
        over.setdefault("association", {})["merge_engine"] = a.engine
    cfg = merge(base, over)
    engine = cfg["association"]["merge_engine"]
    out = os.path.join(HERE, f"merge_debug_00824_{engine}.yaml")
    fb, fc = flat(base), flat(cfg)
    diff = {k: (fb.get(k), fc.get(k)) for k in sorted(set(fb) | set(fc)) if fb.get(k) != fc.get(k)}
    header = ("# GENERATED by make_merge_debug_config.py from lost3dsg/src/perception_module/config.yaml\n"
              "# (merge-algorithm lane, 2026-09-14). Do not edit; regenerate. A GRAPH_API_CONFIG file\n"
              "# REPLACES config.yaml, so everything not listed below is the tracked value verbatim.\n"
              "# KEYS THAT DIFFER FROM THE TRACKED config.yaml:\n")
    for k, (b, c) in diff.items():
        header += f"#   {k}: {b!r} -> {c!r}\n"
    with open(out, "w") as fh:
        fh.write(header)
        yaml.safe_dump(cfg, fh, sort_keys=False, default_flow_style=False, width=100)
    print(out)
    for k, (b, c) in diff.items():
        print(f"  {k}: {b!r} -> {c!r}")
    for key in ("association.merge_engine", "association.merge_ontology_channel",
                "association.merge_attribute_max_log_odds", "association.association_margin_m",
                "similarity.label", "similarity.color", "similarity.material", "similarity.description",
                "vlm.model", "vlm.enable_thinking"):
        print(f"  keep {key} = {fc.get(key)!r}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
