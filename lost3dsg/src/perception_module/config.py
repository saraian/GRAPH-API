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
        # GA-209. Crops per VLM request. 0 or 1 keeps the previous behaviour, one call per
        # crop at crop_concurrency. Measured 2026-09-01: five separate calls at concurrency
        # 8 took 1.69 s, one 3x2 grid took 0.88 s -- 1.91x, by removing the straggler rather
        # than by tuning around it. Above ~6 the cells fall below 224 px and the describer
        # starts losing detail, so crop_grid batches instead of building one large grid.
        # MEASURED OFF, 2026-09-03. The grid was benchmarked at 1.41x against SEQUENTIAL
        # per-crop calls -- but the system runs them CONCURRENTLY at crop_concurrency 8, and
        # my own probe script says so in a comment I then ignored. A number measured under a
        # different execution model is not a prediction about this system.
        #
        # What it cost, on run 20260903_110622: the BLOCKING whole-image label call went from
        # a 2.11 s median (n=26) to 11.03 s (n=16), 5.2x. The grid does not slow the crop
        # describer -- that is asynchronous and never appears in vlm_ms. It SATURATES THE
        # SHARED VLM PROVIDER with 4 large multi-cell images at ~5 s each, and the blocking
        # label call queues behind them.
        #
        # The straggler the grid was meant to remove is real (one call in five at 13.7 s
        # against a 0.55 s median) but costs less than paying 5 s every cycle to insure
        # against it. Set to 6 to re-enable; the code is unchanged and self-checking.
        "grid_cells": 0,
        "base_url": "http://localhost:11434/v1",
        "model": "gemma4:e2b",
        # empty -> use OPENAI_API_KEY env if set, else the legacy api.txt next
        # to cv_utils.py if present, else "ollama" (local server ignores it)
        "api_key": "",
        "timeout": 30.0,
        "retries": 2,
        "crop_concurrency": 4,
        "crop_timeout": 15.0,
        # GA-53: `fallback_labels` stood here. Non-empty, it replaced an unreachable
        # VLM with a static open-vocabulary list and let the cycle continue — working
        # rule 14's third shape, a run reporting success on behalf of something that
        # never ran. WORKING_RULES.md already lists a fallback label list among the
        # substitutions found and removed; that removal was in the other checkout, and
        # this is the tree that runs. A yaml that still sets the key now sets nothing.
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
        # MASTER SWITCH for the evidence-accumulating association (association.py).
        #   legacy   -> the shipped gate chain in _cb_merge_objects. Unchanged, and still
        #               the only path that can merge anything.
        #   evidence -> the six-channel log-odds path.
        # DEFAULT IS legacy, and that is not timidity. Replayed against real bundles the
        # evidence path, WITHOUT observation records, proposed merging pillows INTO THE BED
        # they rest on (containment 0.850) -- containment cannot tell a fragment of X from
        # an object resting on X. It is correct only once co-visibility is available.
        # The old path is never deleted: both must run against one bundle to be compared.
        "mode": "legacy",
        # False-merge cost / missed-merge cost; the commit threshold is log(ratio). A
        # POLICY, not a measurement -- D14 prefers strict because a duplicate is visible and
        # repairable while a wrong merge destroys an identity.
        "cost_ratio": 20.0,
        # Consecutive updates a decision must hold before committing. Persistence, not
        # summation: re-measuring unchanged geometry is not new evidence.
        "min_consecutive": 3,
        # GA-101: how many of the three OPTIONAL terms (colour, material, description) must
        # have been comparable before a merge is allowed. With none of them the divisor is
        # the label weight alone, so two identical labels score exactly 1.0000 -- above any
        # threshold, on zero measured evidence. 0 restores the old behaviour.
        "merge_min_evidence": 1,
        # GA-289. Tracking path (check_tracking_transition): a persistent object with no
        # position covariance yet has no measurable shell, so its search reach is its own
        # extent plus this radius. Objects WITH a covariance use association.search_radius
        # and never read this. `merge_min_evidence` above governs the tracking win as well.
        # Category (c), stated policy: 1.0 m is where 95% of the day's tracking comparisons
        # already sat beyond, and 2.3x the nearest recorded loser (0.355 m).
        "tracking_fallback_radius_m": 1.0,
        # GA-83 / GA-94: input-starvation watchdog. Seconds of /bbox_3d silence per check,
        # and consecutive silent checks before the node ends the run.
        "input_silence_timeout_s": 60.0,
        "input_silence_max_strikes": 3,
        # GA-94b. Robot STOPS that must pass with no detection before the producer is called
        # dead. Detection is gated on the robot stopping, so with dwell=0 a two-minute
        # silence is a normal gap between incidental halts -- run 042828 had 4 cycles and 3
        # stops in 9 minutes, and the time-only guard ended a healthy run. Stops are the unit
        # in which "the producer had its chance and did not take it" is measurable.
        "input_silence_min_stops": 3,
        "tracking_iou_threshold": 0.3,
        "volume_expansion_ratio": 0.01,
        "exploration_frame_limit": 10,
        "object_stability_timeout": 3.0,
        "pov_scale_factor": 1.0,
        "max_volume_threshold": 0.5,
        "bbox_reduction_ratio": 0.30,
        # GA-49: `max_match_distance_m` stood here, documented as the tracking match
        # gate with "0 = off". Nothing has read it since GA-04 replaced it, so the key
        # and its comment were a false statement about the system — a comment that
        # outlived its code, in a file an operator edits to change behaviour.
        # Its replacement is read in object_manager_6.py through a `.get` fallback of
        # 2.0 and was declared nowhere, so the value in force could not be seen in any
        # config file. The default below is that same 2.0: declaring the key documents
        # the behaviour, it does not change it.
        # Radius, in metres, within which a changed object triggers re-evaluation of
        # its neighbours.
        "reevaluation_radius_m": 2.0,
        # GA-11. One object churning must not flood the second-look queue, and one update must not
        # fan out to the whole room. Debounce: an object re-queued within this many seconds of its
        # last queueing is skipped (cost: a genuine second change inside the window is examined
        # once, not twice). Fan-out: at most this many neighbours are queued per update (cost: in a
        # dense room the farthest neighbours are not re-examined on that event).
        "reevaluation_debounce_s": 2.0,
        "reevaluation_max_fanout": 12,
    },
    # GA-270. SERVICE ADDRESSES BELONG IN CONFIG, not in module literals.
    #
    # Two separate outages this session came from the same shape of defect: a hardcoded
    # "http://127.0.0.1:8081" in habitat_feed_host (the belief poller reached nothing, so the
    # Habitat window drew no boxes for a whole run) and another in object_manager_6 (every
    # Graph API POST failed, so 4 admit decisions minted 0 objects and the run ended with an
    # empty map). The bridge moves to 8091 whenever 8081 is taken, which it is on this
    # machine -- and neither literal moved with it.
    #
    # A resource address is configuration. It is stated once, recorded in the bundle with the
    # rest of the config, and every consumer reads it from here. The env var still overrides
    # for a single run; what is gone is the module-level default that nothing can reach.
    "walls": {
        # A wall does not move, so re-fitting one per depth frame buys nothing and costs a
        # core. At 30 fps the detector demanded ~5 cores and starved rtabmap; this is the
        # knob that made it affordable, not the RANSAC speed-up.
        "min_interval_s": 0.5,
        "ransac_iters": 60,
        "max_segments": 12,
    },
    "cloud": {
        # GA-278. Scale applied to the frame BEFORE it is sent to the perception backend.
        # The request is bandwidth-bound to ~1.76 s; 0.75 captures the whole saving (-21%
        # request time) at -4% detections across six frames, which is inside the per-frame
        # variance. 1.0 sends the frame untouched, exactly as before this knob existed.
        "send_scale": 0.75,
    },
    "services": {
        "bridge_host": "127.0.0.1",
        "bridge_port": 8081,
        "feed_host": "127.0.0.1",
        "feed_port": 7790,
    },
    "frames": {
        # GA-236. Frames the motion detector watches. Default is the Habitat camera
        # ALONE, because that is the only one that exists here: the previous literal
        # named two TIAGo head joints and two wheel joints, none of which resolve in
        # this deployment, and their absence was reported as four lookup failures a
        # second rather than once. A TIAGo deployment sets these to its own frames.
        "motion_watch": ["habitat_camera"],
        "motion_watch_base": [],
        # The frame the perception back-projects into. Must be an OPTICAL frame
        # (x right, y down, z forward). Publishing a body pose under this name
        # puts depth into the height axis — see habitat_camera_node.py, which
        # publishes habitat_camera (body) -> habitat_camera_optical.
        "camera": "habitat_camera_optical",
    },
    "paths": {
        "operations_log": "/root/exchange/output/operations.txt",
        # GA-166. THE RUN'S OUTPUT DIRECTORY -- what the bundle collector reads. The
        # per-detection archive resolves here, and NOT from operations_log's directory: a
        # log's directory is not where data goes, and deriving one from the other put an
        # entire run's archive in /tmp.
        "output_dir": "/root/exchange/output",
        # empty -> <this package>/utils/l2_{encoder,decoder}.onnx
        "vitsam_encoder": "",
        "vitsam_decoder": "",
        # empty -> <this package>/prompts/<name>.txt
        "identification_prompt": "",
        "visual_prompt": "",
    },
    "crop": {
        # How the image handed to the describer AND the embedder is built. The A/B arms ARE
        # this switch: one code path, one setting, no parallel implementation.
        #   tight          -> today's behaviour EXACTLY: the detector box, no padding, no
        #                     mask, no contour. The default, so nothing changes silently.
        #   padded         -> mask bbox + adaptive context window (absolute floor included)
        #   contour        -> padded + mask contour + background dimmed (never blacked)
        #   contour_metric -> contour, plus the metric sentence in the TEXT prompt
        #                     (same pixels as `contour`; the difference is the words)
        # DEFAULT IS contour, by owner ruling 2026-09-01 (interactive prompt ~02:00,
        # "contour only"). contour_metric is HELD until box error is measured against
        # GROUND TRUTH rather than against itself: on sub-litre objects the box's own
        # longest side disagrees with itself by a median 2.94x (337 near same-label pairs),
        # monotonically worse as objects shrink, over the same population that produces 59%
        # of the unknown descriptions. contour asserts NOTHING and so cannot mislead; the
        # metric arm would inject its most confident falsehoods exactly where the describer
        # already fails. The self-limiting machinery in crop_context is the honest form of
        # that option, NOT a route back to shipping it -- do not flip this casually.
        "construction": "contour",
        # RESOLUTION IS PER-CONSUMER. One construction, two renderings: identical geometry
        # -- same window, same contour, same dimming -- rendered at whatever each consumer
        # can actually use. Sharing the window was the point; sharing the pixel count was
        # never required by it.
        #
        # At a single fixed 224 the pipeline did both wrong things at once on a 640x480
        # feed: a large object's window is several hundred source px wide, so 224 DISCARDS
        # real detail; the sliver's window is floored at 120 px, so 224 INVENTS pixels the
        # sensor never captured.
        #
        # 224 for the embedder because that is DINOv2's patch grid (16 patches of 14) and a
        # different size buys nothing.
        "embedder_size": 224,
        # 0 -> the describer gets the window at NATIVE source resolution, unresampled.
        # A positive value caps it. COST OF RAISING THIS: more tokens per crop, more upload
        # per cycle, and more latency on a path that already carries ~42 s/cycle of
        # unexplained overhead -- so it is a config value the run can carry as a measured
        # arm rather than an assumption.
        "describer_size": 0,
    },
    "reid": {
        # Explicit enable for the visual re-ID channel. Implicit-by-wiring is how a
        # component becomes impossible to sever from a configuration it should not be in.
        "enabled": False,
        # facebook/dinov2-small (ViT-S/14). ViT-B measured at ~350 MB fp16 on a card shared
        # with the simulator; ViT-S runs in 23 ms and ~138 MiB.
        "model_id": "facebook/dinov2-small",
        # Half-angle of the comparability cone, degrees. Two views outside it are NOT
        # comparable, and the channel ABSTAINS rather than reporting "dissimilar".
        "cone_half_angle_deg": 45.0,
        # Descriptors kept per object, selected for BEARING DIVERSITY rather than recency.
        "max_views": 8,
    },
    "archive": {
        # Per-detection archiving: frame id + the RGB frame + the 2D box + the per-detection
        # mask + the crop construction meta. OFF by default -- a frame per cycle is not free.
        #
        # It is what makes CO-VISIBILITY answerable, and co-visibility is what stops the
        # containment channel merging a pillow into the bed it rests on (measured: ten such
        # pairs in run 20260831_200858, containment up to 0.850). It also ends "the merge
        # destroys its own diagnostic evidence" -- in run 20260831_022033, 100% of merge
        # decisions became unauditable because the objects were gone.
        "per_detection": False,
        # empty -> the directory holding operations_log
        "dir": "",
    },
    "tf": {
        # seconds to wait for a transform lookup before giving up on the frame
        "lookup_timeout": 0.1,
        # GA-95: the TF buffer's cache window. A frame whose stamp is older than this can
        # never be transformed again -- the data has been evicted -- so it is dropped
        # rather than retried. Must match the Buffer(cache_time=...) in perception_2.
        # GA-284. RAISED 30 -> 90, and the reason is a coupling the comment above states as a
        # rule that nothing enforced. `max_frame_age_s` went 5.0 -> 15.0 (GA-281, to break a
        # deadlock where the cycle outlived its own freshness window), which lets a frame wait
        # three times longer before it is transformed. On run 20260903_123748 that produced
        # 7 `habitat_camera_optical->map` extrapolation failures and 16 agent-pose failures --
        # lookups a median 6.4 s, max 31.3 s BEFORE the oldest data still in the buffer.
        #
        # A BOX BUILT ON A FAILED TRANSFORM LANDS BESIDE ITS OBJECT. That is visible in the
        # feed overlay as boxes offset from the furniture they describe, and it is not a
        # projection bug: the overlay projects with the CURRENT pose onto the CURRENT frame,
        # correctly. The world coordinates were wrong before they ever reached it.
        #
        # THE STAMPS WERE NEVER THE PROBLEM. utils.py:109 already looks the transform up at
        # `cached_rgb.header.stamp`, the frame's own timestamp, which is exactly right. You
        # cannot look up a time that has been EVICTED, however precisely you name it.
        #
        # 90 s is 6x the frame window, so a frame that survives max_frame_age can always be
        # transformed. Cost is memory for TF history, which is small. preflight probe a11 now
        # refuses a run where this does not exceed max_frame_age_s with margin.
        "buffer_cache_s": 90.0,
    },
    "rooms": {
        # GA-137. How the GVD skeleton is built.
        #   label_diff -> today's behaviour: mark pixels whose neighbours have different
        #                 nearest-obstacle COMPONENT ids. Provably empty on any floorplan
        #                 whose walls are connected -- which is every floorplan -- so the
        #                 map is never split and every object lands in one room.
        #   ridge      -> the medial axis as a ridge of the distance transform. Works
        #                 regardless of obstacle connectivity.
        # DEFAULT IS ridge, by owner ruling 2026-09-01 (rulings doc item 5, "ON BY DEFAULT").
        # CAVEAT THAT MUST TRAVEL WITH ANY ROOM COUNT FROM THE NEXT RUN: `ridge` is proven
        # on a SYNTHETIC floorplan where the answer was known -- skeleton_px 22278 against
        # label_diff's 0, with the doorway a clean clearance minimum. Whether it segments a
        # REAL occupancy grid into sensible rooms is what the pre-validation run measures.
        # On-by-default is a decision to measure it, not a claim that it works.
        "gvd_method": "ridge",
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
        # GA-258. DWELL MODE, dynamic by default. A fixed dwell is wrong in both
        # directions: it wastes frames when no merge is waiting to be confirmed, and leaves
        # before confirmation when one is. Dynamic dwell asks the object manager what is
        # pending and stays while the answer is above zero, between the bounds below.
        # Set dwell_dynamic false for a fixed-length dwell -- which is what a clean
        # one-variable ablation of merge_min_consecutive needs, since dynamic dwell makes
        # duration co-vary with the parameter under test.
        "dwell_dynamic": True,
        "dwell_min_frames": 8,
        "dwell_max_frames": 90,
        "tour_waypoints": 0,
        "tour_scan_frames": 12,

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
    "visualization": {
        "strict_visibility": False,   # true -> min_visible_points defaults to 5
        "min_visible_points": 0,      # 0 -> 5 if strict else 1; 1 = draw partial (thin), 8 = full
        "depth_tol_abs": 0.10,        # depth-buffer tolerance, metres
        "depth_tol_rel": 0.05,
    },
    "perception": {
        # GA-276. IoU-NMS cannot see a nested box: fully contained at a 5x size
        # difference gives IoU 0.2. Suppress on IoS (intersection over the SMALLER box)
        # as well. Measured 61 fully-contained same-class pairs in one run, every one
        # with IoU < 0.5. Set to 1.01 to disable containment suppression entirely.
        "containment_threshold": 0.85,
        # GA-288. Consecutive perception cycles whose VLM label call FAILED before the run is
        # ended. Mirrors association.input_silence_max_strikes (om6's watchdog), which the
        # owner approved for the same shape. One failed cycle is skipped LOUDLY and counted;
        # this many in a row means the VLM is gone, not blinking. 0 disables the guard and
        # restores crash-on-first-failure.
        "vlm_strikes_max": 3,
        # GA-164. Oldest frame get_synced_data will accept, seconds. Was a hardcoded 1.0
        # that no config could reach. At 1280x960 frames arrived a MEDIAN 7.14 s stale and
        # this check rejected all 1650 of them, so ZERO perception cycles ran in 17 minutes.
        # The backlog itself is fixed by QoS depth=1 (see utils.py); this stays 1.0 so the
        # freshness guarantee is unchanged, and is now a knob rather than a literal.
        "max_frame_age_s": 1.0,
        "backend": "local",  # "modal", "managed", "local"
        "modal_endpoint": "",  # e.g. "https://<user>--lost3dsg-perception-predict.modal.run"
        "score_threshold": 0.15,
        "nms_threshold": 0.50,
        "reachability_strict": False,
        # allow detection passes while moving, proposals marked as unconfirmed
        "detect_while_moving": False,
        # Socket timeout for a remote perception call, seconds. Was hardcoded at
        # 25.0 at one call site; the testing lane measured seven Modal cold starts
        # on 2026-08-30/31 at 25.3, 42.3, 45.8, 41.1, 26.6, 47.7 and 45.6 s, so
        # every one exceeded it and the first call after a cold container could
        # not succeed. The default below clears the measured maximum; raise it
        # here rather than in code if the distribution moves.
        "cloud_timeout_s": 60.0,
    },
}


def _merge(base, override):
    out = dict(base)
    for k, v in (override or {}).items():
        out[k] = _merge(base[k], v) if isinstance(base.get(k), dict) and isinstance(v, dict) else v
    return out


def _load():
    """Return (config, path_actually_read). The path is None when no file was found
    and the defaults are in force.

    GA-52: this computed `path` as a local and discarded it, so no artefact recorded
    which yaml a run had loaded or what value was in force, and no care taken at run
    time could recover it afterwards. A path that merely EXISTS does not prove it is
    the one that was read, which is why the "not found" case returns None rather than
    the location that was searched.
    """
    path = os.environ.get(
        "GRAPH_API_CONFIG",
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.yaml"),
    )
    if not os.path.exists(path):
        return dict(_DEFAULTS), None
    import yaml
    with open(path) as f:
        return _merge(_DEFAULTS, yaml.safe_load(f) or {}), path


# CFG_PATH is the file that was read, or None when the defaults are in force. A
# consumer that logs the configuration must log this beside the values, or the two
# arms of an ablation are indistinguishable in the bundle.
CFG, CFG_PATH = _load()

# Backward compatibility: utils.py does `import config` / `config.simulation`.
simulation = CFG["simulation"]


def visibility(live=None):
    """Resolve the overlay visibility gate to ``(min_visible_points, tol_abs, tol_rel)``.

    `live` is the simulator host's /set_config state (habitat_feed_host.CTRL.config),
    which wins over the yaml. Those values arrive as URL query strings — "true", "5" —
    so nothing here may assume a real bool/int came out of the yaml.
    """
    viz = {**_DEFAULTS["visualization"], **(CFG.get("visualization") or {}), **(live or {})}

    def _num(key, cast, default):
        try:
            return cast(viz.get(key, default))
        except (TypeError, ValueError):
            return default

    strict = str(viz.get("strict_visibility", False)).strip().lower() in ("true", "1", "yes")
    # The host seeds min_visible_points on startup, so it is never absent and a
    # "default when missing" would make the strict toggle a no-op. Strict is a floor
    # instead: at least 5 of the 9 projected corners, and the slider may only raise it.
    min_vis = _num("min_visible_points", int, 0) or 1
    return (max(min_vis, 5) if strict else min_vis,
            _num("depth_tol_abs", float, 0.10),
            _num("depth_tol_rel", float, 0.05))


if __name__ == "__main__":
    assert CFG["vlm"]["model"], CFG
    assert abs(sum(CFG["similarity"].values()) - 1.0) < 1e-6, CFG["similarity"]
    assert _merge({"a": {"b": 1, "c": 2}}, {"a": {"b": 9}}) == {"a": {"b": 9, "c": 2}}
    # visibility(): the host sends query STRINGS, and they must win over the yaml
    assert visibility({"min_visible_points": "7"})[0] == 7
    assert visibility({"strict_visibility": "false"})[0] == 1
    assert visibility({"depth_tol_abs": "0.25"})[1] == 0.25
    assert visibility({"min_visible_points": "junk"})[0] == 1       # garbage never raises
    assert visibility()[1:] == (0.10, 0.05)
    # strict is a floor, not a default: it must bite even though the host always
    # seeds min_visible_points, and the slider may raise it but not lower it
    assert visibility({"strict_visibility": "true", "min_visible_points": "1"})[0] == 5
    assert visibility({"strict_visibility": "true", "min_visible_points": "8"})[0] == 8
    assert CFG_PATH is None or os.path.exists(CFG_PATH), CFG_PATH
    print("config OK:", {k: (list(v) if isinstance(v, dict) else v) for k, v in CFG.items()})
    print("config loaded from:", CFG_PATH if CFG_PATH else "<defaults, no file found>")
