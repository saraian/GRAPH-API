# Pre-PR checklist — lost3dsg cleanup

Ordered. Item 1 gates everything below it: no further behavior-touching change
lands without a green baseline comparison.

## 1. Non-regression baseline (REQUIRED FIRST)

- [ ] On the sim machine, one trusted run: `UPDATE_BASELINE=1 lost3dsg/test/nonregression.sh`
      (HM3D scene from `config.yaml`; baseline is per scene+duration).
- [ ] A second plain run passes (`PASS`) — establishes that run-to-run variance
      fits the 15% tolerance. If it does not, add a scripted deterministic tour
      (publish `/habitat/action` from a seed) before tightening anything else.

## 2. Ruff residue, scheduled

Done already (2026-08-21): F401 (74 unused imports), F811 (duplicate imports +
`graph_api_bridge.py` route-handler rename), F541 (cosmetic f-strings), F841
(dead assignments commented, `scaled_pov_volume` bug fixed), 3× F821 real
undefined-name bugs in `publish_individual_pointclouds_by_id`.

Remaining, in order of value:

- [ ] **F403/F405 (68)** — replace `from utils/nlp_utils/cv_utils import *` in
      `object_manager_6.py` and `object_services.py` with explicit imports.
      Mechanical but wide; do it right before opening the PR, re-run the
      non-regression after.
- [ ] **E722 in `nlp_utils.py`** — narrow the bare `except` in
      `color_name_to_rgb`; same silent-failure class that hid the dead
      embedding signal.
- [ ] **I001 (38)** — import sorting, autofixable, but only with a
      non-regression run behind it (import order can carry side effects here).
- [ ] **E402 (46)** — leave; mostly deliberate (env vars / dlopen flags must be
      set before heavy imports).
- [ ] **E701/E702/E401 (46)** — style only; leave unless the maintainer asks.

## 2b. Found during the first live run (2026-08-21) — all fixed except the last

- [x] `room_manager._nearest_room` returned bare `None` on its guard path but a
      tuple elsewhere — first object before any room existed **crashed
      object_manager_6**. Fixed at the source.
- [x] Vendored `efficientvit` was incomplete upstream (`models/utils`,
      `apps/utils`, `sampler.py` missing) — `perception_2` could never import
      from a clean clone. Completed from mit-han-lab/efficientvit.
- [x] CMake installed neither the sibling modules the nodes import
      (`nlp_utils`, `world_model`, …) nor `efficientvit/` nor the viewer —
      `ros2 run` died with ImportError. Install lists fixed.
- [x] `graph_api_bridge` resolved `viewer/` and its output JSONs relative to
      CWD / a hardcoded `/root/exchange` — now `__file__`-relative and
      `GRAPH_API_OUTPUT_DIR`-overridable.
- [x] Objects were rejected forever without SLAM-derived room polygons
      ("nessuna stanza corrente nota") — added `rooms.default_room_id` config
      seam (empty default keeps strict behaviour).
- [x] rtabmap integrated into the container stack (2026-08-22) with the exact
      arguments of `launch/habitat_launch.py`; `RoomManager` consumes
      `/rtabmap/map` + `/rtabmap/cloud_map`. Two things any camera publisher
      must honour or rtabmap silently gets nothing: **RELIABLE** QoS on
      rgb/depth/camera_info/odom (rtabmap subscribes reliable and rejects
      best-effort publishers), and an `/odom` topic + `odom->base_link` TF.
      Once rooms form from the map, `rooms.default_room_id` can go back to
      strict ("").
- [x] Orientation now survives the Graph API write path (2026-08-22):
      `AddObject.srv`/`UpdateObject.srv` carry yaw/oriented_extents/
      oriented_center, the bridge forwards them, object_services attaches them
      to the bbox dict → persisted JSON → rviz oriented cubes + the habitat
      window overlay (`FEED_OVERLAY=1`).
- [x] **Association bug — the "one box" symptom.** `lost_similarity` scored
      unknown==unknown as agreement on colour, material and description, so
      any two undescribed objects reached 0.95 > 0.85 and every detection
      merged into the first node (1096 PATCHes, one persistent object).
      Unknown is now absence of evidence (weights renormalised), a missing
      description embedding no longer skips a candidate, and a tracking-mode
      spatial gate (`association.max_match_distance_m`) stops same-label
      objects across the map from merging on label alone.
- [x] rtabmap grid hygiene (`RTABMAP_GRID_ARGS`): floor/ceiling/far depth no
      longer painted as obstacles (MaxGroundHeight/MaxObstacleHeight,
      RayTracing, noise filter). Before: whole rooms black in `/rtabmap/map`.
- [x] Feed host mapping phase (`FEED_MAPPING_SECONDS`): continuous navmesh
      coverage tour with a 360° scan per waypoint before detection starts, so
      rooms are segmented from a real map rather than clamped to the first
      frames.

## 2c. Added 2026-08-22 (second live session)
- `cv_utils.box_corners_map` / `draw_boxes_3d`: the 3D boxes (PCA-oriented when present)
  projected back into the frame through the map←optical transform of that frame.
- `perception_2._publish_image_with_bb`: `/image_with_bb` now carries those wireframes
  (2D rectangle only for detections without a 3D box) and is published on **every** cycle,
  empty ones included — a late subscriber no longer sits on "No image".
- `test/habitat_feed_host.py`: tour goals confined to the start floor (`FEED_FLOOR_TOL`;
  HM3D 00861 is two-storey on one navmesh island); belief overlay depth-tested for visibility.
- Checked: synthetic projection (box 2 m ahead lands mid-frame, behind-camera box skipped);
  overlay 9/0/2 visible points for open / walled / two-thirds occluded; ruff: no new findings.
- `box_view.py` (new, ROS-free): box corners + the one visibility rule, shared by
  `cv_utils.draw_boxes_3d(…, depth)` and the host overlay. Self-check: `python3 box_view.py`.
- `hooks.py` (new, ROS-free) + object-manager seam: `Filter` / `Refiner` / `Reevaluation` /
  `DecisionLog` blueprints loaded by dotted path from config `hooks`; admission gate before
  `add_new_object`, neighbour re-evaluation queued on every node update, drained on the 2 s
  timer, revisions logged to `output/hook_decisions.jsonl` (not applied). Unconfigured =
  upstream behaviour. Self-check: `python3 hooks.py`.
- `/pcl_objects` fix: cloud published in `map` through the frame transform (was stamped with
  the body frame while optical → rendered rotated), and no longer wiped on empty cycles
  (`publish_empty_state`). Verified: centroid of a 2 m-deep mask lands at the expected map point.
- config: `habitat.single_floor` / `floor_tolerance_m`, `hooks.*`; `smoke_config.yaml` gains
  `frames.camera: habitat_camera_optical`; host feed reads config.yaml (`GRAPH_API_CONFIG`).
- `hooks.Store` + `load_store`: the map's persistence adapter as a hook (four events +
  `objects()` / `history()` reads). `MapDatabase` (SQLite) subclasses it and gains the two
  reads; `object_services` loads it from config `hooks.store` (empty = unchanged behaviour).
  Checked: events → reads round-trip on a temp DB; container compile of services/manager/bridge.
- `check_tracking_transition` still skipped candidates with a `None` description embedding (the
  third site of the unknown-description bug): with the VLM down every embedding is `None`, so
  the EXPLORATION → TRACKING transition could never fire and the stack never reached updates,
  the uncertain pool or merging. Now the same rule as the two association loops.

## 3. Before opening the PR

- [ ] Rewrite the PR document's line references against the restructured tree
      (`perception_2.py` / `object_manager_6.py`; old files now in `old/`).
- [ ] Re-run the belief-growth soak on HM3D now that the embedding signal is
      alive — the 2710-object claim was measured on the old stack and must not
      be quoted against this one.
- [x] `Bbox3d.msg` extended with `yaw` / `oriented_extents` / `oriented_center`
      (2026-08-21). Requires a `colcon build` before the next run — the old
      generated msg lacks the fields and `_publish_bbox_array` would silently
      drop them (the copy loop now guards on `hasattr`). Orientation flows
      perception → object_manager → `persistent_perception.json`, and rviz
      persistent-bbox markers draw the oriented cube when available.
- [ ] After the baseline is green: wire `pca_box.extent_similarity` (written,
      self-tested) into association behind a config flag, default off — sorted
      oriented extents match across viewpoints where world-axis AABB IoU
      cannot (the measured 4.8% merge-rate ceiling).
- [ ] Nothing is committed yet — shape the working tree into the per-concern
      commit split when the owner says go.
