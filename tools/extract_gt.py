"""Ground-truth inventory extractor: the simulator's own semantic scene graph.

Runs INSIDE the habitat_env (the same interpreter the feed host uses), so labels and
boxes are authoritative — no vertex-colour decoding, no guessing. For every semantic
object with a category and a non-empty AABB it records the label, centroid and extents
in BOTH frames: habitat (y-up) and the feed host's ROS convention
(habitat_pose_to_ros: [-hz, -hx, hy], z-up).

    <habitat_env python> tools/extract_gt.py hm3d_00861 [hm3d_00337 ...]

Writes data/gt/<scene>.json. Read-only w.r.t. the stack; safe while others explore.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

SCENES = {
    "hm3d_00861": ("/DATA/habitat_matterport/hm3d_example/00861-GLAQ4DNUx5U/GLAQ4DNUx5U.basis.glb",
                   "/DATA/habitat_matterport/hm3d_example/hm3d_annotated_basis.scene_dataset_config.json"),
    "hm3d_00337": ("/DATA/habitat_matterport/hm3d_example/00337-CFVBbU9Rsyb/CFVBbU9Rsyb.basis.glb",
                   "/DATA/habitat_matterport/hm3d_example/hm3d_annotated_basis.scene_dataset_config.json"),
    "hm3d_00770": ("/DATA/habitat_matterport/hm3d_example/00770-NBg5UqG3di3/NBg5UqG3di3.basis.glb",
                   "/DATA/habitat_matterport/hm3d_example/hm3d_annotated_basis.scene_dataset_config.json"),
    "mp3d_17DRP": ("/DATA/habitat_matterport/versioned_data/mp3d_example_scene_1.1/17DRP5sb8fy/17DRP5sb8fy.glb",
                   "/DATA/habitat_matterport/versioned_data/mp3d_example_scene_1.1/mp3d.scene_dataset_config.json"),
}

OUT = Path(__file__).resolve().parents[1] / "data" / "gt"


def hab_to_ros(p):
    """The feed host's exact axis convention."""
    return [-float(p[2]), -float(p[0]), float(p[1])]


def extract(scene: str) -> dict:
    import quaternion  # noqa: F401, I001  (must precede habitat_sim: registers the dtype)
    import habitat_sim as hs

    glb, cfg = SCENES[scene]
    cfg0 = hs.SimulatorConfiguration()          # single config, as the feed host builds it
    cfg0.scene_id = glb
    cfg0.scene_dataset_config_file = cfg
    cfg0.load_semantic_mesh = True
    agent_cfg = hs.agent.AgentConfiguration()
    sensor = hs.CameraSensorSpec()
    sensor.uuid = "gt_rgb"
    sensor.resolution = [64, 64]
    agent_cfg.sensor_specifications = [sensor]
    sim = hs.Simulator(hs.Configuration(cfg0, [agent_cfg]))
    try:
        scene_graph = sim.semantic_scene
        objects = []
        for obj in scene_graph.objects:
            if obj is None or obj.category is None:
                continue
            label = (obj.category.name() or "").strip()
            if not label:
                continue
            aabb = obj.aabb
            if aabb is None:
                continue
            c = aabb.center
            ext = aabb.sizes
            if not all(map(lambda v: v == v, list(ext))) or min(ext) <= 0:
                continue
            # axis mapping of habitat_pose_to_ros: hab(x,y,z) -> ros(-z,-x,y); extents
            # are unsigned so they reorder to (hab_z, hab_x, hab_y)
            objects.append({
                # GA-358. The archive's `habitat_gt_instance_id` IS Habitat's semantic_id (the
                # value the semantic sensor paints), so this is the join key; without it a
                # reader joined by LIST POSITION and matched 9 of 870. `hab_id` is Habitat's own
                # object id string, kept beside it for the scene-graph side.
                "semantic_id": int(obj.semantic_id),
                "hab_id": str(obj.id),
                "label": label.split("/")[-1].replace("frl_apartment_", ""),
                "hab_centroid": [round(float(v), 4) for v in c],
                "hab_extents": [round(float(v), 4) for v in ext],
                "pos": [round(float(v), 4) for v in hab_to_ros(c)],
                "extents": [round(float(abs(ext[2])), 4), round(float(abs(ext[0])), 4),
                            round(float(abs(ext[1])), 4)],
                "region": getattr(obj.region, "id", None),
            })
        return {"scene": scene, "source": glb, "n_objects": len(objects), "objects": objects}
    finally:
        sim.close()


def main():
    scenes = sys.argv[1:] or ["hm3d_00861"]
    OUT.mkdir(parents=True, exist_ok=True)
    for scene in scenes:
        rep = extract(scene)
        out = OUT / f"{scene}.json"
        out.write_text(json.dumps(rep, indent=1))
        print(f"{scene}: {rep['n_objects']} objects -> {out}")


if __name__ == "__main__":
    main()
