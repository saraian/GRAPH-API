"""Regenerate a compiler-validated dynamic script from staged scene inputs."""
import argparse
import json
from pathlib import Path

import habitat_sim

from .prepare_cohort import changes
from .select_scenes import inspect as inspect_scene, module


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--scene', type=Path)
    parser.add_argument('--navmesh', type=Path)
    parser.add_argument('--dataset-config', type=Path)
    parser.add_argument('--schedule', type=Path)
    parser.add_argument('--scene-id')
    parser.add_argument('--floor-index', type=int)
    parser.add_argument('--allow-multifloor', action='store_true')
    parser.add_argument('--action-laps', type=int, default=1,
                        help='Distribute object lifecycles over this many complete tour laps')
    parser.add_argument('--exclude-target', action='append', default=[],
                        help='Exclude a support rejected by the exact semantic render preflight')
    parser.add_argument('--objects-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.action_laps < 1:
        parser.error('--action-laps must be positive')
    args.output.mkdir(parents=True, exist_ok=False)
    compiler = module(args.root/'lost3dsg/FOUND-Dataset/scene_script.py', 'dynamic_schedule_compiler')
    runner = module(args.root/'lost3dsg/src/perception_module/script_runner.py', 'dynamic_schedule_runner')
    if args.scene_id:
        dataset = args.root/'lost3dsg/FOUND-Dataset'
        tag = args.scene_id.split('-')[1]
        scene = dataset/'habitat/hm3d-val-habitat-v0.2'/args.scene_id/(tag+'.basis.glb')
        def prepare(sim, levels):
            return changes(sim, levels, scene, args.output, compiler, runner,
                           dataset/'habitat/habitat_objects/configs',
                           action_laps=args.action_laps,
                           excluded_targets=args.exclude_target)
        result = inspect_scene(args.root, args.scene_id, args.output,
                               allow_multifloor=args.allow_multifloor,
                               prepare=prepare, project_goals=True,
                               selected_floor_index=args.floor_index)
        (args.output/'plan.json').write_text(json.dumps(result, indent=2))
        return
    required = [args.scene, args.navmesh, args.dataset_config, args.schedule]
    if any(path is None for path in required):
        parser.error('direct mode requires --scene, --navmesh, --dataset-config and --schedule')
    schedule = json.loads(args.schedule.read_text())['schedule']
    backend = habitat_sim.SimulatorConfiguration()
    backend.scene_id = str(args.scene.resolve())
    backend.scene_dataset_config_file = str(args.dataset_config.resolve())
    backend.enable_physics = True
    backend.requires_textures = True
    agent = habitat_sim.agent.AgentConfiguration()
    with habitat_sim.Simulator(habitat_sim.Configuration(backend, [agent])) as sim:
        if not sim.pathfinder.load_nav_mesh(str(args.navmesh)):
            raise ValueError('Cannot load staged native navmesh')
        plans = changes(sim, schedule, args.scene, args.output, compiler, runner,
                        args.objects_dir, action_laps=args.action_laps,
                        excluded_targets=args.exclude_target)
    (args.output/'dynamic-plans.json').write_text(json.dumps(plans, indent=2))


if __name__ == '__main__':
    main()
