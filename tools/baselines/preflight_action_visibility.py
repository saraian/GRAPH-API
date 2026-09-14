"""Render every planned action state from its fixed scheduled observation scan."""
import argparse
import json
import math
import os
from pathlib import Path

import numpy as np


def sampled_heading_indices(headings, samples):
    return np.linspace(0, headings - 1, min(samples, headings), dtype=int).tolist()


def scan_pixels(sim, agent, xyz, semantic_id, headings):
    import quaternion
    counts = []
    for index in range(headings):
        state = agent.get_state()
        state.position = np.asarray(xyz, dtype=np.float32)
        state.rotation = quaternion.from_rotation_vector(
            np.asarray([0.0, 2.0 * math.pi * index / headings, 0.0]))
        agent.set_state(state)
        semantic = np.asarray(sim.get_sensor_observations()['semantic_sensor'])
        counts.append(int(np.count_nonzero(semantic == int(semantic_id))))
    return counts


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('graph-api-root', 'scene', 'dataset', 'objects', 'schedule',
                 'script', 'config', 'output'):
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--floor', type=float, required=True)
    parser.add_argument('--headings', type=int, default=36)
    parser.add_argument('--scan-samples', type=int, default=8)
    parser.add_argument('--semantic-id-offset', type=int, default=1_000_000)
    args = parser.parse_args(argv)
    if args.headings < 4 or args.scan_samples < 1:
        parser.error('--headings must be at least 4 and --scan-samples must be positive')
    os.environ.update(
        HABITAT_SCENE=args.scene, HABITAT_DATASET=args.dataset,
        HABITAT_EXAMPLE_OBJECTS_DIR=args.objects, FEED_WIDTH='640', FEED_HEIGHT='480',
        FEED_HFOV='90', FEED_SHOW='0', FEED_GT_SEMANTIC='1',
        GRAPH_API_CONFIG=args.config)
    from .runtime import source_module
    feed = source_module(args.graph_api_root, 'habitat_feed_host')
    schedule = feed.load_schedule(args.schedule, args.floor)
    scans = {int(point['stop']): point for point in schedule['trajectory']
             if point.get('scan_deg', 0) > 0}
    script = json.loads(Path(args.script).read_text())
    selected_headings = sampled_heading_indices(args.headings, args.scan_samples)
    sim = feed.make_sim()
    rows = []
    try:
        agent = sim.initialize_agent(0)
        controller = feed.DynamicObjectController(sim, agent)
        names, semantic_ids = {}, {}
        sequence = 0
        for action_index, step in enumerate(script.get('steps', [])):
            action = step.get('action')
            if action not in {'spawn', 'move', 'remove'}:
                continue
            name = step.get('name') or step.get('object')
            command = dict(step)
            if action == 'spawn' and script.get('object_scale') is not None:
                command['object_scale'] = script['object_scale']
            if action != 'spawn':
                command['object_id'] = names[name]
            semantic_id = semantic_ids.get(name)
            before = None
            if action == 'remove':
                trigger = scans[int(step['at_waypoint']['stop'])]
                before = scan_pixels(sim, agent, trigger['xyz'], semantic_id, args.headings)
                if max(before) < 1:
                    raise ValueError(
                        f'Action {action_index} removal target is not visible before deletion')
            result = controller.execute(command)
            if not result.get('success'):
                raise RuntimeError(f'Action {action_index} failed: {result}')
            if action == 'spawn':
                sequence += 1
                names[name] = int(result['object_id'])
                semantic_id = args.semantic_id_offset + sequence
                semantic_ids[name] = semantic_id
                controller.spawned_objects[int(result['object_id'])].semantic_id = semantic_id
            expected = step['expected_observation']
            point = scans[int(expected['stop'])]
            after = scan_pixels(sim, agent, point['xyz'], semantic_id, args.headings)
            sampled_after = [after[index] for index in selected_headings]
            if expected['state'] == 'present' and max(sampled_after) < 1:
                raise ValueError(
                    f'Action {action_index} {action} is not visible in the fixed sampled '
                    f'headings at next scan {expected}')
            if expected['state'] == 'absent' and max(after) != 0:
                raise ValueError(
                    f'Action {action_index} removal remains visible at {expected}')
            rows.append({'action_index': action_index, 'action': action, 'object': name,
                         'semantic_id': semantic_id, 'trigger': step['at_waypoint'],
                         'expected_observation': expected,
                         'pre_action_visible_pixels_by_heading': before,
                         'post_action_visible_pixels_by_heading': after,
                         'sampled_heading_indices': selected_headings,
                         'sampled_post_action_visible_pixels': sampled_after,
                         'maximum_post_action_visible_pixels': max(after)})
        report = {'schema': 'graphapi.action_visibility_preflight.v1', 'complete': True,
                  'route_executed': False, 'headings': args.headings,
                  'scan_samples': args.scan_samples,
                  'sampled_heading_indices': selected_headings,
                  'graph_api_config': str(Path(args.config).resolve()),
                  'actions': rows,
                  'method': ('Actions execute in order in Habitat. Semantic pixels are rendered '
                             'from every fixed scheduled observation pose before acquisition.')}
        Path(args.output).write_text(json.dumps(report, indent=2) + '\n')
        print(json.dumps(report, indent=2))
    finally:
        sim.close()


if __name__ == '__main__':
    main()
