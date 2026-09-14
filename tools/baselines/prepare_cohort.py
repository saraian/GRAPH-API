"""Prepare a shared ten-scene input cohort and waypoint-bound object-change plans.

Imports canonical FOUND placement validation/compiler and GRAPH-API trigger parser.
No LLM, baseline inference, launcher edits or scheduled execution is performed.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

from .replay_model import file_stamp
from .select_scenes import inspect, module

SINGLE = ['00813-svBbv1Pavdk', '00824-Dd4bFSTQ8gi', '00827-BAbdmeyTvMZ',
          '00829-QaLdnwvtxbs', '00880-Nfvxx8J5NCo']
MULTI = ['00820-mL8ThkuaVTM', '00839-zt1RVoi7PcG', '00890-6s7QHgap2fW',
         '00873-bxsVRursffK', '00810-CrMo8WxCyVb']
MAX_TRIGGER_OBJECT_DISTANCE_M = 3.0
MAX_RELAXED_TRIGGER_OBJECT_DISTANCE_M = 4.0
MIN_RELOCATION_DISTANCE_M = 1.0
MEDIUM_OBJECT_TEMPLATE_CANDIDATES = (
    '061_foam_brick', '035_power_drill', '019_pitcher_base', '050_medium_clamp',
)
MIN_MEDIUM_OBJECT_EXTENT_M = 0.15
MAX_MEDIUM_OBJECT_EXTENT_M = 0.75


def select_medium_object_template(sim, compiler, objects_dir):
    """Select an available template by its measured, scaled physical extent."""
    manager = sim.get_rigid_object_manager()
    available = set(compiler.available_template_names(objects_dir))
    measured = []
    for template in MEDIUM_OBJECT_TEMPLATE_CANDIDATES:
        if template not in available:
            continue
        handle, _ = compiler.template_handle_and_support_offset(sim, template, str(objects_dir))
        obj = manager.add_object_by_template_handle(str(handle))
        if obj is None:
            continue
        try:
            compiler.apply_global_object_scale(obj)
            size = obj.aabb.size()
            extent = max(float(size.x), float(size.y), float(size.z))
            measured.append((template, extent))
            if MIN_MEDIUM_OBJECT_EXTENT_M <= extent <= MAX_MEDIUM_OBJECT_EXTENT_M:
                return template, extent
        finally:
            manager.remove_object_by_id(obj.object_id)
    raise ValueError('No configured medium object template has a scaled extent in '
                     f'[{MIN_MEDIUM_OBJECT_EXTENT_M:g}, {MAX_MEDIUM_OBJECT_EXTENT_M:g}]m: {measured}')


def target_visible_from_scan(sim, compiler, template, objects_dir, position, scan_xyz,
                             sensor_height=1.5):
    """Check direct object visibility from an exact scheduled scan position.

    The canonical compiler's ``capture_eye`` proves that some navigable view
    exists.  The benchmark needs a stronger fact: the route itself reaches the
    view.  Instantiate the same scaled object temporarily and raycast from the
    scheduled sensor height to its centre.  This is a cheap plan-time check; it
    never supplies a frame or label to a baseline.
    """
    import habitat_sim
    import magnum as mn

    handle, _ = compiler.template_handle_and_support_offset(sim, template, objects_dir)
    manager = sim.get_rigid_object_manager()
    obj = manager.add_object_by_template_handle(str(handle))
    if obj is None:
        return False
    try:
        compiler.apply_global_object_scale(obj)
        obj.motion_type = habitat_sim.physics.MotionType.KINEMATIC
        obj.translation = mn.Vector3(position)
        size = obj.aabb.size()
        eye = mn.Vector3(float(scan_xyz[0]), float(scan_xyz[1]) + float(sensor_height),
                         float(scan_xyz[2]))
        target = mn.Vector3(float(position[0]),
                            float(position[1]) + max(0.02, 0.5 * float(size.y)),
                            float(position[2]))
        direction = target - eye
        if direction.length() < 0.05:
            return False
        hits = sim.cast_ray(habitat_sim.geo.Ray(eye, direction.normalized()))
        return bool(hits.has_hits() and hits.hits[0].object_id == obj.object_id)
    finally:
        manager.remove_object_by_id(obj.object_id)


def next_waypoint_target_maps(scans, positions, visible_from_scan,
                              maximum_distance=MAX_RELAXED_TRIGGER_OBJECT_DISTANCE_M):
    """Map action stops to targets observed by the next scheduled scan.

    Spawn and move execute after scan ``i`` and place the object where scan
    ``i+1`` has direct line of sight.  Removal candidates are visible at scan
    ``i`` before they are removed, so their absence can be checked at the same
    stop on the following lap.
    """
    placement, removal = {}, {}
    for trigger_index, trigger in enumerate(scans):
        trigger_xyz = np.asarray(trigger['xyz'], dtype=float)[[0, 2]]
        current = []
        for key, position in positions.items():
            point = np.asarray(position, dtype=float)[[0, 2]]
            distance = float(np.linalg.norm(trigger_xyz - point))
            if distance <= maximum_distance and visible_from_scan(trigger_index, key):
                current.append((distance, key))
        removal[trigger_index] = [key for _, key in sorted(current)]

        if trigger_index + 1 >= len(scans):
            placement[trigger_index] = []
            continue
        next_xyz = np.asarray(scans[trigger_index + 1]['xyz'], dtype=float)[[0, 2]]
        following = []
        for key, position in positions.items():
            point = np.asarray(position, dtype=float)[[0, 2]]
            trigger_distance = float(np.linalg.norm(trigger_xyz - point))
            next_distance = float(np.linalg.norm(next_xyz - point))
            if (trigger_distance <= maximum_distance and
                    visible_from_scan(trigger_index + 1, key)):
                following.append((next_distance, trigger_distance, key))
        placement[trigger_index] = [key for _, _, key in sorted(following)]
    return placement, removal


def waypoint_lifecycles(scans, targets_by_scan, positions, template='medium-object'):
    """One physical action at every scan stop with explicit lifecycle closure.

    A lifecycle normally consumes three consecutive stops: spawn, move, remove.
    The leading lifecycle uses extra relocations when the number of stops is not
    divisible by three.  That lets every stop have exactly one action while
    retaining a valid object lifetime and only one concurrent dynamic object.
    For tours with at least two lifecycles, the final object deliberately remains
    in the scene after its final relocation, so persistence is also measured.
    """
    count = len(scans)
    if count < 3:
        raise ValueError('At least three scan stops are required for a complete object lifecycle')
    remainder = count % 3
    first_length = 0 if remainder == 0 else 3 + remainder
    blocks, index = [], 0
    if first_length:
        blocks.append(first_length)
        index = first_length
    while index < count:
        blocks.append(3)
        index += 3

    actions, start = [], 0
    for lifecycle_number, length in enumerate(blocks, 1):
        name = f'dynamic_object_{lifecycle_number:03d}'
        current_key = None
        persistent = lifecycle_number == len(blocks) and len(blocks) > 1
        for offset in range(length):
            scan_index = start + offset
            choices = targets_by_scan[scan_index]
            if offset == 0:
                if not choices:
                    raise ValueError(f'No compiler-validated rear-or-lateral support within {MAX_RELAXED_TRIGGER_OBJECT_DISTANCE_M:g}m of scan stop {scans[scan_index]["stop"]}')
                current_key = choices[0]
                actions.append({'action': 'spawn', 'name': name, 'template': template,
                                'target_point': current_key, 'scan_index': scan_index})
            elif offset == length - 1 and not persistent:
                actions.append({'action': 'remove', 'object': name, 'scan_index': scan_index})
            else:
                destination = next(
                    (key for key in choices
                     if np.linalg.norm(np.asarray(positions[key])[[0, 2]] -
                                       np.asarray(positions[current_key])[[0, 2]]) >= MIN_RELOCATION_DISTANCE_M),
                    None,
                )
                if destination is None:
                    raise ValueError(
                        f'No distinct compiler-validated rear-or-lateral support within {MAX_RELAXED_TRIGGER_OBJECT_DISTANCE_M:g}m '
                        f'of scan stop {scans[scan_index]["stop"]} for lifecycle {name}'
                    )
                actions.append({'action': 'move', 'object': name, 'target_point': destination,
                                'scan_index': scan_index})
                current_key = destination
        start += length
    assert len(actions) == count and start == count
    return actions


def spaced_supported_stops(targets_by_scan, desired_actions):
    """Choose valid stops throughout the tour, preserving deterministic order."""
    supported = [index for index, targets in targets_by_scan.items() if targets]
    if len(supported) < 3:
        raise ValueError('Fewer than three rear-or-lateral compiler-validated scan stops')
    count = min(len(supported), max(3, desired_actions))
    selected = []
    for slot in range(count):
        ideal = slot * (len(supported) - 1) / max(1, count - 1)
        candidate = supported[round(ideal)]
        if not selected or candidate > selected[-1]:
            selected.append(candidate)
    return selected


def spread_valid_lifecycles(scans, targets_by_scan, positions, template, windows=6,
                            removal_targets_by_scan=None):
    """Build the largest action-compatible plan over adaptive tour windows.

    Fixed window boundaries can separate the only compatible spawn and move
    supports.  Try every partition from the requested density down to one
    whole-tour window, then keep the plan with the most actions.  This retains
    spread when the scene supports it and still compiles sparse valid scenes.
    """
    removal_targets_by_scan = removal_targets_by_scan or targets_by_scan
    plans = []
    max_windows = min(windows, max(1, len(scans) // 3))
    for window_count in range(max_windows, 0, -1):
        actions = []
        for window in range(window_count):
            first = round(window * len(scans) / window_count)
            end = round((window + 1) * len(scans) / window_count)
            if end - first < 3:
                continue
            pairs = []
            for spawn_index in range(first, end - 2):
                for spawn_key in targets_by_scan[spawn_index]:
                    for move_index in range(spawn_index + 1, end - 1):
                        for move_key in targets_by_scan[move_index]:
                            distance = float(np.linalg.norm(
                                np.asarray(positions[move_key])[[0, 2]] -
                                np.asarray(positions[spawn_key])[[0, 2]]))
                            remove_indices = [
                                index for index in range(move_index + 1, end)
                                if move_key in removal_targets_by_scan[index]
                            ]
                            if distance >= MIN_RELOCATION_DISTANCE_M and remove_indices:
                                remove_index = remove_indices[-1]
                                pairs.append((remove_index - spawn_index,
                                              move_index - spawn_index, distance,
                                              spawn_index, spawn_key, move_index,
                                              move_key, remove_index))
            if not pairs:
                continue
            (_, _, _, spawn_index, spawn_key, move_index,
             move_key, remove_index) = max(pairs)
            name = f'dynamic_object_{len(actions)//3 + 1:03d}'
            actions.extend([
                {'action': 'spawn', 'name': name, 'template': template,
                 'target_point': spawn_key, 'scan_index': spawn_index},
                {'action': 'move', 'object': name, 'target_point': move_key,
                 'scan_index': move_index},
                {'action': 'remove', 'object': name, 'scan_index': remove_index},
            ])
        if actions:
            # A plan with several lifecycles keeps its last object as the
            # persistence case.  A sparse one-lifecycle plan keeps removal so
            # it still exercises spawn, movement, and deletion.
            if len(actions) > 3:
                actions.pop()
            plans.append(actions)
    if not plans:
        raise ValueError('No action-compatible lifecycle window on this floor')
    return max(plans, key=lambda actions: (
        len(actions),
        actions[-1]['scan_index'] - actions[0]['scan_index'],
    ))


def distribute_lifecycles_across_laps(actions, laps):
    """Place independent lifecycles across complete repeated tour laps.

    Complete lifecycles are assigned round-robin so every lap receives changes.
    The deliberately persistent final lifecycle is kept last on the final lap,
    preserving the one-live-object invariant. Actions within a lap retain their
    waypoint order; the tour itself is never partitioned between laps.
    """
    laps = int(laps)
    if laps < 1:
        raise ValueError('Action lap count must be positive')
    groups = []
    by_object = {}
    for action in actions:
        name = action.get('name') or action.get('object')
        if name not in by_object:
            by_object[name] = []
            groups.append(by_object[name])
        by_object[name].append(dict(action))
    if len(groups) < laps:
        raise ValueError(
            f'Cannot spread {len(groups)} object lifecycles across {laps} laps'
        )
    persistent = [group for group in groups if group[-1]['action'] != 'remove']
    if len(persistent) > 1 or (persistent and persistent[0] is not groups[-1]):
        raise ValueError('Persistent lifecycle must be the final lifecycle')
    complete = groups[:-1] if persistent else groups
    assigned = []
    for index, group in enumerate(complete):
        assigned.append((index % laps, group))
    if persistent:
        assigned.append((laps - 1, persistent[0]))
    distributed = []
    for lap, group in sorted(
            assigned, key=lambda item: (item[0], item[1][0]['scan_index'])):
        for action in group:
            action['lap'] = lap
            distributed.append(action)
    used = {action['lap'] for action in distributed}
    if used != set(range(laps)):
        raise ValueError(
            f'Object actions do not cover every requested lap: {sorted(used)}'
        )
    return distributed


def changes(sim, levels, scene, destination, compiler, runner, objects_dir,
            action_laps=1, excluded_targets=()):
    excluded_targets = {str(value) for value in excluded_targets}
    points = compiler.generate_points(sim, scene=str(scene))
    (destination / 'placement_points.json').write_text(json.dumps(points, indent=2))
    plans = []
    for floor, level in enumerate(levels):
        scans = [p for p in level['trajectory'] if p.get('scan_deg', 0) > 0]
        candidates = {}
        for key, point in points.items():
            if str(key) in excluded_targets:
                continue
            support = np.asarray(point['surface_point'])
            ground = np.asarray(sim.pathfinder.snap_point(support))
            near_route = any(float(np.linalg.norm(
                support[[0, 2]] - np.asarray(scan['xyz'])[[0, 2]]
            )) <= MAX_RELAXED_TRIGGER_OBJECT_DISTANCE_M for scan in scans)
            if (np.isfinite(ground).all() and
                    abs(ground[1] - level['height']) < .4 and near_route):
                candidates[key] = point
        if len(candidates) < 2:
            plans.append({'floor_index': floor, 'height': level['height'], 'events': [],
                          'reason': 'Canonical placement generator returned fewer than two supports on this floor'})
            continue
        template, template_extent_m = select_medium_object_template(sim, compiler, objects_dir)
        # Compile each physical support once. Exact route visibility is checked
        # below; a free-standing compiler capture eye is insufficient evidence.
        preflighted, errors = {}, []
        for key in sorted(candidates):
            trial = {'steps': [{'action': 'spawn', 'name': 'probe', 'template': template,
                                'target_point': key}]}
            try:
                preflighted[key] = compiler.compile_plan(trial, candidates, sim, str(objects_dir))['steps'][0]
            except ValueError as exc:
                errors.append(str(exc))
        positions = {key: step['position'] for key, step in preflighted.items()}
        visibility_cache = {}
        def visible_from_scan(scan_index, key):
            cache_key = (int(scan_index), str(key))
            if cache_key not in visibility_cache:
                visibility_cache[cache_key] = target_visible_from_scan(
                    sim, compiler, template, str(objects_dir), positions[key],
                    scans[scan_index]['xyz'])
            return visibility_cache[cache_key]
        targets_by_scan, observable_by_scan = next_waypoint_target_maps(
            scans, positions, visible_from_scan)
        try:
            planned = spread_valid_lifecycles(
                scans, targets_by_scan, positions, template,
                removal_targets_by_scan=observable_by_scan,
            )
            planned = distribute_lifecycles_across_laps(planned, action_laps)
        except ValueError as exc:
            plans.append({'floor_index': floor, 'height': level['height'], 'events': [],
                          'reason': str(exc), 'placement_rejections': errors[:10],
                          'action_coverage': {'required_scan_stops': len(scans), 'scheduled_actions': 0}})
            continue
        raw_steps = []
        for action in planned:
            step = {key: value for key, value in action.items()
                    if key not in {'scan_index', 'lap'}}
            if step['action'] == 'spawn':
                step['template'] = template
            step['at_waypoint'] = {
                'stop': int(scans[action['scan_index']]['stop']),
                'lap': int(action['lap']),
            }
            raw_steps.append(step)
        raw = {'description': 'Spread compiler-validated medium-object spawn, relocation, removal and persistence lifecycles',
               'steps': raw_steps}
        compiled = compiler.compile_plan(raw, candidates, sim, str(objects_dir))
        # Waypoint barriers replace time estimates. Settling waits from the shared
        # compiler are kept; actual executor timing/robot pose must be recorded later.
        actual = [s for s in compiled['steps'] if s['action'] != 'wait']
        events = []
        object_positions = {}
        for number, (step, planned_action) in enumerate(zip(actual, planned), 1):
            index = planned_action['scan_index']
            step['at_waypoint'] = {
                'stop': int(scans[index]['stop']),
                'lap': int(planned_action['lap']),
            }
            if step['action'] in {'spawn', 'move'}:
                if index + 1 >= len(scans):
                    raise ValueError('Spawn/move at the final scan has no next waypoint observation')
                step['expected_observation'] = {
                    'stop': int(scans[index + 1]['stop']),
                    'lap': int(planned_action['lap']),
                    'state': 'present',
                    'timing': 'next_waypoint_scan',
                }
            else:
                step['expected_observation'] = {
                    'stop': int(scans[index]['stop']),
                    'lap': int(planned_action['lap']) + 1,
                    'state': 'absent',
                    'timing': 'same_waypoint_following_lap',
                }
            runner.HabitatScriptRunner._waypoint_trigger(step, number)
            name = step.get('name') or step.get('object')
            previous_position = object_positions.get(name)
            if step['action'] in {'spawn', 'move'}:
                object_positions[name] = step['position']
            position = step.get('position', object_positions.get(name))
            trigger_distance = (float(np.linalg.norm(np.asarray(scans[index]['xyz'])[[0, 2]] -
                                                      np.asarray(position)[[0, 2]]))
                                if position is not None else None)
            if (trigger_distance is not None and
                    trigger_distance > MAX_RELAXED_TRIGGER_OBJECT_DISTANCE_M + 1e-6):
                raise ValueError(
                    f'{step["action"]} trigger is {trigger_distance:.3f} m from '
                    f'{name}; maximum is {MAX_RELAXED_TRIGGER_OBJECT_DISTANCE_M:g} m'
                )
            events.append({'id': f'F{floor+1}E{number}', 'action': step['action'], 'object': name,
                           'trigger': dict(step['at_waypoint']), 'scan_visit_number': index+1,
                           'expected_observation': dict(step['expected_observation']),
                           'robot_position': scans[index]['xyz'], 'object_position': position,
                           'trigger_object_distance_m': trigger_distance,
                           'range_policy': ('nominal_3m' if trigger_distance is None or trigger_distance <= MAX_TRIGGER_OBJECT_DISTANCE_M
                                            else 'relaxed_4m'),
                           'previous_object_position': previous_position if step['action'] == 'move' else None,
                           'scope': 'expected action and later route observation; execution not recorded'})
        compiled.update(scene=str(scene), floor_height=level['height'],
                        plan_scope='per-floor script; multi-floor orchestration awaits certification',
                        source_compiler=file_stamp(compiler.__file__), trigger_parser=file_stamp(runner.__file__))
        script = destination / f'floor-{floor}.script.json'
        script.write_text(json.dumps(compiled, indent=2))
        plans.append({'floor_index': floor, 'height': level['height'], 'events': events, 'script': file_stamp(script),
                      'action_coverage': {'required_scan_stops': len(scans), 'scheduled_actions': len(events),
                                          'complete': len(events) > 0,
                                          'supported_scan_stops': len([v for v in targets_by_scan.values() if v]),
                                          'supported_removal_scan_stops': len([v for v in observable_by_scan.values() if v]),
                                          'template': template,
                                          'template_scaled_max_extent_m': template_extent_m,
                                          'max_live_objects': 1,
                                          'object_action_laps': sorted({
                                              event['trigger']['lap'] for event in events
                                          }),
                                          'observation_policy': {
                                              'spawn_move': ('action after scan i targets a support '
                                                             'visible from scheduled scan i+1'),
                                              'remove': ('object is visible before removal at scan i; '
                                                         'absence is evaluated at scan i on lap+1'),
                                              'capture_eye_detours': False,
                                          },
                                          'trigger_range_policy': {'nominal_m': MAX_TRIGGER_OBJECT_DISTANCE_M,
                                                                   'relaxed_m': MAX_RELAXED_TRIGGER_OBJECT_DISTANCE_M,
                                                                   'applies_to': ['spawn', 'move', 'remove']},
                                          'persistent_final_object': (
                                              sum(event['action'] == 'spawn' for event in events) >
                                              sum(event['action'] == 'remove' for event in events))}})
    if not any(p['events'] for p in plans):
        raise ValueError('No compiled object changes available for scene')
    return plans


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--resume', action='store_true', help='Reuse completed plans after checking every asset hash')
    parser.add_argument('--action-laps', type=int, default=1,
                        help='Distribute object lifecycles over this many complete tour laps')
    parser.add_argument('--single', nargs='+', default=SINGLE)
    parser.add_argument('--multi', nargs='+', default=MULTI)
    args = parser.parse_args()
    if args.action_laps < 1:
        parser.error('--action-laps must be positive')
    args.output.mkdir(parents=True, exist_ok=args.resume)
    dataset = args.root/'lost3dsg/FOUND-Dataset'
    sys.path.insert(0, str(dataset))
    compiler = module(dataset/'scene_script.py', 'found_cohort_compiler')
    runner = module(args.root/'lost3dsg/src/perception_module/script_runner.py', 'found_cohort_runner')
    records = []
    for kind, scenes in [('single-floor', args.single), ('multi-floor', args.multi)]:
        for scene_id in scenes:
            dest = args.output/scene_id
            if args.resume and (dest/'plan.json').is_file():
                result = json.loads((dest/'plan.json').read_text())
                stamps = list(result['assets'].values()) + [result['ground_truth']]
                stamps += [p['script'] for p in result['dynamic_plans'] if p.get('script')]
                for stamp in stamps:
                    if file_stamp(stamp['path'])['sha256'] != stamp['sha256']:
                        raise ValueError('Cannot resume changed cohort inputs: '+stamp['path'])
                records.append(result)
                print('REUSED', scene_id, flush=True)
                continue
            dest.mkdir(exist_ok=args.resume)
            scene = dataset/'habitat/hm3d-val-habitat-v0.2'/scene_id/(scene_id.split('-')[1]+'.basis.glb')
            def prepare(sim, levels):
                return changes(sim, levels, scene, dest, compiler, runner,
                               dataset/'habitat/habitat_objects/configs',
                               action_laps=args.action_laps)
            result = inspect(args.root, scene_id, dest, allow_multifloor=kind=='multi-floor', prepare=prepare, project_goals=True)
            result['cohort_group'] = kind
            (dest/'plan.json').write_text(json.dumps(result, indent=2))
            records.append(result)
            (args.output/'cohort.json').write_text(json.dumps({'schema': 'graphapi.baseline_cohort.v1',
                'scope': 'validated dataset assets and planned tours/changes; not an executed baseline run',
                'source': file_stamp(__file__), 'scenes': records}, indent=2))
            print('PREPARED', scene_id, result['scan_stops'], 'scans', sum(len(p['events']) for p in result['dynamic_plans']), 'changes', flush=True)

    from .cohort_plot import export_cohort
    from .tour_coverage import annotate_cohort
    from .validate_cohort import validate
    cohort_path = args.output/'cohort.json'
    annotate_cohort(cohort_path)
    validation = validate(cohort_path)
    (args.output/'validation.json').write_text(json.dumps(validation, indent=2))
    export_cohort(cohort_path)


if __name__ == '__main__':
    main()
