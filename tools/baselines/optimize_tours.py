"""Improve prepared tours against the saved physical floor grid; preserve source tours."""
import argparse
import copy
import json
import math
import os
from pathlib import Path

import numpy as np

from .replay_model import file_stamp
from .tour_coverage import in_view, trace


def geodesic(pf, a, b):
    import habitat_sim
    p = habitat_sim.ShortestPath()
    p.requested_start, p.requested_end = np.asarray(a, dtype=np.float32), np.asarray(b, dtype=np.float32)
    if not pf.find_path(p):
        return None
    return p


def visible_cells(sim, xyz, targets, cov, tilts):
    eye = np.asarray(xyz)+[0., cov['camera_height_m'], 0.]
    result = np.zeros(len(targets), dtype=bool)
    for i in np.flatnonzero(in_view(eye, targets, cov['range_m'], cov['vfov_deg'], tilts)):
        delta = targets[i]-eye
        distance = float(np.linalg.norm(delta))
        result[i] = not trace(sim, eye, delta/distance, max(0., distance-.04))
    return result


def greedy_cover(views, covered, fraction=.995):
    """Greedy set cover up to the requested fraction of this candidate set's union."""
    covered = covered.copy()
    ceiling = covered | views.any(axis=0) if len(views) else covered.copy()
    target = math.ceil(int(ceiling.sum())*fraction)
    selected = []
    while covered.sum() < target:
        gains = (views & ~covered).sum(axis=1)
        index = int(gains.argmax())
        if gains[index] == 0:
            break
        selected.append(index)
        covered |= views[index]
    return selected, covered, ceiling


def insert_points(pf, trajectory, additions):
    """Cheapest geodesic insertion while retaining every original point and its order."""
    trajectory = copy.deepcopy(trajectory)
    for point in additions:
        best = None
        # Keep first and final original points: cross-floor transfer endpoints stay stable.
        for i in range(1, len(trajectory)):
            a, b = trajectory[i-1]['xyz'], trajectory[i]['xyz']
            ab, ap, pb = geodesic(pf, a, b), geodesic(pf, a, point['xyz']), geodesic(pf, point['xyz'], b)
            if ab is None or ap is None or pb is None:
                continue
            cost = float(ap.geodesic_distance+pb.geodesic_distance-ab.geodesic_distance)
            if best is None or cost < best[0]:
                best = cost, i
        if best is None:
            raise ValueError('No connected insertion for coverage viewpoint')
        trajectory.insert(best[1], point)
    return trajectory


def improve_floor(sim, level, output, spacing=.5, fraction=.995):
    cov = level['coverage']
    with np.load(cov['grid']['path']) as grid:
        mask, origin = grid['mask'].copy(), grid['origin_xz'].copy()
        res, heights = float(grid['resolution_m']), grid['floor_heights'].copy()
    iz, ix = np.where(mask>=2)
    targets = np.column_stack((origin[0]+(ix+.5)*res, heights[iz, ix]+.02, origin[1]+(iz+.5)*res))
    scans = [p for p in level['trajectory'] if p.get('scan_deg', 0)>0]
    tilts = [0., -45.]
    views = {int(p['stop']): visible_cells(sim, p['xyz'], targets, cov, tilts) for p in scans}
    initial = np.logical_or.reduce(list(views.values()))
    pf = sim.pathfinder
    candidates, keys = [], set()
    for x in np.arange(origin[0]+spacing/2, origin[0]+mask.shape[1]*res, spacing):
        for z in np.arange(origin[1]+spacing/2, origin[1]+mask.shape[0]*res, spacing):
            p = np.asarray(pf.snap_point([float(x), float(level['height']), float(z)]))
            if not np.isfinite(p).all() or abs(p[1]-level['height']) > cov['floor_band_m']:
                continue
            if np.linalg.norm(p[[0, 2]]-[x, z]) > spacing:
                continue
            key = tuple(np.round(p/.10).astype(int))
            if key in keys:
                continue
            keys.add(key)
            path = geodesic(pf, scans[0]['xyz'], p)
            if path is None or np.max(np.abs(np.asarray(path.points)[:, 1]-level['height'])) > .75:
                continue
            candidates.append(p)
    candidate_views = np.array([visible_cells(sim, p, targets, cov, tilts) for p in candidates], dtype=bool)
    if not candidates:
        candidate_views = np.zeros((0, len(targets)), dtype=bool)
    chosen, selected_mask, ceiling = greedy_cover(candidate_views, initial, fraction)
    next_stop = max(int(p['stop']) for p in scans)+1
    additions = []
    for n, j in enumerate(chosen):
        stop = next_stop+n
        additions.append({'xyz': candidates[j].tolist(), 'scan_deg': 360., 'stop': stop,
                          'leg': None, 'coverage_added': True})
        views[stop] = candidate_views[j]
    level['trajectory'] = insert_points(pf, level['trajectory'], additions)
    level['scan_plan'] = {**level.get('scan_plan', {}), 'mode': 'continuous', 'hold_frames': 1,
                          'tilts_deg': tilts, 'turn_step_deg': 10.}
    # Record progressive coverage in final visit order, including original stable stop IDs.
    covered = np.zeros(len(targets), dtype=bool)
    stops = []
    for p in level['trajectory']:
        if not p.get('scan_deg', 0):
            continue
        before = int(covered.sum())
        covered |= views[int(p['stop'])]
        stops.append({'stop': int(p['stop']), 'new_area_m2': round((int(covered.sum())-before)*res**2, 4),
                      'cumulative_area_m2': round(int(covered.sum())*res**2, 4)})
    if not np.array_equal(covered, selected_mask):
        raise ValueError('Inserted tour coverage differs from selected viewpoints')
    if np.any((mask[iz, ix]==3) & ~covered):
        raise ValueError('Coverage regressed at an originally covered cell')
    mask[iz, ix] = np.where(covered, 3, 2)
    np.savez_compressed(output, mask=mask, origin_xz=origin, resolution_m=res, floor_heights=heights)
    outside = np.array([not pf.is_navigable(p) for p in targets])
    result = copy.deepcopy(cov)
    result.update(covered_cells=int(covered.sum()), covered_area_m2=round(int(covered.sum())*res**2, 4),
                  uncovered_area_m2=round(int((~covered).sum())*res**2, 4),
                  coverage_pct=round(100*covered.mean(), 3), tilts_deg=tilts,
                  covered_area_outside_navmesh_m2=round(int((covered & outside).sum())*res**2, 4),
                  per_stop=stops, grid=file_stamp(output))
    result['sources'] = {**result['sources'], 'previous_grid': cov['grid'], 'optimizer': file_stamp(__file__)}
    level['coverage'] = result
    level['coverage_optimization'] = {
        'method': 'two camera tilts plus greedy visibility set cover; cheapest geodesic insertion preserving original order',
        'previous_coverage_pct': cov['coverage_pct'], 'tilt_only_coverage_pct': round(100*initial.mean(), 3),
        'candidate_spacing_m': spacing, 'candidate_count': len(candidates), 'added_scan_stops': len(additions),
        'candidate_visible_area_m2': round(int(ceiling.sum())*res**2, 4),
        'candidate_visibility_ceiling_pct': round(100*ceiling.mean(), 3),
        'candidate_union_target_fraction': fraction,
        'fraction_of_candidate_visible_area_covered': round(float(covered.sum()/ceiling.sum()), 6),
        'unseen_by_candidate_set_m2': round(int((~ceiling).sum())*res**2, 4),
        'limitations': f'Finite {spacing:g}m candidate-grid ceiling, not proof of continuous global optimum. Coverage evaluated at planned target poses; native follower arrival tolerance can alter visibility.'}
    print('OPTIMIZED', round(level['height'], 2), level['coverage_optimization'], 'coverage', result['coverage_pct'], flush=True)


def make_sim(row, turn=10.):
    import habitat_sim
    backend = habitat_sim.SimulatorConfiguration()
    backend.scene_id = row['assets']['scene']['path']
    backend.scene_dataset_config_file = row['assets']['resolved_dataset_config']['path']
    backend.enable_physics = True
    backend.create_renderer = False
    agent = habitat_sim.agent.AgentConfiguration()
    agent.sensor_specifications = []
    agent.action_space = {name: habitat_sim.agent.ActionSpec(name, habitat_sim.agent.ActuationSpec(amount=amount))
                          for name, amount in [('move_forward', .15), ('turn_left', turn), ('turn_right', turn)]}
    return habitat_sim.Simulator(habitat_sim.Configuration(backend, [agent]))


def duration(sim, level, feed, fps):
    """Count ticks of the imported executor, without rendering, object scripts or sleeps."""
    import habitat_sim
    agent = sim.initialize_agent(0)
    state = habitat_sim.AgentState()
    state.position = np.asarray(level['trajectory'][0]['xyz'], dtype=np.float32)
    agent.set_state(state)
    tour = feed.ScheduledTour(sim, level, 3, 'navigate')
    frames, last_lap, lap_counts = 0, 0, []
    previous = np.asarray(agent.get_state().position).copy()
    distance = 0.
    while not tour.house_done:
        tour.step(agent)
        frames += 1
        here = np.asarray(agent.get_state().position)
        distance += float(np.linalg.norm(here-previous))
        previous = here.copy()
        if frames > 1000000:
            raise RuntimeError('Dry action count exceeded diagnostic limit')
        if tour.lap != last_lap:
            lap_counts.append(frames)
            last_lap = tour.lap
        feed.CTRL.scan_events.clear()
    return {'scope': 'nominal action clock from imported ScheduledTour with native sensorless navigation; not acquisition wall time',
            'fps': fps, 'forward_step_m': .15, 'turn_step_deg': feed.TURN_STEP_DEG,
            'first_lap_frames': lap_counts[0], 'three_lap_frames': frames,
            'first_lap_seconds': round(lap_counts[0]/fps, 3), 'three_lap_seconds': round(frames/fps, 3),
            'three_lap_driven_distance_m': round(distance, 3), 'skipped': tour.skipped,
            'valid_complete_tour': not tour.skipped, 'executor': file_stamp(feed.__file__),
            'exclusions': ['Rendering, image writing, IPC and native baseline inference.',
                           'Dynamic object script processing and extra waits.',
                           'Inter-floor transfer, localization/map reset and startup.',
                           'Sensorless action counting does not verify camera tilt application.']}


def validate_optimized(old_doc, new_doc):
    if [r['scene_id'] for r in old_doc['scenes']] != [r['scene_id'] for r in new_doc['scenes']]:
        raise ValueError('Cohort membership changed')
    results = []
    for before, after in zip(old_doc['scenes'], new_doc['scenes']):
        old = json.loads(Path(before['assets']['schedule']['path']).read_text())
        new = json.loads(Path(after['assets']['schedule']['path']).read_text())
        for stamp in [*after['assets'].values(), after['ground_truth']]:
            if file_stamp(stamp['path'])['sha256'] != stamp['sha256']:
                raise ValueError('Source hash changed: '+stamp['path'])
        for a, b in zip(old['schedule'], new['schedule']):
            if [p for p in b['trajectory'] if not p.get('coverage_added')] != a['trajectory']:
                raise ValueError('Original tour points/order changed')
            ids = [p['stop'] for p in b['trajectory'] if p.get('scan_deg', 0)]
            if len(ids) != len(set(ids)):
                raise ValueError('Stop IDs are not unique')
            ca, cb = a['coverage'], b['coverage']
            if ca['free_area_m2'] != cb['free_area_m2'] or ca['range_m'] != cb['range_m']:
                raise ValueError('Coverage denominator/range changed')
            with np.load(ca['grid']['path']) as ga, np.load(cb['grid']['path']) as gb:
                ma, mb = ga['mask'], gb['mask']
                if not np.array_equal(ma>=2, mb>=2) or np.any((ma==3)&(mb!=3)):
                    raise ValueError('Physical coverage domain changed or coverage regressed')
                if int((mb==3).sum()) != cb['covered_cells'] or file_stamp(cb['grid']['path']) != cb['grid']:
                    raise ValueError('Saved coverage grid mismatch')
            if cb['per_stop'][-1]['cumulative_area_m2'] != cb['covered_area_m2']:
                raise ValueError('Cumulative area mismatch')
        for a, b in zip(before['dynamic_plans'], after['dynamic_plans']):
            if a.get('script') != b.get('script'):
                raise ValueError('Compiled dynamic script changed')
            scans = [p for p in new['schedule'][b['floor_index']]['trajectory'] if p.get('scan_deg', 0)]
            for ea, eb in zip(a['events'], b['events']):
                if {k:v for k,v in ea.items() if k!='scan_visit_number'} != {k:v for k,v in eb.items() if k!='scan_visit_number'}:
                    raise ValueError('Object event/robot target changed')
                p = scans[eb['scan_visit_number']-1]
                if p['stop'] != eb['trigger']['stop'] or p['xyz'] != eb['robot_position']:
                    raise ValueError('Dynamic trigger mapping changed')
        results.append({'scene_id': after['scene_id'], 'coverage_pct': after['coverage']['coverage_pct'],
                        'duration': after['duration']})
    return {'complete': True, 'scope': 'optimized input integrity and physical coverage, not baseline execution', 'scenes': results}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cohort', type=Path, default=Path('artifacts/baselines/cohort-10/cohort.json'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--scenes', nargs='+')
    parser.add_argument('--fps', type=float, default=3.)
    parser.add_argument('--graph-api-root', type=Path, default=Path(__file__).resolve().parents[2])
    args = parser.parse_args()
    if not math.isfinite(args.fps) or args.fps <= 0:
        parser.error('fps must be positive and finite')
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    os.environ.update(OUT_DIR=str(args.output/'dry-executor'), RUN_DIR=str(args.output/'dry-executor'),
                      FEED_POST_SCAN_HOOK='', FEED_CAMERA_PITCH_DEG='0', FEED_TURN_STEP_DEG='10')
    from .runtime import source_module
    feed = source_module(args.graph_api_root, 'habitat_feed_host')
    old_doc = json.loads(args.cohort.read_text())
    if args.scenes:
        old_doc['scenes'] = [r for r in old_doc['scenes'] if r['scene_id'] in args.scenes]
    new_doc = copy.deepcopy(old_doc)
    new_doc['source_cohort'] = file_stamp(args.cohort)
    for old_row, row in zip(old_doc['scenes'], new_doc['scenes']):
        print('SCENE', row['scene_id'], flush=True)
        out = args.output/row['scene_id']
        out.mkdir()
        source = row['assets']['schedule']
        doc = json.loads(Path(source['path']).read_text())
        original = copy.deepcopy(doc)
        with make_sim(row) as sim:
            for i, level in enumerate(doc['schedule']):
                improve_floor(sim, level, out/f'floor-{i}.coverage.npz')
                level['previous_duration'] = duration(sim, original['schedule'][i], feed, args.fps)
                level['duration'] = duration(sim, level, feed, args.fps)
                for plan in row['dynamic_plans']:
                    if plan['floor_index']==i:
                        scans = [p for p in level['trajectory'] if p.get('scan_deg', 0)]
                        for event in plan['events']:
                            event['scan_visit_number'] = next(j+1 for j,p in enumerate(scans) if p['stop']==event['trigger']['stop'])
        total = sum(level['coverage']['free_area_m2'] for level in doc['schedule'])
        seen = sum(level['coverage']['covered_area_m2'] for level in doc['schedule'])
        doc['coverage'] = {**doc['coverage'], 'free_area_m2': round(total, 4), 'covered_area_m2': round(seen, 4), 'coverage_pct': round(100*seen/total, 3)}
        doc['coverage_optimization'] = {'source_schedule': source, 'optimizer': file_stamp(__file__)}
        doc['duration'] = {k: round(sum(level['duration'][k] for level in doc['schedule']), 3) for k in ('first_lap_seconds', 'three_lap_seconds')}
        doc['duration'].update(fps=args.fps, valid_complete_tour=all(level['duration']['valid_complete_tour'] for level in doc['schedule']),
                               scope='sum of floor action clocks; excludes cross-floor transfer and acquisition overhead')
        doc['previous_duration'] = {k: round(sum(level['previous_duration'][k] for level in doc['schedule']), 3) for k in ('first_lap_seconds', 'three_lap_seconds')}
        path = out/(row['scene_id']+'.schedule.json')
        path.write_text(json.dumps(doc, indent=2))
        row['assets']['previous_schedule'] = source
        row['assets']['schedule'] = file_stamp(path)
        row['coverage'], row['duration'] = doc['coverage'], doc['duration']
        row['previous_duration'] = doc['previous_duration']
        row['scan_stops'] = sum(p.get('scan_deg', 0)>0 for level in doc['schedule'] for p in level['trajectory'])
        row['schedule_points'] = sum(len(level['trajectory']) for level in doc['schedule'])
        (out/'plan.json').write_text(json.dumps(row, indent=2))
        print('RESULT', row['scene_id'], row['coverage'], row['duration'], flush=True)
    result = validate_optimized(old_doc, new_doc)
    (args.output/'cohort.json').write_text(json.dumps(new_doc, indent=2))
    (args.output/'validation.json').write_text(json.dumps(result, indent=2))
    from .cohort_plot import export_cohort
    export_cohort(args.output/'cohort.json')


if __name__ == '__main__':
    main()
