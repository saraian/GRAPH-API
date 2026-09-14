"""Verify explicit FOUND scene candidates using native Habitat GT and navmeshes.

Run in the Habitat environment with GPU access. Imports the canonical GT and
floor-detection helpers; never changes dataset files or baseline algorithms.
"""
import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np

from .replay_model import file_stamp


def module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def select_floor_ground_truth(objects, floor_heights, selected_index):
    """Scope native GT to semantic regions belonging to one scheduled floor."""
    floor_heights = np.asarray(floor_heights, dtype=float)
    if selected_index < 0 or selected_index >= len(floor_heights):
        raise ValueError('Selected floor index is outside the schedule')
    region_samples = {}
    for obj in objects:
        region = obj.get('region_id')
        if str(obj.get('category_name', '')).strip().lower() != 'floor' or region in (None, '-1'):
            continue
        centre_y = (float(obj['aabb_min_m'][1]) + float(obj['aabb_max_m'][1])) / 2
        region_samples.setdefault(str(region), []).append(centre_y)
    if not region_samples:
        raise ValueError('Cannot scope floor GT: native semantic regions contain no floor objects')
    region_floor = {
        region: int(np.argmin(np.abs(floor_heights - np.median(samples))))
        for region, samples in region_samples.items()
    }
    selected_regions = sorted(region for region, index in region_floor.items()
                              if index == selected_index)
    if not selected_regions:
        raise ValueError('Cannot scope floor GT: selected floor has no semantic floor region')
    included, excluded, fallback = [], [], []
    for obj in objects:
        region = str(obj.get('region_id'))
        if region in region_floor:
            object_floor = region_floor[region]
            source = 'semantic_floor_region'
        else:
            # Region-less native objects are assigned by their supporting (lower)
            # AABB face.  This keeps the fallback explicit and deterministic.
            object_floor = int(np.argmin(np.abs(floor_heights - float(obj['aabb_min_m'][1]))))
            source = 'aabb_lower_face_fallback'
            fallback.append(str(obj['object_id']))
        if object_floor == selected_index:
            included.append(obj)
        else:
            excluded.append(str(obj['object_id']))
    if not included:
        raise ValueError('Selected floor GT scope is empty')
    audit = {
        'policy': 'semantic floor-region assignment; region-less objects use nearest scheduled height to AABB lower face',
        'selected_floor_index': int(selected_index),
        'selected_floor_height_m': float(floor_heights[selected_index]),
        'selected_region_ids': selected_regions,
        'region_floor_index': region_floor,
        'fallback_object_ids': fallback,
        'excluded_native_object_ids': excluded,
        'full_scene_native_object_count': len(objects),
        'selected_floor_native_object_count': len(included),
    }
    return included, audit


def inspect(root, scene_id, output, allow_multifloor=False, prepare=None, project_goals=False,
            selected_floor_index=None):
    import habitat_sim
    dataset = root / 'lost3dsg/FOUND-Dataset'
    helpers = module(root / 'lost3dsg/src/perception_module/hm3d_ground_truth_manifest.py', 'baseline_gt_source')
    schedules = module(root / 'lost3dsg/test/schedule_batch.py', 'baseline_schedule_source')
    tag = scene_id.split('-')[1]
    scene = dataset / 'habitat/hm3d-val-habitat-v0.2' / scene_id / (tag + '.basis.glb')
    navmesh = scene.with_suffix('.navmesh')
    mesh, annotation = helpers._semantic_paths(scene)
    config = dataset / 'habitat/hm3d-val-semantic-configs-v0.2/hm3d_annotated_basis.scene_dataset_config.json'
    schedule_path = dataset / 'schedules' / (scene_id + '.schedule.json')
    full_schedule = json.loads(schedule_path.read_text())['schedule']
    if selected_floor_index is not None:
        if selected_floor_index < 0 or selected_floor_index >= len(full_schedule):
            raise ValueError('Selected floor index is outside the schedule')
        schedule = [full_schedule[selected_floor_index]]
    else:
        schedule = full_schedule
    if len(schedule) != 1 and not allow_multifloor:
        raise ValueError('Candidate does not have exactly one scheduled floor')
    records = list(helpers._semantic_records(annotation).values())
    floor_regions = sorted({r['region_id'] for r in records if r['category_name'] == 'floor' and r['region_id'] != '-1'})
    if len(floor_regions) < 2:
        raise ValueError('Candidate lacks multiple annotated floor regions')
    pf = habitat_sim.PathFinder()
    if not pf.load_nav_mesh(str(navmesh)):
        raise ValueError('Cannot load native navmesh')
    storeys, minor, stairs = schedules.scene_storeys(str(navmesh))
    if len(storeys) != len(full_schedule) or minor or (
            selected_floor_index is None and not allow_multifloor and stairs >= 0.01):
        raise ValueError('Candidate failed floor-count/navmesh sampling checks')
    original_schedule = schedule_path
    projection_max = 0.
    if project_goals:
        for level in schedule:
            for point in level['trajectory']:
                snapped = np.asarray(pf.snap_point(point['xyz']))
                delta = float(np.linalg.norm(np.asarray(point['xyz'])-snapped))
                if not np.isfinite(snapped).all() or delta > .5:
                    raise ValueError('Original goal is too far from the native navmesh to project safely')
                projection_max = max(projection_max, delta)
                point['xyz'] = snapped.tolist()
        schedule_path = output / (scene_id + '.schedule.json')
        schedule_path.write_text(json.dumps({'scene_id': scene_id, 'schedule': schedule,
            'navigation_projection': {'source': file_stamp(original_schedule),
                                      'method': 'native PathFinder.snap_point for every goal; all stops retained',
                                      'max_displacement_m': projection_max}}, indent=2))
    routes, all_heights, distance, max_snap = [], [], 0., 0.
    for level in schedule:
        points = np.array([p['xyz'] for p in level['trajectory']])
        snapped = np.array([pf.snap_point(p) for p in points])
        snaps = np.linalg.norm(points - snapped, axis=1)
        if not np.isfinite(snapped).all() or snaps.max() >= 0.25:
            raise ValueError('Tour contains invalid or distant navmesh goals')
        heights, floor_distance = [], 0.
        for a, b in zip(snapped, np.roll(snapped, -1, axis=0)):
            path = habitat_sim.ShortestPath()
            path.requested_start, path.requested_end = a, b
            if not pf.find_path(path):
                raise ValueError('Disconnected tour leg (including lap closure)')
            floor_distance += float(path.geodesic_distance)
            heights.extend(float(p[1]) for p in path.points)
        if max(heights)-min(heights) >= 0.75:
            raise ValueError('Per-floor tour changes level')
        routes.append({'height': level['height'], 'schedule_points': len(points),
                       'scan_stops': sum(p['scan_deg'] > 0 for p in level['trajectory']),
                       'max_goal_snap_m': float(snaps.max()), 'height_range_m': [min(heights), max(heights)],
                       'route_with_lap_closure_m': floor_distance})
        distance += floor_distance
        max_snap = max(max_snap, float(snaps.max()))
        all_heights.extend(heights)
    transitions = []
    for index, (a, b) in enumerate(zip(schedule, schedule[1:])):
        path = habitat_sim.ShortestPath()
        path.requested_start = pf.snap_point(a['trajectory'][-1]['xyz'])
        path.requested_end = pf.snap_point(b['trajectory'][0]['xyz'])
        if not pf.find_path(path):
            raise ValueError('Disconnected cross-floor route')
        transitions.append({'from_floor': index, 'to_floor': index+1,
                            'geodesic_m': float(path.geodesic_distance),
                            'points': [np.asarray(p).tolist() for p in path.points],
                            'scope': 'navmesh path only; multi-floor execution awaits certification'})
    # The distributed config assumes a different directory layout. Resolve its
    # canonical stage defaults against actual assets in a separate output config.
    defaults = json.loads(config.read_text())['stages']['default_attributes']
    defaults.update(semantic_descriptor_filename=str(annotation.resolve()),
                    semantic_asset=str(mesh.resolve()))
    resolved_config = output / (scene_id + '.scene_dataset_config.json')
    resolved_config.write_text(json.dumps({'stages': {'paths': {'.glb': [str(scene.resolve())]},
                                             'default_attributes': defaults},
                                         'scene_instances': {'default_attributes': {'default_lighting': 'no_lights'}}}))
    backend = habitat_sim.SimulatorConfiguration()
    backend.scene_id = str(scene.resolve())
    backend.scene_dataset_config_file = str(resolved_config.resolve())
    backend.enable_physics = prepare is not None
    backend.load_semantic_mesh = True
    backend.requires_textures = True
    if hasattr(backend, 'use_semantic_textures'):
        backend.use_semantic_textures = False
    agent = habitat_sim.agent.AgentConfiguration()
    sensors = []
    kinds = [('semantic_sensor', habitat_sim.SensorType.SEMANTIC)]
    if prepare:
        kinds += [('color_sensor', habitat_sim.SensorType.COLOR), ('depth_sensor', habitat_sim.SensorType.DEPTH)]
    for uuid, kind in kinds:
        sensor = habitat_sim.CameraSensorSpec()
        sensor.uuid, sensor.sensor_type = uuid, kind
        sensor.resolution = [240, 320] if prepare else [32, 32]
        sensor.position, sensor.hfov = [0.0, 1.5, 0.0], 90.0
        sensors.append(sensor)
    agent.sensor_specifications = sensors
    objects, missing = [], []
    with habitat_sim.Simulator(habitat_sim.Configuration(backend, [agent])) as sim:
        for obj in sim.semantic_scene.objects:
            if obj is None or int(obj.semantic_id) == 0:
                continue
            box = helpers._aabb(obj)
            if box is None:
                missing.append(str(obj.id))
                continue
            name, _ = helpers._category(obj)
            objects.append({'object_id': str(obj.id), 'semantic_id': int(obj.semantic_id),
                            'category_name': name, 'region_id': helpers._region_key(obj.region.id) if obj.region else None,
                            'aabb_min_m': box[0].tolist(), 'aabb_max_m': box[1].tolist(),
                            'geometry_source': 'semantic_object_obb_to_aabb'})
        dynamic_plans = prepare(sim, schedule) if prepare else []
    expected = {r['object_id'] for r in records}
    actual = {o['semantic_id'] for o in objects}
    if missing or not expected or expected != actual:
        raise ValueError(f'Native GT incomplete: missing boxes={missing}, descriptor-only IDs={sorted(expected-actual)}, native-only IDs={sorted(actual-expected)}')
    floor_gt_scope = None
    full_scene_native_gt_objects = len(objects)
    if selected_floor_index is not None:
        objects, floor_gt_scope = select_floor_ground_truth(
            objects, [level['height'] for level in full_schedule], selected_floor_index)
    assets = {k: file_stamp(p) for k, p in {'scene': scene, 'navmesh': navmesh, 'semantic_mesh': mesh,
              'annotations': annotation, 'original_schedule': original_schedule, 'schedule': schedule_path, 'dataset_config': config, 'resolved_dataset_config': resolved_config}.items()}
    gt = {'scene': scene_id, 'ground_truth_objects': objects,
          'geometry_space': {'coordinate_frame': 'Habitat: x,z horizontal; y up'},
          'ground_truth_source': {'scene': str(scene.resolve()), 'selected_floor_index': selected_floor_index,
                                  'object_geometry_source': 'semantic_object_obb_to_aabb',
                                  'native_object_count': len(objects),
                                  'full_scene_native_object_count': full_scene_native_gt_objects,
                                  'descriptor_object_count': len(records),
                                  'floor_scope': floor_gt_scope,
                                  'assets': assets, 'extractor': file_stamp(helpers.__file__)}}
    gt_path = output / (scene_id + '.gt.json')
    with gt_path.open('x') as stream:
        json.dump(gt, stream, indent=2, allow_nan=False)
    result = {'scene_id': scene_id, 'selected': True, 'selected_floor_index': selected_floor_index,
              'annotated_room_regions': len(floor_regions),
              'native_gt_objects': len(objects), 'assets': assets, 'ground_truth': file_stamp(gt_path),
              'original_goal_projection_max_m': projection_max, 'detected_storeys': storeys, 'minor_levels': minor, 'unassigned_navmesh_share': stairs,
              'schedule_points': sum(r['schedule_points'] for r in routes), 'scan_stops': sum(r['scan_stops'] for r in routes),
              'floors': routes, 'floor_transitions': transitions, 'dynamic_plans': dynamic_plans,
              'max_goal_snap_m': max_snap, 'disconnected_legs': 0,
              'route_with_lap_closure_m': distance, 'route_height_range_m': [min(all_heights), max(all_heights)],
              'navmesh_area_m2': float(pf.navigable_area), 'floor_detector': file_stamp(schedules.__file__),
              'limitations': ['Annotated room regions are not proof that a run observes every room.',
                              'Single-floor check uses canonical navmesh sampling and complete route connectivity.',
                              'No baseline run or room reconstruction certification performed by this check.']}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument('--scenes', nargs='+', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    results = []
    for scene in args.scenes:
        results.append(inspect(args.root, scene, args.output))
        print(scene, results[-1]['native_gt_objects'], 'native GT objects', flush=True)
    with (args.output / 'selection.json').open('x') as stream:
        json.dump({'schema': 'graphapi.baseline_scene_selection.v2', 'scenes': results}, stream, indent=2)


if __name__ == '__main__':
    main()
