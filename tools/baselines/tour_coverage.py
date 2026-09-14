"""Planned camera coverage over collision-derived free floor area, not navmesh area.

A grid cell belongs to free floor space when the static stage has an upward-facing
support in this storey's height band and no collision in the vertical body column.
Visibility is ray-tested to the cell's floor surface from each 360-degree scan.
"""
import argparse
import json
import math
from pathlib import Path

import numpy as np

from .replay_model import file_stamp


def in_view(eye, targets, radius, vfov, tilts):
    delta = targets-np.asarray(eye)
    horizontal = np.linalg.norm(delta[:, [0, 2]], axis=1)
    elevation = np.degrees(np.arctan2(delta[:, 1], horizontal))
    visible = (horizontal <= radius) & (horizontal > .001)
    visible &= np.any(np.abs(elevation[:, None]-np.array(tilts)[None, :]) <= vfov/2, axis=1)
    return visible


def trace(sim, origin, direction, max_distance):
    import habitat_sim
    import magnum as mn
    ray = habitat_sim.geo.Ray(mn.Vector3(*map(float, origin)), mn.Vector3(*map(float, direction)))
    hits = sim.cast_ray(ray, float(max_distance))
    return [h for h in hits.hits if 0 <= float(h.ray_distance) <= max_distance]


def coverage(sim, level, bounds, output, resolution=.10, camera_height=1.5, clearance_height=1.5,
             hfov=90., width=640, height=480, floor_band=.4, floor_gt=None):
    if resolution <= 0 or camera_height <= 0 or clearance_height <= 0:
        raise ValueError('Invalid coverage resolution/heights')
    radius = float(level['coverage_radius_m'])
    if not math.isfinite(radius) or radius <= 0:
        raise ValueError('Schedule must supply a positive coverage radius')
    scans = [p for p in level['trajectory'] if p.get('scan_deg', 0)>0]
    if not scans or any(not math.isclose(p['scan_deg'], 360.) for p in scans):
        raise ValueError('Coverage requires explicit full 360-degree scan stops')
    tilts = level.get('scan_plan', {}).get('tilts_deg') or [0.]
    vfov = math.degrees(2*math.atan(math.tan(math.radians(hfov)/2)*height/width))
    low, high = np.asarray(bounds[0]), np.asarray(bounds[1])
    origin = np.floor(low[[0, 2]]/resolution)*resolution
    shape = np.ceil((high[[0, 2]]-origin)/resolution).astype(int)
    mask = np.zeros((shape[1], shape[0]), dtype=np.uint8)
    floor_heights = np.full(mask.shape, np.nan, dtype=np.float32)
    floor_y = float(level['height'])
    if not floor_gt:
        raise ValueError('Coverage requires native semantic floor geometry')
    for iz in range(mask.shape[0]):
        z = float(origin[1]+(iz+.5)*resolution)
        for ix in range(mask.shape[1]):
            x = float(origin[0]+(ix+.5)*resolution)
            hits = trace(sim, [x, floor_y+floor_band+.01, z], [0., -1., 0.], 2*floor_band+.02)
            supports = [h for h in hits if float(h.normal.y) > .75 and any(
                box[0][1]-.06 <= float(h.point.y) <= box[1][1]+.06 for box in floor_gt)]
            if not supports:
                continue
            ground = min(supports, key=lambda h: abs(float(h.point.y)-floor_y))
            y = float(ground.point.y)
            floor_heights[iz, ix] = y
            occupied = trace(sim, [x, y+.04, z], [0., 1., 0.], clearance_height-.04)
            # Centre sampling is explicit: no robot-radius erosion/dilation.
            mask[iz, ix] = 1 if occupied else 2
    iz, ix = np.where(mask==2)
    if not len(ix):
        raise ValueError('Collision-derived free floor domain is empty')
    targets = np.column_stack((origin[0]+(ix+.5)*resolution, floor_heights[iz, ix]+.02,
                               origin[1]+(iz+.5)*resolution))
    outside_navmesh = np.array([not sim.pathfinder.is_navigable(p) for p in targets])
    covered = np.zeros(len(targets), dtype=bool)
    stops = []
    for p in scans:
        eye = np.asarray(p['xyz']) + [0., camera_height, 0.]
        eligible = np.flatnonzero(in_view(eye, targets, radius, vfov, tilts) & ~covered)
        before = int(covered.sum())
        for j in eligible:
            delta = targets[j]-eye
            distance = float(np.linalg.norm(delta))
            if not trace(sim, eye, delta/distance, max(0., distance-.04)):
                covered[j] = True
        stops.append({'stop': int(p['stop']), 'new_area_m2': round((int(covered.sum())-before)*resolution**2, 4),
                      'cumulative_area_m2': round(int(covered.sum())*resolution**2, 4)})
    mask[iz[covered], ix[covered]] = 3
    np.savez_compressed(output, mask=mask, origin_xz=origin, resolution_m=resolution, floor_heights=floor_heights)
    denominator = len(targets)*resolution**2
    numerator = int(covered.sum())*resolution**2
    return {'schema': 'graphapi.tour_coverage.v1', 'scope': 'planned static-scene free-floor visibility at scan stops; not recorded observation coverage',
            'domain': 'native floor-height reference plus full-scene collision-supported floor cell centres with clear vertical body columns; no navmesh mask or robot-radius erosion',
            'free_area_m2': round(denominator, 4), 'covered_area_m2': round(numerator, 4),
            'uncovered_area_m2': round(denominator-numerator, 4), 'coverage_pct': round(100*numerator/denominator, 3),
            'free_area_outside_navmesh_m2': round(int(outside_navmesh.sum())*resolution**2, 4),
            'covered_area_outside_navmesh_m2': round(int((outside_navmesh & covered).sum())*resolution**2, 4),
            'free_cells': len(targets), 'covered_cells': int(covered.sum()), 'obstacle_cells': int((mask==1).sum()),
            'grid_resolution_m': resolution, 'range_m': radius, 'range_source': 'schedule coverage_radius_m (horizontal evaluation range)',
            'camera_height_m': camera_height, 'clearance_height_m': clearance_height,
            'hfov_deg': hfov, 'vfov_deg': vfov, 'image_size': [width, height], 'tilts_deg': tilts,
            'floor_band_m': floor_band, 'grid': file_stamp(output), 'per_stop': stops,
            'limitations': ['Cell-centre grid approximation of physical floor area, not exact polygon area.',
                            'Static stage collisions only; planned dynamic objects are not yet instantiated.',
                            'Coverage includes the floor surface within the configured camera vertical FOV, not object detection quality.',
                            'Motion frames and cross-floor transitions are excluded; only scheduled 360-degree scans contribute.',
                            'Missing/open mesh surfaces cannot contribute a supported floor cell.']}


def annotate_cohort(cohort_path, resolution=.1):
    import habitat_sim
    cohort_path = Path(cohort_path)
    doc = json.loads(cohort_path.read_text())
    for row in doc['scenes']:
        path = Path(row['assets']['schedule']['path'])
        schedule = json.loads(path.read_text())
        if 'coverage' in schedule:
            raise ValueError('Coverage already exists; use a fresh input preparation')
        # Retain all original schedule settings when adding the derived tour.
        original = json.loads(Path(row['assets']['original_schedule']['path']).read_text())
        schedule = {**original, **schedule}
        cfg = habitat_sim.SimulatorConfiguration()
        cfg.scene_id = row['assets']['scene']['path']
        cfg.scene_dataset_config_file = row['assets']['resolved_dataset_config']['path']
        cfg.enable_physics = True
        cfg.load_semantic_mesh = False
        agent = habitat_sim.agent.AgentConfiguration()
        agent.sensor_specifications = []
        summaries = []
        gt = json.loads(Path(row['ground_truth']['path']).read_text())
        floor_gt = [(o['aabb_min_m'], o['aabb_max_m']) for o in gt['ground_truth_objects'] if o['category_name'].strip().lower()=='floor']
        with habitat_sim.Simulator(habitat_sim.Configuration(cfg, [agent])) as sim:
            bb = sim.get_active_scene_graph().get_root_node().cumulative_bb
            bounds = (np.array(bb.min), np.array(bb.max))
            if not np.isfinite(bounds).all() or np.any(bounds[1] <= bounds[0]):
                raise ValueError('Physical scene bounds unavailable')
            for i, level in enumerate(schedule['schedule']):
                grid = path.parent/f'floor-{i}.coverage.npz'
                result = coverage(sim, level, bounds, grid, resolution=resolution, floor_gt=floor_gt)
                result['sources'] = {'scene': row['assets']['scene'], 'native_floor_gt': row['ground_truth'], 'evaluator': file_stamp(__file__)}
                level['coverage'] = result
                summaries.append(result)
        total = sum(r['free_area_m2'] for r in summaries)
        seen = sum(r['covered_area_m2'] for r in summaries)
        schedule['coverage'] = {'schema': 'graphapi.tour_coverage.v1', 'scope': summaries[0]['scope'],
                                'free_area_m2': round(total, 4), 'covered_area_m2': round(seen, 4),
                                'coverage_pct': round(100*seen/total, 3), 'aggregation': 'area-weighted across floors'}
        path.write_text(json.dumps(schedule, indent=2))
        row['assets']['schedule'] = file_stamp(path)
        row['coverage'] = schedule['coverage']
        (path.parent/'plan.json').write_text(json.dumps(row, indent=2))
        print('COVERAGE', row['scene_id'], schedule['coverage'], flush=True)
    cohort_path.write_text(json.dumps(doc, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('cohort', type=Path)
    parser.add_argument('--resolution', type=float, default=.1)
    args = parser.parse_args()
    annotate_cohort(args.cohort, args.resolution)


if __name__ == '__main__':
    main()
