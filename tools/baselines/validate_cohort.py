"""Check the prepared cohort and reject incomplete/altered input plans."""
import argparse
import json
from pathlib import Path

import numpy as np

from .replay_model import file_stamp


def validate(path, expected_per_group=5, require_coverage=True):
    doc = json.loads(Path(path).read_text())
    rows = doc['scenes']
    if len({r['scene_id'] for r in rows}) != len(rows):
        raise ValueError('Duplicate scene')
    for group in ('single-floor', 'multi-floor'):
        if sum(r['cohort_group']==group for r in rows) != expected_per_group:
            raise ValueError('Incomplete cohort group: '+group)
    summary = []
    for row in rows:
        stamps = list(row['assets'].values()) + [row['ground_truth']]
        stamps += [p['script'] for p in row['dynamic_plans'] if p.get('script')]
        for stamp in stamps:
            if file_stamp(stamp['path'])['sha256'] != stamp['sha256']:
                raise ValueError('Asset hash mismatch: '+stamp['path'])
        levels = json.loads(Path(row['assets']['schedule']['path']).read_text())['schedule']
        if require_coverage:
            for level in levels:
                cov = level.get('coverage')
                if not cov or cov['free_cells'] <= 0 or not 0 <= cov['covered_cells'] <= cov['free_cells']:
                    raise ValueError('Missing or invalid free-floor coverage')
                if file_stamp(cov['grid']['path'])['sha256'] != cov['grid']['sha256']:
                    raise ValueError('Coverage grid hash mismatch')
                if not np.isclose(cov['covered_area_m2']/cov['free_area_m2']*100, cov['coverage_pct'], atol=.001):
                    raise ValueError('Coverage percentage has the wrong denominator')
                with np.load(cov['grid']['path']) as grid:
                    if int((grid['mask']>=2).sum()) != cov['free_cells'] or int((grid['mask']==3).sum()) != cov['covered_cells']:
                        raise ValueError('Coverage grid disagrees with reported counts')
        original = json.loads(Path(row['assets']['original_schedule']['path']).read_text())['schedule']
        if len(levels) != len(original) or len(row['floor_transitions']) != len(levels)-1:
            raise ValueError('Missing floor/transition')
        if (len(levels)==1) != (row['cohort_group']=='single-floor'):
            raise ValueError('Floor group mismatch')
        count = 0
        for floor, (level, raw) in enumerate(zip(levels, original)):
            if len(level['trajectory']) != len(raw['trajectory']):
                raise ValueError('Tour was shortened')
            for a, b in zip(level['trajectory'], raw['trajectory']):
                if {k: v for k, v in a.items() if k != 'xyz'} != {k: v for k, v in b.items() if k != 'xyz'}:
                    raise ValueError('Projection changed scan metadata')
            plan = next(p for p in row['dynamic_plans'] if p['floor_index']==floor)
            if not plan['events']:
                if not plan.get('reason'):
                    raise ValueError('Missing explanation for floor without changes')
                continue
            script = json.loads(Path(plan['script']['path']).read_text())
            steps = [s for s in script['steps'] if s['action'] != 'wait']
            if [s['action'] for s in steps] != ['spawn', 'move', 'remove']:
                raise ValueError('Invalid object lifecycle')
            scans = [p for p in level['trajectory'] if p.get('scan_deg', 0)>0]
            previous = -1
            for event, step in zip(plan['events'], steps):
                index = event['scan_visit_number']-1
                if index <= previous or event['trigger'] != step['at_waypoint'] or int(scans[index]['stop']) != step['at_waypoint']['stop']:
                    raise ValueError('Action trigger order/scan mismatch')
                if not np.allclose(event['robot_position'], scans[index]['xyz']):
                    raise ValueError('PNG robot trigger differs from tour')
                target = step.get('position', steps[1]['position'])
                if not np.allclose(event['object_position'], target):
                    raise ValueError('PNG object target differs from compiled script')
                previous = index
                count += 1
        if count < 3:
            raise ValueError('Scene has no complete dynamic sequence')
        gt = json.loads(Path(row['ground_truth']['path']).read_text())
        if len(gt['ground_truth_objects']) != row['native_gt_objects']:
            raise ValueError('GT count mismatch')
        if row['annotated_room_regions'] < 2:
            raise ValueError('Scene lacks multiple room regions')
        summary.append({'scene': row['scene_id'], 'floors': len(levels), 'scans': row['scan_stops'],
                        'gt_objects': row['native_gt_objects'], 'changes': count})
    return {'complete': True, 'scope': 'input plan verification, not executed baseline/certification', 'scenes': summary}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('cohort', type=Path)
    args = parser.parse_args()
    result = validate(args.cohort)
    (args.cohort.parent/'validation.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
