"""Read-only acceptance checks for optimized coverage and native action-clock budgets."""
import argparse
import json
from pathlib import Path

import numpy as np

from .optimize_tours import validate_optimized
from .replay_model import file_stamp


def verify(reference, optimized):
    old, new = json.loads(Path(reference).read_text()), json.loads(Path(optimized).read_text())
    report = validate_optimized(old, new)
    for before, row in zip(old['scenes'], new['scenes']):
        original = json.loads(Path(before['assets']['schedule']['path']).read_text())
        doc = json.loads(Path(row['assets']['schedule']['path']).read_text())
        levels = doc['schedule']
        if len(levels) != len(original['schedule']):
            raise ValueError('Floor count changed')
        for level in levels:
            cov, duration = level['coverage'], level['duration']
            if not duration['valid_complete_tour'] or duration['skipped']:
                raise ValueError('Native executor skipped waypoints')
            for prefix in ('first_lap', 'three_lap'):
                if not np.isclose(duration[prefix+'_frames']/duration['fps'], duration[prefix+'_seconds'], atol=.001):
                    raise ValueError('Action-clock seconds disagree with native ticks')
            scans = sum(p.get('scan_deg', 0)>0 for p in level['trajectory'])
            minimum = scans*len(level['scan_plan']['tilts_deg'])*round(360/duration['turn_step_deg'])
            if duration['first_lap_frames'] < minimum or duration['three_lap_frames'] < 3*minimum:
                raise ValueError('Duration omits scheduled scan rotations')
            if file_stamp(duration['executor']['path']) != duration['executor']:
                raise ValueError('Native executor source changed since action counting')
            with np.load(cov['grid']['path']) as grid:
                free, covered = int((grid['mask']>=2).sum()), int((grid['mask']==3).sum())
                area = float(grid['resolution_m'])**2
                if not np.isclose(free*area, cov['free_area_m2']) or not np.isclose(covered*area, cov['covered_area_m2']):
                    raise ValueError('Physical area differs from saved grid')
                if not np.isclose(100*covered/free, cov['coverage_pct'], atol=.001):
                    raise ValueError('Coverage percentage differs from saved grid')
        for plan in row['dynamic_plans']:
            if plan.get('script') and file_stamp(plan['script']['path']) != plan['script']:
                raise ValueError('Compiled object script changed')
        for field in ('free_area_m2', 'covered_area_m2'):
            if not np.isclose(sum(level['coverage'][field] for level in levels), doc['coverage'][field]):
                raise ValueError('Whole-scene area is not the floor sum')
        if not np.isclose(100*doc['coverage']['covered_area_m2']/doc['coverage']['free_area_m2'],
                          doc['coverage']['coverage_pct'], atol=.001):
            raise ValueError('Whole-scene coverage uses the wrong denominator')
        for field in ('first_lap_seconds', 'three_lap_seconds'):
            if not np.isclose(sum(level['duration'][field] for level in levels), doc['duration'][field], atol=.001):
                raise ValueError('Scene duration is not the floor sum')
        if row['duration'] != doc['duration'] or row['coverage'] != doc['coverage']:
            raise ValueError('Manifest differs from schedule')
    report['action_clock_verified'] = True
    report['native_baseline_algorithms_executed'] = False
    report['multi_floor_localization_certified'] = False
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('reference', type=Path)
    parser.add_argument('optimized', type=Path)
    args = parser.parse_args()
    print(json.dumps(verify(args.reference, args.optimized), indent=2))


if __name__ == '__main__':
    main()
