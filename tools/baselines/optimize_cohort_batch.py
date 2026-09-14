"""Isolated CPU workers for coverage planning; aggregate only completed scene plans."""
import argparse
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .optimize_tours import validate_optimized
from .replay_model import file_stamp


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cohort', type=Path, default=Path('artifacts/baselines/cohort-10/cohort.json'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=3)
    parser.add_argument('--reuse', type=Path, action='append', default=[])
    parser.add_argument('--graph-api-root', type=Path, default=Path(__file__).resolve().parents[2])
    args = parser.parse_args()
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    old = json.loads(args.cohort.read_text())
    reuse = {}
    for path in args.reuse:
        part = json.loads(path.read_text())
        validate_optimized({'scenes': [r for r in old['scenes'] if r['scene_id'] in {p['scene_id'] for p in part['scenes']}]}, part)
        reuse.update({r['scene_id']: r for r in part['scenes']})
    def worker(row):
        scene = row['scene_id']
        if scene in reuse:
            return reuse[scene]
        output = args.output/'parts'/scene
        log = args.output/(scene+'.log')
        with log.open('w') as stream:
            subprocess.run([sys.executable, '-m', 'tools.baselines.optimize_tours', '--cohort', str(args.cohort),
                            '--output', str(output), '--scenes', scene,
                            '--graph-api-root', str(args.graph_api_root)], stdout=stream, stderr=subprocess.STDOUT, check=True)
        result = json.loads((output/'cohort.json').read_text())['scenes'][0]
        print('DONE', scene, result['coverage']['coverage_pct'], result['duration'], flush=True)
        return result
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        rows = list(pool.map(worker, old['scenes']))
    doc = {**old, 'scenes': rows, 'source_cohort': file_stamp(args.cohort), 'optimizer_batch': file_stamp(__file__)}
    validation = validate_optimized(old, doc)
    (args.output/'cohort.json').write_text(json.dumps(doc, indent=2))
    (args.output/'validation.json').write_text(json.dumps(validation, indent=2))
    from .cohort_plot import export_cohort
    export_cohort(args.output/'cohort.json')


if __name__ == '__main__':
    main()
