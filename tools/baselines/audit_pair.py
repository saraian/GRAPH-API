"""Aggregate completed baseline end-to-end reports without recomputing them."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np

from .graphapi_eval import SCHEMA, stamp


def read(path):
    return json.loads(Path(path).read_text())


def latency(values, scope):
    values = np.asarray(values, dtype=float)
    if len(values) and (not np.isfinite(values).all() or (values < 0).any()):
        raise ValueError('Invalid measured latency samples')
    return {'samples': len(values),
        'mean_ms': float(values.mean() * 1000) if len(values) else None,
        'p95_ms': float(np.percentile(values, 95) * 1000) if len(values) else None,
        'scope': scope}


def clio_timing(matched, observer):
    """Compatibility helper retaining the causal latency guard."""
    if observer.get('input_timing_origin') == 'before_rgb_publish':
        return {'latency': latency(matched, observer['latency_scope'])}
    deltas = np.asarray(matched, dtype=float)
    if not np.isfinite(deltas).all():
        raise ValueError('Non-finite observer receipt differences')
    return {'latency': latency([], 'Unavailable: independent subscriber order is not causal'),
        'independent_receipt_delta': {'samples': len(matched),
            'negative_samples': int((deltas < 0).sum()),
            'mean_ms': float(deltas.mean() * 1000) if len(deltas) else None,
            'scope': 'Signed independent observer receipt difference; not latency'}}


def _table_status(table):
    if isinstance(table, dict) and table.get('status'):
        return table['status']
    return 'supported'


def _row(report):
    tables = report['tables']
    object_table = tables.get('table_iv_objects_v2', {})
    temporal = report['temporal_object_actions']
    actions = temporal.get('actions', [])
    removals = [row.get('removal_confirmed_in_final_map',
                        row.get('removal_confirmed_active_layer'))
                for row in actions if row.get('action') == 'remove']
    removals = [value for value in removals if value is not None]
    return {
        'baseline': report['baseline'],
        'table_ii': _table_status(tables.get('table_ii_floor_regions')),
        'table_iii': _table_status(tables.get('table_iii_rooms')),
        'table_iv': _table_status(tables.get('table_iv_objects')),
        'table_vi': _table_status(tables.get('table_vi_room_objects')),
        'table_vii': _table_status(tables.get('table_vii_representation')),
        'object_precision_pct': object_table.get('precision_pct'),
        'object_recall_pct': object_table.get('recall_pct'),
        'object_f1_pct': object_table.get('f1_pct'),
        'semantic_accuracy_pct': object_table.get('semantic_accuracy_pct'),
        'action_count': len(actions),
        'appearance_actions_with_evidence': sum(
            row.get('first_detection_frame', row.get('first_evidence_frame')) is not None
            for row in actions if row.get('action') in ('spawn', 'move')),
        'removals_evaluated': len(removals),
        'removals_confirmed': sum(removals),
        'mean_latency_ms': report['latency'].get('mean_ms'),
        'latency_samples': report['latency'].get('samples'),
    }


def audit(root, output, min_clio_fraction=.95):
    del min_clio_fraction  # retained for compatibility with older callers
    root, output = Path(root).resolve(), Path(output).resolve()
    pair = read(root / 'pair-status.json')
    if pair.get('phase') != 'evaluation_complete' or not pair.get('complete'):
        raise ValueError('The native pair and end-to-end evaluation must complete first')
    reports = {}
    for baseline in ('clio', 'hovsg'):
        path = Path(pair['jobs'][baseline]['evaluation']['output'])
        report = read(path)
        if report.get('schema') != SCHEMA or report.get('baseline') != baseline:
            raise ValueError(f'Invalid {baseline} end-to-end report')
        if report.get('native_algorithm_modified') is not False:
            raise ValueError(f'{baseline} report does not attest an unchanged native algorithm')
        if report['temporal_object_actions'].get('status') != 'supported':
            raise ValueError(f'{baseline} temporal object-action evaluation is unsupported')
        reports[baseline] = report
    rows = [_row(reports[name]) for name in ('clio', 'hovsg')]
    output.mkdir(parents=True, exist_ok=False)
    with (output / 'metrics.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    result = {'schema': 'graphapi.baseline_pair_end_to_end_audit.v1',
        'complete': True, 'scene': reports['clio']['scene'],
        'pair_execution': stamp(root / 'pair-status.json'),
        'baseline_reports': {name: {'source': stamp(
            pair['jobs'][name]['evaluation']['output']), 'summary': _row(report)}
            for name, report in reports.items()},
        'table_v_retrieval': {'status': 'excluded_by_protocol'},
        'native_algorithms_modified': False,
        'interpretation': [
            'Unsupported native tables remain null/unsupported.',
            'Clio semantic-primitive diagnostics are not object-instance scores.',
            'Appearance/disappearance results come from scheduled GT and saved native outputs.'
        ]}
    (output / 'report.json').write_text(json.dumps(result, indent=2) + '\n')
    lines = ['# Baseline end-to-end evaluation', '',
        f"Scene: `{result['scene']}`", '',
        '| Baseline | II | III | IV | VI | VII | Object P/R/F1 | Actions with appearance evidence | Confirmed removals |',
        '|---|---|---|---|---|---|---|---:|---:|']
    for row in rows:
        score = (f"{row['object_precision_pct']}/{row['object_recall_pct']}/"
                 f"{row['object_f1_pct']}" if row['object_precision_pct'] is not None
                 else 'unsupported')
        lines.append(f"| {row['baseline']} | {row['table_ii']} | {row['table_iii']} | "
                     f"{row['table_iv']} | {row['table_vi']} | {row['table_vii']} | "
                     f"{score} | {row['appearance_actions_with_evidence']} | "
                     f"{row['removals_confirmed']}/{row['removals_evaluated']} |")
    lines += ['', 'Table V retrieval/navigation is excluded by this experiment protocol.',
        '', 'See the per-baseline JSON reports for thresholds, denominators, provenance and limitations.']
    (output / 'REPORT.md').write_text('\n'.join(lines) + '\n')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    report = audit(args.run_root, args.output)
    print(json.dumps({'report': str(args.output / 'report.json'),
                      'complete': report['complete']}))


if __name__ == '__main__':
    main()
