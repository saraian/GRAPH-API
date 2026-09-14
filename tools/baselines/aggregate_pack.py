"""Aggregate a completed baseline pack with the authoritative GRAPH-API metrics."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np

from .graphapi_eval import (action_timeline, evaluators, final_ground_truth,
                            hov_manifest, json_default, read_json, read_rows,
                            stamp, _without_retrieval)


def latest_pair(scene_root):
    pairs = sorted(scene_root.glob('pair-attempt-*'),
                   key=lambda path: int(path.name.rsplit('-', 1)[1]))
    for path in reversed(pairs):
        marker = path / 'pair-status.json'
        if marker.is_file():
            value = read_json(marker)
            if value.get('complete') and value.get('phase') == 'evaluation_complete':
                return path, value
    raise ValueError('No completed pair under ' + str(scene_root))


def latency(values, scope):
    values = np.asarray(values, dtype=float)
    if len(values) and (not np.isfinite(values).all() or np.any(values < 0)):
        raise ValueError('Invalid pack latency values')
    return {'samples': len(values),
        'mean_ms': float(values.mean() * 1000) if len(values) else None,
        'p50_ms': float(np.percentile(values, 50) * 1000) if len(values) else None,
        'p95_ms': float(np.percentile(values, 95) * 1000) if len(values) else None,
        'scope': scope}


def temporal_summary(reports):
    actions = []
    for scene, report in reports:
        actions.extend(dict(row, scene=scene)
                       for row in report['temporal_object_actions']['actions'])
    appearances = [row for row in actions if row['action'] in ('spawn', 'move')]
    removals = [row for row in actions if row['action'] == 'remove']
    confirmed = [row.get('removal_confirmed_in_final_map',
                         row.get('removal_confirmed_active_layer')) for row in removals]
    evaluated_removals = [value for value in confirmed if value is not None]
    detected = [row for row in appearances
                if row.get('first_detection_frame', row.get('first_evidence_frame')) is not None]
    return {'status': 'supported', 'actions': actions, 'action_count': len(actions),
        'appearance_action_count': len(appearances),
        'appearance_actions_with_evidence': len(detected),
        'appearance_action_recall_pct': (100 * len(detected) / len(appearances)
                                         if appearances else None),
        'removals_evaluated': len(evaluated_removals),
        'removals_confirmed': sum(evaluated_removals),
        'removal_confirmation_pct': (100 * sum(evaluated_removals) /
                                     len(evaluated_removals)
                                     if evaluated_removals else None)}


def aggregate(pack_root, output):
    pack_root, output = Path(pack_root).resolve(), Path(output).resolve()
    plan = read_json(pack_root / 'pack-manifest.json')
    hov_manifests = []
    reports = {'clio': [], 'hovsg': []}
    latency_values = {'clio': [], 'hovsg': []}
    pair_stamps = []
    for entry in plan['scenes']:
        scene = entry['scene']
        scene_root = pack_root / 'runs' / scene
        pair, pair_state = latest_pair(scene_root)
        recording = scene_root / 'recording'
        pair_stamps.append(stamp(pair / 'pair-status.json'))
        for baseline in ('clio', 'hovsg'):
            report_path = Path(pair_state['jobs'][baseline]['evaluation']['output'])
            reports[baseline].append((scene, read_json(report_path)))
        gt = read_json(recording / 'static-gt.json')
        frames, actions = action_timeline(recording)
        gt = final_ground_truth(gt, frames, actions)
        hov_result = Path(pair_state['jobs']['hovsg']['final_output'])
        hov_manifests.append(hov_manifest(gt, recording, hov_result))
        latency_values['hovsg'].extend(row['stage_wall_time_s'] for row in
            read_rows(hov_result / 'native_observations/observations.jsonl')
            if row['stage'] == 'sam_clip')
        clio_result = Path(pair_state['jobs']['clio']['final_output'])
        observer = read_json(clio_result / 'native_observer.json')
        if observer.get('input_timing_origin') != 'before_rgb_publish':
            raise ValueError('Pack Clio latency is not causally timestamped')
        latency_values['clio'].extend(row['paired_transport_latency_s'] for row in
            read_rows(clio_result / 'native_receipts.jsonl')
            if row['topic'] == '/dominic/forward/semantic/image_raw'
            and row.get('paired_transport_latency_s') is not None)
    metrics, objects = evaluators()
    hov_tables = _without_retrieval(metrics.evaluate(hov_manifests, output))
    filtered = [metrics.filtered_scene(manifest) for manifest in hov_manifests]
    geometry, matches = objects.evaluate_geometry(filtered)
    hov_tables['table_iv_objects_v2'] = {
        **geometry, **objects.evaluate_labels(filtered, matches)}
    clio_sizes = [report['tables']['table_vii_representation']['size_mb_total']
                  for _, report in reports['clio']]
    clio_times = [report['tables']['construction_time_s']
                  for _, report in reports['clio']]
    clio_reason = ('Task-free Clio has no native floor, room, or object-instance layer; '
                   'semantic primitives remain a descriptive proxy.')
    clio_tables = {
        'table_ii_floor_regions': {'status': 'unsupported', 'reason': clio_reason},
        'table_iii_rooms': {'status': 'unsupported', 'reason': clio_reason},
        'table_iv_objects': {'status': 'unsupported', 'reason': clio_reason},
        'table_vi_room_objects': {'status': 'unsupported', 'reason': clio_reason},
        'table_vii_representation': {'status': 'supported',
                                     'size_mb_total': sum(clio_sizes)},
        'construction_time_s': sum(clio_times),
        'protocol_exclusions': ['Table V retrieval/navigation: excluded by experiment protocol']}
    result = {'schema': 'graphapi.baseline_pack_end_to_end_eval.v1', 'pack': plan['pack'],
        'complete': True, 'scenes': [row['scene'] for row in plan['scenes']],
        'baselines': {
            'clio': {'tables': clio_tables,
                'temporal_object_actions': temporal_summary(reports['clio']),
                'latency': latency(latency_values['clio'],
                    'External RGB publish start to native Clio semantic output receipt')},
            'hovsg': {'tables': hov_tables,
                'temporal_object_actions': temporal_summary(reports['hovsg']),
                'latency': latency(latency_values['hovsg'],
                    'Native HOV-SG SAM/CLIP call-to-return per sampled frame')}},
        'pair_executions': pair_stamps, 'table_v_retrieval': {'status': 'excluded_by_protocol'},
        'native_algorithms_modified': False}
    output.mkdir(parents=True, exist_ok=False)
    (output / 'report.json').write_text(json.dumps(
        result, indent=2, allow_nan=False, default=json_default) + '\n')
    object_v2 = hov_tables['table_iv_objects_v2']
    rows = [
        {'baseline': 'clio', 'object_precision_pct': None, 'object_recall_pct': None,
         'object_f1_pct': None, **{key: result['baselines']['clio']['temporal_object_actions'][key]
         for key in ('action_count', 'appearance_action_recall_pct',
                     'removals_evaluated', 'removals_confirmed')}},
        {'baseline': 'hovsg', 'object_precision_pct': object_v2.get('precision_pct'),
         'object_recall_pct': object_v2.get('recall_pct'), 'object_f1_pct': object_v2.get('f1_pct'),
         **{key: result['baselines']['hovsg']['temporal_object_actions'][key]
         for key in ('action_count', 'appearance_action_recall_pct',
                     'removals_evaluated', 'removals_confirmed')}}]
    with (output / 'metrics.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pack-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    result = aggregate(args.pack_root, args.output)
    print(json.dumps({'complete': result['complete'], 'output': str(args.output)}))


if __name__ == '__main__':
    main()
