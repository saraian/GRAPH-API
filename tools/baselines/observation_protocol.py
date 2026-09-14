"""One GT-independent observation contract for GRAPH-API and every baseline."""
from __future__ import annotations

import hashlib
import json
import os
import shutil
from pathlib import Path

import numpy as np


def read_rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line]


def action_steps(script):
    data = json.loads(Path(script).read_text()) if isinstance(script, (str, Path)) else script
    return [step for step in data.get('steps', [])
            if step.get('action') in {'spawn', 'move', 'remove'}]


def validate_script_contract(script, schedule, laps):
    """Reject a plan whose world changes cannot be observed by the fixed tour."""
    steps = action_steps(script)
    scans = [point for point in schedule['trajectory'] if point.get('scan_deg', 0) > 0]
    index_by_stop = {int(point['stop']): index for index, point in enumerate(scans)}
    checked = []
    for index, step in enumerate(steps):
        trigger = step.get('at_waypoint')
        expected = step.get('expected_observation')
        if not isinstance(trigger, dict) or not isinstance(expected, dict):
            raise ValueError(f'Action {index} lacks the waypoint observation contract')
        trigger_stop = int(trigger['stop'])
        trigger_lap = int(trigger.get('lap', 0))
        if trigger_stop not in index_by_stop:
            raise ValueError(f'Action {index} names an unknown trigger stop')
        scan_index = index_by_stop[trigger_stop]
        if step['action'] in {'spawn', 'move'}:
            if scan_index + 1 >= len(scans):
                raise ValueError(f'Action {index} has no next waypoint')
            required = {'stop': int(scans[scan_index + 1]['stop']),
                        'lap': trigger_lap, 'state': 'present',
                        'timing': 'next_waypoint_scan'}
        else:
            required = {'stop': trigger_stop, 'lap': trigger_lap + 1,
                        'state': 'absent',
                        'timing': 'same_waypoint_following_lap'}
        if expected != required:
            raise ValueError(
                f'Action {index} observation contract is {expected}, expected {required}')
        if required['lap'] >= int(laps):
            raise ValueError(f'Action {index} has no observation within the requested laps')
        checked.append({'action_index': index, 'action': step['action'],
                        'trigger': dict(trigger), 'expected_observation': dict(expected)})
    return {'schema': 'graphapi.action_observation_plan.v1', 'complete': True,
            'actions': checked, 'route_changed': False,
            'policy': ('spawn/move after scan i is observed at scan i+1; removal '
                       'after scan i is observed at the same stop on lap+1')}


def scan_windows(frames, position_tolerance_m=.03):
    """Return the contiguous stationary frame window for every completed scan.

    A scan is completed by the TOUR, so only a tour frame may carry the event. The
    runtime also attaches the trigger scan's scan_complete event to the frame it
    captures right after an object action, and counting every carrier saw one scan
    twice and refused the whole recording.

    The window is anchored to the carrier frame's own recorded ``base_position``. The
    event's ``observed_from`` is the SCHEDULED waypoint, and the executed navmesh pose
    stands up to a third of a metre from it, so anchoring there rejected every frame
    and left the window holding the carrier alone.
    """
    windows = {}
    for row in frames:
        if row.get('reason') != 'tour':
            continue
        events = row.get('event') or []
        if isinstance(events, dict):
            events = [events]
        for event in events:
            if event.get('event') != 'scan_complete':
                continue
            end = int(row['index'])
            anchor = np.asarray(row.get('base_position'), dtype=float)
            if anchor.shape != (3,):
                anchor = np.asarray(event.get('observed_from', event['xyz']), dtype=float)
            start = end
            while start > 0:
                previous = frames[start - 1]
                if previous.get('reason') != 'tour':
                    break
                position = np.asarray(previous.get('base_position'), dtype=float)
                if position.shape != (3,) or float(np.linalg.norm(
                        position[[0, 2]] - anchor[[0, 2]])) > position_tolerance_m:
                    break
                start -= 1
            key = (int(event['lap']), int(event['stop']))
            if key in windows:
                raise ValueError(f'Duplicate completed scan event {key}')
            windows[key] = list(range(start, end + 1))
    return windows


def _evaluation_id(result):
    return str(result.get('evaluation_object_id', result['object_id']))


def validate_recording_visibility(recording, script, minimum_visible_frames=1):
    """Use GT only to accept or void a completed acquisition."""
    recording = Path(recording)
    frames = read_rows(recording / 'frames.jsonl')
    actions = read_rows(recording / 'object_actions.jsonl')
    steps = action_steps(script)
    if len(actions) != len(steps):
        raise ValueError('Recorded and planned action counts differ')
    windows = scan_windows(frames)
    semantic_ids = {}
    rows = []
    for index, (action, step) in enumerate(zip(actions, steps)):
        if action['action'] != step['action']:
            raise ValueError(f'Recorded action {index} differs from its plan')
        result = action['result']
        evaluation_id = _evaluation_id(result)
        if result.get('gt_semantic_id') is not None:
            semantic_ids[evaluation_id] = int(result['gt_semantic_id'])
        semantic_id = semantic_ids.get(evaluation_id)
        if semantic_id is None:
            raise ValueError(f'Action {index} has no GT semantic identity')
        expected = step['expected_observation']
        key = (int(expected['lap']), int(expected['stop']))
        indices = windows.get(key)
        if not indices:
            raise ValueError(f'Action {index} expected scan {key} was not recorded')
        if min(indices) <= int(action['after_frame']):
            raise ValueError(f'Action {index} expected observation precedes the action')
        visible = [frame_index for frame_index in indices
                   if semantic_id in frames[frame_index].get('visible_semantic_ids', [])]
        if expected['state'] == 'present':
            passed = len(visible) >= int(minimum_visible_frames)
            if not passed:
                raise ValueError(
                    f'Action {index} {action["action"]} is not GT-visible at '
                    f'its scheduled next-waypoint scan {key}')
        else:
            passed = not visible
            if not passed:
                raise ValueError(
                    f'Action {index} removal is still GT-visible at its following-lap scan {key}')
            semantic_ids.pop(evaluation_id, None)
        rows.append({'action_index': index, 'action': action['action'],
                     'evaluation_object_id': evaluation_id, 'semantic_id': semantic_id,
                     'expected_observation': dict(expected), 'scan_frame_indices': indices,
                     'visible_frame_indices': visible, 'passed': passed})
    return {'schema': 'graphapi.action_observation_gate.v1', 'complete': True,
            'minimum_visible_frames': int(minimum_visible_frames), 'actions': rows,
            'gt_usage': 'post-acquisition accept-or-void gate only; never baseline input selection'}


def select_shared_frames(frames, script, stride, scan_samples=8):
    """Uniform travel coverage plus fixed views from planned observation scans."""
    if stride < 1 or scan_samples < 1:
        raise ValueError('Sampling stride and scan sample count must be positive')
    selected = set(range(0, len(frames), int(stride)))
    reasons = {index: {'uniform_stride'} for index in selected}
    windows = scan_windows(frames)
    for step in action_steps(script):
        expected = step['expected_observation']
        key = (int(expected['lap']), int(expected['stop']))
        indices = windows.get(key)
        if not indices:
            raise ValueError(f'Planned observation scan {key} is absent')
        picks = np.linspace(0, len(indices) - 1,
                            min(int(scan_samples), len(indices)), dtype=int)
        reason = 'scheduled_' + expected['state'] + '_scan'
        for pick in picks:
            frame_index = indices[int(pick)]
            selected.add(frame_index)
            reasons.setdefault(frame_index, set()).add(reason)
    ordered = sorted(selected)
    return ordered, [{'frame_index': index, 'reasons': sorted(reasons[index])}
                     for index in ordered]


def prepare_shared_recording(recording, script, output, stride, scan_samples=8):
    """Build the one GT-free frame sequence consumed by every evaluated system."""
    recording, output = Path(recording), Path(output)
    frames = read_rows(recording / 'frames.jsonl')
    indices, selections = select_shared_frames(frames, script, stride, scan_samples)
    output.mkdir(parents=True, exist_ok=False)
    for folder in ('rgb', 'depth', 'depth_m', 'pose'):
        (output / folder).mkdir()
    sampled = []
    for native_index, source_index in enumerate(indices):
        source = frames[source_index]
        stem = f'{native_index:06d}'
        row = {key: value for key, value in source.items()
               if key not in {'visible_semantic_ids', 'dynamic_ground_truth'}}
        row.update(index=native_index, stem=stem, source_frame_index=source_index,
                   source_stem=source['stem'], source_time_s=source['time_s'])
        sampled.append(row)
        for folder, suffixes in (('rgb', ('.png',)), ('depth', ('.png',)),
                                 ('pose', ('.txt',)), ('depth_m', ('.npz', '.npy'))):
            source_path = next((recording / folder / (source['stem'] + suffix)
                                for suffix in suffixes
                                if (recording / folder / (source['stem'] + suffix)).is_file()), None)
            if source_path is None:
                raise FileNotFoundError(f'Missing {folder} source for frame {source_index}')
            target = output / folder / (stem + source_path.suffix)
            # A HARD link, not a symbolic one. The native containers mount this shared
            # input tree and nothing else, so a symlink carrying the source recording's
            # absolute host path resolves to nothing inside them and the baseline dies
            # on the first frame. A hard link costs no space and needs no second mount.
            # Different devices cannot share an inode, so copy there.
            try:
                os.link(source_path, target)
            except OSError:
                shutil.copy2(source_path, target)
    (output / 'frames.jsonl').write_text(
        ''.join(json.dumps(row) + '\n' for row in sampled))
    acquisition = json.loads((recording / 'acquisition.json').read_text())
    acquisition.update(frames=len(indices), ground_truth_recorded=False,
                       source_recording=str(recording), source_frames=len(frames),
                       observation_protocol='shared GT-independent schedule sampling')
    (output / 'acquisition.json').write_text(json.dumps(acquisition, indent=2) + '\n')
    receipt = {'schema': 'graphapi.shared_observation_manifest.v1', 'complete': True,
        'source_recording': str(recording), 'source_frames': len(frames),
        'source_frames_sha256': hashlib.sha256(
            (recording / 'frames.jsonl').read_bytes()).hexdigest(),
        'uniform_stride': int(stride), 'scheduled_scan_samples': int(scan_samples),
        'sampled_frames': len(indices), 'sampled_source_frame_indices': indices,
        'selections': selections, 'ground_truth_in_baseline_input': False,
        'method': ('Uniform source-frame stride plus fixed evenly spaced headings from every '
                   'planned post-action observation scan; no GT visibility is read for sampling.')}
    (output / 'observation_manifest.json').write_text(json.dumps(receipt, indent=2) + '\n')
    return output, indices, receipt
