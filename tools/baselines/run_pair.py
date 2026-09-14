"""Run unchanged native Clio and HOV-SG against one completed shared recording."""
import argparse
import concurrent.futures
import hashlib
import json
import os
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

from .observation_protocol import (prepare_shared_recording,
                                   validate_recording_visibility,
                                   validate_script_contract)


def write(path, value):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, indent=2) + '\n')
    temp.replace(path)


def prune_regenerable(output, names):
    """Remove large reproducible intermediates after native output is verified."""
    pruned = []
    for name in names:
        path = Path(output) / name
        if not path.is_file():
            continue
        digest = hashlib.sha256()
        with path.open('rb') as stream:
            for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
                digest.update(block)
        pruned.append({'path': str(path), 'bytes': path.stat().st_size,
                       'sha256': digest.hexdigest(),
                       'reconstruction': ('Generated deterministically from the measurement schedule, '
                                          'simulator assets and frozen external adapter configuration; '
                                          'the complete source recording remains through evaluation.')})
        path.unlink()
    return pruned


def validate(recording, schedule_path, script_path, expected_laps):
    meta = json.loads((recording / 'acquisition.json').read_text())
    if not meta['complete'] or meta['requested_laps'] != expected_laps:
        raise ValueError(f'A completed {expected_laps}-lap acquisition is required')
    rows = [json.loads(x) for x in (recording / 'frames.jsonl').read_text().splitlines()]
    if len(rows) != meta['frames']:
        raise ValueError('Input frame count mismatch')
    schedule = json.loads(schedule_path.read_text())
    levels = schedule['schedule']
    if len(levels) != 1:
        raise ValueError('This verification pair requires one floor')
    stops = [p['stop'] for p in levels[0]['trajectory'] if p.get('scan_deg', 0) > 0]
    expected = [(lap, stop) for lap in range(expected_laps) for stop in stops]
    actual = [(e['lap'], e['stop']) for row in rows if row['reason'] == 'tour'
        for e in (row['event'] or [])]
    if actual != expected:
        raise ValueError('Observed scan sequence differs from the complete two-lap schedule')
    if hashlib.sha256(schedule_path.read_bytes()).hexdigest() != meta['schedule_sha256']:
        raise ValueError('Acquisition used another schedule')
    for i, row in enumerate(rows):
        if row['index'] != i or (i and row['time_s'] <= rows[i - 1]['time_s']):
            raise ValueError('Invalid frame ordering or timestamps')
        for folder, suffix in [('rgb', '.png'), ('depth', '.png'), ('depth_m', '.npz'),
                ('pose', '.txt'), ('semantic', '.npz')]:
            if not (recording / folder / (row['stem'] + suffix)).is_file():
                raise ValueError(f'Missing {folder} for frame {i}')
    actions = [json.loads(x) for x in (recording / 'object_actions.jsonl').read_text().splitlines()]
    script = json.loads(script_path.read_text())
    plan_contract = validate_script_contract(script, levels[0], expected_laps)
    planned_actions = [step['action'] for step in script.get('steps', [])
                       if step.get('action') in {'spawn', 'move', 'remove'}]
    if ([x['action'] for x in actions] != planned_actions or
            not all(x['result']['success'] for x in actions)):
        raise ValueError('Dynamic lifecycle did not complete')
    if hashlib.sha256(script_path.read_bytes()).hexdigest() != meta['script_sha256']:
        raise ValueError('Acquisition used another object script')
    visibility = validate_recording_visibility(recording, script_path)
    return {'frames': len(rows), 'scans': len(actual), 'laps': expected_laps,
        'capture_duration_s': rows[-1]['time_s'], 'actions': len(actions),
        'action_observation_plan': plan_contract,
        'action_observation_gate': visibility,
        'acquisition_sha256': hashlib.sha256((recording / 'acquisition.json').read_bytes()).hexdigest(),
        'frames_sha256': hashlib.sha256((recording / 'frames.jsonl').read_bytes()).hexdigest()}


def native_state(root):
    return {name: {'head': subprocess.check_output(['git', '-C', str(root / name), 'rev-parse', 'HEAD'], text=True).strip(),
        'tracked_changes': subprocess.check_output(['git', '-C', str(root / name), 'status', '--porcelain', '--untracked-files=no'], text=True)}
        for name in ('Clio-Baseline', 'HOV-Baseline')}


def run(args):
    root = args.run_root.resolve()
    status_path = root / 'pair-status.json'
    if status_path.exists():
        raise FileExistsError('Pair already submitted; inspect its status before launching again')
    gpu_by_name = {'clio': args.clio_gpu if args.clio_gpu is not None else args.gpu,
                   'hovsg': args.hov_gpu if args.hov_gpu is not None else args.gpu}
    if args.parallel_distinct_gpus and gpu_by_name['clio'] == gpu_by_name['hovsg']:
        raise ValueError('Parallel native execution requires distinct physical GPUs')
    status = {'phase': 'waiting_for_acquisition', 'complete': False, 'gpu': args.gpu,
        'baseline_gpus': gpu_by_name,
        'laps': args.laps, 'clio_mode': args.clio_mode,
        'clio_tasks': args.clio_tasks,
        'clio_segmentation_confidence': args.clio_segmentation_confidence,
        'parallel': args.parallel_distinct_gpus,
        'timing_scope': (('Concurrent native execution on distinct GPUs; acquisition is shared, '
                          'and each baseline wall time is measured independently')
                         if args.parallel_distinct_gpus else
                         ('Sequential native execution; acquisition is shared, and baseline '
                          'wall times do not include the other baseline')),
        'jobs': {}, 'created_at': time.time()}
    write(status_path, status)
    recording = root / 'recording'
    while not json.loads((recording / 'acquisition.json').read_text())['complete']:
        proc = Path(f'/proc/{args.acquisition_pid}/cmdline')
        if not proc.exists() or str(recording).encode() not in proc.read_bytes():
            raise RuntimeError('Acquisition stopped without a completion manifest')
        time.sleep(5)
    status['input_validation'] = validate(
        recording,
        root / 'inputs/schedule.json',
        root / 'inputs/script.json',
        args.laps,
    )
    native_recording, shared_indices, shared_manifest = prepare_shared_recording(
        recording, root / 'inputs/script.json', root / 'shared-input',
        args.hov_skip_frames)
    status['shared_observation_input'] = {
        'path': str(native_recording), 'frames': len(shared_indices),
        'manifest': shared_manifest,
        'used_by': ['clio', 'hovsg'],
    }
    ground_truth = (args.ground_truth.resolve() if args.ground_truth
                    else recording / 'static-gt.json')
    if not ground_truth.is_file():
        raise ValueError('Full static GT is required for end-to-end evaluation: ' +
                         str(ground_truth))
    gt = json.loads(ground_truth.read_text())
    required_gt = ('ground_truth_floors_m', 'ground_truth_regions',
                   'ground_truth_objects', 'rooms', 'categories')
    if any(key not in gt or not gt[key] for key in required_gt):
        raise ValueError('Ground-truth manifest lacks a required nonempty GRAPH-API table input')
    status['ground_truth'] = {'path': str(ground_truth),
        'sha256': hashlib.sha256(ground_truth.read_bytes()).hexdigest()}
    before = native_state(args.baseline_roots)
    if any(row['tracked_changes'] for row in before.values()):
        raise RuntimeError('Native baseline has tracked changes; inspect before running')
    status['native_sources_before'] = before
    results = root / 'results'
    results.mkdir()
    status['phase'] = 'native_runs'
    write(status_path, status)
    state_lock = threading.Lock()
    def launch(name, attempt=1):
        tag = name if attempt == 1 else f'{name}-retry-{attempt}'
        output = results / tag
        container = f'{args.container_prefix}-{tag}'
        env = dict(os.environ, BASELINE_GPU=str(gpu_by_name[name]),
            BASELINE_CONTAINER_NAME=container,
            BASELINE_INTEGRATION_ROOT=str(root / 'native-integration'),
            BASELINE_REPOS_ROOT=str(args.baseline_roots))
        if name == 'hovsg':
            options = ['--skip-frames', '1',
                       '--record-native-observations']
        else:
            options = ['--rate', str(args.clio_rate)]
            if args.clio_segmentation_confidence is not None:
                options += ['--segmentation-confidence',
                            str(args.clio_segmentation_confidence)]
            if args.clio_mode == 'semantic-mapping':
                options += ['--semantic-mapping-only']
            else:
                options += ['--task-conditioned', '--tasks', *args.clio_tasks]
        command = ['bash', str(root / 'native-integration/tools/baselines/gin.sh'), name,
            str(native_recording), str(output)] + options
        with (results / (tag + '.log')).open('w') as stream:
            process = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT, env=env)
        entry = {'tag': tag, 'pid': process.pid, 'output': str(output),
                 'gpu': gpu_by_name[name],
                 'container': container, 'started_at': time.time(), 'command': command}
        with state_lock:
            status['jobs'][name]['attempts'].append(entry)
            write(status_path, status)
        return process, output, container, entry

    def run_native(name, samples, sample_lock):
        verified = False
        final_output = None
        if not args.parallel_distinct_gpus:
            with state_lock:
                status['phase'] = f'native_{name}'
                write(status_path, status)
        for attempt in (1, 2):
            proc, output, container, entry = launch(name, attempt)
            while proc.poll() is None:
                free = shutil.disk_usage('/dev/shm').free
                gpu = subprocess.check_output([
                    'nvidia-smi', '--query-gpu=index,memory.used,utilization.gpu',
                    '--format=csv,noheader,nounits'], text=True).strip()
                with sample_lock:
                    samples.write(json.dumps({'time': time.time(), 'baseline': name,
                        'attempt': attempt, 'scratch_free_bytes': free, 'gpus': gpu}) + '\n')
                    samples.flush()
                if free < 2 * 1024**3:
                    subprocess.run(['docker', 'stop', '-t', '20', container], check=False)
                    raise RuntimeError('Stopped own job before shared RAM scratch exhaustion')
                time.sleep(5)
            code = proc.wait()
            marker = output / 'baseline_result.json'
            result = json.loads(marker.read_text()) if marker.is_file() else {}
            readiness = result.get('evaluation_readiness', {})
            ready = (readiness.get('ready_for_end_to_end_evaluation') if name == 'hovsg'
                     else readiness.get('ready_for_temporal_object_action_evaluation'))
            verified = bool(code == 0 and result.get('complete') and ready)
            with state_lock:
                entry.update(returncode=code, finished_at=time.time(),
                             verified_complete=verified)
                write(status_path, status)
            if verified:
                final_output = output
                names = ('input.bag',) if name == 'clio' else ('full_feats.pt',)
                pruned = prune_regenerable(output, names)
                with state_lock:
                    entry['pruned_regenerable_artifacts'] = pruned
                    write(status_path, status)
                break
            log = (results / (entry['tag'] + '.log')).read_text()
            roslog = output / 'roslaunch.log'
            if roslog.exists():
                log += roslog.read_text()
            explicit_oom = ('CUDA out of memory' in log or
                            'torch.cuda.OutOfMemoryError' in log)
            if attempt == 1 and explicit_oom:
                with state_lock:
                    entry['retry_reason'] = 'Explicit CUDA OOM; one retry on its assigned GPU'
                    write(status_path, status)
                continue
            break
        with state_lock:
            status['jobs'][name]['verified_complete'] = verified
            status['jobs'][name]['final_output'] = (
                str(final_output) if final_output is not None else None)
            write(status_path, status)
        return verified

    sample_lock = threading.Lock()
    with (root / 'pair-resources.jsonl').open('w') as samples:
        for name in ('clio', 'hovsg'):
            status['jobs'][name] = {'baseline': name, 'gpu': gpu_by_name[name], 'attempts': []}
        status['phase'] = ('native_parallel' if args.parallel_distinct_gpus else 'native_runs')
        write(status_path, status)
        if args.parallel_distinct_gpus:
            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
                futures = {name: executor.submit(run_native, name, samples, sample_lock)
                           for name in ('clio', 'hovsg')}
                verified_by_name = {name: future.result() for name, future in futures.items()}
        else:
            verified_by_name = {}
            for name in ('clio', 'hovsg'):
                verified_by_name[name] = run_native(name, samples, sample_lock)
                if not verified_by_name[name]:
                    break
    after = native_state(args.baseline_roots)
    status['native_sources_after'] = after
    if after != before:
        raise RuntimeError('Native tracked source state changed during execution')
    native_failed = [name for name, verified in verified_by_name.items() if not verified]
    if native_failed:
        status.update(phase='native_failure', complete=False,
                      error=(','.join(native_failed) +
                             ' did not produce evaluation-ready native output'),
                      finished_at=time.time())
        write(status_path, status)
        return status
    native_complete = all(status['jobs'][name]['verified_complete']
                          for name in ('clio', 'hovsg'))
    if native_complete:
        status['phase'] = 'end_to_end_evaluation'
        write(status_path, status)
        evaluation = root / 'evaluation'
        evaluation.mkdir()
        for baseline in ('clio', 'hovsg'):
            result = Path(status['jobs'][baseline]['final_output'])
            report = evaluation / f'{baseline}.report.json'
            command = [sys.executable, '-m', 'tools.baselines.graphapi_eval',
                '--baseline', baseline, '--recording', str(recording),
                '--result', str(result), '--ground-truth', str(ground_truth),
                '--output', str(report)]
            completed = subprocess.run(command, check=False)
            status['jobs'][baseline]['evaluation'] = {
                'command': command, 'returncode': completed.returncode,
                'output': str(report),
                'complete': completed.returncode == 0 and report.is_file()}
            write(status_path, status)
            if not status['jobs'][baseline]['evaluation']['complete']:
                raise RuntimeError(f'End-to-end evaluation failed for {baseline}')
    status['complete'] = native_complete
    status['phase'] = ('evaluation_complete' if native_complete else 'native_failure')
    status['finished_at'] = time.time()
    write(status_path, status)
    return status


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-root', type=Path, required=True)
    parser.add_argument('--acquisition-pid', type=int, required=True)
    parser.add_argument('--baseline-roots', type=Path, default=Path('/home/phd_student/Musumeci'))
    parser.add_argument('--gpu', type=int, default=1)
    parser.add_argument('--clio-gpu', type=int)
    parser.add_argument('--hov-gpu', type=int)
    parser.add_argument('--parallel-distinct-gpus', action='store_true')
    parser.add_argument('--laps', type=int, default=2)
    parser.add_argument('--hov-skip-frames', type=int, default=10)
    parser.add_argument('--clio-rate', type=float, default=1.0)
    parser.add_argument('--clio-segmentation-confidence', type=float, default=0.25)
    parser.add_argument('--clio-mode', choices=['semantic-mapping', 'task-conditioned'],
                        default='semantic-mapping')
    parser.add_argument('--clio-tasks', nargs='+', default=[])
    parser.add_argument('--container-prefix', default='graphapi-baselines-824-2lap')
    parser.add_argument('--ground-truth', type=Path,
                        help='Full manifest; defaults to RECORDING/static-gt.json')
    args = parser.parse_args()
    if args.laps < 1:
        parser.error('--laps must be positive')
    if (args.clio_segmentation_confidence is not None and
            not 0 < args.clio_segmentation_confidence < 1):
        parser.error('--clio-segmentation-confidence must be between zero and one')
    if len(set(args.clio_tasks)) != len(args.clio_tasks):
        parser.error('--clio-tasks must be distinct')
    if args.clio_mode == 'semantic-mapping' and args.clio_tasks:
        parser.error('--clio-tasks cannot be used in semantic-mapping mode')
    if args.clio_mode == 'task-conditioned' and not args.clio_tasks:
        parser.error('task-conditioned Clio requires --clio-tasks')
    try:
        status = run(args)
    except (RuntimeError, ValueError, subprocess.CalledProcessError) as error:
        path = args.run_root / 'pair-status.json'
        if path.exists():
            status = json.loads(path.read_text())
            status.update(phase='guard_failure', complete=False, error=str(error))
            write(path, status)
        raise
    print(json.dumps(status, indent=2))
    return 0 if status['complete'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
