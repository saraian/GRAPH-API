"""Resumable Pack A runner for unchanged Clio and HOV-SG on Gin.

One measurement recording is acquired per scene, then the two native baselines
run sequentially on the same recording.  Every retry gets a new pair directory;
failed evidence is retained and successful scenes are never rerun.
"""
import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

from .run_pair import native_state, write


GIB = 1024 ** 3
HABITAT_PYTHON = Path('/home/phd_student/miniconda3/envs/habitat_env/bin/python3.9')
PAIR_PYTHON = Path('/home/phd_student/miniconda3/envs/habitat-3dmem/bin/python3.9')


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def tree_stamp(path):
    """Hash a generated file tree before reclaiming it from the measurement host."""
    path = Path(path)
    files = [path] if path.is_file() else sorted(x for x in path.rglob('*') if x.is_file())
    value = hashlib.sha256()
    total = 0
    for item in files:
        relative = item.name if path.is_file() else item.relative_to(path).as_posix()
        size = item.stat().st_size
        value.update(relative.encode('utf-8') + b'\0' + str(size).encode('ascii') + b'\0')
        with item.open('rb') as stream:
            for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
                value.update(block)
        total += size
    return {'path': str(path), 'files': len(files), 'bytes': total,
            'tree_sha256': value.hexdigest()}


def verify_frozen(root):
    expected = json.loads((root / 'frozen-files.json').read_text())
    for relative, sha256 in expected.items():
        path = root / relative
        if not path.is_file() or digest(path) != sha256:
            raise RuntimeError('Frozen Pack A input/integration changed: ' + relative)
    return len(expected)


def verify_native_revisions(root, baseline_roots=Path('/home/phd_student/Musumeci')):
    expected = json.loads((Path(root) / 'native-revisions.json').read_text())
    if expected.get('schema') != 'graphapi.native_baseline_revisions.v1':
        raise RuntimeError('Native baseline revision receipt has an unknown schema')
    baseline_roots = Path(baseline_roots).resolve()
    if expected.get('baseline_roots') not in (None, str(baseline_roots)):
        raise RuntimeError('Native baseline revision receipt names another checkout root')
    observed = native_state(baseline_roots)
    if observed != expected.get('repositories'):
        raise RuntimeError('Native baseline revision or tracked source state changed')
    return observed


def completed_recording(path):
    marker = path / 'acquisition.json'
    if not marker.is_file():
        return False
    value = json.loads(marker.read_text())
    plan = value.get('action_observation_plan', {})
    return bool(value.get('complete') and value.get('ground_truth_recorded') and
                plan.get('complete') and (path / 'static-gt.json').is_file())


def run_logged(command, log, env, status, status_path):
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open('w') as stream:
        process = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT,
                                   env=env, start_new_session=True,
                                   stdin=subprocess.DEVNULL)
        status.update(child_pid=process.pid, child_command=command,
                      child_log=str(log), progress_at=time.time())
        write(status_path, status)
        code = process.wait()
    status.update(child_pid=None, child_returncode=code, progress_at=time.time())
    write(status_path, status)
    return code


def preserve_failed_intermediates(pair):
    """Bound failed-attempt size while retaining provenance and diagnostic logs."""
    rows = []
    for pattern in ('results/*/input.bag', 'results/*/full_feats.pt'):
        for path in pair.glob(pattern):
            if not path.is_file():
                continue
            rows.append({'path': str(path.relative_to(pair)), 'bytes': path.stat().st_size,
                         'sha256': digest(path), 'reason': 'regenerable failed-run intermediate'})
            path.unlink()
    if rows:
        write(pair / 'failed-attempt-pruned.json', {'files': rows,
            'recording_preserved': True, 'pruned_at': time.time()})


def stage_pair_inputs(root, entry, pair):
    """Expose the declared scene files through run_pair's canonical input names."""
    root, pair = Path(root).resolve(), Path(pair)
    declared_root = (root / entry['input_root']).resolve()
    inputs = pair / 'inputs'
    inputs.mkdir()
    staged = {}
    for name, key in (('schedule.json', 'schedule'), ('script.json', 'script')):
        source = (root / entry[key]).resolve(strict=True)
        try:
            source.relative_to(declared_root)
        except ValueError as error:
            raise RuntimeError(f'Declared {key} escapes its scene input root') from error
        destination = inputs / name
        destination.symlink_to(source)
        staged[name] = str(source)
    return staged


def compact_completed_scene(scene_root, pair):
    """Keep replay/evaluation evidence while bounding five-scene disk use.

    Native execution and both evaluation gates have already consumed lossless depth and
    simulator semantic buffers. The dashboard replay uses RGB, poses, action/frame GT and
    native graph histories, all of which remain. A content hash/count receipt is written
    before generated bulk data is reclaimed, making interrupted cleanup resumable.
    """
    scene_root, pair = Path(scene_root), Path(pair)
    marker = scene_root / 'compaction.json'
    pair_state = json.loads((pair / 'pair-status.json').read_text())
    if not (pair_state.get('complete') and pair_state.get('phase') == 'evaluation_complete'
            and (scene_root / 'metrics/report.json').is_file()):
        raise RuntimeError('Refusing to compact before native, evaluation, and audit gates pass')
    recording = scene_root / 'recording'
    final_outputs = {}
    for baseline in ('clio', 'hovsg'):
        final = Path(pair_state['jobs'][baseline]['final_output']).resolve()
        try:
            final.relative_to((pair / 'results').resolve())
        except ValueError as error:
            raise RuntimeError('Final native output escapes its pair result tree') from error
        final_outputs[baseline] = final
    acquisition = json.loads((recording / 'acquisition.json').read_text())
    frames = [json.loads(row) for row in (recording / 'frames.jsonl').read_text().splitlines()
              if row.strip()]
    expected = int(acquisition['frames'])
    if len(frames) != expected:
        raise RuntimeError('Refusing to compact a recording with inconsistent frame metadata')
    for folder, suffix in (('rgb', '.png'), ('pose', '.txt')):
        if sum(1 for _ in (recording / folder).glob('*' + suffix)) != expected:
            raise RuntimeError('Replay evidence is incomplete before compaction: ' + folder)
    required = [recording / name for name in ('static-gt.json', 'frames.jsonl',
        'object_actions.jsonl', 'scan_events.jsonl')]
    required += [final_outputs['clio'] / 'baseline_result.json',
        final_outputs['clio'] / 'graph/backend/dsg.json',
        final_outputs['clio'] / 'native_graph_history.jsonl.gz',
        final_outputs['hovsg'] / 'baseline_result.json', final_outputs['hovsg'] / 'graph']
    if any(not path.exists() for path in required):
        raise RuntimeError('Replay/evaluation evidence is incomplete before compaction')
    targets = [recording / name for name in ('depth', 'depth_m', 'semantic')]
    targets.append(final_outputs['clio'] / 'native_outputs.bag')
    if marker.is_file():
        receipt = json.loads(marker.read_text())
    else:
        receipt = {'schema': 'graphapi.pack_a_compaction.v1', 'complete': False,
            'created_at': time.time(),
            'reason': ('Lossless native inputs have been consumed by two completed baselines; '
                       'RGB, poses, dynamic GT, graphs, observations, reports and logs remain.'),
            'reclaimed': [tree_stamp(path) for path in targets if path.exists()],
            'retained_for_replay': ['recording/rgb', 'recording/pose',
                'recording/frames.jsonl', 'recording/object_actions.jsonl',
                'recording/static-gt.json', 'pair results and graph histories']}
        write(marker, receipt)
    for path in targets:
        if path.is_dir():
            shutil.rmtree(path)
        elif path.exists():
            path.unlink()
    if any(path.exists() for path in targets):
        raise RuntimeError('Pack A compaction did not reclaim every recorded target')
    receipt.update(complete=True, completed_at=time.time(),
                   retained_rgb_frames=expected, retained_pose_frames=expected,
                   reclaimed_bytes=sum(row['bytes'] for row in receipt['reclaimed']))
    write(marker, receipt)
    return marker


def acquire(root, entry, scene_root, status, status_path, maximum_attempts):
    recording = scene_root / 'recording'
    if completed_recording(recording):
        return recording
    failures = scene_root / 'failed-acquisitions'
    failures.mkdir(parents=True, exist_ok=True)
    if recording.exists():
        recording.rename(failures / f'incomplete-{int(time.time())}')
    for attempt in range(1, maximum_attempts + 1):
        status.update(phase='acquisition', scene=entry['scene'],
                      acquisition_attempt=attempt, progress_at=time.time())
        write(status_path, status)
        command = [str(HABITAT_PYTHON), '-m', 'tools.baselines.runtime',
            '--graph-api-root', entry['graph_api_root'], 'acquire',
            '--scene', entry['mesh'], '--dataset', str(root / entry['dataset']),
            '--objects', entry['objects'], '--schedule', str(root / entry['schedule']),
            '--script', str(root / entry['script']), '--output', str(recording),
            '--config', str(root / entry['config']), '--floor', str(entry['floor']),
            '--laps', '2', '--fps', '3', '--compress-depth', '--shuffle-depth',
            '--record-gt', '--dynamic-semantic-id-offset', '1000000',
            '--purpose', 'measurement']
        env = dict(os.environ, PYTHONPATH=str(root / 'integration'),
                   CUDA_VISIBLE_DEVICES=str(status['gpu']), PYTHONUNBUFFERED='1',
                   PYTHONDONTWRITEBYTECODE='1')
        code = run_logged(command, scene_root / f'acquisition-attempt-{attempt}.log',
                          env, status, status_path)
        if code == 0 and completed_recording(recording):
            return recording
        if recording.exists():
            recording.rename(failures / f'attempt-{attempt}-{int(time.time())}')
    raise RuntimeError(f"Acquisition failed {maximum_attempts} times for {entry['scene']}")


def pair_run(root, entry, scene_root, recording, status, status_path, maximum_attempts):
    for existing in sorted(scene_root.glob('pair-attempt-*')):
        marker = existing / 'pair-status.json'
        if marker.is_file():
            value = json.loads(marker.read_text())
            if value.get('complete') and value.get('phase') == 'evaluation_complete':
                return existing
    attempts = len(list(scene_root.glob('pair-attempt-*')))
    while attempts < maximum_attempts:
        attempts += 1
        pair = scene_root / f'pair-attempt-{attempts}'
        pair.mkdir()
        (pair / 'recording').symlink_to(recording, target_is_directory=True)
        stage_pair_inputs(root, entry, pair)
        (pair / 'native-integration').symlink_to((root / 'integration').resolve(),
                                                target_is_directory=True)
        status.update(phase='native_pair', scene=entry['scene'], pair_attempt=attempts,
                      pair_root=str(pair), progress_at=time.time())
        write(status_path, status)
        command = [str(PAIR_PYTHON), '-m', 'tools.baselines.run_pair',
            '--run-root', str(pair), '--acquisition-pid', str(os.getpid()),
            '--gpu', str(status['gpu']), '--clio-gpu', str(status['clio_gpu']),
            '--hov-gpu', str(status['hov_gpu']), '--parallel-distinct-gpus',
            '--laps', '2', '--hov-skip-frames', str(status['hov_skip_frames']),
            '--clio-rate', '1', '--clio-segmentation-confidence', '.25',
            '--clio-mode', 'semantic-mapping', '--container-prefix',
            'graphapi-pack-a-' + entry['scene'].split('-', 1)[0],
            '--ground-truth', str(recording / 'static-gt.json')]
        env = dict(os.environ, PYTHONPATH=str(root / 'integration'),
                   PYTHONUNBUFFERED='1', PYTHONDONTWRITEBYTECODE='1')
        code = run_logged(command, pair / 'driver.log', env, status, status_path)
        marker = pair / 'pair-status.json'
        result = json.loads(marker.read_text()) if marker.is_file() else {}
        if code == 0 and result.get('complete') and result.get('phase') == 'evaluation_complete':
            return pair
        preserve_failed_intermediates(pair)
    raise RuntimeError(f"Native/evaluation pair failed {maximum_attempts} times for {entry['scene']}")


def audit_scene(root, entry, scene_root, pair, status, status_path):
    output = scene_root / 'metrics'
    if (output / 'report.json').is_file():
        return output
    if output.exists():
        output.rename(scene_root / f'failed-metrics-{int(time.time())}')
    status.update(phase='pair_audit', scene=entry['scene'], progress_at=time.time())
    write(status_path, status)
    command = [str(PAIR_PYTHON), '-m', 'tools.baselines.audit_pair',
               '--run-root', str(pair), '--output', str(output)]
    env = dict(os.environ, PYTHONPATH=str(root / 'integration'),
               PYTHONUNBUFFERED='1', PYTHONDONTWRITEBYTECODE='1')
    code = run_logged(command, scene_root / 'audit.log', env, status, status_path)
    if code or not (output / 'report.json').is_file():
        raise RuntimeError('Pack A audit failed for ' + entry['scene'])
    return output


def run(args):
    root = args.pack_root.resolve()
    status_path = root / 'pack-status.json'
    plan = json.loads((root / 'pack-manifest.json').read_text())
    previous = json.loads(status_path.read_text()) if status_path.is_file() else {}
    completed = previous.get('completed', [])
    status = {'schema': 'graphapi.pack_a_execution.v1', 'complete': False,
        'phase': 'preflight', 'gpu': args.gpu, 'scenes': [x['scene'] for x in plan['scenes']],
        'completed': completed, 'started_at': previous.get('started_at', time.time()),
        'driver_pid': os.getpid(), 'progress_at': time.time(),
        'clio_gpu': args.clio_gpu, 'hov_gpu': args.hov_gpu,
        'hov_skip_frames': args.hov_skip_frames,
        'native_algorithms_modified': False,
        'execution': 'parallel_clio_hovsg_on_distinct_gpus',
        'frozen_manifest_sha256': digest(root / 'frozen-files.json')}
    write(status_path, status)
    status['frozen_files_verified'] = verify_frozen(root)
    status['native_revisions_verified'] = verify_native_revisions(root)
    for entry in plan['scenes']:
        if entry['scene'] in status['completed']:
            continue
        if shutil.disk_usage(root).free < args.minimum_free_gib * GIB:
            raise RuntimeError('Pack A disk reserve reached before ' + entry['scene'])
        verify_frozen(root)
        verify_native_revisions(root)
        scene_root = root / 'runs' / entry['scene']
        scene_root.mkdir(parents=True, exist_ok=True)
        recording = acquire(root, entry, scene_root, status, status_path,
                            args.maximum_attempts)
        pair = pair_run(root, entry, scene_root, recording, status, status_path,
                        args.maximum_attempts)
        metrics = audit_scene(root, entry, scene_root, pair, status, status_path)
        status.update(phase='scene_compaction', scene=entry['scene'], progress_at=time.time())
        write(status_path, status)
        compaction = compact_completed_scene(scene_root, pair)
        status['completed'].append(entry['scene'])
        status.update(phase='scene_complete', scene=entry['scene'],
                      pair_root=str(pair), metrics=str(metrics), compaction=str(compaction),
                      progress_at=time.time())
        write(status_path, status)
    status.update(phase='pack_aggregate', progress_at=time.time())
    write(status_path, status)
    aggregate_output = root / 'pack-metrics'
    if not (aggregate_output / 'report.json').is_file():
        if aggregate_output.exists():
            aggregate_output.rename(root / f'failed-pack-metrics-{int(time.time())}')
        from .aggregate_pack import aggregate
        aggregate(root, aggregate_output)
    status.update(phase='complete', complete=True, child_pid=None,
                  aggregate_metrics=str(aggregate_output),
                  finished_at=time.time(), progress_at=time.time())
    write(status_path, status)
    return status


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pack-root', type=Path, required=True)
    parser.add_argument('--gpu', type=int, default=1)
    parser.add_argument('--clio-gpu', type=int, default=1)
    parser.add_argument('--hov-gpu', type=int, default=0)
    parser.add_argument('--hov-skip-frames', type=int, default=50)
    parser.add_argument('--maximum-attempts', type=int, default=3)
    parser.add_argument('--minimum-free-gib', type=int, default=15)
    args = parser.parse_args(argv)
    if args.clio_gpu == args.hov_gpu:
        parser.error('--clio-gpu and --hov-gpu must be distinct')
    if args.hov_skip_frames < 1:
        parser.error('--hov-skip-frames must be positive')
    try:
        result = run(args)
    except (RuntimeError, ValueError, FileNotFoundError, OSError,
            subprocess.SubprocessError, json.JSONDecodeError) as error:
        path = args.pack_root / 'pack-status.json'
        value = json.loads(path.read_text()) if path.is_file() else {}
        value.update(phase='blocked', complete=False, error=str(error),
                     child_pid=None, failed_at=time.time(), progress_at=time.time())
        write(path, value)
        raise
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
