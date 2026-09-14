"""Bounded Gin operations for the external baseline overnight controller.

Requests arrive as JSON on stdin. Source baselines are never writable mounts.
Only verified, completed recording/results directories in the named lane roots
can be reclaimed from RAM. Lossless archived evidence stays on the controller.
"""
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

from .run_pair import write

ROOT = Path('/dev/shm/graphapi-baselines-overnight-20260913')
CURRENT = Path('/dev/shm/graphapi-baselines-824-two-lap-20260913')
PYTHON = '/home/phd_student/miniconda3/envs/habitat_env/bin/python'
GIB = 1024**3


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(8*1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def checked_root(value):
    path = Path(value).resolve()
    if path != CURRENT and (path.parent != ROOT / 'runs' or not path.name.startswith('00')):
        raise ValueError('Run outside the authorized baseline scratch roots')
    return path


def inventory(root):
    files = {}
    for directory in ('recording', 'results'):
        for path in sorted((root / directory).rglob('*')):
            if path.is_symlink():
                continue  # links are retained by rsync; only regular bytes are hashed
            if path.is_file() and path.name != 'input.bag':
                files[str(path.relative_to(root))] = {'bytes': path.stat().st_size, 'sha256': digest(path)}
    if not files:
        raise ValueError('No completed evidence to inventory')
    return files


def frozen():
    expected = json.loads((ROOT / 'frozen-files.json').read_text())
    for key, sha in expected.items():
        if digest(ROOT / key) != sha:
            raise ValueError('Frozen input/source changed: ' + key)
    return {'checked_files': len(expected)}


def launch(root, kind, command, env):
    marker = root / (kind + '-process.json')
    if marker.exists():
        raise FileExistsError('Process already submitted: ' + str(marker))
    with (root / (kind + '.log')).open('w') as stream:
        process = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT,
            env=env, start_new_session=True, stdin=subprocess.DEVNULL)
    value = {'pid': process.pid, 'command': command, 'started_at': time.time()}
    write(marker, value)
    return value


def status(root):
    value = {'root': str(root), 'scratch_free_bytes': shutil.disk_usage('/dev/shm').free}
    for key, path in [('acquisition', root/'recording/acquisition.json'), ('pair', root/'pair-status.json')]:
        if path.exists():
            value[key] = json.loads(path.read_text())
    for kind in ('acquisition', 'pair'):
        path = root / (kind + '-process.json')
        if path.exists():
            meta = json.loads(path.read_text())
            proc = Path('/proc') / str(meta['pid']) / 'cmdline'
            value[kind + '_process'] = dict(meta, alive=proc.exists() and str(root).encode() in proc.read_bytes())
    return value


def handle(request):
    action = request['action']
    if action == 'preflight':
        report = json.loads((ROOT/'preflight-results.json').read_text())
        depth = json.loads((ROOT/'ros-depth-preflight/ros-depth-preflight.json').read_text())
        relay = json.loads((ROOT/'rgb-relay-preflight/rgb-relay-preflight.json').read_text())
        if not report['passed'] or not depth['passed'] or not relay['passed'] or len(report['scenes']) != 5:
            raise ValueError('Night input preflight is incomplete or failed')
        return dict(frozen(), input_preflight=report, depth_preflight=depth, relay_preflight=relay,
            scratch_free_bytes=shutil.disk_usage('/dev/shm').free)
    root = checked_root(request['root'])
    if action == 'status':
        return status(root)
    if action == 'acquire':
        frozen()
        if shutil.disk_usage('/dev/shm').free < 15*GIB:
            raise RuntimeError('Insufficient RAM scratch for another full recording')
        manifest = json.loads((ROOT/'prepared-scenes.json').read_text())
        entry = next(row for row in manifest['scenes'] if row['scene'] == root.name)
        if root.exists():
            raise FileExistsError('Run already exists: ' + str(root))
        root.mkdir(parents=True)
        # Real input copies keep each downloaded bundle self-contained.
        shutil.copytree(ROOT/'inputs'/root.name, root/'inputs')
        (root/'source').symlink_to(ROOT/'source', target_is_directory=True)
        (root/'native-integration').symlink_to(ROOT/'integration', target_is_directory=True)
        mesh = ROOT/'assets'/root.name/(root.name.split('-',1)[1]+'.basis.glb')
        command = [PYTHON, '-m', 'tools.baselines.runtime', '--graph-api-root', str(root/'source'),
            'acquire', '--scene', str(mesh), '--dataset', str(root/'inputs/dataset.scene_dataset_config.json'),
            '--objects', '/home/phd_student/Musumeci/baseline-integration/objects/configs',
            '--schedule', str(root/'inputs/schedule.json'), '--script', str(root/'inputs/script.json'),
            '--output', str(root/'recording'), '--config', str(root/'inputs/config.yaml'),
            '--floor', str(entry['floor']), '--laps', '2', '--fps', '3', '--compress-depth',
            '--shuffle-depth', '--record-gt', '--dynamic-semantic-id-offset', '1000000']
        env = dict(os.environ, PYTHONPATH=str(ROOT/'integration'), CUDA_VISIBLE_DEVICES='1',
            PYTHONUNBUFFERED='1', PYTHONDONTWRITEBYTECODE='1')
        write(root/'acquisition-command.json',command)
        return launch(root, 'acquisition', [sys.executable,'-m','tools.baselines.acquisition_watch',str(root)], env)
    if action == 'pair':
        frozen()
        if shutil.disk_usage('/dev/shm').free < 18*GIB:
            raise RuntimeError('Insufficient RAM scratch for native bag and outputs')
        pid = json.loads((root/'acquisition-process.json').read_text())['pid']
        command = [sys.executable, '-m', 'tools.baselines.run_pair', '--run-root', str(root),
            '--acquisition-pid', str(pid), '--gpu', '1', '--container-prefix',
            'graphapi-baselines-night-' + root.name.split('-',1)[0]]
        env = dict(os.environ, PYTHONPATH=str(ROOT/'integration'), PYTHONUNBUFFERED='1',
            PYTHONDONTWRITEBYTECODE='1')
        return launch(root, 'pair', command, env)
    if action in ('inventory', 'reclaim'):
        pair = json.loads((root/'pair-status.json').read_text())
        if pair['phase'] not in ('native_complete', 'evaluation_complete') or not pair['complete']:
            raise ValueError('Only completed native results may be reclaimed')
        for kind in ('acquisition', 'pair'):
            proc = status(root).get(kind+'_process')
            if proc and proc['alive']:
                raise ValueError('Run process is still alive')
        files = inventory(root)
        if action == 'inventory':
            return {'files': files, 'bytes': sum(row['bytes'] for row in files.values())}
        if files != request['verified_files']:
            raise ValueError('Completed evidence differs from the verified local archive')
        receipt = {'files': files, 'local_archive': request['archive'],
            'verified_at': time.time(), 'excluded_regenerable_files': ['input.bag'],
            'scope': 'Exact regular bytes verified on both hosts before removing this completed RAM copy'}
        write(root/'archive-receipt.json', receipt)
        # Outputs are container-owned. This isolated CPU container can write only
        # this completed run; names are literal and no baseline source is mounted.
        subprocess.run(['docker','run','--rm','--network','none','-v',str(root)+':/archive',
            '--entrypoint','python3','clio-baseline:noetic','-c',
            "import shutil; shutil.rmtree('/archive/recording'); shutil.rmtree('/archive/results')"], check=True)
        return {'reclaimed': True, 'files': len(files), 'bytes': receipt['files'] and sum(x['bytes'] for x in files.values())}
    raise ValueError('Unknown action: ' + action)


if __name__ == '__main__':
    print(json.dumps(handle(json.load(sys.stdin))))
