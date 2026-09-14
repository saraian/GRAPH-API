"""Twenty-minute operational monitor for the Gin Pack A baseline run."""
import argparse
import hashlib
import json
import os
import re
import signal
import shutil
import subprocess
import time
from pathlib import Path

from .run_pair import write


CLIO_IMAGE = 'sha256:93e83b9864c15f6f358c9b83d6c7633797189b94c17d7ec6ab3b27c8fe5c11d2'
CLIO_DATA = '/home/phd_student/Musumeci/Clio/clio_dataset/office'
PAIR_PYTHON = '/home/phd_student/miniconda3/envs/habitat-3dmem/bin/python3.9'


def command(*values, check=True):
    return subprocess.run(values, text=True, capture_output=True, check=check)


def repair_clio_shell():
    inspected = command('docker', 'inspect', 'clio-gpu', check=False)
    if inspected.returncode:
        command('docker', 'create', '--gpus', 'all', '-it', '--name', 'clio-gpu',
            '--network', 'host', '-e', 'DISPLAY=:0', '-e', 'QT_X11_NO_MITSHM=1',
            '-v', '/tmp/.X11-unix:/tmp/.X11-unix:rw',
            '-v', CLIO_DATA + ':/root/clio_datasets', CLIO_IMAGE, 'bash')
        command('docker', 'start', 'clio-gpu')
        return 'recreated_and_started'
    value = json.loads(inspected.stdout)[0]
    if value['Image'] != CLIO_IMAGE:
        raise RuntimeError('Existing clio-gpu uses an unexpected image; refusing replacement')
    mounts = {row['Destination']: row['Source'] for row in value['Mounts']}
    if mounts.get('/root/clio_datasets') != CLIO_DATA:
        raise RuntimeError('Existing clio-gpu has an unexpected dataset mount')
    if not value['State']['Running']:
        command('docker', 'start', 'clio-gpu')
        return 'started'
    return 'already_running'


def service_state(unit):
    result = command('systemctl', '--user', 'is-active', unit, check=False)
    return result.stdout.strip() or 'unknown'


def start_driver(unit):
    command('systemctl', '--user', 'reset-failed', unit, check=False)
    command('systemctl', '--user', 'start', unit)


def direct_driver_active(root, status):
    pid = status.get('driver_pid')
    if not isinstance(pid, int) or pid < 2:
        return False
    cmdline = Path('/proc') / str(pid) / 'cmdline'
    if not cmdline.is_file():
        return False
    value = cmdline.read_bytes()
    return (b'tools.baselines.pack_a_driver' in value and
            str(Path(root).resolve()).encode() in value)


def direct_child_active(root, status):
    pid = status.get('child_pid')
    if not isinstance(pid, int) or pid < 2:
        return False
    cmdline = Path('/proc') / str(pid) / 'cmdline'
    if not cmdline.is_file():
        return False
    value = cmdline.read_bytes()
    allowed = (b'tools.baselines.runtime', b'tools.baselines.run_pair',
               b'tools.baselines.audit_pair')
    return (any(module in value for module in allowed) and
            str(Path(root).resolve()).encode() in value)


def direct_driver_command(root, args):
    return [PAIR_PYTHON, '-m', 'tools.baselines.pack_a_driver',
        '--pack-root', str(Path(root).resolve()), '--gpu', str(args.clio_gpu),
        '--clio-gpu', str(args.clio_gpu), '--hov-gpu', str(args.hov_gpu),
        '--hov-skip-frames', str(args.hov_skip_frames),
        '--maximum-attempts', str(args.maximum_attempts),
        '--minimum-free-gib', str(args.minimum_free_gib)]


def start_direct_driver(root, args):
    log = (root / 'driver-supervisor.log').open('a')
    env = dict(os.environ, PYTHONPATH=str(root / 'integration'),
               PYTHONDONTWRITEBYTECODE='1', PYTHONUNBUFFERED='1')
    try:
        process = subprocess.Popen(direct_driver_command(root, args), stdin=subprocess.DEVNULL,
            stdout=log, stderr=subprocess.STDOUT, env=env, start_new_session=True)
    finally:
        log.close()
    return process.pid


def terminate_groups(pids, grace_seconds=180):
    """Terminate exact, separately sessioned process-group leaders together."""
    for pid in pids:
        try:
            if os.getpgid(pid) != pid:
                raise RuntimeError(f'Refusing to stop non-leader process {pid}')
        except ProcessLookupError:
            continue
        try:
            os.killpg(pid, signal.SIGTERM)
        except ProcessLookupError:
            continue
    deadline = time.monotonic() + grace_seconds
    while time.monotonic() < deadline:
        if all(not (Path('/proc') / str(pid)).exists() for pid in pids):
            break
        time.sleep(.25)
    for pid in pids:
        if not (Path('/proc') / str(pid)).exists():
            continue
        try:
            os.killpg(pid, signal.SIGKILL)
        except ProcessLookupError:
            continue


def stop_direct_child(root, status, grace_seconds=180):
    if not direct_child_active(root, status):
        return False
    terminate_groups([status['child_pid']], grace_seconds)
    return True


def stop_direct_driver(root, status, grace_seconds=180):
    if not direct_driver_active(root, status):
        return False
    pids = []
    if direct_child_active(root, status):
        pids.append(status['child_pid'])
    pids.append(status['driver_pid'])
    terminate_groups(pids, grace_seconds)
    return True


def owned_containers(root, status):
    """Return only exact Pack A containers named by this run's pair receipt."""
    pair_value = status.get('pair_root')
    if not pair_value:
        return []
    pair = Path(pair_value).resolve()
    try:
        pair.relative_to((root / 'runs').resolve())
    except ValueError:
        return []
    marker = pair / 'pair-status.json'
    if not marker.is_file():
        return []
    value = json.loads(marker.read_text())
    names = []
    for job in value.get('jobs', {}).values():
        for attempt in job.get('attempts', []):
            name = str(attempt.get('container', ''))
            if re.fullmatch(r'graphapi-pack-a-[A-Za-z0-9_.-]+', name):
                names.append(name)
    return sorted(set(names))


def cleanup_owned_containers(root, status):
    cleaned = []
    for name in owned_containers(root, status):
        if command('docker', 'inspect', name, check=False).returncode:
            continue
        command('docker', 'stop', '-t', '30', name, check=False)
        command('docker', 'rm', '-f', name, check=False)
        cleaned.append(name)
    return cleaned


def progress_age(root, status):
    paths = []
    if status.get('child_log'):
        paths.append(Path(status['child_log']))
    if status.get('pair_root'):
        pair = Path(status['pair_root']).resolve()
        try:
            pair.relative_to((root / 'runs').resolve())
        except ValueError:
            pair = None
        if pair is not None:
            # pair-resources is a timer heartbeat. It advances even when a native
            # pipeline is stuck, so it is not evidence of forward progress.
            paths.append(pair / 'pair-status.json')
            paths.extend((pair / 'results').glob('*.log'))
    stamps = [path.stat().st_mtime for path in paths if path.is_file()]
    return time.time() - max(stamps) if stamps else None


def frozen_manifest_digest(root):
    path = Path(root).resolve() / 'frozen-files.json'
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None


def blocked_on_current_snapshot(root, status):
    return (status.get('phase') == 'blocked' and
            status.get('frozen_manifest_sha256') is not None and
            status['frozen_manifest_sha256'] == frozen_manifest_digest(root))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pack-root', type=Path, required=True)
    parser.add_argument('--driver-unit', default='graphapi-pack-a-driver.service')
    parser.add_argument('--direct', action='store_true',
                        help='Supervise a detached driver directly (safe for cron without user linger)')
    parser.add_argument('--clio-gpu', type=int, default=1)
    parser.add_argument('--hov-gpu', type=int, default=0)
    parser.add_argument('--hov-skip-frames', type=int, default=50)
    parser.add_argument('--maximum-attempts', type=int, default=10)
    parser.add_argument('--minimum-free-gib', type=int, default=15)
    args = parser.parse_args(argv)
    root = args.pack_root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    status_path = root / 'pack-status.json'
    status = json.loads(status_path.read_text()) if status_path.is_file() else {}
    row = {'time': time.time(), 'pack_phase': status.get('phase'),
           'pack_complete': bool(status.get('complete')),
           'completed_scenes': status.get('completed', []),
           'disk_free_bytes': shutil.disk_usage(root).free,
           'actions': [], 'errors': []}
    terminal_block = blocked_on_current_snapshot(root, status)
    if terminal_block:
        row['actions'].append('held_terminal_blocked_snapshot')
    try:
        row['clio_gpu'] = repair_clio_shell()
        if row['clio_gpu'] != 'already_running':
            row['actions'].append('repair_' + row['clio_gpu'])
    except (RuntimeError, ValueError, OSError, subprocess.SubprocessError,
            json.JSONDecodeError) as error:
        row['errors'].append('clio-gpu: ' + str(error))
    state = (('active' if direct_driver_active(root, status) else 'inactive')
             if args.direct else service_state(args.driver_unit))
    row['driver_service'] = state
    if (not status.get('complete') and not terminal_block and
            state not in ('active', 'activating')):
        try:
            if args.direct and stop_direct_child(root, status):
                row['actions'].append('stopped_orphaned_child')
            cleaned = cleanup_owned_containers(root, status)
            if cleaned:
                row['actions'].append('removed_stale_containers:' + ','.join(cleaned))
            if args.direct:
                row['driver_pid_after'] = start_direct_driver(root, args)
                row['actions'].append('started_direct_driver')
                row['driver_service_after'] = 'active'
            else:
                start_driver(args.driver_unit)
                row['actions'].append('started_driver_service')
                row['driver_service_after'] = service_state(args.driver_unit)
        except (RuntimeError, ValueError, OSError, subprocess.SubprocessError,
                json.JSONDecodeError) as error:
            row['errors'].append('driver restart: ' + str(error))
    age = progress_age(root, status)
    if age is not None:
        row['progress_age_s'] = age
        if age > 4 * 3600 and state in ('active', 'activating'):
            try:
                if args.direct:
                    stop_direct_driver(root, status)
                else:
                    command('systemctl', '--user', 'stop', args.driver_unit)
                cleaned = cleanup_owned_containers(root, status)
                if args.direct:
                    row['driver_pid_after'] = start_direct_driver(root, args)
                else:
                    start_driver(args.driver_unit)
                row['actions'].append('restarted_stale_driver')
                if cleaned:
                    row['actions'].append('removed_stale_containers:' + ','.join(cleaned))
                row['driver_service_after'] = ('active' if args.direct else
                                               service_state(args.driver_unit))
            except (RuntimeError, ValueError, OSError, subprocess.SubprocessError,
                    json.JSONDecodeError) as error:
                row['errors'].append('stale driver restart: ' + str(error))
    gpu = command('nvidia-smi', '--query-gpu=index,memory.used,utilization.gpu',
                  '--format=csv,noheader,nounits', check=False)
    row['gpus'] = gpu.stdout.strip()
    with (root / 'monitor.jsonl').open('a') as stream:
        stream.write(json.dumps(row) + '\n')
    write(root / 'monitor-latest.json', row)
    print(json.dumps(row, indent=2))
    return 1 if row['errors'] else 0


if __name__ == '__main__':
    raise SystemExit(main())
