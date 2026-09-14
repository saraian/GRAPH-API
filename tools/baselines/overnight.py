"""Persistent local controller for five two-lap baseline pairs on Gin.

Starts after a dated time AND the completed verification audit. One acquisition
may overlap one native pair. Each completed pair is mirrored, byte-verified,
audited, and archived before its RAM scratch is reclaimed. No native algorithm
is patched; an execution/data failure stops future launches and remains visible.
"""
import argparse
import hashlib
import json
import shlex
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

from .audit_pair import audit
from .run_pair import write

REMOTE = '/dev/shm/graphapi-baselines-overnight-20260913'
CURRENT_REMOTE = '/dev/shm/graphapi-baselines-824-two-lap-20260913'
CURRENT_LOCAL = Path('/tmp/baselines-824-two-lap-20260913')
GIB = 1024**3


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8*1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def rpc(action, root=None, **kwargs):
    request = dict(action=action, **kwargs)
    if root is not None:
        request['root'] = root
    command = ['env', 'PYTHONPATH='+REMOTE+'/integration', 'PYTHONDONTWRITEBYTECODE=1',
        'python3', '-m', 'tools.baselines.night_remote']
    for attempt in range(4):
        result = subprocess.run(['ssh','-o','ConnectTimeout=20','Gin',shlex.join(command)],
            input=json.dumps(request), text=True, capture_output=True)
        if result.returncode != 255 or action not in ('status','preflight','inventory') or attempt == 3:
            break
        print(f'Retrying read-only Gin {action} after SSH transport failure, attempt {attempt+1}', flush=True)
        time.sleep(20)
    if result.returncode:
        raise RuntimeError(f'Gin {action} failed ({result.returncode}): {result.stderr[-6000:]}')
    return json.loads(result.stdout)


def mirror(remote, local, final=False):
    local.mkdir(parents=True, exist_ok=True)
    command = ['rsync','-az','--timeout=120','--exclude=input.bag','--exclude=*smoke*',
        '--exclude=__pycache__','--exclude=*.pyc','--exclude=source',
        '--exclude=native-integration','--exclude=integration']
    if not final:
        command += ['--exclude=full_feats.pt']
    for attempt in range(4):
        result = subprocess.run(command+['Gin:'+remote+'/',str(local)+'/'])
        if result.returncode == 0:
            return
        if result.returncode not in (10,12,24,30,35,255) or attempt == 3:
            result.check_returncode()
        print(f'Retrying archive transfer after rsync exit {result.returncode}', flush=True)
        time.sleep(20)


def verify_inventory(local, inventory):
    for key, expected in inventory['files'].items():
        relative = Path(key)
        if relative.is_absolute() or '..' in relative.parts:
            raise ValueError('Invalid archive inventory path')
        path = local / relative
        if not path.is_file() or path.stat().st_size != expected['bytes'] or digest(path) != expected['sha256']:
            raise ValueError('Archive differs from native evidence: '+str(path))


def compress_features(local, inventory):
    archives = {}
    for key, expected in inventory['files'].items():
        if Path(key).name != 'full_feats.pt':
            continue
        path = local/key
        target = path.with_suffix(path.suffix+'.zst')
        subprocess.run(['zstd','-q','-3',str(path),'-o',str(target)],check=True)
        process = subprocess.Popen(['zstd','-q','-d','-c',str(target)],stdout=subprocess.PIPE)
        h = hashlib.sha256()
        for block in iter(lambda: process.stdout.read(8*1024*1024), b''):
            h.update(block)
        process.stdout.close()
        if process.wait() or h.hexdigest() != expected['sha256']:
            raise ValueError('Lossless feature archive failed verification: '+str(path))
        archives[key] = {'archive': str(target), 'archive_sha256': digest(target),
            'original_sha256': expected['sha256'], 'original_bytes': expected['bytes'],
            'archive_bytes': target.stat().st_size}
        # This completed exact derivative is now present losslessly in target.
        path.unlink()
    return archives


def archive(remote, local, report_path, already_audited=False):
    mirror(remote,local,final=True)
    inventory = rpc('inventory',remote)
    verify_inventory(local,inventory)
    if already_audited:
        report = json.loads((report_path/'report.json').read_text())
    else:
        report = audit(local,report_path)
    if not report['gate']['passed']:
        raise ValueError('Completed pair failed its declared data/execution gate')
    compressed = compress_features(local,inventory)
    receipt = dict(inventory, compressed_files=compressed, verified_at=time.time(),
        report=str(report_path), omitted_regenerable_files=['input.bag'])
    write(local/'archive-receipt.json',receipt)
    rpc('reclaim',remote,verified_files=inventory['files'],archive=str(local))
    return {'report':str(report_path), 'archive':str(local), 'verified_files':len(inventory['files'])}


def require_pair_pass(report):
    if not report['gate']['passed'] or set(report['baselines']) != {'clio','hovsg'}:
        raise ValueError('Verification gate did not pass for both baselines')
    if not all(row['required_checks_passed'] for row in report['baselines'].values()):
        raise ValueError('A native baseline failed required checks')


def run(args):
    root=args.local_root.resolve()
    status_path=root/'overnight-status.json'
    if status_path.exists():
        raise FileExistsError('Controller already submitted; inspect state before restarting')
    plan=json.loads((root/'prepared-scenes.json').read_text())
    scenes=[row['scene'] for row in plan['scenes']]
    state={'phase':'waiting_for_time_and_verification','complete':False,
        'not_before':args.not_before,'scene_order':scenes,'laps':2,'gpu':1,
        'started_at':time.time(),'completed':[],'gate_report':str(args.gate_report),
        'scope':'Native Clio + HOV in parallel, one shared input per scene; not a bug-free certification'}
    write(status_path,state)
    deadline=datetime.fromisoformat(args.not_before).timestamp()
    while time.time()<deadline or not args.gate_report.exists():
        post=CURRENT_LOCAL/'postprocess-status.json'
        if post.exists():
            post_status=json.loads(post.read_text())
            if not post_status.get('complete',False):
                raise ValueError('Current pair postprocessing failed: '+json.dumps(post_status))
        time.sleep(30)
    report=json.loads(args.gate_report.read_text())
    require_pair_pass(report)
    preflight=rpc('preflight')
    write(root/'launch-preflight.json',preflight)
    state['phase']='archiving_verification'
    write(status_path,state)
    state['verification_archive']=archive(CURRENT_REMOTE,CURRENT_LOCAL,args.gate_report.parent,True)
    state['free_local_bytes_after_verification']=shutil.disk_usage(root).free
    # Estimate is explicitly a budget, not measured coverage or run time. Final
    # launchability uses live free space and the observed previous recording size.
    state['estimated_input_frames']=round(sum(x['nominal_first_lap_s']*6 for x in plan['scenes']))
    state['input_budget_bytes']=state['estimated_input_frames']*1000000
    state['phase']='acquisition'
    write(status_path,state)
    if shutil.disk_usage(root).free < state['input_budget_bytes']+4*GIB:
        raise RuntimeError('Insufficient archive space for estimated full cohort plus 4 GiB reserve')
    remote_runs={scene:REMOTE+'/runs/'+scene for scene in scenes}
    local_runs={scene:root/'runs'/scene for scene in scenes}
    launched=set()
    for index,scene in enumerate(scenes):
        remote,local=remote_runs[scene],local_runs[scene]
        if scene not in launched:
            rpc('acquire',remote);launched.add(scene)
        state.update(phase='waiting_for_acquisition',scene=scene,next_scene=None)
        write(status_path,state)
        while True:
            info=rpc('status',remote)
            mirror(remote,local)
            if info.get('acquisition',{}).get('complete'):
                break
            if not info.get('acquisition_process',{}).get('alive'):
                raise RuntimeError('Acquisition stopped incomplete: '+scene)
            if shutil.disk_usage(root).free<3*GIB or info['scratch_free_bytes']<3*GIB:
                raise RuntimeError('Storage reserve reached; no further launches')
            time.sleep(30)
        # Native pair starts only after the preceding pair has been archived.
        rpc('pair',remote)
        state.update(phase='native_runs',scene=scene)
        if index+1<len(scenes):
            next_scene=scenes[index+1]
            # Give the native driver time to allocate/advertise its jobs. One
            # next-scene capture is the only permitted GPU overlap.
            time.sleep(5)
            free=rpc('status',remote)['scratch_free_bytes']
            if free>=30*GIB and shutil.disk_usage(root).free>=18*GIB:
                rpc('acquire',remote_runs[next_scene]);launched.add(next_scene)
                state['next_scene']=next_scene
            else:
                state['pipeline_deferred_for_storage']=True
        write(status_path,state)
        while True:
            info=rpc('status',remote)
            mirror(remote,local)
            if state.get('next_scene'):
                upcoming=state['next_scene']
                mirror(remote_runs[upcoming],local_runs[upcoming])
            pair=info.get('pair',{})
            if pair.get('phase') in ('native_complete', 'evaluation_complete'):
                break
            if pair.get('phase') in ('native_failure','guard_failure'):
                raise RuntimeError('Native pair failed: '+json.dumps(pair))
            if not info.get('pair_process',{}).get('alive'):
                raise RuntimeError('Native driver stopped incomplete: '+scene)
            if shutil.disk_usage(root).free<3*GIB:
                raise RuntimeError('Local archive reserve reached; no further launches')
            time.sleep(30)
        state.update(phase='auditing_and_archiving',scene=scene)
        write(status_path,state)
        result=archive(remote,local,args.reports/scene)
        state['completed'].append(dict(scene=scene,**result))
        write(status_path,state)
    state.update(phase='complete',complete=True,finished_at=time.time())
    write(status_path,state)
    return state


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--local-root',type=Path,required=True)
    parser.add_argument('--reports',type=Path,required=True)
    parser.add_argument('--gate-report',type=Path,required=True)
    parser.add_argument('--not-before',required=True)
    args=parser.parse_args()
    try:
        run(args)
    except (RuntimeError,ValueError,OSError,subprocess.CalledProcessError) as error:
        path=args.local_root/'overnight-status.json'
        state=json.loads(path.read_text()) if path.exists() else {}
        state.update(phase='blocked',complete=False,error=str(error),failed_at=time.time())
        write(path,state)
        raise


if __name__=='__main__':
    main()
