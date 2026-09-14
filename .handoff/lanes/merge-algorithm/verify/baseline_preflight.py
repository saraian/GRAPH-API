#!/usr/bin/env python3
"""Can the post-13d75b5 BASELINE re-run start? Checks every input. LAUNCHES NOTHING.

WHY THIS RUN IS NEEDED, in one paragraph. Every merge and association number the lane has
published was measured on `20260914_174342_hm3d_00824` and `20260914_180343_hm3d_00824`.
Both bundles were written BEFORE commit 13d75b5 (2026-09-14 18:54:20 +0200), which deleted
the delivery-time motion gate and made the plausibility gate judge the incoming view. Both
still carry `observation_discard` rows reading `manager_motion_gate` -- 19 and 13 of them --
which is exactly the gate that commit removed. So those maps were built from fewer accepted
detections than the current code produces, and a delta measured against them is a delta
against a tree that no longer exists.

WHAT MUST BE AUTHORISED BEFORE ANYTHING RUNS. This script checks and reports; it starts no
container, no model call and no GPU job. The owner authorises the run itself.

    python3 .handoff/lanes/merge-algorithm/verify/baseline_preflight.py
    python3 .handoff/lanes/merge-algorithm/verify/baseline_preflight.py --json

Exit code 0 when every check passes, 1 when any fails.
"""
import json
import os
import shutil
import subprocess
import sys

REPO = '/DATA/GRAPH-API/.claude/worktrees/found-merge-algorithm'
DATASET = '/DATA/GRAPH-API/lost3dsg/FOUND-Dataset'
WORKSPACE = '/home/xps/graphapi_ws'
# The tree the baseline arm must run: F1/F2 landed, the four correctness repairs NOT applied.
BASELINE_REV = '13d75b5'
IMAGE_TAG = 'graphapi-run:humble-ga290'
# The two bundles the baseline replaces, and the reason each is stale.
SUPERSEDES = ['20260914_174342_hm3d_00824', '20260914_180343_hm3d_00824']
REPAIRED = ['lost3dsg/src/perception_module/association.py',
            'lost3dsg/src/perception_module/object_manager_6.py',
            'lost3dsg/src/perception_module/object_services.py']

checks = []


def check(name, ok, detail):
    checks.append({'check': name, 'ok': bool(ok), 'detail': detail})
    return ok


def sh(*args):
    try:
        return subprocess.run(args, capture_output=True, text=True, timeout=60).stdout.strip()
    except (subprocess.SubprocessError, OSError) as e:
        return f'<{type(e).__name__}: {e}>'


def main():
    # --- the tree ---------------------------------------------------------------------
    head = sh('git', '-C', REPO, 'rev-parse', '--short', 'HEAD')
    check('HEAD is the baseline revision', head.startswith(BASELINE_REV),
          f'HEAD {head}, baseline {BASELINE_REV}')
    dirty = [line[3:] for line in sh('git', '-C', REPO, 'status', '--short').split('\n') if line]
    repaired_dirty = sorted(f for f in dirty if f in REPAIRED)
    # THE ONE THAT MATTERS. The container copies the sources from this checkout, so a dirty
    # working tree means the baseline arm would run the REPAIRS, and the run would measure
    # nothing. Working rule 9.
    check('the four correctness repairs are NOT in the working tree', not repaired_dirty,
          'clean' if not repaired_dirty
          else f'DIRTY, so the baseline arm would run the repairs: {repaired_dirty}')

    # --- the inputs -------------------------------------------------------------------
    scene = os.path.join(DATASET, 'habitat/hm3d-val-habitat-v0.2/00824-Dd4bFSTQ8gi/Dd4bFSTQ8gi.basis.glb')
    cfgjson = os.path.join(DATASET, 'habitat/hm3d-val-semantic-configs-v0.2/'
                                    'hm3d_annotated_basis.scene_dataset_config.json')
    schedule = os.path.join(DATASET, 'schedules/00824-Dd4bFSTQ8gi.schedule.json')
    for label, path in (('scene mesh', scene), ('scene dataset config', cfgjson),
                        ('tour schedule', schedule)):
        check(label, os.path.isfile(path), path)
    for engine in ('evidence', 'legacy'):
        p = os.path.join(REPO, f'lost3dsg/test/debug_configs/merge_debug_00824_{engine}.yaml')
        check(f'{engine} arm config', os.path.isfile(p), p)
    launcher = os.path.join(REPO, 'lost3dsg/test/debug_configs/launch_merge_debug.sh')
    check('launcher', os.path.isfile(launcher), launcher)

    # --- the secrets the perception service needs -------------------------------------
    env_local = os.path.join(REPO, 'lost3dsg/test/env.local.sh')
    has_url = False
    if os.path.isfile(env_local):
        # The VALUE is never printed or recorded; only whether the key is set to something.
        # The `export ` prefix is stripped first: without that this reported the key ABSENT
        # while it was present and exported, which is the shape of a check that cannot come
        # out the other way (working rule 18). Found by running it.
        for line in open(env_local):
            t = line.strip()
            if t.startswith('export '):
                t = t[len('export '):].strip()
            if t.startswith('MODAL_PERCEPTION_URL='):
                has_url = bool(t.split('=', 1)[1].strip().strip('"\''))
    check('env.local.sh sets MODAL_PERCEPTION_URL', has_url,
          env_local if os.path.isfile(env_local) else f'{env_local} MISSING')

    # --- the machine ------------------------------------------------------------------
    images = sh('docker', 'images', '--format', '{{.Repository}}:{{.Tag}}')
    check('docker image present', IMAGE_TAG in images.split('\n'), IMAGE_TAG)
    gpu = sh('nvidia-smi', '--query-gpu=index,memory.used,memory.total', '--format=csv,noheader')
    check('a GPU is visible', bool(gpu) and not gpu.startswith('<'), gpu.replace('\n', ' | '))
    free_gb = shutil.disk_usage(WORKSPACE).free / 1e9 if os.path.isdir(WORKSPACE) else 0.0
    # Each capped bundle on this host measured 1.5-3 GB; two arms plus margin.
    check('workspace has room for two arms', free_gb >= 15.0,
          f'{free_gb:.1f} GB free at {WORKSPACE} (need >= 15)')

    # --- what this run supersedes, and the evidence that it must ----------------------
    for run in SUPERSEDES:
        p = f'{WORKSPACE}/results/{run}/hook_decisions.jsonl'
        gated = 0
        if os.path.isfile(p):
            for line in open(p):
                if '"manager_motion_gate"' in line:
                    gated += 1
        check(f'{run} is pre-13d75b5 (it still has the deleted gate)', gated > 0,
              f'{gated} observation_discard row(s) reading manager_motion_gate')

    failed = [c for c in checks if not c['ok']]
    if '--json' in sys.argv:
        print(json.dumps({'checks': checks, 'failed': len(failed),
                          'baseline_rev': BASELINE_REV}, indent=1))
    else:
        for c in checks:
            print(f"  [{'ok ' if c['ok'] else 'FAIL'}] {c['check']}\n         {c['detail']}")
        print(f"\n{len(checks) - len(failed)} of {len(checks)} checks pass.")
        print("\nTHIS SCRIPT LAUNCHED NOTHING. When the owner authorises the run, the command is:")
        print("  cd /DATA/GRAPH-API/.claude/worktrees/found-merge-algorithm")
        print("  DETACH=1 bash lost3dsg/test/debug_configs/launch_merge_debug.sh evidence")
        print("  DETACH=1 bash lost3dsg/test/debug_configs/launch_merge_debug.sh legacy")
        print("Run the two arms ONE AT A TIME: they share the GPU and the Modal endpoint.")
    return 1 if failed else 0


if __name__ == '__main__':
    sys.exit(main())
