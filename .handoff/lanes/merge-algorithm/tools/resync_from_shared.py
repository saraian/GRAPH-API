#!/usr/bin/env python3
"""Re-sync this worktree to the shared tree's CURRENT bytes and re-apply the lane's changes.

For every tracked path under the synced roots:
  - not edited on this branch since the baseline import -> take the shared tree's bytes
  - edited on this branch, shared tree unchanged since the baseline -> keep ours
  - edited on both -> git merge-file (3-way: ours, baseline, shared); conflicts are reported
Untracked files that exist only in the shared tree are copied; files only here are kept.
Prints a table; exits 1 on any conflict. Run from the worktree root:
    python3 .handoff/lanes/merge-algorithm/tools/resync_from_shared.py [--apply]
"""
import os
import shutil
import subprocess
import sys

WT = '/DATA/GRAPH-API/.claude/worktrees/found-merge-algorithm'
SHARED = '/DATA/GRAPH-API'
BASELINE = '16880b0'
ROOTS = ['lost3dsg/src/perception_module', 'lost3dsg/test', 'lost3dsg/launch', 'lost3dsg/msg', 'lost3dsg/srv',
         'lost3dsg/CMakeLists.txt', 'lost3dsg/package.xml', 'run_sim.sh', 'run_sim_headless.sh']
SKIP_DIRS = {'__pycache__', '.ruff_cache', 'output', 'probe_assets', 'patches'}
SKIP_FILES = {'env.local.sh'}
APPLY = '--apply' in sys.argv


def git(*args, check=True):
    return subprocess.run(['git', *args], cwd=WT, capture_output=True, text=True, check=check).stdout


def show(rev, path):
    r = subprocess.run(['git', 'show', f'{rev}:{path}'], cwd=WT, capture_output=True)
    return r.stdout if r.returncode == 0 else None


def read(p):
    return open(p, 'rb').read() if os.path.exists(p) else None


edited_here = set(git('diff', '--name-only', f'{BASELINE}..HEAD').split())
files = []
for root in ROOTS:
    sp = os.path.join(SHARED, root)
    if os.path.isfile(sp):
        files.append(root)
        continue
    for d, dirs, fs in os.walk(sp):
        dirs[:] = [x for x in dirs if x not in SKIP_DIRS]
        for f in fs:
            if f in SKIP_FILES or f.endswith('.pyc'):
                continue
            files.append(os.path.relpath(os.path.join(d, f), SHARED))

rows = []
conflicts = 0
for rel in sorted(files):
    shared = read(os.path.join(SHARED, rel))
    ours = read(os.path.join(WT, rel))
    base = show(BASELINE, rel)
    if ours == shared:
        continue
    if rel not in edited_here:
        action = 'take shared'
        if APPLY:
            os.makedirs(os.path.dirname(os.path.join(WT, rel)) or '.', exist_ok=True)
            shutil.copyfile(os.path.join(SHARED, rel), os.path.join(WT, rel))
    elif base is not None and shared == base:
        action = 'keep ours (shared unchanged since baseline)'
    elif base is None:
        action = 'CONFLICT: edited here, no baseline'
        conflicts += 1
    else:
        # 3-way merge
        tmp = os.path.join('/home/xps/.claude/jobs/a50b14ef/tmp', 'merge3')
        os.makedirs(tmp, exist_ok=True)
        pb, po, pt = (os.path.join(tmp, n) for n in ('base', 'ours', 'theirs'))
        open(pb, 'wb').write(base)
        open(po, 'wb').write(ours)
        open(pt, 'wb').write(shared)
        r = subprocess.run(['git', 'merge-file', '-p', '-L', 'ours', '-L', 'baseline', '-L', 'shared', po, pb, pt],
                           capture_output=True)
        if r.returncode < 0 or r.returncode > 0 and b'<<<<<<<' in r.stdout:
            action = f'CONFLICT ({r.returncode} hunks)'
            conflicts += 1
            open(os.path.join(tmp, os.path.basename(rel) + '.conflict'), 'wb').write(r.stdout)
        else:
            action = '3-way merged'
            if APPLY:
                open(os.path.join(WT, rel), 'wb').write(r.stdout)
    rows.append((rel, action))

for rel, action in rows:
    print(f'{action:45s} {rel}')
print(f'\n{len(rows)} files differ; conflicts: {conflicts}; applied: {APPLY}')
sys.exit(1 if conflicts else 0)
