"""Which merge engine, ontology flag and similarity weights config.py resolves for each launch config.

Read-only. For each candidate config file it spawns a fresh interpreter with GRAPH_API_CONFIG set,
imports config.py from the worktree, and prints what CFG holds. A fresh process per file because
config.py resolves CFG once at import.
"""
import os
import subprocess
import sys

WT = "/DATA/GRAPH-API/.claude/worktrees/found-merge-algorithm"
PM = f"{WT}/lost3dsg/src/perception_module"
CANDIDATES = [
    ("run_sim.sh default (tracked config.yaml)", f"{PM}/config.yaml"),
    ("schedules/configs/01_reference.yaml", f"{WT}/schedules/configs/01_reference.yaml"),
    ("schedules/configs/06_vlm_online.yaml", f"{WT}/schedules/configs/06_vlm_online.yaml"),
    ("GA-493 debug config (main repo, untracked)", "/DATA/GRAPH-API/lost3dsg/test/debug_configs/ga493_bbox_replay.yaml"),
    ("no file at all (_DEFAULTS)", "/nonexistent/config.yaml"),
]
SNIP = (
    "import sys; sys.path.insert(0, %r); import config as c; a = c.CFG['association']; "
    "print('CFG_PATH=', c.CFG_PATH); "
    "print('merge_engine=', a.get('merge_engine'), 'merge_ontology_channel=', a.get('merge_ontology_channel'), "
    "'merge_attribute_max_log_odds=', a.get('merge_attribute_max_log_odds')); "
    "print('similarity=', c.CFG['similarity'])"
) % PM

for label, path in CANDIDATES:
    env = dict(os.environ, GRAPH_API_CONFIG=path)
    env.pop("MERGE_ENGINE", None)
    r = subprocess.run([sys.executable, "-c", SNIP], env=env, capture_output=True, text=True)
    print(f"== {label}\n   file: {path} exists={os.path.exists(path)}")
    print("   " + r.stdout.strip().replace("\n", "\n   "))
    if r.returncode:
        print("   STDERR:", r.stderr.strip()[-400:])
