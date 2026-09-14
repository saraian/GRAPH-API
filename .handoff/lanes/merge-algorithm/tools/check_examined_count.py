"""The [MERGE AABB] `examined` count is per QUERY. With N identical boxes each query examines N
endpoints and returns N hits; the fixed sum is N*N, the pre-fix sum (once per hit) was N*N*N."""
import os
import sys
from types import SimpleNamespace

PM = '/DATA/GRAPH-API/.claude/worktrees/found-merge-algorithm/lost3dsg/src/perception_module'
sys.path.insert(0, PM)
os.chdir(PM)
import rosstub  # noqa: E402

rosstub.install()
import object_services as osv  # noqa: E402

N = 4
box = {'x_min': 0.0, 'x_max': 1.0, 'y_min': 0.0, 'y_max': 1.0, 'z_min': 0.0, 'z_max': 1.0}
objects = [SimpleNamespace(label=f'o{i}', object_id=f'o{i}', bbox=dict(box)) for i in range(N)]
lines = []
svc = osv.ObjectServices.__new__(osv.ObjectServices)
svc.log_both = lambda level, msg: lines.append(msg)
svc.decision_log = SimpleNamespace(write=lambda *a, **k: None)
svc.get_logger = lambda: rosstub.Any()
assert osv.MERGE_ENGINE == 'legacy'
pairs, _, _ = osv.ObjectServices._merge_candidates(svc, objects, None, 0.8)
assert len(pairs) == N * (N - 1) // 2, len(pairs)
line = [ln for ln in lines if ln.startswith('[MERGE AABB]')][0]
examined = int(line.split('examined=')[1].split()[0])
assert examined == N * N, f"examined={examined}, expected {N * N} (pre-fix would be {N ** 3})"
print(f"OK examined={examined} for {N} identical boxes ({len(pairs)} pairs offered)")
