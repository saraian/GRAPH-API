"""Run ONE check of test_perception_smoke.py in isolation: exec the file up to its runner loop."""
import pathlib
import sys

PM = '/DATA/GRAPH-API/.claude/worktrees/found-merge-algorithm/lost3dsg/src/perception_module'
SMOKE = PM + '/test_perception_smoke.py'
name = sys.argv[1] if len(sys.argv) > 1 else 'merge_path_evidence'
src = pathlib.Path(SMOKE).read_text()
head, sep, _tail = src.partition('\nfor name, fn in [(')
assert sep, "runner loop not found"
ns = {'__file__': SMOKE, '__name__': 'smoke_head'}
exec(compile(head, SMOKE, 'exec'), ns)
fn = ns[name]
try:
    fn()
    print(f"{name}: PASS")
except Exception as exc:  # noqa: BLE001
    import traceback
    traceback.print_exc()
    print(f"{name}: FAIL {type(exc).__name__}: {exc}")
    sys.exit(1)
