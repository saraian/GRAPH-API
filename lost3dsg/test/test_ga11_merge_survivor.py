"""GA-11: after a merge the SURVIVOR is queued for re-evaluation, read from the merge log the
service actually returns (keeper = label string, id in keeper_id / bbox_from_object_id). The
dict-shaped read killed run 20260907_002814 on the first merge."""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src", "perception_module"))
import rosstub  # noqa: E402

rosstub.install()
import object_manager_6 as om  # noqa: E402


class Log:
    def info(self, *_): pass
    def warn(self, *_): pass
    def error(self, *_): pass


node = object.__new__(om.ObjectManagerService if hasattr(om, "ObjectManagerService") else om.ObjectManager)
node.get_logger = lambda: Log()
queued = []
node._note_update = lambda oid, reason="updated", now=None: queued.append((oid, reason))
node.object_services = type("S", (), {"log_both": lambda *a, **k: None})()
real_shape = [{"keeper": "chair", "keeper_id": "obj_keep", "discarded": "chair#2", "bbox_from_object_id": "obj_keep", "similarity": 6.9}]
legacy_shape = [{"keeper": "chair", "discarded": "chair#2", "bbox_from_object_id": "obj_legacy"}]
odd_shape = ["not-a-dict", {"keeper": {"object_id": "obj_dict"}}]
for shape, want in ((real_shape, "obj_keep"), (legacy_shape, "obj_legacy"), (odd_shape, "obj_dict")):
    queued.clear()
    # 2026-09-15: the merge is called IN-PROCESS (object_services._cb_merge_objects), no
    # longer via _call_graph_api -> bridge /merge -> service. The seam moved; the contract
    # under test -- the survivor is queued from the merge log's real shape -- did not.
    node.object_services._cb_merge_objects = (
        lambda req, resp, _s=shape: type("R", (), {
            "success": True, "message": "", "merged_count": 1,
            "merge_log_json": json.dumps(_s)})())
    node.latest_agent_pose = None
    ok = node.merge_duplicate_objects()
    assert ok == 1 and queued == [(want, "merged")], (shape, queued)  # applied COUNT, not a bool
print("OK GA-11: the survivor is queued from the real merge-log shape; label strings and odd entries do not raise")
