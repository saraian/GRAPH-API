"""GA-44: the store exposes the durable id, history() resolves it, and two same-label objects
with colour/material 'unknown' never share a history row (the label fallback is gone).
Also GA-26: on_object_merged deactivates the absorbed row and records the merge."""
import os
import sys
import tempfile
from types import SimpleNamespace as NS

sys.path.insert(0, "/DATA/FOUND/vendor/graph-api/lost3dsg/src/perception_module")
import rosstub  # noqa: E402

rosstub.install()
from map_database import MapDatabase  # noqa: E402

BOX = dict(x_min=0.0, x_max=1.0, y_min=0.0, y_max=1.0, z_min=0.0, z_max=1.0)
FAR = dict(x_min=5.0, x_max=6.0, y_min=0.0, y_max=1.0, z_min=0.0, z_max=1.0)
tmp = tempfile.mkdtemp(dir=os.environ.get("SCRATCH", "/tmp"))
db = MapDatabase(os.path.join(tmp, "t.db"))
mk = lambda oid: NS(object_id=oid, label="doorway", color="unknown", material="unknown", description="", bbox=BOX, room_id="r")  # noqa: E731
a, b = mk("obj_a"), mk("obj_b")
db.on_new_object(a); db.on_new_object(b)  # noqa: E702
db.on_object_moved(a, BOX, FAR, distance=5.0, iou=0.0)
rows = db.objects()
assert {r["object_uuid"] for r in rows} == {"obj_a", "obj_b"}, rows
assert [h["event_type"] for h in db.history("obj_a")] == ["detected", "moved"]
assert [h["event_type"] for h in db.history("obj_b")] == ["detected"], "the move was filed under the other doorway"
assert db.history(next(r["id"] for r in rows if r["object_uuid"] == "obj_a")) == db.history("obj_a"), "rowid readers still work"
ghost = mk("obj_never_stored")
db.on_object_moved(ghost, BOX, FAR, distance=5.0, iou=0.0)   # must skip, not fall back to the label
assert [h["event_type"] for h in db.history("obj_b")] == ["detected"] and len(db.history("obj_a")) == 2
db.on_object_merged(a, b, step=3)
assert [r["object_uuid"] for r in db.objects()] == ["obj_a"], "the absorbed row must go inactive"
assert db.history("obj_b")[-1]["event_type"] == "merged" and "obj_a" in db.history("obj_b")[-1]["notes"]
print("OK GA-44/GA-26: durable id exposed and resolvable; no label fallback; merge recorded")
