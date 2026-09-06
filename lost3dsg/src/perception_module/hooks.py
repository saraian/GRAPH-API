#!/usr/bin/env python3
"""Extension seam of the belief.

Three blueprints, each reproducing today's behaviour so that a stack configured with
none of them runs exactly as before:

    Filter        per-proposal admission: may this detection enter the map?
                  (default: admit everything)
    Refiner       a second look at a node already in the map, given its neighbours:
                  propose revised fields, or nothing (default: nothing)
    Reevaluation  the queue of nodes that deserve that second look, fed by the object
                  manager whenever a node changes (default policy: every spatial
                  neighbour of the changed node; drained by the manager's periodic timer)
    Store         the map's persistence adapter: receives the belief's events (new,
                  moved, disappeared, uncertain) and answers the two reads the tools
                  need (current objects, one object's history). Default: the SQLite
                  temporal map (map_database.MapDatabase); any other backend — a
                  knowledge-graph endpoint, a document store — is a subclass.

A deployment points config `hooks.filter` / `hooks.refiner` at subclasses living in
another package — that package never has to be part of this repository:

    hooks:
      search_paths: ["/path/to/that/package"]   # prepended to sys.path
      filter:  "pkg.module:ClassName"            # subclass of Filter, no-arg constructor
      refiner: "pkg.module:ClassName"            # subclass of Refiner
      store:   "pkg.module:ClassName"            # subclass of Store (empty = SQLite map)
      decisions_log: ""                          # JSONL of decisions/revisions (default: output/)

ROS-free on purpose: the subclasses can be unit-tested without a ROS install.
"""
import importlib
import json
import os
import sys
import time
from dataclasses import dataclass, field

ADMIT, REJECT, ABSTAIN = "admit", "reject", "abstain"


@dataclass
class Decision:
    outcome: str                       # ADMIT | REJECT | ABSTAIN
    reason: str = ""
    annotation: dict = field(default_factory=dict)   # what a dashboard shows next to the node
    # GA-240. PROVISIONAL: the object enters the map, but no ontological reasoning may use it
    # until something else resolves it.
    #
    # WHY A THIRD STATE AND NOT A CHANGE TO `admitted`. Two outcomes were being made to carry
    # three meanings. FOUND maps hold and no-grounds onto ABSTAIN in order to keep them OUT of
    # the map (found/filter.py:8-13, "nothing enters the map except an admit"), while this
    # seam defines ABSTAIN as admissible on purpose -- a blueprint filter that abstains must
    # not empty the map. Both are right for their own side, and the collision put 296 objects
    # into the map unadmitted on run 20260901_174810_hm3d_00861.
    #
    # The owner's ruling is neither of the obvious repairs: a DECLINE stays out of the map, a
    # HOLD or an ungrounded proposal ENTERS but is unusable by the ontological layer until the
    # association/core level resolves it. That is a third state, so it gets a third field
    # rather than an overloaded second one. `admitted` keeps its meaning and its regression.
    provisional: bool = False

    @property
    def admitted(self) -> bool:
        return self.outcome != REJECT   # abstaining is not refusing

    @property
    def ontologically_usable(self) -> bool:
        """False for a provisional admission. The ontology channel must abstain on it."""
        return self.admitted and not self.provisional


class Filter:
    """A proposal is the dict the object manager is about to send to the Graph API:
    label, bbox (AABB keys + optional yaw / oriented_extents / oriented_center),
    color, material, description, room_id."""
    name = "passthrough"

    def judge(self, proposal: dict) -> Decision:
        return Decision(ADMIT, "pass-through")


class Refiner:
    """`node` is a belief node as a dict (object_id, label, bbox, color, material,
    description, room_id); `neighbours` the same for the nodes near it. Return the
    fields to revise (or an `annotation`), or None to leave the node alone."""
    name = "noop"

    def refine(self, node: dict, neighbours: list):
        return None


class Reevaluation:
    """Nodes waiting for a second look, deduplicated by object id."""

    def __init__(self):
        self._pending = {}            # object_id -> reason

    def on_update(self, object_id, neighbour_ids):
        for n in neighbour_ids:
            self._pending.setdefault(n, f"neighbour {object_id} updated")

    def mark(self, object_id, reason):
        self._pending.setdefault(object_id, reason)

    def drain(self):
        items = list(self._pending.items())
        self._pending.clear()
        return items

    def __len__(self):
        return len(self._pending)


class Store:
    """Persistence adapter of the map. `obj` is a belief object (label, color, material,
    description, bbox dict, room_id); bboxes are AABB dicts, optionally with the
    PCA keys. Events are fire-and-forget; reads return plain dicts so callers never
    see the backend. The blueprint stores nothing and answers nothing."""
    name = "null"

    def on_new_object(self, obj, phase="exploration", step=0):
        pass

    def on_object_moved(self, obj, old_bbox, new_bbox, distance, iou, phase="tracking", step=0):
        pass

    def on_object_deleted(self, obj, reason="", phase="tracking", step=0):
        pass

    def on_uncertain_added(self, obj, step=0):
        pass

    def on_object_room_changed(self, obj, old_room, new_room, step=0):
        """GA-45. The shipped SQLite store defined this fifth event and the object manager calls
        it on whatever store is configured (object_manager_6.py, the room-change loop), so a
        store built to THIS blueprint -- four events, as advertised -- crashed the run with an
        AttributeError the first time an object changed room. The blueprint now declares every
        event the manager fires; a store that does not care overrides nothing."""
        pass

    def objects(self, only_active=True):
        """-> [{id, label, color, material, description, bbox, room_id, is_active,
        is_uncertain, first_seen, last_seen, last_event}]"""
        return []

    def history(self, object_id):
        """-> [{timestamp, event_type, phase, step, bbox_old, bbox_new, distance, iou, notes}]"""
        return []


class DecisionLog:
    """Append-only JSONL: one line per admission decision or proposed revision. This is
    what an external dashboard reads; nothing else in the stack depends on it."""

    def __init__(self, path):
        self.path = path
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    def write(self, kind, object_ref, **payload):
        with open(self.path, "a") as f:
            f.write(json.dumps({"t": time.time(), "kind": kind, "object": object_ref, **payload}, default=str) + "\n")


def load_hook(spec, base, search_paths=()):
    """'' -> the blueprint itself; 'pkg.module:Class' -> that class, checked to be a `base`."""
    for p in search_paths:
        if p and p not in sys.path:
            sys.path.insert(0, p)
    if not spec:
        return base()
    mod, _, cls = spec.replace(":", ".").rpartition(".")
    obj = getattr(importlib.import_module(mod), cls)()
    if not isinstance(obj, base):
        raise TypeError(f"hooks: {spec} is not a {base.__name__}")
    return obj


def load_store(cfg, default):
    """`hooks.store` dotted path -> that Store; empty -> default() (the SQLite map)."""
    h = cfg.get("hooks", {}) or {}
    spec = h.get("store", "")
    return load_hook(spec, Store, h.get("search_paths", []) or []) if spec else default()


def load_hooks(cfg):
    """-> (filter, refiner, reevaluation queue) from the `hooks` config section."""
    h = cfg.get("hooks", {}) or {}
    paths = h.get("search_paths", []) or []
    return (load_hook(h.get("filter", ""), Filter, paths),
            load_hook(h.get("refiner", ""), Refiner, paths),
            Reevaluation())


if __name__ == "__main__":
    f, r, q = load_hooks({})
    assert f.judge({"label": "anything"}).admitted and r.refine({}, []) is None
    assert Decision(ABSTAIN).admitted and not Decision(REJECT).admitted
    # GA-240: provisional is a THIRD state, not a synonym for either of the other two.
    assert Decision(ABSTAIN, provisional=True).admitted, "provisional still enters the map"
    assert not Decision(ABSTAIN, provisional=True).ontologically_usable, "but is not usable"
    assert Decision(ADMIT).ontologically_usable, "a plain admit is usable"
    assert not Decision(REJECT).ontologically_usable, "a refusal is neither"
    q.on_update("a", ["b", "c"]); q.on_update("d", ["b"]); q.mark("e", "manual")  # noqa: E702
    assert len(q) == 3 and dict(q.drain())["b"] == "neighbour a updated" and len(q) == 0
    s = load_store({}, lambda: Store())
    s.on_new_object(None); assert s.objects() == [] and s.history(1) == []  # noqa: E702
    try:
        load_hook("json:JSONDecoder", Filter)
        raise AssertionError("a non-Filter class must be refused")
    except TypeError:
        pass
    print("hooks OK")
