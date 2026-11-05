"""Multi-view confirmation of map objects: an option, off by default.

WHY. Measured on six storeys (2 physical storeys x 3 configurations, 2026-10-27 bundles):
889 of 1,141 final objects had been seen in ONE perception cycle, at precision 0.25 against
the scene inventory. Keeping only objects seen in at least two cycles gave precision 0.61
(0.57 without windows, stairs and railings), at the cost of fewer real objects kept
(153 instead of 355). The same filter is what gives a multi-view box fuser its precision.

THE RULE. A sighting is one perception cycle (`ObservationRef.cycle_id`) whose observation was
applied to the object. Every applied mutation passes through one place
(`ObjectTrackingService._write_mutation_receipt`), so add, update, move and the exploration
branch are all counted there, voxels or not -- windows and stairs get a real count, not an
exemption. An object is CONFIRMED when it holds `association.confirm_min_views` distinct
cycles, and it STAYS confirmed: a move carries the state across, and a merge keeps the union
of both sighting sets. Tentative objects stay in the world model as association and merge
targets; they are left out of the published markers, the room semantics and
persistent_perception.json, and are written to tentative_perception.json instead.

THE CONSTANT. `confirm_min_views` is a STATED DESIGN CHOICE: 2 is the smallest multi-view
agreement. Offline sweeps over it are reported, never used to pick it.
null  = off: nothing is recorded, nothing is hidden, every output is unchanged.
1     = record sightings and confirmation fields, hide nothing.
"""
from config import CFG


def _min_views(value):
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(
            f"association.confirm_min_views must be null or an integer >= 1, got {value!r}")
    return value


CONFIRM_MIN_VIEWS = _min_views(CFG["association"].get("confirm_min_views"))

# Copied onto the replacement object when an object moves (object_services move path).
CARRY_ATTRS = ("sighting_cycles", "confirmed", "confirmed_at")


def enabled():
    return CONFIRM_MIN_VIEWS is not None


def note_sighting(obj, cycle_id, when):
    """Attach one perception cycle to `obj`. -> True only when this sighting confirms it."""
    if CONFIRM_MIN_VIEWS is None or obj is None or not cycle_id:
        return False
    cycles = getattr(obj, "sighting_cycles", None)
    if cycles is None:
        cycles = obj.sighting_cycles = set()
    cycles.add(str(cycle_id))
    if getattr(obj, "confirmed", False) or len(cycles) < CONFIRM_MIN_VIEWS:
        return False
    obj.confirmed = True
    obj.confirmed_at = {"by": "sighting", "cycle_id": str(cycle_id), "time": when,
                        "sightings": len(cycles)}
    return True


def absorb(keeper, discard, when):
    """A merge: the survivor holds both sighting sets and is confirmed if either side was,
    or if the union reaches the threshold. -> True only when the merge confirms the keeper."""
    if CONFIRM_MIN_VIEWS is None:
        return False
    cycles = set(getattr(keeper, "sighting_cycles", None) or ())
    cycles |= set(getattr(discard, "sighting_cycles", None) or ())
    keeper.sighting_cycles = cycles
    if getattr(keeper, "confirmed", False):
        return False
    if getattr(discard, "confirmed", False):
        keeper.confirmed = True
        keeper.confirmed_at = dict(getattr(discard, "confirmed_at", None) or {},
                                   inherited_from=getattr(discard, "object_id", None))
        return True
    if len(cycles) >= CONFIRM_MIN_VIEWS:
        keeper.confirmed = True
        keeper.confirmed_at = {"by": "merge", "cycle_id": None, "time": when,
                               "sightings": len(cycles),
                               "merged_from": getattr(discard, "object_id", None)}
        return True
    return False


def is_shown(obj):
    return CONFIRM_MIN_VIEWS is None or bool(getattr(obj, "confirmed", False))


def shown(objects):
    """The objects a consumer may see. Off: the same list object, untouched."""
    if CONFIRM_MIN_VIEWS is None:
        return objects
    return [o for o in objects if getattr(o, "confirmed", False)]


def summary(obj):
    return {"sightings": len(getattr(obj, "sighting_cycles", None) or ()),
            "confirmed": bool(getattr(obj, "confirmed", False)),
            "confirmed_at": getattr(obj, "confirmed_at", None),
            "confirm_min_views": CONFIRM_MIN_VIEWS}
