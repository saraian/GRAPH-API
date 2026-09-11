"""Incremental AABB lookup for detection-to-object association.

The index is deliberately independent of ROS.  It keeps sorted endpoint lists for all six
faces of each axis-aligned bounding box.  A query binary-searches the shortest qualifying
slice and then performs the exact six-sided overlap test on that slice.
"""

from bisect import bisect_left, bisect_right, insort
from math import isfinite, prod
from operator import itemgetter

_coordinate = itemgetter(0)


def bounds(bbox, margin=0.0):
    """Return ``(xmin, ymin, zmin, xmax, ymax, zmax)`` after validating ``bbox``."""
    if not isfinite(margin) or margin < 0:
        raise ValueError("association margin must be finite and nonnegative")
    try:
        lo = (float(bbox["x_min"]), float(bbox["y_min"]), float(bbox["z_min"]))
        hi = (float(bbox["x_max"]), float(bbox["y_max"]), float(bbox["z_max"]))
    except (KeyError, TypeError, ValueError, OverflowError) as exc:
        raise ValueError("bbox requires six finite coordinates") from exc
    if not all(isfinite(v) for v in lo + hi) or any(a > b for a, b in zip(lo, hi)):
        raise ValueError("bbox must be finite with min <= max")
    if margin == 0:
        return lo + hi
    result = tuple(v - margin for v in lo) + tuple(v + margin for v in hi)
    if not all(isfinite(v) for v in result):
        raise ValueError("expanded bbox is not finite")
    return result


def intersects(a, b):
    """Return whether two normalized six-coordinate boxes overlap, including faces."""
    return (a[0] <= b[3] and b[0] <= a[3]
            and a[1] <= b[4] and b[1] <= a[4]
            and a[2] <= b[5] and b[2] <= a[5])


def coverage(observation, stored):
    """Return the fraction of ``stored`` covered by ``observation``.

    Degenerate stored boxes return zero instead of producing a division by zero.  This is
    the old reconciliation helper retained with the index because callers use the same
    validated AABB representation.
    """
    a, b = bounds(observation), bounds(stored)
    volume = prod(b[i + 3] - b[i] for i in range(3))
    if volume <= 1e-12 or not isfinite(volume):
        return 0.0
    intersection = prod(
        max(0.0, min(a[i + 3], b[i + 3]) - max(a[i], b[i]))
        for i in range(3)
    )
    return intersection / volume


class DetectionIndex:
    """Incremental six-endpoint AABB index.

    ``query`` examines the shortest one-dimensional endpoint slice, filters those entries
    with the exact six-axis overlap test, and returns object IDs in insertion order.  Query
    cost is ``O(log n + k + h log h)`` for ``k`` examined endpoints and ``h`` hits.  Updates
    use sorted-list insertion/removal and are ``O(n)`` in the number of indexed objects.

    Callers serialize access.  The index intentionally does not own or mutate the objects
    represented by its IDs.
    """

    def __init__(self):
        self._boxes = {}
        self._order = {}
        self._next_order = 0
        self._ends = [[] for _ in range(6)]
        self.last_examined = 0

    def upsert(self, object_id, bbox):
        # Validate before removing an existing entry, so a bad update cannot erase a good
        # indexed box.
        b = bounds(bbox)
        if not isinstance(object_id, str) or not object_id:
            raise ValueError("index requires a nonempty string object_id")
        if self._boxes.get(object_id) == b:
            return

        order = self._order.get(object_id)
        self.remove(object_id)
        if order is None:
            order = self._next_order
            self._next_order += 1
        self._order[object_id] = order
        self._boxes[object_id] = b
        for axis, entries in enumerate(self._ends):
            insort(entries, (b[axis], object_id))

    def remove(self, object_id):
        old = self._boxes.pop(object_id, None)
        if old is None:
            return False
        for axis, entries in enumerate(self._ends):
            entries.pop(bisect_left(entries, (old[axis], object_id)))
        del self._order[object_id]
        return True

    def build(self, objects):
        """Replace the contents from ``(object_id, bbox)`` pairs atomically."""
        other = type(self)()
        for object_id, bbox in objects:
            other.upsert(object_id, bbox)
        self.__dict__.update(other.__dict__)

    def query(self, bbox, margin=0.0):
        b = bounds(bbox, margin)

        # The endpoint lists contain (coordinate, ID).  Bisecting by the coordinate alone
        # avoids constructing artificial ID sentinels and handles equal coordinates cleanly.
        best = None
        for axis in range(3):
            entries = self._ends[axis]
            stop = bisect_right(entries, b[axis + 3], key=_coordinate)
            if best is None or stop < best[0]:
                best = (stop, entries, 0, stop)

            entries = self._ends[axis + 3]
            start = bisect_left(entries, b[axis], key=_coordinate)
            count = len(entries) - start
            if count < best[0]:
                best = (count, entries, start, len(entries))

        count, entries, start, stop = best
        self.last_examined = count
        hits = (key for _, key in entries[start:stop]
                if intersects(self._boxes[key], b))
        return sorted(hits, key=self._order.__getitem__)

    def __len__(self):
        return len(self._boxes)
