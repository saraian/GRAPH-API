"""Authoritative in-memory world model and derived AABB lookups."""

import logging
import math
import threading

from detection_index import DetectionIndex, bounds


class _TrackedList(list):
    """Keep the legacy live-list API while invalidating derived lookup state."""

    def __init__(self, owner):
        super().__init__()
        self._owner = owner

    def _changed(self):
        owner = self._owner
        owner._spatial = None
        owner._by_id.clear()
        owner._model_order.clear()
        owner._object_keys.clear()
        owner._objects_by_key.clear()
        owner._tracking_spatial = None
        owner._tracking_radius = None
        owner._tracking_unbounded.clear()

    def append(self, value):
        super().append(value)
        self._changed()

    def extend(self, values):
        super().extend(values)
        self._changed()

    def insert(self, index, value):
        super().insert(index, value)
        self._changed()

    def remove(self, value):
        super().remove(value)
        self._changed()

    def pop(self, index=-1):
        value = super().pop(index)
        self._changed()
        return value

    def clear(self):
        super().clear()
        self._changed()

    def __delitem__(self, key):
        super().__delitem__(key)
        self._changed()

    def __setitem__(self, key, value):
        super().__setitem__(key, value)
        self._changed()

    def __iadd__(self, values):
        result = super().__iadd__(values)
        self._changed()
        return result


class WorldModel:
    """Store objects and maintain binary-search spatial indexes as derived state.

    The object list remains live for compatibility with the existing services.  Every list
    mutation invalidates the indexes; the next query rebuilds them once.  Normal bbox updates
    go through ``update_bbox``/``refresh_spatial`` and update an already-built index in place.
    """

    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._actual_perceptions = []
            cls._instance.lock = threading.RLock()
            cls._instance._persistent_perceptions = _TrackedList(cls._instance)
            cls._instance._spatial = None
            cls._instance._by_id = {}
            cls._instance._model_order = {}
            cls._instance._object_keys = {}
            cls._instance._objects_by_key = {}
            cls._instance._tracking_spatial = None
            cls._instance._tracking_radius = None
            cls._instance._tracking_unbounded = set()
        return cls._instance

    @property
    def actual_perceptions(self):
        return self._actual_perceptions

    @property
    def persistent_perceptions(self):
        """The live list used by existing publishers and service handlers.

        Direct mutations are supported and invalidate the derived indexes through
        ``_TrackedList``.  Use ``snapshot`` for iteration and the methods below for bbox
        mutations when possible.
        """
        return self._persistent_perceptions

    def snapshot(self):
        with self.lock:
            return list(self._persistent_perceptions)

    def actual_snapshot(self):
        with self.lock:
            return list(self._actual_perceptions)

    def add_actual_perception(self, obj):
        with self.lock:
            self._actual_perceptions.append(obj)

    def add_persistent_perception(self, obj):
        # Preserve the pre-index API: object admission assigns IDs in ObjectServices, and
        # tests/replayers may temporarily add objects without one.  Such objects remain in
        # the authoritative list but are addressed by an internal index key if their bbox
        # is valid.
        with self.lock:
            self._persistent_perceptions.append(obj)

    def remove_persistent_perception(self, obj):
        with self.lock:
            self._persistent_perceptions.remove(obj)

    def clear_persistent_perceptions(self):
        with self.lock:
            self._persistent_perceptions.clear()

    @staticmethod
    def _valid_object_id(obj):
        key = getattr(obj, "object_id", None)
        return key if isinstance(key, str) and key else None

    def _index_key(self, obj):
        """Return the stable key assigned during the current index build."""
        key = self._object_keys.get(id(obj))
        if key is not None:
            return key
        object_id = self._valid_object_id(obj)
        return object_id or f"__world_model_object_{id(obj)}"

    def rebuild_spatial(self):
        """Rebuild the AABB and ID lookup state from the authoritative object list."""
        with self.lock:
            spatial = DetectionIndex()
            by_id = {}
            model_order = {}
            object_keys = {}
            objects_by_key = {}
            used = set()

            for order, obj in enumerate(self._persistent_perceptions):
                object_id = self._valid_object_id(obj)
                key = object_id if object_id is not None and object_id not in used else None
                if key is None:
                    # The binary index requires unique string IDs.  Runtime objects normally
                    # have graph-assigned IDs; this fallback keeps malformed/replay objects
                    # in the world model without allowing one to overwrite another's entry.
                    key = f"__world_model_object_{order}"
                    while key in used:
                        key += "_"
                used.add(key)
                object_keys[id(obj)] = key
                objects_by_key[key] = obj
                model_order[key] = order
                if object_id is not None and object_id not in by_id:
                    by_id[object_id] = obj

                try:
                    spatial.upsert(key, obj.bbox)
                except ValueError as exc:
                    logging.getLogger("world_model").warning(
                        "Object %s excluded from spatial lookup: %s",
                        object_id or key,
                        exc,
                    )

            self._spatial = spatial
            self._by_id = by_id
            self._model_order = model_order
            self._object_keys = object_keys
            self._objects_by_key = objects_by_key
            self._tracking_spatial = None
            self._tracking_radius = None
            self._tracking_unbounded.clear()

    def _ensure_spatial(self):
        if self._spatial is None:
            self.rebuild_spatial()

    def _index_object(self, obj):
        key = self._index_key(obj)
        try:
            self._spatial.upsert(key, obj.bbox)
        except ValueError as exc:
            self._spatial.remove(key)
            logging.getLogger("world_model").warning(
                "Object %s excluded from spatial lookup: %s",
                getattr(obj, "object_id", None) or key,
                exc,
            )

        if self._tracking_spatial is None:
            return
        try:
            box = bounds(obj.bbox)
            radius = float(self._tracking_radius(obj))
            if not math.isfinite(radius) or radius < 0.0:
                raise ValueError("tracking radius must be finite and nonnegative")
            centre = tuple((box[i] + box[i + 3]) / 2.0 for i in range(3))
            reach = {
                f"{axis}_{edge}": centre[i] + sign * radius
                for i, axis in enumerate("xyz")
                for edge, sign in (("min", -1), ("max", 1))
            }
            self._tracking_spatial.upsert(key, reach)
            self._tracking_unbounded.discard(key)
        except (ValueError, TypeError, OverflowError):
            self._tracking_spatial.remove(key)
            self._tracking_unbounded.add(key)

    def update_bbox(self, obj, bbox):
        """Update a live object's bbox and both derived indexes without a full rebuild."""
        bounds(bbox)
        with self.lock:
            if not any(candidate is obj for candidate in self._persistent_perceptions):
                return False
            self._ensure_spatial()
            obj.bbox = bbox
            self._index_object(obj)
            return True

    def refresh_spatial(self, obj):
        """Re-index an object whose bbox was changed in place by legacy code."""
        with self.lock:
            if not any(candidate is obj for candidate in self._persistent_perceptions):
                return False
            if self._spatial is None:
                return True
            self._index_object(obj)
            return True

    def get_object(self, object_id):
        with self.lock:
            self._ensure_spatial()
            return self._by_id.get(object_id)

    def candidates(self, bbox, margin=0.3):
        """Return objects whose AABBs may overlap ``bbox``.

        ``margin`` retains the old association API.  The query expands by twice that value,
        matching the cleanup branch's conservative broad-phase contract; the caller still
        applies its exact IoU or similarity criterion afterwards.
        """
        with self.lock:
            self._ensure_spatial()
            keys = self._spatial.query(bbox, 2.0 * float(margin))
            result = []
            for key in keys:
                obj = self._object_for_key(key)
                if obj is not None:
                    result.append(obj)
            return result

    def _object_for_key(self, key):
        obj = self._objects_by_key.get(key)
        return obj if obj is not None else self._by_id.get(key)

    def tracking_candidates(self, bbox, radius_fn):
        """Return ``(candidates, excluded)`` for the tracking transition.

        Each stored box is expanded by its own measured/fallback reach and the query box is
        expanded by its half-diagonal.  This is a broad phase only; the existing exact
        centre-distance test remains in ``check_tracking_transition``.

        ``excluded`` is how many live objects this pass did NOT put forward.  It is returned
        rather than left to the caller because the caller cannot recover it: this runs on the
        executor thread with no lock held afterwards, so an object added or removed by the
        HTTP surface between the return and a later subtraction would be charged to locality.
        GA-289's ``pruned_locality`` counter is the reader.
        """
        with self.lock:
            self._ensure_spatial()
            if self._tracking_spatial is None:
                self._tracking_spatial = DetectionIndex()
                self._tracking_radius = radius_fn
                self._tracking_unbounded.clear()
                for obj in self._persistent_perceptions:
                    self._index_object(obj)

            box = bounds(bbox)
            half_diagonal = sum((box[i + 3] - box[i]) ** 2 for i in range(3)) ** 0.5 / 2.0
            centre = tuple((box[i] + box[i + 3]) / 2.0 for i in range(3))
            query = {
                f"{axis}_{edge}": centre[i] + sign * half_diagonal
                for i, axis in enumerate("xyz")
                for edge, sign in (("min", -1), ("max", 1))
            }
            keys = set(self._tracking_spatial.query(query))
            keys.update(self._tracking_unbounded)
            kept = sorted(
                (obj for obj in self._persistent_perceptions
                 if self._object_keys.get(id(obj)) in keys),
                key=lambda obj: self._model_order.get(self._object_keys.get(id(obj)), 0),
            )
            return kept, len(self._persistent_perceptions) - len(kept)

    def clear_actual_perceptions(self):
        with self.lock:
            self._actual_perceptions.clear()

wm = WorldModel()
