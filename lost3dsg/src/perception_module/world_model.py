class WorldModel:
    """
    WorldModel singleton class to manage perceptions.
    Maintains actual and persistent perceptions of objects.
    In actual perceptions, objects detected in the current frame are stored.
    In persistent perceptions, objects that have been consistently detected over time are stored.
    """
    _instance = None
    
    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._actual_perceptions = []
            cls._instance._persistent_perceptions = []
            import threading
            cls._instance.lock = threading.RLock()
        return cls._instance
    
    @property
    def actual_perceptions(self):
        return self._actual_perceptions
    
    @property
    def persistent_perceptions(self):
        """The LIVE list. Mutating it is how nine call sites add and remove objects.

        GA-107. This deliberately still returns the raw list, and the reason is the fix that
        was NOT taken: returning a copy here would silently turn every
        `wm.persistent_perceptions.remove(...)` and `.append(...)` into a no-op -- nine sites
        across object_services alone (:584, :988, :1147, :1385, :1386, :1389) and the smoke
        harness. The map would stop changing and nothing would raise. Use `snapshot()` to
        ITERATE; use this to mutate.
        """
        return self._persistent_perceptions

    def snapshot(self):
        """A copy of the persistent list, taken under the lock. ITERATE THIS.

        GA-107, the defect: `object_tracking_callback` iterates the live list holding NO
        lock, on a four-thread executor, while an HTTP surface can trigger a merge at any
        moment. A removal during that iteration makes Python's list iterator skip the NEXT
        element -- silently, with no exception -- so an object is simply never considered for
        association in that pass.

        THE OBVIOUS FIX WOULD DEADLOCK THE NODE, which is why it is not the one here.
        Decorating the callback with @synchronized_world_model holds this RLock across
        `modify_existing_object`, which makes a BLOCKING HTTP call to the graph API bridge
        whose handler runs on another thread and takes the same lock. The lock is reentrant
        for one thread, not across two.

        A snapshot costs one list copy per pass -- tens of pointers -- and removes the
        skip-the-next-element defect without holding a lock across a network round trip.
        Objects removed mid-pass are still processed once from the copy, which is correct:
        the pass judges the world as it was when the pass began.
        """
        with self.lock:
            return list(self._persistent_perceptions)

    def actual_snapshot(self):
        """A copy of the actual-perception list, taken under the lock."""
        with self.lock:
            return list(self._actual_perceptions)
    
    def add_actual_perception(self, obj):
        self._actual_perceptions.append(obj)
    
    def add_persistent_perception(self, obj):
        self._persistent_perceptions.append(obj)
    
    def clear_actual_perceptions(self):
        self._actual_perceptions.clear()

wm = WorldModel()
