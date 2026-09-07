"""GA-339. Adaptive hold: the feed keeps a STILL camera while the object manager still needs a look.

Pure Python, no habitat, no ROS: the feed host drives it once per frame and this file's own
self-check drives it with a fake signal (`python3 adaptive_hold.py`).

The signal is object_services._publish_merge_pending's merge_pending.json (GA-258):
`pending` pairs over the merge bar and short of the streak, `sweep`, `t`, `needs_max` (vendor e8612c7;
there is no cycle field, so freshness is read from `sweep` and `t`).
Rules, from plan/13-adaptive-dwell/topic.md (owner ruling 2026-09-07 ~13:50):
  * hold at least `min_frames` (18 = gate 0.5 s + one ~5 s cycle at 3 f/s);
  * then release only on a FRESH signal (its `sweep` advanced since the hold began and it is
    younger than `signal_max_age_s`) that says pending == 0, or whose sweep advanced by needs_max;
  * an unknown, absent or stale signal HOLDS, bounded by `max_frames` (45 = 15 s, owner's value).
    Failing toward looking, not toward leaving (rule 11).
"""


class AdaptiveHold:
    HOLD, RELEASE, CAP = "hold", "release", "cap"

    def __init__(self, min_frames=18, max_frames=45, signal_max_age_s=10.0):
        if min_frames < 1 or max_frames < min_frames:
            raise ValueError(f"min_frames {min_frames} / max_frames {max_frames}")
        self.min_frames, self.max_frames, self.max_age = int(min_frames), int(max_frames), float(signal_max_age_s)
        self.active = False
        self.frames = 0
        self._sweep0 = None
        self.last_need = None          # last pending count read, None = never known this hold
        # per-run counters, read by feed_stats.json
        self.episodes = self.frames_total = self.capped = self.released_on_zero = self.unknown_frames = 0
        self.need_at_last_release = None

    def start(self, signal, now):
        self.active, self.frames, self.last_need = True, 0, None
        self.episodes += 1
        self._sweep0 = signal.get("sweep") if signal else None

    def _fresh(self, signal, now):
        if not signal or signal.get("sweep") is None:
            return False
        if self._sweep0 is not None and signal["sweep"] <= self._sweep0:
            return False
        return (now - float(signal.get("t", 0.0))) <= self.max_age

    def step(self, signal, now):
        """One frame of holding. Returns HOLD, RELEASE or CAP; the caller issues NO action on HOLD."""
        assert self.active
        self.frames += 1
        self.frames_total += 1
        if self._fresh(signal, now) and signal.get("pending") is not None:
            self.last_need = int(signal["pending"])
            sweep, needs = signal.get("sweep"), signal.get("needs_max")
            swept = (self._sweep0 is not None and sweep is not None and needs is not None
                     and sweep >= self._sweep0 + int(needs))
            if self.frames >= self.min_frames and (self.last_need == 0 or swept):
                self.active = False
                self.released_on_zero += 1
                self.need_at_last_release = self.last_need
                return self.RELEASE
        else:
            self.unknown_frames += 1
        if self.frames >= self.max_frames:
            self.active = False
            self.capped += 1
            return self.CAP
        return self.HOLD

    def stats(self):
        return {
            "dwell_episodes": self.episodes,
            "dwell_frames_total": self.frames_total,
            "dwell_capped": self.capped,
            "dwell_released_on_zero": self.released_on_zero,
            "dwell_unknown_frames": self.unknown_frames,
            "need_at_last_release": self.need_at_last_release,
        }


def _check():
    # The sequence from the design note: absent -> need 3 at sweep 5 -> need 0 at sweep 6 (fresh)
    # -> release; then a hold where the only zero is STALE -> cap.
    h = AdaptiveHold(min_frames=4, max_frames=9, signal_max_age_s=10.0)
    h.start(None, now=100.0)
    for _ in range(3):
        assert h.step(None, 100.0) == h.HOLD          # absent = unknown = hold
    sig3 = {"pending": 3, "sweep": 5, "t": 100.0, "needs_max": 2}
    assert h.step(sig3, 101.0) == h.HOLD                # fresh, but need 3
    sig0 = {"pending": 0, "sweep": 6, "t": 102.0, "needs_max": 0}
    assert h.step(sig0, 102.0) == h.RELEASE             # fresh zero past the minimum
    assert h.stats()["dwell_released_on_zero"] == 1 and h.need_at_last_release == 0
    assert h.unknown_frames == 3
    # a zero read BEFORE the minimum must not release
    h.start(sig0, now=200.0)
    early = {"pending": 0, "sweep": 7, "t": 200.0, "needs_max": 0}
    assert h.step(early, 200.0) == h.HOLD
    # stale zero: same sweep as at hold start -> unknown -> holds to the cap
    h.start({"sweep": 6, "t": 300.0}, now=300.0)
    stale = {"pending": 0, "sweep": 6, "t": 300.0, "needs_max": 0}
    out = [h.step(stale, 300.0 + i) for i in range(9)]
    assert out[:-1] == [h.HOLD] * 8 and out[-1] == h.CAP, out
    # too old: sweep advanced but older than max age -> unknown
    h.start(None, now=400.0)
    old = {"pending": 0, "sweep": 8, "t": 380.0, "needs_max": 0}
    assert all(h.step(old, 400.0) == h.HOLD for _ in range(8)) and h.step(old, 400.0) == h.CAP
    # sweep advanced by needs_max releases even with pending > 0
    h.start({"sweep": 10, "t": 500.0}, now=500.0)
    for _ in range(4):
        assert h.step({"pending": 2, "sweep": 11, "t": 500.0, "needs_max": 2}, 500.0) == h.HOLD  # 1 of 2 sweeps
    assert h.step({"pending": 2, "sweep": 12, "t": 500.0, "needs_max": 2}, 500.0) == h.RELEASE
    assert h.stats()["dwell_capped"] == 2 and h.stats()["dwell_episodes"] == 5
    print("adaptive_hold.py: OK (absent holds, fresh zero releases past the minimum, early zero holds, "
          "stale zero caps, old signal caps, needs_max sweep releases)")


if __name__ == "__main__":
    _check()
