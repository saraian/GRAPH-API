"""Small, ROS-free state machine for the full-turn completion hook."""


class FullTurnDetector:
    """Emit one monotonically increasing scan id per uninterrupted 360° turn.

    Navigation code owns the actual action execution.  This class only turns
    completed turn actions into a deterministic event, which keeps the hook
    testable without Habitat or ROS.  A translation or a direction reversal
    starts a new scan; partial rotations are never reported as complete.
    """

    def __init__(self, full_turn_degrees=360.0, tolerance_degrees=1e-6):
        if full_turn_degrees <= 0:
            raise ValueError("full_turn_degrees must be positive")
        self.full_turn_degrees = float(full_turn_degrees)
        self.tolerance_degrees = float(tolerance_degrees)
        self._direction = None
        self._degrees = 0.0
        self._scan_id = 0

    @property
    def degrees(self):
        return self._degrees

    def reset(self):
        self._direction = None
        self._degrees = 0.0

    def update(self, action, amount_degrees):
        """Return a new scan id when ``action`` completes a full turn, else None."""
        if action not in ("turn_left", "turn_right"):
            self.reset()
            return None
        amount = abs(float(amount_degrees))
        if amount <= 0.0:
            self.reset()
            return None
        direction = 1 if action == "turn_left" else -1
        if self._direction not in (None, direction):
            self.reset()
        self._direction = direction
        self._degrees += amount
        if self._degrees + self.tolerance_degrees < self.full_turn_degrees:
            return None
        self._scan_id += 1
        scan_id = self._scan_id
        self.reset()
        return scan_id
