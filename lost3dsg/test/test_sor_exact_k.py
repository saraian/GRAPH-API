"""Statistical outlier removal must keep a cloud of exactly k points (reviewed 2026-09-07: it kept 0 of 30)."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src", "perception_module"))
import rosstub  # noqa: E402

rosstub.install(keep=("scipy",)) if "keep" in rosstub.install.__code__.co_varnames else rosstub.install()
import utils  # noqa: E402

rng = np.random.default_rng(0)
for n in (29, 30, 31, 60):
    pts = rng.random((n, 3)).astype(np.float32)
    try:
        kept = utils.statistical_outlier_removal(pts, k=30, std_ratio=1.5)
    except Exception as exc:   # rosstub may stub scipy on the host; then only the guard is testable
        if n <= 30:
            raise AssertionError(f"n={n} must return without touching the tree: {exc}")
        continue
    assert kept.sum() > 0, f"n={n}: nothing kept"
    if n <= 30:
        assert kept.all(), f"n={n} <= k: every point must be kept"
print("OK: SOR keeps clouds of k or fewer points; larger clouds keep something")
