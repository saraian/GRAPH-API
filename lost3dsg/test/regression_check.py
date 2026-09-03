#!/usr/bin/env python3
"""Non-regression check on a belief JSON produced by the lost3dsg stack.

Compares a run's persistent_perception.json against a committed baseline:
object count, distinct labels, and per-axis belief span, each within a
tolerance. Exit 0 = pass, 1 = regression, 2 = usage/IO error.

  regression_check.py BELIEF.json BASELINE.json [--tol 0.15]
  regression_check.py BELIEF.json BASELINE.json --update-baseline

Record the baseline once (on the machine that runs the stack) with
--update-baseline after a run you trust; commit the baseline file.
# ponytail: scalar metrics + relative tolerance; per-object matching if
# association changes need finer regression signal.
"""
import json
import sys
from collections import Counter


def metrics(objs):
    centers = []
    for o in objs:
        b = o.get("bbox") or {}
        if b:
            centers.append([(b[f"{ax}_min"] + b[f"{ax}_max"]) / 2 for ax in "xyz"])
    span = [max(c[i] for c in centers) - min(c[i] for c in centers) for i in range(3)] if centers else [0, 0, 0]
    labels = Counter(o["label"].split("#")[0].strip() for o in objs if "label" in o)
    return {"objects": len(objs), "distinct_labels": len(labels), "span": span}


def compare(cur, base, tol):
    failures = []

    def rel(name, a, b):
        ref = max(abs(b), 1e-9)
        if abs(a - b) / ref > tol:
            failures.append(f"{name}: {a} vs baseline {b} (tol {tol:.0%})")

    rel("objects", cur["objects"], base["objects"])
    rel("distinct_labels", cur["distinct_labels"], base["distinct_labels"])
    for i, ax in enumerate("xyz"):
        rel(f"span_{ax}", round(cur["span"][i], 2), round(base["span"][i], 2))
    return failures


def main(argv):
    if len(argv) < 3:
        print(__doc__)
        return 2
    belief_path, baseline_path = argv[1], argv[2]
    update = "--update-baseline" in argv
    tol = 0.15
    if "--tol" in argv:
        tol = float(argv[argv.index("--tol") + 1])

    try:
        cur = metrics(json.load(open(belief_path)))
    except Exception as e:
        print(f"cannot read belief {belief_path}: {e}")
        return 2

    if update:
        json.dump(cur, open(baseline_path, "w"), indent=1)
        print(f"baseline written: {baseline_path} = {cur}")
        return 0

    try:
        base = json.load(open(baseline_path))
    except Exception as e:
        print(f"cannot read baseline {baseline_path}: {e} (record one with --update-baseline)")
        return 2

    failures = compare(cur, base, tol)
    print(f"current  {cur}")
    print(f"baseline {base}")
    if failures:
        print("REGRESSION:")
        for f in failures:
            print(f"  {f}")
        return 1
    print("PASS")
    return 0


def _selftest():
    objs = [{"label": f"chair #{i}", "bbox": {"x_min": i, "x_max": i + 1, "y_min": 0, "y_max": 1, "z_min": 0, "z_max": 1}} for i in range(10)]
    m = metrics(objs)
    assert m["objects"] == 10 and m["distinct_labels"] == 1 and m["span"][0] == 9
    assert compare(m, m, 0.15) == []
    worse = dict(m, objects=13)  # +30% objects must fail at 15%
    assert any("objects" in f for f in compare(worse, m, 0.15))
    assert compare(dict(m, objects=11), m, 0.15) == []  # +10% passes
    print("selftest OK")


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        _selftest()
    else:
        sys.exit(main(sys.argv))
