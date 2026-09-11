"""Regression tests for the cleanup branch's binary-search AABB index."""

import random
import pathlib
import sys
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src" / "perception_module"))

from detection_index import DetectionIndex, coverage


def box(x=0.0, length=0.4, y=0.0, width=0.4, height=0.4):
    return {
        "x_min": x,
        "x_max": x + length,
        "y_min": y,
        "y_max": y + width,
        "z_min": 0.0,
        "z_max": height,
    }


class DetectionIndexTests(unittest.TestCase):
    def test_boundaries_invalid_and_degenerate_boxes(self):
        index = DetectionIndex()
        index.upsert("negative", box(-2.0, 1.0))
        self.assertEqual(index.query(box(-1.0)), ["negative"])
        self.assertEqual(index.query(box(-0.999)), [])
        index.upsert("huge", box(-1e8, 2e8, y=-1e8, width=2e8))
        self.assertEqual(set(index.query(box(-1e9, 2e9))), {"negative", "huge"})

        for bad in (box(float("nan")), box(float("inf")), box(0.0, -1.0), {}):
            with self.assertRaises(ValueError):
                index.upsert("bad", bad)
            with self.assertRaises(ValueError):
                index.query(bad)

        index.upsert("thin", box(0.0, 0.0))
        self.assertIn("thin", index.query(box()))
        self.assertEqual(coverage(box(), box(0.0, 0.0)), 0.0)

    def test_random_mutations_match_independent_scan(self):
        rng = random.Random(1701)
        index, boxes = DetectionIndex(), {}
        for step in range(500):
            key = str(rng.randrange(120))
            if rng.random() < 0.15:
                index.remove(key)
                boxes.pop(key, None)
            else:
                current = box(
                    rng.uniform(-15, 15),
                    rng.uniform(0, 5),
                    rng.uniform(-15, 15),
                    rng.uniform(0, 5),
                )
                boxes[key] = current
                index.upsert(key, current)

            query = box(
                rng.uniform(-15, 15),
                rng.uniform(0, 10),
                rng.uniform(-15, 15),
                rng.uniform(0, 10),
            )
            margin = 0.6
            expected = {
                key
                for key, current in boxes.items()
                if all(
                    current[f"{axis}_min"] <= query[f"{axis}_max"] + margin
                    and current[f"{axis}_max"] >= query[f"{axis}_min"] - margin
                    for axis in "xyz"
                )
            }
            self.assertEqual(set(index.query(query, margin)), expected, step)

        rebuilt = DetectionIndex()
        rebuilt.build(boxes.items())
        self.assertEqual(set(rebuilt.query(box(-100.0, 200.0, -100.0, 200.0))), set(boxes))

    def test_updates_preserve_insertion_order(self):
        index = DetectionIndex()
        index.upsert("z", box())
        index.upsert("a", box())
        index.upsert("z", box(0.1))
        self.assertEqual(index.query(box()), ["z", "a"])
        index.remove("z")
        index.upsert("z", box())
        self.assertEqual(index.query(box()), ["a", "z"])


if __name__ == "__main__":
    unittest.main()
