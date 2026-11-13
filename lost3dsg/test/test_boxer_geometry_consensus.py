"""Full-object estimates must not grow through repeated updates or merge grouping."""
import ast
import importlib.util
from pathlib import Path
from types import SimpleNamespace as NS
import unittest

import numpy as np

SRC = Path(__file__).resolve().parents[1] / "src/perception_module"
spec = importlib.util.spec_from_file_location("boxer_geometry", SRC / "boxer_geometry.py")
geometry = importlib.util.module_from_spec(spec)
spec.loader.exec_module(geometry)


def box(x=0., scale=1., yaw=0.):
    center = np.array([x, 2., 1.])
    extents = np.array([2., .2, 1.]) * scale
    c, s = abs(np.cos(yaw)), abs(np.sin(yaw))
    half = np.array([c * extents[0] + s * extents[1],
                     s * extents[0] + c * extents[1], extents[2]]) / 2
    lo, hi = center - half, center + half
    return dict(zip(geometry.BOUNDS, [lo[0], hi[0], lo[1], hi[1], lo[2], hi[2]]),
                yaw=yaw, oriented_center=center.tolist(),
                oriented_extents=extents.tolist(), has_orientation=True)


def track(samples):
    bbox, acc = geometry.fuse_samples(samples)
    return NS(bbox=bbox, _yaw_acc=acc)


class GeometryConsensus(unittest.TestCase):
    def test_repeated_noisy_views_do_not_accumulate_an_envelope(self):
        rng = np.random.default_rng(42)
        samples = [box(x=float(rng.normal(0, .03)), yaw=float(rng.normal(0, .06)))
                   for _ in range(80)]
        obj = None
        for incoming in samples:
            bbox, acc = geometry.update_geometry(obj, incoming)
            obj = NS(bbox=bbox, _yaw_acc=acc)
            self.assertIn(bbox, samples)
            self.assertAlmostEqual(np.prod(bbox["oriented_extents"]), .4)
        self.assertEqual(acc["n"], 80)

    def test_one_oversized_prediction_does_not_expand_supported_geometry(self):
        clean = [box(-.02), box(0.), box(.03)]
        obj = track(clean)
        before = obj.bbox.copy()
        result, acc = geometry.update_geometry(obj, box(.2, scale=20.))
        self.assertIn(result, clean)
        self.assertEqual(obj.bbox, before, "proposals must not mutate live state")
        self.assertEqual(acc["n"], 4)

    def test_merge_order_and_grouping_preserve_original_observations(self):
        samples = [box(-.04), box(.01), box(.02), box(.5, scale=4.), box(-.01)]
        expected, _ = geometry.fuse_samples(samples)
        left, right = track(samples[:2]), track(samples[2:])
        for a, b in [(left, right), (right, left)]:
            result, acc = geometry.merge_geometry(a, b)
            self.assertEqual(result, expected)
            self.assertEqual(acc["n"], len(samples))
        reversed_result, _ = geometry.fuse_samples(list(reversed(samples)))
        self.assertEqual(reversed_result, expected)

    def test_native_obb_and_aabb_are_kept_together(self):
        samples = [box(yaw=np.pi / 2 - .01), box(yaw=-np.pi / 2 + .01), box(yaw=1.5)]
        result, _ = geometry.fuse_samples(samples)
        self.assertIn(result, samples)
        expected = box(result["oriented_center"][0], yaw=result["yaw"])
        np.testing.assert_allclose([result[k] for k in geometry.BOUNDS],
                                   [expected[k] for k in geometry.BOUNDS])

    def test_update_and_merge_services_use_consensus_in_boxer_mode(self):
        tree = ast.parse((SRC / "object_services.py").read_text())
        methods = [n for n in tree.body if isinstance(n, ast.FunctionDef)
                   and n.name in ("fuse_orientation", "_combine_object_geometry")]
        env = dict(CFG={"perception": {"detector": "boxer"}}, boxer_geometry=geometry)
        exec(compile(ast.Module(body=methods, type_ignores=[]), "object_services.py", "exec"), env)
        obj = track([box(-.02), box(0.), box(.03)])
        outlier = track([box(.2, scale=20.)])
        for bbox, acc in [env["fuse_orientation"](obj, outlier.bbox),
                          env["_combine_object_geometry"](obj, outlier)]:
            self.assertLess(np.prod(bbox["oriented_extents"]), 1.)
            self.assertEqual(acc["n"], 4)
        # A confirmed relocation starts a fresh history instead of retaining old positions.
        bbox, acc = env["fuse_orientation"](None, box(10.))
        self.assertEqual(bbox["oriented_center"][0], 10.)
        self.assertEqual(acc["n"], 1)

    def test_invalid_boxes_are_not_hidden_in_consensus(self):
        with self.assertRaises(ValueError):
            geometry.fuse_samples([{**box(), "x_max": float("nan")}])


if __name__ == "__main__":
    unittest.main()
