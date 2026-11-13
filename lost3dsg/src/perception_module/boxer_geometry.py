"""Consensus over full-object predictions, without recursively enclosing old boxes.

Retain original geometry samples. Select the observed box nearest the coordinate-wise
median of their AABB boundaries, with each axis normalized by its median span. The
output is an actual prediction (including its orientation), never a union or an
independently averaged yaw. There are no class-specific thresholds. With two discordant
samples the evidence is ambiguous; a deterministic geometry key breaks ties.
"""
from copy import deepcopy

import numpy as np

POLICY = "boxer_observation_consensus"
BOUNDS = ("x_min", "x_max", "y_min", "y_max", "z_min", "z_max")
GEOMETRY = BOUNDS + ("yaw", "oriented_center", "oriented_extents", "has_orientation")


def _sample(box):
    sample = {key: deepcopy(box[key]) for key in GEOMETRY if key in box}
    bounds = np.asarray([sample[k] for k in BOUNDS], dtype=float).reshape(3, 2)
    if not np.all(np.isfinite(bounds)) or np.any(bounds[:, 1] <= bounds[:, 0]):
        raise ValueError("Geometry consensus requires finite boxes with positive spans")
    return sample


def original_samples(obj):
    if obj is None:
        return []
    acc = getattr(obj, "_yaw_acc", None) or {}
    if acc.get("policy") == POLICY:
        return list(acc["samples"])
    # For an old store only its current geometry is available; do not pretend to
    # recover the original measurements from an already inflated envelope.
    box = getattr(obj, "bbox", None)
    return [_sample(box)] if box else []


def fuse_samples(samples):
    if not samples:
        raise ValueError("Geometry consensus requires at least one observation")
    samples = [_sample(box) for box in samples]
    bounds = np.asarray([[box[key] for key in BOUNDS] for box in samples])
    median = np.median(bounds, axis=0)
    spans = np.median(bounds[:, 1::2] - bounds[:, ::2], axis=0)
    scores = np.abs((bounds - median) / np.repeat(spans, 2)).sum(axis=1)

    def key(i):
        box = samples[i]
        return (float(scores[i]), tuple(bounds[i]), float(box.get("yaw", 0.)),
                tuple(box.get("oriented_center", ())), tuple(box.get("oriented_extents", ())))

    selected = min(range(len(samples)), key=key)
    return deepcopy(samples[selected]), {"policy": POLICY, "samples": samples, "n": len(samples)}


def update_geometry(obj, incoming):
    return fuse_samples(original_samples(obj) + [_sample(incoming)])


def merge_geometry(a, b):
    return fuse_samples(original_samples(a) + original_samples(b))
