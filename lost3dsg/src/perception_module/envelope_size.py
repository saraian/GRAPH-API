#!/usr/bin/env python3
"""Is this object the size objects of its kind are measured to be?

The question is asked with the DETECTOR'S OWN LABEL and nothing else. There is no ontology,
no encoder, no alignment step: the label is folded to a lookup key and either the corpus has
an envelope for it or it does not.

WHAT AN ENVELOPE IS. For one class, the 1st and 99th percentile of the smallest, middle and
largest side, in metres, over a measured corpus of that class. `data/envelopes.json` holds
1050 of them, from Metric-Tree (857), ABO (180), HSSD (11) and two dimensional standards.
The file records its own sources and their checksums; regenerate it, never hand-edit it.

WHAT SKIPPING ALIGNMENT COSTS, measured on 340 admission decisions from runs
20260908_194329_hm3d_00861 and 20260909_004443_hm3d_00861, which recorded both paths:

    with an aligner (label -> ontology class -> envelope)   262 of 340 decisions (77.1%)
    this module (label -> envelope, no alignment)           174 of 340 decisions (51.2%)

Where both find an envelope (152 decisions) they choose the same one 145 times, and the
inside/outside answer agrees on 147 of those 152. The disagreements are the reason an
extension's aligner is worth having: "picture frame" lands here on an 85-sample envelope while
the aligner picks a 565-sample one, and the small one is what refused real pictures on a wall.

So this module is the floor, not the ceiling: it gives a size check to a stack that has no
ontology layer at all, and a stack that has one should keep using it.

AN UNKNOWN LABEL IS NOT A PASS. `size_ok` returns None for "no envelope for this class", and
None is not False and not True. A caller that treats it as either is claiming a measurement
nobody made.
"""
import json
import os
import re
from pathlib import Path

from hooks import ABSTAIN, ADMIT, REJECT, Decision, Filter

_DATA = Path(__file__).resolve().parent / "data" / "envelopes.json"
_CACHE = {}


def norm_key(text: str) -> str:
    """'Picture Frame', 'picture_frame' and 'PictureFrame' are one key.

    The corpora key their envelopes as slugs and a detector emits free text, so an exact
    match misses almost everything. Instance numbering goes first ('chair#3' is a chair),
    then everything that is not a letter or a digit.
    """
    s = re.sub(r"[#_]", " ", str(text).lower())
    s = re.sub(r"\d+", " ", s)
    return "".join(ch for ch in s if ch.isalnum())


def load(path=None) -> dict:
    """{normalised key: envelope}. Read once, kept in memory: the file is 169 KB."""
    p = str(path or os.environ.get("GRAPH_API_ENVELOPES") or _DATA)
    if p not in _CACHE:
        with open(p) as fh:
            raw = json.load(fh)
        env = raw.get("envelopes", raw)
        _CACHE[p] = {norm_key(k): v for k, v in env.items()}
    return _CACHE[p]


def envelope(label: str, path=None):
    """The envelope for this label, or None when the corpus has never measured its kind."""
    return load(path).get(norm_key(label))


def size_ok(label: str, extents, path=None):
    """-> (True | False | None, reason).

    True   every side falls inside the measured range for this kind.
    False  a side falls outside it; the reason names the side, the value and the bound.
    None   THE CORPUS HAS NO ENVELOPE FOR THIS KIND. Not a pass and not a failure.

    `extents` is any three side lengths in metres, in any order: they are sorted here,
    because a box lying on its side is the same box.
    """
    e = envelope(label, path)
    if e is None:
        return None, f"no envelope for {label!r}: this kind has never been measured"
    try:
        small, mid, large = sorted(float(x) for x in extents)
    except (TypeError, ValueError) as exc:
        return None, f"extents are not three numbers ({exc})"
    src = f"n={e.get('n', 0)}, {e.get('corpus', '?')}"
    for axis, value in (("small", small), ("mid", mid), ("large", large)):
        lo, hi = e[axis]
        if value < lo:
            return False, f"{axis} side {value:.3f} m is below the 1st pct {lo:.3f} m ({src})"
        if value > hi:
            return False, f"{axis} side {value:.3f} m is above the 99th pct {hi:.3f} m ({src})"
    return True, f"inside the measured range ({src})"


def extents_of(proposal: dict):
    """The three side lengths of a proposal's box, or None.

    The oriented box wins when the proposal carries one: a single view of an object lying
    diagonal to the map axes under-measures it as an axis-aligned box.
    """
    b = proposal.get("bbox") or {}
    if b.get("oriented_extents"):
        return tuple(float(v) for v in b["oriented_extents"])
    if not {"x_min", "x_max", "y_min", "y_max", "z_min", "z_max"} <= set(b):
        return None
    return (abs(b["x_max"] - b["x_min"]), abs(b["y_max"] - b["y_min"]),
            abs(b["z_max"] - b["z_min"]))


class SizeFilter(Filter):
    """A Filter that checks size against the corpus and nothing else.

    `config.yaml` selects it (`hooks.filter`) and configures it (`size_check`):

        size_check:
          enabled: true     # false turns the check into a no-op without editing any code
          enforce: false    # true makes an out-of-range box a refusal

    Both are read from the config FILE, not from the environment: a run is configured in one
    place, and a setting that can also arrive as an environment variable has two places to
    look and no single answer to "what did this run use".

    `enforce` is off by default on purpose. A filter that starts refusing the day it is
    installed changes what every later number means, and the annotation is enough to measure
    the change first.

    It ABSTAINS, never admits, when the corpus has no envelope. Abstain says "I have no
    grounds"; admit would say "I checked and it is fine".
    """
    name = "envelope-size"

    def __init__(self, enabled=None, enforce=None):
        if enabled is None or enforce is None:
            # Imported here, not at module scope, so the pure functions above need no
            # configuration at all: a caller can use size_ok() with nothing wired up.
            from config import CFG
            section = CFG["size_check"]
            enabled = section["enabled"] if enabled is None else enabled
            enforce = section["enforce"] if enforce is None else enforce
        self.enabled, self.enforce = bool(enabled), bool(enforce)

    def judge(self, proposal: dict) -> Decision:
        label = proposal.get("label", "")
        if not self.enabled:
            # A no-op says so. Silence here would be indistinguishable from a check that ran
            # and found nothing to object to.
            return Decision(ABSTAIN, "size check disabled (size_check.enabled: false)",
                            annotation={"size": {"status": "disabled", "label": label}})
        ext = extents_of(proposal)
        if ext is None:
            return Decision(ABSTAIN, "no bounding box on the proposal",
                            annotation={"size": {"status": "no box", "label": label}})
        ok, why = size_ok(label, ext)
        note = {"size": {"status": {True: "inside", False: "outside", None: "unmeasured"}[ok],
                         "label": label, "extents": [round(v, 3) for v in ext], "reason": why}}
        if ok is None:
            return Decision(ABSTAIN, why, annotation=note)
        if ok:
            return Decision(ADMIT, why, annotation=note)
        return Decision(REJECT if self.enforce else ADMIT, why, annotation=note)


if __name__ == "__main__":
    corpus = load()
    assert len(corpus) > 900, f"only {len(corpus)} envelopes: the corpus file is wrong"

    # the three answers are three answers
    ok, why = size_ok("chair", (0.5, 0.5, 0.9))
    assert ok is True, (ok, why)
    ok, why = size_ok("chair", (0.01, 0.01, 0.01))
    assert ok is False and "below" in why, (ok, why)
    ok, why = size_ok("flux capacitor", (0.5, 0.5, 0.5))
    assert ok is None and "never been measured" in why, (ok, why)

    # a label the detector really emits, in the spellings it really emits it in
    assert envelope("Picture Frame") is envelope("picture_frame") is envelope("picture frame#2")

    # order does not matter: a box on its side is the same box
    assert size_ok("chair", (0.9, 0.5, 0.5))[0] == size_ok("chair", (0.5, 0.5, 0.9))[0]

    # the filter abstains on an unmeasured kind and never refuses unless told to
    box = {"x_min": 0, "x_max": 0.01, "y_min": 0, "y_max": 0.01, "z_min": 0, "z_max": 0.01}
    assert SizeFilter(enabled=True, enforce=False).judge(
        {"label": "flux capacitor", "bbox": box}).outcome == ABSTAIN

    # disabled is a no-op that SAYS it is one, and never refuses
    off = SizeFilter(enabled=False, enforce=True).judge({"label": "chair", "bbox": box})
    assert off.outcome == ABSTAIN and off.annotation["size"]["status"] == "disabled", off
    assert "disabled" in off.reason

    lenient = SizeFilter(enabled=True, enforce=False).judge({"label": "chair", "bbox": box})
    assert lenient.outcome == ADMIT and lenient.annotation["size"]["status"] == "outside"
    assert SizeFilter(enabled=True, enforce=True).judge({"label": "chair", "bbox": box}).outcome == REJECT

    # the oriented box wins over the axis-aligned one when both are present
    both = dict(box, oriented_extents=[0.5, 0.5, 0.9])
    assert SizeFilter(enabled=True, enforce=False).judge(
        {"label": "chair", "bbox": both}).annotation["size"]["status"] == "inside"

    print(f"envelope_size self-check OK: {len(corpus)} envelopes")
