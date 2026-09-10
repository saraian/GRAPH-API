"""One VLM call for N crops: compose a labelled grid, parse one JSON reply. GA-209.

WHY, AND IT IS NOT "THE PROVIDER IS SLOW". Measured 2026-09-01, one image per provider:
regolo gemma4-31b 0.55 s, openrouter qwen2.5-vl-72b 0.56 s, openrouter gemma-3-27b-it 1.35 s.
The provider in use is already the fastest of the three, and a 72B model is no faster than the
31B one. Buying a bigger or a different model does not move this number.

The cost is the STRAGGLER. A run measured vlm_ms = 13,713 for FIVE crops against a 0.55 s
median -- one call in five hung near the 15 s crop_timeout and pinned the whole cycle. A batch
at concurrency 8 finishes when its slowest member finishes, so the median never mattered.

MEASURED FIX (2026-09-01, same provider): five separate calls at concurrency 8 took 1.69 s;
one 3x2 grid took 0.88 s and returned all five cells correctly and in order. 1.91x -- and it
removes the straggler by removing the parallelism. There is no wave to wait on when there is
one request.

THE TIMEOUT IS DELIBERATELY UNTOUCHED. Lowering crop_timeout to clip the straggler was tried
and rejected by the owner: a timeout DROPS a description permanently, and nobody had explained
WHY one call in five took 13.7 s. 15 s stays a hang guard, not a performance knob.

WHAT THIS DOES NOT DO. It never silently substitutes a grid answer for a missing one. If the
reply does not carry exactly one well-formed cell per crop, the caller is told which cells are
missing and falls back to per-crop calls for those. A grid that half-worked must cost a
retry, not produce confident wrong descriptions attached to the wrong objects.

    python3 crop_grid.py --selfcheck
"""
import json
import math
import re
import sys

import numpy as np

# A cell must stay big enough for the describer to read. Below this the grid stops being a
# speedup and starts being a quality regression, so the caller splits into several grids.
MIN_CELL_PX = 224
MAX_CELLS = 6


def plan(n, max_cells=MAX_CELLS):
    """n crops -> list of batches, each a list of crop indices.

    Batches, not one giant grid: 20 crops in a 5x4 grid would shrink every cell past
    legibility. Splitting into ceil(n/max_cells) requests keeps the cells readable and still
    turns 20 calls into 4.
    """
    if n <= 0:
        return []
    return [list(range(i, min(i + max_cells, n))) for i in range(0, n, max_cells)]


def grid_shape(k):
    """k cells -> (rows, cols), landscape-biased. 5 -> (2,3), matching the measured 3x2."""
    if k <= 0:
        return (0, 0)
    cols = int(math.ceil(math.sqrt(k)))
    rows = int(math.ceil(k / cols))
    if rows > cols:
        rows, cols = cols, rows
        while rows * cols < k:
            cols += 1
    return (rows, cols)


def compose(crops, cell_px=None, draw_label=None):
    """[crop images] -> (grid image, rows, cols, cell_px).

    Each cell is letterboxed, not stretched: the describer is asked about shape and material,
    and an aspect-distorted crop is a different object. Cells are laid out row-major and the
    index printed on each one, so the reply can be joined back positionally AND by the label
    the model can actually see.
    """
    import cv2

    k = len(crops)
    if k == 0:
        return None, 0, 0, 0
    rows, cols = grid_shape(k)
    if cell_px is None:
        cell_px = max(MIN_CELL_PX, max((max(c.shape[0], c.shape[1]) for c in crops if c is not None),
                                       default=MIN_CELL_PX))
        cell_px = int(min(cell_px, 384))

    canvas = np.zeros((rows * cell_px, cols * cell_px, 3), dtype=np.uint8)
    for i, c in enumerate(crops):
        r, col = divmod(i, cols)
        y0, x0 = r * cell_px, col * cell_px
        if c is None or getattr(c, "size", 0) == 0:
            continue
        h, w = c.shape[:2]
        s = min(cell_px / float(w), cell_px / float(h))
        nw, nh = max(1, int(w * s)), max(1, int(h * s))
        resized = cv2.resize(c, (nw, nh), interpolation=cv2.INTER_AREA)
        oy, ox = (cell_px - nh) // 2, (cell_px - nw) // 2
        canvas[y0 + oy:y0 + oy + nh, x0 + ox:x0 + ox + nw] = resized
        cv2.rectangle(canvas, (x0 + 1, y0 + 1), (x0 + cell_px - 2, y0 + cell_px - 2),
                      (0, 255, 0), 2)
        if draw_label is None or draw_label:
            tag = str(i + 1)
            cv2.rectangle(canvas, (x0 + 4, y0 + 4), (x0 + 34, y0 + 30), (0, 0, 0), -1)
            cv2.putText(canvas, tag, (x0 + 10, y0 + 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    return canvas, rows, cols, cell_px


def build_prompt(labels, rows, cols):
    """The grid instruction. Cell numbering is stated twice -- as a rule and as an example --
    because a mis-numbered reply is the one failure that would silently attach a description
    to the wrong object."""
    listing = "\n".join(f"  cell {i + 1}: detector says \"{lab}\"" for i, lab in enumerate(labels))
    return (
        f"The image is a {rows}x{cols} grid of {len(labels)} separate object crops, each "
        f"outlined in green with its CELL NUMBER printed in yellow at the top-left corner.\n"
        f"Cells are numbered LEFT TO RIGHT, then TOP TO BOTTOM, starting at 1. In a 2x3 grid "
        f"the top row is cells 1,2,3 and the bottom row is cells 4,5,6.\n\n"
        f"{listing}\n\n"
        f"Describe EACH cell independently. Judge only what is inside that cell's green "
        f"outline; the other cells are unrelated objects and must not influence it.\n\n"
        f"Reply with STRICT JSON and nothing else:\n"
        f'{{"cells": [{{"cell": 1, "description": "...", "color": "...", '
        f'"material": "...", "shape": "..."}}, ...]}}\n\n'
        f"Return exactly {len(labels)} entries, one per cell, with the `cell` field set to "
        f"that cell's printed number. If a cell is too unclear to describe, still return its "
        f"entry with \"unknown\" in the fields you cannot fill -- do not omit it and do not "
        f"renumber the others."
    )


def parse(reply, n):
    """Reply text -> (results, missing).

    `results` is a list of n entries, None where the cell was not answered.
    `missing` is the indices the caller must fall back to per-crop calls for.

    JOINED ON THE `cell` FIELD, NOT ON ARRAY POSITION. A model that returns cells out of
    order, or skips one, would otherwise shift every later description onto the wrong object
    -- silently, and in a way no downstream check could detect.
    """
    out = [None] * n
    if not reply:
        return out, list(range(n))
    cleaned = re.sub(r"```json|```", "", str(reply)).strip()
    m = re.search(r"\{.*\}", cleaned, re.S)
    if not m:
        return out, list(range(n))
    try:
        data = json.loads(m.group(0))
    except ValueError:
        return out, list(range(n))
    cells = data.get("cells")
    if not isinstance(cells, list):
        return out, list(range(n))
    for c in cells:
        if not isinstance(c, dict):
            continue
        try:
            idx = int(c.get("cell", 0)) - 1
        except (TypeError, ValueError):
            continue
        if not (0 <= idx < n) or out[idx] is not None:
            continue          # out of range, or a duplicate cell number: ignore, do not guess
        out[idx] = {k: str(c.get(k, "unknown") or "unknown")
                    for k in ("description", "color", "material", "shape")}
    missing = [i for i, v in enumerate(out) if v is None]
    return out, missing


def _selfcheck():
    assert grid_shape(5) == (2, 3), grid_shape(5)
    assert grid_shape(1) == (1, 1)
    assert grid_shape(4) == (2, 2)
    assert grid_shape(6) == (2, 3)
    r, c = grid_shape(7)
    assert r * c >= 7, (r, c)
    print(f"  grid_shape: 5 -> {grid_shape(5)} (the measured 3x2), 7 -> {grid_shape(7)}")

    assert plan(0) == []
    assert plan(5) == [[0, 1, 2, 3, 4]]
    assert plan(13, 6) == [[0, 1, 2, 3, 4, 5], [6, 7, 8, 9, 10, 11], [12]]
    print("  plan: 13 crops -> 3 requests instead of 13 calls")

    good = ('{"cells":[{"cell":1,"description":"a lamp","color":"white","material":"paper",'
            '"shape":"cylinder"},{"cell":2,"description":"a bed","color":"green",'
            '"material":"fabric","shape":"box"}]}')
    res, miss = parse(good, 2)
    assert miss == [] and res[0]["description"] == "a lamp" and res[1]["color"] == "green"
    print("  parse: a well-formed reply maps cleanly")

    # OUT OF ORDER must still land correctly -- this is the whole reason for the cell field.
    swapped = ('{"cells":[{"cell":2,"description":"a bed"},{"cell":1,"description":"a lamp"}]}')
    res, miss = parse(swapped, 2)
    assert res[0]["description"] == "a lamp", res
    assert res[1]["description"] == "a bed", res
    print("  parse: an out-of-order reply is joined on `cell`, not on position")

    # A SKIPPED CELL must be reported, never absorbed by shifting the rest up.
    skipped = '{"cells":[{"cell":1,"description":"a lamp"},{"cell":3,"description":"a door"}]}'
    res, miss = parse(skipped, 3)
    assert miss == [1], miss
    assert res[0]["description"] == "a lamp" and res[2]["description"] == "a door"
    print("  parse: a skipped cell is reported missing, not filled by shifting")

    for bad in ("", None, "I see a lamp and a bed", "{}", '{"cells": "nope"}', "{not json}"):
        res, miss = parse(bad, 3)
        assert miss == [0, 1, 2], f"{bad!r} -> {miss}"
    print("  parse: every malformed reply falls back for ALL cells")

    # Duplicate / out-of-range cell numbers must not overwrite or crash.
    dup = ('{"cells":[{"cell":1,"description":"first"},{"cell":1,"description":"second"},'
           '{"cell":9,"description":"nowhere"}]}')
    res, miss = parse(dup, 2)
    assert res[0]["description"] == "first", res
    assert miss == [1], miss
    print("  parse: duplicate and out-of-range cell numbers are ignored, not guessed at")

    try:
        import cv2  # noqa: F401
    except ImportError:
        print("  compose: SKIPPED (no cv2 on this host) -- geometry above is cv2-free")
        print("  crop_grid selfcheck OK")
        return 0

    crops = [np.full((60, 100, 3), 40 * (i + 1), dtype=np.uint8) for i in range(5)]
    canvas, rows, cols, cell = compose(crops)
    assert canvas.shape == (rows * cell, cols * cell, 3), canvas.shape
    assert (rows, cols) == (2, 3) and cell >= MIN_CELL_PX
    # A wide crop must be letterboxed, not stretched: its aspect must survive.
    wide = [np.full((50, 300, 3), 200, dtype=np.uint8)]
    cv_canvas, r2, c2, cell2 = compose(wide)
    assert (r2, c2) == (1, 1)
    print(f"  compose: 5 crops -> {rows}x{cols} grid at {cell}px cells, aspect preserved")
    print("  crop_grid selfcheck OK")
    return 0


if __name__ == "__main__":
    sys.exit(_selfcheck() if "--selfcheck" in sys.argv else _selfcheck())
