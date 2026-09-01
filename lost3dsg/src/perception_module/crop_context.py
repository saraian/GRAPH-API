#!/usr/bin/env python3
"""GA-108 — the image the describer and the embedder both see.

THE FAILURE THIS FIXES. 49% of run B's persistent objects carry `description == "unknown"`.
The describer is handed a tight axis-aligned crop of the detector box, and for a 6 cm sliver
of an air conditioner that crop is a few dozen pixels of textureless metal with no wall, no
bracket and no ceiling line in it. There is nothing in the image to identify, so "unknown" is
the only honest answer the model can give -- the request was unanswerable before it was sent.

SEVEN DECISIONS, each with the reason it is not the obvious alternative:

1. THE REGION COMES FROM THE MASK, NOT THE DETECTOR BOX. For a thin or diagonal object an
   axis-aligned detector box is mostly background: a broom leaning across a corner fills maybe
   a tenth of its own box. The mask's own bounding box is the tighter, truer extent.

2. PADDING IS ADAPTIVE, AND SMALL OBJECTS GET *MORE* RELATIVE CONTEXT, NOT LESS. This is the
   crux and it inverts the intuitive rule. A constant margin fails exactly where the failure
   is: 20% of a 6 cm sliver is another 1.2 cm of nothing. The pad is chosen so the object
   occupies a bounded FRACTION of the output, so a tiny detection gets a wide context window
   -- the bracket, the wall, the ceiling line -- and a large object settles at the ~20% floor.

3. THE TARGET IS MARKED WITH THE MASK CONTOUR, and the background is DIMMED, never blacked
   out. Blacking out the background is what turns a sliver into an unidentifiable blob: it
   destroys the very context decision 2 went to the trouble of including. A 2-3 px contour in
   a colour that contrasts the local scene makes the referent unambiguous while every
   contextual cue stays legible.

4. RESIZE PRESERVES ASPECT AND PADS TO SQUARE. Never stretch. Shape is one of the attributes
   the model is asked to report, and stretching silently changes the answer.

5. THE TEXT PROMPT CARRIES WHAT THE IMAGE CANNOT SAY. This is the cheapest win here and needs
   no pixels: the metric extent, the height above the floor, the range, the room. "The
   outlined object is roughly 0.98 x 0.38 x 0.64 m, mounted 2.1 m above the floor" separates
   an air conditioner from a picture frame with no image work at all.

6. THE THREE ROUTES TO "unknown" ARE DISTINGUISHED IN THE RECORD. A failed call, a failed
   parse and a genuine abstention are three different problems with three different fixes,
   and today they produce one indistinguishable string. An unanswerable request must never be
   recorded as a model refusal.

7. ONE CONSTRUCTION, TWO CONSUMERS. The describer and the embedder are handed the same image.
   If they see different things, no disagreement between them can be interpreted.
"""

import math

import numpy as np

# --- stated design choices (category c): visible, arguable, not fitted --------------------

# The object should occupy at most this fraction of the output AREA. Drives the adaptive pad:
# a small detection is given a wide window until it reaches this share of the frame.
TARGET_OBJECT_AREA_FRAC = 0.15

# Floor on the relative pad, so a large object still gets a margin.
MIN_PAD_FRAC = 0.20

# ABSOLUTE floor on the context window, as a fraction of the frame's shorter side.
# This is the term that makes decision 2 actually work, and my first version omitted it --
# see the docstring of adaptive_pad. A purely relative target is SCALE-FREE, so a 28 px
# sliver got a 72 px window: the object was duly 15% of it, and the window still contained
# nothing but the object and its immediate surround. Context is only meaningful relative to
# the SCENE, so the floor is a fraction of the frame.
MIN_CONTEXT_FRAC_OF_FRAME = 0.25

# Background dimming. 0.7 keeps context clearly readable; 0.0 would be the black-out this
# design exists to avoid.
DIM_FACTOR = 0.7

CONTOUR_THICKNESS_PX = 2

# The named constructions. `tight` is today's behaviour and is the default everywhere, so
# nothing changes until someone asks for it. The A/B arms ARE these names: one code path,
# one setting, no parallel implementation to drift.
TIGHT = "tight"                      # the detector box, clipped. What ships today.
PADDED = "padded"                    # mask bbox + adaptive context window
CONTOUR = "contour"                  # padded + mask contour + dimmed background
CONTOUR_METRIC = "contour_metric"    # same PIXELS as contour; the metric sentence differs
CONSTRUCTIONS = (TIGHT, PADDED, CONTOUR, CONTOUR_METRIC)

# Result status vocabulary for decision 6. These are the three routes to "unknown".
OK = "ok"
CALL_FAILED = "call_failed"          # the VLM never answered
PARSE_FAILED = "parse_failed"        # it answered and the answer could not be read
MODEL_ABSTAINED = "model_abstained"  # it answered "unknown" -- the only one that is a refusal
UNANSWERABLE = "unanswerable"        # we could not even build a usable image to ask about


# ---------------------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------------------


def _tight_crop(image, detector_box, meta):
    """Today's crop, byte for byte: the detector box clipped to the frame, no resize.

    Reproduced here rather than left in input_output.prepare_crops so that ALL FOUR ARMS
    come out of one function. If `tight` lived somewhere else, the A/B would be comparing
    two implementations rather than four settings, and any difference between them would be
    uninterpretable.
    """
    if detector_box is None:
        meta["status"] = UNANSWERABLE
        meta["reason"] = "no detector box"
        return None, meta
    h, w = image.shape[:2]
    x0 = max(0, min(int(detector_box[0]), w - 1))
    y0 = max(0, min(int(detector_box[1]), h - 1))
    x1 = max(x0 + 1, max(0, min(int(detector_box[2]), w)))
    y1 = max(y0 + 1, max(0, min(int(detector_box[3]), h)))
    crop = image[y0:y1, x0:x1]
    if crop.size == 0:
        meta["status"] = UNANSWERABLE
        meta["reason"] = "empty crop"
        return None, meta
    meta["object_area_frac"] = 1.0     # the box IS the crop
    meta["pad_scale"] = 1.0
    meta["source_window_px"] = [int(crop.shape[1]), int(crop.shape[0])]
    meta["output_px"] = [int(crop.shape[1]), int(crop.shape[0])]
    meta["resampled"] = False
    return crop, meta


def mask_bbox(mask):
    """Tight bbox of the mask as (x1, y1, x2, y2), or None if the mask is empty."""
    if mask is None:
        return None
    m = np.asarray(mask).astype(bool)
    if not m.any():
        return None
    ys, xs = np.nonzero(m)
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def adaptive_pad(box, frame_shape,
                 target_area_frac=TARGET_OBJECT_AREA_FRAC, min_pad_frac=MIN_PAD_FRAC,
                 min_context_frac=MIN_CONTEXT_FRAC_OF_FRAME):
    """-> (x1, y1, x2, y2) context window. Small objects get MORE relative context.

    The window side is chosen so the object's area is at most `target_area_frac` of the
    output. Working in areas, the required linear scale is 1/sqrt(target_area_frac); the
    result is floored at (1 + 2*min_pad_frac) so a large object still gets a margin, then
    floored AGAIN at an absolute fraction of the frame, and clamped at the frame edge.

    THE ABSOLUTE FLOOR IS THE PART I GOT WRONG FIRST AND THE SELF-CHECK CAUGHT. With only
    the relative target, a 28x14 px sliver received a 72x36 px window: the object was
    exactly the intended 15% of it, and the window still contained nothing but the object
    and a few pixels of wall -- the ceiling line that identifies it as a wall-mounted unit
    sat 9 px outside. A relative rule is scale-free, and "more context" for a small object
    has to mean more of the SCENE, not more multiples of a small thing.

    CLAMPING NEVER SHRINKS BACK TO THE TIGHT BOX. An object against the image edge gets
    whatever context exists on the other sides -- the window is pushed inward rather than
    trimmed symmetrically, so the amount of context stays as close to the target as the
    frame allows.
    """
    h, w = frame_shape[:2]
    x1, y1, x2, y2 = [float(v) for v in box]
    bw, bh = max(1.0, x2 - x1), max(1.0, y2 - y1)

    scale = max(1.0 + 2.0 * min_pad_frac, 1.0 / math.sqrt(max(target_area_frac, 1e-6)))
    cw, ch = bw * scale, bh * scale
    # The absolute floor. Without it the relative target is scale-free and a tiny object
    # gets a tiny window -- the object is correctly 15% of it and the window still holds no
    # scene. Measured against the frame's shorter side, so "context" means context in the
    # ROOM rather than context relative to a thing we already know is small.
    floor_px = min_context_frac * float(min(h, w))
    cw, ch = max(cw, floor_px), max(ch, floor_px)
    cw, ch = min(cw, float(w)), min(ch, float(h))

    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    nx1, ny1 = cx - cw / 2.0, cy - ch / 2.0
    # Push inward instead of trimming, so the window keeps its size near an edge.
    nx1 = min(max(0.0, nx1), max(0.0, w - cw))
    ny1 = min(max(0.0, ny1), max(0.0, h - ch))
    return int(round(nx1)), int(round(ny1)), int(round(nx1 + cw)), int(round(ny1 + ch))


def mask_outline(mask, thickness=CONTOUR_THICKNESS_PX):
    """Boolean outline of the mask, `thickness` px thick. Pure numpy — no cv2 needed."""
    m = np.asarray(mask).astype(bool)
    if not m.any():
        return np.zeros_like(m)
    eroded = m.copy()
    for _ in range(max(1, int(thickness))):
        e = eroded.copy()
        e[1:, :] &= eroded[:-1, :]
        e[:-1, :] &= eroded[1:, :]
        e[:, 1:] &= eroded[:, :-1]
        e[:, :-1] &= eroded[:, 1:]
        eroded = e
    return m & ~eroded


def contrasting_colour(image, mask):
    """Pick an outline colour far from the local scene, so the contour is actually visible.

    A fixed green line vanishes against foliage and a fixed red one against a brick wall.
    This takes the mean colour of the region and returns the corner of the RGB cube furthest
    from it — deterministic, and it adapts to the scene rather than assuming one.
    """
    px = np.asarray(image).reshape(-1, 3).astype(float)
    if px.size == 0:
        return (255, 0, 255)
    mean = px.mean(axis=0)
    corners = np.array([(0, 0, 0), (255, 0, 0), (0, 255, 0), (0, 0, 255),
                        (255, 255, 0), (255, 0, 255), (0, 255, 255), (255, 255, 255)],
                       dtype=float)
    return tuple(int(v) for v in corners[np.argmax(np.linalg.norm(corners - mean, axis=1))])


def resize_pad_square(image, size, fill=None):
    """Resize preserving aspect, then pad to a square. NEVER stretches.

    Uses PIL when available (proper resampling); falls back to nearest-neighbour indexing so
    the function still works, and says nothing about quality it cannot deliver.
    """
    img = np.asarray(image)
    h, w = img.shape[:2]
    if h == 0 or w == 0:
        return None
    scale = float(size) / max(h, w)
    nh, nw = max(1, int(round(h * scale))), max(1, int(round(w * scale)))

    try:
        from PIL import Image
        resized = np.asarray(Image.fromarray(img.astype(np.uint8)).resize(
            (nw, nh), Image.BILINEAR))
    except Exception:
        yi = (np.arange(nh) * (h / nh)).astype(int).clip(0, h - 1)
        xi = (np.arange(nw) * (w / nw)).astype(int).clip(0, w - 1)
        resized = img[yi][:, xi]

    if fill is None:
        fill = np.asarray(img).reshape(-1, img.shape[2]).mean(axis=0) if img.ndim == 3 else 0
    out = np.zeros((size, size, img.shape[2]) if img.ndim == 3 else (size, size),
                   dtype=np.uint8)
    out[:, :] = np.asarray(fill, dtype=np.uint8)
    oy, ox = (size - nh) // 2, (size - nw) // 2
    out[oy:oy + nh, ox:ox + nw] = resized
    return out


# ---------------------------------------------------------------------------------------
# The construction
# ---------------------------------------------------------------------------------------


def build_context_crop(image, mask, detector_box=None, size=0,
                       dim_factor=DIM_FACTOR, mark_contour=True,
                       target_area_frac=TARGET_OBJECT_AREA_FRAC,
                       construction=TIGHT):
    """-> (crop, meta) or (None, meta). The one image both consumers get.

    `construction` selects the arm, and TIGHT IS THE DEFAULT AND REPRODUCES TODAY EXACTLY:
    the detector box clipped to the frame, returned at its own resolution, with no mask, no
    padding, no contour and no resize. A default that changed what the next run does would
    not be a switch, it would be a silent edit.

    `meta` always explains what happened, including why nothing was built, so an
    unanswerable request is recorded as UNANSWERABLE rather than becoming a model refusal.
    """
    meta = {"status": OK, "construction": construction, "used_mask_bbox": False,
            "pad_scale": None, "object_area_frac": None, "contour": False}

    if construction not in CONSTRUCTIONS:
        # An unknown arm must not silently become a different one. Crash on the config,
        # not on the frame -- rule 14.
        raise ValueError(f"unknown crop construction {construction!r}; "
                         f"expected one of {CONSTRUCTIONS}")

    if construction == TIGHT:
        return _tight_crop(image, detector_box, meta)

    if image is None:
        meta["status"] = UNANSWERABLE
        meta["reason"] = "no image"
        return None, meta

    box = mask_bbox(mask)
    if box is not None:
        meta["used_mask_bbox"] = True
    else:
        box = detector_box
    if box is None:
        meta["status"] = UNANSWERABLE
        meta["reason"] = "neither a mask nor a detector box"
        return None, meta

    x1, y1, x2, y2 = [int(round(float(v))) for v in box]
    if x2 - x1 < 1 or y2 - y1 < 1:
        meta["status"] = UNANSWERABLE
        meta["reason"] = "degenerate region"
        return None, meta

    wx1, wy1, wx2, wy2 = adaptive_pad((x1, y1, x2, y2), image.shape,
                                      target_area_frac=target_area_frac)
    win = np.array(image[wy1:wy2, wx1:wx2], copy=True)
    if win.size == 0:
        meta["status"] = UNANSWERABLE
        meta["reason"] = "empty context window"
        return None, meta

    win_area = float(win.shape[0] * win.shape[1])
    meta["object_area_frac"] = ((x2 - x1) * (y2 - y1)) / win_area if win_area else None
    meta["pad_scale"] = (wx2 - wx1) / float(max(1, x2 - x1))

    want_marks = construction in (CONTOUR, CONTOUR_METRIC)
    if mask is not None and want_marks:
        sub = np.asarray(mask)[wy1:wy2, wx1:wx2].astype(bool)
        if sub.shape == win.shape[:2] and sub.any():
            if dim_factor is not None and dim_factor < 1.0:
                # DIM, never black: context must stay legible.
                outside = ~sub
                win[outside] = (win[outside].astype(float) * dim_factor).astype(np.uint8)
            if mark_contour and want_marks:
                colour = contrasting_colour(win, sub)
                edge = mask_outline(sub, CONTOUR_THICKNESS_PX)
                win[edge] = colour
                meta["contour"] = True
                meta["contour_colour"] = colour

    meta["source_window_px"] = [int(win.shape[1]), int(win.shape[0])]
    if not size:
        # NATIVE. size=0 means "do not resample": the window is handed over exactly as it
        # was cut from the frame.
        #
        # At one fixed output size the pipeline does both wrong things at once. A large
        # object's window is several hundred source pixels wide, so resizing to 224 THROWS
        # AWAY real detail; the sliver's window is floored at 25% of the shorter side (120 px
        # on a 640x480 feed), so resizing UP to 224 INVENTS pixels that carry no information.
        # Neither is a resolution the sensor produced.
        #
        # GA-108 decision 7 -- one construction, two consumers -- is satisfied by identical
        # GEOMETRY, not identical pixel counts. The embedder needs 224 because that is its
        # patch grid; the describer does not, and upsampling for it only costs tokens.
        meta["output_px"] = [int(win.shape[1]), int(win.shape[0])]
        meta["resampled"] = False
        return win, meta

    out = resize_pad_square(win, size)
    if out is None:
        meta["status"] = UNANSWERABLE
        meta["reason"] = "resize failed"
        return None, meta
    meta["output_px"] = [int(size), int(size)]
    meta["resampled"] = True
    # Whether this rendering added pixels the sensor never captured, or discarded pixels it
    # did. Recorded per crop so the A/B can separate "the window helped" from "the
    # resampling hurt" instead of confounding them.
    meta["upsampled"] = bool(size > max(win.shape[0], win.shape[1]))
    return out, meta


# ---------------------------------------------------------------------------------------
# Decision 5 — the text the image cannot carry
# ---------------------------------------------------------------------------------------


def metric_context_sentence(extent_m=None, height_above_floor_m=None,
                            range_m=None, room=None,
                            extent_uncertainty=None, n_observations=None,
                            min_observations=2):
    """One sentence of metric fact for the prompt. Omits what it does not know.

    READ THIS BEFORE ENABLING THE METRIC ARM. The numbers in this sentence come from the
    SAME perception pass whose output the sentence is meant to improve, so a wrong box
    becomes a confidently wrong measurement fed to the describer. That is worse than silence:
    it does not merely fail to help, it misleads, and it misleads MOST on exactly the small
    objects the arm exists to rescue.

    MEASURED, on 337 near pairs of the same label (<1.5 m apart, so very likely one physical
    object) across every bundle -- the disagreement in their LONGEST SIDE, which is a lower
    bound on the box error since both boxes can be wrong the same way:

        smaller object volume      n     median      p90
        < 0.001 m3                57      2.94x     6.47x
        0.001 - 0.01 m3           68      1.70x     4.18x
        0.01 - 0.1 m3            168      1.28x     2.24x
        > 0.1 m3                  44      1.21x     1.81x

    The error is MONOTONICALLY WORSE FOR SMALLER OBJECTS -- exactly inverse to where the
    sentence would be useful. On a sub-litre object the box's own longest side disagrees with
    itself by a median factor of 3. Telling the model "roughly 0.06 x 0.30 x 0.28 m" about a
    98 cm wall unit is not a hypothetical; it is what this data predicts.

    SO THE EXTENT IS SELF-LIMITING HERE, in two ways:

      * OMITTED when the object's own evidence is too thin to support any statement --
        fewer than `min_observations` sightings, or an extreme aspect ratio (a sliver whose
        longest side is more than 20x its shortest is a segmentation artefact more often
        than an object).
      * Expressed as a RANGE, never a point value, whenever `extent_uncertainty` is given.
        "between 0.3 and 1.2 m across" is defensible; "0.98 x 0.38 x 0.64 m" is not, unless
        somebody has shown the box is that good.

    `height_above_floor_m` HAS NO SOURCE IN THIS TREE. There is no floor plane anywhere --
    the only occurrences of the name are this function and its test. A caller that wants it
    must derive it and say from what: the map's floor plane (does not exist), the agent's
    height (an assumption about where the robot is standing), or the box's own z_min
    (circular -- it is the same box). Passing it means asserting a fourth number of unknown
    provenance, so it stays optional and unused until one of those exists.

    Everything included is measured; anything missing is left out rather than defaulted, so
    the model is never told something the system does not actually know.
    """
    parts = []

    if extent_m is not None and len(extent_m) == 3 and all(e is not None for e in extent_m):
        e = sorted((abs(float(x)) for x in extent_m), reverse=True)
        thin = e[2] > 1e-9 and (e[0] / e[2]) > 20.0
        too_few = n_observations is not None and n_observations < min_observations
        if thin or too_few:
            pass                      # no extent clause at all -- see the docstring
        elif extent_uncertainty and extent_uncertainty > 1.0:
            f = float(extent_uncertainty)
            parts.append("between {:.2f} and {:.2f} m across its longest side".format(
                e[0] / f, e[0] * f))
        else:
            parts.append("roughly {:.2f} x {:.2f} x {:.2f} m".format(*[float(x) for x in extent_m]))

    if height_above_floor_m is not None:
        h = float(height_above_floor_m)
        # "mounted" only above roughly waist height: a wall unit and a floor unit are
        # different objects, and the word carries that without asserting more than the
        # measurement supports.
        template = ("mounted {:.1f} m above the floor" if h > 1.2
                    else "{:.1f} m above the floor")
        parts.append(template.format(h))
    if range_m is not None:
        parts.append("about {:.1f} m from the camera".format(float(range_m)))
    if room:
        parts.append("in the {}".format(room))
    if not parts:
        return ""
    return "The outlined object is " + ", ".join(parts) + "."


def classify_description_result(raw, parsed, call_error=None):
    """Which of the three routes to 'unknown' produced this result. Decision 6.

    `raw` is what the VLM returned (None if the call itself failed), `parsed` the dict the
    parser produced (None if parsing failed). The distinction is not cosmetic: a failed call
    is an infrastructure problem, a failed parse is a brittleness problem, and an abstention
    is a question the model genuinely could not answer -- three different fixes that today
    all read as the same string.
    """
    if call_error is not None or raw is None:
        return CALL_FAILED
    if parsed is None:
        return PARSE_FAILED
    desc = str(parsed.get("description", "")).strip().lower()
    if desc in ("", "unknown", "none", "n/a"):
        return MODEL_ABSTAINED
    return OK


# ---------------------------------------------------------------------------------------
# Self-check
# ---------------------------------------------------------------------------------------


def _scene():
    """A 480x640 room: grey wall, a bright ceiling line, and a small dark object on it."""
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    img[:, :] = (120, 120, 125)
    img[0:40, :] = (230, 230, 235)          # ceiling line — the context that identifies
    mask = np.zeros((480, 640), dtype=bool)
    mask[60:74, 300:328] = True             # a 28x14 px sliver, mounted high
    img[60:74, 300:328] = (40, 40, 45)
    return img, mask


def demo():
    img, mask = _scene()

    # --- 1. mask bbox beats the detector box on a diagonal object -------------------------
    diag = np.zeros((100, 100), dtype=bool)
    for i in range(80):
        diag[10 + i // 4, 10 + i] = True
    mb = mask_bbox(diag)
    det = (0, 0, 100, 100)
    assert mb is not None
    mb_area = (mb[2] - mb[0]) * (mb[3] - mb[1])
    assert mb_area < (det[2] - det[0]) * (det[3] - det[1]) / 4
    print(f"  mask bbox : {mb} — {mb_area} px vs a {100*100} px detector box "
          f"(mostly background for a diagonal object)")

    # --- 2. THE CRUX: small objects get MORE relative context ------------------------------
    small_box = (300, 60, 328, 74)                 # 28 x 14
    big_box = (100, 100, 500, 400)                 # 400 x 300
    ws = adaptive_pad(small_box, img.shape)  # noqa: E501
    wb = adaptive_pad(big_box, img.shape)
    scale_small = (ws[2] - ws[0]) / (small_box[2] - small_box[0])
    scale_big = (wb[2] - wb[0]) / (big_box[2] - big_box[0])
    assert scale_small > scale_big, (scale_small, scale_big)
    assert scale_big >= 1.0
    print(f"  adaptive pad : 28px object -> x{scale_small:.2f} window; "
          f"400px object -> x{scale_big:.2f}. Small gets MORE, which a constant margin "
          f"gets exactly backwards")

    # the sliver's window actually reaches the ceiling line that identifies it
    assert ws[1] <= 40, f"context window missed the ceiling line: {ws}"
    print(f"  the sliver's window is y={ws[1]}..{ws[3]} — it now CONTAINS the ceiling line "
          f"at y<40 that a tight crop excluded")

    # --- 3. dim, never black; contour marks the referent -----------------------------------
    crop, meta = build_context_crop(img, mask, size=224, construction=CONTOUR)
    assert crop is not None and crop.shape == (224, 224, 3)
    assert meta["used_mask_bbox"] and meta["contour"]
    assert crop.max() > 0, "image was blacked out"
    # the background is dimmed but still carries signal: distinct values remain
    assert len(np.unique(crop.reshape(-1, 3), axis=0)) > 3, "context flattened away"
    print(f"  construction : {crop.shape[0]}x{crop.shape[1]}, object is "
          f"{meta['object_area_frac']*100:.1f}% of the window, contour "
          f"{meta['contour_colour']}, background dimmed not blacked")

    # --- 4. aspect preserved, padded to square ---------------------------------------------
    wide = np.zeros((20, 200, 3), dtype=np.uint8)
    wide[:, :] = (10, 200, 10)
    sq = resize_pad_square(wide, 224, fill=(0, 0, 0))
    assert sq.shape == (224, 224, 3)
    # Rows that are not the padding colour. (Measuring per-row std instead would count
    # every row, because R, G and B differ within any coloured row -- my first version of
    # this assertion did exactly that and "failed" on correct code.)
    rows = np.where((sq != np.array([0, 0, 0])).any(axis=(1, 2)))[0]
    band = rows.max() - rows.min() + 1
    assert 15 <= band <= 40, f"aspect not preserved: content band {band}px of 224"
    print(f"  resize : a 200x20 strip stays a {band}px band in 224x224 — "
          f"aspect preserved, not stretched")

    # --- 5. the metric sentence ------------------------------------------------------------
    s = metric_context_sentence(extent_m=(0.98, 0.38, 0.64), height_above_floor_m=2.1,
                                range_m=3.4, room="bedroom")
    assert "0.98 x 0.38 x 0.64" in s and "2.1 m above the floor" in s and "bedroom" in s
    print(f"  prompt text : {s}")
    partial = metric_context_sentence(extent_m=(0.1, 0.1, 0.1))
    assert "above the floor" not in partial and "camera" not in partial
    print(f"  unknown quantities are OMITTED, never defaulted : {partial}")

    # the extent is SELF-LIMITING -- measured box error is 2.94x median under 0.001 m3
    rng = metric_context_sentence(extent_m=(0.98, 0.38, 0.64), extent_uncertainty=2.94)
    assert "between" in rng and "0.33" in rng and "2.88" in rng, rng
    print(f"  uncertain box -> RANGE, not a point value : {rng}")

    thin = metric_context_sentence(extent_m=(2.0, 0.5, 0.05), room="bedroom")
    assert "m" not in thin.split("in the")[0].replace("The outlined object is ", "") or \
        "roughly" not in thin, thin
    assert "roughly" not in thin and "between" not in thin, thin
    print(f"  40:1 sliver -> extent clause DROPPED entirely : {thin}")

    few = metric_context_sentence(extent_m=(0.9, 0.4, 0.6), n_observations=1, room="kitchen")
    assert "roughly" not in few, few
    print(f"  seen once -> extent clause DROPPED : {few}")

    ok = metric_context_sentence(extent_m=(0.9, 0.4, 0.6), n_observations=5)
    assert "roughly" in ok, ok
    print(f"  seen 5x, normal aspect -> extent stated : {ok}")

    # --- 6. the three routes to 'unknown' are distinguishable -------------------------------
    assert classify_description_result(None, None, call_error=RuntimeError("x")) == CALL_FAILED
    assert classify_description_result("garbage {", None) == PARSE_FAILED
    assert classify_description_result("{}", {"description": "unknown"}) == MODEL_ABSTAINED
    assert classify_description_result("{}", {"description": "a white air conditioner"}) == OK
    print(f"  'unknown' routes : {CALL_FAILED} / {PARSE_FAILED} / {MODEL_ABSTAINED} / {OK} "
          f"— today all four record the same string")

    # --- 7. unanswerable is recorded as such, not as a refusal -------------------------------
    _c, m2 = build_context_crop(img, np.zeros_like(mask), detector_box=None,
                                construction=CONTOUR)
    assert _c is None and m2["status"] == UNANSWERABLE, m2
    print(f"  no mask and no box -> status {m2['status']!r} ({m2['reason']}) — "
          f"NOT recorded as the model refusing")

    # --- 8. THE SWITCH: tight is the default and reproduces today EXACTLY -----------------
    det_box = (300, 60, 328, 74)
    tight, m_t = build_context_crop(img, mask, detector_box=det_box)          # default arm
    assert m_t["construction"] == TIGHT
    # byte-for-byte identical to input_output.prepare_crops' slice
    h, w = img.shape[:2]
    x0 = max(0, min(int(det_box[0]), w - 1)); y0 = max(0, min(int(det_box[1]), h - 1))
    x1 = max(x0 + 1, max(0, min(int(det_box[2]), w)))
    y1 = max(y0 + 1, max(0, min(int(det_box[3]), h)))
    assert np.array_equal(tight, img[y0:y1, x0:x1]), "tight is not today's crop"
    assert tight.shape == (14, 28, 3), tight.shape
    print(f"  DEFAULT arm 'tight' : {tight.shape[1]}x{tight.shape[0]}, byte-identical to "
          f"today's prepare_crops slice — no resize, no mask, no contour")

    padded, m_p = build_context_crop(img, mask, detector_box=det_box, construction=PADDED,
                                     size=224)
    contour, m_c = build_context_crop(img, mask, detector_box=det_box, construction=CONTOUR,
                                      size=224)
    cmet, m_cm = build_context_crop(img, mask, detector_box=det_box,
                                    construction=CONTOUR_METRIC, size=224)
    assert padded.shape == contour.shape == (224, 224, 3)
    assert not m_p["contour"] and m_c["contour"] and m_cm["contour"]
    # contour_metric differs from contour in the PROMPT, not the pixels
    assert np.array_equal(contour, cmet), "contour_metric must share contour's pixels"
    print(f"  arms : padded contour={m_p['contour']}, contour contour={m_c['contour']}, "
          f"contour_metric pixels identical to contour ({np.array_equal(contour, cmet)}) "
          f"— the metric arm differs in WORDS")

    try:
        build_context_crop(img, mask, detector_box=det_box, construction="nonsense")
        raise AssertionError("an unknown arm must not be silently accepted")
    except ValueError as e:
        print(f"  unknown arm -> ValueError ({str(e)[:44]}...) — crash on config, not on frame")

    # --- 9. ONE GEOMETRY, TWO RENDERINGS -------------------------------------------------
    emb, m_e = build_context_crop(img, mask, detector_box=det_box, construction=CONTOUR,
                                  size=224)
    des, m_d = build_context_crop(img, mask, detector_box=det_box, construction=CONTOUR,
                                  size=0)
    assert m_e["source_window_px"] == m_d["source_window_px"], "the WINDOW must be identical"
    assert m_e["output_px"] == [224, 224] and m_e["resampled"] is True
    assert m_d["resampled"] is False and m_d["output_px"] == m_d["source_window_px"]
    print(f"  same geometry, two renderings : window {m_d['source_window_px']} -> "
          f"embedder {m_e['output_px']} (resampled), describer {m_d['output_px']} (native)")
    # the sliver's window is 120px; 224 INVENTS pixels the sensor never captured
    assert m_e["upsampled"] is True, m_e
    print(f"  and the meta says so: upsampled={m_e['upsampled']} for a "
          f"{m_d['source_window_px'][0]}px window rendered at 224 — recorded per crop so the "
          f"A/B can separate 'the window helped' from 'the resampling hurt'")

    # A LARGE object goes the other way: 224 discards real detail. mask=None here on
    # purpose -- with a mask the construction rightly prefers the MASK's bbox over the
    # detector box, and passing the sliver's mask with a large box measured the sliver
    # again. (My first version of this assertion did exactly that and "failed" on
    # correct code.)
    big_box = (100, 100, 500, 400)
    _b, m_b = build_context_crop(img, None, detector_box=big_box, construction=PADDED, size=224)
    print(f"  large object: window {m_b['source_window_px']} -> 224 "
          f"(upsampled={m_b['upsampled']}) — real pixels discarded")
    assert m_b["upsampled"] is False

    print("\ncrop_context self-check OK")


if __name__ == "__main__":
    demo()
