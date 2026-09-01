#!/usr/bin/env python3
"""View-tagged visual descriptors for object re-identification.

THE QUESTION THIS ANSWERS is "is this the same object INSTANCE, seen from a different
angle" -- not "are these the same kind of thing". That distinction drives the model choice.
DINOv2 is self-supervised on dense correspondence and its features separate instances;
CLIP is trained to align images with text, so two different white pillows sit almost on top
of each other in CLIP space. Fusing two different white pillows is the exact failure being
fixed, so a text-aligned embedding is the wrong instrument for it.

WHAT IS VERIFIED. The crop construction, the bearing-cone rule, the diversity capping and
the abstention behaviour are pure numpy and are exercised by the self-check at the bottom
(`python3 visual_reid.py`). THE EMBEDDER IS ALSO NOW EXERCISED, on real archived frames,
using /DATA/ASPIRE/.venv/bin/python (torch 2.12.1+cu130) with HF_HOME=/DATA/huggingface_cache:
DINOv2-small on CUDA, 384-d unit-norm descriptors, 23 ms per crop, ~138 MiB of VRAM. Measured
separation on one archived frame: the same region shifted a few pixels scores 0.9307 while a
different region scores 0.5285.

That 0.9307-vs-0.5285 gap is a SANITY CHECK, NOT A VALIDATION. It says the encoder responds
to content rather than to noise. It says nothing about the question that matters -- whether
two views of ONE object score higher than two views of two DIFFERENT objects of the same
kind -- because two crops of one frame are not two viewpoints, and neither crop is labelled.
The real measurement needs per-detection instance labels, which the archive does not carry;
see the note in the A/B harness.

MODEL SIZE, with the measurement rather than the default. The recommendation was ViT-B/14.
Measured on this host: an RTX 4050 Laptop with 6141 MiB total and 2033 MiB FREE -- habitat
and the stack already hold 3741 MiB. ViT-B/14 is ~86M parameters and wants roughly 350 MB
in fp16 for weights alone before activations and the processor's workspace; that fits in
2 GB only if nothing else grows, and the thing it shares the card with is a simulator whose
memory use is not fixed. ViT-S/14 is ~21M parameters, about a quarter of the footprint, and
its features are still strong for instance retrieval. DEFAULT IS ViT-S/14, overridable by
config. If the embedder is ever run on a machine with real headroom, switch it and say so.
"""

import math
import os

import numpy as np

# DINOv2 ViT-S/14 patch size. The processor resizes to a multiple of this.
_PATCH = 14
DEFAULT_MODEL_ID = "facebook/dinov2-small"       # ViT-S/14, ~21M params
LARGER_MODEL_ID = "facebook/dinov2-base"         # ViT-B/14, ~86M params -- needs headroom

# The project's shared cache. The default HF cache location is unwritable here.
DEFAULT_HF_HOME = "/DATA/huggingface_cache"


# ---------------------------------------------------------------------------------------
# Crop construction — the same thing the describer sees
# ---------------------------------------------------------------------------------------


def masked_padded_crop(image, mask, bbox_xyxy, pad_frac=0.25, fill=None, size=224,
                       construction=None):
    """DELEGATES to crop_context.build_context_crop — GA-108 decision 7, one construction.

    This function used to build its own crop and SUPPRESS the background by filling outside
    the mask with a flat colour. GA-108 settled that that is wrong: blacking (or flattening)
    the background is what turns a 6 cm sliver into an unidentifiable blob, and it destroys
    the context that decision 2 exists to include. The construction now dims the surround
    and marks the object with its mask contour instead.

    It also has to be the SAME image the describer sees. If the embedder and the describer
    are handed different pixels, no disagreement between them can be interpreted -- and
    disagreement between them is exactly the signal association wants.

    `pad_frac` and `fill` are accepted and IGNORED, so old call sites keep working while
    the construction is centralised. Returns None on an unusable region, never a blank
    image -- a missing descriptor must be an absence, not a vector.
    """
    from crop_context import TIGHT, build_context_crop
    # The construction is the CALLER's, from CFG["crop"]["construction"], because the
    # describer and the embedder must be handed the same arm. Defaulting to TIGHT here
    # keeps today's behaviour when nobody has chosen.
    crop, _meta = build_context_crop(image, mask, detector_box=bbox_xyxy, size=size,
                                     construction=construction or TIGHT)
    return crop


def mask_sampled_appearance(image, mask, bbox_xyxy):
    """View-invariant colour statistics sampled INSIDE the mask.

    An independent channel from the visual embedding, and deliberately so: it survives when
    the embedder is unavailable, it is cheap, and it fails differently. Mean and standard
    deviation per channel over the masked pixels only -- so it describes the object, not the
    wall behind it.

    Returns None when there is nothing to sample. Never a zero vector.
    """
    if image is None or bbox_xyxy is None:
        return None
    from crop_context import as_2d_mask
    mask = as_2d_mask(mask)          # GA-165: Detection.mask is 3-D
    h, w = image.shape[:2]
    x1, y1, x2, y2 = [int(round(float(v))) for v in bbox_xyxy]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    if x2 - x1 < 1 or y2 - y1 < 1:
        return None
    region = image[y1:y2, x1:x2]
    if mask is not None:
        m = mask[y1:y2, x1:x2]
        if m.shape[:2] != region.shape[:2]:
            return None
        m = m.astype(bool)
        if not m.any():
            return None
        px = region[m]
    else:
        px = region.reshape(-1, region.shape[2]) if region.ndim == 3 else region.reshape(-1, 1)
    px = px.reshape(-1, px.shape[-1]).astype(float) / 255.0
    return np.concatenate([px.mean(axis=0), px.std(axis=0)])


# ---------------------------------------------------------------------------------------
# The embedder — WRITTEN, NOT RUN. See the module docstring.
# ---------------------------------------------------------------------------------------


class DinoV2Embedder:
    """Lazy DINOv2 wrapper. Nothing is imported or downloaded until the first embed call.

    Lazy on purpose: importing torch costs seconds and VRAM, and this module is imported by
    code that may never embed anything. It also means the rest of this file -- the crop
    construction and the comparability logic -- stays importable on a host with no torch,
    which is the only reason any of it could be tested at all.
    """

    def __init__(self, model_id=None, device=None, hf_home=None, input_size=224):
        self.model_id = model_id or DEFAULT_MODEL_ID
        self.hf_home = hf_home or os.environ.get("HF_HOME") or DEFAULT_HF_HOME
        self._device = device
        self.input_size = int(input_size)   # 224 = 16 patches of 14
        self._model = None

    def _load(self):
        if self._model is not None:
            return
        os.environ.setdefault("HF_HOME", self.hf_home)
        import torch
        from transformers import AutoModel

        if self._device is None:
            # Only claim the GPU if there is real headroom. The card is shared with the
            # simulator, and an OOM inside a perception callback would take the run down --
            # a slower CPU descriptor is strictly better than a dead node.
            self._device = "cpu"
            if torch.cuda.is_available():
                free, _total = torch.cuda.mem_get_info()
                if free > 1_500_000_000:
                    self._device = "cuda"
        self._model = AutoModel.from_pretrained(
            self.model_id, cache_dir=self.hf_home).to(self._device).eval()

    def _pixel_values(self, crop_rgb):
        """Build the model input ourselves rather than via AutoImageProcessor.

        TWO REASONS, and the second is the important one.

        (1) `AutoImageProcessor` requires torchvision, which this environment does not have.
        That alone would only be a reason to install something.

        (2) THE PROCESSOR WOULD UNDO GA-108. Its default pipeline resizes the short side and
        then CENTRE-CROPS to 224 -- so a carefully constructed square context window, padded
        to preserve aspect, gets its edges cut off again, and for a small object near the
        window's edge the processor would crop away the very context the window was widened
        to include. The construction is the whole point; nothing downstream may re-crop it.

        So: resize the already-square crop to the model's input size and apply ImageNet
        normalisation, which is what DINOv2 was trained with. No hidden geometry.
        """
        import torch
        arr = np.asarray(crop_rgb)
        if arr.ndim == 2:
            arr = np.stack([arr] * 3, axis=-1)
        if arr.shape[2] > 3:
            arr = arr[:, :, :3]
        side = self.input_size
        if arr.shape[0] != side or arr.shape[1] != side:
            from crop_context import resize_pad_square
            arr = resize_pad_square(arr, side)
            if arr is None:
                return None
        x = arr.astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        x = (x - mean) / std
        t = torch.from_numpy(x.transpose(2, 0, 1)).unsqueeze(0).to(self._device)
        return t

    def embed(self, crop_rgb):
        """-> a unit-norm CLS descriptor, or None. Never a zero vector on failure.

        Returning None rather than zeros matters: a zero vector would compare as a real
        descriptor and produce a similarity, which is the failure mode this whole redesign
        exists to remove. A missing descriptor must be an ABSENCE, so the channel abstains.
        """
        if crop_rgb is None or crop_rgb.size == 0:
            return None
        if min(crop_rgb.shape[:2]) < _PATCH:
            # Smaller than one patch: there is no signal to encode. GA-17's lesson -- do not
            # embed a placeholder, report that there is nothing to embed.
            return None
        self._load()
        import torch

        with torch.no_grad():
            pv = self._pixel_values(crop_rgb)
            if pv is None:
                return None
            out = self._model(pixel_values=pv)
            cls = out.last_hidden_state[:, 0, :].squeeze(0).float().cpu().numpy()
        n = np.linalg.norm(cls)
        return cls / n if n > 1e-9 else None


# ---------------------------------------------------------------------------------------
# The per-object descriptor set, capped by BEARING DIVERSITY rather than recency
# ---------------------------------------------------------------------------------------


class ViewBank:
    """A bounded set of view-tagged descriptors for one object.

    CAPPING BY DIVERSITY, NOT RECENCY, and the difference is the whole point. Keeping the N
    most recent views of an object the robot has been staring at gives N nearly identical
    descriptors from one direction -- a set that looks rich and answers one question. Keeping
    the N most bearing-DIVERSE views maximises the chance that some later object has a view
    inside the comparability cone of one of them, which is what decides whether the channel
    can answer at all.

    Selection is greedy farthest-point on the bearing sphere: repeatedly keep the view whose
    bearing is furthest from everything kept so far. Deterministic, no tuning.
    """

    __slots__ = ("capacity", "items")

    def __init__(self, capacity=8):
        self.capacity = int(capacity)
        self.items = []  # list of (bearing unit vector, descriptor vector, kind, frame_id)

    def add(self, bearing, vector, kind="visual", frame_id=None):
        if vector is None or bearing is None:
            return self
        b = np.asarray(bearing, dtype=float)
        n = np.linalg.norm(b)
        if n < 1e-9:
            return self
        self.items.append((b / n, np.asarray(vector, dtype=float), kind, frame_id))
        if len(self.items) > self.capacity:
            self.items = self._diverse_subset(self.items, self.capacity)
        return self

    @staticmethod
    def _diverse_subset(items, k):
        if len(items) <= k:
            return list(items)
        # Seed with the pair that are furthest apart, then greedily add the item whose
        # minimum angular distance to the kept set is largest.
        bearings = np.array([it[0] for it in items])
        sim = bearings @ bearings.T
        i, j = np.unravel_index(np.argmin(sim), sim.shape)
        kept = [int(i), int(j)]
        while len(kept) < k:
            rest = [t for t in range(len(items)) if t not in kept]
            if not rest:
                break
            # angular distance to the nearest kept bearing; take the largest
            best = max(rest, key=lambda t: -max(sim[t][u] for u in kept))
            kept.append(best)
        return [items[t] for t in sorted(kept)]

    def descriptors(self, kind=None):
        return [(b, v, k, f) for (b, v, k, f) in self.items if kind is None or k == kind]

    def spread(self, kind=None):
        """The object's OWN descriptor spread — the measured scale for comparison.

        Category (b): measured, not chosen. With fewer than two descriptors there is no
        spread to measure and this returns None, which makes the channel abstain rather
        than fall back on an invented sigma.
        """
        vs = [v for (_b, v, k, _f) in self.items if kind is None or k == kind]
        if len(vs) < 2:
            return None
        arr = np.array(vs)
        if arr.ndim != 2:
            return None
        return float(np.mean(np.std(arr, axis=0)))


def build_view_descriptors(bank, kind=None):
    """-> list of association.ViewDescriptor, ready for the appearance channel."""
    from association import ViewDescriptor
    return [ViewDescriptor(b, v, kind=k) for (b, v, k, _f) in bank.descriptors(kind)]


# ---------------------------------------------------------------------------------------
# Self-check — everything here except the embedder
# ---------------------------------------------------------------------------------------


def _img(h, w, colour):
    a = np.zeros((h, w, 3), dtype=np.uint8)
    a[:, :] = colour
    return a


def demo():
    # --- masked padded crop ---------------------------------------------------------------
    img = _img(100, 100, (30, 30, 30))
    img[40:60, 40:60] = (200, 20, 20)          # the object
    mask = np.zeros((100, 100), dtype=bool)
    mask[40:60, 40:60] = True

    from crop_context import CONTOUR
    crop = masked_padded_crop(img, mask, (40, 40, 60, 60), size=224, construction=CONTOUR)
    assert crop is not None and crop.shape == (224, 224, 3), crop.shape
    # The background is DIMMED and still legible -- not flattened, not blacked. More than a
    # couple of distinct colours must survive, or the context is gone.
    assert len(np.unique(crop.reshape(-1, 3), axis=0)) > 3, "context flattened away"
    print(f"  shared construction : -> {crop.shape[1]}x{crop.shape[0]}, "
          f"{len(np.unique(crop.reshape(-1, 3), axis=0))} distinct colours retained "
          f"(the describer gets this same image)")

    empty = np.zeros((100, 100), dtype=bool)
    assert masked_padded_crop(img, empty, None, construction=CONTOUR) is None
    print("  unusable region -> None (never a blank image, never a zero vector)")

    # --- mask-sampled colour ---------------------------------------------------------------
    feat = mask_sampled_appearance(img, mask, (40, 40, 60, 60))
    assert feat is not None and feat.shape == (6,)
    assert feat[0] > 0.7 and feat[1] < 0.15, feat        # red object, sampled inside the mask
    assert np.allclose(feat[3:], 0, atol=1e-6), "uniform patch must have ~zero std"
    print(f"  mask-sampled colour : mean RGB {np.round(feat[:3], 3)} — the object, not the wall")

    # --- diversity capping ------------------------------------------------------------------
    bank = ViewBank(capacity=4)
    # 12 views: 10 crowded around +x, 2 genuinely elsewhere
    for i in range(10):
        bank.add([1.0, 0.02 * i, 0.0], np.array([1.0, 0.0]), frame_id=i)
    bank.add([-1.0, 0.0, 0.0], np.array([0.9, 0.1]), frame_id=90)
    bank.add([0.0, 1.0, 0.0], np.array([0.8, 0.2]), frame_id=91)
    kept = bank.descriptors()
    assert len(kept) == 4, len(kept)
    frames = {f for (_b, _v, _k, f) in kept}
    assert 90 in frames and 91 in frames, f"diversity cap dropped the distinct views: {frames}"
    print("  diversity cap : 12 views -> 4 kept, and BOTH distinct bearings survived "
          "(recency would have kept 4 near-identical ones)")

    # --- spread is measured, and absent when it cannot be -----------------------------------
    assert ViewBank(4).add([1, 0, 0], np.array([1.0, 0.0])).spread() is None, \
        "one descriptor has no measurable spread"
    assert bank.spread() is not None
    print("  spread : measured from the object's own views; None with fewer than two")

    # --- the abstention contract, end to end through the real channel ------------------------
    import association as A
    a_desc = [A.ViewDescriptor([1, 0, 0], np.array([1.0, 0.0]), kind="visual")]
    b_far = [A.ViewDescriptor([-1, 0, 0], np.array([1.0, 0.0]), kind="visual")]
    r = A.channel_appearance(a_desc, b_far, math.radians(45), 0.05, 0.05)
    assert isinstance(r, A.Abstain), r
    print(f"  opposite bearings, IDENTICAL descriptors -> {r} (never 'similar')")

    b_near = [A.ViewDescriptor([1, 0.1, 0], np.array([1.0, 0.0]), kind="visual")]
    r2 = A.channel_appearance(a_desc, b_near, math.radians(45), 0.05, 0.05)
    assert not isinstance(r2, A.Abstain) and r2[0] > 0, r2
    print(f"  same bearing, identical descriptors -> log-odds {r2[0]:+.2f} "
          f"over {r2[1]['comparable_pairs']} comparable pair(s)")

    r3 = A.channel_appearance(a_desc, b_near, math.radians(45), None, None)
    assert isinstance(r3, A.Abstain), r3
    print(f"  comparable views but NO measured spread -> {r3}")

    print(f"\n  model default: {DEFAULT_MODEL_ID} (ViT-S/14) — chosen on measured VRAM, "
          f"not the ViT-B default")
    print("  NOTE: the embedder is verified separately — it needs a torch env "
          "(/DATA/ASPIRE/.venv/bin/python); this self-check is the numpy logic only.")
    print("\nvisual_reid self-check OK")


if __name__ == "__main__":
    demo()
