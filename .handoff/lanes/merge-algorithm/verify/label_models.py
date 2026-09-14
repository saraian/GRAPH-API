"""Which label-similarity instrument separates SAME-kind label pairs from DIFFERENT-kind ones?
Owner 2026-09-14: "the label weight should be well higher but the embedding model should be more
accurate". Candidates that run OFFLINE on this host (no download): MiniLM raw (what production
uses), MiniLM with a sentence template, intfloat/e5-small-v2 (cached), and the OWLv2 text tower
(google/owlv2-base-patch16-ensemble, cached) -- the open-vocabulary detector's own query space.
Probe pairs are the run's real labels; the split is a human judgement stated in the lists."""
import itertools
import os
import sys

import numpy as np
import torch

os.environ.setdefault('HF_HOME', '/DATA/huggingface_cache')
os.environ.setdefault('HF_HUB_OFFLINE', '1')
os.environ.setdefault('TRANSFORMERS_OFFLINE', '1')
CACHE = '/DATA/huggingface_cache'

SAME = [('pillow', 'cushion'), ('sofa', 'couch'), ('countertop', 'counter'), ('chair', 'armchair'),
        ('painting', 'picture'), ('statue', 'sculpture'), ('decor', 'decoration'), ('artwork', 'painting'),
        ('dispenser', 'paper_towel_holder'), ('rug', 'carpet'), ('lamp', 'light'), ('table', 'desk'),
        ('cabinet', 'cupboard'), ('sink', 'basin'), ('tray', 'platter'), ('box', 'crate')]
DIFF = [('pillow', 'bed'), ('pillow', 'sofa'), ('lamp', 'table'), ('plant', 'table'), ('rug', 'floor'),
        ('cabinet', 'countertop'), ('rock', 'statue'), ('mirror', 'painting'), ('bench', 'sofa'),
        ('cushion', 'sofa'), ('chair', 'sofa'), ('seat', 'pillow'), ('cushion', 'seat'), ('stove', 'oven'),
        ('bench', 'cabinet'), ('table', 'object'), ('plant', 'flowers'), ('sink', 'countertop'),
        ('window', 'mirror'), ('towel', 'rug'), ('book', 'box'), ('knob', 'latch')]


def cos(a, b):
    a = a / (np.linalg.norm(a) + 1e-9)
    b = b / (np.linalg.norm(b) + 1e-9)
    return float(a @ b)


def separation(name, sim):
    s_same = np.array([sim(a, b) for a, b in SAME])
    s_diff = np.array([sim(a, b) for a, b in DIFF])
    # AUC by pairwise comparison, and the best single threshold accuracy
    auc = float(np.mean([[1.0 if x > y else 0.5 if x == y else 0.0 for y in s_diff] for x in s_same]))
    best = max(((t, (np.mean(s_same >= t) + np.mean(s_diff < t)) / 2) for t in np.arange(0.0, 1.0, 0.01)),
               key=lambda p: p[1])
    print(f"{name:28s} AUC {auc:.3f}  best thr {best[0]:.2f} acc {best[1]:.3f}  "
          f"same median {np.median(s_same):.3f} (min {s_same.min():.3f})  diff median {np.median(s_diff):.3f} "
          f"(max {s_diff.max():.3f})")
    worst_same = sorted(zip(s_same, SAME))[:3]
    worst_diff = sorted(zip(s_diff, DIFF), reverse=True)[:3]
    print(f"      lowest same-kind: {[(round(float(s), 2), p) for s, p in worst_same]}")
    print(f"      highest diff-kind: {[(round(float(s), 2), p) for s, p in worst_diff]}")


def norm_label(w):
    return w.replace('_', ' ')


# 1. MiniLM raw (production semantic_similarity), and templated
from sentence_transformers import SentenceTransformer  # noqa: E402
mini = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2', cache_folder=CACHE)
cache = {}


def enc_mini(text):
    if text not in cache:
        cache[text] = mini.encode(text)
    return cache[text]


separation('MiniLM raw word', lambda a, b: cos(enc_mini(norm_label(a)), enc_mini(norm_label(b))))
for tpl in ('a photo of a {}', 'a {} in a home', 'the object is a {}'):
    separation(f'MiniLM "{tpl}"', lambda a, b, t=tpl: cos(enc_mini(t.format(norm_label(a))), enc_mini(t.format(norm_label(b)))))

# 2. e5-small-v2 (cached) with its query prefix
try:
    e5 = SentenceTransformer('intfloat/e5-small-v2', cache_folder=CACHE)
    c5 = {}

    def enc_e5(text):
        if text not in c5:
            c5[text] = e5.encode('query: ' + text)
        return c5[text]
    separation('e5-small-v2 word', lambda a, b: cos(enc_e5(norm_label(a)), enc_e5(norm_label(b))))
    separation('e5-small-v2 "a photo of a {}"', lambda a, b: cos(enc_e5('a photo of a ' + norm_label(a)), enc_e5('a photo of a ' + norm_label(b))))
except Exception as exc:
    print('e5-small-v2 unavailable:', type(exc).__name__, str(exc)[:120])

# 3. OWLv2 text tower: the open-vocabulary detector's query space
try:
    from transformers import Owlv2Model, Owlv2Processor
    proc = Owlv2Processor.from_pretrained('google/owlv2-base-patch16-ensemble')
    owl = Owlv2Model.from_pretrained('google/owlv2-base-patch16-ensemble').eval()
    c_owl = {}

    def enc_owl(text):
        if text not in c_owl:
            with torch.no_grad():
                inp = proc(text=[text], return_tensors='pt')
                c_owl[text] = owl.get_text_features(**inp)[0].numpy()
        return c_owl[text]
    separation('OWLv2 text "a photo of a {}"', lambda a, b: cos(enc_owl('a photo of a ' + norm_label(a)), enc_owl('a photo of a ' + norm_label(b))))
    separation('OWLv2 text word', lambda a, b: cos(enc_owl(norm_label(a)), enc_owl(norm_label(b))))
except Exception as exc:
    print('OWLv2 unavailable:', type(exc).__name__, str(exc)[:160])

# 4. how many of the run's 40 labels collide at the best threshold of each candidate is left to the
#    arms script; here only the separation of the judged pairs.
print(f"\nprobe pairs: {len(SAME)} same-kind, {len(DIFF)} different-kind; {len(list(itertools.chain(SAME, DIFF)))} total")
sys.exit(0)
