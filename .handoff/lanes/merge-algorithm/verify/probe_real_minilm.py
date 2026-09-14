"""The smoke pair's attribute score with the REAL MiniLM (what pair_attribute_score computes in production).

Run: HF_HOME=/DATA/huggingface_cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
     /home/xps/miniconda3/envs/mlspaces_310/bin/python probe_real_minilm.py
Preamble copied from tools/sweep_c.py (webcolors stand-in).
"""
import os
import sys
import types

CSS = {'white': (255, 255, 255), 'blue': (0, 0, 255), 'grey': (128, 128, 128), 'gray': (128, 128, 128),
       'brown': (165, 42, 42), 'green': (0, 128, 0), 'black': (0, 0, 0), 'silver': (192, 192, 192),
       'gold': (255, 215, 0), 'beige': (245, 245, 220), 'teal': (0, 128, 128), 'red': (255, 0, 0)}


class _RGB:
    def __init__(self, r, g, b):
        self.red, self.green, self.blue = r, g, b


def _name_to_rgb(name):
    if name not in CSS:
        raise ValueError(name)
    return _RGB(*CSS[name])


sys.modules['webcolors'] = types.SimpleNamespace(name_to_rgb=_name_to_rgb)
PM = '/DATA/GRAPH-API/.claude/worktrees/found-merge-algorithm/lost3dsg/src/perception_module'
sys.path.insert(0, PM)
os.chdir(PM)
os.environ.setdefault('HF_HOME', '/DATA/huggingface_cache')
import association as A  # noqa: E402
from nlp_utils import get_embedding, lost_similarity_detailed, world2vec  # noqa: E402

assert world2vec.st_model is not None and type(world2vec.st_model).__module__.startswith('sentence_transformers'), \
    type(world2vec.st_model)
print("answering component:", type(world2vec.st_model).__module__, type(world2vec.st_model).__name__)

pairs = [
    ("smoke pair (two chair views)", "chair", "chair", "grey", "grey", "fabric", "fabric",
     "dark grey tufted armchair with nailhead trim", "dark grey armchair in the foreground"),
    ("smoke lamp/table", "lamp", "table", "blue", "red", "metal", "wood", "a lamp", "a table"),
    ("smoke cross-kind same attrs", "lamp", "table", "grey", "grey", "fabric", "fabric",
     "dark grey tufted armchair with nailhead trim", "dark grey armchair in the foreground"),
    ("chair vs armchair, same attrs", "chair", "armchair", "grey", "grey", "fabric", "fabric",
     "a grey fabric armchair", "a grey fabric chair with arms"),
    ("same kind, description unknown one side", "chair", "chair", "grey", "grey", "fabric", "fabric",
     "a grey fabric chair", "unknown"),
]
for name, l1, l2, c1, c2, m1, m2, d1, d2 in pairs:
    e1, e2 = get_embedding(world2vec, d1), get_embedding(world2vec, d2)
    score, ev = lost_similarity_detailed(world2vec, l1, l2, c1, c2, m1, m2, e1, e2)
    ch = A.channel_attributes(score, ev["optional_count"], 0.85, 2.0, same_kind=(l1 == l2))
    llr = ch[0] if not isinstance(ch, A.Abstain) else ch
    print(f"  {name:42s} score {score:.4f} optional_count {ev['optional_count']} -> attributes log-odds {llr}")
print("probe_real_minilm done")
