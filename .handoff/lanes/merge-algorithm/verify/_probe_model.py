"""Probe: does the REAL MiniLM load offline under mlspaces_310, and does the real scorer read the new weights."""
import os
import sys
import types

CSS = {'white': (255, 255, 255)}


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
from config import CFG  # noqa: E402
from nlp_utils import get_embedding, lost_similarity_detailed, world2vec  # noqa: E402

print("st_model:", type(world2vec.st_model).__name__, "w2v:", world2vec.w2v_model)
print("CFG similarity:", CFG["similarity"], "sim_threshold:", CFG["association"]["sim_threshold"])
e1 = get_embedding(world2vec, "a wooden chair with a blue cushion")
e2 = get_embedding(world2vec, "a blue chair made of wood")
print("emb len", None if e1 is None else len(e1))
print(lost_similarity_detailed(world2vec, "chair", "chair", "blue", "blue", "wood", "wood", e1, e2))
