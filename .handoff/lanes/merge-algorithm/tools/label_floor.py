"""Measure the label-term similarity (semantic_similarity, real MiniLM) between every pair of
base labels seen in the GA-493 run, to place `merge_attribute_label_floor`: synonyms of one kind
(cushion/pillow, couch/sofa) must sit ABOVE it, different kinds at one spot (pillow/bed,
lamp/table, plant/table) BELOW it."""
import collections
import itertools
import json
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
from nlp_utils import semantic_similarity, world2vec  # noqa: E402

assert world2vec.st_model is not None
R = '/DATA/GRAPH-API/.handoff/lanes/perception/ga493_replay_20260914/'
labels = collections.Counter()
for line in open(R + 'capture/consumer/events.jsonl'):
    e = json.loads(line)
    if e['kind'] != 'consumer_pair':
        continue
    for b in e['payload']['bboxes']['boxes']:
        labels[b['label'].split('#')[0]] += 1
names = sorted(labels)
print(f"{len(names)} base labels: {names}")
sims = {}
for a, b in itertools.combinations(names, 2):
    sims[(a, b)] = semantic_similarity(world2vec, a, b)
ranked = sorted(sims.items(), key=lambda kv: -kv[1])
print("\ntop 25 most similar label pairs:")
for (a, b), s in ranked[:25]:
    print(f"  {s:.3f}  {a} / {b}")
probes = [('pillow', 'cushion'), ('sofa', 'couch'), ('countertop', 'counter'), ('chair', 'armchair'),
          ('pillow', 'bed'), ('pillow', 'sofa'), ('lamp', 'table'), ('plant', 'table'), ('rug', 'floor'),
          ('cabinet', 'countertop'), ('stove', 'oven'), ('painting', 'picture'), ('statue', 'sculpture'),
          ('rock', 'statue'), ('bottle', 'dispenser'), ('mirror', 'painting'), ('bench', 'sofa')]
print("\nprobe pairs (same kind should be high, different kind low):")
for a, b in probes:
    print(f"  {semantic_similarity(world2vec, a, b):.3f}  {a} / {b}")
