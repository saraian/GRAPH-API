"""Arm (b) margin on GA-493 with the REAL MiniLM: the 104 final objects carry no observations, so
covariance is None and separation abstains on EVERY pair -- the regime in which room + attributes
can commit with no positive geometry. How close do the non-intersecting offered pairs come to the
commit threshold, and which ones cross it? Preamble copied from verify/v3_real_model_replay.py.

Run: HF_HOME=/DATA/huggingface_cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
     /home/xps/miniconda3/envs/mlspaces_310/bin/python r15_gap_real_model.py
"""
import collections
import json
import os
import sys
import types

import numpy as np

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
from config import CFG  # noqa: E402
from nlp_utils import get_embedding, lost_similarity_detailed, world2vec  # noqa: E402

assert world2vec.st_model is not None, "real SentenceTransformer did not load"
print("component: nlp_utils.world2vec.st_model =", type(world2vec.st_model).__name__)
print("weights", CFG['similarity'], "attr cap", CFG['association'].get('merge_attribute_max_log_odds'))
R = '/DATA/GRAPH-API/.handoff/lanes/perception/ga493_replay_20260914/'
SIM_T = CFG['association']['sim_threshold']
THR = A.commit_threshold(20.0)
GAP = 0.3


def base(label):
    return str(label).split('#')[0].strip().lower()


def bnd(b):
    return A._as_bounds(b)


_emb = {}


def emb(text):
    if text not in _emb:
        _emb[text] = get_embedding(world2vec, text)
    return _emb[text]


def attr_score(a, b):
    s, ev = lost_similarity_detailed(world2vec, base(a['label']), base(b['label']), a['color'], b['color'],
                                     a['material'], b['material'], emb(a['description']), emb(b['description']))
    return s, int(ev['optional_count']), (base(a['label']) == base(b['label']) and base(a['label']) != '')


final = json.load(open(R + 'bundle/persistent_perception.json'))
objs = [A.AssocObject(o['object_id'], bbox=o['bbox'], room_id=o.get('room_id'), label=o['label'], source=o)
        for o in final]
boxes = [bnd(o['bbox']) for o in final]
xs = [b[0] for b in boxes] + [b[1] for b in boxes]
ys = [b[2] for b in boxes] + [b[3] for b in boxes]
zs = [b[4] for b in boxes] + [b[5] for b in boxes]
vol = max((max(xs) - min(xs)) * (max(ys) - min(ys)) * (max(zs) - min(zs)), 1.0)
rooms = {o.get('room_id') for o in final}
rooms.discard(None)
n_rooms = max(len(rooms), 1)
print(f"{len(final)} objects, map volume {vol:.1f} m3, n_rooms {n_rooms} (room channel +{np.log(n_rooms):.3f} same / "
      f"-{np.log(n_rooms):.3f} different), threshold {THR:.3f}")
ctx = A.AssocContext(map_volume_m3=vol, n_rooms=n_rooms, cost_ratio=20.0, use_ontology=False,
                     attribute_reference=SIM_T, attribute_max_log_odds=2.0, locality_gap_m=GAP,
                     overlap_2d_fn=A.shared_frame_overlap_2d,
                     attribute_score_fn=lambda x, y: attr_score(x.source, y.source))
offered, excluded = A.generate_candidates(objs, ctx)
print(f"offered {len(offered)} {collections.Counter(m.get('offered_by') for _, _, m in offered)}, excluded {len(excluded)}")
print("every offered pair: covariance None on both sides ->", all(a.covariance is None and b.covariance is None for a, b, _ in offered))

rows = []
dec = collections.Counter()
for a, b, meta in offered:
    oa, ob = a.source, b.source
    fa, fb = bnd(oa['fused_bbox']), bnd(ob['fused_bbox'])
    overlapping_f = A.intersection_volume(fa, fb) > 0
    ra, rb = oa.get('room_id'), ob.get('room_id')
    if ra is not None and rb is not None and ra != rb and not overlapping_f:
        dec['room'] += 1
        continue
    if A.geometry_compatible(fa, fb, GAP) is False:
        dec['geometry'] += 1
        continue
    h = A.Hypothesis((a.object_id, b.object_id))
    for f in (1, 2):
        ps = A.score_pair(a, b, ctx)
        h.update(ps, frame_id=f)
        d, why = h.decide(THR, min_evidence=1, min_consecutive=2)
    dec[d] += 1
    ma, mb = bnd(oa['bbox']), bnd(ob['bbox'])
    inter_m = A.intersection_volume(ma, mb) > 0
    ch = {k: round(v.get('log_odds', 0.0), 2) for k, v in ps.channels.items()}
    attrs = ps.channels.get('attributes', {})
    rows.append((d, round(ps.total, 3), inter_m, round(A.box_gap(ma, mb), 3), meta.get('offered_by'),
                 oa['label'], ob['label'], attrs.get('score'), attrs.get('same_kind'), ra == rb, ch,
                 'separation' in ps.abstentions))
print("decisions:", dict(dec))

non_inter = [r for r in rows if not r[2]]
print(f"\nNON-intersecting measured boxes among scored pairs: {len(non_inter)} of {len(rows)}; "
      f"merged {sum(1 for r in non_inter if r[0] == 'merge')}; separation abstained on all: "
      f"{all(r[11] for r in non_inter)}")
non_inter.sort(key=lambda r: -r[1])
print("top 12 by total (decision, total, inter, gap_m, offered_by, labels, attr score, same_kind, same_room, channels):")
for r in non_inter[:12]:
    print("   ", r[:11])
same_room_sk = [r for r in non_inter if r[9] and r[8]]
print(f"\nnon-intersecting, same room, same kind: {len(same_room_sk)}; "
      f"attr score max {max((r[7] or 0) for r in same_room_sk) if same_room_sk else None}; "
      f"totals within 0.5 of threshold: {sum(1 for r in same_room_sk if r[1] >= THR - 0.5)}")
need = SIM_T + (1 - SIM_T) * (THR - np.log(n_rooms) + 0.111) / 2.0
print(f"attribute score a same-room non-intersecting pair needs at giou -0.111 with n_rooms={n_rooms}: {need:.3f}")
print("attr score distribution over same-kind same-room non-intersecting pairs:",
      sorted(round(r[7], 3) for r in same_room_sk if r[7] is not None)[-10:])
