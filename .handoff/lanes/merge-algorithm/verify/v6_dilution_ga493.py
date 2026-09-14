"""GA-493 cross-base-label fused-intersecting offered pairs (the v4 section 1 loop), real MiniLM, with the room
count swept: which REAL pairs cross the guard's dilution edge (overlap < 0.5*total) and would MERGE. Also, at the
run's own room count, how far each guard-held pair sits from that edge in attribute-score units."""
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

assert world2vec.st_model is not None
print("component:", type(world2vec.st_model).__name__, "weights", CFG['similarity'])
R = '/DATA/GRAPH-API/.handoff/lanes/perception/ga493_replay_20260914/'
SIM_T = CFG['association']['sim_threshold']
THR = A.commit_threshold(20.0)
CAP = 2.0


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
print(f"map hull {vol:.1f} m3, rooms in run {len(rooms)}, objects {len(final)}")


def run(n_rooms):
    ctx = A.AssocContext(map_volume_m3=vol, n_rooms=n_rooms, cost_ratio=20.0, use_ontology=False,
                         attribute_reference=SIM_T, attribute_max_log_odds=CAP, locality_gap_m=0.3,
                         overlap_2d_fn=A.shared_frame_overlap_2d,
                         attribute_score_fn=lambda x, y: attr_score(x.source, y.source))
    offered, _ = A.generate_candidates(objs, ctx)
    rows = []
    for a, b, meta in offered:
        oa, ob = a.source, b.source
        if base(oa['label']) == base(ob['label']):
            continue
        fa, fb = bnd(oa['fused_bbox']), bnd(ob['fused_bbox'])
        if A.intersection_volume(fa, fb) <= 0:
            continue
        h = A.Hypothesis((a.object_id, b.object_id))
        d = None
        for f in (1, 2):
            ps = A.score_pair(a, b, ctx)
            h.update(ps, frame_id=f)
            d, why = h.decide(THR, min_evidence=1, min_consecutive=2)
        at = ps.channels.get('attributes', {})
        rows.append(dict(a=oa['label'], b=ob['label'], score=at.get('score'), attrs=at.get('log_odds', 0.0),
                         ov=ps.channels.get('overlap', {}).get('log_odds', 0.0),
                         room=ps.channels.get('room', {}).get('log_odds'), total=ps.total,
                         guard=ps.containment_unchecked, decision=d,
                         covis_uncollected=('covisibility' not in ps.channels
                                            and 'covisibility' not in ps._measured_abstentions)))
    return rows


n_run = len(rooms)
for n in (n_run, 5, 6, 8):
    rows = run(n)
    merges = [r for r in rows if r['decision'] == 'merge']
    held = [r for r in rows if r['guard'] and r['total'] >= THR]
    print(f"\nn_rooms={n} (same-room channel +{np.log(n):.3f}): cross-kind fused-intersecting offered pairs "
          f"n={len(rows)}; co-visibility uncollected on {sum(r['covis_uncollected'] for r in rows)}; "
          f"MERGE {len(merges)}; held only by the guard {len(held)}")
    for r in merges:
        print(f"   MERGE {r['a']}/{r['b']} score {r['score']:.3f} attrs {r['attrs']:+.2f} room {r['room']:+.2f} "
              f"overlap {r['ov']:.2f} total {r['total']:.2f}  (overlap < 0.5*total: {r['ov'] < 0.5 * r['total']})")
    if n == n_run:
        print("   guard-held pairs at the run's own room count, distance to the dilution edge")
        print("   (guard is off when attrs + room > overlap; appearance abstains, so nothing else is positive):")
        for r in sorted(held, key=lambda r: r['ov'] - r['attrs'] - (r['room'] or 0)):
            need_attrs = r['ov'] - (r['room'] or 0.0)
            need_score = SIM_T + need_attrs * (1 - SIM_T) / CAP if need_attrs > 0 else None
            reach = need_score is not None and need_score <= 1.0
            ns = 'n/a' if need_score is None else f'{need_score:.3f}'
            print(f"    {r['a']}/{r['b']}: score {r['score']:.3f}, overlap {r['ov']:.2f}, attrs {r['attrs']:+.2f}, "
                  f"room {(r['room'] or 0):+.2f}, total {r['total']:.2f} -> flips at attrs {need_attrs:+.2f} = "
                  f"score {ns}{'  REACHABLE' if reach else ''}")
