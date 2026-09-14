"""Room-gate refusals: which room pairs, and how many involve overlapping same-label boxes."""
import collections
import json

import numpy as np

R = '/DATA/GRAPH-API/.handoff/lanes/perception/ga493_replay_20260914/'
final = {o['object_id']: o for o in json.load(open(R + 'bundle/persistent_perception.json'))}
rows = [json.loads(line) for line in open(R + 'bundle/hook_decisions.jsonl')]
K = ['x_min', 'y_min', 'z_min', 'x_max', 'y_max', 'z_max']


def iou(a, b):
    lo = np.maximum(a[:3], b[:3])
    hi = np.minimum(a[3:], b[3:])
    d = np.clip(hi - lo, 0, None)
    i = d.prod()
    va = (a[3:] - a[:3]).prod()
    vb = (b[3:] - b[:3]).prod()
    return i / (va + vb - i) if va + vb - i > 0 else 0.0


last = {}
for r in rows:
    if r['kind'] == 'merge_refused' and r['reason'] == 'room':
        last[tuple(sorted((r['object'], r['candidate'])))] = r
rooms = collections.Counter()
above_gate = 0
overlap_above_gate = []
for p, r in last.items():
    rooms[(r['room_a'], r['room_b'])] += 1
    oa, ob = final.get(p[0]), final.get(p[1])
    if not oa or not ob:
        continue
    fa = np.array([oa['fused_bbox'][k] for k in K])
    fb = np.array([ob['fused_bbox'][k] for k in K])
    ma = np.array([oa['bbox'][k] for k in K])
    mb = np.array([ob['bbox'][k] for k in K])
    same = oa['label'].split('#')[0] == ob['label'].split('#')[0]
    if r['similarity'] is not None and r['similarity'] >= 0.925:
        above_gate += 1
        if same and iou(ma, mb) > 0:
            overlap_above_gate.append((oa['label'], ob['label'], round(iou(ma, mb), 3), round(iou(fa, fb), 3),
                                       round(r['similarity'], 3), r['room_a'], r['room_b']))
print(f"distinct room-refused pairs: {len(last)}; room pairs: {dict(rooms)}")
print(f"room-refused pairs with similarity >= 0.925 (would otherwise have MERGED): {above_gate}")
print("  ...of which same-label with overlapping MEASURED boxes (label, label, measIoU, fusedIoU, sim, room_a, room_b):")
for e in overlap_above_gate:
    print("   ", e)
# how are rooms laid out? centre of each final object per room
by_room = collections.defaultdict(list)
for o in final.values():
    by_room[o.get('room_id')].append(o['label'])
print("final objects per room_id:", {k: len(v) for k, v in by_room.items()})
