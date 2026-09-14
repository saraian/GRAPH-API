"""Dump raw records for same-label pairs refused on similarity whose FINAL fused boxes overlap
at IoU >= 0.7. Verifies the labeller's claim against the artefact itself."""
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


seen = set()
for r in rows:
    if r['kind'] != 'merge_refused' or r['reason'] != 'similarity':
        continue
    p = tuple(sorted((r['object'], r['candidate'])))
    if p in seen:
        continue
    oa, ob = final.get(p[0]), final.get(p[1])
    if not oa or not ob:
        continue
    if oa['label'].split('#')[0] != ob['label'].split('#')[0]:
        continue
    fa = np.array([oa['fused_bbox'][k] for k in K])
    fb = np.array([ob['fused_bbox'][k] for k in K])
    ma = np.array([oa['bbox'][k] for k in K])
    mb = np.array([ob['bbox'][k] for k in K])
    if iou(fa, fb) < 0.7:
        continue
    seen.add(p)
    hist = [x for x in rows if x['kind'] == 'merge_refused'
            and tuple(sorted((x['object'], x['candidate']))) == p]
    print("=" * 100)
    print(f"PAIR {p[0][:16]} / {p[1][:16]}  fusedIoU={iou(fa, fb):.3f} measuredIoU={iou(ma, mb):.3f}")
    for o in (oa, ob):
        print(f"  {o['label']:12s} color={o['color']!r:12s} material={o['material']!r:12s} "
              f"views={o['bbox_fusion']['view_count']} created={o['creation_time']:.1f}")
        print(f"     desc={o['description']!r}")
        print(f"     bbox     ={[round(o['bbox'][k], 2) for k in K]}")
        print(f"     fused    ={[round(o['fused_bbox'][k], 2) for k in K]}")
    print(f"  refusal history ({len(hist)} records): " + "; ".join(
        f"{h['reason']}@{None if h.get('similarity') is None else round(h['similarity'], 3)} "
        f"ev={h.get('evidence_count')}" for h in hist))
    if len(seen) >= 5:
        break
