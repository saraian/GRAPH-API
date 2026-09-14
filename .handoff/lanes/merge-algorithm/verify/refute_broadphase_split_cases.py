"""The two same-label misses from refute_broadphase_split.py Part A, in detail: which
same-label objects existed at that cycle (creation_time vs the detection's time), and which
of them the broad phase (wm.candidates on the measured box) still offered."""
import json
import sys

PM = '/DATA/GRAPH-API/.claude/worktrees/found-merge-algorithm/lost3dsg/src/perception_module'
sys.path.insert(0, PM)
import rosstub  # noqa: E402

rosstub.install()
import association as A  # noqa: E402
import object_info  # noqa: E402
import object_manager_6 as OM6  # noqa: E402
from world_model import wm  # noqa: E402

R = '/DATA/GRAPH-API/.handoff/lanes/perception/ga493_replay_20260914/'
GAP = OM6.ASSOCIATION_MARGIN_M


def base(label):
    return str(label).split('#')[0].strip().lower()


final = json.load(open(R + 'bundle/persistent_perception.json'))
wm.persistent_perceptions.clear()
objs = []
for o in final:
    ob = object_info.Object(o['label'], None, dict(o['bbox']))
    ob.object_id = o['object_id']
    ob.fused_bbox = dict(o['fused_bbox'])
    ob.creation_time = o.get('creation_time')
    ob.view_count = (o.get('fused_bbox') or {}).get('view_count')
    objs.append(ob)
    wm.persistent_perceptions.append(ob)

for line in open(R + 'capture/consumer/events.jsonl'):
    e = json.loads(line)
    if e.get('kind') != 'consumer_pair':
        continue
    if not e['cycle_id'].startswith(('347dc7', 'ccd5b6')):
        continue
    t_ev = e.get('recorded_at')
    for d in e['payload']['bboxes']['boxes']:
        if base(d['label']) not in ('rug', 'cushion'):
            continue
        det = {k: d[k] for k in ('x_min', 'x_max', 'y_min', 'y_max', 'z_min', 'z_max')}
        broad = set(id(c) for c in OM6._association_candidates(det))
        same = [ob for ob in objs if base(ob.label) == base(d['label'])]
        print(f"cycle {e['cycle_id'][:6]} t={t_ev} det {d['label']} box="
              f"{[round(det[k], 2) for k in ('x_min', 'x_max', 'y_min', 'y_max', 'z_min', 'z_max')]}")
        for ob in same:
            dd, m, f = A._as_bounds(det), A._as_bounds(ob.bbox), A._as_bounds(ob.fused_bbox)
            existed = (ob.creation_time is not None and t_ev is not None
                       and float(ob.creation_time) <= float(t_ev))
            print(f"   {ob.label:12s} id={ob.object_id[-6:]} created={ob.creation_time} "
                  f"existed_at_cycle={existed} views={ob.view_count} "
                  f"gate={OM6.locality_ok(det, ob, GAP)} broad={id(ob) in broad} "
                  f"gap_fused={A.box_gap(dd, f):.2f} gap_measured={A.box_gap(dd, m):.2f}")
wm.persistent_perceptions.clear()
