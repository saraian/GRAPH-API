"""Shape of the GA-493 bundle artefacts this verification reads. Read-only."""
import collections
import json

R = '/DATA/GRAPH-API/.handoff/lanes/perception/ga493_replay_20260914/'
d = json.load(open(R + 'bundle/persistent_perception.json'))
objs = d if isinstance(d, list) else d.get('objects', d)
print('persistent_perception.json:', type(d).__name__, len(objs))
o = objs[0]
print('keys:', sorted(o.keys()))
for k in ('label', 'object_id', 'bbox', 'fused_bbox', 'color', 'material', 'room_id',
          'creation_time', 'geometry_epoch', 'object_revision'):
    print(' ', k, '=', o.get(k))
print('  description =', (o.get('description') or '')[:80])
obs = o.get('observations')
print('  observations type', type(obs).__name__, 'len', len(obs) if obs else obs)
if obs:
    print('  observation[0] =', obs[0])
print('has fused_bbox:', collections.Counter(bool(x.get('fused_bbox')) for x in objs))
print('n observations:', collections.Counter(len(x.get('observations') or []) for x in objs).most_common(10))
print('labels:', collections.Counter(str(x['label']).split('#')[0] for x in objs).most_common(40))

# consumer events
n = 0
kinds = collections.Counter()
sample = None
for line in open(R + 'capture/consumer/events.jsonl'):
    e = json.loads(line)
    kinds[e['kind']] += 1
    if e['kind'] == 'consumer_pair' and sample is None:
        sample = e
print('event kinds:', kinds)
p = sample['payload']
print('consumer_pair payload keys:', sorted(p.keys()))
print('cycle_id:', sample.get('cycle_id'), 'exploration_mode:', p.get('exploration_mode'),
      'robot_has_moved:', p.get('robot_has_moved'))
b = p['bboxes']['boxes'][0]
print('box keys:', sorted(b.keys()))
print('box sample:', {k: b[k] for k in b if k not in ('fusion_voxel_keys',)})
dd = p['descriptions']['descriptions'][0]
print('desc keys:', sorted(dd.keys()))
