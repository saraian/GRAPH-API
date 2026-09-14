"""Probe: bundle shape. Counts adds / updates, observation coverage, cycle ids, modes, unknown attributes."""
import collections
import json

R = '/DATA/GRAPH-API/.handoff/lanes/perception/ga493_replay_20260914/'
obs = {}
n_events = 0
for line in open(R + 'capture/consumer/events.jsonl'):
    e = json.loads(line)
    if e['kind'] != 'consumer_pair':
        continue
    n_events += 1
    p = e['payload']
    descs = {(d.get('observation') or {}).get('observation_id'): d for d in p['descriptions']['descriptions']}
    for b in p['bboxes']['boxes']:
        oid = (b.get('observation') or {}).get('observation_id')
        d = descs.get(oid, {})
        obs[oid] = {'label': b['label'], 'expl': p.get('exploration_mode'), 'moved': p.get('robot_has_moved'),
                    'color': d.get('color'), 'material': d.get('material'), 'description': d.get('description'),
                    'cycle': e.get('cycle_id'), 'has_desc': oid in descs}
print("consumer_pair events:", n_events, "detections with box:", len(obs))
print("detections without paired description:", sum(1 for o in obs.values() if not o['has_desc']))
print("exploration_mode values:", collections.Counter(o['expl'] for o in obs.values()))
print("robot_has_moved values:", collections.Counter(o['moved'] for o in obs.values()))
print("cycles:", len({o['cycle'] for o in obs.values()}), "None cycle:", sum(1 for o in obs.values() if o['cycle'] is None))
print("colour unknown/empty:", sum(1 for o in obs.values() if not o['color'] or str(o['color']).lower() in ('unknown', 'none', 'n/a')),
      "material unknown/empty:", sum(1 for o in obs.values() if not o['material'] or str(o['material']).lower() in ('unknown', 'none', 'n/a')),
      "description unknown/empty:", sum(1 for o in obs.values() if not o['description'] or str(o['description']).lower() in ('unknown', 'none', 'n/a')))
print("colour words:", collections.Counter(str(o['color']).lower() for o in obs.values()).most_common(30))
recs = [json.loads(line) for line in open(R + 'bundle/mutation_receipts.jsonl')]
print("receipts by (op, state):", collections.Counter((r['operation'], r['mutation_state']) for r in recs))
applied = [r for r in recs if r['mutation_state'] == 'applied_in_memory']
print("applied adds with obs in capture:", sum(1 for r in applied if r['operation'] == 'add' and (r.get('observation') or {}).get('observation_id') in obs),
      "applied updates with obs in capture:", sum(1 for r in applied if r['operation'] == 'update' and (r.get('observation') or {}).get('observation_id') in obs))
print("receipt keys:", sorted(recs[0].keys()))
upd = next((r for r in applied if r['operation'] == 'update'), None)
print("update example:", json.dumps(upd)[:600])
hooks = [json.loads(line) for line in open(R + 'bundle/hook_decisions.jsonl')]
print("hook kinds:", collections.Counter(h['kind'] for h in hooks))
final = json.load(open(R + 'bundle/persistent_perception.json'))
print("final objects:", len(final), "with fused_bbox:", sum(1 for o in final if o.get('fused_bbox')), "keys:", sorted(final[0].keys()))
