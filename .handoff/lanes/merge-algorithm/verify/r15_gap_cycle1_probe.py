"""GA-493: perception_2 publishes the description array BEFORE the agent pose in each cycle, so
objects admitted in the FIRST cycle may have been created with latest_agent_pose None and hence no
sighting. How many objects were admitted in the first cycle, and how many of those survived to the
final map without a later update/match (a proxy for 'seen once')?"""
import collections
import json

R = '/DATA/GRAPH-API/.handoff/lanes/perception/ga493_replay_20260914/'
first_cycle = None
cycles = []
for line in open(R + 'capture/consumer/events.jsonl'):
    e = json.loads(line)
    if e['kind'] == 'consumer_pair':
        cycles.append(e['cycle_id'])
        if first_cycle is None:
            first_cycle = e['cycle_id']
print('consumer_pair cycles', len(cycles), 'first', first_cycle)

adm_by_cycle = collections.Counter()
links = {}
for line in open(R + 'bundle/hook_decisions.jsonl'):
    r = json.loads(line)
    if r.get('kind') == 'admission':
        adm_by_cycle[r.get('cycle_id')] += 1
    if r.get('kind') == 'link':
        links[r.get('object')] = r
print('admissions per cycle (first 5 cycles):', [(c, adm_by_cycle.get(c, 0)) for c in cycles[:5]])

ops = collections.Counter()
per_obj = collections.defaultdict(list)
for line in open(R + 'bundle/mutation_receipts.jsonl'):
    r = json.loads(line)
    ops[(r.get('operation'), r.get('mutation_state'))] += 1
    per_obj[r.get('object_id')].append((r.get('operation'), r.get('cycle_id')))
print('mutation ops', dict(ops))
final = json.load(open(R + 'bundle/persistent_perception.json'))
final_ids = {o['object_id'] for o in final}
added_first = [oid for oid, lst in per_obj.items() if any(op == 'add' and cyc == first_cycle for op, cyc in lst)]
print('objects ADDED in the first cycle', len(added_first), 'of', len(per_obj), 'objects with receipts;',
      'of those in the final map', sum(1 for o in added_first if o in final_ids),
      '; with a later update receipt', sum(1 for o in added_first if any(op == 'update' for op, _ in per_obj[o])))
# perception_latencies may say when the pose arrived relative to descriptions
try:
    n = 0
    for line in open(R + 'bundle/perception_latencies.jsonl'):
        n += 1
        if n <= 2:
            print('latency row', line[:300].rstrip())
    print('latency rows', n)
except FileNotFoundError:
    pass
