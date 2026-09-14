"""GA-493 capture: did agent poses arrive before the first object was admitted (so every object
recorded a sighting on its creation cycle)?"""
import collections
import json

E = '/DATA/GRAPH-API/.handoff/lanes/perception/ga493_replay_20260914/capture/consumer/events.jsonl'
H = '/DATA/GRAPH-API/.handoff/lanes/perception/ga493_replay_20260914/bundle/hook_decisions.jsonl'

kinds = collections.Counter()
first = {}
n = 0
with open(E) as f:
    for line in f:
        n += 1
        try:
            r = json.loads(line)
        except Exception:
            continue
        k = r.get('event') or r.get('kind') or r.get('type')
        kinds[k] += 1
        if k not in first:
            first[k] = (n, r.get('t') or r.get('ts') or r.get('time'))
print('events lines', n)
print('kinds', kinds.most_common(20))
for k, v in sorted(first.items(), key=lambda kv: kv[1][0]):
    print(f'  first {k!s:32s} line {v[0]:7d} t {v[1]}')

# hook_decisions: first admission and first observation_discard, and what observation_discard is
adm_t = None
with open(H) as f:
    for line in f:
        r = json.loads(line)
        if r.get('kind') == 'admission' and adm_t is None:
            adm_t = r.get('t')
            print('first admission t', adm_t, 'object', r.get('object'))
        if r.get('kind') == 'observation_discard':
            print('observation_discard example:', line[:300].rstrip())
            break
