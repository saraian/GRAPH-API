"""Did the GA-493 merge path see a separation channel (= observations reached the merge sweep)?
And what do the persisted objects carry?"""
import collections
import json

B = '/DATA/GRAPH-API/.handoff/lanes/perception/ga493_replay_20260914/bundle'

kinds = collections.Counter()
sep_vals = collections.Counter()
sep_abst = collections.Counter()
engines = collections.Counter()
examples = []
n = 0


def walk(o):
    if isinstance(o, dict):
        if 'channels' in o and isinstance(o['channels'], dict):
            yield o
        for v in o.values():
            yield from walk(v)
    elif isinstance(o, list):
        for v in o:
            yield from walk(v)


with open(B + '/hook_decisions.jsonl') as f:
    for line in f:
        n += 1
        try:
            r = json.loads(line)
        except Exception:
            continue
        kinds[r.get('kind') or r.get('event') or r.get('type')] += 1
        for o in walk(r):
            engines[o.get('engine')] += 1
            ch = o['channels']
            ab = o.get('abstentions') or {}
            if 'separation' in ch:
                v = ch['separation']
                tag = 'None' if v is None else ('<=-7.99' if v <= -7.99 else ('>0' if v > 0 else 'other'))
                sep_vals[tag] += 1
            elif 'separation' in ab:
                sep_abst[str(ab['separation'])[:70]] += 1
            else:
                sep_abst['<neither>'] += 1
            if len(examples) < 2:
                examples.append({k: o.get(k) for k in ('channels', 'abstentions', 'engine',
                                                       'decision_reason', 'room_a', 'room_b',
                                                       'hypothesis_total', 'updates', 'reason')})
print('hook_decisions lines', n)
print('kinds', kinds.most_common(15))
print('engines on records with channels', engines)
print('separation channel values', sep_vals)
print('separation abstentions', sep_abst.most_common(5))
for e in examples:
    print(e)

# tracking_scan_summary rows
cnt = 0
with open(B + '/hook_decisions.jsonl') as f:
    for line in f:
        if 'tracking_scan_summary' in line or 'reach_fallback' in line:
            cnt += 1
            if cnt <= 3:
                print('TSS', line[:400].rstrip())
print('rows mentioning tracking_scan_summary/reach_fallback', cnt)

d = json.load(open(B + '/persistent_perception.json'))
objs = d if isinstance(d, list) else (d.get('objects') or list(d.values())[0])
print('persistent type', type(d).__name__, 'n objects', len(objs))
o = objs[0]
print('keys', sorted(o.keys()) if isinstance(o, dict) else o)
print('room_id', collections.Counter(str(x.get('room_id')) for x in objs))
print('has observations key', sum(1 for x in objs if 'observations' in x),
      'n_obs>0', sum(1 for x in objs if x.get('observations')))
print('n_observations field', collections.Counter(str(x.get('n_observations', x.get('observation_count', '?'))) for x in objs).most_common(8))
