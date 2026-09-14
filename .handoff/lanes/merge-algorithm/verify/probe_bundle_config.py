"""What the GA-493 bundle says it ran with, and what its debug config pins.

Read-only. Reads run_metadata.json, the bundle's config.yaml copy, the engine field of
hook_decisions.jsonl, and the debug config the bundle names.
"""
import json
import os
import re

B = "/DATA/GRAPH-API/.handoff/lanes/perception/ga493_replay_20260914/bundle"
KEYS = ("_config", "merge_engine", "merge_ontology", "similarity", "_local", "merge_attribute")

m = json.load(open(f"{B}/run_metadata.json"))
ec = m.get("effective_config") or {}
print("run_metadata top-level keys:", sorted(m.keys()))
print("config_name:", m.get("config_name"))
for k in sorted(ec):
    if any(s in k for s in KEYS):
        print("  effective_config", k, "=", ec[k])

print("--- bundle config.yaml lines of interest")
for i, line in enumerate(open(f"{B}/config.yaml"), 1):
    if re.search(r"merge_engine|merge_ontology_channel|generated_from|^\s+(label|color|material|description):", line):
        print(f"  {i}: {line.rstrip()}")

print("--- hook_decisions.jsonl engine field")
c = {}
n = 0
first_keys = None
for l in open(f"{B}/hook_decisions.jsonl"):
    try:
        d = json.loads(l)
    except Exception:
        continue
    n += 1
    if first_keys is None:
        first_keys = sorted(d.keys())
    e = d.get("engine") or (d.get("detail") or {}).get("engine") or (d.get("evidence") or {}).get("engine")
    c[e] = c.get(e, 0) + 1
print("  rows:", n, "engine counts:", c)
print("  first row keys:", first_keys)

dbg = ec.get("_config_file") or ""
print("--- debug config named by the bundle:", dbg)
for cand in (dbg, "/DATA/GRAPH-API/lost3dsg/test/debug_configs/ga493_bbox_replay.yaml"):
    if cand and os.path.exists(cand):
        text = open(cand).read().splitlines()
        print(f"  {cand}: {len(text)} lines")
        for i, line in enumerate(text, 1):
            if re.search(r"merge_engine|merge_ontology_channel|generated_from|^similarity|^\s+(label|color|material|description):", line):
                print(f"    {i}: {line.rstrip()}")
        break
    else:
        print(f"  {cand}: absent")
