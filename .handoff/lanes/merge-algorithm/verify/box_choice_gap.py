"""Which box does each locality stage read, and does it matter on the GA-493 bundle?

Broad phases (world_model.candidates for association, generate_candidates' gap test for the
merge) read the MEASURED bbox; the exact gates (locality_ok, the service geometry gate) read
the FUSED box via association.locality_bounds. Measures, on the 104 final objects:
  (a) per object, how far the fused box extends beyond the measured box (max over 6 faces);
  (b) over all pairs, agreement of `gap <= 0.3 m` on measured vs fused boxes;
  (c) kinds and refusal reasons in the bundle's hook_decisions.jsonl (denominators).
"""
import itertools
import json
import statistics

P = "/DATA/GRAPH-API/.handoff/lanes/perception/ga493_replay_20260914/bundle/persistent_perception.json"
H = "/DATA/GRAPH-API/.handoff/lanes/perception/ga493_replay_20260914/bundle/hook_decisions.jsonl"
KEYS = ("x_min", "x_max", "y_min", "y_max", "z_min", "z_max")
GAP = 0.3

d = json.load(open(P))
objs = d if isinstance(d, list) else (d.get("objects") or d.get("persistent_perceptions") or d)
if isinstance(objs, dict):
    objs = list(objs.values())


def bounds(b):
    if not b:
        return None
    try:
        return tuple(float(b[k]) for k in KEYS)
    except (KeyError, TypeError, ValueError):
        return None


def gap(a, b):
    return max(0.0, max(a[0] - b[1], b[0] - a[1]), max(a[2] - b[3], b[2] - a[3]),
               max(a[4] - b[5], b[4] - a[5]))


def base(o):
    lab = str(o.get("label", ""))
    return lab.split("#")[0].strip().lower()


n = len(objs)
have_m = [o for o in objs if bounds(o.get("bbox"))]
have_f = [o for o in objs if bounds(o.get("fused_bbox"))]
print(f"objects: {n}; with measured bbox: {len(have_m)}; with fused_bbox: {len(have_f)}")

ext = []
for o in objs:
    m, f = bounds(o.get("bbox")), bounds(o.get("fused_bbox"))
    if m and f:
        ext.append((max(m[0] - f[0], f[1] - m[1], m[2] - f[2], f[3] - m[3], m[4] - f[4], f[5] - m[5]),
                    o.get("label")))
ext.sort(reverse=True)
over = [e for e in ext if e[0] > GAP]
if ext:
    print(f"(a) fused box extends beyond measured box: n={len(ext)} objects with both; "
          f"median {statistics.median([e[0] for e in ext]):.3f} m, max {ext[0][0]:.3f} m ({ext[0][1]}); "
          f"{len(over)} of {len(ext)} extend > {GAP} m (the amount at which the 2x0.3 m association "
          f"broad phase on the measured box can hide a candidate the fused-box gate would accept)")
    print("    top 5:", [(round(e, 3), lab) for e, lab in ext[:5]])

pairs = f_ok = m_ok = f_only = m_only = 0
f_only_same = []
for a, b in itertools.combinations(objs, 2):
    ma, mb = bounds(a.get("bbox")), bounds(b.get("bbox"))
    if not (ma and mb):
        continue
    fa, fb = bounds(a.get("fused_bbox")) or ma, bounds(b.get("fused_bbox")) or mb
    pairs += 1
    gm, gf = gap(ma, mb) <= GAP, gap(fa, fb) <= GAP
    f_ok += gf
    m_ok += gm
    if gf and not gm:
        f_only += 1
        if base(a) == base(b):
            f_only_same.append((a.get("label"), b.get("label"), round(gap(ma, mb), 3), round(gap(fa, fb), 3)))
    if gm and not gf:
        m_only += 1
print(f"(b) pairs with both measured boxes: {pairs}; gap<=0.3 on fused: {f_ok}; on measured: {m_ok}; "
      f"fused-compatible but measured-incompatible: {f_only} (same base label: {len(f_only_same)}); "
      f"measured-compatible but fused-incompatible: {m_only}")
print("    same-label fused-only pairs (a, b, measured gap, fused gap):", f_only_same[:12])

kinds = {}
reasons = {}
with open(H) as fh:
    for line in fh:
        try:
            r = json.loads(line)
        except ValueError:
            continue
        k = r.get("kind")
        kinds[k] = kinds.get(k, 0) + 1
        if k == "merge_refused":
            reasons[r.get("reason")] = reasons.get(r.get("reason"), 0) + 1
print("(c) hook_decisions.jsonl kinds:", kinds)
print("    merge_refused reasons (legacy engine, GA-493):", reasons)
