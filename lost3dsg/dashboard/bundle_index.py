"""What every archived run IS: its configuration, its statistics, its map, its verdict.

WHY A SEPARATE MODULE. Choosing which bundle to replay is a question about the runs, not about
the replay page, and the answer is spread across eight artefacts. Gathering it in one place
means a reader compares runs by what they RECORDED rather than by their timestamps.

THE ONE PERFORMANCE RULE THAT SHAPES THIS FILE: `hook_decisions.jsonl` reaches 475 MB, so it
is never json-parsed to count anything. Kinds are counted by a byte scan over fixed keys, and
the result is CACHED OUTSIDE THE BUNDLE — rule 12, an archived bundle is never written to.

Every field is read or reported absent. Nothing here is inferred from a neighbouring value.
"""
import json
import os
import re
import time
from pathlib import Path

try:
    from found.dashboard import dash_env
except ImportError:
    import dash_env

RUNS_DIR = dash_env.runs_dir()
CACHE_DIR = Path(dash_env.env("DASH_BUNDLE_CACHE", "/tmp/dash_bundle_cache"))
# Kinds counted by scanning bytes. Extending this list is cheap; parsing the file is not.
KINDS = ("admission", "merge_refused", "merge", "link", "update", "not_offered",
         # GA-232: not-offered pairs are summarised per sweep now, not written one by
         # one. A bundle written after 2026-09-01 carries this instead of 2.35 M rows.
         "not_offered_summary")
_KIND_RE = {k: re.compile((f'"kind": "{k}"').encode()) for k in KINDS}


def _read_json(p, default=None):
    try:
        with open(p) as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return default


def _count_lines(p):
    try:
        with open(p, "rb") as f:
            return sum(buf.count(b"\n") for buf in iter(lambda: f.read(1 << 20), b""))
    except OSError:
        return 0


def _count_kinds(p):
    """Count decision kinds by BYTE SCAN. `merge_refused` contains `merge`, so each line is
    matched against the longest key first and attributed once — the exact mistake working
    rule 50 was written for, avoided by construction rather than by care."""
    counts = dict.fromkeys(KINDS, 0)
    order = sorted(KINDS, key=len, reverse=True)
    try:
        with open(p, "rb") as f:
            for line in f:
                for k in order:
                    if _KIND_RE[k].search(line):
                        counts[k] += 1
                        break
    except OSError:
        pass
    return counts


def _human(n):
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.0f} {unit}" if unit == "B" else f"{n:.1f} {unit}"
        n /= 1024.0
    return f"{n:.1f} GB"


def _dir_bytes(p):
    total = 0
    try:
        for root, _dirs, files in os.walk(p):
            for f in files:
                try:
                    total += os.path.getsize(os.path.join(root, f))
                except OSError:
                    pass
    except OSError:
        pass
    return total


def describe(name, use_cache=True):
    """-> one dict describing a run. Cached on (bundle mtime, decision-file size)."""
    d = RUNS_DIR / name
    if not d.is_dir():
        return None
    dec = d / "hook_decisions.jsonl"
    stamp = f"{int(d.stat().st_mtime)}_{dec.stat().st_size if dec.is_file() else 0}"
    cache = CACHE_DIR / f"{name}.{stamp}.json"
    if use_cache and cache.is_file():
        got = _read_json(cache)
        if got:
            return got

    meta = _read_json(d / "run_metadata.json", {}) or {}
    eff = (meta.get("resolved_config") or {}).get("effective_config") or {}
    policy = meta.get("policy") or {}
    build = _read_json(d / "build_cache.json", {}) or {}
    room = _read_json(d / "room.json", {}) or {}
    rooms = ((room.get("building") or {}).get("rooms")) or room.get("rooms") or []
    wm = _read_json(d / "persistent_perception.json", []) or []

    frame_files = sorted((d / "frames").glob("*.jpg")) if (d / "frames").is_dir() else []
    n_frames = len(frame_files)
    # MEASURED resolution, from a frame the run actually wrote. `habitat.width` in the
    # effective config is NOT trustworthy for this: run_sim.sh records an incident where it
    # "read 1280 from my own code default while the merged config supplied 640 and the sensor
    # stayed at 640x480", and bundle 20260901_055513 still stamps 1280x960 beside 640x480
    # JPEGs. A page that printed the config value would republish a number the project has
    # already found to be wrong, so the pixels win and the config value is shown beside it
    # only when the two disagree.
    measured_res = None
    if frame_files:
        try:
            from found.dashboard import replay_view as _rv
        except ModuleNotFoundError:
            import replay_view as _rv
        wh = _rv._jpeg_size(frame_files[0])
        if wh:
            measured_res = f"{wh[0]}x{wh[1]}"
    n_det = _count_lines(d / "detections.jsonl") if (d / "detections.jsonl").is_file() else 0

    gt_joined = None
    if n_det:
        # Sampled, and SAID to be sampled. The file is small enough to read whole today; the
        # cap is here so a future 10x run degrades to an estimate instead of a stall.
        seen = joined = 0
        try:
            with open(d / "detections.jsonl") as f:
                for line in f:
                    if seen >= 20000:
                        break
                    seen += 1
                    if '"habitat_gt_instance_id": null' not in line and \
                       '"habitat_gt_instance_id"' in line:
                        joined += 1
        except OSError:
            seen = 0
        gt_joined = round(100.0 * joined / seen) if seen else None

    kinds = _count_kinds(dec) if dec.is_file() else dict.fromkeys(KINDS, 0)
    verdicts = {}
    if kinds.get("admission"):
        v = {"admit": 0, "reject": 0, "abstain": 0}
        try:
            with open(dec, "rb") as f:
                for line in f:
                    if b'"kind": "admission"' not in line:
                        continue
                    for k in v:
                        if f'"{k}"'.encode() in line:
                            v[k] += 1
                            break
        except OSError:
            pass
        verdicts = v

    info = {
        "name": name,
        "scene": meta.get("run_id", name).split("_", 2)[-1],
        # WHICH MACHINE produced this bundle. Written by the launcher from 2026-09-10; None for
        # every bundle recorded before that, and the reader must show the difference rather than
        # guess a default -- a bundle that does not say is not a bundle from this machine.
        "machine": meta.get("machine"),
        "started": name[:15],
        "size": _human(_dir_bytes(d)),
        "frames": n_frames,
        "detections": n_det,
        "gt_joined_pct": gt_joined,
        "objects": len(wm),
        "rooms": len(rooms),
        "room_ids": [r.get("room_id") for r in rooms][:6],
        "decisions": kinds,
        "verdicts": verdicts,
        "replayable": bool(n_frames and n_det),
        "why_not": ("" if n_frames and n_det else
                    ("no detections.jsonl — per-detection recording was off" if not n_det
                     else "no frames/ directory")),
        "config": {
            "merge_engine": eff.get("association.merge_engine", "legacy (default)"),
            "gvd_method": eff.get("rooms.gvd_method"),
            "crop": eff.get("crop.construction"),
            "per_detection": eff.get("archive.per_detection"),
            "backend": eff.get("perception.backend"),
            "resolution": measured_res or (
                f'{eff.get("habitat.width")}x{eff.get("habitat.height")} (config)'
                if eff.get("habitat.width") else None),
            "resolution_config": (f'{eff.get("habitat.width")}x{eff.get("habitat.height")}'
                                  if eff.get("habitat.width") else None),
            "resolution_source": "frames" if measured_res else "config",
            "config_file": meta.get("config_name"),
        },
        "policy": {k: policy.get(k) for k in
                   ("enforce", "hold_band", "min_support", "rooms_enforced", "aligner",
                    "corpus_order", "kg_aliases") if k in policy},
        "map": (meta.get("feed") or {}).get("localize_db") or eff.get("rooms.map_db"),
        "build": {"mode": build.get("mode"), "seconds": build.get("seconds"),
                  "key": build.get("build_key")},
        "generated": time.time(),
    }
    try:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        for old in CACHE_DIR.glob(f"{name}.*.json"):
            old.unlink(missing_ok=True)
        cache.write_text(json.dumps(info))
    except OSError:
        pass          # a cache that cannot be written is not a reason to fail the page
    return info


def index(limit=40):
    if not RUNS_DIR.is_dir():
        return []
    dirs = [p for p in RUNS_DIR.iterdir() if p.is_dir() and not p.is_symlink()]
    names = sorted((p.name for p in dirs), reverse=True)[:limit]
    return [i for i in (describe(n) for n in names) if i]


def _cell(v, dash="—"):
    return dash if v in (None, "", 0) and v is not False else str(v)


def page():
    rows = index()
    n_ok = sum(1 for r in rows if r["replayable"])
    body = []
    for r in rows:
        c, p, k, v = r["config"], r["policy"], r["decisions"], r["verdicts"]
        split = (f'{v.get("admit",0)} / {v.get("reject",0)} / {v.get("abstain",0)}'
                 if v else "—")
        engine = c["merge_engine"]
        eng_cls = "ev" if engine and "evidence" in str(engine) else "lg"
        act = (f'<a class="go" href="/replay?bundle={r["name"]}">REPLAY</a>'
               if r["replayable"] else f'<span class="no" title="{r["why_not"]}">—</span>')
        body.append(f"""<tr>
<td class="id">{r['name']}<div class="sub">{_cell(c['config_file'])} · {r['size']}</div></td>
<td><span class="tag {eng_cls}">{engine}</span>
    <div class="sub">enforce={_cell(p.get('enforce'))} · {_cell(c['gvd_method'])} ·
    crop {_cell(c['crop'])} · {_cell(c['resolution'])}{
      f" <span title=\"config says {c['resolution_config']}; the frames are {c['resolution']}\">&#9888;</span>"
      if c.get('resolution_source') == 'frames' and c.get('resolution_config')
      and c['resolution_config'] != c['resolution'] else ""}</div></td>
<td class="num">{_cell(r['frames'])}</td>
<td class="num">{_cell(r['detections'])}</td>
<td class="num">{_cell(r['gt_joined_pct'] and str(r['gt_joined_pct']) + '%')}</td>
<td class="num">{_cell(r['objects'])}</td>
<td class="num">{_cell(r['rooms'])}<div class="sub">{' '.join(x for x in r['room_ids'] if x) or ''}</div></td>
<td class="num">{split}</td>
<td class="num">{_cell(k.get('merge'))}<div class="sub">{_cell(k.get('merge_refused'))} refused</div></td>
<td>{act}</td></tr>""")
    return f"""<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Bundles</title>
<style>
:root{{--ground:#f7f6f3;--ink:#15161a;--dim:#5a5f68;--faint:#8d939c;--rule:#dfdcd6;--card:#fffefc;
  --ev:#1d7a52;--ev-bg:#e6f2ec;--lg:#9a5b06;--lg-bg:#faf0de}}
@media (prefers-color-scheme:dark){{:root{{--ground:#111318;--ink:#e8e8ea;--dim:#9aa0aa;
  --faint:#6d747e;--rule:#282c34;--card:#181b21;--ev:#57c48d;--ev-bg:#12281e;
  --lg:#e0a34e;--lg-bg:#2a2013}}}}
*{{box-sizing:border-box}}
body{{margin:0;background:var(--ground);color:var(--ink);
  font:14px/1.55 ui-sans-serif,system-ui,-apple-system,sans-serif}}
.wrap{{max-width:1400px;margin:0 auto;padding:22px 20px 60px}}
a.back{{font:600 11px ui-monospace,monospace;color:var(--dim);text-decoration:none;
  border:1px solid var(--rule);border-radius:3px;padding:5px 10px}}
h1{{font-size:25px;margin:15px 0 5px;letter-spacing:-.015em}}
p.sub2{{color:var(--dim);margin:0 0 16px;max-width:76ch}}
.tw{{overflow-x:auto;border:1px solid var(--rule);border-radius:3px;background:var(--card)}}
table{{border-collapse:collapse;width:100%;font-size:13px}}
th{{text-align:left;font:600 10px ui-monospace,monospace;text-transform:uppercase;
  letter-spacing:.08em;color:var(--faint);padding:9px 10px;border-bottom:1px solid var(--rule);
  white-space:nowrap}}
td{{padding:9px 10px;border-bottom:1px solid var(--rule);vertical-align:top}}
tr:last-child td{{border-bottom:none}}
td.id{{font:700 11px ui-monospace,monospace;white-space:nowrap}}
td.num{{font:12px ui-monospace,monospace;text-align:right;white-space:nowrap;
  font-variant-numeric:tabular-nums}}
.sub{{font:10px ui-monospace,monospace;color:var(--faint);font-weight:400;margin-top:3px}}
.tag{{display:inline-block;font:600 10px ui-monospace,monospace;padding:2px 7px;border-radius:2px}}
.tag.ev{{color:var(--ev);background:var(--ev-bg);border:1px solid var(--ev)}}
.tag.lg{{color:var(--lg);background:var(--lg-bg);border:1px solid var(--lg)}}
a.go{{font:600 10px ui-monospace,monospace;color:var(--ev);border:1px solid var(--ev);
  border-radius:3px;padding:4px 9px;text-decoration:none;white-space:nowrap}}
.no{{color:var(--faint);cursor:help}}
</style></head><body><div class="wrap">
<a class="back" href="/">&larr; DASHBOARD</a>
<h1>Bundles</h1>
<p class="sub2">Every archived run, by what it RECORDED rather than by when it happened.
<b>{n_ok} of {len(rows)} can be replayed</b> — replay needs frames and detections, which a run
only writes with <code>archive.per_detection</code> on. Hover a dash to see why not.</p>
<div class="tw"><table>
<tr><th>run</th><th>engine &amp; config</th><th>frames</th><th>dets</th><th>GT</th>
<th>objects</th><th>rooms</th><th>admit/rej/abst</th><th>merges</th><th></th></tr>
{''.join(body)}
</table></div>
</div></body></html>"""


if __name__ == "__main__":
    t0 = time.time()
    rows = index(limit=8)
    print(f"described {len(rows)} bundles in {time.time() - t0:.1f}s "
          f"(cache {CACHE_DIR})")
    for r in rows:
        print(f"  {r['name']}  engine={r['config']['merge_engine']:<18} "
              f"frames={r['frames']:<5} dets={r['detections']:<6} obj={r['objects']:<5} "
              f"rooms={r['rooms']}  merges={r['decisions'].get('merge')}  "
              f"{'REPLAYABLE' if r['replayable'] else r['why_not']}")
    t1 = time.time()
    index(limit=8)
    print(f"\n  second pass (cached): {time.time() - t1:.2f}s")
    assert "<table" in page()
    print("bundle_index self-check OK")
