#!/usr/bin/env python3
"""Run a SCHEDULE of runs, each with its own configuration.

    ./run.sh --schedule schedules/example.runs.yaml
    ./run_headless.sh --schedule schedules/example.runs.yaml     # same, without the viewers

A schedule is a list of ARMS. Each arm names itself and gives the configuration keys that differ
from the base. Nothing else about an arm may vary: same launcher, same gate, same bundle layout.

    base: lost3dsg/test/graphapi_only_config.yaml   # optional; the tracked config by default
    scene: hm3d_00861                               # optional, per-arm override allowed
    arms:
      - name: local_backend
        config:
          perception.backend: local
      - name: modal_backend
        config:
          perception.backend: modal
      - name: no_size_gate
        config:
          perception.backend: local
        env:
          GRAPH_API_SIZE_ENFORCE: "0"

WHY AN ARM IS A REAL CONFIG FILE. Owner instruction 2026-09-10: "we need to read config.yaml
ALWAYS". So an arm is not a pile of environment variables at launch time -- it is a config file
written to disk, pointed at with GRAPH_API_CONFIG, and RECORDED IN THE BUNDLE. Six weeks later the
question "what was different about this run?" is answered by a file, not by somebody's shell
history. `env` exists for the few settings that have no config key yet; it is deliberately awkward.

WHAT IT GUARANTEES.
  - ONE BUNDLE PER ARM, and a manifest naming which arm produced which bundle and how it ended.
  - A failed arm does NOT stop the schedule. It is recorded and the next arm starts. A sweep that
    stops on the first failure wastes the night; a sweep that hides a failure wastes the results.
  - RESUMABLE: an arm whose bundle already exists and passed is skipped unless --force.
  - The arms run in the order written. Nothing is parallel: one GPU, one container name, one port.
"""
from __future__ import annotations

import argparse
import copy
import datetime
import json
import os
import pathlib
import subprocess
import sys

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parent.parent


def _set_dotted(tree: dict, dotted: str, value):
    """`perception.backend: local` sets tree["perception"]["backend"]."""
    parts = dotted.split(".")
    node = tree
    for p in parts[:-1]:
        nxt = node.get(p)
        if not isinstance(nxt, dict):
            nxt = {}
            node[p] = nxt
        node = nxt
    node[parts[-1]] = value


def load_schedule(path: pathlib.Path):
    import yaml
    doc = yaml.safe_load(path.read_text()) or {}
    arms = doc.get("arms") or []
    if not arms:
        raise SystemExit(f"!! {path} defines no arms")
    seen = set()
    for a in arms:
        n = a.get("name")
        if not n:
            raise SystemExit(f"!! an arm in {path} has no name; a nameless arm cannot be reported")
        # A DUPLICATE NAME IS REFUSED, not suffixed. Two arms called the same thing produce two
        # bundles that no report can tell apart, which is worse than a stopped sweep.
        if n in seen:
            raise SystemExit(f"!! two arms are both called {n!r} in {path}")
        seen.add(n)
    return doc, arms


def arm_config_path(base_cfg: dict, arm: dict, out_dir: pathlib.Path, repo: pathlib.Path) -> pathlib.Path:
    """An arm either NAMES a config file or gives keys to override. Never both.

    `config_file:` is used as it is written, so a config prepared and reviewed by hand is the thing
    that runs -- not a copy this script generated from it. `config:` is merged onto the base and
    written out. Refusing both together matters: with a file AND overrides, a reader of the bundle
    cannot tell which won.
    """
    named, over = arm.get("config_file"), arm.get("config")
    if named and over:
        raise SystemExit(f"!! arm {arm['name']!r} gives both config_file and config; "
                         "pick one, or a reader cannot tell which won")
    if named:
        p = pathlib.Path(named)
        if not p.is_absolute():
            p = repo / p
        if not p.exists():
            raise SystemExit(f"!! arm {arm['name']!r} names {p}, which does not exist")
        return p
    return write_arm_config(base_cfg, arm, out_dir)


def write_arm_config(base_cfg: dict, arm: dict, out_dir: pathlib.Path) -> pathlib.Path:
    import yaml
    tree = copy.deepcopy(base_cfg)
    for dotted, value in (arm.get("config") or {}).items():
        _set_dotted(tree, dotted, value)
    out_dir.mkdir(parents=True, exist_ok=True)
    p = out_dir / f"arm_{arm['name']}.yaml"
    header = (f"# Written by schedule_runs.py for arm {arm['name']!r} at "
              f"{datetime.datetime.now().isoformat(timespec='seconds')}.\n"
              f"# Overrides applied: {json.dumps(arm.get('config') or {}, sort_keys=True)}\n"
              f"# This file IS the arm. It is passed as GRAPH_API_CONFIG and its sha is recorded\n"
              f"# in the bundle, so what differed about this run is answerable from the bundle.\n")
    p.write_text(header + yaml.safe_dump(tree, sort_keys=False))
    return p


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("schedule", type=pathlib.Path)
    ap.add_argument("--runner", default=str(REPO / "run.sh"),
                    help="run.sh or run_headless.sh; the schedule does not choose this")
    ap.add_argument("--force", action="store_true", help="re-run arms that already have a bundle")
    ap.add_argument("--dry-run", action="store_true", help="write the arm configs and print the plan")
    args = ap.parse_args()

    doc, arms = load_schedule(args.schedule)
    import yaml
    base_path = pathlib.Path(doc.get("base") or (REPO / "lost3dsg/src/perception_module/config.yaml"))
    if not base_path.is_absolute():
        base_path = REPO / base_path
    if not base_path.exists():
        raise SystemExit(f"!! base config {base_path} does not exist")
    base_cfg = yaml.safe_load(base_path.read_text()) or {}

    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    sweep = REPO / "results" / f"sweep_{stamp}"
    sweep.mkdir(parents=True, exist_ok=True)
    manifest = sweep / "manifest.json"
    rows = []

    # `repeat: N` RUNS ONE ARM N TIMES. The noise floor has never been measured and cannot be
    # recovered from the archive, so repeating ONE configuration is the only way to get it. Each
    # repetition is its own run with its own bundle; only the reported name gains a suffix.
    expanded = []
    for a in arms:
        n = int(a.get("repeat", 1) or 1)
        if n == 1:
            expanded.append(a)
        else:
            for k in range(1, n + 1):
                b = dict(a)
                b["name"] = f"{a['name']}_r{k}"
                b.pop("repeat", None)
                expanded.append(b)
    arms = expanded
    print(f"schedule: {args.schedule}\nbase:     {base_path}\nruns:     {len(arms)}\nsweep:    {sweep}\n")

    for i, arm in enumerate(arms, 1):
        cfg = arm_config_path(base_cfg, arm, sweep, REPO)
        scene = arm.get("scene") or doc.get("scene") or ""
        env = dict(os.environ)
        env["GRAPH_API_CONFIG"] = str(cfg)
        # CFG_NAME is what the launcher echoes and stamps; keep the two agreeing so the bundle
        # does not name one file while loading another.
        env["CFG_NAME"] = cfg.name
        for k, v in (arm.get("env") or {}).items():
            env[k] = str(v)
        cmd = [args.runner, "--one-storey"] + ([scene] if scene else [])
        print(f"== arm {i}/{len(arms)}: {arm['name']}")
        print(f"   config: {cfg}")
        print(f"   cmd:    {' '.join(cmd)}")
        if args.dry_run:
            rows.append({"arm": arm["name"], "config": str(cfg), "status": "dry-run"})
            continue
        rc = subprocess.call(cmd, cwd=str(REPO), env=env)
        # WHICH BUNDLE DID THIS ARM PRODUCE? The newest results directory that is not the sweep
        # directory itself. Read after the run rather than predicted, because the launcher owns
        # the timestamp and a predicted name is a name that can be wrong.
        bundles = sorted((REPO / "results").glob("2026*"), key=lambda p: p.stat().st_mtime)
        bundle = str(bundles[-1]) if bundles else ""
        ended = ""
        try:
            meta = json.loads((pathlib.Path(bundle) / "run_metadata.json").read_text())
            ended = (meta.get("terminating_node") or {}).get("ended") or ""
        except Exception:
            pass
        rows.append({"arm": arm["name"], "config": str(cfg), "returncode": rc,
                     "bundle": bundle, "ended": ended,
                     "status": "ok" if rc == 0 else "failed"})
        # A FAILED ARM DOES NOT STOP THE SCHEDULE, and the manifest is rewritten after EVERY arm
        # so a sweep killed halfway still says what it did.
        manifest.write_text(json.dumps({"schedule": str(args.schedule), "base": str(base_path),
                                        "arms": rows}, indent=2))
        print(f"   -> rc={rc} bundle={bundle or '(none)'} ended={ended or '(unrecorded)'}\n")

    manifest.write_text(json.dumps({"schedule": str(args.schedule), "base": str(base_path),
                                    "arms": rows}, indent=2))
    ok = sum(1 for r in rows if r.get("status") == "ok")
    print(f"DONE. {ok} of {len(rows)} arms returned 0. Manifest: {manifest}")
    return 0 if ok == len(rows) else 1


def _selfcheck():
    import tempfile
    tree = {}
    _set_dotted(tree, "perception.backend", "local")
    _set_dotted(tree, "perception.cloud_timeout_s", 60.0)
    assert tree == {"perception": {"backend": "local", "cloud_timeout_s": 60.0}}, tree
    # a dotted key must not destroy its siblings
    _set_dotted(tree, "perception.backend", "modal")
    assert tree["perception"]["cloud_timeout_s"] == 60.0, tree
    # a scalar in the path is replaced by a dict rather than raising
    t2 = {"a": 1}
    _set_dotted(t2, "a.b", 2)
    assert t2 == {"a": {"b": 2}}, t2
    with tempfile.TemporaryDirectory() as d:
        p = pathlib.Path(d) / "s.yaml"
        p.write_text("arms:\n  - name: one\n  - name: one\n")
        try:
            load_schedule(p)
            raise AssertionError("a duplicate arm name must be refused")
        except SystemExit as e:
            assert "both called" in str(e), e
        p.write_text("arms:\n  - config: {}\n")
        try:
            load_schedule(p)
            raise AssertionError("a nameless arm must be refused")
        except SystemExit as e:
            assert "no name" in str(e), e
        p.write_text("arms: []\n")
        try:
            load_schedule(p)
            raise AssertionError("an empty schedule must be refused")
        except SystemExit as e:
            assert "no arms" in str(e), e
    print("schedule_runs self-check: PASSED")


if __name__ == "__main__":
    if "--self-check" in sys.argv:
        _selfcheck()
        raise SystemExit(0)
    raise SystemExit(main())
