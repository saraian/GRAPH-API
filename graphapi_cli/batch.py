"""Existing schedule schema, shared launch service and exact child result attribution."""
import copy
import hashlib
import json
from pathlib import Path
import uuid

import yaml

from .configuration import merge, path_from, read_yaml, resolve_config
from .launch import start
from .registry import write_json
from .credentials import prepare_vlm_credentials, uses_vlm


def dotted(tree, key, value):
    parts = key.split(".")
    node = tree
    for part in parts[:-1]:
        if not isinstance(node.get(part), dict):
            node[part] = {}
        node = node[part]
    node[parts[-1]] = value


def run_batch(root, env, args):
    schedule = path_from(args.schedule, root)
    document = read_yaml(schedule)
    arms = document.get("arms")
    if not isinstance(arms, list) or not arms:
        raise ValueError(f"{schedule}: no experiment arms")
    _, base = resolve_config(root, document.get("base") or args.config, args.local)
    seen, signatures, prepared = set(), {}, []
    for arm in arms:
        name = arm.get("name")
        if not name or name in seen:
            raise ValueError(f"missing/duplicate arm name: {name}")
        seen.add(name)
        if arm.get("config_file") and arm.get("config"):
            raise ValueError(f"{name}: config_file and config overrides are mutually exclusive")
        if arm.get("config_file"):
            _, config = resolve_config(root, arm["config_file"], args.local)
        else:
            config = copy.deepcopy(base)
            for key, value in arm.get("config", {}).items():
                dotted(config, key, value)
        inputs = {"config": config, "env": arm.get("env", {}),
                  "scene": arm.get("scene") or document.get("scene")}
        signature = json.dumps(inputs, sort_keys=True)
        duplicate = signatures.get(signature)
        if duplicate and arm.get("replicate_of") != duplicate:
            raise ValueError(f"{name} duplicates {duplicate}; declare replicate_of: {duplicate} without changing experimental values")
        if arm.get("replicate_of") and signature != next((s for s, n in signatures.items() if n == arm["replicate_of"]), None):
            raise ValueError(f"{name}: replicate_of does not match the named earlier arm")
        signatures.setdefault(signature, name)
        count = arm.get("repeat", 1)
        if not isinstance(count, int) or isinstance(count, bool) or count < 1:
            raise ValueError(f"{name}: repeat must be a positive integer")
        for index in range(count):
            prepared.append({"name": name if count == 1 else f"{name}_{index + 1}",
                             "config": config, "environment": arm.get("env", {}), "scene": inputs["scene"]})
    digest = hashlib.sha256(schedule.read_bytes() + json.dumps(prepared, sort_keys=True).encode()).hexdigest()[:16]
    manifest = Path(env["WORKSPACE_ROOT"]) / "results" / f"batch_{digest}.json"
    old = json.loads(manifest.read_text()) if manifest.exists() else {"arms": []}
    previous = {row["arm"]: row for row in old["arms"]}
    rows = []
    gpus = [value.strip() for value in args.gpus.split(",") if value.strip()]
    if not gpus:
        raise ValueError("--gpus requires at least one GPU index")
    if args.dry_run:
        print(json.dumps({"schedule": str(schedule), "arms": [{k: v for k, v in row.items() if k != "config"} for row in prepared]}, indent=2))
        return 0
    # Check every pending arm before starting the first one, so a missing key
    # cannot leave an expensive batch partially acquired.
    for arm in prepared:
        prior = previous.get(arm["name"])
        if prior and prior.get("state") == "COMPLETED" and not args.force:
            continue
        child_env = dict(env, **{k: str(v) for k, v in arm["environment"].items()})
        if uses_vlm({"mode": "sim"}, child_env):
            prepare_vlm_credentials(root, arm["config"], child_env)
    for index, arm in enumerate(prepared):
        prior = previous.get(arm["name"])
        if prior and prior.get("state") == "COMPLETED" and not args.force:
            rows.append(prior)
            continue
        path = root / "config/.runtime" / f"batch_{uuid.uuid4().hex}.yaml"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(yaml.safe_dump(arm["config"], sort_keys=False))
        path.chmod(0o600)
        child_env = dict(env)
        child_env.update({k: str(v) for k, v in arm["environment"].items()})
        child = start(root, {"mode": "sim", "config": str(path), "local": args.local,
                             "scene": arm["scene"], "one_storey": True, "gpu": gpus[index % len(gpus)]}, inherited=child_env)
        rows.append({"arm": arm["name"], "operation_id": child["operation_id"], "bundles": child["bundles"],
                     "state": child["state"], "returncode": child["returncode"]})
        write_json(manifest, {"schedule": str(schedule), "arms": rows})
        if child["state"] != "COMPLETED" and not args.continue_on_failure:
            break
    return 0 if len(rows) == len(prepared) and all(r["state"] == "COMPLETED" for r in rows) else 1
