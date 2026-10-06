"""Generate the CLI registry from tracked entrypoints; never execute/import them."""
import argparse
import ast
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[2]


def generate(root):
    result = {}
    paths = subprocess.check_output(["git", "ls-files", "--cached", "--others", "--exclude-standard", "*.py", "*.sh"], cwd=root, text=True).splitlines()
    for relative in paths:
        p = root / relative
        if p.suffix == ".py":
            tree = ast.parse(p.read_text())
            guarded = any(isinstance(n, ast.If) and "__name__" in ast.unparse(n.test)
                          and "__main__" in ast.unparse(n.test) for n in tree.body)
            if not guarded and not ("launch" in p.name and "def generate_launch_description" in p.read_text()):
                continue
        slug = p.stem.replace("_", "-").replace(".launch", "")
        if p.suffix == ".py" and "def generate_launch_description" in p.read_text():
            slug = "launch-" + slug.replace("-launch", "")
            kind, env, gpu = "launch", "pal" if p.name.startswith("bag_slam") else "ros", True
        else:
            kind = "script"
            env = "ros"
            gpu = False
            if "tools/baselines/" in relative:
                slug = "baseline-" + slug
                env = "orchestrator"
            if "tiago/found-docker/" in relative:
                env = "pal"
            if "habitat" in slug or slug in {"schedule-batch", "voronoi-roadmap", "hm3d-ground-truth-manifest", "sample-tour"}:
                env, gpu = "habitat", True
            if relative == "tiago/tiago-host-dds.sh":
                env = "host"
        if slug in result:
            slug = relative.replace("/", "-").rsplit(".", 1)[0].replace("_", "-")
        visibility = "diagnostic" if ("test" in p.stem or "/old/" in relative or "/efficientvit/" in relative) else "public"
        entry = {"path": relative, "kind": kind, "environment": env, "gpu": gpu, "visibility": visibility}
        if relative == "lost3dsg/test/live_stack_container.sh":
            entry.update(kind="internal", use="Internal ROS supervisor; use graphapi run sim")
        if relative == "tiago/found-docker/found-robot-stack.sh":
            entry.update(kind="internal", use="Use graphapi run tiago physical / run tiago bag for managed private sessions")
        if relative == "lost3dsg/test/schedule_runs.py":
            entry.update(kind="internal", use="Use graphapi batch SCHEDULE")
        if relative.startswith("tools/baselines/") and p.suffix == ".py":
            entry["module"] = relative[:-3].replace("/", ".")
        if relative.startswith("graphapi_cli/"):
            entry.update(kind="internal", use="Internal managed implementation; use graphapi --help")
        public_runtime = {
            "graphapi_cli/runtime/connect_gin.sh": "host",
            "graphapi_cli/runtime/baseline_native.sh": "host",
            "graphapi_cli/runtime/run_pipelines.sh": "orchestrator",
        }
        if relative in public_runtime:
            entry.update(kind="script", environment=public_runtime[relative])
            entry.pop("use", None)
        result[slug] = entry
    # Library-style baseline entrypoint deliberately has no main guard.
    result["baseline-entrypoint"] = {"path": "tools/baselines/entrypoint.py", "module": "tools.baselines.entrypoint",
                                      "kind": "script", "environment": "orchestrator", "gpu": False, "visibility": "public"}
    result["baseline-native"]["gpu"] = True
    result["baseline-native"]["path"] = "graphapi_cli/runtime/baseline_native.sh"
    return dict(sorted(result.items()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    content = json.dumps(generate(ROOT), indent=2) + "\n"
    path = ROOT / "config/entrypoints.json"
    if args.check:
        if not path.exists() or path.read_text() != content:
            raise SystemExit("entrypoint registry is stale: python tools/startup/generate_catalogue.py")
    else:
        path.write_text(content)
    print("registered entrypoints:", len(json.loads(content)))


if __name__ == "__main__":
    main()
