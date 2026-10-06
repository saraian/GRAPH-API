"""Idempotent preparation, with no host Habitat or hardware certification."""
from pathlib import Path
import os
import subprocess
import sys

import yaml

from .configuration import environment, initialize, path_from, read_yaml
from .docker import ensure_image, ensure_gazebo_image, run_python


def setup(root, env, args):
    target = initialize(root, args.local)
    settings = read_yaml(target)
    for key, attr in [("dataset", "dataset"), ("models", "models"), ("cache", "cache"),
                      ("workspace", "workspace"), ("pal_bundle", "pal_bundle"), ("baseline_repos", "baseline_repos")]:
        value = getattr(args, attr, None)
        if value:
            settings.setdefault("paths", {})[key] = str(path_from(value, root))
    if args.container:
        settings.setdefault("docker", {})["pal_container"] = args.container
    target.write_text(yaml.safe_dump(settings, sort_keys=False))
    target.chmod(0o600)
    # Explicit flags override any inherited environment for this preparation.
    for attr, key in [("dataset", "HM3D_ROOT"), ("models", "SAM_MODEL_DIR"), ("cache", "HF_SHARED_CACHE"),
                      ("workspace", "WORKSPACE_ROOT"), ("pal_bundle", "TIAGO_ISO_DIR"), ("baseline_repos", "BASELINE_REPOS_ROOT")]:
        if getattr(args, attr, None):
            env[key] = str(path_from(getattr(args, attr), root))
    env, _ = environment(root, args.local, env)
    for name in ("results", "maps", "ws", "schedules"):
        (Path(env["WORKSPACE_ROOT"]) / name).mkdir(parents=True, exist_ok=True)
    modes = ("sim", "gazebo", "tiago", "baselines", "cloud") if args.mode == "all" else (args.mode,)
    for mode in modes:
        if mode in ("sim", "cloud", "baselines"):
            ensure_image(root, env, rebuild=args.rebuild)
        if mode == "sim":
            for key in ("SAM_MODEL_DIR", "HF_SHARED_CACHE"):
                Path(env[key]).mkdir(parents=True, exist_ok=True)
            env["HF_HOME"] = env["HF_SHARED_CACHE"]
            env["GRAPHAPI_PREPARE"] = "1"
            result = run_python(root, env, [str(root / "graphapi_cli/prepare_models.py"), env["SAM_MODEL_DIR"]])
            if result:
                raise ValueError("model preparation failed; see the downloader's error above")
            dataset = env.get("HM3D_ROOT")
            if not dataset or not Path(dataset).is_dir():
                raise ValueError("register your licensed scene library: graphapi setup sim --dataset /path/containing/scene_datasets")
        elif mode == "gazebo":
            ensure_gazebo_image(root, env)
        elif mode == "tiago":
            from .launch import runtime
            container = args.container or env.get("GRAPHAPI_PAL_CONTAINER", env["TIAGO_DOCKER_TARGET"] + "-dev")
            # Existing creation, helper sync and workspace dependency installation are preserved.
            result = subprocess.run(["bash", str(runtime(root, "run_tiago")), "build", container], cwd=root, env=env).returncode
            if result:
                raise ValueError("private PAL preparation failed; see TIAGO_ISO/README.md")
        elif mode == "baselines":
            base = env.get("BASELINE_REPOS_ROOT")
            if not base:
                raise ValueError("set --baseline-repos to the directory containing the existing external baseline repositories")
            base = Path(base)
            for name in ("Clio-Baseline", "HOV-Baseline"):
                if not (base / name).is_dir():
                    raise ValueError(f"missing external baseline checkout: {base / name}")
            for name, variable, default in (("Clio-Baseline", "CLIO_IMAGE", "clio-baseline:noetic"),
                                             ("HOV-Baseline", "HOV_IMAGE", "hov-baseline:clean")):
                image = env.get(variable, default)
                if subprocess.run(["docker", "image", "inspect", image], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode:
                    candidates = [base / name / "Dockerfile", base / name / "docker/Dockerfile"]
                    dockerfile = next((path for path in candidates if path.is_file()), None)
                    if dockerfile is None:
                        raise ValueError(f"supply the existing {image} image or a standard Dockerfile in {base / name}; its native models/cache remain external inputs")
                    subprocess.run(["docker", "build", "-f", str(dockerfile), "-t", image, str(base / name)], check=True)
            subprocess.run([sys.executable, str(root / "tools/baselines/install_entrypoints.py"),
                            "--integration-root", str(root), "--clio-root", str(base / "Clio-Baseline"),
                            "--hovsg-root", str(base / "HOV-Baseline")], check=True)
        print(f"prepared {mode}; local configuration: {target}")
