"""One configuration resolution path, without importing ROS or inference libraries."""
from __future__ import annotations

import copy
import os
from pathlib import Path

import yaml


class ConfigurationError(ValueError):
    pass


def workflow_name(mode):
    return {"tiago": "tiago-physical", "bag": "tiago-bag",
            "launch-simulation": "tiago-gazebo", "launch-simulation2": "tiago-navigation",
            "launch-bag-slam": "tiago-bag-slam", "launch-bag-slam-1": "tiago-bag-slam-1",
            "launch-bag-slam-2": "tiago-bag-slam-2"}.get(mode, mode)


def tiago_map_source(mode, selected, env):
    if mode not in ("tiago", "bag"):
        return None
    allowed = ("slam", "robot") if mode == "tiago" else ("slam", "recorded")
    if selected is not None:
        if selected not in allowed:
            raise ConfigurationError(f"{workflow_name(mode)} map source must be one of {allowed}")
        return selected
    start = env.get("FOUND_START_RTABMAP", "1" if mode == "tiago" else "0")
    if start not in ("0", "1"):
        raise ConfigurationError("FOUND_START_RTABMAP must be 0 or 1")
    return "slam" if start == "1" else ("robot" if mode == "tiago" else "recorded")


def node_profile(mode, map_source=None):
    if mode == "tiago":
        return "config/tiago_robot.yaml"
    if mode == "bag":
        return "config/tiago_bag_rtabmap.yaml" if map_source == "slam" else "config/tiago_bag.yaml"
    return None


class UniqueLoader(yaml.SafeLoader):
    pass


def _mapping(loader, node, deep=False):
    result = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if key in result:
            raise ConfigurationError(f"duplicate YAML key: {key}")
        result[key] = loader.construct_object(value_node, deep=deep)
    return result


UniqueLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _mapping)


def read_yaml(path: Path, optional=False):
    if optional and not path.exists():
        return {}
    try:
        value = yaml.load(path.read_text(), Loader=UniqueLoader)
    except (OSError, yaml.YAMLError) as exc:
        raise ConfigurationError(f"{path}: {exc}") from exc
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ConfigurationError(f"{path}: expected a YAML mapping")
    return value


def merge(base, override):
    out = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(out.get(key), dict) and isinstance(value, dict):
            out[key] = merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def project_root(explicit=None):
    if explicit:
        root = Path(explicit).expanduser()
        if not root.is_absolute() and os.environ.get("GRAPHAPI_HOST_CWD"):
            root = Path(os.environ["GRAPHAPI_HOST_CWD"]) / root
        root = root.resolve()
    else:
        candidates = [Path.cwd(), *Path.cwd().parents, Path(__file__).resolve().parent.parent]
        root = next((p for p in candidates if (p / "config/graphapi.yaml").is_file()), None)
        if root is None:
            raise ConfigurationError("project not found; pass --project /path/to/GRAPH-API")
    if not (root / "lost3dsg").is_dir():
        raise ConfigurationError(f"{root}: not a GRAPH-API checkout")
    return root


def path_from(value, base):
    path = Path(str(value)).expanduser()
    return (path if path.is_absolute() else base / path).resolve()


PATH_ENV = {
    "workspace": "WORKSPACE_ROOT", "dataset": "HM3D_ROOT", "models": "SAM_MODEL_DIR",
    "cache": "HF_SHARED_CACHE", "bags": "TIAGO_BAG_DIR", "pal_bundle": "TIAGO_ISO_DIR",
    "baseline_repos": "BASELINE_REPOS_ROOT",
}


def local_settings(root, selected=None):
    path = path_from(selected, root) if selected else root / "config/local.yaml"
    settings = read_yaml(path, optional=True)
    allowed = {"paths", "docker", "environment", "overrides"}
    unknown = set(settings) - allowed
    if unknown:
        raise ConfigurationError(f"{path}: unknown local sections: {', '.join(sorted(unknown))}")
    for key in allowed:
        if key in settings and not isinstance(settings[key], dict):
            raise ConfigurationError(f"{path}: {key} must be a mapping")
    unknown = set(settings.get("paths", {})) - set(PATH_ENV)
    if unknown:
        raise ConfigurationError(f"{path}: unknown paths: {', '.join(sorted(unknown))}")
    return path, settings


def environment(root, selected=None, inherited=None):
    """Explicit caller environment wins over registered machine defaults."""
    local_path, settings = local_settings(root, selected)
    env = dict(os.environ if inherited is None else inherited)
    for key, value in settings.get("environment", {}).items():
        if not isinstance(key, str) or not key.isidentifier() or isinstance(value, (dict, list)):
            raise ConfigurationError(f"{local_path}: invalid environment entry {key}")
        env.setdefault(key, str(value))
    defaults = {"workspace": root, "models": root / "models/efficientvit_sam",
                "cache": root / "models/huggingface", "bags": root / "bags",
                "pal_bundle": root / "TIAGO_ISO"}
    for key, env_key in PATH_ENV.items():
        value = settings.get("paths", {}).get(key, defaults.get(key))
        if value is not None:
            env.setdefault(env_key, str(path_from(value, local_path.parent)))
        if env.get(env_key):
            env[env_key] = str(path_from(env[env_key], root))
    docker = settings.get("docker", {})
    unknown = set(docker) - {"sim_image", "pal_container", "pal_target"}
    if unknown:
        raise ConfigurationError(f"{local_path}: unknown docker settings: {', '.join(sorted(unknown))}")
    env.setdefault("IMAGE_TAG", str(docker.get("sim_image", "graphapi-sim:latest")))
    env.setdefault("TIAGO_DOCKER_TARGET", str(docker.get("pal_target", "tiago-127")))
    if docker.get("pal_container"):
        env.setdefault("GRAPHAPI_PAL_CONTAINER", str(docker["pal_container"]))
    env["GRAPHAPI_ENV_KEYS"] = ",".join(settings.get("environment", {}))
    env["GRAPHAPI_ROOT"] = str(root)
    env["GRAPHAPI_LOCAL_CONFIG"] = str(local_path)
    env["GRAPHAPI_MANAGED"] = "1"
    return env, settings


def resolve_config(root, selected=None, local=None, overrides=None):
    """Preserve the existing defaults + selected YAML merge, then merge local overrides once."""
    path = path_from(selected, root) if selected else root / "config/graphapi.yaml"
    original = path
    aliases = read_yaml(root / "config/aliases.yaml", optional=True)
    try:
        relative = str(path.relative_to(root))
    except ValueError:
        relative = None
    if relative in aliases:
        path = root / aliases[relative]
    cfg = read_yaml(path)
    # Import _DEFAULTS without executing config.py or importing CUDA/ROS modules.
    import ast
    module = ast.parse((root / "lost3dsg/src/perception_module/config.py").read_text())
    defaults = next(ast.literal_eval(node.value) for node in module.body
                    if isinstance(node, ast.Assign)
                    and any(isinstance(t, ast.Name) and t.id == "_DEFAULTS" for t in node.targets))
    cfg = merge(defaults, cfg)
    # Explicitly preserve old deployment overlays during migration; never source shell settings.
    adjacent = original.parent / "config.local.yaml"
    if not adjacent.exists() and selected is None:
        adjacent = root / "lost3dsg/src/perception_module/config.local.yaml"
    if adjacent.exists():
        cfg = merge(cfg, read_yaml(adjacent))
    _, settings = local_settings(root, local)
    cfg = merge(cfg, settings.get("overrides", {}))
    if overrides:
        cfg = merge(cfg, overrides)
    for key, value in cfg.items():
        if isinstance(defaults.get(key), dict) and not isinstance(value, dict):
            raise ConfigurationError(f"{path}: {key} must be a mapping")
    mode = cfg.get("habitat", {}).get("localization_mode")
    if mode not in ("ground_truth", "rtabmap"):
        raise ConfigurationError(f"{path}: habitat.localization_mode must be ground_truth or rtabmap")
    return path, cfg


def initialize(root, selected=None):
    target = path_from(selected, root) if selected else root / "config/local.yaml"
    if not target.exists():
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text((root / "config/local.example.yaml").read_text())
        target.chmod(0o600)
    return target
