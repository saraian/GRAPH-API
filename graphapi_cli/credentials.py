"""Check VLM credentials before allocating or starting an application workload."""
from pathlib import Path
from urllib.parse import urlparse

from .configuration import ConfigurationError


LOCAL_HOSTS = {"localhost", "127.0.0.1", "::1", "0.0.0.0", "host.docker.internal"}
GEMINI_PROVIDERS = {"gemini", "google", "google_gemini", "vertex", "vertex_ai"}
GEMINI_HOSTS = {"aiplatform.googleapis.com", "generativelanguage.googleapis.com"}


def is_gemini(vlm):
    provider = str(vlm.get("provider", "auto") or "auto").strip().lower()
    if provider in GEMINI_PROVIDERS:
        return True
    if provider in {"openai", "openai_compatible", "openai-compatible"}:
        return False
    return (urlparse(str(vlm.get("base_url", ""))).hostname or "").lower() in GEMINI_HOSTS


def prepare_vlm_credentials(root, cfg, env):
    """Use the inference client's credential order; never log or persist a key."""
    vlm = cfg.get("vlm", {}) or {}
    base_url = str(vlm.get("base_url", ""))
    gemini = is_gemini(vlm)
    names = (("GEMINI_API_KEY", "GOOGLE_API_KEY") if gemini else ()) + (
        "OPENAI_API_KEY", "REGOLO_API_KEY", "OPENROUTER_API_KEY")
    key = str(vlm.get("api_key") or "").strip()
    if not key:
        key = next((str(env[name]).strip() for name in names
                    if str(env.get(name, "")).strip()), "")
    if not key:
        legacy = Path(root) / "lost3dsg/src/perception_module/api.txt"
        if legacy.is_file():
            key = legacy.read_text().strip()
    if not key and urlparse(base_url).hostname not in LOCAL_HOSTS:
        host = urlparse(base_url).hostname or "the configured remote endpoint"
        hint = ("GEMINI_API_KEY" if gemini else
                "REGOLO_API_KEY" if host == "api.regolo.ai" else
                "OPENROUTER_API_KEY" if host == "openrouter.ai" else "OPENAI_API_KEY")
        raise ConfigurationError(
            f"VLM API key is missing for {host}.\n"
            "Run this in the same terminal, then retry:\n"
            f"  export {hint}='your-key'\n"
            "You can also set overrides.vlm.api_key in config/local.yaml."
        )
    if key:
        # Explicit config keys and legacy files must also reach private PAL,
        # where api.txt may belong to a different checkout. Keep the same key
        # precedence after removing secrets from the resolved YAML.
        env["GEMINI_API_KEY" if gemini else "OPENAI_API_KEY"] = key
    env["GRAPHAPI_VLM_CHECKED"] = "1"


def uses_vlm(request, env):
    if request["mode"] == "sim":
        return not (request.get("no_perception") or request.get("mapping_only")
                    or env.get("MAPPING_ONLY") == "1")
    enabled = env.get("FOUND_START_PERCEPTION", "1") != "0"
    if request.get("no_perception"):
        enabled = False
    for flag in request.get("legacy_args", []):
        if flag in ("--perception", "--no-perception"):
            enabled = flag == "--perception"
    return enabled


VLM_TOOLS = {
    "lost3dsg/launch/habitat_launch.py",
    "lost3dsg/src/perception_module/perception_2.py",
    "lost3dsg/src/perception_module/object_manager_6.py",
    "lost3dsg/src/perception_module/object_services.py",
    "lost3dsg/perception_save.py",
}


def prepare_tool_credentials(root, entry, env):
    from .configuration import resolve_config
    path = entry["path"]
    parallel = path == "graphapi_cli/runtime/run_pipelines.sh"
    if parallel and not uses_vlm({"mode": "sim"}, env):
        return None
    if path in VLM_TOOLS or parallel:
        _, cfg = resolve_config(root, env.get("GRAPH_API_CONFIG"), env.get("GRAPHAPI_LOCAL_CONFIG"))
        prepare_vlm_credentials(root, cfg, env)
        return cfg
    elif path == "lost3dsg/src/perception_module/openai_habitat_assistant.py":
        # This separate assistant uses OpenAI rather than the perception VLM.
        key = str(env.get("OPENAI_API_KEY") or "").strip()
        legacy = Path(root) / "lost3dsg/src/perception_module/api.txt"
        if not key and legacy.is_file():
            key = legacy.read_text().strip()
        if not key:
            raise ConfigurationError("OpenAI API key is missing. Before starting the assistant, run: "
                                     "export OPENAI_API_KEY='your-key' (in the same terminal).")
        env["OPENAI_API_KEY"] = key
