"""Runtime configuration for the perception module.

Everything here used to be a constant somewhere in the code: the model id, the
output path, the camera frame convention, the VLM timeout. Constants are fine
until someone runs this on a different robot, a different provider, or in CI,
at which point each one needs a patch.

Values are read, in order of precedence:

  1. an explicit path passed to load_config()
  2. $PERCEPTION_CONFIG
  3. perception_config.yaml next to this file
  4. the defaults below

Nothing here reaches the network or the filesystem at import time.
"""
import os
import threading

try:
    import yaml
except ImportError:  # config file support is optional, defaults still work
    yaml = None

DEFAULTS = {
    "vlm": {
        # Provider model id. Was pinned to "gpt-5-nano" in cv_utils.
        "model": "gpt-5-nano",
        "base_url": None,
        # Seconds. Without this a single slow response stalls the whole
        # detection cycle before it reaches the /bbox_3d publish.
        "timeout_s": 20.0,
        "max_retries": 2,
        # Set false to skip per-object descriptions entirely. They exist to feed
        # object association; deployments that do not need them pay ~1 call per
        # detected object per cycle for nothing.
        "enabled": True,
        # File holding the API key. Read lazily, only when a call is made.
        "api_key_file": None,
        "api_key_env": "OPENAI_API_KEY",
    },
    "camera": {
        # ---------------------------------------------------------------
        # READ THIS BEFORE CHANGING `frame_convention`.
        #
        # mask_list_to_centroid_and_bbox back-projects pixels using the OPTICAL
        # convention: x right, y down, z forward. The resulting points are then
        # transformed out of the frame named by the incoming image header.
        #
        # So that frame MUST also be optical. On a real TIAGo it is, because the
        # driver publishes head_front_camera_color_optical_frame. A simulator
        # bridge that publishes a ROS BODY frame (x forward, y left, z up) under
        # the same field will project every detection along the wrong ray, and
        # nothing will report an error: objects simply land in the wrong place,
        # typically with height and depth swapped.
        #
        #   "optical" -> trust the image header frame as-is (correct default)
        #   "body"    -> the publisher gives a body frame; insert the fixed
        #                body->optical rotation named below before projecting
        #
        # If you are configuring this for a new robot: publish an optical frame
        # if you can, and only set "body" if you cannot change the publisher.
        # ---------------------------------------------------------------
        "frame_convention": "optical",
        # Applied only when frame_convention == "body". Quaternion (x, y, z, w)
        # taking ROS body axes to optical axes.
        "body_to_optical_quat": [-0.5, 0.5, -0.5, 0.5],
        # Look the transform up at the image's own stamp rather than "latest".
        # Set false only to reproduce the old behaviour.
        "use_image_stamp": True,
        # How far back the TF buffer must reach to still hold that stamp.
        "tf_buffer_s": 10.0,
    },
    "output": {
        # Was hardcoded to /root/exchange/lost3dsg/output/.
        # Relative paths resolve against the package root.
        "dir": "output",
    },
    "models": {
        # Cache for downloaded weights. Downloaded once, then reused.
        "cache_dir": None,   # defaults to ~/.cache/lost3dsg
        "auto_download": True,
        "semantic_backend": "sentence-transformers",
        "sentence_model": "all-MiniLM-L6-v2",
    },
    "exploration": {
        # Prompt on stdin for the exploration->tracking switch. Requires a TTY,
        # so it must be off for launch files, containers and CI.
        "interactive_gate": False,
        # Publish belief markers during exploration too, not only after the
        # switch. Without this an unattended run builds a belief nobody sees.
        "publish_during_exploration": True,
    },
}

_lock = threading.Lock()
_config = None


def _merge(base, override):
    out = dict(base)
    for k, v in (override or {}).items():
        out[k] = _merge(base[k], v) if isinstance(v, dict) and isinstance(base.get(k), dict) else v
    return out


def load_config(path=None):
    """Load and cache the config. Safe to call from anywhere, any number of times."""
    global _config
    with _lock:
        if _config is not None and path is None:
            return _config
        path = path or os.environ.get("PERCEPTION_CONFIG") or \
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "perception_config.yaml")
        data = {}
        if yaml is not None and os.path.isfile(path):
            with open(path) as f:
                data = yaml.safe_load(f) or {}
        _config = _merge(DEFAULTS, data)
        return _config


def get(section, key):
    return load_config()[section][key]


def package_root():
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def output_dir():
    d = get("output", "dir")
    if not os.path.isabs(d):
        d = os.path.join(package_root(), d)
    os.makedirs(d, exist_ok=True)
    return d


def model_cache_dir():
    d = get("models", "cache_dir") or os.path.join(
        os.path.expanduser("~"), ".cache", "lost3dsg")
    os.makedirs(d, exist_ok=True)
    return d
