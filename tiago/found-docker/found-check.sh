#!/bin/bash
# Verify the Graph API / LOST-3DSG Python and ROS runtime inside this container.
set -euo pipefail
# shellcheck disable=SC1091
. /etc/profile.d/99-found.sh

if [ -f /ws/install/setup.bash ]; then
  # The base image does not source the Graph-API overlay automatically.  Without
  # this, `ros2 pkg list` gives a false negative even after a successful build.
  set +u
  . /ws/install/setup.bash
  set -u
fi

python3 - <<'PY'
import importlib
import os
mods = [
    "numpy", "scipy", "cv2", "PIL", "yaml",
    "torch", "torchvision", "transformers", "sentence_transformers",
    "rdflib", "pyoxigraph", "openai", "fastapi", "gensim",
    "onnxruntime", "timm", "webcolors", "requests",
]
failed = []
for name in mods:
    try:
        importlib.import_module(name)
    except Exception as exc:
        failed.append(f"{name}: {type(exc).__name__}: {exc}")
if failed:
    print("MISSING PYTHON MODULES:")
    print("\n".join(failed))
    raise SystemExit(1)
import torch
print("python runtime ok")
print("  torch", torch.__version__, "cuda", torch.cuda.is_available())
import onnxruntime as ort
ort_providers = ort.get_available_providers()
print("  onnxruntime", ort.__version__, "providers", ort_providers)
require_cuda = os.environ.get("VITSAM_REQUIRE_CUDA", "1").strip().lower() in {
    "1", "true", "yes", "on"
}
if require_cuda and "CUDAExecutionProvider" not in ort_providers:
    raise SystemExit("VitSAM GPU check failed: CUDAExecutionProvider is unavailable")

# Checking only get_available_providers() is insufficient: ONNX Runtime can expose
# CUDAExecutionProvider while a session silently falls back to CPU when the container
# has stale NVIDIA device nodes, missing shared libraries, or an unusable CUDA context.
# Instantiate the same two sessions used by perception_2.py so `check` exercises the
# real boundary that previously failed only after the tmux stack had been launched.
from models import VitSam
from utils import ENCODER_VITSAM_PATH, DECODER_VITSAM_PATH

for model_path in (ENCODER_VITSAM_PATH, DECODER_VITSAM_PATH):
    if not os.path.isfile(model_path):
        raise SystemExit(f"VitSAM model is missing: {model_path}")

try:
    vitsam = VitSam(ENCODER_VITSAM_PATH, DECODER_VITSAM_PATH)
except Exception as exc:
    raise SystemExit(
        f"VitSAM session check failed: {type(exc).__name__}: {exc}"
    ) from exc

encoder_providers = vitsam.encoder.session.get_providers()
decoder_providers = vitsam.decoder.session.get_providers()
print("  VitSAM encoder session", encoder_providers)
print("  VitSAM decoder session", decoder_providers)
if require_cuda and not all(
    "CUDAExecutionProvider" in providers
    for providers in (encoder_providers, decoder_providers)
):
    raise SystemExit(
        "VitSAM GPU check failed: both encoder and decoder sessions must select "
        "CUDAExecutionProvider"
    )
print("  VitSAM sessions ok (device", vitsam.device + ")")

# The normal host wrapper prompts for this credential before it enters Docker,
# but direct calls to found-robot-stack used to launch three failing VLM cycles
# and only then kill perception. Make the stricter startup check opt-in so the
# lightweight `run_tiago.sh check` remains useful without credentials.
require_vlm = os.environ.get("FOUND_CHECK_REQUIRE_VLM", "0").strip().lower() in {
    "1", "true", "yes", "on"
}
if require_vlm:
    from config import CFG
    from cv_utils import _endpoint_is_local, _resolve_api_key

    base_url = CFG.get("vlm", {}).get("base_url", "")
    if not _endpoint_is_local(base_url):
        try:
            _resolve_api_key()
        except Exception as exc:
            raise SystemExit(
                f"VLM credential check failed for {base_url!r}: "
                f"{type(exc).__name__}: {exc}"
            ) from exc
        print("  VLM credential available for", base_url)
    else:
        print("  VLM endpoint is local; no remote credential required")
PY

if command -v ros2 >/dev/null 2>&1; then
  if ros2 pkg prefix lost3dsg >/dev/null 2>&1; then
    echo "lost3dsg ROS package ok"
  else
    echo "note: lost3dsg is not built yet; run found-build-ws"
  fi
else
  echo "note: ros2 is not on PATH in this shell"
fi
