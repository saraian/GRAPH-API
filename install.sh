#!/usr/bin/env bash
# RUN THIS ONCE. It is safe to run again: every step checks before it acts.
#
# Owner, 2026-09-10: "ONE config file (which anyway should be in the project root), ONE install
# script and ONE launch script." This is the install script. It replaces the README's Install
# section, which was 62 lines of manual steps.
#
# It does five things and prints what it did:
#   1. build the container image
#   2. check the conda environment the host renderer needs
#   3. cache every model, and PROVE it by loading each one with the hub switched off
#   4. create your local settings file and NAME the values you must fill in
#   5. verify all of it, and refuse with a cause rather than half-installing
#
# WHY STEP 3 PROVES RATHER THAN WARMS. A warm cache shows a download happened once. Loading with
# HF_HUB_OFFLINE=1 shows none will happen DURING a run, which is what the owner asked for: "No
# in-run fetches." The difference is not academic -- a cache in the old flat layout is found by a
# file search and NOT by the loader, which reads $HF_HOME/hub and nothing else. That exact shape
# produced a false pass on this machine on 2026-09-10.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FAIL=0
step() { printf '\n== %s\n' "$1"; }
ok()   { printf '   ok    %s\n' "$1"; }
bad()  { printf '   FAIL  %s\n' "$1"; FAIL=1; }
found(){ printf '   found %-18s %s\n' "$1" "$2"; }

# EVERY VALUE IS DISCOVERED, NOT ASSUMED, and anything already set in the environment wins.
# Owner, 2026-09-10: "Can't we have the values prefilled to work straight away on any machine
# (ofc customizable)?" A hardcoded default is right on exactly one machine -- that is how
# WORKSPACE_ROOT came to compute to "/" and how HM3D_ROOT pointed at a path only one host has.
# So each value below is SEARCHED FOR, the first real hit wins, and the search order is printed.
_first_dir() { for d in "$@"; do [ -d "$d" ] && { echo "$d"; return; }; done; }
_first_file_dir() { # $1 = filename to find, rest = roots to search, shallowly
  local f="$1"; shift
  for r in "$@"; do
    [ -d "$r" ] || continue
    local hit; hit="$(find "$r" -maxdepth 4 -name "$f" -print -quit 2>/dev/null)"
    [ -n "$hit" ] && { dirname "$hit"; return; }
  done
}

step "0/5  finding what this machine already has"

# The container image: an exact tag first, then ANY graphapi-run image, so a tag like
# graphapi-run:humble-ga290 is used instead of triggering a 16 GB rebuild beside it.
if [ -n "${IMAGE_TAG:-}" ]; then found IMAGE_TAG "$IMAGE_TAG (from the environment)"
elif docker image inspect graphapi-run:humble >/dev/null 2>&1; then IMAGE_TAG=graphapi-run:humble; found IMAGE_TAG "$IMAGE_TAG"
else
  IMAGE_TAG="$(docker images --format '{{.Repository}}:{{.Tag}}' 2>/dev/null | /bin/grep -E '^graphapi-run:' | head -1)"
  [ -n "$IMAGE_TAG" ] && found IMAGE_TAG "$IMAGE_TAG (the only one present)" || { IMAGE_TAG=graphapi-run:humble; found IMAGE_TAG "$IMAGE_TAG (none present; will build)"; }
fi

# The host renderer. miniconda, anaconda and mambaforge all appear on our machines.
CONDA_ENV="${CONDA_ENV:-habitat_env}"
if [ -z "${CONDA_PY:-}" ]; then
  for _base in "$HOME/miniconda3" "$HOME/anaconda3" "$HOME/mambaforge" "$HOME/miniforge3" /opt/conda; do
    [ -x "$_base/envs/$CONDA_ENV/bin/python" ] && { CONDA_PY="$_base/envs/$CONDA_ENV/bin/python"; break; }
  done
  CONDA_PY="${CONDA_PY:-$HOME/miniconda3/envs/$CONDA_ENV/bin/python}"
fi
found CONDA_PY "$CONDA_PY"

# The scene library: the directory that CONTAINS the scene datasets, wherever it sits.
if [ -z "${HM3D_ROOT:-}" ]; then
  HM3D_ROOT="$(_first_dir "$HOME/Musumeci/habitat_matterport/hm3d_example" /DATA/habitat_matterport/hm3d_example \
                          "$HOME/habitat_matterport/hm3d_example" "$HERE/../habitat_matterport/hm3d_example")"
  [ -z "$HM3D_ROOT" ] && HM3D_ROOT="$(_first_file_dir scene_datasets "$HOME/Musumeci/habitat_matterport" /DATA/habitat_matterport "$HOME/habitat_matterport")"
fi
[ -n "${HM3D_ROOT:-}" ] && found HM3D_ROOT "$HM3D_ROOT" || bad "no scene library found. Set HM3D_ROOT to the directory holding scene_datasets/."

# The segmenter weights: find the encoder file and take its directory.
if [ -z "${SAM_MODEL_DIR:-}" ]; then
  SAM_MODEL_DIR="$(_first_file_dir l2_encoder.onnx "$HOME/Musumeci/models" /DATA/models "$HOME/models" "$HERE/lost3dsg/src/perception_module/utils")"
fi
[ -n "${SAM_MODEL_DIR:-}" ] && found SAM_MODEL_DIR "$SAM_MODEL_DIR" || bad "no l2_encoder.onnx found. Set SAM_MODEL_DIR to the directory holding the two EfficientViT-SAM files."

# The model cache: a directory that already has the hub layout beats an empty candidate.
if [ -z "${HF_SHARED_CACHE:-}" ]; then
  for _c in "$HOME/Musumeci/gin_data/.hf_cache" /DATA/huggingface_cache "$HOME/.cache/huggingface" "$HOME/hf_cache"; do
    [ -d "$_c/hub" ] && { HF_SHARED_CACHE="$_c"; break; }
  done
  HF_SHARED_CACHE="${HF_SHARED_CACHE:-$HOME/.cache/huggingface}"
fi
found HF_SHARED_CACHE "$HF_SHARED_CACHE"

# The workspace. If none exists, CREATE one rather than refuse: this is the value that must never
# be derived, and a machine with nowhere to put a bundle is a machine that cannot run at all.
if [ -z "${WORKSPACE_ROOT:-}" ]; then
  for _w in "$HOME/Musumeci/gin_data" /DATA/workspace "$HOME/graphapi_workspace"; do
    { [ -d "$_w/maps" ] || [ -d "$_w/runs" ] || [ -d "$_w/results" ]; } && { WORKSPACE_ROOT="$_w"; break; }
  done
  if [ -z "${WORKSPACE_ROOT:-}" ]; then
    WORKSPACE_ROOT="$HOME/graphapi_workspace"
    mkdir -p "$WORKSPACE_ROOT"/{maps,runs,results}
    found WORKSPACE_ROOT "$WORKSPACE_ROOT (created: maps/ runs/ results/)"
  else found WORKSPACE_ROOT "$WORKSPACE_ROOT"; fi
else found WORKSPACE_ROOT "$WORKSPACE_ROOT (from the environment)"; fi

step "1/5  container image ($IMAGE_TAG)"
if docker image inspect "$IMAGE_TAG" >/dev/null 2>&1; then
  ok "already built. Delete it to rebuild: docker rmi $IMAGE_TAG"
else
  # MEASURED ON GIN, 2026-09-10: the image was there as graphapi-run:humble-ga290, so the exact-tag
  # test missed it and this step would have started a 16 GB build on a box at 90% disk. Name what
  # exists rather than build a second copy of it.
  OTHER="$(docker images --format '{{.Repository}}:{{.Tag}}' | /bin/grep -E '^graphapi-run:' | head -1 || true)"
  if [ -n "$OTHER" ]; then
    bad "no image tagged $IMAGE_TAG, but $OTHER exists. Set IMAGE_TAG=$OTHER, or delete it to build fresh."
  else
  echo "   building — this pulls ROS 2 Humble, rtabmap, navigation2 and torch, about 16 GB"
  docker build -t "$IMAGE_TAG" "$HERE" || bad "docker build failed; the output above says why"
  docker image inspect "$IMAGE_TAG" >/dev/null 2>&1 && ok "built"
  fi
fi

step "2/5  host renderer (conda env '$CONDA_ENV')"
# The launcher calls this interpreter by absolute path, so a env on a different prefix is not a
# substitute -- it must be THIS path or the launcher must be told another one.
if [ -x "$CONDA_PY" ]; then
  if "$CONDA_PY" -c 'import habitat_sim' 2>/dev/null; then ok "$CONDA_PY imports habitat_sim"
  else bad "$CONDA_PY exists but cannot import habitat_sim — install per https://github.com/facebookresearch/habitat-sim"; fi
else
  bad "no interpreter at $CONDA_PY. Create the env (conda create -n $CONDA_ENV python=3.9), install habitat-sim into it, or set CONDA_PY."
fi

step "3/5  models cached at $HF_SHARED_CACHE, and proved offline"
mkdir -p "$HF_SHARED_CACHE"
# BOTH dinov2 SIZES ON PURPOSE. visual_reid.py picks -base over -small on measured VRAM, so caching
# the declared default alone leaves the run fetching the one it actually chooses. Measured
# 2026-09-10: neither size was cached anywhere on this host while every run loaded one of them.
HUB_MODELS="google/owlv2-base-patch16-ensemble sentence-transformers/all-MiniLM-L6-v2 facebook/dinov2-small facebook/dinov2-base"
# THE CONTAINER IS THE AUTHORITY, not this shell. Measured on Gin 2026-09-10: the host python has
# no huggingface_hub, so a check run here reported every model missing on a machine where all four
# load correctly at run time. The run loads them inside the container, from the mount at /models/hf,
# so that is where the question must be asked. Same lesson as the /ext fault: ask what answers at
# the path the run actually reads.
if docker image inspect "$IMAGE_TAG" >/dev/null 2>&1; then
  # --entrypoint python3 ON PURPOSE. The image's entrypoint colcon-builds the package from the
  # /graph_api mount, so `docker run <image> python3 ...` runs the BUILD, not python, and fails on
  # "cannot stat /graph_api/lost3dsg". Measured on Gin 2026-09-10. This check needs the image's
  # interpreter and its site-packages, not its build.
  docker run --rm --entrypoint python3 \
      -v "$HF_SHARED_CACHE":/models/hf -e HF_HOME=/models/hf \
      -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 "$IMAGE_TAG" \
      -c '
import sys
names = ["google/owlv2-base-patch16-ensemble", "sentence-transformers/all-MiniLM-L6-v2",
         "facebook/dinov2-small", "facebook/dinov2-base"]
try:
    from huggingface_hub import snapshot_download
except Exception as e:
    print(f"   the image has no huggingface_hub ({e}); cannot prove the cache")
    sys.exit(3)
missing = []
for n in names:
    try:
        snapshot_download(n, local_files_only=True)
        print(f"   resolves offline: {n}")
    except Exception as e:
        missing.append(f"{n}: {type(e).__name__}")
if missing:
    print("   NOT RESOLVABLE OFFLINE: " + "; ".join(missing))
    print("   Fetch them into the cache, then run this again. A flat blobs-and-refs cache at the")
    print("   root is found by a file search and NOT by the loader, which reads $HF_HOME/hub.")
    sys.exit(4)
print("   all four hub models resolve inside the container with the hub switched off")
' || bad "the models do not load inside the container (see above)"
else
  bad "cannot check the model cache without the image"
fi
# The two segmenter files come from disk, not the hub, and the gate asserts them by path.
SAM_DIR="$SAM_MODEL_DIR"
for f in l2_encoder.onnx l2_decoder.onnx; do
  [ -s "$SAM_DIR/$f" ] && ok "$SAM_DIR/$f" || bad "missing $SAM_DIR/$f (set SAM_MODEL_DIR to where the EfficientViT-SAM files are)"
done

step "4/5  writing your settings"
# PREFILLED WITH WHAT THIS MACHINE HAS, and every line is yours to edit afterwards. An existing file is
# never overwritten -- a person who has customised it must not lose that by running this again.
LOCAL="$HERE/lost3dsg/test/env.local.sh"
if [ -f "$LOCAL" ]; then
  ok "$LOCAL exists — left alone. Delete it to regenerate."
else
  cat > "$LOCAL" <<EOF
# Written by install.sh on $(date -Is). EDIT FREELY: this file is yours, it is gitignored, and
# install.sh will not overwrite it. Every value was discovered on this machine.
export WORKSPACE_ROOT=$WORKSPACE_ROOT
export HF_SHARED_CACHE=$HF_SHARED_CACHE
export HM3D_ROOT=$HM3D_ROOT
export SAM_MODEL_DIR=$SAM_MODEL_DIR
export IMAGE_TAG=$IMAGE_TAG
export CONDA_PY=$CONDA_PY

# THE LABELLING ENDPOINT IS THE ONE VALUE NO SEARCH CAN FIND, because it is a CREDENTIAL: the URL
# alone spends the account's GPU budget, so it is never committed anywhere (GA-319). Leave it
# unset to run the LOCAL detector instead -- install.sh has configured that below.
# export MODAL_PERCEPTION_URL=https://<workspace>--lost3dsg-predict.modal.run
EOF
  chmod 600 "$LOCAL"
  ok "wrote $LOCAL with the values above"
fi

# THE BACKEND. Both shipped configs set perception.backend to "modal", which needs the endpoint.
# Without it probe a4 skips and a skipped probe FAILS the gate -- so a machine with no credential
# could not run at all. Writing the local backend into config.local.yaml makes it run straight
# away, and config.local.yaml is the documented per-machine override: gitignored, merged key by
# key, and it PRINTS which keys it changed, so no run is silently different from the tracked file.
CFG_LOCAL="$HERE/lost3dsg/src/perception_module/config.local.yaml"
# shellcheck disable=SC1090
[ -f "$LOCAL" ] && . "$LOCAL" 2>/dev/null || true
if [ -n "${MODAL_PERCEPTION_URL:-}" ]; then
  ok "labelling endpoint set — the shipped modal backend will be used"
elif [ -f "$CFG_LOCAL" ]; then
  ok "$CFG_LOCAL exists — left alone"
else
  # THE SEGMENTER PATHS GO IN TOO, and without them the local backend cannot start. The tracked
  # config leaves paths.vitsam_* EMPTY and falls back to <package>/utils/l2_encoder.onnx, which is
  # not in the image -- so probe a4 refuses with "the segmenter files do not resolve". Measured on
  # Gin 2026-09-10 with the default config. /models/vitsam is where the launcher mounts
  # SAM_MODEL_DIR, so these are the paths the CONTAINER will see, not this machine's.
  { printf 'perception:\n  backend: "local"   # written by install.sh: no MODAL_PERCEPTION_URL on this machine\n'
    printf 'paths:\n'
    printf '  vitsam_encoder: "/models/vitsam/l2_encoder.onnx"   # the container mount of SAM_MODEL_DIR\n'
    printf '  vitsam_decoder: "/models/vitsam/l2_decoder.onnx"\n'
  } > "$CFG_LOCAL"
  ok "no endpoint, so wrote $CFG_LOCAL with backend: local (the detector runs in the container)"
fi

step "5/5  verify"
# A VALUE THE CALLER SET WINS OVER THE FILE, and the message says which answered. Measured on Gin
# 2026-09-10: this sourced env.local.sh AFTER the caller had passed WORKSPACE_ROOT, so it reported
# a path the caller never asked for and the cause read as the caller's mistake.
WS_FROM="the caller"; WS="${WORKSPACE_ROOT:-}"
# shellcheck disable=SC1090
[ -f "$LOCAL" ] && . "$LOCAL" 2>/dev/null || true
if [ -z "$WS" ]; then WS="${WORKSPACE_ROOT:-}"; WS_FROM="$LOCAL"; fi
if [ -z "$WS" ]; then bad "WORKSPACE_ROOT is set neither in the environment nor in $LOCAL"
elif [ ! -d "$WS/maps" ] && [ ! -d "$WS/runs" ] && [ ! -d "$WS/results" ]; then
  bad "WORKSPACE_ROOT=$WS (from $WS_FROM) holds no maps/, runs/ or results/ — the launcher refuses this rather than writing a bundle where nobody looks"
else ok "workspace $WS (from $WS_FROM)"; fi
# THE EFFECTIVE BACKEND, not the presence of a URL. Step 4 writes config.local.yaml with
# backend: local when there is no endpoint, so checking for the URL here contradicted the step
# above and made INSTALLED unreachable on a machine with no credential. Ask what the run will
# actually use.
# THE TRAILING COMMENT BROKE THE FIRST VERSION: `backend: "local"   # written by install.sh`
# survived tr and cut as `local#writtenby...`, so the check failed on the file this script had
# just written. Strip the comment before reading the value.
_backend="$( { [ -f "$CFG_LOCAL" ] && /bin/grep -hE '^[[:space:]]+backend:' "$CFG_LOCAL"; } 2>/dev/null \
             | head -1 | sed 's/#.*//' | cut -d: -f2 | tr -d ' "'"'"'\t')"
if [ -n "${MODAL_PERCEPTION_URL:-}" ]; then ok "labelling endpoint set; the modal backend will be used"
elif [ "$_backend" = "local" ]; then ok "no endpoint, and the backend is local — the detector runs in the container"
else bad "no labelling endpoint and the backend is not local. Set MODAL_PERCEPTION_URL in $LOCAL, or perception.backend to \"local\" in $CFG_LOCAL."
fi

# A MISSING DISPLAY IS NOT A FAILURE, IT IS A CHOICE OF SCRIPT. rviz, the preview window and the
# overlay all default to on and each aborts without an X server -- measured on Gin 2026-09-10,
# where six runs passed their gate and then died on rviz. run_headless.sh exists for exactly this,
# so name it rather than refusing the install.
if [ -n "$(ls /tmp/.X11-unix/ 2>/dev/null)" ]; then
  ok "an X display is available ($(ls /tmp/.X11-unix/ | tr '\n' ' ')) — use ./run.sh"
else
  ok "no X display — USE ./run_headless.sh, not ./run.sh. It turns rviz and the preview window off;
         ./run.sh would pass its gate and then die when rviz aborts with no display."
fi

if [ "$FAIL" = "0" ]; then
  if [ -n "$(ls /tmp/.X11-unix/ 2>/dev/null)" ]; then
    printf '\nINSTALLED. Next:  ./run.sh\n'
  else
    printf '\nINSTALLED. Next:  ./run_headless.sh          (this machine has no display)\n'
  fi
else
  printf '\nNOT INSTALLED. Fix the FAIL lines above and run this again — it is safe to repeat.\n'
fi
exit "$FAIL"
