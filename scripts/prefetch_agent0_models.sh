#!/bin/bash
# Pre-fetch all external model assets that cap-x agent0 + Qwen evaluation
# needs. Compute nodes on Leonardo have no outbound internet, so this MUST
# be run from the login node (which does).
#
# What this script does NOT need to download:
#   - Qwen3.6-27B           : already at $MODELS_ROOT/Qwen3.6-27B
#   - Contact GraspNet      : vendored in capx/third_party/contact_graspnet_pytorch
#
# What it downloads:
#   - facebook/sam3              -> $HF_HOME (sam3.pt + config.json, ~few GB)
#   - allenai/Molmo2-8B          -> $HF_HOME (object→pixel-point VLM, ~35 GB)
#   - panda_description URDF     -> $ROBOT_DESCRIPTIONS_CACHE
#
# After this completes, the run_agent0_qwen36.sh script reads from these
# paths with HF_HUB_OFFLINE=1, so the compute node never reaches out.

set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

# Model-cache root — the SAME knob the run scripts use, so we download to
# exactly where they read from. Default scratch; override to a healthy FS:
#   CAPX_CACHE_ROOT=$WORK/zwang003 bash scripts/prefetch_agent0_models.sh
# (HF_HOME is $CAPX_CACHE_ROOT/cache/huggingface — matches the run scripts.
#  The old default was $MODELS_ROOT/hf_cache, which did NOT match them.)
# An explicit CAPX_CACHE_ROOT wins even over an HF_HOME exported by ~/.bashrc,
# so we download to exactly the root you asked for (not the profile's scratch).
if [[ -n "${CAPX_CACHE_ROOT:-}" ]]; then
    MODELS_ROOT="$CAPX_CACHE_ROOT/models"
    export HF_HOME="$CAPX_CACHE_ROOT/cache/huggingface"
    export ROBOT_DESCRIPTIONS_CACHE="$CAPX_CACHE_ROOT/models/robot_descriptions"
else
    CAPX_CACHE_ROOT="/leonardo_scratch/fast/EUHPC_D33_222/zwang003"
    MODELS_ROOT="${MODELS_ROOT:-$CAPX_CACHE_ROOT/models}"
    export HF_HOME="${HF_HOME:-$CAPX_CACHE_ROOT/cache/huggingface}"
    export ROBOT_DESCRIPTIONS_CACHE="${ROBOT_DESCRIPTIONS_CACHE:-$MODELS_ROOT/robot_descriptions}"
fi

mkdir -p "$HF_HOME" "$ROBOT_DESCRIPTIONS_CACHE"

# Sanity: must have outbound HTTPS (we are on a login node)
if ! curl -sSf -m 10 https://huggingface.co/api/models/facebook/sam3 >/dev/null 2>&1; then
    echo "ERROR: cannot reach huggingface.co. Run this on the login node." >&2
    exit 1
fi

source .venv/bin/activate

echo "=== [1/3] facebook/sam3 -> $HF_HOME ==="
python - <<'PY'
import os
from huggingface_hub import hf_hub_download
repo = "facebook/sam3"
for fn in ("config.json", "sam3.pt"):
    p = hf_hub_download(repo_id=repo, filename=fn)
    print(f"  {fn} -> {p}")
PY

echo ""
echo "=== [2/3] allenai/Molmo2-8B -> $HF_HOME ==="
python - <<'PY'
from huggingface_hub import snapshot_download
p = snapshot_download(repo_id="allenai/Molmo2-8B", max_workers=8)
print(f"  -> {p}")
PY

echo ""
echo "=== [3/3] robot_descriptions: panda_description -> $ROBOT_DESCRIPTIONS_CACHE ==="
python - <<'PY'
from robot_descriptions.loaders.yourdfpy import load_robot_description
urdf = load_robot_description("panda_description")
print(f"  loaded: {len(urdf.actuated_joint_names)} joints")
PY

echo ""
echo "=== Done. Cache sizes ==="
du -sh "$HF_HOME" "$ROBOT_DESCRIPTIONS_CACHE"

cat <<EOF

Persist these for the eval node:

  export HF_HOME=$HF_HOME
  export HF_HUB_OFFLINE=1
  export TRANSFORMERS_OFFLINE=1
  export ROBOT_DESCRIPTIONS_CACHE=$ROBOT_DESCRIPTIONS_CACHE

(scripts/run_agent0_qwen36.sh sets these automatically.)
EOF
