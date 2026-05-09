#!/bin/bash
# Start a local vLLM server for allenai/Molmo2-8B on port 8122. Used by
# capx.integrations.vision.molmo.init_molmo() for object→pixel-point detection
# inside reduced-API skills (e.g. plan_grasp).
#
# Runs on the eval node, pinned to a dedicated GPU (default: 2). vLLM lives in
# the `vllm` conda env, not the cap-x .venv.
#
# Usage:
#   bash scripts/serve_molmo_local.sh           # foreground
#   nohup bash scripts/serve_molmo_local.sh &   # background

set -euo pipefail

PORT="${MOLMO_PORT:-8122}"
GPU="${MOLMO_GPU:-2}"
HOST="${MOLMO_HOST:-127.0.0.1}"

# Skip if already up.
if curl -sf -m 3 "http://${HOST}:${PORT}/v1/models" >/dev/null 2>&1; then
    echo "Molmo already serving on ${HOST}:${PORT}"
    exit 0
fi

# Models live on scratch (downloaded by prefetch_agent0_models.sh).
export HF_HOME="${HF_HOME:-/leonardo_scratch/fast/EUHPC_D33_222/zwang003/cache/huggingface}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# vLLM lives in a dedicated conda env (same one used by qwen36_27b/serve_qwen36.sh).
# shellcheck disable=SC1091
source /leonardo/home/userexternal/zwang003/miniconda3/etc/profile.d/conda.sh
conda activate vllm

# Force the system NVML (matches kernel driver) — CUDA toolkit's NVML on
# LD_LIBRARY_PATH otherwise fails to initialize and vLLM's device probe
# crashes with "Device string must not be empty" / "Can't initialize NVML".
export LD_PRELOAD="${LD_PRELOAD:+$LD_PRELOAD:}/usr/lib64/libnvidia-ml.so.1"

# Pin to a single GPU so we don't fight SAM3/GraspNet/MuJoCo for memory.
export CUDA_VISIBLE_DEVICES="$GPU"

# 8192 tokens is plenty for "Point at <object>" prompts; keeps KV cache small.
# --max-num-batched-tokens must be >= max image-feature tokens (Molmo emits
# ~8134 tokens per image), otherwise vLLM refuses to start.
exec vllm serve "allenai/Molmo2-8B" \
    --served-model-name "allenai/Molmo2-8B" \
    --port "$PORT" \
    --host "$HOST" \
    --tensor-parallel-size 1 \
    --gpu-memory-utilization 0.85 \
    --dtype bfloat16 \
    --max-model-len 8192 \
    --max-num-batched-tokens 16384 \
    --trust-remote-code
