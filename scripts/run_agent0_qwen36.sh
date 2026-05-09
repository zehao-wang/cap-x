#!/bin/bash
# Run cap-x agent0 LIBERO eval against a separately-launched Qwen3.6-27B vLLM
# server.
#
# Prereqs (do these once, on the login node):
#   1. sbatch qwen36_27b/serve_qwen36.sh        # boots vLLM on a 4xA100 node
#   2. squeue -u $USER                          # wait for it to be RUNNING
#   3. salloc a GPU compute node and ssh in    # this node runs the eval +
#                                                local SAM3/GraspNet/PyRoKi
#   4. bash scripts/run_agent0_qwen36.sh [debug|full]
#
# The eval node talks to the Qwen node over Leonardo's internal network on
# port 8000; no outbound internet is required, so /network_forward.txt is NOT
# needed.

set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

# Scratch paths for outputs/logs (home is small; /leonardo_scratch is fast).
SCRATCH_ROOT="${SCRATCH_ROOT:-/leonardo_scratch/fast/EUHPC_D33_222}"
LOG_DIR="${LOG_DIR:-$SCRATCH_ROOT/cap-x/logs}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$SCRATCH_ROOT/cap-x/outputs}"
mkdir -p "$LOG_DIR" "$OUTPUT_ROOT"

# ---------------------------------------------------------------------------
# Offline asset paths (populated once by scripts/prefetch_agent0_models.sh
# on the login node — compute node has no outbound internet)
# ---------------------------------------------------------------------------
MODELS_ROOT="${MODELS_ROOT:-/leonardo_scratch/fast/EUHPC_D33_222/zwang003/models}"
# HF_HOME is the user's pre-existing cache (used by prefetch + Qwen serve).
export HF_HOME="${HF_HOME:-/leonardo_scratch/fast/EUHPC_D33_222/zwang003/cache/huggingface}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export ROBOT_DESCRIPTIONS_CACHE="${ROBOT_DESCRIPTIONS_CACHE:-$MODELS_ROOT/robot_descriptions}"

if [[ ! -d "$HF_HOME" || ! -d "$ROBOT_DESCRIPTIONS_CACHE" ]]; then
    echo "ERROR: missing offline caches under $MODELS_ROOT." >&2
    echo "       Run on login node: bash scripts/prefetch_agent0_models.sh" >&2
    exit 1
fi

# ---------------------------------------------------------------------------
# Args / mode
# ---------------------------------------------------------------------------
MODE="${1:-debug}"   # debug | full
NUM_WORKERS="${NUM_WORKERS:-}"

case "$MODE" in
    debug)
        SUITES=( libero_object_swap )
        TOTAL_TRIALS=1
        : "${NUM_WORKERS:=1}"
        DEBUG_FLAG="--args.debug"
        # Run only the first task in the suite (10 → 1) for fast iteration.
        EXTRA_FLAGS="--args.max-tasks-per-suite 1"
        OUTPUT_SUFFIX="debug"
        ;;
    full)
        SUITES=( libero_object_swap libero_object_task libero_goal_swap libero_goal_task libero_spatial_swap libero_spatial_task )
        TOTAL_TRIALS=""   # use all init states (50 each)
        : "${NUM_WORKERS:=8}"
        DEBUG_FLAG=""
        EXTRA_FLAGS=""
        OUTPUT_SUFFIX="full"
        ;;
    *)
        echo "Usage: $0 [debug|full]" >&2
        exit 1
        ;;
esac

# ---------------------------------------------------------------------------
# Locate the Qwen vLLM server
# ---------------------------------------------------------------------------
# Override with: QWEN_HOST=lrdnXXXX QWEN_PORT=8000 bash run_agent0_qwen36.sh ...
QWEN_PORT="${QWEN_PORT:-8000}"
QWEN_MODEL="${QWEN_MODEL:-Qwen3.6-27B}"

# Must match --max-model-len in qwen36_27b/serve_qwen36.sh.
# stress_results.md confirmed inputs up to 98304 tokens completed end-to-end;
# 131072 was never actually validated (skipped because input+output > cap).
# 100000 is the conservative ceiling — close to the proven limit with a small
# safety margin. KV cache has ~10x headroom on 4xA100-64GB if you want to push.
QWEN_MAX_MODEL_LEN="${QWEN_MAX_MODEL_LEN:-100000}"
# Reserve this many tokens for the model's response. Input budget per request
# is therefore (QWEN_MAX_MODEL_LEN - QWEN_MAX_TOKENS).
QWEN_MAX_TOKENS="${QWEN_MAX_TOKENS:-20480}"

if [[ -z "${QWEN_HOST:-}" ]]; then
    echo "Auto-detecting Qwen serving node from squeue..."
    QWEN_JOB=$(squeue -u "$USER" -h -o "%i %j %T %N" \
        | awk '$2=="qwen36-27b" && $3=="RUNNING" {print $1" "$4; exit}')
    if [[ -z "$QWEN_JOB" ]]; then
        echo "ERROR: no RUNNING job named 'qwen36-27b' for $USER." >&2
        echo "       sbatch qwen36_27b/serve_qwen36.sh first, or set QWEN_HOST=..." >&2
        exit 1
    fi
    QWEN_JOBID=$(echo "$QWEN_JOB" | awk '{print $1}')
    QWEN_HOST=$(echo "$QWEN_JOB" | awk '{print $2}')
    echo "  Qwen job=$QWEN_JOBID host=$QWEN_HOST"
fi

# vLLM exposes the OpenAI-compatible endpoint at /v1/chat/completions
SERVER_URL="http://${QWEN_HOST}:${QWEN_PORT}/v1/chat/completions"

echo "=== Probing Qwen server ==="
echo "  URL: $SERVER_URL"
if ! curl -sSf -m 10 "http://${QWEN_HOST}:${QWEN_PORT}/v1/models" >/dev/null; then
    echo "ERROR: cannot reach Qwen server at $QWEN_HOST:$QWEN_PORT." >&2
    echo "       Check the SLURM job log: qwen36_27b/logs/qwen36-27b_*.out" >&2
    exit 1
fi
echo "  OK"

# ---------------------------------------------------------------------------
# Local helper servers (SAM3 / GraspNet / PyRoKi) — needed by agent0
# These run on the eval node's GPU. Skip if already up.
# ---------------------------------------------------------------------------
echo "=== Helper servers ==="
source .venv/bin/activate

start_if_down() {
    local port="$1"; local name="$2"; shift 2
    # Any HTTP response (incl. 404 from vLLM root) means the port is bound.
    if curl -s -o /dev/null --connect-timeout 2 "http://127.0.0.1:$port/" 2>/dev/null; then
        echo "  $name (port $port): UP"
    else
        echo "  $name (port $port): starting..."
        nohup "$@" > "$LOG_DIR/${name}.log" 2>&1 &
    fi
}

# GPU plan for the eval node (4×A100-64GB), one tool per GPU:
#   GPU 0 → SAM3   GPU 1 → GraspNet   GPU 2 → Molmo   GPU 3 → PyRoKi + MuJoCo
start_if_down 8114 sam3 \
    env CUDA_VISIBLE_DEVICES=0 \
    python -m capx.serving.launch_sam3_server --device cuda --port 8114 --host 127.0.0.1
start_if_down 8115 graspnet \
    env CUDA_VISIBLE_DEVICES=1 \
    python -m capx.serving.launch_contact_graspnet_server --port 8115 --host 127.0.0.1
start_if_down 8116 pyroki \
    env CUDA_VISIBLE_DEVICES=3 \
    python -m capx.serving.launch_pyroki_server --port 8116 --host 127.0.0.1 \
        --robot panda_description --target-link panda_hand
# Molmo VLM (object→pixel-point), used by reduced-API skills (plan_grasp etc.)
start_if_down 8122 molmo \
    env MOLMO_GPU=2 \
    bash scripts/serve_molmo_local.sh

echo "Waiting 30s for helper servers..."
sleep 30
for p in 8114 8115 8116 8122; do
    code=$(curl -s -o /dev/null -w "%{http_code}" --connect-timeout 2 "http://127.0.0.1:$p/" 2>/dev/null || echo 000)
    echo "  port $p: $([ "$code" != "000" ] && echo "UP ($code)" || echo DOWN)"
done

# ---------------------------------------------------------------------------
# Launch the eval (Libero env runs in .venv-libero)
# ---------------------------------------------------------------------------
deactivate || true
source .venv-libero/bin/activate

OUTPUT_DIR="$OUTPUT_ROOT/agent0_qwen36_${OUTPUT_SUFFIX}"
mkdir -p "$OUTPUT_DIR"

# Persist every SAM3 / Molmo call (overlay PNG + meta JSON) so we can
# verify them post-hoc. Set to empty to disable either.
export CAPX_SAM3_DUMP_DIR="${CAPX_SAM3_DUMP_DIR:-$OUTPUT_DIR/sam3_dumps}"
export CAPX_MOLMO_DUMP_DIR="${CAPX_MOLMO_DUMP_DIR:-$OUTPUT_DIR/molmo_dumps}"

echo ""
echo "=== Launching agent0 eval ==="
echo "  mode=$MODE  suites=${SUITES[*]}  workers=$NUM_WORKERS  trials=${TOTAL_TRIALS:-all}"
echo "  output=$OUTPUT_DIR"
echo "  model=$QWEN_MODEL  via=$SERVER_URL"
echo ""

# Build the cmd. Note: run_libero_batch.py takes --args.* flags via tyro and
# forwards server_url/model into LaunchArgs.
CMD=(
    python -m capx.envs.scripts.run_libero_batch
    --args.base-config-path env_configs/libero/franka_libero_cap_agent0_qwen.yaml
    --args.suites "${SUITES[@]}"
    --args.models "$QWEN_MODEL"
    --args.server-url "$SERVER_URL"
    --args.num-workers "$NUM_WORKERS"
    --args.output-dir "$OUTPUT_DIR"
    --args.max-tokens "$QWEN_MAX_TOKENS"
    --args.reasoning-effort medium
    # Qwen3.6-27B is a VL model — reuse the same vLLM server for image differencing.
    --args.visual-differencing-model "$QWEN_MODEL"
    --args.visual-differencing-model-server-url "$SERVER_URL"
)
[[ -n "$TOTAL_TRIALS" ]] && CMD+=( --args.total-trials "$TOTAL_TRIALS" )
[[ -n "$DEBUG_FLAG"   ]] && CMD+=( $DEBUG_FLAG )
# shellcheck disable=SC2206
CMD+=( $EXTRA_FLAGS )

# MuJoCo offscreen rendering needs an EGL-capable GPU. Pin it to GPU 0; the
# helper servers (SAM3/GraspNet/PyRoKi) already share this node's GPUs.
MUJOCO_EGL_DEVICE_ID=3 MUJOCO_GL=egl TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 \
    "${CMD[@]}" 2>&1 | tee "$LOG_DIR/agent0_qwen36_${OUTPUT_SUFFIX}.log"
