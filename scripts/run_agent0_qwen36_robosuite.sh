#!/bin/bash
# Run cap-x agent0 ROBOSUITE eval against a separately-launched Qwen3.6-27B
# vLLM server. Uses the base .venv (robosuite 1.5.x lives here, NOT in
# .venv-libero which has its own pinned robosuite 1.4 fork).
#
# Prereqs (do these once, on the login node):
#   1. sbatch qwen36_27b/serve_qwen36.sh     # boots vLLM on a 4xA100 node
#   2. squeue -u $USER                       # wait for it to be RUNNING
#   3. salloc a GPU compute node and ssh in  # this node runs the eval +
#                                              local SAM3/GraspNet/PyRoKi
#   4. bash scripts/run_agent0_qwen36_robosuite.sh [debug|full]

set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

# Scratch paths for outputs/logs.
SCRATCH_ROOT="${SCRATCH_ROOT:-/leonardo_scratch/fast/EUHPC_D33_222}"
LOG_DIR="${LOG_DIR:-$SCRATCH_ROOT/cap-x/logs}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$SCRATCH_ROOT/cap-x/outputs}"
mkdir -p "$LOG_DIR" "$OUTPUT_ROOT"

# Offline asset paths (populated once by scripts/prefetch_agent0_models.sh).
MODELS_ROOT="${MODELS_ROOT:-/leonardo_scratch/fast/EUHPC_D33_222/zwang003/models}"
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
# Mode → which Robosuite agent0 (multiturn + VDM + reduced API + skill lib)
# configs to run, and how many trials.
# ---------------------------------------------------------------------------
MODE="${1:-debug}"
NUM_WORKERS="${NUM_WORKERS:-}"

# All six available agent0-style robosuite configs.
ALL_CONFIGS=(
    env_configs/cube_lifting/franka_robosuite_cube_lifting_multiturn_vdm_reduced_api_skill_lib.yaml
    env_configs/cube_stack/franka_robosuite_cube_stack_multiturn_vdm_reduced_api_skill_lib.yaml
    env_configs/cube_restack/franka_robosuite_cube_restack_multiturn_vdm_reduced_api_skill_lib.yaml
    env_configs/nut_assembly/franka_robosuite_nut_assembly_multiturn_vdm_reduced_api_skill_lib.yaml
    env_configs/spill_wipe/franka_robosuite_spill_wipe_multiturn_vdm_reduced_api_skill_lib.yaml
    env_configs/two_arm_lift/franka_robosuite_two_arm_lift_multiturn_vdm_reduced_api_skill_lib.yaml
)

case "$MODE" in
    debug)
        # Smoke test: cube_lifting, 1 trial, single worker.
        CONFIGS=( env_configs/cube_lifting/franka_robosuite_cube_lifting_multiturn_vdm_reduced_api_skill_lib.yaml )
        TOTAL_TRIALS=1
        : "${NUM_WORKERS:=1}"
        DEBUG_FLAG="--args.debug"
        OUTPUT_SUFFIX="debug"
        ;;
    full)
        CONFIGS=( "${ALL_CONFIGS[@]}" )
        TOTAL_TRIALS=""   # use YAML default (100)
        : "${NUM_WORKERS:=8}"
        DEBUG_FLAG=""
        OUTPUT_SUFFIX="full"
        ;;
    custom)
        # Subset run. Caller must export CAPX_CONFIGS (space-separated list of
        # YAML paths relative to repo root) and CAPX_OUTPUT_SUFFIX. Optional:
        # CAPX_TOTAL_TRIALS (empty → YAML default).
        if [[ -z "${CAPX_CONFIGS:-}" || -z "${CAPX_OUTPUT_SUFFIX:-}" ]]; then
            echo "ERROR: custom mode requires CAPX_CONFIGS and CAPX_OUTPUT_SUFFIX." >&2
            exit 1
        fi
        # shellcheck disable=SC2206
        CONFIGS=( $CAPX_CONFIGS )
        TOTAL_TRIALS="${CAPX_TOTAL_TRIALS:-}"
        : "${NUM_WORKERS:=8}"
        DEBUG_FLAG=""
        OUTPUT_SUFFIX="$CAPX_OUTPUT_SUFFIX"
        ;;
    *)
        echo "Usage: $0 [debug|full|custom]" >&2
        exit 1
        ;;
esac

# ---------------------------------------------------------------------------
# Locate the Qwen vLLM server.
# ---------------------------------------------------------------------------
QWEN_PORT="${QWEN_PORT:-8000}"
QWEN_MODEL="${QWEN_MODEL:-Qwen3.6-27B}"
QWEN_MAX_MODEL_LEN="${QWEN_MAX_MODEL_LEN:-100000}"
QWEN_MAX_TOKENS="${QWEN_MAX_TOKENS:-20480}"

if [[ -z "${QWEN_HOST:-}" ]]; then
    echo "Auto-detecting Qwen serving node from squeue..."
    QWEN_JOB=$(squeue -u "$USER" -h -o "%i %j %T %N" \
        | awk '$2=="qwen36-27b" && $3=="RUNNING" {print $1" "$4; exit}')
    if [[ -z "$QWEN_JOB" ]]; then
        echo "ERROR: no RUNNING job named 'qwen36-27b' for $USER." >&2
        exit 1
    fi
    QWEN_JOBID=$(echo "$QWEN_JOB" | awk '{print $1}')
    QWEN_HOST=$(echo "$QWEN_JOB" | awk '{print $2}')
    echo "  Qwen job=$QWEN_JOBID host=$QWEN_HOST"
fi

SERVER_URL="http://${QWEN_HOST}:${QWEN_PORT}/v1/chat/completions"
# The cap-x harness routes by model name: "Qwen3.6-27B" -> the qwen lane, whose
# endpoint comes from CAPX_QWEN_URL (else localhost:8000). Export it so the
# remote vLLM host is honored (the per-run --server-url flag is now ignored).
export CAPX_QWEN_URL="$SERVER_URL"

echo "=== Probing Qwen server ==="
echo "  URL: $SERVER_URL"
if ! curl -sSf -m 10 "http://${QWEN_HOST}:${QWEN_PORT}/v1/models" >/dev/null; then
    echo "ERROR: cannot reach Qwen server at $QWEN_HOST:$QWEN_PORT." >&2
    exit 1
fi
echo "  OK"

# ---------------------------------------------------------------------------
# Helper servers (SAM3 / GraspNet / PyRoKi). Same .venv hosts both helpers
# and robosuite eval, so we activate it once.
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

# GPU plan for the eval node (4×A100-64GB), one tool per GPU to avoid contention:
#   GPU 0 → SAM3 (vision FM, largest single tool)
#   GPU 1 → ContactGraspNet
#   GPU 2 → Molmo-2-8B vLLM (dedicated, ~17 GB bf16 + KV cache)
#   GPU 3 → PyRoKi (light) + MuJoCo offscreen rendering
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
# Molmo VLM (object→pixel-point) on its own GPU. vLLM compilation is slow
# (~3 min cold start), so leave it running across runs.
start_if_down 8122 molmo \
    env MOLMO_GPU=2 \
    bash scripts/serve_molmo_local.sh

echo "Waiting 30s for helper servers..."
sleep 30
# Use any HTTP response (incl. 404 from vLLM root) as 'up' — only no-connection is down.
for p in 8114 8115 8116 8122; do
    code=$(curl -s -o /dev/null -w "%{http_code}" --connect-timeout 2 "http://127.0.0.1:$p/" 2>/dev/null || echo 000)
    echo "  port $p: $([ "$code" != "000" ] && echo "UP ($code)" || echo DOWN)"
done

# ---------------------------------------------------------------------------
# Launch the eval.
# ---------------------------------------------------------------------------
OUTPUT_DIR="$OUTPUT_ROOT/agent0_qwen36_robosuite_${OUTPUT_SUFFIX}"
mkdir -p "$OUTPUT_DIR"

# Persist every SAM3 / Molmo call (overlay PNG + meta JSON) for post-hoc
# verification. Set to empty to disable either.
export CAPX_SAM3_DUMP_DIR="${CAPX_SAM3_DUMP_DIR:-$OUTPUT_DIR/sam3_dumps}"
export CAPX_MOLMO_DUMP_DIR="${CAPX_MOLMO_DUMP_DIR:-$OUTPUT_DIR/molmo_dumps}"

echo ""
echo "=== Launching agent0 ROBOSUITE eval ==="
echo "  mode=$MODE  configs=${#CONFIGS[@]}  workers=$NUM_WORKERS  trials=${TOTAL_TRIALS:-yaml-default}"
echo "  output=$OUTPUT_DIR"
echo "  model=$QWEN_MODEL  via=$SERVER_URL"
echo ""

CMD=(
    python -m capx.envs.scripts.run_batch
    --args.config-paths "${CONFIGS[@]}"
    --args.models "$QWEN_MODEL"
    --args.num-workers "$NUM_WORKERS"
    --args.output-dir "$OUTPUT_DIR"
    --args.max-tokens "$QWEN_MAX_TOKENS"
    --args.reasoning-effort medium
    # Qwen3.6-27B is a VL model — the VDM routes to the same qwen lane by name.
    --args.visual-differencing-model "$QWEN_MODEL"
)
[[ -n "$TOTAL_TRIALS" ]] && CMD+=( --args.total-trials "$TOTAL_TRIALS" )
[[ -n "$DEBUG_FLAG"   ]] && CMD+=( $DEBUG_FLAG )

# Pin MuJoCo offscreen rendering to GPU 3 (PyRoKi node, lightest); keep heavy
# tool models (SAM3 / GraspNet / Molmo) free of MuJoCo contention.
MUJOCO_EGL_DEVICE_ID=3 MUJOCO_GL=egl TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 \
    "${CMD[@]}" 2>&1 | tee "$LOG_DIR/agent0_qwen36_robosuite_${OUTPUT_SUFFIX}.log"
