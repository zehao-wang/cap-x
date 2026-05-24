#!/bin/bash
# Run cap-x agent0 LIBERO-PRO eval against a separately-launched Qwen3.6-27B
# vLLM server. Uses the .venv-libero venv (LIBERO ships its own pinned
# robosuite 1.4 fork that conflicts with the main .venv).
#
# Prereqs (do these once, on the login node):
#   1. sbatch ../qwen36_27b/serve_qwen36.sh   # boots vLLM on a 4xA100 node
#   2. squeue -u $USER                        # wait for it to be RUNNING
#   3. salloc a GPU compute node and ssh in   # this node runs the eval +
#                                               local SAM3/GraspNet/PyRoKi
#   4. bash scripts/run_agent0_qwen36_libero.sh [debug|full|custom]
#
# Helper server allocation (4xA100 node, NO Molmo — LIBERO config doesn't use it):
#   GPU 0 → SAM3, GPU 1 → ContactGraspNet, GPU 2 → PyRoKi+MuJoCo, GPU 3 → spare

set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

# ---------------------------------------------------------------------------
# Paths / offline caches (same conventions as the robosuite script).
# ---------------------------------------------------------------------------
SCRATCH_ROOT="${SCRATCH_ROOT:-/leonardo_scratch/fast/EUHPC_D33_222}"
LOG_DIR="${LOG_DIR:-$SCRATCH_ROOT/cap-x/logs}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$SCRATCH_ROOT/cap-x/outputs}"
mkdir -p "$LOG_DIR" "$OUTPUT_ROOT"

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

# LIBERO needs ~/.libero/config.yaml on headless nodes; create on demand.
if [[ ! -f "$HOME/.libero/config.yaml" ]]; then
    mkdir -p "$HOME/.libero"
    LIBERO_ROOT="$(pwd)/capx/third_party/LIBERO-PRO/libero/libero"
    cat > "$HOME/.libero/config.yaml" <<EOF
assets: $LIBERO_ROOT/assets
bddl_files: $LIBERO_ROOT/bddl_files
benchmark_root: $LIBERO_ROOT
datasets: $LIBERO_ROOT/../datasets
init_states: $LIBERO_ROOT/init_files
EOF
    echo "Created ~/.libero/config.yaml"
fi

# ---------------------------------------------------------------------------
# Mode → which LIBERO suites + per-suite task cap + worker count.
# ---------------------------------------------------------------------------
MODE="${1:-debug}"
NUM_WORKERS="${NUM_WORKERS:-}"

ALL_SUITES=(
    libero_object_swap
    libero_object_task
    libero_goal_swap
    libero_goal_task
    libero_spatial_swap
    libero_spatial_task
)

BASE_CONFIG="${BASE_CONFIG:-env_configs/libero/franka_libero_cap_agent0_qwen.yaml}"
if [[ ! -f "$BASE_CONFIG" ]]; then
    echo "ERROR: base config not found: $BASE_CONFIG" >&2
    exit 1
fi

case "$MODE" in
    debug)
        # Smoke test: object_swap, 1 task, 1 worker.
        SUITES=( libero_object_swap )
        MAX_TASKS=1
        : "${NUM_WORKERS:=1}"
        DEBUG_FLAG="--args.debug"
        OUTPUT_SUFFIX="debug"
        ;;
    full)
        SUITES=( "${ALL_SUITES[@]}" )
        MAX_TASKS=""   # all tasks per suite
        : "${NUM_WORKERS:=8}"
        DEBUG_FLAG=""
        OUTPUT_SUFFIX="full"
        ;;
    custom)
        # Caller must export CAPX_SUITES (space-separated suite names) and
        # CAPX_OUTPUT_SUFFIX. Optional: CAPX_MAX_TASKS, CAPX_BASE_CONFIG.
        if [[ -z "${CAPX_SUITES:-}" || -z "${CAPX_OUTPUT_SUFFIX:-}" ]]; then
            echo "ERROR: custom mode requires CAPX_SUITES and CAPX_OUTPUT_SUFFIX." >&2
            exit 1
        fi
        # shellcheck disable=SC2206
        SUITES=( $CAPX_SUITES )
        MAX_TASKS="${CAPX_MAX_TASKS:-}"
        : "${NUM_WORKERS:=8}"
        DEBUG_FLAG=""
        OUTPUT_SUFFIX="$CAPX_OUTPUT_SUFFIX"
        [[ -n "${CAPX_BASE_CONFIG:-}" ]] && BASE_CONFIG="$CAPX_BASE_CONFIG"
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

echo "=== Probing Qwen server ==="
echo "  URL: $SERVER_URL"
if ! curl -sSf -m 10 "http://${QWEN_HOST}:${QWEN_PORT}/v1/models" >/dev/null; then
    echo "ERROR: cannot reach Qwen server at $QWEN_HOST:$QWEN_PORT." >&2
    exit 1
fi
echo "  OK"

# ---------------------------------------------------------------------------
# Helper servers (SAM3 / GraspNet / PyRoKi). LIBERO doesn't use Molmo, so we
# skip it. The LIBERO YAML's api_servers also lists these three only.
# ---------------------------------------------------------------------------
echo "=== Helper servers ==="
source .venv-libero/bin/activate

start_if_down() {
    local port="$1"; local name="$2"; shift 2
    if curl -s -o /dev/null --connect-timeout 2 "http://127.0.0.1:$port/" 2>/dev/null; then
        echo "  $name (port $port): UP"
    else
        echo "  $name (port $port): starting..."
        nohup "$@" > "$LOG_DIR/${name}.log" 2>&1 &
    fi
}

start_if_down 8114 sam3 \
    env CUDA_VISIBLE_DEVICES=0 \
    python -m capx.serving.launch_sam3_server --device cuda --port 8114 --host 127.0.0.1
start_if_down 8115 graspnet \
    env CUDA_VISIBLE_DEVICES=1 \
    python -m capx.serving.launch_contact_graspnet_server --port 8115 --host 127.0.0.1
start_if_down 8116 pyroki \
    env CUDA_VISIBLE_DEVICES=2 \
    python -m capx.serving.launch_pyroki_server --port 8116 --host 127.0.0.1 \
        --robot panda_description --target-link panda_hand

echo "Waiting 30s for helper servers..."
sleep 30
for p in 8114 8115 8116; do
    code=$(curl -s -o /dev/null -w "%{http_code}" --connect-timeout 2 "http://127.0.0.1:$p/" 2>/dev/null)
    if [[ -z "$code" || "$code" == "000" ]]; then
        echo "  port $p: DOWN (still booting — SAM3 cold-load ~1-2 min)"
    else
        echo "  port $p: UP ($code)"
    fi
done

# ---------------------------------------------------------------------------
# Launch the LIBERO batch.
# ---------------------------------------------------------------------------
OUTPUT_DIR="$OUTPUT_ROOT/agent0_qwen36_libero_${OUTPUT_SUFFIX}"
mkdir -p "$OUTPUT_DIR"

export CAPX_SAM3_DUMP_DIR="${CAPX_SAM3_DUMP_DIR:-$OUTPUT_DIR/sam3_dumps}"

echo ""
echo "=== Launching agent0 LIBERO eval ==="
echo "  mode=$MODE  suites=${#SUITES[@]}  workers=$NUM_WORKERS  max_tasks_per_suite=${MAX_TASKS:-all}"
echo "  base config=$BASE_CONFIG"
echo "  output=$OUTPUT_DIR"
echo "  model=$QWEN_MODEL  via=$SERVER_URL"
echo ""

CMD=(
    python -m capx.envs.scripts.run_libero_batch
    --args.base-config-path "$BASE_CONFIG"
    --args.suites "${SUITES[@]}"
    --args.models "$QWEN_MODEL"
    --args.server-url "$SERVER_URL"
    --args.num-workers "$NUM_WORKERS"
    --args.output-dir "$OUTPUT_DIR"
    --args.max-tokens "$QWEN_MAX_TOKENS"
    --args.reasoning-effort medium
    --args.visual-differencing-model "$QWEN_MODEL"
    --args.visual-differencing-model-server-url "$SERVER_URL"
)
[[ -n "$MAX_TASKS"  ]] && CMD+=( --args.max-tasks-per-suite "$MAX_TASKS" )
[[ -n "$DEBUG_FLAG" ]] && CMD+=( $DEBUG_FLAG )

# Pin MuJoCo offscreen rendering to GPU 2 (PyRoKi node, the lightest GPU user).
MUJOCO_EGL_DEVICE_ID=2 MUJOCO_GL=egl TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 \
    "${CMD[@]}" 2>&1 | tee "$LOG_DIR/agent0_qwen36_libero_${OUTPUT_SUFFIX}.log"
