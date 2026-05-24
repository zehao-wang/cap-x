#!/bin/bash
# Run cap-x agent0 LIBERO-PRO in INTERACTIVE web-UI mode against a separately-
# launched Qwen3.6-27B vLLM server. Single trial per session — use
# scripts/run_agent0_qwen36_libero.sh for batch runs.
#
# Prereqs (do these once, on the login node):
#   1. sbatch ../qwen36_27b/serve_qwen36.sh   # boots vLLM on a 4xA100 node
#   2. squeue -u $USER                        # wait for it to be RUNNING
#   3. salloc a GPU compute node and ssh in   # this node runs the web UI +
#                                               local SAM3/GraspNet/PyRoKi
#   4. bash scripts/run_agent0_qwen36_libero_interactive.sh [config_yaml]
#
# Then from your LOCAL machine, in another terminal:
#   ssh -N -L 8200:<compute-host>:8200 leo
# and open http://localhost:8200 in your browser. The script prints the
# exact ssh command with the current compute hostname filled in.

set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

# ---------------------------------------------------------------------------
# Output / cache paths (same conventions as the robosuite interactive script).
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

# Web-UI frontend must be pre-built on the login node — _ensure_frontend_built()
# downloads Node.js via nodeenv, which needs internet that compute nodes lack.
if [[ ! -f "web-ui/dist/index.html" ]]; then
    echo "ERROR: web-ui/dist/index.html missing — frontend not built." >&2
    echo "       Compute nodes have no internet, so the build must happen on" >&2
    echo "       the login node. Run there (once, and after any web-ui edit):" >&2
    echo "         source .venv/bin/activate" >&2
    echo "         python -c 'from capx.envs.launch import _ensure_frontend_built; _ensure_frontend_built()'" >&2
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
# Which LIBERO YAML to load on startup. Web UI lets you switch from the
# browser anyway. Note: each LIBERO YAML pins one suite_name+task_id, so to
# try a different task pass another YAML or copy and edit.
# ---------------------------------------------------------------------------
CONFIG="${1:-env_configs/libero/franka_libero_cap_agent0_qwen.yaml}"
if [[ ! -f "$CONFIG" ]]; then
    echo "ERROR: config not found: $CONFIG" >&2
    echo "Usage: $0 [path/to/env_config.yaml]" >&2
    exit 1
fi

WEB_UI_PORT="${WEB_UI_PORT:-8200}"

# ---------------------------------------------------------------------------
# Locate the Qwen vLLM server.
# ---------------------------------------------------------------------------
QWEN_PORT="${QWEN_PORT:-8000}"
QWEN_MODEL="${QWEN_MODEL:-Qwen3.6-27B}"
QWEN_MAX_TOKENS="${QWEN_MAX_TOKENS:-20480}"

if [[ -z "${QWEN_HOST:-}" ]]; then
    echo "Auto-detecting Qwen serving node from squeue..."
    QWEN_JOB=$(squeue -u "$USER" -h -o "%i %j %T %N" \
        | awk '$2=="qwen36-27b" && $3=="RUNNING" {print $1" "$4; exit}')
    if [[ -z "$QWEN_JOB" ]]; then
        echo "ERROR: no RUNNING job named 'qwen36-27b' for $USER." >&2
        echo "       Submit one with: sbatch ../qwen36_27b/serve_qwen36.sh" >&2
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
# Helper servers (SAM3 / GraspNet / PyRoKi). LIBERO doesn't use Molmo.
#   GPU 0 → SAM3, GPU 1 → GraspNet, GPU 2 → PyRoKi+MuJoCo, GPU 3 → spare
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
# Output dump dir.
# ---------------------------------------------------------------------------
SESSION_TAG="$(date +%Y%m%d_%H%M%S)"
OUTPUT_DIR="$OUTPUT_ROOT/agent0_qwen36_libero_interactive_${SESSION_TAG}"
mkdir -p "$OUTPUT_DIR"
export CAPX_SAM3_DUMP_DIR="${CAPX_SAM3_DUMP_DIR:-$OUTPUT_DIR/sam3_dumps}"

# ---------------------------------------------------------------------------
# Print the local-machine port-forward hint, then launch the web UI.
# ---------------------------------------------------------------------------
THIS_HOST="$(hostname -s)"
cat <<EOF

============================================================================
  Launching CaP-X interactive web UI (LIBERO)
    config:  $CONFIG
    model:   $QWEN_MODEL  via  $SERVER_URL
    output:  $OUTPUT_DIR
    bind:    0.0.0.0:$WEB_UI_PORT  (on $THIS_HOST)

  >>> ON YOUR LOCAL MACHINE, OPEN A SEPARATE TERMINAL AND RUN: <<<

      ssh -N -L $WEB_UI_PORT:$THIS_HOST:$WEB_UI_PORT leo

  Leonardo blocks direct ssh into compute nodes, so we use 'leo' as a
  pure TCP relay: local:$WEB_UI_PORT --(ssh)--> leo --(internal TCP)--> $THIS_HOST:$WEB_UI_PORT.

  Then in your browser open:  http://localhost:$WEB_UI_PORT
  Viser 3D is reverse-proxied through the same port — no extra forward needed.

  Ctrl-C here stops the web UI. Helper servers (SAM3/GraspNet/PyRoKi)
  stay up so the next interactive session can reuse them.
============================================================================

EOF

# Pin MuJoCo offscreen rendering to GPU 2 (PyRoKi node, lightest GPU user).
MUJOCO_EGL_DEVICE_ID=2 MUJOCO_GL=egl TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 \
    python -m capx.envs.launch \
        --config-path "$CONFIG" \
        --model "$QWEN_MODEL" \
        --server-url "$SERVER_URL" \
        --max-tokens "$QWEN_MAX_TOKENS" \
        --reasoning-effort medium \
        --visual-differencing-model "$QWEN_MODEL" \
        --visual-differencing-model-server-url "$SERVER_URL" \
        --output-dir "$OUTPUT_DIR" \
        --web-ui True \
        --web-ui-port "$WEB_UI_PORT"
