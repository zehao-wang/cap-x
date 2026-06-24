#!/bin/bash
# Run cap-x agent0 ROBOSUITE in INTERACTIVE web-UI mode against a separately-
# launched Qwen3.6-27B vLLM server. Single trial per session — to run multiple
# trials use scripts/run_agent0_qwen36_robosuite.sh instead.
#
# Prereqs (do these once, on the login node):
#   1. sbatch ../qwen36_27b/serve_qwen36.sh   # boots vLLM on a 4xA100 node
#   2. squeue -u $USER                        # wait for it to be RUNNING
#   3. salloc a GPU compute node and ssh in   # this node runs the web UI +
#                                               local SAM3/GraspNet/PyRoKi/Molmo
#   4. bash scripts/run_agent0_qwen36_interactive.sh [config_yaml]
#
# Then from your LOCAL machine (one-shot port-forward through leo login):
#   ssh -N -L 8200:localhost:8200 <compute_node_short_name>
# and open http://localhost:8200 in your browser. Viser 3D is reverse-proxied
# through 8200 — you do NOT need to forward 8080 separately.

set -euo pipefail
REPO_ROOT="$(git rev-parse --show-toplevel)"
cd "$REPO_ROOT"

# ---------------------------------------------------------------------------
# Logs root. Everything this interactive run produces is unified under ONE
# timestamped directory below logs/ (a symlink to durable scratch) — the
# timestamped subdir + console tee are set up in the "Session log dir" block
# just before launch. Override LOGS_ROOT to relocate; the model-cache paths
# below are unrelated and keep their own defaults.
# ---------------------------------------------------------------------------
LOGS_ROOT="${LOGS_ROOT:-$REPO_ROOT/logs}"

# Model-cache root — the ONE knob for where the models are read from: SAM3 +
# Molmo via HF_HOME, robot URDFs via robot_descriptions. Defaults to scratch.
# Point it at a healthy filesystem (e.g. $WORK) to survive a Lustre/scratch
# outage — same var that prefetch uses, so you read from where you downloaded:
#   CAPX_CACHE_ROOT=$WORK/zwang003 bash scripts/prefetch_agent0_models.sh   # once, on login
#   CAPX_CACHE_ROOT=$WORK/zwang003 bash scripts/run_agent0_qwen36_robosuite_interactive.sh
# NOTE: an explicit CAPX_CACHE_ROOT must win even over an HF_HOME exported by
# your ~/.bashrc (which sets HF_HOME to scratch). So when CAPX_CACHE_ROOT is
# given, derive all three paths from it unconditionally; otherwise fall back to
# scratch while still honoring any pre-set HF_HOME/MODELS_ROOT.
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
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

if [[ ! -d "$HF_HOME" || ! -d "$ROBOT_DESCRIPTIONS_CACHE" ]]; then
    echo "ERROR: missing model caches:" >&2
    echo "         HF_HOME=$HF_HOME" >&2
    echo "         ROBOT_DESCRIPTIONS_CACHE=$ROBOT_DESCRIPTIONS_CACHE" >&2
    echo "       Pre-fetch them on the login node with the SAME CAPX_CACHE_ROOT:" >&2
    echo "         CAPX_CACHE_ROOT=$CAPX_CACHE_ROOT bash scripts/prefetch_agent0_models.sh" >&2
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

# ---------------------------------------------------------------------------
# Which task config to load on startup. The web UI lets you switch configs
# from the browser anyway, so this is just the initial one.
# ---------------------------------------------------------------------------
CONFIG="${1:-env_configs/cube_lifting/franka_robosuite_cube_lifting_multiturn_vdm_reduced_api_skill_lib.yaml}"
if [[ ! -f "$CONFIG" ]]; then
    echo "ERROR: config not found: $CONFIG" >&2
    echo "Usage: $0 [path/to/env_config.yaml]" >&2
    exit 1
fi

# Which backend families this launch exposes in the web-UI config dropdown.
# The helper servers + MuJoCo/EGL env below are set up for MuJoCo/Franka, so we
# only offer robosuite + LIBERO (both panda). The server further hides any whose
# sim package isn't importable — so LIBERO stays hidden until it's installed,
# and R1Pro (OmniGibson) / real-robot configs never show up here.
export CAPX_CONFIG_FAMILIES="${CAPX_CONFIG_FAMILIES:-robosuite,libero}"

WEB_UI_PORT="${WEB_UI_PORT:-8200}"

# ---------------------------------------------------------------------------
# Session log dir — ONE timestamped directory under logs/ holds this whole run:
#   web_ui.log                          console of this script + the web-UI
#                                       process (uvicorn + python logging +
#                                       viser server + robosuite)
#   {sam3,graspnet,pyroki,molmo}.log    helper-server stdout (only for servers
#                                       this run actually starts; reused ones
#                                       keep logging into their first session)
#   trial_NN/                           per-trial artifacts the web UI writes:
#                                       trace/, viser_history/*.npz,
#                                       sam3_dumps/, molmo_dumps/, videos
#                                       (routed here via --output-dir below)
# The tee runs via process substitution so a foreground Ctrl-C can't sever
# python's stdout mid-shutdown; the terminal still shows everything live.
# ---------------------------------------------------------------------------
SESSION_TAG="$(date +%Y%m%d_%H%M%S)"
SESSION_DIR="$LOGS_ROOT/interactive_${SESSION_TAG}"
mkdir -p "$SESSION_DIR"
exec > >(tee "$SESSION_DIR/web_ui.log") 2>&1
echo "Session log dir: $SESSION_DIR"
# Fallback dump dirs: the web UI normally writes SAM3/Molmo dumps under each
# trial's artifact dir (trial_NN/), so these env vars only apply if that is
# unset (e.g. a headless reuse of this env).
export CAPX_SAM3_DUMP_DIR="${CAPX_SAM3_DUMP_DIR:-$SESSION_DIR/sam3_dumps}"
export CAPX_MOLMO_DUMP_DIR="${CAPX_MOLMO_DUMP_DIR:-$SESSION_DIR/molmo_dumps}"

# ---------------------------------------------------------------------------
# Locate the Qwen vLLM server (auto-detect from squeue, override via QWEN_HOST).
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
# Helper servers (SAM3 / GraspNet / PyRoKi / Molmo). Same GPU plan as batch:
#   GPU 0 → SAM3, GPU 1 → ContactGraspNet, GPU 2 → Molmo, GPU 3 → PyRoKi+MuJoCo
# ---------------------------------------------------------------------------
echo "=== Helper servers ==="
source .venv/bin/activate

start_if_down() {
    local port="$1"; local name="$2"; shift 2
    if curl -s -o /dev/null --connect-timeout 2 "http://127.0.0.1:$port/" 2>/dev/null; then
        echo "  $name (port $port): UP"
    else
        echo "  $name (port $port): starting..."
        nohup "$@" > "$SESSION_DIR/${name}.log" 2>&1 &
    fi
}

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
start_if_down 8122 molmo \
    env MOLMO_GPU=2 \
    bash scripts/serve_molmo_local.sh

# These are GPU model servers (SAM3 / GraspNet / PyRoKi / vLLM-Molmo) that take
# minutes to load — at a short wait they will almost always still be DOWN. That
# is fine: they keep loading in the background and just need to be UP before you
# start a trial in the browser. The web UI itself does NOT wait for them. The
# wait below is purely so the status line below is informative; bump it via
# HELPER_WAIT_SECS if you want a better chance of seeing them come UP here.
HELPER_WAIT_SECS="${HELPER_WAIT_SECS:-60}"
echo "Waiting ${HELPER_WAIT_SECS}s for helper servers (they load in the background)..."
sleep "$HELPER_WAIT_SECS"
for p in 8114 8115 8116 8122; do
    # -w prints the HTTP code; on a failed connection it prints "000". Don't add
    # a `|| echo 000` — that doubled the output to "000000" and falsely read UP.
    code=$(curl -s -o /dev/null -w "%{http_code}" --connect-timeout 2 "http://127.0.0.1:$p/" 2>/dev/null)
    [ -z "$code" ] && code=000
    echo "  port $p: $([ "$code" != "000" ] && echo "UP ($code)" || echo "still loading / down")"
done

# ---------------------------------------------------------------------------
# Print the local-machine port-forward hint, then launch the web UI.
# ---------------------------------------------------------------------------
THIS_HOST="$(hostname -s)"
cat <<EOF

============================================================================
  Launching CaP-X interactive web UI
    config:  $CONFIG
    model:   $QWEN_MODEL  via  $SERVER_URL
    logs:    $SESSION_DIR  (web_ui.log + helper logs + trial_NN/ artifacts)
    bind:    0.0.0.0:$WEB_UI_PORT  (on $THIS_HOST)

  >>> ON YOUR LOCAL MACHINE, OPEN A SEPARATE TERMINAL AND RUN: <<<

      ssh -N -L $WEB_UI_PORT:$THIS_HOST:$WEB_UI_PORT leo

  Leonardo blocks direct ssh into compute nodes, so we use 'leo' as a
  pure TCP relay: local:$WEB_UI_PORT --(ssh)--> leo --(internal TCP)--> $THIS_HOST:$WEB_UI_PORT.
  Don't ProxyJump — leo just opens a socket to $THIS_HOST on your behalf.

  Then in your browser open:  http://localhost:$WEB_UI_PORT
  Viser 3D is reverse-proxied through the same port — no extra forward needed.

  Ctrl-C here stops the web UI. Helper servers (SAM3/GraspNet/PyRoKi/Molmo)
  stay up so the next interactive session can reuse them — kill them with
  'pkill -f launch_sam3_server' etc. when you're done with the allocation.
============================================================================

EOF

MUJOCO_EGL_DEVICE_ID=3 MUJOCO_GL=egl TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 \
    python -m capx.envs.launch \
        --config-path "$CONFIG" \
        --model "$QWEN_MODEL" \
        --max-tokens "$QWEN_MAX_TOKENS" \
        --reasoning-effort medium \
        --visual-differencing-model "$QWEN_MODEL" \
        --output-dir "$SESSION_DIR" \
        --web-ui True \
        --web-ui-port "$WEB_UI_PORT"
