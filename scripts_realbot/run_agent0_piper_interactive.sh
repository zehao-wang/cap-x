#!/bin/bash
# Run cap-x agent0 on the REAL Agilex PIPER arm in INTERACTIVE web-UI mode.
#
# This is the real-robot sibling of run_agent0_qwen36_robosuite_interactive.sh.
# Differences, all because this runs on a LOCAL workstation wired to hardware
# (not a Leonardo compute node):
#   - LLM is OpenRouter Gemini via the local proxy on :8110 — NOT Qwen3.6.
#     (robosuite/libero use the Qwen vLLM server; the real robot never does.)
#   - No slurm / squeue / scratch / HF_HOME / MuJoCo-EGL plumbing.
#   - Only SAM3 + Contact-GraspNet helper servers are started (Piper does IK
#     in-process via pyroki, and the VDM differencing model is Gemini too, so
#     no PyRoKi/Molmo servers).
#
# Prereqs (one-time, see README_piper.md §1-§3):
#   1. uv sync ; uv pip install piper-sdk python-can pyrealsense2 pyyaml
#   2. echo "sk-or-v1-..." > .openrouterkey        # repo root, git-ignored
#   3. conda env `zed_bridge` (python 3.10 + pyzed + numpy<2)
#   4. export the real-robot env vars (README_piper §3):
#        ZED_BRIDGE_PYTHON  PIPER_ZED_BRIDGE  PIPER_URDF_PATH  PIPER_CAMERA_EXTRINSICS
#        PIPER_CAN_CHANNEL / PIPER_CAN_INTERFACE / PIPER_CAN_BITRATE (CAN link)
#   5. Bring CAN up:  sudo scripts_realbot/setup_can.sh
#   6. Build the web-UI once (and after any web-ui edit):
#        .venv/bin/python -c 'from capx.envs.launch import _ensure_frontend_built; _ensure_frontend_built()'
#
# Run:
#   bash scripts_realbot/run_agent0_piper_interactive.sh [config_yaml]
# Then open http://localhost:8200 in your browser (Viser 3D is reverse-proxied
# through the same port — no extra forward needed when local).

set -euo pipefail
REPO_ROOT="$(git rev-parse --show-toplevel)"
cd "$REPO_ROOT"

PY="${CAPX_PYTHON:-.venv/bin/python}"

# ---------------------------------------------------------------------------
# Knobs (env-overridable). The LLM intentionally defaults to OpenRouter Gemini;
# override PIPER_LLM_MODEL only with another OpenRouter-served model id.
# ---------------------------------------------------------------------------
CONFIG="${1:-env_configs/real/piper_real.yaml}"
WEB_UI_PORT="${WEB_UI_PORT:-8200}"
PIPER_LLM_MODEL="${PIPER_LLM_MODEL:-google/gemini-3.1-pro-preview}"
PIPER_MAX_TOKENS="${PIPER_MAX_TOKENS:-20480}"
OPENROUTER_PORT="${OPENROUTER_PORT:-8110}"
OPENROUTER_URL="http://localhost:${OPENROUTER_PORT}/chat/completions"
SAM3_PORT="${SAM3_PORT:-8114}"
GRASPNET_PORT="${GRASPNET_PORT:-8115}"

# ---------------------------------------------------------------------------
# Prechecks — fail early with an actionable message instead of a stack trace.
# ---------------------------------------------------------------------------
[[ -x "$PY" ]] || { echo "ERROR: python not found at $PY (set CAPX_PYTHON)." >&2; exit 1; }
[[ -f "$CONFIG" ]] || { echo "ERROR: config not found: $CONFIG" >&2; exit 1; }
if [[ ! -f .openrouterkey ]]; then
    echo "ERROR: .openrouterkey missing in repo root." >&2
    echo "       echo 'sk-or-v1-...' > .openrouterkey   (README_piper §2)" >&2
    exit 1
fi
if [[ ! -f web-ui/dist/index.html ]]; then
    echo "ERROR: web-ui/dist/index.html missing — frontend not built." >&2
    echo "       $PY -c 'from capx.envs.launch import _ensure_frontend_built; _ensure_frontend_built()'" >&2
    exit 1
fi
missing=()
for v in ZED_BRIDGE_PYTHON PIPER_ZED_BRIDGE PIPER_URDF_PATH PIPER_CAMERA_EXTRINSICS; do
    [[ -n "${!v:-}" ]] || missing+=("$v")
done
if (( ${#missing[@]} )); then
    echo "ERROR: required real-robot env vars unset (README_piper §3): ${missing[*]}" >&2
    exit 1
fi
for v in PIPER_ZED_BRIDGE PIPER_URDF_PATH PIPER_CAMERA_EXTRINSICS; do
    [[ -f "${!v}" ]] || echo "WARN: $v points at a missing file: ${!v}" >&2
done

# Real-robot configs are gated behind CAPX_ENABLE_REAL and only the "real"
# family is exposed in the dropdown for this launch.
export CAPX_ENABLE_REAL=1
export CAPX_CONFIG_FAMILIES="${CAPX_CONFIG_FAMILIES:-real}"

# ---------------------------------------------------------------------------
# Session log dir — ONE timestamped directory holds this whole run.
# ---------------------------------------------------------------------------
LOGS_ROOT="${LOGS_ROOT:-$REPO_ROOT/logs}"
SESSION_TAG="$(date +%Y%m%d_%H%M%S)"
SESSION_DIR="$LOGS_ROOT/piper_interactive_${SESSION_TAG}"
mkdir -p "$SESSION_DIR"
exec > >(tee "$SESSION_DIR/web_ui.log") 2>&1
echo "Session log dir: $SESSION_DIR"

# ---------------------------------------------------------------------------
# OpenRouter proxy (:8110). launch.py routes LLM calls here but does NOT start
# it — bring it up if it isn't already serving.
# ---------------------------------------------------------------------------
start_if_down() {
    local port="$1"; local name="$2"; shift 2
    if curl -s -o /dev/null --connect-timeout 2 "http://127.0.0.1:$port/" 2>/dev/null; then
        echo "  $name (port $port): UP"
    else
        echo "  $name (port $port): starting..."
        nohup "$@" > "$SESSION_DIR/${name}.log" 2>&1 &
    fi
}

echo "=== OpenRouter proxy ==="
start_if_down "$OPENROUTER_PORT" openrouter \
    "$PY" capx/serving/openrouter_server.py --key-file .openrouterkey --port "$OPENROUTER_PORT"

# ---------------------------------------------------------------------------
# Helper servers. In web-UI mode launch.py does NOT start the config's
# api_servers (they bind slowly and would block the UI), so we start them here.
# Piper needs only SAM3 (segmentation) + Contact-GraspNet (grasp planning).
# ---------------------------------------------------------------------------
echo "=== Helper servers (SAM3 / Contact-GraspNet) ==="
start_if_down "$SAM3_PORT" sam3 \
    "$PY" -m capx.serving.launch_sam3_server --device cuda --port "$SAM3_PORT" --host 127.0.0.1
start_if_down "$GRASPNET_PORT" graspnet \
    "$PY" -m capx.serving.launch_contact_graspnet_server --device cuda --port "$GRASPNET_PORT" --host 127.0.0.1

# These GPU servers take a while to load; they keep loading in the background
# and just need to be UP before you start a trial in the browser.
HELPER_WAIT_SECS="${HELPER_WAIT_SECS:-30}"
echo "Waiting ${HELPER_WAIT_SECS}s for helper servers (they load in the background)..."
sleep "$HELPER_WAIT_SECS"
for p in "$OPENROUTER_PORT" "$SAM3_PORT" "$GRASPNET_PORT"; do
    code=$(curl -s -o /dev/null -w "%{http_code}" --connect-timeout 2 "http://127.0.0.1:$p/" 2>/dev/null)
    [ -z "$code" ] && code=000
    echo "  port $p: $([ "$code" != "000" ] && echo "UP ($code)" || echo "still loading / down")"
done

# ---------------------------------------------------------------------------
# Launch the web UI against the real robot.
# ---------------------------------------------------------------------------
cat <<EOF

============================================================================
  Launching CaP-X interactive web UI  (REAL Agilex PIPER)
    config:  $CONFIG
    model:   $PIPER_LLM_MODEL  via  $OPENROUTER_URL  (OpenRouter — NOT Qwen)
    logs:    $SESSION_DIR

  Open in your browser:  http://localhost:$WEB_UI_PORT
  Ctrl-C here stops the web UI. Helper servers (SAM3/GraspNet/OpenRouter) stay
  up for the next session — kill with 'pkill -f launch_sam3_server' etc.
============================================================================

EOF

"$PY" -m capx.envs.launch \
    --config-path "$CONFIG" \
    --model "$PIPER_LLM_MODEL" \
    --server-url "$OPENROUTER_URL" \
    --max-tokens "$PIPER_MAX_TOKENS" \
    --reasoning-effort medium \
    --visual-differencing-model "$PIPER_LLM_MODEL" \
    --visual-differencing-model-server-url "$OPENROUTER_URL" \
    --output-dir "$SESSION_DIR" \
    --web-ui True \
    --web-ui-port "$WEB_UI_PORT"
