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
# On launch this prints a PRE-FLIGHT CHECKLIST and pauses: it starts OpenRouter +
# SAM3 + GraspNet itself, but YOU must bring up the arm (power + CAN, and the
# piper_state_service for *_service configs) and — when piper_zed_source=service —
# the ZED camera service (scripts_realbot/zed_service/run_zed_service.sh).
# Set CAPX_SKIP_CHECKLIST=1 to skip the pause.
#
# Standalone service launchers (this script auto-starts OpenRouter via the first):
#   scripts_realbot/openrouter_service/run_openrouter_service.sh   # LLM/VDM proxy :8110
#   scripts_realbot/zed_service/run_zed_service.sh                 # ZED depth (service mode)
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
# Viser 3D scene port for the Piper env. Pinned to a distinctive port (NOT viser's
# common default 8080, NOT the calibration scripts' 8201) so the web-UI proxy
# targets THIS session's viser and never latches onto a stale one. The env binds
# this and capx/web/server.py::_find_viser_port honours it.
CAPX_VISER_PORT="${CAPX_VISER_PORT:-8211}"
export CAPX_VISER_PORT

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

# ---------------------------------------------------------------------------
# Pre-flight checklist — things YOU must bring up by hand before a trial.
# This script starts OpenRouter + SAM3 + GraspNet itself, but it CANNOT start
# the arm or the ZED camera service. We probe what we can and then pause so you
# can confirm everything is live. Set CAPX_SKIP_CHECKLIST=1 to skip the pause.
# ---------------------------------------------------------------------------
read_cfg() {  # read_cfg <key>  -> first matching scalar anywhere in the config, else ""
    "$PY" - "$CONFIG" "$1" <<'PY' 2>/dev/null || true
import sys, yaml
cfg = yaml.safe_load(open(sys.argv[1])) or {}
key = sys.argv[2]
def find(d):
    if isinstance(d, dict):
        if key in d:
            return d[key]
        for v in d.values():
            r = find(v)
            if r is not None:
                return r
    return None
v = find(cfg)
print("" if v is None else v)
PY
}

ZED_SOURCE="$(read_cfg piper_zed_source)"; ZED_SOURCE="${ZED_SOURCE:-bridge}"
ZED_SOCK="$(read_cfg piper_zed_service_socket)"; ZED_SOCK="${ZED_SOCK:-/tmp/piper/zed.sock}"
LOW_LEVEL="$(read_cfg low_level)"
STATE_URL="$(read_cfg piper_state_service_url)"
# The arm lives on can0 (can1 is the other adapter). EXPORT it so the preflight
# check below AND the env subprocess use the same channel (the env reads
# PIPER_CAN_CHANNEL; without the export it falls back to its own code default).
# PIPER_CAN_INTERFACE is the backend type (socketcan/gs_usb), NOT a channel name,
# so it's not a fallback here.
export PIPER_CAN_CHANNEL="${PIPER_CAN_CHANNEL:-can0}"
CAN_CH="$PIPER_CAN_CHANNEL"

mark() { case "$1" in ok) echo "  [✓]";; no) echo "  [✗]";; *) echo "  [?]";; esac; }

# 1) Robot arm: CAN link up (+ piper_state_service when this config talks to it).
if ip link show "$CAN_CH" 2>/dev/null | grep -q "state UP\|<.*UP.*>"; then
    CAN_STAT=ok; CAN_MSG="CAN '$CAN_CH' is UP"
else
    CAN_STAT=no; CAN_MSG="CAN '$CAN_CH' is DOWN — run: sudo scripts_realbot/setup_can.sh"
fi
ROBOT_LINE="$(mark "$CAN_STAT") Robot arm powered + E-stop released; $CAN_MSG"

STATE_LINE=""
if [[ "$LOW_LEVEL" == *service* || -n "$STATE_URL" ]]; then
    hp="${STATE_URL#ws://}"; hp="${hp#wss://}"; host="${hp%%:*}"; port="${hp##*:}"; port="${port%%/*}"
    if [[ -n "$host" && -n "$port" ]] && (exec 3<>"/dev/tcp/$host/$port") 2>/dev/null; then
        exec 3>&- 3<&- 2>/dev/null || true
        STATE_LINE="$(mark ok) piper_state_service reachable at $STATE_URL"
    else
        STATE_LINE="$(mark no) piper_state_service NOT reachable at ${STATE_URL:-<unset>} — start it (piper_real_service.yaml §1)"
    fi
fi

# 2) ZED camera service (only in service mode; in bridge mode this script's
#    subprocess opens the camera, so nothing to start by hand).
if [[ "$ZED_SOURCE" == "service" ]]; then
    if "$PY" - "$ZED_SOCK" <<'PY' 2>/dev/null; then
import socket, sys
s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM); s.settimeout(2.0)
s.connect(sys.argv[1]); s.close()
PY
        ZED_LINE="$(mark ok) ZED camera service reachable at $ZED_SOCK"
    else
        ZED_LINE="$(mark no) ZED camera service NOT reachable at $ZED_SOCK — start it: scripts_realbot/zed_service/run_zed_service.sh --socket $ZED_SOCK"
    fi
else
    ZED_LINE="$(mark '?') ZED in 'bridge' mode (piper_zed_source=$ZED_SOURCE) — this script opens the camera itself; no service to start"
fi

# 3) OpenRouter proxy — this script auto-starts it below; just needs the key.
#    Manual / standalone start: scripts_realbot/openrouter_service/run_openrouter_service.sh
OR_LINE="$(mark ok) OpenRouter proxy auto-started by this script on :$OPENROUTER_PORT (.openrouterkey present) — manual: scripts_realbot/openrouter_service/run_openrouter_service.sh --port $OPENROUTER_PORT"

cat <<EOF

============================================================================
  PRE-FLIGHT CHECKLIST  (config: $CONFIG)
$ROBOT_LINE
${STATE_LINE:+$STATE_LINE
}$ZED_LINE
$OR_LINE
============================================================================
EOF

if [[ "${CAPX_SKIP_CHECKLIST:-0}" != "1" ]]; then
    if [[ -t 0 ]]; then
        read -r -p "Bring up anything marked [✗], then press Enter to continue (Ctrl-C to abort)... " _ || true
    else
        echo "(stdin not a TTY — not pausing; set CAPX_SKIP_CHECKLIST=1 to silence this note)"
    fi
fi

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
# it — bring it up if it isn't already serving. Delegates to the standalone
# launcher (scripts_realbot/openrouter_service/run_openrouter_service.sh) so
# there's one source of truth for how this service starts.
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
    bash scripts_realbot/openrouter_service/run_openrouter_service.sh --key-file .openrouterkey --port "$OPENROUTER_PORT"

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
HELPER_WAIT_SECS="${HELPER_WAIT_SECS:-60}"
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
    viser:   :$CAPX_VISER_PORT  (proxied through the web UI; not opened directly)
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
