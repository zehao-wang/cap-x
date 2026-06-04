#!/usr/bin/env bash
# Launch the OpenRouter chat-completions proxy used by the REAL Agilex PIPER arm.
# The real robot's LLM (and the VDM differencing model) is OpenRouter Gemini
# served through this local proxy on :8110 — NOT the Qwen vLLM server that
# robosuite/libero use. run_agent0_piper_interactive.sh auto-starts this for you;
# this script is the single source of truth so you can also start it by hand.
#
#   ./run_openrouter_service.sh                       # key-file .openrouterkey, port 8110
#   ./run_openrouter_service.sh --port 8110           # any extra args pass straight through
#   OPENROUTER_PORT=8110 ./run_openrouter_service.sh  # env overrides the defaults
#
# All args are forwarded to capx/serving/openrouter_server.py.
set -euo pipefail

REPO_ROOT="$(git rev-parse --show-toplevel)"
cd "$REPO_ROOT"

PY="${CAPX_PYTHON:-.venv/bin/python}"
KEY_FILE="${OPENROUTER_KEY_FILE:-.openrouterkey}"
PORT="${OPENROUTER_PORT:-8110}"

if [[ ! -x "$PY" ]]; then
  echo "ERROR: python not found at $PY (set CAPX_PYTHON)." >&2
  exit 1
fi
if [[ ! -f "$KEY_FILE" ]]; then
  echo "ERROR: OpenRouter key file missing: $KEY_FILE" >&2
  echo "       echo 'sk-or-v1-...' > $KEY_FILE   (repo root, git-ignored)" >&2
  exit 1
fi

# When invoked bare, fall back to the env-driven defaults; otherwise honour
# whatever the caller passed (e.g. the interactive launcher's --key-file/--port).
if (( $# == 0 )); then
  set -- --key-file "$KEY_FILE" --port "$PORT"
fi

exec "$PY" capx/serving/openrouter_server.py "$@"
