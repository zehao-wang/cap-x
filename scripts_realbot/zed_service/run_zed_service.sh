#!/usr/bin/env bash
# Launch the ZED 2i depth service with the raiden venv (pyzed + TRI-Stereo weights).
# All args are forwarded to zed_depth_service.py, e.g.:
#   ./run_zed_service.sh --socket /tmp/piper/zed.sock --variant c64
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RAIDEN_VENV="${RAIDEN_VENV:-/home/zwa0839/Documents/Projects/HumanOnlyRobotLearning/src/packages/raiden/.venv}"
PY="$RAIDEN_VENV/bin/python"

if [[ ! -x "$PY" ]]; then
  echo "raiden venv python not found: $PY" >&2
  echo "Set RAIDEN_VENV to the raiden .venv that has pyzed + TRI-Stereo weights." >&2
  exit 1
fi

exec "$PY" "$HERE/zed_depth_service.py" "$@"
