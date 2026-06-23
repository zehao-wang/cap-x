#!/usr/bin/env bash
# Regenerate figures/tables then compile the track-judge paper to main.pdf.
# Mirrors self_evolve/paper: make_figs.py (cap-x venv) + tectonic (conda pkgs cache binary,
# which needs the icu + base libs on LD_LIBRARY_PATH).
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
CAPX_PY="${CAPX_PY:-$HERE/../../.venv/bin/python}"
TECTONIC="${TECTONIC:-/home/zwa0839/miniconda3/pkgs/tectonic-0.16.9-ha39f199_0/bin/tectonic}"
TEC_LIBS="/home/zwa0839/miniconda3/pkgs/icu-78.3-h33c6efd_0/lib:/home/zwa0839/miniconda3/lib"

"$CAPX_PY" "$HERE/make_figs.py"
( cd "$HERE" && LD_LIBRARY_PATH="$TEC_LIBS:${LD_LIBRARY_PATH:-}" "$TECTONIC" main.tex )
echo "[build] wrote $HERE/main.pdf"
