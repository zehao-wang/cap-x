#!/usr/bin/env bash
# Compile the cap-x gaps writeup to gaps.pdf.
# Mirrors track_judge/paper/build.sh: tectonic (conda pkgs cache binary,
# which needs icu + base libs on LD_LIBRARY_PATH). No figure generation step
# (this is a text writeup); add make_figs.py here later if we want plots.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
TECTONIC="${TECTONIC:-/home/zwa0839/miniconda3/pkgs/tectonic-0.16.9-ha39f199_0/bin/tectonic}"
TEC_LIBS="/home/zwa0839/miniconda3/pkgs/icu-78.3-h33c6efd_0/lib:/home/zwa0839/miniconda3/lib"

( cd "$HERE" && LD_LIBRARY_PATH="$TEC_LIBS:${LD_LIBRARY_PATH:-}" "$TECTONIC" gaps.tex )
echo "[build] wrote $HERE/gaps.pdf"
