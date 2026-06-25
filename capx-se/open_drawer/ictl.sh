#!/bin/bash
# Client for interactive.py. Reads a Python command body from stdin, sends it to
# the persistent harness, waits for completion, prints the captured output.
#   echo 'state("hi")' | bash ictl.sh <iodir>
#   bash ictl.sh <iodir> <<'PY'
#   ...multi-line...
#   PY
IO="${1:?usage: ictl.sh <iodir>}"
TIMEOUT="${2:-180}"
TOK="t$(date +%s%N)"
BODY="$(cat)"
printf '#TOKEN %s\n%s\n' "$TOK" "$BODY" > "$IO/in.py.tmp"
mv "$IO/in.py.tmp" "$IO/in.py"          # atomic swap so the server never reads a partial file
i=0
while (( i < TIMEOUT*10 )); do
    if [[ -f "$IO/done" && "$(cat "$IO/done" 2>/dev/null)" == "$TOK" ]]; then
        cat "$IO/out.txt"
        exit 0
    fi
    sleep 0.1; i=$((i+1))
done
echo "ICTL TIMEOUT after ${TIMEOUT}s (server busy or dead)"; exit 1
