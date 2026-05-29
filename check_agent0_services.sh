#!/bin/bash
# Health-check for an agent0 interactive session: the Qwen vLLM server, the web
# UI, the four helper servers (SAM3 / GraspNet / PyRoKi / Molmo), and the Lustre
# IO that any of them can hang on. Safe to run repeatedly; read-only — it starts
# nothing and kills nothing.
#
# Run it from the LOGIN NODE (it reaches into the compute job via `srun
# --overlap`). Running it on the compute node itself also works — it just skips
# the overlap hop. Typical use:
#
#   bash scripts/check_agent0_services.sh
#
# What "remote can't access" usually means once this prints all-green: the
# server side is fine and the problem is your LOCAL `ssh -L` forward — the exact
# command to copy is printed at the very end.
#
# Overrides (all auto-detected from squeue otherwise):
#   QWEN_JOBNAME   (default qwen36-27b)   QWEN_PORT  (default 8000)
#   CAPX_JOBNAME   (default capx-i*)      WEB_UI_PORT(default 8200)
#   COMPUTE_NODE / COMPUTE_JOBID          force the web-UI/helper node + job
#   QWEN_HOST                             force the Qwen node

set -uo pipefail   # NOT -e: a single failed check must not abort the rest.

QWEN_JOBNAME="${QWEN_JOBNAME:-qwen36-27b}"
QWEN_PORT="${QWEN_PORT:-8000}"
QWEN_MODEL="${QWEN_MODEL:-Qwen3.6-27B}"
CAPX_JOBNAME="${CAPX_JOBNAME:-capx-i}"
WEB_UI_PORT="${WEB_UI_PORT:-8200}"
SCRATCH="${CAPX_CACHE_ROOT:-/leonardo_scratch/fast/EUHPC_D33_222/zwang003}"

# Helper servers bound to 127.0.0.1 on the compute node: port -> name.
HELPERS=("8114:sam3" "8115:graspnet" "8116:pyroki" "8122:molmo")

# ---- pretty pass/fail ------------------------------------------------------
if [[ -t 1 ]]; then G=$'\e[32m'; R=$'\e[31m'; Y=$'\e[33m'; B=$'\e[1m'; Z=$'\e[0m'
else G=; R=; Y=; B=; Z=; fi
FAILED=0
ok()   { echo "  ${G}OK${Z}   $*"; }
bad()  { echo "  ${R}FAIL${Z} $*"; FAILED=1; }
warn() { echo "  ${Y}WARN${Z} $*"; }
hdr()  { echo; echo "${B}=== $* ===${Z}"; }

# Find a RUNNING job by name prefix -> "<jobid> <node>" (empty if none).
find_job() {
    squeue -u "$USER" -h -o "%i %j %T %N" \
        | awk -v n="$1" '$3=="RUNNING" && index($2,n)==1 {print $1" "$4; exit}'
}

THIS_HOST="$(hostname -s)"

# ---------------------------------------------------------------------------
hdr "Jobs"
squeue -u "$USER" -h -o "  %.10i %.14j %.8T %.10N %.12L" || true

# ---------------------------------------------------------------------------
hdr "Qwen vLLM server"
if [[ -z "${QWEN_HOST:-}" ]]; then
    read -r _ QWEN_HOST < <(find_job "$QWEN_JOBNAME") || true
fi
if [[ -z "${QWEN_HOST:-}" ]]; then
    bad "no RUNNING '$QWEN_JOBNAME' job (sbatch ../qwen36_27b/serve_qwen36.sh)"
else
    QURL="http://${QWEN_HOST}:${QWEN_PORT}"
    if curl -sSf -m 10 "$QURL/v1/models" >/dev/null 2>&1; then
        ok "/v1/models  ($QURL)"
        # Tiny real completion — proves the model actually generates, not just
        # that the HTTP layer is up.
        RESP=$(curl -sS -m 60 "$QURL/v1/chat/completions" \
            -H 'Content-Type: application/json' \
            -d "{\"model\":\"$QWEN_MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"Reply with exactly: pong\"}],\"max_tokens\":2048,\"temperature\":0}" 2>/dev/null)
        if echo "$RESP" | grep -qi 'pong'; then
            ok "chat completion returns content"
        else
            bad "chat completion gave no usable content: $(echo "$RESP" | head -c 160)"
        fi
    else
        bad "cannot reach $QURL/v1/models"
    fi
fi

# ---------------------------------------------------------------------------
hdr "Compute node (web UI + helpers)"
COMPUTE_JOBID="${COMPUTE_JOBID:-}"
COMPUTE_NODE="${COMPUTE_NODE:-}"
if [[ -z "$COMPUTE_NODE" || -z "$COMPUTE_JOBID" ]]; then
    read -r COMPUTE_JOBID COMPUTE_NODE < <(find_job "$CAPX_JOBNAME") || true
fi
if [[ -z "$COMPUTE_NODE" ]]; then
    bad "no RUNNING '$CAPX_JOBNAME*' job — is the interactive session allocated?"
    echo;  echo "${B}Result: $FAILED failure(s).${Z}";  exit 1
fi
echo "  job=$COMPUTE_JOBID  node=$COMPUTE_NODE"

# Web UI reachable over the internal network from wherever we are. This is the
# SAME hop the `ssh -L ... leo` relay performs, so a 200 here means the server
# side is good and any browser failure is a LOCAL ssh-forward problem.
if [[ "$THIS_HOST" == "$COMPUTE_NODE" ]]; then WEB_TGT="127.0.0.1"; else WEB_TGT="$COMPUTE_NODE"; fi
CODE=$(curl -s -m 10 -o /dev/null -w "%{http_code}" "http://${WEB_TGT}:${WEB_UI_PORT}/" 2>/dev/null)
if [[ "$CODE" == "200" ]]; then
    ok "web UI http://${WEB_TGT}:${WEB_UI_PORT}/  (HTTP 200, internal relay path works)"
elif [[ -n "$CODE" && "$CODE" != "000" ]]; then
    warn "web UI answered HTTP $CODE (up, but not 200)"
else
    bad "web UI http://${WEB_TGT}:${WEB_UI_PORT}/ unreachable from $THIS_HOST"
fi

# ---------------------------------------------------------------------------
# Deep checks that must run ON the compute node. If we're already there, run
# them directly; otherwise reach in with `srun --overlap` against the job.
# ---------------------------------------------------------------------------
run_on_node() {
    if [[ "$THIS_HOST" == "$COMPUTE_NODE" ]]; then
        bash -c "$1"
    else
        timeout 45 srun --overlap --jobid="$COMPUTE_JOBID" -N1 -n1 bash -c "$1" 2>&1
    fi
}

hdr "On-node deep check ($COMPUTE_NODE)"
# Build the remote script. Single-quoted heredoc so $vars expand on the node,
# except the ones we splice in (ports/scratch) which are safe literals.
PORTS_CSV="$(IFS=,; echo "${HELPERS[*]%%:*}")"
REMOTE=$(cat <<REMOTE_EOF
echo "host: \$(hostname -s)"

echo "--- D-state (uninterruptible IO = Lustre hang) procs ---"
DS=\$(ps -eo pid,stat,wchan:20,comm | awk '\$2 ~ /D/ {print "    "\$0}')
[ -n "\$DS" ] && { echo "\$DS"; echo "DSTATE_FOUND"; } || echo "    none"

echo "--- our defunct (zombie) procs ---"
ps -u \$USER -o pid,stat,etime,comm | awk '\$2 ~ /Z/ {print "    "\$0}' | grep . || echo "    none"

echo "--- listening sockets ---"
ss -ltnp 2>/dev/null | grep -E ":${WEB_UI_PORT}|$(echo "$PORTS_CSV" | sed 's/,/|:/g; s/^/:/')" || echo "    none of the expected ports listening"

echo "--- scratch Lustre read (hang => Lustre stuck) ---"
T0=\$SECONDS
if timeout 15 ls -la $SCRATCH/models >/dev/null 2>&1; then
    echo "    ls $SCRATCH/models : ok (\$((SECONDS-T0))s)"
else
    echo "    LUSTRE_HANG ls $SCRATCH/models timed out (>15s)"
fi

echo "--- helper health (localhost on node) ---"
for pn in $(printf '%s ' "${HELPERS[@]}"); do
    p=\${pn%%:*}; n=\${pn#*:}
    # molmo is a vLLM server: probe /v1/models; the rest 404 on / but that still
    # proves the HTTP server is alive.
    if [ "\$n" = molmo ]; then path=/v1/models; else path=/; fi
    c=\$(curl -s -m 8 -o /dev/null -w "%{http_code}" http://127.0.0.1:\$p\$path 2>/dev/null)
    [ -z "\$c" ] && c=000
    if [ "\$c" != "000" ]; then echo "    \$n (\$p): UP (\$c)"; else echo "    \$n (\$p): DOWN"; fi
done
REMOTE_EOF
)

OUT="$(run_on_node "$REMOTE")"
echo "$OUT" | sed 's/^/  /'
# Roll the on-node findings up into the pass/fail tally.
echo "$OUT" | grep -q "DSTATE_FOUND"   && bad "D-state processes present — Lustre IO is hanging on $COMPUTE_NODE"
echo "$OUT" | grep -q "LUSTRE_HANG"    && bad "scratch Lustre ls timed out on $COMPUTE_NODE"
echo "$OUT" | grep -qE "DOWN"          && warn "one or more helper servers DOWN (may still be loading — recheck in a minute)"
echo "$OUT" | grep -q "none of the expected ports" && bad "expected ports not listening on $COMPUTE_NODE"

# ---------------------------------------------------------------------------
hdr "Local port-forward (run in a terminal on YOUR machine)"
echo "    ssh -N -L ${WEB_UI_PORT}:${COMPUTE_NODE}:${WEB_UI_PORT} leo"
echo "  then open:  http://localhost:${WEB_UI_PORT}"
echo "  (if local :${WEB_UI_PORT} is busy, use e.g. -L 1${WEB_UI_PORT}:${COMPUTE_NODE}:${WEB_UI_PORT} and open http://localhost:1${WEB_UI_PORT})"

# ---------------------------------------------------------------------------
echo
if [[ "$FAILED" -eq 0 ]]; then
    echo "${G}${B}Result: all server-side checks passed.${Z} If the browser still can't connect, it's the LOCAL ssh forward above."
else
    echo "${R}${B}Result: $FAILED failing check(s) above.${Z}"
fi
exit "$FAILED"
