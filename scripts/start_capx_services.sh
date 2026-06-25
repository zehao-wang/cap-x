#!/bin/bash
# Start ALL cap-x default services used by the dogfooding / self-evolve tasks.
#
# These are the perception + planning tools the reduced-API skills call into:
#   - sam3      (8114)  text-prompt segmentation   [GPU]
#   - graspnet  (8115)  Contact-GraspNet grasp proposals  [GPU]
#   - pyroki    (8116)  IK / goto_pose / motion solving   [CPU]
#   - molmo     (8122)  point-prompt VLM (reduced API point_prompt_molmo) [GPU, vLLM]
#
# Policy (per AGENT.md): start EVERY default service EXCEPT openrouter before you
# begin coding a task. openrouter is the agent's own LLM endpoint, launched
# separately; everything else here is a tool the solution code may need.
#
# Idempotent: a service already listening on its port is left alone. Safe to
# re-run. Logs go to logs/services/<name>.log.
#
# Usage:
#   bash scripts/start_capx_services.sh            # start all, wait for ready
#   bash scripts/start_capx_services.sh --status   # just report up/down, start nothing
#
# Env overrides: SAM3_PORT GRASPNET_PORT PYROKI_PORT MOLMO_PORT, GPU (default cuda).

set -uo pipefail   # NOT -e: one service failing must not abort the rest.

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
VENV="${CAPX_VENV:-$ROOT/.venv/bin/python}"
LOGDIR="$ROOT/logs/services"
mkdir -p "$LOGDIR"

SAM3_PORT="${SAM3_PORT:-8114}"
GRASPNET_PORT="${GRASPNET_PORT:-8115}"
PYROKI_PORT="${PYROKI_PORT:-8116}"
MOLMO_PORT="${MOLMO_PORT:-8122}"
GPU="${GPU:-cuda}"

if [[ -t 1 ]]; then G=$'\e[32m'; R=$'\e[31m'; Y=$'\e[33m'; B=$'\e[1m'; Z=$'\e[0m'
else G=; R=; Y=; B=; Z=; fi

port_up() { python3 -c "import socket,sys
s=socket.socket(); s.settimeout(1.0)
sys.exit(0 if s.connect_ex(('127.0.0.1',int(sys.argv[1])))==0 else 1)" "$1" 2>/dev/null; }

# start_svc <name> <port> <cmd...>
start_svc() {
    local name="$1" port="$2"; shift 2
    if port_up "$port"; then
        echo "  ${G}UP${Z}    $name (port $port) — already running, skip"
        return 0
    fi
    echo "  ${Y}START${Z} $name (port $port) -> $LOGDIR/$name.log"
    nohup "$@" > "$LOGDIR/$name.log" 2>&1 &
    echo "        pid $!"
}

wait_ready() {
    local name="$1" port="$2" timeout="${3:-180}"
    local i=0
    while (( i < timeout )); do
        port_up "$port" && { echo "  ${G}READY${Z} $name (port $port) after ${i}s"; return 0; }
        sleep 3; i=$((i+3))
    done
    echo "  ${R}TIMEOUT${Z} $name (port $port) not ready in ${timeout}s — see $LOGDIR/$name.log"
    return 1
}

# ---- status-only mode ------------------------------------------------------
if [[ "${1:-}" == "--status" ]]; then
    echo "${B}cap-x service status${Z}"
    for pair in "sam3:$SAM3_PORT" "graspnet:$GRASPNET_PORT" "pyroki:$PYROKI_PORT" "molmo:$MOLMO_PORT"; do
        n="${pair%%:*}"; p="${pair##*:}"
        if port_up "$p"; then echo "  ${G}UP${Z}   $n ($p)"; else echo "  ${R}DOWN${Z} $n ($p)"; fi
    done
    exit 0
fi

echo "${B}Starting cap-x services (all except openrouter)${Z}"
echo "venv: $VENV"

# ---- GPU perception + CPU planning (main venv) -----------------------------
start_svc sam3     "$SAM3_PORT"     "$VENV" -m capx.serving.launch_sam3_server     --port "$SAM3_PORT"     --device "$GPU"
start_svc graspnet "$GRASPNET_PORT" "$VENV" -m capx.serving.launch_contact_graspnet_server --port "$GRASPNET_PORT" --device "$GPU"
start_svc pyroki   "$PYROKI_PORT"   "$VENV" -m capx.serving.launch_pyroki_server   --port "$PYROKI_PORT" \
          --robot-urdf-name panda_description --target-link-name panda_hand

# ---- molmo (vLLM point-prompt VLM) -----------------------------------------
# vLLM is the `molmo` extra (vllm==0.15.0). Serve Molmo2-8B from whichever uv
# venv actually has vLLM installed; serve it directly (the HPC serve_molmo_local.sh
# wants a conda `vllm` env + Leonardo paths, so it's not used locally). If no venv
# has vLLM, warn + skip: install with `uv sync --extra molmo`. init_molmo is lazy,
# so only point_prompt_molmo is unavailable when molmo is down (SAM3 still covers
# segmentation).
# Prefer the dedicated .venv-molmo (vLLM is heavy + pins conflict with cap-x's
# robosuite/transformers, so it lives in its own env; molmo.py talks to it over
# HTTP, fully decoupled). Fall back to any other venv that happens to have vLLM.
MOLMO_VENV=""
for cand in "$ROOT/.venv-molmo/bin/python" "$VENV" "$ROOT/.venv-libero/bin/python"; do
    [[ -x "$cand" ]] && "$cand" -c "import vllm" 2>/dev/null && { MOLMO_VENV="$cand"; break; }
done
if port_up "$MOLMO_PORT"; then
    echo "  ${G}UP${Z}    molmo (port $MOLMO_PORT) — already running, skip"
elif [[ -n "$MOLMO_VENV" ]]; then
    echo "  ${Y}START${Z} molmo (port $MOLMO_PORT) via vLLM in $MOLMO_VENV"
    # GPU 1 by default (GPU 0 holds SAM3/graspnet); --enforce-eager skips vLLM's
    # slow first-run cudagraph capture + inductor compile (~20 min) — point-prompt
    # latency doesn't need it, and startup drops to ~2 min and is far more reliable.
    CUDA_VISIBLE_DEVICES="${MOLMO_GPU:-1}" HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-0}" \
        nohup "$MOLMO_VENV" -m vllm.entrypoints.openai.api_server \
        --model allenai/Molmo2-8B --served-model-name allenai/Molmo2-8B \
        --port "$MOLMO_PORT" --host 127.0.0.1 --trust-remote-code --enforce-eager \
        --dtype bfloat16 --max-model-len 8192 --max-num-batched-tokens 16384 \
        --gpu-memory-utilization 0.85 > "$LOGDIR/molmo.log" 2>&1 &
    echo "        pid $!"
else
    echo "  ${Y}SKIP${Z}  molmo (port $MOLMO_PORT) — vLLM not installed in any venv."
    echo "        install (dedicated env):  uv venv .venv-molmo --python 3.12 &&"
    echo "                                  uv pip install --python .venv-molmo vllm==0.15.0"
    echo "        point_prompt_molmo unavailable meanwhile; SAM3 covers segmentation."
fi

echo "${B}Waiting for readiness...${Z}"
wait_ready sam3     "$SAM3_PORT"     180
wait_ready graspnet "$GRASPNET_PORT" 180
wait_ready pyroki   "$PYROKI_PORT"   180
if port_up "$MOLMO_PORT"; then wait_ready molmo "$MOLMO_PORT" 600; fi

echo "${B}Done.${Z} Re-run with --status to recheck."
