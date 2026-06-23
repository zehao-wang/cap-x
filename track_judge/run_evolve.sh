#!/usr/bin/env bash
# Launch the track-judge self-evolve loop FULLY AUTONOMOUSLY (no permission prompts).
#
#   bash track_judge/run_evolve.sh [num_generations]     # default 6
#
# Design (mirrors motion_plan/self_evolve/run_evolve.sh): ONE fresh Claude Code session PER
# generation. Each session starts from the CURRENT on-disk state (git HEAD, results/BEST,
# latest results/genNNN), reads track_judge/AGENT.md, does EXACTLY ONE generation (one
# hypothesis -> edit track_judge/algo/ -> eval one arm -> accept/reject -> commit), and exits.
# Generation N+1 therefore builds on the committed result of generation N — never in parallel,
# never sharing context. The loop is RESUMABLE: every generation is committed and BEST
# persists. NEVER git push. No human feedback anywhere.
#
# Stops early on 3 consecutive REJECT commits (subject contains "REJECT").
set -euo pipefail

N="${1:-6}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"   # .../track_judge
REPO="$(cd "$HERE/.." && pwd)"
cd "$REPO"

# Raise the spawned -p sessions' Bash timeout cap so a full libero-pro arm eval runs as ONE
# FOREGROUND blocking call (without this the 10-min cap forces backgrounding -> the session
# ends with no commit).
export BASH_DEFAULT_TIMEOUT_MS=300000      # 5 min default for ordinary commands
export BASH_MAX_TIMEOUT_MS=3600000         # 60 min ceiling; the eval sets its own large timeout

PER_GEN_PROMPT="Run EXACTLY ONE generation of the track-judge self-evolve loop FULLY \
AUTONOMOUSLY, then exit. FIRST read track_judge/AGENT.md (the autonomous loop, the accept \
gate §6/§7, the honesty rules §7, logging/commit §8, and the 'Two levels' section) — it is \
your only briefing. Start from the CURRENT on-disk state: inspect git HEAD, \
track_judge/results/BEST, the latest track_judge/results/genNNN/summary.json, and \
track_judge/results/baseline_vdm/. Form ONE hypothesis on the binding metric, edit ONLY \
track_judge/algo/ (one judge primitive / mask strategy / codegen-prompt tweak / TAPIP3D \
param), append one CHANGELOG line in algo/params.py, and save tracking-viz. At DECISION time \
run the arm-B eval of YOUR algo with ONE foreground command: \
  python track_judge/benchmark.py --arm tracking --suites <dev> --gen <N> \
then compare to BEST: python track_judge/compare.py gen<N> <BEST> (and vs baseline_vdm). \
ACCEPT iff success >= BEST within tol AND median task time <= BEST within tol AND at least \
one strictly improves. On ACCEPT: echo gen<N> > track_judge/results/BEST, keep algo/. On \
REJECT: git checkout -- track_judge/algo/ (keep results/genNNN + the CHANGELOG line). Update \
EVOLUTION.md either way and git-commit EVERY generation (subject 'gen<N>(track-judge): <change> \
— ACCEPTED' or '— REJECTED'; NO Co-Authored-By / Claude trailer; NEVER git push). CRITICAL: \
this is a headless -p session with NO async notification — NEVER use run_in_background, \
Monitor, or any 'I will wait for the notification' pattern; doing so silently ENDS the \
session and LOSES the generation. Run EVERY long command as a foreground Bash tool call with \
a large timeout (~3000000 ms) and WAIT inline. Do NOT ask for permission — decide and act. \
Finish with the one-line tagged summary."

echo "[run_evolve] launching up to ${N} generations, one fresh session each …"

consec_rejects=0; consec_fails=0
for i in $(seq 1 "$N"); do
  echo "[run_evolve] === generation session ${i}/${N} ==="
  before="$(git -C "$REPO" rev-parse HEAD)"
  log="track_judge/evolve_run_$(date +%Y%m%d_%H%M%S)_sess${i}.log"
  claude --dangerously-skip-permissions -p "$PER_GEN_PROMPT" 2>&1 | tee "$log"
  after="$(git -C "$REPO" rev-parse HEAD)"

  # No commit -> almost always the -p background-and-die failure. Restore any half-finished
  # algo/ edit, count it as a FAILURE (not a reject), abort if it keeps happening.
  if [ "$before" = "$after" ]; then
    git -C "$REPO" checkout -- track_judge/algo/ 2>/dev/null || true
    consec_fails=$((consec_fails + 1)); consec_rejects=0
    echo "[run_evolve] session ${i}: NO COMMIT (likely backgrounded-and-died) — consecutive fails: ${consec_fails}"
    if [ "$consec_fails" -ge 3 ]; then
      echo "[run_evolve] stopping: 3 consecutive sessions made no commit (loop not progressing)."
      break
    fi
    continue
  fi
  consec_fails=0

  subj="$(git -C "$REPO" log -1 --format=%s 2>/dev/null || echo '')"
  echo "[run_evolve] session ${i} head commit: ${subj}"
  if printf '%s' "$subj" | grep -qi 'REJECT'; then
    consec_rejects=$((consec_rejects + 1))
    echo "[run_evolve] consecutive rejects: ${consec_rejects}"
    if [ "$consec_rejects" -ge 3 ]; then
      echo "[run_evolve] stopping early: 3 consecutive rejects."
      break
    fi
  else
    consec_rejects=0
  fi
done

echo "[run_evolve] done."
