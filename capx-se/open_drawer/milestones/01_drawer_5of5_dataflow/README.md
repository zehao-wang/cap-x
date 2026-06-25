# Milestone 01 — robust drawer-open + dataflow-instrumented run (2026-06-25)

A snapshot of the `open_drawer` dogfooding at this stage. Sensing-only solve (no
privileged sim state), success **and** safe, generalizes across a perturbed layout,
and now ships cap-x-style run artifacts (multi-view replay + a log that shows where an
agent would call the LLM and on what data).

## What works at this milestone
- **Deterministic 5/5** on `libero_goal/task0` (open the middle drawer), cold runner,
  drawer fully open (`qpos` −0.16), object disturbance single-/low-double-digit mm.
- **Generalizes**: `libero_goal_swap/task0` (cabinet repositioned) succeeds too —
  the failure at the default horizon was pure execution-step budget (gap F), not the skill.
- **Safe-aborts** the unreachable bottom drawer (`libero_goal_task/task0`) instead of
  forcing it.
- **Multi-view replays + dataflow-tagged log** saved per run (this folder is one example).

## Files here
- `example_summary.txt` / `example_trace.json` — the **example log** from one success
  run (`libero_goal/task0` seed3). The log tags each key step `llm` (a decision boundary
  — purpose + the data it uses + what it decides) vs `local` (SAM3 / pyroki-IK / HORL
  planner). This run: **9 decision boundaries, 8 local-model calls**.
- `example_replay_agentview.mp4` / `example_replay_wrist.mp4` — both camera views of the
  same run.
- `CHANGES_vs_capx.md` — what we changed in cap-x core + what we added on top.

## The one-line takeaway from the log
Every *control* decision (seat / pull progress) uses **proprioception only — no image
frames**; *perception* decisions use a **single agentview frame**; only axis estimation
is single-view (the lone decision that structurally wants a sequence/probe). Useful for
deciding which steps justify a richer (tracker/sequence) input vs a per-turn VLM.
