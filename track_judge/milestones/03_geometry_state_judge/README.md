# Milestone 03 — geometric state-judge: removing the per-turn LLM (VDM) (2026-06-26)

A self-contained snapshot of the **track-judge** paradigm: replace cap-x's per-turn **Visual
Differencing Model** (a *second* LLM call that judges task state every turn) with **agent-written
geometric judgment code** driven by local models (SAM segmentation + TAPIP3D 3D tracking) — so a
*weak* runtime LLM can author the judge in ONE line and there is **zero LLM at judgment time**.

Milestones 01–02 (`capx-se/open_drawer/`) proved a single task can be solved sensing-only with
zero runtime-LLM judgment. Milestone 03 generalizes that into a **reusable judgment paradigm**
and validates it across relations and a compositional task — on tasks the official cap-x VDM
baseline FAILS.

## The headline results (all on real libero data, runtime LLM simulated as allowed)

1. **Call count (the agreed metric).** The judge is pure code → **0 LLM calls per turn** vs the
   VDM's 1/turn. With codegen 1/turn, total LLM calls **halve** (arm B 6 vs arm A 12 over a
   6-turn run). Structural, not stochastic.

2. **Localization on real data.** plain-word name → local SAM mask → TAPIP3D world track →
   centroid lands **0.6–2.7 cm** of the true object pose — well under the DSL's decision
   thresholds.

3. **Coverage.** Three relations — `place_in / on_top_of / opened` (+ `all_of`) — cover **28/30**
   libero-pro dev tasks. The judge is one declarative line:
   `J.judge(ctx, "place_in", target="bowl", reference="plate")`.

4. **GT-validated loops** (judge verdict vs LIBERO `task_completed()`), REGENERATE→FINISH:
   - `opened` (open the middle drawer) — ✅
   - `on_top_of` (put the bowl on the plate) — ✅
   - **compositional** `all_of(opened, place_in)` (**open the top drawer and put the bowl
     inside**, libero_goal task3, **which cap-x VDM FAILS on both swap+task**) — ✅ all 3 turns
     agree with GT; the per-subgoal residual ("drawer open ✓ | bowl still 0% inside") is the
     signal a frame-diff VDM cannot give and why it fails multi-step tasks worst.

5. **Honesty + robustness.** Where an object can't be grounded the judge SAYS so ("could not
   resolve target") instead of faking a verdict; a cross-turn cache reuses a prior turn's
   localization when per-frame SAM drops out (state-dependent flakiness).

## The recurring finding
Across every generalization test the geometry/composition is **sound**; the recurring limiter is
**per-frame semantic GROUNDING** (SAM-by-text on small/branded objects — cheese 0.02 — or
state-dependent — "top drawer" 0.33→0.24 once open). Geometry is not the bottleneck; grounding is.
Next lever: a stronger grounder (molmo point-prompt) behind the same DSL.

## What's in this paradigm (the deliverable, `track_judge/algo/`)
- `judge_dsl.py` — declarative relations (`place_in/on_top_of/next_to/opened/closed/lifted/`
  `grasped/removed_from` + `all_of`); graded, metric, early-abort verdicts.
- `track_judge.py` — the harness seam: dense RGB-D → TAPIP3D → `ctx`; `ctx.points_of(name)`
  (local SAM, score-gated, dense mask-seeded `track_mask` for small objects, cross-turn cache).
- `judge_prompt.py` — tells a weak LLM to write the one-line judge.
- `COVERAGE.md` — the 28/30 mapping. `EVOLUTION.md` / `algo/params.py` — gen log.

## Reproduce
Services: `bash scripts/start_capx_services.sh` (SAM3) + TAPIP3D server (`run_server.sh` in
`../HumanOnlyRobotLearning/.../TAPIP3D`, tapip3d conda env, GPU). Then e.g.:
```
MUJOCO_GL=egl HF_HUB_OFFLINE=1 .venv-libero/bin/python \
  track_judge/results/gen7_compositional/demo_compositional.py /tmp/out
```
Per-result findings + scripts: `track_judge/results/gen{1,4,5,6,7}_*/`.

## Still open
Real **success/speed A/B numbers** vs the VDM need the runtime LLM endpoint on :8110 (a parallel
session is mid-migrating it to a Codex CLI server). The judge *signal* and *mechanism* are
validated; the comparison run is the remaining step. See `CHANGES_vs_capx.md` for the cap-x-side
seam and the gaps this surfaced.
