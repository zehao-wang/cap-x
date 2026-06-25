# EVOLUTION — track-judge self-evolve log (human-readable mirror)

**Authority = `algo/params.py:CHANGELOG` + the git commits.** This file is the *readable*
mirror, one block per generation; if it ever disagrees with the CHANGELOG/commits, those win.
Numbers stay authoritative in `results/*/summary.json`.

## Required setup (before gen1)

The whole experiment is A (VDM baseline) vs B (tracking). **There is no comparison basis until
the VDM baseline exists**, so the REQUIRED first step is to run arm A on all six suites:

```bash
python track_judge/benchmark.py --baseline --suites \
  libero_object_swap,libero_object_task,libero_goal_swap,libero_goal_task,\
libero_spatial_swap,libero_spatial_task
```

`--baseline` = `--arm vdm --out baseline_vdm`; it writes `results/baseline_vdm/summary.json`,
the exact path `compare.py` and `paper/make_figs.py` read. Re-measure it only when the suites
change. Then begin gen1 (`run_evolve.sh`).

## Accept gate (summary; full rules in AGENT.md §6–§7)

Per gen, compare arm-B `gen<N>` to the current `BEST`:
**ACCEPT iff** success_rate ≥ BEST (within ~1–2 trials' tol) AND median task time ≤ BEST
(within ~5%) AND at least one strictly improves. **HARD REJECT** if the run crashed or the
judge is degenerate (gaming — calls everything "done"). Commit EVERY generation; revert
`algo/` on reject. Held-out suites (`libero_spatial_*`) are never tuned on.

---

## Per-generation template (copy for each gen)

### genN — <one-line title> ✅ ACCEPTED / ❌ REJECTED
- **改动 (change):** the ONE thing edited in `track_judge/algo/` this generation (judge
  primitive / mask-selection strategy / codegen-prompt wording / TAPIP3D param). One hypothesis.
- **结果 (result):**
  - success: BEST `<a>` → gen`<N>` `<b>`  (vs baseline_vdm `<v>`)
  - speed (median task time s): BEST `<a>` → `<b>`  (vs baseline_vdm `<v>`)
  - profile (median s): codegen `<>` / exec `<>` / judge `<>` / track `<>`; judge p90 `<>`
  - judge-vs-GT (arm B): true_done/false_done/missed_done/true_notdone = `<>/<>/<>/<>`
- **教训 (lesson):** why it accepted/rejected; the root cause the tracking-viz showed.
- **下一步 (next):** the binding axis now + the next hypothesis it suggests.

### gen1 — declarative goal-relation DSL (weak-LLM-friendly judge authoring) 🟢 SIGNAL VALIDATED · A/B PENDING
- **改动 (change):** new `algo/judge_dsl.py` — a DECLARATIVE relation layer so a *weak*
  runtime LLM writes the per-turn judge in ONE line instead of raw numpy over `coords`.
  Agent names a goal RELATION + OBJECTS in plain words:
  `J.judge(ctx, "place_in", target="bowl", reference="bin")`. Relations implemented (graded,
  metric, early-abort): `place_in / on_top_of / next_to / opened / closed / lifted /
  grasped(co-moving, slip→abort) / removed_from` (+ aliases open/close/pick_up/stack/...).
  Wired `ctx.points_of("name")` in `algo/track_judge.py` → local **SAM3** segments the named
  object on frame 0 → grid tracks seeded in that mask → `[T,Nt,3]`; graceful whole-frame
  fallback when SAM is down (never crashes the judge). Rewrote `algo/judge_prompt.py` to make
  the one-liner the PRIMARY path and raw-`ctx` code only an escape hatch.
- **结果 (result):** the judge SIGNAL PATH is validated on real libero data (no A/B yet — the
  runtime LLM endpoint :8110 is mid-migration to the Codex CLI server). See
  `results/gen1_validation/findings.md`:
  - (A) **plain-word → local SAM** resolves distinct objects at score **0.58–0.93** (plate
    0.92, bowl 0.92, bottle 0.93, drawer-handle 0.79) while failures sit ≤0.08 (robot gripper
    0.010) → added a **MIN_SEG_SCORE=0.30 gate** so a wrong low-score mask is rejected, not tracked.
  - (B) **full chain on real data** (176 dense frames → TAPIP3D world tracks): a plain-word
    object's tracked **world centroid lands 0.6–2.7 cm from its GT pose** (plate 0.6 cm, wine
    1.7 cm, bowl 2.7 cm), and the DSL relations read the scene correctly (handle "moved 0.2 cm
    of ~15 cm"; bowl "19.2 cm from plate, 0% inside"). Noise ≪ DSL thresholds.
  - offline: `judge_dsl` self-test (all 8 relations + dispatch), import/plumbing, SAM-down
    whole-grid fallback all pass.
- **教训 (lesson):** the north-star ("make a weak model succeed") is an *authoring-ergonomics*
  problem as much as a geometry one — the win is collapsing the judge from "write correct 3D
  numpy" to "name a relation + objects". The metric `feedback` residual (e.g. "4.0cm from
  container center") is what makes the verdict drive the NEXT code revision, not just FINISH.
- **下一步 (next):** (1) run the A/B benchmark vs the VDM baseline once :8110 is up (Codex CLI
  server) — the only thing blocking real success+speed numbers. (2) gen2: take the gripper
  trajectory for `grasped()` from **proprioception** (SAM can't resolve "robot gripper", 0.010),
  exposing an end-effector track on `ctx`. (3) consider 2-condition goals (open+place) and a
  `ctx.state` open-baseline for `closed`.

<!-- newest generations go ABOVE this line as they happen -->
