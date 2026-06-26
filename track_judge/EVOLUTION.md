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

### gen2 — `all_of` multi-condition helper + DSL coverage map 🟢 OFFLINE-VALIDATED · A/B PENDING
- **改动 (change):** (1) `judge_dsl.all_of(ctx, (rel, kwargs), ...)` AND-combines relations so
  2-condition goals are still one call (done=all, abort=any, progress=mean). Prompt updated to
  use it. (2) `track_judge/COVERAGE.md` — mapped every base dev task to a relation.
- **结果 (result):** quantified compatibility (the user's "兼容性强"): **place_in / on_top_of /
  opened cover 28/30 dev tasks** — all 10 `libero_object` (place_in basket), all 10
  `libero_spatial` (on_top_of plate), 8/10 `libero_goal`. Only 2 residuals, both the anticipated
  geometry boundary and neither needing a *per-turn* VLM: goal[5] "push to front of stove"
  (proximity → approx `next_to`) and goal[7] "turn on the stove" (non-spatial state). self-test
  (relations + dispatch + all_of) passes.
- **教训 (lesson):** the spatial qualifiers in libero_spatial ("the bowl between the plate and
  the ramekin") are *grounding*, not judgment — the per-turn state test is identical for all 10,
  which is why so few relations generalize. The VDM is structurally redundant for ~93% of tasks.
- **下一步 (next):** run the A/B benchmark once :8110 is up; gen3 candidate = proprioceptive
  gripper track for `grasped()` (early-abort on slip). Optional: a `moved_to_region` for goal[5].

### gen3 — positive done=True validation on real tracks + on_top_of progress fix 🟢 OFFLINE-VALIDATED · A/B PENDING
- **改动 (change):** added `results/gen1_validation/val_positive.py` (teleport bowl onto plate
  over a rendered window → TAPIP3D → `on_top_of` must flip). Fixed `on_top_of` progress to be
  the last-k region-occupancy fraction (agrees with `done`).
- **结果 (result):** `on_top_of(bowl, plate)` flips to **done=True, progress=1.0** on real
  tracks. The judge is now verified in all three directions on real data: not-done + metric
  residual (A, B) and done=True (C) — closing the degenerate-judge gap AGENT.md §7 warns about.
- **教训 (lesson):** a graded `progress` must agree with `done` (it read 0.097 while done=True
  via a dz-penalty); occupancy-fraction is the consistent form.
- **下一步 (next):** the judge signal is now validated in all directions; the only thing left
  for real numbers is the A/B benchmark (blocked on the :8110 Codex CLI endpoint). gen4
  candidate stays proprioceptive-gripper `grasped()`.

### gen4 — arm-B loop with the runtime LLM SIMULATED (no :8110) + 2 harness fixes 🟢 DEMONSTRATED · A/B PENDING
- **改动 (change):** `results/gen4_loop_demo/` — the experimenter stands in for the cap-agent0
  runtime LLM (writes the 1-line judge per `judge_prompt`) and runs the REAL `TrackJudge.judge_turn`
  seam across 2 turns on `libero_goal/task0`, no endpoint. Two `algo/` fixes the loop surfaced:
  (1) `opened()` keeps a persistent CLOSED baseline in `ctx.state` (multi-turn openness
  accumulates); (2) `ctx.points_of` densely seeds query points inside a small object's mask and
  runs a dedicated TAPIP3D track (`ctx.track_mask`) when the 24×24 grid yields <6 hits.
- **结果 (result):** geometry drives the loop correctly and agrees with GT every turn — turn1
  short pull → not-done → **REGENERATE**; turn2 full pull → **done** ("moved 15.8cm of ~16cm",
  prog 0.99) → **FINISH**; judge-vs-GT AGREE both turns. Before fix (2), the small drawer handle
  read "moved 0.2cm" while fully open (0–1 grid hits → static-background centroid) and the loop
  FAILED — dense mask-seeding tracks the full 15.8cm.
- **教训 (lesson):** acting as the runtime LLM (instead of waiting on :8110) was the fastest way
  to surface gaps a synthetic self-test cannot: small-object seeding and cross-turn baselines.
  The metric residual ("15.8cm of 16cm") is precisely the actionable signal for the next revision.
- **下一步 (next):** A/B *numbers* still need the real runtime LLM endpoint; the mechanism is now
  demonstrated end-to-end on a real task across turns. gen5 candidate: proprioceptive gripper for
  `grasped()`; extend the simulated-LLM loop to a pick-place task (place_in / on_top_of).

<!-- newest generations go ABOVE this line as they happen -->
