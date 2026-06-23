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

<!-- newest generations go ABOVE this line as they happen -->
