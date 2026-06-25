# Changes at milestone 02 (vs milestone 01 / stock cap-x)

Milestone 01's changes still hold (see `../01_drawer_5of5_dataflow/CHANGES_vs_capx.md`).
This milestone adds **no new cap-x-core patches** — the horizon win is entirely skill-side,
plus a measurement hook in the runner. Full rationale in `../../GAPS.md` §F /
`../../DISCUSSION.md`.

## Skill-side (`robust_skill.py`)
- **Proprioceptive pull open-detector.** New constants `PULL_MIN_OPEN`, `STALL_DELTA`,
  `DRAWER_OPEN_ADV`; the pull loop stops as soon as the grip is holding and the TCP has
  advanced ~the full drawer travel, or a pull command stalls against the hard stop. Replaces
  the old "always grind 10 steps until `advanced() > PULL_TRAVEL`" (which never fired because
  the advance plateaus just under the threshold).
- **Pull-trajectory subsampling.** `run()` gained an `every=k` arg; the pull calls
  `run(tr, every=PULL_SUBSAMPLE=2)`. Transit/seat stay full-resolution (collision routing
  matters there); only the straight-line pull is subsampled.

## Runner-side (`run_robust.py`)
- `_dbg` now records `env._sim_step_count` per stage (`sim_steps` / `stage_sim_steps` in
  `trace.json`). This is the measurement that localized gap F to the pull (95.6% of the
  budget) — pure instrumentation, measurement-only, not in the solve path.

## NOT changed (deliberately)
- No cap-x-core executor change. The real fix (horizon-aware / non-blocking executor +
  convergence-failure signal) belongs in cap-x; here it is worked around in the skill so the
  skill fits a normal horizon. Logged as the standing gap F.
- Residual object disturbance (~4–30 mm, seed-dependent) is left as-is: it is the single-view
  sensing safety floor (occluded geometry absent from the collision world), accepted rather
  than overfit away.
