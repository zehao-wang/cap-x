# CHANGES vs stock cap-x — milestone 03 (geometric state-judge)

What this paradigm adds/changes relative to stock cap-x-se, and the cap-x gaps it surfaced.
Per the experiment's rule, runtime behaviour changes only through the `state_judge` switch +
the dense-RGBD recording seam (built once, frozen); everything else is new code under
`track_judge/algo/`.

## The change in one line
Stock cap-x judges per-turn task state with the **VDM** — a second VLM call
(`_get_visual_differencing_feedback`) that diffs before/after frames and returns free text. This
milestone routes that per-turn feedback through **agent-written geometric code** instead
(`state_judge: tracking`), so judgment is **deterministic, local, and LLM-free**.

## What's new (the paradigm, `track_judge/algo/`)
- **A declarative relation DSL** (`judge_dsl.py`) so a *weak* LLM writes the judge as ONE line
  (name a RELATION over OBJECTS in plain words) instead of raw numpy over `[T,N,3]`.
- **`ctx.points_of(name)`** (`track_judge.py`): plain-word name → local **SAM3** mask (score-gated
  at 0.30) → TAPIP3D world tracks. Robustness added as real data demanded it:
  - **dense mask-seeded `track_mask`** — the global 24×24 grid gets 0–1 hits on a small object
    (drawer handle ≈ 0.2% of frame) → its centroid is static background; we densely seed
    `query_point`s inside the mask and run a dedicated track.
  - **cross-turn resolution cache** — SAM-by-text is state-dependent (a drawer resolves closed,
    not open+occluded); a resolved object's last-known world position is reused when a later turn
    fails, rather than faking or giving up.
  - **honest non-resolution** — a named object that can't be grounded yields a "could not resolve"
    verdict (progress unknown), never a confident-wrong verdict from the background grid.
- **Metric, graded, early-abort verdicts** `{done, abort, progress, feedback}` where `feedback`
  is a **metric residual** ("bowl 16.3 cm from container, 0% inside") that directly guides the
  next code revision — not a vague free-text opinion the policy LLM has to re-interpret.

## The integration seam (Phase-0 infra, frozen — not part of a generation)
- `capx/envs/trial.py`: a `state_judge: vdm|tracking` switch; when `tracking`, per-turn feedback
  comes from `track_judge.judge_turn` (the agent's `judge_state(ctx)`), not the VDM.
- `capx/envs/simulators/libero.py`: dense RGB-D capture (`enable_dense_rgbd_capture`) + exact
  `camera_params()` (K + world_to_cam) for the tracker. Gated behind the tracking arm so arm A is
  untouched.

## cap-x gaps this surfaced
1. **The per-turn VDM is a removable, redundant LLM call.** Its errors correlate with the policy
   LLM (same modality/family); geometry is a cheaper, decorrelated, physically grounded signal.
   Removing it = −1 LLM call/turn (−50% total here) with no judgment LLM at all.
2. **Grounding, not geometry, is the generalization limiter.** SAM-by-text fails on small/branded
   objects and is state-dependent. cap-x needs a stronger language-grounded detector (e.g. molmo
   point-prompt) behind the judge, and/or persistence of grounded objects across turns (added
   here as the cross-turn cache).
3. **Judgment wants proprioception too.** `grasped()` (slip / missed-grasp) needs the gripper
   trajectory, which SAM cannot ground ("robot gripper" 0.01); the principled source is robot
   proprioception threaded into `ctx` (a seam extension, not yet done).
4. **A weak runtime model needs reasoning pushed into tooling.** The one-line DSL + local models
   is the concrete realization of "make a weak model succeed by moving judgment out of the LLM".

## Not changed
The cap-agent0 loop, the libero-pro suites, the VDM (arm A), the success metric, and
`benchmark.py`/`compare.py` are untouched, so an A/B remains clean once the runtime LLM endpoint
(:8110) is up.
