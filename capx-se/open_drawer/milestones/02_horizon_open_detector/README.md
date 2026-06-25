# Milestone 02 — horizon-optimized pull: open-detector + subsample (2026-06-25)

Builds on milestone 01 (deterministic 5/5, sensing-only, dataflow-instrumented). This
milestone **solves gap F** (the blocking executor exhausting the episode horizon) at the
skill level and cuts the pull's sim-step cost **~8×**, so a *correct* drawer-open now fits a
normal LIBERO horizon with large margin — including the harder repositioned cabinet that
previously failed purely on step budget.

## The headline result
On `libero_goal_swap/task0` seed1 (cabinet repositioned), which used to **FAIL at the
default 30,000-step horizon** (needed 80k — pure gap-F exhaustion, not a skill error):

| | milestone 01 (at 80k) | milestone 02 (at 30k) |
|---|---|---|
| pull sim-steps | 38,924 | **3,910** (10× fewer) |
| total sim-steps | 40,724 | **5,173** (7.9× fewer) |
| drawer qpos | −0.153 | **−0.160 (fully open)** |
| outcome | needed 80k horizon | **SUCCESS at 30k** |

Base task `libero_goal/task0` seeds 3/4/5 all still pass, **fully open −0.16**, at
5,160 / 6,373 / 7,709 sim-steps, object disturbance 11.1 / 24.4 / 4.0 mm — no regression.

## What changed (two sensing-only fixes, both in `robust_skill.py`)
1. **Proprioceptive open-detector.** The pull loop used to run all 10 steps even though the
   drawer bottomed out by step ~2 (TCP advance plateaued at 167–169 mm, just under the old
   `PULL_TRAVEL`=170 break, so it never fired). Now it stops the instant the grip is *holding*
   AND the TCP has dragged ~the full drawer travel (`DRAWER_OPEN_ADV`), or a pull command
   stalls against the hard stop (`STALL_DELTA`). qpos is privileged, so "open" is inferred
   from the advance plateau, not the joint; the grip-holding guard keeps it from over-reading
   on slip. This alone cut the pull from ~38.9k → ~7.5k sim-steps.
2. **Pull-trajectory subsampling.** The pull is a straight +Y drag (waypoints near-collinear,
   so skipping them can't cut a corner toward the plate, which sits *below*), and each blocking
   move otherwise maxes the 120-step convergence cap while fighting the damped drawer. `run()`
   now takes `every=k`; the pull uses `PULL_SUBSAMPLE=2` (command every 2nd waypoint, final
   always kept). Endpoint/openness unchanged; the open-detector self-corrects if a coarse
   command lands short. ~halved the pull again → ~3.9k.

## On the residual object disturbance — accepted, not a bug
The solve sees only the **single agentview depth cloud**. Occluded geometry is simply absent
from the collision world, so when the gripper dips in to seat on a low handle tucked behind
front table objects, a few-mm graze is the **information limit of single-view sensing**, not a
skill error. We treat small grazes (≤ ~3 cm, nothing knocked over) as acceptable — success +
"no gross disturbance" is the SAFE bar, not zero contact. Chasing 0 would overfit to this one
camera pose. The real lever is *more scene info* (second view / wrist-fused cloud / contact
feedback), which is a standing cap-x gap. See `../../DISCUSSION.md` ("Single-view sensing
imposes a SAFETY FLOOR").

## Files here
- `example_summary.txt` / `example_trace.json` — the example log from the headline success run
  (`libero_goal_swap/task0` seed1 @ 30k). Note the new `open-detector` decision step and the
  per-stage `sim_steps` accounting added this milestone.
- `example_replay_agentview.mp4` / `example_replay_wrist.mp4` — both views of that run.
- `CHANGES_vs_capx.md` — what changed vs milestone 01 / stock cap-x.

## cap-x takeaway
Gap F's *workaround* lives in the skill, but the underlying gap is unchanged: cap-x still needs
a **horizon-aware / non-blocking executor** and a **per-waypoint convergence-failure signal**,
so skills don't each have to hand-roll an open/stall detector + trajectory subsampling just to
stay in budget. See `../../GAPS.md` §F.
