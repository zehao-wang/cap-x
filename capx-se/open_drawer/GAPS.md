# cap-x gaps found while hand-writing `open_the_middle_drawer_of_the_cabinet`

Dogfooding result: I hand-authored the LIBERO drawer-opening skill using only the
legitimate `FrankaLiberoApiReducedSkillLibrary` vocabulary (cameras + proprio in,
joint actions out) and tried to run it live (suite `libero_goal_task`, task 0,
seed 1) with the SAM3 (8114) + pyroki (8116) servers and in-process CuRobo.

Legend: **[patched]** = fixed in this session · **[flag]** = needs a decision.

---

## A. Task is mis-specified by cap-x — the headline gap  **[patched]**

LIBERO-PRO's `_task` variant perturbs the BDDL of
`libero_goal_task/open_the_middle_drawer_of_the_cabinet.bddl`:

| source | value |
|---|---|
| filename | …`open_the_middle_drawer`… |
| BDDL `(:language …)` | **"open the bottom drawer of the cabinet"** |
| BDDL `(:goal …)` | `(Open wooden_cabinet_1_**bottom**_region)` |
| suite metadata `task.language` | **"open the middle drawer of the cabinet"** |

`load_libero_task` (capx/integrations/libero/__init__.py) **deliberately used the
suite metadata language** ("middle"), but `check_success()` scores the BDDL goal
("bottom"). So the agent is instructed to open the *middle* drawer while success
requires the *bottom* drawer — **following the instruction can never pass**, and
the whole `_task`/`_swap` arm is silently mis-scored.

Verified: setting `wooden_cabinet_1_middle_level` fully open → `check_success()=False`;
setting `wooden_cabinet_1_bottom_level=-0.16` → `True`.

**[patched]** `load_libero_task` now reads the BDDL `(:language …)` (same source as
the scored goal), prefers it, and logs a warning when it diverges from suite
metadata. Non-perturbed suites are unaffected (languages match).
**[flag]** Confirm this is the intended scoring for the PRO `_task` suites — it
flips the instruction shown to the agent across every perturbed task.

## B. CuRobo planning was completely broken in the reduced API  **[patched ×2]**

The CuRobo trajectory primitives are the *only* collision-aware planning cap-x
has (the pyroki `/plan` route ignores obstacles, and the HTTP CuRobo client
`capx/integrations/motion/curobo.py` is dead code). Yet they were unusable:

1. **`get_ee_pose` never existed.** `update_curobo_world_with_object` and
   `plan_with_grasped_object` (libero_reduced.py) call `self.get_ee_pose()`, but
   no class defines it (the example codes in `franka_libero_env.py` call it too —
   they could never have run against this API). **[patched]** added
   `FrankaLiberoApiReduced.get_ee_pose()` (reads `robot_cartesian_pos` — real-robot
   legitimate).
2. **warp incompatibility.** Vendored CuRobo `geom/sdf/world_mesh.py:67` calls
   `wp.torch.device_from_torch`, gone in warp 1.14 (it's `wp.device_from_torch`).
   Every collision-world build raised `AttributeError`. **[patched]** compat shim.
3. **Not exposed.** `plan_grasp_trajectory` / `plan_with_grasped_object` /
   `execute_joint_trajectory` / `parse_grasp_poses_for_curobo` were commented out
   of `functions()`. **[patched]** exposed (now that #1/#2 make them work). The
   user explicitly wanted CuRobo usable.

After A/B the CuRobo IK path works and is **accurate (5 mm)** — much better than
pyroki for low grasps.

## C. CuRobo collision-aware planning fails reaching into the cabinet  **[flag]**

`plan_grasp_trajectory(use_world_collision=True)` returns `GRAPH_FAIL` /
"Start or End state in collision" for a handle adjacent to the cabinet mesh
(start home config flagged in-collision; goal touches the scene mesh). Workable
only with `use_world_collision=False` (free-space joint plan to the accurate IK).
Needs better object/scene collision-ignoring or sphere-buffer tuning for
reach-into-container tasks.

## D. `solve_ik` silently substitutes orientation  **[flag]**

`FrankaLiberoApiReduced.solve_ik` tries the requested quaternion, then silently
falls back to top-down / 45° / side orientations and returns whatever solves —
the caller cannot tell which orientation it got. For a directional grasp (drawer
handle) this breaks Cartesian continuity and the grasp. The first run jittered
because of this. Suggest a `strict=True` mode (raise instead of silently
re-orienting), or return the achieved orientation.

## E. pyroki IK is inaccurate at low/awkward poses  **[flag]**

The pyroki `/ik` soft least-squares solver (pos_weight 50) left ~5 cm error at
the low handle (z≈0.04) → grasped above the handle. CuRobo IK hit it to 5 mm.
For precision grasps the skill should prefer CuRobo IK; pyroki is fine for
coarse free-space moves. Consider routing `solve_ik`/`goto_pose` through CuRobo
IK when available, or tightening the pyroki cost / iterations.

## F. The blocking joint controller exhausts the episode horizon  **[flag]**

`move_to_joints_blocking` spends up to `max_steps` sim-steps *per waypoint*;
`execute_joint_trajectory(subsample=1, max_steps=400)` over a 30-waypoint plan
blew the default horizon → robosuite `"executing action in terminated episode"`.
Also the default `max_steps=120` under-converged (achieved joints 0.39 rad short,
~5 cm off) until raised to ~250–400. Needs a horizon-aware execution budget (or a
non-blocking controller) so trajectories fit a normal LIBERO horizon, plus a
convergence-failure signal instead of silently stopping short.

**Session 4 (2026-06-25) — measured + skill-side fix (largely resolved for the pull).**
Instrumented `run_robust.py` to record `_sim_step_count` per stage. On the gap-F case
(`libero_goal_swap/task0` seed1) the **pull was 38,924 of 40,724 total sim-steps (95.6%)**
— and the loop ran all 10 pull steps even though the drawer was fully open by step ~2,
because the open-detector was missing. Each waypoint maxes the 120-step convergence cap
(the position controller can't settle to 0.01 rad while dragging the damped drawer:
38,924 / 10 steps / 32 waypoints ≈ 121 ≈ the cap), so every wasted pull step costs
~3.9k sim-steps. **Fix (skill-side, sensing-only):** a PROPRIOCEPTIVE open-detector in
`robust_skill.py` — once the grip is holding AND the TCP has advanced ~the full drawer
travel (`DRAWER_OPEN_ADV=0.15`), or a pull command stalls against the hard stop
(`STALL_DELTA`), stop pulling (qpos is privileged, so "open" is inferred from the TCP
advance plateau, not the joint). Result: the gap-F case now **SUCCEEDS at the original
30,000 horizon** (8,739 sim-steps total, pull 7,512, drawer fully open −0.16) — was a
FAIL needing 80k. Base task seeds 3/4/5 all still pass fully-open at ~9–14k steps, with
no disturbance regression (seed 4 even improved 32→22 mm). **The cap-x-side gap remains
real** for any skill that issues many waypoints near a load: cap-x still needs a
horizon-aware / non-blocking executor + a per-waypoint convergence-failure signal, so
skills don't each have to hand-roll an open/stall detector to stay in budget.

## G. No articulation-axis estimation primitive  **[worked around in skill]**

Opening a drawer needs the prismatic (slide) axis. cap-x has `select_top_down_grasp`
(top-down only) but nothing for a *front/horizontal* grasp or for estimating an
articulation axis. PCA plane-fit on the cabinet points gave the wrong axis
(−X; truth +Y) because depth sees front+top faces. The LLM example
(`lgs_task3_open_drawer_code`) just hardcodes/guesses the pull direction. The
robust legitimate fix (implemented in `skill.py`): estimate from cabinet-centroid
→ handle, then **probe** (pull a little, re-observe the handle's 3D displacement)
to recover the true axis. Good candidates for the atomic-task library:
`front_grasp_from_normal`, `estimate_prismatic_axis_by_probe`, `pull_open`.

## H. Coupling / ergonomics  **[flag]**

- **No standalone skill runner.** Every skill runs inside the LLM exec() loop
  (`trial.py`); there's no "inject `functions()`, run code, report
  `check_success()`". Built one here (`harness.py`) — it's also the shape of the
  `eval_fn` the self-evolve Benchmark Evaluator (GOAL.md §⑤) still needs.
- **API ctor warms up pyroki** (eager IK in `__init__`) — constructing the control
  API requires the pyroki server even for a skill that wouldn't use it. (SAM3 /
  graspnet / molmo inits are lazy, which is good.)
- **pyroki server has no `/health` route** (returns 404); harness/launch readiness
  checks should hit a real endpoint.
- **Depth shape inconsistency**: raw obs depth is `(H,W,1)`; the reduced API's
  `get_observation()` squeezes to `(H,W)`. Helpers accept both, but it's a
  contract foot-gun.
- **Example code rot**: the `*_CODE` blocks in `franka_libero_env.py` call
  functions not in the reduced `functions()` (`get_ee_pose`,
  `get_object_3d_points_and_masks_from_language`, `query_VLM`,
  `get_top_down_grasp_from_obb`) — stale against the current API.

---

## VALIDATION: full success on a reachable drawer scenario

`libero_goal` task 0 (`open_the_middle_drawer_of_the_cabinet`, the NON-perturbed
suite where instruction == goal == middle region): the general skill opens it
**end-to-end, `check_success()=True`, 3/3 seeds (1,2,3)**. Perception-only:
detect 3 handles → pick "middle" from the instruction → estimate axis from
cabinet→handle → probe-refine to +Y → CuRobo reach + deep grasp (gripper holds)
→ pull middle qpos 0 → −0.16. No scene-specific constants; handle positions and
the refined axis vary per seed.

The bottom drawer (`libero_goal_task`) remains blocked by the reach/control
limits below (to be revisited with AnyGrasp + possibly compliant control).

## What actually happened on the sim (evidence)

- **Pipeline validated** on the reachable **middle** drawer: CuRobo collision-off
  reach (accurate) + deep grasp (gripper opening 0.168 → holding the bar) + +Y
  pull → `middle_level` qpos 0.0 → **−0.160 (fully open)**.
- **Bottom drawer (the scored target) is unreachable** by a front grasp: every
  reach stalls at z≈0.07–0.11 (handle at 0.042), gripper closes empty
  (opening ≈ 0) — a physical reach/collision limit of this scene, on top of the
  instruction/goal mismatch (A).

Net: the skill correctly opens the drawer it is told to (middle), but
`check_success()` stays False because cap-x's instruction (middle) contradicts
the perturbed scored goal (bottom), and the bottom drawer is additionally
unreachable. Fixing A makes instruction == goal; C/E/F are what stand between the
current stack and reliably opening even a reachable drawer.

---

## G. Stochastic trajopt seat is not REPEATABLE — the precision-grasp gap  **[worked around]** (session 2)

> **Resolution:** replaced the trajopt seat with a **deterministic pyroki `solve_ik`
> seat** (solve_ik at the bar pose + linear joint interp over the sensing-verified
> clear corridor). Lands the TCP repeatably deep+centered (z 0.109/0.110/0.110 vs the
> trajopt's 0.10–0.17); on-bar on the first try on the cold runner. Caveat: pyroki IK
> accuracy depends on a good warm-start (a high arm config → a z≈0.14 solution), so the
> skill re-descends + re-solves if the IK lands off-bar. Underlying gap (no first-class
> deterministic short-range seat primitive) stands; worked around with solve_ik.

The HORL pyroki trajopt is the motion engine. For TRANSIT it's fine. For the final
~5 cm **seat** onto a thin handle it is the wrong tool:

- The same seat target lands the TCP at **z scattering 0.10–0.17 by RNG seed** (and
  y at the tip −0.119 vs the bar center −0.137). A high-z landing grips *above* the
  bar and pulls nothing; only a vertically-centered (z≈0.10) grip drags the drawer.
- It intermittently returns **NaN final_cost** (diverged) and **phantom ~1e10
  collision cost** for a gripper passing 7 cm *above* a flat object — a real
  disturbance of only ~3 mm. So the cost cannot be trusted as a safety gate; we gate
  on the *executed* TCP (sensing/proprioception) + grip instead.
- Net effect: with a **warm** planner (RNG advanced) we hit good seats → 5/5
  in-harness; with a **cold** planner (one process per seed) the seat RNG gates
  success. The skill stays SAFE (rejects bad seats, aborts with 0 disturbance) but
  is not deterministic.

**What cap-x needs:** a deterministic short-range **seat primitive** — direct IK at
the grasp pose (curobo / a seeded pyroki IK) + linear joint interp over the
sensing-verified clear corridor — so precision grasping doesn't ride on trajopt RNG.

## H. RRT-to-a-near-structure goal fails silently  **[flag]** (session 2)

`planner.plan(...)` to a pre-grasp close to the cabinet intermittently returns
not-planned with `RRTConnect: Motion planning start tree could not be initialized`
(the start config reads as in-collision against the depth cloud). The caller gets a
silent no-op and, unless it VERIFIES the achieved pose, proceeds from the wrong
place (stranded at the high standoff → a too-high seat). Needs either a reliable
near-contact RRT or a documented "verify the move landed" contract.

## I. No standalone multi-round sim-driving loop  **[built]** (session 2)

cap-x had no way to "run a segment → observe sensing+GT → write/run the next
segment" against a live env while keeping the (expensive) planner warm. Built
`interactive.py` + `ictl.sh` (persistent namespace, file-inbox protocol,
`reset_env(s)` for the reset-on-collision rule). This is also the shape a
self-evolve interactive evaluator would need.
