# What the cap-x agent framework is missing — and how to improve it

Derived from hand-writing & live-running the LIBERO `open_…_drawer` skill. This
is the framework-level discussion (the per-bug list is in `GAPS.md`).

## What worked
- The legitimate vocabulary (SAM3 text seg + depth→3D + proprio + IK/CuRobo) is
  enough to perceive the handles, derive a grasp, and pull. The **middle drawer
  opens fully** (qpos 0 → −0.16) end-to-end; the probe step recovers the true
  prismatic axis (+Y) from perception alone. So the primitives are sound.

## Where the framework falls short

### 1. Task/eval contract is not self-consistent, and nothing checks it
The instruction the agent sees (suite `task.language` = "middle") contradicted
the scored goal (BDDL = "bottom"). An agent that perfectly follows instructions
scores 0, silently. **Improvement:** instruction and reward must come from one
source (the BDDL), and the loader should *assert* they agree (or log loudly).
Patched here; should be a framework invariant, not a per-task accident.

### 2. No "run one skill against a task" execution path
Everything is wrapped in the LLM exec() loop. There is no clean
`run(code) → check_success()` entrypoint, which is needed for (a) authoring/
debugging skills, (b) the self-evolve Benchmark Evaluator's `eval_fn`
(GOAL.md §⑤ — still missing), and (c) regression tests. **Improvement:**
promote `harness.py` into `capx/` as the canonical headless skill runner; have
the Evaluator call it.

### 3. The motion stack is a collection of half-wired pieces
- The only collision-aware planner (CuRobo) was *broken on three independent
  counts* (missing `get_ee_pose`, warp API drift, not exposed) — i.e. it had
  never run from the reduced API. The HTTP CuRobo client is dead code; the
  pyroki `/plan` route silently ignores obstacles. **Improvement:** one motion
  facade with a tested contract (IK, free-space plan, collision-aware plan), a
  CI smoke test that actually plans+executes one grasp, and deletion of the dead
  paths. A capability the agent is told it has but that errors on first call is
  worse than not having it.
- `solve_ik` silently swaps in a different orientation on failure; `pyroki` IK is
  ~5 cm off at low poses while CuRobo IK is 5 mm. The agent can't see any of
  this. **Improvement:** route precision moves through CuRobo IK; make `solve_ik`
  return/raise on orientation substitution; surface achieved-vs-commanded error.

### 4. Execution model fights the fixed-horizon eval
`move_to_joints_blocking` spends up to `max_steps` sim-steps *per waypoint* and
silently under-converges; a normal trajectory either blows the episode horizon or
stops cm short of the target. **Improvement:** a horizon-aware trajectory
executor (time-parameterized, budget-checked) and an explicit convergence/
failure signal the skill can branch on.

### 5. No contact/collision feedback to the skill
The bottom drawer fails because the hand-collision box hits the table and the
gripper clips the stacked upper handles and a plate — but the skill is **blind**
to all of it (we only saw it via privileged `sim.data.contact`). With perception
only, the skill cannot tell "didn't reach" from "reached but blocked".
**Improvement:** expose proprioceptive collision signals a real robot also has
(commanded-vs-achieved pose error, joint-effort/contact spikes, gripper-didn't-
close-on-anything) as first-class API outputs, and let CuRobo plan against the
*observed* scene mesh so it routes around obstacles (plate/table/upper drawers)
instead of GRAPH_FAIL-ing because the goal grazes a mesh.

### 6. No manipulation skill library for articulated objects
The library has top-down-grasp helpers but nothing for: front/normal grasp,
articulation-axis estimation, or constrained pull/push. The one LLM example just
guesses the pull direction. **Improvement:** seed `capx/atomic_task_library/`
with `estimate_front_normal`, `front_grasp`, `probe_prismatic_axis`,
`pull_open` — exactly the reusable atoms this task needed (and the self-evolve
loop is meant to distill).

### 7. Grasp synthesis is too weak for thin/cluttered handles
Contact-GraspNet on a thin handle and a top-down heuristic don't cover a
low handle wedged above a table, below stacked handles, behind no clearance, with
a plate in front. A scripted force-closure grasp is near-infeasible there; this
is the kind of case that needs either a hook/non-prehensile primitive or
collision-aware grasp+motion co-optimization. **Improvement:** add a
non-prehensile "hook and drag" primitive and integrate grasp selection with the
collision world so only *reachable* grasps are proposed.

### 8. Control mode: position-only, no compliance — the likely root cause
The deepest blocker for the bottom drawer is the control mode. cap-x drives the
arm with a **blocking `JOINT_POSITION` controller** (`move_to_joints_blocking`)
against a **coarse gripper collision box** (`gripper0_hand_collision` half-extent
0.104). Measured floor: the wrist cannot descend below z≈0.09 anywhere near the
table, but the bottom handle is at z=0.042 — so the end-effector physically
cannot reach it from the front/side (grasp *or* hook), and top-down is blocked by
the stacked upper drawers. Every primitive (CuRobo, pyroki IK, hook/drag,
graspnet) hits this same wall; graspnet additionally only returns
low-confidence/degenerate grasps on the thin handles. Trained LIBERO policies
open this drawer because they use **compliant OSC** that pushes through the soft
table/cabinet contacts that a position controller + strict planner treat as hard
stops. **Improvement:** expose an operational-space / impedance control mode (or
a contact-tolerant guarded-move primitive) so skills can make compliant contact
moves; pure position control + coarse collision cannot reach low/cluttered
targets. This is probably the single most important missing capability.

### 9. Robustness: a "success" that knocks over other objects isn't a solution
Measured on the *successful* middle-drawer run: the arm displaced the bowl
~210–240 mm, cream cheese ~100 mm, plate ~40 mm. On a real robot this is a
failure. Findings:
- **Collision-OFF motion** (pyroki IK / curobo `use_world_collision=False`) plows
  through the scene. The reach corridor for the front-normal grasp passes right
  over the bowl that sits in front of the cabinet.
- **Collision-AWARE curobo works but is finicky.** With default robot spheres the
  start config is falsely in-collision → `GRAPH_FAIL`; shrinking the spheres
  (buffer −0.04) plans but the under-sized robot model then *grazes* obstacles for
  real. A full-size robot (buffer 0) + larger `robot_distance_threshold` (0.40)
  both plans and cuts reach-disturbance to ~29 mm in isolation.
- **But it's non-deterministic in the full pipeline**: `plan_to_grasp_poses` does
  IK then a **joint-space plan to the nearest IK solution (`plan_single_js`)**,
  which isn't collision-checked against the world the way the pose plan is — so the
  executed path still sometimes grazes the bowl (~120 mm). The collision guarantee
  is only as good as the weakest stage.
- **Single-view world mesh** (marching cubes from one agentview depth)
  underestimates partially-occluded objects (the bowl), so even a correct planner
  routes too close.
- **The pull is not collision-aware/constrained** and over-pulls past the drawer
  limit (gripper slips, arm flails into the table objects). Mitigated with a
  no-progress stop, a bounded pull distance, and a lift-then-home retreat, but the
  pull should be a guarded, collision-aware constrained motion.

**Improvements:** (a) a single collision-aware motion facade whose *every* stage
(IK seed, joint-space plan, Cartesian pull) is collision-checked; (b) a
**multi-view (agentview+wrist) fused collision world** so obstacles aren't
under-modeled; (c) **obstacle-aware grasp/approach selection** (pick the approach
corridor and the grasp on the handle that are clear of other objects — exactly
what a modern grasp model like AnyGrasp + a real collision world would give);
(d) report object-disturbance as a first-class eval metric alongside
`check_success` so "robust" is measured, not assumed.

## Bottom line
The perception + planning *ingredients* are good, but they're not assembled into
a reliable, observable, eval-consistent control loop. The highest-leverage fixes
are: (1) one consistent task/eval contract, (2) one tested motion facade with a
headless runner, (3) contact/collision feedback + observed-scene collision
planning, (4) a small articulated-manipulation atomic library. With those, both
of the framework's stated north-stars (benchmark accuracy; faster human-in-the-
loop success) get materially easier.
