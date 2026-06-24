# START HERE — open_drawer continuation handoff

Dogfooding exercise: hand-write the LIBERO `open_..._drawer` skill with cap-x's
legitimate tools (camera RGB-D + proprioception only; NO privileged object poses
in the solving path) to (a) find what cap-x lacks and (b) build a robust solution.
All code here is committed under `[tmp]` commits (latest: `d6d28e5`).

## Where things stand (one line)
Robust collision-aware open-drawer works: zero-disturbance grasp, object
disturbance cut from ~210 mm → 0–32 mm, drawer opens to **−0.10 of −0.16**.
**One step left**: a *partial-release* re-grasp "pump" to reach the −0.15 success
threshold (the thin bar shears out of the gripper ~10 cm into the pull).

## Files (all in `capx-se/open_drawer/`, committed)
- `skill.py` — baseline general skill (cap-x primitives only). Opens the middle
  drawer **3/3 seeds** on `libero_goal/task0` but **NOT robust** (knocks objects).
- `harness.py` — standalone runner for `skill.py` (the missing "inject functions()
  → run → check_success()" path). `--suite/--task-id/--seed/--instruction`.
- `horl_planner.py` — **HORL RRT+trajopt planner wrapped for cap-x** (the robust
  motion engine). In-process, `.venv-libero`, cap-x Franka, collision world from
  depth point cloud. Fixed T=32 + sphere-pool N=96 ⇒ compiles ONCE (~2 min), then
  ~0.6 s/solve. `.plan()` = RRT+trajopt; `.plan_trajopt()` = soft collision-aware.
- `robust_skill.py` — `solve_robust()`: RRT→clear standoff → trajopt final segment
  (zero-disturbance grasp) → collision-aware pull. **This is the robust deliverable.**
- `run_robust.py` — runs `robust_skill` end-to-end + measures object disturbance.
- `GAPS.md` (per-bug cap-x gaps) · `DISCUSSION.md` (framework gaps + §9 robustness,
  §10 the planner integration & the exact remaining tuning).
- `logs/` — validation logs + `agentview.png` etc.

## How to run (next session)
ENV: `.venv-libero/bin/python`, always `MUJOCO_GL=egl HF_HUB_OFFLINE=1`.
Servers needed: **SAM3 (8114)** for perception, **pyroki (8116)** for the control
API's IK warmup / `goto_pose`. (graspnet 8115 only if using `plan_grasp`; the
robust path doesn't.) Start them from the main venv:
```
.venv/bin/python -m capx.serving.launch_pyroki_server --port 8116 &
.venv/bin/python -m capx.serving.launch_sam3_server --port 8114 --device cuda:0 &
```
Baseline skill (reachable middle drawer, should be SUCCESS, not robust):
```
MUJOCO_GL=egl HF_HUB_OFFLINE=1 .venv-libero/bin/python capx-se/open_drawer/harness.py --suite libero_goal --task-id 0
```
Robust skill (zero-disturbance grasp; opens to ~−0.10; ~2 min first compile):
```
MUJOCO_GL=egl HF_HUB_OFFLINE=1 .venv-libero/bin/python capx-se/open_drawer/run_robust.py --suite libero_goal --task-id 0
```
HORL planner source: `HORL_SRC` env (default
`/home/zwa0839/Documents/Projects/HumanOnlyRobotLearning/src`). OMPL+pyroki+jax all
present in `.venv-libero`.

## The exact next step (to finish)
In `robust_skill.py`, the pull is a single collision-aware trajopt that reaches
~−0.10 because the **thin handle bar shears out of the gripper ~10 cm in**.
Implement a **partial-release pump**: pull → (gripper opening to ~0.3, NOT full
open) → nudge back −pull*~0.03 onto the now-more-open handle → re-close → pull
again, 2–3×, until EE stops advancing. (A FULL `open_gripper()` mid-pull loses the
drawer — that regressed it to dq≈0; tested.) Then it should hit qpos ≤ −0.15.
Keep the pull collision-aware (straight pull arm-sweeps the plate → 105 mm).

## Task facts (verified, libero, base frame = robot base at origin)
- Handles (base z): **bottom 0.042 / middle 0.11 / top 0.18**, all at x≈0.70, y≈−0.137.
- Drawer slides along world **+Y** to open (qpos 0 → −0.16); success needs qpos ≤ −0.15.
- Table top base z≈−0.012; the **plate** sits at base x[0.673,0.769], y[−0.054,0.043]
  in front of the drawer — main obstacle. Bowl x≈0.558, cheese x≈0.622.
- **`libero_goal_task/task0`**: instruction (suite metadata) says "middle" but the
  BDDL goal/`:language` say **BOTTOM** — and the **bottom drawer (z=0.042) is
  physically unreachable** by a front grasp (hand-collision box vs table; top-down
  blocked by the stacked drawers above). Use **`libero_goal/task0`** for a
  reachable, self-consistent demo (instruction==goal==middle).
- cap-x patch already applied: `load_libero_task` now sources the instruction from
  the BDDL (matches the scored goal) and warns on mismatch.

## cap-x patches made (besides capx-se/)
- `capx/integrations/libero/__init__.py` — BDDL-language instruction alignment.
- `capx/integrations/franka/libero_reduced.py` — added `get_ee_pose` (was
  referenced-but-undefined → curobo path was dead); exposed curobo fns in
  `functions()`.
- `capx/third_party/curobo/.../geom/sdf/world_mesh.py` — warp 1.14 compat
  (`wp.torch.device_from_torch` → `wp.device_from_torch`). **In the curobo submodule
  working tree — NOT committed to the submodule; re-apply if the submodule is reset.**

## Biggest cap-x gaps found (full list in DISCUSSION.md)
1. Task/eval contract was inconsistent (instruction vs scored goal) — patched.
2. No standalone skill runner (built `harness.py`; also the shape the self-evolve
   Benchmark Evaluator `eval_fn` needs).
3. curobo path was dead (missing `get_ee_pose` + warp drift + not exposed) — patched.
4. **No collision-aware execution + no contact feedback** ⇒ a benchmark "success"
   that knocks objects over. Now addressed via the HORL planner + depth collision world.
5. `solve_ik` silently swaps orientation on failure; pyroki IK ~5 cm off vs curobo
   5 mm; blocking joint controller is step-expensive / exhausts the horizon.
