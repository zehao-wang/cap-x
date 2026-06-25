# START HERE — open_drawer continuation handoff

Dogfooding: hand-write the LIBERO `open_..._drawer` skill with cap-x's legitimate
tools (camera RGB-D + proprioception only; NO privileged poses in the solve path)
to (a) find what cap-x lacks and (b) build a robust + SAFE solution. Commits: old
ones `[tmp]`, new ones `[auto]` (per AGENT.md). **Start every session by launching
all services: `bash scripts/start_capx_services.sh`** (SAM3/graspnet/pyroki/molmo).

## Session 3 (2026-06-25): gaps writeup compiled + other drawer settings probed
- **cap-x gaps PDF** built: `paper/gaps.tex` + `paper/build.sh` (mirrors
  `track_judge/paper/`'s tectonic toolchain) → `paper/gaps.pdf`. Synthesizes
  GAPS.md/DISCUSSION.md/DESIGN_SUMMARY.md. Rebuild: `bash paper/build.sh`.
- **Tried other drawer settings (generalization).** Drawer tasks in the dev pool:
  `libero_goal{,_task,_swap}` task0 (middle drawer) and task3 ("open the top drawer
  and put the bowl inside", a multi-step pick-place — not attempted yet).
  - **`libero_goal_swap/task0` seed1** (perturbed layout; handle at x≈0.409 vs
    x≈0.70 in the base suite — cabinet repositioned): the **deterministic IK seat
    GENERALIZED** — `seat 0` landed on-bar on the FIRST try, grip 0.164, despite the
    new cabinet pose (`logs/swap0_seed1.log`). But the run **hit the episode horizon
    mid-pull** ("executing action in terminated episode") = **gap F** (the blocking
    controller's cumulative per-waypoint cost over the ratcheting pull exceeds the
    fixed `max_steps`=30000 horizon). Added `--max-steps` to `run_robust.py` to probe
    this; **at 80000 it SUCCEEDS** (`logs/swap0_seed1_bighorizon.log`): drawer fully
    open (qpos −0.159), disturbance only **21.9 mm** (wine bottle; bowl/cheese/plate
    all 0.0). So the failure at 30000 was **pure gap-F horizon exhaustion**, NOT a
    pull-axis problem — the skill *generalizes* to the repositioned cabinet; the
    blocking controller's cumulative per-waypoint cost over the ratcheting pull just
    doesn't fit a 30000-step horizon there (the deeper/further cabinet pose makes the
    pull more step-expensive). On a real FIXED-horizon eval this correct skill would
    fail purely on execution-step budget. Strong cross-suite evidence for gap F:
    cap-x needs a horizon-aware / non-blocking executor (fewer sim-steps per waypoint),
    or the ratcheting pull needs fewer re-seat cycles.
  - **`libero_goal_task/task0` seed1** (BDDL patch active → instruction is now
    `'open the bottom drawer of the cabinet'`, matching the scored goal; metadata-
    divergence WARNING logged as designed): the skill correctly detects the **bottom**
    handle @ [0.701,−0.137,**0.042**] and **safe-aborts** ("could not reach a clear
    pre-grasp") → SUCCESS=False, no grasp (`logs/task0_bottom_seed1.log`). Confirms the
    bottom drawer is **unreachable** (handle z=0.042 below the ~0.09 wrist floor) — the
    documented skip case. **New nuance:** the abort logged "zero disturbance" but the
    plate actually moved **36.3 mm** — the descend-to-pre fallback trajopt (collision
    cost 1.97e9) grazed the plate *before* giving up. So the "safe abort" path is itself
    not collision-verified on execution; reinforces "trust executed-sensing, not trajopt
    cost" and that even abort motions need a guarded/collision-checked executor.
- Per the standing rule: settings where graspnet/reach finds no viable grasp (awaiting
  AnyGrasp) are simply skipped (the bottom drawer is one such).
- **Replays:** previously we saved NO success videos (only text logs + static PNG snaps).
  Added `--video` to `run_robust.py` (env `enable_video_capture` + `get_recorded_frames`
  + imageio, same as `harness.py`). Replays land in `replays/`.

## Where things stand (updated 2026-06-25, session 2)
Target = `libero_goal/task0` (middle drawer; instruction==goal==middle, reachable).
Success = drawer qpos < **−0.14**. Solve path is sensing-only; the runner reads
qpos/disturbance for scoring.

### The big realization this session: the handle is a **D-bracket**, not a knob
From the asset (`wooden_cabinet_middle_handle.xml`) + the wrist cam: the graspable
part is a **horizontal bar along world X (~9 cm)**, held off the drawer face by two
posts, **center at z≈0.110**. Because the bar is **perpendicular to the pull (+Y)**,
a grip that is **vertically centered on the bar (TCP z≈0.10)** drags the drawer with
no slip. The earlier "shear" was NOT axial slip — it was seats landing **1–6 cm too
HIGH** (z≈0.14–0.17) that gripped above the bar / its top edge and held weakly.

### What now works — **cold-runner 5/5 (up from baseline 3/5)**
A redesigned `robust_skill.py` (sensing-only) with the **deterministic IK seat** +
tuned pull + trajopt retreat. Official `run_robust.py`, fresh process per seed
(`logs/rr8_seed*.log`): **5/5 SUCCESS, all qpos −0.160**, plate disturbance
17.7 / 10.7 / 8.9 / 25.4 / 4.2 mm (seeds 1–5). The IK seat lands on-bar on the first
try every time (no RNG), so the cold runner matches the warm harness.

Three things fixed this session beyond the IK seat:
- **Tuned pull** (`PULL_TRAVEL` 0.22→0.17, `PULL_STEP`→0.07): a deep centered grip
  barely slips, so stopping right at the drawer travel keeps the pull arc from
  sweeping further over the plate (≈32 mm → ≈5–18 mm; one seed still 25 mm).
- **Trajopt retreat** (was RRT): after the pull the arm sits at the open drawer where
  RRT reads the start as in-collision and **HANGS** ("start tree could not be
  initialized") — it blocked whole runs even though the drawer was already open.
  Trajopt accepts the start and lifts away. (gap H)

**Remaining = SAFETY tail: plate disturbance still spikes to ~25 mm on some seeds**
(stochastic pull trajopt + the seat brushing the plate ~5 mm on descent). Lower it by:
(1) keeping the elbow higher / a straighter pull, (2) a sensing open-detector to stop
the pull the instant the drawer is open, (3) lower `pos_weight` in the pull so the
plate-in-`obstacles` avoidance dominates. The success problem is solved; this is the
polish left.

Key fixes that made the wins possible (all in `robust_skill.py`):
1. **Gap approach** — descend into the CLEAR gap between the plate's near edge and the
   handle (pre = handle +Y 3.7 cm), instead of transiting at handle height *over* the
   flat plate (z<0.04, ~7 cm below) which makes the trajopt flag a **phantom collision**
   (cost ~1e10 though real disturbance is ~3 mm).
2. **z-centering** — seat target z = bar_z − 2 cm so the TCP lands centered on the bar.
3. **Robust descend-to-pre** — RRT-to-pre *silently fails* on some seeds (strands the
   arm at the high standoff → a too-high seat → shear). Retry RRT, fall back to a
   collision-aware trajopt, and **verify** the arm actually reached pre before seating.
   (This was THE fix that took the warm-harness run to 5/5.)
4. **Seat acceptance by RESULT** — reject fly-off (cost>1e8) and high-z/shallow seats
   (require TCP on the bar: y<−0.116, z<0.125); retry. Safe-abort if none found.
5. **Ratcheting over-pull** — pull +Y in collision-aware steps, re-closing each step;
   on grip-loss re-seat on the now-more-protruding bar. Over-pull (0.30 m EE travel)
   so a slipping smooth bar still reaches the −0.16 stop.

### SOLVED: the seat is now DETERMINISTIC (pyroki `solve_ik`, not the trajopt)
The HORL **trajopt** seat's landing z scattered **0.10–0.17 by RNG** (the old gap) —
the wrong tool for a 5 cm precision seat. **Fix: deterministic IK seat.** The pre→bar
corridor is sensing-verified clear, so `solve_ik(bar_pose, quat)` + a linear joint
interp lands the TCP **repeatably DEEP and CENTERED** (y≈−0.137, z≈0.11) — measured
0.109/0.110/0.110 over 3 reps vs the trajopt's 0.10–0.17. On the cold runner the IK
seat lands on-bar on the **first try** (seeds 1,2,3 confirmed: TCP=[·,−0.138,0.106] /
[·,−0.134,0.106] / [·,−0.143,0.106], grip ~0.168).

Caveat found: pyroki IK quality depends on a **good warm-start** — if called from a
high arm config (the RRT-to-pre-fail bug) it returns a z≈0.14 solution. So the seat
**re-descends + re-solves** if the IK lands high/shallow (`on_bar` check). With the
robust descend first, the re-solve rarely triggers.

Result with the IK seat: **cold-runner 5/5** (`logs/rr6_seed*.log`, all qpos −0.160).

### THE remaining work: cut the plate disturbance (success is solved)
The +Y pull at handle height sweeps the arm back across the flat plate → 3–33 mm
plate disturbance (seed-dependent: seeds 5/3 are clean at 2.9/6.1 mm; seeds 1/4 hit
32–33 mm). Success is no longer the problem; **safety is**. Next-step ideas, in order:
1. Inspect WHAT touches the plate (wrist+agentview snaps mid-pull) — is it the
   forearm/elbow sweeping low, or the open fingers on release? (use the interactive
   harness — it has `snap()`).
2. If it's the arm arc: bias the pull to keep the elbow up / pull along a higher path
   while the TCP stays on the handle, or shorten `PULL_TRAVEL` (a deep IK grip barely
   slips, so it may already reach −0.16 with less travel).
3. Put the plate explicitly in the pull's collision cloud (it's currently in
   `obstacles`, but the pull's `pos_weight` may override it — lower it during the pull).
4. Release + retreat earlier (stop pulling the instant the drawer is open — needs a
   sensing open-detector, e.g. re-detect the handle moved +Y ~0.15, since qpos is
   privileged).

## Files (all in `capx-se/open_drawer/`)
- `robust_skill.py` — **the deliverable** (sensing-only solve_robust). Gap approach →
  robust descend → result-gated z-centered seat → ratcheting over-pull → safe retreat.
- `interactive.py` + `ictl.sh` — **persistent multi-round harness** (NEW). Loads env +
  HORL planner ONCE (~2 min compile amortized), execs Python snippets from a file inbox
  in a persistent namespace, supports `reset_env(s)` (reset to initial / to seed s, per
  the "reset-on-collision then re-drive" rule). This is the missing cap-x "drive the sim
  a segment at a time, observe sensing+GT between moves" loop. Launch via a TRACKED
  background task (nohup children get killed when the wrapper exits — lesson learned):
  `MUJOCO_GL=egl HF_HUB_OFFLINE=1 .venv-libero/bin/python capx-se/open_drawer/interactive.py --seed 1 --io <iodir>`
  then `echo '<python>' | bash capx-se/open_drawer/ictl.sh <iodir>`.
  Helpers in the namespace: `state/reset_env/snap/detect_handles/obstacle_cloud/jts/ee/
  tcp/run/frame/quat_from_z`.
- `diagnose_corridors.py` — fast (no-planner) per-seed characterizer (frontal vs
  top-down corridor clearance + GT object positions). Showed corridors are clear for all
  seeds (the failures were seat variance, not blockage).
- `horl_planner.py` — HORL RRT+trajopt wrapped for cap-x (unchanged).
- `harness.py` / `skill.py` — the baseline primitive skill + its standalone runner.
- `run_robust.py` — official runner for `robust_skill` (measures qpos + disturbance).
- `GAPS.md` / `DISCUSSION.md` — cap-x gaps (now incl. the seat-z variance).
- `logs/` — `rr*_seed*.log` validation logs, `*_wrist.png` etc.

## How to run (next session)
ENV: `.venv-libero/bin/python`, always `MUJOCO_GL=egl HF_HUB_OFFLINE=1`. Services:
`bash scripts/start_capx_services.sh`. Then:
```
MUJOCO_GL=egl HF_HUB_OFFLINE=1 .venv-libero/bin/python capx-se/open_drawer/run_robust.py --suite libero_goal --task-id 0 --seed 3
```
For fast iteration use the interactive harness (planner stays warm across seeds via
`reset_env(s)`), NOT one run_robust per change (each pays the ~2 min compile).

## Task facts (verified)
- Handles (base z): bottom 0.042 / middle 0.110 / top 0.183; x≈0.70, y≈−0.137 (bar
  FRONT detected ~−0.119; bar center ~−0.137). Drawer slides +Y, qpos 0→−0.16,
  damping 50, success qpos<−0.14.
- Middle handle = D-bracket: bar along X, size ~0.015×0.016×0.089, at local
  y=−0.102 z=0.110, + two posts at y=−0.083.
- The **plate** lies flat in front (x[0.50,0.79], y[−0.05,0.16], z[0.005,0.04]) — the
  only real obstacle; it sits ~7 cm BELOW the handle corridor.
- TCP cannot be planned past y≈−0.119 (a kinematic wall at the bar front) for this
  −Y orientation, regardless of target depth — but a grip at the bar front still drags
  the drawer IF vertically centered.

## cap-x gaps found this session (full list in GAPS.md / DISCUSSION.md)
1. **Stochastic trajopt seat is not repeatable** (z scatters 0.10–0.17 by RNG; also
   returns NaN / phantom 1e10 collision cost over a 7-cm-distant flat object). Needs a
   deterministic short-range seat primitive (direct IK + linear interp) for precision
   grasping — the planner is fine for transit, wrong for the final seat.
2. **RRT-to-a-near-structure goal silently fails** ("start tree could not be
   initialized" / not-planned) with no signal to the caller — must be retried/verified.
3. **No standalone multi-round sim-driving loop** in cap-x — built `interactive.py`.
4. (carried) instruction/goal contract, dead curobo path, contact feedback — see GAPS.md.
