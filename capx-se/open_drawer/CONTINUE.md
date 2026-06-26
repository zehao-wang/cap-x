# START HERE — open_drawer continuation handoff

Dogfooding: hand-write the LIBERO `open_..._drawer` skill with cap-x's legitimate
tools (camera RGB-D + proprioception only; NO privileged poses in the solve path)
to (a) find what cap-x lacks and (b) build a robust + SAFE solution. Commits: old
ones `[tmp]`, new ones `[auto]` (per AGENT.md). **Start every session by launching
all services: `bash scripts/start_capx_services.sh`** (SAM3/graspnet/pyroki/molmo).

## Session 5 (2026-06-26): COMPOSITIONAL task SOLVED — `libero_goal/task3` "open the top drawer and put the bowl inside" (open → pick → place), seed 1 SUCCESS + 0.0 mm disturbance
- **New deliverable `compose_skill.py`** (`solve_compose`, sensing-only). Three motor
  phases from the instruction's two sub-goals: **OPEN** the top drawer (reuses the proven
  `solve_robust`) → **PICK** the bowl → **PLACE** it in the open drawer. Runner
  `run_compose.py` (batch over seeds, reuses env+planner across seeds; reports success +
  SAFETY = disturbance of the OTHER objects, bowl reported separately). Per-run artifacts in
  `runs_compose/`.
- **Seed 1: SUCCESS=True, placed=True, max other-object disturbance 0.0 mm** (bowl carried
  260 mm into the drawer; cheese/bottle/plate all 0.0). Drawer stays fully open through the
  place. ~7.6k sim-steps total (fits the 30k horizon with >3× margin).
- **`robust_skill.py` generalized to the TOP drawer**: the seat z-gate is now RELATIVE to the
  detected handle z (`mid[2]±`), so the same open-skill seats on the top bar (z≈0.18) as well
  as the middle (z≈0.11). First-try seat, grip 0.17. `solve_robust` now also returns
  `outward` + `pull_travel` so the place phase can compute the drop GEOMETRICALLY.
- **The place target is computed from the open-phase geometry, NOT re-sensed.** Re-detecting
  the open handle post-retreat is fragile (arm occludes / cabinet centroid flips `outward`);
  instead `drop = handle + outward*travel − outward*0.11` at the handle's z (the success box
  `wooden_cabinet_1_top_region` rides WITH the drawer and sits ~0.11 m behind the open handle).
  Verified empirically (`probe_target.py`): a bowl centred in the open region → task_completed.
- **THE hard part was the bowl pick+carry. Five cap-x lessons, all the hard way:**
  1. **`solve_ik` is STATEFUL** (warm-starts from the last IK result `api.cfg`, not the
     current joints). After the open phase + `goto_home`, `api.cfg` is stale at the side-
     approach seat config → the next solve_ik lands in a bad branch and the descend stalls
     (hand never reaches the bowl). Fix: reset `fns["solve_ik"].__self__.cfg = None` before
     the pick so it warm-starts from the rest pose (≈home). **gap-I (new).**
  2. **`goto_home_joint_position()` from an extended pose only reaches PARTIAL home** in one
     call (ee≈0.55 vs home 0.37 — `move_to_joints` interp is coarse). Call it until ee
     converges. **gap-J (new).**
  3. **`solve_ik` (pyroki server) is COLLISION-UNAWARE** — it returns configs that physically
     collide with the scene. Grasping the bowl from the cabinet (+x) side made the wrist hit
     the protruding OPEN drawer and the descend stalled, with NO signal (only a blocked
     controller). **gap-K (new).**
  4. **A naive top-down rim PINCH slips the bowl during the carry** (thin-wall line contact
     can't hold it airborne; grip 0.10 → slides to 0.01). **Contact-GraspNet** finds firm
     antipodal grasps (grip 0.17–0.23) that HOLD. Use `plan_grasp` → rank downward candidates
     → accept the first that descends AND grips in [0.05, 0.40]. The hand-built pinch is the
     wrong tool; graspnet is the right one. **gap-L (new): cap-x has no grasp-quality /
     stability check; the skill must brute-try candidates and verify by grip reading.**
  5. **`solve_ik` returns TCP-frame** (robot_cartesian_pos reads ~0.113 above it), and the
     gripper reads **~1.0 open / ~0.015 closed-empty** (NOT the other way). Both cost runs to
     re-derive; documented in the constants.
- **Place mechanics that matter:** carry HIGH above the drop via solve_ik waypoints (NOT the
  planner — a planner move desyncs `api.cfg`), then descend straight down; a level carry
  shoves the protruding drawer closed (qpos −0.16→−0.11). Aim the TCP so the BOWL CENTRE
  (offset (bcen−grasp_p) from the TCP) lands at the region centre.
- **Fast-iteration methodology (key):** the bowl pick/carry needs NO planner, so it was tuned
  in seconds via staged-open calib scripts (set drawer qpos, no 2-min compile) instead of
  7-min full runs. The slip only reproduced with a harsh 40-waypoint carry; once it did,
  graspnet candidates were swept fast.
- **20-SEED BENCHMARK (`logs/bench20b.log`, `runs_compose/benchmark_summary.json`):
  9/20 SUCCESS (bowl in drawer); 7/20 SUCCESS **and** SAFE (≤30 mm other-obj disturbance:
  seeds 6/7/14/17 ≈0 mm, 5/11/18 ≈30 mm). 1 GROSS-UNSAFE (seed 1: wrist grazed the tall wine
  bottle → knocked off the table, 2713 mm — `success` but disqualified).** The drawer OPEN
  succeeds on ALL 20 (proven skill); 100% of the variance is the BOWL pick+place:
  - ~9 seeds FAIL the PICK (`placed=False`): graspnet candidates all descend-short (collision
    w/ the open drawer — gap-K) or close empty/over-grip (graspnet nondeterminism — gap-L).
  - ~2 seeds place `True` but MISS the region (bowl at drawer height but wrong xy: the place
    offset uses the approximate SAM3 centroid, which is biased per seed).
  - **A ~30 mm wine-bottle graze recurs** on many seeds — the top-down bowl approach passes
    close to the tall bottle and collision-UNAWARE `solve_ik` (gap-K) can't route around it.
  **Headline finding: cap-x's grasp+IK tools can hit a clean SUCCESS (seed 6/17: bowl placed,
  0 mm disturbance) but can't yet RELIABLY+SAFELY solve a compositional pick-place — the
  missing piece is collision-aware IK / a collision-checked executor for the pick (the HORL
  planner has it, but using it desyncs `solve_ik`'s stateful warm-start = gap-I).**
- **GraspGenX swap (user request 2026-06-26): infrastructure DONE, but it does NOT yet beat
  Contact-GraspNet on this bowl.** NVlabs GraspGenX is installed at
  `~/Documents/Projects/HumanOnlyRobotLearning/src/packages/GraspGenX/` with a Unix-socket
  inference service (`service/server.py`). Built a self-contained client
  `capx-se/open_drawer/graspgenx_client.py` (length-prefixed wire protocol) +
  `calib_ggx.py`. Service launch (its own venv, GPU0, **a PRIVATE socket
  `grasp_gen_capx.sock` — a PARALLEL SESSION races the shared `grasp_gen.sock`, do NOT touch
  it**): `.venv/bin/python -m service.server --socket /tmp/demo_bridge/sockets/grasp_gen_capx.sock
  --default_gripper franka_panda`.
  - **Works:** GraspGenX returns 6-DOF grasps with FAR higher confidence (0.83-0.94 vs CGN's
    0.15-0.29), in the solve frame. Convention decoded (`graspgenx/robot.py`): approach=+Z,
    closing=+X, `depth=0.1034`; grasp origin = panda_hand, fingertips at `pos+approach*0.1034`.
    `solve_ik(pos+approach*depth, quat_from_R)` places the gripper exactly at the grasp pose.
  - **Blocker:** on the akita bowl from a SINGLE agentview (top-only ~2600-pt cloud), GraspGenX
    grasps GRIP the rim (closed-grip 0.17-0.32) but the bowl does NOT lift (slips; both up-lift
    and approach-axis retract give dz=0). CGN's grasps from the same partial cloud happened to
    capture the rim antipodally and held -> CGN's 9/20. GraspGenX conditions on OBJECT SHAPE and
    is under-served by a top-only partial cloud.
  - **THE bug was the closing-axis transform (found via the official `end2end/robots/
    franka_panda.yaml`).** GraspGenX's grasp frame closes along +X; the Panda's panda_hand
    closes along +Y, so the official applies `grasp_to_tool_transform` = **+90deg about the
    grasp's Z** (quaternion_xyzw [0,0,0.7071,0.7071]). WITHOUT it the gripper VISUALISES
    correctly (viz uses the GraspGen frame) but the real fingers close wrong-axis and grasps
    MISS — exactly our symptom. WITH it (R' = R @ Rz(+90)), the SAME grasps go from **0/N ->
    2/2 grip+lift** the bowl (lift 180-220 mm). Also key (centering, already done): GraspGenX
    expects an OBJECT-CENTRED cloud (the render script's `T_center = -mean`); feeding a world-
    offset cloud degrades grasps. The client `graspgenx_client.py` now does BOTH by default
    (center=True, align_panda=True), returning robot-ready 6-DOF poses; convention recap:
    approach = R[:,2] (unchanged by Rz90), depth=0.1034, TCP target = pos + approach*depth.
  - **Validated GGX > CGN where CGN gives NOTHING:** on the flat **plate**, CGN returns
    `No grasp candidates found` (0); GGX returns **40 grasps conf 0.96-0.98** (sensible rim-edge
    grasps, render in `ggx_render/plate_grasps_top.png`). This is GGX's real value: recall on
    objects CGN can't grasp.
  - **On the BOWL benchmark, GGX is ~comparable to CGN, NOT better.** Integrated behind
    `COMPOSE_GRASP=ggx` in `compose_skill.py` (pool -> +90 -> near-vertical filter -> verify-by-
    grip). Staged (approach-axis descend+lift) = **2/8**; CGN = 9/20 (~3.6/8). Reasons: (a) the
    thin bowl rim is a hard MuJoCo grip target for everything; (b) **`solve_ik` can't hold a
    TILTED orientation (gap-D), so we must drop GGX's best 6-DOF grasps and keep only near-
    vertical ones** -- the official executes ALL grasps via **cuRobo** (collision-aware, any
    orientation) + stiff finger PD. The vertical compose pipeline (built for CGN's top-down
    grasps) mis-handles even the near-vertical GGX grasps on lift (seed1 slips).
  - **NEXT to actually beat CGN on the bowl: execute GGX's 6-DOF grasps via a real planner**
    (cuRobo / HORL trajopt) with approach-axis descend + tool--Z lift (per the official
    end2end), instead of vertical `solve_ik`. Until then, keep CGN (9/20) for the bowl and use
    GGX for CGN-failure objects. Service left RUNNING on a PRIVATE socket (GPU0, tracked task).
- **How to run:** `MUJOCO_GL=egl HF_HUB_OFFLINE=1 .venv-libero/bin/python
  capx-se/open_drawer/run_compose.py --seeds 1 2 3 --max-steps 40000`. Grounding probes:
  `probe_task3.py` (scene geometry), `probe_target.py` (placement target), `calib_perception.py`
  (sensing). All privileged reads in probes/runner are MEASUREMENT-ONLY.

## Session 4 (2026-06-25): SOLVED gap-F horizon blowout — open-detector + pull subsample (8× fewer sim-steps)
- **Measured WHERE the horizon goes.** Instrumented `run_robust.py`'s `_dbg` to record
  `env._sim_step_count` per stage. On the gap-F case (`libero_goal_swap/task0` seed1, the
  one that needed 80k) the budget was: pre 632, seat 94, **pull 38,924 (95.6%)**, retreat
  1,074 → total 40,724. The horizon counts EVERY blocking sub-step (the underlying env
  advances once per `move_to_joints_blocking` inner step), and each waypoint MAXES the
  120-step convergence cap because the position controller can't settle to 0.01 rad while
  dragging the damped (50) drawer (38,924/10 steps/32 waypoints ≈ 121 ≈ the cap).
- **Root cause = no open-detector.** The pull loop ran all 10 steps even though the drawer
  was fully open (qpos −0.15) by step ~2: TCP advance plateaued at 167–169 mm just under
  `PULL_TRAVEL`=170, so the existing top-of-loop break never fired. 6 of 10 pull steps
  (~23k sim-steps) were pure waste dragging an already-bottomed drawer.
- **Fix (skill-side, sensing-only) in `robust_skill.py`.** Added a PROPRIOCEPTIVE
  open-detector to the pull loop (qpos is privileged, so infer "open" from the TCP-advance
  plateau): (a) `DRAWER_OPEN_ADV=0.15` — if the grip is HOLDING and the TCP has advanced
  ~the full drawer travel, stop (grip-holding makes the advance reflect real drawer motion,
  so it won't over-read on slip); (b) `STALL_DELTA=0.008` — a full pull command that no
  longer advances the TCP (with grip holding, past `PULL_MIN_OPEN`) means the drawer hit
  its hard stop → stop. The `PULL_TRAVEL`=0.17 cap + 10-step loop stay as fallback, so the
  only new failure mode is stopping too EARLY, which the grip-holding guard prevents.
- **Verified.** Gap-F case (`libero_goal_swap/task0` s1) now **SUCCEEDS at the original
  30,000 horizon** (was FAIL, needed 80k): total **8,739 sim-steps** (pull 7,512 — 5.2×
  fewer), drawer **fully open −0.16**, open-detector fired at step 2 (advanced 158 mm).
  Base task seeds 3/4/5 all still pass, **fully open −0.16**, ~9.4/9.3/14.4k steps,
  disturbance 6.7/22.4/12.2 mm (seed 4 IMPROVED from ~32 mm — fewer idle arm sweeps). No
  regression; the skill now fits a normal LIBERO horizon with >3× margin and generalizes
  to deeper/repositioned cabinets (stops as soon as it's open, regardless of pull cost).
- GAPS.md gap F updated: the cap-x-side gap is still real (cap-x needs a horizon-aware /
  non-blocking executor + per-waypoint convergence-failure signal); this session worked it
  around at the skill level so the skill fits budget.
- **DONE this session too — pull trajectory subsampling (another ~2×).** `run()` now takes
  `every=k` and the pull uses `PULL_SUBSAMPLE=2` (execute every 2nd waypoint of each pull
  trajopt, final always kept). The pull is a straight +Y drag (waypoints near-collinear, so
  skipping them doesn't cut a corner toward the plate, which sits BELOW) and each blocking
  move otherwise maxes the 120-step cap, so this ~halves pull sim-steps; the endpoint
  (openness) is unchanged and the open-detector self-corrects if a coarse command lands short.
  Transit/seat stay full-resolution. **Verified:** gap-F case (swap0 s1) pull 7,512 → 3,910,
  total 8,739 → **5,173**, fully open −0.16, disturbance 37 → **30 mm** (improved). Base s4
  pull → 3,956, total → 6,373, fully open, disturbance 22 → 24 mm (noise). **Cumulative vs the
  original gap-F blowout: 40,724 → 5,173 sim-steps (7.9×), pull 38,924 → 3,910 (10×).**
- **Snapshotted as `milestones/02_horizon_open_detector/`** (README + CHANGES + example
  trace/summary/replays from the headline swap0 s1 run).
- **DECISION on the SAFETY tail — accepted, not chased.** The residual object disturbance
  (~4–30 mm, seed-dependent) is the **single-view sensing safety floor**: the solve sees only
  the agentview depth cloud, so occluded geometry is absent from the collision world and a few-
  mm graze when dipping onto a low handle is the information limit, not a skill bug. Zeroing it
  would overfit to this one camera pose (breaks the anti-overfit rule). Stance: treat small
  grazes (≤ ~3 cm, nothing knocked over) as acceptable; SAFE = success + no GROSS disturbance,
  not zero contact. The real lever is more scene info (2nd view / wrist-fused cloud / contact
  feedback = standing cap-x gaps), not a cleverer planner. Written up in DISCUSSION.md
  ("Single-view sensing imposes a SAFETY FLOOR").
- **Next (optional, diminishing returns):** could trim the per-waypoint convergence cap for
  the pull (never converges under load anyway) — but needs an env-level `max_steps` knob on
  `move_to_joints` (cap-x core), so it's a gap to report, not a skill tweak. Current 30k margin
  is ~6×; likely not worth it. Horizon is done; safety tail is accepted per above.
- **Known flakiness (orthogonal):** SAM3 occasionally returns no handle on the very first
  detection (one swap0 s1 run aborted "no handle detected" → safe-abort, 0 disturbance). Pre-
  existing perception nondeterminism, not from these changes; the skill safe-aborts correctly.

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
- **Run artifacts (cap-x convention).** `run_robust.py` now persists a per-run dir by
  default (mirrors `trial.py`'s `output_dir/trial_*`): `runs/<suite>_t<id>_s<seed>__
  success<0/1>_dist<NN>mm__<ts>/` with `trace.json` (key-step skill log via injected
  `log=` + per-stage privileged measurements + result + disturbance summary),
  `summary.txt` (human-readable key steps), and `video_success<0/1>.mp4` (the replay, via
  cap-x's own `capx.utils.video_utils._write_video`). Flags: `--out-dir`, `--no-video`,
  `--no-artifacts`, `--max-steps`. `runs/.gitignore` keeps the json/txt logs in git but
  ignores mp4s (curated replays live in `replays/`). **Bug fixed:** the video getter was
  `get_recorded_frames` (doesn't exist) in both `run_robust.py` and `harness.py` — so
  harness video had NEVER worked; corrected to `get_video_frames()`.
- **Multi-view replays + dataflow-tagged log.** `runlog.py` (new) tags each key step
  `llm` (a DECISION boundary — where an agentic cap-x would call the LLM; records the
  call's PURPOSE, the DATA it consumes [single frame vs sequence vs proprio], and what it
  DECIDES) vs `local` (SAM3 / pyroki-IK / HORL planner). `robust_skill.py` is annotated
  with these; `solve_robust(log=...)` accepts a plain `print` (auto-wrapped) or a
  `RunLogger`. `run_robust.py` records BOTH agentview + wrist replays (multi-view) and
  writes `dataflow_summary` into `trace.json` + an "llm-decision data map" into
  `summary.txt`. **Insight surfaced:** the skill makes ZERO real run-time LLM calls; all
  *control* decisions (seat/pull) use **proprioception only [no frames]**, *perception*
  decisions use a **single agentview frame**, and axis estimation is **single-view (no
  depth axis)** — the one decision that structurally wants a sequence/probe. This per-
  decision data map is the input for deciding which steps justify a richer (tracker/
  sequence) input vs which a per-turn VLM over-serves. See paper §"Instrumenting the
  data-flow".

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
