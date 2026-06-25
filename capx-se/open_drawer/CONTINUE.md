# START HERE — open_drawer continuation handoff

Dogfooding: hand-write the LIBERO `open_..._drawer` skill with cap-x's legitimate
tools (camera RGB-D + proprioception only; NO privileged poses in the solve path)
to (a) find what cap-x lacks and (b) build a robust + SAFE solution. Commits: old
ones `[tmp]`, new ones `[auto]` (per AGENT.md). **Start every session by launching
all services: `bash scripts/start_capx_services.sh`** (SAM3/graspnet/pyroki/molmo).

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

### What now works — **cold-runner 4/5 (up from baseline 3/5), all SAFE**
A redesigned `robust_skill.py` (sensing-only). Official `run_robust.py`, one fresh
process per seed (`logs/rr4_seed*.log`):
- **seed 2: SUCCESS −0.160** (disturb 26 mm — over-pull arc grazes the plate)
- **seed 3: SUCCESS −0.160** (disturb 4.2 mm, clean)
- **seed 4: SUCCESS −0.160** (disturb 17.9 mm)
- **seed 5: SUCCESS −0.160** (disturb 13.6 mm) — the seed the old handoff called
  "safe-unsolvable" now opens.
- **seed 1: FAIL, disturb 5.1 mm — SAFE** (no good low-z seat this cold run →
  safe-abort, zero plow), the RNG-unlucky one (below).
- Warm-harness reproduces **5/5** (≤6.6 mm). The 4/5-vs-5/5 delta is purely the seat
  RNG, and the per-seed count varies run-to-run — but it is **always SAFE** (rejects
  bad seats, never plows). Disturbance is up vs the old 3-mm because of the over-pull
  (a smooth bar slips; the EE must over-travel) grazing the plate — a success/safety
  knob (`PULL_TRAVEL`) to revisit once the seat is deterministic.

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

Result with the IK seat: **harness 3/3 (seeds 1,2,4) all SUCCESS** (qpos −0.160),
disturb 4–23 mm; cold-runner seed 1 ✓ (was failing), seed 3 ✓. Full 5-seed cold
re-validation (with the ratchet IK re-seat fix) in `logs/rr6_seed*.log`.

**Remaining knob: pull disturbance.** A deep IK grip barely slips, so `PULL_TRAVEL`
is back to 0.22, but the +Y pull arc still grazes the plate (seed 1 ~32 mm; seeds 2–4
4–9 mm). Lower it further by pulling with the elbow held higher / a shorter travel now
that the grip is reliable — the main remaining tuning.

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
