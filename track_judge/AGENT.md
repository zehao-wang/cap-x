# AGENT.md — playbook for the **track-judge** experiment (AUTONOMOUS, cap-x-se update)

You extend **cap-x-se** with a new way for the agent to judge task state, and you prove it
out **one generation at a time, fully autonomously**. This file is the only briefing you
need each cycle. It supersedes the *human-feedback* workflow of the repo-root `AGENT.md`
for this experiment; it **reuses** that file's code-style rules (clear structure, don't
over-guard, ≤500 lines/file, no `Co-Authored-By`/Claude trailer on commits).

A **generation** = one focused change under `track_judge/algo/` + one libero-pro eval run +
one commit.

**The paradigm: a "self-evaluate" agent.** The new agent self-**plans** (writes execution
code) AND self-**evaluates** (writes its own state-judgment code) — **both are code, neither
calls an LLM** at judgment time. The judgment code may import **any packages** (numpy, scipy,
the TAPIP3D tracker, geometry libs, …); it is not restricted to a fixed library. This is the
whole point: replace the per-turn VLM judgment (the VDM) with code the agent writes once for
the task. Tracking (TAPIP3D 3D trajectories) is the primary signal that code uses.

### Two levels — keep them separate
1. **RUNTIME (the agent under test).** Runs in **cap-x's standard environment** (`.venv` /
   `.venv-libero`), executing libero-pro exactly as cap-x normally does. The **only** LLM
   call is the agent **writing** the plan + judge code (single Gemini, cap-agent0). After
   that, execution and judgment are **pure code — no LLM, no human**. This is what
   `benchmark.py` measures. Nothing in this file changes how runtime behaves except the
   `state_judge` switch and the design files the agent's code-writing is conditioned on.
2. **SELF-EVOLVE (this playbook — an EXTERNAL meta-loop, exactly like motion_plan/self_evolve).**
   Separate, fresh Claude Code sessions that, **based on the experiment results**, edit the
   agent's *design* (`track_judge/algo/`: the judge-codegen prompt, `judge_lib`, the harness)
   from outside, re-benchmark, accept/reject, and commit. The self-evolve loop is **not part
   of the runtime agent** and never runs during a robot rollout — it has the full runnable
   cap-x system as its evaluation environment. You, reading this, are the self-evolve level.

## ▶ AUTONOMOUS MODE — pre-authorized; never ask permission
Run the loop end-to-end. You decide the hypothesis, edit `track_judge/algo/`, run the eval,
judge accept/reject by the rules below, update `track_judge/results/BEST`, and `git commit`
**every** generation (accepted *or* rejected, so every step is recoverable). Then start the
next. Do **NOT** ask "should I proceed / which direction / is this OK" — act, then report
what you did and why. Stop early only on a genuine blocker (env or eval harness broken).
- **Loop budget:** the number of generations the launch prompt asks for (default ~6), or
  until ~3 consecutive rejects with no fresh idea — then write a short summary and stop.
- **Never `git push`.** **No human feedback anywhere in this experiment.**
- One hypothesis per generation; change the smallest thing that tests it.
- You MAY decompose a generation's coding work and **spawn sub-agents in parallel** for
  independent pieces (see §10). One generation still = one hypothesis + one commit.

---

## 0. THESIS (what we are trying to show)
The VDM is a **redundant LLM call.** In today's cap-x-se the per-turn **Visual Differencing
Model** is a *second* VLM invocation whose errors are **correlated** with the policy LLM
(same modality/family) — it largely hands the agent back its own opinion, at real latency and
cost. We claim it can be **removed entirely** and replaced with **agent-written geometric
state-judgment code driven by 3D point tracking (TAPIP3D)** — a cheaper, *decorrelated*,
physically grounded signal — and that doing so makes the single-LLM cap-agent0 agent
**(1) succeed more and (2) run faster** on libero-pro.

Two headline metrics, reported together, every generation:
- **task success** — libero-pro ground-truth `check_success()` success rate (higher better).
- **speed** — median wall-clock per task, end-to-end (lower better). Removing a per-turn VLM
  call is a direct speed lever; early-abort on geometric failure is another.

**Win condition for the experiment:** the tracking arm (B), with the VDM **fully removed**,
beats the VDM baseline (A) on **both** — success rate ≥ A (within tolerance, ideally strictly
higher) **and** median task time < A. A generation that trades one for the other is a partial
result; record it honestly, don't claim the thesis.

### 0.1 Why a 3D trajectory beats a frame-diff VLM (capability targets for `judge_lib`)
The judge sees **continuous, metric 3D trajectories** of chosen masks, not a discrete
semantic snapshot. `judge_lib.py` should expose primitives that exploit this — these are the
source of the win, and the things the VDM structurally cannot do:
1. **Collision / near-miss** — min inter-object distance over time (object vs arm / other
   objects) in metric cm; flag contact and risky clearance.
2. **Grasp instability / slip** — object points tracked **relative to gripper points**; if
   they stop co-moving (relative displacement or in-hand rotation grows) the grasp slipped —
   detected mid-motion, not at the end.
3. **Early failure → early abort** — a continuous signal detects "dropped/failed at frame k"
   *during* execution → stop and retry immediately instead of running the whole trajectory
   then querying a VLM. Primary source of BOTH speed and success.
4. **Drop / fall** — descent past support height + free-fall acceleration signature.
5. **3D containment / placement** — object points entering the target container's 3D *volume*
   and persisting across the last K frames (robust to viewpoint/occlusion).
6. **Occlusion robustness** — TAPIP3D predicts occluded points' positions, so state is
   judged even when the object is hidden behind the gripper/another object.
7. **Appearance/viewpoint invariance** — geometry ignores lighting, texture, distractors,
   prompt phrasing (all of which perturb a VLM diff).
8. **Multi-object relations** — track several masks → exact on-top-of / inside / aligned /
   stacking-order tests.
9. **Determinism + debuggability** — deterministic, inspectable rules (with saved
   tracking-viz) vs a stochastic, opaque VLM call.
10. **Dense + cheap + graded** — local, so it can run every turn (or sub-step); the
    continuous distance-to-goal can serve as a **graded progress** signal, not just done/not.
11. **"Right motion, wrong outcome"** — gripper moved correctly but the object didn't follow
    (missed grasp) is immediately visible in the relative trajectory.

Design implication: do NOT reduce the tracking judge to a discrete "done?" boolean that just
imitates the VDM — that throws away the advantage. Prefer continuous, relational, early-abort
signals.

---

## 1. The two arms (what is actually being compared)
In libero-pro, **success is always ground-truth** (`FrankaLiberoEnv.task_completed()` →
`check_success()`). The thing we swap is the **per-turn state feedback the agent uses to
self-correct** (the REGENERATE/FINISH signal in `trial.py:_handle_multi_turn_step`):

- **Arm A — VDM baseline (frozen reference).** The stock cap-x-se path:
  `_get_visual_differencing_feedback` / `_get_video_differencing_feedback` send before/after
  (or the turn video) to a VLM (`google/gemini-3.1-pro-preview`) and return free-text. We do
  NOT modify this; it is the number to beat.
- **Arm B — self-evaluate (the new thing).** For the same initial static frame the agent
  emits **two** code artifacts: the usual **execution code** (self-plan) AND a
  **state-judgment function** (self-evaluate). The judgment function is **pure code — it
  calls NO LLM** and **may import any packages**; it runs on the turn's **dense RGB-D video**,
  using TAPIP3D 3D trajectories as its primary signal, and decides progress / done / failure
  **geometrically** (e.g. object dropped = tracked point set falls + leaves support; object
  placed = its tracked points enter the target container's 3D region). It chooses which
  **masks** to track itself. **No VLM anywhere in arm B's judgment** — that is exactly the
  redundant call we remove.

The arm is a single config switch: `state_judge: vdm | tracking` (§3). Everything else
(the cap-agent0 loop, the LLM, the suites, the seeds) is identical between arms, so the A/B
is clean.

---

## 2. Repo layout — what is evolvable vs frozen
Everything for this experiment lives under `track_judge/` (self-contained, like
`motion_plan/self_evolve/`).

```
track_judge/
  AGENT.md            ← this playbook
  REPORT.md           ← the living experiment report (paper-oriented; maintained on ACCEPT)
  EVOLUTION.md        ← human-readable changelog (one entry per gen; mirror of CHANGELOG)
  algo/               ← THE ONLY CODE THE LOOP MAY MUTATE
    __init__.py
    judge_prompt.py   ← the prompt fragment that tells cap-agent0 how to WRITE judge code
    judge_lib.py      ← CONVENIENCE geometric primitives (in_region_3d, dropped, ...) — the
                         self-evaluate judge MAY use these but may also import ANY package; not a cap
    track_judge.py    ← the harness: dense RGBD video → TAPIP3D → run agent's judge fn → verdict
    params.py         ← DEFAULTS + append-only CHANGELOG (one line/gen)
  tapip3d_client.py   ← vendored copy of Tapip3DClient + wire (pure socket+numpy, no deps)
  benchmark.py        ← runs libero-pro for ONE arm, writes results/genNNN/, auto-diffs vs BEST
  compare.py          ← accept/reject verdict engine (success + speed, tolerances, hard gate)
  run_evolve.sh       ← driver: one fresh headless cap-agent0 session per generation
  results/
    BEST              ← one line: the current champion gen for arm B (e.g. "gen7")
    baseline_vdm/     ← arm A reference numbers (frozen; re-measured only when suites change)
    genNNN/           ← per-gen summary.json + report.md + tracking-viz/
```

**FROZEN (never edit to make a generation "win"):** the cap-agent0 loop (`capx/envs/trial.py`,
`runner.py`, `launch.py`), the libero-pro suites (`capx/third_party/LIBERO-PRO/`), the VDM
(arm A), `benchmark.py`/`compare.py`, the success metric. The single integration seam you
add into `trial.py` (the `state_judge` switch + dense-RGBD recording) is **infrastructure**,
built once in Phase 0 and then frozen — it is not where generations happen.

---

## 3. cap-agent0 single-LLM Gemini config + libero-pro
- **Single LLM, no ensemble.** Use a config with `use_parallel_ensemble: false` and
  `use_multimodel: false` (the qwen variant does this) and `--model
  google/gemini-3.1-pro-preview`. ⚠️ `env_configs/libero/franka_libero_cap_agent0.yaml` is
  currently a *multi-LLM* ensemble — do not use it as-is; make a
  `franka_libero_track_judge.yaml` with both flags false + Gemini.
- **Gemini is already the default model** (`launch.py:LaunchArgs.model`,
  `visual_differencing_model`), served via the proxy at `server_url`
  (`http://127.0.0.1:8110/chat/completions`). Make sure that proxy is up before a run.
- **Suites:** libero-pro only, via `capx/envs/scripts/run_libero_batch.py`. Default 6:
  `libero_object_swap, libero_object_task, libero_goal_swap, libero_goal_task,
  libero_spatial_swap, libero_spatial_task`. Pick a **dev subset** to iterate on (cheap,
  discriminating) and hold the rest out (§7). Success + speed are aggregated by
  `launch_utils._print_and_save_summary` → `summaries.txt`.
- **The arm switch** is a new config key `state_judge: vdm | tracking` threaded into the
  trial config; `tracking` routes `_handle_multi_turn_step`'s feedback through
  `track_judge.algo.track_judge` instead of the VDM functions.

---

## 4. Frame recording change (dense RGB-D video for the tracker)
TAPIP3D needs a **dense** RGB-D video + camera intrinsics + extrinsics. Today
`libero.py:_record_frame()` grabs `agentview` RGB only, `depth=False`, subsampled ×4. Phase-0
infra change (then frozen):
1. **Dense:** add a capture mode that records **every** intermediate motion frame
   (`_subsample_rate = 1`, or drop the modulo gate at `libero.py:~265` and `~300`). Keep the
   ×4 default for arm A so the baseline is unchanged.
2. **Depth + camera params:** render depth too (`sim.render(..., depth=True)`) and expose the
   MuJoCo `agentview` intrinsics (from fovy + W/H) and extrinsics (camera pose). LIBERO is
   simulated, so depth/intrinsics/extrinsics are **exact** — no ZED/stereo service needed.
3. Frames still flow through `get_video_frames_range` per turn; the judge harness consumes
   the per-turn dense RGB-D + camera params.

Gate the dense+depth path behind `state_judge == tracking` so arm A's cost/behaviour is
untouched.

---

## 5. TAPIP3D integration + tracking visualizations
- **Reuse the socket service** built at
  `…/HumanOnlyRobotLearning/src/packages/TAPIP3D/service/` (server in the `tapip3d` conda
  env; `Tapip3DClient` is pure socket+numpy). **Vendor** a thin copy of `client.py`+`wire.py`
  into `track_judge/tapip3d_client.py` so cap-x has no cross-repo import; it talks to the
  running tapip3d server over the unix socket. The harness ensures the server is up (ping;
  optionally auto-start) before a run.
- **Masks are chosen by the agent's judge code.** Provide it primitives in `judge_lib.py`
  (e.g. segment-by-text via the existing SAM API, or MuJoCo segmentation masks) → mask →
  query points → `client.track(window, masks…)` → 3D tracks in the **world/table frame**
  (sim extrinsics) → geometric tests.
- **Always save tracking visualizations for debug** — REQUIRED, both full-image and the
  per-object/per-mask tracks the judge used — under
  `track_judge/results/genNNN/tracking-viz/<suite>/<task>/<trial>/` (overlay video/images of
  the tracked points + the geometric regions the verdict used). A generation that produces no
  viz is incomplete.
- Recall the tracker's limits (it is NOT real-time): window must be ≥12; ~1–2.5 fps;
  `filter_depth` off; coords are `[T,N,3]`. Judge code runs once per turn on the turn video.

---

## 6. Procedure — each generation
1. **Inspect:** `git HEAD`, `results/BEST`, the latest `results/genNNN/summary.json`, and
   `results/baseline_vdm/`. Identify the binding metric (success gap, or a slow judgment) and
   the suites/tasks driving it. Read the saved `tracking-viz/` of recent failures — the
   geometry usually shows you the bug.
2. **Hypothesise:** ONE hypothesis. Edit ONLY `track_judge/algo/` (a judge primitive, the
   mask-selection strategy, the judge-codegen prompt, or a TAPIP3D param). Append one
   `CHANGELOG` line in `algo/params.py`.
3. **Iterate fast** on a small dev subset (e.g. one suite, `--max-tasks-per-suite N`,
   reduced trials) to get signal cheaply.
4. **DECISION — full dev eval, arm B:**
   `python track_judge/benchmark.py --arm tracking --suites <dev> --gen <N>`
   then compare to BEST and to `results/baseline_vdm/`:
   `python track_judge/compare.py gen<N> BEST` and `… gen<N> baseline_vdm`.
   **ACCEPT iff ALL:** (a) **success rate ≥ BEST** (not worse beyond tolerance) on the dev
   subset, AND (b) **median task time ≤ BEST** (not slower beyond tolerance), AND (c) at
   least one of {success, speed} **strictly improves**. The standing experiment goal (B vs A)
   is reported every gen but is not the per-gen gate — the per-gen gate is "beat the current
   champion B".
5. **ACCEPT:** `echo gen<N> > results/BEST`, keep `algo/`, commit. **REJECT:** revert
   `algo/` (`git checkout -- track_judge/algo/`), keep `results/genNNN/` + the CHANGELOG line,
   commit. Update `EVOLUTION.md` either way. Go to 1.

---

## 7. Accept/reject gate — metrics, tolerances, honesty
- **Gated axes (can REJECT):** `success_rate` (tolerance ~1–2 trials worth, i.e. don't reject
  on a single flaky trial), `task_time_median_s` (tolerance ~5%). A **hard gate**: a
  generation that crashes the judge / throws on any task, or that silently judges every task
  "done" (degenerate), is an automatic REJECT regardless of numbers.
- **Informational (report, never gate) — but ALWAYS recorded for the next session.** An
  agentic solution is multi-step / multi-interaction; per-step time and per-step error are the
  real-deployment cost we optimize. `benchmark.py` profiles and writes to `summary.json`:
  per-step timing (`codegen/exec/judge/track` medians + p90), interaction counts (turns, VLM
  calls/turn, aborts/finishes), and the error breakdown (judge-vs-ground-truth confusion:
  `false_done`/`missed_done`, plus `failure_categories`). Read these each generation to see
  where time and errors accumulate, and target them — that is how the thesis (faster AND more
  successful) is actually won.
- **Held-out split.** Tune on the **dev subset** only. Keep ≥2 libero-pro suites **held out**;
  run them only as an occasional generalisation check, **never** in the per-gen loop. Tuning
  on the held-out suites is a FAILURE of the experiment.
- **Evaluate the PURE method — no backstop.** Arm B is scored on its OWN verdicts driving
  the agent. Do not let arm B silently fall back to the VDM, or to GT success, when its
  geometry is unsure — that would borrow the baseline's score. If the judge is uncertain it
  must say so and the agent acts on that; report the resulting failures as failures.
- **Honest reporting.** Report real success + speed numbers with before→after. Never report a
  metric you didn't run. If you cap tasks/trials/suites for speed, **log what was dropped** —
  a partial run must not read as a full one. Simulator/LLM runs are noisy: re-run a borderline
  success flip 2–3× before trusting it.
- **No overfitting.** A win must come from a geometric/judgment **principle that transfers**
  (e.g. "a placed object's points must persist inside the container region across the last K
  frames"), not from task-specific constants fitted to make particular libero tasks pass.

---

## 8. Logging + committing
- **`algo/params.py:CHANGELOG`** — append-only, newest first, one line per gen:
  `(N, "<change> — ACCEPTED/REJECTED. hypothesis. success A→B, time A→B s. root-cause/finding.")`.
  Keep rejected entries (so dead ends aren't retried).
- **`EVOLUTION.md`** — human-readable mirror, one block per gen:
  `### genN — <title> ✅ACCEPTED / ❌REJECTED` then `**改动**` / `**结果**`(success+speed,
  vs BEST and vs baseline_vdm) / `**教训**` / `**下一步**`.
- **Commit EVERY generation** (accepted or rejected). Never `git push`. No `Co-Authored-By` /
  Claude trailer. Subject convention:
  ```
  gen<N>(track-judge): <one-line change> — ACCEPTED      (or — REJECTED)

  <hypothesis + result>. success <a>→<b> (vs VDM <v>), task_time <a>→<b>s (vs VDM <v>).
  ```
  Phase-0 infra commits (not generations) use `[track-judge] <feature>` and are committed as
  they are verified.
- **`REPORT.md`** — the living experiment report (the research deliverable, like
  self_evolve's paper). On every **ACCEPT** that moves a headline number, update its result
  tables (§6) and add the durable finding to its findings log (§7). Keep it honest and
  paper-ready: it is what the experiment is *for* (proving the VDM is a removable, redundant
  LLM call). Numbers stay authoritative in `summary.json`/`CHANGELOG`; `REPORT.md` synthesises.
- Commit the `algo/` change (or its revert), `results/genNNN/` incl. `tracking-viz/`,
  `results/BEST`, the CHANGELOG line, `EVOLUTION.md`, and (on accept) `REPORT.md`.

---

## 9. Hard rules (do not violate)
1. **No human feedback.** Success = GT `check_success()`. Promotion = the automatic accept
   gate here, NOT a human-merged PR. (This is the deliberate departure from cap-x-se.)
2. **Only edit `track_judge/algo/`** during generations. Never touch the cap-agent0 loop, the
   suites, the VDM (arm A), or `benchmark.py`/`compare.py` — that makes generations
   incomparable. The Phase-0 integration seam in `trial.py`/`libero.py` is built once, then
   frozen.
3. **One hypothesis per generation; commit every generation; revert code on reject.**
4. **Keep it elegant.** General geometric principles over piles of per-task constants. A
   constant is OK when it is physical (a drop height in metres, a container-margin in cm),
   never when fitted to specific tasks. Overfitting libero-pro is a FAILURE.
5. **Always save tracking-viz** (§5). A generation without debug visualizations is incomplete.
6. **Report both metrics, honestly** (§7). The thesis is a *pair* (success AND speed); never
   claim it on one.

---

## 10. Task decomposition + parallel sub-agents (encouraged)
A generation's *coding* may be split and run in parallel — spawn sub-agents for independent
pieces and integrate their results yourself. Natural splits:
- one agent on a `judge_lib.py` primitive (e.g. `points_in_container_3d`),
- one on the judge-codegen prompt wording,
- one on a mask-selection strategy,
- one on the tracking-viz renderer.
Keep the *evaluation* single-threaded and deterministic (one arm, fixed seeds) so the
accept/reject verdict stays clean. Still: one hypothesis, one commit per generation.

---

## 11. PHASE 0 — bootstrap (build the infra, THEN start generations)
The harness does not exist yet. Phase 0 is **infrastructure** (committed as `[track-judge]`,
not as generations). Do it first; it is the only time you touch frozen files. Suggested
decomposition for parallel sub-agents:
1. **Frame recording** — dense + depth + camera intrinsics/extrinsics in `libero.py`, gated
   behind `state_judge == tracking` (§4).
2. **Arm switch** — `state_judge: vdm|tracking` config key threaded into the trial config;
   route `_handle_multi_turn_step` feedback to `track_judge.algo.track_judge` when `tracking`.
3. **TAPIP3D client + harness** — vendor `tapip3d_client.py`; write `algo/track_judge.py`
   (dense RGB-D + cam params → tracker → run the agent's judge fn → verdict + viz).
4. **judge_lib + judge_prompt** — initial geometric primitives and the codegen prompt fragment.
5. **Single-Gemini config** — `franka_libero_track_judge.yaml` (ensemble off, Gemini).
6. **benchmark.py / compare.py / run_evolve.sh** — port the self_evolve structure: per-gen
   `results/genNNN/`, a `BEST` pointer, success+speed scoring, the tiered accept gate, and a
   one-fresh-session-per-generation driver.
7. **baseline_vdm** — run arm A once on all suites → freeze `results/baseline_vdm/` as the
   number to beat (re-measured only if suites change).
Verify Phase 0 with a 1-task smoke per arm (both produce a verdict; arm B writes viz), commit
each piece as it works, then begin **gen1**.
