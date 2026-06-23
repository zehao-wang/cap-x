# Removing the VDM: 3D-Tracking Geometric State Judgment for Code-as-Policies Agents

**A cap-x-se experiment report.** Living document — maintained on every accepted generation
(see `AGENT.md` §8). Authoritative numbers live in `results/genNNN/summary.json` and the
`algo/params.py:CHANGELOG`; this file is the human-readable synthesis toward a paper.

Status: **Phase 0 (bootstrap) — results pending.** Tables below are scaffolds to be filled.

---

## 1. Research question
Current agentic Code-as-Policies solutions (cap-x-se) call a **Visual Differencing Model
(VDM)** — a VLM — every turn to judge task state and drive the agent's self-correction
(REGENERATE/FINISH). We ask:

> **Is the VDM a redundant LLM call?** Can it be *removed entirely* and replaced by
> **agent-written geometric state-judgment code over 3D point trajectories** (TAPIP3D), such
> that the agent becomes **both more successful and faster**?

## 2. Thesis & hypotheses
We call the new agent **self-evaluate**: it self-**plans** (execution code) and
self-**evaluates** (state-judgment code), and **both are code — neither calls an LLM** at
judgment time; the judgment code may import **any packages**. The VDM, by contrast, is a
*second* VLM call whose errors are **correlated** with the policy LLM (same modality/family):
it largely returns the agent's own opinion, at latency + cost. A geometric judge over 3D
tracks is a **decorrelated, physically grounded** signal and is local/cheap.

- **H1 (effectiveness):** tracking-judge success rate ≥ VDM, ideally strictly higher.
- **H2 (speed):** tracking-judge median task time < VDM (no per-turn VLM call; early abort).
- **H3 (mechanism):** the gains come from capabilities the VDM structurally lacks —
  continuous metric signal, collision/clearance, grasp-slip, early failure detection,
  occlusion-robust 3D containment (§5).

## 3. Background / system under test
- **Agent:** cap-agent0, the single-LLM Code-as-Policies loop
  (`capx/envs/trial.py:_run_single_trial`); writes raw Python executed against bound robot
  APIs; multi-turn REGENERATE/FINISH self-correction. **Single LLM = Gemini**
  (`google/gemini-3.1-pro-preview`), ensemble disabled.
- **Benchmark:** libero-pro (`capx/third_party/LIBERO-PRO`), success = ground-truth
  `check_success()`. Dev suites for iteration; ≥2 suites held out for generalisation.
- **Baseline (Arm A):** stock VDM (`_get_visual_differencing_feedback` /
  `_get_video_differencing_feedback`), Gemini VLM.
- **Treatment (Arm B):** VDM removed; per-turn feedback comes from agent-written geometric
  judge code run over the turn's **dense RGB-D video** via the TAPIP3D socket service. The
  agent chooses masks; judgment is geometric. Single switch `state_judge: vdm|tracking`.

## 4. Method
- **Frame recording:** dense (every motion frame) RGB **+ depth** + MuJoCo agentview
  intrinsics/extrinsics, gated behind the tracking arm (sim → exact RGB-D).
- **Tracking judge:** dense RGB-D + camera params → TAPIP3D 3D tracks (world frame) → the
  agent's judge function applies `judge_lib` geometric primitives → verdict
  {progress, done, failure, abort} feeding REGENERATE/FINISH. Tracking visualizations saved
  every trial for debug.
- **Protocol:** autonomous self-evolve loop (`AGENT.md`); one hypothesis/generation; accept
  iff it beats the current champion on success AND speed (tolerances in `compare.py`);
  commit every generation; held-out suites never tuned on; the tracking arm is scored as the
  **pure method** (no VDM/GT backstop).
- **Controls:** identical agent, LLM, suites, seeds, init states across arms; only the
  feedback mechanism differs.
- **Two levels (do not conflate).** *Runtime* is the agent under test in cap-x's standard
  environment, whose only LLM call is writing the plan+judge code (judgment then runs as pure
  code, no LLM). *Self-evolve* is an external meta-loop (like motion_plan/self_evolve) that
  edits the agent's design from outside based on results; it is absent at runtime.

## 5. Why a 3D trajectory beats a frame-diff VLM (qualitative contribution)
(Mirror of `AGENT.md` §0.1 — these are the mechanism claims H3.)
collision/near-miss · grasp-slip (relative-to-gripper) · early failure → early abort · drop
detection · 3D containment/placement · occlusion robustness (occluded points predicted) ·
appearance/viewpoint invariance · multi-object relations · determinism/debuggability ·
dense+cheap+graded progress · "right motion, wrong outcome" (missed grasp). The VDM, a
discrete 2D semantic snapshot from a correlated VLM, can do none of these.

## 6. Results (to be filled as generations land)

### 6.1 Headline — Arm B (best) vs Arm A (VDM), dev suites
| arm | success rate | median task time (s) | per-turn judge latency (s) | VLM calls/turn |
|---|---|---|---|---|
| A — VDM baseline | _TBD_ | _TBD_ | _TBD_ | 1 |
| B — tracking (BEST) | _TBD_ | _TBD_ | _TBD_ | 0 |

### 6.2 Per-suite success + speed (dev + held-out)
| suite | A success | B success | A time | B time | dev/held-out |
|---|---|---|---|---|---|
| libero_object_swap | _TBD_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ |
| … | | | | | |

### 6.3 Ablations / mechanism evidence (planned)
- early-abort on vs off (isolates H2 speed source);
- which `judge_lib` capabilities each accepted gen used (collision / slip / containment / …);
- failure cases where geometry beat VDM and vice-versa (with tracking-viz).

## 7. Findings log (per accepted generation)
_Synced with `EVOLUTION.md` / `CHANGELOG`. One durable finding per accepted gen._
- _gen1: TBD_

## 8. Limitations / threats to validity
- TAPIP3D is **not real-time** (~1–2.5 fps, window ≥12); the per-turn judge cost must still
  net out faster than a VLM call — verify, don't assume. Early-abort is the main speed lever.
- **Sim depth is exact**; a real-robot port (ZED + stereo depth) is noisier — results here
  are a sim upper bound, flagged as such.
- Mask/segmentation quality bounds the judge; report when a failure is mask-driven.
- Overfitting risk: wins must come from transferable geometric principles, checked on the
  held-out suites.
- Decorrelation claim is argued, not yet measured; §6.3 ablations should substantiate it.

## 9. Conclusion
_TBD — does removing the VDM, replaced by 3D-tracking geometric judgment, improve both
success and speed? State the verdict with numbers once generations converge._
