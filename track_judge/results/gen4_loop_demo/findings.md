# gen4 — arm-B loop with the runtime LLM SIMULATED (no :8110), + two harness fixes it surfaced

Instead of starting the (in-flight, blocked) Codex endpoint, the experimenter stands in for the
cap-agent0 runtime LLM: reads the task and writes the ONE-LINE declarative judge per
`judge_prompt`. Execution is stubbed by driving the drawer joint (standing in for the LLM's
motion plan). This exercises the REAL arm-B decision path — `TrackJudge.judge_turn` (the
production seam) → geometric verdict → FINISH/REGENERATE — end to end, without the endpoint.

Run: `MUJOCO_GL=egl HF_HUB_OFFLINE=1 .venv-libero/bin/python demo_loop_no_llm.py <out>`
Judge written by the "LLM": `J.judge(ctx, "opened", target="drawer handle", travel=0.16)`.

## Result — geometry drives the code-revision loop, agrees with GT every turn
| turn | execution (LLM) | judge verdict | agent decision | GT (qpos / done) | judge-vs-GT |
|---|---|---|---|---|---|
| 1 | short pull → −0.05 | not done, "moved …cm of ~16cm" | **REGENERATE** | −0.050 / False | ✅ AGREE |
| 2 | full pull → −0.16 (revised) | **done**, "moved **15.8cm** of ~16cm", prog 0.99 | **FINISH** | −0.160 / True | ✅ AGREE |

The metric residual ("moved 15.8 cm of ~16 cm") is exactly the actionable signal that tells the
next code revision what to fix — no LLM at judgment time.

## Two real harness gaps this surfaced (both fixed in `algo/`, gen4)
1. **Multi-turn openness needs a persistent CLOSED baseline.** `opened()` measured displacement
   within ONE window; across turns (each opening a bit) it never reaches travel. Fixed: `opened()`
   stashes the handle's first-ever (closed) position in `ctx.state` and measures absolute openness
   from it — so turn 2 reads 15.8 cm *from closed*, not just its own window.
2. **The global 24×24 grid is too coarse to seed a small object (drawer handle ≈ 0.2% of frame).**
   It got 0–1 grid hits → the centroid was the static background → "moved 0.2 cm" even when the
   drawer was fully open (turn 2 FAILED before this fix). Fixed: `ctx.points_of` now DENSELY
   seeds query points inside the object's mask and runs a dedicated TAPIP3D track
   (`ctx.track_mask`, back-projecting masked pixels to world `query_point`s) when the grid yields
   < 6 hits. With it, the handle tracks the full 15.8 cm.

## Takeaway
The arm-B value loop works end to end with a one-line geometric judge standing in for the VDM:
correct FINISH/REGENERATE, agreeing with ground truth, zero LLM at judgment. Acting as the
runtime LLM (rather than waiting on :8110) was the fastest way to surface the two harness gaps
that a synthetic self-test could not. The A/B *numbers* still need the real runtime LLM, but the
mechanism is now demonstrated on a real task across turns.
