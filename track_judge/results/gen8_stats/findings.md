# gen8 — STATISTICS: the geometric-judge advantage, quantified

Two halves: (A) what the per-turn VDM costs, mined from the official cap-x VDM baseline logs;
(B) the geometric judge's accuracy vs ground truth over a sampled sweep.

## A. The VDM baseline is statistically poor — and the failure is a JUDGMENT pathology
Mined from `outputs/track_judge_*vdm*/**/summaries.txt` (N=68 VDM-arm runs, Gemini, mostly n=1
per task across the dev suites):

| statistic | value | reading |
|---|---|---|
| GT task success | **8/68 = 12%** | the VDM-driven agent almost always fails |
| runs that never FINISH (`finishes==0`) | **31/68 = 46%** | the VDM never recognizes "done" → REGENERATEs to the horizon |
| GT-successes the agent never self-finished | 1/8 | a real success the VDM missed (missed-done) |
| valid code yet task fails | 50/68 = 74% | the agent writes runnable code but doesn't succeed |
| mean regenerations / run | **4.5** (median 3.5, max 11; 24% hit ≥10) | in arm A each REGEN = one extra VDM call |
| per-suite success | 6–27% | goal_task 7%, object_task 6% (worst) |

The key point: **46% of runs never converge a verdict at all** — the per-turn VDM, the signal that
should say "done, stop" or "failed, change tack", doesn't. This is the redundant/harmful-call
thesis in numbers, not just an anecdote.

## B. The geometric judge matches ground truth (accuracy sweep)
`stats_sweep_judge_accuracy.py`: `on_top_of(bowl, plate)` on libero_goal task8, sampling states
by teleporting the bowl across **3 seeds × 7 placements** (3 seated-on-plate via physics settle,
3 clearly-off, 1 hovering over the plate but 12 cm too high). Judge verdict vs `task_completed()`:

| metric | value |
|---|---|
| accuracy (judge == GT) | **21/21 = 100%** |
| confusion | true_done 9 · **false_done 0** · **missed_done 0** · true_notdone 12 |
| precision / recall | **1.00 / 1.00** |

Notably the **hover** case (bowl over the plate but 12 cm up) is correctly judged NOT done — the
judge discriminates "seated on" from "above", which a frame-diff that sees the bowl overlapping
the plate in 2D can get wrong. 0 false-done and 0 missed-done is exactly the pair the VDM gets
nonzero on (≥1 missed-done in just 8 successes above).

## The advantage, stated statistically
- **Reliability:** geometric judge 21/21 = 100% (0 false/missed-done) on a controlled sweep,
  vs a VDM baseline where 46% of runs never even produce a "done" and ≥1/8 successes were missed.
- **Call count:** the geometric judge makes **0 LLM calls per turn** vs the VDM's 1; the baseline
  averages **4.5 regenerations/run**, i.e. ~4.5 extra VDM calls/run that the geometric arm removes.
- **Determinism:** every repeated placement above returns the same verdict (the three "on" and
  three "off" samples per seed are identical-verdict), vs an LLM judge sampled at temperature.

## Honest limits
n is modest and the sweep states are controlled (teleported, clean on/off), one relation, one
task — this measures the judge's *accuracy ceiling*, not a same-loop A/B against the VDM (which
needs the runtime LLM endpoint). The baseline numbers are n=1/task and noisy. Both are real and
point the same way; the head-to-head success+speed A/B is the remaining confirmation.
