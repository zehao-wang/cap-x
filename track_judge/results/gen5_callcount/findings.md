# gen5 — CALL-COUNT A/B across tasks (runtime LLM simulated), + honest-unresolved fix

Metric of record (per the user): the NUMBER of LLM calls, not speed. The experimenter stands
in for the cap-agent0 runtime LLM (writes the one-line declarative judge); execution is stubbed
by driving each task's target object/joint (GT used ONLY to place the stub, never in the judge).
Each task loads its REAL libero task so `env.task_completed()` is true ground truth. 2-turn loop
per task (turn 1 insufficient → REGENERATE, turn 2 complete → FINISH).

Run: `MUJOCO_GL=egl HF_HUB_OFFLINE=1 .venv-libero/bin/python demo_callcount.py <out>`

## Call count — the headline
| | judgment LLM calls | codegen calls (1/turn) | total LLM calls |
|---|---|---|---|
| **arm A (VDM)** | **6** (1 per turn) | 6 | **12** |
| **arm B (geometry judge)** | **0** | 6 | **6** |

**arm B removes the per-turn VDM call entirely → 50% fewer LLM calls** (6 of 12) on this 3-task,
6-turn run. The reduction equals the number of turns; it grows with every turn and every
early-abort the geometry enables. This is structural, not stochastic: arm B's judgment is pure
code.

## Per-task behaviour (honesty)
| task | relation | turn1 | turn2 | judge-vs-GT |
|---|---|---|---|---|
| 0 open the middle drawer | `opened` | not-done → REGEN | **done → FINISH** | ✅ AGREE both (joint GT, faithful stub) |
| 8 put the bowl on the plate | `on_top_of` | not-done → REGEN | done → FINISH | geometry correct; GT predicate needs physical contact a teleport can't make |
| 6 put the cream cheese on the bowl | `place_in` | "could not resolve target" | "could not resolve target" | SAM can't ground "cream cheese" (0.02) |

- **task0** is fully GT-validated: the drawer's GT is joint-position based, so the stub drives
  true GT and the geometric judge AGREES every turn (the same result as gen4).
- **task8**: the judge geometrically detects the bowl reaching the plate (REGENERATE→FINISH),
  but LIBERO's "on plate" GT predicate needs physical contact/settling that a *positional*
  teleport stub does not produce — a stub limitation, not a judge error. Real motion execution
  (what the A/B benchmark runs) provides the contact.
- **task6** surfaced and drove a real fix: SAM cannot ground "cream cheese" (score 0.02), and
  the DSL used to silently fall back to the whole-frame grid → a confident-but-wrong verdict
  ("50.8cm from container"). Now `_resolve` returns None for an unresolvable named object and
  `judge()` returns an **honest "could not resolve target"** verdict (progress n/a) — the
  semantic-grounding boundary reported as uncertainty, never as a fake geometric pass.

## Takeaway
On call count — the agreed metric — removing the VDM is a clean **−1 LLM call per turn (−50%
total here)**, with zero judgment LLM calls in arm B. Geometry correctly gates FINISH/REGENERATE
where the object is resolvable (task0 GT-validated; task8 geometrically correct). Where the
object is NOT resolvable (task6 cheese), the judge now says so honestly rather than guessing —
the one residual is grounding, not the per-turn judgment.
