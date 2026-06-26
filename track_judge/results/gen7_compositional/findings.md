# gen7 — COMPOSITIONAL task generalization: "open the top drawer and put the bowl inside"

`libero_goal` task3 — a MULTI-STEP task the official cap-x VDM baseline FAILS on **both**
`libero_goal_swap` and `libero_goal_task` (objects are SAM-resolvable, so the failure is
operation/judgment, not grounding — the class the user asked to target). The strongest
generalization test for the paradigm: TWO subgoals judged by ONE declarative `all_of`.

Judge (the whole thing the "runtime LLM" writes):
```python
def judge_state(ctx):
    return J.all_of(ctx,
        ("opened",   {"target": "top drawer handle", "travel": 0.16}),
        ("place_in", {"target": "bowl", "reference": "top drawer"}))
```
GT (from the BDDL): `(In akita_black_bowl_1 wooden_cabinet_1_top_region)`. Our judge is
STRICTER — it also requires the drawer open — matching the instruction's two subgoals.

## Result — full task, 3-turn loop, judge-vs-GT AGREE every turn
| turn | execution (stub) | judge | subgoal residual | GT | agree |
|---|---|---|---|---|---|
| 1 | open halfway, bowl out | not-done (0.24) → REGEN | "handle 7.8/16cm \| bowl 29.4cm, 0% inside" | False | ✅ |
| 2 | open fully, bowl out | not-done (0.49) → REGEN | "handle **15.8/16cm** \| bowl 16.3cm, **0% inside**" | False | ✅ |
| 3 | place bowl in drawer | **done (0.97) → FINISH** | "handle 15.8/16cm \| **in container (xy off 2.4cm)**" | **True** | ✅ |

The compositional **residual is the headline**: at every turn it says WHICH subgoal remains —
turn 2's "drawer open ✓, bowl still 0% inside" is exactly the signal that tells the next code
revision to stop re-opening and start placing. This is what a frame-diff VDM cannot give and
why multi-step tasks (where it must track progress across subgoals) are where it fails worst.

## The generalization gap this surfaced — and the fix (gen7, in `algo/`)
First run FAILED at turn 3: SAM-by-text is **state-dependent** — "top drawer handle" (0.477→0.145)
and "top drawer" (0.330→0.236) both resolved on the CLOSED scene but dropped below the 0.30 gate
once the drawer was open + the bowl occluded it. The judge HONESTLY returned "could not resolve"
(the gen5 behaviour) rather than faking — but the final containment went unjudged.

Fix: **cross-turn resolution cache** in `ctx.points_of`. When an object resolves, its last-frame
world points are stashed in `ctx.state`; when a later turn fails to resolve it, the last-known
position is REUSED (a static trajectory) instead of giving up — valid because a structure that
was just localized has not teleported (the drawer is open and static by turn 3). With it, turn 3
reuses the turn-2 drawer/handle localization → `place_in` fires → done, AGREE with GT.

## Takeaway
The paradigm **generalizes to a compositional task cap-x fails**: one `all_of` line, two subgoals
judged geometrically, a per-subgoal metric residual driving revision, correct FINISH agreeing
with GT — zero LLM at judgment. The recurring limiter is per-frame SAM grounding (here
state-dependent), now mitigated by cross-turn caching; the geometry/composition itself is sound.
