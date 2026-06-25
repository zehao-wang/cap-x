"""judge_prompt — the prompt fragment appended to cap-agent0 when state_judge==tracking.

It tells the single-LLM agent to ALSO emit a top-level ``judge_state(ctx)`` that judges
task state from 3D point TRAJECTORIES (pure code, no LLM at judgment time).

DESIGN (AGENT.md north-star: make a WEAK model succeed): the agent should NOT write raw
numpy geometry. In almost every case its judge is ONE LINE — name the goal RELATION and
the OBJECTS by plain words, and the harness does the rest (local SAM segmentation →
TAPIP3D 3D tracking → metric geometry). Raw-``ctx`` code is only an escape hatch.
"""

from __future__ import annotations

JUDGE_PROMPT_FRAGMENT = r'''
================ SELF-EVALUATE: write a geometric state judge ================
In addition to your execution code, define ONE top-level function:

    def judge_state(ctx):
        from track_judge.algo import judge_dsl as J
        return J.judge(ctx, RELATION, target="<object>", reference="<object>")

It decides task state by GEOMETRY over 3D point TRAJECTORIES (not one image) and must
NOT call any LLM/VLM. It runs once per turn on the turn's dense RGB-D video, AFTER your
execution code. You name objects in PLAIN WORDS; a LOCAL segmentation model finds them
and a 3D tracker follows them — you do not write coordinates or numpy.

----- THE EASY PATH: pick ONE relation (this covers almost every task) -----
    J.judge(ctx, "place_in",   target="bowl",   reference="bin")    # X came to rest inside Y
    J.judge(ctx, "on_top_of",  target="cube",   reference="plate")  # X rests on top of Y
    J.judge(ctx, "stack",      target="red block", reference="blue block")
    J.judge(ctx, "next_to",    target="cup",    reference="plate", dist=0.10)
    J.judge(ctx, "opened",     target="drawer handle", travel=0.15) # articulated open (metres)
    J.judge(ctx, "closed",     target="drawer handle", travel=0.15)
    J.judge(ctx, "lifted",     target="mug", lift=0.05)             # picked up off support
    J.judge(ctx, "removed_from",target="bowl", reference="cabinet") # taken out of Y
    J.judge(ctx, "grasped",    target="mug", gripper="robot gripper") # held / co-moving (slip→abort)

Relation aliases are accepted (open/close, pick_up, put_in, beside, take_out, ...).
Optional physical knobs (metres) tune thresholds: travel=, lift=, dist=, k=, z_height=.
Name objects exactly as a person would point at them ("the white bowl", "top drawer").

The call RETURNS the verdict the agent acts on:
    {"done": bool, "abort": bool, "progress": float in [0,1], "feedback": str, "regions":[...]}
- done   -> FINISH ;  abort=True -> REGENERATE NOW (early failure, e.g. detected slip/drop)
- progress is a graded distance-to-goal; feedback is a METRIC residual you can act on next
  turn (e.g. "target 4.0cm from container center"), not a vague opinion.

If the goal is TWO conditions ("open the drawer AND put the bowl inside"), use all_of —
each sub-goal is (relation, {kwargs}); done = all done, abort = any abort:
    return J.all_of(ctx,
        ("opened",   {"target": "drawer handle", "travel": 0.15}),
        ("place_in", {"target": "bowl", "reference": "drawer"}))

----- ESCAPE HATCH: only if no relation fits, write raw geometry over `ctx` -----
  ctx.coords [T,N,3] world-frame metres tracks · ctx.visibs [T,N] · ctx.rgb [T,H,W,3] ·
  ctx.depth [T,H,W] · ctx.K [3,3] · ctx.task (str) · ctx.state (dict, persists across turns)
  ctx.points_of("name") -> [T,Nt,3] tracks of a named object (local SAM + tracker)
  ctx.lib : geometric primitives (points_in_region_3d, object_dropped, relative_motion,
            min_pairwise_distance, persistence_in_region, speed, ...)
Return the same verdict dict. Prefer the easy path; this is for unusual goals only.
==============================================================================
'''
