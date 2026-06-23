"""judge_prompt — the prompt fragment appended to cap-agent0 when state_judge==tracking.

It tells the single-LLM agent that, in addition to the usual execution code, it must
ALSO emit a top-level ``judge_state(ctx)`` that judges task state from 3D point
TRAJECTORIES (pure code, no LLM at judgment time). Keep it concrete.
"""

from __future__ import annotations

JUDGE_PROMPT_FRAGMENT = r'''
================ SELF-EVALUATE: write a geometric state judge ================
In addition to your execution code, define ONE top-level function:

    def judge_state(ctx):
        ...
        return {"done": bool, "abort": bool, "progress": float, "feedback": str}

This function decides task state by GEOMETRY over 3D point TRAJECTORIES — NOT from a
single image, and it must NOT call any LLM/VLM. It runs once per turn, on the turn's
dense RGB-D video, after your execution code runs. You may import ANY package (numpy,
scipy, your own geometry); convenience primitives are in `ctx.lib`.

Because you see motion over time (not one frame), test things a frame-diff cannot:
object dropped / fell (descent past support), grasp slip (object stops co-moving with
the gripper), collision / clearance (min inter-set distance over time), 3D containment
that PERSISTS across the last K frames, and early failure -> set abort=True to retry
immediately instead of running the whole trajectory.

`ctx` exposes:
  ctx.coords   : np.ndarray [T, N, 3]  tracked points in WORLD frame, metres, over T frames
  ctx.visibs   : np.ndarray [T, N]     visibility (TAPIP3D predicts occluded points too)
  ctx.rgb      : np.ndarray [T, H, W, 3] uint8   the (sub-sampled) turn video
  ctx.depth    : np.ndarray [T, H, W] float32    depth in metres
  ctx.K        : np.ndarray [3, 3]     camera intrinsics
  ctx.lib      : the judge_lib module (points_in_region_3d, centroid, fraction_in_region,
                 object_dropped, relative_motion, min_pairwise_distance, speed,
                 persistence_in_region, ...)
  ctx.task     : str  the task description
  ctx.state    : dict persistent ACROSS turns — stash baselines/counters here
  ctx.mask_points(mask2d) -> int[]  : indices of tracked points seeded inside a 2D bool
                                      mask [H, W]; default tracking is a whole-frame grid

Verdict fields (all optional except you should set `done`):
  done     : True when the task goal is geometrically satisfied (drives FINISH)
  abort    : True on detected failure -> REGENERATE/retry now (early abort)
  progress : optional float in [0, 1], a graded distance-to-goal signal
  feedback : optional short string for the agent's next turn
  regions  : optional list of region dicts to draw in the saved tracking-viz, e.g.
             {"type": "aabb", "lo": [x,y,z], "hi": [x,y,z]}
A bare bool or bare string return is tolerated.

Example — "place the bowl in the bin" (object placed inside a 3D region and stays):

    def judge_state(ctx):
        lib = ctx.lib
        bin_region = {"type": "aabb", "lo": [0.30, -0.10, 0.00], "hi": [0.50, 0.10, 0.12]}
        obj = ctx.coords  # whole-frame grid; refine with ctx.mask_points(...) if you segment
        placed = lib.persistence_in_region(obj, bin_region, k=4)
        dropped, frame = lib.object_dropped(obj, support_z=0.0)
        frac = lib.fraction_in_region(obj[-1], bin_region)
        if dropped and not placed:
            return {"done": False, "abort": True,
                    "feedback": f"object fell at frame {frame}; regrasp and retry",
                    "regions": [bin_region]}
        return {"done": placed, "abort": False, "progress": frac,
                "feedback": "in bin" if placed else "not yet in bin region",
                "regions": [bin_region]}
==============================================================================
'''
