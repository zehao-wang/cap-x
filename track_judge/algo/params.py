"""params — DEFAULTS + append-only CHANGELOG for the tracking-judge harness.

Keep CHANGELOG append-only: one entry per generation, newest last. Each entry is
``(gen:int, "<change> — ACCEPTED/REJECTED. hypothesis. metrics. finding.")``.
"""

from __future__ import annotations

DEFAULTS = {
    "window": 12,               # frames sub-sampled per turn for the tracker (>= 12)
    "resolution_factor": 1.0,   # TAPIP3D internal resolution scale
    "num_iters": 6,             # TAPIP3D refinement iterations
    "query_grid": 24,           # default whole-frame query grid (NxN seed)
    "filter_depth": False,      # TAPIP3D depth filtering (off per AGENT.md §5)
}

CHANGELOG: list[tuple[int, str]] = [
    (0, "gen0 — seed: initial self-evaluate tracking-judge harness + judge_lib + prompt."),
    (1, "gen1 — declarative goal-relation DSL (judge_dsl) + ctx.points_of (local SAM, "
        "score-gated) + weak-LLM judge prompt — SIGNAL-PATH VALIDATED on real libero data; "
        "A/B benchmark still PENDING (runtime LLM endpoint :8110 mid-migration to Codex CLI). "
        "hypothesis: a WEAK runtime LLM can author the per-turn judge in ONE line by naming "
        "a RELATION (place_in/on_top_of/opened/lifted/grasped/...) over OBJECTS in plain "
        "words; all segmentation (local SAM), tracking (TAPIP3D) and metric geometry pushed "
        "into deterministic harness; verdict is graded {done,abort,progress,feedback} with a "
        "METRIC residual to drive the next code revision. VALIDATED (results/gen1_validation/): "
        "(A) plain-word->SAM resolves distinct objects at score 0.58-0.93 vs failures <=0.08 "
        "-> added MIN_SEG_SCORE=0.30 gate; (B) full chain plain-word->SAM->TAPIP3D world track "
        "centroid lands 0.6-2.7cm from GT object pose, well under DSL thresholds; DSL relations "
        "read the real scene correctly. FINDING: 'robot gripper' unresolvable by SAM (0.010) -> "
        "gen2 should take the gripper traj from proprioception for grasped(). NEXT: run A/B."),
    (2, "gen2 — all_of() multi-condition helper + DSL coverage map (COVERAGE.md) — offline. "
        "Mapped every base task of the dev suites to a relation: place_in covers all 10 "
        "libero_object (place in basket), on_top_of covers all 10 libero_spatial (bowl on "
        "plate), and place_in/on_top_of/opened cover 8/10 libero_goal -> 28/30 dev tasks with "
        "THREE relations. all_of(ctx, (rel,kwargs), ...) AND-combines for 2-condition goals "
        "like goal[3] 'open the top drawer and put the bowl inside'. Residuals (anticipated "
        "geometry boundary, neither needs a per-turn VLM): goal[5] push-to-front (proximity, "
        "approx via next_to) and goal[7] turn-on-stove (non-spatial state). self-test passes "
        "(relations + dispatch + all_of). Still PENDING A/B (endpoint)."),
    (3, "gen3 — POSITIVE done=True validation on real tracks + on_top_of progress fix — offline. "
        "Guards the degenerate-judge failure mode (a judge that never says done would pass all "
        "the not-done cases): teleport the bowl onto the plate over a rendered window -> TAPIP3D "
        "-> on_top_of(bowl,plate) FLIPS to done=True, progress=1.0 (val_positive.py). Fixed "
        "on_top_of progress to be the last-k region-occupancy fraction so the graded signal "
        "AGREES with done (was 0.097 while done=True via a dz-penalty artifact; now 1.0 seated). "
        "judge now verified in all 3 directions on real data: not-done/metric (A,B), done (C). "
        "Still PENDING A/B (endpoint)."),
    (4, "gen4 — arm-B loop with the runtime LLM SIMULATED (no :8110) + 2 harness fixes it "
        "surfaced. Stood in for cap-agent0's LLM: wrote the 1-line judge "
        "J.judge(ctx,'opened',target='drawer handle') and ran the REAL TrackJudge.judge_turn "
        "seam across 2 turns on libero_goal/task0. RESULT: geometry drives the loop correctly "
        "(turn1 short pull -> not-done -> REGENERATE; turn2 full pull -> done 'moved 15.8cm of "
        "16cm' -> FINISH), judge-vs-GT AGREE both turns. FIX1: opened() keeps a persistent "
        "CLOSED baseline in ctx.state so multi-turn openness accumulates (was per-window -> "
        "never reached travel). FIX2: ctx.points_of densely seeds query_points INSIDE a small "
        "object's mask + dedicated TAPIP3D track (ctx.track_mask) when the 24x24 grid yields "
        "<6 hits -- a drawer handle (~0.2% of frame) got 0-1 grid hits so its centroid was "
        "static background ('moved 0.2cm' while fully open); dense seeding tracks the full "
        "15.8cm. A/B numbers still need the real runtime LLM, but the mechanism is demonstrated."),
]
