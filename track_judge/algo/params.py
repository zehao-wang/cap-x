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
]
