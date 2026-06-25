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
    (1, "gen1 — declarative goal-relation DSL (judge_dsl) + ctx.points_of (local SAM) + "
        "weak-LLM judge prompt — PENDING-EVAL (LLM + TAPIP3D endpoints down, no A/B yet). "
        "hypothesis: a WEAK runtime LLM can author the per-turn judge in ONE line by naming "
        "a RELATION (place_in/on_top_of/opened/lifted/grasped/...) over OBJECTS named in "
        "plain words; all segmentation (local SAM), tracking (TAPIP3D) and metric geometry "
        "are pushed into deterministic harness. judge returns graded {done,abort,progress,"
        "feedback} with a METRIC residual to drive the next code revision. Offline-validated: "
        "judge_dsl self-test + import/plumbing + graceful whole-grid fallback when SAM is "
        "down. NEXT: bring up SAM3+TAPIP3D, validate named-object resolution on a real "
        "libero frame, then run the A/B benchmark."),
]
