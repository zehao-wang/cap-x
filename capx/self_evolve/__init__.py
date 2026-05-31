"""Self-evolve pipeline: shared storage scaffold + config (module ⓪).

This package is the code home for the cap-x-se self-evolve system. So far it
provides the shared contract that modules ③④⑤ build on:

- :mod:`capx.self_evolve.config`  — central hyper-params (``docs-se/config.md``)
- :mod:`capx.self_evolve.schemas` — ``mem/`` data schemas (``docs-se/storage.md``)
- :mod:`capx.self_evolve.storage` — ``MemStore`` read/write helpers over ``mem/``

See ``docs-se/README.md`` for the end-to-end pipeline and ``GOAL.md`` for status.
"""

from __future__ import annotations

from capx.self_evolve.config import (
    BenchmarkEvalConfig,
    ExperienceDistillConfig,
    SelfEvolveConfig,
    UpdatePlannerConfig,
)
from capx.self_evolve.schemas import (
    DIGEST_SECTIONS,
    CandidateStats,
    Digest,
    SuccessLog,
    slugify_task,
)
from capx.self_evolve.storage import MemStore, default_mem_root, make_history_id
from capx.self_evolve.feedback_postprocessor import (
    PostprocessResult,
    distill_experience,
    generalize_by_rewrite,
    run_feedback_postprocessor,
)
from capx.self_evolve.history_reader import HistoryReader
from capx.self_evolve.proposal import (
    Proposal,
    ProposalCandidate,
    validate_proposal,
    write_accepted,
)
from capx.self_evolve.update_planner import (
    UpdatePlannerResult,
    run_update_planner,
    should_trigger,
)
from capx.self_evolve.benchmark_eval import (
    CandidateDecision,
    CandidateInfo,
    EvalRoundResult,
    EvaluatorResult,
    accumulate,
    candidate_infos,
    candidate_names_in_code,
    decide,
    render_report,
    run_benchmark_evaluator,
    scan_referenced_names,
)
from capx.self_evolve.library_pr import (
    PlannedPR,
    apply_planned_changes,
    create_library_pr,
    plan_pr,
    stamp_docstring_date,
)

__all__ = [
    "SelfEvolveConfig",
    "ExperienceDistillConfig",
    "UpdatePlannerConfig",
    "BenchmarkEvalConfig",
    "SuccessLog",
    "Digest",
    "CandidateStats",
    "DIGEST_SECTIONS",
    "slugify_task",
    "MemStore",
    "default_mem_root",
    "make_history_id",
    "run_feedback_postprocessor",
    "generalize_by_rewrite",
    "distill_experience",
    "PostprocessResult",
    "HistoryReader",
    "Proposal",
    "ProposalCandidate",
    "validate_proposal",
    "write_accepted",
    "run_update_planner",
    "should_trigger",
    "UpdatePlannerResult",
    "CandidateInfo",
    "EvalRoundResult",
    "CandidateDecision",
    "EvaluatorResult",
    "run_benchmark_evaluator",
    "candidate_infos",
    "scan_referenced_names",
    "candidate_names_in_code",
    "accumulate",
    "decide",
    "render_report",
    "PlannedPR",
    "plan_pr",
    "apply_planned_changes",
    "create_library_pr",
    "stamp_docstring_date",
]
