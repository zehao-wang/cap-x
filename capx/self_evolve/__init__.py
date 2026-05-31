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
]
