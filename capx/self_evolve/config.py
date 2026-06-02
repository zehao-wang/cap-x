"""Central hyper-params for the self-evolve pipeline.

Shared contract — every tunable lives here so it can be adjusted in one place;
defaults mirror ``docs-se/config.md``. Each downstream module references its own
group: Experience Distill (module ③), Update Planner (④), Benchmark Evaluator (⑤).

Usage::

    from capx.self_evolve.config import SelfEvolveConfig

    cfg = SelfEvolveConfig()                 # all defaults
    cfg.experience_distill.max_words         # -> 200
    cfg = SelfEvolveConfig.from_dict({"update_planner": {"trigger_history_count": 8}})
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from typing import Any


@dataclass
class ExperienceDistillConfig:
    """Module ③ (Feedback Postprocessor / Experience Distill)."""

    # Word cap on the ``<id>.digest.md`` distilled before entering the pool;
    # forces a "small" sourced report (aligns with Agent Debugger's word budget).
    max_words: int = 200


@dataclass
class UpdatePlannerConfig:
    """Module ④ (Update Planner)."""

    # Unprocessed-history count that auto-triggers a planning run.
    trigger_history_count: int = 5
    # LLM-1 <-> LLM-2 revise rounds before a forced finalize into the candidate pool.
    max_revise_iterations: int = 5
    # Hard cap on History-Reader tool-loop iterations within a single LLM call
    # (read budget for digest/drill); independent of the revise cap.
    max_read_iterations: int = 20


@dataclass
class BenchmarkEvalConfig:
    """Module ⑤ (Benchmark Evaluator)."""

    # Minimum ``eval_runs`` required before a promote is suggested.
    min_samples: int = 5
    # ``positive`` count at which a candidate is suggested for promotion.
    promote_threshold: int = 10
    # ``eval_runs`` after which a still-unused (``used_runs==0``), unreferenced
    # candidate becomes an abandon candidate. Set high: a whole task set may
    # legitimately never touch a given candidate.
    max_idle_evals: int = 50
    # Schedule hint (nightly cron, fired once). The cron expression itself lives
    # with module ⑥; this records the intent for reference.
    schedule: str = "nightly"


@dataclass
class HeartbeatConfig:
    """Module ⑥ (Heartbeat / Cron) scheduling layer."""

    # Cron expression for the nightly Benchmark Evaluator (⑤) trigger. Default
    # 02:00 daily. The schedule *intent* is recorded in ``benchmark_eval.schedule``;
    # the concrete expression lives here (per docs-se/05 + 06).
    evaluator_cron: str = "0 2 * * *"
    # Cron expression for the daily-task heartbeat (hand low-accuracy tasks to the
    # human-in-the-loop). Default 09:00 daily.
    daily_task_cron: str = "0 9 * * *"
    # How many low-accuracy tasks the daily heartbeat hands to the interactive loop.
    daily_task_count: int = 3
    # A task counts as "low accuracy" (eligible for the daily heartbeat) when its
    # success rate is at or below this threshold.
    low_accuracy_threshold: float = 0.5


@dataclass
class SelfEvolveConfig:
    """Top-level config aggregating every pipeline group."""

    experience_distill: ExperienceDistillConfig = field(default_factory=ExperienceDistillConfig)
    update_planner: UpdatePlannerConfig = field(default_factory=UpdatePlannerConfig)
    benchmark_eval: BenchmarkEvalConfig = field(default_factory=BenchmarkEvalConfig)
    heartbeat: HeartbeatConfig = field(default_factory=HeartbeatConfig)

    _GROUPS = {
        "experience_distill": ExperienceDistillConfig,
        "update_planner": UpdatePlannerConfig,
        "benchmark_eval": BenchmarkEvalConfig,
        "heartbeat": HeartbeatConfig,
    }

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "SelfEvolveConfig":
        """Build a config from a (partial) nested dict, filling missing values
        with defaults. Unknown groups or keys raise ``ValueError`` so typos in a
        config file surface loudly instead of being silently ignored."""
        data = data or {}
        unknown_groups = set(data) - set(cls._GROUPS)
        if unknown_groups:
            raise ValueError(f"unknown config group(s): {sorted(unknown_groups)}")

        kwargs: dict[str, Any] = {}
        for name, group_cls in cls._GROUPS.items():
            group_data = data.get(name, {}) or {}
            valid = {f.name for f in fields(group_cls)}
            unknown_keys = set(group_data) - valid
            if unknown_keys:
                raise ValueError(f"unknown key(s) in '{name}': {sorted(unknown_keys)}")
            kwargs[name] = group_cls(**group_data)
        return cls(**kwargs)

    def to_dict(self) -> dict[str, Any]:
        """Nested dict of all groups, suitable for JSON/YAML serialization."""
        return {name: asdict(getattr(self, name)) for name in self._GROUPS}
