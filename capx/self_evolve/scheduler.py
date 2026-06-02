"""Module ⑥: Heartbeat / Cron — the scheduling layer that threads the pipeline.

Two roles (``docs-se/06-heartbeat-cron.md``):

- **Heartbeat (daily task)**: periodically pull *low-accuracy* tasks back to the
  human-in-the-loop (module ①), so new human-confirmed successes flow into the
  ``history_pool`` and drive library updates.
- **Cron**: fire jobs at fixed times — typically the **nightly Benchmark
  Evaluator** (module ⑤), which turns vetted candidates into a library-update PR.

Following the rest of the pipeline, the *live* parts are injected behind
callbacks (running a sim eval, waking the interactive loop, the long-lived sleep
of ``run_forever``), while the deterministic scheduling logic is unit-testable:

- :class:`CronSpec` — a small 5-field cron matcher;
- :func:`select_daily_tasks` — which low-accuracy tasks to hand back;
- :class:`Heartbeat` — owns jobs and fires the due ones on each :meth:`tick`,
  de-duplicating within a minute.
"""

from __future__ import annotations

import datetime as _dt
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable

from capx.self_evolve.benchmark_eval import EvalFn, EvaluatorResult, run_benchmark_evaluator
from capx.self_evolve.config import BenchmarkEvalConfig, HeartbeatConfig
from capx.self_evolve.storage import MemStore

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# cron matcher (minute hour day-of-month month day-of-week)
# --------------------------------------------------------------------------- #
def _parse_field(field_str: str, lo: int, hi: int) -> set[int]:
    """Parse one cron field into the set of allowed ints in ``[lo, hi]``.

    Supports ``*``, integer, ``a,b`` lists, ``a-b`` ranges, and ``*/n`` / ``a-b/n``
    steps — enough for the fixed daily/nightly triggers this layer needs.
    """
    allowed: set[int] = set()
    for part in field_str.split(","):
        part = part.strip()
        if not part:
            continue
        step = 1
        rng = part
        if "/" in part:
            rng, step_s = part.split("/", 1)
            step = int(step_s)
            if step <= 0:
                raise ValueError(f"cron step must be positive: {part!r}")
        if rng == "*":
            start, end = lo, hi
        elif "-" in rng:
            a, b = rng.split("-", 1)
            start, end = int(a), int(b)
        else:
            start = end = int(rng)
        if start < lo or end > hi or start > end:
            raise ValueError(f"cron field {part!r} out of range [{lo},{hi}]")
        allowed.update(range(start, end + 1, step))
    if not allowed:
        raise ValueError(f"empty cron field: {field_str!r}")
    return allowed


@dataclass
class CronSpec:
    """A parsed 5-field cron expression with a ``matches(dt)`` predicate.

    Fields: ``minute hour day-of-month month day-of-week`` (cron dow: Sun=0..Sat=6).
    Simplification vs. Vixie cron: when both day-of-month and day-of-week are
    restricted we AND them (not OR). Our triggers leave both as ``*``, so this
    never bites; it is documented rather than hidden.
    """

    minute: set[int]
    hour: set[int]
    dom: set[int]
    month: set[int]
    dow: set[int]
    expr: str = ""

    @classmethod
    def parse(cls, expr: str) -> "CronSpec":
        fields = expr.split()
        if len(fields) != 5:
            raise ValueError(f"cron expression must have 5 fields, got {len(fields)}: {expr!r}")
        m, h, dom, mon, dow = fields
        return cls(
            minute=_parse_field(m, 0, 59),
            hour=_parse_field(h, 0, 23),
            dom=_parse_field(dom, 1, 31),
            month=_parse_field(mon, 1, 12),
            dow=_parse_field(dow, 0, 6),
            expr=expr,
        )

    def matches(self, when: _dt.datetime) -> bool:
        cron_dow = (when.weekday() + 1) % 7  # python Mon=0..Sun=6 -> cron Sun=0..Sat=6
        return (
            when.minute in self.minute
            and when.hour in self.hour
            and when.day in self.dom
            and when.month in self.month
            and cron_dow in self.dow
        )


# --------------------------------------------------------------------------- #
# low-accuracy task selection (heartbeat daily task)
# --------------------------------------------------------------------------- #
def select_daily_tasks(
    accuracies: dict[str, float],
    *,
    count: int,
    max_accuracy: float,
) -> list[str]:
    """Pick the ``count`` lowest-accuracy tasks at or below ``max_accuracy``.

    Sorted ascending by accuracy, then by name for a stable, deterministic order
    (so repeated heartbeats with unchanged scores hand back the same worst tasks).
    """
    eligible = [(acc, task) for task, acc in accuracies.items() if acc <= max_accuracy]
    eligible.sort(key=lambda pair: (pair[0], pair[1]))
    return [task for _, task in eligible[: max(0, count)]]


# --------------------------------------------------------------------------- #
# jobs + heartbeat scheduler
# --------------------------------------------------------------------------- #
@dataclass
class HeartbeatJob:
    """A named action fired when its cron matches the wake time."""

    name: str
    cron: CronSpec
    action: Callable[[_dt.datetime], Any]


@dataclass
class Heartbeat:
    """Owns jobs and fires the due ones on each :meth:`tick`.

    State is just the last minute each job fired, so a tick is idempotent within
    a clock minute (calling tick twice in the same minute fires each job once).
    The driver (:meth:`run_forever`) is a thin live wrapper around :meth:`tick`.
    """

    jobs: list[HeartbeatJob] = field(default_factory=list)
    _last_fired: dict[str, str] = field(default_factory=dict)

    def add(self, job: HeartbeatJob) -> "Heartbeat":
        self.jobs.append(job)
        return self

    def tick(self, now: _dt.datetime | None = None) -> list[tuple[str, Any]]:
        """Fire every job whose cron matches ``now`` and hasn't fired this minute.

        Returns ``(job_name, action_result)`` for each job fired. A failing action
        is logged and recorded as the raised exception rather than aborting the
        tick, so one broken job can't starve the others.
        """
        now = now or _dt.datetime.now()
        minute_key = now.strftime("%Y%m%d-%H%M")
        fired: list[tuple[str, Any]] = []
        for job in self.jobs:
            if not job.cron.matches(now):
                continue
            if self._last_fired.get(job.name) == minute_key:
                continue
            self._last_fired[job.name] = minute_key
            try:
                result = job.action(now)
            except Exception as exc:  # noqa: BLE001 - one job must not starve others
                logger.exception("heartbeat job %r failed", job.name)
                result = exc
            fired.append((job.name, result))
        return fired

    def run_forever(self, *, sleep: Callable[[float], None] = time.sleep, _max_ticks: int | None = None) -> None:
        """Live driver: tick once per clock minute, forever.

        Side-effecting and not unit-tested (it sleeps); ``tick`` carries all the
        logic. ``_max_ticks`` bounds the loop for a smoke test.
        """
        ticks = 0
        while _max_ticks is None or ticks < _max_ticks:
            self.tick()
            ticks += 1
            now = _dt.datetime.now()
            sleep(60 - now.second - now.microsecond / 1e6)


# --------------------------------------------------------------------------- #
# job factories (wire the live callbacks)
# --------------------------------------------------------------------------- #
def make_evaluator_job(
    store: MemStore,
    eval_fn: EvalFn,
    *,
    name: str = "benchmark_evaluator",
    cron: str | None = None,
    config: HeartbeatConfig | None = None,
    eval_config: BenchmarkEvalConfig | None = None,
    on_result: Callable[[EvaluatorResult], Any] | None = None,
) -> HeartbeatJob:
    """Nightly cron job: run the Benchmark Evaluator (⑤) over the candidate pool.

    ``run_benchmark_evaluator`` already no-ops on an empty pool, so the cron may
    fire harmlessly when there is nothing to evaluate. ``on_result`` is an optional
    hook (e.g. open the approval PR via :mod:`capx.self_evolve.library_pr`) — kept
    injected because turning a result into a PR is a live, gated action.
    """
    config = config or HeartbeatConfig()
    spec = CronSpec.parse(cron or config.evaluator_cron)

    def action(_now: _dt.datetime) -> EvaluatorResult:
        result = run_benchmark_evaluator(store, eval_fn, config=eval_config, now=_now)
        if result.triggered and on_result is not None:
            on_result(result)
        return result

    return HeartbeatJob(name=name, cron=spec, action=action)


def make_daily_task_job(
    accuracy_provider: Callable[[], dict[str, float]],
    run_interactive_task: Callable[[str], Any],
    *,
    name: str = "daily_task",
    cron: str | None = None,
    config: HeartbeatConfig | None = None,
) -> HeartbeatJob:
    """Daily heartbeat: hand the worst low-accuracy tasks back to the human loop.

    ``accuracy_provider`` yields current per-task success rates (its source —
    benchmark logs — is live and injected); ``run_interactive_task`` wakes module
    ① for one task. Returns the list of tasks dispatched this fire.
    """
    config = config or HeartbeatConfig()
    spec = CronSpec.parse(cron or config.daily_task_cron)

    def action(_now: _dt.datetime) -> list[str]:
        tasks = select_daily_tasks(
            accuracy_provider(),
            count=config.daily_task_count,
            max_accuracy=config.low_accuracy_threshold,
        )
        for task in tasks:
            run_interactive_task(task)
        logger.info("daily-task heartbeat dispatched %d task(s): %s", len(tasks), tasks)
        return tasks

    return HeartbeatJob(name=name, cron=spec, action=action)
