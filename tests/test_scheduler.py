"""Tests for module ⑥ (Heartbeat / Cron).

The cron matcher, low-accuracy selection, and tick de-duplication are pure and
fully tested; the live callbacks (sim eval, interactive loop) are scripted.
"""

from __future__ import annotations

import datetime as dt

import pytest

from capx.self_evolve import MemStore
from capx.self_evolve.benchmark_eval import EvalRoundResult
from capx.self_evolve.config import BenchmarkEvalConfig, HeartbeatConfig
from capx.self_evolve.proposal import ProposalCandidate, write_accepted
from capx.self_evolve.scheduler import (
    CronSpec,
    Heartbeat,
    HeartbeatJob,
    make_daily_task_job,
    make_evaluator_job,
    select_daily_tasks,
)


# --------------------------------------------------------------------------- #
# cron matcher
# --------------------------------------------------------------------------- #
def test_cron_nightly_match():
    spec = CronSpec.parse("0 2 * * *")
    assert spec.matches(dt.datetime(2026, 5, 31, 2, 0))
    assert not spec.matches(dt.datetime(2026, 5, 31, 2, 1))
    assert not spec.matches(dt.datetime(2026, 5, 31, 3, 0))


def test_cron_steps_lists_ranges():
    assert CronSpec.parse("*/15 * * * *").minute == {0, 15, 30, 45}
    assert CronSpec.parse("0 9-11 * * *").hour == {9, 10, 11}
    assert CronSpec.parse("0 0 1,15 * *").dom == {1, 15}


def test_cron_dow_sunday_is_zero():
    spec = CronSpec.parse("0 0 * * 0")  # Sundays
    assert spec.matches(dt.datetime(2026, 6, 7, 0, 0))  # 2026-06-07 is a Sunday
    assert not spec.matches(dt.datetime(2026, 6, 8, 0, 0))  # Monday


def test_cron_invalid_raises():
    with pytest.raises(ValueError):
        CronSpec.parse("0 2 * *")  # 4 fields
    with pytest.raises(ValueError):
        CronSpec.parse("99 2 * * *")  # minute out of range
    with pytest.raises(ValueError):
        CronSpec.parse("*/0 * * * *")  # zero step


# --------------------------------------------------------------------------- #
# low-accuracy selection
# --------------------------------------------------------------------------- #
def test_select_daily_tasks_picks_worst_below_threshold():
    acc = {"a": 0.1, "b": 0.9, "c": 0.4, "d": 0.5, "e": 0.55}
    # threshold 0.5 -> {a,c,d}; worst-2 ascending = [a, c]
    assert select_daily_tasks(acc, count=2, max_accuracy=0.5) == ["a", "c"]


def test_select_daily_tasks_stable_tiebreak_by_name():
    acc = {"z": 0.2, "a": 0.2, "m": 0.2}
    assert select_daily_tasks(acc, count=3, max_accuracy=0.5) == ["a", "m", "z"]


def test_select_daily_tasks_empty_when_all_above():
    assert select_daily_tasks({"a": 0.9}, count=3, max_accuracy=0.5) == []


# --------------------------------------------------------------------------- #
# heartbeat tick
# --------------------------------------------------------------------------- #
def test_tick_fires_due_job_once_per_minute():
    fired_log = []
    hb = Heartbeat().add(HeartbeatJob("nightly", CronSpec.parse("0 2 * * *"), lambda now: fired_log.append(now)))

    at_2 = dt.datetime(2026, 5, 31, 2, 0, 0)
    assert [n for n, _ in hb.tick(at_2)] == ["nightly"]
    # same minute again -> de-duplicated
    assert hb.tick(at_2.replace(second=30)) == []
    # not due at a different time
    assert hb.tick(dt.datetime(2026, 5, 31, 2, 1)) == []
    # due again next day
    assert [n for n, _ in hb.tick(dt.datetime(2026, 6, 1, 2, 0))] == ["nightly"]
    assert len(fired_log) == 2


def test_tick_isolates_failing_job():
    def boom(now):
        raise RuntimeError("nope")

    hb = Heartbeat()
    hb.add(HeartbeatJob("bad", CronSpec.parse("* * * * *"), boom))
    hb.add(HeartbeatJob("good", CronSpec.parse("* * * * *"), lambda now: "ok"))
    results = dict(hb.tick(dt.datetime(2026, 5, 31, 2, 0)))
    assert isinstance(results["bad"], RuntimeError)
    assert results["good"] == "ok"


def test_run_forever_bounded_smoke():
    calls = {"n": 0}
    hb = Heartbeat().add(HeartbeatJob("j", CronSpec.parse("* * * * *"), lambda now: calls.__setitem__("n", calls["n"] + 1)))
    hb.run_forever(sleep=lambda s: None, _max_ticks=3)
    # fired at most once per distinct minute; at least once
    assert calls["n"] >= 1


# --------------------------------------------------------------------------- #
# job factories
# --------------------------------------------------------------------------- #
def _seed_candidate(store, name="winner"):
    return write_accepted(
        ProposalCandidate(name, "skill_library", f"def {name}():\n    pass\n", ["h__1"]),
        store, now=dt.datetime(2026, 5, 31),
    )


def test_evaluator_job_runs_on_pool_and_noops_when_empty():
    import tempfile, pathlib
    tmp = pathlib.Path(tempfile.mkdtemp())
    store = MemStore(tmp / "mem")

    eval_calls = {"n": 0}

    def eval_fn(infos):
        eval_calls["n"] += 1
        return EvalRoundResult(positive_hits={"winner": 1}, used={"winner"})

    seen_results = []
    job = make_evaluator_job(
        store, eval_fn, cron="0 2 * * *",
        eval_config=BenchmarkEvalConfig(promote_threshold=10, min_samples=5),
        on_result=seen_results.append,
    )
    now = dt.datetime(2026, 5, 31, 2, 0)

    # empty pool -> evaluator no-ops, eval_fn never called, on_result not invoked
    res_empty = job.action(now)
    assert not res_empty.triggered
    assert eval_calls["n"] == 0 and seen_results == []

    # non-empty pool -> eval_fn runs, stats accumulate, on_result fires
    _seed_candidate(store, "winner")
    res = job.action(now)
    assert res.triggered and eval_calls["n"] == 1
    assert seen_results == [res]
    assert store.read_candidate_stats("winner").positive == 1


def test_daily_task_job_dispatches_low_accuracy_tasks():
    dispatched = []
    job = make_daily_task_job(
        accuracy_provider=lambda: {"easy": 0.95, "hard": 0.1, "mid": 0.45},
        run_interactive_task=dispatched.append,
        cron="0 9 * * *",
        config=HeartbeatConfig(daily_task_count=5, low_accuracy_threshold=0.5),
    )
    out = job.action(dt.datetime(2026, 5, 31, 9, 0))
    assert out == ["hard", "mid"]  # easy filtered out, worst-first
    assert dispatched == ["hard", "mid"]
