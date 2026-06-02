"""Tests for module ⑤ (Benchmark Evaluator) + approval-by-PR.

The sim run is a scripted ``eval_fn`` returning a canned :class:`EvalRoundResult`,
so stats accumulation, the abandon reference scan, the promote/abandon/keep
decision, the report, and the PR plan/apply all run with no sim and no git.
"""

from __future__ import annotations

import datetime as dt

from capx.self_evolve import MemStore
from capx.self_evolve.benchmark_eval import (
    ABANDON,
    KEEP,
    PROMOTE,
    CandidateInfo,
    EvalRoundResult,
    accumulate,
    candidate_names_in_code,
    decide,
    run_benchmark_evaluator,
    scan_referenced_names,
)
from capx.self_evolve.config import BenchmarkEvalConfig
from capx.self_evolve.library_pr import (
    apply_planned_changes,
    create_library_pr,
    plan_pr,
    stamp_docstring_date,
)
from capx.self_evolve.proposal import ProposalCandidate, write_accepted
from capx.self_evolve.schemas import CandidateStats

NOW = dt.datetime(2026, 5, 31, 22, 0, 0)


def _seed_candidate(store, name, code="def f():\n    pass\n", target="skill_library", history=("h__1",)):
    cand = ProposalCandidate(name, target, code, list(history))
    # seed a fake history id so write_accepted's source list is non-trivial
    return write_accepted(cand, store, now=NOW)


# --------------------------------------------------------------------------- #
# usage detection + reference scan
# --------------------------------------------------------------------------- #
def test_candidate_names_in_code():
    code = "def main():\n    grasp(obj)\n    other.thing()\n"
    assert candidate_names_in_code(code, {"grasp", "place"}) == {"grasp"}
    assert candidate_names_in_code(code, ["thing"]) == set()  # attribute, not a bare name


def test_scan_referenced_names():
    infos = [
        CandidateInfo("approach_and_grasp", "skill_library", "def approach_and_grasp(o):\n    grasp(o)\n"),
        CandidateInfo("grasp", "atomic_task_library", "def grasp(o):\n    pass\n"),
        CandidateInfo("lonely", "skill_library", "def lonely():\n    pass\n"),
    ]
    referenced = scan_referenced_names(infos)
    assert referenced == {"grasp"}  # grasp is called by approach_and_grasp; lonely by no one


def test_scan_ignores_self_reference():
    infos = [CandidateInfo("recurse", "skill_library", "def recurse(n):\n    return recurse(n - 1)\n")]
    assert scan_referenced_names(infos) == set()


# --------------------------------------------------------------------------- #
# accumulate + decide
# --------------------------------------------------------------------------- #
def test_accumulate_round():
    stats = CandidateStats("f", "skill_library")
    accumulate(stats, EvalRoundResult(positive_hits={"f": 3}, used={"f"}), now=NOW.date())
    assert (stats.eval_runs, stats.used_runs, stats.positive) == (1, 1, 3)
    accumulate(stats, EvalRoundResult(positive_hits={}, used=set()), now=NOW.date())
    assert (stats.eval_runs, stats.used_runs, stats.positive) == (2, 1, 3)  # unused round: only eval_runs++
    assert stats.last_eval_date == "2026-05-31"


def test_decide_promote():
    cfg = BenchmarkEvalConfig(promote_threshold=10, min_samples=5)
    stats = CandidateStats("f", "skill_library", positive=10, eval_runs=5, used_runs=5)
    kind, _ = decide(stats, set(), cfg)
    assert kind == PROMOTE


def test_decide_promote_needs_min_samples():
    cfg = BenchmarkEvalConfig(promote_threshold=10, min_samples=5)
    stats = CandidateStats("f", "skill_library", positive=10, eval_runs=4, used_runs=4)
    assert decide(stats, set(), cfg)[0] == KEEP  # not enough eval_runs yet


def test_decide_abandon():
    cfg = BenchmarkEvalConfig(max_idle_evals=50)
    stats = CandidateStats("f", "skill_library", positive=0, eval_runs=51, used_runs=0)
    assert decide(stats, set(), cfg)[0] == ABANDON


def test_decide_referenced_blocks_abandon():
    cfg = BenchmarkEvalConfig(max_idle_evals=50)
    stats = CandidateStats("f", "skill_library", positive=0, eval_runs=51, used_runs=0)
    kind, reason = decide(stats, {"f"}, cfg)
    assert kind == KEEP and "referenced" in reason


def test_decide_used_not_abandoned():
    cfg = BenchmarkEvalConfig(max_idle_evals=50)
    stats = CandidateStats("f", "skill_library", positive=2, eval_runs=51, used_runs=3)
    assert decide(stats, set(), cfg)[0] == KEEP  # used at least once -> never abandoned


# --------------------------------------------------------------------------- #
# orchestration
# --------------------------------------------------------------------------- #
def test_run_evaluator_empty_pool_is_noop(tmp_path):
    store = MemStore(tmp_path / "mem")
    result = run_benchmark_evaluator(store, lambda infos: EvalRoundResult())
    assert not result.triggered
    assert result.decisions == []


def test_run_evaluator_accumulates_and_decides(tmp_path):
    store = MemStore(tmp_path / "mem")
    _seed_candidate(store, "approach_and_grasp", "def approach_and_grasp(o):\n    grasp(o)\n")
    _seed_candidate(store, "grasp", "def grasp(o):\n    pass\n", target="atomic_task_library")
    _seed_candidate(store, "idle", "def idle():\n    pass\n")

    # Pre-age 'idle' so one more round crosses max_idle_evals while still unused.
    s = store.read_candidate_stats("idle")
    s.eval_runs = 50
    store.update_candidate_stats(s)

    def eval_fn(infos):
        # 'approach_and_grasp' lands in 12 final-success programs; 'grasp' only ever
        # used via approach_and_grasp's reference (not called directly this round).
        return EvalRoundResult(positive_hits={"approach_and_grasp": 12}, used={"approach_and_grasp"})

    cfg = BenchmarkEvalConfig(promote_threshold=10, min_samples=1, max_idle_evals=50)
    result = run_benchmark_evaluator(store, eval_fn, config=cfg, now=NOW)

    assert result.triggered
    by = {d.func_name: d for d in result.decisions}
    assert by["approach_and_grasp"].decision == PROMOTE
    assert by["grasp"].decision == KEEP  # unused but referenced by approach_and_grasp
    assert by["idle"].decision == ABANDON  # eval_runs 50->51, never used, unreferenced

    # stats persisted in place
    assert store.read_candidate_stats("approach_and_grasp").positive == 12
    assert store.read_candidate_stats("idle").eval_runs == 51
    # report mentions the headline counts
    assert "1 promote" in result.report and "1 abandon" in result.report
    assert "approval" not in result.report.lower() or "merge" in result.report.lower()


# --------------------------------------------------------------------------- #
# docstring date-stamp
# --------------------------------------------------------------------------- #
def test_stamp_docstring_none():
    out = stamp_docstring_date("def f(x):\n    return x\n", "2026-05-31")
    assert '"""Updated: 2026-05-31"""' in out
    compile(out, "<t>", "exec")  # still valid python


def test_stamp_docstring_single_line():
    out = stamp_docstring_date('def f():\n    """Grasp."""\n    pass\n', "2026-05-31")
    assert "Grasp." in out and "Updated: 2026-05-31" in out
    compile(out, "<t>", "exec")


def test_stamp_docstring_multiline():
    code = 'def f():\n    """Grasp.\n\n    Long desc.\n    """\n    pass\n'
    out = stamp_docstring_date(code, "2026-05-31")
    assert "Long desc." in out and "Updated: 2026-05-31" in out
    compile(out, "<t>", "exec")


def test_stamp_docstring_no_function_unchanged():
    code = "x = 1\n"
    assert stamp_docstring_date(code, "2026-05-31") == code


# --------------------------------------------------------------------------- #
# PR plan + apply
# --------------------------------------------------------------------------- #
def test_plan_pr_promote_and_abandon(tmp_path):
    store = MemStore(tmp_path / "mem")
    _seed_candidate(store, "approach_and_grasp", "def approach_and_grasp(o):\n    return o\n")
    _seed_candidate(store, "idle", "def idle():\n    pass\n")

    s = store.read_candidate_stats("approach_and_grasp")
    s.positive, s.eval_runs, s.used_runs = 12, 6, 6
    store.update_candidate_stats(s)
    s = store.read_candidate_stats("idle")
    s.eval_runs, s.used_runs = 60, 0
    store.update_candidate_stats(s)

    result = run_benchmark_evaluator(
        store, lambda infos: EvalRoundResult(),
        config=BenchmarkEvalConfig(promote_threshold=10, min_samples=1, max_idle_evals=50),
        now=NOW,
    )
    planned = plan_pr(result, now=NOW)

    promote_rel = "capx/skill_library/approach_and_grasp.py"
    assert promote_rel in planned.writes
    assert "Updated: 2026-05-31" in planned.writes[promote_rel]
    assert "capx/skill_library/__init__.py" in planned.writes
    assert "mem/func_candidate_pool/idle.py" in planned.deletes
    assert "mem/func_candidate_pool/idle.stats.json" in planned.deletes
    assert planned.branch.startswith("se/library-update-")
    assert result.report == planned.pr_body
    assert not planned.is_empty


def test_plan_pr_empty_when_all_keep(tmp_path):
    store = MemStore(tmp_path / "mem")
    _seed_candidate(store, "f")
    result = run_benchmark_evaluator(
        store, lambda infos: EvalRoundResult(used={"f"}),
        config=BenchmarkEvalConfig(promote_threshold=10, min_samples=5),
        now=NOW,
    )
    planned = plan_pr(result, now=NOW)
    assert planned.is_empty


def test_apply_planned_changes(tmp_path):
    store = MemStore(tmp_path / "mem")
    _seed_candidate(store, "winner", "def winner():\n    return 1\n")
    _seed_candidate(store, "loser", "def loser():\n    pass\n")
    s = store.read_candidate_stats("winner")
    s.positive, s.eval_runs, s.used_runs = 12, 6, 6
    store.update_candidate_stats(s)
    s = store.read_candidate_stats("loser")
    s.eval_runs = 60
    store.update_candidate_stats(s)

    result = run_benchmark_evaluator(
        store, lambda infos: EvalRoundResult(),
        config=BenchmarkEvalConfig(promote_threshold=10, min_samples=1, max_idle_evals=50),
        now=NOW,
    )
    planned = plan_pr(result, now=NOW, mem_reldir="mem/func_candidate_pool")

    # apply against a worktree rooted at tmp_path (mem/ lives there too)
    written, deleted = apply_planned_changes(planned, tmp_path)
    assert (tmp_path / "capx/skill_library/winner.py").exists()
    assert not (tmp_path / "mem/func_candidate_pool/loser.py").exists()
    assert "capx/skill_library/winner.py" in written
    assert "mem/func_candidate_pool/loser.py" in deleted


# --------------------------------------------------------------------------- #
# live PR mechanism (mocked runner — no real git/gh)
# --------------------------------------------------------------------------- #
def test_create_library_pr_sequences_git_then_gh(tmp_path):
    store = MemStore(tmp_path / "mem")
    _seed_candidate(store, "winner", "def winner():\n    return 1\n")
    s = store.read_candidate_stats("winner")
    s.positive, s.eval_runs, s.used_runs = 12, 6, 6
    store.update_candidate_stats(s)
    result = run_benchmark_evaluator(
        store, lambda infos: EvalRoundResult(),
        config=BenchmarkEvalConfig(promote_threshold=10, min_samples=1),
        now=NOW,
    )
    planned = plan_pr(result, now=NOW)

    calls = []

    def fake_runner(cmd, cwd):
        calls.append(cmd)

    summary = create_library_pr(planned, repo_root=tmp_path, runner=fake_runner)
    verbs = [c[0] for c in calls]
    assert verbs == ["git", "git", "git", "git", "gh"]
    assert calls[0][:3] == ["git", "checkout", "-b"]  # branch first, never base
    assert "winner.py" in " ".join(str(p) for c in calls for p in c) or summary["written"]
    assert summary["branch"] == planned.branch
    assert summary["pushed"] and summary["pr_opened"]
    # files actually applied on the branch worktree
    assert (tmp_path / "capx/skill_library/winner.py").exists()


def test_create_library_pr_empty_raises(tmp_path):
    from capx.self_evolve.library_pr import PlannedPR
    import pytest

    with pytest.raises(ValueError):
        create_library_pr(PlannedPR(branch="x"), runner=lambda cmd, cwd: None)
