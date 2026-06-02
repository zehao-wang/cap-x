"""Module ⑤: Benchmark Evaluator — short-term → long-term promotion gate.

Triggered nightly (scheduled by module ⑥) or woken manually; runs only when
``func_candidate_pool`` is non-empty. It injects the candidate functions as
*optional* tools (see ``docs-se/integration.md``) and runs the benchmark in sim
with **VDM auto-judging success** — that sim run is abstracted behind an
injected ``eval_fn`` (mirroring how module ④ injects ``query_fn``), so all the
deterministic logic here is exercisable without a live sim:

- per-candidate **stats accumulation** (``eval_runs`` / ``used_runs`` /
  ``positive``; we never record negatives, by design);
- the abandon **reference scan** (does any *other* living candidate call it?);
- the **promote / abandon / keep** decision;
- the **report**.

Hard constraint (``docs-se/05-benchmark-evaluator.md``): the Evaluator NEVER
directly mutates a long-term library or deletes a candidate. It only accumulates
stats in place (explicitly allowed — stats follow the candidate) and returns a
*proposal*: a report plus the promote/abandon decisions. Turning that into
reality goes through a PR the user merges — see :mod:`capx.self_evolve.library_pr`.
"""

from __future__ import annotations

import ast
import datetime as _dt
import logging
from dataclasses import dataclass, field
from typing import Callable

from capx.self_evolve.config import BenchmarkEvalConfig
from capx.self_evolve.schemas import CandidateStats
from capx.self_evolve.storage import MemStore

logger = logging.getLogger(__name__)

PROMOTE, ABANDON, KEEP = "promote", "abandon", "keep"


# --------------------------------------------------------------------------- #
# data
# --------------------------------------------------------------------------- #
@dataclass
class CandidateInfo:
    """A candidate handed to ``eval_fn`` for injection as an optional tool."""

    func_name: str
    target_library: str
    code: str


@dataclass
class EvalRoundResult:
    """What one benchmark run (over the dataset) observed about each candidate.

    - ``positive_hits[name]``: number of tasks whose **final successful** code
      used the candidate this round (VDM-judged). Each adds to ``positive``.
    - ``used``: candidates invoked *anywhere* this round (chat exploration or a
      final answer). Membership adds 1 to ``used_runs`` (per round, not per task).

    We deliberately carry no negative signal: a candidate tried and dropped is
    not penalised.
    """

    positive_hits: dict[str, int] = field(default_factory=dict)
    used: set[str] = field(default_factory=set)


# ``eval_fn(candidates) -> EvalRoundResult``: run the benchmark in sim with the
# candidates injected as optional tools and VDM judging success. The live wiring
# lives at the call site (sim-gated); the orchestration below only needs this.
EvalFn = Callable[[list[CandidateInfo]], EvalRoundResult]


@dataclass
class CandidateDecision:
    func_name: str
    target_library: str
    decision: str  # PROMOTE | ABANDON | KEEP
    reason: str
    stats: CandidateStats
    code: str = ""  # candidate source — carried so a promote can be written to a PR


@dataclass
class EvaluatorResult:
    triggered: bool
    decisions: list[CandidateDecision] = field(default_factory=list)
    report: str = ""

    def _of(self, kind: str) -> list[CandidateDecision]:
        return [d for d in self.decisions if d.decision == kind]

    @property
    def promotions(self) -> list[CandidateDecision]:
        return self._of(PROMOTE)

    @property
    def abandons(self) -> list[CandidateDecision]:
        return self._of(ABANDON)

    @property
    def keeps(self) -> list[CandidateDecision]:
        return self._of(KEEP)


# --------------------------------------------------------------------------- #
# reference scan + usage detection (shared, pure)
# --------------------------------------------------------------------------- #
def _names_in_code(code: str) -> set[str]:
    """All bare identifiers referenced in a snippet (best-effort via ``ast``).

    Used both for the abandon reference scan and for detecting which candidates
    a final answer called. Attribute access (``obj.foo``) is intentionally not
    treated as a free name — candidates are top-level functions, called bare.
    """
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return set()
    return {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}


def candidate_names_in_code(code: str, names: set[str] | list[str]) -> set[str]:
    """Which of ``names`` appear as bare references in ``code``.

    Helper for a live ``eval_fn`` to compute ``positive_hits`` / ``used`` from a
    trial's final/executed code without re-implementing the parse.
    """
    return _names_in_code(code) & set(names)


def scan_referenced_names(infos: list[CandidateInfo]) -> set[str]:
    """Candidate names that some *other* living candidate's source references.

    Computed fresh from the current pool snapshot (not stored), per the design:
    a candidate referenced by another living candidate must not be abandoned even
    if it is itself unused, or the referrer would break.
    """
    names = {i.func_name for i in infos}
    referenced: set[str] = set()
    for info in infos:
        used = _names_in_code(info.code) - {info.func_name}
        referenced |= used & names
    return referenced


# --------------------------------------------------------------------------- #
# stats + decision
# --------------------------------------------------------------------------- #
def accumulate(stats: CandidateStats, result: EvalRoundResult, *, now: _dt.date) -> CandidateStats:
    """Fold one eval round into a candidate's running stats (mutates + returns).

    Every round in the pool is one ``eval_runs``; being used anywhere this round
    is one ``used_runs``; each task whose final success code used it adds to
    ``positive``. Negatives are never recorded.
    """
    stats.eval_runs += 1
    if stats.func_name in result.used:
        stats.used_runs += 1
    stats.positive += max(0, int(result.positive_hits.get(stats.func_name, 0)))
    stats.last_eval_date = now.strftime("%Y-%m-%d")
    return stats


def decide(
    stats: CandidateStats,
    referenced: set[str],
    config: BenchmarkEvalConfig,
) -> tuple[str, str]:
    """Decide promote / abandon / keep for one candidate (with a reason string).

    Priority: promote before abandon (a promotable candidate has been used, so it
    can never also be an abandon candidate, but we check promote first regardless).
    """
    if stats.positive >= config.promote_threshold and stats.eval_runs >= config.min_samples:
        return PROMOTE, (
            f"positive={stats.positive} ≥ promote_threshold={config.promote_threshold} "
            f"and eval_runs={stats.eval_runs} ≥ min_samples={config.min_samples}"
        )
    if (
        stats.eval_runs > config.max_idle_evals
        and stats.used_runs == 0
        and stats.func_name not in referenced
    ):
        return ABANDON, (
            f"unused (used_runs=0) through eval_runs={stats.eval_runs} "
            f"> max_idle_evals={config.max_idle_evals}, and no living candidate references it"
        )
    if stats.func_name in referenced and stats.used_runs == 0:
        return KEEP, "unused but referenced by another living candidate"
    return KEEP, (
        f"positive={stats.positive} (< {config.promote_threshold}) — kept for more evidence"
    )


# --------------------------------------------------------------------------- #
# orchestration
# --------------------------------------------------------------------------- #
def candidate_infos(store: MemStore) -> list[CandidateInfo]:
    """Load every candidate currently in the pool (name, target library, code)."""
    infos: list[CandidateInfo] = []
    for name in store.list_candidate_names():
        stats = store.read_candidate_stats(name)
        infos.append(CandidateInfo(name, stats.target_library, store.read_candidate_code(name)))
    return infos


def run_benchmark_evaluator(
    store: MemStore,
    eval_fn: EvalFn,
    *,
    config: BenchmarkEvalConfig | None = None,
    now: _dt.datetime | None = None,
) -> EvaluatorResult:
    """Run one Benchmark Evaluator pass over the current candidate pool.

    No-op (``triggered=False``) when the pool is empty. Otherwise: run one eval
    round via ``eval_fn``, fold the outcome into each candidate's stats **in place**
    (the one allowed mutation), scan references, decide, and return the report +
    decisions. Promotions/abandons are only *proposed* here — applying them is the
    PR step in :mod:`capx.self_evolve.library_pr`.
    """
    config = config or BenchmarkEvalConfig()
    now = now or _dt.datetime.now()

    infos = candidate_infos(store)
    if not infos:
        logger.info("benchmark evaluator: candidate pool empty, nothing to do")
        return EvaluatorResult(triggered=False, report="candidate pool empty; nothing to evaluate.\n")

    result = eval_fn(infos)
    referenced = scan_referenced_names(infos)

    decisions: list[CandidateDecision] = []
    for info in infos:
        stats = store.read_candidate_stats(info.func_name)
        accumulate(stats, result, now=now.date())
        store.update_candidate_stats(stats)  # allowed: stats follow the candidate
        kind, reason = decide(stats, referenced, config)
        decisions.append(
            CandidateDecision(info.func_name, info.target_library, kind, reason, stats, info.code)
        )

    report = render_report(decisions, now=now)
    logger.info(
        "benchmark evaluator: %d candidate(s) -> promote=%d abandon=%d keep=%d",
        len(decisions),
        sum(d.decision == PROMOTE for d in decisions),
        sum(d.decision == ABANDON for d in decisions),
        sum(d.decision == KEEP for d in decisions),
    )
    return EvaluatorResult(triggered=True, decisions=decisions, report=report)


# --------------------------------------------------------------------------- #
# report
# --------------------------------------------------------------------------- #
def render_report(decisions: list[CandidateDecision], *, now: _dt.datetime) -> str:
    """A small markdown report: what the Evaluator proposes, and why.

    The report is the proposal the user approves: it becomes the body of the PR
    (:mod:`capx.self_evolve.library_pr`). Nothing here is applied — every promote
    (write to a long-term library) and abandon (delete a candidate) takes effect
    only when the user MERGES that PR.
    """
    n_prom = sum(d.decision == PROMOTE for d in decisions)
    n_aban = sum(d.decision == ABANDON for d in decisions)
    n_keep = sum(d.decision == KEEP for d in decisions)
    lines = [
        f"# Benchmark Evaluator report · {now:%Y-%m-%d %H:%M:%S}",
        "",
        f"Evaluated {len(decisions)} candidate(s): "
        f"**{n_prom} promote**, **{n_aban} abandon**, {n_keep} keep.",
        "",
        "> ⚠️ Proposal only. Every promote/abandon below takes effect **only after "
        "you merge the PR** (close = reject). The Evaluator does not touch the "
        "long-term libraries or delete candidates on its own.",
        "",
    ]

    def block(title: str, kind: str) -> None:
        chosen = [d for d in decisions if d.decision == kind]
        lines.append(f"## {title} ({len(chosen)})")
        if not chosen:
            lines.append("_none_")
            lines.append("")
            return
        for d in chosen:
            s = d.stats
            target = f" → `capx/{d.target_library}/`" if kind == PROMOTE else ""
            lines.append(f"- **{d.func_name}**{target}")
            lines.append(
                f"  - stats: positive={s.positive}, used_runs={s.used_runs}, "
                f"eval_runs={s.eval_runs}"
            )
            lines.append(f"  - {d.reason}")
        lines.append("")

    block("Promote", PROMOTE)
    block("Abandon", ABANDON)
    block("Keep", KEEP)
    return "\n".join(lines).rstrip() + "\n"
