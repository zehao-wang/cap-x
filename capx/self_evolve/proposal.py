"""Module ④: candidate proposal contract + mechanical review guardrails.

The ``propose(...)`` terminal tool and LLM-2's finalize both speak the same JSON
contract (``docs-se/04-update-planner.md``). This module gives that contract a
typed form, the *mechanical* review checks LLM-2 must not skip (real source ids,
valid identifier, granularity ceiling, target library, dedup), and the persister
that writes an accepted candidate into ``func_candidate_pool`` with its stats.

The mechanical checks are deterministic and unit-testable; they back LLM-2's
judgement (which additionally reasons about generality / bugs / true reuse) but
do not replace it.
"""

from __future__ import annotations

import datetime as _dt
import re
from dataclasses import dataclass, field
from typing import Any

from capx.self_evolve.schemas import VALID_TARGET_LIBRARIES, CandidateStats
from capx.self_evolve.storage import MemStore

_IDENT_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")

# Verbs that signal a generic atomic task (allowed). Object-bound names like
# ``put_apple`` / ``stack_red_block`` are rejected by the granularity rule.
_GENERIC_VERBS = {
    "pick", "place", "grasp", "release", "push", "pull", "approach", "lift",
    "move", "reach", "insert", "align", "rotate", "open", "close", "press",
    "pour", "wipe", "screw", "unscrew", "slide", "retreat", "home",
}
# Common object/color words; a function name embedding these (beyond a generic
# arg) is too task-specific to be an atomic task.
_OBJECT_HINTS = {
    "apple", "banana", "cube", "block", "mug", "cup", "bottle", "can", "bowl",
    "plate", "pot", "lid", "drawer", "door", "nut", "peg", "bread", "milk",
    "red", "green", "blue", "yellow", "square", "round",
}


@dataclass
class ProposalCandidate:
    func_name: str
    target_library: str
    code: str
    source_history: list[str] = field(default_factory=list)
    rationale: str = ""

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ProposalCandidate":
        missing = {"func_name", "target_library", "code"} - set(data)
        if missing:
            raise ValueError(f"candidate missing field(s): {sorted(missing)}")
        return cls(
            func_name=data["func_name"],
            target_library=data["target_library"],
            code=data["code"],
            source_history=list(data.get("source_history", [])),
            rationale=data.get("rationale", ""),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "func_name": self.func_name,
            "target_library": self.target_library,
            "code": self.code,
            "source_history": self.source_history,
            "rationale": self.rationale,
        }


@dataclass
class Proposal:
    """The ``propose`` payload: a (possibly empty) batch of candidates."""

    candidates: list[ProposalCandidate] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Proposal":
        return cls(candidates=[ProposalCandidate.from_dict(c) for c in data.get("candidates", [])])

    def to_dict(self) -> dict[str, Any]:
        return {"candidates": [c.to_dict() for c in self.candidates]}


def is_object_bound_name(func_name: str) -> bool:
    """Heuristic granularity check: True if the name binds a specific object/task.

    Atomic tasks must not bind a concrete object/task (``put_apple`` rejected;
    ``pick`` / ``place`` allowed). We flag a name when any of its underscore /
    camelCase tokens is a known object/color hint.
    """
    tokens = re.split(r"[_\W]+", re.sub(r"(?<=[a-z])(?=[A-Z])", "_", func_name))
    return any(tok.lower() in _OBJECT_HINTS for tok in tokens if tok)


def validate_candidate(
    candidate: ProposalCandidate,
    store: MemStore,
    *,
    valid_history_ids: set[str] | None = None,
    existing_candidates: set[str] | None = None,
) -> list[str]:
    """Mechanical review of one candidate; return a list of issue strings (empty = OK).

    Checks: valid identifier name, valid ``target_library``, non-empty code,
    ``source_history`` references real history ids, granularity ceiling for the
    atomic task library, and no name collision with an existing candidate.
    """
    issues: list[str] = []
    if not _IDENT_RE.match(candidate.func_name):
        issues.append(f"func_name {candidate.func_name!r} is not a valid identifier")
    if candidate.target_library not in VALID_TARGET_LIBRARIES:
        issues.append(
            f"target_library must be one of {VALID_TARGET_LIBRARIES}, got {candidate.target_library!r}"
        )
    if not candidate.code.strip():
        issues.append("code is empty")
    if not candidate.source_history:
        issues.append("source_history is empty (every candidate must cite its evidence)")

    known = valid_history_ids if valid_history_ids is not None else set(store.list_history_ids())
    for hid in candidate.source_history:
        if hid not in known:
            issues.append(f"source_history id {hid!r} does not exist")

    if candidate.target_library == "atomic_task_library" and is_object_bound_name(candidate.func_name):
        issues.append(
            f"func_name {candidate.func_name!r} is object/task-bound; atomic tasks must be generic "
            "(e.g. pick(obj), not put_apple())"
        )

    existing = existing_candidates if existing_candidates is not None else set(store.list_candidate_names())
    if candidate.func_name in existing:
        issues.append(f"candidate {candidate.func_name!r} already exists in the pool")

    return issues


def validate_proposal(proposal: Proposal, store: MemStore) -> dict[str, list[str]]:
    """Run mechanical checks over all candidates; map func_name -> issues.

    Names with an empty issue list pass the mechanical gate. Reads the store's
    history/candidate ids once and shares them across candidates.
    """
    valid_ids = set(store.list_history_ids())
    existing = set(store.list_candidate_names())
    report: dict[str, list[str]] = {}
    for c in proposal.candidates:
        report[c.func_name] = validate_candidate(
            c, store, valid_history_ids=valid_ids, existing_candidates=existing
        )
    return report


def write_accepted(
    candidate: ProposalCandidate,
    store: MemStore,
    *,
    now: _dt.datetime | None = None,
) -> CandidateStats:
    """Persist an accepted candidate (``.py`` + fresh ``.stats.json``) to the pool.

    Stats start at zero counters; ``source_history`` is taken verbatim from the
    proposal and ``created_date`` is stamped. Returns the written stats.
    """
    now = now or _dt.datetime.now()
    stats = CandidateStats(
        func_name=candidate.func_name,
        target_library=candidate.target_library,
        source_history=list(candidate.source_history),
        created_date=now.strftime("%Y-%m-%d"),
    )
    store.write_candidate(stats, candidate.code)
    return stats
