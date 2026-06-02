"""Module ④: Update Planner — LLM-1 propose → LLM-2 review → candidate pool.

Triggered when ``history_pool`` has ``>= trigger_history_count`` unprocessed
entries. Two LLM roles both run the History Reader loop (digest-first, drill on
demand, capped by ``max_read_iterations``):

- **LLM-1** proposes a batch of new/updated candidate functions (each citing
  ``source_history`` + a rationale) via the terminal ``propose`` action.
- **LLM-2** reviews them — drilling back to verify references, dedup, generality,
  and atomic-task granularity — then *writes* the ones it accepts into
  ``func_candidate_pool``; anything it rejects becomes feedback that wakes LLM-1
  to revise, up to ``max_revise_iterations`` (then a forced finalize).

Writes are additionally gated by the deterministic checks in
:mod:`capx.self_evolve.proposal`, so a hallucinated reference or object-bound
name can never reach the pool regardless of what LLM-2 says.

The model is reached only through an injected ``query_fn(messages) -> str``; the
tool loop, dispatch, revise control flow, gating and cursor advance are all
exercisable with a scripted ``query_fn`` (no live server needed). Whether a real
model emits the documented action JSON is the one live-only unknown.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Callable

from capx.self_evolve.config import UpdatePlannerConfig
from capx.self_evolve.history_reader import HistoryReader
from capx.self_evolve.proposal import (
    Proposal,
    ProposalCandidate,
    validate_candidate,
    write_accepted,
)
from capx.self_evolve.storage import MemStore

logger = logging.getLogger(__name__)

QueryFn = Callable[[list[dict]], str]

# Read tools exposed in the loop -> (HistoryReader method, allowed arg names).
_TOOLS = {
    "list_unprocessed": (lambda r, a: r.list_unprocessed()),
    "read_digest": (lambda r, a: r.read_digest(a["id"])),
    "read_history": (lambda r, a: r.read_history(
        a["id"], a["field"], int(a.get("offset", 0)),
        None if a.get("limit") is None else int(a["limit"]),
    )),
    "grep_history": (lambda r, a: r.grep_history(a["regex"], a.get("ids"), a.get("field", "chat_history"))),
    "read_library": (lambda r, a: r.read_library(a.get("target"))),
}


# --------------------------------------------------------------------------- #
# action parsing
# --------------------------------------------------------------------------- #
def extract_json_action(text: str) -> dict[str, Any] | None:
    """Extract the last top-level JSON object from a model response.

    Models wrap the action in a ```json fence or emit it raw, often after some
    reasoning text. We scan for balanced ``{...}`` spans and return the last one
    that parses — that is the model's final action.
    """
    candidates: list[str] = []
    for m in re.finditer(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL):
        candidates.append(m.group(1))
    if not candidates:
        depth, start = 0, None
        for i, ch in enumerate(text):
            if ch == "{":
                if depth == 0:
                    start = i
                depth += 1
            elif ch == "}" and depth:
                depth -= 1
                if depth == 0 and start is not None:
                    candidates.append(text[start:i + 1])
    for blob in reversed(candidates):
        try:
            obj = json.loads(blob)
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            continue
    return None


# --------------------------------------------------------------------------- #
# tool loop
# --------------------------------------------------------------------------- #
def run_tool_loop(
    query_fn: QueryFn,
    system_prompt: str,
    user_prompt: str,
    reader: HistoryReader,
    terminal_key: str,
    *,
    max_read_iterations: int,
) -> dict[str, Any]:
    """Drive one LLM through read tools until it emits ``terminal_key``.

    Returns the terminal action's payload (the value under ``terminal_key``). A
    read tool call dispatches to :class:`HistoryReader` and its result is fed back
    as a user turn. On the last allowed iteration the model is told to finalize;
    if it still does not, an empty terminal payload is returned so the caller can
    proceed deterministically rather than hang.
    """
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    for it in range(max_read_iterations):
        last = it == max_read_iterations - 1
        if last:
            messages.append({
                "role": "user",
                "content": (
                    f"Read budget reached. Emit your final {terminal_key!r} action "
                    "now as a single JSON object, based on the evidence so far."
                ),
            })
        action = extract_json_action(query_fn(messages) or "")
        if action is None:
            messages.append({"role": "user", "content": (
                "No JSON action found. Respond with a single JSON object: either "
                f'{{"tool": ..., "args": {{...}}}} or your terminal {{"{terminal_key}": ...}}.'
            )})
            continue
        if terminal_key in action:
            return action[terminal_key] or {}
        tool = action.get("tool")
        if tool in _TOOLS:
            try:
                result = _TOOLS[tool](reader, action.get("args", {}) or {})
                payload = json.dumps(result, ensure_ascii=False, default=str)
            except Exception as e:  # noqa: BLE001 - surface tool errors to the model
                payload = json.dumps({"error": f"{type(e).__name__}: {e}"})
            messages.append({"role": "assistant", "content": json.dumps(action)})
            messages.append({"role": "user", "content": f"Result of {tool}:\n{payload}"})
        else:
            messages.append({"role": "user", "content": (
                f"Unknown tool {tool!r}. Available read tools: {sorted(_TOOLS)}; "
                f"or emit your terminal {{\"{terminal_key}\": ...}}."
            )})
    logger.warning("tool loop hit read budget without a terminal %r action", terminal_key)
    return {}


# --------------------------------------------------------------------------- #
# prompts
# --------------------------------------------------------------------------- #
_TOOLS_DOC = (
    "Read-only tools (call one per step as JSON {\"tool\": name, \"args\": {...}}):\n"
    "- list_unprocessed(): index {id,task,datetime,digest_path} of unprocessed history.\n"
    "- read_digest(id): the small sourced digest (READ THIS BY DEFAULT).\n"
    "- read_history(id, field, offset?, limit?): drill the full json; field in "
    "{final_code,chat_history,settings}; use the digest's [#idx] as offset.\n"
    "- grep_history(regex, ids?, field?): find where a name/keyword appears.\n"
    "- read_library(target?): current skill_library / atomic_task_library signatures+docstrings.\n"
    "Work digest-first; only drill to verify a claim before committing to it."
)

_PROPOSE_SYSTEM = (
    "You are LLM-1, the proposer in cap-x library management. From recently "
    "successful, human-confirmed trials you propose NEW or UPDATED reusable "
    "functions to add to the long-term libraries. Prefer few, genuinely general "
    "candidates over many narrow ones.\n\n" + _TOOLS_DOC + "\n\n"
    "Granularity: atomic_task_library functions must NOT bind a specific object or "
    "task (pick(obj)/place(...) yes; put_apple() no). skill_library functions "
    "compose primitives into general workflows.\n"
    "Finish with the terminal action:\n"
    '{"propose": {"candidates": [{"func_name","target_library","code",'
    '"source_history":[ids],"rationale"}]}}\n'
    "candidates may be [] if nothing is worth distilling. source_history MUST cite "
    "real ids from list_unprocessed()."
)

_REVIEW_SYSTEM = (
    "You are LLM-2, the reviewer in cap-x library management. You scrutinize LLM-1's "
    "proposed candidates by drilling back into the source history and the current "
    "library: is each source_history reference real, is the function a duplicate, "
    "does it have bugs, is it general enough, and is its granularity within an "
    "atomic task (reject object/task-bound names)?\n\n" + _TOOLS_DOC + "\n\n"
    "Finish with the terminal action listing the candidates you ACCEPT (you may "
    "edit their code) and feedback for any you reject:\n"
    '{"review": {"accepted": [{"func_name","target_library","code",'
    '"source_history":[ids],"rationale"}], "feedback": "<why you rejected the rest, '
    'or empty if all accepted>"}}'
)


def _proposal_user_prompt() -> str:
    return (
        "Review the unprocessed successful trials and propose reusable functions. "
        "Start by calling list_unprocessed(), then read digests."
    )


def _review_user_prompt(proposal: Proposal, feedback_round: int) -> str:
    return (
        f"Review round {feedback_round}. LLM-1 proposed these candidates:\n"
        f"{json.dumps(proposal.to_dict(), ensure_ascii=False, indent=2)}\n\n"
        "Verify each against the source history and current library, then emit your "
        "review action with the accepted candidates and feedback for the rest."
    )


def _revise_user_prompt(feedback: str, rejected: list[str]) -> str:
    return (
        "LLM-2 reviewed your proposal. Revise the candidates that were NOT accepted "
        f"({rejected or 'none'}). Reviewer feedback:\n{feedback}\n\n"
        "You may read more history to address it. Emit an updated propose action "
        "containing ONLY the revised/remaining candidates."
    )


# --------------------------------------------------------------------------- #
# orchestration
# --------------------------------------------------------------------------- #
@dataclass
class Review:
    accepted: list[ProposalCandidate] = field(default_factory=list)
    feedback: str = ""

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> "Review":
        return cls(
            accepted=[ProposalCandidate.from_dict(c) for c in payload.get("accepted", [])],
            feedback=payload.get("feedback", ""),
        )


@dataclass
class UpdatePlannerResult:
    triggered: bool
    processed_ids: list[str] = field(default_factory=list)
    written: list[str] = field(default_factory=list)
    revise_rounds: int = 0
    rejected_feedback: str = ""


def should_trigger(store: MemStore, config: UpdatePlannerConfig | None = None) -> bool:
    """True when unprocessed history has reached the trigger count."""
    config = config or UpdatePlannerConfig()
    return len(store.list_unprocessed()) >= config.trigger_history_count


def _gate_and_write(
    accepted: list[ProposalCandidate],
    store: MemStore,
    now,
    valid_ids: set[str],
) -> tuple[list[str], list[str]]:
    """Write mechanically-valid accepted candidates; return (written, bounced_msgs).

    Existing-candidate set is recomputed per call so names written earlier in the
    same run also count as collisions.
    """
    written: list[str] = []
    bounced: list[str] = []
    for cand in accepted:
        issues = validate_candidate(
            cand, store, valid_history_ids=valid_ids,
            existing_candidates=set(store.list_candidate_names()),
        )
        if issues:
            bounced.append(f"{cand.func_name}: {'; '.join(issues)}")
            continue
        write_accepted(cand, store, now=now)
        written.append(cand.func_name)
    return written, bounced


def run_update_planner(
    store: MemStore,
    query_fn: QueryFn,
    *,
    config: UpdatePlannerConfig | None = None,
    now=None,
    force: bool = False,
) -> UpdatePlannerResult:
    """Run one Update Planner pass over the current unprocessed batch.

    No-op (``triggered=False``) unless the trigger count is met or ``force``. On
    success the whole snapshotted batch is marked processed (files retained), even
    if zero candidates were written — the batch has been considered.
    """
    import datetime as _dt

    config = config or UpdatePlannerConfig()
    now = now or _dt.datetime.now()
    if not force and not should_trigger(store, config):
        return UpdatePlannerResult(triggered=False)

    batch_ids = store.list_unprocessed()
    if not batch_ids:
        return UpdatePlannerResult(triggered=False)
    valid_ids = set(store.list_history_ids())
    reader = HistoryReader(store)

    # LLM-1: initial proposal.
    proposal = Proposal.from_dict(run_tool_loop(
        query_fn, _PROPOSE_SYSTEM, _proposal_user_prompt(), reader,
        terminal_key="propose", max_read_iterations=config.max_read_iterations,
    ))

    written: list[str] = []
    last_feedback = ""
    rounds = 0
    for rounds in range(1, config.max_revise_iterations + 1):
        review = Review.from_payload(run_tool_loop(
            query_fn, _REVIEW_SYSTEM, _review_user_prompt(proposal, rounds), reader,
            terminal_key="review", max_read_iterations=config.max_read_iterations,
        ))
        round_written, bounced = _gate_and_write(review.accepted, store, now, valid_ids)
        written.extend(round_written)
        last_feedback = review.feedback

        accepted_names = {c.func_name for c in review.accepted}
        remaining = [c.func_name for c in proposal.candidates if c.func_name not in accepted_names]
        # Converged: reviewer gave no feedback, nothing bounced, nothing left over.
        if not review.feedback.strip() and not bounced and not remaining:
            break
        if rounds >= config.max_revise_iterations:
            logger.info("update planner: revise cap reached; finalizing")
            break

        # LLM-1 revises the leftovers using reviewer feedback + mechanical bounces.
        combined = "\n".join(filter(None, [review.feedback, *bounced]))
        proposal = Proposal.from_dict(run_tool_loop(
            query_fn, _PROPOSE_SYSTEM, _revise_user_prompt(combined, remaining), reader,
            terminal_key="propose", max_read_iterations=config.max_read_iterations,
        ))

    store.mark_processed(batch_ids)
    logger.info(
        "update planner processed %d history -> wrote %d candidate(s) in %d round(s)",
        len(batch_ids), len(written), rounds,
    )
    return UpdatePlannerResult(
        triggered=True, processed_ids=batch_ids, written=written,
        revise_rounds=rounds, rejected_feedback=last_feedback,
    )
