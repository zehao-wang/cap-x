"""Module ③: Feedback Postprocessor + Experience Distill.

Runs *after* a human confirms success (live loop ``human_finished``) and *before*
the trial is written to ``history_pool``. Two serial steps (see
``docs-se/03-feedback-postprocessor.md``):

1. **Generalize-by-rewrite** — multiple code-rewrite rounds that *still interact
   with the environment*; each round is judged correct/incorrect by a human only
   (no detailed feedback). Goal: strip the code's dependence on the one-off extra
   info a human gave this time (e.g. hoist a hard-coded ``z += 0.03`` into a
   grasp hyper-param). The last human-approved rewrite becomes ``final_code``.
2. **Experience Distill** — once ``final_code`` is finalized, ask fixed
   debugger-style questions and distil the whole trial into a small, sourced
   ``Digest`` (length-capped, every claim carrying a ``[#idx]`` backtrace into
   ``chat_history``). The ``.json`` (full) + ``.digest.md`` then enter the pool
   together.

This module is the **dependency-injected core**: it talks to the model and the
environment through two injected callables, so it is unit-testable with fakes and
the web layer supplies the real ones:

- ``query_fn(messages) -> str``  — query the LLM, return its text content.
- ``judge_fn(code) -> bool``     — run ``code`` in the env, ask the human, return
  whether the human judged it correct.
"""

from __future__ import annotations

import datetime as _dt
import logging
from dataclasses import dataclass
from typing import Any, Callable

from capx.self_evolve.config import ExperienceDistillConfig
from capx.self_evolve.schemas import DIGEST_SECTIONS, Digest, SuccessLog
from capx.self_evolve.storage import MemStore

logger = logging.getLogger(__name__)

QueryFn = Callable[[list[dict]], str]
JudgeFn = Callable[[str], bool]


# --------------------------------------------------------------------------- #
# transcript rendering (for sourcing)
# --------------------------------------------------------------------------- #
def _message_text(content: Any) -> str:
    """Flatten a chat message ``content`` (str or list-of-parts) to plain text."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, dict):
                if part.get("type") == "text":
                    parts.append(part.get("text", ""))
                elif "image_url" in part:
                    parts.append("[image]")
            else:
                parts.append(str(part))
        return "\n".join(p for p in parts if p)
    return "" if content is None else str(content)


def render_indexed_transcript(chat_history: list[dict], max_chars_per_msg: int = 600) -> str:
    """Render ``chat_history`` with explicit 0-based ``[#i]`` indices.

    The distiller cites these indices so every digest claim is traceable back to
    a concrete turn. Long messages are truncated for prompt economy — the full
    text always remains in the ``.json``.
    """
    lines = []
    for i, msg in enumerate(chat_history):
        role = msg.get("role", "?") if isinstance(msg, dict) else "?"
        text = _message_text(msg.get("content") if isinstance(msg, dict) else msg)
        text = text.strip()
        if len(text) > max_chars_per_msg:
            text = text[:max_chars_per_msg] + " …[truncated]"
        lines.append(f"[#{i}] ({role}): {text}")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# prompt builders (pure)
# --------------------------------------------------------------------------- #
def build_generalize_rewrite_prompt(
    task_description: str,
    current_code: str,
    *,
    round_index: int,
    last_was_rejected: bool,
) -> list[dict]:
    """Prompt asking the model to rewrite the code into a *more general* form.

    The success was likely reached using one-off info from human feedback. This
    asks the model to remove that direct dependence — hoist magic numbers into
    named hyper-params, replace object-specific assumptions with measured /
    parameterized ones — WITHOUT changing what the code accomplishes. The rewrite
    is then re-executed in the env and judged by the human.
    """
    system = (
        "You generalize a robot-control program that already succeeded. The "
        "success may have relied on one-off information a human gave during this "
        "session. Rewrite the code so it no longer depends on that one-off info: "
        "hoist hard-coded magic numbers into clearly-named hyper-parameters at the "
        "top, replace object/task-specific assumptions with measured or "
        "parameterized values, and keep the externally-observable behavior "
        "identical. Do NOT add new capabilities. Output ONLY the rewritten Python "
        "code, no fences, no explanation."
    )
    note = (
        "The previous rewrite was judged INCORRECT when run (it changed behavior "
        "or broke the task). Be more conservative: generalize less aggressively "
        "and preserve exactly what worked.\n\n"
        if last_was_rejected
        else ""
    )
    user = (
        f"Task:\n{task_description.strip()}\n\n"
        f"{note}"
        f"Current working code (rewrite round {round_index}):\n```python\n"
        f"{current_code.strip()}\n```\n\n"
        "Return the generalized version of this code."
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def build_distill_prompt(
    task_description: str,
    final_code: str,
    chat_history: list[dict],
    *,
    max_words: int,
) -> list[dict]:
    """Debugger-style fixed-question prompt that produces the sourced digest.

    Asks the four fixed sections (KEY STRATEGY / REUSABLE PATTERN / KEY
    HYPER-PARAMS / FEEDBACK / FRAGILITY), enforces a hard word cap, and requires a
    ``[#idx]`` backtrace on every claim. The expected output is the markdown of
    the digest *sections only* (the header/settings line are added by the writer).
    """
    transcript = render_indexed_transcript(chat_history)
    sections_spec = "\n".join(f"## {name}\n<...>  [#idx]" for name in DIGEST_SECTIONS)
    system = (
        "You are an experience distiller for a robot-control coding agent, in the "
        "style of an agent debugger: you turn a long successful trial into a SMALL, "
        "SOURCED report. You do not summarize vaguely — you answer fixed questions "
        "and cite evidence. Every claim MUST end with a backtrace marker [#i] "
        "pointing at the 0-based index of the chat message that supports it. You "
        "never write code."
    )
    user = (
        f"Task:\n{task_description.strip()}\n\n"
        f"Finalized (generalized) code:\n```python\n{final_code.strip()}\n```\n\n"
        f"Indexed trial transcript (cite these indices):\n{transcript}\n\n"
        "Write the digest as EXACTLY these markdown sections and nothing else "
        "(no preamble, no code, no extra sections):\n\n"
        f"{sections_spec}\n\n"
        "Rules:\n"
        f"- HARD LIMIT: at most {max_words} words across all section bodies.\n"
        "- Every bullet/claim ends with a [#i] backtrace into the transcript.\n"
        "- KEY STRATEGY: the core idea of the code; why it succeeded.\n"
        "- REUSABLE PATTERN: which steps generalize and could become a "
        "skill_library / atomic_task_library function, at what granularity.\n"
        "- KEY HYPER-PARAMS / FEEDBACK: hyper-params locked in during "
        "generalization + which human feedback gave each value.\n"
        "- FRAGILITY: anything fragile / lucky; write 'none' if nothing."
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def build_shorten_prompt(distill_text: str, max_words: int, current_words: int) -> list[dict]:
    """Ask the distiller to compress an over-long digest, keeping structure + sources."""
    return [
        {
            "role": "system",
            "content": (
                "You compress a sourced digest. Keep the exact same markdown "
                "section headers and keep a [#i] backtrace on every claim, but cut "
                "it down. Output only the digest sections."
            ),
        },
        {
            "role": "user",
            "content": (
                f"This digest is {current_words} words; the hard limit is "
                f"{max_words}. Compress it to under the limit without dropping "
                f"sections or sources:\n\n{distill_text}"
            ),
        },
    ]


# --------------------------------------------------------------------------- #
# distill parsing
# --------------------------------------------------------------------------- #
def parse_distill_response(
    text: str,
    *,
    task: str,
    datetime_str: str,
    settings_line: str,
) -> Digest:
    """Parse a distiller's sectioned markdown into a :class:`Digest`.

    Tolerates a leading task/settings header if the model echoed one; otherwise
    fills header fields from the passed-in values. Missing sections are left for
    :meth:`Digest.validate` to reject.
    """
    parsed = Digest.parse(text)
    # The writer owns the header; only fall back to parsed values if it had them.
    return Digest(
        task=task,
        datetime=datetime_str,
        settings_line=settings_line,
        sections={name: parsed.sections.get(name, "").strip() for name in DIGEST_SECTIONS},
    )


# --------------------------------------------------------------------------- #
# orchestration
# --------------------------------------------------------------------------- #
@dataclass
class PostprocessResult:
    history_id: str
    final_code: str
    rewrite_rounds: int
    digest: Digest


def generalize_by_rewrite(
    task_description: str,
    original_code: str,
    *,
    query_fn: QueryFn,
    judge_fn: JudgeFn,
    max_rounds: int = 3,
) -> tuple[str, int]:
    """Run the generalize-by-rewrite loop; return ``(final_code, accepted_rounds)``.

    ``original_code`` is already human-confirmed, so it is the safe baseline. Each
    round proposes a more general version; if the human judges it correct it
    becomes the new baseline and we try to generalize further, else we stop and
    keep the last approved baseline. A round that returns no/empty code stops too.
    """
    baseline = original_code
    accepted = 0
    last_rejected = False
    for r in range(max_rounds):
        prompt = build_generalize_rewrite_prompt(
            task_description, baseline, round_index=r, last_was_rejected=last_rejected
        )
        candidate = (query_fn(prompt) or "").strip()
        if not candidate or candidate == baseline.strip():
            logger.info("generalize round %d: no change proposed; stopping", r)
            break
        if judge_fn(candidate):
            logger.info("generalize round %d: human accepted rewrite", r)
            baseline = candidate
            accepted += 1
            last_rejected = False
        else:
            logger.info("generalize round %d: human rejected; keeping baseline", r)
            if last_rejected:
                break  # two strikes -> stop pushing
            last_rejected = True
    return baseline, accepted


def distill_experience(
    task_description: str,
    final_code: str,
    chat_history: list[dict],
    *,
    query_fn: QueryFn,
    datetime_str: str,
    settings_line: str,
    config: ExperienceDistillConfig | None = None,
    max_shorten_retries: int = 2,
) -> Digest:
    """Run the distill step with a length guard; return a validated-ish Digest.

    Enforces ``config.max_words`` by re-asking the model to compress up to
    ``max_shorten_retries`` times. If it is still over after that, the digest is
    returned anyway (a confirmed-success trial must not be lost over a stubborn
    word count) but a warning is logged.
    """
    config = config or ExperienceDistillConfig()
    prompt = build_distill_prompt(
        task_description, final_code, chat_history, max_words=config.max_words
    )
    text = (query_fn(prompt) or "").strip()
    digest = parse_distill_response(
        text, task=task_description_to_task(task_description), datetime_str=datetime_str, settings_line=settings_line
    )

    for _ in range(max_shorten_retries):
        if digest.word_count() <= config.max_words:
            break
        logger.info(
            "digest over cap (%d > %d words); asking to compress",
            digest.word_count(), config.max_words,
        )
        text = (query_fn(build_shorten_prompt(text, config.max_words, digest.word_count())) or "").strip()
        digest = parse_distill_response(
            text, task=digest.task, datetime_str=datetime_str, settings_line=settings_line
        )

    if digest.word_count() > config.max_words:
        logger.warning(
            "digest still over cap after retries (%d > %d words); keeping it anyway",
            digest.word_count(), config.max_words,
        )
    return digest


def task_description_to_task(task_description: str) -> str:
    """Derive a short task name from a (possibly long) task description.

    The digest header wants a name, not the whole API-laden prompt. Use the first
    non-empty line, trimmed; callers with a real task name should pass it directly
    via the orchestrator instead of relying on this.
    """
    for line in task_description.splitlines():
        line = line.strip().lstrip("# ").strip()
        if line:
            return line[:120]
    return "task"


def run_feedback_postprocessor(
    *,
    task: str,
    task_description: str,
    settings: dict[str, Any],
    original_code: str,
    chat_history: list[dict],
    query_fn: QueryFn,
    judge_fn: JudgeFn,
    store: MemStore,
    config: ExperienceDistillConfig | None = None,
    max_rewrite_rounds: int = 3,
    now: _dt.datetime | None = None,
) -> PostprocessResult:
    """End-to-end module ③: generalize -> distill -> write the history pair.

    ``task`` is the canonical short task name (used for the history id + digest
    header); ``task_description`` is the full prompt shown to the model.
    """
    config = config or ExperienceDistillConfig()
    now = now or _dt.datetime.now()
    datetime_str = now.strftime("%Y-%m-%d %H:%M:%S")

    final_code, rounds = generalize_by_rewrite(
        task_description, original_code,
        query_fn=query_fn, judge_fn=judge_fn, max_rounds=max_rewrite_rounds,
    )

    settings_line = _settings_line(settings)
    digest = distill_experience(
        task_description, final_code, chat_history,
        query_fn=query_fn, datetime_str=datetime_str, settings_line=settings_line, config=config,
    )
    # The header task name is authoritative from the caller, not the description.
    digest.task = task

    log = SuccessLog(
        task=task,
        final_code=final_code,
        chat_history=chat_history,
        settings=settings,
        datetime=datetime_str,
    )
    history_id = store.write_history(log, digest)
    logger.info("feedback postprocessor wrote history %s (%d rewrite rounds)", history_id, rounds)
    return PostprocessResult(
        history_id=history_id, final_code=final_code, rewrite_rounds=rounds, digest=digest
    )


def _settings_line(settings: dict[str, Any]) -> str:
    """Compact one-line settings summary for the digest header."""
    keys = ("env", "dataset", "llm", "model", "task_suite")
    parts = [f"{k}={settings[k]}" for k in keys if k in settings]
    if not parts:
        parts = [f"{k}={v}" for k, v in list(settings.items())[:4]]
    return ", ".join(parts) if parts else "(none)"
