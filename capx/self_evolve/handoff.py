"""Live-loop → Feedback Postprocessor input contract (``postprocess_handoff.json``).

The interactive live loop (``capx/utils/trace_logger.py`` :meth:`write_handoff`,
called on human-confirmed success) drops one ``postprocess_handoff.json`` at the
trial root. That file is the **single artifact module ③ consumes** — it does not
re-parse the per-attempt traces.

This module is the *reader* half of that contract: it loads the JSON, validates
the fields the postprocessor needs, and maps them onto
:func:`capx.self_evolve.feedback_postprocessor.run_feedback_postprocessor`'s
keyword arguments. Keeping the mapping here (not in the debug CLI or the web
layer) means both the offline debugger and the eventual automated bridge read the
contract through one place.

See ``docs-se/storage.md`` (postprocess handoff schema) and
``docs-se/03-feedback-postprocessor.md``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

HANDOFF_FILENAME = "postprocess_handoff.json"
HANDOFF_SCHEMA = "postprocess_handoff/v1"


@dataclass
class Handoff:
    """A parsed, validated ``postprocess_handoff.json``."""

    path: Path
    task: str
    settings: dict[str, Any]
    final_code: str
    chat_history: list[dict[str, Any]]
    human_feedback: list[dict[str, Any]] = field(default_factory=list)
    success: dict[str, Any] = field(default_factory=dict)
    datetime: str = ""
    raw: dict[str, Any] = field(default_factory=dict)

    @property
    def short_task(self) -> str:
        """A compact task name for ids / digest headers (first non-empty line)."""
        for line in self.task.splitlines():
            line = line.strip().lstrip("# ").strip()
            if line:
                return line[:120]
        return "task"

    def feedback_texts(self) -> list[str]:
        """The verbatim human-feedback strings, in order (convenience view)."""
        return [str(f.get("text", "")) for f in self.human_feedback]

    def postprocessor_kwargs(self) -> dict[str, Any]:
        """Map the handoff onto ``run_feedback_postprocessor`` keyword args.

        ``original_code`` is the human-confirmed success code (pre-generalization)
        — module ③ generalizes *from* it. ``chat_history`` carries the verbatim
        human feedback as first-class turns, so the distiller can ``[#idx]``-cite
        it directly.
        """
        return {
            "task": self.short_task,
            "task_description": self.task,
            "settings": self.settings,
            "original_code": self.final_code,
            "chat_history": self.chat_history,
        }


def _resolve(path: str | Path) -> Path:
    """Accept either the JSON file or the trial dir that contains it."""
    p = Path(path)
    if p.is_dir():
        p = p / HANDOFF_FILENAME
    return p


def load_handoff(path: str | Path) -> Handoff:
    """Load + validate a handoff. Raises ``ValueError`` with a precise reason.

    The errors are deliberately specific — this doubles as the "did the live
    loop log the success correctly?" contract check used by the debug CLI.
    """
    p = _resolve(path)
    if not p.exists():
        raise FileNotFoundError(f"no handoff at {p} (live loop writes it on human Finish)")
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"handoff {p} is not valid JSON: {exc}") from exc

    schema = data.get("schema")
    if schema != HANDOFF_SCHEMA:
        raise ValueError(f"handoff {p}: unexpected schema {schema!r} (want {HANDOFF_SCHEMA!r})")
    task = data.get("task")
    if not isinstance(task, str) or not task.strip():
        raise ValueError(f"handoff {p}: 'task' must be a non-empty string")
    final_code = data.get("final_code")
    if not isinstance(final_code, str) or not final_code.strip():
        raise ValueError(f"handoff {p}: 'final_code' must be a non-empty string (success code)")
    chat_history = data.get("chat_history")
    if not isinstance(chat_history, list) or not chat_history:
        raise ValueError(f"handoff {p}: 'chat_history' must be a non-empty list")
    settings = data.get("settings") or {}
    if not isinstance(settings, dict):
        raise ValueError(f"handoff {p}: 'settings' must be a dict")

    return Handoff(
        path=p,
        task=task,
        settings=settings,
        final_code=final_code,
        chat_history=chat_history,
        human_feedback=data.get("human_feedback") or [],
        success=data.get("success") or {},
        datetime=data.get("datetime", ""),
        raw=data,
    )


def find_handoffs(logs_root: str | Path) -> list[Path]:
    """All ``postprocess_handoff.json`` under ``logs_root``, newest first."""
    root = Path(logs_root)
    if not root.exists():
        return []
    hits = list(root.glob(f"**/{HANDOFF_FILENAME}"))
    return sorted(hits, key=lambda p: p.stat().st_mtime, reverse=True)
