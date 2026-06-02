"""Data schemas for the ``mem/`` pools (shared contract).

Mirrors ``docs-se/storage.md``. Three artifact types flow through the pipeline:

- **SuccessLog** -> ``history_pool/<id>.json``  (full success trial; drill-down source)
- **digest**     -> ``history_pool/<id>.digest.md``  (lossy, sourced entry view)
- **CandidateStats** -> ``func_candidate_pool/<func>.stats.json``  (running stats)

Each dataclass round-trips through plain dicts (``from_dict`` / ``to_dict``) and
validates required fields, so a malformed file fails loudly at read time rather
than corrupting a downstream consumer.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

# Digest fixed sections, in order. The renderer/parser key off these.
DIGEST_SECTIONS = (
    "KEY STRATEGY",
    "REUSABLE PATTERN",
    "KEY HYPER-PARAMS / FEEDBACK",
    "FRAGILITY",
)

# A backtrace marker tying a digest claim to chat_history[idx], e.g. "[#3]".
SOURCE_RE = re.compile(r"\[#(\d+)\]")

VALID_TARGET_LIBRARIES = ("skill_library", "atomic_task_library")


def slugify_task(task: str) -> str:
    """Filesystem-safe slug for a task name (used in ``<task>__<ts>`` ids)."""
    slug = re.sub(r"[^0-9A-Za-z._-]+", "-", task.strip()).strip("-")
    return slug or "task"


# --------------------------------------------------------------------------- #
# success log
# --------------------------------------------------------------------------- #
@dataclass
class SuccessLog:
    """``history_pool/<id>.json`` — a human-confirmed, generalized success trial."""

    task: str
    final_code: str
    chat_history: list[dict[str, Any]] = field(default_factory=list)
    settings: dict[str, Any] = field(default_factory=dict)
    datetime: str = ""

    def validate(self) -> None:
        if not self.task or not isinstance(self.task, str):
            raise ValueError("SuccessLog.task must be a non-empty string")
        if not isinstance(self.final_code, str) or not self.final_code.strip():
            raise ValueError("SuccessLog.final_code must be a non-empty string")
        if not isinstance(self.chat_history, list):
            raise ValueError("SuccessLog.chat_history must be a list")
        if not isinstance(self.settings, dict):
            raise ValueError("SuccessLog.settings must be a dict")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return {
            "task": self.task,
            "settings": self.settings,
            "final_code": self.final_code,
            "chat_history": self.chat_history,
            "datetime": self.datetime,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SuccessLog":
        missing = {"task", "final_code"} - set(data)
        if missing:
            raise ValueError(f"SuccessLog missing required field(s): {sorted(missing)}")
        obj = cls(
            task=data["task"],
            final_code=data["final_code"],
            chat_history=data.get("chat_history", []),
            settings=data.get("settings", {}),
            datetime=data.get("datetime", ""),
        )
        obj.validate()
        return obj


# --------------------------------------------------------------------------- #
# digest
# --------------------------------------------------------------------------- #
@dataclass
class Digest:
    """``history_pool/<id>.digest.md`` — the small, sourced entry view.

    ``sections`` maps each fixed section name to its markdown body. The body
    should carry ``[#idx]`` backtrace markers into ``chat_history``; this view is
    lossy by design — the ``.json`` is the source of truth.
    """

    task: str
    datetime: str
    settings_line: str
    sections: dict[str, str] = field(default_factory=dict)

    def validate(self, max_words: int | None = None) -> None:
        if not self.task:
            raise ValueError("Digest.task must be non-empty")
        missing = set(DIGEST_SECTIONS) - set(self.sections)
        if missing:
            raise ValueError(f"Digest missing required section(s): {sorted(missing)}")
        if max_words is not None and self.word_count() > max_words:
            raise ValueError(
                f"Digest exceeds max_words={max_words} (got {self.word_count()} words)"
            )

    def word_count(self) -> int:
        """Words across all section bodies (the budget governed by max_words)."""
        return sum(len(body.split()) for body in self.sections.values())

    def source_indices(self) -> list[int]:
        """All ``[#idx]`` backtrace targets referenced across the digest."""
        seen: list[int] = []
        for body in self.sections.values():
            for m in SOURCE_RE.finditer(body):
                idx = int(m.group(1))
                if idx not in seen:
                    seen.append(idx)
        return seen

    def render(self) -> str:
        """Render to the fixed markdown layout from ``storage.md``."""
        lines = [f"# {self.task}  ·  {self.datetime}", f"- settings: {self.settings_line}", ""]
        for name in DIGEST_SECTIONS:
            lines.append(f"## {name}")
            lines.append(self.sections.get(name, "").strip())
            lines.append("")
        return "\n".join(lines).rstrip() + "\n"

    @classmethod
    def parse(cls, text: str) -> "Digest":
        """Parse a rendered digest back into structured form."""
        task, datetime_str, settings_line = "", "", ""
        sections: dict[str, str] = {}
        current: str | None = None
        buf: list[str] = []

        def flush() -> None:
            if current is not None:
                sections[current] = "\n".join(buf).strip()

        for line in text.splitlines():
            if line.startswith("# ") and not line.startswith("## "):
                header = line[2:].strip()
                if "·" in header:
                    task, datetime_str = (p.strip() for p in header.split("·", 1))
                else:
                    task = header
            elif line.startswith("- settings:"):
                settings_line = line[len("- settings:"):].strip()
            elif line.startswith("## "):
                flush()
                current = line[3:].strip()
                buf = []
            elif current is not None:
                buf.append(line)
        flush()
        return cls(task=task, datetime=datetime_str, settings_line=settings_line, sections=sections)


# --------------------------------------------------------------------------- #
# candidate stats
# --------------------------------------------------------------------------- #
@dataclass
class CandidateStats:
    """``func_candidate_pool/<func>.stats.json`` — running stats per candidate."""

    func_name: str
    target_library: str
    source_history: list[str] = field(default_factory=list)
    positive: int = 0
    eval_runs: int = 0
    used_runs: int = 0
    created_date: str = ""
    last_eval_date: str = ""

    def validate(self) -> None:
        if not self.func_name:
            raise ValueError("CandidateStats.func_name must be non-empty")
        if self.target_library not in VALID_TARGET_LIBRARIES:
            raise ValueError(
                f"CandidateStats.target_library must be one of {VALID_TARGET_LIBRARIES}, "
                f"got {self.target_library!r}"
            )
        for name in ("positive", "eval_runs", "used_runs"):
            value = getattr(self, name)
            if not isinstance(value, int) or value < 0:
                raise ValueError(f"CandidateStats.{name} must be a non-negative int")
        if self.used_runs > self.eval_runs:
            raise ValueError("CandidateStats.used_runs cannot exceed eval_runs")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return {
            "func_name": self.func_name,
            "target_library": self.target_library,
            "positive": self.positive,
            "eval_runs": self.eval_runs,
            "used_runs": self.used_runs,
            "source_history": self.source_history,
            "created_date": self.created_date,
            "last_eval_date": self.last_eval_date,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CandidateStats":
        missing = {"func_name", "target_library"} - set(data)
        if missing:
            raise ValueError(f"CandidateStats missing required field(s): {sorted(missing)}")
        obj = cls(
            func_name=data["func_name"],
            target_library=data["target_library"],
            source_history=data.get("source_history", []),
            positive=data.get("positive", 0),
            eval_runs=data.get("eval_runs", 0),
            used_runs=data.get("used_runs", 0),
            created_date=data.get("created_date", ""),
            last_eval_date=data.get("last_eval_date", ""),
        )
        obj.validate()
        return obj
