"""``mem/`` storage scaffold (shared contract).

Implements the layout in ``docs-se/storage.md`` as read/write helpers so each
pipeline stage talks to disk through one place:

    mem/
    ├── history_pool/        <id>.json + <id>.digest.md   (module ③ writes, ④ reads)
    ├── func_candidate_pool/ <func>.py + <func>.stats.json (④ writes, ⑤ updates)
    └── .processed_history   incremental cursor of handled history ids (④)

``MemStore`` is intentionally thin: paths, atomic-ish writes, schema validation
on the way in/out, and the processed-history cursor. It does not embed pipeline
logic — that lives in modules ③④⑤.
"""

from __future__ import annotations

import datetime as _dt
import json
import re
from pathlib import Path

from capx.self_evolve.schemas import (
    CandidateStats,
    Digest,
    SuccessLog,
    slugify_task,
)

# Repo root = three levels up from this file (capx/self_evolve/storage.py).
_REPO_ROOT = Path(__file__).resolve().parents[2]

# A python identifier (candidate func names map 1:1 to filenames).
_IDENT_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def default_mem_root() -> Path:
    """The repo-root ``mem/`` directory (pools live here, per the hard constraint)."""
    return _REPO_ROOT / "mem"


def make_history_id(task: str, when: _dt.datetime | None = None) -> str:
    """``<task-slug>__<YYYYMMDD-HHMMSS>`` id for a history pair."""
    when = when or _dt.datetime.now()
    return f"{slugify_task(task)}__{when:%Y%m%d-%H%M%S}"


class MemStore:
    """Read/write helper over a ``mem/`` root."""

    def __init__(self, root: Path | str | None = None) -> None:
        self.root = Path(root) if root is not None else default_mem_root()
        self.history_pool = self.root / "history_pool"
        self.candidate_pool = self.root / "func_candidate_pool"
        self.processed_history_file = self.root / ".processed_history"

    def ensure_dirs(self) -> None:
        """Create the pool directories if missing (idempotent)."""
        self.history_pool.mkdir(parents=True, exist_ok=True)
        self.candidate_pool.mkdir(parents=True, exist_ok=True)

    # ----------------------------------------------------------------- io util
    @staticmethod
    def _write_text(path: Path, text: str) -> None:
        """Write atomically via a temp file + rename to avoid torn reads."""
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(text, encoding="utf-8")
        tmp.replace(path)

    # ------------------------------------------------------------- history pool
    def write_history(self, log: SuccessLog, digest: Digest, history_id: str | None = None) -> str:
        """Write a ``.json`` + ``.digest.md`` pair; returns the history id.

        The two files always land together (digest is the entry view onto the
        full json). Validation runs before anything touches disk.
        """
        self.ensure_dirs()
        log.validate()
        digest.validate()
        history_id = history_id or make_history_id(log.task)
        self._write_text(
            self.history_pool / f"{history_id}.json",
            json.dumps(log.to_dict(), indent=2, ensure_ascii=False),
        )
        self._write_text(self.history_pool / f"{history_id}.digest.md", digest.render())
        return history_id

    def read_history(self, history_id: str) -> SuccessLog:
        """Read the full ``.json`` for a history id (drill-down source of truth)."""
        path = self.history_pool / f"{history_id}.json"
        if not path.exists():
            raise FileNotFoundError(f"no history json for id {history_id!r}: {path}")
        return SuccessLog.from_dict(json.loads(path.read_text(encoding="utf-8")))

    def read_digest(self, history_id: str) -> Digest:
        """Read the ``.digest.md`` for a history id (consumers' default view)."""
        path = self.history_pool / f"{history_id}.digest.md"
        if not path.exists():
            raise FileNotFoundError(f"no digest for id {history_id!r}: {path}")
        return Digest.parse(path.read_text(encoding="utf-8"))

    def list_history_ids(self) -> list[str]:
        """All history ids present in the pool (sorted; ts in id => chrono order)."""
        if not self.history_pool.exists():
            return []
        return sorted(p.stem for p in self.history_pool.glob("*.json"))

    # --------------------------------------------------------- candidate pool
    def write_candidate(self, stats: CandidateStats, code: str) -> None:
        """Write a candidate's ``.py`` source + ``.stats.json``."""
        self.ensure_dirs()
        if not _IDENT_RE.match(stats.func_name):
            raise ValueError(f"func_name is not a valid identifier: {stats.func_name!r}")
        stats.validate()
        if not code.strip():
            raise ValueError("candidate code must be non-empty")
        self._write_text(self.candidate_pool / f"{stats.func_name}.py", code)
        self._write_text(
            self.candidate_pool / f"{stats.func_name}.stats.json",
            json.dumps(stats.to_dict(), indent=2, ensure_ascii=False),
        )

    def read_candidate_stats(self, func_name: str) -> CandidateStats:
        path = self.candidate_pool / f"{func_name}.stats.json"
        if not path.exists():
            raise FileNotFoundError(f"no stats for candidate {func_name!r}: {path}")
        return CandidateStats.from_dict(json.loads(path.read_text(encoding="utf-8")))

    def read_candidate_code(self, func_name: str) -> str:
        path = self.candidate_pool / f"{func_name}.py"
        if not path.exists():
            raise FileNotFoundError(f"no code for candidate {func_name!r}: {path}")
        return path.read_text(encoding="utf-8")

    def list_candidate_names(self) -> list[str]:
        """All candidate func names with a ``.stats.json`` present (sorted)."""
        if not self.candidate_pool.exists():
            return []
        return sorted(p.name[: -len(".stats.json")] for p in self.candidate_pool.glob("*.stats.json"))

    def update_candidate_stats(self, stats: CandidateStats) -> None:
        """Overwrite just the ``.stats.json`` (Evaluator accumulates in place)."""
        self.ensure_dirs()
        stats.validate()
        if not (self.candidate_pool / f"{stats.func_name}.py").exists():
            raise FileNotFoundError(
                f"cannot update stats for {stats.func_name!r}: no candidate code on disk"
            )
        self._write_text(
            self.candidate_pool / f"{stats.func_name}.stats.json",
            json.dumps(stats.to_dict(), indent=2, ensure_ascii=False),
        )

    # ------------------------------------------------- processed-history cursor
    def read_processed_history(self) -> list[str]:
        """History ids already handled by Update Planner (incremental cursor)."""
        if not self.processed_history_file.exists():
            return []
        return [
            line.strip()
            for line in self.processed_history_file.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]

    def mark_processed(self, history_ids: list[str] | str) -> None:
        """Append history id(s) to the cursor (dedup, preserving first-seen order)."""
        if isinstance(history_ids, str):
            history_ids = [history_ids]
        self.ensure_dirs()
        merged = self.read_processed_history()
        for hid in history_ids:
            if hid not in merged:
                merged.append(hid)
        self._write_text(self.processed_history_file, "\n".join(merged) + "\n")

    def list_unprocessed(self) -> list[str]:
        """History ids present in the pool but not yet in the processed cursor."""
        processed = set(self.read_processed_history())
        return [hid for hid in self.list_history_ids() if hid not in processed]
