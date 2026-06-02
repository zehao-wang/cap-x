"""Module ④: History Reader — read-only tool set over the pools.

cap-x's own implementation (no external ``adb`` CLI). Both LLM-1 (propose) and
LLM-2 (review) drive these tools in a *digest-first, drill-on-demand* loop
(``docs-se/04-update-planner.md``): start from the lightweight unprocessed index,
read small ``.digest.md`` entries by default, and only drill back into the full
``.json`` (by the digest's ``[#idx]`` markers) when a claim needs verifying.

Every method here is read-only and side-effect-free, so the tool loop can be
exercised without a live model. ``read_library`` parses the *current* long-term
libraries' signatures + docstrings (for dedup / generality / granularity checks);
the library dirs may not exist yet, in which case it returns an empty list.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path
from typing import Any

from capx.self_evolve.storage import MemStore

# Repo root = two levels up from capx/self_evolve/history_reader.py.
_REPO_ROOT = Path(__file__).resolve().parents[2]
_LIBRARY_DIRS = {
    "skill_library": _REPO_ROOT / "capx" / "skill_library",
    "atomic_task_library": _REPO_ROOT / "capx" / "atomic_task_library",
}
_DRILLABLE_FIELDS = ("final_code", "chat_history", "settings")


def _flatten_message(content: Any) -> str:
    """Flatten one chat message's ``content`` (str or list-of-parts) to text."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        out = []
        for part in content:
            if isinstance(part, dict):
                if part.get("type") == "text":
                    out.append(part.get("text", ""))
                elif "image_url" in part:
                    out.append("[image]")
            else:
                out.append(str(part))
        return "\n".join(p for p in out if p)
    return "" if content is None else str(content)


class HistoryReader:
    """Read-only tools over a :class:`MemStore` + the long-term libraries."""

    def __init__(self, store: MemStore, library_dirs: dict[str, Path] | None = None) -> None:
        self.store = store
        self.library_dirs = library_dirs or _LIBRARY_DIRS

    # --------------------------------------------------------------- entry tool
    def list_unprocessed(self) -> list[dict[str, Any]]:
        """Lightweight index of unprocessed history (NO bodies). The entry point.

        Each item: ``{id, task, datetime, digest_path}``. Reads only the small
        digest header to fill ``task``/``datetime`` so the index stays cheap.
        """
        items = []
        for hid in self.store.list_unprocessed():
            task, datetime_str = hid, ""
            try:
                digest = self.store.read_digest(hid)
                task, datetime_str = digest.task or hid, digest.datetime
            except (FileNotFoundError, ValueError):
                pass  # malformed/absent digest: still index it by id
            items.append({
                "id": hid,
                "task": task,
                "datetime": datetime_str,
                "digest_path": str(self.store.history_pool / f"{hid}.digest.md"),
            })
        return items

    # --------------------------------------------------------- default = digest
    def read_digest(self, history_id: str) -> str:
        """Read the rendered ``<id>.digest.md`` (small, sourced). The default read."""
        return self.store.read_digest(history_id).render()

    # ----------------------------------------------------------- drill-down
    def read_history(
        self,
        history_id: str,
        field: str,
        offset: int = 0,
        limit: int | None = None,
    ) -> Any:
        """Drill into one field of the full ``.json``.

        - ``chat_history`` (list): ``offset``/``limit`` paginate over messages —
          pass the digest's ``[#idx]`` as ``offset`` (with ``limit=1``) to fetch
          exactly that turn.
        - ``final_code`` (str): ``offset``/``limit`` paginate over lines.
        - ``settings`` (dict): returned whole (small).
        """
        if field not in _DRILLABLE_FIELDS:
            raise ValueError(f"field must be one of {_DRILLABLE_FIELDS}, got {field!r}")
        log = self.store.read_history(history_id)
        value = getattr(log, field)
        if isinstance(value, list):  # chat_history
            end = len(value) if limit is None else offset + limit
            return value[offset:end]
        if isinstance(value, str):  # final_code
            lines = value.splitlines()
            end = len(lines) if limit is None else offset + limit
            return "\n".join(lines[offset:end])
        return value  # settings dict

    def grep_history(
        self,
        regex: str,
        ids: list[str] | None = None,
        field: str = "chat_history",
    ) -> list[dict[str, Any]]:
        """Regex-search across histories; return hit locations, not bodies.

        Each hit: ``{id, field, message_index}`` — ``message_index`` is the
        ``chat_history`` turn for that field, else ``None``. Lets the model
        pinpoint where a tool/API/object name appears before drilling.
        """
        pattern = re.compile(regex)
        ids = ids if ids is not None else self.store.list_unprocessed()
        hits = []
        for hid in ids:
            try:
                log = self.store.read_history(hid)
            except (FileNotFoundError, ValueError):
                continue
            if field == "chat_history":
                for i, msg in enumerate(log.chat_history):
                    text = _flatten_message(msg.get("content") if isinstance(msg, dict) else msg)
                    if pattern.search(text):
                        hits.append({"id": hid, "field": field, "message_index": i})
            else:
                value = getattr(log, field, None)
                text = value if isinstance(value, str) else str(value)
                if pattern.search(text):
                    hits.append({"id": hid, "field": field, "message_index": None})
        return hits

    # ------------------------------------------------- current long-term library
    def read_library(self, target: str | None = None) -> list[dict[str, str]]:
        """Signatures + docstrings of the *current* long-term library functions.

        Used to check for duplicates, judge generality, and compare granularity.
        ``target`` restricts to one library; ``None`` reads both. Non-existent
        library dirs yield no entries (they are created only after promotion).
        """
        targets = [target] if target else list(self.library_dirs)
        out: list[dict[str, str]] = []
        for name in targets:
            directory = self.library_dirs.get(name)
            if directory is None:
                raise ValueError(f"unknown library target: {name!r}")
            if not directory.exists():
                continue
            for py in sorted(directory.glob("*.py")):
                if py.name == "__init__.py":
                    continue
                out.extend(self._extract_funcs(py, name))
        return out

    @staticmethod
    def _extract_funcs(path: Path, library: str) -> list[dict[str, str]]:
        """Top-level function signatures + docstrings from a module via ``ast``."""
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except (OSError, SyntaxError):
            return []
        funcs = []
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                funcs.append({
                    "library": library,
                    "module": path.stem,
                    "func_name": node.name,
                    "signature": f"{node.name}({_format_args(node.args)})",
                    "docstring": (ast.get_docstring(node) or "").strip(),
                })
        return funcs


def _format_args(args: ast.arguments) -> str:
    """Render a function's positional/keyword arg names (best-effort, no types)."""
    names = [a.arg for a in getattr(args, "posonlyargs", [])]
    names += [a.arg for a in args.args]
    if args.vararg:
        names.append(f"*{args.vararg.arg}")
    names += [a.arg for a in args.kwonlyargs]
    if args.kwarg:
        names.append(f"**{args.kwarg.arg}")
    return ", ".join(names)
