"""Structured per-trial trace of agent <-> LLM interactions and tool-call flow.

Each trial gets three artifacts in its output directory (``trial_XX/``):

  - ``llm_trace.jsonl`` : one JSON record per LLM call — phase, turn, model, the
    *full* input messages, and the model's full output (content + reasoning),
    plus the parsed decision / extracted code blocks when known. Inline image
    data-URLs in the input are stripped out (saved as PNGs under
    ``trace_images/``) so the file stays small and ``jq``-queryable.
  - ``events.jsonl``    : a chronological merge of LLM calls (one summary line
    each) and tool-execution steps (from ``execution_logger``), so the
    tool-call flow and the conversation interleave in real execution order.
  - ``trace.md``        : a human-readable rendering of ``events.jsonl``, written
    at finalize time.

Usage::

    trace = TraceLogger(trial_dir, model=args.model)
    trace.log_llm(phase="initial", turn=1, input_messages=msgs,
                  output_content=text, output_reasoning=reasoning)
    trace.log_tool(tool_name="SAM3", text="...", block_index=0, n_images=1)
    trace.finalize()  # writes trace.md

``NullTraceLogger`` is a no-op drop-in for runs without an output directory.
"""

from __future__ import annotations

import base64
import json
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class TraceLogger:
    """Writes the per-trial LLM / tool trace. Thread-safe (tool steps are logged
    from the env worker thread; LLM calls from the asyncio thread)."""

    def __init__(self, trial_dir: str | Path, model: str | None = None) -> None:
        self.dir = Path(trial_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.img_dir = self.dir / "trace_images"
        self.llm_path = self.dir / "llm_trace.jsonl"
        self.events_path = self.dir / "events.jsonl"
        self.md_path = self.dir / "trace.md"
        self.default_model = model

        self._lock = threading.Lock()
        self._seq = 0          # global event ordinal
        self._llm_seq = 0      # LLM-call ordinal
        self._events: list[dict[str, Any]] = []

        # Start fresh so a re-run of the same trial dir doesn't append to stale
        # files. (Images are overwritten in place by name.)
        for p in (self.llm_path, self.events_path, self.md_path):
            try:
                p.unlink()
            except FileNotFoundError:
                pass

    # ------------------------------------------------------------------ utils
    def _append(self, path: Path, rec: dict[str, Any]) -> None:
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    def _record_event(self, ev: dict[str, Any]) -> None:
        self._events.append(ev)
        self._append(self.events_path, ev)

    def _save_image(self, data_url: Any, llm_idx: int, img_i: int) -> str:
        if not isinstance(data_url, str):
            return "<non-string image>"
        try:
            b64 = data_url.split(",", 1)[-1] if data_url.startswith("data:") else data_url
            raw = base64.b64decode(b64)
            self.img_dir.mkdir(parents=True, exist_ok=True)
            name = f"llm{llm_idx:03d}_img{img_i}.png"
            (self.img_dir / name).write_bytes(raw)
            return f"trace_images/{name}"
        except Exception:
            return "<unsavable image>"

    def _sanitize_messages(self, messages: Any, llm_idx: int) -> list[dict[str, Any]]:
        """Deep copy of ``messages`` with inline image data-URLs replaced by a
        reference to a saved PNG, keeping the JSONL compact."""
        out: list[dict[str, Any]] = []
        img_i = 0
        for msg in messages or []:
            if not isinstance(msg, dict):
                out.append({"role": "?", "content": str(msg)})
                continue
            role = msg.get("role", "")
            content = msg.get("content")
            if isinstance(content, list):
                parts: list[dict[str, Any]] = []
                for part in content:
                    if isinstance(part, dict) and part.get("type") == "image_url":
                        url = part.get("image_url")
                        if isinstance(url, dict):
                            url = url.get("url")
                        ref = self._save_image(url, llm_idx, img_i)
                        img_i += 1
                        parts.append({
                            "type": "image_url",
                            "image_ref": ref,
                            "elided_chars": len(url) if isinstance(url, str) else 0,
                        })
                    elif isinstance(part, dict):
                        parts.append(part)
                    else:
                        parts.append({"type": "text", "text": str(part)})
                out.append({"role": role, "content": parts})
            else:
                out.append({"role": role, "content": content})
        return out

    # ------------------------------------------------------------------- API
    def log_llm(
        self,
        *,
        phase: str,
        turn: int,
        input_messages: Any,
        output_content: str | None,
        output_reasoning: str | None = None,
        model: str | None = None,
        decision: str | None = None,
        code_blocks: list[str] | None = None,
        duration_s: float | None = None,
    ) -> None:
        with self._lock:
            self._llm_seq += 1
            self._seq += 1
            llm_idx = self._llm_seq
            seq = self._seq
            ts = _now_iso()
            rec = {
                "seq": seq,
                "llm_call": llm_idx,
                "ts": ts,
                "phase": phase,
                "turn": turn,
                "model": model or self.default_model,
                "duration_s": round(duration_s, 3) if duration_s is not None else None,
                "input_messages": self._sanitize_messages(input_messages, llm_idx),
                "output": {
                    "content": output_content,
                    "reasoning": output_reasoning,
                },
                "decision": decision,
                "code_blocks": code_blocks,
            }
            self._append(self.llm_path, rec)
            self._record_event({
                "seq": seq,
                "ts": ts,
                "type": "llm",
                "llm_call": llm_idx,
                "phase": phase,
                "turn": turn,
                "model": rec["model"],
                "decision": decision,
                "n_code_blocks": len(code_blocks) if code_blocks else 0,
                "output_preview": (output_content or "")[:240],
            })

    def log_tool(
        self,
        *,
        tool_name: str,
        text: str,
        block_index: int | None = None,
        step_index: int | None = None,
        n_images: int = 0,
    ) -> None:
        with self._lock:
            self._seq += 1
            self._record_event({
                "seq": self._seq,
                "ts": _now_iso(),
                "type": "tool",
                "tool_name": tool_name,
                "text": text,
                "block_index": block_index,
                "step_index": step_index,
                "n_images": n_images,
            })

    def finalize(self) -> None:
        """Render the chronological ``events`` into a readable ``trace.md``."""
        with self._lock:
            lines = ["# Trial trace", ""]
            for ev in self._events:
                if ev.get("type") == "llm":
                    lines.append(
                        f"## [{ev['seq']}] LLM · {ev['phase']} · turn {ev['turn']} · {ev.get('model')}"
                    )
                    if ev.get("decision"):
                        lines.append(f"- decision: `{ev['decision']}`")
                    lines.append(f"- code blocks: {ev.get('n_code_blocks', 0)}")
                    lines.append(f"- full I/O: `llm_trace.jsonl` call #{ev['llm_call']}")
                    preview = (ev.get("output_preview") or "").strip()
                    if preview:
                        lines.append("")
                        lines.append("> " + preview.replace("\n", "\n> "))
                    lines.append("")
                else:
                    head = f"- 🛠 [{ev['seq']}] {ev.get('tool_name', 'tool')}"
                    if ev.get("block_index") is not None:
                        head += f" (block {ev['block_index']})"
                    if ev.get("n_images"):
                        head += f" · {ev['n_images']} img"
                    lines.append(head)
                    text = (ev.get("text") or "").strip()
                    if text:
                        lines.append(f"  - {text[:240]}")
            self.md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


class NullTraceLogger:
    """No-op trace logger used when no output directory is configured."""

    def log_llm(self, **kwargs: Any) -> None:  # noqa: D102
        pass

    def log_tool(self, **kwargs: Any) -> None:  # noqa: D102
        pass

    def finalize(self) -> None:  # noqa: D102
        pass
