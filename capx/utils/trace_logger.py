"""Structured per-trial trace of agent <-> LLM interactions and tool-call flow.

The trace is split **per attempt** so each reset-and-retry round has its own
self-contained conversation log. A trial dir holds one subfolder per attempt::

    trial_XX/
      attempt_00/{llm_trace.jsonl, events.jsonl, trace.md, trace_images/}
      attempt_01/...

with these three artifacts in each attempt folder:

  - ``llm_trace.jsonl`` : one JSON record per LLM call — phase, turn, model, the
    *full* input messages, and the model's full output (content + reasoning),
    plus the parsed decision / extracted code blocks when known. Inline image
    data-URLs in the input are stripped out (saved as PNGs under
    ``trace_images/``) so the file stays small and ``jq``-queryable.
  - ``events.jsonl``    : a chronological merge of LLM calls (one summary line
    each) and tool-execution steps (from ``execution_logger``), so the
    tool-call flow and the conversation interleave in real execution order.
  - ``trace.md``        : a human-readable rendering of ``events.jsonl``, written
    when the attempt rolls over (``new_attempt``) or the trial finalizes.

The attempt index is kept in lockstep with the viser ``frame_history`` segments
(one segment per attempt) by the trial runner: it calls :meth:`new_attempt` at
the same point a reset starts a fresh segment, so ``attempt_NN/`` here and
``attempt_NN/observations.npz`` (the viser history) refer to the same attempt.

Usage::

    trace = TraceLogger(trial_dir, model=args.model)   # opens attempt_00/
    trace.log_llm(phase="initial", turn=1, input_messages=msgs,
                  output_content=text, output_reasoning=reasoning)
    trace.log_tool(tool_name="SAM3", text="...", block_index=0, n_images=1)
    trace.new_attempt()   # on reset/retry -> rolls over to attempt_01/
    trace.finalize()      # writes the current attempt's trace.md

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
        self.base_dir = Path(trial_dir)
        self.default_model = model
        self._lock = threading.Lock()
        self._attempt = 0
        self._open_attempt()

    def _open_attempt(self) -> None:
        """Point the trace artifacts at ``attempt_{N}/`` and reset per-attempt
        counters. Starts fresh so a re-run of the same dir doesn't append to
        stale files (images are overwritten in place by name)."""
        self.dir = self.base_dir / f"attempt_{self._attempt:02d}"
        self.dir.mkdir(parents=True, exist_ok=True)
        self.img_dir = self.dir / "trace_images"
        self.llm_path = self.dir / "llm_trace.jsonl"
        self.events_path = self.dir / "events.jsonl"
        self.md_path = self.dir / "trace.md"

        self._seq = 0          # global event ordinal (per attempt)
        self._llm_seq = 0      # LLM-call ordinal (per attempt)
        self._events: list[dict[str, Any]] = []

        for p in (self.llm_path, self.events_path, self.md_path):
            try:
                p.unlink()
            except FileNotFoundError:
                pass

    def new_attempt(self) -> None:
        """Roll over to a fresh attempt subfolder, retaining prior ones.

        Renders the current attempt's ``trace.md`` first, then advances the
        attempt index and reopens the artifacts under the new ``attempt_NN/``.
        The runner calls this in lockstep with the viser ``frame_history``
        ``new_segment`` so attempt indices match across both."""
        with self._lock:
            self._finalize_unlocked()
            self._attempt += 1
            self._open_attempt()

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

    def log_human(self, *, kind: str, text: str = "", turn: int | None = None) -> None:
        """Record a human action as a first-class event.

        ``kind`` is ``"feedback"`` (verbatim operator feedback that triggers a
        reset-and-retry) or ``"finish"`` (the human-confirmed success signal —
        the *only* success signal in interactive mode). Logging these as events
        means the Feedback Postprocessor can read the human turns and the
        success marker straight from the trace, instead of reverse-parsing them
        out of LLM prompt scaffolding."""
        with self._lock:
            self._seq += 1
            self._record_event({
                "seq": self._seq,
                "ts": _now_iso(),
                "type": "human",
                "kind": kind,
                "turn": turn,
                "text": text,
            })

    def finalize(self) -> None:
        """Render the current attempt's ``events`` into a readable ``trace.md``."""
        with self._lock:
            self._finalize_unlocked()

    def _finalize_unlocked(self) -> None:
        lines = [f"# Attempt {self._attempt:02d} trace", ""]
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
            elif ev.get("type") == "human":
                kind = ev.get("kind", "?")
                lines.append(f"## [{ev['seq']}] 🧑 HUMAN · {kind}")
                text = (ev.get("text") or "").strip()
                if text:
                    lines.append("")
                    lines.append("> " + text.replace("\n", "\n> "))
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

    # ------------------------------------------------------- postprocess handoff
    def _read_jsonl(self, path: Path) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        try:
            with path.open(encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        out.append(json.loads(line))
        except FileNotFoundError:
            pass
        return out

    def _build_chat_history(self, task: str) -> list[dict[str, Any]]:
        """Reconstruct one linear, sourced conversation across every attempt.

        Walks each ``attempt_NN/`` in order and merges its ``events.jsonl``
        (chronological LLM + tool + human steps) with the *full* LLM output
        pulled from the same attempt's ``llm_trace.jsonl``. Every entry gets a
        stable 0-based ``index`` — this is the ``[#message_index]`` the digest
        step sources against (see docs-se/storage.md)."""
        chat: list[dict[str, Any]] = [{
            "index": 0, "role": "task", "attempt": None, "content": task,
        }]
        attempt_dirs = sorted(
            d for d in self.base_dir.glob("attempt_*") if d.is_dir()
        )
        for adir in attempt_dirs:
            attempt = int(adir.name.split("_")[-1])
            # llm_call -> full output content (events.jsonl only keeps a preview)
            out_by_call: dict[int, dict[str, Any]] = {}
            for rec in self._read_jsonl(adir / "llm_trace.jsonl"):
                out_by_call[rec.get("llm_call")] = rec.get("output") or {}
            for ev in self._read_jsonl(adir / "events.jsonl"):
                etype = ev.get("type")
                if etype == "llm":
                    out = out_by_call.get(ev.get("llm_call"), {})
                    chat.append({
                        "index": len(chat), "role": "assistant",
                        "attempt": attempt, "phase": ev.get("phase"),
                        "turn": ev.get("turn"),
                        "content": out.get("content"),
                        "reasoning": out.get("reasoning"),
                    })
                elif etype == "tool":
                    chat.append({
                        "index": len(chat), "role": "tool", "attempt": attempt,
                        "tool_name": ev.get("tool_name"),
                        "content": ev.get("text"),
                    })
                elif etype == "human":
                    chat.append({
                        "index": len(chat),
                        "role": f"human_{ev.get('kind', 'action')}",
                        "attempt": attempt, "content": ev.get("text"),
                    })
        return chat

    def write_handoff(
        self,
        *,
        task: str,
        settings: dict[str, Any],
        final_code: str,
        success_attempt: int | None = None,
        filename: str = "postprocess_handoff.json",
    ) -> Path:
        """Write the Feedback Postprocessor's input contract at the trial root.

        Assembles the human-confirmed success trial into one JSON aligned with
        the success-log schema (docs-se/storage.md): ``task`` / ``settings`` /
        ``final_code`` (the human-confirmed code, *pre*-generalization) /
        ``chat_history`` (full conversation incl. the verbatim human feedback) /
        ``datetime``, plus an explicit ``success`` marker and a convenience
        ``human_feedback`` list. The postprocessor reads this directly — it does
        not have to re-parse per-attempt traces. Returns the written path."""
        chat = self._build_chat_history(task)
        if success_attempt is None:
            finishes = [e for e in chat if e["role"] == "human_finish"]
            if finishes:
                success_attempt = finishes[-1]["attempt"]
        human_feedback = [
            {"index": e["index"], "attempt": e["attempt"], "text": e["content"]}
            for e in chat if e["role"] == "human_feedback"
        ]
        handoff = {
            "schema": "postprocess_handoff/v1",
            "task": task,
            "settings": settings,
            "success": {"signal": "human_finished", "attempt": success_attempt},
            "final_code": final_code,
            "human_feedback": human_feedback,
            "chat_history": chat,
            "datetime": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }
        path = self.base_dir / filename
        path.write_text(
            json.dumps(handoff, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return path


class NullTraceLogger:
    """No-op trace logger used when no output directory is configured."""

    def log_llm(self, **kwargs: Any) -> None:  # noqa: D102
        pass

    def log_tool(self, **kwargs: Any) -> None:  # noqa: D102
        pass

    def log_human(self, **kwargs: Any) -> None:  # noqa: D102
        pass

    def new_attempt(self) -> None:  # noqa: D102
        pass

    def finalize(self) -> None:  # noqa: D102
        pass

    def write_handoff(self, **kwargs: Any) -> Any:  # noqa: D102
        return None
