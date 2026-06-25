"""Structured run logger for the open-drawer skill.

Goal: make the agent's data-flow legible. Each step is tagged by *kind* so we can
see, at a glance:
  - **llm**   : a DECISION boundary — where an agentic cap-x would call the LLM.
                Records the call's PURPOSE, the DATA it consumes (`inputs`), and
                what it DECIDES (`decides`). `inputs` states modality + cardinality
                so we can tell a single frame from a frame *sequence*, e.g.
                "agentview.rgb[1 frame]", "wrist.rgbd[1 frame]", "agentview[2: before/after]",
                "proprio(TCP,grip) [no frames]", "instruction(text)".
  - **local** : a LOCAL model / solver execution (SAM3, pyroki-IK, HORL trajopt/RRT) —
                NOT an LLM call. Records which model, the op, and its `inputs`/`outputs`.
  - **act**   : a robot motion (executing a planned trajectory / gripper).
  - **info**  : misc.

Our hand-written skill makes ZERO real LLM calls at run time (every decision is
deterministic code + local models). The `llm` tags therefore mark *where an agentic
version would invoke the LLM and on what data* — exactly the map needed to decide,
per decision, whether it needs a video clip or just one frame.
"""
from __future__ import annotations

import time
from collections import Counter

_TAG = {"llm": "LLM", "local": "loc", "act": "act", "info": "   "}


class RunLogger:
    def __init__(self) -> None:
        self.t0 = time.time()
        self.steps: list[dict] = []

    def _add(self, kind: str, msg: str, **extra) -> dict:
        extra = {k: v for k, v in extra.items() if v is not None}
        rec = {"i": len(self.steps), "t": round(time.time() - self.t0, 2),
               "kind": kind, "msg": msg, **extra}
        self.steps.append(rec)
        head = f"  [{_TAG.get(kind, '   ')}] {msg}"
        if "inputs" in extra:
            head += f"   <= {extra['inputs']}"
        if "decides" in extra:
            head += f"   -> {extra['decides']}"
        print(head, flush=True)
        return rec

    def __call__(self, msg) -> dict:                       # plain info line
        return self._add("info", str(msg))

    def llm(self, purpose: str, *, inputs: str | None = None,
            decides: str | None = None, detail: str | None = None) -> dict:
        return self._add("llm", detail or purpose, purpose=purpose,
                         inputs=inputs, decides=decides)

    def local(self, model: str, op: str, *, inputs: str | None = None,
              outputs: str | None = None) -> dict:
        return self._add("local", f"{model}: {op}", model=model,
                         inputs=inputs, outputs=outputs)

    def act(self, msg: str, **extra) -> dict:
        return self._add("act", str(msg), **extra)

    def summary(self) -> dict:
        kinds = Counter(s["kind"] for s in self.steps)
        return {
            "n_steps": len(self.steps),
            "n_llm_decisions": kinds.get("llm", 0),
            "n_local_model_calls": kinds.get("local", 0),
            "local_models_used": sorted({s["model"] for s in self.steps
                                         if s["kind"] == "local"}),
            # the data each would-be-LLM decision consumes — the map for agent design
            "llm_decisions": [{"t": s["t"], "purpose": s.get("purpose"),
                               "inputs": s.get("inputs"), "decides": s.get("decides")}
                              for s in self.steps if s["kind"] == "llm"],
        }


class _Adapter:
    """Wrap a plain callable (e.g. ``print``) so .llm/.local/.act still work when a
    caller passes ``log=print`` instead of a RunLogger."""

    def __init__(self, fn):
        self._fn = fn
        self.steps = []

    def __call__(self, m):
        self._fn(str(m))

    def llm(self, purpose, *, inputs=None, decides=None, detail=None):
        s = f"[LLM] {detail or purpose}"
        if inputs:
            s += f" <= {inputs}"
        if decides:
            s += f" -> {decides}"
        self._fn(s)

    def local(self, model, op, *, inputs=None, outputs=None):
        self._fn(f"[local:{model}] {op}" + (f" <= {inputs}" if inputs else ""))

    def act(self, m, **extra):
        self._fn(f"[act] {m}")

    def summary(self):
        return {}


def as_logger(log):
    """Return `log` unchanged if it's already a RunLogger, else wrap a plain callable."""
    return log if hasattr(log, "llm") else _Adapter(log)
