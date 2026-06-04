"""Modular debug harness for the self-evolve pipeline (develop-phase tooling).

The final system is automated end-to-end, but while building it we want to drive
each module **in isolation** and *see* what it read and what it produced. Every
module already takes its model/env through injected callables
(``query_fn`` / ``judge_fn``), so this harness only has to supply *inspectable*
versions of those plus a thin CLI that dumps inputs and outputs.

One subcommand per debug concern (run ``python -m capx.self_evolve.debug -h``):

- ``handoff <trial>``    — module ① output check: validate ``postprocess_handoff.json``
  and print the human feedback / final_code / chat_history it logged. Answers
  "did the feedback loop land the log correctly?". No LLM.
- ``postprocessor <handoff>`` — module ③: feed a handoff in, run generalize +
  distill with a real LLM, print final_code + digest + the written history pair.
  ``--skip-rewrite`` isolates the distill step. Answers "does it read the log and
  is the agent's output what I expect?".
- ``planner [--mem ...]`` — module ④: first print exactly what the History Reader
  tools expose over the pool (``--tools-only`` stops there), then run the
  proposer/reviewer loop and print the candidates. Answers "can the planner see
  the history pool through its tools and produce a sane plan?".

Every LLM call is mirrored to ``outputs/se_debug/<cmd>_<ts>/llm_calls.jsonl`` and
summarized on stdout, so the agent's intermediate reasoning is auditable.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import sys
import tempfile
from pathlib import Path
from typing import Any, Callable

from capx.self_evolve.handoff import find_handoffs, load_handoff
from capx.self_evolve.history_reader import HistoryReader
from capx.self_evolve.storage import MemStore, default_mem_root

QueryFn = Callable[[list[dict]], str]
JudgeFn = Callable[[str], bool]

_RULE = "─" * 72


# --------------------------------------------------------------------------- #
# small print helpers
# --------------------------------------------------------------------------- #
def _h(title: str) -> None:
    print(f"\n{_RULE}\n{title}\n{_RULE}")


def _kv(key: str, val: Any) -> None:
    print(f"  {key:<16} {val}")


def _block(text: str, *, limit: int | None = None) -> None:
    text = text if limit is None or len(text) <= limit else text[:limit] + " …[truncated]"
    for line in text.splitlines() or [""]:
        print(f"  | {line}")


# --------------------------------------------------------------------------- #
# injectable callables (the inspectable query_fn / judge_fn)
# --------------------------------------------------------------------------- #
def make_logged_query_fn(
    model: str,
    *,
    server_url: str,
    temperature: float = 0.2,
    max_tokens: int = 4096,
    log_path: Path | None = None,
    verbose: bool = True,
) -> QueryFn:
    """A real-LLM ``query_fn(messages)->str`` that records every call.

    Wraps :func:`capx.llm.client.query_model` (which returns a dict) down to the
    plain text the pipeline modules expect, while appending each
    ``{messages, content, reasoning}`` to ``log_path`` and printing a summary.
    ``fn.calls`` keeps the records in memory for assertions/inspection.
    """
    from capx.llm.client import ModelQueryArgs, query_model

    args = ModelQueryArgs(model=model, server_url=server_url,
                          temperature=temperature, max_tokens=max_tokens)
    calls: list[dict[str, Any]] = []

    def fn(messages: list[dict]) -> str:
        out = query_model(args, messages)
        content = out["content"] if isinstance(out, dict) else str(out)
        rec = {
            "n": len(calls) + 1,
            "messages": messages,
            "content": content,
            "reasoning": out.get("reasoning") if isinstance(out, dict) else None,
        }
        calls.append(rec)
        if log_path is not None:
            with log_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        if verbose:
            last = messages[-1] if messages else {}
            print(f"  · LLM call #{rec['n']} ({model}) "
                  f"in={_brief(last)} → out[{len(content)}ch]: {content[:120]!r}")
        return content

    fn.calls = calls  # type: ignore[attr-defined]
    return fn


def _brief(message: dict) -> str:
    content = message.get("content") if isinstance(message, dict) else message
    if isinstance(content, list):
        return f"{sum(len(str(p)) for p in content)}ch/{len(content)}parts"
    return f"{len(str(content))}ch"


def auto_accept_judge(verbose: bool = True) -> JudgeFn:
    """Accept every rewrite (offline: no env). Lets generalization run to its cap."""
    def fn(code: str) -> bool:
        if verbose:
            print(f"  · judge: AUTO-ACCEPT rewrite [{len(code)}ch]")
        return True
    return fn


def reject_judge(verbose: bool = True) -> JudgeFn:
    """Reject every rewrite — keeps the original code (isolates 'no generalization')."""
    def fn(code: str) -> bool:
        if verbose:
            print(f"  · judge: REJECT rewrite [{len(code)}ch]")
        return False
    return fn


def interactive_judge() -> JudgeFn:
    """Eyeball each rewrite in the terminal and answer y/n (no env execution)."""
    def fn(code: str) -> bool:
        print("\n  --- proposed rewrite ---")
        _block(code)
        ans = input("  accept this rewrite? [y/N] ").strip().lower()
        return ans in ("y", "yes")
    return fn


def _judge_from_name(name: str) -> JudgeFn:
    return {"auto": auto_accept_judge(), "reject": reject_judge(),
            "interactive": interactive_judge()}[name]


def _session_dir(cmd: str) -> Path:
    ts = _dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    d = Path("outputs") / "se_debug" / f"{cmd}_{ts}"
    d.mkdir(parents=True, exist_ok=True)
    return d


# --------------------------------------------------------------------------- #
# cmd: handoff  (module ① output / feedback-loop log check)
# --------------------------------------------------------------------------- #
def cmd_handoff(args: argparse.Namespace) -> int:
    target = args.target or _newest_handoff(args.logs_root)
    if target is None:
        print(f"No postprocess_handoff.json found under {args.logs_root}", file=sys.stderr)
        return 2
    ho = load_handoff(target)  # raises with a precise reason if the contract is broken

    _h(f"HANDOFF CONTRACT CHECK · {ho.path}")
    _kv("task", ho.short_task)
    _kv("datetime", ho.datetime)
    _kv("success", ho.success)
    _kv("settings", ho.settings)
    _kv("chat turns", len(ho.chat_history))
    _kv("human feedback", f"{len(ho.human_feedback)} turn(s)")

    _h("HUMAN FEEDBACK (verbatim)")
    if not ho.human_feedback:
        print("  (none — model finished without feedback, or feedback was not logged)")
    for fb in ho.human_feedback:
        _kv(f"[#{fb.get('index')}] att{fb.get('attempt')}", "")
        _block(str(fb.get("text", "")))

    _h("FINAL CODE (human-confirmed, pre-generalization)")
    _block(ho.final_code)

    if args.show_chat:
        _h("CHAT HISTORY (role · phase/tool · content)")
        for e in ho.chat_history:
            tag = e.get("phase") or e.get("tool_name") or ""
            c = e.get("content")
            c = "" if c is None else (c if isinstance(c, str) else str(c))
            print(f"  [#{e.get('index')}] {e.get('role'):16s} {tag:18s} {c[:90].replace(chr(10), ' ')}")

    print(f"\n✅ contract OK — usable by `debug postprocessor {ho.path.parent.name}`")
    return 0


def _newest_handoff(logs_root: str) -> Path | None:
    hits = find_handoffs(logs_root)
    return hits[0] if hits else None


# --------------------------------------------------------------------------- #
# cmd: postprocessor  (module ③)
# --------------------------------------------------------------------------- #
def cmd_postprocessor(args: argparse.Namespace) -> int:
    from capx.self_evolve.config import ExperienceDistillConfig
    from capx.self_evolve.feedback_postprocessor import run_feedback_postprocessor

    ho = load_handoff(args.handoff)
    kwargs = ho.postprocessor_kwargs()
    rounds = 0 if args.skip_rewrite else args.rewrite_rounds

    api_ref = kwargs.get("api_reference")
    _h("POSTPROCESSOR INPUTS (from handoff)")
    _kv("task", kwargs["task"])
    _kv("settings", kwargs["settings"])
    _kv("chat turns", len(kwargs["chat_history"]))
    _kv("feedback", ho.feedback_texts())
    _kv("api_reference", f"{len(api_ref)} chars" if api_ref else "MISSING — (B) re-derivation will degrade")
    _kv("rewrite rounds", f"{rounds} ({'skipped' if rounds == 0 else 'judge=' + args.judge})")
    if rounds > 0 and args.judge == "auto":
        print("  ⚠ offline --judge auto rubber-stamps every rewrite (no env). It shows the "
              "agent's\n    refine DIRECTION, not a validated result — real (B) re-derivation "
              "needs env-in-the-loop.")

    sess = _session_dir("postprocessor")
    query_fn = make_logged_query_fn(
        args.model, server_url=args.server_url, log_path=sess / "llm_calls.jsonl"
    )
    judge_fn = _judge_from_name(args.judge)

    # Dry-run by default: write into a throwaway mem/ so the real pool is untouched.
    mem_root = Path(args.mem) if args.mem else Path(tempfile.mkdtemp(prefix="se_debug_mem_"))
    store = MemStore(mem_root)

    _h(f"RUNNING module ③  (mem={mem_root}{'  [DRY temp]' if not args.mem else ''})")
    result = run_feedback_postprocessor(
        query_fn=query_fn, judge_fn=judge_fn, store=store,
        config=ExperienceDistillConfig(), max_rewrite_rounds=rounds, **kwargs,
    )

    _h("OUTPUT · final_code")
    _block(result.final_code)
    _h(f"OUTPUT · digest ({result.digest.word_count()} words)")
    _block(result.digest.render())
    _h("WRITTEN history pair")
    _kv("history_id", result.history_id)
    _kv("json", store.history_pool / f"{result.history_id}.json")
    _kv("digest", store.history_pool / f"{result.history_id}.digest.md")
    _kv("llm calls", sess / "llm_calls.jsonl")
    return 0


# --------------------------------------------------------------------------- #
# cmd: planner  (module ④)
# --------------------------------------------------------------------------- #
def cmd_planner(args: argparse.Namespace) -> int:
    store = MemStore(Path(args.mem) if args.mem else default_mem_root())
    reader = HistoryReader(store)

    _h(f"HISTORY READER TOOLS over pool  (mem={store.root})")
    unprocessed = reader.list_unprocessed()
    _kv("list_unprocessed()", f"{len(unprocessed)} item(s)")
    for item in unprocessed:
        print(f"    - {item.get('id')}  ({item.get('task')}, {item.get('datetime')})")
    for item in unprocessed:
        _h(f"read_digest({item.get('id')!r})")
        try:
            _block(reader.read_digest(item["id"]))
        except Exception as exc:  # noqa: BLE001 - debug surface
            print(f"  ⚠ {exc}")

    if not unprocessed:
        print("\n(no unprocessed history — nothing for the planner to plan)")
        return 0
    if args.tools_only:
        print("\n--tools-only: stopping before the LLM proposer/reviewer loop.")
        return 0

    from capx.self_evolve.config import UpdatePlannerConfig
    from capx.self_evolve.update_planner import run_update_planner

    sess = _session_dir("planner")
    query_fn = make_logged_query_fn(
        args.model, server_url=args.server_url, log_path=sess / "llm_calls.jsonl"
    )
    _h("RUNNING module ④  (proposer ↔ reviewer)")
    result = run_update_planner(
        store, query_fn, config=UpdatePlannerConfig(), force=args.force
    )

    _h("OUTPUT · update planner result")
    _kv("triggered", result.triggered)
    _kv("processed", result.processed_ids)
    _kv("written", result.written)
    _kv("revise rounds", result.revise_rounds)
    _kv("reviewer feedback", (result.rejected_feedback or "")[:300])
    for name in result.written:
        _h(f"candidate · {name}")
        _kv("stats", store.read_candidate_stats(name).to_dict())
        _block(store.read_candidate_code(name), limit=1200)
    _kv("llm calls", sess / "llm_calls.jsonl")
    return 0


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m capx.self_evolve.debug",
        description="Modular debug harness for the self-evolve pipeline.",
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    ph = sub.add_parser("handoff", help="validate + dump a postprocess_handoff.json (no LLM)")
    ph.add_argument("target", nargs="?", help="trial dir or handoff json (default: newest under --logs-root)")
    ph.add_argument("--logs-root", default="logs", help="where to look for the newest handoff")
    ph.add_argument("--show-chat", action="store_true", help="also dump the full chat_history index")
    ph.set_defaults(func=cmd_handoff)

    pp = sub.add_parser("postprocessor", help="run module ③ on a handoff with a real LLM")
    pp.add_argument("handoff", help="trial dir or handoff json")
    pp.add_argument("--model", default="openrouter/google/gemini-2.5-pro-preview")
    pp.add_argument("--server-url", default="http://localhost:8000/v1/chat/completions",
                    help="ignored for openrouter/* models (routed to the proxy)")
    pp.add_argument("--judge", choices=["auto", "reject", "interactive"], default="auto",
                    help="rewrite judge stand-in (offline: no env execution)")
    pp.add_argument("--rewrite-rounds", type=int, default=3)
    pp.add_argument("--skip-rewrite", action="store_true", help="distill-only (rewrite rounds=0)")
    pp.add_argument("--mem", help="mem/ root to write into (default: throwaway temp = dry run)")
    pp.set_defaults(func=cmd_postprocessor)

    pl = sub.add_parser("planner", help="inspect History Reader tools + run module ④")
    pl.add_argument("--mem", help="mem/ root to read (default: repo mem/)")
    pl.add_argument("--tools-only", action="store_true", help="only show what the reader tools expose")
    pl.add_argument("--force", action="store_true", help="run even if below trigger_history_count")
    pl.add_argument("--model", default="openrouter/google/gemini-2.5-pro-preview")
    pl.add_argument("--server-url", default="http://localhost:8000/v1/chat/completions")
    pl.set_defaults(func=cmd_planner)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
