"""Run ONE arm of the track-judge experiment on libero-pro and persist the results.

    python track_judge/benchmark.py --arm tracking --suites a,b,c --gen 7 \
        [--max-tasks-per-suite M] [--repeat R] [--output DIR]

What it does (the self-evolve DECISION step, AGENT.md §6.4):
  * flips ``state_judge`` to the chosen arm (``vdm`` = baseline A, ``tracking`` = arm B) on
    ``env_configs/libero/franka_libero_track_judge.yaml`` and drives the FROZEN
    ``run_libero_batch`` / ``launch.main`` path with it,
  * collects every trial's ``track_judge_trace.json`` (written by the runtime, agent B) plus
    each suite's ``summaries.txt`` (ground-truth success, written by
    ``launch_utils._print_and_save_summary``),
  * aggregates into ``results/genNNN/summary.json`` (the schema make_figs/compare read) and a
    readable ``report.md``, then auto-diffs against ``results/BEST``.

REQUIRED PRE-GEN1 STEP — produce the VDM baseline (arm A) the whole experiment compares to:

    python track_judge/benchmark.py --baseline --suites \
        libero_object_swap,libero_object_task,libero_goal_swap,libero_goal_task,\
libero_spatial_swap,libero_spatial_task

  ``--baseline`` is shorthand for ``--arm vdm --out baseline_vdm``. It MUST land in
  ``results/baseline_vdm/summary.json`` because both ``compare.py`` and
  ``paper/make_figs.py`` read EXACTLY that path as the comparison basis — there is no
  comparison without it. Re-measure it only when the suites change (AGENT.md §2).

Output naming: ``--out NAME`` writes to ``results/<NAME>/`` (default ``gen<N>`` from --gen,
or ``baseline_<arm>`` if neither is given). ``--output DIR`` overrides with an explicit path.

Honesty (AGENT.md §7): if trial traces are missing (e.g. a dry run with no GPU/LLM proxy),
fields that need them degrade to ``null`` — we never fabricate numbers. What was capped
(suites / max-tasks / repeat) is recorded in the summary so a partial run can't read as full.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from statistics import median
from typing import Any, Dict, List, Optional

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
RESULTS_DIR = os.path.join(HERE, "results")
BEST_PIN = os.path.join(RESULTS_DIR, "BEST")
CONFIG = os.path.join(REPO, "env_configs", "libero", "franka_libero_track_judge.yaml")
TRACE_NAME = "track_judge_trace.json"

# Held-out suites: NEVER tune on these (AGENT.md §7). Reported with held_out=True so a
# generalisation check is visibly distinct from the dev loop.
HELD_OUT_SUITES = {"libero_spatial_swap", "libero_spatial_task"}

DEFAULT_SUITES = [
    "libero_object_swap", "libero_object_task",
    "libero_goal_swap", "libero_goal_task",
    "libero_spatial_swap", "libero_spatial_task",
]


def _git_hash() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=HERE,
            stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "unknown"


def read_best_pin() -> Optional[str]:
    if not os.path.exists(BEST_PIN):
        return None
    try:
        return open(BEST_PIN).read().strip() or None
    except Exception:
        return None


def _median(xs: List[float]) -> Optional[float]:
    xs = [x for x in xs if x is not None]
    return float(median(xs)) if xs else None


def _p90(xs: List[float]) -> Optional[float]:
    xs = sorted(x for x in xs if x is not None)
    if not xs:
        return None
    k = max(0, int(round(0.9 * (len(xs) - 1))))
    return float(xs[k])


def _mean(xs: List[float]) -> Optional[float]:
    xs = [x for x in xs if x is not None]
    return float(sum(xs) / len(xs)) if xs else None


# ── parse a suite's summaries.txt (ground-truth success_rate) ───────────────────

_RATE_RE = re.compile(r"^\s*([0-9.]+)\s*/\s*([0-9.]+)\s*/\s*([0-9.]+)")


def parse_summaries_txt(path: str) -> Optional[Dict[str, Any]]:
    """Return {'success_rate', 'n_trials'} parsed from a suite's summaries.txt, or None."""
    if not os.path.exists(path):
        return None
    try:
        lines = open(path).read().splitlines()
    except Exception:
        return None
    rate = n_trials = None
    for i, ln in enumerate(lines):
        if ln.startswith("Total number of trials:"):
            try:
                n_trials = int(ln.split(":", 1)[1].strip())
            except Exception:
                pass
        if "success rate / Average reward / Task completed" in ln and i + 1 < len(lines):
            m = _RATE_RE.match(lines[i + 1])
            if m:
                rate = float(m.group(1))
    if rate is None and n_trials is None:
        return None
    return {"success_rate": rate, "n_trials": n_trials}


def discover(output_dir: str, suites: List[str]) -> Dict[str, Any]:
    """Walk the run output tree, collect per-suite summaries + every trace.

    The batch writes ``<output_dir>/<suite>/<task>/...`` with a ``summaries.txt`` and
    per-trial dirs (which agent B drops a ``track_judge_trace.json`` into)."""
    per_suite_runs: Dict[str, List[Dict[str, Any]]] = {s: [] for s in suites}
    traces: List[Dict[str, Any]] = []
    for suite in suites:
        sdir = os.path.join(output_dir, suite)
        if not os.path.isdir(sdir):
            continue
        for root, _dirs, files in os.walk(sdir):
            if "summaries.txt" in files:
                parsed = parse_summaries_txt(os.path.join(root, "summaries.txt"))
                if parsed is not None:
                    per_suite_runs[suite].append(parsed)
            if TRACE_NAME in files:
                try:
                    tr = json.load(open(os.path.join(root, TRACE_NAME)))
                    tr.setdefault("suite", suite)
                    traces.append(tr)
                except Exception:
                    pass
    return {"per_suite_runs": per_suite_runs, "traces": traces}


# ── aggregation into the summary.json schema ────────────────────────────────────

def _per_turn(traces: List[Dict[str, Any]], key: str) -> List[float]:
    """Flatten a per-turn field across all traces (each trace has a 'turns' list)."""
    out: List[float] = []
    for tr in traces:
        for turn in tr.get("turns", []) or []:
            v = turn.get(key)
            if v is not None:
                out.append(float(v))
    return out


def _suite_success(runs: List[Dict[str, Any]]) -> Optional[float]:
    rates = [r["success_rate"] for r in runs if r.get("success_rate") is not None]
    ns = [r["n_trials"] for r in runs if r.get("n_trials")]
    if not rates:
        return None
    if ns and len(ns) == len(rates):  # trial-weighted mean across tasks
        tot = sum(ns)
        return float(sum(r * n for r, n in zip(rates, ns)) / tot) if tot else _mean(rates)
    return _mean(rates)


def aggregate(arm: str, gen: Optional[int], suites: List[str],
              found: Dict[str, Any]) -> Dict[str, Any]:
    traces = found["traces"]
    per_suite_runs = found["per_suite_runs"]

    # trial-level
    gt = [bool(t.get("gt_success")) for t in traces if t.get("gt_success") is not None]
    wall = [t.get("wallclock_s") for t in traces]
    vlm_per_turn = _per_turn(traces, "vlm_calls") if traces else []

    overall_success = _mean([1.0 if g else 0.0 for g in gt]) if gt else None
    overall = {
        "success_rate": overall_success,
        "task_time_median_s": _median(wall),
        "judge_latency_s": _median(_per_turn(traces, "judge_s")),
        "vlm_calls_per_turn": _mean(vlm_per_turn) if vlm_per_turn else None,
    }

    timing = {
        "codegen_s_median": _median(_per_turn(traces, "codegen_s")),
        "exec_s_median": _median(_per_turn(traces, "exec_s")),
        "judge_s_median": _median(_per_turn(traces, "judge_s")),
        "track_s_median": _median(_per_turn(traces, "track_s")),
        "judge_s_p90": _p90(_per_turn(traces, "judge_s")),
    }

    diagnostics = _diagnostics(traces)

    per_suite: Dict[str, Any] = {}
    for s in suites:
        runs = per_suite_runs.get(s, [])
        s_traces = [t for t in traces if t.get("suite") == s]
        sr = _suite_success(runs)
        if sr is None and s_traces:  # fall back to trace GT if summaries.txt missing
            sg = [1.0 if t.get("gt_success") else 0.0 for t in s_traces
                  if t.get("gt_success") is not None]
            sr = _mean(sg) if sg else None
        per_suite[s] = {
            "success_rate": sr,
            "task_time_median_s": _median([t.get("wallclock_s") for t in s_traces]),
            "held_out": s in HELD_OUT_SUITES,
        }

    return {
        "arm": arm,
        "gen": gen,
        "git_hash": _git_hash(),
        "n_trials": len(traces),
        "overall": overall,
        "timing": timing,
        "diagnostics": diagnostics,
        "per_suite": per_suite,
    }


def _diagnostics(traces: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Arm-B confusion of the judge's final 'done' vs ground truth, + failure buckets.

    judge_final_done = the judge said the task was complete at the end of the trial.
    gt_success       = libero check_success() ground truth.
    """
    tp = fp = miss = tn = 0
    cats: Dict[str, int] = {}
    aborts = finishes = 0
    have_judge = False
    for t in traces:
        jd = t.get("judge_final_done")
        gs = t.get("gt_success")
        if jd is not None and gs is not None:
            have_judge = True
            jd, gs = bool(jd), bool(gs)
            if jd and gs:
                tp += 1
            elif jd and not gs:
                fp += 1
            elif (not jd) and gs:
                miss += 1
            else:
                tn += 1
        cat = t.get("failure_category")
        if cat:
            cats[cat] = cats.get(cat, 0) + 1
        # abort vs finish: judge_final_done True => finished; explicit abort flag wins
        if t.get("aborted"):
            aborts += 1
        elif jd:
            finishes += 1
    return {
        "judge_vs_gt": ({"true_done": tp, "false_done": fp,
                         "missed_done": miss, "true_notdone": tn} if have_judge else None),
        "failure_categories": cats,
        "n_aborts": aborts,
        "n_finishes": finishes,
    }


# ── run the eval ────────────────────────────────────────────────────────────────

def _patch_state_judge(arm: str) -> None:
    """Flip `state_judge:` in the shared config to the chosen arm (in place)."""
    lines = open(CONFIG).read().splitlines()
    for i, ln in enumerate(lines):
        if re.match(r"^\s*state_judge\s*:", ln):
            lines[i] = f"state_judge: {arm}"
            break
    with open(CONFIG, "w") as f:
        f.write("\n".join(lines) + "\n")


def run_eval(arm: str, suites: List[str], output_dir: str,
             max_tasks_per_suite: Optional[int], repeat: int) -> int:
    """Drive run_libero_batch for `arm`. `repeat` re-runs the batch (noise control)."""
    _patch_state_judge(arm)
    py = os.path.join(REPO, ".venv", "bin", "python")
    py = py if os.path.exists(py) else sys.executable
    rc = 0
    for r in range(max(1, repeat)):
        cmd = [py, os.path.join(REPO, "capx", "envs", "scripts", "run_libero_batch.py"),
               "--base-config-path", CONFIG,
               "--suites", *suites,
               "--output-dir", output_dir,
               "--record-video", "True"]
        if max_tasks_per_suite is not None:
            cmd += ["--max-tasks-per-suite", str(max_tasks_per_suite)]
        print(f"[benchmark] eval arm={arm} repeat {r + 1}/{repeat}: {' '.join(cmd)}")
        rc = subprocess.call(cmd, cwd=REPO)
        if rc != 0:
            print(f"[benchmark] WARNING: batch exited {rc} (repeat {r + 1})")
    return rc


# ── persistence + report ─────────────────────────────────────────────────────────

def _persist(out_dir: str, summary: Dict[str, Any]) -> None:
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    with open(os.path.join(out_dir, "report.md"), "w") as f:
        f.write(render_report(summary))
    print(f"[benchmark] wrote {out_dir}/summary.json + report.md")


def _f(v, fmt="%.3f"):
    return "—" if v is None else (fmt % v)


def render_report(s: Dict[str, Any]) -> str:
    o, t, d = s["overall"], s["timing"], s["diagnostics"]
    L = [f"# track-judge benchmark — arm `{s['arm']}` · gen {s.get('gen')}\n",
         f"- git `{s['git_hash']}` · n_trials **{s['n_trials']}**", ""]
    if s["n_trials"] == 0:
        L.append("> ⚠ NO trial traces found — this is a dry/partial run. Headline fields are "
                 "null; numbers were NOT fabricated.\n")
    L += ["## Headline (success + speed)",
          "| success rate | median task time (s) | judge latency (s) | VLM calls/turn |",
          "|---|---|---|---|",
          f"| {_f(o['success_rate'])} | {_f(o['task_time_median_s'], '%.1f')} "
          f"| {_f(o['judge_latency_s'])} | {_f(o['vlm_calls_per_turn'])} |\n",
          "## Per-step timing profile (median, s)",
          "| codegen | exec | judge | track | judge p90 |",
          "|---|---|---|---|---|",
          f"| {_f(t['codegen_s_median'])} | {_f(t['exec_s_median'])} "
          f"| {_f(t['judge_s_median'])} | {_f(t['track_s_median'])} "
          f"| {_f(t['judge_s_p90'])} |\n",
          "## Diagnostics"]
    jvg = d.get("judge_vs_gt")
    if jvg:
        L += ["judge-vs-GT confusion (arm B):",
              "| true_done | false_done | missed_done | true_notdone |",
              "|---|---|---|---|",
              f"| {jvg['true_done']} | {jvg['false_done']} | {jvg['missed_done']} "
              f"| {jvg['true_notdone']} |\n"]
    else:
        L.append("_judge-vs-GT confusion: n/a (no per-trial judge verdicts)._\n")
    fc = d.get("failure_categories") or {}
    L.append("failure categories: " +
             (", ".join(f"`{k}`={v}" for k, v in sorted(fc.items())) or "—"))
    L.append(f"aborts={d['n_aborts']} · finishes={d['n_finishes']}\n")
    L += ["## Per-suite",
          "| suite | success | median time (s) | held-out |",
          "|---|---|---|---|"]
    for name, ps in s["per_suite"].items():
        L.append(f"| {name} | {_f(ps['success_rate'])} "
                 f"| {_f(ps['task_time_median_s'], '%.1f')} "
                 f"| {'yes' if ps['held_out'] else ''} |")
    return "\n".join(L) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="track-judge benchmark (one arm, one gen)")
    ap.add_argument("--arm", choices=["vdm", "tracking"], default=None)
    ap.add_argument("--baseline", action="store_true",
                    help="shorthand for --arm vdm --out baseline_vdm (the REQUIRED pre-gen1 "
                         "VDM baseline arm A; lands in results/baseline_vdm/)")
    ap.add_argument("--suites", type=str, default=",".join(DEFAULT_SUITES),
                    help="comma list of libero-pro suites")
    ap.add_argument("--gen", type=int, default=None, help="generation index")
    ap.add_argument("--out", type=str, default=None,
                    help="results/<NAME> subdir name (default gen<NNN> or baseline_<arm>)")
    ap.add_argument("--max-tasks-per-suite", type=int, default=None)
    ap.add_argument("--repeat", type=int, default=1, help="re-runs of the batch (noise)")
    ap.add_argument("--output", type=str, default=None,
                    help="explicit results dir path (overrides --out)")
    ap.add_argument("--no-eval", action="store_true",
                    help="skip launching the eval; only aggregate an existing output tree "
                         "(used by tests / re-aggregation)")
    ap.add_argument("--eval-output", type=str, default=None,
                    help="run output tree to read traces/summaries from "
                         "(default ./outputs/track_judge_gen<NNN>)")
    args = ap.parse_args(argv)

    arm = args.arm
    out_name = args.out
    if args.baseline:  # shorthand: VDM baseline arm A -> results/baseline_vdm/
        arm = arm or "vdm"
        out_name = out_name or "baseline_vdm"
    if arm is None:
        ap.error("--arm is required (or pass --baseline)")

    suites = [s.strip() for s in args.suites.split(",") if s.strip()]
    gen = args.gen
    tag = out_name or (f"gen{gen:03d}" if gen is not None else f"baseline_{arm}")
    args.arm = arm  # used downstream
    out_dir = args.output or os.path.join(RESULTS_DIR, tag)
    eval_output = args.eval_output or os.path.join(REPO, "outputs", f"track_judge_{tag}")

    if not args.no_eval:
        run_eval(args.arm, suites, eval_output, args.max_tasks_per_suite, args.repeat)

    found = discover(eval_output, suites)
    summary = aggregate(args.arm, gen, suites, found)
    summary["dropped"] = {  # what was capped — a partial run must not read as full
        "max_tasks_per_suite": args.max_tasks_per_suite,
        "repeat": args.repeat,
        "suites_run": suites,
        "held_out_suites": sorted(HELD_OUT_SUITES),
        "eval_output": eval_output,
    }
    summary["timestamp"] = time.strftime("%Y-%m-%d %H:%M:%S")
    _persist(out_dir, summary)
    print("\n" + render_report(summary))

    # auto-diff vs BEST.
    best = read_best_pin()
    if best and best != tag and os.path.exists(os.path.join(RESULTS_DIR, best, "summary.json")):
        print(f"[benchmark] diffing gen vs BEST ({best}):")
        from compare import compare_files  # local import; same dir on sys.path
        print(compare_files(os.path.join(RESULTS_DIR, best, "summary.json"),
                            os.path.join(out_dir, "summary.json")))
    else:
        print(f"[benchmark] no BEST pin to diff against (BEST={best}).")
    return 0


if __name__ == "__main__":
    sys.path.insert(0, HERE)  # make `import compare` work when run as a script
    raise SystemExit(main())
