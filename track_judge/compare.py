"""Accept/reject verdict engine for the track-judge experiment (AGENT.md §6.4, §7).

    python track_judge/compare.py genA genB        # reads results/genX/summary.json
    python track_judge/compare.py results/baseline_vdm/summary.json results/gen007/summary.json

genB is the candidate; genA is the baseline to beat (the current BEST, or baseline_vdm).

Verdict rules (tiered):
  * GATED axes (can REJECT):
      - success_rate     — tolerance ~1–2 trials' worth (SUCCESS_TOL, an absolute rate).
      - task_time_median_s — tolerance ~5% relative (TIME_REL_TOL).
  * HARD gate (auto-REJECT regardless of numbers):
      - either run crashed / produced no trials, OR
      - arm-B `diagnostics.judge_vs_gt` is degenerate: the judge is gaming, not judging —
        e.g. it calls (almost) everything "done" (false_done dominates, true_notdone≈0), so
        it carries no information. That borrows the baseline's behaviour instead of judging.
  * ACCEPT iff: success ≥ baseline within tol AND time ≤ baseline within tol AND at least one
    of {success, time} STRICTLY improves (beyond tol). Otherwise REJECT.
  * Informational (printed, never gate): the per-step timing profile + judge-vs-GT deltas.

`verdict(a_summary, b_summary) -> {"accept": bool, "reasons": [...]}` is importable by tests.
"""

from __future__ import annotations

import json
import os
import sys
from typing import Any, Dict, List, Optional

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(HERE, "results")

SUCCESS_TOL = 0.04   # ~1–2 trials' worth of success-rate noise (absolute)
TIME_REL_TOL = 0.05  # 5% relative on median task time
# A judge is degenerate if it almost never says "not done" while a real fraction of trials
# are incomplete — i.e. it has no discriminative power and is gaming the gate.
DEGEN_NOTDONE_FRAC = 0.05   # < 5% true_notdone among all judged trials
DEGEN_FALSEDONE_FRAC = 0.50  # AND > 50% of "done" verdicts are wrong


def _load(path_or_gen: str) -> Dict[str, Any]:
    p = path_or_gen
    if not p.endswith(".json"):
        p = os.path.join(RESULTS_DIR, p, "summary.json")
    with open(p) as f:
        return json.load(f)


def _crashed(s: Dict[str, Any]) -> bool:
    return int(s.get("n_trials", 0) or 0) == 0 or s.get("overall", {}).get("success_rate") is None


def _degenerate_judge(s: Dict[str, Any]) -> bool:
    """True if arm-B's judge confusion shows it just says 'done' for everything."""
    jvg = (s.get("diagnostics") or {}).get("judge_vs_gt")
    if not jvg:
        return False  # no judge verdicts (e.g. VDM arm) — not applicable
    total = sum(int(jvg.get(k, 0)) for k in
                ("true_done", "false_done", "missed_done", "true_notdone"))
    if total == 0:
        return False
    done = jvg.get("true_done", 0) + jvg.get("false_done", 0)
    notdone_frac = (jvg.get("true_notdone", 0) + jvg.get("missed_done", 0)) / total
    false_done_frac = (jvg.get("false_done", 0) / done) if done else 0.0
    return notdone_frac < DEGEN_NOTDONE_FRAC and false_done_frac > DEGEN_FALSEDONE_FRAC


def verdict(a: Dict[str, Any], b: Dict[str, Any]) -> Dict[str, Any]:
    """Accept/reject b (candidate) vs a (baseline). Importable by tests."""
    reasons: List[str] = []

    # ── HARD gate ─────────────────────────────────────────────────────────────
    if _crashed(b):
        return {"accept": False,
                "reasons": ["HARD: candidate run crashed / produced no scored trials"]}
    if _degenerate_judge(b):
        jvg = b["diagnostics"]["judge_vs_gt"]
        return {"accept": False,
                "reasons": [f"HARD: judge is degenerate (gaming) — judge_vs_gt={jvg}"]}

    oa, ob = a.get("overall", {}), b.get("overall", {})
    sa, sb = oa.get("success_rate"), ob.get("success_rate")
    ta, tb = oa.get("task_time_median_s"), ob.get("task_time_median_s")

    # ── GATED axis: success (higher better, absolute tol) ──────────────────────
    succ_ok = succ_better = True
    if sa is not None and sb is not None:
        succ_ok = sb >= sa - SUCCESS_TOL
        succ_better = sb > sa + SUCCESS_TOL
        if not succ_ok:
            reasons.append(f"success regressed: {sa:.3f} -> {sb:.3f} (tol {SUCCESS_TOL})")
        elif succ_better:
            reasons.append(f"success improved: {sa:.3f} -> {sb:.3f}")
    else:
        succ_better = False
        reasons.append("success: missing baseline/candidate value (not gated)")

    # ── GATED axis: time (lower better, relative tol) ──────────────────────────
    time_ok = time_better = True
    if ta is not None and tb is not None:
        margin = TIME_REL_TOL * abs(ta) if ta else 0.0
        time_ok = tb <= ta + margin
        time_better = tb < ta - margin
        if not time_ok:
            reasons.append(f"time regressed: {ta:.2f}s -> {tb:.2f}s (tol {TIME_REL_TOL:.0%})")
        elif time_better:
            reasons.append(f"time improved: {ta:.2f}s -> {tb:.2f}s")
    else:
        time_better = False
        reasons.append("time: missing baseline/candidate value (not gated)")

    accept = succ_ok and time_ok and (succ_better or time_better)
    if accept and not reasons:
        reasons.append("within tolerance on both axes with a strict improvement")
    if not accept and succ_ok and time_ok and not (succ_better or time_better):
        reasons.append("no strict improvement on either gated axis (tie) -> REJECT")
    return {"accept": accept, "reasons": reasons}


# ── rendering ─────────────────────────────────────────────────────────────────

def _d(va, vb):
    if va is None or vb is None:
        return "—"
    return f"{vb - va:+.3f}"


def _row(label, va, vb, fmt="%.3f"):
    fa = "—" if va is None else fmt % va
    fb = "—" if vb is None else fmt % vb
    return f"| {label} | {fa} | {fb} | {_d(va, vb)} |"


def compare_files(a_path: str, b_path: str) -> str:
    a = _load(a_path) if isinstance(a_path, str) else a_path
    b = _load(b_path) if isinstance(b_path, str) else b_path
    oa, ob = a.get("overall", {}), b.get("overall", {})
    ta, tb = a.get("timing", {}), b.get("timing", {})

    L = [f"## Compare  [{a.get('arm')}] gen {a.get('gen')}  →  "
         f"[{b.get('arm')}] gen {b.get('gen')}", ""]
    L += ["### Gated axes (success + speed)",
          "| axis | baseline | candidate | Δ |",
          "|---|---|---|---|",
          _row("success_rate", oa.get("success_rate"), ob.get("success_rate")),
          _row("task_time_median_s", oa.get("task_time_median_s"),
               ob.get("task_time_median_s"), "%.2f"),
          ""]
    L += ["### Timing profile (informational, median s)",
          "| step | baseline | candidate | Δ |",
          "|---|---|---|---|",
          _row("codegen_s", ta.get("codegen_s_median"), tb.get("codegen_s_median")),
          _row("exec_s", ta.get("exec_s_median"), tb.get("exec_s_median")),
          _row("judge_s", ta.get("judge_s_median"), tb.get("judge_s_median")),
          _row("track_s", ta.get("track_s_median"), tb.get("track_s_median")),
          _row("judge_s_p90", ta.get("judge_s_p90"), tb.get("judge_s_p90")),
          ""]

    da = (a.get("diagnostics") or {}).get("judge_vs_gt")
    db = (b.get("diagnostics") or {}).get("judge_vs_gt")
    L.append("### judge-vs-GT (arm B confusion)")
    if da or db:
        L += ["| cell | baseline | candidate | Δ |", "|---|---|---|---|"]
        for k in ("true_done", "false_done", "missed_done", "true_notdone"):
            va = (da or {}).get(k)
            vb = (db or {}).get(k)
            L.append(_row(k, va, vb, "%.0f"))
    else:
        L.append("_no judge verdicts on either side (VDM arm)._")
    L.append("")

    v = verdict(a, b)
    L.append(f"**Verdict: {'ACCEPT ✅' if v['accept'] else 'REJECT ❌'}**")
    for r in v["reasons"]:
        L.append(f"- {r}")
    L.append("")
    L.append("_Gated: success (≥ baseline within tol) AND time (≤ baseline within tol) AND a "
             "strict improvement. Hard gate: crash or a degenerate (gaming) judge._")
    return "\n".join(L) + "\n"


def main(argv=None) -> int:
    args = argv if argv is not None else sys.argv[1:]
    if len(args) != 2:
        print("usage: compare <genA|pathA> <genB|pathB>")
        return 2
    print(compare_files(args[0], args[1]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
