"""Module ⑤: approval-by-PR — turn evaluator decisions into a mergeable PR.

The Benchmark Evaluator only *proposes* (see :mod:`capx.self_evolve.benchmark_eval`).
Nothing reaches the long-term libraries (``capx/skill_library/`` /
``capx/atomic_task_library/``) and no candidate is deleted from
``func_candidate_pool`` until the user **merges** the PR this module builds
(closing it = rejection). That is the hard constraint from
``docs-se/05-benchmark-evaluator.md``: the Evaluator never mutates the long-term
state directly.

Split by side-effect, so the planning is unit-testable:

- :func:`stamp_docstring_date` — stamp a promoted function's docstring with its
  update date (the date only; not *what* changed).
- :func:`plan_pr` — pure: an :class:`EvaluatorResult` → a :class:`PlannedPR`
  (branch name, file writes, deletes, commit message, PR title/body). No disk, no git.
- :func:`apply_planned_changes` — write/delete the planned files under a root
  (the file mutations a PR branch carries). Pure I/O, tmp-dir testable.
- :func:`create_library_pr` — the live, git+``gh`` mechanism: branch → apply →
  commit → push → open PR. Side-effecting and environment-gated (needs git+gh).
"""

from __future__ import annotations

import ast
import datetime as _dt
import logging
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

from capx.self_evolve.benchmark_eval import EvaluatorResult

logger = logging.getLogger(__name__)

# Repo root = two levels up from capx/self_evolve/library_pr.py.
_REPO_ROOT = Path(__file__).resolve().parents[2]
_LIBRARY_RELDIR = {
    "skill_library": Path("capx") / "skill_library",
    "atomic_task_library": Path("capx") / "atomic_task_library",
}


# --------------------------------------------------------------------------- #
# docstring date-stamp
# --------------------------------------------------------------------------- #
def stamp_docstring_date(code: str, date: str) -> str:
    """Stamp the first top-level function's docstring with ``Updated: <date>``.

    Records *when* a function was (re)solidified, not what changed — per the
    design. Handles the three cases: no docstring (insert a one-line one),
    single-line docstring (expand to multi-line), multi-line docstring (insert a
    line before the closing quotes). Returns ``code`` unchanged if it does not
    parse or has no top-level function.
    """
    stamp = f"Updated: {date}"
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return code
    func = next(
        (n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))),
        None,
    )
    if func is None or not func.body:
        return code

    trailing_nl = code.endswith("\n")
    lines = code.splitlines()
    first = func.body[0]
    indent = " " * first.col_offset
    has_doc = (
        isinstance(first, ast.Expr)
        and isinstance(getattr(first, "value", None), ast.Constant)
        and isinstance(first.value.value, str)
    )

    if not has_doc:
        lines.insert(first.lineno - 1, f'{indent}"""{stamp}"""')
    else:
        start, end = first.lineno - 1, first.end_lineno - 1
        if start == end:  # single-line docstring -> expand to multi-line
            raw = lines[start]
            line_indent = raw[: len(raw) - len(raw.lstrip())]
            inner = raw.strip()
            quote = inner[:3]
            text = inner[3:-3].rstrip()
            lines[start] = (
                f"{line_indent}{quote}{text}\n\n{line_indent}{stamp}\n{line_indent}{quote}"
            )
        else:  # multi-line -> insert before the closing-quote line
            lines.insert(end, f"{indent}{stamp}")

    out = "\n".join(lines)
    return out + "\n" if trailing_nl else out


# --------------------------------------------------------------------------- #
# plan
# --------------------------------------------------------------------------- #
@dataclass
class PlannedPR:
    """Everything needed to open the approval PR — computed without side effects.

    ``writes`` / ``deletes`` are repo-root-relative POSIX paths so the same plan
    can be applied to any worktree (and asserted against in tests).
    """

    branch: str
    writes: dict[str, str] = field(default_factory=dict)   # relpath -> content
    deletes: list[str] = field(default_factory=list)       # relpath
    commit_message: str = ""
    pr_title: str = ""
    pr_body: str = ""

    @property
    def is_empty(self) -> bool:
        return not self.writes and not self.deletes


def plan_pr(
    result: EvaluatorResult,
    *,
    now: _dt.datetime | None = None,
    mem_reldir: str = "mem/func_candidate_pool",
) -> PlannedPR:
    """Translate evaluator decisions into a concrete, side-effect-free PR plan.

    - **Promote** → write the (date-stamped) candidate source to
      ``capx/<target_library>/<func_name>.py`` and ensure that package's
      ``__init__.py`` exists.
    - **Abandon** → delete the candidate's ``.py`` + ``.stats.json`` from the pool.

    Keep/empty results produce an empty plan (``is_empty``); the caller skips
    opening a PR in that case.
    """
    now = now or _dt.datetime.now()
    date = now.strftime("%Y-%m-%d")
    writes: dict[str, str] = {}
    deletes: list[str] = []

    touched_libs: set[str] = set()
    for d in result.promotions:
        if not d.code.strip():
            raise ValueError(
                f"promotion for {d.func_name!r} carries no code; the evaluator must "
                "attach candidate source to the decision before planning the PR"
            )
        reldir = _LIBRARY_RELDIR[d.target_library]
        writes[(reldir / f"{d.func_name}.py").as_posix()] = stamp_docstring_date(d.code, date)
        touched_libs.add(d.target_library)

    for lib in sorted(touched_libs):
        init_rel = (_LIBRARY_RELDIR[lib] / "__init__.py").as_posix()
        writes.setdefault(init_rel, f'"""cap-x {lib} (self-evolve solidified skills)."""\n')

    pool = Path(mem_reldir)
    for d in result.abandons:
        deletes.append((pool / f"{d.func_name}.py").as_posix())
        deletes.append((pool / f"{d.func_name}.stats.json").as_posix())

    n_prom, n_aban = len(result.promotions), len(result.abandons)
    branch = f"se/library-update-{now:%Y%m%d-%H%M%S}"
    title = f"[cap-x-se] library update: {n_prom} promote, {n_aban} abandon ({date})"
    return PlannedPR(
        branch=branch,
        writes=writes,
        deletes=deletes,
        commit_message=title,
        pr_title=title,
        pr_body=result.report,
    )


# --------------------------------------------------------------------------- #
# apply (the file mutations a PR branch carries)
# --------------------------------------------------------------------------- #
def apply_planned_changes(planned: PlannedPR, root: Path | str) -> tuple[list[str], list[str]]:
    """Write/delete the planned files under ``root``; return (written, deleted).

    This is what the PR branch contains. Calling it on the repo's working tree on
    the *default* branch would violate the hard constraint, so
    :func:`create_library_pr` always calls it on a fresh branch.
    """
    root = Path(root)
    written: list[str] = []
    deleted: list[str] = []
    for rel, content in planned.writes.items():
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        written.append(rel)
    for rel in planned.deletes:
        path = root / rel
        if path.exists():
            path.unlink()
            deleted.append(rel)
    return written, deleted


# --------------------------------------------------------------------------- #
# live PR mechanism (git + gh) — environment-gated
# --------------------------------------------------------------------------- #
def _run(cmd: list[str], cwd: Path) -> subprocess.CompletedProcess:
    logger.debug("run: %s", " ".join(cmd))
    return subprocess.run(cmd, cwd=cwd, check=True, capture_output=True, text=True)


def create_library_pr(
    planned: PlannedPR,
    *,
    repo_root: Path | str | None = None,
    base: str = "main",
    push: bool = True,
    open_pr: bool = True,
    runner=_run,
) -> dict[str, object]:
    """Stage ``planned`` on a fresh branch, commit, push, and open a PR.

    The approval mechanism: it never merges and never writes on ``base`` — the
    user merges (approve) or closes (reject) the resulting PR. ``push`` / ``open_pr``
    are toggleable for dry runs / sandboxes (this dev env has no ``gh``); ``runner``
    is injectable so the git/gh sequence can be asserted without a real repo.

    Returns a summary dict (branch, written/deleted relpaths, whether pushed/PR
    opened). Raises if the plan is empty — nothing to propose.
    """
    if planned.is_empty:
        raise ValueError("planned PR is empty; nothing to propose")
    repo_root = Path(repo_root) if repo_root is not None else _REPO_ROOT

    runner(["git", "checkout", "-b", planned.branch], repo_root)
    written, deleted = apply_planned_changes(planned, repo_root)
    runner(["git", "add", "-A"], repo_root)
    runner(["git", "commit", "-m", planned.commit_message], repo_root)

    pushed = pr_opened = False
    if push:
        runner(["git", "push", "-u", "origin", planned.branch], repo_root)
        pushed = True
        if open_pr:
            runner(
                ["gh", "pr", "create", "--base", base, "--head", planned.branch,
                 "--title", planned.pr_title, "--body", planned.pr_body],
                repo_root,
            )
            pr_opened = True
    logger.info(
        "library PR staged on %s (%d write, %d delete); pushed=%s pr=%s",
        planned.branch, len(written), len(deleted), pushed, pr_opened,
    )
    return {
        "branch": planned.branch,
        "written": written,
        "deleted": deleted,
        "pushed": pushed,
        "pr_opened": pr_opened,
    }
