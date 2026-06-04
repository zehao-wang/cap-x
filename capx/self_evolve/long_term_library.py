"""Loader for the fixed long-term library (skill_library + atomic_task_library).

These are the self-evolve system's promoted, validated capabilities
(``docs-se/storage.md``):

- ``capx/skill_library/*.py``        general functions built from primitives
- ``capx/atomic_task_library/*.py``  one file per atomic task = function + config

The loader exposes them to the agent the same two ways primitives are exposed
(``docs-se/integration.md``):

- :func:`long_term_docs` — signature + docstring of every public function, for
  the agent prompt (so it knows the function exists).
- :func:`inject_long_term` — exec each module's source into the code-execution
  namespace, so an atomic task can call the primitives that live alongside it.

Both are **no-ops when the dirs are empty/absent**, so wiring them in is safe
before anything has been promoted. Exec-into-namespace (not import) is required:
the functions reference primitives (``goto_pose`` …) as free globals and must
resolve them from the *agent's* namespace, exactly like user code does.
"""

from __future__ import annotations

import ast
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
SKILL_LIBRARY_DIR = _REPO_ROOT / "capx" / "skill_library"
ATOMIC_TASK_LIBRARY_DIR = _REPO_ROOT / "capx" / "atomic_task_library"


def _module_files() -> list[Path]:
    """Every promoted ``*.py`` across both library dirs (skills first), sorted."""
    files: list[Path] = []
    for d in (SKILL_LIBRARY_DIR, ATOMIC_TASK_LIBRARY_DIR):
        if d.is_dir():
            files.extend(p for p in sorted(d.glob("*.py")) if p.name != "__init__.py")
    return files


def _render_signature(fn: ast.FunctionDef) -> str:
    """Reconstruct a readable ``name(args)`` signature from an AST def."""
    a = fn.args
    parts: list[str] = []
    positional = a.posonlyargs + a.args
    n_no_default = len(positional) - len(a.defaults)
    for i, arg in enumerate(positional):
        s = arg.arg
        if arg.annotation is not None:
            s += f": {ast.unparse(arg.annotation)}"
        di = i - n_no_default
        if di >= 0:
            s += f"={ast.unparse(a.defaults[di])}"
        parts.append(s)
        if a.posonlyargs and arg is a.posonlyargs[-1]:
            parts.append("/")
    if a.vararg:
        parts.append("*" + a.vararg.arg)
    elif a.kwonlyargs:
        parts.append("*")
    for arg, default in zip(a.kwonlyargs, a.kw_defaults):
        s = arg.arg
        if arg.annotation is not None:
            s += f": {ast.unparse(arg.annotation)}"
        if default is not None:
            s += f"={ast.unparse(default)}"
        parts.append(s)
    if a.kwarg:
        parts.append("**" + a.kwarg.arg)
    return f"{fn.name}({', '.join(parts)})"


def long_term_docs() -> str:
    """Prompt documentation for every public long-term function.

    One entry per top-level non-underscore function: its signature + docstring,
    tagged with the source library so the agent knows the granularity. Returns an
    empty string when nothing is promoted yet.
    """
    entries: list[str] = []
    for path in _module_files():
        library = path.parent.name
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except (SyntaxError, OSError):
            continue
        for node in tree.body:
            if isinstance(node, ast.FunctionDef) and not node.name.startswith("_"):
                doc = (ast.get_docstring(node) or "").strip()
                block = [f"{_render_signature(node)}    [{library}]"]
                if doc:
                    block.append("  " + doc.replace("\n", "\n  "))
                entries.append("\n".join(block))
    return "\n\n".join(entries)


def inject_long_term(namespace: dict) -> list[str]:
    """Exec each long-term module's source into ``namespace``.

    Returns the public function names that became callable. A module that fails
    to load is skipped with a printed warning (one bad promoted file must not
    break the whole run).
    """
    added: list[str] = []
    for path in _module_files():
        src = path.read_text(encoding="utf-8")
        try:
            exec(compile(src, str(path), "exec"), namespace)  # noqa: S102
        except Exception as exc:  # noqa: BLE001 - degrade gracefully
            print(f"[long_term_library] failed to load {path.name}: {exc}")
            continue
        try:
            for node in ast.parse(src).body:
                if isinstance(node, ast.FunctionDef) and not node.name.startswith("_"):
                    added.append(node.name)
        except SyntaxError:
            pass
    return added
