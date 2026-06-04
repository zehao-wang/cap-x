"""Tests for the long-term library loader (skill_library + atomic_task_library).

Exercises the two exposure mechanisms in isolation against a temp library dir,
plus a smoke check that the real promoted ``pick`` atomic task is discoverable.
"""

from __future__ import annotations

import capx.self_evolve.long_term_library as ltl


def _point_dirs_at(monkeypatch, skill_dir, atomic_dir):
    monkeypatch.setattr(ltl, "SKILL_LIBRARY_DIR", skill_dir)
    monkeypatch.setattr(ltl, "ATOMIC_TASK_LIBRARY_DIR", atomic_dir)


def test_empty_dirs_are_noop(monkeypatch, tmp_path):
    _point_dirs_at(monkeypatch, tmp_path / "skill", tmp_path / "atomic")  # absent
    assert ltl.long_term_docs() == ""
    ns: dict = {}
    assert ltl.inject_long_term(ns) == []


def test_docs_and_injection_roundtrip(monkeypatch, tmp_path):
    atomic = tmp_path / "atomic"
    atomic.mkdir()
    (atomic / "__init__.py").write_text("", encoding="utf-8")
    (atomic / "place.py").write_text(
        "import numpy as np\n"
        "PLACE_MARGIN = 0.05\n"
        "def _helper():\n    return 1\n"
        "def place(object_name, target, *, lift=0.1):\n"
        '    """Place object at target."""\n'
        "    return goto(target)\n",
        encoding="utf-8",
    )
    _point_dirs_at(monkeypatch, tmp_path / "skill", atomic)

    docs = ltl.long_term_docs()
    assert "place(object_name, target, *, lift=0.1)    [atomic]" in docs
    assert "Place object at target." in docs
    assert "_helper" not in docs  # underscore-private functions are not advertised

    ns: dict = {"goto": lambda t: ("went", t)}
    added = ltl.inject_long_term(ns)
    assert added == ["place"]  # only the public function is reported
    assert ns["place"]("cube", "binA")[1] == "binA"  # resolves the free global `goto`
    assert ns["PLACE_MARGIN"] == 0.05  # config constant landed in the namespace


def test_real_pick_atomic_task_is_discoverable():
    """The committed pick atomic task shows up with its signature + docstring."""
    docs = ltl.long_term_docs()
    assert "pick(object_name)    [atomic_task_library]" in docs
    assert "collision-aware" in docs


def test_injection_skips_a_broken_module(monkeypatch, tmp_path, capsys):
    atomic = tmp_path / "atomic"
    atomic.mkdir()
    (atomic / "bad.py").write_text("def oops(:\n    pass\n", encoding="utf-8")  # syntax error
    (atomic / "good.py").write_text("def good():\n    return 1\n", encoding="utf-8")
    _point_dirs_at(monkeypatch, tmp_path / "skill", atomic)

    ns: dict = {}
    added = ltl.inject_long_term(ns)
    assert added == ["good"] and "good" in ns
    assert "failed to load bad.py" in capsys.readouterr().out
