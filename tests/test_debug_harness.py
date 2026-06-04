"""Tests for the modular debug harness + handoff contract reader.

Covers the live-loop → postprocessor seam without a model server or env:
``load_handoff`` validation, the handoff → ``run_feedback_postprocessor``
mapping, and the injectable judge stand-ins. Matches the fake-``query_fn`` /
``tmp_path`` MemStore style of ``test_feedback_postprocessor.py``.
"""

from __future__ import annotations

import json

import pytest

from capx.self_evolve import MemStore, load_handoff, run_feedback_postprocessor
from capx.self_evolve.debug import auto_accept_judge, reject_judge
from capx.self_evolve.handoff import HANDOFF_SCHEMA, Handoff

FINAL_CODE = 'open_gripper()\npos, quat = sample_grasp_pose("spring onion")\npos[2] += 0.03\ngoto_pose(pos, quat)\nclose_gripper()'

CHAT_HISTORY = [
    {"index": 0, "role": "task", "attempt": None, "content": "pick up the spring onion"},
    {"index": 1, "role": "assistant", "attempt": 0, "phase": "initial", "content": "open_gripper()"},
    {"index": 2, "role": "human_feedback", "attempt": 0, "content": "加 pregrasp，grasp z 加 offset"},
    {"index": 3, "role": "assistant", "attempt": 1, "phase": "initial", "content": FINAL_CODE},
    {"index": 4, "role": "human_finish", "attempt": 1, "content": ""},
]

GOOD_DIGEST = """## KEY STRATEGY
approach then grasp with a vertical margin [#3]

## REUSABLE PATTERN
pick(obj) generalizes as an atomic task [#1]

## KEY HYPER-PARAMS / FEEDBACK
z_margin=0.03 from the operator [#2]

## FRAGILITY
none [#3]
"""


def _write_handoff(dir_path, *, drop=None, schema=HANDOFF_SCHEMA):
    payload = {
        "schema": schema,
        "task": "pick up the spring onion",
        "settings": {"model": "google/gemini-3.1-pro-preview", "env_config": "piper_real.yaml"},
        "success": {"signal": "human_finished", "attempt": 1},
        "final_code": FINAL_CODE,
        "human_feedback": [{"index": 2, "attempt": 0, "text": "加 pregrasp，grasp z 加 offset"}],
        "chat_history": CHAT_HISTORY,
        "datetime": "2026-06-04 15:31:07",
    }
    if drop:
        payload.pop(drop)
    p = dir_path / "postprocess_handoff.json"
    p.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return p


# --------------------------------------------------------------------------- #
# load_handoff contract validation
# --------------------------------------------------------------------------- #
def test_load_handoff_accepts_dir_or_file(tmp_path):
    _write_handoff(tmp_path)
    from_dir = load_handoff(tmp_path)
    from_file = load_handoff(tmp_path / "postprocess_handoff.json")
    assert from_dir.short_task == from_file.short_task == "pick up the spring onion"
    assert from_dir.feedback_texts() == ["加 pregrasp，grasp z 加 offset"]
    assert from_dir.success == {"signal": "human_finished", "attempt": 1}


def test_load_handoff_rejects_missing_final_code(tmp_path):
    _write_handoff(tmp_path, drop="final_code")
    with pytest.raises(ValueError, match="final_code"):
        load_handoff(tmp_path)


def test_load_handoff_rejects_bad_schema(tmp_path):
    _write_handoff(tmp_path, schema="something/else")
    with pytest.raises(ValueError, match="schema"):
        load_handoff(tmp_path)


def test_load_handoff_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_handoff(tmp_path)


# --------------------------------------------------------------------------- #
# handoff -> postprocessor mapping
# --------------------------------------------------------------------------- #
def test_postprocessor_kwargs_shape():
    ho = Handoff(
        path=None, task="pick up the spring onion", settings={}, final_code=FINAL_CODE,
        chat_history=CHAT_HISTORY, human_feedback=[], success={}, datetime="",
    )
    kw = ho.postprocessor_kwargs()
    assert kw["original_code"] == FINAL_CODE
    assert kw["task"] == "pick up the spring onion"
    assert kw["chat_history"] is CHAT_HISTORY


def test_handoff_feeds_postprocessor_end_to_end(tmp_path):
    """The contract a handoff promises actually runs module ③ to a history pair."""
    _write_handoff(tmp_path)
    ho = load_handoff(tmp_path)

    def fake_query(messages):
        # Distill prompt enumerates the digest sections; everything else is the
        # generalize-rewrite step — return the code unchanged so it stops at once.
        user = messages[-1]["content"]
        if "KEY STRATEGY" in user:
            return GOOD_DIGEST
        return FINAL_CODE

    store = MemStore(tmp_path / "mem")
    result = run_feedback_postprocessor(
        query_fn=fake_query, judge_fn=auto_accept_judge(verbose=False), store=store,
        max_rewrite_rounds=3, **ho.postprocessor_kwargs(),
    )

    assert result.final_code == FINAL_CODE  # no-op rewrite kept the baseline
    assert (store.history_pool / f"{result.history_id}.json").exists()
    assert (store.history_pool / f"{result.history_id}.digest.md").exists()
    reloaded = store.read_history(result.history_id)
    assert reloaded.chat_history == CHAT_HISTORY  # verbatim feedback survived into the pool
    assert result.digest.source_indices() == [3, 1, 2]


# --------------------------------------------------------------------------- #
# injectable judges
# --------------------------------------------------------------------------- #
def test_judge_stand_ins():
    assert auto_accept_judge(verbose=False)("x") is True
    assert reject_judge(verbose=False)("x") is False
