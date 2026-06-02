"""Tests for module ④ (Update Planner): reader, proposal gate, tool loop, run.

All model interaction is a scripted ``query_fn`` that branches on the role in the
system prompt, so the full propose → review → write → cursor flow runs with no
server or sim.
"""

from __future__ import annotations

import datetime as dt

from capx.self_evolve import MemStore
from capx.self_evolve.config import UpdatePlannerConfig
from capx.self_evolve.history_reader import HistoryReader
from capx.self_evolve.proposal import (
    Proposal,
    ProposalCandidate,
    is_object_bound_name,
    validate_candidate,
    write_accepted,
)
from capx.self_evolve.schemas import Digest, SuccessLog
from capx.self_evolve.update_planner import (
    extract_json_action,
    run_tool_loop,
    run_update_planner,
    should_trigger,
)


def _digest(task="pick cube"):
    return Digest(
        task=task, datetime="2026-05-31 12:00:00", settings_line="env=robosuite",
        sections={
            "KEY STRATEGY": "approach then grasp [#1]",
            "REUSABLE PATTERN": "approach_and_grasp is reusable [#0]",
            "KEY HYPER-PARAMS / FEEDBACK": "z_margin=0.03 [#2]",
            "FRAGILITY": "none [#1]",
        },
    )


def _seed_history(store: MemStore, n=1, start=0):
    ids = []
    for i in range(start, start + n):
        log = SuccessLog(
            task=f"task{i}",
            final_code=f"def main():\n    line_a_{i}()\n    line_b_{i}()\n    line_c_{i}()\n",
            chat_history=[
                {"role": "user", "content": f"do task {i}"},
                {"role": "assistant", "content": [{"type": "text", "text": "moving to grasp the cube"}]},
                {"role": "user", "content": "keep 0.03 margin"},
            ],
            settings={"env": "robosuite"},
            datetime="2026-05-31 12:00:00",
        )
        ids.append(store.write_history(log, _digest(f"task{i}"), history_id=f"task{i}__2026053{i % 10}-120000"))
    return ids


# --------------------------------------------------------------------------- #
# HistoryReader
# --------------------------------------------------------------------------- #
def test_reader_index_digest_and_drill(tmp_path):
    store = MemStore(tmp_path / "mem")
    (hid,) = _seed_history(store, 1)
    reader = HistoryReader(store)

    index = reader.list_unprocessed()
    assert index[0]["id"] == hid and index[0]["task"] == "task0"
    assert "task0" in reader.read_digest(hid)

    # drill chat_history by [#idx]
    turn = reader.read_history(hid, "chat_history", offset=2, limit=1)
    assert turn[0]["content"] == "keep 0.03 margin"
    # drill final_code by lines
    assert reader.read_history(hid, "final_code", offset=1, limit=1) == "    line_a_0()"


def test_reader_grep_and_processed_filtering(tmp_path):
    store = MemStore(tmp_path / "mem")
    ids = _seed_history(store, 2)
    reader = HistoryReader(store)
    hits = reader.grep_history("grasp the cube")
    assert {h["id"] for h in hits} == set(ids)
    assert all(h["message_index"] == 1 for h in hits)

    store.mark_processed(ids[0])
    assert [i["id"] for i in reader.list_unprocessed()] == [ids[1]]


def test_reader_read_library_parses_signatures(tmp_path):
    skill_dir = tmp_path / "skill_library"
    skill_dir.mkdir()
    (skill_dir / "grasping.py").write_text(
        'def approach_and_grasp(obj, margin=0.03):\n    """Approach then grasp an object."""\n    pass\n'
    )
    (skill_dir / "__init__.py").write_text("")
    reader = HistoryReader(MemStore(tmp_path / "mem"), library_dirs={"skill_library": skill_dir})
    funcs = reader.read_library("skill_library")
    assert len(funcs) == 1
    assert funcs[0]["signature"] == "approach_and_grasp(obj, margin)"
    assert "Approach then grasp" in funcs[0]["docstring"]

    # non-existent dir -> empty, no crash
    empty = HistoryReader(MemStore(tmp_path / "mem"), library_dirs={"skill_library": tmp_path / "nope"})
    assert empty.read_library("skill_library") == []


# --------------------------------------------------------------------------- #
# proposal gate
# --------------------------------------------------------------------------- #
def test_is_object_bound_name():
    assert is_object_bound_name("put_apple")
    assert is_object_bound_name("stackRedBlock")
    assert not is_object_bound_name("pick")
    assert not is_object_bound_name("approach_and_grasp")


def test_validate_candidate_flags_issues(tmp_path):
    store = MemStore(tmp_path / "mem")
    (hid,) = _seed_history(store, 1)

    ok = ProposalCandidate("approach_and_grasp", "skill_library", "def approach_and_grasp(): pass", [hid])
    assert validate_candidate(ok, store) == []

    bad = ProposalCandidate("put_apple", "atomic_task_library", "def put_apple(): pass", ["nope__1"])
    issues = validate_candidate(bad, store)
    assert any("object/task-bound" in m for m in issues)
    assert any("does not exist" in m for m in issues)


def test_write_accepted_persists_code_and_stats(tmp_path):
    store = MemStore(tmp_path / "mem")
    (hid,) = _seed_history(store, 1)
    cand = ProposalCandidate("approach_and_grasp", "skill_library", "def approach_and_grasp(): pass", [hid])
    stats = write_accepted(cand, store, now=dt.datetime(2026, 5, 31))
    assert store.list_candidate_names() == ["approach_and_grasp"]
    on_disk = store.read_candidate_stats("approach_and_grasp")
    assert on_disk.source_history == [hid]
    assert on_disk.created_date == "2026-05-31"
    assert stats.positive == 0 and stats.eval_runs == 0


# --------------------------------------------------------------------------- #
# action parsing + tool loop
# --------------------------------------------------------------------------- #
def test_extract_json_action_variants():
    assert extract_json_action('```json\n{"tool": "list_unprocessed"}\n```')["tool"] == "list_unprocessed"
    assert extract_json_action('thinking... {"propose": {"candidates": []}}')["propose"] == {"candidates": []}
    # last object wins
    assert extract_json_action('{"a": 1} then {"tool": "read_digest"}')["tool"] == "read_digest"
    assert extract_json_action("no json here") is None


def test_tool_loop_dispatches_then_terminates(tmp_path):
    store = MemStore(tmp_path / "mem")
    _seed_history(store, 1)
    reader = HistoryReader(store)
    scripted = iter([
        '{"tool": "list_unprocessed"}',
        '{"propose": {"candidates": [{"func_name": "f", "target_library": "skill_library", '
        '"code": "def f(): pass", "source_history": ["task0__20260531-120000"]}]}}',
    ])
    seen = {"list_result": False}

    def query_fn(messages):
        # confirm the tool result was fed back
        if any("Result of list_unprocessed" in (m.get("content") or "") for m in messages):
            seen["list_result"] = True
        return next(scripted)

    payload = run_tool_loop(query_fn, "sys", "go", reader, terminal_key="propose", max_read_iterations=20)
    assert seen["list_result"]
    assert Proposal.from_dict(payload).candidates[0].func_name == "f"


def test_tool_loop_budget_exhaustion_returns_empty(tmp_path):
    reader = HistoryReader(MemStore(tmp_path / "mem"))
    payload = run_tool_loop(
        lambda m: '{"tool": "list_unprocessed"}', "sys", "go", reader,
        terminal_key="propose", max_read_iterations=3,
    )
    assert payload == {}


# --------------------------------------------------------------------------- #
# end-to-end orchestration
# --------------------------------------------------------------------------- #
def _role(messages):
    sys = messages[0]["content"]
    return "llm2" if "LLM-2, the reviewer" in sys else "llm1"


def test_should_trigger(tmp_path):
    store = MemStore(tmp_path / "mem")
    _seed_history(store, 4)
    assert not should_trigger(store, UpdatePlannerConfig(trigger_history_count=5))
    _seed_history(store, 1, start=4)
    assert should_trigger(store, UpdatePlannerConfig(trigger_history_count=5))


def test_run_update_planner_accepts_and_advances_cursor(tmp_path):
    store = MemStore(tmp_path / "mem")
    ids = _seed_history(store, 2)
    cand = {
        "func_name": "approach_and_grasp", "target_library": "skill_library",
        "code": "def approach_and_grasp(obj):\n    pass\n", "source_history": [ids[0]],
        "rationale": "general approach+grasp",
    }

    def query_fn(messages):
        if _role(messages) == "llm1":
            return '{"propose": {"candidates": [%s]}}' % __import__("json").dumps(cand)
        return '{"review": {"accepted": [%s], "feedback": ""}}' % __import__("json").dumps(cand)

    result = run_update_planner(
        store, query_fn, config=UpdatePlannerConfig(trigger_history_count=2),
        now=dt.datetime(2026, 5, 31),
    )
    assert result.triggered
    assert result.written == ["approach_and_grasp"]
    assert result.revise_rounds == 1
    assert set(result.processed_ids) == set(ids)
    assert store.list_unprocessed() == []  # cursor advanced
    assert store.list_candidate_names() == ["approach_and_grasp"]


def test_run_update_planner_gate_bounces_hallucinated_reference(tmp_path):
    store = MemStore(tmp_path / "mem")
    ids = _seed_history(store, 2)
    # LLM-2 "accepts" a candidate citing a non-existent history id; the mechanical
    # gate must refuse to write it no matter what the reviewer says.
    bad = {
        "func_name": "ghost", "target_library": "skill_library",
        "code": "def ghost(): pass", "source_history": ["nonexistent__1"], "rationale": "x",
    }

    def query_fn(messages):
        if _role(messages) == "llm1":
            return '{"propose": {"candidates": [%s]}}' % __import__("json").dumps(bad)
        return '{"review": {"accepted": [%s], "feedback": ""}}' % __import__("json").dumps(bad)

    result = run_update_planner(
        store, query_fn, config=UpdatePlannerConfig(trigger_history_count=2, max_revise_iterations=2),
        now=dt.datetime(2026, 5, 31),
    )
    assert result.triggered
    assert result.written == []  # never written despite reviewer acceptance
    assert store.list_candidate_names() == []
    assert set(result.processed_ids) == set(ids)  # batch still marked processed


def test_run_update_planner_no_trigger_is_noop(tmp_path):
    store = MemStore(tmp_path / "mem")
    _seed_history(store, 1)
    result = run_update_planner(store, lambda m: "{}", config=UpdatePlannerConfig(trigger_history_count=5))
    assert not result.triggered
    assert store.list_unprocessed()  # nothing consumed
