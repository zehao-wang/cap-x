"""Round-trip tests for the self-evolve storage scaffold (module ⓪).

Covers the three ``mem/`` schemas, the digest source-marker handling, the
processed-history cursor, and config defaults/overrides. Uses ``tmp_path`` so no
real ``mem/`` is touched.
"""

from __future__ import annotations

import pytest

from capx.self_evolve import (
    CandidateStats,
    Digest,
    MemStore,
    SelfEvolveConfig,
    SuccessLog,
    make_history_id,
)
from capx.self_evolve.schemas import DIGEST_SECTIONS


# --------------------------------------------------------------------------- #
# config
# --------------------------------------------------------------------------- #
def test_config_defaults_match_docs():
    cfg = SelfEvolveConfig()
    assert cfg.experience_distill.max_words == 200
    assert cfg.update_planner.trigger_history_count == 5
    assert cfg.update_planner.max_revise_iterations == 5
    assert cfg.update_planner.max_read_iterations == 20
    assert cfg.benchmark_eval.min_samples == 5
    assert cfg.benchmark_eval.promote_threshold == 10
    assert cfg.benchmark_eval.max_idle_evals == 50


def test_config_partial_override_and_roundtrip():
    cfg = SelfEvolveConfig.from_dict({"update_planner": {"trigger_history_count": 8}})
    assert cfg.update_planner.trigger_history_count == 8
    assert cfg.update_planner.max_revise_iterations == 5  # untouched default
    assert SelfEvolveConfig.from_dict(cfg.to_dict()).to_dict() == cfg.to_dict()


def test_config_rejects_unknown_group_and_key():
    with pytest.raises(ValueError):
        SelfEvolveConfig.from_dict({"nope": {}})
    with pytest.raises(ValueError):
        SelfEvolveConfig.from_dict({"benchmark_eval": {"bogus": 1}})


# --------------------------------------------------------------------------- #
# schemas
# --------------------------------------------------------------------------- #
def _sample_log():
    return SuccessLog(
        task="pick red cube",
        final_code="def main():\n    pick('red cube')\n",
        chat_history=[{"role": "user", "content": "pick it"}, {"role": "assistant", "content": "ok"}],
        settings={"env": "robosuite", "llm": "qwen3.6-27B"},
        datetime="2026-05-31 12:00:00",
    )


def _sample_digest():
    return Digest(
        task="pick red cube",
        datetime="2026-05-31 12:00:00",
        settings_line="env=robosuite, llm=qwen3.6-27B",
        sections={
            "KEY STRATEGY": "approach then grasp [#1]",
            "REUSABLE PATTERN": "pick(obj) generalizes [#0]",
            "KEY HYPER-PARAMS / FEEDBACK": "z margin 0.03 from human [#1]",
            "FRAGILITY": "none",
        },
    )


def test_success_log_roundtrip():
    log = _sample_log()
    assert SuccessLog.from_dict(log.to_dict()).to_dict() == log.to_dict()


def test_success_log_validation():
    with pytest.raises(ValueError):
        SuccessLog(task="", final_code="x").validate()
    with pytest.raises(ValueError):
        SuccessLog(task="t", final_code="  ").validate()
    with pytest.raises(ValueError):
        SuccessLog.from_dict({"task": "t"})  # missing final_code


def test_digest_render_parse_roundtrip_and_sources():
    digest = _sample_digest()
    reparsed = Digest.parse(digest.render())
    assert reparsed.task == digest.task
    assert reparsed.datetime == digest.datetime
    assert reparsed.settings_line == digest.settings_line
    for name in DIGEST_SECTIONS:
        assert reparsed.sections[name] == digest.sections[name]
    assert reparsed.source_indices() == [1, 0]  # first-seen order across sections


def test_digest_validation_missing_section_and_word_cap():
    bad = Digest(task="t", datetime="", settings_line="", sections={"KEY STRATEGY": "x"})
    with pytest.raises(ValueError):
        bad.validate()
    long_digest = _sample_digest()
    long_digest.sections["KEY STRATEGY"] = "word " * 300
    with pytest.raises(ValueError):
        long_digest.validate(max_words=200)


def test_candidate_stats_roundtrip_and_validation():
    stats = CandidateStats(
        func_name="approach_pose",
        target_library="atomic_task_library",
        source_history=["pick-red-cube__20260531-120000"],
        positive=3,
        eval_runs=4,
        used_runs=2,
        created_date="2026-05-31",
        last_eval_date="2026-05-31",
    )
    assert CandidateStats.from_dict(stats.to_dict()).to_dict() == stats.to_dict()
    with pytest.raises(ValueError):
        CandidateStats(func_name="f", target_library="wrong_lib").validate()
    with pytest.raises(ValueError):
        CandidateStats(func_name="f", target_library="skill_library", used_runs=5, eval_runs=2).validate()


# --------------------------------------------------------------------------- #
# MemStore
# --------------------------------------------------------------------------- #
def test_history_write_read_pair(tmp_path):
    store = MemStore(tmp_path / "mem")
    hid = store.write_history(_sample_log(), _sample_digest(), history_id="t__20260531-120000")
    assert hid == "t__20260531-120000"
    assert store.list_history_ids() == [hid]
    assert store.read_history(hid).final_code == _sample_log().final_code
    assert store.read_digest(hid).sections["FRAGILITY"] == "none"


def test_history_auto_id_from_task(tmp_path):
    store = MemStore(tmp_path / "mem")
    hid = store.write_history(_sample_log(), _sample_digest())
    assert hid.startswith("pick-red-cube__")


def test_candidate_write_read_and_update(tmp_path):
    store = MemStore(tmp_path / "mem")
    stats = CandidateStats(func_name="approach_pose", target_library="atomic_task_library")
    store.write_candidate(stats, code="def approach_pose():\n    pass\n")
    assert store.list_candidate_names() == ["approach_pose"]
    assert "approach_pose" in store.read_candidate_code("approach_pose")

    stats.eval_runs = 1
    stats.used_runs = 1
    store.update_candidate_stats(stats)
    assert store.read_candidate_stats("approach_pose").eval_runs == 1

    with pytest.raises(ValueError):
        store.write_candidate(CandidateStats(func_name="bad name", target_library="skill_library"), "x")


def test_processed_history_cursor(tmp_path):
    store = MemStore(tmp_path / "mem")
    for ts in ("20260531-120000", "20260531-130000", "20260531-140000"):
        store.write_history(_sample_log(), _sample_digest(), history_id=f"t__{ts}")
    assert len(store.list_unprocessed()) == 3

    store.mark_processed("t__20260531-120000")
    store.mark_processed(["t__20260531-130000", "t__20260531-120000"])  # dedup
    assert store.read_processed_history() == ["t__20260531-120000", "t__20260531-130000"]
    assert store.list_unprocessed() == ["t__20260531-140000"]


def test_make_history_id_slugifies():
    assert make_history_id("Pick / Place!", ) .startswith("Pick-Place__")
