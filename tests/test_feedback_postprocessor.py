"""Tests for module ③ (Feedback Postprocessor + Experience Distill) core.

Uses fake ``query_fn`` / ``judge_fn`` callables and a ``tmp_path`` MemStore, so
no model server, env, or real ``mem/`` is touched.
"""

from __future__ import annotations

from capx.self_evolve import MemStore, run_feedback_postprocessor
from capx.self_evolve.config import ExperienceDistillConfig
from capx.self_evolve.feedback_postprocessor import (
    _human_feedback_from_chat,
    build_distill_prompt,
    build_generalize_rewrite_prompt,
    distill_experience,
    generalize_by_rewrite,
    parse_distill_response,
    render_indexed_transcript,
)
from capx.self_evolve.schemas import DIGEST_SECTIONS

CHAT = [
    {"role": "user", "content": "pick up the red cube"},
    {"role": "assistant", "content": [{"type": "text", "text": "moving to grasp"}, {"type": "image_url", "image_url": {"url": "data:..."}}]},
    {"role": "user", "content": "keep eef 0.03 above the table"},
]

GOOD_DIGEST = """## KEY STRATEGY
approach then grasp with a vertical margin [#1]

## REUSABLE PATTERN
pick(obj) generalizes as an atomic task [#0]

## KEY HYPER-PARAMS / FEEDBACK
z_margin=0.03 from the operator [#2]

## FRAGILITY
none [#1]
"""


# --------------------------------------------------------------------------- #
# transcript / prompt builders
# --------------------------------------------------------------------------- #
def test_indexed_transcript_has_indices_and_flattens_parts():
    text = render_indexed_transcript(CHAT)
    assert "[#0] (user): pick up the red cube" in text
    assert "[#1] (assistant): moving to grasp" in text
    assert "[image]" in text  # image part flattened, not crashed
    assert "[#2] (user): keep eef 0.03 above the table" in text


def test_distill_prompt_lists_all_sections_and_cap():
    msgs = build_distill_prompt("pick the cube", "def main(): pass", CHAT, max_words=200)
    user = msgs[-1]["content"]
    for name in DIGEST_SECTIONS:
        assert name in user
    assert "200 words" in user


# --------------------------------------------------------------------------- #
# parse
# --------------------------------------------------------------------------- #
def test_parse_distill_response_fills_header_and_sections():
    digest = parse_distill_response(
        GOOD_DIGEST, task="pick red cube", datetime_str="2026-05-31 12:00:00", settings_line="env=robosuite"
    )
    digest.validate(max_words=200)
    assert digest.task == "pick red cube"
    assert digest.sections["FRAGILITY"] == "none [#1]"
    assert digest.source_indices() == [1, 0, 2]


# --------------------------------------------------------------------------- #
# generalize-by-rewrite
# --------------------------------------------------------------------------- #
def test_generalize_accepts_then_stops_on_no_change():
    queries = ["def main():\n    z_margin=0.03\n    pick()\n", "def main():\n    z_margin=0.03\n    pick()\n"]
    calls = {"i": 0}

    def query_fn(_msgs):
        out = queries[calls["i"]]
        calls["i"] += 1
        return out

    final, accepted = generalize_by_rewrite(
        "pick", "def main():\n    z=0.03\n    pick()\n",
        query_fn=query_fn, judge_fn=lambda code: True, max_rounds=3,
    )
    assert accepted == 1  # first rewrite accepted; second was identical -> stop
    assert "z_margin" in final


def test_generalize_reverts_on_rejection():
    def query_fn(_msgs):
        return "def main():\n    broken()\n"

    final, accepted = generalize_by_rewrite(
        "pick", "def main():\n    works()\n",
        query_fn=query_fn, judge_fn=lambda code: False, max_rounds=3,
    )
    assert accepted == 0
    assert final == "def main():\n    works()\n"  # baseline kept


# --------------------------------------------------------------------------- #
# (A)/(B) discrimination prompt + feedback extraction
# --------------------------------------------------------------------------- #
def test_human_feedback_from_chat_pulls_only_feedback_turns():
    chat = [
        {"role": "task", "content": "pick the onion"},
        {"role": "assistant", "content": "code..."},
        {"role": "human_feedback", "content": "z 加 2cm margin"},
        {"role": "tool", "content": "moved"},
        {"role": "human_feedback", "content": "多加几个中间 waypoint"},
        {"role": "human_finish", "content": ""},
    ]
    assert _human_feedback_from_chat(chat) == ["z 加 2cm margin", "多加几个中间 waypoint"]


def test_generalize_prompt_carries_classification_feedback_and_api():
    msgs = build_generalize_rewrite_prompt(
        "pick the onion", "pos[2]+=0.02\nwp1=pos+[-0.05,0,0.30]",
        round_index=0, last_was_rejected=False,
        human_feedback=["z 加 2cm margin", "多加几个中间 waypoint 防撞"],
        api_reference="get_object_pose(...)\nget_scene_view(...)\nrefresh_point_clouds()",
    )
    system, user = msgs[0]["content"], msgs[1]["content"]
    # system must teach the (A) keep-as-hyper-param vs (B) re-derive distinction
    assert "(A)" in system and "(B)" in system
    assert "hyper-parameter" in system and "re-derive" in system.lower()
    # the one-off guidance + API surface are both in the user prompt
    assert "z 加 2cm margin" in user and "防撞" in user
    assert "get_scene_view" in user and "refresh_point_clouds" in user


def test_generalize_prompt_degrades_without_feedback_or_api():
    msgs = build_generalize_rewrite_prompt(
        "pick", "pos[2]+=0.02", round_index=1, last_was_rejected=True,
    )
    user = msgs[1]["content"]
    assert "Available APIs" not in user  # api block omitted
    assert "Human guidance given this session" not in user  # feedback block omitted
    assert "more conservative" in user  # rejection note still present


# --------------------------------------------------------------------------- #
# distill length guard
# --------------------------------------------------------------------------- #
def test_distill_shortens_when_over_cap():
    long_digest = GOOD_DIGEST.replace("none [#1]", "word " * 300 + "[#1]")
    responses = [long_digest, GOOD_DIGEST]  # 1st over cap, shorten -> good
    calls = {"i": 0}

    def query_fn(_msgs):
        out = responses[min(calls["i"], len(responses) - 1)]
        calls["i"] += 1
        return out

    digest = distill_experience(
        "pick the cube", "def main(): pass", CHAT,
        query_fn=query_fn, datetime_str="2026-05-31 12:00:00", settings_line="env=robosuite",
        config=ExperienceDistillConfig(max_words=50),
    )
    assert digest.word_count() <= 50
    assert calls["i"] == 2  # one distill + one shorten


# --------------------------------------------------------------------------- #
# end-to-end
# --------------------------------------------------------------------------- #
def test_run_feedback_postprocessor_writes_pair(tmp_path):
    store = MemStore(tmp_path / "mem")
    # query_fn: first the generalize rewrite, then the distill.
    responses = ["def main():\n    z_margin=0.03\n    pick()\n", GOOD_DIGEST]
    calls = {"i": 0}

    def query_fn(_msgs):
        out = responses[min(calls["i"], len(responses) - 1)]
        calls["i"] += 1
        return out

    result = run_feedback_postprocessor(
        task="pick red cube",
        task_description="Goal: pick up the red cube.\nAPIs: ...",
        settings={"env": "robosuite", "llm": "qwen3.6-27B"},
        original_code="def main():\n    z=0.03\n    pick()\n",
        chat_history=CHAT,
        query_fn=query_fn,
        judge_fn=lambda code: True,
        store=store,
        config=ExperienceDistillConfig(max_words=200),
        max_rewrite_rounds=1,
    )

    assert result.rewrite_rounds == 1
    assert "z_margin" in result.final_code

    # The pair landed and round-trips through the store.
    assert store.list_history_ids() == [result.history_id]
    log = store.read_history(result.history_id)
    assert log.task == "pick red cube"
    assert log.final_code == result.final_code
    digest = store.read_digest(result.history_id)
    assert digest.task == "pick red cube"
    assert digest.sections["KEY STRATEGY"]
    assert digest.source_indices()  # has [#idx] backtraces
