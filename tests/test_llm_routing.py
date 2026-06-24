"""Tests for the 3-lane LLM backend router in capx.llm.client.

The harness contract: the user only sets `model`; the model name alone picks
the backend lane, its endpoint, and the request shape. Only gpt-5.5 (Codex CLI
proxy) and Qwen3.6 (local vLLM) are special; everything else is OpenRouter.
"""

import pytest

from capx.llm.client import (
    ModelQueryArgs,
    build_payload,
    is_openrouter_model,
    lane_server_url,
    resolve_lane,
)


@pytest.mark.parametrize(
    "model, lane",
    [
        ("gpt-5.5", "codex"),
        ("Qwen3.6-27B", "qwen"),
        ("qwen3.6", "qwen"),
        ("anthropic/claude-opus-4-5", "openrouter"),
        ("google/gemini-3.1-pro-preview", "openrouter"),
        ("openai/gpt-5.4", "openrouter"),
        ("some/unknown-model", "openrouter"),
    ],
)
def test_resolve_lane(model, lane):
    assert resolve_lane(model) == lane


def test_is_openrouter_model_is_the_default_lane():
    assert is_openrouter_model("anything/at-all")
    assert not is_openrouter_model("gpt-5.5")
    assert not is_openrouter_model("Qwen3.6-27B")


def test_lane_default_urls():
    assert lane_server_url("codex").endswith(":8110/chat/completions")
    assert lane_server_url("qwen").endswith(":8000/v1/chat/completions")
    assert lane_server_url("openrouter").endswith(":8110/chat/completions")


def test_lane_url_env_override(monkeypatch):
    monkeypatch.setenv("CAPX_QWEN_URL", "http://gpu-box:9000/v1/chat/completions")
    assert lane_server_url("qwen") == "http://gpu-box:9000/v1/chat/completions"


def test_codex_payload_omits_temperature():
    args = ModelQueryArgs(model="gpt-5.5", server_url="", temperature=0.7, max_tokens=512)
    payload = build_payload("codex", "gpt-5.5", args, [{"role": "user", "content": "hi"}])
    assert payload == {
        "model": "gpt-5.5",
        "messages": [{"role": "user", "content": "hi"}],
        "max_completion_tokens": 512,
    }


def test_openrouter_payload_is_plain_chat_shape():
    args = ModelQueryArgs(model="x", server_url="", temperature=0.3, max_tokens=256)
    payload = build_payload("openrouter", "anthropic/claude-opus-4-5", args, [{"role": "user", "content": "hi"}])
    assert payload == {
        "model": "anthropic/claude-opus-4-5",
        "messages": [{"role": "user", "content": "hi"}],
        "temperature": 0.3,
        "max_tokens": 256,
    }


def test_stream_flag_added():
    args = ModelQueryArgs(model="x", server_url="")
    assert build_payload("openrouter", "x", args, [], stream=True)["stream"] is True
