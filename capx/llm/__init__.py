"""LLM client module for querying language models.

This module provides utilities for querying various LLM providers
(OpenAI, Claude, open-source models, OpenRouter) with support for
streaming, ensemble queries, and backward-compatible aliases.
"""

from capx.llm.client import (
    ENSEMBLE_CONFIGS,
    VLM_MODELS,
    ModelQueryArgs,
    _completions_to_responses_convert_prompt,
    build_payload,
    collapse_text_image_inputs,
    is_openrouter_model,
    lane_server_url,
    query_model,
    query_model_ensemble,
    query_model_streaming,
    query_single_model_ensemble,
    resolve_lane,
)

__all__ = [
    "ENSEMBLE_CONFIGS",
    "VLM_MODELS",
    "ModelQueryArgs",
    "_completions_to_responses_convert_prompt",
    "build_payload",
    "collapse_text_image_inputs",
    "is_openrouter_model",
    "lane_server_url",
    "query_model",
    "query_model_ensemble",
    "query_model_streaming",
    "query_single_model_ensemble",
    "resolve_lane",
]
