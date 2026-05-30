"""Visual-differencing model (VDM) feedback orchestration.

Split out of :mod:`capx.web.async_trial_runner`. Two task-scoped VDM calls that
ground the agent with a vision model:

* :func:`describe_initial_state` — a one-off description of the reset scene,
  folded into the initial task prompt.
* :func:`diff_states` — the per-turn before/after difference (with the executed
  code's console stdout for grounding) used as multi-turn visual feedback.

Both keep the VDM context task-only (no code-generation rules / API reference),
query the VDM off the event loop, log the call to the trace, and emit the
matching UI event. They return the produced text (or ``None``) and never mutate
the caller's prompt — the runner decides where to splice the result in.
"""

from __future__ import annotations

import asyncio
from typing import Any, Awaitable, Callable

from capx.llm.client import query_model as _query_model
from capx.web.models import EnvironmentInitEvent, ImageAnalysisEvent, WSEventBase
from capx.web.trial_support import build_initial_state_prompt, build_state_diff_prompt

Emit = Callable[[WSEventBase], Awaitable[None]]


async def describe_initial_state(
    *,
    task_description: str,
    image_b64: str,
    vdm_args: Any,
    session: Any,
    emit: Emit,
    trace: Any,
) -> str | None:
    """Describe the reset scene with the VDM; return the description text."""
    await emit(EnvironmentInitEvent(
        session_id=session.session_id,
        status="building_description",
        message="Building initial environment description...",
    ))

    prompt = build_initial_state_prompt(task_description, image_b64)
    out = await asyncio.to_thread(_query_model, vdm_args, prompt)
    description = out["content"] if out else None
    trace.log_llm(
        phase="env_description",
        turn=0,
        input_messages=prompt,
        output_content=description,
        model=vdm_args.model,
    )

    await emit(EnvironmentInitEvent(
        session_id=session.session_id,
        status="description_complete",
        message="Environment description ready",
        description_content=description,
    ))
    return description


async def diff_states(
    *,
    task_description: str,
    prev_b64: str,
    cur_b64: str,
    console_output: str | None,
    vdm_args: Any,
    turn_number: int,
    session: Any,
    emit: Emit,
    trace: Any,
) -> str | None:
    """Compare before/after states with the VDM; return the difference text."""
    prompt = build_state_diff_prompt(
        task_description,
        prev_b64,
        cur_b64,
        console_output=console_output,
    )
    response = await asyncio.to_thread(_query_model, vdm_args, prompt)
    feedback = response.get("content") if response else None
    trace.log_llm(
        phase="img_differencing",
        turn=turn_number,
        input_messages=prompt,
        output_content=feedback,
        model=vdm_args.model,
    )
    if feedback:
        await emit(ImageAnalysisEvent(
            session_id=session.session_id,
            analysis_type="state_comparison",
            content=feedback,
            model_used=vdm_args.model,
        ))
    return feedback


__all__ = ["describe_initial_state", "diff_states"]
