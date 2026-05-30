"""Streaming model-query helper.

Split out of :mod:`capx.web.async_trial_runner`. Drives one model query
end-to-end: collapses consecutive same-role messages, pumps the blocking
streaming generator on a worker thread, forwards delta events over the
WebSocket, logs the call to the trace, and returns ``(content, reasoning)``.
Shared by initial generation, the multi-turn decision, and feedback re-planning
so the streaming plumbing lives in exactly one place.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any, Awaitable, Callable

from capx.llm.client import query_model_streaming as _query_model_streaming
from capx.web.models import (
    ModelStreamingDeltaEvent,
    ModelStreamingStartEvent,
    WSEventBase,
)
from capx.web.trial_support import merge_consecutive_messages


async def stream_query(
    prompt: list[dict],
    phase: Any,
    turn_no: int,
    *,
    args: Any,
    session: Any,
    emit: Callable[[WSEventBase], Awaitable[None]],
    is_cancelled: Callable[[], bool],
    trace: Any,
) -> tuple[str, str | None]:
    """Run one streaming model query and return ``(content, reasoning)``."""
    # Collapse any consecutive same-role messages (feedback / reset turns can
    # create adjacent user messages) so role-alternating servers don't choke.
    prompt = merge_consecutive_messages(prompt)
    _t0 = time.time()
    await emit(ModelStreamingStartEvent(
        session_id=session.session_id,
        phase=phase,
        turn_number=turn_no,
        model_name=args.model,
    ))
    q: asyncio.Queue = asyncio.Queue()
    cap_loop = asyncio.get_running_loop()

    def _pump():
        gen = _query_model_streaming(args, prompt)
        try:
            for chunk in gen:
                if is_cancelled():
                    # Close the generator so its underlying streaming HTTP
                    # connection is released now, instead of leaking until the
                    # request timeout fires.
                    gen.close()
                    break
                asyncio.run_coroutine_threadsafe(q.put(chunk), cap_loop)
        except Exception as e:  # surface as a queued error chunk
            asyncio.run_coroutine_threadsafe(
                q.put({"type": "error", "error": str(e)}), cap_loop
            )

    pump_task = cap_loop.run_in_executor(None, _pump)
    content = ""
    reasoning_out = None
    try:
        while True:
            if is_cancelled():
                raise asyncio.CancelledError("Cancelled during model streaming")
            try:
                chunk = await asyncio.wait_for(q.get(), timeout=1.0)
            except asyncio.TimeoutError:
                if pump_task.done():
                    break
                continue
            if chunk["type"] == "content_delta":
                await emit(ModelStreamingDeltaEvent(
                    session_id=session.session_id, content_delta=chunk["content"],
                ))
            elif chunk["type"] == "reasoning_delta":
                await emit(ModelStreamingDeltaEvent(
                    session_id=session.session_id, reasoning_delta=chunk["content"],
                ))
            elif chunk["type"] == "done":
                content = chunk["content"]
                reasoning_out = chunk.get("reasoning")
                break
            elif chunk["type"] == "error":
                raise RuntimeError(f"Streaming error: {chunk['error']}")
        await pump_task
    finally:
        # On cancel / error we stop waiting on the pump. The executor thread
        # can't be force-killed, but the is_cancelled() check in _pump +
        # gen.close() let it unwind on the next chunk and free the connection
        # rather than leaking it.
        if not pump_task.done():
            pump_task.cancel()
    trace.log_llm(
        phase=getattr(phase, "value", str(phase)),
        turn=turn_no,
        input_messages=prompt,
        output_content=content,
        output_reasoning=reasoning_out,
        model=args.model,
        duration_s=time.time() - _t0,
    )
    return content, reasoning_out


__all__ = ["stream_query"]
