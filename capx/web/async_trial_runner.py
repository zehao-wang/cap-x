"""Async trial runner for interactive web UI."""

from __future__ import annotations

import asyncio
import copy
import gc
import logging
import os
import threading
import time
from dataclasses import dataclass
from typing import Any

from capx.envs.configs.instantiate import instantiate
from capx.llm.client import (
    VLM_MODELS,
    ModelQueryArgs,
    query_model as _query_model,
    query_model_streaming as _query_model_streaming,
)
from capx.utils.launch_utils import (
    TrialSummary,
    _build_multi_turn_decision_prompt,
    _extract_code,
    _get_visual_feedback,
    _parse_multi_turn_decision,
    _save_trial_artifacts,
)
from capx.utils.video_utils import _write_video
from capx.web.models import (
    CodeExecutionResultEvent,
    CodeExecutionStartEvent,
    DecisionType,
    EnvironmentInitEvent,
    ErrorEvent,
    ExecutionStepEvent,
    ImageAnalysisEvent,
    ModelResponseEvent,
    ModelStreamingDeltaEvent,
    ModelStreamingEndEvent,
    ModelStreamingStartEvent,
    ModelThinkingEvent,
    SessionState,
    StateUpdateEvent,
    ThinkingPhase,
    TrialCompleteEvent,
    UserPromptRequestEvent,
    VisualFeedbackEvent,
    WSEventBase,
)
from capx.utils import execution_logger
from capx.web.session_manager import (
    FINISH_COMMAND,
    RESET_COMMAND,
    Session,
    run_blocking_with_interrupt,
)

logger = logging.getLogger(__name__)

MULTITURN_LIMIT = 30


def _encode_frame_png(frame) -> tuple[Any, str]:
    """Encode an RGB frame to a (PIL image, ``data:image/png;base64,...`` URL)."""
    from PIL import Image
    import io as _io
    import base64 as _b64

    pil_img = Image.fromarray(frame)
    buf = _io.BytesIO()
    pil_img.save(buf, format="png")
    data_url = f"data:image/png;base64,{_b64.b64encode(buf.getvalue()).decode('utf-8')}"
    return pil_img, data_url


def _merge_consecutive_messages(messages: list[dict]) -> list[dict]:
    """Merge adjacent messages with the same role into one.

    Some chat-template servers require strictly alternating roles. Appending
    human feedback / a reset note as its own user turn (§9.1/§9.2) can produce
    consecutive ``user`` messages; this collapses them just before sending,
    concatenating their content parts. Content is normalized to the list-of-
    parts form so text and image parts merge cleanly. Input is not mutated.
    """
    def _as_parts(content: Any) -> list[dict]:
        if isinstance(content, list):
            return list(content)
        return [{"type": "text", "text": content if content is not None else ""}]

    merged: list[dict] = []
    for msg in messages:
        if merged and merged[-1]["role"] == msg.get("role"):
            merged[-1]["content"] = _as_parts(merged[-1]["content"]) + _as_parts(msg.get("content"))
        else:
            merged.append({**msg, "content": msg.get("content")})
    return merged


@dataclass
class LaunchArgsCompat:
    """Compatible args structure for _query_model."""

    model: str
    server_url: str
    api_key: str | None
    max_tokens: int
    temperature: float
    reasoning_effort: str
    debug: bool

    # Image differencing
    visual_differencing_model: str | None
    visual_differencing_model_server_url: str | None
    visual_differencing_model_api_key: str | None


async def run_trial_async(
    session: Session,
    args: LaunchArgsCompat,
) -> TrialSummary | None:
    """Run a single trial asynchronously with WebSocket event emission.

    Args:
        session: The session containing config, env_factory, and WebSocket connections.
            The session.await_user_input_each_turn setting can be toggled during execution.
        args: Model query arguments.

    Returns:
        TrialSummary on completion, None if cancelled.
    """
    trial_start_time = time.time()
    trial = 1  # Interactive mode runs one trial at a time

    # Helper to emit events
    async def emit(event: WSEventBase) -> None:
        await session.emit(event)

    # Helper to check cancellation
    def is_cancelled() -> bool:
        return session.cancel_event.is_set()

    try:
        # Clear any previous execution histories at trial start
        execution_logger.clear_all_histories()

        # Wait for WebSocket connection before proceeding
        # This ensures early events aren't lost
        logger.info(f"Waiting for WebSocket connection for session {session.session_id}")
        for _ in range(50):  # Wait up to 5 seconds
            if session.websockets:
                logger.info(f"WebSocket connected for session {session.session_id}")
                break
            await asyncio.sleep(0.1)
        else:
            logger.warning(f"No WebSocket connected after 5s for session {session.session_id}")

        # Update state
        session.state = SessionState.RUNNING
        await emit(StateUpdateEvent(
            session_id=session.session_id,
            state=SessionState.RUNNING,
        ))

        # Instantiate environment
        await emit(EnvironmentInitEvent(
            session_id=session.session_id,
            status="starting",
            message="Initializing environment...",
        ))
        # Force enable_render and viser for the web UI so users get 3D visualization
        if "cfg" in session.env_factory:
            session.env_factory["cfg"]["enable_render"] = True
            session.env_factory["cfg"]["viser_debug"] = True

        # Use a single-worker thread pool for ALL env operations so that MuJoCo's
        # thread-local osmesa GL context is always available for rendering.
        import concurrent.futures
        env_executor = concurrent.futures.ThreadPoolExecutor(max_workers=1, thread_name_prefix="env")
        loop = asyncio.get_running_loop()

        async def run_in_env_thread(func, *args):
            return await loop.run_in_executor(env_executor, func, *args)

        logger.info(f"Instantiating environment for session {session.session_id}")
        env = await run_in_env_thread(instantiate, session.env_factory)

        # Interactive mode has no max-step limit: the user drives the episode
        # by hand and is protected by exec_timeout instead of a step horizon.
        # Lift both the high-level truncation bound (low_level_env.max_steps,
        # read in CodeExecutionEnvBase.step) and the underlying robosuite horizon
        # (ignore_done) so move_to_joints never raises "executing action in
        # terminated episode" mid-session. Headless/batch runs keep their limit.
        low_level_env = getattr(env, "low_level_env", None)
        if low_level_env is not None:
            low_level_env.max_steps = 10**9
            robosuite_env = getattr(low_level_env, "robosuite_env", None)
            if robosuite_env is not None:
                robosuite_env.ignore_done = True

        # Store env reference in session for safety interrupt
        session.env = env

        # Enable web UI logging on all API instances
        if hasattr(env, "_apis"):
            for api in env._apis.values():
                api.enable_webui(True)
            logger.info(f"Enabled web UI logging on {len(env._apis)} API(s)")

        # Get prompts from config
        multi_turn_prompt = session.env_factory["cfg"].get("multi_turn_prompt", None)
        task_only_prompt = session.env_factory["cfg"].get("task_only_prompt", None)

        # Reset environment
        await emit(EnvironmentInitEvent(
            session_id=session.session_id,
            status="resetting",
            message="Resetting environment...",
        ))
        # Reset and capture initial frame in the same thread to avoid
        # MuJoCo OpenGL context issues (osmesa contexts are thread-local).
        def _reset_and_render():
            obs, info = env.reset()
            frame = env.render() if hasattr(env, "render") else None
            return obs, info, frame

        obs, _, initial_frame = await run_in_env_thread(_reset_and_render)
        # Snapshot the episode-start state so a human-requested reset can send
        # the simulator back here without sampling a new episode (§9.1).
        episode_snapshot = (
            await run_in_env_thread(env.snapshot_state)
            if low_level_env is not None and hasattr(env, "snapshot_state")
            else None
        )
        # Per-trial artifact dir for SAM3/Molmo intermediate dumps. The reduced
        # APIs are constructed against env.low_level_env, so set both to be
        # robust to either being read by resolve_dump_dir.
        if session.config.get("output_dir"):
            trial_dir_inflight = os.path.join(session.config["output_dir"], f"trial_{trial:02d}")
            env.trial_artifact_dir = trial_dir_inflight
            if hasattr(env, "low_level_env"):
                env.low_level_env.trial_artifact_dir = trial_dir_inflight
        obs["full_prompt"] = copy.deepcopy(obs["full_prompt"])

        # Patch LIBERO task language into prompt template
        from capx.envs.trial import _patch_libero_goal
        _patch_libero_goal(env, obs)

        # Extract the actual (substituted) task prompt for the UI
        actual_task_prompt = None
        try:
            prompt_content = obs["full_prompt"][-1]["content"]
            if isinstance(prompt_content, list):
                actual_task_prompt = prompt_content[0].get("text", "")
            elif isinstance(prompt_content, str):
                actual_task_prompt = prompt_content
        except (KeyError, IndexError):
            pass

        await emit(EnvironmentInitEvent(
            session_id=session.session_id,
            status="complete",
            message="Environment ready",
            description_content=actual_task_prompt,
        ))

        # Enable video capture if configured
        if session.config.get("record_video") and hasattr(env, "enable_video_capture"):
            env.enable_video_capture(True, clear=True)

        # Initialize tracking variables
        raw_code = None
        code_blocks: list[str] = []
        code_block_metadata: list[dict] = []
        code_block_idx = 0
        num_regenerations = 0
        num_finishes = 0
        all_responses: list[dict] = []
        visual_feedback_imgs = []
        visual_feedback_base64_history: list[str] = []
        stderr_history: list[str] = []

        info_step = {"sandbox_rc": -1, "stdout": "", "stderr": "", "task_completed": False}
        reward = 0.0
        terminated = truncated = False

        # Build image differencing args if needed
        visual_differencing_args = ModelQueryArgs(
            model=args.visual_differencing_model,
            server_url=args.visual_differencing_model_server_url,
            api_key=args.visual_differencing_model_api_key,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            reasoning_effort=args.reasoning_effort,
            debug=args.debug,
        )

        use_visual_feedback = session.config.get("use_visual_feedback", False)
        use_img_differencing = session.config.get("use_img_differencing", False)

        # Build initial visual feedback from the frame captured in the reset thread
        initial_visual_feedback_base64 = None
        if (
            (use_visual_feedback and args.model in VLM_MODELS)
            or (use_img_differencing and visual_differencing_args.model in VLM_MODELS)
        ) and initial_frame is not None:
            initial_visual_feedback_img, initial_visual_feedback_base64 = _encode_frame_png(initial_frame)
            visual_feedback_imgs.append(initial_visual_feedback_img)
            visual_feedback_base64_history.append(initial_visual_feedback_base64)

            # Emit initial visual feedback
            await emit(VisualFeedbackEvent(
                session_id=session.session_id,
                image_base64=initial_visual_feedback_base64,
                description="Initial environment state",
            ))

        # Snapshot the clean, text-only task prompt before any visual feedback /
        # image-differencing text is appended. On a human-requested reset we
        # rebuild the conversation from this clean base + only the most recent
        # code as context (§9.1), rather than carrying the full history.
        clean_base_prompt = copy.deepcopy(obs["full_prompt"])

        # Add visual feedback to prompt if enabled
        if use_visual_feedback and initial_visual_feedback_base64:
            obs["full_prompt"][-1]["content"][0]["text"] += (
                "\n\nIncluded below is an image of the initial state of the environment."
            )
            obs["full_prompt"][-1]["content"].append(
                {"type": "image_url", "image_url": {"url": initial_visual_feedback_base64}}
            )

        # Handle image differencing for initial state
        # Use task_only_prompt if provided, otherwise extract from the full prompt
        # (matches launch.py behaviour which deep-copies the prompt text)
        task_description = task_only_prompt or copy.deepcopy(
            obs["full_prompt"][-1]["content"][0]["text"]
        )
        if use_img_differencing and initial_visual_feedback_base64:

            await emit(EnvironmentInitEvent(
                session_id=session.session_id,
                status="building_description",
                message="Building initial environment description...",
            ))

            initial_env_description_prompt = [
                {
                    "role": "system",
                    "content": "You are a helpful assistant that describes the initial state of the environment with the goal of the task in mind. Do *NOT* write any code. Provide ONLY task-relevant information.",
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": task_description},
                        {"type": "text", "text": "Describe the initial state of the environment with the goal of the task in mind. Do *NOT* write any code. Provide ONLY task-relevant information."},
                        {"type": "image_url", "image_url": {"url": initial_visual_feedback_base64}},
                    ],
                },
            ]

            initial_env_description_out = await asyncio.to_thread(
                _query_model, visual_differencing_args, initial_env_description_prompt
            )
            initial_env_description = initial_env_description_out["content"]
            initial_visual_differencing_feedback = f"The initial state of the environment is described as follows:\n{initial_env_description}"
            obs["full_prompt"][-1]["content"][0]["text"] += f"\n\n{initial_visual_differencing_feedback}"

            await emit(EnvironmentInitEvent(
                session_id=session.session_id,
                status="description_complete",
                message="Environment description ready",
                description_content=initial_env_description,
            ))

        # Display the prompt to the user before generation starts
        prompt_text = ""
        for msg in obs["full_prompt"]:
            content = msg.get("content", "")
            if isinstance(content, list):
                content = "\n".join(p.get("text", "") for p in content if isinstance(p, dict))
            if content:
                prompt_text += f"**[{msg.get('role', 'unknown')}]**\n{content}\n\n"
        if prompt_text:
            await emit(EnvironmentInitEvent(
                session_id=session.session_id,
                status="description_complete",
                message="Task prompt",
                description_content=prompt_text.strip(),
            ))

        if is_cancelled():
            raise asyncio.CancelledError("Cancelled before initial code generation")

        # ====================================================================
        # Streaming helper — drives one model query end-to-end, emitting the
        # streaming delta events and returning (content, reasoning). Shared by
        # initial generation, the multi-turn decision, and reset re-planning so
        # the streaming plumbing lives in exactly one place.
        # ====================================================================
        async def _stream_query(prompt, phase, turn_no):
            # Collapse any consecutive same-role messages (feedback / reset turns
            # can create adjacent user messages) so role-alternating servers
            # don't choke. See _merge_consecutive_messages.
            prompt = _merge_consecutive_messages(prompt)
            await emit(ModelStreamingStartEvent(
                session_id=session.session_id,
                phase=phase,
                turn_number=turn_no,
                model_name=args.model,
            ))
            q: asyncio.Queue = asyncio.Queue()
            cap_loop = asyncio.get_running_loop()

            def _pump():
                try:
                    for chunk in _query_model_streaming(args, prompt):
                        asyncio.run_coroutine_threadsafe(q.put(chunk), cap_loop)
                except Exception as e:  # surface as a queued error chunk
                    asyncio.run_coroutine_threadsafe(
                        q.put({"type": "error", "error": str(e)}), cap_loop
                    )

            pump_task = cap_loop.run_in_executor(None, _pump)
            content = ""
            reasoning_out = None
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
            return content, reasoning_out

        # ====================================================================
        # Human-pause helper — between turns we surface the state and block for
        # a human action. Returns one of:
        #   ("finish",   "")    human confirms success -> end trial (§4.1)
        #   ("reset",    "")    legacy reset command (button removed; harmless)
        #   ("feedback", text)  free-text feedback -> reset scene & re-attempt
        #   ("continue", "")    empty send -> no-op (keep waiting)
        # Waits indefinitely: interactive sessions have NO turn timeout / auto-
        # restart — the human drives every step. The Stop button cancels the
        # task, which interrupts this await.
        # ====================================================================
        async def _wait_for_human(reward_val):
            # Drain leftovers from a previous resume so we don't auto-continue.
            while not session.user_injection_queue.empty():
                try:
                    session.user_injection_queue.get_nowait()
                except asyncio.QueueEmpty:
                    break
            session.state = SessionState.AWAITING_USER_INPUT
            await emit(StateUpdateEvent(
                session_id=session.session_id,
                state=SessionState.AWAITING_USER_INPUT,
            ))
            await emit(UserPromptRequestEvent(
                session_id=session.session_id,
                current_state_summary=(
                    f"Executed {code_block_idx} of {len(code_blocks)} code blocks. "
                    f"Reward: {reward_val:.3f}"
                ),
                executed_code_blocks=code_block_idx,
            ))
            payload = await session.user_injection_queue.get()
            session.state = SessionState.RUNNING
            await emit(StateUpdateEvent(
                session_id=session.session_id, state=SessionState.RUNNING,
            ))
            if payload == RESET_COMMAND:
                return "reset", ""
            if payload == FINISH_COMMAND:
                return "finish", ""
            if payload:
                return "feedback", payload
            return "continue", ""

        # Human feedback targets only the most recent code and is not retained
        # long-term (§9.2): each turn's decision prompt is rebuilt from the clean
        # task base plus the (unchanged) multi-turn template, with the current
        # turn's feedback appended as its own user turn — no accumulation, no
        # extra multi-step scaffolding.
        # Most recent executed code — the only context retained across a reset.
        last_executed_code = ""
        # Set when the human confirms success; the sole success signal in
        # interactive mode (§4.1).
        human_finished = False

        # ========================================================================
        # Initial code generation (with streaming)
        # ========================================================================
        turn_number = 1  # Track multi-turn iterations (starts at 1)
        logger.info("Querying model for initial code generation (streaming)")
        raw_code, reasoning = await _stream_query(
            obs["full_prompt"], ThinkingPhase.INITIAL, turn_number
        )

        if not raw_code:
            raise ValueError("Model returned empty response during streaming")

        initial_blocks = _extract_code(raw_code)
        code_blocks.extend(initial_blocks)
        code_block_metadata.extend([{"generation": 0, "regenerated": False}] * len(initial_blocks))

        session.total_code_blocks = len(code_blocks)

        await emit(ModelStreamingEndEvent(
            session_id=session.session_id,
            content=raw_code,
            reasoning=reasoning,
            code_blocks=initial_blocks,
            decision=DecisionType.INITIAL,
        ))

        all_responses.append({
            "block_idx": [code_block_idx],
            "code_blocks": initial_blocks,
            "decision": "initial",
            "reasoning": reasoning if reasoning else "",
        })

        # ========================================================================
        # Code execution loop
        # ========================================================================
        # In interactive mode the human decides when to stop, so the MULTITURN
        # turn cap does not apply; headless/non-interactive runs keep it.
        while code_block_idx < len(code_blocks) and (
            session.await_user_input_each_turn or code_block_idx <= MULTITURN_LIMIT
        ):
            if is_cancelled():
                raise asyncio.CancelledError("Cancelled during code execution")

            code = code_blocks[code_block_idx]
            last_executed_code = code
            session.current_block_index = code_block_idx

            # Check for cancellation before executing (safety check)
            if is_cancelled():
                raise asyncio.CancelledError("Cancelled before code execution")

            await emit(CodeExecutionStartEvent(
                session_id=session.session_id,
                block_index=code_block_idx,
                code=code,
            ))

            logger.info(f"Executing code block {code_block_idx}")

            # Set up execution logger to capture detailed execution steps
            # The emit callback sends events via WebSocket in real-time
            step_counter = [0]  # Use list to allow mutation in closure

            # Capture event loop BEFORE thread execution (critical!)
            # The callback runs in a thread pool executor, so we need the main loop reference
            exec_main_loop = asyncio.get_running_loop()

            def emit_execution_step(step: execution_logger.ExecutionStep) -> None:
                """Sync callback to emit execution steps during code execution."""
                try:
                    # Create and emit the event by scheduling in the captured main event loop
                    event = ExecutionStepEvent(
                        session_id=session.session_id,
                        block_index=code_block_idx,
                        step_index=step.step_index,
                        tool_name=step.tool_name,
                        text=step.text,
                        images=step.images,
                        highlight=step.highlight,
                    )
                    asyncio.run_coroutine_threadsafe(session.emit(event), exec_main_loop)
                    step_counter[0] += 1
                except Exception as e:
                    logger.warning(f"Failed to emit execution step: {e}")

            # Initialize execution logger for this code block
            execution_logger.init_execution_context(
                code_block_index=code_block_idx,
                emit_callback=emit_execution_step,
            )

            try:
                # Run step + render in the same thread to keep MuJoCo GL context.
                # We capture the frame here because MuJoCo's osmesa GL context is
                # thread-local — rendering must happen on the same thread as the
                # simulation step.
                def _step_and_render(code):
                    result = env.step(code)
                    try:
                        frame = env.render() if hasattr(env, "render") else None
                    except Exception:
                        frame = None
                    return result, frame

                def _step_render_with_interrupt(code):
                    session.execution_thread_id = threading.get_ident()
                    try:
                        return _step_and_render(code)
                    finally:
                        session.execution_thread_id = None

                exec_timeout = getattr(session, 'execution_timeout', 180)
                try:
                    (obs_next, reward, terminated, truncated, info_step), post_step_frame = await asyncio.wait_for(
                        run_in_env_thread(_step_render_with_interrupt, code),
                        timeout=exec_timeout,
                    )
                except asyncio.TimeoutError:
                    logger.warning(f"Code block {code_block_idx} timed out after {exec_timeout}s")
                    await emit(CodeExecutionResultEvent(
                        session_id=session.session_id,
                        block_index=code_block_idx,
                        success=False,
                        stdout="",
                        stderr=f"Execution timed out after {exec_timeout} seconds. The code may be stuck in a loop or waiting for an unreachable target.",
                        reward=0.0,
                        task_completed=False,
                    ))
                    # Reset env for next attempt
                    try:
                        obs, _ = await run_in_env_thread(lambda: env.reset())
                    except Exception:
                        pass
                    break  # Exit code block loop, go to multi-turn decision
            finally:
                # Finalize execution logger and get history
                exec_history = execution_logger.finalize_execution_context()
                if exec_history and step_counter[0] > 0:
                    logger.info(f"Code block {code_block_idx} had {len(exec_history.steps)} execution steps")

            # Check for cancellation after executing (might have been stopped during execution)
            if is_cancelled():
                raise asyncio.CancelledError("Cancelled after code execution")

            await emit(CodeExecutionResultEvent(
                session_id=session.session_id,
                block_index=code_block_idx,
                success=info_step["sandbox_rc"] == 0,
                stdout=info_step["stdout"],
                stderr=info_step["stderr"],
                reward=reward,
                task_completed=info_step.get("task_completed"),
            ))

            code_block_idx += 1
            obs = obs_next

            # ====================================================================
            # Multi-turn decision
            # ====================================================================
            if multi_turn_prompt:
                # Compute the executed and remaining code blocks
                # import pdb; pdb.set_trace()
                executed_code = "# Prior executed code blocks:\n"
                for block_idx in range(code_block_idx):
                    if block_idx < code_block_idx - 1:
                        executed_code += f"# Code block {block_idx}\n{code_blocks[block_idx]}\n"
                    else:
                        executed_code += f"\n\n# Last executed code block (Code block {block_idx}):\n{code_blocks[block_idx]}\n"
                # executed_code = "\n".join(code_blocks[:code_block_idx])
                # remaining_code = "\n".join(code_blocks[code_block_idx:])
                # Check for episode termination
                if "terminated episode" in info_step["stderr"]:
                    truncated = True
                    break

                # Build multi-turn prompt
                complete_multi_turn_prompt = multi_turn_prompt.format(
                    executed_code=executed_code,
                    console_stdout=info_step["stdout"],
                    console_stderr=info_step["stderr"],
                )

                if info_step["stderr"]:
                    stderr_history.append(info_step["stderr"])

                # Build visual feedback from the frame captured in the step thread
                visual_feedback_base64 = None
                if (
                    (use_visual_feedback and args.model in VLM_MODELS)
                    or (use_img_differencing and visual_differencing_args.model in VLM_MODELS)
                ) and post_step_frame is not None:
                    visual_feedback_img, visual_feedback_base64 = _encode_frame_png(post_step_frame)
                    visual_feedback_imgs.append(visual_feedback_img)
                    visual_feedback_base64_history.append(visual_feedback_base64)

                    await emit(VisualFeedbackEvent(
                        session_id=session.session_id,
                        image_base64=visual_feedback_base64,
                    ))

                # Image differencing
                visual_differencing_feedback = None
                if use_img_differencing and len(visual_feedback_base64_history) >= 2:
                    visual_differencing_prompt = [
                        {
                            "role": "system",
                            "content": "You are a helpful assistant that describes the difference between the current state of the environment and the previous state of the environment with the goal of the task in mind and whether the task has been completed. Do *NOT* write any code.",
                        },
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": task_description},
                                {"type": "text", "text": "Describe the difference between the current state of the environment and the previous state of the environment with the goal of the task in mind and whether the task has been completed. Do *NOT* write any code.."},
                                {"type": "text", "text": "Previous state:"},
                                {"type": "image_url", "image_url": {"url": visual_feedback_base64_history[-2]}},
                                {"type": "text", "text": "Current state:"},
                                {"type": "image_url", "image_url": {"url": visual_feedback_base64_history[-1]}},
                            ],
                        },
                    ]
                    img_diff_response = await asyncio.to_thread(
                        _query_model, visual_differencing_args, visual_differencing_prompt
                    )
                    visual_differencing_feedback = img_diff_response.get("content") if img_diff_response else None

                    # Emit image analysis event for visualization
                    if visual_differencing_feedback:
                        await emit(ImageAnalysisEvent(
                            session_id=session.session_id,
                            analysis_type="state_comparison",
                            content=visual_differencing_feedback,
                            model_used=visual_differencing_args.model,
                        ))

                # Zero out visual feedback if not using it
                if not use_visual_feedback:
                    visual_feedback_base64 = None

                # ----------------------------------------------------------------
                # Turn handling — diverges by mode.
                #
                # Interactive (§9, simplified): reached only *after* a code block
                # executed, so the human always reviews a fresh result. They either
                # Finish (the sole success signal, §4.1) or send — and *any* send
                # resets the scene to the episode start and regenerates a full fresh
                # attempt from [task + previous code + feedback]. There is no
                # incremental multi-turn and no separate Reset action; resetting
                # every attempt also keeps the viser playback timeline clean (one
                # episode per attempt).
                #
                # Headless/benchmark: unchanged — the model drives REGENERATE /
                # FINISH itself with no human in the loop.
                # ----------------------------------------------------------------
                headless_finish = False

                if session.await_user_input_each_turn:
                    has_snapshot = episode_snapshot is not None and hasattr(env, "restore_state")

                    def _restore_and_render():
                        # Exact restore to the episode start, or a fresh reset if
                        # no snapshot was captured (non-sim env) — task restarts.
                        if has_snapshot:
                            o = env.restore_state(episode_snapshot)
                        else:
                            o, _i = env.reset()
                        f = env.render() if hasattr(env, "render") else None
                        return o, f

                    # Pause for the human. Finish ends the trial; only actual
                    # feedback resets the scene and re-attempts. An empty send
                    # is a no-op (keep waiting) — there is no turn timeout / auto
                    # restart; we only ever modify this one trial via feedback.
                    new_blocks: list[str] = []
                    while True:
                        action, fb_text = await _wait_for_human(reward)
                        if action == "finish":
                            human_finished = True
                            break
                        if action != "feedback":
                            # Empty send / legacy reset with no text: nothing to
                            # modify — re-pause and keep waiting.
                            continue
                        feedback_text = fb_text

                        if not has_snapshot:
                            logger.warning("No episode snapshot; falling back to env.reset()")
                            await emit(ErrorEvent(
                                session_id=session.session_id,
                                message=(
                                    "No episode snapshot was available — restarting from a "
                                    "fresh reset instead of the exact episode start."
                                ),
                                recoverable=True,
                            ))
                        obs, reset_frame = await run_in_env_thread(_restore_and_render)

                        # Regeneration context: task + previous attempt's code +
                        # the human's feedback (the only added context).
                        gen_prompt = copy.deepcopy(clean_base_prompt)
                        reset_note = (
                            "The simulator has been reset to the start of this episode; "
                            "none of the earlier steps persist. Your previous attempt's "
                            "code was:\n```python\n"
                            f"{last_executed_code}\n```"
                        )
                        gen_prompt.append({
                            "role": "user",
                            "content": [{"type": "text", "text": reset_note}],
                        })
                        if use_visual_feedback and reset_frame is not None:
                            _img, _b64url = _encode_frame_png(reset_frame)
                            gen_prompt[-1]["content"].append(
                                {"type": "image_url", "image_url": {"url": _b64url}}
                            )
                            await emit(VisualFeedbackEvent(
                                session_id=session.session_id,
                                image_base64=_b64url,
                                description="Environment reset",
                            ))
                        if feedback_text:
                            logger.info(f"Human feedback: {feedback_text[:100]}...")
                            gen_prompt.append({
                                "role": "user",
                                "content": [{"type": "text", "text": feedback_text}],
                            })

                        turn_number += 1
                        raw_code, reasoning = await _stream_query(
                            gen_prompt, ThinkingPhase.INITIAL, turn_number
                        )
                        new_blocks = _extract_code(raw_code) if raw_code else []
                        await emit(ModelStreamingEndEvent(
                            session_id=session.session_id,
                            content=raw_code,
                            reasoning=reasoning,
                            code_blocks=new_blocks,
                            decision=DecisionType.INITIAL,
                        ))
                        all_responses.append({
                            "block_idx": [0],
                            "code_blocks": new_blocks,
                            "decision": "feedback_retry",
                            "reasoning": reasoning if reasoning else "",
                        })
                        if new_blocks:
                            break  # got code -> execute the fresh attempt
                        await emit(ErrorEvent(
                            session_id=session.session_id,
                            message="The model produced no code. Add feedback and try again.",
                            recoverable=True,
                        ))

                    if human_finished:
                        break  # exit execution loop

                    code_blocks = list(new_blocks)
                    code_block_metadata = [{"generation": 0, "regenerated": False}] * len(new_blocks)
                    code_block_idx = 0
                    session.total_code_blocks = len(code_blocks)
                    continue  # restart execution loop with the fresh attempt

                # Headless/benchmark: model decides REGENERATE / FINISH (unchanged).
                turn_number += 1
                multi_turn_decision_prompt = _build_multi_turn_decision_prompt(
                    {"full_prompt": clean_base_prompt},
                    complete_multi_turn_prompt,
                    visual_feedback_base64,
                    visual_differencing_feedback,
                )
                mt_content, mt_reasoning = await _stream_query(
                    multi_turn_decision_prompt, ThinkingPhase.MULTI_TURN, turn_number
                )
                if not mt_content:
                    reasoning = None
                    decision, new_code = "finish", None
                else:
                    reasoning = mt_reasoning
                    decision, new_code = _parse_multi_turn_decision(mt_content)

                if decision == "regenerate":
                    new_blocks = _extract_code(new_code)
                    await emit(ModelStreamingEndEvent(
                        session_id=session.session_id,
                        content=mt_content,
                        reasoning=reasoning,
                        code_blocks=new_blocks,
                        decision=DecisionType.REGENERATE,
                    ))
                    all_responses.append({
                        "multi_turn_prompt": multi_turn_decision_prompt,
                        "block_idx": [code_block_idx],
                        "code_blocks": new_blocks,
                        "decision": "regenerate",
                        "reasoning": reasoning if reasoning else "",
                    })
                    del code_blocks[code_block_idx:]
                    del code_block_metadata[code_block_idx:]
                    code_blocks.extend(new_blocks)
                    code_block_metadata.extend([
                        {"generation": num_regenerations + 1, "regenerated": True, "regenerated_at_idx": code_block_idx}
                    ] * len(new_blocks))
                    num_regenerations += 1
                    session.num_regenerations = num_regenerations
                    session.total_code_blocks = len(code_blocks)
                else:  # finish
                    await emit(ModelStreamingEndEvent(
                        session_id=session.session_id,
                        content=mt_content,
                        reasoning=reasoning,
                        code_blocks=[],
                        decision=DecisionType.FINISH,
                    ))
                    all_responses.append({
                        "multi_turn_prompt": multi_turn_decision_prompt,
                        "decision": "finish",
                        "reasoning": reasoning if reasoning else new_code if new_code else "",
                    })
                    num_finishes += 1
                    headless_finish = True

                if headless_finish:
                    break

            logger.info(f"Code block {code_block_idx} done, {len(code_blocks)} total blocks")

        # ========================================================================
        # Trial complete
        # ========================================================================
        logger.info("Trial execution complete")

        # Build final code with annotations
        annotated_blocks = []
        for i, (block, metadata) in enumerate(zip(code_blocks, code_block_metadata, strict=False)):
            header = f"# Code block {i}"
            if metadata.get("regenerated"):
                header += f" (regenerated at step {metadata.get('regenerated_at_idx', '?')})"
            annotated_blocks.append(f"{header}\n{block}")

        final_code = "\n\n".join(annotated_blocks)

        # Handle sandbox_rc override for max steps
        if "executing action in terminated episode" in info_step.get("stderr", ""):
            info_step["sandbox_rc"] = 0

        stderr = "\n\n".join(stderr_history) if stderr_history else info_step.get("stderr", "")

        log_lines = [
            "-" * 100,
            "Generated program:",
            final_code,
            "\n\nEnvironment response:",
            f"  Sandbox failed: {info_step['sandbox_rc']}",
            f"  Stdout: {info_step['stdout']}",
            f"  Stderr: {stderr}",
            f"  Reward: {reward}",
            f"  Task Completed: {info_step.get('task_completed', 'N/A')}",
            f"  Terminated: {terminated}, Truncated: {truncated}",
            f"  Num Regenerations: {num_regenerations}",
            f"  Num Finishes: {num_finishes}",
            f"  Num Code Blocks: {len(code_blocks)}",
            "-" * 100,
        ]

        # Save artifacts if output_dir configured
        code_path = None
        output_dir = session.config.get("output_dir")
        if output_dir:
            logger.info(f"Saving trial artifacts to: {output_dir}")
            code_path = await asyncio.to_thread(
                _save_trial_artifacts,
                session.config,
                trial,
                info_step["sandbox_rc"],
                reward,
                info_step.get("task_completed", False),
                final_code,
                raw_code,
                all_responses,
                log_lines,
                visual_feedback_imgs,
            )
            logger.info(f"Trial artifacts saved to: {code_path}")

            # Save execution histories
            all_exec_histories = execution_logger.get_all_histories()
            if all_exec_histories:
                from pathlib import Path
                exec_history_dir = Path(output_dir) / "execution_history"
                for history in all_exec_histories:
                    await asyncio.to_thread(history.save_to_directory, exec_history_dir)
                logger.info(f"Saved {len(all_exec_histories)} execution histories to: {exec_history_dir}")

            # Save video if configured
            if session.config.get("record_video") and hasattr(env, "get_video_frames"):
                frames = env.get_video_frames(clear=True)
                if frames:
                    video_dir = os.path.join(
                        output_dir,
                        f"trial_{trial:02d}_sandboxrc_{info_step['sandbox_rc']}_reward_{reward:.3f}_taskcompleted_{int(info_step.get('task_completed', False))}",
                    )
                    await asyncio.to_thread(
                        _write_video,
                        frames,
                        video_dir,
                        suffix=f"{reward:.3f}",
                    )
        else:
            logger.info("No output_dir configured, skipping artifact save")

        # task_completed reflects the environment's own reward signal (shown for
        # reference). The success signal differs by mode:
        #   - interactive: only the human can declare success (§4.1), via the
        #     Finish action (human_finished). The model's FINISH never counts.
        #   - headless/non-interactive: fall back to the env / model finish.
        task_completed = bool(info_step.get("task_completed", False)) or (terminated and reward > 0)
        if session.await_user_input_each_turn:
            success = human_finished
        else:
            success = task_completed or num_finishes > 0

        # Emit completion
        session.state = SessionState.COMPLETE
        await emit(TrialCompleteEvent(
            session_id=session.session_id,
            success=success,
            total_reward=reward,
            task_completed=task_completed,
            num_regenerations=num_regenerations,
            num_code_blocks=len(code_blocks),
            summary="\n".join(log_lines),
        ))
        await emit(StateUpdateEvent(
            session_id=session.session_id,
            state=SessionState.COMPLETE,
        ))

        trial_end_time = time.time()
        logger.info(f"Trial completed in {trial_end_time - trial_start_time:.2f} seconds")

        # Cleanup
        gc.collect()

        return TrialSummary(
            trial=trial,
            success=success,
            reward=reward,
            terminated=terminated,
            truncated=truncated,
            sandbox_rc=info_step["sandbox_rc"],
            log="\n".join(log_lines),
            task_completed=task_completed,
            code_path=code_path,
            num_regenerations=num_regenerations,
            num_finishes=num_finishes,
            num_code_blocks=len(code_blocks),
        )

    except asyncio.CancelledError as e:
        logger.info(f"Trial cancelled: {e}")
        session.state = SessionState.IDLE
        await emit(StateUpdateEvent(
            session_id=session.session_id,
            state=SessionState.IDLE,
        ))
        return None

    except KeyboardInterrupt:
        logger.info("Trial interrupted by user (stop button)")
        session.state = SessionState.IDLE
        await emit(StateUpdateEvent(
            session_id=session.session_id,
            state=SessionState.IDLE,
        ))
        return None

    except Exception as e:
        import traceback
        tb_str = traceback.format_exc()
        logger.exception(f"Trial error: {e}")
        session.state = SessionState.ERROR
        error_msg = str(e) if str(e) else f"{type(e).__name__}: {tb_str.splitlines()[-2].strip()}"
        await emit(ErrorEvent(
            session_id=session.session_id,
            message=error_msg,
            recoverable=True,
        ))
        await emit(StateUpdateEvent(
            session_id=session.session_id,
            state=SessionState.ERROR,
        ))
        # Don't re-raise - error has been communicated via WebSocket
        # This allows clean recovery without task exception handling issues
        return None
