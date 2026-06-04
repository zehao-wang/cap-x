"""Async trial runner for interactive web UI."""

from __future__ import annotations

import asyncio
import copy
import gc
import logging
import os
import threading
import time

from capx.envs.configs.instantiate import instantiate
from capx.llm.client import (
    VLM_MODELS,
    ModelQueryArgs,
    query_model as _query_model,
)
from capx.utils.launch_utils import (
    TrialSummary,
    _build_multi_turn_decision_prompt,
    _extract_code,
    _get_visual_feedback,
    _parse_multi_turn_decision,
    _save_trial_artifacts,
)
from capx.harnesses.prompt import (
    get_vdm_task_description_from_env,
    prepare_multiturn_console_text,
)
from capx.web.models import (
    CodeExecutionResultEvent,
    CodeExecutionStartEvent,
    DecisionType,
    EnvironmentInitEvent,
    ErrorEvent,
    ExecutionStepEvent,
    ImageAnalysisEvent,
    ModelResponseEvent,
    ModelStreamingEndEvent,
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
from capx.utils.trace_logger import NullTraceLogger, TraceLogger
from capx.web.session_manager import (
    FINISH_COMMAND,
    RESET_COMMAND,
    WIZARD_CONFIRM,
    Session,
    run_blocking_with_interrupt,
)
from capx.web.trial_support import (
    LaunchArgsCompat,
    build_feedback_distill_prompt,
    build_feedback_regeneration_block,
    encode_frame_png as _encode_frame_png,
)
from capx.web.model_stream import stream_query
from capx.web.reset_wizard import GuidedResetWizard
from capx.web.trial_artifacts import save_auxiliary_artifacts
from capx.web import vdm_feedback

logger = logging.getLogger(__name__)

MULTITURN_LIMIT = 30


def _append_task_to_prompt(full_prompt: list, task_text: str) -> None:
    """Fold the operator's runtime task into the last prompt message in place.

    Handles both content shapes the harness uses: a list of content parts whose
    first part is the ``{"type": "text", "text": ...}`` block, or a plain string.
    """
    addition = f"\n\nThe task for this trial is:\n{task_text}"
    if not full_prompt:
        return
    msg = full_prompt[-1]
    content = msg.get("content")
    if isinstance(content, list):
        for part in content:
            if isinstance(part, dict) and "text" in part:
                part["text"] = (part.get("text") or "") + addition
                return
        content.insert(0, {"type": "text", "text": addition.strip()})
    elif isinstance(content, str):
        msg["content"] = content + addition
    else:
        msg["content"] = addition.strip()


def _annotate_code(code_blocks: list, code_block_metadata: list) -> str:
    """Join code blocks into one annotated program (the form saved as code.py).

    Shared by the trial's top-level code.py and the per-attempt
    trial_NN/attempt_NN/code.py so they have identical formatting.
    """
    parts = []
    for i, (block, meta) in enumerate(zip(code_blocks, code_block_metadata, strict=False)):
        header = f"# Code block {i}"
        if isinstance(meta, dict) and meta.get("regenerated"):
            header += f" (regenerated at step {meta.get('regenerated_at_idx', '?')})"
        parts.append(f"{header}\n{block}")
    return "\n\n".join(parts)


def _next_trial_number(output_dir: str | None) -> int:
    """Next free ``trial_NN`` index in ``output_dir`` (1 if none / no dir).

    Interactive runs one trial per ``run_trial_async`` call but reuses the same
    per-session ``output_dir``; numbering by the next free index keeps each new
    trial's trace / handoff / artifacts in its own ``trial_NN`` instead of
    overwriting ``trial_01``.
    """
    if not output_dir or not os.path.isdir(output_dir):
        return 1
    import re
    nums = [
        int(m.group(1))
        for name in os.listdir(output_dir)
        if (m := re.match(r"trial_(\d+)", name))
    ]
    return (max(nums) + 1) if nums else 1


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
    # Number this trial as the next free trial_NN in the (per-session, reused)
    # output_dir so "new trial" doesn't overwrite the previous trial's trace /
    # handoff / artifacts (they all key off this number).
    trial = _next_trial_number(
        session.config.get("output_dir") if getattr(session, "config", None) else None
    )

    # Per-trial agent<->LLM / tool trace. Reassigned to a real TraceLogger once
    # the output dir is known; the no-op default keeps the finally block safe if
    # we fail before then.
    trace: TraceLogger | NullTraceLogger = NullTraceLogger()

    # Single-worker pool for ALL env ops (created below once the trial starts).
    # Tracked here so the finally block can tear it down on every exit path and
    # not leak a worker thread per trial.
    env_executor = None
    # Real-robot low-level env running a background live-preview daemon, if any.
    # Tracked so the finally block stops it on every exit path.
    live_preview_env = None

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

        # Real-robot backends expose a guided-reset capability (connection check
        # + return-to-rest-pose). Detected by duck typing on the low-level env so
        # the runner stays agnostic of the concrete hardware class. When set, the
        # initial and feedback resets run the interactive wizard instead of a
        # plain env.reset() (sim restore_state path is untouched).
        is_real_robot = low_level_env is not None and hasattr(
            low_level_env, "return_to_rest_pose"
        )

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

        # Reset and capture initial frame in the same thread to avoid
        # MuJoCo OpenGL context issues (osmesa contexts are thread-local).
        def _reset_and_render():
            obs, info = env.reset()
            frame = env.render() if hasattr(env, "render") else None
            return obs, info, frame

        # Guided real-robot reset wizard (real backends only); the sim path
        # never uses it. See capx.web.reset_wizard.GuidedResetWizard.
        reset_wizard = GuidedResetWizard(
            session=session,
            emit=emit,
            is_cancelled=is_cancelled,
            run_in_env_thread=run_in_env_thread,
            low_level_env=low_level_env,
            reset_and_render=_reset_and_render,
        )
        _guided_real_robot_reset = reset_wizard.run

        # Reset environment
        await emit(EnvironmentInitEvent(
            session_id=session.session_id,
            status="resetting",
            message="Resetting environment...",
        ))

        if is_real_robot:
            obs, initial_frame = await _guided_real_robot_reset("initial")
        else:
            obs, _, initial_frame = await run_in_env_thread(_reset_and_render)
        # Snapshot the episode-start state so a human-requested reset can send
        # the simulator back here without sampling a new episode (§9.1).
        episode_snapshot = (
            await run_in_env_thread(env.snapshot_state)
            if low_level_env is not None and hasattr(env, "snapshot_state")
            else None
        )
        # Real-robot envs expose a background live-preview daemon: keep the viser
        # Camera View showing a live ZED RGB-D feed while the session is idle
        # (waiting for the task / a human / while the LLM streams), not just
        # during commanded motion. No-op for sim backends.
        if low_level_env is not None and hasattr(low_level_env, "start_live_preview"):
            await run_in_env_thread(low_level_env.start_live_preview)
            live_preview_env = low_level_env
        # Per-trial artifact dir for SAM3/Molmo intermediate dumps. The reduced
        # APIs are constructed against env.low_level_env, so set both to be
        # robust to either being read by resolve_dump_dir.
        # `trace` records every agent<->LLM exchange and tool-call step into this
        # dir for debugging / behaviour tracing (llm_trace.jsonl, events.jsonl,
        # trace.md); it is a no-op when no output dir is configured.
        if session.config.get("output_dir"):
            trial_dir_inflight = os.path.join(session.config["output_dir"], f"trial_{trial:02d}")
            env.trial_artifact_dir = trial_dir_inflight
            if hasattr(env, "low_level_env"):
                env.low_level_env.trial_artifact_dir = trial_dir_inflight
            trace = TraceLogger(trial_dir_inflight, model=args.model)
        else:
            trace = NullTraceLogger()
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

        # ====================================================================
        # Runtime task gate (prompt_for_task configs, e.g. the real robot).
        # The env is up — arm homed and, with the live-preview daemon running,
        # the viser Camera View now shows a live ZED feed. Instead of
        # auto-starting whatever the baked-in prompt implies, pause and ask the
        # operator to type the task. The typed text is folded into the task
        # prompt BEFORE the clean-base snapshot / visual-feedback / VDM steps
        # below, so every downstream consumer (and feedback re-runs) sees it.
        # ====================================================================
        prompt_for_task = bool(session.env_factory.get("cfg", {}).get("prompt_for_task", False))
        if prompt_for_task:
            # Drain any stale injections so we don't auto-consume a leftover.
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
                    "请输入本次 trial 的任务（例如：把红色方块放到盘子里）。"
                ),
                executed_code_blocks=0,
            ))
            task_text = None
            while True:
                if is_cancelled():
                    raise asyncio.CancelledError("Cancelled while awaiting task")
                payload = await session.user_injection_queue.get()
                # Finish before any task → nothing to run; end the trial cleanly.
                if payload == FINISH_COMMAND:
                    task_text = None
                    break
                # Reset / empty send before a task is meaningless here — keep
                # waiting rather than starting an unspecified task.
                if payload == RESET_COMMAND or not payload or not payload.strip():
                    continue
                task_text = payload.strip()
                break

            if task_text is None:
                session.state = SessionState.COMPLETE
                await emit(TrialCompleteEvent(
                    session_id=session.session_id,
                    success=False,
                    total_reward=0.0,
                    task_completed=False,
                    num_regenerations=0,
                    num_code_blocks=0,
                    summary="No task was given before the session ended.",
                ))
                await emit(StateUpdateEvent(
                    session_id=session.session_id,
                    state=SessionState.COMPLETE,
                ))
                return None

            session.state = SessionState.RUNNING
            await emit(StateUpdateEvent(
                session_id=session.session_id,
                state=SessionState.RUNNING,
            ))
            _append_task_to_prompt(obs["full_prompt"], task_text)
            actual_task_prompt = (actual_task_prompt or "") + f"\n\nTask: {task_text}"

        # Enable video capture if configured
        if session.config.get("record_video") and hasattr(env, "enable_video_capture"):
            env.enable_video_capture(True, clear=True)

        # Initialize tracking variables
        raw_code = None
        code_blocks: list[str] = []
        code_block_metadata: list[dict] = []
        code_block_idx = 0
        # Whether the current attempt has executed any code yet. Mirrors the
        # viser frame_history "empty segment" guard so the per-attempt trace
        # rolls over (trace.new_attempt) in lockstep with new viser segments:
        # a feedback retry that produced no code does not advance either.
        attempt_has_run = False
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

        # Handle image differencing for initial state. Keep the VDM context
        # task-scoped: no code-generation rules or API reference.
        task_description = get_vdm_task_description_from_env(
            env,
            obs["full_prompt"][-1]["content"][0]["text"],
            task_only_prompt=task_only_prompt,
        )
        if use_img_differencing and initial_visual_feedback_base64:
            initial_env_description = await vdm_feedback.describe_initial_state(
                task_description=task_description,
                image_b64=initial_visual_feedback_base64,
                vdm_args=visual_differencing_args,
                session=session,
                emit=emit,
                trace=trace,
            )
            obs["full_prompt"][-1]["content"][0]["text"] += (
                "\n\nThe initial state of the environment is described as "
                f"follows:\n{initial_env_description}"
            )

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

        # Streaming helper — drives one model query end-to-end, emitting delta
        # events and returning (content, reasoning). Thin adapter over
        # capx.web.model_stream.stream_query that binds this trial's
        # collaborators; shared by initial generation, the multi-turn decision,
        # and feedback re-planning.
        async def _stream_query(prompt, phase, turn_no):
            return await stream_query(
                prompt, phase, turn_no,
                args=args, session=session, emit=emit,
                is_cancelled=is_cancelled, trace=trace,
            )

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

        # Each regeneration is rebuilt from the clean task base + a single
        # labelled block (cumulative operator guidance + the previous attempt's
        # key failure + the verbatim latest feedback). The previous attempt's
        # full code is NOT carried; cumulative feedback is kept concise by a
        # dedicated distiller call (§9.2), so context stays bounded.
        human_finished = False  # set on human-confirmed success (sole signal, §4.1)
        operator_guidance = ""  # distilled, deduplicated guidance carried across rounds

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
        # Execution loop
        # ========================================================================
        # The model runs its OWN self-driven multi-turn loop (REGENERATE/FINISH
        # scaffold + optional visual feedback / differencing), exactly as in
        # headless. In INTERACTIVE mode that loop is wrapped in a human cycle:
        # once the model's loop ends — it FINISHed, the episode terminated, it
        # errored/timed out, or it hit the turn cap — we pause and ask the human
        # (§1, §3):
        #   - Finish   -> the SOLE success signal; ends the trial.
        #   - feedback -> reset the scene to the episode start and re-run the
        #                 WHOLE model multi-turn from [task prompt (incl. API /
        #                 tool docs) + previous attempt's full code + feedback].
        #   - empty    -> no-op, keep waiting (no turn timeout / auto restart).
        # Headless runs the loop once and stops on the model's FINISH / cap.
        interactive = session.await_user_input_each_turn
        force_end = False          # request the model multi-turn loop to end now
        post_step_frame = None     # latest rendered frame (for the human to judge)
        # Console output of the last executed block — carried into a feedback
        # retry as the attempt's "key failure" evidence (not the full code).
        last_exec = {"stdout": "", "stderr": ""}
        while True:
            if is_cancelled():
                raise asyncio.CancelledError("Cancelled during code execution")

            # ----------------------------------------------------------------
            # Has the model's multi-turn loop ended this pass?
            # ----------------------------------------------------------------
            if force_end or code_block_idx >= len(code_blocks) or code_block_idx > MULTITURN_LIMIT:
                if not interactive:
                    break  # headless: the trial ends here

                # Surface the latest frame (cheap encode; no differencing LLM) so
                # the human reviews the result they are about to judge.
                if (
                    use_visual_feedback
                    and args.model in VLM_MODELS
                    and post_step_frame is not None
                ):
                    _img, _b64 = _encode_frame_png(post_step_frame)
                    visual_feedback_imgs.append(_img)
                    await emit(VisualFeedbackEvent(
                        session_id=session.session_id, image_base64=_b64,
                    ))

                has_snapshot = episode_snapshot is not None and hasattr(env, "restore_state")

                def _restore_and_render():
                    # Exact restore to the episode start, or a fresh reset if no
                    # snapshot was captured (non-sim env) — task restarts.
                    if has_snapshot:
                        logger.info("interactive feedback: restoring episode snapshot")
                        o = env.restore_state(episode_snapshot)
                    else:
                        logger.info("interactive feedback: no snapshot -> env.reset()")
                        o, _i = env.reset()
                    f = env.render() if hasattr(env, "render") else None
                    return o, f

                # Pause for the human. Finish ends the trial; feedback distils
                # into cumulative operator guidance, resets the scene, and
                # re-attempts; an empty send is a no-op (keep waiting).
                new_blocks: list[str] = []
                while True:
                    action, fb_text = await _wait_for_human(reward)
                    if action == "finish":
                        human_finished = True
                        # The sole interactive success signal — record it as a
                        # first-class human turn so the postprocessor handoff
                        # can mark which attempt the human confirmed.
                        trace.log_human(kind="finish", turn=turn_number)
                        break
                    if action != "feedback":
                        continue  # empty send / legacy reset: keep waiting
                    feedback_text = fb_text
                    logger.info(f"Human feedback: {feedback_text[:100]}...")
                    # Log the verbatim feedback as a human turn (before the
                    # distiller / reset) so it lands on the attempt it judged.
                    trace.log_human(kind="feedback", text=feedback_text, turn=turn_number)

                    # Key failure of the attempt just run — carried into the
                    # regeneration as console evidence (NOT the full code), bounded
                    # by the prompt harness so the context stays small.
                    failure_console = prepare_multiturn_console_text(
                        last_exec["stdout"], last_exec["stderr"],
                    )

                    # Dedicated distiller call: fold this feedback (+ the failure)
                    # into one SHORT, deduplicated operator-guidance block that
                    # persists across rounds. Keeps specialised work specialised
                    # and the regeneration context bounded (§9.2).
                    distill_prompt = build_feedback_distill_prompt(
                        task_description,
                        operator_guidance,
                        feedback_text,
                        failure_console.stdout,
                        failure_console.stderr,
                    )
                    # A distiller failure must not abort the trial or discard the
                    # human's feedback: keep the prior guidance and let the retry
                    # proceed on the verbatim feedback + failure block below.
                    try:
                        distill_out = await asyncio.to_thread(_query_model, args, distill_prompt)
                        distilled = (distill_out or {}).get("content")
                    except Exception as e:  # noqa: BLE001 - degrade gracefully
                        logger.warning(f"Feedback distillation failed; keeping prior guidance: {e}")
                        distilled = None
                    if distilled and distilled.strip():
                        operator_guidance = distilled.strip()
                    trace.log_llm(
                        phase="feedback_distill",
                        turn=turn_number,
                        input_messages=distill_prompt,
                        output_content=operator_guidance,
                        model=args.model,
                    )
                    await emit(ImageAnalysisEvent(
                        session_id=session.session_id,
                        analysis_type="operator_guidance",
                        content=operator_guidance,
                        model_used=args.model,
                    ))

                    if is_real_robot:
                        # Real hardware: run the guided 3-step reset wizard
                        # (connection check -> auto-home + confirm -> rearrange).
                        obs, reset_frame = await _guided_real_robot_reset("feedback")
                    else:
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

                    # The reset above started a fresh viser segment (new_segment
                    # in restore_state); roll the trace over to the matching
                    # attempt_NN/ so this retry's conversation is logged on its
                    # own. Guarded like the viser empty-segment check: a retry
                    # that produced no code does not advance the attempt index.
                    if attempt_has_run:
                        # Persist the just-finished attempt's code into its own
                        # attempt_NN/ before the trace rolls over to the next one.
                        trace.save_attempt_code(_annotate_code(code_blocks, code_block_metadata))
                        trace.new_attempt()
                        attempt_has_run = False

                    # Regeneration context = clean task base (KEEPS the API / tool
                    # docs) + one labelled block: cumulative operator guidance, the
                    # previous attempt's key failure, and the verbatim latest
                    # feedback. The previous attempt's full code is NOT carried.
                    gen_prompt = copy.deepcopy(clean_base_prompt)
                    gen_prompt.append({
                        "role": "user",
                        "content": [{
                            "type": "text",
                            "text": build_feedback_regeneration_block(
                                operator_guidance,
                                feedback_text,
                                failure_console.stdout,
                                failure_console.stderr,
                            ),
                        }],
                    })
                    if reset_frame is not None:
                        _img, _b64url = _encode_frame_png(reset_frame)
                        # Always surface the post-reset frame so the human can
                        # visually confirm the scene returned to the episode start
                        # (independent of use_visual_feedback / viser playback).
                        await emit(VisualFeedbackEvent(
                            session_id=session.session_id,
                            image_base64=_b64url,
                            description="Environment reset",
                        ))
                        # Only feed it to the model when visual feedback is on.
                        if use_visual_feedback:
                            gen_prompt[-1]["content"].append(
                                {"type": "image_url", "image_url": {"url": _b64url}}
                            )

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
                        break  # got code -> re-run the model multi-turn on it
                    await emit(ErrorEvent(
                        session_id=session.session_id,
                        message="The model produced no code. Add feedback and try again.",
                        recoverable=True,
                    ))

                if human_finished:
                    break  # exit execution loop -> trial complete

                # Re-run the WHOLE model multi-turn on the fresh attempt.
                code_blocks = list(new_blocks)
                code_block_metadata = [{"generation": 0, "regenerated": False}] * len(new_blocks)
                code_block_idx = 0
                force_end = False
                session.total_code_blocks = len(code_blocks)
                continue

            code = code_blocks[code_block_idx]
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
            # This attempt is now running code -> its viser segment will hold
            # frames and its trace dir is "used", so the next reset rolls over.
            attempt_has_run = True

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
                    trace.log_tool(
                        tool_name=step.tool_name,
                        text=step.text,
                        block_index=code_block_idx,
                        step_index=step.step_index,
                        n_images=len(step.images),
                    )
                except Exception as e:
                    logger.warning(f"Failed to emit execution step: {e}")

            # Initialize execution logger for this code block
            execution_logger.init_execution_context(
                code_block_index=code_block_idx,
                emit_callback=emit_execution_step,
            )

            timed_out = False
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
                    timeout_msg = (
                        f"Execution timed out after {exec_timeout} seconds. The code "
                        "may be stuck in a loop or waiting for an unreachable target."
                    )
                    last_exec = {"stdout": "", "stderr": timeout_msg}
                    await emit(CodeExecutionResultEvent(
                        session_id=session.session_id,
                        block_index=code_block_idx,
                        success=False,
                        stdout="",
                        stderr=timeout_msg,
                        reward=0.0,
                        task_completed=False,
                    ))
                    # Reset env for next attempt
                    try:
                        obs, _ = await run_in_env_thread(lambda: env.reset())
                    except Exception:
                        pass
                    timed_out = True
            finally:
                # Finalize execution logger and get history
                exec_history = execution_logger.finalize_execution_context()
                if exec_history and step_counter[0] > 0:
                    logger.info(f"Code block {code_block_idx} had {len(exec_history.steps)} execution steps")

            # A timeout leaves obs_next/reward unset: end the model loop now.
            # Headless stops; interactive pauses for the human at the loop top.
            if timed_out:
                force_end = True
                continue

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
            last_exec = {"stdout": info_step["stdout"], "stderr": info_step["stderr"]}

            code_block_idx += 1
            obs = obs_next

            if info_step.get("stderr"):
                stderr_history.append(info_step["stderr"])
            if "terminated episode" in (info_step.get("stderr") or ""):
                truncated = True
                force_end = True
                continue

            # ====================================================================
            # Model self-driven multi-turn decision (REGENERATE / FINISH).
            # Drives BOTH interactive and headless when multi_turn_prompt is
            # configured. In interactive, FINISH ends THIS model loop (not the
            # trial) and the human pause at the loop top then fires.
            # ====================================================================
            if multi_turn_prompt:
                # Compute the executed code blocks for the decision prompt.
                executed_code = "# Prior executed code blocks:\n"
                for block_idx in range(code_block_idx):
                    if block_idx < code_block_idx - 1:
                        executed_code += f"# Code block {block_idx}\n{code_blocks[block_idx]}\n"
                    else:
                        executed_code += f"\n\n# Last executed code block (Code block {block_idx}):\n{code_blocks[block_idx]}\n"

                # A hard execution error (non-zero sandbox rc -> stderr traceback)
                # means the task cannot be complete via this turn. Route straight
                # to a code revision: skip the VDM (it only confuses things and
                # has misjudged broken states as "done"), and instruct the model
                # to fix the error rather than FINISH.
                errored = info_step.get("sandbox_rc", 0) != 0

                # Build multi-turn prompt
                console_text = prepare_multiturn_console_text(
                    info_step["stdout"],
                    info_step["stderr"],
                )
                complete_multi_turn_prompt = multi_turn_prompt.format(
                    executed_code=executed_code,
                    console_stdout=console_text.stdout,
                    console_stderr=console_text.stderr,
                )

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

                # Image differencing — skipped on error (don't route errors
                # through the VDM). Otherwise the VDM also gets the console
                # stdout for grounding (a single camera view often can't show a
                # small lift / height change).
                visual_differencing_feedback = None
                if use_img_differencing and not errored and len(visual_feedback_base64_history) >= 2:
                    visual_differencing_feedback = await vdm_feedback.diff_states(
                        task_description=task_description,
                        prev_b64=visual_feedback_base64_history[-2],
                        cur_b64=visual_feedback_base64_history[-1],
                        console_output=info_step.get("stdout"),
                        vdm_args=visual_differencing_args,
                        turn_number=turn_number,
                        session=session,
                        emit=emit,
                        trace=trace,
                    )

                # Zero out visual feedback if not using it
                if not use_visual_feedback:
                    visual_feedback_base64 = None

                # Model decides REGENERATE / FINISH. On FINISH we set force_end
                # so the loop top ends the pass — headless stops, interactive
                # pauses for the human.
                turn_number += 1
                multi_turn_decision_prompt = _build_multi_turn_decision_prompt(
                    {"full_prompt": clean_base_prompt},
                    complete_multi_turn_prompt,
                    visual_feedback_base64,
                    visual_differencing_feedback,
                )
                if errored:
                    # Push hard toward a revision: the code raised an error, so
                    # the task is not done — fix it, don't FINISH.
                    multi_turn_decision_prompt.append({
                        "role": "user",
                        "content": [{
                            "type": "text",
                            "text": (
                                "The code above raised an error (see stderr); the task is "
                                "NOT complete. Respond with REGENERATE and corrected Python "
                                "code that fixes the error. Do NOT respond FINISH."
                            ),
                        }],
                    })
                mt_content, mt_reasoning = await _stream_query(
                    multi_turn_decision_prompt, ThinkingPhase.MULTI_TURN, turn_number
                )
                if not mt_content:
                    reasoning = None
                    decision, new_code = "finish", None
                else:
                    reasoning = mt_reasoning
                    decision, new_code = _parse_multi_turn_decision(mt_content)

                # Never let a hard error end the trial as "finished": if the model
                # still chose FINISH (or returned nothing parseable as code) on an
                # error, treat it as a regeneration of the failed code so the loop
                # keeps trying instead of stopping on a broken state.
                if errored and decision == "finish":
                    logger.warning("Model chose FINISH despite a hard error; forcing regenerate")
                    decision = "regenerate"
                    if not _extract_code(new_code or ""):
                        new_code = mt_content or ""

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
                    force_end = True

            logger.info(f"Code block {code_block_idx} done, {len(code_blocks)} total blocks")

        # ========================================================================
        # Trial complete
        # ========================================================================
        logger.info("Trial execution complete")

        # Build final code with annotations; also drop it into the last attempt's
        # folder so every attempt — including the final/winning one — has its own
        # attempt_NN/code.py (this one equals the trial's top-level code.py).
        final_code = _annotate_code(code_blocks, code_block_metadata)
        trace.save_attempt_code(final_code)

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
            # Unified per-trial folder: code/logs land in trial_NN/ next to the
            # LLM trace + handoff (instead of a separate metric-named sibling), so
            # everything for one trial lives in one place.
            trial_dir = os.path.join(output_dir, f"trial_{trial:02d}")
            logger.info(f"Saving trial artifacts to: {trial_dir}")
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
                trial_dir=trial_dir,
            )
            logger.info(f"Trial artifacts saved to: {code_path}")

            # On human-confirmed success, write the Feedback Postprocessor's
            # input contract (success-log schema draft): task / settings /
            # final_code / chat_history (incl. verbatim human feedback) /
            # datetime. This is the single artifact the postprocessor consumes
            # — it does not re-parse per-attempt traces.
            if human_finished:
                try:
                    handoff_settings = {
                        "model": args.model,
                        "env_config": getattr(session, "config_path", None),
                        "use_visual_feedback": use_visual_feedback,
                        "use_img_differencing": use_img_differencing,
                    }
                    # The full API/tool prompt (perception APIs etc.) the agent saw;
                    # module ③ needs it to re-derive scene-specific parts from
                    # perception during the generalize-rewrite step.
                    api_reference = "\n\n".join(
                        part["text"]
                        for msg in clean_base_prompt
                        for part in (msg.get("content") if isinstance(msg.get("content"), list) else [])
                        if isinstance(part, dict) and part.get("type") == "text" and part.get("text")
                    ) or None
                    handoff_path = await asyncio.to_thread(
                        trace.write_handoff,
                        task=task_description,
                        settings=handoff_settings,
                        final_code=final_code,
                        api_reference=api_reference,
                    )
                    logger.info(f"Postprocessor handoff written to: {handoff_path}")
                except Exception as _handoff_exc:  # noqa: BLE001
                    logger.warning(f"Failed to write postprocessor handoff: {_handoff_exc}")

            # Auxiliary saves that hang off the live env: per-attempt viser
            # observations, execution-step histories, and the recorded video.
            await save_auxiliary_artifacts(
                env=env,
                output_dir=output_dir,
                trial=trial,
                info_step=info_step,
                reward=reward,
                record_video=bool(session.config.get("record_video")),
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

    finally:
        # Stop the real-robot live-preview daemon (no-op for sim) so it doesn't
        # keep reading the cameras after the trial ends.
        if live_preview_env is not None:
            try:
                live_preview_env.stop_live_preview()
            except Exception as _preview_exc:
                logger.warning(f"Failed to stop live preview: {_preview_exc}")
        # Render the human-readable trace.md from the live-written events on
        # every exit path (success, Stop-button cancel, or error) — the JSONL
        # files are written live, this just renders the readable timeline.
        try:
            await asyncio.to_thread(trace.finalize)
        except Exception as _trace_exc:
            logger.warning(f"Failed to render trace.md: {_trace_exc}")
        # Release the env worker pool so a finished/cancelled trial doesn't leak
        # its thread. wait=False: an in-flight env.step() can't be force-killed,
        # but idle workers exit and queued futures are dropped instead of piling
        # up across trials. (The process-exit hang is handled by the Ctrl-C
        # watchdog in launch._run_web_ui.)
        if env_executor is not None:
            env_executor.shutdown(wait=False, cancel_futures=True)
