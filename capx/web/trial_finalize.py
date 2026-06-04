"""Trial finalize phase, split out of :mod:`capx.web.async_trial_runner`.

Runs once the trial's execution loop exits normally: build the final annotated
code, persist artifacts into the unified ``trial_NN/`` folder, write the
postprocess handoff on human-confirmed success, save the auxiliary (viser / video
/ execution-history) artifacts, decide success, emit completion, and return the
:class:`TrialSummary`. All inputs are passed explicitly (keyword-only) so this is
a pure tail of the runner with no hidden state.
"""

from __future__ import annotations

import asyncio
import gc
import logging
import os
import time
from typing import Any, Awaitable, Callable

from capx.utils.launch_utils import TrialSummary, _save_trial_artifacts
from capx.web.models import SessionState, StateUpdateEvent, TrialCompleteEvent, WSEventBase
from capx.web.session_manager import Session
from capx.web.trial_artifacts import save_auxiliary_artifacts
from capx.web.trial_helpers import annotate_code

logger = logging.getLogger(__name__)


async def finalize_trial(
    *,
    session: Session,
    args: Any,
    trace: Any,
    env: Any,
    emit: Callable[[WSEventBase], Awaitable[None]],
    trial: int,
    trial_start_time: float,
    code_blocks: list,
    code_block_metadata: list,
    info_step: dict,
    stderr_history: list,
    reward: float,
    terminated: bool,
    truncated: bool,
    num_regenerations: int,
    num_finishes: int,
    raw_code: str | None,
    all_responses: list,
    visual_feedback_imgs: list,
    human_finished: bool,
    use_visual_feedback: bool,
    use_img_differencing: bool,
    clean_base_prompt: list,
    task_description: str,
) -> TrialSummary:
    """Persist + report a completed trial; returns its :class:`TrialSummary`."""
    logger.info("Trial execution complete")

    # Build final code with annotations; also drop it into the last attempt's
    # folder so every attempt — including the final/winning one — has its own
    # attempt_NN/code.py (this one equals the trial's top-level code.py).
    final_code = annotate_code(code_blocks, code_block_metadata)
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
