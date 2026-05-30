"""Auxiliary trial-artifact saving.

Split out of :mod:`capx.web.async_trial_runner`. The primary code/response dump
is handled by ``launch_utils._save_trial_artifacts``; this module saves the
extras that hang off the live env at trial end: the viser observation history
(one ``observations.npz`` per attempt), the execution-step histories, and the
recorded video. Each save is best-effort and isolated so one failing does not
abort the others or the trial.
"""

from __future__ import annotations

import asyncio
import logging
import os
from pathlib import Path
from typing import Any

from capx.utils import execution_logger
from capx.utils.video_utils import _write_video

logger = logging.getLogger(__name__)


async def save_auxiliary_artifacts(
    *,
    env: Any,
    output_dir: str,
    trial: int,
    info_step: dict,
    reward: float,
    record_video: bool,
) -> None:
    """Save the viser history, execution histories, and video for a trial."""
    # Viser observations: one observations.npz per attempt, written into the
    # SAME attempt_NN/ folder as that attempt's LLM trace
    # (trial_NN/attempt_NN/observations.npz). Only the task config's standard
    # views are recorded, so the saved cameras match what the agent observes.
    for cand in (getattr(env, "low_level_env", None), env):
        fh = getattr(cand, "frame_history", None) if cand is not None else None
        if fh is None:
            continue
        try:
            trial_dir = os.path.join(output_dir, f"trial_{trial:02d}")
            written = await asyncio.to_thread(fh.save_segments, trial_dir)
            logger.info(f"Saved {len(written)} attempt observation file(s) under: {trial_dir}")
        except Exception as exc:
            logger.warning(f"viser segment save failed: {exc}")
        break

    # Execution-step histories.
    all_exec_histories = execution_logger.get_all_histories()
    if all_exec_histories:
        exec_history_dir = Path(output_dir) / "execution_history"
        for history in all_exec_histories:
            await asyncio.to_thread(history.save_to_directory, exec_history_dir)
        logger.info(f"Saved {len(all_exec_histories)} execution histories to: {exec_history_dir}")

    # Recorded video (result-suffixed dir so it sits next to the trial dump).
    if record_video and hasattr(env, "get_video_frames"):
        frames = env.get_video_frames(clear=True)
        if frames:
            video_dir = os.path.join(
                output_dir,
                f"trial_{trial:02d}_sandboxrc_{info_step['sandbox_rc']}"
                f"_reward_{reward:.3f}"
                f"_taskcompleted_{int(info_step.get('task_completed', False))}",
            )
            await asyncio.to_thread(_write_video, frames, video_dir, suffix=f"{reward:.3f}")


__all__ = ["save_auxiliary_artifacts"]
