"""Guided real-robot reset wizard.

Split out of :mod:`capx.web.async_trial_runner`. Real-robot backends expose a
guided-reset capability (connection check + return-to-rest-pose); on the initial
reset and on every feedback retry the runner drives this interactive 3-step
wizard instead of a plain ``env.reset()`` (the sim ``restore_state`` path never
touches this). Each step blocks on a human button press delivered via
``session.wizard_response_queue`` (Ready / ✓ / ✗).

The wizard is given its collaborators explicitly (the runner's ``emit`` /
``is_cancelled`` / ``run_in_env_thread`` helpers, the low-level env, and the
shared ``reset_and_render`` closure) so it carries no hidden state and stays
trivially testable.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Awaitable, Callable

from capx.web.models import ResetWizardEvent, SessionState, StateUpdateEvent, WSEventBase
from capx.web.session_manager import WIZARD_READY, WIZARD_REJECT, Session

logger = logging.getLogger(__name__)


class GuidedResetWizard:
    """Runs the interactive 3-step real-robot reset and returns ``(obs, frame)``.

    Step 1: loop until the arm middleware is connected (Ready re-checks).
    Step 2: auto-home, then confirm rest pose (✗ re-homes, ✓ continues).
    Step 3: confirm the environment has been rearranged.
    """

    def __init__(
        self,
        *,
        session: Session,
        emit: Callable[[WSEventBase], Awaitable[None]],
        is_cancelled: Callable[[], bool],
        run_in_env_thread: Callable[..., Awaitable[Any]],
        low_level_env: Any,
        reset_and_render: Callable[[], Any],
    ) -> None:
        self._session = session
        self._emit = emit
        self._is_cancelled = is_cancelled
        self._run_in_env_thread = run_in_env_thread
        self._low_level_env = low_level_env
        self._reset_and_render = reset_and_render

    async def _wait(self) -> str:
        """Pause for a wizard button press; return 'ready'/'confirm'/'reject'."""
        session = self._session
        # Drop any presses queued before this prompt (e.g. an impatient
        # double-click on the previous step) so we never auto-advance.
        while not session.wizard_response_queue.empty():
            try:
                session.wizard_response_queue.get_nowait()
            except asyncio.QueueEmpty:
                break
        session.state = SessionState.AWAITING_USER_INPUT
        await self._emit(StateUpdateEvent(
            session_id=session.session_id,
            state=SessionState.AWAITING_USER_INPUT,
        ))
        payload = await session.wizard_response_queue.get()
        session.state = SessionState.RUNNING
        await self._emit(StateUpdateEvent(
            session_id=session.session_id, state=SessionState.RUNNING,
        ))
        if payload == WIZARD_READY:
            return "ready"
        if payload == WIZARD_REJECT:
            return "reject"
        return "confirm"

    async def _show(
        self,
        step: str,
        message: str,
        actions: list[str] | None = None,
        busy: bool = False,
    ) -> None:
        await self._emit(ResetWizardEvent(
            session_id=self._session.session_id,
            step=step,
            message=message,
            actions=actions or [],
            busy=busy,
        ))

    async def run(self, label: str):
        """Run the 3-step guided reset and return ``(obs, frame)``."""
        logger.info(f"Starting guided real-robot reset ({label})")
        # Step 1 — connection check loop
        while True:
            if self._is_cancelled():
                raise asyncio.CancelledError("Cancelled during reset wizard")
            if await self._run_in_env_thread(self._low_level_env.is_connected):
                break
            await self._show(
                "connection",
                "机械臂未连接。请重启机械臂服务/中间件，完成后点击 Ready。",
                actions=["ready"],
            )
            await self._wait()
            await self._show("connection", "正在检测机械臂连接…", busy=True)

        # Step 2a — automatic return to rest pose
        await self._show("rest_pose", "正在自动回到 rest pose…", busy=True)
        await self._run_in_env_thread(self._low_level_env.return_to_rest_pose)

        # Step 2b — confirm rest pose (✗ re-homes)
        while True:
            if self._is_cancelled():
                raise asyncio.CancelledError("Cancelled during reset wizard")
            await self._show(
                "rest_pose",
                "机械臂是否已回到 rest pose？是 → ✓；否 → ✗（将再次自动回归）。",
                actions=["confirm", "reject"],
            )
            if await self._wait() == "confirm":
                break
            await self._show("rest_pose", "正在回到 rest pose…", busy=True)
            await self._run_in_env_thread(self._low_level_env.return_to_rest_pose)

        # Step 3 — confirm environment rearranged
        await self._show(
            "rearrange",
            "请重新摆放环境，完成后点击 ✓。",
            actions=["confirm"],
        )
        await self._wait()

        await self._show("complete", "真机 reset 完成。")
        logger.info(f"Guided real-robot reset complete ({label})")

        obs_out, _info, frame_out = await self._run_in_env_thread(self._reset_and_render)
        return obs_out, frame_out


__all__ = ["GuidedResetWizard"]
