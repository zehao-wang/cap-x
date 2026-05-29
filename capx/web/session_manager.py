"""Session manager for tracking active trial sessions."""

from __future__ import annotations

import asyncio
import ctypes
import logging
import threading
import uuid
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Any, Callable, Awaitable

from fastapi import WebSocket

from capx.web.models import SessionState, StateUpdateEvent, WSEventBase

if TYPE_CHECKING:
    from capx.web.async_trial_runner import TrialContext

logger = logging.getLogger(__name__)

# Sentinel payloads put on ``user_injection_queue`` to signal human control
# actions that are distinct from plain feedback text. The runner's human-pause
# wait point recognises these and acts accordingly (see async_trial_runner).
# Wrapped in NUL bytes so they can never collide with real typed feedback.
RESET_COMMAND = "\x00__CAPX_RESET__\x00"
FINISH_COMMAND = "\x00__CAPX_FINISH__\x00"

# Sentinels for the guided real-robot reset wizard, delivered on a dedicated
# wizard_response_queue (separate from feedback). Mapped from the UI button ids
# "ready" / "confirm" / "reject" by SessionManager.request_wizard_action.
WIZARD_READY = "\x00__CAPX_WIZARD_READY__\x00"
WIZARD_CONFIRM = "\x00__CAPX_WIZARD_CONFIRM__\x00"
WIZARD_REJECT = "\x00__CAPX_WIZARD_REJECT__\x00"

_WIZARD_ACTION_SENTINELS = {
    "ready": WIZARD_READY,
    "confirm": WIZARD_CONFIRM,
    "reject": WIZARD_REJECT,
}

# Cap the replay buffer so an unbounded interactive session can't grow it
# without limit. Heavy base64 image payloads are also stripped from the stored
# copy (see Session._record_for_replay), so the bound is on event count, not bytes.
EVENT_HISTORY_MAXLEN = 5000


async def run_blocking_with_interrupt(
    session: "Session",
    func: Callable[..., Any],
    *args: Any,
    **kwargs: Any,
) -> Any:
    """Run a blocking function in a thread while tracking the thread ID for interruption.

    This allows the stop_session method to interrupt long-running code execution.
    """
    def wrapper():
        # Store the current thread ID so it can be interrupted
        session.execution_thread_id = threading.get_ident()
        try:
            return func(*args, **kwargs)
        finally:
            session.execution_thread_id = None

    return await asyncio.to_thread(wrapper)


def _raise_exception_in_thread(thread_id: int, exception_type: type) -> bool:
    """Raise an exception in another thread.

    This is a safety mechanism to interrupt code execution.
    Returns True if successful, False otherwise.
    """
    try:
        res = ctypes.pythonapi.PyThreadState_SetAsyncExc(
            ctypes.c_ulong(thread_id),
            ctypes.py_object(exception_type)
        )
        if res == 0:
            logger.warning(f"Thread {thread_id} not found")
            return False
        elif res > 1:
            # If more than one thread was affected, reset
            ctypes.pythonapi.PyThreadState_SetAsyncExc(ctypes.c_ulong(thread_id), None)
            logger.error(f"Multiple threads affected when interrupting {thread_id}")
            return False
        return True
    except Exception as e:
        logger.error(f"Failed to raise exception in thread: {e}")
        return False


@dataclass
class Session:
    """Represents an active trial session."""

    session_id: str
    state: SessionState = SessionState.IDLE
    config_path: str | None = None
    config: dict[str, Any] = field(default_factory=dict)
    env_factory: dict[str, Any] | None = None

    # Settings that can be changed during a trial
    await_user_input_each_turn: bool = False
    execution_timeout: int = 180  # seconds per code block

    # Event history for replay on reconnect (bounded; see EVENT_HISTORY_MAXLEN)
    event_history: deque[str] = field(
        default_factory=lambda: deque(maxlen=EVENT_HISTORY_MAXLEN)
    )

    # Async coordination
    cancel_event: asyncio.Event = field(default_factory=asyncio.Event)
    user_injection_queue: asyncio.Queue[str] = field(default_factory=asyncio.Queue)
    # Responses to the guided real-robot reset wizard (Ready / ✓ / ✗), kept
    # separate from feedback so they never collide with typed user input.
    wizard_response_queue: asyncio.Queue[str] = field(default_factory=asyncio.Queue)
    # Serializes all WebSocket sends for this session. Emits can be scheduled
    # concurrently onto the loop (e.g. execution-step callbacks fired from the
    # env worker thread via run_coroutine_threadsafe), and Starlette/uvicorn
    # forbids overlapping sends on a single socket — this lock prevents that.
    _send_lock: asyncio.Lock = field(default_factory=asyncio.Lock)

    # Running task reference
    task: asyncio.Task | None = None

    # Environment reference for forced shutdown
    env: Any = None

    # Thread tracking for interruption
    execution_thread_id: int | None = None

    # Connected WebSocket clients
    websockets: list[WebSocket] = field(default_factory=list)

    # Execution state
    current_block_index: int = 0
    total_code_blocks: int = 0
    num_regenerations: int = 0

    # Timestamps
    created_at: datetime = field(default_factory=datetime.utcnow)
    started_at: datetime | None = None
    completed_at: datetime | None = None

    async def emit(self, event: WSEventBase) -> None:
        """Broadcast event to all connected WebSocket clients and store for replay.

        Sends are serialized under ``_send_lock`` and iterate over a snapshot of
        the connection list, so concurrently-scheduled emits never overlap sends
        on one socket nor mutate the list mid-iteration.
        """
        message = event.model_dump_json()
        self._record_for_replay(event, message)

        async with self._send_lock:
            disconnected = []
            for ws in list(self.websockets):
                try:
                    await ws.send_text(message)
                except Exception as e:
                    logger.warning(f"Failed to send to WebSocket: {e}")
                    disconnected.append(ws)
            # Clean up disconnected clients
            for ws in disconnected:
                if ws in self.websockets:
                    self.websockets.remove(ws)

    def _record_for_replay(self, event: WSEventBase, message: str) -> None:
        """Append an event to the bounded replay history.

        High-frequency streaming deltas are skipped entirely; heavy base64 image
        payloads are stripped from the *stored* copy (the live broadcast above
        still carries them) so the buffer stays small over a long session.
        """
        if event.type == "model_streaming_delta":
            return
        if event.type == "execution_step" and getattr(event, "images", None):
            message = event.model_copy(update={"images": []}).model_dump_json()
        elif event.type == "visual_feedback" and getattr(event, "image_base64", None):
            message = event.model_copy(update={"image_base64": ""}).model_dump_json()
        self.event_history.append(message)

    async def attach_websocket(self, websocket: WebSocket) -> None:
        """Replay history + current state to a freshly-accepted ws, then register it.

        Runs under the same send lock as ``emit`` and registers the socket only
        after replay completes, so a running trial's emits can neither interleave
        sends on this socket nor duplicate/reorder the replayed history.
        """
        async with self._send_lock:
            if self.event_history:
                logger.info(
                    f"Replaying {len(self.event_history)} events for session {self.session_id}"
                )
                for event_json in list(self.event_history):
                    try:
                        await websocket.send_text(event_json)
                    except Exception:
                        return  # client vanished mid-replay; don't register it
            try:
                await websocket.send_text(
                    StateUpdateEvent(
                        session_id=self.session_id, state=self.state
                    ).model_dump_json()
                )
            except Exception:
                return
            self.websockets.append(websocket)

    def reset(self) -> None:
        """Reset session state for a new trial."""
        self.state = SessionState.IDLE
        self.cancel_event = asyncio.Event()
        self.user_injection_queue = asyncio.Queue()
        self.wizard_response_queue = asyncio.Queue()
        self.event_history = deque(maxlen=EVENT_HISTORY_MAXLEN)
        self.task = None
        self.env = None
        self.execution_thread_id = None
        self.current_block_index = 0
        self.total_code_blocks = 0
        self.num_regenerations = 0
        self.started_at = None
        self.completed_at = None


class SessionManager:
    """Manages all active trial sessions.

    Only one session can be active at a time. Creating a new session
    will automatically stop and clean up any existing sessions.
    """

    def __init__(self):
        self._sessions: dict[str, Session] = {}
        self._lock = asyncio.Lock()

    async def create_session(self) -> Session:
        """Create a new session, stopping any existing sessions first."""
        async with self._lock:
            # Stop and clean up ALL existing sessions first
            for session_id in list(self._sessions.keys()):
                await self._cleanup_session_unlocked(session_id)

            session_id = str(uuid.uuid4())
            session = Session(session_id=session_id)
            self._sessions[session_id] = session
            logger.info(f"Created session: {session_id}")
            return session

    async def _cleanup_session_unlocked(self, session_id: str) -> None:
        """Clean up a session (must be called with lock held)."""
        if session_id not in self._sessions:
            return

        session = self._sessions[session_id]
        logger.info(f"Cleaning up session: {session_id}")

        # Cancel any running task
        if session.task and not session.task.done():
            session.cancel_event.set()
            session.task.cancel()
            try:
                await asyncio.wait_for(session.task, timeout=2.0)
            except (asyncio.TimeoutError, asyncio.CancelledError):
                pass

        # Close all WebSocket connections
        for ws in session.websockets:
            try:
                await ws.close(code=4001, reason="Session replaced")
            except Exception:
                pass

        # Stop the env's viser server so its port frees up for the next task.
        # Without this the old server is orphaned: the next env's viser bumps
        # to a higher port while the proxy stays pinned to the stale one, so
        # the new task's scene never reaches the browser (blank viewer).
        if session.env is not None:
            try:
                await asyncio.to_thread(session.env.close)
            except Exception as exc:
                logger.warning(f"Error closing env during cleanup: {exc}")
            session.env = None

        del self._sessions[session_id]
        logger.info(f"Session cleaned up: {session_id}")

    async def get_session(self, session_id: str) -> Session | None:
        """Get a session by ID."""
        return self._sessions.get(session_id)

    async def remove_session(self, session_id: str) -> None:
        """Remove a session."""
        async with self._lock:
            await self._cleanup_session_unlocked(session_id)

    async def stop_session(self, session_id: str) -> bool:
        """Stop a running session immediately.

        This is a safety-critical operation that should interrupt code execution
        as quickly as possible.
        """
        session = await self.get_session(session_id)
        if not session:
            return False

        if session.task and not session.task.done():
            logger.info(f"STOPPING session (safety interrupt): {session_id}")
            session.cancel_event.set()

            # Interrupt the execution thread if code is running
            if session.execution_thread_id is not None:
                logger.info(f"Interrupting execution thread {session.execution_thread_id}")
                if _raise_exception_in_thread(session.execution_thread_id, KeyboardInterrupt):
                    logger.info("Thread interrupt sent successfully")
                else:
                    logger.warning("Thread interrupt failed")

            # Try to close the environment immediately to interrupt any running code
            if session.env is not None:
                try:
                    logger.info(f"Attempting to close environment for session {session_id}")
                    if hasattr(session.env, 'close'):
                        session.env.close()
                    elif hasattr(session.env, 'shutdown'):
                        session.env.shutdown()
                except Exception as e:
                    logger.warning(f"Error closing environment: {e}")

            # Cancel the task immediately (don't wait for graceful shutdown)
            session.task.cancel()
            try:
                await asyncio.wait_for(session.task, timeout=2.0)
            except (asyncio.TimeoutError, asyncio.CancelledError):
                pass

            session.state = SessionState.IDLE
            session.env = None  # Clear env reference
            session.execution_thread_id = None  # Clear thread reference
            logger.info(f"Session {session_id} stopped")
            return True

        return False

    async def inject_prompt(self, session_id: str, text: str) -> bool:
        """Inject user prompt text into a session."""
        session = await self.get_session(session_id)
        if not session:
            return False

        if session.state == SessionState.AWAITING_USER_INPUT:
            await session.user_injection_queue.put(text)
            logger.info(f"Injected prompt into session {session_id}: {text[:50]}...")
            return True

        return False

    async def request_reset(self, session_id: str) -> bool:
        """Ask a paused session to reset its environment.

        Only honoured while the session is awaiting user input (the natural
        decision point between turns). The runner keeps only this attempt's
        most recent code as context and replans from the freshly reset state.
        """
        session = await self.get_session(session_id)
        if not session:
            return False
        if session.state == SessionState.AWAITING_USER_INPUT:
            await session.user_injection_queue.put(RESET_COMMAND)
            logger.info(f"Reset requested for session {session_id}")
            return True
        return False

    async def request_finish(self, session_id: str) -> bool:
        """Mark a paused session as human-confirmed success and end it.

        In interactive mode the model's own FINISH never ends the trial; the
        success signal must come from the human via this call (§9 / §4.1).
        """
        session = await self.get_session(session_id)
        if not session:
            return False
        if session.state == SessionState.AWAITING_USER_INPUT:
            await session.user_injection_queue.put(FINISH_COMMAND)
            logger.info(f"Finish (human success) requested for session {session_id}")
            return True
        return False

    async def request_wizard_action(self, session_id: str, action: str) -> bool:
        """Deliver a guided-reset wizard button press (Ready / ✓ / ✗).

        Honoured only while the session is awaiting user input (the wizard sets
        that state between steps). Unknown actions are ignored.
        """
        sentinel = _WIZARD_ACTION_SENTINELS.get(action)
        if sentinel is None:
            logger.warning(f"Ignoring unknown wizard action {action!r}")
            return False
        session = await self.get_session(session_id)
        if not session:
            return False
        if session.state == SessionState.AWAITING_USER_INPUT:
            await session.wizard_response_queue.put(sentinel)
            logger.info(f"Wizard action {action!r} for session {session_id}")
            return True
        return False

    def list_sessions(self) -> list[dict[str, Any]]:
        """List all sessions with their status."""
        return [
            {
                "session_id": s.session_id,
                "state": s.state.value,
                "config_path": s.config_path,
                "created_at": s.created_at.isoformat(),
            }
            for s in self._sessions.values()
        ]

    def get_active_session(self) -> Session | None:
        """Get the currently active session (if any).

        Since only one session is allowed at a time, this returns
        the single session if it exists and is still running.
        """
        for session in self._sessions.values():
            if session.state in (SessionState.RUNNING, SessionState.AWAITING_USER_INPUT, SessionState.LOADING_CONFIG):
                return session
        return None

    async def on_websocket_disconnect(self, session_id: str) -> None:
        """Handle WebSocket disconnection.

        If the session has no more connected WebSockets and is still running,
        we'll keep it alive briefly in case of reconnection. If it's complete
        or errored, clean it up.
        """
        session = await self.get_session(session_id)
        if not session:
            return

        # If session is complete/error and no WebSockets, clean up
        if session.state in (SessionState.COMPLETE, SessionState.ERROR, SessionState.IDLE):
            if not session.websockets:
                logger.info(f"Session {session_id} has no connections and is {session.state}, cleaning up")
                await self.remove_session(session_id)


# Global singleton
_manager: SessionManager | None = None


def get_session_manager() -> SessionManager:
    """Get the global session manager instance."""
    global _manager
    if _manager is None:
        _manager = SessionManager()
    return _manager
