"""Client-side env that talks to ``launch_piper_state_service`` over websockets.

Drop-in replacement for :class:`PiperRealLowLevel` from the agent process's
point of view: the same ``obs`` shape, the same control methods, the same
overlay attributes — but the heavy lifting (CAN, cameras, viser, point cloud)
runs in a separate service process.

The client:

  * opens a websocket to the service URL given via ``service_url``,
  * receives a ``hello`` message describing per-camera SHM regions and attaches
    to them,
  * spawns a daemon thread that consumes ``state`` pushes and updates
    ``self.obs`` in place (joints, intrinsics, pose matrices, plus RGB / depth
    refreshed from SHM whenever the seq advances),
  * sends control RPCs synchronously (``goto_pose`` etc. block until the reply
    arrives),
  * forwards overlay assignments (``cube_center``, ``grasp_sample`` ...) to the
    service via ``OP_SET`` so the service-side viser scene reflects them.

``viser_server`` is always ``None`` in the agent process; the service hosts
viser. UI-side methods on the env (``_update_viser_server`` ,
``_refresh_viser_point_cloud``) become RPCs.
"""

from __future__ import annotations

import asyncio
import threading
import time
from concurrent.futures import Future
from typing import Any

import numpy as np
from gymnasium import spaces

from capx.envs.base import BaseEnv
from capx.utils.video_utils import resize_with_pad
from capx.envs.simulators.piper.service_protocol import (
    OP_HELLO,
    OP_RPC,
    OP_RPC_REPLY,
    OP_SET,
    OP_STATE,
    PROTOCOL_VERSION,
    RPC_EXECUTE_JOINT_TRAJECTORY,
    RPC_GOTO_HOME_BLOCKING,
    RPC_MOVE_TO_JOINTS_BLOCKING,
    RPC_REFRESH_POINT_CLOUD,
    RPC_RESET,
    RPC_SET_GRIPPER,
    RPC_STEP_ONCE,
    RPC_UPDATE_VISER_SERVER,
    SETTABLE_OVERLAY_KEYS,
    ReqIdAllocator,
    decode,
    encode,
)
from capx.envs.simulators.piper.service_shm import (
    CameraShmDescriptor,
    CameraShmReader,
)


_RPC_TIMEOUT_SEC = 120.0


class _ServiceSideViserMarker:
    """Sentinel placed on PiperRealServiceClient.viser_server.

    Tells callers that *some* viser exists (so the control API's
    ``if env.viser_server is not None`` guards stay live), without exposing a
    real viser handle from the agent process. The actual viser server runs
    inside the service process; updates land there via RPC.
    """

    def __repr__(self) -> str:  # pragma: no cover -- diagnostic only
        return "<service-side viser>"


class PiperRealServiceClient(BaseEnv):
    """Thin client env. See module docstring."""

    HOME_JOINTS_RAD = np.zeros(6, dtype=np.float64)

    def __init__(
        self,
        service_url: str,
        *,
        seed: int | None = None,
        viser_debug: bool = False,
        privileged: bool = False,
        enable_render: bool = False,
        connect_timeout_sec: float = 30.0,
        # Tolerated for parity with PiperRealLowLevel; cameras live on the service.
        wrist_camera_enabled: bool | None = None,
        wrist_camera_serial: str | None = None,
        wrist_camera_fps: int | None = None,
        wrist_camera_width: int | None = None,
        wrist_camera_height: int | None = None,
        wrist_camera_use_depth: bool | None = None,
        wrist_camera_link_name: str | None = None,
        wrist_camera_position: list[float] | None = None,
        wrist_camera_rpy_radians: list[float] | None = None,
        **_unused: Any,
    ) -> None:
        super().__init__()
        # Plain attributes go through object.__setattr__ to bypass the overlay
        # forwarding hook.
        object.__setattr__(self, "_service_url", str(service_url))
        object.__setattr__(self, "obs", {})
        # Truthy sentinel: the service hosts viser, but PiperControlApi guards
        # every viser update with ``if env.viser_server is not None``. Setting
        # this to a non-None marker (rather than the real viser handle) keeps
        # the control API code path active so update_viser_server RPCs fire.
        object.__setattr__(self, "viser_server", _ServiceSideViserMarker())
        object.__setattr__(self, "preview_waypoints", None)
        object.__setattr__(self, "last_ik_target", None)
        object.__setattr__(self, "_local_overlay_cache", {})

        # RPC plumbing.
        object.__setattr__(self, "_loop", asyncio.new_event_loop())
        object.__setattr__(self, "_loop_thread", None)
        object.__setattr__(self, "_ws", None)
        object.__setattr__(self, "_pending", {})  # req_id -> Future
        object.__setattr__(self, "_pending_lock", threading.Lock())
        object.__setattr__(self, "_req_alloc", ReqIdAllocator())
        object.__setattr__(self, "_shm_readers", {})
        object.__setattr__(self, "_camera_seq_seen", {})
        object.__setattr__(self, "_obs_lock", threading.Lock())
        object.__setattr__(self, "_hello_received", threading.Event())
        object.__setattr__(self, "_first_state_received", threading.Event())
        object.__setattr__(self, "_closed", False)

        # Video recording (agent-side; RGB already arrives via SHM/state push,
        # so the client just buffers resized copies whenever enabled).
        object.__setattr__(self, "_record_frames", False)
        object.__setattr__(self, "_record_wrist_camera", False)
        object.__setattr__(self, "_frame_buffer", [])
        object.__setattr__(self, "_wrist_frame_buffer", [])
        object.__setattr__(self, "_recording_lock", threading.Lock())

        self.action_space = spaces.Box(low=-np.inf, high=np.inf, shape=(7,))
        self.observation_space = spaces.Dict({})

        self._start_loop_thread()
        self._connect(timeout_sec=connect_timeout_sec)

    # =================================================================== #
    # Forwarding overlay attributes to the service.
    # =================================================================== #
    def __setattr__(self, name: str, value: Any) -> None:
        if name in SETTABLE_OVERLAY_KEYS:
            self._local_overlay_cache[name] = value
            self._send_set(name, value)
        # Always store locally so the control API can read back what it wrote.
        object.__setattr__(self, name, value)

    # =================================================================== #
    # asyncio loop bringup.
    # =================================================================== #
    def _start_loop_thread(self) -> None:
        loop = self._loop

        def _run() -> None:
            asyncio.set_event_loop(loop)
            loop.run_forever()

        t = threading.Thread(target=_run, name="piper-client-asyncio", daemon=True)
        t.start()
        object.__setattr__(self, "_loop_thread", t)

    def _run_coro(self, coro: Any) -> Any:
        return asyncio.run_coroutine_threadsafe(coro, self._loop).result()

    # =================================================================== #
    # Connection bringup.
    # =================================================================== #
    def _connect(self, *, timeout_sec: float) -> None:
        async def _do_connect() -> None:
            from websockets.asyncio.client import connect

            ws = await connect(self._service_url, max_size=64 * 1024 * 1024)
            object.__setattr__(self, "_ws", ws)
            asyncio.create_task(self._reader_loop(ws))

        self._run_coro(_do_connect())
        if not self._hello_received.wait(timeout=timeout_sec):
            raise TimeoutError(
                f"Did not receive hello from {self._service_url} within {timeout_sec}s"
            )
        # Wait for the first state push so the agent sees a populated obs.
        self._first_state_received.wait(timeout=timeout_sec)

    async def _reader_loop(self, ws) -> None:
        try:
            async for frame in ws:
                try:
                    msg = decode(frame)
                except Exception as e:
                    print(f"[piper_client] bad frame: {e}")
                    continue
                op = msg.get("op")
                if op == OP_HELLO:
                    self._handle_hello(msg)
                elif op == OP_STATE:
                    self._handle_state(msg)
                elif op == OP_RPC_REPLY:
                    self._handle_rpc_reply(msg)
                else:
                    print(f"[piper_client] unknown op: {op!r}")
        except Exception as e:
            print(f"[piper_client] reader loop ended: {e}")

    def _handle_hello(self, msg: dict[str, Any]) -> None:
        version = int(msg.get("version") or 0)
        if version != PROTOCOL_VERSION:
            print(
                f"[piper_client] WARNING: protocol version mismatch "
                f"(client={PROTOCOL_VERSION}, server={version})"
            )
        cameras = msg.get("cameras") or []
        readers: dict[str, CameraShmReader] = {}
        for cam in cameras:
            try:
                desc = CameraShmDescriptor.from_dict(cam)
                readers[desc.camera_key] = CameraShmReader(desc)
            except Exception as e:
                print(f"[piper_client] failed to attach SHM for {cam}: {e}")
        object.__setattr__(self, "_shm_readers", readers)
        self._hello_received.set()
        print(
            f"[piper_client] connected; cameras: {list(readers.keys())}"
        )

    def _handle_state(self, msg: dict[str, Any]) -> None:
        joints = msg.get("joints")
        cameras = msg.get("cameras") or {}
        with self._obs_lock:
            obs = self.obs
            if joints is not None:
                obs["robot_joint_pos"] = np.asarray(joints, dtype=np.float32)
            for cam_key, cam_state in cameras.items():
                view = obs.setdefault(cam_key, {})
                images = view.setdefault("images", {})
                if cam_state.get("intrinsics") is not None:
                    view["intrinsics"] = np.asarray(cam_state["intrinsics"])
                if cam_state.get("pose_mat") is not None:
                    view["pose_mat"] = np.asarray(cam_state["pose_mat"])
                if cam_state.get("pose") is not None:
                    view["pose"] = np.asarray(cam_state["pose"])
                shm_seq = int(cam_state.get("shm_seq") or 0)
                last_seq = int(self._camera_seq_seen.get(cam_key, 0))
                if shm_seq > last_seq:
                    reader = self._shm_readers.get(cam_key)
                    if reader is not None:
                        seq, _ts, rgb, depth = reader.read_latest()
                        if rgb is not None:
                            images["rgb"] = rgb
                            self._maybe_record_frame(cam_key, rgb)
                        if depth is not None:
                            images["depth"] = depth
                        self._camera_seq_seen[cam_key] = seq
        self._first_state_received.set()

    def _maybe_record_frame(self, cam_key: str, rgb: np.ndarray) -> None:
        if not self._record_frames:
            return
        if cam_key == "robot0_robotview":
            try:
                frame = resize_with_pad(rgb, 480, 640)
            except Exception:
                return
            with self._recording_lock:
                self._frame_buffer.append(frame)
        elif cam_key == "robot0_eye_in_hand" and self._record_wrist_camera:
            try:
                frame = resize_with_pad(rgb, 480, 640)
            except Exception:
                return
            with self._recording_lock:
                self._wrist_frame_buffer.append(frame)

    def _handle_rpc_reply(self, msg: dict[str, Any]) -> None:
        req_id = msg.get("req_id")
        with self._pending_lock:
            fut: Future | None = self._pending.pop(req_id, None)
        if fut is None:
            return
        if msg.get("ok"):
            fut.set_result(msg.get("result"))
        else:
            fut.set_exception(RuntimeError(str(msg.get("error"))))

    # =================================================================== #
    # Outgoing messages.
    # =================================================================== #
    def _send_frame(self, payload: dict[str, Any]) -> None:
        ws = self._ws
        if ws is None:
            raise RuntimeError("client not connected")
        frame = encode(payload)
        self._run_coro(ws.send(frame))

    def _send_set(self, key: str, value: Any) -> None:
        if self._ws is None or self._closed:
            return
        try:
            self._send_frame({"op": OP_SET, "key": key, "value": value})
        except Exception as e:
            print(f"[piper_client] set {key} failed: {e}")

    def _rpc(self, method: str, **kwargs: Any) -> Any:
        if method not in {
            RPC_EXECUTE_JOINT_TRAJECTORY,
            RPC_GOTO_HOME_BLOCKING,
            RPC_MOVE_TO_JOINTS_BLOCKING,
            RPC_REFRESH_POINT_CLOUD,
            RPC_RESET,
            RPC_SET_GRIPPER,
            RPC_STEP_ONCE,
            RPC_UPDATE_VISER_SERVER,
        }:
            raise ValueError(f"client tried to call unsupported method: {method}")
        req_id = self._req_alloc.next_id()
        fut: Future = Future()
        with self._pending_lock:
            self._pending[req_id] = fut
        self._send_frame(
            {"op": OP_RPC, "req_id": req_id, "method": method, "args": kwargs}
        )
        return fut.result(timeout=_RPC_TIMEOUT_SEC)

    # =================================================================== #
    # BaseEnv interface.
    # =================================================================== #
    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        self._rpc(RPC_RESET)
        # Wait for at least one fresh state push after reset.
        self._first_state_received.clear()
        self._first_state_received.wait(timeout=30.0)
        return self.obs, {}

    def step(self, action: Any):
        raise NotImplementedError(
            "PiperRealServiceClient is driven through the control API, not step()"
        )

    def get_observation(self) -> dict[str, Any]:
        return self.obs

    def compute_reward(self) -> float:
        return 0.0

    def task_completed(self) -> bool:
        return False

    def render(self, mode: str = "rgb_array") -> np.ndarray:
        rgb = (
            self.obs.get("robot0_robotview", {})
            .get("images", {})
            .get("rgb")
        )
        if rgb is not None:
            return rgb
        return np.zeros((480, 640, 3), dtype=np.uint8)

    def render_wrist(self) -> np.ndarray | None:
        return (
            self.obs.get("robot0_eye_in_hand", {})
            .get("images", {})
            .get("rgb")
        )

    # =================================================================== #
    # Video recording (agent-side; frames are sourced from the SHM-backed
    # RGB the state-push handler refreshes).
    # =================================================================== #
    def enable_video_capture(
        self,
        enabled: bool = True,
        *,
        clear: bool = True,
        wrist_camera: bool = False,
    ) -> None:
        with self._recording_lock:
            object.__setattr__(self, "_record_frames", bool(enabled))
            object.__setattr__(
                self, "_record_wrist_camera", bool(enabled and wrist_camera)
            )
            if clear:
                self._frame_buffer.clear()
                self._wrist_frame_buffer.clear()
        if enabled:
            # Capture one frame immediately so callers that record then
            # immediately fetch always get at least one image.
            rgb = (
                self.obs.get("robot0_robotview", {})
                .get("images", {})
                .get("rgb")
            )
            if rgb is not None:
                self._maybe_record_frame("robot0_robotview", rgb)
            if self._record_wrist_camera:
                wrist = (
                    self.obs.get("robot0_eye_in_hand", {})
                    .get("images", {})
                    .get("rgb")
                )
                if wrist is not None:
                    self._maybe_record_frame("robot0_eye_in_hand", wrist)

    def get_video_frames(self, *, clear: bool = False) -> list[np.ndarray]:
        with self._recording_lock:
            frames = [f.copy() for f in self._frame_buffer]
            if clear:
                self._frame_buffer.clear()
        return frames

    def get_video_frame_count(self) -> int:
        with self._recording_lock:
            return len(self._frame_buffer)

    def get_video_frames_range(self, start: int, end: int) -> list[np.ndarray]:
        with self._recording_lock:
            return [f.copy() for f in self._frame_buffer[start:end]]

    def get_wrist_video_frames(self, *, clear: bool = False) -> list[np.ndarray]:
        with self._recording_lock:
            frames = [f.copy() for f in self._wrist_frame_buffer]
            if clear:
                self._wrist_frame_buffer.clear()
        return frames

    def get_wrist_video_frames_range(self, start: int, end: int) -> list[np.ndarray]:
        with self._recording_lock:
            return [f.copy() for f in self._wrist_frame_buffer[start:end]]

    # =================================================================== #
    # Control surface mirrored from PiperRealLowLevel + PiperMotionMixin.
    # =================================================================== #
    def execute_joint_trajectory(
        self,
        trajectory: np.ndarray,
        *,
        command_dt: float = 0.02,
        final_tolerance: float = 0.02,
        final_max_steps: int = 260,
    ) -> None:
        self._rpc(
            RPC_EXECUTE_JOINT_TRAJECTORY,
            trajectory=np.asarray(trajectory, dtype=np.float64),
            command_dt=float(command_dt),
            final_tolerance=float(final_tolerance),
            final_max_steps=int(final_max_steps),
        )

    def move_to_joints_blocking(
        self,
        joints: np.ndarray,
        *,
        tolerance: float = 0.02,
        max_steps: int = 500,
    ) -> None:
        self._rpc(
            RPC_MOVE_TO_JOINTS_BLOCKING,
            joints=np.asarray(joints, dtype=np.float64),
            tolerance=float(tolerance),
            max_steps=int(max_steps),
        )

    def goto_home_blocking(
        self, *, tolerance: float = 0.02, max_steps: int = 500
    ) -> None:
        self._rpc(
            RPC_GOTO_HOME_BLOCKING,
            tolerance=float(tolerance),
            max_steps=int(max_steps),
        )

    def _set_gripper(self, fraction: float) -> None:
        self._rpc(RPC_SET_GRIPPER, fraction=float(fraction))

    def _step_once(self) -> None:
        self._rpc(RPC_STEP_ONCE)

    # ------------------------------------------------------------ viser --
    def _sync_overlays(self) -> None:
        """Re-push every overlay attribute to the service.

        Catches in-place mutations like ``preview_waypoints.extend(...)`` that
        bypass ``__setattr__`` forwarding. Cheap: a handful of small dicts /
        ndarrays. Skips attrs that were never assigned.
        """
        for key in SETTABLE_OVERLAY_KEYS:
            value = getattr(self, key, None)
            if value is None and key not in self._local_overlay_cache:
                continue
            self._send_set(key, value)

    def _refresh_viser_point_cloud(self) -> None:
        try:
            self._rpc(RPC_REFRESH_POINT_CLOUD)
        except Exception as e:
            print(f"[piper_client] refresh_point_cloud failed: {e}")

    def _update_viser_server(self, *, force_scene: bool = False) -> None:
        try:
            self._sync_overlays()
            self._rpc(RPC_UPDATE_VISER_SERVER, force_scene=bool(force_scene))
        except Exception as e:
            print(f"[piper_client] update_viser_server failed: {e}")

    # =================================================================== #
    # Cleanup.
    # =================================================================== #
    def close(self) -> None:
        if self._closed:
            return
        object.__setattr__(self, "_closed", True)
        try:
            if self._ws is not None:
                self._run_coro(self._ws.close())
        except Exception:
            pass
        for reader in self._shm_readers.values():
            reader.close()
        try:
            self._loop.call_soon_threadsafe(self._loop.stop)
        except Exception:
            pass

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass


__all__ = ["PiperRealServiceClient"]
