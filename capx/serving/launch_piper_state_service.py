"""Long-running service that owns the Piper arm + cameras + viser.

Run as::

    uv run --no-sync --active python -m capx.serving.launch_piper_state_service \
        --config env_configs/real/piper_real.yaml \
        --host 127.0.0.1 --port 8210 --shm-prefix piper_svc

Add ``--no-wrist-camera`` when the wrist RealSense is not attached.

The service:

  1. Instantiates a normal :class:`PiperRealLowLevel` (which connects CAN,
     starts both cameras, opens viser, and homes the arm).
  2. Allocates POSIX SHM regions for each camera's RGB/depth buffers.
  3. Runs a state-poll thread at ``--poll-hz`` that calls
     ``_update_from_hardware``, copies images into SHM, and broadcasts a
     state push to all connected websocket clients.
  4. Hosts a websocket server that dispatches RPC requests
     (``execute_joint_trajectory``, ``goto_home_blocking``, ``set_gripper``,
     ...) onto the wrapped env. Motion RPCs pause the poll thread so the
     CAN handle is touched by exactly one thread at a time.

The agent process talks to this service via
``capx.envs.simulators.piper_real_service_client.PiperRealServiceClient``.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import threading
import time
import traceback
from typing import Any

import numpy as np
import yaml

from capx.envs.simulators.piper.service_protocol import (
    OP_HELLO,
    OP_RPC,
    OP_RPC_REPLY,
    OP_SET,
    OP_STATE,
    PROTOCOL_VERSION,
    RPC_EXECUTE_JOINT_TRAJECTORY,
    RPC_GET_OBSERVATION,
    RPC_GOTO_HOME_BLOCKING,
    RPC_MOVE_TO_JOINTS_BLOCKING,
    RPC_REFRESH_POINT_CLOUD,
    RPC_RESET,
    RPC_SET_GRIPPER,
    RPC_STEP_ONCE,
    RPC_UPDATE_VISER_SERVER,
    SETTABLE_OVERLAY_KEYS,
    decode,
    encode,
)
from capx.envs.simulators.piper.service_shm import CameraShmWriter

# Import lazily so missing piper-sdk doesn't crash --help.
PiperRealLowLevel = None  # populated in main()


class _DropHandshakeNoiseFilter(logging.Filter):
    """Drop tracebacks from non-WS HTTP probes hitting the service port.

    Browsers, healthchecks, and curl regularly hit ws://host:port over plain
    HTTP, which the websockets library logs as ``InvalidUpgrade`` /
    ``InvalidHandshake`` exceptions. Those are expected and not actionable, so
    silence them. Real handshake failures from would-be WS clients are still
    rare and surface via other paths.
    """

    _NEEDLES = ("opening handshake failed", "InvalidUpgrade", "InvalidHandshake")

    def filter(self, record: logging.LogRecord) -> bool:
        msg = record.getMessage()
        if any(needle in msg for needle in self._NEEDLES):
            return False
        if record.exc_info is not None:
            exc_type = record.exc_info[0]
            name = getattr(exc_type, "__name__", "")
            if name in {"InvalidUpgrade", "InvalidHandshake"}:
                return False
        return True


def _quiet_handshake_noise() -> None:
    for name in ("websockets.server", "websockets.asyncio.server"):
        logging.getLogger(name).addFilter(_DropHandshakeNoiseFilter())


_MOTION_RPC_METHODS = {
    RPC_EXECUTE_JOINT_TRAJECTORY,
    RPC_MOVE_TO_JOINTS_BLOCKING,
    RPC_GOTO_HOME_BLOCKING,
    RPC_SET_GRIPPER,
    RPC_STEP_ONCE,
    RPC_RESET,
}


class PiperStateService:
    def __init__(
        self,
        env: Any,  # PiperRealLowLevel; typed Any to keep import lazy
        shm_prefix: str,
        poll_hz: float = 15.0,
        pointcloud_refresh_sec: float = 1.0,
    ) -> None:
        self.env = env
        self.shm_prefix = shm_prefix
        self.poll_dt = 1.0 / max(float(poll_hz), 1.0)
        # 0 disables auto-refresh; any positive value is the minimum interval
        # between merged-pointcloud refreshes triggered by the poll thread.
        self.pointcloud_refresh_sec = float(pointcloud_refresh_sec)
        self._last_pointcloud_refresh = 0.0
        self._writers: dict[str, CameraShmWriter] = {}
        self._stop = threading.Event()
        self._motion_active = threading.Event()
        self._loop: asyncio.AbstractEventLoop | None = None
        self._clients: set[Any] = set()
        self._latest_state: dict[str, Any] | None = None
        self._latest_state_lock = threading.Lock()

    # ---------------------------------------------------------------- shm --
    def _ensure_writers(self) -> None:
        for cam_key, view in list(self.env.obs.items()):
            if cam_key in self._writers:
                continue
            if not isinstance(view, dict):
                continue
            images = view.get("images", {})
            rgb = images.get("rgb")
            if rgb is None or rgb.ndim < 2:
                continue
            depth = images.get("depth")
            h, w = rgb.shape[:2]
            self._writers[cam_key] = CameraShmWriter(
                camera_key=cam_key,
                prefix=self.shm_prefix,
                height=h,
                width=w,
                has_depth=depth is not None,
            )
            print(
                f"[piper_svc] allocated SHM for {cam_key}: {h}x{w} "
                f"(depth={'yes' if depth is not None else 'no'})"
            )

    def _close_writers(self) -> None:
        for w in self._writers.values():
            w.close()
        self._writers.clear()

    # ------------------------------------------------------------- state --
    def _build_state_msg(self) -> dict[str, Any]:
        cameras: dict[str, Any] = {}
        for cam_key, writer in self._writers.items():
            view = self.env.obs.get(cam_key, {}) or {}
            cameras[cam_key] = {
                "intrinsics": view.get("intrinsics"),
                "pose_mat": view.get("pose_mat"),
                "pose": view.get("pose"),
                "shm_seq": int(writer._seq),
            }
        joints = self.env.obs.get("robot_joint_pos")
        return {
            "op": OP_STATE,
            "ts": time.time(),
            "joints": np.asarray(joints) if joints is not None else None,
            "cameras": cameras,
        }

    def _state_poll_thread(self) -> None:
        while not self._stop.is_set():
            if self._motion_active.is_set():
                # Motion RPC owns the hardware; wait it out without touching CAN.
                time.sleep(self.poll_dt)
                continue
            try:
                self.env._update_from_hardware()
            except Exception as e:
                print(f"[piper_svc] poll error: {e}")
                time.sleep(self.poll_dt)
                continue
            try:
                self._ensure_writers()
                for cam_key, writer in self._writers.items():
                    view = self.env.obs.get(cam_key, {}) or {}
                    images = view.get("images", {})
                    rgb = images.get("rgb")
                    if rgb is None:
                        continue
                    writer.write(rgb, images.get("depth"))
            except Exception as e:
                print(f"[piper_svc] shm write error: {e}")
            try:
                # Keep service-side viser panels live at idle — motion methods
                # already update viser on their own.
                if getattr(self.env, "viser_server", None) is not None:
                    self.env._update_viser_server()
            except Exception:
                pass

            # Auto-refresh the merged point cloud so viser shows it without
            # waiting for an agent to call refresh_point_clouds().
            if self.pointcloud_refresh_sec > 0.0:
                now = time.time()
                if now - self._last_pointcloud_refresh >= self.pointcloud_refresh_sec:
                    try:
                        if hasattr(self.env, "_refresh_viser_point_cloud"):
                            self.env._refresh_viser_point_cloud()
                    except Exception as e:
                        print(f"[piper_svc] pointcloud refresh error: {e}")
                    self._last_pointcloud_refresh = now

            msg = self._build_state_msg()
            with self._latest_state_lock:
                self._latest_state = msg
            self._broadcast(msg)
            time.sleep(self.poll_dt)

    # --------------------------------------------------------- broadcast --
    def _broadcast(self, msg: dict[str, Any]) -> None:
        if not self._clients or self._loop is None:
            return
        frame = encode(msg)
        loop = self._loop
        for ws in list(self._clients):
            try:
                asyncio.run_coroutine_threadsafe(ws.send(frame), loop)
            except Exception:
                pass

    # ----------------------------------------------------------- ws hub ---
    async def _serve_client(self, ws) -> None:
        peer = getattr(ws, "remote_address", None)
        print(f"[piper_svc] client connected: {peer}")
        hello = {
            "op": OP_HELLO,
            "version": PROTOCOL_VERSION,
            "shm_prefix": self.shm_prefix,
            "cameras": [w.descriptor.to_dict() for w in self._writers.values()],
        }
        await ws.send(encode(hello))
        with self._latest_state_lock:
            latest = self._latest_state
        if latest is not None:
            await ws.send(encode(latest))
        self._clients.add(ws)
        try:
            async for frame in ws:
                try:
                    msg = decode(frame)
                except Exception as e:
                    print(f"[piper_svc] bad frame from {peer}: {e}")
                    continue
                op = msg.get("op")
                if op == OP_RPC:
                    asyncio.create_task(self._handle_rpc(ws, msg))
                elif op == OP_SET:
                    self._handle_set(msg)
                else:
                    print(f"[piper_svc] unknown op from {peer}: {op!r}")
        except Exception as e:
            print(f"[piper_svc] client {peer} error: {e}")
        finally:
            self._clients.discard(ws)
            print(f"[piper_svc] client disconnected: {peer}")

    async def _handle_rpc(self, ws, msg: dict[str, Any]) -> None:
        req_id = msg.get("req_id")
        method = msg.get("method")
        args = msg.get("args") or {}
        loop = asyncio.get_running_loop()
        is_motion = method in _MOTION_RPC_METHODS

        def _run() -> Any:
            fn = self._method_table().get(method)
            if fn is None:
                raise ValueError(f"unknown method: {method}")
            if is_motion:
                self._motion_active.set()
            try:
                return fn(**args)
            finally:
                if is_motion:
                    self._motion_active.clear()

        try:
            result = await loop.run_in_executor(None, _run)
            reply = {
                "op": OP_RPC_REPLY,
                "req_id": req_id,
                "ok": True,
                "result": result,
            }
        except Exception as e:
            traceback.print_exc()
            reply = {
                "op": OP_RPC_REPLY,
                "req_id": req_id,
                "ok": False,
                "error": f"{type(e).__name__}: {e}",
            }
        try:
            await ws.send(encode(reply))
        except Exception:
            pass

    def _handle_set(self, msg: dict[str, Any]) -> None:
        key = msg.get("key")
        if key not in SETTABLE_OVERLAY_KEYS:
            print(f"[piper_svc] ignored set for non-allowlisted key: {key!r}")
            return
        try:
            setattr(self.env, key, msg.get("value"))
        except Exception as e:
            print(f"[piper_svc] set {key} failed: {e}")

    def _method_table(self) -> dict[str, Any]:
        env = self.env

        def _exec_traj(
            trajectory,
            command_dt: float = 0.02,
            final_tolerance: float = 0.02,
            final_max_steps: int = 260,
        ) -> None:
            traj = np.asarray(trajectory, dtype=np.float64)
            env.execute_joint_trajectory(
                traj,
                command_dt=float(command_dt),
                final_tolerance=float(final_tolerance),
                final_max_steps=int(final_max_steps),
            )

        def _move_to(joints, tolerance: float = 0.02, max_steps: int = 500) -> None:
            env.move_to_joints_blocking(
                np.asarray(joints, dtype=np.float64),
                tolerance=float(tolerance),
                max_steps=int(max_steps),
            )

        def _go_home(tolerance: float = 0.02, max_steps: int = 500) -> None:
            env.goto_home_blocking(tolerance=float(tolerance), max_steps=int(max_steps))

        def _set_gripper(fraction: float) -> None:
            env._set_gripper(float(fraction))

        def _step_once() -> None:
            env._step_once()

        def _refresh_pc() -> None:
            if hasattr(env, "_refresh_viser_point_cloud"):
                env._refresh_viser_point_cloud()

        def _update_viser(force_scene: bool = False) -> None:
            if hasattr(env, "_update_viser_server"):
                env._update_viser_server(force_scene=bool(force_scene))

        def _reset() -> None:
            if hasattr(env, "reset"):
                env.reset()

        def _get_obs() -> dict[str, Any]:
            # Return a lightweight snapshot. Image arrays go through SHM, so
            # we only echo non-image obs entries here. The state-push channel
            # is the primary path; this RPC is for "give me something now".
            with self._latest_state_lock:
                latest = self._latest_state
            return latest or {}

        return {
            RPC_EXECUTE_JOINT_TRAJECTORY: _exec_traj,
            RPC_MOVE_TO_JOINTS_BLOCKING: _move_to,
            RPC_GOTO_HOME_BLOCKING: _go_home,
            RPC_SET_GRIPPER: _set_gripper,
            RPC_STEP_ONCE: _step_once,
            RPC_REFRESH_POINT_CLOUD: _refresh_pc,
            RPC_UPDATE_VISER_SERVER: _update_viser,
            RPC_RESET: _reset,
            RPC_GET_OBSERVATION: _get_obs,
        }

    # --------------------------------------------------------------- run --
    def _prime_initial_frame(self) -> None:
        """Capture one hardware frame and allocate SHM writers synchronously.

        Without this, there's a race where a client can connect before the
        poll thread has run its first iteration: the ``hello`` message would
        carry an empty cameras list, the client would attach zero SHM readers,
        and subsequent state pushes would silently drop their RGB/depth.
        """
        try:
            self.env._update_from_hardware()
        except Exception as e:
            print(f"[piper_svc] prime _update_from_hardware error: {e}")
            return
        try:
            self._ensure_writers()
            for cam_key, writer in self._writers.items():
                view = self.env.obs.get(cam_key, {}) or {}
                images = view.get("images", {})
                rgb = images.get("rgb")
                if rgb is None:
                    continue
                writer.write(rgb, images.get("depth"))
        except Exception as e:
            print(f"[piper_svc] prime SHM write error: {e}")
            return
        self._latest_state = self._build_state_msg()

    async def run(self, host: str, port: int) -> None:
        from websockets.asyncio.server import serve

        self._loop = asyncio.get_running_loop()
        # Prime BEFORE starting the WS server so any connecting client sees a
        # fully-populated hello (camera SHM descriptors).
        self._prime_initial_frame()
        threading.Thread(
            target=self._state_poll_thread, name="piper-svc-poll", daemon=True
        ).start()
        try:
            async with serve(
                self._serve_client,
                host,
                port,
                max_size=64 * 1024 * 1024,
            ):
                print(f"[piper_svc] websocket server listening on ws://{host}:{port}")
                await asyncio.Future()
        finally:
            self._stop.set()
            self._close_writers()


# ----------------------------------------------------- yaml -> env kwargs --
def _build_env_from_yaml(
    yaml_path: str,
    *,
    wrist_camera_enabled_override: bool | None = None,
):
    global PiperRealLowLevel
    from capx.envs.simulators.piper_real import (
        PiperRealLowLevel as _PRL,  # noqa: F401  -- triggers piper_sdk import
    )

    PiperRealLowLevel = _PRL

    with open(yaml_path) as f:
        config = yaml.safe_load(f)
    cfg = ((config.get("env") or {}).get("cfg") or {})

    wrist_camera_enabled = cfg.get("piper_wrist_camera_enabled")
    if wrist_camera_enabled_override is not None:
        wrist_camera_enabled = wrist_camera_enabled_override

    kwargs: dict[str, Any] = {
        "viser_debug": True,
        "wrist_camera_enabled": wrist_camera_enabled,
        "wrist_camera_serial": cfg.get("piper_wrist_camera_serial") or None,
        "wrist_camera_fps": cfg.get("piper_wrist_camera_fps"),
        "wrist_camera_width": cfg.get("piper_wrist_camera_width"),
        "wrist_camera_height": cfg.get("piper_wrist_camera_height"),
        "wrist_camera_use_depth": cfg.get("piper_wrist_camera_use_depth"),
        "wrist_camera_link_name": cfg.get("piper_wrist_camera_link_name"),
        "wrist_camera_position": cfg.get("piper_wrist_camera_position"),
        "wrist_camera_rpy_radians": cfg.get("piper_wrist_camera_rpy_radians"),
        "zed_source": cfg.get("piper_zed_source"),
        "zed_service_socket": cfg.get("piper_zed_service_socket"),
    }
    kwargs = {k: v for k, v in kwargs.items() if v is not None}
    env = PiperRealLowLevel(**kwargs)
    if cfg.get("piper_planner_timesteps") is not None:
        env.piper_planner_timesteps = int(cfg["piper_planner_timesteps"])
    if cfg.get("piper_min_target_z") is not None:
        env.piper_min_target_z = float(cfg["piper_min_target_z"])
    return env


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument(
        "--config",
        required=True,
        help="Path to a piper_real yaml; only the env.cfg.piper_* keys are read.",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8210)
    parser.add_argument("--shm-prefix", default="piper_svc")
    parser.add_argument("--poll-hz", type=float, default=15.0)
    wrist_group = parser.add_mutually_exclusive_group()
    wrist_group.add_argument(
        "--wrist-camera",
        dest="wrist_camera_enabled",
        action="store_true",
        default=None,
        help="Force-enable the wrist RealSense camera, overriding the YAML.",
    )
    wrist_group.add_argument(
        "--no-wrist-camera",
        dest="wrist_camera_enabled",
        action="store_false",
        help="Disable the wrist RealSense camera, overriding the YAML.",
    )
    parser.add_argument(
        "--pointcloud-refresh-sec",
        type=float,
        default=1.0,
        help="Seconds between merged-pointcloud refreshes in viser. 0 disables.",
    )
    args = parser.parse_args()

    _quiet_handshake_noise()
    env = _build_env_from_yaml(
        args.config,
        wrist_camera_enabled_override=args.wrist_camera_enabled,
    )
    service = PiperStateService(
        env,
        args.shm_prefix,
        poll_hz=args.poll_hz,
        pointcloud_refresh_sec=args.pointcloud_refresh_sec,
    )
    try:
        asyncio.run(service.run(args.host, args.port))
    except KeyboardInterrupt:
        print("[piper_svc] interrupted, shutting down")
    finally:
        service._stop.set()
        service._close_writers()


if __name__ == "__main__":
    main()
