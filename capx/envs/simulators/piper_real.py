"""Low-level environment for a real Agilex PIPER arm + cameras.

This module follows the same public shape as ``franka_real.py``: the simulator
entrypoint class lives here, while Piper-specific setup, motion, rendering, and
camera helpers live under ``capx.envs.simulators.piper``.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

from capx.envs.base import BaseEnv
from capx.envs.simulators.piper.cameras import _RealSenseD435Bridge, _Zed2iBridge
from capx.envs.simulators.piper.common import (
    DEFAULT_PIPER_EEF_LINK,
    DEFAULT_PIPER_URDF,
    PIPER_GRIPPER_RANGE_M,
    RAW_MDEG_TO_RAD,
    _UrdfLinkFk,
    _env_bool,
    _env_float_list,
    _pose_xyz_wxyz_from_matrix,
    _resolve_piper_mesh,
)
from capx.envs.simulators.piper.io import PiperIOMixin
from capx.envs.simulators.piper.motion import PiperMotionMixin
from capx.envs.simulators.piper.setup import PiperSetupMixin
from capx.envs.simulators.piper.viser import (
    PiperViserMixin,
    _apply_display_flip,
    _wrist_image_flip_mode,
)


class PiperRealLowLevel(
    PiperSetupMixin,
    PiperMotionMixin,
    PiperIOMixin,
    PiperViserMixin,
    BaseEnv,
):
    """Real Agilex PIPER over CAN with ZED scene + optional wrist RealSense."""

    CAN_INTERFACE = os.environ.get("PIPER_CAN_INTERFACE", "socketcan")
    CAN_CHANNEL = os.environ.get("PIPER_CAN_CHANNEL", "can1")
    CAN_BITRATE = int(os.environ.get("PIPER_CAN_BITRATE", "1000000"))
    MOTION_SPEED = int(os.environ.get("PIPER_MOTION_SPEED", "30"))
    CAMERA_EXTRINSICS_FILE = os.environ.get("PIPER_CAMERA_EXTRINSICS", "") or None
    ZED_BRIDGE_SCRIPT = os.environ.get(
        "PIPER_ZED_BRIDGE",
        str(Path.home() / "Documents/Projects/robodata_Agilex/camera/zed_bridge.py"),
    )
    ZED_BRIDGE_PYTHON = os.environ.get("ZED_BRIDGE_PYTHON", sys.executable)
    ZED_FPS = int(os.environ.get("PIPER_ZED_FPS", "15"))
    ZED_WIDTH = int(os.environ.get("PIPER_ZED_WIDTH", "1280"))
    ZED_HEIGHT = int(os.environ.get("PIPER_ZED_HEIGHT", "720"))
    ZED_DEPTH_MODE = os.environ.get("PIPER_ZED_DEPTH_MODE", "NEURAL")
    ZED_AUTO_EXPOSURE_GAIN = (
        _env_bool("PIPER_ZED_AUTO_EXPOSURE_GAIN", False)
        if os.environ.get("PIPER_ZED_AUTO_EXPOSURE_GAIN") is not None
        else None
    )
    ZED_EXPOSURE = (
        int(os.environ["PIPER_ZED_EXPOSURE"])
        if os.environ.get("PIPER_ZED_EXPOSURE")
        else None
    )
    ZED_GAIN = (
        int(os.environ["PIPER_ZED_GAIN"])
        if os.environ.get("PIPER_ZED_GAIN")
        else None
    )
    ZED_OPEN_TIMEOUT_SEC = float(os.environ.get("PIPER_ZED_OPEN_TIMEOUT_SEC", "15"))
    ZED_OPEN_DEADLINE_SEC = float(os.environ.get("PIPER_ZED_OPEN_DEADLINE_SEC", "180"))
    # Scene ZED source: "bridge" (local pyzed subprocess, default) or "service"
    # (read RGB+depth from the standalone ZED service; cap-x does no depth compute).
    ZED_SOURCE = os.environ.get("PIPER_ZED_SOURCE", "bridge")
    ZED_SERVICE_SOCKET = os.environ.get("PIPER_ZED_SERVICE_SOCKET", "/tmp/piper/zed.sock")

    WRIST_CAMERA_ENABLED = _env_bool("PIPER_WRIST_CAMERA_ENABLED", False)
    WRIST_CAMERA_SERIAL = os.environ.get("PIPER_WRIST_CAMERA_SERIAL") or None
    WRIST_CAMERA_FPS = int(os.environ.get("PIPER_WRIST_CAMERA_FPS", "30"))
    WRIST_CAMERA_WIDTH = int(os.environ.get("PIPER_WRIST_CAMERA_WIDTH", "640"))
    WRIST_CAMERA_HEIGHT = int(os.environ.get("PIPER_WRIST_CAMERA_HEIGHT", "480"))
    WRIST_CAMERA_USE_DEPTH = _env_bool("PIPER_WRIST_CAMERA_USE_DEPTH", True)
    WRIST_CAMERA_LINK_NAME = os.environ.get(
        "PIPER_WRIST_CAMERA_LINK_NAME", DEFAULT_PIPER_EEF_LINK
    )
    WRIST_CAMERA_POSITION = _env_float_list(
        "PIPER_WRIST_CAMERA_POSITION", [0.0, 0.0, 0.0], 3
    )
    WRIST_CAMERA_RPY_RADIANS = _env_float_list(
        "PIPER_WRIST_CAMERA_RPY_RADIANS", [0.0, 0.0, 0.0], 3
    )
    HOME_JOINTS_RAD = np.zeros(6, dtype=np.float64)

    def __init__(
        self,
        seed: int | None = None,
        viser_debug: bool = True,
        privileged: bool = False,
        enable_render: bool = False,
        wrist_camera_enabled: bool | None = None,
        wrist_camera_serial: str | None = None,
        wrist_camera_fps: int | None = None,
        wrist_camera_width: int | None = None,
        wrist_camera_height: int | None = None,
        wrist_camera_use_depth: bool | None = None,
        wrist_camera_link_name: str | None = None,
        wrist_camera_position: list[float] | None = None,
        wrist_camera_rpy_radians: list[float] | None = None,
        zed_fps: int | None = None,
        zed_width: int | None = None,
        zed_height: int | None = None,
        zed_depth_mode: str | None = None,
        zed_auto_exposure_gain: bool | None = None,
        zed_exposure: int | None = None,
        zed_gain: int | None = None,
        zed_source: str | None = None,
        zed_service_socket: str | None = None,
    ) -> None:
        super().__init__()
        self._init_runtime_state(
            wrist_camera_enabled,
            wrist_camera_link_name,
            wrist_camera_position,
            wrist_camera_rpy_radians,
        )
        self._init_camera_extrinsics()
        self._connect_piper()
        self._start_cameras(
            wrist_camera_serial,
            wrist_camera_fps,
            wrist_camera_width,
            wrist_camera_height,
            wrist_camera_use_depth,
            zed_fps=zed_fps,
            zed_width=zed_width,
            zed_height=zed_height,
            zed_depth_mode=zed_depth_mode,
            zed_auto_exposure_gain=zed_auto_exposure_gain,
            zed_exposure=zed_exposure,
            zed_gain=zed_gain,
            zed_source=zed_source,
            zed_service_socket=zed_service_socket,
        )
        self._init_recording_state()
        self._init_viser(viser_debug)
        print("[piper_real] Moving arm to HOME pose (all joints = 0 deg)...")
        self.goto_home_blocking()

    def _update_from_hardware(self) -> None:
        joints_rad = self._read_arm_joints()
        gripper_m = self._read_gripper_width()
        self.obs["robot_joint_pos"] = np.concatenate(
            [joints_rad, [gripper_m]]
        ).astype(np.float32)
        self._update_zed_observation()
        self._update_wrist_observation(joints_rad)

    def _read_arm_joints(self) -> np.ndarray:
        js = self._piper.GetArmJointMsgs().joint_state
        joints_raw = np.array(
            [
                js.joint_1,
                js.joint_2,
                js.joint_3,
                js.joint_4,
                js.joint_5,
                js.joint_6,
            ],
            dtype=np.float64,
        )
        return joints_raw * RAW_MDEG_TO_RAD

    def _read_gripper_width(self) -> float:
        try:
            gripper_state = self._piper.GetArmGripperMsgs().gripper_state
            return float(gripper_state.grippers_angle) * 1e-6
        except Exception:
            return self._gripper_fraction * PIPER_GRIPPER_RANGE_M

    def _update_zed_observation(self) -> None:
        rgb, depth = self._zed.read_frames()
        if rgb is None:
            return
        view = self.obs.setdefault("robot0_robotview", {})
        images = view.setdefault("images", {})
        images["rgb"] = rgb
        if depth is not None:
            images["depth"] = depth
        view["intrinsics"] = self._intrinsics
        view["pose"] = self._cam_pose_xyz_wxyz
        view["pose_mat"] = self._cam_pose_mat

    def _update_wrist_observation(self, joints_rad: np.ndarray) -> None:
        if self._wrist_cam is None:
            return
        wrist_rgb, wrist_depth = self._wrist_cam.read_frames()
        if wrist_rgb is None:
            return
        # Wrist RealSense is mounted upside-down on the gripper. Flip once
        # here so every consumer (agent API, viser panels, point cloud,
        # frustum thumbnails) sees the same human-readable orientation.
        flip_mode = _wrist_image_flip_mode()
        wrist_rgb = _apply_display_flip(wrist_rgb, flip_mode)
        if wrist_depth is not None:
            wrist_depth = _apply_display_flip(wrist_depth, flip_mode)
        view = self.obs.setdefault("robot0_eye_in_hand", {})
        images = view.setdefault("images", {})
        images["rgb"] = wrist_rgb
        if wrist_depth is not None:
            images["depth"] = wrist_depth
        if self._wrist_intrinsics is not None:
            view["intrinsics"] = self._wrist_intrinsics
        if self._wrist_fk is None:
            return
        try:
            pose_mat = self._wrist_fk.transform(joints_rad) @ self._wrist_link_to_camera_mat
            view["pose_mat"] = pose_mat
            view["pose"] = _pose_xyz_wxyz_from_matrix(pose_mat)
        except Exception:
            pass

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        print("[piper_real] reset: moving arm to HOME pose (all joints = 0 deg)...")
        self.goto_home_blocking()
        self._set_gripper(1.0)
        for _ in range(100):
            self._update_from_hardware()
            if self.obs.get("robot0_robotview", {}).get("images", {}).get("rgb") is not None:
                break
            print("[piper_real] Waiting for first camera frame...")
            time.sleep(0.5)
        if self.viser_server is not None:
            self._update_viser_server(force_scene=True)
            self._refresh_viser_point_cloud()
        return self.obs, {}

    def step(self, action: Any):
        raise NotImplementedError("PiperRealLowLevel is driven through the control API")


__all__ = [
    "DEFAULT_PIPER_EEF_LINK",
    "DEFAULT_PIPER_URDF",
    "PiperRealLowLevel",
    "_RealSenseD435Bridge",
    "_UrdfLinkFk",
    "_Zed2iBridge",
    "_resolve_piper_mesh",
]
