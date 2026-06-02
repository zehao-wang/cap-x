"""Startup helpers for the real Piper low-level environment."""

from __future__ import annotations

import os
import time
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation as SciRotation

from capx.envs.simulators.piper.cameras import _RealSenseD435Bridge, _Zed2iBridge
from capx.envs.simulators.piper.common import (
    DEFAULT_PIPER_URDF,
    PIPER_GRIPPER_RANGE_M,
    _UrdfLinkFk,
    _env_bool,
    _load_extrinsics,
    _resolve_piper_mesh,
    _transform_from_xyz_rpy,
)
from capx.envs.simulators.piper.viser import (
    _apply_intrinsics_flip,
    _wrist_image_flip_mode,
)


class PiperSetupMixin:
    def _init_runtime_state(
        self,
        wrist_camera_enabled: bool | None,
        wrist_camera_link_name: str | None,
        wrist_camera_position: list[float] | None,
        wrist_camera_rpy_radians: list[float] | None,
    ) -> None:
        self.obs: dict[str, Any] = {}
        self.latest_action: dict[str, Any] = {}
        self._current_joints = np.zeros(6, dtype=np.float64)
        self._gripper_fraction = 1.0
        self._gripper_hold_um = -1
        self._wrist_camera_enabled = (
            self.WRIST_CAMERA_ENABLED
            if wrist_camera_enabled is None
            else bool(wrist_camera_enabled)
        )
        self._wrist_camera_required = _env_bool("PIPER_WRIST_CAMERA_REQUIRED", False)
        self._wrist_camera_link_name = (
            wrist_camera_link_name or self.WRIST_CAMERA_LINK_NAME
        )
        wrist_pos = np.asarray(
            wrist_camera_position
            if wrist_camera_position is not None
            else self.WRIST_CAMERA_POSITION,
            dtype=np.float64,
        ).reshape(3)
        wrist_rpy = np.asarray(
            wrist_camera_rpy_radians
            if wrist_camera_rpy_radians is not None
            else self.WRIST_CAMERA_RPY_RADIANS,
            dtype=np.float64,
        ).reshape(3)
        self._wrist_link_to_camera_mat = _transform_from_xyz_rpy(wrist_pos, wrist_rpy)
        self._wrist_fk: _UrdfLinkFk | None = None
        self._wrist_cam: _RealSenseD435Bridge | None = None
        self._wrist_intrinsics: np.ndarray | None = None

    def _init_camera_extrinsics(self) -> None:
        cam_pos, cam_quat_wxyz = _load_extrinsics(self.CAMERA_EXTRINSICS_FILE)
        rot = SciRotation.from_quat(
            [cam_quat_wxyz[1], cam_quat_wxyz[2], cam_quat_wxyz[3], cam_quat_wxyz[0]]
        )
        cam_pose_mat = np.eye(4, dtype=np.float64)
        cam_pose_mat[:3, :3] = rot.as_matrix()
        cam_pose_mat[:3, 3] = cam_pos
        self._cam_pose_xyz_wxyz = np.concatenate([cam_pos, cam_quat_wxyz])
        self._cam_pose_mat = cam_pose_mat

    def _connect_piper(self) -> None:
        from piper_sdk import C_PiperInterface_V2  # type: ignore

        print(
            f"[piper_real] Connecting to PIPER via {self.CAN_INTERFACE} "
            f"channel={self.CAN_CHANNEL} @ {self.CAN_BITRATE} bps"
        )
        if self.CAN_INTERFACE == "gs_usb":
            self._piper = C_PiperInterface_V2(
                can_name=self.CAN_CHANNEL, judge_flag=False, can_auto_init=False
            )
            self._piper.CreateCanBus(
                can_name=self.CAN_CHANNEL,
                bustype="gs_usb",
                expected_bitrate=self.CAN_BITRATE,
                judge_flag=False,
            )
            self._piper.ConnectPort(can_init=False)
        else:
            self._piper = C_PiperInterface_V2(
                can_name=self.CAN_CHANNEL, judge_flag=True, can_auto_init=True
            )
            self._piper.ConnectPort()

        print("[piper_real] Enabling arm...")
        t0 = time.time()
        while time.time() - t0 < 5.0:
            if self._piper.EnablePiper():
                break
            time.sleep(0.05)
        else:
            print("[piper_real] WARNING: EnablePiper did not confirm within 5s")
        self._send_gripper(int(PIPER_GRIPPER_RANGE_M * 1e6))
        time.sleep(0.2)

    def _start_cameras(
        self,
        wrist_camera_serial: str | None,
        wrist_camera_fps: int | None,
        wrist_camera_width: int | None,
        wrist_camera_height: int | None,
        wrist_camera_use_depth: bool | None,
        *,
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
        source = (zed_source or self.ZED_SOURCE).lower()
        if source == "service":
            # Read RGB+depth from the standalone ZED service (cap-x does no depth
            # compute). Drop-in for the local bridge on the observation path.
            from capx.envs.simulators.piper.zed_service_client import _ZedServiceClient

            # No give-up: the client heartbeat-waits until the service is reachable
            # (warning on screen with the fail reason each beat).
            self._zed = _ZedServiceClient(
                socket_path=zed_service_socket or self.ZED_SERVICE_SOCKET,
                connect_timeout_sec=self.ZED_OPEN_TIMEOUT_SEC,
            )
            print(f"[piper_real] Connecting to ZED service at {self._zed.socket_path}...")
        else:
            self._zed = _Zed2iBridge(
                bridge_script=self.ZED_BRIDGE_SCRIPT,
                bridge_python=self.ZED_BRIDGE_PYTHON,
                fps=int(zed_fps) if zed_fps is not None else self.ZED_FPS,
                width=int(zed_width) if zed_width is not None else self.ZED_WIDTH,
                height=int(zed_height) if zed_height is not None else self.ZED_HEIGHT,
                use_depth=True,
                depth_mode=str(zed_depth_mode) if zed_depth_mode is not None else self.ZED_DEPTH_MODE,
                open_timeout_sec=self.ZED_OPEN_TIMEOUT_SEC,
                open_deadline_sec=self.ZED_OPEN_DEADLINE_SEC,
            )
            print("[piper_real] Starting ZED2i bridge...")
        self._zed.start()
        self._intrinsics = self._zed.intrinsics_matrix()
        print(f"[piper_real] ZED ready at {self._zed.width}x{self._zed.height}")
        if source != "service":
            # Exposure/gain are a camera bring-up concern owned by the service in
            # service mode; cap-x only reads there.
            self._apply_zed_color_controls(
                zed_auto_exposure_gain=zed_auto_exposure_gain,
                zed_exposure=zed_exposure,
                zed_gain=zed_gain,
            )

        if not self._wrist_camera_enabled:
            return
        try:
            self._wrist_fk = _UrdfLinkFk(DEFAULT_PIPER_URDF, self._wrist_camera_link_name)
            self._wrist_cam = _RealSenseD435Bridge(
                serial=wrist_camera_serial or self.WRIST_CAMERA_SERIAL,
                fps=wrist_camera_fps or self.WRIST_CAMERA_FPS,
                width=wrist_camera_width or self.WRIST_CAMERA_WIDTH,
                height=wrist_camera_height or self.WRIST_CAMERA_HEIGHT,
                use_depth=(
                    self.WRIST_CAMERA_USE_DEPTH
                    if wrist_camera_use_depth is None
                    else bool(wrist_camera_use_depth)
                ),
                align_depth_to_color=True,
            )
            serial_msg = f" serial={self._wrist_cam.serial}" if self._wrist_cam.serial else ""
            print(f"[piper_real] Starting wrist RealSense D435{serial_msg}...")
            self._wrist_cam.start()
            # Wrist images are flipped at obs-write time (see
            # _update_wrist_observation); pre-flip the intrinsics once so the
            # cx/cy used by depth_color_to_pointcloud match the flipped frame.
            raw_K = self._wrist_cam.intrinsics_matrix()
            self._wrist_intrinsics = _apply_intrinsics_flip(
                raw_K,
                _wrist_image_flip_mode(),
                int(self._wrist_cam.width),
                int(self._wrist_cam.height),
            )
            print(
                "[piper_real] Wrist RealSense ready at "
                f"{self._wrist_cam.width}x{self._wrist_cam.height}"
            )
        except Exception as e:
            self._wrist_cam = None
            self._wrist_fk = None
            self._wrist_intrinsics = None
            if self._wrist_camera_required:
                raise
            print(
                "[piper_real] Wrist RealSense disabled "
                f"({type(e).__name__}: {e})"
            )

    def _apply_zed_color_controls(
        self,
        *,
        zed_auto_exposure_gain: bool | None,
        zed_exposure: int | None,
        zed_gain: int | None,
    ) -> None:
        auto_exposure_gain = (
            self.ZED_AUTO_EXPOSURE_GAIN
            if zed_auto_exposure_gain is None
            else bool(zed_auto_exposure_gain)
        )
        exposure = self.ZED_EXPOSURE if zed_exposure is None else int(zed_exposure)
        gain = self.ZED_GAIN if zed_gain is None else int(zed_gain)

        if auto_exposure_gain is True:
            self._zed.set_auto_exposure_gain()
            print("[piper_real] ZED2i auto exposure/gain enabled")
            return
        if exposure is not None:
            self._zed.set_exposure(exposure)
            print(f"[piper_real] ZED2i exposure={int(np.clip(exposure, 0, 100))}")
        if gain is not None:
            self._zed.set_gain(gain)
            print(f"[piper_real] ZED2i gain={int(np.clip(gain, 0, 100))}")

    def _init_recording_state(self) -> None:
        self._record_frames = False
        self._frame_buffer: list[np.ndarray] = []
        self._record_wrist_camera = False
        self._wrist_frame_buffer: list[np.ndarray] = []

    def _init_viser(self, viser_debug: bool) -> None:
        self.viser_server = None
        self.urdf_vis = None
        self.viser_img_handle = None
        self.viser_depth_handle = None
        self.viser_wrist_img_handle = None
        self.viser_wrist_depth_handle = None
        self._depth_view_sliders: dict[str, Any] | None = None
        self.image_frustum_handle = None
        self.image_frustum_handles: dict[str, Any] = {}
        self.grasp_mesh_handle = None
        self.cube_frame_handle = None
        self.cube_point_cloud_handle = None
        self.ik_target_handle = None
        self.preview_path_handle = None
        self.preview_target_handle = None
        self._viser_scene_update_interval = float(
            os.environ.get("PIPER_VISER_SCENE_UPDATE_SEC", "0.25")
        )
        self._last_viser_scene_update_time = 0.0
        self._urdf_n_actuated = 0
        if not viser_debug:
            return
        try:
            import viser  # type: ignore
            import yourdfpy  # type: ignore
            from viser.extras import ViserUrdf  # type: ignore

            self.viser_server = viser.ViserServer(
                port=int(os.environ.get("CAPX_VISER_PORT", "8201"))
            )
            urdf = yourdfpy.URDF.load(
                DEFAULT_PIPER_URDF,
                filename_handler=_resolve_piper_mesh,
                build_collision_scene_graph=False,
                load_collision_meshes=False,
            )
            self._urdf_n_actuated = len(urdf.actuated_joint_names)
            self.urdf_vis = ViserUrdf(
                self.viser_server, urdf_or_path=urdf, load_meshes=True
            )
            print(
                f"[piper_real] Viser server up on port "
                f"{os.environ.get('CAPX_VISER_PORT', '8201')} "
                f"(URDF actuated joints: {self._urdf_n_actuated})"
            )
        except Exception as e:
            print(
                f"[piper_real] Viser init failed ({type(e).__name__}: {e}); "
                "3D viz disabled."
            )
            self.viser_server = None


__all__ = ["PiperSetupMixin"]
