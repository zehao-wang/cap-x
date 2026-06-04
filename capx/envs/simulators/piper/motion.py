"""Motion and gripper commands for the real Piper low-level env."""

from __future__ import annotations

import time

import numpy as np

from capx.envs.simulators.piper.common import (
    PIPER_GRIPPER_CONTACT_EFFORT,
    PIPER_GRIPPER_EFFORT,
    PIPER_GRIPPER_RANGE_M,
    PIPER_GRIPPER_STATUS,
    RAD_TO_RAW_MDEG,
)


class PiperMotionMixin:
    def goto_home_blocking(
        self,
        *,
        tolerance: float = 0.02,
        max_steps: int = 500,
    ) -> None:
        self.move_to_joints_blocking(
            self.HOME_JOINTS_RAD,
            tolerance=tolerance,
            max_steps=max_steps,
        )

    def return_to_rest_pose(self) -> None:
        """Open the gripper and drive all joints back to the home (rest) pose.

        This is the real-robot reset primitive the guided reset wizard calls on
        the initial reset and on every feedback retry — a physical arm has no
        sim snapshot to restore to, so a re-home is the canonical "reset".
        """
        self._set_gripper(1.0)
        self.goto_home_blocking()

    def move_to_joints_blocking(
        self,
        joints: np.ndarray,
        *,
        tolerance: float = 0.02,
        max_steps: int = 500,
    ) -> None:
        target_rad = np.asarray(joints, dtype=np.float64).reshape(6)
        self._current_joints = target_rad.copy()
        target_raw = (target_rad * RAD_TO_RAW_MDEG).astype(np.int64).tolist()

        for step in range(max_steps):
            self._piper.MotionCtrl_2(0x01, 0x01, self.MOTION_SPEED, 0x00)
            self._piper.JointCtrl(*target_raw)
            self._update_from_hardware()
            current = self.obs["robot_joint_pos"][:6].astype(np.float64)
            if np.max(np.abs(current - target_rad)) < tolerance:
                break
            if self._record_frames and step % 4 == 0:
                self._record_frame()
            if self.viser_server is not None and step % 4 == 0:
                self._update_viser_server()
            time.sleep(0.01)
        if self.viser_server is not None:
            self._update_viser_server()

    def execute_joint_trajectory(
        self,
        trajectory: np.ndarray,
        *,
        command_dt: float = 0.02,
        final_tolerance: float = 0.02,
        final_max_steps: int = 260,
    ) -> None:
        traj = np.asarray(trajectory, dtype=np.float64)
        if traj.ndim != 2 or traj.shape[1] != 6:
            raise ValueError(f"trajectory must be (T, 6), got shape {traj.shape}")
        if traj.shape[0] == 0:
            return

        for i, target_rad in enumerate(traj):
            self._current_joints = target_rad.copy()
            target_raw = (target_rad * RAD_TO_RAW_MDEG).astype(np.int64).tolist()
            self._piper.MotionCtrl_2(0x01, 0x01, self.MOTION_SPEED, 0x00)
            self._piper.JointCtrl(*target_raw)
            self._update_from_hardware()
            if self._record_frames and i % 2 == 0:
                self._record_frame()
            if self.viser_server is not None and i % 2 == 0:
                self._update_viser_server()
            time.sleep(command_dt)

        final_target = traj[-1]
        final_raw = (final_target * RAD_TO_RAW_MDEG).astype(np.int64).tolist()
        for step in range(final_max_steps):
            self._piper.MotionCtrl_2(0x01, 0x01, self.MOTION_SPEED, 0x00)
            self._piper.JointCtrl(*final_raw)
            self._update_from_hardware()
            current = self.obs["robot_joint_pos"][:6].astype(np.float64)
            if np.max(np.abs(current - final_target)) < final_tolerance:
                break
            if self._record_frames and step % 4 == 0:
                self._record_frame()
            if self.viser_server is not None and step % 4 == 0:
                self._update_viser_server()
            time.sleep(0.01)

        if self.viser_server is not None:
            self._update_viser_server()

    def _send_gripper(self, target_um: int) -> None:
        target_um = int(abs(target_um))
        is_close = target_um < int(PIPER_GRIPPER_RANGE_M * 1e6) // 2

        if not is_close:
            self._gripper_hold_um = -1
            cmd_um = target_um
        elif self._gripper_hold_um >= 0:
            cmd_um = self._gripper_hold_um
        else:
            try:
                fb = self._piper.GetArmGripperMsgs().gripper_state
                if abs(fb.grippers_effort) >= PIPER_GRIPPER_CONTACT_EFFORT:
                    self._gripper_hold_um = int(fb.grippers_angle)
                    cmd_um = self._gripper_hold_um
                else:
                    cmd_um = target_um
            except Exception:
                cmd_um = target_um

        self._piper.GripperCtrl(
            cmd_um, PIPER_GRIPPER_EFFORT, PIPER_GRIPPER_STATUS, 0
        )

    def _set_gripper(self, fraction: float) -> None:
        self._gripper_fraction = float(np.clip(fraction, 0.0, 1.0))
        range_um = int(self._gripper_fraction * PIPER_GRIPPER_RANGE_M * 1e6)
        self._send_gripper(range_um)
        time.sleep(0.2)

    def _step_once(self) -> None:
        target_raw = (
            (self._current_joints * RAD_TO_RAW_MDEG).astype(np.int64).tolist()
        )
        self._piper.MotionCtrl_2(0x01, 0x01, self.MOTION_SPEED, 0x00)
        self._piper.JointCtrl(*target_raw)
        range_um = int(self._gripper_fraction * PIPER_GRIPPER_RANGE_M * 1e6)
        self._send_gripper(range_um)
        time.sleep(0.02)
        if self._record_frames:
            self._record_frame()
        if self.viser_server is not None:
            self._update_viser_server()


__all__ = ["PiperMotionMixin"]
