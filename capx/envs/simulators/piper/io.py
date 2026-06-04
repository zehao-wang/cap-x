"""Recording, rendering, reward stubs, and cleanup for Piper real env."""

from __future__ import annotations

import numpy as np

from capx.utils.camera_utils import obs_get_rgb
from capx.utils.video_utils import resize_with_pad


class PiperIOMixin:
    def compute_reward(self) -> float:
        return 0.0

    def task_completed(self) -> bool:
        return False

    def is_connected(self, max_age: float = 2.0) -> bool:
        """Best-effort check that the PIPER CAN link is up and streaming joints.

        The guided reset wizard's connection step loops on this. We treat the
        arm as connected when a joint-state message reads back; a raised
        exception or a missing message means CAN is down / the arm is unpowered.
        (``max_age`` mirrors the franka signature; the piper SDK does not expose
        a per-message timestamp, so a successful read is the liveness signal.)
        """
        piper = getattr(self, "_piper", None)
        if piper is None:
            return False
        try:
            msgs = piper.GetArmJointMsgs()
        except Exception:
            return False
        return msgs is not None and getattr(msgs, "joint_state", None) is not None

    def get_observation(self) -> dict:
        self._update_from_hardware()
        return self.obs

    def enable_video_capture(
        self,
        enabled: bool = True,
        *,
        clear: bool = True,
        wrist_camera: bool = False,
    ) -> None:
        self._record_frames = enabled
        self._record_wrist_camera = bool(enabled and wrist_camera)
        if clear:
            self._frame_buffer.clear()
            self._wrist_frame_buffer.clear()
        if enabled:
            self._record_frame()

    def get_video_frames(self, *, clear: bool = False) -> list[np.ndarray]:
        frames = [frame.copy() for frame in self._frame_buffer]
        if clear:
            self._frame_buffer.clear()
        return frames

    def get_video_frame_count(self) -> int:
        return len(self._frame_buffer)

    def get_video_frames_range(self, start: int, end: int) -> list[np.ndarray]:
        return [frame.copy() for frame in self._frame_buffer[start:end]]

    def get_wrist_video_frames(self, *, clear: bool = False) -> list[np.ndarray]:
        frames = [frame.copy() for frame in self._wrist_frame_buffer]
        if clear:
            self._wrist_frame_buffer.clear()
        return frames

    def get_wrist_video_frames_range(self, start: int, end: int) -> list[np.ndarray]:
        return [frame.copy() for frame in self._wrist_frame_buffer[start:end]]

    def _record_frame(self) -> None:
        rgb_imgs = obs_get_rgb(self.obs)
        if not rgb_imgs:
            return
        frame = resize_with_pad(list(rgb_imgs.values())[0], 480, 640)
        self._frame_buffer.append(frame)
        if self._record_wrist_camera:
            wrist = (
                self.obs.get("robot0_eye_in_hand", {})
                .get("images", {})
                .get("rgb")
            )
            if wrist is not None:
                self._wrist_frame_buffer.append(resize_with_pad(wrist, 480, 640))

    def render(self, mode: str = "rgb_array") -> np.ndarray:
        obs = self.get_observation()
        rgb = obs.get("robot0_robotview", {}).get("images", {}).get("rgb")
        if rgb is not None:
            return rgb
        return np.zeros((480, 640, 3), dtype=np.uint8)

    def render_wrist(self) -> np.ndarray | None:
        obs = self.get_observation()
        return obs.get("robot0_eye_in_hand", {}).get("images", {}).get("rgb")

    def close(self) -> None:
        # Stop the live-preview daemon first so it isn't mid-read when we tear
        # down the cameras / viser.
        if hasattr(self, "stop_live_preview"):
            try:
                self.stop_live_preview()
            except Exception:
                pass
        if self._wrist_cam is not None:
            self._wrist_cam.stop()
        if self._zed is not None:
            self._zed.stop()
        # Release the viser server so its port (CAPX_VISER_PORT) is freed for the
        # next env instead of forcing viser to auto-increment and leaving a stale
        # server the web proxy could latch onto.
        viser_server = getattr(self, "viser_server", None)
        if viser_server is not None:
            try:
                viser_server.stop()
            except Exception:
                pass
            self.viser_server = None

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass
        try:
            if hasattr(self, "_piper") and self._piper is not None:
                self._piper.DisconnectPort()
        except Exception:
            pass


__all__ = ["PiperIOMixin"]
