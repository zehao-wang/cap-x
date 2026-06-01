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
        if self._wrist_cam is not None:
            self._wrist_cam.stop()
        if self._zed is not None:
            self._zed.stop()

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
