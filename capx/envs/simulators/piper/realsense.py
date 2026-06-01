"""RealSense bridge used by the real Piper wrist camera."""

from __future__ import annotations

import time
from typing import Any

import numpy as np


class _RealSenseD435Bridge:
    """In-process pyrealsense2 capture for a wrist-mounted D435-class camera."""

    def __init__(
        self,
        *,
        serial: str | None = None,
        fps: int = 30,
        width: int = 640,
        height: int = 480,
        use_depth: bool = True,
        align_depth_to_color: bool = True,
    ) -> None:
        self.serial = serial
        self.fps = int(fps)
        self.width = int(width)
        self.height = int(height)
        self.use_depth = bool(use_depth)
        self.align_depth_to_color = bool(align_depth_to_color)
        self._pipeline: Any | None = None
        self._align: Any | None = None
        self._depth_scale = 0.001
        self._intrinsics = np.eye(3, dtype=np.float64)
        self._distortion = np.zeros(5, dtype=np.float64)

    def start(self, timeout_sec: float = 10.0) -> None:
        try:
            import pyrealsense2 as rs  # type: ignore
        except ImportError as e:
            raise RuntimeError(
                "pyrealsense2 is required for the Piper wrist RealSense camera. "
                "Install it with `uv pip install pyrealsense2` or `uv sync --extra piper`."
            ) from e

        self._pipeline = rs.pipeline()
        cfg = rs.config()
        if self.serial:
            cfg.enable_device(self.serial)
        cfg.enable_stream(
            rs.stream.color,
            self.width,
            self.height,
            rs.format.rgb8,
            self.fps,
        )
        if self.use_depth:
            cfg.enable_stream(
                rs.stream.depth,
                self.width,
                self.height,
                rs.format.z16,
                self.fps,
            )

        profile = self._pipeline.start(cfg)
        if self.use_depth:
            if self.align_depth_to_color:
                self._align = rs.align(rs.stream.color)
            depth_sensor = profile.get_device().first_depth_sensor()
            self._depth_scale = float(depth_sensor.get_depth_scale())

        color_profile = profile.get_stream(rs.stream.color).as_video_stream_profile()
        intr = color_profile.get_intrinsics()
        self.width = int(intr.width)
        self.height = int(intr.height)
        self._intrinsics = np.array(
            [[intr.fx, 0.0, intr.ppx], [0.0, intr.fy, intr.ppy], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )
        coeffs = list(getattr(intr, "coeffs", []) or [])
        coeffs = (coeffs + [0.0] * 5)[:5]
        self._distortion = np.asarray(coeffs, dtype=np.float64)

        deadline = time.time() + float(timeout_sec)
        while time.time() < deadline:
            rgb, _depth = self.read_frames(timeout_ms=500)
            if rgb is not None:
                return
        raise TimeoutError(f"RealSense D435 did not produce a frame within {timeout_sec}s")

    def intrinsics_matrix(self) -> np.ndarray:
        return self._intrinsics.copy()

    def distortion_coeffs(self) -> np.ndarray:
        return self._distortion.copy()

    def read_frames(
        self, *, timeout_ms: int = 0
    ) -> tuple[np.ndarray | None, np.ndarray | None]:
        if self._pipeline is None:
            return None, None
        try:
            frames = (
                self._pipeline.wait_for_frames(timeout_ms)
                if timeout_ms > 0
                else self._pipeline.poll_for_frames()
            )
        except Exception:
            return None, None
        if not frames:
            return None, None

        if self._align is not None:
            frames = self._align.process(frames)

        color_frame = frames.get_color_frame()
        if not color_frame:
            return None, None
        rgb = np.asanyarray(color_frame.get_data()).copy()

        depth_m: np.ndarray | None = None
        if self.use_depth:
            depth_frame = frames.get_depth_frame()
            if depth_frame:
                depth_z16 = np.asanyarray(depth_frame.get_data()).astype(np.float32)
                depth_m = depth_z16 * self._depth_scale
                depth_m[depth_z16 == 0] = np.nan
                depth_m = depth_m[:, :, None]
        return rgb, depth_m

    def stop(self) -> None:
        if self._pipeline is not None:
            try:
                self._pipeline.stop()
            except Exception:
                pass
        self._pipeline = None
        self._align = None


__all__ = ["_RealSenseD435Bridge"]
