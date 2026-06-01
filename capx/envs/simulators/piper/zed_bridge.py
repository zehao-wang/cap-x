"""ZED bridge used by the real Piper environment."""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
import time
from multiprocessing.shared_memory import SharedMemory
from typing import Any

import numpy as np

from capx.envs.simulators.piper.common import RAW_MM_TO_M


_ZED_MAX_W, _ZED_MAX_H = 2208, 1242
_ZED_META_BYTES = 25
_ZED_CTRL_UNSET = -1000
_ZED_CTRL_AUTO = -1


class _Zed2iBridge:
    """Subprocess/shared-memory bridge for a ZED 2i camera."""

    def __init__(
        self,
        bridge_script: str,
        bridge_python: str,
        fps: int = 30,
        width: int = 1280,
        height: int = 720,
        use_depth: bool = True,
        depth_mode: str = "PERFORMANCE",
        open_timeout_sec: float = 15.0,
        open_deadline_sec: float = 180.0,
    ) -> None:
        self.bridge_script = bridge_script
        self.bridge_python = bridge_python
        self.fps = fps
        self.width = width
        self.height = height
        self.use_depth = use_depth
        self.depth_mode = depth_mode
        self.open_timeout_sec = float(open_timeout_sec)
        self.open_deadline_sec = float(open_deadline_sec)

        self._shm_color: SharedMemory | None = None
        self._shm_depth: SharedMemory | None = None
        self._shm_meta: SharedMemory | None = None
        self._proc: subprocess.Popen | None = None
        self._color_arr: np.ndarray | None = None
        self._depth_arr: np.ndarray | None = None
        self._params_file: str | None = None
        self._params: dict[str, Any] = {}

    def start(self, timeout: float | None = None) -> None:
        if timeout is None:
            timeout = self.open_deadline_sec + 30.0
        if not os.path.exists(self.bridge_script):
            raise FileNotFoundError(
                f"ZED bridge script not found: {self.bridge_script}. "
                f"Point PIPER_ZED_BRIDGE at robodata_Agilex/camera/zed_bridge.py."
            )

        streams = "rgbd" if self.use_depth else "rgb"
        self._shm_color = SharedMemory(create=True, size=_ZED_MAX_H * _ZED_MAX_W * 3)
        self._shm_depth = (
            SharedMemory(create=True, size=_ZED_MAX_H * _ZED_MAX_W * 2)
            if self.use_depth
            else None
        )
        self._shm_meta = SharedMemory(create=True, size=_ZED_META_BYTES)
        self._shm_meta.buf[16] = 0
        self._shm_meta.buf[17:21] = np.int32(_ZED_CTRL_UNSET).tobytes()
        self._shm_meta.buf[21:25] = np.int32(_ZED_CTRL_UNSET).tobytes()
        self._params_file = tempfile.mktemp(suffix="_zed2_params.json")

        cmd = [
            self.bridge_python,
            self.bridge_script,
            "--shm-color",
            self._shm_color.name,
            "--shm-meta",
            self._shm_meta.name,
            "--fps",
            str(self.fps),
            "--width",
            str(self.width),
            "--height",
            str(self.height),
            "--streams",
            streams,
            "--params-file",
            self._params_file,
            "--open-timeout-sec",
            str(self.open_timeout_sec),
            "--open-deadline-sec",
            str(self.open_deadline_sec),
        ]
        if self.use_depth:
            cmd += [
                "--shm-depth",
                self._shm_depth.name,
                "--depth-mode",
                self.depth_mode,
            ]

        self._proc = subprocess.Popen(cmd, stderr=subprocess.PIPE)

        deadline = time.time() + timeout
        while time.time() < deadline:
            status = self._shm_meta.buf[16]
            if status == 255:
                err = (
                    self._proc.stderr.read().decode(errors="replace")
                    if self._proc.stderr
                    else ""
                )
                self.stop()
                raise RuntimeError(
                    f"ZED bridge failed to open camera.\n"
                    f"Bridge python: {self.bridge_python}\nBridge stderr:\n{err}"
                )
            if status >= 1:
                break
            time.sleep(0.05)
        else:
            self.stop()
            raise TimeoutError(f"ZED bridge did not start within {timeout}s")

        w = int(np.frombuffer(self._shm_meta.buf[0:4], dtype=np.int32)[0])
        h = int(np.frombuffer(self._shm_meta.buf[4:8], dtype=np.int32)[0])
        self.width, self.height = w, h
        self._color_arr = np.ndarray(
            (h, w, 3), dtype=np.uint8, buffer=self._shm_color.buf
        )
        if self.use_depth:
            self._depth_arr = np.ndarray(
                (h, w), dtype=np.uint16, buffer=self._shm_depth.buf
            )

        if self._params_file and os.path.exists(self._params_file):
            with open(self._params_file) as f:
                self._params = json.load(f)

    def intrinsics_matrix(self) -> np.ndarray:
        left = self._params.get("left", {})
        fx = float(left.get("fx", 0.0))
        fy = float(left.get("fy", 0.0))
        cx = float(left.get("cx", 0.0))
        cy = float(left.get("cy", 0.0))
        return np.array(
            [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64
        )

    def _write_ctrl(self, offset: int, value: int) -> None:
        if self._shm_meta is None:
            return
        self._shm_meta.buf[offset:offset + 4] = np.int32(value).tobytes()

    def set_gain(self, value: int) -> None:
        self._write_ctrl(17, int(np.clip(value, 0, 100)))

    def set_exposure(self, value: int) -> None:
        self._write_ctrl(21, int(np.clip(value, 0, 100)))

    def set_auto_exposure_gain(self) -> None:
        self._write_ctrl(17, _ZED_CTRL_AUTO)
        self._write_ctrl(21, _ZED_CTRL_AUTO)

    def read_frames(self) -> tuple[np.ndarray | None, np.ndarray | None]:
        if self._shm_meta is None or self._shm_meta.buf[16] != 2:
            return None, None
        rgb = self._color_arr.copy() if self._color_arr is not None else None
        depth_m: np.ndarray | None = None
        if self._depth_arr is not None:
            depth_mm = self._depth_arr.astype(np.float32)
            depth_m = depth_mm * RAW_MM_TO_M
            depth_m[depth_mm == 0] = np.nan
            depth_m = depth_m[:, :, None]
        return rgb, depth_m

    def stop(self) -> None:
        if self._proc is not None:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._proc.kill()
            self._proc = None
        for shm in (self._shm_color, self._shm_depth, self._shm_meta):
            if shm is not None:
                try:
                    shm.close()
                    shm.unlink()
                except Exception:
                    pass
        self._shm_color = self._shm_depth = self._shm_meta = None
        if self._params_file and os.path.exists(self._params_file):
            try:
                os.unlink(self._params_file)
            except Exception:
                pass

__all__ = ["_Zed2iBridge"]
