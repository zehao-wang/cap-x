#!/usr/bin/env python3
"""ZED 2i depth service — owns the camera, serves aligned RGB-D over a Unix socket.

Implements ``capx/envs/simulators/piper/ZED_SERVICE_REQUIREMENTS.md`` (wire format v1).
The service **exclusively owns** the ZED 2i, computes **metric depth** from the left/right
stereo pair via TRI-Stereo (learned), and returns RGB (uint8, RGB order) + depth (float32
metres, NaN = invalid) that are spatially aligned and come from the same grab.

Run it with the raiden venv (pyzed + onnxruntime/tensorrt + TRI-Stereo weights), e.g.::

    .../raiden/.venv/bin/python zed_depth_service.py --socket /tmp/piper/zed.sock

Readiness contract (§4): the socket is only created **after** the camera opens and one
self-test ``get_frame`` succeeds. So a reachable socket guarantees valid data.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import socket
import struct
import sys
import threading
import time
from pathlib import Path
from typing import Any, Optional

import numpy as np

_LEN = struct.Struct(">I")  # 4-byte big-endian length prefix (frame + header)
_MAX_FRAME = 256 * 1024 * 1024  # 256 MiB guard, matches the client

# (width, height) -> ZED resolution enum name. Resolved against sl.RESOLUTION at runtime.
_RESOLUTION_BY_WH = {
    (2208, 1242): "HD2K",
    (1920, 1080): "HD1080",
    (1280, 720): "HD720",
    (672, 376): "VGA",
}


# ─────────────────────────────────────────────────────────────────────────────
# Depth backend
# ─────────────────────────────────────────────────────────────────────────────
class _TriStereoBackend:
    """Metric depth from the ZED stereo pair via TRI-Stereo (TensorRT > ONNX)."""

    def __init__(self, variant: str, weights_dir: Optional[str]) -> None:
        from raiden.depth.tri_stereo import (  # noqa: PLC0415
            TRIStereoOnnxDepthPredictor,
            TRIStereoTrtDepthPredictor,
        )

        engine_path = onnx_path = None
        if weights_dir:
            engine_path = str(Path(weights_dir) / f"stereo_{variant}.engine")
            onnx_path = str(Path(weights_dir) / f"stereo_{variant}.onnx")

        if TRIStereoTrtDepthPredictor.engine_available(variant=variant, engine_path=engine_path):
            pred = TRIStereoTrtDepthPredictor(variant=variant, engine_path=engine_path)
            try:
                pred._ensure_loaded()
                self._pred, self.label = pred, f"TRI-Stereo-{variant.upper()} TRT"
                return
            except RuntimeError as exc:
                print(f"[zed_service] TRT engine unusable ({exc}); falling back to ONNX", flush=True)

        if TRIStereoOnnxDepthPredictor.model_available(variant=variant, onnx_path=onnx_path):
            pred = TRIStereoOnnxDepthPredictor(variant=variant, onnx_path=onnx_path)
            pred._ensure_loaded()
            self._pred, self.label = pred, f"TRI-Stereo-{variant.upper()} ONNX"
            return

        raise RuntimeError(
            f"No TRI-Stereo model found for variant {variant!r} "
            f"(weights_dir={weights_dir!r}). Run `git lfs pull` in the raiden repo."
        )

    def depth_m(self, left_bgr: np.ndarray, right_bgr: np.ndarray, fx: float, baseline: float) -> np.ndarray:
        """Return float32 (H, W) depth in metres; 0/invalid pixels become NaN."""
        depth = self._pred.predict(left_bgr, right_bgr, fx, baseline).astype(np.float32, copy=False)
        depth[~np.isfinite(depth)] = np.nan
        depth[depth <= 0.0] = np.nan
        return depth


# ─────────────────────────────────────────────────────────────────────────────
# Camera (exclusive owner)
# ─────────────────────────────────────────────────────────────────────────────
class _ZedCamera:
    """Opens and owns the ZED 2i; grabs synchronized RGB-D under a lock."""

    def __init__(self, args: argparse.Namespace) -> None:
        self._args = args
        self._lock = threading.Lock()
        self._cam = None  # type: ignore[assignment]
        self._left = None
        self._right = None
        self._backend = _TriStereoBackend(args.variant, args.weights_dir)
        self.width = 0
        self.height = 0
        self.fx = self.fy = self.cx = self.cy = 0.0
        self.baseline_m = 0.0

    def open(self) -> None:
        """Open the camera with bounded retry/backoff (§4). Raises on total timeout."""
        import pyzed.sl as sl  # noqa: PLC0415

        res = _RESOLUTION_BY_WH.get((self._args.width, self._args.height))
        if res is None:
            raise RuntimeError(
                f"Unsupported resolution {self._args.width}x{self._args.height}; "
                f"supported: {sorted(_RESOLUTION_BY_WH)}"
            )

        init = sl.InitParameters()
        init.camera_resolution = getattr(sl.RESOLUTION, res)
        init.camera_fps = self._args.fps
        init.depth_mode = sl.DEPTH_MODE.NONE  # we compute depth ourselves (TRI-Stereo)
        init.coordinate_units = sl.UNIT.METER
        init.open_timeout_sec = self._args.open_timeout_sec
        if self._args.serial:
            init.set_from_serial_number(self._args.serial)

        cam = sl.Camera()
        deadline = time.time() + self._args.open_deadline_sec
        attempt = 0
        status = sl.ERROR_CODE.FAILURE
        while time.time() < deadline:
            attempt += 1
            status = cam.open(init)
            if status == sl.ERROR_CODE.SUCCESS:
                break
            print(f"[zed_service] open attempt {attempt} failed ({status}); retrying...", flush=True)
            cam.close()
            time.sleep(min(5.0, max(0.5, deadline - time.time())))
        if status != sl.ERROR_CODE.SUCCESS:
            raise RuntimeError(f"could not open ZED within {self._args.open_deadline_sec}s: {status}")

        info = cam.get_camera_information()
        cal = info.camera_configuration.calibration_parameters
        res_info = info.camera_configuration.resolution
        self.width, self.height = int(res_info.width), int(res_info.height)
        self.fx, self.fy = float(cal.left_cam.fx), float(cal.left_cam.fy)
        self.cx, self.cy = float(cal.left_cam.cx), float(cal.left_cam.cy)
        self.baseline_m = float(abs(cal.get_camera_baseline()))

        self._apply_exposure_gain(sl, cam)

        self._cam = cam
        self._left, self._right = sl.Mat(), sl.Mat()
        print(
            f"[zed_service] camera open: {self.width}x{self.height}@{self._args.fps} "
            f"baseline={self.baseline_m:.4f}m backend={self._backend.label}",
            flush=True,
        )

    def _apply_exposure_gain(self, sl, cam) -> None:  # noqa: ANN001
        # -1 = leave on ZED auto-exposure/gain (default). 0..100 sets a fixed value.
        if self._args.exposure >= 0:
            cam.set_camera_settings(sl.VIDEO_SETTINGS.EXPOSURE, int(self._args.exposure))
        if self._args.gain >= 0:
            cam.set_camera_settings(sl.VIDEO_SETTINGS.GAIN, int(self._args.gain))

    def grab_rgbd(self) -> Optional[tuple[np.ndarray, np.ndarray, int]]:
        """Grab one frame; return (rgb RGB uint8 HxWx3, depth float32 HxW m NaN-invalid, ts_ns).

        Returns None if the grab fails. rgb and depth come from the same grab (§3.3).
        """
        import pyzed.sl as sl  # noqa: PLC0415

        with self._lock:
            if self._cam.grab(sl.RuntimeParameters()) != sl.ERROR_CODE.SUCCESS:
                return None
            self._cam.retrieve_image(self._left, sl.VIEW.LEFT)
            self._cam.retrieve_image(self._right, sl.VIEW.RIGHT)
            left_bgr = self._left.get_data()[:, :, :3].copy()
            right_bgr = self._right.get_data()[:, :, :3].copy()
            ts_ns = self._cam.get_timestamp(sl.TIME_REFERENCE.IMAGE).get_nanoseconds()

        depth = self._backend.depth_m(left_bgr, right_bgr, self.fx, self.baseline_m)
        rgb = np.ascontiguousarray(left_bgr[:, :, ::-1])  # BGR -> RGB
        return rgb, depth, int(ts_ns)

    def intrinsics(self) -> list[list[float]]:
        return [[self.fx, 0.0, self.cx], [0.0, self.fy, self.cy], [0.0, 0.0, 1.0]]

    def close(self) -> None:
        if self._cam is not None:
            with self._lock:
                self._cam.close()
                self._cam = None


# ─────────────────────────────────────────────────────────────────────────────
# Wire protocol (v1)
# ─────────────────────────────────────────────────────────────────────────────
def _recv_exactly(sock: socket.socket, n: int) -> bytes:
    chunks: list[bytes] = []
    got = 0
    while got < n:
        chunk = sock.recv(min(n - got, 1 << 20))
        if not chunk:
            raise ConnectionError("client closed mid-frame")
        chunks.append(chunk)
        got += len(chunk)
    return b"".join(chunks)


def _recv_request(sock: socket.socket) -> dict[str, Any]:
    (n,) = _LEN.unpack(_recv_exactly(sock, 4))
    if n > _MAX_FRAME:
        raise ValueError(f"request too large: {n}")
    return json.loads(_recv_exactly(sock, n).decode("utf-8"))


def _send_payload(sock: socket.socket, payload: bytes) -> None:
    sock.sendall(_LEN.pack(len(payload)) + payload)


def _send_json(sock: socket.socket, obj: dict[str, Any]) -> None:
    _send_payload(sock, json.dumps(obj).encode("utf-8"))


def _build_frame_payload(camera: _ZedCamera, rgb: np.ndarray, depth: np.ndarray, ts_ns: int) -> bytes:
    rgb_bytes = rgb.tobytes()
    depth_bytes = np.ascontiguousarray(depth, dtype=np.float32).tobytes()
    header = {
        "v": 1,
        "ok": True,
        "width": camera.width,
        "height": camera.height,
        "timestamp_ns": ts_ns,
        "rgb": {"dtype": "uint8", "shape": list(rgb.shape), "order": "RGB", "nbytes": len(rgb_bytes)},
        "depth": {
            "dtype": "float32",
            "shape": [camera.height, camera.width],
            "unit": "m",
            "invalid": "nan",
            "nbytes": len(depth_bytes),
        },
        "intrinsics": camera.intrinsics(),
        "baseline_m": camera.baseline_m,
    }
    hbytes = json.dumps(header).encode("utf-8")
    return _LEN.pack(len(hbytes)) + hbytes + rgb_bytes + depth_bytes


# ─────────────────────────────────────────────────────────────────────────────
# Server
# ─────────────────────────────────────────────────────────────────────────────
class _ZedService:
    def __init__(self, args: argparse.Namespace) -> None:
        self._args = args
        self._camera = _ZedCamera(args)
        self._srv: Optional[socket.socket] = None
        self._stop = threading.Event()

    def _handle_get_frame(self, conn: socket.socket) -> None:
        result = self._camera.grab_rgbd()
        if result is None:
            _send_json(conn, {"v": 1, "ok": False, "error": "grab failed"})
            return
        rgb, depth, ts_ns = result
        _send_payload(conn, _build_frame_payload(self._camera, rgb, depth, ts_ns))

    def _handle_ping(self, conn: socket.socket) -> None:
        _send_json(
            conn,
            {
                "v": 1,
                "ok": True,
                "camera_open": self._camera.width > 0,
                "width": self._camera.width,
                "height": self._camera.height,
                "fps": self._args.fps,
            },
        )

    def _serve_conn(self, conn: socket.socket) -> None:
        try:
            while not self._stop.is_set():
                try:
                    req = _recv_request(conn)
                except (ConnectionError, OSError, ValueError, json.JSONDecodeError):
                    break
                method = req.get("method")
                if method == "get_frame":
                    self._handle_get_frame(conn)
                elif method == "ping":
                    self._handle_ping(conn)
                else:
                    _send_json(conn, {"v": 1, "ok": False, "error": f"unknown method {method!r}"})
        finally:
            conn.close()

    def _self_test(self) -> None:
        """One real grab + depth before exposing the socket (§4)."""
        deadline = time.time() + 30.0
        while time.time() < deadline:
            result = self._camera.grab_rgbd()
            if result is not None:
                rgb, depth, _ = result
                if (
                    rgb.shape == (self._camera.height, self._camera.width, 3)
                    and rgb.dtype == np.uint8
                    and depth.shape == (self._camera.height, self._camera.width)
                    and depth.dtype == np.float32
                ):
                    valid = int(np.isfinite(depth).sum())
                    print(f"[zed_service] self-test ok: depth valid px={valid}", flush=True)
                    return
                raise RuntimeError(
                    f"self-test shape/dtype mismatch: rgb={rgb.shape}/{rgb.dtype} "
                    f"depth={depth.shape}/{depth.dtype}"
                )
            time.sleep(0.05)
        raise RuntimeError("self-test failed: no valid frame within 30s")

    def _bind_socket(self) -> None:
        path = self._args.socket
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        if os.path.exists(path):
            os.unlink(path)
        srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        srv.bind(path)
        srv.listen(8)
        srv.settimeout(1.0)  # so accept() wakes to check the stop flag
        self._srv = srv
        print(f"[zed_service] ready; serving on {path}", flush=True)

    def run(self) -> None:
        self._camera.open()
        self._self_test()  # only past here do we expose the socket → reachable ⇒ valid data
        self._bind_socket()
        try:
            while not self._stop.is_set():
                try:
                    conn, _ = self._srv.accept()
                except socket.timeout:
                    continue
                except OSError:
                    break
                threading.Thread(target=self._serve_conn, args=(conn,), daemon=True).start()
        finally:
            self.shutdown()

    def shutdown(self) -> None:
        self._stop.set()
        if self._srv is not None:
            try:
                self._srv.close()
            except OSError:
                pass
            self._srv = None
        if os.path.exists(self._args.socket):
            try:
                os.unlink(self._args.socket)
            except OSError:
                pass
        self._camera.close()
        print("[zed_service] stopped cleanly.", flush=True)


def _parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--socket", default="/tmp/piper/zed.sock", help="Unix socket path (default: /tmp/piper/zed.sock)")
    p.add_argument("--width", type=int, default=1920, help="capture width (default: 1920)")
    p.add_argument("--height", type=int, default=1080, help="capture height (default: 1080)")
    p.add_argument("--fps", type=int, default=30, help="capture fps (default: 30)")
    p.add_argument("--serial", type=int, default=0, help="ZED serial (default: 0 = first available)")
    p.add_argument("--exposure", type=int, default=-1, help="exposure 0..100, -1 = auto (default)")
    p.add_argument("--gain", type=int, default=-1, help="gain 0..100, -1 = auto (default)")
    p.add_argument("--variant", choices=["c32", "c64"], default="c64", help="TRI-Stereo variant (default: c64)")
    p.add_argument("--weights-dir", default=None, help="override TRI-Stereo weights dir (default: raiden search path)")
    p.add_argument("--open-timeout-sec", type=float, default=15.0, help="per-open() timeout (default: 15)")
    p.add_argument("--open-deadline-sec", type=float, default=180.0, help="total open budget (default: 180)")
    return p.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = _parse_args(argv)
    service = _ZedService(args)

    def _on_signal(_sig, _frame):  # noqa: ANN001
        print("[zed_service] signal received; shutting down...", flush=True)
        service.shutdown()
        sys.exit(0)

    signal.signal(signal.SIGTERM, _on_signal)
    signal.signal(signal.SIGINT, _on_signal)
    service.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
