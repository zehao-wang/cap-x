"""Thin read-only client for the external ZED camera service (Piper real robot).

Per GOAL.md §5: the ZED 2i runs as a **standalone service** that owns the camera and
does all depth compute (e.g. TRI-Stereo). cap-x does **not** care how the camera is
brought up — it only accesses data over a fixed protocol. This side owns *harness
access behavior only*; camera readiness is the service's responsibility (it runs a
startup read self-test and only then reports "up", so a reachable socket means data is
guaranteed — see ZED_SERVICE_REQUIREMENTS.md §0/§3).

Access policy: **no give-up timeout.** If the service is unreachable (at startup or if
it drops mid-run), the client **heartbeat-waits until it comes back**, printing an
on-screen warning with the current fail reason each beat. So reads block through an
outage rather than failing the agent.

Drop-in for ``_Zed2iBridge`` on the observation path:
``start() / read_frames() -> (rgb, depth) / intrinsics_matrix() / stop()``.

Wire format v1 (ZED_SERVICE_REQUIREMENTS.md §3): Unix domain socket, every message is
``[4B big-endian length][payload]``. Requests are UTF-8 JSON. A ``get_frame`` reply
payload is ``[4B header-len][JSON header][rgb raw bytes][depth raw bytes]``; depth is
float32 metres with NaN = invalid; rgb is uint8 (H, W, 3) in RGB order.
"""

from __future__ import annotations

import json
import socket
import struct
import time
from typing import Any

import numpy as np

_LEN = struct.Struct(">I")  # 4-byte big-endian length prefix
_MAX_FRAME = 256 * 1024 * 1024  # 256 MiB guard

# Transport-level failures that mean "service unreachable / bad frame" → heartbeat-wait.
_TRANSPORT_ERRORS = (
    OSError,
    ConnectionError,
    ValueError,
    struct.error,
    json.JSONDecodeError,
    KeyError,
)


def _recv_exactly(sock: socket.socket, n: int) -> bytes:
    chunks: list[bytes] = []
    got = 0
    while got < n:
        chunk = sock.recv(min(n - got, 1 << 20))
        if not chunk:
            raise ConnectionError("ZED service socket closed mid-frame")
        chunks.append(chunk)
        got += len(chunk)
    return b"".join(chunks)


def _send_frame(sock: socket.socket, obj: dict[str, Any]) -> None:
    payload = json.dumps(obj).encode("utf-8")
    sock.sendall(_LEN.pack(len(payload)) + payload)


def _recv_frame(sock: socket.socket) -> bytes:
    (n,) = _LEN.unpack(_recv_exactly(sock, 4))
    if n > _MAX_FRAME:
        raise ValueError(f"ZED service frame too large: {n} bytes")
    return _recv_exactly(sock, n)


class _ZedServiceClient:
    """Read-only Unix-socket client to the ZED camera service (heartbeat-waits on outage)."""

    def __init__(
        self,
        socket_path: str,
        *,
        connect_timeout_sec: float = 15.0,
        heartbeat_sec: float = 3.0,
    ) -> None:
        self.socket_path = socket_path
        self.connect_timeout_sec = float(connect_timeout_sec)
        self.heartbeat_sec = float(heartbeat_sec)
        self._sock: socket.socket | None = None
        self._intrinsics: np.ndarray | None = None
        self.width = 0
        self.height = 0

    # ── Connection (harness access behaviour) ───────────────────────────────────
    def _open_sock(self) -> socket.socket:
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        sock.settimeout(self.connect_timeout_sec)
        sock.connect(self.socket_path)
        self._sock = sock
        return sock

    def _close_sock(self) -> None:
        if self._sock is not None:
            try:
                self._sock.close()
            except Exception:
                pass
            self._sock = None

    def _warn(self, reason: str) -> None:
        print(
            f"[piper_real] ⚠ ZED camera service unavailable — please check the camera "
            f"service status. Reason: {reason} (socket {self.socket_path}). "
            f"Waiting {self.heartbeat_sec:.0f}s, will keep retrying...",
            flush=True,
        )

    # ── Frames (read-only, heartbeat-wait on outage) ────────────────────────────
    def _get_frame_once(self) -> tuple[np.ndarray | None, np.ndarray | None]:
        """Single attempt: connect if needed, request a frame, parse. Raises on transport error."""
        sock = self._sock or self._open_sock()
        _send_frame(sock, {"v": 1, "method": "get_frame"})
        payload = _recv_frame(sock)
        (hlen,) = _LEN.unpack(payload[:4])
        header = json.loads(payload[4 : 4 + hlen].decode("utf-8"))
        if not header.get("ok", False):
            return None, None  # service up but signalling not-ready; skip this frame

        off = 4 + hlen
        rgb_meta = header["rgb"]
        depth_meta = header["depth"]
        rgb_n = int(rgb_meta["nbytes"])
        depth_n = int(depth_meta["nbytes"])
        rgb = np.frombuffer(payload[off : off + rgb_n], dtype=np.uint8).reshape(
            rgb_meta["shape"]
        )
        off += rgb_n
        depth = np.frombuffer(payload[off : off + depth_n], dtype=np.float32).reshape(
            depth_meta["shape"]
        )

        self.width = int(header["width"])
        self.height = int(header["height"])
        self._intrinsics = np.asarray(header["intrinsics"], dtype=np.float64)

        # Match the obs contract used downstream: depth is (H, W, 1) metres, NaN invalid.
        depth_m = depth.astype(np.float32, copy=True)
        depth_m[~np.isfinite(depth_m)] = np.nan
        depth_m[depth_m <= 0.0] = np.nan
        return rgb.copy(), depth_m[:, :, None]

    def read_frames(self) -> tuple[np.ndarray | None, np.ndarray | None]:
        """One synchronized RGB-D frame. Heartbeat-waits (forever) through an outage.

        Returns (None, None) only when the service is reachable but signals not-ready
        (lets the obs loop skip a frame); on an unreachable service it keeps retrying.
        """
        while True:
            try:
                return self._get_frame_once()
            except _TRANSPORT_ERRORS as e:
                self._close_sock()
                self._warn(f"{type(e).__name__}: {e}")
                time.sleep(self.heartbeat_sec)

    def start(self) -> None:
        """Wait for the service and read one frame to populate dims + intrinsics.

        No give-up: ``read_frames`` heartbeat-waits until the service is reachable. The
        service guarantees readiness via its own startup read self-test, so once a frame
        arrives we have valid intrinsics.
        """
        while True:
            rgb, _depth = self.read_frames()
            if rgb is not None and self._intrinsics is not None:
                print(
                    f"[piper_real] ZED service ready at {self.width}x{self.height} "
                    f"(socket {self.socket_path})",
                    flush=True,
                )
                return
            # Reachable but not-ready (ok:false): per the readiness contract this
            # shouldn't persist; warn and keep waiting.
            self._warn("service connected but returned no frame (camera not ready?)")
            time.sleep(self.heartbeat_sec)

    def stop(self) -> None:
        self._close_sock()

    def intrinsics_matrix(self) -> np.ndarray:
        if self._intrinsics is None:
            raise RuntimeError("intrinsics not available; call start() first")
        return self._intrinsics


__all__ = ["_ZedServiceClient"]
