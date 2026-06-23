"""Tapip3DClient — talk to the TAPIP3D service over its Unix socket.

Self-contained vendored copy: the pure-socket client + its length-prefixed
wire framing merged into one module. Pure stdlib + numpy (no torch), so it
imports cleanly anywhere. Keeps ONE persistent connection.

This is a verbatim merge of TAPIP3D/service/client.py and TAPIP3D/service/wire.py
from the demo-bridge repo, with the wire functions inlined and the sys.path
import shim dropped, so cap-x has no cross-repo import. It talks to a running
tapip3d server over the unix socket.

Wire frame layout (after the outer 4-byte total-length prefix):

    [4B header_len][header JSON utf-8][raw tensor bytes ...]

The header's ``_tensors`` field lists ``{name, dtype, shape}`` in body order; the
receiver rebuilds each array with ``np.frombuffer`` (near zero-copy).
"""

from __future__ import annotations

import json
import socket
import struct
import time
from typing import Any, Dict, Optional, Tuple

import numpy as np

# ── wire framing (inlined from wire.py) ──────────────────────────────────────────
_LEN = struct.Struct(">I")          # 4-byte unsigned length prefix
MAX_FRAME = 512 * 1024 * 1024       # 512 MiB guard
Tensors = Dict[str, np.ndarray]


def _recv_exactly(sock: socket.socket, n: int) -> bytes:
    chunks = []
    got = 0
    while got < n:
        chunk = sock.recv(min(n - got, 1 << 20))
        if not chunk:
            raise ConnectionError("socket closed mid-frame")
        chunks.append(chunk)
        got += len(chunk)
    return b"".join(chunks)


def send_msg(sock: socket.socket, header: Dict[str, Any], tensors: Optional[Tensors] = None) -> None:
    """Send one frame: a JSON ``header`` plus optional named numpy ``tensors``."""
    tensors = tensors or {}
    manifest = []
    buffers = []
    for name, arr in tensors.items():
        a = np.ascontiguousarray(arr)
        manifest.append({"name": name, "dtype": a.dtype.str, "shape": list(a.shape)})
        buffers.append(a.tobytes())
    head = json.dumps({**header, "_tensors": manifest}).encode("utf-8")
    body = b"".join(buffers)
    payload = _LEN.pack(len(head)) + head + body
    if len(payload) > MAX_FRAME:
        raise ValueError(f"frame too large: {len(payload)} bytes")
    sock.sendall(_LEN.pack(len(payload)) + payload)


def recv_msg(sock: socket.socket) -> Tuple[Dict[str, Any], Tensors]:
    """Receive one frame → (header, tensors). Inverse of :func:`send_msg`."""
    (plen,) = _LEN.unpack(_recv_exactly(sock, _LEN.size))
    if plen > MAX_FRAME:
        raise ValueError(f"frame too large: {plen} bytes")
    payload = _recv_exactly(sock, plen)
    (hlen,) = _LEN.unpack(payload[:4])
    header = json.loads(payload[4:4 + hlen].decode("utf-8"))
    body = memoryview(payload)[4 + hlen:]
    tensors: Tensors = {}
    off = 0
    for t in header.pop("_tensors", []):
        shape = tuple(int(s) for s in t["shape"])
        count = int(np.prod(shape)) if shape else 1
        arr = np.frombuffer(body, dtype=np.dtype(t["dtype"]), count=count, offset=off)
        tensors[t["name"]] = arr.reshape(shape)
        off += arr.nbytes
    return header, tensors


# ── client ───────────────────────────────────────────────────────────────────────
class Tapip3DError(RuntimeError):
    pass


class Tapip3DClient:
    def __init__(self, socket_path: str = "/tmp/demo_bridge/sockets/tapip3d.sock",
                 *, connect_timeout: float = 60.0):
        self.socket_path = socket_path
        self._connect_timeout = connect_timeout
        self._sock: Optional[socket.socket] = None

    def _conn(self) -> socket.socket:
        if self._sock is not None:
            return self._sock
        deadline = time.time() + self._connect_timeout
        last: Optional[Exception] = None
        while time.time() < deadline:
            s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            try:
                s.connect(self.socket_path)
                self._sock = s
                return s
            except OSError as e:  # server not up yet
                last = e
                s.close()
                time.sleep(0.2)
        raise Tapip3DError(f"could not connect to {self.socket_path}: {last}")

    def _request(self, method: str, job: Dict[str, Any] | None = None, tensors=None):
        sock = self._conn()
        try:
            send_msg(sock, {"method": method, "job": job or {}}, tensors)
            header, out_tensors = recv_msg(sock)
        except (ConnectionError, BrokenPipeError) as e:
            self.close()
            raise Tapip3DError(f"{method}: connection lost ({e})") from e
        if not header.get("ok", False):
            raise Tapip3DError(f"{method}: {header.get('error')}")
        return header.get("result", {}), out_tensors

    # ── API ──────────────────────────────────────────────────────────────────────
    def ping(self) -> bool:
        res, _ = self._request("ping")
        return bool(res.get("pong"))

    def status(self) -> Dict[str, Any]:
        res, _ = self._request("status")
        return res

    def configure(self, **knobs) -> Dict[str, Any]:
        """Set runtime knobs: resolution_factor, num_iters, vis_threshold, query_grid."""
        res, _ = self._request("configure", {k: v for k, v in knobs.items() if v is not None})
        return res

    def track(self, *, video: np.ndarray, depths: np.ndarray, intrinsics: np.ndarray,
              extrinsics: Optional[np.ndarray] = None, query_point: Optional[np.ndarray] = None,
              num_iters: Optional[int] = None, resolution_factor: Optional[float] = None,
              vis_threshold: Optional[float] = None, query_grid: Optional[int] = None,
              bidirectional: Optional[bool] = None
              ) -> Dict[str, Any]:
        """Track one window of RGB-D frames → {coords [T,N,3], visibs [T,N], info}.

        ``video`` [T,H,W,3] uint8, ``depths`` [T,H,W] float32 (metres),
        ``intrinsics`` [T,3,3] or [3,3]. ``extrinsics`` optional (default identity =
        camera frame). ``query_point`` optional (default = grid on frame 0).
        """
        tensors: Dict[str, np.ndarray] = {
            "video": np.ascontiguousarray(video, dtype=np.uint8),
            "depths": np.ascontiguousarray(depths, dtype=np.float32),
            "intrinsics": np.ascontiguousarray(intrinsics, dtype=np.float32),
        }
        if extrinsics is not None:
            tensors["extrinsics"] = np.ascontiguousarray(extrinsics, dtype=np.float32)
        if query_point is not None:
            tensors["query_point"] = np.ascontiguousarray(query_point, dtype=np.float32)
        job = {k: v for k, v in {
            "num_iters": num_iters, "resolution_factor": resolution_factor,
            "vis_threshold": vis_threshold, "query_grid": query_grid,
            "bidirectional": bidirectional}.items() if v is not None}
        res, out = self._request("track", job, tensors)
        return {"coords": out.get("coords"), "visibs": out.get("visibs"),
                "info": res.get("info", {})}

    def close(self) -> None:
        if self._sock is not None:
            try:
                self._sock.close()
            finally:
                self._sock = None
