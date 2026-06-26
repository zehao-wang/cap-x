"""Tiny self-contained client for the GraspGenX Unix-socket service (NVlabs GraspGenX).

The service (its own uv venv) hosts 6-DOF grasp inference: send an object point cloud
(N,3) and get back grasps (K,4,4) + confidences IN THE SAME FRAME as the input cloud.
Wire format mirrors the service's wire.py: an outer 4-byte length prefix, then
[4B header_len][header JSON][raw tensor bytes]; the header's _tensors lists the body
tensors. Torch-free -- runs in the cap-x solve venv.

Start the service (GraspGenX repo, its venv):
    .venv/bin/python -m service.server --default_gripper franka_panda
"""
from __future__ import annotations

import json
import socket
import struct

import numpy as np

# Private socket for the cap-x dogfooding service (a parallel session also runs a GraspGenX
# service on the shared grasp_gen.sock — keep ours separate to avoid racing the same socket).
SOCKET_PATH = "/tmp/demo_bridge/sockets/grasp_gen_capx.sock"
_LEN = struct.Struct(">I")


def _recv_exactly(sock, n):
    chunks, got = [], 0
    while got < n:
        c = sock.recv(min(n - got, 1 << 20))
        if not c:
            raise ConnectionError("socket closed mid-frame")
        chunks.append(c); got += len(c)
    return b"".join(chunks)


def _send(sock, header, tensors):
    manifest, buffers = [], []
    for name, arr in (tensors or {}).items():
        a = np.ascontiguousarray(arr)
        manifest.append({"name": name, "dtype": a.dtype.str, "shape": list(a.shape)})
        buffers.append(a.tobytes())
    head = json.dumps({**header, "_tensors": manifest}).encode("utf-8")
    payload = _LEN.pack(len(head)) + head + b"".join(buffers)
    sock.sendall(_LEN.pack(len(payload)) + payload)


def _recv(sock):
    (plen,) = _LEN.unpack(_recv_exactly(sock, _LEN.size))
    payload = _recv_exactly(sock, plen)
    (hlen,) = _LEN.unpack(payload[:4])
    header = json.loads(payload[4:4 + hlen].decode("utf-8"))
    body = memoryview(payload)[4 + hlen:]
    tensors, off = {}, 0
    for t in header.pop("_tensors", []):
        shape = tuple(int(s) for s in t["shape"])
        count = int(np.prod(shape)) if shape else 1
        arr = np.frombuffer(body, dtype=np.dtype(t["dtype"]), count=count, offset=off)
        tensors[t["name"]] = arr.reshape(shape); off += arr.nbytes
    return header, tensors


def graspgenx_grasps(pc, *, gripper="franka_panda", num_grasps=200, topk=50,
                     grasp_threshold=-1.0, socket_path=SOCKET_PATH, timeout=60.0):
    """Return (grasps (K,4,4) float32, confidences (K,) float32) for object cloud ``pc`` (N,3),
    in the SAME frame as ``pc``. Raises on service error / no connection."""
    pc = np.asarray(pc, dtype=np.float32)
    if pc.ndim != 2 or pc.shape[1] != 3:
        raise ValueError(f"pc must be (N,3); got {pc.shape}")
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    sock.settimeout(timeout)
    sock.connect(socket_path)
    try:
        _send(sock, {"method": "infer",
                     "job": {"gripper_name": gripper, "num_grasps": int(num_grasps),
                             "topk_num_grasps": int(topk), "grasp_threshold": float(grasp_threshold)}},
              {"point_cloud": pc})
        header, tensors = _recv(sock)
    finally:
        sock.close()
    if header.get("ok") is False:
        raise RuntimeError(f"GraspGenX service error: {header.get('error')}")
    g = tensors.get("grasps", np.zeros((0, 4, 4), np.float32))
    c = tensors.get("confidences", np.zeros((0,), np.float32))
    return g, c


def ping(socket_path=SOCKET_PATH, timeout=5.0):
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    sock.settimeout(timeout)
    sock.connect(socket_path)
    try:
        _send(sock, {"method": "ping"}, {})
        header, _ = _recv(sock)
    finally:
        sock.close()
    return header


if __name__ == "__main__":
    print("ping:", ping())
