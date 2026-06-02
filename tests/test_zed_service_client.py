"""Round-trip test for the ZED service client against a fake Unix-socket server.

Validates the wire format v1 (see piper/ZED_SERVICE_REQUIREMENTS.md §3) end-to-end
without any hardware: a fake server serves one synthetic RGB-D frame; the client must
decode rgb (H,W,3) uint8 RGB, depth (H,W,1) float32 metres with NaN for invalid, and
expose the intrinsics from the header.
"""

import json
import socket
import struct
import threading
import time

import numpy as np

from capx.envs.simulators.piper.zed_service_client import _ZedServiceClient

_LEN = struct.Struct(">I")
_W, _H = 8, 6
_FX, _FY, _CX, _CY = 1050.0, 1050.0, 4.0, 3.0


def _recv_frame(sock):
    (n,) = _LEN.unpack(_recv_exactly(sock, 4))
    return _recv_exactly(sock, n)


def _recv_exactly(sock, n):
    buf = b""
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            raise ConnectionError("closed")
        buf += chunk
    return buf


def _make_frame_payload():
    rgb = np.arange(_H * _W * 3, dtype=np.uint8).reshape(_H, _W, 3)
    depth = np.full((_H, _W), 1.5, dtype=np.float32)
    depth[0, 0] = 0.0  # invalid → NaN
    depth[1, 1] = np.nan  # already invalid
    header = {
        "v": 1, "ok": True, "width": _W, "height": _H, "timestamp_ns": 123,
        "rgb": {"dtype": "uint8", "shape": [_H, _W, 3], "order": "RGB", "nbytes": rgb.nbytes},
        "depth": {"dtype": "float32", "shape": [_H, _W], "unit": "m", "invalid": "nan",
                  "nbytes": depth.nbytes},
        "intrinsics": [[_FX, 0, _CX], [0, _FY, _CY], [0, 0, 1]],
        "baseline_m": 0.12,
    }
    hbytes = json.dumps(header).encode("utf-8")
    return _LEN.pack(len(hbytes)) + hbytes + rgb.tobytes() + depth.tobytes(), rgb, depth


def _serve(sock_path, ready):
    srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    srv.bind(sock_path)
    srv.listen(1)
    ready.set()
    conn, _ = srv.accept()
    try:
        while True:
            try:
                req = json.loads(_recv_frame(conn).decode("utf-8"))
            except (ConnectionError, OSError):
                break
            method = req.get("method")
            if method == "get_frame":
                payload, _, _ = _make_frame_payload()
                conn.sendall(_LEN.pack(len(payload)) + payload)
            else:  # ping / set_video_settings → JSON-only ack
                ack = json.dumps({"v": 1, "ok": True}).encode("utf-8")
                conn.sendall(_LEN.pack(len(ack)) + ack)
    finally:
        conn.close()
        srv.close()


def test_zed_service_client_roundtrip(tmp_path):
    sock_path = str(tmp_path / "zed.sock")
    ready = threading.Event()
    t = threading.Thread(target=_serve, args=(sock_path, ready), daemon=True)
    t.start()
    assert ready.wait(timeout=5.0)

    client = _ZedServiceClient(sock_path, connect_timeout_sec=2.0, heartbeat_sec=0.2)
    client.start()  # heartbeat-waits for the service, then one frame for dims + intrinsics

    assert (client.width, client.height) == (_W, _H)
    np.testing.assert_allclose(
        client.intrinsics_matrix(),
        np.array([[_FX, 0, _CX], [0, _FY, _CY], [0, 0, 1]]),
    )

    rgb, depth = client.read_frames()
    assert rgb is not None and depth is not None
    assert rgb.shape == (_H, _W, 3) and rgb.dtype == np.uint8
    assert depth.shape == (_H, _W, 1) and depth.dtype == np.float32
    # invalid pixels (0 and NaN at source) come back NaN; valid stays 1.5
    assert np.isnan(depth[0, 0, 0])
    assert np.isnan(depth[1, 1, 0])
    assert depth[2, 2, 0] == np.float32(1.5)
    client.stop()


def test_zed_service_client_heartbeat_waits_for_service(tmp_path):
    """start() must heartbeat-wait (not give up) and recover when the service appears late."""
    sock_path = str(tmp_path / "zed.sock")
    result: dict = {}

    def run_client():
        c = _ZedServiceClient(sock_path, connect_timeout_sec=1.0, heartbeat_sec=0.1)
        c.start()  # no server yet → must heartbeat-wait, then recover
        result["wh"] = (c.width, c.height)
        c.stop()

    ct = threading.Thread(target=run_client, daemon=True)
    ct.start()
    time.sleep(0.6)  # client is heartbeat-waiting while no server exists
    assert ct.is_alive(), "client should still be waiting (service not up yet)"

    ready = threading.Event()
    threading.Thread(target=_serve, args=(sock_path, ready), daemon=True).start()
    assert ready.wait(timeout=5.0)

    ct.join(timeout=8.0)
    assert not ct.is_alive(), "client did not recover after the service came up"
    assert result.get("wh") == (_W, _H)
