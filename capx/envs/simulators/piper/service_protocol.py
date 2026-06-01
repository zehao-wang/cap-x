"""Wire protocol for the Piper state service.

Frames are msgpack dicts sent over a single websocket connection. There are
three message families:

  * ``hello``        : server -> client on connect; describes available cameras
                      and the names of the POSIX SHM regions backing each
                      camera's RGB / depth frames.
  * ``state``        : server -> client; periodic snapshot of joints, gripper,
                      pose matrices, intrinsics, and the latest SHM frame seq
                      counters per camera. Image arrays are NOT inlined — the
                      client reads them from SHM.
  * ``rpc`` / ``rpc_reply`` : client <-> server; request/response with a
                      monotonically increasing ``req_id``.

A client may also send ``set`` messages to update overlay state on the server
(e.g. ``cube_center`` / ``grasp_sample`` for the viser scene running on the
service side).

The protocol is intentionally schema-light: msgpack dicts with stable string
keys. Add fields freely; older clients ignore unknown keys.
"""

from __future__ import annotations

import itertools
from typing import Any

import msgpack
import msgpack_numpy

# Register numpy hooks so np.ndarray serializes/deserializes transparently.
# Used for poses, intrinsics, joint vectors — but NOT for camera images, which
# always go through SHM.
msgpack_numpy.patch()

PROTOCOL_VERSION = 1

# Op names for top-level messages.
OP_HELLO = "hello"
OP_STATE = "state"
OP_RPC = "rpc"
OP_RPC_REPLY = "rpc_reply"
OP_SET = "set"

# RPC method names. Adding a new method? Add it here so both sides agree.
RPC_GET_OBSERVATION = "get_observation"
RPC_EXECUTE_JOINT_TRAJECTORY = "execute_joint_trajectory"
RPC_MOVE_TO_JOINTS_BLOCKING = "move_to_joints_blocking"
RPC_GOTO_HOME_BLOCKING = "goto_home_blocking"
RPC_SET_GRIPPER = "set_gripper"
RPC_STEP_ONCE = "step_once"
RPC_REFRESH_POINT_CLOUD = "refresh_point_cloud"
RPC_UPDATE_VISER_SERVER = "update_viser_server"
RPC_RESET = "reset"

ALL_RPC_METHODS = frozenset(
    {
        RPC_GET_OBSERVATION,
        RPC_EXECUTE_JOINT_TRAJECTORY,
        RPC_MOVE_TO_JOINTS_BLOCKING,
        RPC_GOTO_HOME_BLOCKING,
        RPC_SET_GRIPPER,
        RPC_STEP_ONCE,
        RPC_REFRESH_POINT_CLOUD,
        RPC_UPDATE_VISER_SERVER,
        RPC_RESET,
    }
)

# Overlay keys settable via OP_SET. The service applies them via setattr on its
# wrapped low-level env so the existing PiperViserMixin renders them.
SETTABLE_OVERLAY_KEYS = frozenset(
    {
        "cube_center",
        "cube_rot",
        "cube_points",
        "cube_color",
        "grasp_sample",
        "grasp_scores",
        "grasp_contact_pts",
        "grasp_last_object_name",
        "last_ik_target",
        "preview_waypoints",
    }
)


def encode(message: dict[str, Any]) -> bytes:
    """Pack a message dict into a msgpack frame."""
    return msgpack.packb(message, use_bin_type=True)


def decode(frame: bytes) -> dict[str, Any]:
    """Unpack a msgpack frame into a message dict."""
    return msgpack.unpackb(frame, raw=False)


class ReqIdAllocator:
    """Monotonic request-id allocator (single-threaded callers)."""

    def __init__(self) -> None:
        self._counter = itertools.count(1)

    def next_id(self) -> int:
        return next(self._counter)


__all__ = [
    "ALL_RPC_METHODS",
    "OP_HELLO",
    "OP_RPC",
    "OP_RPC_REPLY",
    "OP_SET",
    "OP_STATE",
    "PROTOCOL_VERSION",
    "RPC_EXECUTE_JOINT_TRAJECTORY",
    "RPC_GET_OBSERVATION",
    "RPC_GOTO_HOME_BLOCKING",
    "RPC_MOVE_TO_JOINTS_BLOCKING",
    "RPC_REFRESH_POINT_CLOUD",
    "RPC_RESET",
    "RPC_SET_GRIPPER",
    "RPC_STEP_ONCE",
    "RPC_UPDATE_VISER_SERVER",
    "ReqIdAllocator",
    "SETTABLE_OVERLAY_KEYS",
    "decode",
    "encode",
]
