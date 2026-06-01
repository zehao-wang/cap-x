"""POSIX shared-memory image bus for the Piper state service.

Each camera owns three shared-memory regions:

  * ``<prefix>_<cam>_rgb``   : ``H * W * 3`` uint8 RGB buffer.
  * ``<prefix>_<cam>_depth`` : ``H * W * 1`` float32 depth (metres; NaN = invalid).
                               Allocated even when the camera has no depth — in
                               that case the producer never writes to it and the
                               meta region's ``has_depth`` flag is 0.
  * ``<prefix>_<cam>_meta``  : 32-byte header.

The meta header layout (little-endian):

    offset 0  : uint64  seq            -- monotonic, incremented after each write
    offset 8  : float64 ts             -- producer wallclock when frame was written
    offset 16 : uint8   has_rgb
    offset 17 : uint8   has_depth
    offset 18 : uint8   _reserved[14]

Producers (the service) call :meth:`CameraShm.write` with rgb and optional
depth. Consumers (the client) call :meth:`CameraShm.read_latest` which returns
copies of the latest valid arrays plus the seq it observed.

The protocol-side ``hello`` message advertises shape/dtype/name for each
camera's regions; the client uses :class:`CameraShmReader` to attach.
"""

from __future__ import annotations

import struct
import time
from dataclasses import dataclass
from multiprocessing import shared_memory
from typing import Any

import numpy as np

_META_SIZE = 32
_META_STRUCT = struct.Struct("<QdBB14x")  # seq, ts, has_rgb, has_depth, padding


def _meta_pack(seq: int, ts: float, has_rgb: bool, has_depth: bool) -> bytes:
    return _META_STRUCT.pack(int(seq), float(ts), 1 if has_rgb else 0, 1 if has_depth else 0)


def _meta_unpack(buf: bytes) -> tuple[int, float, bool, bool]:
    seq, ts, has_rgb, has_depth = _META_STRUCT.unpack(buf[:_META_SIZE])
    return int(seq), float(ts), bool(has_rgb), bool(has_depth)


@dataclass
class CameraShmDescriptor:
    """Wire-shareable description of a camera's SHM regions."""

    camera_key: str  # e.g. "robot0_robotview"
    height: int
    width: int
    has_depth: bool
    rgb_name: str
    depth_name: str
    meta_name: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "camera_key": self.camera_key,
            "height": int(self.height),
            "width": int(self.width),
            "has_depth": bool(self.has_depth),
            "rgb_name": self.rgb_name,
            "depth_name": self.depth_name,
            "meta_name": self.meta_name,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "CameraShmDescriptor":
        return cls(
            camera_key=str(d["camera_key"]),
            height=int(d["height"]),
            width=int(d["width"]),
            has_depth=bool(d["has_depth"]),
            rgb_name=str(d["rgb_name"]),
            depth_name=str(d["depth_name"]),
            meta_name=str(d["meta_name"]),
        )


class CameraShmWriter:
    """Service-side producer: allocates the three SHM regions and writes frames."""

    def __init__(
        self,
        *,
        camera_key: str,
        prefix: str,
        height: int,
        width: int,
        has_depth: bool,
    ) -> None:
        self.camera_key = camera_key
        self.height = int(height)
        self.width = int(width)
        self.has_depth = bool(has_depth)
        self._seq = 0

        rgb_bytes = self.height * self.width * 3
        depth_bytes = self.height * self.width * 4  # float32
        rgb_name = f"{prefix}_{camera_key}_rgb"
        depth_name = f"{prefix}_{camera_key}_depth"
        meta_name = f"{prefix}_{camera_key}_meta"

        self._rgb_shm = shared_memory.SharedMemory(
            name=rgb_name, create=True, size=rgb_bytes
        )
        # Always allocate depth so the descriptor is uniform; if the camera
        # doesn't produce depth, the buffer is just never written.
        self._depth_shm = shared_memory.SharedMemory(
            name=depth_name, create=True, size=depth_bytes
        )
        self._meta_shm = shared_memory.SharedMemory(
            name=meta_name, create=True, size=_META_SIZE
        )

        self._rgb_view = np.ndarray(
            (self.height, self.width, 3), dtype=np.uint8, buffer=self._rgb_shm.buf
        )
        self._depth_view = np.ndarray(
            (self.height, self.width), dtype=np.float32, buffer=self._depth_shm.buf
        )

        # Initialize meta to seq=0, has_rgb=0, has_depth=0.
        self._meta_shm.buf[:_META_SIZE] = _meta_pack(0, 0.0, False, False)

        self.descriptor = CameraShmDescriptor(
            camera_key=camera_key,
            height=self.height,
            width=self.width,
            has_depth=self.has_depth,
            rgb_name=rgb_name,
            depth_name=depth_name,
            meta_name=meta_name,
        )

    def write(self, rgb: np.ndarray, depth: np.ndarray | None) -> int:
        """Copy a new frame in and bump seq. Returns the new seq."""
        if rgb.shape[:2] != (self.height, self.width):
            raise ValueError(
                f"rgb shape {rgb.shape} doesn't match SHM "
                f"({self.height}x{self.width}) for {self.camera_key}"
            )
        np.copyto(self._rgb_view, rgb.astype(np.uint8, copy=False))
        has_depth_now = False
        if self.has_depth and depth is not None:
            d = depth
            if d.ndim == 3 and d.shape[-1] == 1:
                d = d[:, :, 0]
            if d.shape != (self.height, self.width):
                raise ValueError(
                    f"depth shape {d.shape} doesn't match SHM "
                    f"({self.height}x{self.width}) for {self.camera_key}"
                )
            np.copyto(self._depth_view, d.astype(np.float32, copy=False))
            has_depth_now = True
        self._seq += 1
        self._meta_shm.buf[:_META_SIZE] = _meta_pack(
            self._seq, time.time(), True, has_depth_now
        )
        return self._seq

    def close(self) -> None:
        for shm in (self._rgb_shm, self._depth_shm, self._meta_shm):
            try:
                shm.close()
                shm.unlink()
            except FileNotFoundError:
                pass
            except Exception:
                pass


class CameraShmReader:
    """Client-side consumer: attaches to a producer's SHM regions."""

    _ATTACH_RETRY_TIMEOUT_S = 5.0
    _ATTACH_RETRY_INTERVAL_S = 0.1

    def __init__(self, descriptor: CameraShmDescriptor) -> None:
        self.descriptor = descriptor
        self._rgb_shm = self._attach_with_retry(descriptor.rgb_name)
        self._depth_shm = self._attach_with_retry(descriptor.depth_name)
        self._meta_shm = self._attach_with_retry(descriptor.meta_name)
        self._rgb_view = np.ndarray(
            (descriptor.height, descriptor.width, 3),
            dtype=np.uint8,
            buffer=self._rgb_shm.buf,
        )
        self._depth_view = np.ndarray(
            (descriptor.height, descriptor.width),
            dtype=np.float32,
            buffer=self._depth_shm.buf,
        )

    @classmethod
    def _attach_with_retry(cls, name: str) -> shared_memory.SharedMemory:
        """Attach to a SHM segment, retrying briefly if it isn't there yet.

        The state service primes SHM before opening the WS port, but a client
        that connects mid-service-restart can race against the writer's
        allocation. A short retry loop closes that window without papering
        over a real misconfiguration.
        """
        deadline = time.monotonic() + cls._ATTACH_RETRY_TIMEOUT_S
        last_exc: Exception | None = None
        attempt = 0
        while True:
            try:
                return shared_memory.SharedMemory(name=name)
            except FileNotFoundError as e:
                last_exc = e
                if time.monotonic() >= deadline:
                    raise
                attempt += 1
                if attempt == 1:
                    print(
                        f"[piper_client] SHM '{name}' not ready yet; "
                        f"retrying for up to {cls._ATTACH_RETRY_TIMEOUT_S:.1f}s..."
                    )
                time.sleep(cls._ATTACH_RETRY_INTERVAL_S)

    def read_latest(self) -> tuple[int, float, np.ndarray | None, np.ndarray | None]:
        """Return (seq, ts, rgb, depth). rgb/depth are None if not yet written.

        We read the meta header twice around the array copy. If the seq advanced
        mid-copy we accept it — at 30 Hz the staleness is bounded and the only
        consumers are visualization + IK warm-starts, neither of which care.
        """
        seq, ts, has_rgb, has_depth = _meta_unpack(bytes(self._meta_shm.buf[:_META_SIZE]))
        if seq == 0:
            return 0, 0.0, None, None
        rgb = self._rgb_view.copy() if has_rgb else None
        depth = self._depth_view.copy()[..., None] if has_depth else None
        return seq, ts, rgb, depth

    def close(self) -> None:
        # Consumers do not unlink — only the producer owns the lifetime.
        for shm in (self._rgb_shm, self._depth_shm, self._meta_shm):
            try:
                shm.close()
            except Exception:
                pass


__all__ = [
    "CameraShmDescriptor",
    "CameraShmReader",
    "CameraShmWriter",
]
