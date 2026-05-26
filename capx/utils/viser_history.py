"""Per-trial observation history with a Viser GUI for scrubbing.

Each simulator owns one ``ViserFrameHistory`` and calls ``record(...)`` once per
visual update with the current cameras + robot joints. This module is the thin
**data + lifecycle** facade:

* it groups frames into **segments** — one per attempt. A fresh trial calls
  ``clear()`` (drop everything); each in-trial reset calls ``new_segment()`` so
  earlier attempts stay browsable;
* it caps each segment at ``max_frames`` and binary-subsamples on overflow so
  coverage stays time-uniform;
* it persists segments via :mod:`capx.utils.viser_history_io` — ``save(path)``
  dumps the latest segment (headless trial saver), ``save_segments(dir)`` dumps
  every attempt as ``attempt_00.npz``, ``attempt_01.npz``, … (interactive web UI).

Everything that touches the viser server — the Attempt/Timestep/Live/Camera/
Reset-View widgets, the concatenated multi-view image, the per-camera frustums
(drawn at each camera's real FOV), the camera parent frames and orbit-camera
snapping — lives in :class:`~capx.utils.viser_playback_panel.ViserPlaybackPanel`.
The facade forwards three notifications to it (recorded / new segment / cleared)
and reads back only ``live``. Scene decorations that aren't observations
(pointclouds, grasps, cube frames) stay with the simulator; they always reflect
"now".
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from capx.utils.viser_history_io import save_frames
from capx.utils.viser_playback_panel import ViserPlaybackPanel


@dataclass
class _CameraSnapshot:
    image: np.ndarray
    pose_xyz_wxyz: Optional[np.ndarray]  # 7-vector [x, y, z, qw, qx, qy, qz] or None
    fov: Optional[float] = None  # vertical field of view in radians, or None


@dataclass
class _Frame:
    step: int
    cameras: dict[str, _CameraSnapshot] = field(default_factory=dict)
    joints: Optional[np.ndarray] = None
    gripper_fraction: Optional[float] = None


class ViserFrameHistory:
    """Records per-step observations; delegates the viser GUI to a panel."""

    def __init__(
        self,
        viser_server,
        urdf_vis=None,
        max_frames: int = 100,
        render_aspect: float = 4.0 / 3.0,
        folder_label: str = "Trial Playback",
    ) -> None:
        self.max_frames = max(2, int(max_frames))

        # Frames grouped into segments (one per attempt). The last segment is
        # the live recording target. Empty until the first record().
        self._segments: list[list[_Frame]] = []

        # The panel owns all viser state and reads our segments via this getter.
        self._panel = ViserPlaybackPanel(
            viser_server,
            lambda: self._segments,
            urdf_vis=urdf_vis,
            max_frames=self.max_frames,
            render_aspect=render_aspect,
            folder_label=folder_label,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def live(self) -> bool:
        """True when playback is locked to the latest frame of the latest
        segment. Simulators gate their own live URDF updates on this."""
        return self._panel.live

    @property
    def num_segments(self) -> int:
        return len(self._segments)

    @property
    def count(self) -> int:
        """Number of frames in the latest (recording) segment."""
        return len(self._segments[-1]) if self._segments else 0

    def clear(self) -> None:
        """Drop every segment (call at the start of a new trial)."""
        self._segments = []
        self._panel.notify_cleared()

    def new_segment(self) -> None:
        """Begin a new attempt segment, retaining the prior ones.

        Called on an in-trial reset (§9.4): the prior attempt stays browsable
        via the Attempt dropdown while recording resumes on a fresh segment. A
        no-op when the latest segment is still empty (avoids blank attempts from
        back-to-back resets).
        """
        if not (self._segments and len(self._segments[-1]) == 0):
            self._segments.append([])
        self._panel.notify_new_segment()

    def save(self, path: str) -> None:
        """Dump the latest segment to ``path`` as a compressed ``.npz``.

        No-op when empty. Used by the headless trial saver (one segment per
        episode); see :func:`capx.utils.viser_history_io.save_frames`.
        """
        if self._segments:
            save_frames(self._segments[-1], path)

    def save_segments(self, directory: str, prefix: str = "attempt") -> list[str]:
        """Dump every non-empty segment in order to ``directory``.

        Files are named ``{prefix}_{i:02d}.npz`` (i = attempt index). Returns
        the paths written. No-op for empty segments.
        """
        import os

        os.makedirs(directory, exist_ok=True)
        written: list[str] = []
        for i, frames in enumerate(self._segments):
            if not frames:
                continue
            path = os.path.join(directory, f"{prefix}_{i:02d}.npz")
            save_frames(frames, path)
            written.append(path)
        return written

    def record(
        self,
        cameras: dict[str, dict],
        joints: Optional[np.ndarray] = None,
        gripper_fraction: Optional[float] = None,
        step: Optional[int] = None,
    ) -> None:
        """Append one observation to the live (latest) segment.

        Args:
            cameras: ``{camera_name: {"image": HxWx3 uint8, "pose_xyz_wxyz":
                7-vec [x, y, z, qw, qx, qy, qz] or None, "fov": vertical FOV in
                radians or None}}``. New camera names extend the dropdown.
            joints: vector accepted by ``urdf_vis.update_cfg`` (caller decides
                whether to append gripper width).
            gripper_fraction: 0..1 gripper opening (informational).
            step: integer label shown next to the slider; defaults to the
                segment's frame count.
        """
        if not self._segments:
            self._segments.append([])
        target = self._segments[-1]
        frame = _Frame(
            step=int(step) if step is not None else len(target),
            cameras={
                name: _CameraSnapshot(
                    image=np.asarray(cam["image"]),
                    pose_xyz_wxyz=(
                        np.asarray(cam["pose_xyz_wxyz"], dtype=np.float64)
                        if cam.get("pose_xyz_wxyz") is not None
                        else None
                    ),
                    fov=float(cam["fov"]) if cam.get("fov") is not None else None,
                )
                for name, cam in cameras.items()
                if cam.get("image") is not None
            },
            joints=None if joints is None else np.asarray(joints, dtype=np.float64),
            gripper_fraction=gripper_fraction,
        )
        target.append(frame)

        # Binary subsample on overflow — halve the segment and keep the latest.
        if len(target) > self.max_frames:
            kept = target[::2]
            if kept[-1] is not target[-1]:
                kept.append(target[-1])
            self._segments[-1] = kept

        # The buffering above is server-independent and must stay intact for
        # save(); the panel handles all GUI/scene work (and going inert when the
        # server is gone).
        self._panel.notify_recorded(frame)


__all__ = ["ViserFrameHistory"]
