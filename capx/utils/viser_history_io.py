"""Persistence for :class:`~capx.utils.viser_history.ViserFrameHistory`.

A single free function serializes a list of frames to a compressed ``.npz``.
Kept separate from the GUI so the on-disk layout has one obvious home and the
main module stays focused on recording + the viser panel. Frames are duck-typed
(``.step``, ``.cameras``, ``.joints``, ``.gripper_fraction``; each camera snap
exposing ``.image``, ``.pose_xyz_wxyz``, ``.fov``) so this module imports
nothing from the history.
"""

from __future__ import annotations

import os
from typing import Sequence

import numpy as np


def save_frames(frames: Sequence, path: str) -> None:
    """Serialize ``frames`` to ``path`` as a compressed ``.npz`` file.

    Layout (all arrays length N == number of frames):

    * ``steps``: int64, the per-frame step labels.
    * ``camera_names``: 1-D str array — the cameras observed.
    * ``images_{name}``: uint8 ``(N, H, W, 3)``; zero-filled where missing.
    * ``poses_{name}``: float64 ``(N, 7)`` ``[x, y, z, qw, qx, qy, qz]``;
      NaN where the simulator did not provide a pose for that frame.
    * ``fov_{name}``: float64 ``(N,)`` vertical FOV (radians); NaN where
      not provided.
    * ``joints``: float64 ``(N, J)`` or shape ``(0,)`` if none; NaN where
      not provided for a frame.
    * ``gripper_fraction``: float64 ``(N,)``; NaN where not provided.

    No-op when ``frames`` is empty. Creates parent directories as needed.
    """
    if not frames:
        return
    if os.path.dirname(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)

    n = len(frames)
    steps = np.asarray([f.step for f in frames], dtype=np.int64)

    cameras_seen: list[str] = []
    for f in frames:
        for name in f.cameras:
            if name not in cameras_seen:
                cameras_seen.append(name)

    out: dict[str, np.ndarray] = {
        "steps": steps,
        "camera_names": np.asarray(cameras_seen, dtype=object),
    }

    for cam in cameras_seen:
        ref_image = next(
            (f.cameras[cam].image for f in frames if cam in f.cameras),
            None,
        )
        if ref_image is None:
            continue
        img_shape = ref_image.shape
        images = np.zeros((n, *img_shape), dtype=np.uint8)
        poses = np.full((n, 7), np.nan, dtype=np.float64)
        fovs = np.full((n,), np.nan, dtype=np.float64)
        for i, f in enumerate(frames):
            snap = f.cameras.get(cam)
            if snap is None:
                continue
            if snap.image.shape == img_shape:
                images[i] = snap.image
            if snap.pose_xyz_wxyz is not None:
                poses[i] = snap.pose_xyz_wxyz
            if snap.fov is not None:
                fovs[i] = snap.fov
        out[f"images_{cam}"] = images
        out[f"poses_{cam}"] = poses
        out[f"fov_{cam}"] = fovs

    joint_widths = {f.joints.shape[0] for f in frames if f.joints is not None}
    if joint_widths:
        j = max(joint_widths)
        joints = np.full((n, j), np.nan, dtype=np.float64)
        for i, f in enumerate(frames):
            if f.joints is not None and f.joints.shape[0] == j:
                joints[i] = f.joints
        out["joints"] = joints
    else:
        out["joints"] = np.zeros((0,), dtype=np.float64)

    out["gripper_fraction"] = np.asarray(
        [np.nan if f.gripper_fraction is None else f.gripper_fraction for f in frames],
        dtype=np.float64,
    )

    np.savez_compressed(path, **out)


__all__ = ["save_frames"]
