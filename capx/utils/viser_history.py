"""Per-trial observation history with a Viser GUI for scrubbing.

Each simulator owns one ``ViserFrameHistory``. The simulator calls
``record(...)`` once per visual update with the current cameras + robot joints.
The helper:

* caps the buffer at ``max_frames`` (default 100) and binary-subsamples on
  overflow so coverage stays time-uniform;
* exposes a Viser GUI panel with a timestep slider, a Live toggle, a Step
  label, and a Camera dropdown auto-populated from the cameras the simulator
  passes in;
* owns the on-scene image + camera-frustum handles and updates them to the
  selected frame (latest when Live, or the slider position when scrubbing);
* drives ``urdf_vis.update_cfg(...)`` only while Live is on, so scrubbing
  freezes the robot pose. Callers should gate their own live URDF updates on
  ``frame_history.live`` to avoid fighting the scrubbed frame;
* snaps newly-connecting clients to the first observation-camera pose it
  ever sees, and exposes a "Reset View" button that returns there;
* owns the scene-tree parent frame ``{camera_name}`` at the live camera pose
  so any scene children the simulator adds under that path (e.g.
  ``"{camera_name}/point_cloud"`` for depth-derived clouds in camera-local
  coords) land in world coordinates aligned with the frustum and the robot.
  The parent is only refreshed on ``record(...)`` — scrubbing the slider
  does not move it, so live point clouds stay anchored to the camera pose
  that generated them.

Scene decorations that aren't part of an observation (pointclouds, grasps,
cube frames) are left to the simulator — they always reflect "now", which is
acceptable because they're rare and tied to specific decision points.

Persistence: ``save(path)`` dumps the buffer to a single compressed ``.npz``
file. Trial-end save is wired up in ``capx/envs/trial.py``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass
class _CameraSnapshot:
    image: np.ndarray
    pose_xyz_wxyz: Optional[np.ndarray]  # 7-vector [x, y, z, qw, qx, qy, qz] or None


@dataclass
class _Frame:
    step: int
    cameras: dict[str, _CameraSnapshot] = field(default_factory=dict)
    joints: Optional[np.ndarray] = None
    gripper_fraction: Optional[float] = None


class ViserFrameHistory:
    """Records per-step observations and exposes a Viser GUI for scrubbing."""

    # Distance between camera position and look_at point when snapping a
    # client's view to a stored pose. Affects orbit-controls feel only,
    # not the resulting rotation.
    _VIEW_DISTANCE: float = 0.6

    def __init__(
        self,
        viser_server,
        urdf_vis=None,
        max_frames: int = 100,
        render_aspect: float = 4.0 / 3.0,
        folder_label: str = "Trial Playback",
    ) -> None:
        self.server = viser_server
        self.urdf_vis = urdf_vis
        self.max_frames = max(2, int(max_frames))
        self._render_aspect = float(render_aspect)
        self._folder_label = folder_label

        self._frames: list[_Frame] = []
        self._live: bool = True
        self._active_camera: Optional[str] = None
        self._known_cameras: list[str] = []
        self._suppress_callbacks: bool = False  # block re-entry while we set widgets

        # GUI + scene handles, all created lazily so we don't pollute the
        # viewer until the simulator actually records something.
        self._gui_ready = False
        self._slider = None
        self._live_toggle = None
        self._camera_dropdown = None
        self._step_label = None
        self._image_handle = None  # single GUI image, swapped to active camera
        self._frustum_handles: dict[str, object] = {}
        # Parent scene-tree frames keyed by camera name. Children added by the
        # simulator under "{name}/..." (most importantly the depth-derived
        # point clouds) get transformed by this frame, so they line up with
        # the frustum, the robot URDF and the Reset View pose.
        self._camera_parent_handles: dict[str, object] = {}
        self._reset_button = None

        # First pose we ever see per camera name — the "default view" the
        # client should be snapped to on connect and on Reset View. Stored as
        # 7-vec [x, y, z, qw, qx, qy, qz], OpenCV convention (+Z forward, +Y
        # down) — same layout the simulators hand us under the "pose_xyz_wxyz"
        # key in record().
        self._initial_view_poses: dict[str, np.ndarray] = {}

        # Snap newly-connecting clients to the active observation camera's
        # init pose. viser fires this callback after the client sends its
        # first camera state, so setters on client.camera are safe.
        self.server.on_client_connect(self._on_client_connect)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def live(self) -> bool:
        """True when the slider is locked to the latest frame.

        Simulators should consult this in tight live-update paths (e.g.
        ``_update_viser_robot_only``) and skip URDF updates while False so
        the scrubbed pose isn't immediately overwritten.
        """
        return self._live

    @property
    def count(self) -> int:
        return len(self._frames)

    def clear(self) -> None:
        """Reset the buffer (call at the start of a new trial)."""
        self._frames.clear()
        # Don't tear down GUI/scene handles — reusing them across trials avoids
        # widget flicker. Reset slider to 0 and snap back to Live.
        if self._gui_ready:
            self._suppress_callbacks = True
            try:
                if self._slider is not None:
                    self._slider.max = 0
                    self._slider.value = 0
                if self._live_toggle is not None:
                    self._live_toggle.value = True
                if self._step_label is not None:
                    self._step_label.value = "0 / 0"
            finally:
                self._suppress_callbacks = False
        self._live = True

    def save(self, path: str) -> None:
        """Dump the current buffer to ``path`` as a compressed ``.npz`` file.

        Layout (all arrays length N == number of frames currently held):

        * ``steps``: int64, the per-frame step labels.
        * ``camera_names``: 1-D str array — the cameras observed in the trial.
          Frames where a particular camera's image was missing get a uint8
          zero-filled image of the trial's dominant resolution; pose entries
          fall back to NaN.
        * ``images_{name}``: uint8 ``(N, H, W, 3)``.
        * ``poses_{name}``: float64 ``(N, 7)`` ``[x, y, z, qw, qx, qy, qz]``;
          NaN where the simulator did not provide a pose for that frame.
        * ``joints``: float64 ``(N, J)`` or shape ``(0,)`` if no joints were
          ever recorded; NaN where joints were not provided for a frame.
        * ``gripper_fraction``: float64 ``(N,)``; NaN where not provided.

        No-op when the buffer is empty. Creates parent directories as needed.
        """
        if not self._frames:
            return
        import os

        os.makedirs(os.path.dirname(path), exist_ok=True) if os.path.dirname(path) else None

        n = len(self._frames)
        steps = np.asarray([f.step for f in self._frames], dtype=np.int64)

        # Determine the union of camera names seen across the buffer; for any
        # given camera infer the dominant image shape and the joint width.
        cameras_seen: list[str] = []
        for f in self._frames:
            for name in f.cameras:
                if name not in cameras_seen:
                    cameras_seen.append(name)

        out: dict[str, np.ndarray] = {
            "steps": steps,
            "camera_names": np.asarray(cameras_seen, dtype=object),
        }

        for cam in cameras_seen:
            ref_image = next(
                (f.cameras[cam].image for f in self._frames if cam in f.cameras),
                None,
            )
            if ref_image is None:
                continue
            img_shape = ref_image.shape
            images = np.zeros((n, *img_shape), dtype=np.uint8)
            poses = np.full((n, 7), np.nan, dtype=np.float64)
            for i, f in enumerate(self._frames):
                snap = f.cameras.get(cam)
                if snap is None:
                    continue
                if snap.image.shape == img_shape:
                    images[i] = snap.image
                if snap.pose_xyz_wxyz is not None:
                    poses[i] = snap.pose_xyz_wxyz
            out[f"images_{cam}"] = images
            out[f"poses_{cam}"] = poses

        joint_widths = {f.joints.shape[0] for f in self._frames if f.joints is not None}
        if joint_widths:
            j = max(joint_widths)
            joints = np.full((n, j), np.nan, dtype=np.float64)
            for i, f in enumerate(self._frames):
                if f.joints is not None and f.joints.shape[0] == j:
                    joints[i] = f.joints
            out["joints"] = joints
        else:
            out["joints"] = np.zeros((0,), dtype=np.float64)

        out["gripper_fraction"] = np.asarray(
            [np.nan if f.gripper_fraction is None else f.gripper_fraction for f in self._frames],
            dtype=np.float64,
        )

        np.savez_compressed(path, **out)

    def record(
        self,
        cameras: dict[str, dict],
        joints: Optional[np.ndarray] = None,
        gripper_fraction: Optional[float] = None,
        step: Optional[int] = None,
    ) -> None:
        """Append one observation.

        Args:
            cameras: ``{camera_name: {"image": HxWx3 uint8, "pose_xyz_wxyz":
                7-vec [x, y, z, qw, qx, qy, qz] or None}}``. New camera names
                extend the dropdown.
            joints: vector accepted by ``urdf_vis.update_cfg`` (caller decides
                whether to append gripper width).
            gripper_fraction: 0..1 gripper opening (informational, not used
                for URDF — caller bakes that into ``joints``).
            step: integer label shown next to the slider; defaults to buffer
                length.
        """
        frame = _Frame(
            step=int(step) if step is not None else len(self._frames),
            cameras={
                name: _CameraSnapshot(
                    image=np.asarray(cam["image"]),
                    pose_xyz_wxyz=(
                        np.asarray(cam["pose_xyz_wxyz"], dtype=np.float64)
                        if cam.get("pose_xyz_wxyz") is not None
                        else None
                    ),
                )
                for name, cam in cameras.items()
                if cam.get("image") is not None
            },
            joints=None if joints is None else np.asarray(joints, dtype=np.float64),
            gripper_fraction=gripper_fraction,
        )
        self._frames.append(frame)

        # Binary subsample on overflow — halve the buffer and keep the latest.
        if len(self._frames) > self.max_frames:
            kept = self._frames[::2]
            if kept[-1] is not self._frames[-1]:
                kept.append(self._frames[-1])
            self._frames = kept

        self._ensure_gui()
        self._sync_cameras(frame)
        self._refresh_slider_bounds()

        # Update the live camera parent frames so any scene children the
        # simulator added (or is about to add) under "{name}/..." reflect the
        # current camera pose. Intentionally outside ``_render_frame`` so
        # scrubbing the slider does NOT move parent frames, keeping any
        # already-published live point cloud anchored to its capture pose.
        for cam_name, snap in frame.cameras.items():
            if snap.pose_xyz_wxyz is not None:
                self._update_camera_parent(cam_name, snap.pose_xyz_wxyz)

        if self._live:
            self._render_frame(self._frames[-1])

    # ------------------------------------------------------------------
    # GUI plumbing
    # ------------------------------------------------------------------

    def _ensure_gui(self) -> None:
        """Create the slider/Live/label widgets. The Camera dropdown is built
        lazily by ``_sync_cameras`` once we know at least one camera name —
        viser's ``add_dropdown`` reads ``options[0]`` for its initial value
        and crashes on an empty tuple.
        """
        if self._gui_ready:
            return
        self._gui_folder = self.server.gui.add_folder(self._folder_label)
        with self._gui_folder:
            # max must be > min at creation: viser clamps `step` to
            # `min(step, max - min)` internally (see _gui_api.add_slider), and
            # only the `max` prop is re-sent on later updates. A degenerate
            # max=0 here would bake step=0 in the frontend and dragging would
            # divide by zero, emitting NaN. Use max_frames-1 so step stays 1.
            self._slider = self.server.gui.add_slider(
                "Timestep",
                min=0,
                max=max(1, self.max_frames - 1),
                step=1,
                initial_value=0,
            )
            self._step_label = self.server.gui.add_text("Step", initial_value="0 / 0")
            self._step_label.disabled = True
            self._live_toggle = self.server.gui.add_checkbox("Live", initial_value=True)
            self._reset_button = self.server.gui.add_button("Reset View")

        self._slider.on_update(self._on_slider_change)
        self._live_toggle.on_update(self._on_live_toggle)
        self._reset_button.on_click(self._on_reset_view)
        self._gui_ready = True

    def _sync_cameras(self, frame: _Frame) -> None:
        """Extend (or first-time create) the camera dropdown."""
        new = [c for c in frame.cameras.keys() if c not in self._known_cameras]
        if not new:
            return
        self._known_cameras.extend(new)

        self._suppress_callbacks = True
        try:
            if self._camera_dropdown is None:
                # First camera ever — create the widget now with a valid initial value.
                with self._gui_folder:
                    self._camera_dropdown = self.server.gui.add_dropdown(
                        "Camera",
                        options=tuple(self._known_cameras),
                        initial_value=self._known_cameras[0],
                    )
                self._active_camera = self._known_cameras[0]
                self._camera_dropdown.on_update(self._on_camera_change)
            else:
                self._camera_dropdown.options = tuple(self._known_cameras)
                if self._active_camera is None:
                    self._active_camera = self._known_cameras[0]
                    self._camera_dropdown.value = self._active_camera
        finally:
            self._suppress_callbacks = False

    def _refresh_slider_bounds(self) -> None:
        new_max = max(0, len(self._frames) - 1)
        self._suppress_callbacks = True
        try:
            self._slider.max = new_max
            if self._live:
                self._slider.value = new_max
            self._step_label.value = f"{self._slider.value} / {new_max}"
        finally:
            self._suppress_callbacks = False

    # ------------------------------------------------------------------
    # Callbacks
    # ------------------------------------------------------------------

    def _on_slider_change(self, _event) -> None:
        if self._suppress_callbacks or not self._frames:
            return
        idx = int(self._slider.value)
        idx = max(0, min(idx, len(self._frames) - 1))
        # Any manual slider move turns Live off so the next record() doesn't
        # immediately drag us back to the latest frame.
        if idx != len(self._frames) - 1 and self._live:
            self._live = False
            self._suppress_callbacks = True
            try:
                self._live_toggle.value = False
            finally:
                self._suppress_callbacks = False
        self._step_label.value = f"{idx} / {len(self._frames) - 1}"
        self._render_frame(self._frames[idx])

    def _on_camera_change(self, _event) -> None:
        if self._suppress_callbacks or not self._frames:
            return
        self._active_camera = self._camera_dropdown.value
        idx = int(self._slider.value)
        idx = max(0, min(idx, len(self._frames) - 1))
        self._render_frame(self._frames[idx])

    def _on_live_toggle(self, _event) -> None:
        if self._suppress_callbacks:
            return
        self._live = bool(self._live_toggle.value)
        if self._live and self._frames:
            last = len(self._frames) - 1
            self._suppress_callbacks = True
            try:
                self._slider.value = last
                self._step_label.value = f"{last} / {last}"
            finally:
                self._suppress_callbacks = False
            self._render_frame(self._frames[-1])

    def _on_client_connect(self, client) -> None:
        pose = self._active_initial_pose()
        if pose is None:
            return
        try:
            self._apply_pose_to_client(client, pose)
        except Exception:
            # Don't let a display-side issue tear the client's connect path.
            pass

    def _on_reset_view(self, _event) -> None:
        pose = self._active_initial_pose()
        if pose is None:
            return
        for client in self.server.get_clients().values():
            try:
                self._apply_pose_to_client(client, pose)
            except Exception:
                pass

    def _active_initial_pose(self) -> Optional[np.ndarray]:
        if self._active_camera is None:
            return None
        return self._initial_view_poses.get(self._active_camera)

    @staticmethod
    def _wxyz_to_rotation_matrix(q: np.ndarray) -> np.ndarray:
        w, x, y, z = (float(v) for v in q)
        return np.array(
            [
                [1 - 2 * (y * y + z * z), 2 * (x * y - z * w),     2 * (x * z + y * w)],
                [2 * (x * y + z * w),     1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                [2 * (x * z - y * w),     2 * (y * z + x * w),     1 - 2 * (x * x + y * y)],
            ],
            dtype=np.float64,
        )

    def _apply_pose_to_client(self, client, pose_xyz_wxyz: np.ndarray) -> None:
        """Drive a viser client's orbit camera to the given OpenCV pose.

        Pose layout is ``[x, y, z, qw, qx, qy, qz]`` in the OpenCV camera
        convention (+Z forward, +Y down, +X right) — matching what we hand
        ``add_camera_frustum``. We give viser ``position``, ``up_direction``,
        and ``look_at``; its internal ``_update_wxyz`` then reconstructs the
        rotation matching the source pose.
        """
        pose = np.asarray(pose_xyz_wxyz, dtype=np.float64)
        position = pose[:3]
        wxyz = pose[3:7]
        R = self._wxyz_to_rotation_matrix(wxyz)
        forward = R[:, 2]
        up = -R[:, 1]  # OpenCV +Y is down -> world-up is -Y of the camera
        look_at = position + forward * self._VIEW_DISTANCE

        client.camera.position = position
        client.camera.up_direction = up
        client.camera.look_at = look_at

    # ------------------------------------------------------------------
    # Scene update
    # ------------------------------------------------------------------

    def _render_frame(self, frame: _Frame) -> None:
        cam_name = self._active_camera
        if cam_name is None and frame.cameras:
            # First frame may arrive before _sync_cameras has run.
            cam_name = next(iter(frame.cameras.keys()))
            self._active_camera = cam_name

        if cam_name is not None and cam_name in frame.cameras:
            snap = frame.cameras[cam_name]
            self._set_image(snap.image)
            self._set_frustum(cam_name, snap)
            # Hide stale frustums for other cameras so the scene isn't
            # cluttered with frozen viewpoints.
            for other_name, handle in self._frustum_handles.items():
                if other_name != cam_name:
                    handle.visible = False

        if frame.joints is not None and self.urdf_vis is not None:
            try:
                self.urdf_vis.update_cfg(frame.joints)
            except Exception:
                # urdf_vis may reject a joints vector of unexpected length
                # (e.g. simulator passed 7 instead of 8). Swallow silently —
                # we don't want a display-side issue to crash the trial.
                pass

    def _set_image(self, image: np.ndarray) -> None:
        if self._image_handle is None:
            self._image_handle = self.server.gui.add_image(
                image, label="Observation"
            )
        else:
            self._image_handle.image = image

    def _update_camera_parent(self, name: str, pose_xyz_wxyz: np.ndarray) -> None:
        """Create or move the scene-tree parent frame for camera ``name``.

        The frame lives at the bare ``{name}`` scene path so simulators that
        push depth-derived point clouds to ``{name}/point_cloud`` (camera-local
        coords) get them transformed into world coordinates. The axes are
        rendered small but visible — they double as a debug gizmo at each
        camera origin.
        """
        position = pose_xyz_wxyz[:3]
        wxyz = pose_xyz_wxyz[3:7]
        handle = self._camera_parent_handles.get(name)
        if handle is None:
            handle = self.server.scene.add_frame(
                name,
                position=position,
                wxyz=wxyz,
                axes_length=0.05,
                axes_radius=0.005,
            )
            self._camera_parent_handles[name] = handle
        else:
            handle.position = position
            handle.wxyz = wxyz

    def _set_frustum(self, name: str, snap: _CameraSnapshot) -> None:
        if snap.pose_xyz_wxyz is None:
            return
        # Remember the very first pose we see for this camera as the canonical
        # "init view" used by client-connect snap and the Reset View button.
        # We don't refresh on later frames — Reset View is supposed to return
        # to the original viewpoint even if the camera moved during a trial.
        if name not in self._initial_view_poses:
            self._initial_view_poses[name] = np.asarray(
                snap.pose_xyz_wxyz, dtype=np.float64
            ).copy()
        handle = self._frustum_handles.get(name)
        if handle is None:
            handle = self.server.scene.add_camera_frustum(
                name=f"history/{name}",
                position=snap.pose_xyz_wxyz[:3],
                wxyz=snap.pose_xyz_wxyz[3:7],
                fov=1.0,
                aspect=self._render_aspect,
                scale=0.05,
                image=snap.image,
            )
            self._frustum_handles[name] = handle
        else:
            handle.position = snap.pose_xyz_wxyz[:3]
            handle.wxyz = snap.pose_xyz_wxyz[3:7]
            handle.image = snap.image
            handle.visible = True


__all__ = ["ViserFrameHistory"]
