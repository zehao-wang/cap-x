"""Viser GUI/scene panel for :class:`ViserFrameHistory`.

Owns everything that touches the viser server: the **Attempt / Timestep / Live /
Camera / Reset View** widgets, the on-scene concatenated observation image, the
per-camera frustums, the camera parent frames, and the orbit-camera snapping.

It also owns the *view* state (which attempt/segment + frame is shown, whether
playback is Live, which camera Reset View targets); the frame *data* lives in
:class:`ViserFrameHistory`, which the panel reads through a ``segments_getter``
callback. Frames are duck-typed so this module imports nothing from the history.

The history calls three notifications: :meth:`notify_recorded` (a frame was
appended to the live segment), :meth:`notify_new_segment` (an in-trial reset
started a fresh attempt), and :meth:`notify_cleared` (a new trial dropped
everything). Each notification is a no-op once the server's event loop has
closed (task switch), so a stale panel goes inert instead of crashing.
"""

from __future__ import annotations

from typing import Callable, Optional

import numpy as np


class ViserPlaybackPanel:
    """The viser-facing half of the trial playback history."""

    # Distance between camera position and look_at point when snapping a
    # client's view to a stored pose. Affects orbit-controls feel only.
    _VIEW_DISTANCE: float = 0.6
    # Fallback frustum FOV (radians) when the simulator didn't supply one.
    _DEFAULT_FOV: float = 1.0

    def __init__(
        self,
        viser_server,
        segments_getter: Callable[[], list],
        *,
        urdf_vis=None,
        max_frames: int = 100,
        render_aspect: float = 4.0 / 3.0,
        folder_label: str = "Trial Playback",
    ) -> None:
        self.server = viser_server
        self._segments_getter = segments_getter
        self.urdf_vis = urdf_vis
        self.max_frames = max(2, int(max_frames))
        self._render_aspect = float(render_aspect)
        self._folder_label = folder_label

        # View state (which segment/frame is shown; Live; Reset View target).
        self._active_idx: int = 0
        self._live: bool = True
        self._active_camera: Optional[str] = None
        self._known_cameras: list[str] = []
        self._suppress_callbacks: bool = False  # block re-entry while we set widgets
        self._defunct: bool = False  # latched once the server's loop closes

        # GUI + scene handles, created lazily on the first recorded frame.
        self._gui_ready = False
        self._gui_folder = None
        self._attempt_dropdown = None
        self._slider = None
        self._live_toggle = None
        self._camera_dropdown = None
        self._step_label = None
        self._image_handle = None  # single GUI image holding the concatenated view
        self._frustum_handles: dict[str, object] = {}
        self._camera_parent_handles: dict[str, object] = {}
        self._reset_button = None

        # Snap newly-connecting clients to the active camera's current pose.
        self.server.on_client_connect(self._on_client_connect)

    # ------------------------------------------------------------------
    # Public API (called by ViserFrameHistory)
    # ------------------------------------------------------------------

    @property
    def live(self) -> bool:
        """True when the slider is locked to the latest frame of the latest
        segment. Simulators gate their own live URDF updates on this so the
        scrubbed pose isn't overwritten."""
        return self._live

    def notify_recorded(self, frame) -> None:
        """A frame was appended to the live (latest) segment — refresh the GUI."""
        if not self._server_alive():
            return
        segments = self._segments_getter()
        live_on_latest = self._live and self._active_idx == len(segments) - 1
        try:
            self._ensure_gui()
            self._sync_cameras(frame)
            self._refresh_attempt_options()
            if live_on_latest:
                self._refresh_slider_bounds()
            # Update the live camera parent frames so scene children the
            # simulator adds under "{name}/..." (e.g. point clouds) track the
            # current camera pose. Outside _render_frame so scrubbing doesn't
            # move them — live point clouds stay anchored to their capture pose.
            for cam_name, snap in frame.cameras.items():
                if snap.pose_xyz_wxyz is not None:
                    self._update_camera_parent(cam_name, snap.pose_xyz_wxyz)
            if live_on_latest:
                self._render_frame(frame)
        except RuntimeError:
            # Server stopped mid-update (task-switch teardown race). Go inert.
            self._defunct = True

    def notify_new_segment(self) -> None:
        """An in-trial reset started a fresh attempt — follow it, stay Live."""
        segments = self._segments_getter()
        self._active_idx = max(0, len(segments) - 1)
        self._live = True
        if not self._gui_ready or not self._server_alive():
            return
        try:
            self._refresh_attempt_options()
            self._refresh_slider_bounds()
        except RuntimeError:
            self._defunct = True

    def notify_cleared(self) -> None:
        """A new trial dropped every segment — reset the widgets to empty."""
        self._active_idx = 0
        self._live = True
        if not self._gui_ready or not self._server_alive():
            return
        self._suppress_callbacks = True
        try:
            if self._attempt_dropdown is not None:
                self._attempt_dropdown.options = ("Attempt 1",)
                self._attempt_dropdown.value = "Attempt 1"
            if self._slider is not None:
                self._slider.max = max(1, self.max_frames - 1)
                self._slider.value = 0
            if self._live_toggle is not None:
                self._live_toggle.value = True
            if self._step_label is not None:
                self._step_label.value = "0 / 0"
        except RuntimeError:
            self._defunct = True
        finally:
            self._suppress_callbacks = False

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @property
    def _display_frames(self) -> list:
        """Frames of the segment the GUI currently reflects."""
        segments = self._segments_getter()
        if not segments:
            return []
        idx = max(0, min(self._active_idx, len(segments) - 1))
        return segments[idx]

    def _server_alive(self) -> bool:
        """False once the viser server's background event loop has closed.

        The web UI stops the server on every task switch; a simulator may still
        hold this panel and notify it. Every GUI/scene mutation then raises
        ``RuntimeError('Event loop is closed')``. We check the loop up front and
        latch ``_defunct`` so a stale panel goes quietly inert.
        """
        if self._defunct:
            return False
        loop = getattr(
            getattr(self.server, "_websock_server", None),
            "_background_event_loop",
            None,
        )
        if loop is not None and loop.is_closed():
            self._defunct = True
            return False
        return True

    def _ensure_gui(self) -> None:
        """Create the Attempt/slider/Live/label widgets. The Camera dropdown is
        built lazily by ``_sync_cameras`` once we know at least one camera name —
        viser's ``add_dropdown`` reads ``options[0]`` for its initial value and
        crashes on an empty tuple.
        """
        if self._gui_ready:
            return
        self._gui_folder = self.server.gui.add_folder(self._folder_label)
        with self._gui_folder:
            self._attempt_dropdown = self.server.gui.add_dropdown(
                "Attempt",
                options=("Attempt 1",),
                initial_value="Attempt 1",
            )
            # max must be > min at creation: viser clamps `step` to
            # `min(step, max - min)` internally, and only `max` is re-sent on
            # later updates. A degenerate max=0 would bake step=0 in the
            # frontend and dragging would divide by zero. Use max_frames-1.
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

        self._attempt_dropdown.on_update(self._on_attempt_change)
        self._slider.on_update(self._on_slider_change)
        self._live_toggle.on_update(self._on_live_toggle)
        self._reset_button.on_click(self._on_reset_view)
        self._gui_ready = True

    def _sync_cameras(self, frame) -> None:
        """Extend (or first-time create) the camera dropdown."""
        new = [c for c in frame.cameras.keys() if c not in self._known_cameras]
        if not new:
            return
        self._known_cameras.extend(new)

        self._suppress_callbacks = True
        try:
            if self._camera_dropdown is None:
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

    def _refresh_attempt_options(self) -> None:
        """Keep the Attempt dropdown options in sync with the segment count."""
        if self._attempt_dropdown is None:
            return
        n = max(1, len(self._segments_getter()))
        options = tuple(f"Attempt {i + 1}" for i in range(n))
        want = f"Attempt {self._active_idx + 1}"
        self._suppress_callbacks = True
        try:
            if tuple(self._attempt_dropdown.options) != options:
                self._attempt_dropdown.options = options
            # Follow the live segment when Live is on.
            if self._live and self._attempt_dropdown.value != want:
                self._attempt_dropdown.value = want
        finally:
            self._suppress_callbacks = False

    def _refresh_slider_bounds(self) -> None:
        frames = self._display_frames
        new_max = max(0, len(frames) - 1)
        self._suppress_callbacks = True
        try:
            self._slider.max = max(new_max, 1)
            if self._live:
                self._slider.value = new_max
            value = min(int(self._slider.value), new_max)
            self._step_label.value = f"{value} / {new_max}"
        finally:
            self._suppress_callbacks = False

    def _render_current(self) -> None:
        """Render the slider-pointed frame of the active segment."""
        frames = self._display_frames
        if not frames:
            return
        idx = max(0, min(int(self._slider.value), len(frames) - 1))
        self._render_frame(frames[idx])

    # ------------------------------------------------------------------
    # Callbacks
    # ------------------------------------------------------------------

    def _on_attempt_change(self, _event) -> None:
        if self._suppress_callbacks:
            return
        segments = self._segments_getter()
        if not segments:
            return
        try:
            idx = int(self._attempt_dropdown.value.rsplit(" ", 1)[1]) - 1
        except (ValueError, IndexError):
            return
        self._active_idx = max(0, min(idx, len(segments) - 1))
        # Selecting a non-latest attempt freezes Live so record() doesn't drag
        # the view back to the recording segment.
        if self._active_idx != len(segments) - 1:
            self._set_live(False)
        self._refresh_slider_bounds()
        self._render_current()

    def _on_slider_change(self, _event) -> None:
        if self._suppress_callbacks:
            return
        frames = self._display_frames
        if not frames:
            return
        idx = max(0, min(int(self._slider.value), len(frames) - 1))
        # Any manual slider move turns Live off so record() doesn't drag us back.
        if idx != len(frames) - 1:
            self._set_live(False)
        self._step_label.value = f"{idx} / {len(frames) - 1}"
        self._render_frame(frames[idx])

    def _on_camera_change(self, _event) -> None:
        # The on-scene image concatenates all cameras and every frustum is
        # drawn, so the only effect of the Camera dropdown is which camera
        # Reset View / client-connect snaps to. No re-render needed.
        if self._suppress_callbacks:
            return
        self._active_camera = self._camera_dropdown.value

    def _on_live_toggle(self, _event) -> None:
        if self._suppress_callbacks:
            return
        self._live = bool(self._live_toggle.value)
        if self._live and self._segments_getter():
            # Live always follows the latest (recording) segment.
            self._active_idx = len(self._segments_getter()) - 1
            self._refresh_attempt_options()
            self._refresh_slider_bounds()
            self._render_current()

    def _set_live(self, value: bool) -> None:
        """Set Live and mirror it onto the toggle without re-entering callbacks."""
        if self._live == value:
            return
        self._live = value
        if self._live_toggle is None:
            return
        self._suppress_callbacks = True
        try:
            self._live_toggle.value = value
        finally:
            self._suppress_callbacks = False

    def _on_client_connect(self, client) -> None:
        pose = self._active_current_pose()
        if pose is None:
            return
        try:
            self._apply_pose_to_client(client, pose)
        except Exception:
            # Don't let a display-side issue tear the client's connect path.
            pass

    def _on_reset_view(self, _event) -> None:
        pose = self._active_current_pose()
        if pose is None:
            return
        for client in self.server.get_clients().values():
            try:
                self._apply_pose_to_client(client, pose)
            except Exception:
                pass

    def _active_current_pose(self) -> Optional[np.ndarray]:
        """Pose of the selected camera at the currently displayed frame.

        Reset View / client-connect snap to *this* — wherever the chosen camera
        is right now in the displayed attempt, not a remembered initial pose.
        Falls back to any posed camera in the frame if the selected one is
        missing there.
        """
        frames = self._display_frames
        if not frames:
            return None
        idx = int(self._slider.value) if self._slider is not None else len(frames) - 1
        idx = max(0, min(idx, len(frames) - 1))
        frame = frames[idx]
        snap = frame.cameras.get(self._active_camera) if self._active_camera else None
        if (snap is None or snap.pose_xyz_wxyz is None) and frame.cameras:
            snap = next(
                (s for s in frame.cameras.values() if s.pose_xyz_wxyz is not None),
                None,
            )
        return None if snap is None else snap.pose_xyz_wxyz

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
        ``add_camera_frustum``. viser reconstructs the rotation from position +
        up_direction + look_at.
        """
        pose = np.asarray(pose_xyz_wxyz, dtype=np.float64)
        position = pose[:3]
        R = self._wxyz_to_rotation_matrix(pose[3:7])
        forward = R[:, 2]
        up = -R[:, 1]  # OpenCV +Y is down -> world-up is -Y of the camera
        client.camera.position = position
        client.camera.up_direction = up
        client.camera.look_at = position + forward * self._VIEW_DISTANCE

    # ------------------------------------------------------------------
    # Scene update
    # ------------------------------------------------------------------

    def _render_frame(self, frame) -> None:
        # 2D: a single image concatenating every camera in this frame.
        if frame.cameras:
            ordered = [c for c in self._known_cameras if c in frame.cameras]
            ordered += [c for c in frame.cameras if c not in ordered]
            self._set_image(self._concat_images([frame.cameras[c].image for c in ordered]))

        # 3D: one frustum per camera at its own pose/FOV; hide cameras absent
        # from this frame.
        for name, snap in frame.cameras.items():
            self._set_frustum(name, snap)
        for name, handle in self._frustum_handles.items():
            handle.visible = name in frame.cameras

        if frame.joints is not None and self.urdf_vis is not None:
            try:
                self.urdf_vis.update_cfg(frame.joints)
            except Exception:
                # urdf_vis may reject a joints vector of unexpected length —
                # swallow so a display-side issue never crashes the trial.
                pass

    @staticmethod
    def _concat_images(images: list[np.ndarray]) -> np.ndarray:
        """Horizontally concatenate camera images, matching heights.

        Single image -> returned as-is. Differing heights are resized (nearest)
        to the max height so the hstack is rectangular.
        """
        if len(images) == 1:
            return images[0]
        target_h = max(img.shape[0] for img in images)
        resized = []
        for img in images:
            h, w = img.shape[:2]
            if h == target_h:
                resized.append(img)
                continue
            new_w = max(1, int(round(w * target_h / h)))
            rows = (np.arange(target_h) * h / target_h).astype(np.int64)
            cols = (np.arange(new_w) * w / new_w).astype(np.int64)
            resized.append(img[rows][:, cols])
        return np.concatenate(resized, axis=1)

    def _set_image(self, image: np.ndarray) -> None:
        if self._image_handle is None:
            self._image_handle = self.server.gui.add_image(image, label="Observation")
        else:
            self._image_handle.image = image

    def _update_camera_parent(self, name: str, pose_xyz_wxyz: np.ndarray) -> None:
        """Create or move the scene-tree parent frame for camera ``name``.

        Lives at the bare ``{name}`` scene path so simulators that push
        depth-derived point clouds to ``{name}/point_cloud`` (camera-local
        coords) get them transformed into world coordinates. The small axes
        double as a debug gizmo at each camera origin.
        """
        position = pose_xyz_wxyz[:3]
        wxyz = pose_xyz_wxyz[3:7]
        handle = self._camera_parent_handles.get(name)
        if handle is None:
            self._camera_parent_handles[name] = self.server.scene.add_frame(
                name,
                position=position,
                wxyz=wxyz,
                axes_length=0.05,
                axes_radius=0.005,
            )
        else:
            handle.position = position
            handle.wxyz = wxyz

    def _set_frustum(self, name: str, snap) -> None:
        if snap.pose_xyz_wxyz is None:
            return
        handle = self._frustum_handles.get(name)
        if handle is None:
            self._frustum_handles[name] = self.server.scene.add_camera_frustum(
                name=f"history/{name}",
                position=snap.pose_xyz_wxyz[:3],
                wxyz=snap.pose_xyz_wxyz[3:7],
                fov=snap.fov if snap.fov is not None else self._DEFAULT_FOV,
                aspect=self._render_aspect,
                scale=0.05,
                image=snap.image,
            )
        else:
            # fov/aspect are fixed at creation (CameraFrustumHandle exposes no
            # fov setter); camera intrinsics are static within a trial, so only
            # pose + image need refreshing here.
            handle.position = snap.pose_xyz_wxyz[:3]
            handle.wxyz = snap.pose_xyz_wxyz[3:7]
            handle.image = snap.image
            handle.visible = True


__all__ = ["ViserPlaybackPanel"]
