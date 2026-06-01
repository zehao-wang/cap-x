from __future__ import annotations

import os
import threading
import time
from typing import Any

import numpy as np

from capx.envs.simulators.piper.common import (
    VISER_DEPTH_FAR_M,
    VISER_DEPTH_NEAR_M,
    VISER_WRIST_DEPTH_FAR_M,
    VISER_WRIST_DEPTH_NEAR_M,
    _depth_to_color_image,
)
from capx.utils.camera_utils import obs_get_rgb
from capx.utils.depth_utils import depth_color_to_pointcloud


def _apply_display_flip(image: np.ndarray, mode: str) -> np.ndarray:
    if mode == "h":
        return np.ascontiguousarray(image[:, ::-1])
    if mode == "v":
        return np.ascontiguousarray(image[::-1, :])
    if mode == "180":
        return np.ascontiguousarray(image[::-1, ::-1])
    return image


def _apply_intrinsics_flip(K: np.ndarray, mode: str, width: int, height: int) -> np.ndarray:
    """Adjust a pinhole intrinsic matrix to match a flipped image.

    For a 180/h/v flip applied to the image, cx and cy must be remapped so the
    resulting matrix still describes the same physical camera in pixel space.
    """
    if mode not in ("h", "v", "180"):
        return K
    K = np.asarray(K, dtype=np.float64).copy()
    if mode in ("h", "180"):
        K[0, 2] = (width - 1) - K[0, 2]
    if mode in ("v", "180"):
        K[1, 2] = (height - 1) - K[1, 2]
    return K


def _wrist_image_flip_mode() -> str:
    """Source of truth for the wrist-camera flip applied at obs-write time."""
    return os.environ.get("PIPER_WRIST_CAMERA_VISER_FLIP", "180")


class PiperViserMixin:
    # Viser GUI sliders that control depth-panel colormap range and the
    # merged point-cloud clip range. They do NOT affect the raw depth that
    # the agent observes via obs[<key>]["images"]["depth"].
    _DEPTH_VIEW_SLIDER_MAX_M = 5.0

    def _ensure_depth_view_sliders(self) -> None:
        if self.viser_server is None:
            return
        if getattr(self, "_depth_view_sliders", None) is not None:
            return
        sliders: dict[str, Any] = {}
        try:
            with self.viser_server.gui.add_folder("Depth view ranges (m)"):
                sliders["zed_near"] = self.viser_server.gui.add_slider(
                    "ZED near",
                    min=0.0,
                    max=self._DEPTH_VIEW_SLIDER_MAX_M,
                    step=0.01,
                    initial_value=float(VISER_DEPTH_NEAR_M),
                )
                sliders["zed_far"] = self.viser_server.gui.add_slider(
                    "ZED far",
                    min=0.0,
                    max=self._DEPTH_VIEW_SLIDER_MAX_M,
                    step=0.01,
                    initial_value=float(VISER_DEPTH_FAR_M),
                )
                sliders["wrist_near"] = self.viser_server.gui.add_slider(
                    "Wrist near",
                    min=0.0,
                    max=self._DEPTH_VIEW_SLIDER_MAX_M,
                    step=0.01,
                    initial_value=float(VISER_WRIST_DEPTH_NEAR_M),
                )
                sliders["wrist_far"] = self.viser_server.gui.add_slider(
                    "Wrist far",
                    min=0.0,
                    max=self._DEPTH_VIEW_SLIDER_MAX_M,
                    step=0.01,
                    initial_value=float(VISER_WRIST_DEPTH_FAR_M),
                )
        except Exception:
            return
        self._depth_view_sliders = sliders

    def _ensure_home_button(self) -> None:
        if self.viser_server is None:
            return
        if getattr(self, "_home_button_handle", None) is not None:
            return
        try:
            button = self.viser_server.gui.add_button("Return to 0")
        except Exception:
            return
        self._home_button_handle = button
        self._home_motion_thread: threading.Thread | None = None

        @button.on_click
        def _on_return_to_zero(_event) -> None:
            existing = getattr(self, "_home_motion_thread", None)
            if existing is not None and existing.is_alive():
                print("[piper_real] Return to 0 already in progress; ignoring click")
                return

            def _run_home() -> None:
                try:
                    button.disabled = True
                except Exception:
                    pass
                try:
                    if hasattr(self, "goto_home_blocking"):
                        self.goto_home_blocking()
                    else:
                        home = np.zeros(6, dtype=np.float64)
                        self.move_to_joints_blocking(home)
                    print("[piper_real] Return to 0: home pose reached.")
                except Exception as exc:
                    print(f"[piper_real] Return to 0 failed: {exc}")
                finally:
                    try:
                        button.disabled = False
                    except Exception:
                        pass

            thread = threading.Thread(
                target=_run_home, name="piper-viser-return-to-zero", daemon=True
            )
            self._home_motion_thread = thread
            thread.start()

    def _slider_pair(self, near_key: str, far_key: str, fallback: tuple[float, float]) -> tuple[float, float]:
        sliders = getattr(self, "_depth_view_sliders", None)
        if not sliders:
            return fallback
        try:
            near = float(sliders[near_key].value)
            far = float(sliders[far_key].value)
        except Exception:
            return fallback
        if far <= near:
            far = near + 1e-3
        return near, far

    def _zed_depth_view_range(self) -> tuple[float, float]:
        return self._slider_pair(
            "zed_near", "zed_far", (VISER_DEPTH_NEAR_M, VISER_DEPTH_FAR_M)
        )

    def _wrist_depth_view_range(self) -> tuple[float, float]:
        return self._slider_pair(
            "wrist_near",
            "wrist_far",
            (VISER_WRIST_DEPTH_NEAR_M, VISER_WRIST_DEPTH_FAR_M),
        )

    def _viser_init_check(self) -> None:
        if self.viser_server is None:
            return
        # Side-panel order: external view, external depth, wrist view, wrist depth.
        if self.viser_img_handle is None:
            self.viser_img_handle = self.viser_server.gui.add_image(
                np.zeros((480, 640, 3), dtype=np.uint8), label="Camera View"
            )
        if self.viser_depth_handle is None:
            self.viser_depth_handle = self.viser_server.gui.add_image(
                np.zeros((480, 640, 3), dtype=np.uint8),
                label="Depth (range from slider)",
            )
        wrist_present = (
            getattr(self, "_wrist_camera_enabled", False)
            or "robot0_eye_in_hand" in self.obs
        )
        if self.viser_wrist_img_handle is None and wrist_present:
            self.viser_wrist_img_handle = self.viser_server.gui.add_image(
                np.zeros((480, 640, 3), dtype=np.uint8),
                label="Wrist Camera View",
            )
        if self.viser_wrist_depth_handle is None and wrist_present:
            self.viser_wrist_depth_handle = self.viser_server.gui.add_image(
                np.zeros((480, 640, 3), dtype=np.uint8),
                label="Wrist Depth (range from slider)",
            )
        self._ensure_depth_view_sliders()
        self._ensure_home_button()
        if self.image_frustum_handle is None:
            self.image_frustum_handle = self.viser_server.scene.add_camera_frustum(
                name="robot0_robotview",
                position=(0, 0, 0),
                wxyz=(1, 0, 0, 0),
                fov=1.0,
                aspect=1.0,
                scale=0.05,
            )
            self.image_frustum_handles["robot0_robotview"] = self.image_frustum_handle

    def update_viser_image(self, frame: np.ndarray) -> None:
        if self.viser_server is None:
            return
        self._viser_init_check()
        if self.viser_img_handle is not None:
            self.viser_img_handle.image = frame

    def _refresh_viser_point_cloud(self) -> None:
        if self.viser_server is None:
            return
        self._viser_init_check()
        points_by_view: list[np.ndarray] = []
        colors_by_view: list[np.ndarray] = []
        for image_key in ("robot0_robotview", "robot0_eye_in_hand"):
            view = self.obs.get(image_key, {})
            depth = view.get("images", {}).get("depth")
            rgb = view.get("images", {}).get("rgb")
            pose_mat = view.get("pose_mat")
            if depth is None or rgb is None or pose_mat is None or "intrinsics" not in view:
                continue
            if depth.ndim == 3 and depth.shape[-1] == 1:
                depth = depth[:, :, 0]
            if image_key == "robot0_eye_in_hand":
                clip_range = self._wrist_depth_view_range()
            else:
                clip_range = self._zed_depth_view_range()
            try:
                points_camera, colors = depth_color_to_pointcloud(
                    depth,
                    rgb,
                    view["intrinsics"],
                    depth_clip_range=clip_range,
                )
                points_hom = np.concatenate(
                    [points_camera, np.ones((points_camera.shape[0], 1))],
                    axis=1,
                )
                points_by_view.append((pose_mat @ points_hom.T).T[:, :3])
                colors_by_view.append(colors)
            except Exception:
                continue
        if not points_by_view:
            return
        try:
            points = np.concatenate(points_by_view, axis=0)
            colors = np.concatenate(colors_by_view, axis=0)
            self.viser_server.scene.add_point_cloud(
                "merged_camera_point_cloud",
                points,
                colors,
                point_size=0.001,
                point_shape="square",
            )
        except Exception:
            pass

    def _update_viser_server(self, *, force_scene: bool = False) -> None:
        if self.viser_server is None:
            return
        try:
            import viser.transforms as vtf  # type: ignore
        except Exception:
            return

        self._viser_init_check()
        self._update_robot_mesh()
        now = time.time()
        if (
            not force_scene
            and now - self._last_viser_scene_update_time
            < self._viser_scene_update_interval
        ):
            return
        self._last_viser_scene_update_time = now
        self._update_camera_frustums(vtf)
        self._update_object_overlays(vtf)
        self._update_ik_target()
        self._update_preview_path()

    def _update_robot_mesh(self) -> None:
        if self.urdf_vis is None or self._urdf_n_actuated <= 0:
            return
        cfg = np.zeros(self._urdf_n_actuated, dtype=np.float64)
        joint_pos = np.asarray(
            self.obs.get("robot_joint_pos", self._current_joints), dtype=np.float64
        )
        n_arm = min(6, self._urdf_n_actuated)
        cfg[:n_arm] = joint_pos[:n_arm]
        if self._urdf_n_actuated >= 8 and joint_pos.shape[0] >= 7:
            gripper_m = float(joint_pos[6])
            cfg[6] = 0.5 * gripper_m
            cfg[7] = -0.5 * gripper_m
        try:
            self.urdf_vis.update_cfg(cfg)
        except Exception:
            pass

    def _update_camera_frustums(self, vtf) -> None:
        panels = {
            "robot0_robotview": (self.viser_img_handle, self.viser_depth_handle),
            "robot0_eye_in_hand": (
                self.viser_wrist_img_handle,
                self.viser_wrist_depth_handle,
            ),
        }
        for image_key, rgb in obs_get_rgb(self.obs).items():
            view = self.obs.get(image_key, {})
            rgb_handle, depth_handle = panels.get(image_key, (None, None))
            if rgb_handle is not None:
                rgb_handle.image = self._viser_panel_image(image_key, rgb)
            depth_img = view.get("images", {}).get("depth")
            if depth_img is not None and depth_handle is not None:
                if image_key == "robot0_eye_in_hand":
                    near_m, far_m = self._wrist_depth_view_range()
                else:
                    near_m, far_m = self._zed_depth_view_range()
                try:
                    depth_handle.image = self._viser_panel_image(
                        image_key,
                        _depth_to_color_image(depth_img, near=near_m, far=far_m),
                    )
                except Exception:
                    pass
            frustum = self._get_frustum(image_key, rgb)
            pose_mat = view.get("pose_mat")
            pose = view.get("pose")
            if pose_mat is not None:
                frustum.position = pose_mat[:3, 3]
                frustum.wxyz = vtf.SE3.from_matrix(pose_mat).rotation().wxyz
            elif pose is not None:
                frustum.position = pose[:3]
                frustum.wxyz = pose[3:]
            else:
                frustum.visible = False
                continue
            # Frustum thumbnails project through the calibrated extrinsic;
            # wrist obs is already flipped to match the calibration frame.
            frustum.image = rgb
            frustum.visible = True

    def _viser_panel_image(self, image_key: str, image: np.ndarray) -> np.ndarray:
        # Wrist frames are already flipped at obs-write time
        # (see _update_wrist_observation), so no extra rotation here.
        return image

    def _get_frustum(self, image_key: str, rgb: np.ndarray):
        frustum = self.image_frustum_handles.get(image_key)
        if frustum is not None:
            return frustum
        h, w = rgb.shape[:2]
        frustum = self.viser_server.scene.add_camera_frustum(
            name=image_key,
            position=(0, 0, 0),
            wxyz=(1, 0, 0, 0),
            fov=1.0,
            aspect=float(w) / max(float(h), 1.0),
            scale=0.05,
        )
        self.image_frustum_handles[image_key] = frustum
        return frustum

    def _update_object_overlays(self, vtf) -> None:
        cube_center = getattr(self, "cube_center", None)
        cube_rot = getattr(self, "cube_rot", None)
        if cube_center is not None and cube_rot is not None:
            self._show_cube_frame(vtf, cube_center, cube_rot)
        elif self.cube_frame_handle is not None:
            self.cube_frame_handle.visible = False
        cube_points = getattr(self, "cube_points", None)
        cube_color = getattr(self, "cube_color", None)
        if cube_points is not None and cube_color is not None:
            self._show_cube_cloud(cube_points, cube_color)
        elif self.cube_point_cloud_handle is not None:
            self.cube_point_cloud_handle.visible = False
        self._show_best_grasp(vtf)

    def _show_cube_frame(self, vtf, center, rot) -> None:
        try:
            pos = np.asarray(center, dtype=np.float64)
            wxyz = vtf.SO3.from_matrix(np.asarray(rot)).wxyz
            if self.cube_frame_handle is None:
                self.cube_frame_handle = self.viser_server.scene.add_frame(
                    "robot0_robotview/cube_frame",
                    position=pos,
                    wxyz=wxyz,
                    axes_length=0.05,
                    axes_radius=0.005,
                )
            else:
                self.cube_frame_handle.position = pos
                self.cube_frame_handle.wxyz = wxyz
                self.cube_frame_handle.visible = True
        except Exception:
            pass

    def _show_cube_cloud(self, points, colors) -> None:
        try:
            points = np.asarray(points)
            colors = np.asarray(colors)
            if self.cube_point_cloud_handle is None:
                self.cube_point_cloud_handle = self.viser_server.scene.add_point_cloud(
                    "robot0_robotview/cube_point_cloud",
                    points,
                    colors,
                    point_size=0.002,
                    point_shape="square",
                )
            else:
                self.cube_point_cloud_handle.points = points
                self.cube_point_cloud_handle.colors = colors
                self.cube_point_cloud_handle.visible = True
        except Exception:
            pass

    def _show_best_grasp(self, vtf) -> None:
        grasp_sample = getattr(self, "grasp_sample", None)
        scores = getattr(self, "grasp_scores", None)
        if grasp_sample is None or scores is None or len(scores) == 0:
            if self.grasp_mesh_handle is not None:
                self.grasp_mesh_handle.visible = False
            return
        try:
            best_idx = int(np.argmax(scores))
            tf = vtf.SE3.from_matrix(grasp_sample[best_idx]) @ vtf.SE3.from_translation(
                np.array([0.0, 0.0, 0.12])
            )
            if self.grasp_mesh_handle is None:
                self.grasp_mesh_handle = self.viser_server.scene.add_frame(
                    "robot0_robotview/grasp",
                    position=tf.wxyz_xyz[-3:],
                    wxyz=tf.wxyz_xyz[:4],
                    axes_length=0.065,
                    axes_radius=0.003,
                )
            else:
                self.grasp_mesh_handle.position = tf.wxyz_xyz[-3:]
                self.grasp_mesh_handle.wxyz = tf.wxyz_xyz[:4]
                self.grasp_mesh_handle.visible = True
        except Exception:
            pass

    def _update_ik_target(self) -> None:
        ik_target = getattr(self, "last_ik_target", None)
        if ik_target is None:
            return
        try:
            pos, quat_wxyz = ik_target
            if self.ik_target_handle is None:
                self.ik_target_handle = self.viser_server.scene.add_frame(
                    "ik_target",
                    position=np.asarray(pos, dtype=np.float64),
                    wxyz=np.asarray(quat_wxyz, dtype=np.float64),
                    axes_length=0.06,
                    axes_radius=0.002,
                )
            else:
                self.ik_target_handle.position = pos
                self.ik_target_handle.wxyz = quat_wxyz
                self.ik_target_handle.visible = True
        except Exception:
            pass

    def _update_preview_path(self) -> None:
        preview_wps = getattr(self, "preview_waypoints", None)
        if not preview_wps:
            self._hide_preview()
            return
        try:
            pts = np.asarray([wp["position"] for wp in preview_wps], dtype=np.float32)
            last_wp = preview_wps[-1]
            if len(pts) >= 2:
                if self.preview_path_handle is None:
                    self.preview_path_handle = self.viser_server.scene.add_spline_catmull_rom(
                        "preview/path", points=pts, color=(50, 200, 50), line_width=3.0
                    )
                else:
                    self.preview_path_handle.points = pts
                    self.preview_path_handle.visible = True
            elif self.preview_path_handle is not None:
                self.preview_path_handle.visible = False
            if self.preview_target_handle is None:
                self.preview_target_handle = self.viser_server.scene.add_frame(
                    "preview/target",
                    position=np.asarray(last_wp["position"], dtype=np.float64),
                    wxyz=np.asarray(last_wp["quat_wxyz"], dtype=np.float64),
                    axes_length=0.05,
                    axes_radius=0.003,
                )
            else:
                self.preview_target_handle.position = last_wp["position"]
                self.preview_target_handle.wxyz = last_wp["quat_wxyz"]
                self.preview_target_handle.visible = True
        except Exception as e:
            print(f"[piper_real] preview render failed: {e}")

    def _hide_preview(self) -> None:
        try:
            if self.preview_path_handle is not None:
                self.preview_path_handle.visible = False
            if self.preview_target_handle is not None:
                self.preview_target_handle.visible = False
        except Exception:
            pass

__all__ = ["PiperViserMixin"]
