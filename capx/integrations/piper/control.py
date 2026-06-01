"""Control API for the real Agilex PIPER arm (6-DOF + parallel-jaw gripper).

Mirrors FrankaControlApi but adapted for 6-DOF, a local (in-process) pyroki IK
solver using the PIPER URDF, and a `gripper_base` target link. Contact-GraspNet
is optional — if the extra isn't installed, a top-down grasp is synthesised
from the object pose instead.
"""

from __future__ import annotations

import os
import pathlib
import time
from typing import Any

import numpy as np
import open3d as o3d
import viser.transforms as vtf
from PIL import Image

from capx.envs.base import BaseEnv
from capx.envs.simulators.piper.common import (
    DEFAULT_PIPER_EEF_LINK,
    DEFAULT_PIPER_URDF,
    _resolve_piper_mesh,
)
from capx.integrations.base_api import ApiBase
from capx.integrations.franka.common import (
    close_gripper as _close_gripper,
    open_gripper as _open_gripper,
)
from capx.integrations.vision.sam3 import init_sam3, visualize_sam3_results
from capx.utils.camera_utils import obs_get_rgb
from capx.utils.depth_utils import depth_color_to_pointcloud, depth_to_rgb
from capx.utils.visualization_utils import draw_detections, overlay_segmentation_masks


GRASP_TCP_OFFSET_M = 0.12

# Contact-GraspNet uses the Panda gripper template: z = approach, x = finger
# open/close axis, y = orthogonal. PIPER's gripper TCP frame instead uses y as
# the open/close axis, so we post-multiply each predicted grasp by Rz(angle)
# (rotation around the approach axis) to swap x ↔ y. The default is -90°; if
# your gripper actually closes in the opposite direction, set this to +90°.
PIPER_GRASP_ROT_Z_DEG = float(os.environ.get("PIPER_GRASP_ROT_Z_DEG", "-90"))


def _grasp_frame_fix(angle_deg: float) -> np.ndarray:
    a = np.deg2rad(angle_deg)
    c, s = np.cos(a), np.sin(a)
    return np.array(
        [[c, -s, 0.0, 0.0],
         [s,  c, 0.0, 0.0],
         [0.0, 0.0, 1.0, 0.0],
         [0.0, 0.0, 0.0, 1.0]],
        dtype=np.float64,
    )


_GRASP_FRAME_FIX = _grasp_frame_fix(PIPER_GRASP_ROT_Z_DEG)


def _init_piper_ik(urdf_path: str, target_link_name: str):
    """Build a local pyroki IK solver for the PIPER URDF.

    Returns a callable matching `init_pyroki()`'s interface:
        ik_solve_fn(target_pose_wxyz_xyz, prev_cfg=None) -> np.ndarray joint vector
    """
    import pyroki as pk  # type: ignore
    import yourdfpy

    import capx.integrations.motion.pyroki_snippets as pks

    urdf = yourdfpy.URDF.load(
        urdf_path,
        filename_handler=_resolve_piper_mesh,
        build_collision_scene_graph=False,
        load_collision_meshes=False,
    )
    robot = pk.Robot.from_urdf(urdf)

    def ik_solve_fn(
        target_pose_wxyz_xyz: np.ndarray, prev_cfg: np.ndarray | None = None
    ) -> np.ndarray:
        target = np.asarray(target_pose_wxyz_xyz, dtype=np.float64)
        if prev_cfg is None:
            q = pks.solve_ik(
                robot=robot,
                target_link_name=target_link_name,
                target_position=target[-3:],
                target_wxyz=target[:-3],
            )
        else:
            q = pks.solve_ik_vel_cost(
                robot=robot,
                target_link_name=target_link_name,
                target_position=target[-3:],
                target_wxyz=target[:-3],
                prev_cfg=prev_cfg,
            )
        return np.asarray(q, dtype=np.float64)

    return ik_solve_fn


def _init_piper_pose_planner(urdf_path: str, target_link_name: str):
    """Build a local pose-space planner that returns smooth joint trajectories.

    The planner follows the same spirit as the PyRoKi `/plan` endpoint:
    interpolate cartesian poses over `timesteps`, solve IK waypoint-by-waypoint
    with velocity regularization, and return a multi-step joint trajectory.
    """
    import pyroki as pk  # type: ignore
    import yourdfpy
    from scipy.spatial.transform import Rotation as SciRotation, Slerp

    import capx.integrations.motion.pyroki_snippets as pks

    urdf = yourdfpy.URDF.load(
        urdf_path,
        filename_handler=_resolve_piper_mesh,
        build_collision_scene_graph=False,
        load_collision_meshes=False,
    )
    robot = pk.Robot.from_urdf(urdf)
    target_link_index = robot.links.names.index(target_link_name)
    num_actuated = int(robot.joints.num_actuated_joints)
    default_cfg = np.asarray(robot.joint_var_cls.default_factory(), dtype=np.float64).reshape(
        num_actuated
    )

    def _to_robot_cfg(joints: np.ndarray) -> np.ndarray:
        """Map arm-only joints to full robot actuated-joint vector expected by PyRoKi."""
        q_in = np.asarray(joints, dtype=np.float64).reshape(-1)
        if q_in.shape[0] == num_actuated:
            return q_in
        q_full = default_cfg.copy()
        n = min(q_in.shape[0], num_actuated)
        q_full[:n] = q_in[:n]
        return q_full

    def fk_target_pose_wxyz_xyz(joints: np.ndarray) -> np.ndarray:
        q = _to_robot_cfg(joints)
        Ts = np.asarray(robot.forward_kinematics(q), dtype=np.float64)
        pose = np.asarray(Ts[target_link_index], dtype=np.float64).reshape(7)
        return pose

    def _slerp_wxyz(q0_wxyz: np.ndarray, q1_wxyz: np.ndarray, num_steps: int) -> np.ndarray:
        q0_xyzw = np.array([q0_wxyz[1], q0_wxyz[2], q0_wxyz[3], q0_wxyz[0]], dtype=np.float64)
        q1_xyzw = np.array([q1_wxyz[1], q1_wxyz[2], q1_wxyz[3], q1_wxyz[0]], dtype=np.float64)
        r0 = SciRotation.from_quat(q0_xyzw)
        r1 = SciRotation.from_quat(q1_xyzw)
        slerp = Slerp([0.0, 1.0], SciRotation.concatenate([r0, r1]))
        t = np.linspace(0.0, 1.0, int(num_steps), dtype=np.float64)
        q_xyzw = slerp(t).as_quat()
        return np.column_stack([q_xyzw[:, 3], q_xyzw[:, 0], q_xyzw[:, 1], q_xyzw[:, 2]])

    def plan_fn(waypoint_poses_wxyz_xyz: np.ndarray, timesteps: int = 20) -> np.ndarray:
        poses = np.asarray(waypoint_poses_wxyz_xyz, dtype=np.float64).reshape(-1, 7)
        if poses.shape[0] < 2:
            raise ValueError("planner requires at least 2 poses (start + goal)")
        timesteps = int(max(2, timesteps))

        n_seg = poses.shape[0] - 1
        # Build a dense multi-segment cartesian path first, then resample to an
        # exact fixed-N trajectory (timesteps) so goto_pose always executes a
        # deterministic number of planner waypoints.
        seg_steps = [timesteps] * n_seg

        pose_path: list[np.ndarray] = []
        for i in range(n_seg):
            p0 = poses[i, 4:]
            p1 = poses[i + 1, 4:]
            q0 = poses[i, :4]
            q1 = poses[i + 1, :4]
            n = seg_steps[i]
            pos_interp = np.linspace(p0, p1, n, dtype=np.float64)
            quat_interp = _slerp_wxyz(q0, q1, n)
            seg_path = np.concatenate([quat_interp, pos_interp], axis=1)
            if i > 0:
                seg_path = seg_path[1:]  # drop duplicate boundary
            pose_path.append(seg_path)

        dense_poses = np.concatenate(pose_path, axis=0)
        if dense_poses.shape[0] != timesteps:
            sample_idx = np.linspace(
                0,
                dense_poses.shape[0] - 1,
                timesteps,
                dtype=np.int64,
            )
            dense_poses = dense_poses[sample_idx]

        traj: list[np.ndarray] = []
        prev_cfg: np.ndarray | None = None
        for i, pose in enumerate(dense_poses):
            if i == 0:
                cfg = pks.solve_ik(
                    robot=robot,
                    target_link_name=target_link_name,
                    target_position=pose[-3:],
                    target_wxyz=pose[:4],
                )
            else:
                cfg = pks.solve_ik_vel_cost(
                    robot=robot,
                    target_link_name=target_link_name,
                    target_position=pose[-3:],
                    target_wxyz=pose[:4],
                    prev_cfg=prev_cfg,
                )
            cfg_np = np.asarray(cfg, dtype=np.float64)
            traj.append(cfg_np[:6].copy())
            prev_cfg = cfg_np
        return np.asarray(traj, dtype=np.float64)

    return plan_fn, fk_target_pose_wxyz_xyz


class PiperControlApi(ApiBase):
    """Robot control helpers for Agilex PIPER.

    Functions:
      - get_object_pose(object_name) -> (position, quaternion_wxyz, bbox_extent|None)
      - sample_grasp_pose(object_name) -> (position, quaternion_wxyz)
      - refresh_point_clouds() -> None
      - goto_pose(position, quaternion_wxyz, z_approach=0.0) -> None
      - go_home() -> None
      - open_gripper() / close_gripper() -> None
    """

    def __init__(
        self,
        env: BaseEnv,
        tcp_offset: list[float] = [0.0, 0.0, -0.13],
        urdf_path: str = DEFAULT_PIPER_URDF,
        target_link_name: str = DEFAULT_PIPER_EEF_LINK,
        debug: bool = False,
    ) -> None:
        super().__init__(env)
        self._TCP_OFFSET = np.array(tcp_offset, dtype=np.float64)
        self.debug = debug
        self._target_link_name = target_link_name
        self._planner_timesteps = int(
            getattr(self._env, "piper_planner_timesteps", 20)
        )
        self._min_target_z = float(
            getattr(self._env, "piper_min_target_z", 0.02)
        )

        print("[piper_api] Init SAM3 segmentation...")
        self.sam3_seg_fn = init_sam3()

        print("[piper_api] Init PIPER IK (local pyroki)...")
        self.ik_solve_fn = _init_piper_ik(urdf_path, target_link_name)
        print("[piper_api] Init PIPER pose planner (local pyroki)...")
        self.pose_plan_fn, self.fk_target_pose_fn = _init_piper_pose_planner(
            urdf_path, target_link_name
        )
        self.cfg: np.ndarray | None = None

        # Contact-GraspNet is optional — only needed for sample_grasp_pose.
        self.grasp_net_plan_fn = None
        try:
            from capx.integrations.vision.graspnet import init_contact_graspnet

            self.grasp_net_plan_fn = init_contact_graspnet()
            print("[piper_api] Contact-GraspNet loaded.")
        except Exception as e:  # pragma: no cover
            print(
                "[piper_api] Contact-GraspNet unavailable "
                f"({type(e).__name__}); sample_grasp_pose will fall back to "
                "a top-down grasp at the detected object position."
            )

    # ------------------------------------------------------------------ #
    def functions(self) -> dict[str, Any]:
        return {
            "detect_objects": self.detect_objects,
            "get_object_pose": self.get_object_pose,
            "sample_grasp_pose": self.sample_grasp_pose,
            "preview_pose": self.preview_pose,
            "clear_preview": self.clear_preview,
            "refresh_point_clouds": self.refresh_point_clouds,
            "goto_pose": self.goto_pose,
            "go_home": self.go_home,
            "open_gripper": self.open_gripper,
            "close_gripper": self.close_gripper,
            "get_wrist_view": self.get_wrist_view,
            "get_scene_view": self.get_scene_view,
        }

    # ------------------------------------------------------------------ #
    def _get_view(self, view_key: str, view_label: str) -> dict[str, np.ndarray]:
        obs = self._env.get_observation()
        view = obs.get(view_key) or {}
        images = view.get("images") or {}
        rgb = images.get("rgb")
        depth = images.get("depth")
        if rgb is None:
            raise RuntimeError(
                f"{view_label} RGB not available — is the camera enabled and streaming?"
            )
        out: dict[str, np.ndarray] = {"rgb": np.asarray(rgb)}
        if depth is not None:
            out["depth"] = np.asarray(depth)

        self._log_step(f"get_{view_label}_view", f"Captured {view_label} RGB+depth.")
        log_images: list[np.ndarray] = [out["rgb"]]
        if "depth" in out:
            depth_arr = out["depth"]
            depth_2d = depth_arr[:, :, 0] if depth_arr.ndim == 3 else depth_arr
            log_images.append(depth_to_rgb(depth_2d))
        self._log_step_update(images=log_images)
        return out

    def get_wrist_view(self) -> dict[str, np.ndarray]:
        """Capture the current wrist (eye-in-hand) camera frame.

        Returns:
            Dict with key ``"rgb"`` (HxWx3 uint8) and, when available,
            ``"depth"`` (HxWx1 float32, metres). Raises if the wrist camera
            is not enabled.
        """
        return self._get_view("robot0_eye_in_hand", "wrist")

    def get_scene_view(self) -> dict[str, np.ndarray]:
        """Capture the current scene (ZED) camera frame.

        Returns:
            Dict with key ``"rgb"`` (HxWx3 uint8) and ``"depth"`` (HxWx1
            float32, metres) when the scene camera provides depth.
        """
        return self._get_view("robot0_robotview", "scene")

    # ------------------------------------------------------------------ #
    def refresh_point_clouds(self) -> None:
        """Refresh the full-scene Viser depth point cloud on explicit request.

        Normal preview and motion updates do not refresh this point cloud; it is
        otherwise refreshed at the start of each trial/reset.
        """
        if hasattr(self._env, "get_observation"):
            self._env.get_observation()
        if hasattr(self._env, "_refresh_viser_point_cloud"):
            self._env._refresh_viser_point_cloud()
        self._log_step(
            "refresh_point_clouds",
            "Refreshed the full-scene Viser point cloud on explicit request.",
        )

    # ------------------------------------------------------------------ #
    def preview_pose(
        self,
        position: np.ndarray,
        quaternion_wxyz: np.ndarray,
        z_approach: float = 0.0,
        label: str | None = None,
    ) -> None:
        """Draw a planned EEF trajectory preview on Viser.

        Does not move the robot. Use this to visualise where ``goto_pose``
        would take the gripper. This expands a planner trajectory and renders
        it as a path with a final target frame.

        Multiple previews accumulate — call it more than once to lay out a
        multi-step plan, then call ``clear_preview()`` when you're ready to
        execute.

        Args:
            position: (3,) XYZ in metres, world frame.
            quaternion_wxyz: (4,) WXYZ unit quaternion.
            z_approach: If > 0, inserts a pre-approach target
                ``z_approach`` metres back along the EEF -Z axis before final.
            label: Optional name used only for logging.
        """
        from scipy.spatial.transform import Rotation as SciRotation

        pos = np.asarray(position, dtype=np.float64).reshape(3)
        quat_wxyz = np.asarray(quaternion_wxyz, dtype=np.float64).reshape(4)

        if not hasattr(self._env, "preview_waypoints") or self._env.preview_waypoints is None:
            self._env.preview_waypoints = []

        idx = len(self._env.preview_waypoints)
        name = label if label else f"wp_{idx}"

        quat_xyzw = np.array(
            [quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]], dtype=np.float64
        )
        rot = SciRotation.from_quat(quat_xyzw)
        offset_pos = pos + rot.apply(self._TCP_OFFSET)

        # Build planner waypoints in the same space used by goto_pose
        # (target link pose with TCP offset applied).
        if len(self._env.preview_waypoints) > 0:
            last_wp = self._env.preview_waypoints[-1]
            start_pose = np.concatenate(
                [
                    np.asarray(last_wp["quat_wxyz"], dtype=np.float64).reshape(4),
                    np.asarray(last_wp["position"], dtype=np.float64).reshape(3),
                ]
            )
        else:
            current_joints = np.asarray(
                self._env.get_observation()["robot_joint_pos"][:6], dtype=np.float64
            )
            start_pose = self.fk_target_pose_fn(current_joints)

        planner_waypoints: list[np.ndarray] = [start_pose]
        if z_approach != 0.0:
            approach_pos = offset_pos + rot.apply(np.array([0, 0, -z_approach]))
            planner_waypoints.append(np.concatenate([quat_wxyz, approach_pos]))
        planner_waypoints.append(np.concatenate([quat_wxyz, offset_pos]))

        timesteps = int(max(2, self._planner_timesteps))
        cmd_traj = self.pose_plan_fn(np.asarray(planner_waypoints, dtype=np.float64), timesteps=timesteps)
        traj_poses = np.asarray([self.fk_target_pose_fn(q) for q in cmd_traj], dtype=np.float64)

        start_idx = 1 if len(self._env.preview_waypoints) > 0 else 0
        new_entries: list[dict[str, Any]] = []
        for local_i, pose in enumerate(traj_poses[start_idx:]):
            new_entries.append(
                {
                    "name": f"{name}_wp{local_i}",
                    "position": pose[-3:].copy(),
                    "quat_wxyz": pose[:4].copy(),
                }
            )
        self._env.preview_waypoints.extend(new_entries)

        self._log_step(
            "preview_pose",
            f"Preview '{name}': pos={np.array2string(pos, precision=4)}, "
            f"z_approach={z_approach:.3f}, rendered planner waypoints={len(new_entries)} (no motion)",
        )

        # Force the Viser scene to re-render so the new waypoint shows up
        # immediately. preview_pose never moves the arm, so the usual
        # post-motion redraw won't fire on its own.
        viser_server = getattr(self._env, "viser_server", None)
        if viser_server is not None and hasattr(self._env, "_update_viser_server"):
            try:
                self._env._update_viser_server(force_scene=True)
            except Exception as e:
                print(f"[piper_api] preview_pose: Viser update failed: {e}")

    def clear_preview(self) -> None:
        """Erase all previously-drawn preview waypoints."""
        self._env.preview_waypoints = []
        viser_server = getattr(self._env, "viser_server", None)
        if viser_server is not None and hasattr(self._env, "_update_viser_server"):
            try:
                self._env._update_viser_server(force_scene=True)
            except Exception as e:
                print(f"[piper_api] clear_preview: Viser update failed: {e}")
        self._log_step("clear_preview", "Cleared preview waypoints.")

    # ------------------------------------------------------------------ #
    def detect_objects(
        self, candidates: list[str], min_score: float = 0.05
    ) -> list[dict[str, Any]]:
        """Run SAM3 over a list of candidate object names and visualize results.

        For each candidate, keeps the highest-scoring detection above
        ``min_score``. The composite overlay (masks + boxes + labels) is
        pushed to the web UI so the user can see what was found.

        Args:
            candidates: Natural-language object names to look for.
            min_score: Discard detections below this confidence score.

        Returns:
            List of dicts (one per detected candidate, in input order) with
            keys ``"label"`` (str), ``"score"`` (float), and
            ``"box"`` ([x1, y1, x2, y2] in pixel coordinates). Masks are not
            returned to keep stdout small.
        """
        if isinstance(candidates, str):
            candidates = [candidates]

        self._log_step(
            "detect_objects",
            f"Running SAM3 over {len(candidates)} candidate(s): {candidates}",
        )
        obs = self._env.get_observation()
        rgb_imgs = obs_get_rgb(obs)
        assert rgb_imgs, "No RGB images in observation"
        rgb = next(iter(rgb_imgs.values()))

        detections: list[dict[str, Any]] = []
        for name in candidates:
            try:
                results = self.sam3_seg_fn(rgb, text_prompt=name)
            except Exception as e:
                print(f"[piper_api] detect_objects: SAM3 failed for '{name}': {e}")
                continue
            if not results:
                continue
            best = max(results, key=lambda r: r["score"])
            if best["score"] < min_score:
                continue
            detections.append({
                "label": name,
                "score": float(best["score"]),
                "box": [float(v) for v in best["box"]],
                "mask": best["mask"],
            })

        if detections:
            overlay = draw_detections(rgb, detections)
            self._log_step_update(
                text=", ".join(f"{d['label']} ({d['score']:.2f})" for d in detections),
                images=overlay,
            )
        else:
            self._log_step_update(text="No detections.", images=rgb)

        return [
            {"label": d["label"], "score": d["score"], "box": d["box"]}
            for d in detections
        ]

    # ------------------------------------------------------------------ #
    def _camera_to_world(self, obs: dict[str, Any]) -> vtf.SE3:
        pose = obs["robot0_robotview"]["pose"]
        return vtf.SE3.from_rotation_and_translation(
            rotation=vtf.SO3(wxyz=pose[3:]), translation=pose[:3]
        )

    def _segment(
        self, rgb: np.ndarray, object_name: str
    ) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
        self._log_step(
            "SAM3 Segmentation",
            f"Running SAM3 text-prompt segmentation for '{object_name}' …",
        )
        results = self.sam3_seg_fn(rgb, text_prompt=object_name)
        if len(results) == 0:
            raise ValueError(f"No SAM3 detections for '{object_name}'")
        scores = [r["score"] for r in results]
        best = results[int(np.argmax(scores))]

        if self.debug:
            visualize_sam3_results(
                Image.fromarray(rgb), object_name, results,
                output_dir=pathlib.Path("."), show=False,
            )
        vis_masks = [r["mask"] for r in results if r.get("score", 0) > 0.05]
        if vis_masks:
            self._log_step_update(
                text=f"Best score: {max(scores):.3f}",
                images=overlay_segmentation_masks(rgb, vis_masks),
            )
        return best["box"], best["mask"], results

    # ------------------------------------------------------------------ #
    def get_object_pose(
        self, object_name: str, return_bbox_extent: bool = False
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
        """Get the pose of an object given a natural-language description.

        Args:
            object_name: Name of the object to detect.
            return_bbox_extent: If True, also return the OBB extent (xyz side lengths).

        Returns:
            position: (3,) world-frame XYZ in metres.
            quaternion_wxyz: (4,) world-frame orientation (may be unreliable —
              prefer (0,0,1,0) for top-down placement).
            bbox_extent: (3,) xyz extent or None.
        """
        self._log_step("get_object_pose", f"Detecting object **'{object_name}'** …")
        t0 = time.time()
        obs = self._env.get_observation()

        rgb_imgs = obs_get_rgb(obs)
        assert rgb_imgs, "No RGB images in observation"
        rgb = next(iter(rgb_imgs.values()))
        self._log_step_update(images=rgb)

        depth = obs["robot0_robotview"]["images"]["depth"]
        if self.debug:
            Image.fromarray(depth_to_rgb(depth[:, :, 0])).save("depth_image.jpg")

        valid_mask = ~np.isnan(depth[:, :, 0])
        _box, mask, _ = self._segment(rgb, object_name)
        idxs = np.where(mask.flatten()[valid_mask.flatten()].astype(bool))

        self._log_step(
            "Point Cloud + OBB",
            "Computing oriented bounding box from depth point cloud …",
        )
        points, colors = depth_color_to_pointcloud(
            depth[:, :, 0], rgb, obs["robot0_robotview"]["intrinsics"]
        )
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points[idxs])
        pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
        obb = pcd.get_oriented_bounding_box()

        self._env.cube_center = obb.center
        self._env.cube_rot = obb.R
        self._env.cube_points = points[idxs]
        self._env.cube_color = colors[idxs]

        cam_tf = self._camera_to_world(obs)
        obb_tf_world = cam_tf @ vtf.SE3.from_rotation_and_translation(
            rotation=vtf.SO3.from_matrix(obb.R), translation=obb.center
        )
        self._log_step_update(
            text=(
                f"Position: {np.array2string(obb_tf_world.wxyz_xyz[-3:], precision=4)} "
                f"({time.time() - t0:.1f}s)"
            )
        )
        if return_bbox_extent:
            return (
                obb_tf_world.wxyz_xyz[-3:],
                obb_tf_world.wxyz_xyz[:4],
                obb.extent,
            )
        return obb_tf_world.wxyz_xyz[-3:], obb_tf_world.wxyz_xyz[:4], None

    def _plan_grasp_pose(self, object_name: str) -> tuple[dict[str, Any], int, float]:
        """Run segmentation + GraspNet and cache grasps on env."""
        obs = self._env.get_observation()
        rgb = next(iter(obs_get_rgb(obs).values()))
        depth = obs["robot0_robotview"]["images"]["depth"]
        valid_mask = ~np.isnan(depth[:, :, 0])
        _box, mask, _ = self._segment(rgb, object_name)
        segmentation = mask[:, :, None]

        idxs = np.where(segmentation.flatten()[valid_mask.flatten()].astype(bool))
        points, colors = depth_color_to_pointcloud(
            depth[:, :, 0], rgb, obs["robot0_robotview"]["intrinsics"]
        )
        self._env.cube_center = None
        self._env.cube_rot = None
        self._env.cube_points = points[idxs]
        self._env.cube_color = colors[idxs]

        grasp_sample, grasp_scores, grasp_contact_pts = self.grasp_net_plan_fn(
            depth[:, :, 0],
            obs["robot0_robotview"]["intrinsics"],
            segmentation[:, :, 0],
            1,
        )
        grasp_scores = np.asarray(grasp_scores, dtype=np.float64).reshape(-1)
        if grasp_scores.size == 0:
            raise RuntimeError("Contact-GraspNet returned 0 grasps.")

        # Convert from Contact-GraspNet's Panda-template frame (x = open axis)
        # to PIPER's gripper TCP frame (y = open axis). Rotates each grasp
        # around its own approach axis; translation and approach direction are
        # untouched. Override the angle via PIPER_GRASP_ROT_Z_DEG if needed.
        grasp_sample = np.asarray(grasp_sample, dtype=np.float64)
        grasp_sample = grasp_sample @ _GRASP_FRAME_FIX

        best_idx = int(np.argmax(grasp_scores))

        self._env.grasp_sample = grasp_sample
        self._env.grasp_scores = grasp_scores
        self._env.grasp_contact_pts = grasp_contact_pts
        self._env.grasp_last_object_name = object_name
        return obs, best_idx, float(grasp_scores[best_idx])

    def sample_grasp_pose(self, object_name: str) -> tuple[np.ndarray, np.ndarray]:
        """Sample a grasp pose in world coordinates.

        The highest-scoring Contact-GraspNet grasp is selected automatically.
        """
        if self.grasp_net_plan_fn is None:
            pos, _quat, _ = self.get_object_pose(object_name)
            top_down_wxyz = np.array([0.0, 0.0, 1.0, 0.0], dtype=np.float64)
            print(
                "[piper_api] Using top-down fallback grasp "
                "(install --extra contactgraspnet for learned grasps)."
            )
            return pos, top_down_wxyz

        self._log_step("sample_grasp_pose", f"Planning grasp for **'{object_name}'** …")
        t0 = time.time()
        obs, best_idx, score = self._plan_grasp_pose(object_name)
        grasp_tf_cam = vtf.SE3.from_matrix(
            self._env.grasp_sample[best_idx]
        ) @ vtf.SE3.from_translation(np.array([0.0, 0.0, GRASP_TCP_OFFSET_M]))
        self._env.grasp_sample_tf = grasp_tf_cam
        grasp_world = self._camera_to_world(obs) @ grasp_tf_cam
        if hasattr(self._env, "_update_viser_server") and getattr(self._env, "viser_server", None) is not None:
            self._env._update_viser_server(force_scene=True)

        self._log_step_update(
            text=(
                f"{len(self._env.grasp_scores)} total grasps; "
                f"selected highest-score grasp (src={best_idx}, score={score:.3f}) "
                f"({time.time() - t0:.1f}s)"
            )
        )
        return grasp_world.wxyz_xyz[-3:], grasp_world.wxyz_xyz[:4]

    # ------------------------------------------------------------------ #
    def _ik_joints(
        self,
        quat_wxyz: np.ndarray,
        pos: np.ndarray,
        prev_cfg: np.ndarray | None = None,
    ) -> np.ndarray:
        """Run IK and return first 6 arm joints."""
        ik_prev_cfg = self.cfg if prev_cfg is None else prev_cfg
        if ik_prev_cfg is None:
            self.cfg = self.ik_solve_fn(
                target_pose_wxyz_xyz=np.concatenate([quat_wxyz, pos]),
            )
        else:
            self.cfg = self.ik_solve_fn(
                target_pose_wxyz_xyz=np.concatenate([quat_wxyz, pos]),
                prev_cfg=ik_prev_cfg,
            )
        return np.asarray(self.cfg[:6], dtype=np.float64).reshape(6)

    def goto_pose(
        self,
        position: np.ndarray,
        quaternion_wxyz: np.ndarray,
        z_approach: float = 0.0,
    ) -> None:
        """Go to a target EEF pose via planner-generated smooth trajectory.

        Args:
            position: (3,) XYZ in metres (world frame).
            quaternion_wxyz: (4,) WXYZ unit quaternion.
            z_approach: If > 0, inserts an intermediate approach pose
              `position + z_approach` along EEF -Z before final descent.
              The final planned trajectory is still fixed to N waypoints.
        """
        from scipy.spatial.transform import Rotation as SciRotation

        pos = np.asarray(position, dtype=np.float64).reshape(3)
        quat_wxyz = np.asarray(quaternion_wxyz, dtype=np.float64).reshape(4)
        if pos[2] < self._min_target_z:
            old_z = float(pos[2])
            pos[2] = self._min_target_z
            self._log_step(
                "goto_pose",
                f"Input z={old_z:.4f} < {self._min_target_z:.4f}; "
                f"clamped to safety floor z={self._min_target_z:.4f}.",
            )
        # Expose IK target to the low-level env's Viser viz.
        self._env.last_ik_target = (pos.copy(), quat_wxyz.copy())
        self._log_step(
            "goto_pose",
            f"Moving to {np.array2string(pos, precision=4)} "
            f"(z_approach={z_approach:.3f})",
        )

        quat_xyzw = np.array(
            [quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]], dtype=np.float64
        )
        rot = SciRotation.from_quat(quat_xyzw)
        offset_pos = pos + rot.apply(self._TCP_OFFSET)

        current_joints = np.asarray(
            self._env.get_observation()["robot_joint_pos"][:6], dtype=np.float64
        )
        start_pose = self.fk_target_pose_fn(current_joints)

        planner_waypoints: list[np.ndarray] = [start_pose]
        if z_approach != 0.0:
            approach_pos = offset_pos + rot.apply(np.array([0, 0, -z_approach]))
            planner_waypoints.append(np.concatenate([quat_wxyz, approach_pos]))
        planner_waypoints.append(np.concatenate([quat_wxyz, offset_pos]))
        waypoints_arr = np.asarray(planner_waypoints, dtype=np.float64)

        timesteps = int(max(2, self._planner_timesteps))
        cmd_traj = self.pose_plan_fn(waypoints_arr, timesteps=timesteps)
        if hasattr(self._env, "execute_joint_trajectory"):
            self._env.execute_joint_trajectory(
                cmd_traj,
                command_dt=0.02,
                final_tolerance=0.02,
                final_max_steps=260,
            )
        else:
            # Fallback for envs without streamed trajectory execution.
            for joints in cmd_traj[1:]:
                self._env.move_to_joints_blocking(
                    joints,
                    tolerance=0.02,
                    max_steps=260,
                )
        # Do not write a 6-DoF arm-only vector into self.cfg, since IK solvers
        # may expect full robot actuated-joint dimensionality for warm starts.
        self.cfg = None

        self._log_step_update(
            text=(
                f"Motion complete. planner waypoints={int(cmd_traj.shape[0])}, "
                f"fixed_timesteps={timesteps}"
            )
        )

    def go_home(self) -> None:
        """Move the arm back to its joint zero / rest pose (all 6 joints = 0 rad).

        Use this whenever the task asks for the arm to "return to home", "reset",
        or "回零". Do NOT improvise a cartesian zero pose with goto_pose — the
        canonical home is defined in joint space (PiperRealLowLevel.HOME_JOINTS_RAD)
        and a cartesian approximation can leave the arm in a different IK branch.
        """
        self._log_step("go_home", "Returning to joint-zero home pose …")
        if hasattr(self._env, "goto_home_blocking"):
            self._env.goto_home_blocking()
        else:
            home = np.zeros(6, dtype=np.float64)
            self._env.move_to_joints_blocking(home)
        self.cfg = None
        self._env.last_ik_target = None
        self._log_step_update(text="Home pose reached.")

    def open_gripper(self) -> None:
        """Open the gripper fully."""
        self._log_step("open_gripper", "Opening gripper …")
        _open_gripper(self._env, steps=15)
        self._log_step_update(text="Gripper opened.")

    def close_gripper(self) -> None:
        """Close the gripper fully."""
        self._log_step("close_gripper", "Closing gripper …")
        _close_gripper(self._env, steps=15)
        self._log_step_update(text="Gripper closed.")


__all__ = ["PiperControlApi"]
