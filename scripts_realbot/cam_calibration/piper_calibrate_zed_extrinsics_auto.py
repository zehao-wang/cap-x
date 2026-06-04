#!/usr/bin/env python3
"""Automatic PIPER/ZED eye-to-hand calibration from editable joint waypoints.

Workflow:
  1. Load one persistent waypoint JSON. Buttons can override any waypoint with
     the current robot joints; unreachable waypoints are also auto-overwritten
     with the actual reached joints.
  2. Normal mode starts ZED immediately and shows a continuous live preview in
     viser. Dry-run skips ZED and only visits the waypoints.
  3. Start auto calibration moves through waypoints, waits for the arm to
     settle, detects the checkerboard/ChArUco board, solves eye-to-hand, writes
     the ZED extrinsics YAML, and dumps a separate metrics JSON.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np
import yaml
from scipy.spatial.transform import Rotation as SciRotation

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from capx.envs.simulators.piper_real import _Zed2iBridge  # noqa: E402
from capx.envs.simulators.piper.zed_service_client import _ZedServiceClient  # noqa: E402
from capx.envs.simulators.piper.common import (  # noqa: E402
    PIPER_GRIPPER_CONTACT_EFFORT,
    PIPER_GRIPPER_EFFORT,
    PIPER_GRIPPER_RANGE_M,
    PIPER_GRIPPER_STATUS,
    RAD_TO_RAW_MDEG,
    RAW_MDEG_TO_RAD,
)
from capx.integrations.piper.control import (  # noqa: E402
    DEFAULT_PIPER_EEF_LINK,
    DEFAULT_PIPER_URDF,
    _init_piper_pose_planner,
    _resolve_piper_mesh,
)
from scripts_realbot.cam_calibration.piper_calibrate_zed_extrinsics import (  # noqa: E402
    detect_charuco,
    detect_checkerboard,
    grab_board,
    read_piper_pose,
)

DEFAULT_WAYPOINTS_DEG: list[tuple[str, list[float]]] = [
    ("pose_01", [0.0, 18.0, -28.0, 0.0, 38.0, 0.0]),
    ("pose_02", [12.0, 20.0, -32.0, 18.0, 36.0, -22.0]),
    ("pose_03", [-12.0, 20.0, -32.0, -18.0, 36.0, 22.0]),
    ("pose_04", [22.0, 14.0, -24.0, 34.0, 44.0, -34.0]),
    ("pose_05", [-22.0, 14.0, -24.0, -34.0, 44.0, 34.0]),
    ("pose_06", [0.0, 30.0, -42.0, 45.0, 32.0, -48.0]),
    ("pose_07", [0.0, 30.0, -42.0, -45.0, 32.0, 48.0]),
    ("pose_08", [18.0, 28.0, -48.0, 62.0, 42.0, 18.0]),
    ("pose_09", [-18.0, 28.0, -48.0, -62.0, 42.0, -18.0]),
    ("pose_10", [8.0, 38.0, -54.0, 24.0, 26.0, 62.0]),
    ("pose_11", [-8.0, 38.0, -54.0, -24.0, 26.0, -62.0]),
    ("pose_12", [0.0, 16.0, -28.0, 88.0, 40.0, 0.0]),
]

ZED2_HD2K_WIDTH = 2208
ZED2_HD2K_HEIGHT = 1242
ZED2_HD2K_MAX_FPS = 15


@dataclass
class Waypoint:
    name: str
    joints_rad: np.ndarray


@dataclass
class Capture:
    waypoint_index: int
    waypoint_name: str
    R_g2b: np.ndarray
    t_g2b: np.ndarray
    R_t2c: np.ndarray
    t_t2c: np.ndarray
    rgb: np.ndarray | None


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with open(tmp_path, "w") as f:
        json.dump(payload, f, indent=2)
        f.write("\n")
    os.replace(tmp_path, path)


def _default_waypoints() -> list[Waypoint]:
    return [
        Waypoint(name, np.deg2rad(np.asarray(joints_deg, dtype=np.float64)))
        for name, joints_deg in DEFAULT_WAYPOINTS_DEG
    ]


def _load_waypoints(path: Path) -> list[Waypoint]:
    if not path.exists():
        return _default_waypoints()
    with open(path) as f:
        data = json.load(f)
    raw_poses = data.get("poses")
    if not isinstance(raw_poses, list) or not raw_poses:
        raise ValueError(f"{path} must contain a non-empty 'poses' list")

    waypoints: list[Waypoint] = []
    for i, item in enumerate(raw_poses):
        if not isinstance(item, dict):
            raise ValueError(f"pose #{i + 1} in {path} is not an object")
        name = str(item.get("name", f"pose_{i + 1:02d}"))
        joints = np.asarray(item.get("joints"), dtype=np.float64).reshape(-1)
        if joints.shape != (6,):
            raise ValueError(f"{name} in {path} must have exactly 6 joint values")
        waypoints.append(Waypoint(name, joints))
    return waypoints


def _save_waypoints(path: Path, waypoints: list[Waypoint]) -> None:
    _write_json_atomic(
        path,
        {
            "version": 1,
            "poses": [
                {"name": wp.name, "joints": [float(v) for v in wp.joints_rad]}
                for wp in waypoints
            ],
        },
    )


def _wxyz_from_R(R: np.ndarray) -> np.ndarray:
    xyzw = SciRotation.from_matrix(R).as_quat()
    return np.array([xyzw[3], xyzw[0], xyzw[1], xyzw[2]], dtype=np.float64)


def _zed_capture_config() -> tuple[int, int, int]:
    width = int(os.environ.get("PIPER_ZED_WIDTH", str(ZED2_HD2K_WIDTH)))
    height = int(os.environ.get("PIPER_ZED_HEIGHT", str(ZED2_HD2K_HEIGHT)))
    fps = int(os.environ.get("PIPER_ZED_FPS", str(ZED2_HD2K_MAX_FPS)))
    return width, height, fps


class PiperDriver:
    MOTION_SPEED = 30

    def __init__(self, can_interface: str, can_channel: str, can_bitrate: int) -> None:
        from piper_sdk import C_PiperInterface_V2  # type: ignore

        self._lock = threading.RLock()
        if can_interface == "gs_usb":
            self._piper = C_PiperInterface_V2(
                can_name=can_channel, judge_flag=False, can_auto_init=False
            )
            self._piper.CreateCanBus(
                can_name=can_channel,
                bustype="gs_usb",
                expected_bitrate=can_bitrate,
                judge_flag=False,
            )
            self._piper.ConnectPort(can_init=False)
        else:
            self._piper = C_PiperInterface_V2(
                can_name=can_channel, judge_flag=True, can_auto_init=True
            )
            self._piper.ConnectPort()

        for _ in range(100):
            with self._lock:
                if self._piper.EnablePiper():
                    joints = self.read_joints_rad()
                    print(
                        "[auto-calib] Piper control ready; joints(deg)="
                        f"{np.round(np.degrees(joints), 1).tolist()}"
                    )
                    return
            time.sleep(0.05)
        raise RuntimeError("EnablePiper did not confirm within 5s")

    def read_joints_rad(self) -> np.ndarray:
        with self._lock:
            js = self._piper.GetArmJointMsgs().joint_state
        return np.array(
            [js.joint_1, js.joint_2, js.joint_3, js.joint_4, js.joint_5, js.joint_6],
            dtype=np.float64,
        ) * RAW_MDEG_TO_RAD

    def read_end_pose(self) -> tuple[np.ndarray, np.ndarray]:
        with self._lock:
            return read_piper_pose(self._piper)

    def read_gripper_um(self) -> tuple[int, int]:
        with self._lock:
            fb = self._piper.GetArmGripperMsgs().gripper_state
        return int(fb.grippers_angle), int(fb.grippers_effort)

    def move_to_joints(self, joints_rad: np.ndarray, *, tolerance: float,
                       max_steps: int, command_dt: float = 0.01) -> bool:
        target = np.asarray(joints_rad, dtype=np.float64).reshape(6)
        target_raw = (target * RAD_TO_RAW_MDEG).astype(np.int64).tolist()
        for _ in range(max_steps):
            with self._lock:
                self._piper.MotionCtrl_2(0x01, 0x01, self.MOTION_SPEED, 0x00)
                self._piper.JointCtrl(*target_raw)
            if np.max(np.abs(self.read_joints_rad() - target)) < tolerance:
                return True
            time.sleep(command_dt)
        return False

    def _step_gripper(self, target_um: int) -> None:
        target_raw = (self.read_joints_rad() * RAD_TO_RAW_MDEG).astype(np.int64).tolist()
        with self._lock:
            self._piper.MotionCtrl_2(0x01, 0x01, self.MOTION_SPEED, 0x00)
            self._piper.JointCtrl(*target_raw)
            self._piper.GripperCtrl(
                int(abs(target_um)), PIPER_GRIPPER_EFFORT, PIPER_GRIPPER_STATUS, 0
            )

    def open_gripper(self, *, steps: int = 50) -> bool:
        target_um = int(PIPER_GRIPPER_RANGE_M * 1e6)
        for _ in range(steps):
            self._step_gripper(target_um)
            try:
                angle_um, _ = self.read_gripper_um()
                if angle_um >= target_um - 5000:
                    return True
            except Exception:
                pass
            time.sleep(0.03)
        try:
            angle_um, _ = self.read_gripper_um()
            return angle_um >= target_um - 5000
        except Exception:
            return False

    def close_gripper_on_contact(self, *, hold_seconds: float = 2.0) -> bool:
        deadline = time.time() + hold_seconds + 4.0
        latched_um: int | None = None
        while time.time() < deadline:
            try:
                angle_um, effort = self.read_gripper_um()
            except Exception:
                angle_um, effort = 0, 0
            if latched_um is None and abs(effort) >= PIPER_GRIPPER_CONTACT_EFFORT:
                latched_um = angle_um
                print(
                    "[auto-calib] Gripper contact at "
                    f"{angle_um * 1e-6 * 1000:.1f} mm (effort={abs(effort)}); latching."
                )
            self._step_gripper(latched_um if latched_um is not None else 0)
            time.sleep(0.05)
            if latched_um is not None and time.time() > deadline - hold_seconds:
                return True
        print("[auto-calib] WARNING: gripper closed without contact latch.")
        try:
            angle_um, _ = self.read_gripper_um()
            return angle_um <= 5000
        except Exception:
            return False

    def disconnect(self) -> None:
        try:
            with self._lock:
                self._piper.DisconnectPort()
        except Exception:
            pass


def _start_zed(args) -> tuple[Any, np.ndarray, np.ndarray, Callable]:
    if args.zed_source == "service":
        zed = _ZedServiceClient(socket_path=args.zed_socket)
        print(f"[auto-calib] Connecting to ZED service at {args.zed_socket} ...")
        zed.start()
        print(f"[auto-calib] ZED service ready at {zed.width}x{zed.height}")
    else:
        zed_width, zed_height, zed_fps = _zed_capture_config()
        zed = _Zed2iBridge(
            bridge_script=os.environ.get(
                "PIPER_ZED_BRIDGE",
                str(Path.home() / "Documents/Projects/robodata_Agilex/camera/zed_bridge.py"),
            ),
            bridge_python=os.environ.get("ZED_BRIDGE_PYTHON", sys.executable),
            fps=zed_fps,
            width=zed_width,
            height=zed_height,
            use_depth=False,
        )
        print(f"[auto-calib] Starting ZED bridge at {zed_width}x{zed_height}@{zed_fps} ...")
        zed.start()
    K = zed.intrinsics_matrix()
    D = np.zeros(5, dtype=np.float64)
    print(f"[auto-calib] ZED intrinsics: fx={K[0,0]:.1f} fy={K[1,1]:.1f} "
          f"cx={K[0,2]:.1f} cy={K[1,2]:.1f}")
    if args.board == "checkerboard":
        detect_fn = lambda rgb: detect_checkerboard(rgb, args.cols, args.rows, args.square, K, D)
        print(f"[auto-calib] Checkerboard: {args.cols}x{args.rows} inner corners, "
              f"square={args.square * 1000:.1f}mm")
    else:
        detect_fn = lambda rgb: detect_charuco(
            rgb, args.cols, args.rows, args.square, args.marker, K, D
        )
        print(f"[auto-calib] ChArUco: {args.cols}x{args.rows} squares, "
              f"square={args.square * 1000:.1f}mm, marker={args.marker * 1000:.1f}mm")
    return zed, K, D, detect_fn


def _launch_viser(port: int, urdf_path: str):
    import viser  # type: ignore
    import yourdfpy  # type: ignore
    from viser.extras import ViserUrdf  # type: ignore

    server = viser.ViserServer(port=port)
    urdf_vis = None
    num_actuated = 0
    if urdf_path and os.path.exists(urdf_path):
        urdf = yourdfpy.URDF.load(
            urdf_path,
            filename_handler=_resolve_piper_mesh,
            build_collision_scene_graph=False,
            load_collision_meshes=False,
        )
        urdf_vis = ViserUrdf(server, urdf_or_path=urdf, load_meshes=True)
        num_actuated = len(urdf.actuated_joint_names)

    server.scene.add_frame("/base", axes_length=0.1, axes_radius=0.005)
    zed_width, zed_height, _ = _zed_capture_config()
    img_handle = server.gui.add_image(
        np.zeros((zed_height, zed_width, 3), dtype=np.uint8),
        label="ZED live",
    )
    return server, urdf_vis, num_actuated, img_handle


def _draw_fk_frame(server, path: str, fk_fn, joints_rad: np.ndarray, label: str) -> None:
    pose = np.asarray(fk_fn(joints_rad), dtype=np.float64).reshape(7)
    position = pose[4:]
    server.scene.add_frame(
        path,
        position=position.astype(np.float32),
        wxyz=pose[:4].astype(np.float32),
        axes_length=0.045,
        axes_radius=0.003,
    )
    server.scene.add_label(
        f"{path}/label",
        text=label,
        position=tuple((position + np.array([0, 0, 0.035])).astype(np.float32)),
    )


def _annotate(rgb: np.ndarray, detect_fn, K: np.ndarray, D: np.ndarray):
    R, t = detect_fn(rgb)
    if R is None:
        return rgb
    rvec, _ = cv2.Rodrigues(R)
    return cv2.drawFrameAxes(rgb.copy(), K, D, rvec, np.asarray(t).reshape(3, 1), 0.05, 3)


def _refresh_urdf(stop_evt, drv: PiperDriver, urdf_vis, num_actuated: int):
    while not stop_evt.is_set():
        if urdf_vis is not None:
            try:
                cfg = np.zeros(num_actuated, dtype=np.float64)
                joints = drv.read_joints_rad()
                cfg[: min(6, num_actuated)] = joints[: min(6, num_actuated)]
                urdf_vis.update_cfg(cfg)
            except Exception:
                pass
        time.sleep(0.1)


def _refresh_camera(stop_evt, zed, img_handle):
    while not stop_evt.is_set():
        try:
            rgb, _ = zed.read_frames()
            if rgb is not None:
                img_handle.image = rgb
        except Exception:
            pass
        time.sleep(0.2)


def _solve_and_write(captures: list[Capture], output_path: Path):
    R_g2b = [c.R_g2b for c in captures]
    t_g2b = [c.t_g2b for c in captures]
    R_t2c = [c.R_t2c for c in captures]
    t_t2c = [c.t_t2c for c in captures]
    R_b2g = [R.T for R in R_g2b]
    t_b2g = [-R.T @ t for R, t in zip(R_g2b, t_g2b)]
    R_base_cam, t_base_cam = cv2.calibrateHandEye(
        R_b2g, t_b2g, R_t2c, t_t2c, method=cv2.CALIB_HAND_EYE_PARK,
    )
    t_base_cam = np.asarray(t_base_cam).reshape(3)

    T_base_cam = np.eye(4)
    T_base_cam[:3, :3] = R_base_cam
    T_base_cam[:3, 3] = t_base_cam
    tgt_translations = []
    for c in captures:
        T_bg = np.eye(4)
        T_bg[:3, :3] = c.R_g2b
        T_bg[:3, 3] = c.t_g2b
        T_ct = np.eye(4)
        T_ct[:3, :3] = c.R_t2c
        T_ct[:3, 3] = c.t_t2c
        tgt_translations.append((np.linalg.inv(T_bg) @ T_base_cam @ T_ct)[:3, 3])
    tgt = np.array(tgt_translations)
    residual_std_mm = tgt.std(axis=0) * 1000.0
    per_pose_residual_mm = np.linalg.norm(tgt - tgt.mean(axis=0), axis=1) * 1000.0

    rpy = SciRotation.from_matrix(R_base_cam).as_euler("xyz", degrees=False)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        f.write(
            "# ZED 2i pose in the Agilex PIPER base frame.\n"
            f"# Solved from {len(captures)} auto waypoints using OpenCV "
            "CALIB_HAND_EYE_PARK (eye-to-hand).\n"
            f"# Board-on-gripper residual std (mm): {residual_std_mm.round(1).tolist()}\n"
            "#\n"
            "# position:    [x, y, z] in metres\n"
            "# rpy_radians: [roll, pitch, yaw] extrinsic XYZ, radians\n\n"
        )
        yaml.safe_dump(
            {
                "position": [float(v) for v in t_base_cam],
                "rpy_radians": [float(v) for v in rpy],
            },
            f,
            default_flow_style=None,
            sort_keys=False,
        )
    return R_base_cam, t_base_cam, rpy, residual_std_mm, per_pose_residual_mm


def _draw_calibrated_frustum(server, K, R_base_cam, t_base_cam, zed_wh):
    w, h = zed_wh
    fov_y = 2.0 * float(np.arctan2(0.5 * float(h), float(K[1, 1])))
    server.scene.add_camera_frustum(
        "/calibrated_zed",
        position=t_base_cam.astype(np.float32),
        wxyz=_wxyz_from_R(R_base_cam).astype(np.float32),
        fov=fov_y,
        aspect=float(w) / float(h),
        scale=0.15,
        color=(50, 200, 50),
    )


def _build_metrics(args, captures: list[Capture], pose_results: list[dict[str, Any]],
                   output_path: Path, poses_path: Path, solved=None) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "success": solved is not None,
        "valid_detections": len(captures),
        "total_waypoints": len(pose_results),
        "output_yaml": str(output_path),
        "poses_json": str(poses_path),
        "pose_results": pose_results,
        "parameters": {
            "board": args.board,
            "cols": args.cols,
            "rows": args.rows,
            "square_m": args.square,
            "marker_m": args.marker,
            "settle_sec": args.settle_sec,
            "detect_timeout_sec": args.detect_timeout,
            "joint_tolerance_rad": args.joint_tolerance,
            "max_move_steps": args.max_move_steps,
        },
    }
    if solved is None:
        payload["reason"] = "not_enough_valid_detections"
        return payload

    R_base_cam, t_base_cam, rpy, residual_std_mm, per_pose_residual_mm = solved
    for c, err in zip(captures, per_pose_residual_mm):
        pose_results[c.waypoint_index]["residual_mm"] = float(err)
    payload.update(
        {
            "position_m": [float(v) for v in t_base_cam],
            "rpy_radians": [float(v) for v in rpy],
            "residual_std_xyz_mm": [float(v) for v in residual_std_mm],
            "residual_rms_mm": float(np.sqrt(np.mean(per_pose_residual_mm ** 2))),
            "residual_max_mm": float(np.max(per_pose_residual_mm)),
        }
    )
    return payload


def _print_metrics(captures: list[Capture], metrics: dict[str, Any]) -> None:
    print()
    print("[auto-calib] Calibration metrics")
    print(f"  valid detections : {metrics['valid_detections']}/{metrics['total_waypoints']}")
    print(f"  position (m)     : {[round(v, 4) for v in metrics['position_m']]}")
    print(f"  rpy (rad)        : {[round(v, 4) for v in metrics['rpy_radians']]}")
    print(f"  residual std xyz : {[round(v, 1) for v in metrics['residual_std_xyz_mm']]} mm")
    print(f"  residual rms     : {metrics['residual_rms_mm']:.1f} mm")
    print(f"  residual max     : {metrics['residual_max_mm']:.1f} mm")
    print("  per-pose residual:")
    for c, err in zip(captures, [r["residual_mm"] for r in metrics["pose_results"] if r.get("used")]):
        print(f"    {c.waypoint_index + 1:02d} {c.waypoint_name}: {err:.1f} mm")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--board", choices=["checkerboard", "charuco"], default="checkerboard")
    p.add_argument("--cols", type=int, default=4)
    p.add_argument("--rows", type=int, default=5)
    p.add_argument("--square", type=float, default=0.020)
    p.add_argument("--marker", type=float, default=0.015)
    p.add_argument("--output", default=os.environ.get(
        "PIPER_CAMERA_EXTRINSICS",
        str(ROOT / "env_configs/real/piper_zed_extrinsics.yaml"),
    ))
    p.add_argument("--poses-json", default=str(
        ROOT / "env_configs/real/piper_zed_calibration_poses.json"
    ))
    p.add_argument("--captures-dir", default="")
    p.add_argument(
        "--zed-source",
        choices=["service", "bridge"],
        default=os.environ.get("PIPER_ZED_SOURCE", "service"),
        help="ZED frame source: 'service' (zed_depth_service.py over a Unix socket) "
        "or 'bridge' (legacy subprocess zed_bridge.py). Default: service.",
    )
    p.add_argument(
        "--zed-socket",
        default=os.environ.get("PIPER_ZED_SERVICE_SOCKET", "/tmp/piper/zed.sock"),
        help="Unix socket path for the ZED service (when --zed-source=service).",
    )
    p.add_argument("--can-interface", default=os.environ.get("PIPER_CAN_INTERFACE", "socketcan"))
    p.add_argument("--can-channel", default=os.environ.get("PIPER_CAN_CHANNEL", "can0"))
    p.add_argument("--can-bitrate", type=int, default=int(os.environ.get("PIPER_CAN_BITRATE", "1000000")))
    p.add_argument("--viser-port", type=int, default=8201)
    p.add_argument("--detect-timeout", type=float, default=2.5)
    p.add_argument("--settle-sec", type=float, default=2.0)
    p.add_argument("--joint-tolerance", type=float, default=0.025)
    p.add_argument("--max-move-steps", type=int, default=600)
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    poses_path = Path(args.poses_json)
    output_path = Path(args.output)
    metrics_path = poses_path.with_name("piper_zed_calibration_metrics.json")

    try:
        waypoints = _load_waypoints(poses_path)
        source = str(poses_path) if poses_path.exists() else "built-in defaults"
    except Exception as e:
        print(f"[auto-calib] Failed to load {poses_path}: {e}")
        waypoints = _default_waypoints()
        source = "built-in defaults"
    print(f"[auto-calib] Loaded {len(waypoints)} waypoints from {source}")

    urdf_path = os.environ.get("PIPER_URDF_PATH", DEFAULT_PIPER_URDF)
    print("[auto-calib] Building FK helper for waypoint EEF visualization ...")
    _, fk_fn = _init_piper_pose_planner(urdf_path, DEFAULT_PIPER_EEF_LINK)

    drv = None
    zed = None
    try:
        print(f"[auto-calib] Connecting Piper via {args.can_interface}/{args.can_channel}")
        drv = PiperDriver(args.can_interface, args.can_channel, args.can_bitrate)
        zed, K, D, detect_fn = (None, None, None, None) if args.dry_run else _start_zed(args)
        server, urdf_vis, num_actuated, img_handle = _launch_viser(args.viser_port, urdf_path)
    except Exception as e:
        print(f"[auto-calib] FATAL: {type(e).__name__}: {e}")
        if zed is not None:
            try:
                zed.stop()
            except Exception:
                pass
        if drv is not None:
            drv.disconnect()
        return

    print(f"[viser] Open http://localhost:{args.viser_port}")
    captures_dir = Path(args.captures_dir) if args.captures_dir else None
    if captures_dir:
        captures_dir.mkdir(parents=True, exist_ok=True)

    stop_evt = threading.Event()
    threading.Thread(target=_refresh_urdf, args=(stop_evt, drv, urdf_vis, num_actuated), daemon=True).start()
    if zed is not None:
        threading.Thread(target=_refresh_camera, args=(stop_evt, zed, img_handle), daemon=True).start()

    state_lock = threading.Lock()
    running_lock = threading.Lock()
    running = False

    status_md = server.gui.add_markdown("**Status:** ready.")
    progress_md = server.gui.add_markdown(
        f"**Waypoint JSON:** `{poses_path}`\n\n"
        f"**Calibration output:** `{output_path}`\n\n"
        f"**Mode:** `{'dry-run' if args.dry_run else 'calibration'}`"
    )

    def set_status(text: str) -> None:
        print(f"[auto-calib] {text}")
        status_md.content = f"**Status:** {text}"

    def override_waypoint(idx: int, joints: np.ndarray) -> str:
        with state_lock:
            waypoints[idx].joints_rad = joints.copy()
            name = waypoints[idx].name
            _save_waypoints(poses_path, waypoints)
        _draw_fk_frame(server, f"/waypoints/{idx:02d}", fk_fn, joints, f"{idx + 1:02d} {name}")
        return name

    for i, wp in enumerate(waypoints):
        _draw_fk_frame(server, f"/waypoints/{i:02d}", fk_fn, wp.joints_rad, f"{i + 1:02d} {wp.name}")

    @server.gui.add_button("Open gripper").on_click
    def _(_):
        set_status("opening gripper ...")
        set_status("gripper opened." if drv.open_gripper() else "open gripper was not confirmed.")

    @server.gui.add_button("Close gripper").on_click
    def _(_):
        set_status("closing gripper with contact latch ...")
        ok = drv.close_gripper_on_contact(hold_seconds=1.0)
        set_status("gripper closed." if ok else "close gripper was not confirmed.")

    @server.gui.add_button("Return to 0 pose").on_click
    def _(_):
        set_status("returning to 0 pose ...")
        reached = drv.move_to_joints(
            np.zeros(6, dtype=np.float64),
            tolerance=args.joint_tolerance,
            max_steps=args.max_move_steps,
        )
        set_status("returned to 0 pose." if reached else "0 pose was not reached.")

    if zed is not None and hasattr(zed, "set_auto_exposure_gain"):
        with server.gui.add_folder("Camera (ZED)"):
            auto_cb = server.gui.add_checkbox("Auto exposure / gain", initial_value=True)
            exposure_slider = server.gui.add_slider(
                "Exposure %", min=0, max=100, step=1, initial_value=50, disabled=True
            )
            gain_slider = server.gui.add_slider(
                "Gain %", min=0, max=100, step=1, initial_value=50, disabled=True
            )

        @auto_cb.on_update
        def _(_):
            if auto_cb.value:
                zed.set_auto_exposure_gain()
                exposure_slider.disabled = True
                gain_slider.disabled = True
                set_status("camera: auto exposure + gain.")
            else:
                exposure_slider.disabled = False
                gain_slider.disabled = False
                zed.set_exposure(int(exposure_slider.value))
                zed.set_gain(int(gain_slider.value))
                set_status(
                    f"camera: manual exposure={int(exposure_slider.value)} "
                    f"gain={int(gain_slider.value)}."
                )

        @exposure_slider.on_update
        def _(_):
            if not auto_cb.value:
                zed.set_exposure(int(exposure_slider.value))

        @gain_slider.on_update
        def _(_):
            if not auto_cb.value:
                zed.set_gain(int(gain_slider.value))

    for i, wp in enumerate(waypoints):
        btn = server.gui.add_button(f"Override {i + 1:02d}: {wp.name}")

        @btn.on_click
        def _(_, idx=i):
            try:
                name = override_waypoint(idx, drv.read_joints_rad())
                set_status(f"overrode {idx + 1:02d} `{name}` and saved JSON.")
            except Exception as e:
                set_status(f"failed to override waypoint {idx + 1}: {type(e).__name__}: {e}")

    def run_sequence() -> None:
        nonlocal running
        with running_lock:
            if running:
                set_status("sequence is already running.")
                return
            running = True
        try:
            with state_lock:
                run_waypoints = [Waypoint(wp.name, wp.joints_rad.copy()) for wp in waypoints]
            captures: list[Capture] = []
            pose_results = [
                {"index": i, "name": wp.name, "used": False, "reason": "not_attempted"}
                for i, wp in enumerate(run_waypoints)
            ]
            set_status(f"starting {'dry run' if args.dry_run else 'auto calibration'} on {len(run_waypoints)} waypoints.")

            for idx, wp in enumerate(run_waypoints):
                progress_md.content = f"**Running:** {idx + 1}/{len(run_waypoints)} `{wp.name}`"
                set_status(f"moving to {idx + 1:02d} `{wp.name}` ...")
                reached = drv.move_to_joints(
                    wp.joints_rad,
                    tolerance=args.joint_tolerance,
                    max_steps=args.max_move_steps,
                )
                if not reached:
                    actual = drv.read_joints_rad()
                    name = override_waypoint(idx, actual)
                    wp.joints_rad = actual.copy()
                    set_status(f"warning: {idx + 1:02d} `{name}` not reached; saved actual joints.")

                set_status(f"settling at {idx + 1:02d} `{wp.name}` for {args.settle_sec:.1f}s ...")
                time.sleep(args.settle_sec)

                actual = drv.read_joints_rad()
                _draw_fk_frame(
                    server,
                    f"/{'dry_run' if args.dry_run else 'captures'}/pose_{idx:02d}",
                    fk_fn,
                    actual,
                    f"{idx + 1:02d} {wp.name}",
                )
                if args.dry_run:
                    pose_results[idx] = {"index": idx, "name": wp.name, "used": True, "reason": "dry_run"}
                    continue

                assert zed is not None and detect_fn is not None and K is not None and D is not None
                set_status(f"detecting board at {idx + 1:02d} `{wp.name}` ...")
                R_bd, t_bd, rgb = grab_board(zed, detect_fn, timeout=args.detect_timeout)
                if R_bd is None:
                    pose_results[idx] = {"index": idx, "name": wp.name, "used": False, "reason": "board_not_detected"}
                    set_status(f"board not detected at {idx + 1:02d} `{wp.name}`; skipping.")
                    continue
                try:
                    R_gb, t_gb = drv.read_end_pose()
                except Exception as e:
                    pose_results[idx] = {
                        "index": idx,
                        "name": wp.name,
                        "used": False,
                        "reason": "end_pose_read_failed",
                        "error": f"{type(e).__name__}: {e}",
                    }
                    continue

                capture = Capture(idx, wp.name, R_gb, t_gb, R_bd, t_bd, rgb)
                captures.append(capture)
                pose_results[idx] = {
                    "index": idx,
                    "name": wp.name,
                    "used": True,
                    "capture_index": len(captures) - 1,
                    "gripper_xyz_m": [float(v) for v in t_gb],
                    "board_xyz_in_cam_m": [float(v) for v in t_bd],
                }
                if rgb is not None:
                    img_handle.image = _annotate(rgb, detect_fn, K, D)
                    if captures_dir:
                        import imageio.v3 as iio  # noqa: E402
                        iio.imwrite(str(captures_dir / f"capture_{len(captures)-1:03d}.png"), rgb)
                set_status(f"captured {len(captures)} from {idx + 1:02d} `{wp.name}`.")

            if args.dry_run:
                progress_md.content = f"**Result:** dry-run complete. Visited `{len(run_waypoints)}` waypoints."
                set_status("dry-run complete.")
                return

            if len(captures) < 3:
                metrics = _build_metrics(args, captures, pose_results, output_path, poses_path)
                _write_json_atomic(metrics_path, metrics)
                progress_md.content = f"**Result:** failed. Valid detections: `{len(captures)}/{len(run_waypoints)}`."
                set_status(f"only {len(captures)} valid detections; need at least 3.")
                return

            set_status(f"solving calibration from {len(captures)} valid detections ...")
            solved = _solve_and_write(captures, output_path)
            metrics = _build_metrics(args, captures, pose_results, output_path, poses_path, solved)
            _write_json_atomic(metrics_path, metrics)
            _print_metrics(captures, metrics)
            print(f"  metrics json     : {metrics_path}")
            _draw_calibrated_frustum(server, K, solved[0], solved[1], (zed.width, zed.height))

            warn = " Residual is high (>20 mm)." if max(metrics["residual_std_xyz_mm"]) > 20.0 else ""
            progress_md.content = (
                f"**Result:** saved `{output_path}`\n\n"
                f"metrics: `{metrics_path}`\n\n"
                f"valid detections: `{len(captures)}/{len(run_waypoints)}`\n\n"
                f"residual std mm: `{[round(v, 1) for v in metrics['residual_std_xyz_mm']]}`\n\n"
                f"residual rms/max mm: `{metrics['residual_rms_mm']:.1f} / {metrics['residual_max_mm']:.1f}`"
            )
            set_status(f"solved and saved calibration.{warn}")
        finally:
            with running_lock:
                running = False

    btn_start = server.gui.add_button(
        "Start dry run" if args.dry_run else "Start auto calibration"
    )

    @btn_start.on_click
    def _(_):
        threading.Thread(target=run_sequence, daemon=True).start()

    print()
    print("=" * 70)
    print("Automatic calibration ready.")
    print(f"  viser      : http://localhost:{args.viser_port}")
    print(f"  poses JSON : {poses_path}")
    print(f"  output YAML: {output_path}")
    print(f"  mode       : {'dry-run' if args.dry_run else 'calibration'}")
    print("=" * 70)

    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        print("\n[auto-calib] Exiting.")
    finally:
        stop_evt.set()
        if zed is not None:
            try:
                zed.stop()
            except Exception:
                pass
        drv.disconnect()


if __name__ == "__main__":
    main()
