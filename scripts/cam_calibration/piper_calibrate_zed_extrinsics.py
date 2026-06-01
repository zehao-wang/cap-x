#!/usr/bin/env python3
"""Eye-to-hand calibration: solve the ZED 2i pose in the Agilex PIPER base frame.

Setup
-----
1. Tape/clamp your printed calibration board rigidly to the gripper face so it
   moves with the arm. A checkerboard PDF is fine; a ChArUco board is more
   robust.
2. Power on the arm and ZED. Make sure `PIPER_ZED_BRIDGE`, `ZED_BRIDGE_PYTHON`,
   socketcan (`can0`) are ready.
3. Position the arm so the board is clearly in ZED's view. You'll then move
   the arm through ~12 poses — use both translation AND orientation changes
   (tilt/roll the board in different directions). Poor rotational diversity
   is the #1 reason hand-eye calibration comes out wrong.

Capture the poses
-----------------
After each arm pose, press ENTER in this script. It'll:
  - grab one RGB frame from ZED and detect the board (solvePnP → T_cam_board)
  - read Piper's SDK end pose (T_base_gripper) via `GetArmEndPoseMsgs`
  - store the pair

Solve + write
-------------
Once ≥3 captures are collected (12 recommended), OpenCV's
`cv2.calibrateHandEye` runs in eye-to-hand mode (gripper↔base inputs
inverted) and writes the resulting camera pose to

    env_configs/real/piper_zed_extrinsics.yaml

Example
-------
    uv run --no-sync --active scripts/cam_calibration/piper_calibrate_zed_extrinsics.py

Defaults match `generate_calibration_board.py`: a 5×6-square checkerboard
with 20 mm squares (→ 4×5 inner corners, board = 100×120 mm). Override with
`--cols/--rows/--square` for other boards. For ChArUco, `--cols/--rows`
become `squares_x` / `squares_y`.
"""
from __future__ import annotations

import argparse
import os
import sys
import threading
import time
from pathlib import Path

import cv2
import numpy as np
import yaml
from scipy.spatial.transform import Rotation as SciRotation

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from capx.envs.simulators.piper.common import RAW_MDEG_TO_RAD  # noqa: E402
from capx.envs.simulators.piper_real import _Zed2iBridge  # noqa: E402
from capx.integrations.piper.control import (  # noqa: E402
    DEFAULT_PIPER_URDF,
    _resolve_piper_mesh,
)

# PIPER SDK raw-unit conversions (end_pose is in 0.001 mm / 0.001 deg)
RAW_TO_M = 1e-6
RAW_TO_DEG = 1e-3


# --------------------------------------------------------------------------- #
# Board detectors — all return (R_target2cam (3,3), t_target2cam (3,)) or None #
# --------------------------------------------------------------------------- #
def detect_checkerboard(rgb, cols, rows, square, K, D):
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    found, corners = cv2.findChessboardCorners(
        gray,
        (cols, rows),
        flags=cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE,
    )
    if not found:
        return None, None
    corners = cv2.cornerSubPix(
        gray,
        corners,
        (11, 11),
        (-1, -1),
        (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-3),
    )
    objp = np.zeros((cols * rows, 3), np.float32)
    objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2) * square
    ok, rvec, tvec = cv2.solvePnP(objp, corners, K, D)
    if not ok:
        return None, None
    R, _ = cv2.Rodrigues(rvec)
    return R, tvec.reshape(3)


def detect_charuco(rgb, sx, sy, square_len, marker_len, K, D):
    try:
        aruco = cv2.aruco
    except AttributeError as e:
        raise RuntimeError(
            "cv2.aruco is not available. Install `opencv-contrib-python` or "
            "pass --board checkerboard."
        ) from e
    adict = aruco.getPredefinedDictionary(aruco.DICT_5X5_100)
    # New-API CharucoBoard takes (squares_x, squares_y).
    board = aruco.CharucoBoard((sx, sy), square_len, marker_len, adict)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)

    if hasattr(aruco, "CharucoDetector"):  # OpenCV ≥ 4.7
        detector = aruco.CharucoDetector(board)
        ch_corners, ch_ids, _, _ = detector.detectBoard(gray)
        if ch_corners is None or len(ch_corners) < 6:
            return None, None
        obj, img = board.matchImagePoints(ch_corners, ch_ids)
        if obj is None or len(obj) < 6:
            return None, None
        ok, rvec, tvec = cv2.solvePnP(obj, img, K, D)
        if not ok:
            return None, None
    else:  # Legacy API
        marker_corners, marker_ids, _ = aruco.detectMarkers(gray, adict)
        if marker_ids is None or len(marker_ids) == 0:
            return None, None
        ret, ch_corners, ch_ids = aruco.interpolateCornersCharuco(
            marker_corners, marker_ids, gray, board
        )
        if ret is None or ret < 6:
            return None, None
        ok, rvec, tvec = aruco.estimatePoseCharucoBoard(
            ch_corners, ch_ids, board, K, D, None, None
        )
        if not ok:
            return None, None
    R, _ = cv2.Rodrigues(rvec)
    return R, tvec.reshape(3)


# --------------------------------------------------------------------------- #
def read_piper_pose(piper):
    """Return (R_base_gripper (3,3), t_base_gripper (3,)) from the SDK end pose."""
    pose = piper.GetArmEndPoseMsgs().end_pose
    xyz_m = (
        np.array([pose.X_axis, pose.Y_axis, pose.Z_axis], dtype=np.float64)
        * RAW_TO_M
    )
    rxyz_deg = (
        np.array([pose.RX_axis, pose.RY_axis, pose.RZ_axis], dtype=np.float64)
        * RAW_TO_DEG
    )
    R = SciRotation.from_euler("xyz", rxyz_deg, degrees=True).as_matrix()
    return R, xyz_m


def grab_board(zed, detect_fn, timeout=3.0):
    """Keep grabbing frames until the board is detected or timeout."""
    t0 = time.time()
    last_rgb = None
    while time.time() - t0 < timeout:
        rgb, _ = zed.read_frames()
        if rgb is None:
            time.sleep(0.05)
            continue
        last_rgb = rgb
        R, t = detect_fn(rgb)
        if R is not None:
            return R, t, rgb
        time.sleep(0.05)
    return None, None, last_rgb


# --------------------------------------------------------------------------- #
# Viser visualization helpers                                                 #
# --------------------------------------------------------------------------- #
def _wxyz_from_R(R):
    xyzw = SciRotation.from_matrix(R).as_quat()
    return np.array([xyzw[3], xyzw[0], xyzw[1], xyzw[2]], dtype=np.float64)


def _annotate(rgb, detect_fn, K, D, axis_len=0.05):
    """Run the board detector and draw XYZ axes on success. Returns annotated rgb."""
    R, t = detect_fn(rgb)
    if R is None:
        return rgb, None, None
    rvec, _ = cv2.Rodrigues(R)
    annotated = cv2.drawFrameAxes(
        rgb.copy(), K, D, rvec, np.asarray(t).reshape(3, 1), axis_len, 3
    )
    return annotated, R, t


def _launch_viser(port, urdf_path, image_shape):
    """Spin up a viser server, load the PIPER URDF + placeholder image.

    Returns (server, urdf_vis, num_actuated_joints, img_handle, status_md).
    If viser/yourdfpy fail to load, returns (None, ...) so the caller can skip.
    """
    try:
        import viser  # type: ignore
        import yourdfpy  # type: ignore
        from viser.extras import ViserUrdf  # type: ignore
    except Exception as e:  # pragma: no cover
        print(f"[viser] import failed ({e}); visualization disabled.")
        return None, None, 0, None, None

    server = viser.ViserServer(port=port)
    urdf_vis = None
    num_actuated = 0
    if urdf_path and os.path.exists(urdf_path):
        try:
            urdf = yourdfpy.URDF.load(
                urdf_path,
                filename_handler=_resolve_piper_mesh,
                build_collision_scene_graph=False,
                load_collision_meshes=False,
            )
            urdf_vis = ViserUrdf(server, urdf_or_path=urdf, load_meshes=True)
            num_actuated = len(urdf.actuated_joint_names)
        except Exception as e:
            print(f"[viser] URDF load failed ({e}); arm model skipped.")
    else:
        print(f"[viser] URDF not found at {urdf_path}; arm model skipped.")

    server.scene.add_frame("/base", axes_length=0.1, axes_radius=0.005)
    img_handle = server.gui.add_image(
        np.zeros(image_shape, dtype=np.uint8),
        label="ZED (axes drawn when board detected)",
    )
    status_md = server.gui.add_markdown("**Captured:** 0 poses")
    return server, urdf_vis, num_actuated, img_handle, status_md


def _refresh_loop(stop_evt, zed, piper, detect_fn, K, D,
                  urdf_vis, num_actuated, img_handle):
    """Background thread: live ZED preview + arm URDF follow at ~10 Hz."""
    while not stop_evt.is_set():
        rgb, _ = zed.read_frames()
        if rgb is not None and img_handle is not None:
            try:
                annotated, _R, _t = _annotate(rgb, detect_fn, K, D)
                img_handle.image = annotated
            except Exception:
                pass
        if urdf_vis is not None:
            try:
                js = piper.GetArmJointMsgs().joint_state
                joints = np.array(
                    [js.joint_1, js.joint_2, js.joint_3,
                     js.joint_4, js.joint_5, js.joint_6],
                    dtype=np.float64,
                ) * RAW_MDEG_TO_RAD
                cfg = np.zeros(num_actuated, dtype=np.float64)
                cfg[: min(6, num_actuated)] = joints[: min(6, num_actuated)]
                urdf_vis.update_cfg(cfg)
            except Exception:
                pass
        time.sleep(0.1)


# --------------------------------------------------------------------------- #
def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--num-poses", type=int, default=12)
    p.add_argument("--board", choices=["checkerboard", "charuco"], default="checkerboard")
    p.add_argument("--cols", type=int, default=4,
                   help="inner corners along x (checker) OR squares_x (charuco). "
                        "Default 4 = 5-square-wide board (matches generate_calibration_board.py).")
    p.add_argument("--rows", type=int, default=5,
                   help="inner corners along y (checker) OR squares_y (charuco). "
                        "Default 5 = 6-square-long board (matches generate_calibration_board.py).")
    p.add_argument("--square", type=float, default=0.020,
                   help="square side length in metres (default 20mm; "
                        "5x6 squares of 20mm = 100x120mm board)")
    p.add_argument("--marker", type=float, default=0.015,
                   help="ChArUco marker side length in metres (default 15mm)")
    p.add_argument(
        "--output",
        default=os.environ.get(
            "PIPER_CAMERA_EXTRINSICS",
            str(ROOT / "env_configs/real/piper_zed_extrinsics.yaml"),
        ),
    )
    p.add_argument("--captures-dir",
                   default=str(ROOT / "data/calib_debug/zed"),
                   help="Directory for per-pose debug images (annotated on success, "
                        "raw '*_fail.png' on detection failure). Pass empty string to disable.")
    p.add_argument("--can-interface",
                   default=os.environ.get("PIPER_CAN_INTERFACE", "socketcan"))
    p.add_argument("--can-channel",
                   default=os.environ.get("PIPER_CAN_CHANNEL", "can1"))
    p.add_argument("--can-bitrate", type=int,
                   default=int(os.environ.get("PIPER_CAN_BITRATE", "1000000")))
    p.add_argument("--viser-port", type=int, default=8201,
                   help="Port for the viser visualization server (default 8201)")
    p.add_argument("--no-viser", action="store_true",
                   help="Disable viser visualization")
    args = p.parse_args()

    # ---------- Piper (read-only; no motion commands) ------------------- #
    from piper_sdk import C_PiperInterface_V2  # type: ignore
    print(f"[calib] Connecting Piper via {args.can_interface} / {args.can_channel}")
    if args.can_interface == "gs_usb":
        piper = C_PiperInterface_V2(can_name=args.can_channel, judge_flag=False,
                                    can_auto_init=False)
        piper.CreateCanBus(can_name=args.can_channel, bustype="gs_usb",
                           expected_bitrate=args.can_bitrate, judge_flag=False)
        piper.ConnectPort(can_init=False)
    else:
        piper = C_PiperInterface_V2(can_name=args.can_channel, judge_flag=True,
                                    can_auto_init=True)
        piper.ConnectPort()
    time.sleep(0.5)

    # ---------- ZED bridge (RGB only) ----------------------------------- #
    zed = _Zed2iBridge(
        bridge_script=os.environ.get(
            "PIPER_ZED_BRIDGE",
            str(Path.home()
                / "Documents/Projects/robodata_Agilex/camera/zed_bridge.py"),
        ),
        bridge_python=os.environ.get("ZED_BRIDGE_PYTHON", sys.executable),
        fps=int(os.environ.get("PIPER_ZED_FPS", "15")),
        width=int(os.environ.get("PIPER_ZED_WIDTH", "1280")),
        height=int(os.environ.get("PIPER_ZED_HEIGHT", "720")),
        use_depth=False,
    )
    print("[calib] Starting ZED bridge...")
    zed.start()
    K = zed.intrinsics_matrix()
    # ZED VIEW.LEFT is already rectified → distortion ≈ 0
    D = np.zeros(5, dtype=np.float64)
    print(f"[calib] ZED intrinsics: fx={K[0,0]:.1f} fy={K[1,1]:.1f} "
          f"cx={K[0,2]:.1f} cy={K[1,2]:.1f}")

    # ---------- Detector -------------------------------------------------- #
    if args.board == "checkerboard":
        detect_fn = (lambda rgb: detect_checkerboard(
            rgb, args.cols, args.rows, args.square, K, D))
        print(f"[calib] Checkerboard: {args.cols}x{args.rows} inner corners, "
              f"square={args.square*1000:.1f}mm")
    else:
        detect_fn = (lambda rgb: detect_charuco(
            rgb, args.cols, args.rows, args.square, args.marker, K, D))
        print(f"[calib] ChArUco: {args.cols}x{args.rows} squares, "
              f"square={args.square*1000:.1f}mm, marker={args.marker*1000:.1f}mm")

    captures_dir = Path(args.captures_dir) if args.captures_dir else None
    if captures_dir:
        captures_dir.mkdir(parents=True, exist_ok=True)

    # ---------- Viser visualization (optional) --------------------------- #
    stop_evt = threading.Event()
    viser_server = urdf_vis = img_handle = status_md = refresh_th = None
    if not args.no_viser:
        # Wait briefly for first ZED frame so we can size the GUI image correctly.
        first_rgb = None
        t0 = time.time()
        while first_rgb is None and time.time() - t0 < 5.0:
            first_rgb, _ = zed.read_frames()
            if first_rgb is None:
                time.sleep(0.1)
        image_shape = first_rgb.shape if first_rgb is not None else (720, 1280, 3)
        viser_server, urdf_vis, num_actuated, img_handle, status_md = _launch_viser(
            args.viser_port,
            os.environ.get("PIPER_URDF_PATH", DEFAULT_PIPER_URDF),
            image_shape,
        )
        if viser_server is not None:
            print(f"[viser] Open http://localhost:{args.viser_port} to watch.")
            refresh_th = threading.Thread(
                target=_refresh_loop,
                args=(stop_evt, zed, piper, detect_fn, K, D,
                      urdf_vis, num_actuated, img_handle),
                daemon=True,
            )
            refresh_th.start()

    # ---------- Interactive capture loop -------------------------------- #
    R_g2b, t_g2b, R_t2c, t_t2c = [], [], [], []
    print()
    print("Move the arm into a new pose with the board fully visible in the ZED.")
    print("Use BOTH translation and orientation diversity (tilt the board).")
    print(f"Collecting {args.num_poses} poses. Press ENTER to capture, "
          "'s' to skip, 'q' to stop early.\n")
    i = 0
    while i < args.num_poses:
        ans = input(f"[{i+1}/{args.num_poses}] ENTER=capture, s=skip, q=quit: "
                    ).strip().lower()
        if ans == "q":
            break
        if ans == "s":
            continue
        R_bd, t_bd, rgb = grab_board(zed, detect_fn, timeout=2.0)
        if R_bd is None:
            print("  ✗ board not detected — reposition (fully in view, well lit) and retry.")
            if captures_dir and rgb is not None:
                import imageio.v3 as iio  # noqa: E402
                iio.imwrite(str(captures_dir / f"capture_{i:03d}_fail.png"), rgb)
            continue
        R_gb, t_gb = read_piper_pose(piper)
        R_g2b.append(R_gb); t_g2b.append(t_gb)
        R_t2c.append(R_bd); t_t2c.append(t_bd)
        print(f"  ✓ gripper_xyz={np.round(t_gb, 3).tolist()}  "
              f"board_xyz_in_cam={np.round(t_bd, 3).tolist()}")
        if captures_dir and rgb is not None:
            import imageio.v3 as iio  # noqa: E402
            annotated, _R, _t = _annotate(rgb, detect_fn, K, D)
            iio.imwrite(str(captures_dir / f"capture_{i:03d}.png"), annotated)
        # Add the captured gripper pose as a small coordinate frame in the 3D view
        if viser_server is not None:
            try:
                viser_server.scene.add_frame(
                    f"/captures/pose_{i:02d}",
                    position=t_gb.astype(np.float32),
                    wxyz=_wxyz_from_R(R_gb).astype(np.float32),
                    axes_length=0.03,
                    axes_radius=0.003,
                )
                if status_md is not None:
                    status_md.content = f"**Captured:** {i + 1} poses"
            except Exception:
                pass
        i += 1

    n = len(R_g2b)
    if n < 3:
        print(f"[calib] Only {n} valid captures; need ≥3 for calibrateHandEye. Aborting.")
        stop_evt.set()
        if refresh_th is not None:
            refresh_th.join(timeout=1.0)
        zed.stop(); piper.DisconnectPort()
        return

    # ---------- Eye-to-hand solve --------------------------------------- #
    # OpenCV's calibrateHandEye solves the eye-IN-hand problem. For eye-TO-hand
    # (camera fixed, board on gripper) we invert the gripper2base transforms;
    # the returned "cam2gripper" is then interpreted as cam-pose-in-base → T_base_cam.
    R_b2g = [R.T for R in R_g2b]
    t_b2g = [-R.T @ t for R, t in zip(R_g2b, t_g2b)]
    R_base_cam, t_base_cam = cv2.calibrateHandEye(
        R_b2g, t_b2g, R_t2c, t_t2c, method=cv2.CALIB_HAND_EYE_PARK,
    )
    t_base_cam = np.asarray(t_base_cam).reshape(3)

    # ---------- Residual sanity check ----------------------------------- #
    # If the solve is correct, T_gripper_target should be constant across poses.
    T_base_cam = np.eye(4); T_base_cam[:3, :3] = R_base_cam; T_base_cam[:3, 3] = t_base_cam
    tgt_translations = []
    for R_g, t_g, R_t, t_t in zip(R_g2b, t_g2b, R_t2c, t_t2c):
        T_bg = np.eye(4); T_bg[:3, :3] = R_g; T_bg[:3, 3] = t_g
        T_ct = np.eye(4); T_ct[:3, :3] = R_t; T_ct[:3, 3] = t_t
        T_gt = np.linalg.inv(T_bg) @ T_base_cam @ T_ct
        tgt_translations.append(T_gt[:3, 3])
    tgt = np.array(tgt_translations)
    trans_std_mm = tgt.std(axis=0) * 1000.0
    print(f"\n[calib] Residual — board-on-gripper translation std (mm): "
          f"x={trans_std_mm[0]:.1f} y={trans_std_mm[1]:.1f} z={trans_std_mm[2]:.1f}")
    if trans_std_mm.max() > 20.0:
        print("  ⚠ High residual (>20mm std). Re-capture with more rotational "
              "diversity, or check that the board is rigidly fixed to the gripper.")

    # ---------- Write YAML ---------------------------------------------- #
    rpy = SciRotation.from_matrix(R_base_cam).as_euler("xyz", degrees=False)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        f.write(
            "# ZED 2i pose in the Agilex PIPER base frame.\n"
            f"# Solved from {n} poses using OpenCV CALIB_HAND_EYE_PARK "
            f"(eye-to-hand).\n"
            f"# Board-on-gripper residual std (mm): "
            f"{trans_std_mm.round(1).tolist()}\n"
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
    print(f"\n[calib] Wrote {out_path}")
    print(f"  position   : [{t_base_cam[0]:.4f}, {t_base_cam[1]:.4f}, {t_base_cam[2]:.4f}] m")
    print(f"  rpy_radians: [{rpy[0]:+.4f}, {rpy[1]:+.4f}, {rpy[2]:+.4f}] rad")

    # ---------- Visualize the solved camera pose in 3D ------------------ #
    if viser_server is not None:
        try:
            h, w = first_rgb.shape[:2] if first_rgb is not None else (720, 1280)
            fy = float(K[1, 1])
            fov_y = 2.0 * float(np.arctan2(0.5 * float(h), fy))
            aspect = float(w) / float(h)
            viser_server.scene.add_camera_frustum(
                "/calibrated_zed",
                position=t_base_cam.astype(np.float32),
                wxyz=_wxyz_from_R(R_base_cam).astype(np.float32),
                fov=fov_y,
                aspect=aspect,
                scale=0.12,
                color=(50, 200, 50),
            )
            # Also show an image thumbnail in the scene
            last_rgb, _ = zed.read_frames()
            if last_rgb is not None:
                annotated, _R, _t = _annotate(last_rgb, detect_fn, K, D)
                try:
                    viser_server.scene.add_camera_frustum(
                        "/calibrated_zed/image",
                        position=np.zeros(3, dtype=np.float32),
                        wxyz=np.array([1, 0, 0, 0], dtype=np.float32),
                        fov=fov_y,
                        aspect=aspect,
                        scale=0.12,
                        image=annotated,
                    )
                except Exception:
                    pass
            print(f"[viser] Calibrated camera frustum added — inspect at "
                  f"http://localhost:{args.viser_port}")
            print(
                "[viser] Sanity check: the green frustum should sit where the "
                "ZED physically is and point at the workspace.\n"
                "        Press ENTER to exit and tear everything down."
            )
            try:
                input()
            except (EOFError, KeyboardInterrupt):
                pass
        except Exception as e:
            print(f"[viser] post-solve visualization failed: {e}")

    stop_evt.set()
    if refresh_th is not None:
        refresh_th.join(timeout=1.0)
    zed.stop()
    piper.DisconnectPort()


if __name__ == "__main__":
    main()
