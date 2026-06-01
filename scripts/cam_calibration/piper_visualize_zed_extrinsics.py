#!/usr/bin/env python3
"""Visualize the saved ZED 2i → PIPER base extrinsics in viser.

Loads `env_configs/real/piper_zed_extrinsics.yaml`, starts a viser server,
and draws:
  - the PIPER base frame and URDF (joints follow the live SDK readings)
  - the calibrated ZED pose as a green camera frustum
  - the live ZED RGB image attached to the frustum's image plane

Use this to sanity-check a calibration without re-running the capture loop.

Example
-------
    uv run --no-sync --active scripts/cam_calibration/piper_visualize_zed_extrinsics.py
"""
from __future__ import annotations

import argparse
import os
import sys
import threading
import time
from pathlib import Path

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


def _wxyz_from_R(R):
    xyzw = SciRotation.from_matrix(R).as_quat()
    return np.array([xyzw[3], xyzw[0], xyzw[1], xyzw[2]], dtype=np.float64)


def _load_extrinsics(path: Path):
    with open(path) as f:
        data = yaml.safe_load(f)
    pos = np.asarray(data["position"], dtype=np.float64).reshape(3)
    rpy = np.asarray(data["rpy_radians"], dtype=np.float64).reshape(3)
    R = SciRotation.from_euler("xyz", rpy).as_matrix()
    return pos, R


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument(
        "--extrinsics",
        default=os.environ.get(
            "PIPER_CAMERA_EXTRINSICS",
            str(ROOT / "env_configs/real/piper_zed_extrinsics.yaml"),
        ),
    )
    p.add_argument("--viser-port", type=int, default=8201)
    p.add_argument("--no-piper", action="store_true",
                   help="Skip connecting to the Piper SDK (URDF shown at zero pose)")
    p.add_argument("--no-zed", action="store_true",
                   help="Skip starting the ZED bridge (no live image on frustum)")
    p.add_argument("--can-interface",
                   default=os.environ.get("PIPER_CAN_INTERFACE", "socketcan"))
    p.add_argument("--can-channel",
                   default=os.environ.get("PIPER_CAN_CHANNEL", "can1"))
    p.add_argument("--can-bitrate", type=int,
                   default=int(os.environ.get("PIPER_CAN_BITRATE", "1000000")))
    args = p.parse_args()

    extr_path = Path(args.extrinsics)
    if not extr_path.exists():
        print(f"[viz] extrinsics file not found: {extr_path}", file=sys.stderr)
        sys.exit(1)
    t_base_cam, R_base_cam = _load_extrinsics(extr_path)
    rpy_deg = np.degrees(SciRotation.from_matrix(R_base_cam).as_euler("xyz"))
    print(f"[viz] Loaded {extr_path}")
    print(f"  position   : [{t_base_cam[0]:.4f}, {t_base_cam[1]:.4f}, "
          f"{t_base_cam[2]:.4f}] m")
    print(f"  rpy_degrees: [{rpy_deg[0]:+.2f}, {rpy_deg[1]:+.2f}, "
          f"{rpy_deg[2]:+.2f}] deg")

    # ---------- Piper (optional, read-only) ----------------------------- #
    piper = None
    if not args.no_piper:
        try:
            from piper_sdk import C_PiperInterface_V2  # type: ignore
            print(f"[viz] Connecting Piper via {args.can_interface} / "
                  f"{args.can_channel}")
            if args.can_interface == "gs_usb":
                piper = C_PiperInterface_V2(
                    can_name=args.can_channel, judge_flag=False,
                    can_auto_init=False,
                )
                piper.CreateCanBus(
                    can_name=args.can_channel, bustype="gs_usb",
                    expected_bitrate=args.can_bitrate, judge_flag=False,
                )
                piper.ConnectPort(can_init=False)
            else:
                piper = C_PiperInterface_V2(
                    can_name=args.can_channel, judge_flag=True,
                    can_auto_init=True,
                )
                piper.ConnectPort()
            time.sleep(0.5)
        except Exception as e:
            print(f"[viz] Piper connect failed ({e}); URDF stays at zero pose.")
            piper = None

    # ---------- ZED bridge (optional) ----------------------------------- #
    zed = None
    if not args.no_zed:
        try:
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
            print("[viz] Starting ZED bridge...")
            zed.start()
            # warm up for first frame
            t0 = time.time()
            while time.time() - t0 < 5.0:
                rgb, _ = zed.read_frames()
                if rgb is not None:
                    break
                time.sleep(0.1)
        except Exception as e:
            print(f"[viz] ZED start failed ({e}); frustum will have no image.")
            zed = None

    # ---------- viser --------------------------------------------------- #
    import viser  # type: ignore
    server = viser.ViserServer(port=args.viser_port)
    print(f"[viz] Open http://localhost:{args.viser_port}")

    server.scene.add_frame("/base", axes_length=0.1, axes_radius=0.005)

    # URDF
    urdf_vis = None
    num_actuated = 0
    urdf_path = os.environ.get("PIPER_URDF_PATH", DEFAULT_PIPER_URDF)
    if urdf_path and os.path.exists(urdf_path):
        try:
            import yourdfpy  # type: ignore
            from viser.extras import ViserUrdf  # type: ignore
            urdf = yourdfpy.URDF.load(
                urdf_path,
                filename_handler=_resolve_piper_mesh,
                build_collision_scene_graph=False,
                load_collision_meshes=False,
            )
            urdf_vis = ViserUrdf(server, urdf_or_path=urdf, load_meshes=True)
            num_actuated = len(urdf.actuated_joint_names)
        except Exception as e:
            print(f"[viz] URDF load failed ({e}); arm model skipped.")
    else:
        print(f"[viz] URDF not found at {urdf_path}; arm model skipped.")

    # Camera frustum — use ZED intrinsics if available, else a reasonable default
    if zed is not None:
        K = zed.intrinsics_matrix()
        # ZED bridge reports the left-rectified stream size via env vars
        h = int(os.environ.get("PIPER_ZED_HEIGHT", "720"))
        w = int(os.environ.get("PIPER_ZED_WIDTH", "1280"))
        fy = float(K[1, 1])
        fov_y = 2.0 * float(np.arctan2(0.5 * float(h), fy))
        aspect = float(w) / float(h)
    else:
        fov_y = np.deg2rad(54.0)  # typical ZED 2i vertical FoV
        aspect = 16.0 / 9.0

    frustum = server.scene.add_camera_frustum(
        "/calibrated_zed",
        position=t_base_cam.astype(np.float32),
        wxyz=_wxyz_from_R(R_base_cam).astype(np.float32),
        fov=fov_y,
        aspect=aspect,
        scale=0.15,
        color=(50, 200, 50),
    )

    # GUI: numbers + close button
    server.gui.add_markdown(
        f"**Extrinsics:** `{extr_path.name}`\n\n"
        f"position (m): `[{t_base_cam[0]:+.4f}, {t_base_cam[1]:+.4f}, "
        f"{t_base_cam[2]:+.4f}]`\n\n"
        f"rpy (deg): `[{rpy_deg[0]:+.2f}, {rpy_deg[1]:+.2f}, "
        f"{rpy_deg[2]:+.2f}]`"
    )

    # ---------- live refresh thread ------------------------------------- #
    stop_evt = threading.Event()

    def _refresh():
        while not stop_evt.is_set():
            if zed is not None:
                try:
                    rgb, _ = zed.read_frames()
                    if rgb is not None:
                        frustum.image = rgb
                except Exception:
                    pass
            if piper is not None and urdf_vis is not None:
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

    th = threading.Thread(target=_refresh, daemon=True)
    th.start()

    print("[viz] Press Ctrl-C to exit.")
    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        pass
    finally:
        stop_evt.set()
        th.join(timeout=1.0)
        if zed is not None:
            zed.stop()
        if piper is not None:
            try:
                piper.DisconnectPort()
            except Exception:
                pass


if __name__ == "__main__":
    main()
