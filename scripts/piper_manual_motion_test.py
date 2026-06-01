#!/usr/bin/env python3
"""Manual motion test for the PIPER arm (no SAM3, no LLM).

Drives the arm to user-specified EEF poses via pyroki IK. Useful before SAM3
access is granted — tests CAN + ZED + extrinsics + IK + execution + gripper
all in one interactive REPL, with every target visualized in viser.

Everything is expressed in the robot **base frame** (metres, degrees).

Commands (type at the prompt):
    goto X Y Z [RX RY RZ]        move EEF to (X,Y,Z), rpy in deg (default
                                  [180 0 0] = gripper pointing straight down)
    above X Y Z [H] [RX RY RZ]    goto (X, Y, Z+H) — convenient "above" pose,
                                  default H=0.15 m
    joints J1 J2 J3 J4 J5 J6      direct joint angles in degrees
    home                          all joints → 0 deg (safe rest)
    open / close                  gripper open / close
    where                         print current EEF pose (from URDF FK)
    quit / q                      exit

Every `goto` / `above` / `joints` / `home` command:
  - draws a blue target coordinate frame in viser (before execution)
  - waits for you to press ENTER to confirm
  - then commands the arm (motion is blocking)

Example
-------
    uv run --no-sync --active scripts/piper_manual_motion_test.py
    # then inside:
    > where
    > above 0.35 0.00 0.10           # go 15 cm above (0.35, 0, 0.10)
    > goto 0.35 0.00 0.10            # drop to (0.35, 0, 0.10)
    > close
    > above 0.35 0.00 0.10 0.20      # lift 20 cm
    > open
    > home
    > q
"""
from __future__ import annotations

import argparse
import os
import sys
import threading
import time
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as SciRotation

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from capx.envs.simulators.piper_real import PiperRealLowLevel  # noqa: E402
from capx.integrations.franka.common import (  # noqa: E402
    close_gripper as _close_gripper,
    open_gripper as _open_gripper,
)
from capx.integrations.piper.control import (  # noqa: E402
    DEFAULT_PIPER_EEF_LINK,
    DEFAULT_PIPER_URDF,
    _init_piper_ik,
    _resolve_piper_mesh,
)

RAW_MDEG_TO_RAD = np.pi / 180.0 * 1e-3
TCP_OFFSET = np.array([0.0, 0.0, -0.13], dtype=np.float64)  # same as PiperControlApi


# --------------------------------------------------------------------------- #
# Viser helper                                                                 #
# --------------------------------------------------------------------------- #
class Viz:
    def __init__(self, env, port: int):
        import viser  # type: ignore

        self.env = env
        self.server = viser.ViserServer(port=port)
        print(f"[viz] Open http://localhost:{port}")
        self.server.scene.add_frame("/base", axes_length=0.1, axes_radius=0.005)

        pose = env._cam_pose_xyz_wxyz
        K = env._intrinsics
        h, w = env._zed.height, env._zed.width
        fov_y = 2.0 * float(np.arctan2(float(K[1, 2]), float(K[1, 1])))
        aspect = float(w) / float(h)
        self.frustum = self.server.scene.add_camera_frustum(
            "/calibrated_zed",
            position=pose[:3].astype(np.float32),
            wxyz=pose[3:].astype(np.float32),
            fov=fov_y, aspect=aspect, scale=0.15, color=(50, 200, 50),
        )

        self.urdf_vis = None
        self.num_actuated = 0
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
                self.urdf_vis = ViserUrdf(self.server, urdf_or_path=urdf, load_meshes=True)
                self.num_actuated = len(urdf.actuated_joint_names)
            except Exception as e:
                print(f"[viz] URDF load failed ({e}); arm model skipped.")

        self.status = self.server.gui.add_markdown("**Status:** ready")
        self._stop = threading.Event()
        self._th = threading.Thread(target=self._loop, daemon=True)
        self._th.start()
        self._target_counter = 0

    def _loop(self):
        while not self._stop.is_set():
            try:
                rgb, _ = self.env._zed.read_frames()
                if rgb is not None:
                    self.frustum.image = rgb
            except Exception:
                pass
            if self.urdf_vis is not None:
                try:
                    js = self.env._piper.GetArmJointMsgs().joint_state
                    joints = np.array(
                        [js.joint_1, js.joint_2, js.joint_3,
                         js.joint_4, js.joint_5, js.joint_6],
                        dtype=np.float64,
                    ) * RAW_MDEG_TO_RAD
                    cfg = np.zeros(self.num_actuated, dtype=np.float64)
                    cfg[: min(6, self.num_actuated)] = joints[: min(6, self.num_actuated)]
                    self.urdf_vis.update_cfg(cfg)
                except Exception:
                    pass
            time.sleep(0.1)

    def set_status(self, text: str):
        try:
            self.status.content = f"**Status:** {text}"
        except Exception:
            pass

    def draw_target(self, pos, quat_wxyz, label: str = ""):
        self._target_counter += 1
        name = f"/targets/{self._target_counter:02d}_{label or 'pose'}"
        try:
            self.server.scene.add_frame(
                name,
                position=np.asarray(pos, dtype=np.float32),
                wxyz=np.asarray(quat_wxyz, dtype=np.float32),
                axes_length=0.08, axes_radius=0.004,
            )
        except Exception:
            pass

    def stop(self):
        self._stop.set()
        self._th.join(timeout=1.0)


# --------------------------------------------------------------------------- #
def _wxyz_from_rpy_deg(rpy_deg):
    rot = SciRotation.from_euler("xyz", rpy_deg, degrees=True)
    xyzw = rot.as_quat()
    return np.array([xyzw[3], xyzw[0], xyzw[1], xyzw[2]], dtype=np.float64), rot


def _ik_joints(ik_solve_fn, quat_wxyz, pos):
    target = np.concatenate([quat_wxyz, pos])
    q = ik_solve_fn(target_pose_wxyz_xyz=target)
    return np.asarray(q[:6], dtype=np.float64).reshape(6)


def _read_current_joints(env):
    js = env._piper.GetArmJointMsgs().joint_state
    return np.array(
        [js.joint_1, js.joint_2, js.joint_3, js.joint_4, js.joint_5, js.joint_6],
        dtype=np.float64,
    ) * RAW_MDEG_TO_RAD


def _parse_floats(tokens, expected_min=None, expected_max=None):
    vals = [float(t) for t in tokens]
    if expected_min is not None and len(vals) < expected_min:
        raise ValueError(f"expected at least {expected_min} numbers, got {len(vals)}")
    if expected_max is not None and len(vals) > expected_max:
        raise ValueError(f"expected at most {expected_max} numbers, got {len(vals)}")
    return vals


# --------------------------------------------------------------------------- #
def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--viser-port", type=int, default=8201)
    p.add_argument("--no-viser", action="store_true")
    p.add_argument("--no-confirm", action="store_true",
                   help="Skip ENTER confirmation before every motion (NOT RECOMMENDED)")
    args = p.parse_args()

    env = PiperRealLowLevel()
    env.reset()

    print("[motion_test] Building pyroki IK solver (no SAM3 needed)...")
    ik_solve_fn = _init_piper_ik(DEFAULT_PIPER_URDF, DEFAULT_PIPER_EEF_LINK)

    viz = None if args.no_viser else Viz(env, port=args.viser_port)

    def _confirm(desc: str) -> bool:
        if args.no_confirm:
            return True
        ans = input(f"    [ENTER=execute / s=skip]: ").strip().lower()
        return ans != "s"

    def _goto_tcp(pos, rpy_deg, label: str):
        """Move TCP (fingertip) to `pos` with `rpy_deg` orientation.

        Matches PiperControlApi.goto_pose semantics: pos is the TCP, the IK
        target passed to the URDF is pos + R * TCP_OFFSET (TCP_OFFSET is the
        gripper_base -> fingertip offset in the EEF frame).
        """
        quat_wxyz, rot = _wxyz_from_rpy_deg(rpy_deg)
        ik_target_pos = pos + rot.apply(TCP_OFFSET)
        try:
            joints = _ik_joints(ik_solve_fn, quat_wxyz, ik_target_pos)
        except Exception as e:
            print(f"    ✗ IK failed: {type(e).__name__}: {e}")
            return
        print(f"    TCP target (base) : {np.array2string(pos, precision=3)}")
        print(f"    rpy (deg)         : {rpy_deg}")
        print(f"    IK joint solution : {np.array2string(np.degrees(joints), precision=1)} deg")
        if viz is not None:
            viz.draw_target(pos, quat_wxyz, label=label)
            viz.set_status(
                f"{label} target drawn (blue). ENTER to execute, s to skip."
            )
        if not _confirm(label):
            print("    skipped.")
            return
        env.move_to_joints_blocking(joints)
        if viz is not None:
            viz.set_status(f"{label} reached.")

    def _goto_joints(joints_deg):
        joints_rad = np.deg2rad(np.asarray(joints_deg, dtype=np.float64))
        print(f"    joint target (deg): {np.array2string(np.asarray(joints_deg), precision=1)}")
        if viz is not None:
            viz.set_status(f"joints target — waiting confirmation")
        if not _confirm("joints"):
            print("    skipped.")
            return
        env.move_to_joints_blocking(joints_rad)
        if viz is not None:
            viz.set_status("joints target reached.")

    def _where():
        joints = _read_current_joints(env)
        print(f"    joints (deg): {np.array2string(np.degrees(joints), precision=1)}")
        # rough end pose from the SDK
        try:
            from scripts.piper_calibrate_zed_extrinsics import read_piper_pose
            R, t = read_piper_pose(env._piper)
            rpy = np.degrees(SciRotation.from_matrix(R).as_euler("xyz"))
            print(f"    SDK end pose xyz (m)   : {np.array2string(t, precision=3)}")
            print(f"    SDK end pose rpy (deg) : {np.array2string(rpy, precision=1)}")
        except Exception:
            pass

    # ------------- REPL ------------------------------------------------- #
    print("""
Commands:
  goto X Y Z [RX RY RZ]       move to (x,y,z), rpy deg (default 180 0 0 = down)
  above X Y Z [H] [RX RY RZ]  go to (X,Y,Z+H), default H=0.15
  joints J1..J6                direct joint angles in DEGREES
  home                         all joints → 0
  open / close                 gripper
  where                        print current pose
  quit / q                     exit
""")
    try:
        while True:
            raw = input("> ").strip()
            if not raw:
                continue
            tok = raw.split()
            cmd = tok[0].lower()
            try:
                if cmd in ("q", "quit", "exit"):
                    break
                elif cmd == "where":
                    _where()
                elif cmd == "open":
                    _open_gripper(env, steps=15)
                elif cmd == "close":
                    _close_gripper(env, steps=15)
                elif cmd == "home":
                    _goto_joints([0, 0, 0, 0, 0, 0])
                elif cmd == "joints":
                    vals = _parse_floats(tok[1:], expected_min=6, expected_max=6)
                    _goto_joints(vals)
                elif cmd == "goto":
                    vals = _parse_floats(tok[1:], expected_min=3, expected_max=6)
                    pos = np.array(vals[:3], dtype=np.float64)
                    rpy = vals[3:6] if len(vals) == 6 else [180.0, 0.0, 0.0]
                    _goto_tcp(pos, rpy, "goto")
                elif cmd == "above":
                    vals = _parse_floats(tok[1:], expected_min=3, expected_max=7)
                    pos = np.array(vals[:3], dtype=np.float64)
                    h = vals[3] if len(vals) >= 4 else 0.15
                    rpy = vals[4:7] if len(vals) == 7 else [180.0, 0.0, 0.0]
                    pos_above = pos.copy()
                    pos_above[2] += h
                    _goto_tcp(pos_above, rpy, "above")
                else:
                    print(f"    unknown command: {cmd!r}")
            except ValueError as e:
                print(f"    parse error: {e}")
            except Exception as e:
                print(f"    error: {type(e).__name__}: {e}")
    except KeyboardInterrupt:
        pass
    finally:
        if viz is not None:
            viz.stop()
        try:
            env._zed.stop()
        except Exception:
            pass
        try:
            env._piper.DisconnectPort()
        except Exception:
            pass


if __name__ == "__main__":
    main()
