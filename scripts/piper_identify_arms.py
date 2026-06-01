#!/usr/bin/env python3
"""Identify which `canN` interface corresponds to which physical PIPER arm.

Connects to every available CAN interface (can0, can1, ...) via piper_sdk and
live-prints each arm's current joint + end-pose. Physically nudge the arm you
plan to use — the canN whose numbers change is the one you want.

Set `PIPER_CAN_CHANNEL` in ~/.bashrc (or pass `--can-channel` at runtime) to
that interface and you're done.

Usage:
    uv run --no-sync --active scripts/piper_identify_arms.py
    # Ctrl-C to stop.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time

import numpy as np

RAW_TO_DEG = 1e-3


def list_up_can_ifaces() -> list[str]:
    out = subprocess.run(
        ["ip", "-brief", "link", "show"], capture_output=True, text=True, check=True
    ).stdout
    ifaces = []
    for line in out.splitlines():
        parts = line.split()
        if parts and parts[0].startswith("can") and len(parts) > 1 and "UP" in parts[1]:
            ifaces.append(parts[0])
    return sorted(ifaces)


def connect(channel: str, bitrate: int):
    from piper_sdk import C_PiperInterface_V2  # type: ignore

    piper = C_PiperInterface_V2(
        can_name=channel, judge_flag=True, can_auto_init=True
    )
    piper.ConnectPort()
    return piper


def read_state(piper) -> dict:
    try:
        js = piper.GetArmJointMsgs().joint_state
        joints_deg = np.array(
            [js.joint_1, js.joint_2, js.joint_3, js.joint_4, js.joint_5, js.joint_6],
            dtype=np.float64,
        ) * RAW_TO_DEG
    except Exception:
        joints_deg = np.full(6, np.nan)
    try:
        pose = piper.GetArmEndPoseMsgs().end_pose
        xyz_mm = np.array([pose.X_axis, pose.Y_axis, pose.Z_axis]) * 1e-3
        rxyz_deg = np.array([pose.RX_axis, pose.RY_axis, pose.RZ_axis]) * 1e-3
    except Exception:
        xyz_mm = np.full(3, np.nan)
        rxyz_deg = np.full(3, np.nan)
    return {"joints_deg": joints_deg, "xyz_mm": xyz_mm, "rxyz_deg": rxyz_deg}


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--bitrate", type=int, default=1_000_000)
    p.add_argument("--hz", type=float, default=5.0)
    args = p.parse_args()

    ifaces = list_up_can_ifaces()
    if not ifaces:
        print("No UP canN interface found. Run: bash scripts/setup_can.sh first.")
        sys.exit(1)
    print(f"Connecting to: {', '.join(ifaces)}")

    pipers = {}
    for ch in ifaces:
        try:
            pipers[ch] = connect(ch, args.bitrate)
            print(f"  ✓ {ch} connected")
        except Exception as e:
            print(f"  ✗ {ch} failed: {e}")
    if not pipers:
        print("No arm connected. Check CAN cables / arm power.")
        sys.exit(1)

    print()
    print("Physically move / nudge the arm you want to use.")
    print("Watch which row's 'xyz_mm' numbers actually change — that's your arm.")
    print("Press Ctrl-C to stop.\n")

    period = 1.0 / max(args.hz, 0.5)
    try:
        while True:
            lines = []
            for ch, piper in pipers.items():
                s = read_state(piper)
                xyz = s["xyz_mm"]
                joints = s["joints_deg"]
                lines.append(
                    f"{ch}  xyz_mm=[{xyz[0]:+7.1f} {xyz[1]:+7.1f} {xyz[2]:+7.1f}]  "
                    f"j1..j6_deg=["
                    + " ".join(f"{v:+6.1f}" for v in joints)
                    + "]"
                )
            # Overwrite in place (ANSI cursor up per line after first print)
            print("\x1b[2K" + "  |  ".join(lines), end="\r", flush=True)
            time.sleep(period)
    except KeyboardInterrupt:
        print()
    finally:
        for piper in pipers.values():
            try:
                piper.DisconnectPort()
            except Exception:
                pass


if __name__ == "__main__":
    main()
