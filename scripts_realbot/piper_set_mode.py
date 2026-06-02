#!/usr/bin/env python3
"""Switch an Agilex PIPER arm between master and slave mode.

cap-x needs a SLAVE-mode arm: master-mode arms broadcast control frames
(0x155-0x157) themselves and do NOT respond to JointCtrl / EndPoseCtrl
commands. A newly-powered-on arm in master mode must be flipped to slave
before cap-x can drive it.

IMPORTANT (per Agilex SDK note):
  After sending the mode switch command to an arm currently in MASTER
  mode, you MUST power-cycle the arm for the change to take effect.
  Slave → master requires no power-cycle.

Usage:
    # Default: set arm on PIPER_CAN_CHANNEL (can1) to slave mode
    uv run --no-sync --active scripts_realbot/piper_set_mode.py

    # Explicit
    uv run --no-sync --active scripts_realbot/piper_set_mode.py --channel can1 --mode slave
    uv run --no-sync --active scripts_realbot/piper_set_mode.py --channel can0 --mode master
"""
from __future__ import annotations

import argparse
import os
import sys
import time


MODE_CODES = {
    "slave": 0xFC,   # standard / operation
    "master": 0xFA,  # teaching / broadcaster
}


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--channel", default=os.environ.get("PIPER_CAN_CHANNEL", "can1"),
                   help="canN interface of the arm to configure (default: can1)")
    p.add_argument("--mode", choices=list(MODE_CODES.keys()), default="slave",
                   help="target mode (default: slave — required for cap-x)")
    p.add_argument("--bitrate", type=int,
                   default=int(os.environ.get("PIPER_CAN_BITRATE", "1000000")))
    args = p.parse_args()

    from piper_sdk import C_PiperInterface_V2  # type: ignore

    print(f"[set_mode] Connecting to {args.channel}...")
    piper = C_PiperInterface_V2(
        can_name=args.channel, judge_flag=True, can_auto_init=True
    )
    piper.ConnectPort()
    time.sleep(0.5)

    code = MODE_CODES[args.mode]
    print(f"[set_mode] Sending MasterSlaveConfig(0x{code:02X}, 0, 0, 0) → {args.mode}")
    piper.MasterSlaveConfig(code, 0, 0, 0)
    time.sleep(0.3)
    piper.DisconnectPort()

    print()
    if args.mode == "slave":
        print("If this arm was previously in MASTER mode, power-cycle it now.")
        print("After reboot, verify it responds to commands:")
        print("   uv run --no-sync --active scripts_realbot/piper_identify_arms.py")
        print("and confirm the arm accepts JointCtrl by running a cap-x eval.")
    else:
        print("Arm switched to master. No power-cycle needed for slave→master.")


if __name__ == "__main__":
    main()
