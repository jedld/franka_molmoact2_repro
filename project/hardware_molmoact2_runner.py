#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""MolmoAct2 closed loop on a real Franka using USB cameras + motion_server.cpp.

Requires:
  - ``motion_server`` running (libfranka HTTP API on :34568)
  - MolmoAct2 inference server (see start_molmoact2_3090.md)
  - USB cameras configured in ``usb_cameras.json`` or env vars (see README)

Example:
  python3 hardware_molmoact2_runner.py \\
    --motion-url http://127.0.0.1:34568 \\
    --molmoact2-url http://127.0.0.1:8012 \\
    --instruction "Pick up the apple and place it on the plate"
"""

from __future__ import annotations

import argparse
import signal
import sys
import time
from pathlib import Path

_PROJECT_DIR = Path(__file__).resolve().parent
if str(_PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(_PROJECT_DIR))

from motion_server_http_robot import MotionServerHttpRobot
from motion_server_molmoact2 import MolmoAct2Controller
from motion_server_usb_cameras import UsbCameraStreamManager, load_usb_camera_config


def main() -> int:
    parser = argparse.ArgumentParser(description="MolmoAct2 hardware runner (USB cameras + motion_server)")
    parser.add_argument("--motion-url", default="http://127.0.0.1:34568", help="motion_server.cpp base URL")
    parser.add_argument("--molmoact2-url", default="http://127.0.0.1:8012", help="MolmoAct2 inference URL")
    parser.add_argument("--instruction", required=True, help="Natural-language task string")
    parser.add_argument(
        "--external-camera-mode",
        default="single",
        choices=["single", "dual", "duplicate"],
        help="MolmoAct2 exterior layout (default: single)",
    )
    parser.add_argument(
        "--external-slot",
        default="external_1",
        choices=["external_1", "external_2"],
        help="Primary exterior camera when mode=single (default: external_1)",
    )
    parser.add_argument("--usb-config", default=None, help="Path to usb_cameras.json (overrides default)")
    parser.add_argument("--hz", type=float, default=60.0, help="Main loop rate (default: 60)")
    args = parser.parse_args()

    usb_config = load_usb_camera_config(args.usb_config)
    cameras = UsbCameraStreamManager(usb_config)
    robot = MotionServerHttpRobot(args.motion_url)
    molmo = MolmoAct2Controller(robot, cameras)

    stop = False

    def _handle_sigint(_signum, _frame) -> None:
        nonlocal stop
        stop = True

    signal.signal(signal.SIGINT, _handle_sigint)
    signal.signal(signal.SIGTERM, _handle_sigint)

    print("Opening USB cameras...")
    cameras.initialize()
    molmo.configure(
        server_url=args.molmoact2_url,
        instruction=args.instruction,
        external_slot=args.external_slot,
        external_camera_mode=args.external_camera_mode,
    )
    health = molmo.refresh_health(async_ok=False)
    if not health.get("ok"):
        print(f"Warning: MolmoAct2 health check failed: {health}")
    else:
        print("MolmoAct2 inference server: ok")

    print("Starting MolmoAct2 control loop (Ctrl+C to stop)...")
    molmo.start()

    dt = 1.0 / max(args.hz, 1.0)
    try:
        while not stop:
            t0 = time.monotonic()
            molmo.tick(dt)
            elapsed = time.monotonic() - t0
            time.sleep(max(0.0, dt - elapsed))
    finally:
        molmo.stop()
        cameras.shutdown()
        print("Stopped.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
