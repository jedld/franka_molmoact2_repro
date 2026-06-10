# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Compute and print the auto wrist camera pose for the DROID Franka scene.

Run from the Isaac Sim Python environment after a repo build:

  project\\launch_compute_wrist_camera_pose.bat
  project\\launch_compute_wrist_camera_pose.bat --behind 0.04

Requires the Franka on the DROID desk with the default ready joint pose applied.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_PROJECT_DIR = Path(__file__).resolve().parent
if str(_PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(_PROJECT_DIR))

parser = argparse.ArgumentParser(description="Compute wrist camera pose behind gripper (RealSense D435)")
parser.add_argument("--headless", action="store_true")
parser.add_argument(
    "--robot-path",
    default="/World/RobotMount/Franka",
    help="Franka articulation root prim path",
)
parser.add_argument(
    "--behind",
    type=float,
    default=None,
    help="Distance behind gripper center along −hand Z (m); default from wrist camera module",
)
parser.add_argument("--spawn-robot", action="store_true", help="Spawn DROID scene if robot is missing")
args, _ = parser.parse_known_args()

from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": args.headless, "width": 1280, "height": 720})

import isaacsim.core.experimental.utils.app as app_utils
import numpy as np
import omni
import warp as wp
from isaacsim.core.experimental.prims import Articulation
from isaacsim.core.simulation_manager import SimulationManager
from isaacsim.storage.native import get_assets_root_path

from motion_server_scene import spawn_droid_scene
from motion_server_wrist_camera import (
    DEFAULT_BEHIND_GRIPPER_M,
    compute_wrist_camera_local_pose,
    format_wrist_camera_pose_report,
)


def _prim_exists(path: str) -> bool:
    return omni.usd.get_context().get_stage().GetPrimAtPath(path).IsValid()


def _set_droid_ready_pose(robot_path: str) -> None:
    robot = Articulation(robot_path)
    default_q = np.array([0.0, -1.3, 0.0, -2.87, 0.0, 2.0, 0.75, 0.04, 0.04], dtype=np.float32)
    robot.set_dof_positions(wp.from_numpy(default_q.reshape(1, -1), dtype=wp.float32))


def main() -> None:
    robot_path = args.robot_path
    assets_root = get_assets_root_path()
    if assets_root is None:
        print("ERROR: Isaac Sim assets root not found.")
        simulation_app.close()
        sys.exit(1)

    if args.spawn_robot or not _prim_exists(robot_path):
        spawn_droid_scene(assets_root, robot_path, task="empty")
        print(f"Spawned DROID scene at {robot_path}")

    simulation_app.update()
    SimulationManager.setup_simulation(dt=1.0 / 60.0, device="cpu")
    simulation_app.update()
    app_utils.play()
    simulation_app.update()

    _set_droid_ready_pose(robot_path)
    simulation_app.update()

    hand_path = f"{robot_path}/panda_hand"
    behind = DEFAULT_BEHIND_GRIPPER_M if args.behind is None else args.behind
    stage = omni.usd.get_context().get_stage()

    _translation, _rotation, meta = compute_wrist_camera_local_pose(
        stage,
        hand_path,
        behind_distance_m=behind,
    )
    print(format_wrist_camera_pose_report(hand_path, meta))

    app_utils.stop()
    simulation_app.close()


if __name__ == "__main__":
    main()
