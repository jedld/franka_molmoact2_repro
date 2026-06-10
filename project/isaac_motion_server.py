# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Isaac Sim motion server — drop-in HTTP replacement for project/motion_server.cpp.

Preserves the libfranka motion_server REST interface on port 34568 by default:

  curl -X POST http://127.0.0.1:34568 -H "Content-Type: application/json" \\
       -d '{"moveToCartesian": [0.5, 0, 0.3]}'

Supported JSON keys (each value is a numeric array):
  moveToCartesian, moveToJointPose, closeGripper, openGripper, readState, readJointState

Launch from a built Isaac Sim tree:
  project/launch_isaac_motion_server.bat
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_PROJECT_DIR = Path(__file__).resolve().parent
if str(_PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(_PROJECT_DIR))

parser = argparse.ArgumentParser(description="Isaac Sim HTTP motion server (motion_server.cpp compatible)")
parser.add_argument("--headless", action="store_true", help="Run Isaac Sim without a GUI")
parser.add_argument("--port", type=int, default=34568, help="HTTP listen port (default: 34568)")
parser.add_argument("--host", default="0.0.0.0", help="HTTP bind address (default: 0.0.0.0)")
parser.add_argument(
    "--robot-path",
    default="/World/RobotMount/Franka",
    help="USD path to Franka articulation root (default: /World/RobotMount/Franka)",
)
parser.add_argument(
    "--spawn-robot",
    action="store_true",
    help="Spawn the DROID scene + robot (default when the prim is missing)",
)
parser.add_argument(
    "--task",
    default="apple_on_plate",
    choices=[
        "apple_on_plate",
        "pipette_in_tray",
        "red_cube_in_tape_roll",
        "knife_in_box",
        "objects_in_bowl",
        "empty",
    ],
    help="MolmoAct2 Table 6 task props (default: apple_on_plate)",
)
parser.add_argument("--skip-home", action="store_true", help="Skip homing move on startup")
parser.add_argument("--test", action="store_true", help="Exit after a short smoke run")
parser.add_argument("--no-ui", action="store_true", help="Disable web UI (motion API only)")
parser.add_argument("--no-cameras", action="store_true", help="Do not spawn or capture simulation cameras")
parser.add_argument(
    "--usb-cameras",
    action="store_true",
    help="Use USB RealSense/V4L2 cameras instead of Isaac Sim rendered cameras",
)
parser.add_argument(
    "--usb-config",
    default=None,
    help="Path to usb_cameras.json (default: project/usb_cameras.json or MOLMO_USB_CONFIG)",
)
parser.add_argument(
    "--molmoact2-url",
    default="http://192.168.0.233:8012",
    help="MolmoAct2 inference server base URL (default: http://192.168.0.233:8012)",
)
args, _ = parser.parse_known_args()

from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": args.headless, "width": 1280, "height": 720})

import carb
import isaacsim.core.experimental.utils.app as app_utils
import numpy as np
import omni
import warp as wp
from isaacsim.core.experimental.prims import Articulation
from isaacsim.core.rendering_manager import ViewportManager
from isaacsim.core.simulation_manager import SimulationManager
from isaacsim.storage.native import get_assets_root_path

from motion_server_api import MotionRestServer
from motion_server_cameras import CameraStreamManager, POLICY_CAPTURE_HZ, discover_stage_cameras
from motion_server_usb_cameras import UsbCameraStreamManager, load_usb_camera_config
from motion_server_handler import IsaacMotionServerHandler
from motion_server_molmoact2 import MolmoAct2Controller, TASK_INSTRUCTIONS
from motion_server_scene import DEFAULT_FRANKA_PATH, SCENE_CAMERA_TARGET, spawn_droid_scene, setup_droid_cameras


def _prim_exists(path: str) -> bool:
    stage = omni.usd.get_context().get_stage()
    return stage.GetPrimAtPath(path).IsValid()


def _set_default_joint_pose(robot_path: str) -> None:
    robot = Articulation(robot_path)
    default_q = np.array([0.0, -1.3, 0.0, -2.87, 0.0, 2.0, 0.75, 0.04, 0.04], dtype=np.float32)
    robot.set_dof_positions(wp.from_numpy(default_q.reshape(1, -1), dtype=wp.float32))
    robot.set_dof_position_targets(default_q.reshape(1, -1))


def main() -> None:
    assets_root = get_assets_root_path()
    if assets_root is None:
        carb.log_error("Could not find Isaac Sim assets folder.")
        simulation_app.close()
        sys.exit(1)

    spawn = args.spawn_robot or not _prim_exists(args.robot_path)
    if spawn:
        spawn_droid_scene(assets_root, args.robot_path, task=args.task)
        print(f"Spawned MolmoAct2-DROID scene at {args.robot_path} (task={args.task})")
    elif not _prim_exists(args.robot_path):
        carb.log_error(f"Robot prim not found at {args.robot_path}. Pass --spawn-robot or open a stage first.")
        simulation_app.close()
        sys.exit(1)

    ViewportManager.set_camera_view(
        "/OmniverseKit_Persp",
        eye=[0.0, -0.95, 1.25],
        target=list(SCENE_CAMERA_TARGET),
    )

    simulation_app.update()
    SimulationManager.setup_simulation(dt=1.0 / 60.0, device="cpu")
    simulation_app.update()
    app_utils.play()
    simulation_app.update()

    if spawn:
        _set_default_joint_pose(args.robot_path)
        simulation_app.update()

    camera_manager = None
    if not args.no_cameras:
        if args.usb_cameras:
            usb_config = load_usb_camera_config(args.usb_config)
            camera_manager = UsbCameraStreamManager(usb_config)
            camera_manager.initialize()
            print(
                f"USB cameras: {', '.join(s.camera_id for s in camera_manager.specs)} "
                f"(policy @ {POLICY_CAPTURE_HZ} Hz, background capture)"
            )
        else:
            if spawn:
                camera_specs = setup_droid_cameras(args.robot_path)
            else:
                camera_specs = discover_stage_cameras()
                if not camera_specs:
                    camera_specs = setup_droid_cameras(args.robot_path)
            if camera_specs:
                camera_manager = CameraStreamManager(camera_specs)
                simulation_app.update()
                camera_manager.initialize()
                camera_manager.capture()  # seed RGB cache before the main loop
                print(f"Sim cameras: {', '.join(s.camera_id for s in camera_specs)} (policy @ {POLICY_CAPTURE_HZ} Hz)")

    handler = IsaacMotionServerHandler(args.robot_path, step_fn=simulation_app.update)

    molmoact2: MolmoAct2Controller | None = None
    if camera_manager is not None:
        molmoact2 = MolmoAct2Controller(handler, camera_manager)
        default_instruction = TASK_INSTRUCTIONS.get(args.task, TASK_INSTRUCTIONS["apple_on_plate"])
        molmoact2.configure(server_url=args.molmoact2_url, instruction=default_instruction)
        print(f"MolmoAct2 inference: {args.molmoact2_url}")

    api = MotionRestServer(
        handler,
        host=args.host,
        port=args.port,
        camera_manager=camera_manager,
        enable_ui=not args.no_ui,
        molmoact2=molmoact2,
    )
    api.start()

    print("Press Ctrl+C in the terminal running Isaac Sim to exit.")
    if not args.no_ui:
        print(f"Web UI: http://127.0.0.1:{args.port}/")
        print(f"  MolmoAct2 panel: http://127.0.0.1:{args.port}/#molmoact2-panel")
    print("Example motion API:")
    print(
        f'  curl -X POST http://127.0.0.1:{args.port} '
        '-H "Content-Type: application/json" '
        '-d \'{"readState": []}\''
    )

    try:
        handler.initialize(skip_home=args.skip_home)
    except Exception as exc:
        carb.log_warn(f"Startup initialize() warning: {exc}")

    frame = 0
    sim_dt = 1.0 / 60.0
    while simulation_app.is_running():
        api.process_pending()
        simulation_app.update()
        if camera_manager is not None:
            camera_manager.tick_policy(sim_dt)
        if molmoact2 is not None:
            molmoact2.tick(sim_dt)
        frame += 1
        if args.test and frame >= 120:
            break

    api.stop()
    if isinstance(camera_manager, UsbCameraStreamManager):
        camera_manager.shutdown()
    app_utils.stop()
    simulation_app.close()


if __name__ == "__main__":
    main()
