# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""MolmoAct2-DROID workstation scene for the Isaac Sim motion server.

Layout, cameras, and task props match
source/standalone_examples/api/isaacsim.ros2.bridge/molmoact2_droid_validation.py
so web UI feeds and manipulation targets align with MolmoAct2 experiments.
"""

from __future__ import annotations

import numpy as np
import omni
import isaacsim.core.experimental.utils.stage as stage_utils
from isaacsim.core.experimental.objects import Cube, Cylinder, DomeLight, GroundPlane, RectLight, Sphere
from isaacsim.core.experimental.prims import GeomPrim, RigidPrim
from pxr import Gf, Sdf, Usd, UsdGeom

from motion_server_cameras import CAMERA_HEIGHT, CAMERA_WIDTH, CameraSpec
from motion_server_wrist_camera import (
    configure_realsense_d435_prim,
    compute_wrist_camera_local_pose,
    format_wrist_camera_pose_report,
)

FRANKA_USD_PATH = "/Isaac/Robots/FrankaRobotics/FrankaPanda/franka.usd"
ROBOT_MOUNT_PATH = "/World/RobotMount"
DEFAULT_FRANKA_PATH = f"{ROBOT_MOUNT_PATH}/Franka"
DESK_TOP_PATH = "/World/Desk/Top"

# DROID standing-desk layout (MolmoAct2 Table 6).
DESK_TOP_Z = 0.75
DESK_THICKNESS = 0.03
DESK_WIDTH = 1.00
DESK_DEPTH = 0.72
DESK_CENTER = (0.0, 0.05, DESK_TOP_Z - DESK_THICKNESS / 2.0)
ROBOT_POSITION = (0.0, -0.28, DESK_TOP_Z)
ROBOT_YAW_DEG = 90.0
WORKSPACE_FORWARD_Y = 0.25
SCENE_CAMERA_TARGET = (0.0, -0.10, DESK_TOP_Z + 0.06)

# DROID hardware (RSS 2024 / droid-dataset.github.io):
#   - 2× Stereolabs ZED 2 on adjustable tripods (exterior_1_left, exterior_2_left)
#   - 1× wrist RGB (sim: Intel RealSense D435 intrinsics; real DROID uses ZED Mini)
# Native capture: 1280×720 RGB @ 15 Hz; MolmoAct2 uses 320×180.
# Tripod base positions (DROID reference layout); pullback moves them farther along the look-at ray.
_EXT1_EYE = (-0.65, 0.12, 0.34)  # left-side tripod (ZED 2 clamp)
_EXT2_EYE = (0.0, 0.56, 0.36)  # front tripod (+Y), faces robot reach axis
EXTERIOR_CAMERA_PULLBACK = 1.35

# ZED 2 LEFT_CAM_HD rectified intrinsics @ 1280×720 (Stereolabs factory calib reference).
# https://support.stereolabs.com/hc/en-us/articles/360007497173
ZED_2_NATIVE_WIDTH = 1280
ZED_2_NATIVE_HEIGHT = 720
ZED_2_FX = 699.595
ZED_2_FY = 699.595
ZED_2_CX = 614.61
ZED_2_CY = 346.3235
ZED_2_FOCUS_DISTANCE = 1.0
ZED_2_F_STOP = 4.0
ZED_2_CLIP_NEAR = 0.01
ZED_2_CLIP_FAR = 100.0

# Soft indoor lighting (DomeLight alone at 3500 blows out wrist cameras).
DOME_LIGHT_INTENSITY = 850.0
DOME_LIGHT_COLOR = (0.82, 0.85, 0.90)
WORKSPACE_KEY_LIGHT_INTENSITY = 2200.0
WORKSPACE_KEY_LIGHT_COLOR = (1.0, 0.96, 0.90)


def _on_table(x_off: float, y_off: float, z_off: float = 0.03) -> tuple[float, float, float]:
    return (
        DESK_CENTER[0] + x_off,
        DESK_CENTER[1] + WORKSPACE_FORWARD_Y + y_off,
        DESK_TOP_Z + z_off,
    )


def _desk_leg_positions() -> list[tuple[float, float]]:
    cx, cy, _ = DESK_CENTER
    inset = 0.06
    return [
        (cx - DESK_WIDTH / 2 + inset, cy - DESK_DEPTH / 2 + inset),
        (cx + DESK_WIDTH / 2 - inset, cy - DESK_DEPTH / 2 + inset),
        (cx - DESK_WIDTH / 2 + inset, cy + DESK_DEPTH / 2 - inset),
        (cx + DESK_WIDTH / 2 - inset, cy + DESK_DEPTH / 2 - inset),
    ]


def _make_static_collider(path: str) -> None:
    GeomPrim(path, apply_collision_apis=True)


def _scale_camera_intrinsics(
    native_width: int,
    native_height: int,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    width: int,
    height: int,
) -> tuple[float, float, float, float]:
    sx = width / native_width
    sy = height / native_height
    return (fx * sx, fy * sy, cx * sx, cy * sy)


def _scale_zed_2_intrinsics(width: int, height: int) -> tuple[float, float, float, float]:
    return _scale_camera_intrinsics(
        ZED_2_NATIVE_WIDTH,
        ZED_2_NATIVE_HEIGHT,
        ZED_2_FX,
        ZED_2_FY,
        ZED_2_CX,
        ZED_2_CY,
        width,
        height,
    )


def _configure_pinhole_camera_prim(
    camera: UsdGeom.Camera,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    width: int,
    height: int,
    *,
    focus_distance: float,
    clip_near: float,
    clip_far: float,
    f_stop: float,
    exposure_time: float | None = None,
) -> None:
    """Apply OpenCV pinhole intrinsics with a USD pinhole fallback."""
    camera.GetProjectionAttr().Set("perspective")
    camera.GetFocusDistanceAttr().Set(focus_distance)
    camera.GetClippingRangeAttr().Set(Gf.Vec2f(clip_near, clip_far))

    focal_length = ((fx + fy) * 0.5) * 1.4e-3 / 10.0
    horizontal_aperture = 1.4e-3 * width / 10.0
    vertical_aperture = 1.4e-3 * height / 10.0
    camera.GetFocalLengthAttr().Set(focal_length)
    camera.GetHorizontalApertureAttr().Set(horizontal_aperture)
    camera.GetVerticalApertureAttr().Set(vertical_aperture)

    prim = camera.GetPrim()
    prim.CreateAttribute("exposure:fStop", Sdf.ValueTypeNames.Float).Set(f_stop)
    if exposure_time is not None:
        prim.CreateAttribute("exposure:time", Sdf.ValueTypeNames.Float).Set(exposure_time)
        prim.CreateAttribute("exposure:responsivity", Sdf.ValueTypeNames.Float).Set(1.0)
    _apply_opencv_pinhole_schema(prim, fx, fy, cx, cy, width, height)


def _configure_zed_2_prim(camera: UsdGeom.Camera, width: int, height: int) -> None:
    """Stereolabs ZED 2 left RGB intrinsics scaled to the render resolution."""
    fx, fy, cx, cy = _scale_zed_2_intrinsics(width, height)
    _configure_pinhole_camera_prim(
        camera,
        fx,
        fy,
        cx,
        cy,
        width,
        height,
        focus_distance=ZED_2_FOCUS_DISTANCE,
        clip_near=ZED_2_CLIP_NEAR,
        clip_far=ZED_2_CLIP_FAR,
        f_stop=ZED_2_F_STOP,
        exposure_time=0.008,
    )


def _apply_opencv_pinhole_schema(
    prim,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    width: int,
    height: int,
) -> None:
    """Apply OpenCV pinhole intrinsics (zero distortion — rectified ZED left view)."""
    schema = "OmniLensDistortionOpenCvPinholeAPI"
    if not prim.HasAPI(schema):
        prim.ApplyAPI(schema)
    prefix = "omni:lensdistortion:opencvPinhole"
    prim.CreateAttribute("omni:lensdistortion:model", Sdf.ValueTypeNames.Token).Set("opencvPinhole")
    prim.CreateAttribute(f"{prefix}:fx", Sdf.ValueTypeNames.Float).Set(float(fx))
    prim.CreateAttribute(f"{prefix}:fy", Sdf.ValueTypeNames.Float).Set(float(fy))
    prim.CreateAttribute(f"{prefix}:cx", Sdf.ValueTypeNames.Float).Set(float(cx))
    prim.CreateAttribute(f"{prefix}:cy", Sdf.ValueTypeNames.Float).Set(float(cy))
    prim.CreateAttribute(f"{prefix}:imageSize", Sdf.ValueTypeNames.Int2).Set(Gf.Vec2i(int(width), int(height)))
    for coeff in ("k1", "k2", "p1", "p2", "k3", "k4", "k5", "k6", "s1", "s2", "s3", "s4"):
        prim.CreateAttribute(f"{prefix}:{coeff}", Sdf.ValueTypeNames.Float).Set(0.0)


def _look_at_rotation(eye: Gf.Vec3d, target: Gf.Vec3d, up: Gf.Vec3d | None = None) -> Gf.Rotation:
    if up is None:
        up = Gf.Vec3d(0.0, 0.0, 1.0)
    return Gf.Matrix4d().SetLookAt(eye, target, up).GetInverse().ExtractRotation()


def _set_prim_world_transform(prim: Usd.Prim, translation: Gf.Vec3d, rotation: Gf.Rotation) -> None:
    xformable = UsdGeom.Xformable(prim)
    xformable.ClearXformOpOrder()
    xformable.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble).Set(translation)
    xformable.AddOrientOp(UsdGeom.XformOp.PrecisionDouble).Set(Gf.Quatd(rotation.GetQuat()))


def _set_prim_local_transform(prim: Usd.Prim, translation: Gf.Vec3d, rotation: Gf.Rotation) -> None:
    _set_prim_world_transform(prim, translation, rotation)


def _create_world_camera(
    path: str,
    eye: tuple[float, float, float],
    look_at: tuple[float, float, float],
    width: int = CAMERA_WIDTH,
    height: int = CAMERA_HEIGHT,
) -> str:
    stage = omni.usd.get_context().get_stage()
    camera = UsdGeom.Camera.Define(stage, path)
    _configure_zed_2_prim(camera, width, height)
    eye_v = Gf.Vec3d(*eye)
    target_v = Gf.Vec3d(*look_at)
    _set_prim_world_transform(camera.GetPrim(), eye_v, _look_at_rotation(eye_v, target_v))
    return path


def _create_wrist_camera(
    path: str,
    hand_path: str,
    width: int = CAMERA_WIDTH,
    height: int = CAMERA_HEIGHT,
) -> str:
    """Mount RealSense D435 on panda_hand; pose from gripper geometry (see motion_server_wrist_camera)."""
    stage = omni.usd.get_context().get_stage()
    hand_prim = stage.GetPrimAtPath(hand_path)
    if not hand_prim.IsValid():
        raise ValueError(f"Franka hand prim not found: {hand_path}")

    camera = UsdGeom.Camera.Define(stage, path)
    configure_realsense_d435_prim(camera, width, height)
    translation, rotation, meta = compute_wrist_camera_local_pose(stage, hand_path)
    _set_prim_local_transform(camera.GetPrim(), translation, rotation)
    print(format_wrist_camera_pose_report(hand_path, meta))

    wrist_prim = camera.GetPrim()
    if wrist_prim.GetParent().GetPath() != hand_prim.GetPath():
        print(f"Warning: wrist camera parent is {wrist_prim.GetParent().GetPath()}, expected {hand_path}")
    return path


def _spawn_rigid_prop(path: str, shape: str, **kwargs) -> None:
    if shape == "sphere":
        radii = kwargs.pop("radii", kwargs.pop("sizes", 0.03))
        Sphere(path, radii=radii, **kwargs)
    elif shape == "cube":
        sizes = kwargs.pop("sizes", 0.04)
        if isinstance(sizes, (tuple, list)) and len(sizes) == 3:
            Cube(path, sizes=1.0, scales=sizes, **kwargs)
        else:
            Cube(path, sizes=sizes, **kwargs)
    elif shape == "cylinder":
        radius = kwargs.pop("radius")
        height = kwargs.pop("height")
        Cylinder(path, radii=radius, heights=height, **kwargs)
    else:
        raise ValueError(shape)
    GeomPrim(path, apply_collision_apis=True)
    RigidPrim(path, masses=[0.05])


def _add_task_props(task: str) -> None:
    tasks = {
        "apple_on_plate": [
            ("sphere", "/World/Tasks/Apple", {"sizes": 0.04, "positions": _on_table(-0.12, -0.22), "colors": (0.85, 0.1, 0.1)}),
            ("cylinder", "/World/Tasks/Plate", {"radius": 0.08, "height": 0.01, "positions": _on_table(0.10, -0.22, 0.005), "colors": (0.9, 0.9, 0.95)}),
        ],
        "pipette_in_tray": [
            ("cylinder", "/World/Tasks/Pipette", {"radius": 0.008, "height": 0.18, "positions": _on_table(-0.10, 0.10, 0.08), "colors": (0.2, 0.2, 0.25)}),
            ("cube", "/World/Tasks/Tray", {"sizes": (0.18, 0.12, 0.02), "positions": _on_table(0.05, 0.10, 0.01), "colors": (0.75, 0.75, 0.8)}),
        ],
        "red_cube_in_tape_roll": [
            ("cube", "/World/Tasks/RedCube", {"sizes": 0.04, "positions": _on_table(-0.10, 0.0), "colors": (0.9, 0.05, 0.05)}),
            ("cylinder", "/World/Tasks/TapeRoll", {"radius": 0.05, "height": 0.03, "positions": _on_table(0.10, 0.0, 0.015), "colors": (0.15, 0.15, 0.15)}),
        ],
        "knife_in_box": [
            ("cube", "/World/Tasks/Knife", {"sizes": (0.18, 0.025, 0.01), "positions": _on_table(-0.12, -0.05), "colors": (0.7, 0.72, 0.75)}),
            ("cube", "/World/Tasks/Box", {"sizes": (0.22, 0.14, 0.06), "positions": _on_table(0.10, -0.05, 0.03), "colors": (0.45, 0.35, 0.25)}),
        ],
        "objects_in_bowl": [
            ("cylinder", "/World/Tasks/Bowl", {"radius": 0.09, "height": 0.05, "positions": _on_table(0.05, 0.06, 0.025), "colors": (0.3, 0.45, 0.7)}),
            ("sphere", "/World/Tasks/ObjectA", {"sizes": 0.025, "positions": _on_table(-0.05, 0.10), "colors": (0.9, 0.75, 0.1)}),
            ("cube", "/World/Tasks/ObjectB", {"sizes": 0.025, "positions": _on_table(0.0, 0.03), "colors": (0.2, 0.7, 0.3)}),
            ("sphere", "/World/Tasks/ObjectC", {"sizes": 0.02, "positions": _on_table(0.10, 0.08), "colors": (0.8, 0.4, 0.9)}),
        ],
    }
    if task not in tasks:
        return
    for shape, path, params in tasks[task]:
        _spawn_rigid_prop(path, shape, **params)


def _add_lab_backdrop() -> None:
    """Neutral walls so exterior cameras see a lab corner instead of an infinite gray void."""
    wall_height = 2.4
    cx, cy, _ = DESK_CENTER
    wall_color = (0.78, 0.76, 0.73)
    thin = 0.06

    Cube(
        "/World/Room/BackWall",
        sizes=1.0,
        scales=(4.5, thin, wall_height),
        positions=(cx, cy + 1.85, wall_height / 2.0),
        colors=wall_color,
    )
    Cube(
        "/World/Room/LeftWall",
        sizes=1.0,
        scales=(thin, 3.5, wall_height),
        positions=(cx - 1.6, cy + 0.6, wall_height / 2.0),
        colors=(0.74, 0.72, 0.70),
    )
    Cube(
        "/World/Room/RightWall",
        sizes=1.0,
        scales=(thin, 2.8, wall_height),
        positions=(cx + 1.6, cy + 0.4, wall_height / 2.0),
        colors=(0.76, 0.74, 0.71),
    )


def _setup_lighting() -> None:
    """Soft dome + overhead key light (avoids blown-out wrist views)."""
    dome = DomeLight("/World/DomeLight")
    dome.set_intensities(DOME_LIGHT_INTENSITY)
    dome.set_colors(DOME_LIGHT_COLOR)

    cx, cy, _ = DESK_CENTER
    key = RectLight(
        "/World/WorkspaceKeyLight",
        widths=1.4,
        heights=1.0,
        positions=(cx, cy + 0.12, DESK_TOP_Z + 1.35),
    )
    key.set_intensities(WORKSPACE_KEY_LIGHT_INTENSITY)
    key.set_colors(WORKSPACE_KEY_LIGHT_COLOR)


def build_droid_workspace() -> None:
    """Standing desk at 0.75 m, ground plane, backdrop, and soft lighting."""
    stage_utils.set_stage_units(meters_per_unit=1.0)
    GroundPlane("/World/GroundPlane", sizes=10.0, colors=(0.42, 0.42, 0.44))

    Cube(
        DESK_TOP_PATH,
        sizes=1.0,
        scales=(DESK_WIDTH, DESK_DEPTH, DESK_THICKNESS),
        positions=DESK_CENTER,
        colors=(0.55, 0.42, 0.30),
    )
    _make_static_collider(DESK_TOP_PATH)

    leg_height = DESK_TOP_Z - DESK_THICKNESS
    for idx, (lx, ly) in enumerate(_desk_leg_positions()):
        Cylinder(
            f"/World/Desk/Leg_{idx}",
            radii=0.025,
            heights=leg_height,
            positions=(lx, ly, leg_height / 2.0),
            colors=(0.4, 0.4, 0.42),
        )

    _add_lab_backdrop()
    _setup_lighting()


def load_franka_on_desk(assets_root: str, franka_path: str = DEFAULT_FRANKA_PATH) -> None:
    stage = omni.usd.get_context().get_stage()
    mount_path = ROBOT_MOUNT_PATH
    UsdGeom.Xform.Define(stage, mount_path)
    mount_xform = UsdGeom.XformCommonAPI(stage.GetPrimAtPath(mount_path))
    mount_xform.SetTranslate(Gf.Vec3d(*ROBOT_POSITION))
    mount_xform.SetRotate((0.0, 0.0, ROBOT_YAW_DEG), UsdGeom.XformCommonAPI.RotationOrderXYZ)

    stage_utils.add_reference_to_stage(assets_root + FRANKA_USD_PATH, franka_path)
    robot = stage.GetPrimAtPath(franka_path)
    robot.GetVariantSet("Gripper").SetVariantSelection("AlternateFinger")
    robot.GetVariantSet("Mesh").SetVariantSelection("Quality")


def _pull_back_eye(
    eye: tuple[float, float, float],
    look: tuple[float, float, float],
    scale: float,
) -> tuple[float, float, float]:
    """Move the camera along its view ray; look-at orientation stays the same."""
    return tuple(look[i] + scale * (eye[i] - look[i]) for i in range(3))


def setup_droid_cameras(franka_path: str = DEFAULT_FRANKA_PATH) -> list[CameraSpec]:
    """DROID ZED 2 tripods + RealSense D435 wrist camera (320×180)."""
    stage = omni.usd.get_context().get_stage()
    UsdGeom.Xform.Define(stage, "/World/Cameras")

    wx, wy, _ = DESK_CENTER
    look = SCENE_CAMERA_TARGET
    hand_path = f"{franka_path}/panda_hand"
    ext1_path = "/World/Cameras/external_1"
    ext2_path = "/World/Cameras/external_2"
    wrist_path = f"{hand_path}/wrist_camera"

    ext1_base = (wx + _EXT1_EYE[0], wy + _EXT1_EYE[1], DESK_TOP_Z + _EXT1_EYE[2])
    ext2_base = (wx + _EXT2_EYE[0], wy + _EXT2_EYE[1], DESK_TOP_Z + _EXT2_EYE[2])
    ext1_eye = _pull_back_eye(ext1_base, look, EXTERIOR_CAMERA_PULLBACK)
    ext2_eye = _pull_back_eye(ext2_base, look, EXTERIOR_CAMERA_PULLBACK)
    _create_world_camera(ext1_path, ext1_eye, look)
    _create_world_camera(ext2_path, ext2_eye, look)
    _create_wrist_camera(wrist_path, hand_path)

    specs = [
        CameraSpec("external_1", "Exterior 1 (left, ZED 2)", ext1_path),
        CameraSpec("external_2", "Exterior 2 (front, ZED 2)", ext2_path),
        CameraSpec("wrist", "Wrist (RealSense D435)", wrist_path),
    ]
    return [s for s in specs if stage.GetPrimAtPath(s.prim_path).IsValid()]


def spawn_droid_scene(assets_root: str, franka_path: str = DEFAULT_FRANKA_PATH, task: str = "apple_on_plate") -> None:
    """Full MolmoAct2-DROID validation layout for the motion server."""
    build_droid_workspace()
    load_franka_on_desk(assets_root, franka_path)
    if task != "empty":
        _add_task_props(task)
