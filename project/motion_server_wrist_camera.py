# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Auto-compute Franka wrist (gripper) camera pose and RealSense D435 intrinsics.

Mount model: centered behind the gripper finger plane in ``panda_hand`` local frame,
optical axis aligned to world −Z (straight down) with image up aligned to finger spread (+Y).

``panda_hand`` frame (Isaac Franka URDF): +Z → fingertips, +Y → finger spread, +X → lateral.
"""

from __future__ import annotations

from dataclasses import dataclass

from pxr import Gf, Sdf, Usd, UsdGeom

from motion_server_gripper import get_gripper_center_z_offset_hand

# Intel RealSense D435 color stream @ 640×480 (ROS camera_info from Isaac camera_ros example).
# source/standalone_examples/deprecated/api/isaacsim.sensors.camera/camera_ros.py
REALSENSE_D435_NATIVE_WIDTH = 640
REALSENSE_D435_NATIVE_HEIGHT = 480
REALSENSE_D435_FX = 612.4178466796875
REALSENSE_D435_FY = 612.362060546875
REALSENSE_D435_CX = 309.72296142578125
REALSENSE_D435_CY = 245.35870361328125
REALSENSE_D435_FOCUS_DISTANCE = 0.5
REALSENSE_D435_F_STOP = 5.6
REALSENSE_D435_CLIP_NEAR = 0.01
REALSENSE_D435_CLIP_FAR = 100.0
REALSENSE_D435_EXPOSURE_TIME = 0.015

# URDF fallback when finger links are unavailable at setup time.
FRANKA_FINGER_JOINT_LOCAL_Z = 0.0584
DEFAULT_BEHIND_GRIPPER_M = 0.03

# Manual tweaks applied after auto-compute (hand-local frame: +Z fingers, +Y finger spread).
WRIST_CAMERA_USER_TRANSLATION_OFFSET = (0.0, 0.0, -0.065)
WRIST_CAMERA_USER_YAW_DEG = -90.0  # yaw right about hand +Z (world vertical when level)
WRIST_CAMERA_USER_ROLL_DEG = 180.0  # spin about optical axis (−Z) to flip captured image

LEFT_FINGER_NAME = "panda_leftfinger"
RIGHT_FINGER_NAME = "panda_rightfinger"


@dataclass(frozen=True)
class WristCameraPose:
    """Wrist camera mount in ``panda_hand`` local coordinates."""

    local_translation: tuple[float, float, float]
    local_rotation_deg_xyz: tuple[float, float, float]
    gripper_center_local: tuple[float, float, float]
    behind_distance_m: float
    used_finger_geometry: bool


def scale_realsense_d435_intrinsics(width: int, height: int) -> tuple[float, float, float, float]:
    sx = width / REALSENSE_D435_NATIVE_WIDTH
    sy = height / REALSENSE_D435_NATIVE_HEIGHT
    return (
        REALSENSE_D435_FX * sx,
        REALSENSE_D435_FY * sy,
        REALSENSE_D435_CX * sx,
        REALSENSE_D435_CY * sy,
    )


def _project_on_plane(vector: Gf.Vec3d, plane_normal: Gf.Vec3d) -> Gf.Vec3d:
    n = plane_normal.GetNormalized()
    return vector - n * Gf.Dot(vector, n)


def _look_at_rotation(eye: Gf.Vec3d, target: Gf.Vec3d, up: Gf.Vec3d) -> Gf.Rotation:
    return Gf.Matrix4d().SetLookAt(eye, target, up).GetInverse().ExtractRotation()


def _rotation_to_euler_xyz_deg(rotation: Gf.Rotation) -> tuple[float, float, float]:
    """Extract XYZ Euler angles in degrees (for logging / manual tweaks)."""
    deg = rotation.Decompose(Gf.Vec3d.XAxis(), Gf.Vec3d.YAxis(), Gf.Vec3d.ZAxis())
    return (deg[0], deg[1], deg[2])


def _hand_local_origin(stage: Usd.Stage, hand_path: str, child_path: str) -> Gf.Vec3d | None:
    hand_prim = stage.GetPrimAtPath(hand_path)
    child_prim = stage.GetPrimAtPath(child_path)
    if not hand_prim.IsValid() or not child_prim.IsValid():
        return None

    hand_world = UsdGeom.Xformable(hand_prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    child_world = UsdGeom.Xformable(child_prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    child_origin_world = child_world.Transform(Gf.Vec3d(0.0, 0.0, 0.0))
    return hand_world.GetInverse().Transform(child_origin_world)


def _gripper_center_local(stage: Usd.Stage, hand_path: str) -> tuple[Gf.Vec3d, bool]:
    left = _hand_local_origin(stage, hand_path, f"{hand_path}/{LEFT_FINGER_NAME}")
    right = _hand_local_origin(stage, hand_path, f"{hand_path}/{RIGHT_FINGER_NAME}")
    if left is not None and right is not None:
        return (left + right) * 0.5, True
    return Gf.Vec3d(0.0, 0.0, FRANKA_FINGER_JOINT_LOCAL_Z), False


def _apply_user_wrist_camera_offsets(
    local_translation: Gf.Vec3d,
    local_rotation: Gf.Rotation,
) -> tuple[Gf.Vec3d, Gf.Rotation]:
    ox, oy, oz = WRIST_CAMERA_USER_TRANSLATION_OFFSET
    local_translation = local_translation + Gf.Vec3d(ox, oy, oz)
    if WRIST_CAMERA_USER_YAW_DEG != 0.0:
        local_rotation = local_rotation * Gf.Rotation(Gf.Vec3d.ZAxis(), WRIST_CAMERA_USER_YAW_DEG)
    if WRIST_CAMERA_USER_ROLL_DEG != 0.0:
        # USD camera looks along −Z; roll spins the image in the sensor plane.
        local_rotation = local_rotation * Gf.Rotation(Gf.Vec3d(0.0, 0.0, -1.0), WRIST_CAMERA_USER_ROLL_DEG)
    return local_translation, local_rotation


def compute_wrist_camera_local_pose(
    stage: Usd.Stage,
    hand_path: str,
    *,
    behind_distance_m: float = DEFAULT_BEHIND_GRIPPER_M,
    world_down: Gf.Vec3d | None = None,
) -> tuple[Gf.Vec3d, Gf.Rotation, WristCameraPose]:
    """Compute wrist camera translation + rotation in ``panda_hand`` local frame.

    Position: gripper center (midpoint of finger links) offset along −hand Z (behind fingers).
    Orientation: USD camera −Z → ``world_down`` (default world −Z), image up → hand +Y.
    """
    if world_down is None:
        world_down = Gf.Vec3d(0.0, 0.0, -1.0)

    hand_prim = stage.GetPrimAtPath(hand_path)
    if not hand_prim.IsValid():
        raise ValueError(f"Hand prim not found: {hand_path}")

    hand_xform = UsdGeom.Xformable(hand_prim)
    hand_world = hand_xform.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    hand_rot = hand_world.ExtractRotation()

    gripper_center, used_fingers = _gripper_center_local(stage, hand_path)
    tip_offset_z = get_gripper_center_z_offset_hand()
    if tip_offset_z != 0.0:
        gripper_center = gripper_center + Gf.Vec3d(0.0, 0.0, tip_offset_z)
    # Behind gripper: −hand Z from finger-base center (toward palm / wrist).
    local_translation = gripper_center + Gf.Vec3d(-0.0668, 0.0, -behind_distance_m)

    cam_pos_world = hand_world.Transform(local_translation)
    target_world = cam_pos_world + world_down.GetNormalized()

    hand_y_world = hand_rot.TransformDir(Gf.Vec3d(0.0, 1.0, 0.0))
    up_world = _project_on_plane(hand_y_world, world_down)
    if up_world.GetLength() < 1e-6:
        hand_x_world = hand_rot.TransformDir(Gf.Vec3d(1.0, 0.0, 0.0))
        up_world = _project_on_plane(hand_x_world, world_down)
    up_world.Normalize()

    cam_rot_world = _look_at_rotation(cam_pos_world, target_world, up_world)
    local_rotation = hand_rot.GetInverse() * cam_rot_world
    local_translation, local_rotation = _apply_user_wrist_camera_offsets(local_translation, local_rotation)

    meta = WristCameraPose(
        local_translation=(local_translation[0], local_translation[1], local_translation[2]),
        local_rotation_deg_xyz=_rotation_to_euler_xyz_deg(local_rotation),
        gripper_center_local=(gripper_center[0], gripper_center[1], gripper_center[2]),
        behind_distance_m=behind_distance_m,
        used_finger_geometry=used_fingers,
    )
    return local_translation, local_rotation, meta


def apply_opencv_pinhole_schema(
    prim: Usd.Prim,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    width: int,
    height: int,
) -> None:
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


def configure_realsense_d435_prim(camera: UsdGeom.Camera, width: int, height: int) -> None:
    """Apply RealSense D435 color intrinsics scaled to the render resolution."""
    fx, fy, cx, cy = scale_realsense_d435_intrinsics(width, height)
    camera.GetProjectionAttr().Set("perspective")
    camera.GetFocusDistanceAttr().Set(REALSENSE_D435_FOCUS_DISTANCE)
    camera.GetClippingRangeAttr().Set(Gf.Vec2f(REALSENSE_D435_CLIP_NEAR, REALSENSE_D435_CLIP_FAR))

    focal_length = ((fx + fy) * 0.5) * 1.4e-3 / 10.0
    horizontal_aperture = 1.4e-3 * width / 10.0
    vertical_aperture = 1.4e-3 * height / 10.0
    camera.GetFocalLengthAttr().Set(focal_length)
    camera.GetHorizontalApertureAttr().Set(horizontal_aperture)
    camera.GetVerticalApertureAttr().Set(vertical_aperture)

    prim = camera.GetPrim()
    prim.CreateAttribute("exposure:fStop", Sdf.ValueTypeNames.Float).Set(REALSENSE_D435_F_STOP)
    prim.CreateAttribute("exposure:time", Sdf.ValueTypeNames.Float).Set(REALSENSE_D435_EXPOSURE_TIME)
    prim.CreateAttribute("exposure:responsivity", Sdf.ValueTypeNames.Float).Set(1.0)
    apply_opencv_pinhole_schema(prim, fx, fy, cx, cy, width, height)


def format_wrist_camera_pose_report(hand_path: str, meta: WristCameraPose) -> str:
    tx, ty, tz = meta.local_translation
    rx, ry, rz = meta.local_rotation_deg_xyz
    cx, cy, cz = meta.gripper_center_local
    source = "finger link geometry" if meta.used_finger_geometry else "URDF finger-joint fallback"
    return (
        f"Wrist camera pose ({hand_path} local frame)\n"
        f"  gripper center ({source}): ({cx:.4f}, {cy:.4f}, {cz:.4f}) m\n"
        f"  behind offset (−Z): {meta.behind_distance_m:.4f} m\n"
        f"  translation (x,y,z): ({tx:.4f}, {ty:.4f}, {tz:.4f}) m\n"
        f"  rotation euler XYZ deg: ({rx:.2f}, {ry:.2f}, {rz:.2f})\n"
        f"  user offset (x,y,z): {WRIST_CAMERA_USER_TRANSLATION_OFFSET} m, yaw {WRIST_CAMERA_USER_YAW_DEG} deg, roll {WRIST_CAMERA_USER_ROLL_DEG} deg\n"
        f"  view: world −Z, image up: hand +Y (finger spread)\n"
        f"  intrinsics: RealSense D435 color @ {REALSENSE_D435_NATIVE_WIDTH}×{REALSENSE_D435_NATIVE_HEIGHT}"
    )
