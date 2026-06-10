# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Long-finger gripper mod for the Isaac Sim Franka Panda.

Replaces the stock finger visuals/collisions with ``panda_long_finger.stl`` on
``panda_leftfinger`` / ``panda_rightfinger``. Tune the constants below if the mesh
is exported in different units or axis conventions.
"""

from __future__ import annotations

import struct
from pathlib import Path

from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics

_PROJECT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _PROJECT_DIR.parent

# --- User-tunable mount parameters -------------------------------------------------

# Set False to keep the stock Isaac ``AlternateFinger`` meshes.
ENABLE_LONG_FINGER_MOD = True

# STL path (mm-style CAD export; 20×58×14.5 mm bbox in the shipped file).
LONG_FINGER_STL_PATH = _PROJECT_DIR / "assets" / "panda_long_finger.stl"

# Scale STL vertices to meters (0.001 when the STL is in millimeters).
LONG_FINGER_STL_SCALE = 0.001

# Pose of the mesh in each finger *link* local frame (before right-finger mirror).
# Start with identity; adjust if the finger points the wrong way in sim.
LONG_FINGER_LOCAL_TRANSLATION = (0.0, 0.0, 0.0)
LONG_FINGER_LOCAL_ROTATION_DEG_XYZ = (0.0, 0.0, 0.0)

# Mirror the mesh on ``panda_rightfinger`` across the finger link Y axis.
LONG_FINGER_MIRROR_RIGHT_FINGER_Y = True

# Extra +Z offset (hand frame, meters) applied to the wrist-camera gripper center
# after mesh install. Leave at 0 to auto-estimate from the STL bounds; set manually
# if the wrist view is mis-framed (see ``launch_compute_wrist_camera_pose.bat``).
LONG_FINGER_GRIPPER_CENTER_Z_OFFSET_M: float | None = None

LEFT_FINGER_NAME = "panda_leftfinger"
RIGHT_FINGER_NAME = "panda_rightfinger"
LONG_FINGER_MESH_NAME = "long_finger_mesh"

# Populated by :func:`apply_long_finger_mod` for wrist-camera placement.
_gripper_center_z_offset_hand: float = 0.0


def get_gripper_center_z_offset_hand() -> float:
    """Hand-frame +Z shift for gripper center (fingertip vs. joint-origin midpoint)."""
    return _gripper_center_z_offset_hand


def _load_binary_stl(path: Path) -> tuple[list[Gf.Vec3f], list[int], list[int]]:
    data = path.read_bytes()
    if len(data) < 84:
        raise ValueError(f"STL too small: {path}")
    if data.lstrip().startswith(b"solid"):
        raise ValueError(f"ASCII STL not supported: {path}")

    (tri_count,) = struct.unpack_from("<I", data, 80)
    expected = 84 + tri_count * 50
    if len(data) < expected:
        raise ValueError(f"Truncated binary STL ({len(data)} bytes, expected {expected}): {path}")

    points: list[Gf.Vec3f] = []
    index_map: dict[tuple[float, float, float], int] = {}
    face_vertex_counts: list[int] = []
    face_vertex_indices: list[int] = []

    offset = 84
    for _ in range(tri_count):
        offset += 12  # normal
        face = []
        for _ in range(3):
            x, y, z = struct.unpack_from("<3f", data, offset)
            offset += 12
            key = (round(x, 6), round(y, 6), round(z, 6))
            if key not in index_map:
                index_map[key] = len(points)
                points.append(Gf.Vec3f(x, y, z))
            face.append(index_map[key])
        offset += 2  # attribute byte count
        face_vertex_counts.append(3)
        face_vertex_indices.extend(face)

    return points, face_vertex_counts, face_vertex_indices


def _mesh_bounds(points: list[Gf.Vec3f]) -> tuple[Gf.Vec3d, Gf.Vec3d]:
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    zs = [p[2] for p in points]
    return Gf.Vec3d(min(xs), min(ys), min(zs)), Gf.Vec3d(max(xs), max(ys), max(zs))


def _scaled_rotated_points(
    points: list[Gf.Vec3f],
    scale: float,
    translation: tuple[float, float, float],
    rotation_deg_xyz: tuple[float, float, float],
    mirror_y: bool,
) -> list[Gf.Vec3f]:
    rotation = (
        Gf.Rotation(Gf.Vec3d.XAxis(), rotation_deg_xyz[0])
        * Gf.Rotation(Gf.Vec3d.YAxis(), rotation_deg_xyz[1])
        * Gf.Rotation(Gf.Vec3d.ZAxis(), rotation_deg_xyz[2])
    )
    tx, ty, tz = translation
    out: list[Gf.Vec3f] = []
    for p in points:
        v = Gf.Vec3d(p[0] * scale, p[1] * scale, p[2] * scale)
        if mirror_y:
            v[1] *= -1.0
        v = rotation.TransformDir(v)
        out.append(Gf.Vec3f(v[0] + tx, v[1] + ty, v[2] + tz))
    return out


def _hide_and_disable_collision(prim: Usd.Prim) -> None:
    if not prim.IsValid():
        return
    if prim.IsA(UsdGeom.Mesh) or prim.IsA(UsdGeom.Capsule) or prim.IsA(UsdGeom.Cube):
        UsdGeom.Imageable(prim).MakeInvisible()
        if prim.HasAPI(UsdPhysics.CollisionAPI):
            UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Set(False)
    for child in prim.GetChildren():
        _hide_and_disable_collision(child)


def _define_mesh_prim(
    stage: Usd.Stage,
    path: str,
    points: list[Gf.Vec3f],
    face_vertex_counts: list[int],
    face_vertex_indices: list[int],
) -> Usd.Prim:
    mesh = UsdGeom.Mesh.Define(stage, path)
    mesh.GetPointsAttr().Set(points)
    mesh.GetFaceVertexCountsAttr().Set(face_vertex_counts)
    mesh.GetFaceVertexIndicesAttr().Set(face_vertex_indices)
    mesh.GetSubdivisionSchemeAttr().Set(UsdGeom.Tokens.none)
    prim = mesh.GetPrim()
    UsdPhysics.CollisionAPI.Apply(prim)
    UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Set(True)
    return prim


def _finger_tip_z_in_hand(stage: Usd.Stage, hand_path: str, finger_path: str, points: list[Gf.Vec3f]) -> float:
    hand_prim = stage.GetPrimAtPath(hand_path)
    finger_prim = stage.GetPrimAtPath(finger_path)
    if not hand_prim.IsValid() or not finger_prim.IsValid():
        return 0.0

    hand_world = UsdGeom.Xformable(hand_prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    finger_world = UsdGeom.Xformable(finger_prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    hand_from_finger = hand_world.GetInverse() * finger_world

    tip_z = max(hand_from_finger.Transform(Gf.Vec3d(p[0], p[1], p[2]))[2] for p in points)
    return float(tip_z)


def apply_long_finger_mod(stage: Usd.Stage, franka_path: str) -> None:
    """Attach long-finger STL meshes and hide stock finger geometry."""
    global _gripper_center_z_offset_hand

    if not ENABLE_LONG_FINGER_MOD:
        _gripper_center_z_offset_hand = 0.0
        return

    stl_path = Path(LONG_FINGER_STL_PATH)
    if not stl_path.is_file():
        print(f"Warning: long finger STL not found at {stl_path}; keeping stock gripper meshes.")
        _gripper_center_z_offset_hand = 0.0
        return

    raw_points, face_counts, face_indices = _load_binary_stl(stl_path)
    hand_path = f"{franka_path}/panda_hand"

    tip_z_values: list[float] = []
    for finger_name, mirror_y in (
        (LEFT_FINGER_NAME, False),
        (RIGHT_FINGER_NAME, LONG_FINGER_MIRROR_RIGHT_FINGER_Y),
    ):
        finger_path = f"{franka_path}/{finger_name}"
        finger_prim = stage.GetPrimAtPath(finger_path)
        if not finger_prim.IsValid():
            print(f"Warning: finger prim missing: {finger_path}")
            continue

        _hide_and_disable_collision(finger_prim)

        points = _scaled_rotated_points(
            raw_points,
            LONG_FINGER_STL_SCALE,
            LONG_FINGER_LOCAL_TRANSLATION,
            LONG_FINGER_LOCAL_ROTATION_DEG_XYZ,
            mirror_y=mirror_y,
        )
        mesh_path = f"{finger_path}/{LONG_FINGER_MESH_NAME}"
        _define_mesh_prim(stage, mesh_path, points, face_counts, face_indices)
        tip_z_values.append(_finger_tip_z_in_hand(stage, hand_path, finger_path, points))

    if LONG_FINGER_GRIPPER_CENTER_Z_OFFSET_M is not None:
        _gripper_center_z_offset_hand = LONG_FINGER_GRIPPER_CENTER_Z_OFFSET_M
    elif tip_z_values:
        # Link origins sit near Z≈0.058 m; shift gripper center toward fingertip +Z extent.
        joint_origin_z = 0.0584
        avg_tip_z = sum(tip_z_values) / len(tip_z_values)
        _gripper_center_z_offset_hand = max(0.0, avg_tip_z - joint_origin_z)
    else:
        _gripper_center_z_offset_hand = 0.0

    lo, hi = _mesh_bounds(
        _scaled_rotated_points(
            raw_points,
            LONG_FINGER_STL_SCALE,
            LONG_FINGER_LOCAL_TRANSLATION,
            LONG_FINGER_LOCAL_ROTATION_DEG_XYZ,
            mirror_y=False,
        )
    )
    print(
        "Long finger mod applied:\n"
        f"  STL: {stl_path}\n"
        f"  scale: {LONG_FINGER_STL_SCALE}, translation: {LONG_FINGER_LOCAL_TRANSLATION}, "
        f"rotation deg XYZ: {LONG_FINGER_LOCAL_ROTATION_DEG_XYZ}\n"
        f"  mesh bounds in finger link frame (m): "
        f"X[{lo[0]:.4f},{hi[0]:.4f}] Y[{lo[1]:.4f},{hi[1]:.4f}] Z[{lo[2]:.4f},{hi[2]:.4f}]\n"
        f"  wrist gripper-center +Z offset (hand frame): {_gripper_center_z_offset_hand:.4f} m"
    )
