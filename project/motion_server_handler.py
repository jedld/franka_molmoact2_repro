# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Isaac Sim backend for the motion_server HTTP API.

Implements the same command semantics as project/motion_server.cpp (libfranka)
so existing HTTP clients can drive a simulated Franka Panda without changes.
"""

from __future__ import annotations

import math
import threading
from typing import Callable

import numpy as np
import warp as wp
from isaacsim.core.experimental.prims import Articulation, RigidPrim
from isaacsim.core.experimental.utils.impl.transform import quaternion_conjugate, quaternion_multiplication


StepFn = Callable[[], None]

# Franka Panda joint limits (radians) — matches motion_server.cpp
_Q_MIN = np.array([-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973])
_Q_MAX = np.array([2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973])

# Canonical home pose used by the real motion_server GUI.
_Q_HOME = np.array([0.0, -math.pi / 4, 0.0, -3.0 * math.pi / 4, 0.0, math.pi / 2, math.pi / 4])

DEFAULT_MOTION_TIME = 5.0
WORKSPACE_SPHERE_RADIUS = 0.855
MIN_TABLE_Z = 0.015
GRIPPER_OPEN = 0.04
GRIPPER_MAX_WIDTH = 0.08


def _quat_wxyz_to_matrix(quat: np.ndarray) -> np.ndarray:
    w, x, y, z = quat
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def _matrix_to_column_major_4x4(rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    pose = np.zeros(16, dtype=np.float64)
    pose[0:3] = rotation[:, 0]
    pose[4:7] = rotation[:, 1]
    pose[8:11] = rotation[:, 2]
    pose[12:15] = translation
    pose[15] = 1.0
    return pose


def _invert_rigid_transform(rotation: np.ndarray, translation: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    inv_r = rotation.T
    inv_t = -inv_r @ translation
    return inv_r, inv_t


def _compose_rigid_transform(
    r_a: np.ndarray, t_a: np.ndarray, r_b: np.ndarray, t_b: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    r = r_a @ r_b
    t = r_a @ t_b + t_a
    return r, t


def _rotation_zyx(alpha: float, beta: float, gamma: float) -> np.ndarray:
    """ZYX Euler (Alpha=Z, Beta=Y, Gamma=X) — matches motion_server.cpp."""
    ca, sa = math.cos(alpha), math.sin(alpha)
    cb, sb = math.cos(beta), math.sin(beta)
    cg, sg = math.cos(gamma), math.sin(gamma)
    return np.array(
        [
            [ca * cb, ca * sb * sg - sa * cg, ca * sb * cg + sa * sg],
            [sa * cb, sa * sb * sg + ca * cg, sa * sb * cg - ca * sg],
            [-sb, cb * sg, cb * cg],
        ],
        dtype=np.float64,
    )


def _euler_zyx_from_ot_ee(pose: np.ndarray) -> tuple[float, float, float]:
    """Extract ZYX Euler angles from a column-major O_T_EE matrix."""
    beta = math.atan2(-pose[2], math.sqrt(pose[0] ** 2 + pose[1] ** 2))
    cos_beta = math.cos(beta)
    if abs(cos_beta) < 1e-5:
        if beta > 0:
            beta = math.pi / 2
            gamma = math.atan2(pose[4], pose[5])
            alpha = 0.0
        else:
            beta = -math.pi / 2
            gamma = -math.atan2(pose[4], pose[5])
            alpha = 0.0
    else:
        alpha = math.atan2(pose[1] / cos_beta, pose[0] / cos_beta)
        gamma = math.atan2(pose[6] / cos_beta, pose[10] / cos_beta)
    return alpha, beta, gamma


def _cubic_scalar(x0: float, xf: float, time: float, tf: float) -> tuple[float, float]:
    t2 = time * time
    t3 = time * time * time
    tf2 = tf * tf
    tf3 = tf * tf * tf
    delta = xf - x0
    a2 = 3.0 * delta / tf2
    a3 = -2.0 * delta / tf3
    value = x0 + a2 * t2 + a3 * t3
    velocity = 2.0 * a2 * time + 3.0 * a3 * t2
    return value, velocity


class IsaacMotionServerHandler:
    """Robot command handler backed by an Isaac Sim Franka articulation."""

    def __init__(self, robot_path: str, step_fn: StepFn) -> None:
        self._robot_path = robot_path
        self._step = step_fn
        self._mutex = threading.Lock()
        self._robot = Articulation(robot_path)
        self._ee = RigidPrim(f"{robot_path}/panda_hand")
        self._ee_link_index = self._robot.get_link_indices("panda_hand").list()[0]

    def initialize(self, skip_home: bool = False) -> None:
        if not skip_home:
            self._go_home_joints(tf=10.0)
        self.open_gripper([])

        pose = self._read_ot_ee()
        print("Initial O_T_EE (column-major):")
        for i in range(16):
            print(f"{pose[i]}, ", end="")
            if i % 4 == 3:
                print()
        alpha, beta, gamma = _euler_zyx_from_ot_ee(pose)
        print(
            f"Alpha(Z): {alpha:.2f}, Beta(Y): {beta:.2f}, Gamma(X): {gamma:.2f} radians | "
            f"X: {pose[12]:.3f}, Y: {pose[13]:.3f}, Z: {pose[14]:.3f} m"
        )

    def _read_ot_ee(self) -> np.ndarray:
        base_pos, base_ori = self._robot.get_world_poses()
        ee_pos, ee_ori = self._ee.get_world_poses()
        base_p = base_pos.numpy()[0]
        base_q = base_ori.numpy()[0]
        ee_p = ee_pos.numpy()[0]
        ee_q = ee_ori.numpy()[0]

        r_world_base = _quat_wxyz_to_matrix(base_q)
        r_world_ee = _quat_wxyz_to_matrix(ee_q)
        r_base_world, t_base_world = _invert_rigid_transform(r_world_base, base_p)
        r_base_ee, t_base_ee = _compose_rigid_transform(r_base_world, t_base_world, r_world_ee, ee_p)
        return _matrix_to_column_major_4x4(r_base_ee, t_base_ee)

    def _state_vector_from_ot_ee(self, pose: np.ndarray) -> list[float]:
        alpha, beta, gamma = _euler_zyx_from_ot_ee(pose)
        return [
            float(pose[12]),
            float(pose[13]),
            float(pose[14]),
            math.degrees(alpha),
            math.degrees(beta),
            math.degrees(gamma),
        ]

    def _current_arm_q(self) -> np.ndarray:
        return self._robot.get_dof_positions().numpy()[0, :7]

    def _current_gripper_q(self) -> np.ndarray:
        return self._robot.get_dof_positions().numpy()[0, 7:9]

    def _set_arm_targets(self, q: np.ndarray) -> None:
        self._robot.set_dof_position_targets(q.reshape(1, 7), dof_indices=list(range(7)))

    def _set_gripper_targets(self, width: float) -> None:
        half = float(np.clip(width / 2.0, 0.0, GRIPPER_OPEN))
        self._robot.set_dof_position_targets(
            np.array([[half, half]], dtype=np.float32), dof_indices=[7, 8]
        )

    def _simulate_for(self, seconds: float) -> None:
        steps = max(1, int(round(seconds / (1.0 / 60.0))))
        for _ in range(steps):
            self._step()

    def _go_home_joints(self, tf: float = 10.0) -> None:
        q_initial = self._current_arm_q()
        elapsed = 0.0
        dt = 1.0 / 60.0
        while elapsed < tf:
            t2 = elapsed * elapsed
            t3 = t2 * elapsed
            tf2 = tf * tf
            tf3 = tf2 * tf
            q_target = np.zeros(7)
            for i in range(7):
                delta = _Q_HOME[i] - q_initial[i]
                a2 = 3.0 * delta / tf2
                a3 = -2.0 * delta / tf3
                q_target[i] = q_initial[i] + a2 * t2 + a3 * t3
            self._set_arm_targets(q_target)
            self._step()
            elapsed += dt
        print("[goHomeJoints] reached home pose")

    def _differential_ik(
        self,
        goal_position: np.ndarray,
        goal_orientation: np.ndarray,
        method: str = "damped-least-squares",
    ) -> np.ndarray:
        current_q = self._current_arm_q()
        _, ee_pos, ee_ori = self._get_arm_state()
        jacobian = self._robot.get_jacobian_matrices().numpy()
        jacobian_ee = jacobian[:, self._ee_link_index - 1, :, :7]

        scale = 1.0
        damping = 0.05
        goal_quat_wp = wp.from_numpy(goal_orientation.reshape(1, 4), dtype=wp.float32)
        current_quat_wp = wp.from_numpy(ee_ori.reshape(1, 4), dtype=wp.float32)
        q_diff_wp = quaternion_multiplication(goal_quat_wp, quaternion_conjugate(current_quat_wp))
        q_diff = q_diff_wp.numpy()[0]
        error = np.concatenate(
            [goal_position - ee_pos, q_diff[1:] * np.sign(q_diff[0])]
        ).reshape(6, 1)

        if method == "damped-least-squares":
            j_t = np.swapaxes(jacobian_ee, 1, 2)
            lmbda = np.eye(jacobian_ee.shape[1]) * (damping**2)
            delta_q = scale * (j_t @ np.linalg.inv(jacobian_ee @ j_t + lmbda) @ error).squeeze(-1)
        else:
            delta_q = scale * (np.linalg.pinv(jacobian_ee) @ error).squeeze(-1)
        return current_q + delta_q[0]

    def _get_arm_state(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        q = self._current_arm_q()
        ee_pos, ee_ori = self._ee.get_world_poses()
        return q, ee_pos.numpy()[0], ee_ori.numpy()[0]

    def _ot_ee_to_world_target(self, pose: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        base_pos, base_ori = self._robot.get_world_poses()
        base_p = base_pos.numpy()[0]
        base_q = base_ori.numpy()[0]
        r_world_base = _quat_wxyz_to_matrix(base_q)
        r_base_ee = np.column_stack((pose[0:3], pose[4:7], pose[8:11]))
        t_base_ee = pose[12:15]
        r_world_ee = r_world_base @ r_base_ee
        t_world_ee = r_world_base @ t_base_ee + base_p

        # rotation matrix -> quaternion (wxyz)
        trace = np.trace(r_world_ee)
        if trace > 0:
            s = 0.5 / math.sqrt(trace + 1.0)
            w = 0.25 / s
            x = (r_world_ee[2, 1] - r_world_ee[1, 2]) * s
            y = (r_world_ee[0, 2] - r_world_ee[2, 0]) * s
            z = (r_world_ee[1, 0] - r_world_ee[0, 1]) * s
        elif r_world_ee[0, 0] > r_world_ee[1, 1] and r_world_ee[0, 0] > r_world_ee[2, 2]:
            s = 2.0 * math.sqrt(1.0 + r_world_ee[0, 0] - r_world_ee[1, 1] - r_world_ee[2, 2])
            w = (r_world_ee[2, 1] - r_world_ee[1, 2]) / s
            x = 0.25 * s
            y = (r_world_ee[0, 1] + r_world_ee[1, 0]) / s
            z = (r_world_ee[0, 2] + r_world_ee[2, 0]) / s
        elif r_world_ee[1, 1] > r_world_ee[2, 2]:
            s = 2.0 * math.sqrt(1.0 + r_world_ee[1, 1] - r_world_ee[0, 0] - r_world_ee[2, 2])
            w = (r_world_ee[0, 2] - r_world_ee[2, 0]) / s
            x = (r_world_ee[0, 1] + r_world_ee[1, 0]) / s
            y = 0.25 * s
            z = (r_world_ee[1, 2] + r_world_ee[2, 1]) / s
        else:
            s = 2.0 * math.sqrt(1.0 + r_world_ee[2, 2] - r_world_ee[0, 0] - r_world_ee[1, 1])
            w = (r_world_ee[1, 0] - r_world_ee[0, 1]) / s
            x = (r_world_ee[0, 2] + r_world_ee[2, 0]) / s
            y = (r_world_ee[1, 2] + r_world_ee[2, 1]) / s
            z = 0.25 * s
        return t_world_ee.astype(np.float32), np.array([w, x, y, z], dtype=np.float32)

    @staticmethod
    def _is_valid_cartesian_pose(numbers: list[float]) -> None:
        xf, yf, zf = numbers[0], numbers[1], numbers[2]
        if xf * xf + yf * yf + zf * zf > WORKSPACE_SPHERE_RADIUS**2:
            raise RuntimeError("The desired position is outside the workspace.")
        if zf < MIN_TABLE_Z:
            raise RuntimeError("The desired position is too close to the table.")

    @staticmethod
    def _is_valid_joint_pose(q: np.ndarray) -> bool:
        for i in range(7):
            if q[i] < _Q_MIN[i] or q[i] > _Q_MAX[i]:
                print(
                    f"Joint {i + 1} target {q[i]} rad outside [{_Q_MIN[i]}, {_Q_MAX[i]}]"
                )
                return False
        return True

    def move_to_joint_pose(self, numbers: list[float]) -> list[float]:
        with self._mutex:
            if len(numbers) < 7:
                raise RuntimeError("moveToJointPose requires at least 7 joint angles (rad).")
            q_target = np.array(numbers[:7], dtype=np.float64)
            if not self._is_valid_joint_pose(q_target):
                raise RuntimeError("Joint target outside Franka joint limits.")

            tf = DEFAULT_MOTION_TIME if len(numbers) < 8 else max(float(numbers[7]), DEFAULT_MOTION_TIME)
            q_initial = self._current_arm_q()
            elapsed = 0.0
            dt = 1.0 / 60.0
            try:
                while elapsed < tf:
                    # Cubic joint trajectory (same polynomial as motion_server.cpp).
                    t2 = elapsed * elapsed
                    t3 = t2 * elapsed
                    tf2 = tf * tf
                    tf3 = tf2 * tf
                    q_cmd = np.zeros(7)
                    for i in range(7):
                        delta = q_target[i] - q_initial[i]
                        a2 = 3.0 * delta / tf2
                        a3 = -2.0 * delta / tf3
                        q_cmd[i] = q_initial[i] + a2 * t2 + a3 * t3
                    self._set_arm_targets(q_cmd)
                    self._step()
                    elapsed += dt
            except Exception as exc:
                raise RuntimeError(f"collision_recovery: {exc}") from exc

            return self._current_arm_q().tolist()

    def move_to_cartesian(self, numbers: list[float]) -> list[float]:
        with self._mutex:
            if len(numbers) < 3:
                raise RuntimeError("Please provide at least 3 float numbers (x,y,z,t).")
            self._is_valid_cartesian_pose(numbers)

            xf, yf, zf = float(numbers[0]), float(numbers[1]), float(numbers[2])
            tf = DEFAULT_MOTION_TIME if len(numbers) < 4 else max(float(numbers[3]), DEFAULT_MOTION_TIME)

            delta_alpha = delta_beta = delta_gamma = 0.0
            is_rotation = False
            if len(numbers) >= 5:
                delta_alpha = float(numbers[4])
                if delta_alpha < -90 or delta_alpha > 90:
                    delta_alpha = 0.0
                    print("Error: deltaAlpha must be between -90 and 90 degrees.")
                else:
                    delta_alpha = math.radians(delta_alpha)
                    is_rotation = True
            if len(numbers) >= 6:
                delta_beta = float(numbers[5])
                if delta_beta < -90 or delta_beta > 90:
                    delta_beta = 0.0
                    print("Error: deltaBeta must be between -90 and 90 degrees.")
                else:
                    delta_beta = math.radians(delta_beta)
            if len(numbers) >= 7:
                delta_gamma = float(numbers[6])
                if delta_gamma < -90 or delta_gamma > 90:
                    delta_gamma = 0.0
                    print("Error: deltaGamma must be between -90 and 90 degrees.")
                else:
                    delta_gamma = math.radians(delta_gamma)

            initial_pose = self._read_ot_ee()
            alpha = beta = gamma = 0.0
            alpha_f = beta_f = gamma_f = 0.0
            if is_rotation:
                alpha, beta, gamma = _euler_zyx_from_ot_ee(initial_pose)
                alpha_f = alpha + delta_alpha
                beta_f = beta + delta_beta
                gamma_f = gamma + delta_gamma

            elapsed = 0.0
            dt = 1.0 / 60.0
            x0, y0, z0 = initial_pose[12], initial_pose[13], initial_pose[14]

            try:
                while elapsed < tf:
                    xt, vx = _cubic_scalar(x0, xf, elapsed, tf)
                    yt, vy = _cubic_scalar(y0, yf, elapsed, tf)
                    zt, vz = _cubic_scalar(z0, zf, elapsed, tf)
                    stop = abs(vx) < 0.0001 and abs(vy) < 0.0001 and abs(vz) < 0.0001 and elapsed > 1.0

                    pose_cmd = initial_pose.copy()
                    pose_cmd[12] = xt
                    pose_cmd[13] = yt
                    pose_cmd[14] = zt

                    if is_rotation:
                        at, va = _cubic_scalar(alpha, alpha_f, elapsed, tf)
                        bt, vb = _cubic_scalar(beta, beta_f, elapsed, tf)
                        gt, vg = _cubic_scalar(gamma, gamma_f, elapsed, tf)
                        rot = _rotation_zyx(at, bt, gt)
                        pose_cmd[0:3] = rot[:, 0]
                        pose_cmd[4:7] = rot[:, 1]
                        pose_cmd[8:11] = rot[:, 2]
                        stop = stop and abs(va) < 0.0001 and abs(vb) < 0.0001 and abs(vg) < 0.0001

                    goal_pos, goal_ori = self._ot_ee_to_world_target(pose_cmd)
                    q_cmd = self._differential_ik(goal_pos, goal_ori)
                    self._set_arm_targets(q_cmd)
                    self._step()
                    elapsed += dt
                    if elapsed >= tf or stop:
                        print(f"\n{elapsed:.3f}sec : End of motion .............")
                        break
            except Exception as exc:
                raise RuntimeError(f"collision_recovery: {exc}") from exc

            final_pose = self._read_ot_ee()
            return self._state_vector_from_ot_ee(final_pose)

    def close_gripper(self, numbers: list[float]) -> str:
        with self._mutex:
            try:
                grasping_width = 0.01 if not numbers else float(numbers[0])
                if grasping_width > GRIPPER_MAX_WIDTH:
                    return (
                        f"Object is too large for the current fingers on the gripper: {GRIPPER_MAX_WIDTH}"
                    )
                self._set_gripper_targets(grasping_width)
                self._simulate_for(1.0)
                fingers = self._current_gripper_q()
                achieved = float(fingers[0] + fingers[1])
                if achieved > grasping_width + 0.015:
                    return "Failed to grasp object."
                return "Object grasped successfully."
            except Exception as exc:
                return str(exc)

    def open_gripper(self, numbers: list[float]) -> str:
        with self._mutex:
            try:
                print("Grasped object, will release it now.")
                self._set_gripper_targets(GRIPPER_MAX_WIDTH)
                self._simulate_for(1.0)
            except Exception as exc:
                return str(exc)
            return "Gripper opened successfully."

    def read_state(self) -> list[float]:
        with self._mutex:
            return self._state_vector_from_ot_ee(self._read_ot_ee())

    def read_joint_state(self) -> list[float]:
        with self._mutex:
            return self._current_arm_q().tolist()

    def read_gripper_state(self) -> list[float]:
        with self._mutex:
            fingers = self._current_gripper_q()
            return [float(fingers[0] + fingers[1])]

    def set_joint_pose(self, numbers: list[float]) -> list[float]:
        """Immediate joint targets for MolmoAct2 policy steps (no cubic trajectory)."""
        with self._mutex:
            if len(numbers) < 7:
                raise RuntimeError("setJointPose requires at least 7 joint angles (rad).")
            q_target = np.array(numbers[:7], dtype=np.float32)
            if not self._is_valid_joint_pose(q_target.astype(np.float64)):
                raise RuntimeError("Joint target outside Franka joint limits.")
            self._set_arm_targets(q_target.astype(np.float64))
            if len(numbers) >= 8:
                self._set_gripper_targets(float(numbers[7]))
            self._step()
            return self._current_arm_q().tolist()

    def get_joint_positions_q9(self) -> np.ndarray:
        """Return 9-DOF joint vector (7 arm + 2 finger joints) for policy observation."""
        with self._mutex:
            return np.concatenate([self._current_arm_q(), self._current_gripper_q()]).astype(np.float64)

    def apply_policy_joint_targets(self, q9: np.ndarray) -> None:
        """Apply absolute joint targets from MolmoAct2 (single sim step, no trajectory)."""
        q = np.asarray(q9, dtype=np.float64).reshape(9)
        cmd: list[float] = q[:7].tolist()
        cmd.append(float((q[7] + q[8]) / 2.0))
        self.set_joint_pose(cmd)
