# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""HTTP client for ``motion_server.cpp`` — used by the hardware MolmoAct2 runner."""

from __future__ import annotations

import json
import urllib.error
import urllib.request

import numpy as np


class MotionServerHttpRobot:
    """Drive a real Franka via the motion_server REST API (port 34568 by default)."""

    def __init__(self, base_url: str = "http://127.0.0.1:34568", timeout_s: float = 30.0) -> None:
        self._base_url = base_url.rstrip("/")
        self._timeout_s = timeout_s
        self._last_gripper_width = 0.04

    def _post(self, payload: dict[str, list[float]]) -> dict:
        body = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            self._base_url,
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self._timeout_s) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"motion_server HTTP {exc.code}: {detail}") from exc

    def get_joint_positions_q9(self) -> np.ndarray:
        payload = self._post({"readJointState": [], "readGripperState": []})
        q7 = payload.get("readJointState")
        if not q7 or len(q7) < 7:
            raise RuntimeError(f"readJointState failed: {payload}")
        grip = payload.get("readGripperState")
        if grip and len(grip) >= 1:
            width = float(grip[0])
            self._last_gripper_width = width
        else:
            width = self._last_gripper_width
        half = width / 2.0
        return np.array(list(q7[:7]) + [half, half], dtype=np.float64)

    def apply_policy_joint_targets(self, q9: np.ndarray) -> None:
        q = np.asarray(q9, dtype=np.float64).reshape(9)
        gripper = float((q[7] + q[8]) / 2.0)
        cmd: list[float] = q[:7].tolist()
        if abs(gripper - self._last_gripper_width) > 0.002:
            cmd.append(gripper)
        self._post({"setJointPose": cmd})
        self._last_gripper_width = gripper
