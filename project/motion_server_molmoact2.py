# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""MolmoAct2-DROID inference loop for the Isaac Sim motion server.

Mirrors the control pattern in molmoact2_droid_remote_client.py: capture cameras +
joint state at 15 Hz, POST to the MolmoAct2 HTTP API (see start_molmoact2_3090.md),
apply absolute joint targets from the returned action chunk.

Endpoints:
  - single exterior → POST /act on port 8012
  - dual / duplicate exterior → POST /act_dual on port 8101 (MOLMOACT2_DUAL_EXTERIOR=1)
"""

from __future__ import annotations

import base64
import json
import threading
import time
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np
from numpy.lib.format import descr_to_dtype, dtype_to_descr

from motion_server_cameras import MOLMOACT2_POLICY_SHAPE, prepare_molmoact2_image

DEFAULT_MOLMOACT2_URL = "http://192.168.0.233:8012"


class PolicyRobotBackend(Protocol):
    def get_joint_positions_q9(self) -> np.ndarray: ...

    def apply_policy_joint_targets(self, q9: np.ndarray) -> None: ...


class PolicyCameraBackend(Protocol):
    def observations_ready(self, external_slot: str = "external_1", external_camera_mode: str = "single") -> bool: ...

    def get_rgb(self, camera_id: str) -> np.ndarray | None: ...


DEFAULT_MOLMOACT2_DUAL_URL = "http://192.168.0.233:8101"
INFERENCE_TIMEOUT_S = 120.0
CONTROL_HZ = 15.0
EXPECTED_ACTION_SHAPE = (15, 8)

# MolmoAct2-DROID training / HF model card camera order:
#   [exterior_1, exterior_2, wrist]
# HTTP API field mapping (start_molmoact2_3090.md):
#   external_cam     ← sim external_1 (left ZED 2)
#   external_cam_2   ← sim external_2 (front ZED 2)
#   wrist_cam        ← sim wrist (RealSense D435)
# Single /act uses one exterior in external_cam + wrist_cam (2-view path).
# Duplicate /act_dual sends [ext, ext, wrist] per HF workaround.

TASK_INSTRUCTIONS: dict[str, str] = {
    "apple_on_plate": "Pick up the apple and place it on the plate",
    "pipette_in_tray": "Place the pipette in the tray",
    "red_cube_in_tape_roll": "Place the red cube in the center of the tape roll",
    "knife_in_box": "Put the knife in the box",
    "objects_in_bowl": "Move the objects into the bowl",
}


def state8_from_joints(q9: np.ndarray) -> np.ndarray:
    """Collapse 9-DOF Franka state to MolmoAct2 8-D observation."""
    gripper = 0.5 * (float(q9[7]) + float(q9[8]))
    return np.concatenate([q9[:7], [gripper]]).astype(np.float32)


def actions8_to_q9(action8: np.ndarray, current_q9: np.ndarray) -> np.ndarray:
    """Expand 8-D MolmoAct2 action to 9-DOF joint command."""
    q = current_q9.copy()
    q[:7] = action8[:7]
    g = float(action8[7])
    q[7] = g
    q[8] = g
    return q


def _numpy_json_default(obj: Any) -> Any:
    """json-numpy compatible encoder (Crimson-Crow/json-numpy format)."""
    if isinstance(obj, np.ndarray):
        arr = np.ascontiguousarray(obj)
        return {
            "__numpy__": base64.b64encode(arr.tobytes()).decode("ascii"),
            "dtype": dtype_to_descr(arr.dtype),
            "shape": list(arr.shape),
        }
    if isinstance(obj, np.generic):
        return obj.item()
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def _numpy_json_object_hook(dct: dict[str, Any]) -> Any:
    if "__numpy__" in dct:
        buf = base64.b64decode(dct["__numpy__"])
        arr = np.frombuffer(buf, dtype=descr_to_dtype(dct["dtype"]))
        shape = dct.get("shape")
        return arr.reshape(shape) if shape else arr[0]
    return dct


def _json_dumps(payload: dict[str, Any]) -> bytes:
    try:
        import json_numpy

        json_numpy.patch()
        return json_numpy.dumps(payload).encode("utf-8")
    except ImportError:
        return json.dumps(payload, default=_numpy_json_default).encode("utf-8")


def _json_loads(data: bytes) -> dict[str, Any]:
    try:
        import json_numpy

        json_numpy.patch()
        return json_numpy.loads(data)
    except ImportError:
        return json.loads(data.decode("utf-8"), object_hook=_numpy_json_object_hook)


def prepare_molmoact2_state(state8: np.ndarray) -> np.ndarray:
    """Normalize robot state to ``(8,) float32`` C-contiguous per API spec."""
    return np.ascontiguousarray(np.asarray(state8, dtype=np.float32).reshape(8))


def _prepare_payload_image(image: np.ndarray) -> np.ndarray:
    """Policy image: strict ``(180, 320, 3) uint8`` RGB layout for json_numpy wire encoding."""
    return prepare_molmoact2_image(image, expected_shape=MOLMOACT2_POLICY_SHAPE)


def build_molmoact2_request(
    external_camera_mode: str,
    ext_rgb: np.ndarray,
    wrist_rgb: np.ndarray,
    state8: np.ndarray,
    instruction: str,
    ext2_rgb: np.ndarray | None = None,
) -> tuple[str, dict[str, Any]]:
    """Build endpoint + payload per start_molmoact2_3090.md / molmoact2_server_droid.py.

    Camera slot order must match MolmoAct2-DROID training:
      dual:       external_cam=external_1, external_cam_2=external_2, wrist_cam=wrist
      duplicate:  external_cam=external_cam_2=chosen exterior, wrist_cam=wrist
      single:     external_cam=chosen exterior, wrist_cam=wrist

    - ``single`` → ``POST /act`` with ``external_cam`` + ``wrist_cam`` (port 8012)
    - ``dual`` / ``duplicate`` → ``POST /act_dual`` with three image fields (port 8101)

    Image layout (all camera fields): ``(H, W, 3)`` = ``(180, 320, 3)`` ``uint8`` RGB, C-contiguous.
    """
    state8 = prepare_molmoact2_state(state8)
    wrist_rgb = _prepare_payload_image(wrist_rgb)

    if external_camera_mode == "dual":
        if ext2_rgb is None:
            raise ValueError("dual mode requires external_cam and external_cam_2")
        payload = {
            "external_cam": _prepare_payload_image(ext_rgb),
            "external_cam_2": _prepare_payload_image(ext2_rgb),
            "wrist_cam": wrist_rgb,
            "instruction": instruction,
            "state": state8,
        }
        return "/act_dual", payload

    if external_camera_mode == "duplicate":
        ext = _prepare_payload_image(ext_rgb)
        payload = {
            "external_cam": ext,
            "external_cam_2": np.ascontiguousarray(ext),
            "wrist_cam": wrist_rgb,
            "instruction": instruction,
            "state": state8,
        }
        return "/act_dual", payload

    payload = {
        "external_cam": _prepare_payload_image(ext_rgb),
        "wrist_cam": wrist_rgb,
        "instruction": instruction,
        "state": state8,
    }
    return "/act", payload


def query_molmoact2(
    url: str,
    endpoint: str,
    payload: dict[str, Any],
    timeout: float = INFERENCE_TIMEOUT_S,
) -> tuple[np.ndarray, float | None]:
    """POST json_numpy payload to MolmoAct2; returns ((N, 8) actions, dt_ms)."""
    import urllib.error
    import urllib.request

    body = _json_dumps(payload)
    req = urllib.request.Request(
        f"{url.rstrip('/')}{endpoint}",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            out = _json_loads(resp.read())
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"MolmoAct2 {endpoint} HTTP {exc.code}: {detail}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"MolmoAct2 {endpoint} unreachable: {exc.reason}") from exc

    actions = np.asarray(out["actions"], dtype=np.float32)
    if actions.ndim == 1:
        actions = actions[None, :]
    dt_ms = out.get("dt_ms")
    return actions, None if dt_ms is None else float(dt_ms)


def query_molmoact2_observation(
    url: str,
    external_camera_mode: str,
    ext_rgb: np.ndarray,
    wrist_rgb: np.ndarray,
    state8: np.ndarray,
    instruction: str,
    ext2_rgb: np.ndarray | None = None,
    timeout: float = INFERENCE_TIMEOUT_S,
) -> tuple[np.ndarray, float | None, str]:
    """Build request from observation tensors and POST to the correct inference endpoint."""
    endpoint, payload = build_molmoact2_request(
        external_camera_mode, ext_rgb, wrist_rgb, state8, instruction, ext2_rgb=ext2_rgb
    )
    actions, dt_ms = query_molmoact2(url, endpoint, payload, timeout=timeout)
    return actions, dt_ms, endpoint


def check_molmoact2_health(url: str, timeout: float = 5.0) -> dict[str, Any]:
    """GET /healthz — expect ``{"status": "ok"}`` per start_molmoact2_3090.md."""
    import urllib.error
    import urllib.request

    health_url = f"{url.rstrip('/')}/healthz"
    try:
        with urllib.request.urlopen(health_url, timeout=timeout) as resp:
            body = resp.read().decode("utf-8", errors="replace")
            ok = False
            if resp.status == 200:
                try:
                    ok = json.loads(body).get("status") == "ok"
                except json.JSONDecodeError:
                    ok = False
            return {"ok": ok, "status": resp.status, "body": body}
    except urllib.error.HTTPError as exc:
        return {"ok": False, "status": exc.code, "body": exc.read().decode("utf-8", errors="replace")}
    except Exception as exc:
        return {"ok": False, "status": 0, "body": str(exc)}


@dataclass
class MolmoAct2Status:
    running: bool = False
    server_url: str = DEFAULT_MOLMOACT2_URL
    instruction: str = ""
    external_slot: str = "external_1"
    external_camera_mode: str = "single"
    control_hz: float = CONTROL_HZ
    action_idx: int = 0
    chunk_size: int = 0
    chunks_fetched: int = 0
    steps_applied: int = 0
    fetching: bool = False
    last_error: str = ""
    last_endpoint: str = ""
    last_dt_ms: float | None = None
    last_fetch_s: float | None = None
    inference_healthy: bool | None = None
    inference_health_body: str = ""


class MolmoAct2Controller:
    """Runs MolmoAct2 policy inference on the Isaac Sim main loop thread."""

    def __init__(
        self,
        handler: PolicyRobotBackend,
        camera_manager: PolicyCameraBackend,
    ) -> None:
        self._handler = handler
        self._cameras = camera_manager
        self._lock = threading.Lock()
        self._status = MolmoAct2Status()
        self._action_chunk: np.ndarray | None = None
        self._action_idx = 0
        self._control_accum = 0.0
        self._fetch_thread: threading.Thread | None = None
        self._fetch_error: str | None = None
        self._pending_chunk: np.ndarray | None = None
        self._pending_dt_ms: float | None = None
        self._health_thread: threading.Thread | None = None

    @property
    def is_running(self) -> bool:
        with self._lock:
            return self._status.running

    def configure(
        self,
        *,
        server_url: str | None = None,
        instruction: str | None = None,
        external_slot: str | None = None,
        external_camera_mode: str | None = None,
    ) -> MolmoAct2Status:
        with self._lock:
            if server_url is not None:
                self._status.server_url = server_url.rstrip("/")
            if instruction is not None:
                self._status.instruction = instruction
            if external_slot is not None:
                if external_slot not in ("external_1", "external_2"):
                    raise ValueError("external_slot must be external_1 or external_2")
                self._status.external_slot = external_slot
            if external_camera_mode is not None:
                if external_camera_mode not in ("single", "dual", "duplicate"):
                    raise ValueError("external_camera_mode must be single, dual, or duplicate")
                self._status.external_camera_mode = external_camera_mode
            return MolmoAct2Status(**vars(self._status))

    def start(self) -> MolmoAct2Status:
        with self._lock:
            if not self._status.instruction.strip():
                raise RuntimeError("Instruction is empty.")
            self._status.running = True
            self._status.last_error = ""
            self._action_chunk = None
            self._action_idx = 0
            self._control_accum = 0.0
            self._fetch_error = None
            self._pending_chunk = None
        self.refresh_health(async_ok=True)
        return self.get_status()

    def stop(self) -> MolmoAct2Status:
        with self._lock:
            self._status.running = False
            self._status.fetching = False
        if self._fetch_thread is not None and self._fetch_thread.is_alive():
            self._fetch_thread.join(timeout=0.1)
        return self.get_status()

    def refresh_health(self, async_ok: bool = True) -> dict[str, Any]:
        url = self.get_status().server_url

        def _worker() -> None:
            result = check_molmoact2_health(url)
            with self._lock:
                self._status.inference_healthy = result["ok"]
                self._status.inference_health_body = result.get("body", "")

        if async_ok:
            self._health_thread = threading.Thread(target=_worker, daemon=True)
            self._health_thread.start()
            return {"started": True}
        result = check_molmoact2_health(url)
        with self._lock:
            self._status.inference_healthy = result["ok"]
            self._status.inference_health_body = result.get("body", "")
        return result

    def get_status(self) -> MolmoAct2Status:
        with self._lock:
            status = MolmoAct2Status(**vars(self._status))
            status.action_idx = self._action_idx
            status.chunk_size = 0 if self._action_chunk is None else int(self._action_chunk.shape[0])
            status.fetching = self._fetch_thread is not None and self._fetch_thread.is_alive()
            if self._fetch_error:
                status.last_error = self._fetch_error
            return status

    def tick(self, sim_dt: float) -> None:
        with self._lock:
            if not self._status.running:
                return
            external_slot = self._status.external_slot
            external_camera_mode = self._status.external_camera_mode
            instruction = self._status.instruction
            server_url = self._status.server_url

        if not self._cameras.observations_ready(external_slot, external_camera_mode):
            return

        self._control_accum += sim_dt
        period = 1.0 / CONTROL_HZ
        if self._control_accum < period:
            return
        self._control_accum -= period

        if self._fetch_error:
            with self._lock:
                self._status.running = False
                self._status.last_error = self._fetch_error
            return

        self._consume_pending_chunk()

        ext_rgb, ext2_rgb = self._resolve_external_images(external_slot, external_camera_mode)
        wrist_rgb = self._cameras.get_rgb("wrist")
        if ext_rgb is None or wrist_rgb is None:
            return
        if external_camera_mode == "dual" and ext2_rgb is None:
            return

        q9 = self._handler.get_joint_positions_q9()
        state8 = state8_from_joints(q9)

        if self._action_chunk is None or self._action_idx >= len(self._action_chunk):
            self._maybe_start_fetch(
                ext_rgb, wrist_rgb, state8, instruction, server_url, ext2_rgb=ext2_rgb
            )
            return

        target_q9 = actions8_to_q9(self._action_chunk[self._action_idx], q9)
        self._handler.apply_policy_joint_targets(target_q9)
        self._action_idx += 1
        with self._lock:
            self._status.steps_applied += 1

    def _resolve_external_images(
        self, external_slot: str, external_camera_mode: str
    ) -> tuple[np.ndarray | None, np.ndarray | None]:
        """Map sim camera IDs to MolmoAct2 exterior slots (see module-level camera order)."""
        if external_camera_mode == "dual":
            return self._cameras.get_rgb("external_1"), self._cameras.get_rgb("external_2")
        primary = self._cameras.get_rgb(external_slot)
        if external_camera_mode == "duplicate" and primary is not None:
            return primary, primary.copy()
        return primary, None

    def _consume_pending_chunk(self) -> None:
        with self._lock:
            if self._pending_chunk is not None:
                self._action_chunk = self._pending_chunk
                self._pending_chunk = None
                self._action_idx = 0
                self._status.chunks_fetched += 1
                if self._pending_dt_ms is not None:
                    self._status.last_dt_ms = self._pending_dt_ms
                    self._pending_dt_ms = None

    def _maybe_start_fetch(
        self,
        ext_rgb: np.ndarray,
        wrist_rgb: np.ndarray,
        state8: np.ndarray,
        instruction: str,
        server_url: str,
        ext2_rgb: np.ndarray | None = None,
    ) -> None:
        if self._fetch_thread is not None and self._fetch_thread.is_alive():
            return

        ext_copy = ext_rgb.copy()
        ext2_copy = None if ext2_rgb is None else ext2_rgb.copy()
        wrist_copy = wrist_rgb.copy()
        state_copy = state8.copy()
        camera_mode = self.get_status().external_camera_mode

        def _worker() -> None:
            t0 = time.perf_counter()
            try:
                chunk, dt_ms, endpoint = query_molmoact2_observation(
                    server_url,
                    camera_mode,
                    ext_copy,
                    wrist_copy,
                    state_copy,
                    instruction,
                    ext2_rgb=ext2_copy,
                )
                elapsed = time.perf_counter() - t0
                with self._lock:
                    self._pending_chunk = chunk
                    self._pending_dt_ms = dt_ms
                    self._status.last_fetch_s = elapsed
                    self._status.last_endpoint = endpoint
                    self._fetch_error = None
            except Exception as exc:
                self._fetch_error = str(exc)

        with self._lock:
            self._status.fetching = True
        self._fetch_thread = threading.Thread(target=_worker, daemon=True)
        self._fetch_thread.start()
