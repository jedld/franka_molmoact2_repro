# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""RGB camera capture for the Isaac Sim motion server web UI."""

from __future__ import annotations

import io
import threading
from dataclasses import dataclass

import numpy as np
import omni
from isaacsim.sensors.camera import Camera
from pxr import UsdGeom

CAMERA_WIDTH = 320
CAMERA_HEIGHT = 180

# MolmoAct2-DROID wire format (start_molmoact2_3090.md): (H, W, 3) uint8 RGB, C-contiguous.
MOLMOACT2_POLICY_SHAPE = (CAMERA_HEIGHT, CAMERA_WIDTH, 3)

# MolmoAct2 / DROID policy observation rate — only this path runs on the sim thread.
POLICY_CAPTURE_HZ = 15.0

# Web UI JPEGs are encoded lazily on HTTP worker threads (never on the sim thread).
PREVIEW_JPEG_QUALITY = 68


def _to_numpy_cpu(image) -> np.ndarray:
    """Copy annotator output (numpy, warp, torch) to a host ``np.ndarray``."""
    if isinstance(image, np.ndarray):
        return np.asarray(image)
    if hasattr(image, "detach") and hasattr(image, "cpu") and hasattr(image, "numpy"):
        return np.asarray(image.detach().cpu().numpy())
    if hasattr(image, "numpy") and callable(image.numpy):
        return np.asarray(image.numpy())
    try:
        import warp as wp

        if isinstance(image, wp.array):
            return wp.to_numpy(image)
    except ImportError:
        pass
    return np.asarray(image)


def prepare_molmoact2_image(
    image,
    *,
    expected_shape: tuple[int, int, int] | None = MOLMOACT2_POLICY_SHAPE,
) -> np.ndarray:
    """Normalize a frame to MolmoAct2-DROID wire format: C-contiguous ``(H, W, 3) uint8`` RGB.

    Matches ``start_molmoact2_3090.md`` and AllenAI ``host_server_droid.py`` (RGB, not BGR).
    """
    arr = _to_numpy_cpu(image)

    # Accept CHW (3, H, W) from mistaken callers; policy path uses HWC from Isaac Camera.
    if arr.ndim == 3 and arr.shape[0] == 3 and arr.shape[-1] != 3:
        arr = np.transpose(arr, (1, 2, 0))

    if arr.ndim != 3:
        raise ValueError(f"MolmoAct2 image must be HxWx3, got ndim={arr.ndim} shape={getattr(arr, 'shape', '?')}")

    if arr.shape[-1] == 4:
        arr = arr[:, :, :3]
    elif arr.shape[-1] != 3:
        raise ValueError(f"MolmoAct2 image requires 3 RGB channels, got shape={arr.shape}")

    if np.issubdtype(arr.dtype, np.floating):
        peak = float(np.nanmax(arr)) if arr.size else 0.0
        if peak <= 1.0:
            arr = arr * 255.0
        rgb = np.clip(arr, 0, 255).astype(np.uint8)
    else:
        rgb = np.clip(arr, 0, 255).astype(np.uint8)

    rgb = np.ascontiguousarray(rgb, dtype=np.uint8)

    if expected_shape is not None and rgb.shape != expected_shape:
        raise ValueError(
            f"MolmoAct2 image shape {rgb.shape} != expected {expected_shape} "
            f"(height={expected_shape[0]}, width={expected_shape[1]}, RGB)"
        )
    return rgb


def ensure_uint8_rgb(image: np.ndarray) -> np.ndarray:
    """Convert a camera frame to HxWx3 uint8 RGB (any resolution, for web preview)."""
    return prepare_molmoact2_image(image, expected_shape=None)


@dataclass(frozen=True)
class CameraSpec:
    camera_id: str
    label: str
    prim_path: str


def discover_stage_cameras() -> list[CameraSpec]:
    """Return MolmoAct2-style cameras on the stage (external_1, external_2, wrist)."""
    stage = omni.usd.get_context().get_stage()
    preferred = [
        ("external_1", "Exterior 1 (left, ZED 2)", "/World/Cameras/external_1"),
        ("external_2", "Exterior 2 (front, ZED 2)", "/World/Cameras/external_2"),
    ]
    specs: list[CameraSpec] = []
    for camera_id, label, path in preferred:
        if stage.GetPrimAtPath(path).IsValid():
            specs.append(CameraSpec(camera_id, label, path))

    for prim in stage.Traverse():
        if not prim.IsA(UsdGeom.Camera):
            continue
        path = str(prim.GetPath())
        if path.endswith("/wrist_camera"):
            specs.append(CameraSpec("wrist", "Wrist (RealSense D435)", path))
            break

    if specs:
        return specs

    for prim in stage.Traverse():
        if not prim.IsA(UsdGeom.Camera):
            continue
        path = str(prim.GetPath())
        if path.startswith("/OmniverseKit_"):
            continue
        camera_id = path.rstrip("/").split("/")[-1]
        label = camera_id.replace("_", " ").title()
        specs.append(CameraSpec(camera_id, label, path))
    return specs


def _encode_jpeg(rgb: np.ndarray, quality: int = PREVIEW_JPEG_QUALITY) -> bytes:
    rgb_u8 = np.ascontiguousarray(rgb[:, :, :3], dtype=np.uint8)
    try:
        import cv2

        bgr = cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2BGR)
        ok, buf = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
        if ok:
            return buf.tobytes()
    except ImportError:
        pass

    from PIL import Image

    bio = io.BytesIO()
    Image.fromarray(rgb_u8).save(bio, format="JPEG", quality=quality)
    return bio.getvalue()


class CameraStreamManager:
    """Policy RGB on the sim thread @ 15 Hz; web preview JPEGs encoded lazily on HTTP threads."""

    def __init__(
        self,
        specs: list[CameraSpec],
        resolution: tuple[int, int] = (CAMERA_WIDTH, CAMERA_HEIGHT),
        *,
        policy_hz: float = POLICY_CAPTURE_HZ,
    ) -> None:
        self._specs = specs
        self._resolution = resolution
        self._policy_hz = policy_hz
        self._policy_accum = 0.0
        self._cameras: dict[str, Camera] = {}
        self._jpeg_cache: dict[str, bytes] = {}
        self._rgb_cache: dict[str, np.ndarray] = {}
        self._rgb_seq: dict[str, int] = {}
        self._jpeg_seq: dict[str, int] = {}
        self._lock = threading.Lock()
        self._encode_locks: dict[str, threading.Lock] = {s.camera_id: threading.Lock() for s in specs}
        self._placeholder = _encode_jpeg(np.zeros((resolution[1], resolution[0], 3), dtype=np.uint8))

    @property
    def specs(self) -> list[CameraSpec]:
        return list(self._specs)

    def initialize(self) -> None:
        for spec in self._specs:
            if spec.camera_id in self._cameras:
                continue
            self._cameras[spec.camera_id] = Camera(
                prim_path=spec.prim_path,
                name=spec.camera_id,
                resolution=self._resolution,
                annotator_device="cpu",
            )
        for camera in self._cameras.values():
            camera.initialize()

    def tick_policy(self, sim_dt: float) -> None:
        """Capture RGB for MolmoAct2 at ``policy_hz``; JPEG encoding is not performed here."""
        if self._policy_hz <= 0.0:
            return
        self._policy_accum += sim_dt
        period = 1.0 / self._policy_hz
        if self._policy_accum < period:
            return
        self._policy_accum -= period
        self._capture_rgb_all()

    def capture(self) -> None:
        """Legacy alias: capture RGB immediately (prefer ``tick_policy`` in the main loop)."""
        self._capture_rgb_all()

    def _capture_rgb_all(self) -> None:
        for spec in self._specs:
            camera = self._cameras.get(spec.camera_id)
            if camera is None:
                continue
            try:
                frame = camera.get_rgb()
                if frame is None:
                    frame = camera.get_rgba()
                if frame is None:
                    continue
                rgb = prepare_molmoact2_image(frame)
            except Exception:
                continue
            with self._lock:
                self._rgb_cache[spec.camera_id] = rgb
                self._rgb_seq[spec.camera_id] = self._rgb_seq.get(spec.camera_id, 0) + 1

    def get_jpeg(self, camera_id: str) -> bytes:
        """Return cached JPEG, encoding on this (HTTP) thread only if RGB has updated."""
        with self._lock:
            rgb = self._rgb_cache.get(camera_id)
            seq = self._rgb_seq.get(camera_id, 0)
            if rgb is None:
                return self._placeholder
            if self._jpeg_seq.get(camera_id) == seq and camera_id in self._jpeg_cache:
                return self._jpeg_cache[camera_id]
            rgb_copy = rgb.copy()

        encode_lock = self._encode_locks.get(camera_id)
        if encode_lock is None:
            return _encode_jpeg(rgb_copy)
        with encode_lock:
            with self._lock:
                if self._jpeg_seq.get(camera_id) == seq and camera_id in self._jpeg_cache:
                    return self._jpeg_cache[camera_id]
            jpeg = _encode_jpeg(rgb_copy)
            with self._lock:
                if self._rgb_seq.get(camera_id) == seq:
                    self._jpeg_cache[camera_id] = jpeg
                    self._jpeg_seq[camera_id] = seq
                    return jpeg
                return self._jpeg_cache.get(camera_id, jpeg)

    def get_rgb(self, camera_id: str) -> np.ndarray | None:
        with self._lock:
            rgb = self._rgb_cache.get(camera_id)
            return None if rgb is None else rgb.copy()

    def observations_ready(
        self,
        external_slot: str = "external_1",
        external_camera_mode: str = "single",
    ) -> bool:
        with self._lock:
            if "wrist" not in self._rgb_cache:
                return False
            if external_camera_mode == "dual":
                return "external_1" in self._rgb_cache and "external_2" in self._rgb_cache
            return external_slot in self._rgb_cache

    def list_cameras(self) -> list[dict[str, str]]:
        return [{"id": s.camera_id, "label": s.label, "path": s.prim_path} for s in self._specs]
