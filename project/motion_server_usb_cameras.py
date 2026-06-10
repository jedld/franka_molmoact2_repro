# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""USB camera capture for MolmoAct2-DROID (RealSense wrist + V4L2 exterior cameras).

Frames are captured on a background thread, resized to 320×180 RGB for policy input,
and exposed through the same interface as :class:`CameraStreamManager` (sim cameras).
"""

from __future__ import annotations

import json
import os
import re
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from motion_server_cameras import (
    CAMERA_HEIGHT,
    CAMERA_WIDTH,
    POLICY_CAPTURE_HZ,
    PREVIEW_JPEG_QUALITY,
    CameraSpec,
    _encode_jpeg,
    prepare_molmoact2_image,
)

_PROJECT_DIR = Path(__file__).resolve().parent
_DEFAULT_CONFIG_PATH = _PROJECT_DIR / "usb_cameras.json"


@dataclass
class UsbCameraDeviceConfig:
    """One camera slot: device selector and native capture resolution."""

    device: str = ""
    width: int = 640
    height: int = 480
    fps: int = 30


@dataclass
class UsbCameraConfig:
    """DROID-style USB layout: two exterior UVC cameras + one wrist RealSense."""

    wrist: UsbCameraDeviceConfig = field(
        default_factory=lambda: UsbCameraDeviceConfig(device="realsense", width=640, height=480, fps=30)
    )
    external_1: UsbCameraDeviceConfig = field(
        default_factory=lambda: UsbCameraDeviceConfig(device="0", width=1280, height=720, fps=15)
    )
    external_2: UsbCameraDeviceConfig = field(
        default_factory=lambda: UsbCameraDeviceConfig(device="1", width=1280, height=720, fps=15)
    )
    policy_shape: tuple[int, int, int] = (CAMERA_HEIGHT, CAMERA_WIDTH, 3)

    @staticmethod
    def from_dict(data: dict[str, Any]) -> UsbCameraConfig:
        def _slot(name: str, defaults: UsbCameraDeviceConfig) -> UsbCameraDeviceConfig:
            raw = data.get(name, {})
            if isinstance(raw, str):
                return UsbCameraDeviceConfig(device=raw, width=defaults.width, height=defaults.height, fps=defaults.fps)
            if not isinstance(raw, dict):
                return defaults
            return UsbCameraDeviceConfig(
                device=str(raw.get("device", defaults.device)),
                width=int(raw.get("width", defaults.width)),
                height=int(raw.get("height", defaults.height)),
                fps=int(raw.get("fps", defaults.fps)),
            )

        cfg = UsbCameraConfig(
            wrist=_slot("wrist", UsbCameraDeviceConfig("realsense", 640, 480, 30)),
            external_1=_slot("external_1", UsbCameraDeviceConfig("0", 1280, 720, 15)),
            external_2=_slot("external_2", UsbCameraDeviceConfig("1", 1280, 720, 15)),
        )
        shape = data.get("policy_shape")
        if isinstance(shape, list) and len(shape) == 3:
            cfg.policy_shape = (int(shape[0]), int(shape[1]), int(shape[2]))
        return cfg


def load_usb_camera_config(path: str | Path | None = None) -> UsbCameraConfig:
    """Load config from ``MOLMO_USB_CONFIG``, explicit path, or ``project/usb_cameras.json``."""
    config_path = path or os.environ.get("MOLMO_USB_CONFIG")
    if config_path:
        p = Path(config_path)
        if p.is_file():
            return UsbCameraConfig.from_dict(json.loads(p.read_text(encoding="utf-8")))

    if _DEFAULT_CONFIG_PATH.is_file():
        return UsbCameraConfig.from_dict(json.loads(_DEFAULT_CONFIG_PATH.read_text(encoding="utf-8")))

    cfg = UsbCameraConfig()
    if os.environ.get("MOLMO_USB_WRIST"):
        cfg.wrist.device = os.environ["MOLMO_USB_WRIST"]
    if os.environ.get("MOLMO_USB_EXTERNAL_1"):
        cfg.external_1.device = os.environ["MOLMO_USB_EXTERNAL_1"]
    if os.environ.get("MOLMO_USB_EXTERNAL_2"):
        cfg.external_2.device = os.environ["MOLMO_USB_EXTERNAL_2"]
    for slot_name, slot in (
        ("MOLMO_USB_WRIST_WIDTH", cfg.wrist),
        ("MOLMO_USB_WRIST_HEIGHT", cfg.wrist),
        ("MOLMO_USB_EXT1_WIDTH", cfg.external_1),
        ("MOLMO_USB_EXT1_HEIGHT", cfg.external_1),
        ("MOLMO_USB_EXT2_WIDTH", cfg.external_2),
        ("MOLMO_USB_EXT2_HEIGHT", cfg.external_2),
    ):
        env = os.environ.get(slot_name)
        if env:
            if slot_name.endswith("_WIDTH"):
                slot.width = int(env)
            else:
                slot.height = int(env)
    return cfg


class _CameraBackend:
    def open(self) -> None:
        raise NotImplementedError

    def close(self) -> None:
        raise NotImplementedError

    def read_rgb(self) -> np.ndarray | None:
        raise NotImplementedError


class _OpenCvBackend(_CameraBackend):
    def __init__(self, device: str, width: int, height: int, fps: int) -> None:
        self._device = device
        self._width = width
        self._height = height
        self._fps = fps
        self._cap = None

    def open(self) -> None:
        import cv2

        source: str | int
        if re.fullmatch(r"\d+", self._device):
            source = int(self._device)
        else:
            source = self._device
        cap = cv2.VideoCapture(source, cv2.CAP_V4L2)
        if not cap.isOpened():
            cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open USB camera: {self._device}")
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, float(self._width))
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, float(self._height))
        cap.set(cv2.CAP_PROP_FPS, float(self._fps))
        self._cap = cap

    def close(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def read_rgb(self) -> np.ndarray | None:
        import cv2

        if self._cap is None:
            return None
        ok, frame = self._cap.read()
        if not ok or frame is None:
            return None
        if frame.ndim == 2:
            return cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


class _RealSenseBackend(_CameraBackend):
    def __init__(self, device: str, width: int, height: int, fps: int) -> None:
        self._device = device
        self._width = width
        self._height = height
        self._fps = fps
        self._pipeline = None

    def open(self) -> None:
        try:
            import pyrealsense2 as rs
        except ImportError as exc:
            raise RuntimeError(
                "pyrealsense2 is required for wrist camera device='realsense'. "
                "Install: pip install pyrealsense2"
            ) from exc

        pipeline = rs.pipeline()
        config = rs.config()
        serial: str | None = None
        if self._device.startswith("realsense:"):
            serial = self._device.split(":", 1)[1]
            config.enable_device(serial)
        config.enable_stream(rs.stream.color, self._width, self._height, rs.format.rgb8, self._fps)
        pipeline.start(config)
        self._pipeline = pipeline
        label = serial or "default"
        print(f"RealSense color stream started ({label}) @ {self._width}×{self._height} {self._fps} fps")

    def close(self) -> None:
        if self._pipeline is not None:
            self._pipeline.stop()
            self._pipeline = None

    def read_rgb(self) -> np.ndarray | None:
        if self._pipeline is None:
            return None
        frames = self._pipeline.wait_for_frames(timeout_ms=500)
        color = frames.get_color_frame()
        if not color:
            return None
        return np.asanyarray(color.get_data())


def _create_backend(device: str, width: int, height: int, fps: int) -> _CameraBackend:
    if device.lower().startswith("realsense"):
        return _RealSenseBackend(device, width, height, fps)
    return _OpenCvBackend(device, width, height, fps)


class UsbCameraStreamManager:
    """Policy RGB @ 15 Hz from USB devices; JPEG preview uses the same cache as sim cameras."""

    def __init__(
        self,
        config: UsbCameraConfig | None = None,
        *,
        policy_hz: float = POLICY_CAPTURE_HZ,
    ) -> None:
        self._config = config or load_usb_camera_config()
        self._policy_hz = policy_hz
        self._policy_shape = self._config.policy_shape
        self._specs = [
            CameraSpec("external_1", "Exterior 1 (USB)", "usb://external_1"),
            CameraSpec("external_2", "Exterior 2 (USB)", "usb://external_2"),
            CameraSpec("wrist", "Wrist (USB RealSense / V4L2)", "usb://wrist"),
        ]
        self._backends: dict[str, _CameraBackend] = {}
        self._jpeg_cache: dict[str, bytes] = {}
        self._rgb_cache: dict[str, np.ndarray] = {}
        self._rgb_seq: dict[str, int] = {}
        self._jpeg_seq: dict[str, int] = {}
        self._lock = threading.Lock()
        self._encode_locks = {s.camera_id: threading.Lock() for s in self._specs}
        self._placeholder = _encode_jpeg(np.zeros((CAMERA_HEIGHT, CAMERA_WIDTH, 3), dtype=np.uint8))
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._running = False

    @property
    def specs(self) -> list[CameraSpec]:
        return list(self._specs)

    def initialize(self) -> None:
        slots = {
            "wrist": self._config.wrist,
            "external_1": self._config.external_1,
            "external_2": self._config.external_2,
        }
        for camera_id, dev in slots.items():
            if not dev.device:
                print(f"Warning: USB camera '{camera_id}' has empty device — skipping.")
                continue
            backend = _create_backend(dev.device, dev.width, dev.height, dev.fps)
            backend.open()
            self._backends[camera_id] = backend
            print(f"USB camera {camera_id}: device={dev.device} capture={dev.width}×{dev.height}")

        if not self._backends:
            raise RuntimeError("No USB cameras opened. Set MOLMO_USB_CONFIG or project/usb_cameras.json.")

        self._stop.clear()
        self._running = True
        self._thread = threading.Thread(target=self._capture_loop, name="usb-camera-capture", daemon=True)
        self._thread.start()
        deadline = time.monotonic() + 3.0
        while time.monotonic() < deadline:
            if self._rgb_cache:
                break
            time.sleep(0.05)

    def shutdown(self) -> None:
        self._stop.set()
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        for backend in self._backends.values():
            backend.close()
        self._backends.clear()

    def tick_policy(self, sim_dt: float) -> None:
        """No-op — USB capture runs on its own thread (hardware / non-sim hosts)."""

    def capture(self) -> None:
        self._capture_once()

    def _capture_loop(self) -> None:
        period = 1.0 / self._policy_hz if self._policy_hz > 0 else 0.0
        while not self._stop.is_set():
            t0 = time.monotonic()
            self._capture_once()
            if period > 0:
                time.sleep(max(0.0, period - (time.monotonic() - t0)))

    def _capture_once(self) -> None:
        for camera_id, backend in self._backends.items():
            try:
                rgb = backend.read_rgb()
                if rgb is None:
                    continue
                h, w = self._policy_shape[0], self._policy_shape[1]
                if rgb.shape[0] != h or rgb.shape[1] != w:
                    import cv2

                    rgb = cv2.resize(rgb, (w, h), interpolation=cv2.INTER_AREA)
                rgb = prepare_molmoact2_image(rgb, expected_shape=self._policy_shape)
            except Exception as exc:
                print(f"USB camera {camera_id} capture error: {exc}")
                continue
            with self._lock:
                self._rgb_cache[camera_id] = rgb
                self._rgb_seq[camera_id] = self._rgb_seq.get(camera_id, 0) + 1

    def get_jpeg(self, camera_id: str) -> bytes:
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
            return _encode_jpeg(rgb_copy, PREVIEW_JPEG_QUALITY)
        with encode_lock:
            with self._lock:
                if self._jpeg_seq.get(camera_id) == seq and camera_id in self._jpeg_cache:
                    return self._jpeg_cache[camera_id]
            jpeg = _encode_jpeg(rgb_copy, PREVIEW_JPEG_QUALITY)
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
