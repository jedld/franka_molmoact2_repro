# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""HTTP REST API compatible with project/motion_server.cpp, plus web UI routes."""

from __future__ import annotations

import json
import mimetypes
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

from motion_server_handler import IsaacMotionServerHandler
from motion_server_molmoact2 import MolmoAct2Controller, TASK_INSTRUCTIONS

_WEB_UI_DIR = Path(__file__).resolve().parent / "web_ui"
_CLIENT_GONE_ERRORS = (BrokenPipeError, ConnectionAbortedError, ConnectionResetError, OSError)


def _safe_write(handler: BaseHTTPRequestHandler, data: bytes) -> bool:
    """Write response body; return False if the browser closed the connection."""
    try:
        handler.wfile.write(data)
        return True
    except _CLIENT_GONE_ERRORS:
        return False


class _QuietThreadingHTTPServer(ThreadingHTTPServer):
    """HTTP server that does not log tracebacks when clients drop connections."""

    def process_request_thread(self, request, client_address) -> None:
        try:
            self.finish_request(request, client_address)
        except _CLIENT_GONE_ERRORS:
            pass
        except Exception:
            self.handle_error(request, client_address)
        finally:
            self.shutdown_request(request)


class _MotionRequest:
    __slots__ = ("commands", "response", "error", "event")

    def __init__(self, commands: dict[str, list[float]]) -> None:
        self.commands = commands
        self.response: dict[str, Any] | None = None
        self.error: Exception | None = None
        self.event = threading.Event()


class MotionRestServer:
    """Thread-safe bridge between HTTP clients and the Isaac Sim main loop."""

    def __init__(
        self,
        handler: IsaacMotionServerHandler,
        host: str = "0.0.0.0",
        port: int = 34568,
        camera_manager: Any | None = None,
        enable_ui: bool = True,
        molmoact2: MolmoAct2Controller | None = None,
    ) -> None:
        self._handler = handler
        self._host = host
        self._port = port
        self._camera_manager = camera_manager
        self._enable_ui = enable_ui
        self._molmoact2 = molmoact2
        self._queue_lock = threading.Lock()
        self._pending: _MotionRequest | None = None
        self._httpd: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        server = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, format: str, *args: Any) -> None:
                return

            def handle(self) -> None:
                try:
                    super().handle()
                except _CLIENT_GONE_ERRORS:
                    pass

            def do_GET(self) -> None:
                if not server._enable_ui:
                    self.send_error(404)
                    return
                parsed = urlparse(self.path)
                path = unquote(parsed.path)

                if path == "/api/molmoact2/status":
                    self._send_json(200, server._molmoact2_status_dict())
                    return

                if path == "/api/molmoact2/health":
                    if server._molmoact2 is None:
                        self._send_json(503, {"ok": False, "error": "MolmoAct2 controller not configured."})
                        return
                    self._send_json(200, server._molmoact2.refresh_health(async_ok=False))
                    return

                if path == "/api/molmoact2/tasks":
                    self._send_json(200, TASK_INSTRUCTIONS)
                    return

                if path in ("/", "/index.html"):
                    server._serve_file(self, _WEB_UI_DIR / "index.html")
                    return

                if path == "/api/cameras":
                    if server._camera_manager is None:
                        self._send_json(200, [])
                        return
                    self._send_json(200, server._camera_manager.list_cameras())
                    return

                if path.startswith("/api/camera/"):
                    camera_id = path.rsplit("/", 1)[-1]
                    if camera_id.endswith(".jpg"):
                        camera_id = camera_id[:-4]
                    if server._camera_manager is None:
                        self.send_error(404)
                        return
                    jpeg = server._camera_manager.get_jpeg(camera_id)
                    self.send_response(200)
                    self.send_header("Content-Type", "image/jpeg")
                    self.send_header("Cache-Control", "no-store")
                    self.send_header("Content-Length", str(len(jpeg)))
                    self.end_headers()
                    _safe_write(self, jpeg)
                    return

                static_path = _WEB_UI_DIR / path.lstrip("/")
                if static_path.is_file() and static_path.resolve().is_relative_to(_WEB_UI_DIR.resolve()):
                    server._serve_file(self, static_path)
                    return

                self.send_error(404)

            def do_POST(self) -> None:
                parsed = urlparse(self.path)
                path = unquote(parsed.path)

                if path == "/api/molmoact2/start":
                    server._handle_molmoact2_start(self)
                    return
                if path == "/api/molmoact2/stop":
                    server._handle_molmoact2_stop(self)
                    return
                if path == "/api/molmoact2/configure":
                    server._handle_molmoact2_configure(self)
                    return

                if path not in ("", "/"):
                    self.send_error(404)
                    return
                length = int(self.headers.get("Content-Length", "0"))
                raw = self.rfile.read(length)
                try:
                    body = json.loads(raw.decode("utf-8"))
                except json.JSONDecodeError as exc:
                    self._send_json(400, str(exc))
                    return
                if not isinstance(body, dict):
                    self._send_json(400, "JSON body must be an object.")
                    return
                try:
                    commands = server._parse_body(body)
                except Exception as exc:
                    self._send_json(400, str(exc))
                    return
                request = _MotionRequest(commands)
                with server._queue_lock:
                    server._pending = request
                if not request.event.wait(timeout=600.0):
                    self._send_json(400, "Motion request timed out.")
                    return
                if request.error is not None:
                    self._send_json(400, str(request.error))
                    return
                self._send_json(200, request.response)

            def _send_json(self, status: int, payload: Any) -> None:
                if isinstance(payload, str):
                    body = json.dumps(payload).encode("utf-8")
                else:
                    body = json.dumps(payload).encode("utf-8")
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                _safe_write(self, body)

        self._httpd = _QuietThreadingHTTPServer((self._host, self._port), Handler)
        self._thread = threading.Thread(target=self._httpd.serve_forever, daemon=True)
        self._thread.start()
        print(f"Listening at: http://{self._host}:{self._port}")
        if self._enable_ui:
            print(f"Web UI: http://127.0.0.1:{self._port}/")

    @staticmethod
    def _serve_file(handler: BaseHTTPRequestHandler, path: Path) -> None:
        data = path.read_bytes()
        content_type = mimetypes.guess_type(str(path))[0] or "application/octet-stream"
        handler.send_response(200)
        handler.send_header("Content-Type", content_type)
        handler.send_header("Content-Length", str(len(data)))
        if path.name == "index.html":
            handler.send_header("Cache-Control", "no-store, no-cache, must-revalidate")
        handler.end_headers()
        _safe_write(handler, data)

    def stop(self) -> None:
        if self._httpd is not None:
            self._httpd.shutdown()
            self._httpd.server_close()

    @staticmethod
    def _read_json_body(handler: BaseHTTPRequestHandler) -> dict[str, Any]:
        length = int(handler.headers.get("Content-Length", "0"))
        raw = handler.rfile.read(length)
        body = json.loads(raw.decode("utf-8"))
        if not isinstance(body, dict):
            raise ValueError("JSON body must be an object.")
        return body

    def _molmoact2_status_dict(self) -> dict[str, Any]:
        if self._molmoact2 is None:
            return {"configured": False, "running": False}
        status = self._molmoact2.get_status()
        return {"configured": True, **vars(status)}

    def _handle_molmoact2_configure(self, handler: BaseHTTPRequestHandler) -> None:
        if self._molmoact2 is None:
            handler._send_json(503, {"error": "MolmoAct2 requires cameras (--no-cameras disables it)."})
            return
        try:
            body = self._read_json_body(handler)
            status = self._molmoact2.configure(
                server_url=body.get("server_url"),
                instruction=body.get("instruction"),
                external_slot=body.get("external_slot"),
                external_camera_mode=body.get("external_camera_mode"),
            )
            handler._send_json(200, vars(status))
        except Exception as exc:
            handler._send_json(400, {"error": str(exc)})

    def _handle_molmoact2_start(self, handler: BaseHTTPRequestHandler) -> None:
        if self._molmoact2 is None:
            handler._send_json(503, {"error": "MolmoAct2 requires cameras (--no-cameras disables it)."})
            return
        try:
            body = self._read_json_body(handler)
            if body:
                self._molmoact2.configure(
                    server_url=body.get("server_url"),
                    instruction=body.get("instruction"),
                    external_slot=body.get("external_slot"),
                    external_camera_mode=body.get("external_camera_mode"),
                )
            status = self._molmoact2.start()
            handler._send_json(200, vars(status))
        except Exception as exc:
            handler._send_json(400, {"error": str(exc)})

    def _handle_molmoact2_stop(self, handler: BaseHTTPRequestHandler) -> None:
        if self._molmoact2 is None:
            handler._send_json(503, {"error": "MolmoAct2 controller not configured."})
            return
        status = self._molmoact2.stop()
        handler._send_json(200, vars(status))

    def process_pending(self) -> None:
        with self._queue_lock:
            request = self._pending
        if request is None:
            return
        try:
            request.response = self._execute(request.commands)
        except Exception as exc:
            request.error = exc
        finally:
            with self._queue_lock:
                if self._pending is request:
                    self._pending = None
            request.event.set()

    @staticmethod
    def _parse_body(body: dict[str, Any]) -> dict[str, list[float]]:
        commands: dict[str, list[float]] = {}
        for key, value in body.items():
            if not isinstance(value, list):
                raise RuntimeError("All values must be arrays.")
            numbers: list[float] = []
            for item in value:
                if not isinstance(item, (int, float)):
                    raise RuntimeError("All array elements must be numbers.")
                numbers.append(float(item))
            commands[key] = numbers
        return commands

    def _execute(self, commands: dict[str, list[float]]) -> dict[str, Any]:
        if self._molmoact2 is not None and self._molmoact2.is_running:
            manual_keys = {"moveToCartesian", "moveToJointPose", "closeGripper", "openGripper"}
            if manual_keys.intersection(commands):
                self._molmoact2.stop()
        response: dict[str, Any] = {}
        for key, numbers in commands.items():
            if key == "moveToCartesian":
                final_coords = self._handler.move_to_cartesian(numbers)
                response[key] = final_coords
            elif key == "moveToJointPose":
                final_q = self._handler.move_to_joint_pose(numbers)
                response[key] = final_q
            elif key == "closeGripper":
                response[key] = self._handler.close_gripper(numbers)
            elif key == "openGripper":
                response[key] = self._handler.open_gripper(numbers)
            elif key == "readState":
                response[key] = self._handler.read_state()
            elif key == "readJointState":
                response[key] = self._handler.read_joint_state()
            elif key == "readGripperState":
                response[key] = self._handler.read_gripper_state()
            elif key == "setJointPose":
                response[key] = self._handler.set_joint_pose(numbers)
            else:
                print(f"Invalid command: {key}")
                response["Response"] = "Invalid command."
        return response
