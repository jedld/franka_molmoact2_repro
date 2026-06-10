MolmoAct2 inference Server
==========================

Authoritative API reference: `start_molmoact2_3090.md` (3090 launcher + HTTP contract).

USB cameras (RealSense wrist + V4L2 exterior on the control PC): configure `usb_cameras.json`
or `MOLMO_USB_*` env vars; run `hardware_molmoact2_runner.py` with `motion_server.cpp`, or
Isaac Sim with `--usb-cameras`. See repo `README.md` § USB cameras.

Server ready when startup warmup completes.

| Mode | Port | Endpoint | Image fields |
|------|------|----------|--------------|
| Single exterior | `8012` | `POST /act` | `external_cam`, `wrist_cam` |
| Dual / duplicate exterior | `8101` | `POST /act_dual` | `external_cam`, `external_cam_2`, `wrist_cam` |

Health: `GET /healthz` → `{"status":"ok"}`

Wire format: `json_numpy` (`Content-Type: application/json`). State is `(8,)` float32
(Franka q1–q7 + gripper). Response: `actions` `(15, 8)` float32, optional `dt_ms`.

Start dual-exterior server on GPU host:

```bash
MOLMOACT2_DUAL_EXTERIOR=1 ./start_molmoact2_3090.sh
```

### DROID camera layout (3 views)

MolmoAct2-DROID was trained on **three** RGB views: `[external_1, external_2, wrist]`.
The motion server maps sim cameras to the inference API via **Exterior cameras for policy**:

| Mode | Client behavior | GPU server |
|------|-----------------|------------|
| **Single** | `POST /act` with one exterior + wrist | Default `:8012` |
| **Both exteriors** | `POST /act_dual` with ext1, ext2, wrist | `:8101` (`MOLMOACT2_DUAL_EXTERIOR=1`) |
| **Duplicate** | `POST /act_dual` with same exterior twice + wrist | `:8101` (HF model-card workaround) |

Simulated DROID cameras (Isaac scene)
-------------------------------------

Hardware mirrored from the [DROID platform](https://droid-dataset.github.io/) and
[MolmoAct2-DROID](https://huggingface.co/allenai/MolmoAct2-DROID):

| Slot | Real hardware | Sim intrinsics | Placement |
|------|---------------|----------------|-----------|
| `external_1` | ZED 2 left RGB | LEFT_CAM_HD @ 1280×720 → 320×180 | Left tripod, elevated view |
| `external_2` | ZED 2 left RGB | same | Front tripod (+Y) |
| `wrist` | RealSense D435 color | D435 ROS K @ 640×480 → 320×180 | auto: behind gripper center, view −Z world |

Wrist pose is computed at spawn from finger link geometry (`motion_server_wrist_camera.py`).

Camera performance: MolmoAct2 reads RGB at **15 Hz** on the sim thread. Web UI JPEG
preview is encoded lazily on HTTP worker threads and does not run during physics ticks.
Preview / tune with:

```bat
project\launch_compute_wrist_camera_pose.bat
project\launch_compute_wrist_camera_pose.bat --behind 0.04
```

Tripod positions match `molmoact2_droid_validation.py`. Per-lab DROID data varies tripod
pose; this is the Table 6 reference layout, not a specific collection site.

Isaac Sim motion server integration
-----------------------------------

Launch the DROID scene + web UI (default inference URL matches single-exterior server):

```bat
project\launch_isaac_motion_server.bat --spawn-robot --task apple_on_plate
```

Optional overrides:

```bat
project\launch_isaac_motion_server.bat --spawn-robot --molmoact2-url http://192.168.0.233:8012
project\launch_isaac_motion_server.bat --spawn-robot --molmoact2-url http://192.168.0.233:8101
```

Web UI:

- Main page: `http://127.0.0.1:34568/`
- **MolmoAct2 panel (scroll down or use header link):** `http://127.0.0.1:34568/#molmoact2-panel`

Steps:

1. Pick exterior mode; set URL to `:8012` (single) or `:8101` (dual/duplicate).
2. Confirm **Check Server Health** reports `ok` (warmup finished on GPU host).
3. Enter or pick a Table 6 task instruction.
4. **DROID Ready First** (recommended) or use quick action, then **Start Inference**.

The server runs a 15 Hz closed loop: cameras + 8-D joint state → `/act` or `/act_dual` →
absolute joint targets on the simulated Franka. Manual motion commands auto-stop the loop.

HTTP control API (same port):

- `GET /api/molmoact2/status`
- `GET /api/molmoact2/health` — proxy to inference `/healthz` (validates `status == "ok"`)
- `GET /api/molmoact2/tasks` — Table 6 instruction presets
- `POST /api/molmoact2/configure` — `{ server_url, instruction, external_slot, external_camera_mode }`
  (`external_camera_mode`: `single` | `dual` | `duplicate`)
- `POST /api/molmoact2/start`
- `POST /api/molmoact2/stop`
