# Franka MolmoAct2 Reproduction

Isaac Sim motion server that runs a closed-loop **MolmoAct2-DROID** policy on a simulated Franka arm. The sim exposes the same HTTP motion API as the original `motion_server.cpp` (libfranka) and adds DROID-style cameras, a web UI, and a 15 Hz inference loop against a remote MolmoAct2 GPU server.

## Architecture

```
┌─────────────────────────────┐     RGB + joint state      ┌──────────────────────────┐
│  Isaac Sim (this repo)      │ ─────────────────────────▶ │  MolmoAct2 inference     │
│  Franka + DROID cameras     │     POST /act or /act_dual │  (GPU host, ~22 GiB)     │
│  Web UI :34568              │ ◀───────────────────────── │  :8012 or :8101          │
└─────────────────────────────┘     action chunk (15×8)    └──────────────────────────┘
```

| Component | Port | Role |
|-----------|------|------|
| Motion server + web UI | `34568` | Sim control, camera preview, MolmoAct2 panel |
| MolmoAct2 single exterior | `8012` | `POST /act` — one exterior + wrist |
| MolmoAct2 dual exterior | `8101` | `POST /act_dual` — two exteriors + wrist |

## Prerequisites

**Simulation host**

- Built [Isaac Sim](https://developer.nvidia.com/isaac-sim) tree with `_build/linux-x86_64/release` (Linux) or `_build/windows-x86_64/release` (Windows) at the **repository root** (parent of `project/`).
- Run `./build.sh` (or `build.bat`) from your Isaac Sim source checkout before using the launch scripts.

**Inference host** (can be the same or a separate GPU machine)

- NVIDIA GPU with CUDA (tested on RTX 3090 24 GB).
- MolmoAct2 server — see [`project/start_molmoact2_3090.md`](project/start_molmoact2_3090.md) for the launcher script, API contract, and client examples.

## Quick start

### 1. Start MolmoAct2 on the GPU host

Follow [`project/start_molmoact2_3090.md`](project/start_molmoact2_3090.md). Wait until warmup completes:

```bash
curl -s http://localhost:8012/healthz
# {"status":"ok"}
```

For two exterior cameras (DROID layout), use `MOLMOACT2_DUAL_EXTERIOR=1` and port `8101`.

### 2. Launch Isaac Sim motion server

**Linux**

```bash
./project/launch_isaac_motion_server.sh --spawn-robot --task apple_on_plate
```

**Windows**

```bat
project\launch_isaac_motion_server.bat --spawn-robot --task apple_on_plate
```

Point inference at your GPU host if needed:

```bash
./project/launch_isaac_motion_server.sh --spawn-robot \
  --molmoact2-url http://192.168.0.233:8012
```

### 3. Run inference from the web UI

1. Open `http://127.0.0.1:34568/` (MolmoAct2 panel: `#molmoact2-panel`).
2. Set exterior mode and inference URL (`:8012` or `:8101`).
3. Confirm **Check Server Health** returns `ok`.
4. Pick a Table 6 task instruction, then **DROID Ready First** → **Start Inference**.

## Project layout

```
project/
  isaac_motion_server.py      # Main entry — Isaac Sim app + motion API
  motion_server_api.py        # HTTP server (motion + MolmoAct2 control routes)
  motion_server_handler.py    # Franka articulation control
  motion_server_scene.py      # DROID workstation + Table 6 task props
  motion_server_cameras.py    # ZED 2 / RealSense D435 camera streams
  motion_server_molmoact2.py  # 15 Hz closed-loop inference client
  motion_server_wrist_camera.py
  compute_wrist_camera_pose.py
  web_ui/index.html           # Browser UI
  launch_isaac_motion_server.{sh,bat}
  start_molmoact2_3090.md     # MolmoAct2 GPU server guide (authoritative API)
  AGENTS.md                   # Integration reference for automation
  motion_server.cpp           # Original libfranka HTTP server (reference)
```

Generated paths (not committed): `_build/` (Isaac Sim), `molmoact2/` and `models/` (inference weights).

## Tasks

Built-in Table 6 instructions (`--task`):

| Key | Instruction |
|-----|-------------|
| `apple_on_plate` | Pick up the apple and place it on the plate |
| `pipette_in_tray` | Place the pipette in the tray |
| `red_cube_in_tape_roll` | Place the red cube in the center of the tape roll |
| `knife_in_box` | Put the knife in the box |
| `objects_in_bowl` | Move the objects into the bowl |

## HTTP APIs

**Motion API** (compatible with `motion_server.cpp`) — `POST http://127.0.0.1:34568` with JSON keys such as `moveToCartesian`, `moveToJointPose`, `openGripper`, `closeGripper`, `readState`.

**MolmoAct2 control** (same port):

- `GET /api/molmoact2/status`
- `GET /api/molmoact2/health`
- `GET /api/molmoact2/tasks`
- `POST /api/molmoact2/configure`
- `POST /api/molmoact2/start`
- `POST /api/molmoact2/stop`

See [`project/AGENTS.md`](project/AGENTS.md) for camera layout, exterior modes, and integration details.

## References

- [MolmoAct2 (Ai2)](https://allenai.org/blog/molmoact2)
- [allenai/molmoact2](https://github.com/allenai/molmoact2)
- [MolmoAct2-DROID on Hugging Face](https://huggingface.co/allenai/MolmoAct2-DROID)
- [DROID dataset platform](https://droid-dataset.github.io/)

## License

Source files in `project/` are marked SPDX `Apache-2.0` (Copyright NVIDIA CORPORATION & AFFILIATES). `motion_server.cpp` is included as reference for the original libfranka interface.
