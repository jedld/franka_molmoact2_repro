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

## USB cameras (RealSense + exterior)

Use physical cameras on the control PC instead of Isaac Sim rendered views. Frames are resized to **320×180 RGB** for MolmoAct2.

### Two deployment modes

| Mode | Entry point | Robot | Cameras |
|------|-------------|-------|---------|
| **Hardware** | `hardware_molmoact2_runner.py` | Real Franka via `motion_server.cpp` | USB |
| **Sim + USB** | `launch_isaac_motion_server.sh --usb-cameras` | Isaac Sim Franka | USB (sim robot, real images) |

### Install dependencies (control PC)

```bash
pip install -r project/requirements-usb.txt
```

- **Wrist** — Intel RealSense D435 via `pyrealsense2` (`device: "realsense"`).
- **Exterior** — ZED 2 / UVC webcams via OpenCV V4L2 (`device: "0"`, `"/dev/video2"`, or `/dev/v4l/by-id/...`).

List devices:

```bash
# V4L2 indices and by-id symlinks
ls -l /dev/v4l/by-id/
v4l2-ctl --list-devices

# RealSense serial numbers
rs-enumerate-devices | grep Serial
```

### Configuration

Edit [`project/usb_cameras.json`](project/usb_cameras.json) (checked in as a template):

```json
{
  "wrist": { "device": "realsense", "width": 640, "height": 480, "fps": 30 },
  "external_1": { "device": "0", "width": 1280, "height": 720, "fps": 15 },
  "external_2": { "device": "1", "width": 1280, "height": 720, "fps": 15 }
}
```

| Field | Meaning |
|-------|---------|
| `wrist.device` | `realsense`, `realsense:<serial>`, or V4L2 path / index for a UVC wrist cam |
| `external_*.device` | V4L2 index (`"0"`) or `/dev/v4l/by-id/usb-...` (stable across reboots) |
| `width` / `height` / `fps` | Native USB capture resolution (resized to 320×180 for policy) |

**Environment overrides** (alternative to editing JSON):

| Variable | Example |
|----------|---------|
| `MOLMO_USB_CONFIG` | `/path/to/usb_cameras.json` |
| `MOLMO_USB_WRIST` | `realsense:123456789` |
| `MOLMO_USB_EXTERNAL_1` | `/dev/v4l/by-id/usb-ZED_...` |
| `MOLMO_USB_EXTERNAL_2` | `1` |
| `MOLMO_USB_EXT1_WIDTH` / `MOLMO_USB_EXT1_HEIGHT` | `1280` / `720` |

### Hardware MolmoAct2 loop (real Franka)

Three processes:

```bash
# Terminal 1 — libfranka motion server (rebuild after pulling setJointPose / readGripperState)
cd project && ./motion_server <robot-fci-ip>

# Terminal 2 — MolmoAct2 GPU server
./start_molmoact2_3090.sh

# Terminal 3 — USB cameras + 15 Hz policy loop
pip install -r project/requirements-usb.txt
./project/launch_hardware_molmoact2.sh \
  --instruction "Pick up the apple and place it on the plate" \
  --external-camera-mode single
```

Dual exterior (port `8101`, `MOLMOACT2_DUAL_EXTERIOR=1` on GPU host):

```bash
./project/launch_hardware_molmoact2.sh \
  --instruction "Pick up the cup" \
  --external-camera-mode dual \
  --molmoact2-url http://127.0.0.1:8101
```

### Isaac Sim with USB cameras

Simulated robot, real camera feeds (useful for sim2real camera testing):

```bash
./project/launch_isaac_motion_server.sh --spawn-robot --usb-cameras
./project/launch_isaac_motion_server.sh --spawn-robot --usb-cameras --usb-config /path/to/usb_cameras.json
```

### motion_server.cpp additions for hardware policy

The hardware runner uses high-rate joint streaming (not 5 s `moveToJointPose` trajectories):

| Command | Purpose |
|---------|---------|
| `setJointPose` | 7 arm joints (rad); optional 8th = gripper width (m) — MolmoAct2 @ 15 Hz |
| `readGripperState` | `[width]` for 8-D observation construction |

Rebuild `motion_server` after updating `motion_server.cpp`.

## Hardware motion server (`motion_server.cpp`)

`project/motion_server.cpp` is the original **libfranka** HTTP server for a physical Franka Panda over FCI. The Isaac Sim stack above implements the same JSON API so clients can swap sim for hardware without changes.

This file is **not** built or run by the Isaac Sim launch scripts. You compile it on a Linux control PC that can reach the robot's FCI interface.

### Prerequisites

- Franka Research robot with FCI enabled and Desk unlocked for external control.
- Control PC on the same network as the robot (default FCI IP in code: `192.168.2.100`).
- [libfranka](https://frankarobotics.github.io/docs/libfranka/docs/installation.html) (C++ library + headers).
- [C++ REST SDK](https://github.com/microsoft/cpprestsdk) (`libcpprest-dev` on Ubuntu/Debian).
- `common.h` and `common.cpp` from your robotics codebase — **not included in this repo**. They provide `goHome(franka::Robot&)` used during startup homing. Place both files next to `motion_server.cpp` before building.

On Ubuntu/Debian, system packages for cpprest and Boost:

```bash
sudo apt-get install build-essential libcpprest-dev libssl-dev libboost-system-dev
```

Install libfranka per the [official docs](https://frankarobotics.github.io/docs/libfranka/docs/installation.html) (Debian package or CMake build from [frankarobotics/libfranka](https://github.com/frankarobotics/libfranka)).

### Build

From the `project/` directory, with `common.cpp` present:

```bash
cd project

g++ -std=c++11 motion_server.cpp common.cpp -o motion_server \
  -lfranka -lcpprest -lssl -lcrypto -lboost_system -pthread
```

If libfranka was built from source and is not on the default linker path, add include and library flags, for example:

```bash
g++ -std=c++11 motion_server.cpp common.cpp -o motion_server \
  -I"$LIBFRANKA/include" -L"$LIBFRANKA/build" \
  -lfranka -lcpprest -lssl -lcrypto -lboost_system -pthread
```

Replace `$LIBFRANKA` with your libfranka checkout root.

### Run

```bash
./motion_server                  # robot at 192.168.2.100, listen on 0.0.0.0:34568
./motion_server 172.16.0.42      # custom FCI IP
```

Startup sequence:

1. Connects to the arm, clears stale errors, and relaxes collision thresholds for non-RT kernels (`RealtimeConfig::kIgnore`).
2. Homing — tries `goHome()` from `common.cpp` (needs a real-time kernel); on failure falls back to an in-file cubic joint trajectory (`goHomeJoints`). You may be prompted to confirm motion.
3. Homes the gripper and prints the initial end-effector pose.
4. Listens for HTTP `POST` requests until you press **Enter** in the terminal.

Keep a hand on the E-stop whenever the arm is under automatic control.

### HTTP API

All commands are `POST http://<host>:34568` with `Content-Type: application/json`. Each JSON key maps to a numeric array (use `[]` when no arguments are needed). Multiple keys in one request are executed in order.

| Command | Arguments | Response |
|---------|-----------|----------|
| `moveToCartesian` | `[x, y, z]` or `[x, y, z, tf_sec]` or `[x, y, z, tf, dα, dβ, dγ]` (positions in m; optional motion time ≥ 5 s; optional relative rotations in degrees) | `[x, y, z, α, β, γ]` (position m, ZYX Euler degrees) |
| `moveToJointPose` | `[q1…q7]` radians, optional 8th element `tf_sec` (≥ 5 s) | final `[q1…q7]` |
| `readState` | `[]` | `[x, y, z, α, β, γ]` |
| `readJointState` | `[]` | `[q1…q7]` radians |
| `readGripperState` | `[]` | `[width]` meters between fingers |
| `setJointPose` | `[q1…q7]` rad; optional 8th = gripper width (m) | final `[q1…q7]` — for MolmoAct2 policy loop |
| `closeGripper` | `[]` (default 1 cm grasp) or `[width_m]` | status string |
| `openGripper` | `[]` (default speed 0.1) or `[speed]` | status string |

Examples:

```bash
# Read end-effector pose and joint angles
curl -s -X POST http://127.0.0.1:34568 -H "Content-Type: application/json" \
  -d '{"readState": [], "readJointState": []}'

# Move to Cartesian target (meters, robot base frame)
curl -s -X POST http://127.0.0.1:34568 -H "Content-Type: application/json" \
  -d '{"moveToCartesian": [0.5, 0.0, 0.3]}'

# Move to joint pose (radians) — same layout MolmoAct2 uses for actions
curl -s -X POST http://127.0.0.1:34568 -H "Content-Type: application/json" \
  -d '{"moveToJointPose": [0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785]}'

# Gripper
curl -s -X POST http://127.0.0.1:34568 -H "Content-Type: application/json" \
  -d '{"closeGripper": []}'
curl -s -X POST http://127.0.0.1:34568 -H "Content-Type: application/json" \
  -d '{"openGripper": []}'
```

On collision or reflex errors during motion, the server attempts `automaticErrorRecovery()` and returns HTTP 400 with a `collision_recovery:` prefix so callers can retry or replan.

See [Long-finger gripper mod → Real Franka](#real-franka-motion_servercpp) if you run this server with extended gripper fingers.

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
  motion_server_gripper.py      # Long-finger STL mount + tuneables
  motion_server_usb_cameras.py  # USB RealSense + V4L2 capture
  hardware_molmoact2_runner.py  # Real robot + USB cameras + MolmoAct2
  launch_hardware_molmoact2.sh
  usb_cameras.json              # USB device template (edit for your PC)
  requirements-usb.txt
  assets/panda_long_finger.stl
  compute_wrist_camera_pose.py
  web_ui/index.html           # Browser UI
  launch_isaac_motion_server.{sh,bat}
  start_molmoact2_3090.md     # MolmoAct2 GPU server guide (authoritative API)
  AGENTS.md                   # Integration reference for automation
  motion_server.cpp           # libfranka HTTP server for real hardware (see above)
```

Generated paths (not committed): `_build/` (Isaac Sim), `molmoact2/` and `models/` (inference weights).

## Long-finger gripper mod

CAD mesh: `project/assets/panda_long_finger.stl` (also at repo root as `panda_long_finger.stl`). Sim and hardware paths differ below.

### Isaac Sim

The sim loads the STL on both `panda_leftfinger` / `panda_rightfinger` at spawn time (stock finger meshes are hidden). Implementation: `project/motion_server_gripper.py`.

#### Sim parameters to tune

| Location | Constant | When to change |
|----------|----------|----------------|
| `motion_server_gripper.py` | `ENABLE_LONG_FINGER_MOD` | Set `False` to revert to stock Isaac `AlternateFinger` meshes. |
| `motion_server_gripper.py` | `LONG_FINGER_STL_SCALE` | Default `0.001` (STL in **millimeters**). Use `1.0` if your CAD exports meters. |
| `motion_server_gripper.py` | `LONG_FINGER_LOCAL_TRANSLATION` | Mesh sits wrong at the finger base — shift in the finger link frame (m). |
| `motion_server_gripper.py` | `LONG_FINGER_LOCAL_ROTATION_DEG_XYZ` | Finger points the wrong way — rotate so the tip aligns with hand **+Z** (see `motion_server_wrist_camera.py`). |
| `motion_server_gripper.py` | `LONG_FINGER_MIRROR_RIGHT_FINGER_Y` | Right finger looks flipped — toggle (default `True`). |
| `motion_server_gripper.py` | `LONG_FINGER_GRIPPER_CENTER_Z_OFFSET_M` | Wrist camera framing is off — set manually (m, hand +Z) or leave `None` for auto-estimate from STL bounds. |
| `motion_server_wrist_camera.py` | `WRIST_CAMERA_USER_TRANSLATION_OFFSET` | Fine-tune wrist RGB after the long finger is mounted. |
| `motion_server_wrist_camera.py` | `DEFAULT_BEHIND_GRIPPER_M` / `--behind` | Camera too close to / far from the longer fingers (`launch_compute_wrist_camera_pose.bat`). |
| `motion_server_handler.py` | `GRIPPER_OPEN` / `GRIPPER_MAX_WIDTH` | Only if the mod changes max jaw opening (default 8 cm — usually unchanged). |

On first launch, check the console line `Long finger mod applied:` for mesh bounds. Preview the wrist view with:

```bat
project\launch_compute_wrist_camera_pose.bat
project\launch_compute_wrist_camera_pose.bat --behind 0.04
```

MolmoAct2 joint/gripper semantics (`state` 8-D, action chunk) are unchanged in sim — only geometry and wrist-camera placement shift.

### Real Franka (`motion_server.cpp`)

**No mandatory code changes.** `motion_server.cpp` talks to libfranka’s gripper API (jaw width, force, homing). It does not model finger mesh geometry — that is a mechanical install plus Franka Desk configuration.

#### Hardware setup (not in C++)

1. Mount the long fingers on the gripper.
2. Configure them in **Franka Desk** if your lab requires it.
3. Start `motion_server` — `initialize()` calls `gripper.homing()` on boot.
4. Confirm runtime max opening: `gripper.readOnce().max_width` (often still ~0.08 m unless the mod changes jaw travel).

#### What stays the same in `motion_server.cpp`

| Area | Why |
|------|-----|
| HTTP API | Same commands (`moveToCartesian`, `readJointState`, `closeGripper`, etc.) |
| `openGripper` | Uses runtime `gripper_state.max_width` from libfranka — no hardcoded max |
| `readState` / `readJointState` | Still flange pose and arm joints; gripper width semantics unchanged |
| `moveToJointPose` | Joint targets unaffected by finger length |
| MolmoAct2 8-D state | Still q1–q7 + gripper scalar from finger joints |

#### Optional C++ tweaks (only if you see problems)

**1. Table clearance in `moveToCartesian`** — most likely change. `isValidCartesianPose()` rejects targets with flange `z < 0.015` m:

```cpp
if (zf < 0.015) {
    throw std::runtime_error("The desired position is too close to the table.");
}
```

That limit applies to the **flange** (`O_T_EE`), not the fingertip. Longer fingers reach the table sooner — raise `0.015` by roughly the extra finger length beyond stock (often **+0.02 to +0.04 m**). Tune with a few manual `moveToCartesian` tests.

**2. `closeGripper` grasp parameters** — only if grasps fail or feel wrong:

```cpp
gripper.grasp(grasping_width, 0.1, 40.0, 0.05, 0.05);
```

| Parameter | Default | When to change |
|-----------|---------|----------------|
| `grasping_width` (empty payload) | `0.01` m | Usually fine — target gap between pads |
| speed | `0.1` | Slow down if fingers bounce |
| force | `40.0` N | Reduce if long fingers flex or slip; increase if objects drop |
| `epsilon_inner` / `epsilon_outer` | `0.05` m | Widen if pad contact is inconsistent with longer fingers |

**3. Collision thresholds** — constructor `setCollisionBehavior()` is already loosened for non-RT kernels. Tune only if you get more `cartesian_reflex` with heavier/longer fingers.

**4. Workspace sphere (`0.855` m)** — rarely needs changing.

#### What you do not need in `motion_server.cpp`

- No STL or mesh loading (sim-only; see `motion_server_gripper.py`).
- No `GRIPPER_MAX_WIDTH` constant — max width comes from libfranka at runtime.
- No wrist-camera offsets — physical cameras are fixed on the hardware rig.

#### Flange frame vs fingertip contact

`readState` and `moveToCartesian` use the **hand/flange frame** (`O_T_EE`). Fingertips sit further along the tool axis with long fingers. If you plan motions in **contact-point** coordinates, apply that offset in your client, or raise the table Z guard in `isValidCartesianPose()` — `motion_server.cpp` does not apply a fingertip offset today.

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

**Motion API** (same contract as `motion_server.cpp`) — `POST http://127.0.0.1:34568` with JSON keys such as `moveToCartesian`, `moveToJointPose`, `openGripper`, `closeGripper`, `readState`, `readJointState`. See [Hardware motion server](#hardware-motion-server-motion_servercpp) for the full command table.

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
