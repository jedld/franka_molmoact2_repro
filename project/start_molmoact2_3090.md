# MolmoAct2 on RTX 3090 — Server & Client Guide

This document describes `./start_molmoact2_3090.sh`: what it starts, how to configure it, and how to build client applications against the HTTP API.

**MolmoAct2** is an open vision-language-action (VLA) model from [Ai2](https://allenai.org/blog/molmoact2) for robot control. The launcher script clones [allenai/molmoact2](https://github.com/allenai/molmoact2), downloads a fine-tuned checkpoint (~22 GiB), and runs a FastAPI inference server tuned for a single **RTX 3090 (24 GB)**.

---

## Quick reference

| Item | Value |
|------|-------|
| Default embodiment | `droid` (Franka + DROID cameras) |
| Default bind address | `0.0.0.0` (all interfaces) |
| Default port (DROID, single exterior) | `8012` |
| Default port (DROID, dual exterior) | `8101` |
| Default port (YAM bimanual) | `8013` |
| Wire format | `json_numpy` over HTTP JSON bodies |
| Health endpoint | `GET /healthz` → `{"status":"ok"}` |
| Inference endpoint | `POST /act` (and `POST /act_dual` when dual exterior is enabled) |
| Server implementation (DROID) | `molmoact2_server_droid.py` (project root) |
| Server implementation (YAM) | `molmoact2/examples/yam/host_server_yam.py` |

---

## Architecture

```
┌──────────────────┐     json_numpy POST      ┌─────────────────────────────┐
│  Robot client    │ ───────────────────────▶ │  FastAPI server             │
│  (NUC, sim, etc.)│ ◀─────────────────────── │  :8012 / :8013 / :8101      │
│                  │     actions + dt_ms      │  MolmoAct2 + CUDA (3090)    │
└──────────────────┘                          └─────────────────────────────┘
```

The server is **stateless**: each `POST /act` is one observation in, one action chunk out. There is no session ID, auth layer, or WebSocket — plain HTTP with a custom JSON encoding for NumPy arrays.

---

## Starting the server

### Prerequisites

- NVIDIA GPU with CUDA (script targets **RTX 3090 24 GB**, compute capability sm_86)
- NVIDIA drivers (`nvidia-smi` must work)
- `git`
- Network access for first run (clone repo + download ~22 GiB model weights)

The script installs [uv](https://github.com/astral-sh/uv) if missing and manages Python deps inside `molmoact2/.venv`.

### Basic usage

```bash
chmod +x start_molmoact2_3090.sh
./start_molmoact2_3090.sh
```

Wait until startup **warmup** completes and logs show the server listening. Then:

```bash
curl -s http://localhost:8012/healthz
# {"status":"ok"}
```

### Common launch variants

```bash
# Bimanual YAM embodiment (3 cameras, 14-D state)
MOLMOACT2_EMBODIMENT=yam ./start_molmoact2_3090.sh

# DROID with two exterior cameras (+ wrist); enables POST /act_dual
MOLMOACT2_DUAL_EXTERIOR=1 ./start_molmoact2_3090.sh

# Faster inference (~2×) if VRAM allows
MOLMOACT2_CUDA_GRAPH=1 ./start_molmoact2_3090.sh

# Skip re-download when weights are already cached
MOLMOACT2_SKIP_DOWNLOAD=1 ./start_molmoact2_3090.sh

# Custom port / GPU
MOLMOACT2_PORT=9000 MOLMOACT2_GPU=0 ./start_molmoact2_3090.sh
```

### Environment variables

| Variable | Default | Description |
|----------|---------|-------------|
| `MOLMOACT2_EMBODIMENT` | `droid` | `droid` or `yam` |
| `MOLMOACT2_PORT` | `8012` / `8101` / `8013` | Listen port (see port table below) |
| `MOLMOACT2_HOST` | `0.0.0.0` | Bind address |
| `MOLMOACT2_GPU` | `0` | Physical GPU index (`CUDA_VISIBLE_DEVICES`) |
| `MOLMOACT2_DTYPE` | `bfloat16` | `bfloat16`, `float16`, or `float32` |
| `MOLMOACT2_CUDA_GRAPH` | `0` | `1` enables CUDA graph capture (~2× faster, +VRAM) |
| `MOLMOACT2_NO_WARMUP` | `0` | `1` skips the startup dummy inference pass |
| `MOLMOACT2_SKIP_DOWNLOAD` | `0` | `1` skips Hugging Face download if cached |
| `MOLMOACT2_FORCE_SYNC` | `0` | `1` re-runs `uv sync` even if venv exists |
| `MOLMOACT2_DUAL_EXTERIOR` | `0` | `1` enables `/act_dual` (DROID only) |
| `MOLMOACT2_MODEL` | embodiment default | Override HF repo id |
| `MOLMOACT2_DIR` | `./molmoact2` | Clone location for allenai/molmoact2 |
| `MOLMOACT2_HF_HOME` | `./models/huggingface` | Hugging Face cache directory |
| `MOLMOACT2_REPO_REF` | `main` | Git branch/tag for molmoact2 repo |

### Embodiments, models, and ports

| `MOLMOACT2_EMBODIMENT` | Hugging Face model | Default port | Cameras | State dim |
|------------------------|-------------------|--------------|---------|-----------|
| `droid` | `allenai/MolmoAct2-DROID` | `8012` | `external_cam`, `wrist_cam` | 8 |
| `droid` + `MOLMOACT2_DUAL_EXTERIOR=1` | same | `8101` | `external_cam`, `external_cam_2`, `wrist_cam` | 8 |
| `yam` | `allenai/MolmoAct2-BimanualYAM` | `8013` | `top_cam`, `left_cam`, `right_cam` | 14 |

Other MolmoAct2 checkpoints (LIBERO, SO-100/101) are not served by this script; use LeRobot or in-process `predict_action()` instead.

### OOM troubleshooting (3090)

If startup or inference fails with CUDA OOM:

1. `MOLMOACT2_DTYPE=bfloat16` (default; ~16 GiB VRAM)
2. `MOLMOACT2_CUDA_GRAPH=0` (CUDA graphs add ~2 GiB)
3. `MOLMOACT2_NO_WARMUP=1` (skips one heavy pass at boot)

`float32` needs ~24–26 GiB and often OOMs on a 3090.

---

## HTTP API

Base URL: `http://<host>:<port>` where `<port>` is from the table above.

All inference requests and responses use **`Content-Type: application/json`** bodies encoded with the **`json_numpy`** protocol (see next section). Standard `json.dumps` without `json_numpy` will not round-trip arrays.

### `GET /healthz`

Liveness check. No body.

**Response 200**

```json
{"status": "ok"}
```

Use this to wait until the model is loaded and warmup finished before sending observations.

### `POST /act`

Predict a chunk of robot actions from the current observation.

**DROID request fields (required)**

| Field | Type | Shape / notes |
|-------|------|---------------|
| `external_cam` | `uint8` ndarray | `(H, W, 3)` RGB, any resolution |
| `wrist_cam` | `uint8` ndarray | `(H, W, 3)` RGB |
| `instruction` | string | Natural-language task, e.g. `"pick up the cup"` |
| `state` | `float32` ndarray | `(8,)` — Franka joint positions `q1..q7` + gripper |

**YAM request fields (required)**

| Field | Type | Shape / notes |
|-------|------|---------------|
| `top_cam` | `uint8` ndarray | `(H, W, 3)` RGB — camera order matters |
| `left_cam` | `uint8` ndarray | `(H, W, 3)` RGB |
| `right_cam` | `uint8` ndarray | `(H, W, 3)` RGB |
| `instruction` | string | Natural-language task |
| `state` | `float32` ndarray | `(14,)` — bimanual joint state (7-D per arm) |

**Optional request fields (both embodiments)**

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `num_steps` | int | `10` | Flow-matching denoising steps (quality/latency tradeoff) |
| `enable_cuda_graph` | bool | server default | Per-request CUDA graph override |
| `timestamp` | float | — | Ignored by server; clients may send for logging |

**Success response 200**

| Field | Type | Description |
|-------|------|-------------|
| `actions` | `float32` ndarray | Action chunk; see action semantics below |
| `dt_ms` | float | Server-side inference time in milliseconds |

**Error responses**

| Status | Body | Typical cause |
|--------|------|---------------|
| 400 | `{"error": "..."}` | Missing field, bad array shape/dtype, invalid JSON |
| 500 | `{"error": "..."}` | Model/inference failure |

### `POST /act_dual` (DROID + dual exterior only)

Same as `/act` but requires **three** image fields:

| Field | Type |
|-------|------|
| `external_cam` | `(H, W, 3)` uint8 RGB |
| `external_cam_2` | `(H, W, 3)` uint8 RGB |
| `wrist_cam` | `(H, W, 3)` uint8 RGB |
| `instruction` | string |
| `state` | `(8,)` float32 |

Only available when the server was started with `MOLMOACT2_DUAL_EXTERIOR=1` (default port `8101`). Returns the same response shape as `/act`.

### YAM-only: `GET /act` health detail

The upstream YAM server also exposes `GET /act` with metadata (`repo_id`, `norm_tag`, `state_dim`, etc.). The DROID server in this project uses `GET /healthz` only for health.

---

## `json_numpy` wire format

NumPy arrays are embedded in JSON objects with a `__numpy__` key. **Clients must use the `json_numpy` Python package** (or reimplement the same encoding).

### Array encoding

Each ndarray becomes:

```json
{
  "__numpy__": "<base64-encoded raw bytes>",
  "dtype": "<numpy dtype string>",
  "shape": [<dim0>, <dim1>, ...]
}
```

Examples:

| Array | `dtype` | `shape` |
|-------|---------|---------|
| RGB image `uint8` | `"|u1"` | `[H, W, 3]` |
| State / actions `float32` | `"<f4"` | `[8]` or `[15, 8]` |

Full request example (conceptual structure):

```json
{
  "external_cam": {"__numpy__": "...", "dtype": "|u1", "shape": [480, 640, 3]},
  "wrist_cam":    {"__numpy__": "...", "dtype": "|u1", "shape": [480, 640, 3]},
  "instruction":  "open the drawer",
  "state":        {"__numpy__": "...", "dtype": "<f4", "shape": [8]}
}
```

### Python client dependency

```bash
pip install json-numpy requests numpy
```

Always serialize with `json_numpy.dumps()` and parse responses with `json_numpy.loads()`. Do not use plain `json.dumps` on ndarrays.

---

## Action semantics

### DROID (`franka_droid`)

- **State**: 8 floats — 7 arm joints + 1 gripper value, in the same units/normalization used during DROID training.
- **Actions**: `(15, 8)` `float32` — **15 future timesteps**, each 8-D (same layout as state). The server pads or truncates model output to exactly 15 rows.
- **Control loop**: Clients typically execute actions open-loop for several steps, then re-query the server with a fresh observation (~5 Hz is common).

### YAM (`yam_dual_molmoact2`)

- **State**: 14 floats — 7-D per arm (order matches training).
- **Actions**: `(N, D)` `float32` where `N` and `D` come from the checkpoint's `norm_stats.json` (commonly ~25 steps). Do not hardcode `N`/`D` in portable clients; read `actions.shape` at runtime.
- **Camera order** for the model: `[top, left, right]` — must match training.

### Images

- **Color space**: RGB (not BGR). Convert OpenCV frames with `cv2.cvtColor(img, cv2.COLOR_BGR2RGB)`.
- **Dtype**: `uint8` required for DROID server; YAM server clips non-uint8 to `[0, 255]`.
- **Resolution**: Arbitrary; the processor resizes internally. Warmup uses `180×320`.

---

## Client examples

### Python — DROID minimal client

```python
import json_numpy
import numpy as np
import requests

BASE = "http://127.0.0.1:8012"

def health() -> bool:
    r = requests.get(f"{BASE}/healthz", timeout=5)
    return r.status_code == 200 and r.json().get("status") == "ok"

def act(external_rgb: np.ndarray, wrist_rgb: np.ndarray, instruction: str, state: np.ndarray) -> dict:
    payload = {
        "external_cam": np.asarray(external_rgb, dtype=np.uint8),
        "wrist_cam": np.asarray(wrist_rgb, dtype=np.uint8),
        "instruction": instruction,
        "state": np.asarray(state, dtype=np.float32).reshape(8),
    }
    body = json_numpy.dumps(payload)
    r = requests.post(
        f"{BASE}/act",
        headers={"Content-Type": "application/json"},
        data=body,
        timeout=120,
    )
    r.raise_for_status()
    return json_numpy.loads(r.text)

# Example with dummy data
if health():
    out = act(
        np.zeros((480, 640, 3), dtype=np.uint8),
        np.zeros((480, 640, 3), dtype=np.uint8),
        "pick up the red block",
        np.zeros(8, dtype=np.float32),
    )
    actions = out["actions"]  # shape (15, 8)
    latency_ms = out["dt_ms"]
```

### Python — DROID dual exterior

```python
payload = {
    "external_cam": ext1,      # uint8 HxWx3
    "external_cam_2": ext2,    # uint8 HxWx3
    "wrist_cam": wrist,
    "instruction": instruction,
    "state": state,            # float32 (8,)
}
# POST http://127.0.0.1:8101/act_dual
```

### Python — YAM client

Reference implementation: `molmoact2/examples/yam/molmoact_client.py`. Payload keys:

```python
payload = {
    "top_cam": top_rgb,
    "left_cam": left_rgb,
    "right_cam": right_rgb,
    "instruction": instruction,
    "state": joint_positions,  # float32 (14,)
}
# POST http://127.0.0.1:8013/act
```

### Remote / LAN clients

The server binds `0.0.0.0` by default. Replace `127.0.0.1` with the host's LAN IP (e.g. `http://192.168.1.50:8012`). There is no TLS or API key — secure the network accordingly.

### Languages other than Python

Reimplement `json_numpy` encoding:

1. Flatten array in C-contiguous order.
2. Base64-encode raw bytes.
3. Set `dtype` and `shape` as in the table above.
4. POST JSON to `/act`, parse JSON response, decode `actions` the same way.

For images, sending pre-resized `uint8` RGB arrays keeps payloads smaller than float tensors.

---

## Operational behavior

| Topic | Behavior |
|-------|----------|
| **Warmup** | On start (unless `MOLMOACT2_NO_WARMUP=1`), server runs one dummy inference. First real request is faster after warmup. |
| **Concurrency** | Inference is serialized with a lock. Concurrent `POST /act` calls queue; do not rely on parallel requests for throughput. |
| **CUDA graphs** | When enabled (`MOLMOACT2_CUDA_GRAPH=1`), faster but not safe under concurrent calls; matches the single-lock design. |
| **Polling rate** | ~5 Hz observation rate is typical for robot loops. |
| **Timeouts** | Allow generous HTTP timeouts (30–120 s) on first request after cold start. |

---

## Machine-readable API schema (for LLM agents)

Use this block as structured context when generating client code.

```yaml
service: molmoact2-inference
launcher: ./start_molmoact2_3090.sh
protocol: http+json_numpy
auth: none

endpoints:
  - method: GET
    path: /healthz
    response_json: { status: ok }

  - method: POST
    path: /act
    content_type: application/json
    encoding: json_numpy
    embodiments:
      droid:
        port_default: 8012
        required_fields:
          external_cam: { dtype: uint8, shape: [H, W, 3], color: RGB }
          wrist_cam:    { dtype: uint8, shape: [H, W, 3], color: RGB }
          instruction:  { type: string }
          state:        { dtype: float32, shape: [8] }
        response_fields:
          actions: { dtype: float32, shape: [15, 8] }
          dt_ms:   { type: float }
      yam:
        port_default: 8013
        required_fields:
          top_cam:      { dtype: uint8, shape: [H, W, 3], color: RGB }
          left_cam:     { dtype: uint8, shape: [H, W, 3], color: RGB }
          right_cam:    { dtype: uint8, shape: [H, W, 3], color: RGB }
          instruction:  { type: string }
          state:        { dtype: float32, shape: [14] }
        response_fields:
          actions: { dtype: float32, shape: [N, D], note: "N,D from checkpoint" }
          dt_ms:   { type: float }

  - method: POST
    path: /act_dual
    condition: MOLMOACT2_DUAL_EXTERIOR=1 and embodiment=droid
    port_default: 8101
    required_fields:
      external_cam:   { dtype: uint8, shape: [H, W, 3] }
      external_cam_2: { dtype: uint8, shape: [H, W, 3] }
      wrist_cam:      { dtype: uint8, shape: [H, W, 3] }
      instruction:    { type: string }
      state:          { dtype: float32, shape: [8] }
    response_fields:
      actions: { dtype: float32, shape: [15, 8] }
      dt_ms:   { type: float }

client_libraries:
  python:
    - json-numpy  # json_numpy.dumps / json_numpy.loads
    - requests or httpx
  errors:
    400: { error: string }
    500: { error: string }
```

---

## Related files

| Path | Role |
|------|------|
| `start_molmoact2_3090.sh` | Launcher script |
| `molmoact2_server_droid.py` | DROID FastAPI server (+ `/act_dual`) |
| `molmoact2/examples/yam/host_server_yam.py` | YAM FastAPI server |
| `molmoact2/examples/yam/molmoact_client.py` | Reference YAM HTTP client |
| `molmoact2/CLAUDE.md` | Upstream wire-protocol notes |

## References

- [MolmoAct2 blog post](https://allenai.org/blog/molmoact2)
- [MolmoAct2 GitHub](https://github.com/allenai/molmoact2)
- [allenai/MolmoAct2-DROID](https://huggingface.co/allenai/MolmoAct2-DROID)
- [allenai/MolmoAct2-BimanualYAM](https://huggingface.co/allenai/MolmoAct2-BimanualYAM)
