# MuseTalk API

FastAPI wrapper around MuseTalk for single-GPU lip-sync video generation.

## Endpoints

| Method | Path | Description |
|---|---|---|
| `POST` | `/api/v1/upload-generate-video` | Multipart upload: `avatar_video` + `audio`. Returns `{job_id, status}`. |
| `GET`  | `/api/v1/job/{job_id}` | Job status: `queued` → `processing` → `completed` / `failed`. |
| `GET`  | `/api/v1/video/{job_id}` | Download the generated MP4 once `status == completed`. |
| `GET`  | `/health` | Liveness probe. |
| `GET`  | `/docs` | OpenAPI/Swagger UI. |

## Architecture

- **One process, one GPU.** Models load once at startup and stay in VRAM.
- **Async job queue** (`asyncio.Queue`) in-process. Jobs run sequentially — no concurrent GPU use.
- **SQLite** (`api/storage/jobs.db`) tracks job state; survives restarts.
- **Local disk** at `api/storage/{uploads,videos,jobs}/`.
- Uploads are cleaned up after the job completes.

## Setup (local / RunPod pod)

From the MuseTalk project root:

```bash
# 1. System deps
apt install -y ffmpeg

# 2. Python deps (once)
pip install -r requirements.txt
pip install -r api/requirements-api.txt
pip install -U openmim
mim install mmengine "mmcv==2.0.1" "mmdet==3.1.0" "mmpose==1.1.0"

# 3. Weights (once, ~5 GB)
bash download_weights.sh        # Linux/RunPod
# or: download_weights.bat      # Windows

# 4. Run
bash api/run.sh                 # Linux
# or: api\run.bat               # Windows
```

Server listens on `http://0.0.0.0:8000`. Swagger UI at `/docs`.

## Docker (RunPod recommended)

```bash
# Build from project root
docker build -f api/Dockerfile -t musetalk-api .

# Run (weights must be mounted or already in image at /app/models)
docker run --rm --gpus all -p 8000:8000 \
  -v $(pwd)/models:/app/models \
  -v $(pwd)/api/storage:/app/api/storage \
  musetalk-api
```

On RunPod: use an NVIDIA CUDA 11.8 base, mount a persistent volume at `/app/models` for weights and `/app/api/storage` for outputs.

## Configuration (env vars)

| Var | Default | Notes |
|---|---|---|
| `STORAGE_DIR` | `api/storage` | Uploads, videos, SQLite db |
| `MODELS_DIR` | `<project>/models` | MuseTalk weights |
| `USE_FLOAT16` | `1` | fp16 inference (strongly recommended) |
| `GPU_ID` | `0` | Which CUDA device |
| `BATCH_SIZE` | `8` | UNet batch size |
| `FPS` | `25` | Output FPS for image-only avatars |
| `EXTRA_MARGIN` | `10` | Chin margin (v15) |
| `PARSING_MODE` | `jaw` | `jaw` or `raw` |
| `LEFT_CHEEK_WIDTH` | `90` | Jaw mask left width |
| `RIGHT_CHEEK_WIDTH` | `90` | Jaw mask right width |
| `AUDIO_PAD_LEFT` | `2` | Whisper context frames (left) |
| `AUDIO_PAD_RIGHT` | `2` | Whisper context frames (right) |
| `MAX_UPLOAD_MB` | `500` | Per-file size cap |
| `CORS_ALLOW` | `*` | CORS allow-origin |
| `HOST` / `PORT` | `0.0.0.0` / `8000` | |

## Usage example

### Submit a job

```bash
curl -X POST http://localhost:8000/api/v1/upload-generate-video \
  -F "avatar_video=@avatar.mp4" \
  -F "audio=@speech.wav"
# → {"job_id": "abc123...", "status": "queued"}
```

### Poll status

```bash
curl http://localhost:8000/api/v1/job/abc123...
# → {"job_id": "abc123", "status": "processing", "progress": 55, "stage": "running_unet", ...}
```

Statuses: `queued`, `processing`, `completed`, `failed`
Stages (during `processing`): `starting`, `extracting_frames`, `extracting_landmarks`,
`encoding_latents`, `encoding_audio`, `running_unet`, `blending_frames`, `muxing_video`, `done`.

### Download result

```bash
curl -o result.mp4 http://localhost:8000/api/v1/video/abc123...
```

### Python client

```python
import requests, time

BASE = "http://localhost:8000/api/v1"

# Submit
with open("avatar.mp4", "rb") as v, open("speech.wav", "rb") as a:
    r = requests.post(
        f"{BASE}/upload-generate-video",
        files={"avatar_video": v, "audio": a},
    )
job_id = r.json()["job_id"]

# Poll
while True:
    s = requests.get(f"{BASE}/job/{job_id}").json()
    print(s["status"], s.get("progress"), s.get("stage"))
    if s["status"] in ("completed", "failed"):
        break
    time.sleep(2)

# Download
if s["status"] == "completed":
    with requests.get(f"{BASE}/video/{job_id}", stream=True) as resp:
        with open("out.mp4", "wb") as f:
            for chunk in resp.iter_content(1 << 20):
                f.write(chunk)
```

## Input requirements

**avatar_video** — mp4/mov/avi/mkv/webm, or a single image (mp4 strongly preferred).
A clearly visible face must be present. Static images produce a "frozen face with moving
lips" result — use a short idle clip (3–10 sec) for natural output. MuseTalk automatically
ping-pong cycles the clip to match any audio length.

**audio** — wav/mp3/m4a/flac/ogg/aac. Any length, any language.

## Notes on single-GPU behavior

- Only one inference runs at a time (serialized by `_GPU_LOCK`).
- Concurrent POST requests queue up and execute FIFO.
- Model load is ~10–20 sec on first startup; subsequent requests reuse the loaded state.
- Expected time for 1-min output on RTX 4090 with fp16: ~40–70 sec.
