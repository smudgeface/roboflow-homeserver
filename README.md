# Roboflow Home Server

Local AI computer vision pipeline running on a TrueNAS SCALE home server. Uses an NVIDIA GPU to run object detection on UniFi security camera RTSP streams, triggering Apple HomeKit notifications via Homebridge when people are detected.

## Architecture

```
UniFi Camera (RTSP) → [Camera Watcher Container] → [Roboflow Inference Server :9001 (GPU)]
                              │
                              ▼ (on person detection)
                       [Homebridge Webhook] → HomeKit → iPhone notification
```

**Camera Watcher** grabs frames from an RTSP stream, sends them to a local **Roboflow Inference Server** for object detection, and triggers a **Homebridge** virtual motion sensor when a person is detected. Everything runs on the local network — no cloud inference.

## Hardware

- **Host OS:** TrueNAS SCALE 25 (Fangtooth) — native Docker Compose
- **GPU:** NVIDIA RTX 2000 Ada Generation (16GB ECC VRAM) — runs inference
- **CPU:** Intel 13th Gen (iGPU reserved for Plex)
- **Storage:** Mirrored Samsung 990 Evo NVMes — model cache
- **Cameras:** UniFi NVR, hardwired

## Project Structure

```
├── docker-compose.yml           # Local dev: inference server + watcher
├── truenas-app.yaml             # TrueNAS "Install via YAML" deployment
├── .env.example                 # Template — copy to .env with real values
├── .github/workflows/
│   └── build-push.yml           # CI: build watcher image → ghcr.io
└── watcher/
    ├── watcher.py               # Detection loop: RTSP → inference → webhook
    ├── Dockerfile               # Python 3.11-slim + OpenCV + ffmpeg
    └── requirements.txt
```

## Configuration

Copy `.env.example` to `.env` and fill in:

| Variable | Required | Default | Description |
|---|---|---|---|
| `RTSP_URL` | Yes | — | Camera RTSP stream URL |
| `ROBOFLOW_API_KEY` | Yes | — | From Roboflow account dashboard |
| `MODEL_ID` | No | `yolov8n-640` | Detection model (try `yolov8s-640` for better accuracy) |
| `CONFIDENCE_THRESHOLD` | No | `0.5` | Minimum confidence to trigger |
| `COOLDOWN_SECONDS` | No | `30` | Min seconds between notifications |
| `FRAME_INTERVAL` | No | `1.0` | Seconds between inference calls |

## Deployment

### TrueNAS (production)

1. Push to `main` — GitHub Actions builds and pushes `ghcr.io/smudgeface/camera-watcher:latest`

2. On TrueNAS, authenticate to ghcr.io (one-time):
   ```bash
   docker login ghcr.io -u smudgeface
   ```

3. In TrueNAS UI: **Apps → Discover → Custom App → Install via YAML**
   - Paste `truenas-app.yaml` contents
   - Replace placeholder values (`YOUR_CAMERA_TOKEN`, `your_roboflow_api_key_here`)

### Local development

```bash
cp .env.example .env
# Fill in RTSP_URL and ROBOFLOW_API_KEY
docker compose up -d --build
docker logs -f camera-watcher
```

## Roadmap

- [ ] Deploy and verify end-to-end pipeline
- [ ] Multi-camera support
- [ ] Custom model training via Roboflow
- [ ] Experiment with different detection models
