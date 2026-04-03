# Roboflow Home Server

## Project Overview
Local AI computer vision pipeline running on a TrueNAS SCALE home server. Uses an NVIDIA GPU to run object detection on UniFi security camera RTSP streams, triggering Apple HomeKit automations (phone notifications) via Homebridge when people are detected.

## Architecture

```
UniFi Camera (RTSP) → [Camera Watcher Container] → [Roboflow Inference Server :9001 (GPU)]
                              │
                              ▼ (on person detection)
                       [Homebridge Webhook] → HomeKit → iPhone notification
```

## Hardware
- **Host OS:** TrueNAS SCALE 25 (Fangtooth) — native Docker Compose for apps
- **CPU:** Intel 13th Gen (iGPU reserved exclusively for Plex 4K transcoding)
- **GPU:** NVIDIA RTX 2000 Ada Generation (16GB ECC VRAM, 70W TDP) — exclusive to Roboflow inference
- **Storage:** "fast" pool (mirrored Samsung 990 Evo NVMes)
- **Cameras:** UniFi NVR, hardwired to same network switch as TrueNAS

## Network Map
| Service | IP | Port | Notes |
|---|---|---|---|
| TrueNAS host | 192.168.20.10 | — | Hosts all Docker containers |
| Roboflow Inference API | 192.168.20.10 | 9001 | GPU container |
| UniFi NVR (RTSP) | 192.168.20.1 | 7447 | Unencrypted local streams |
| Homebridge UI | 192.168.20.4 | 8581 | Web management interface |
| Homebridge Webhooks | 192.168.20.4 | 51828 | `homebridge-http-webhooks` plugin (CONFIRM port in plugin config — 8581 is the UI, not webhooks) |

- **RTSP URL format:** `rtsp://192.168.20.1:7447/<token>` (per-camera tokens from UniFi)
- **Homebridge sensor:** `ai_person_detector` — virtual motion sensor triggered via HTTP GET

## Software Stack
- **Roboflow Inference Server:** GPU Docker container, model cache at `/mnt/fast/apps/roboflow`
- **Camera Watcher:** CPU-only Docker container (this project's `watcher/` directory)
- **OpenClaw (AI Agent):** CPU-only Ubuntu VM on TrueNAS — sandboxed, uses cloud LLMs only
- **Plex:** Native TrueNAS app using Intel iGPU
- **Homebridge:** Separate device (192.168.20.4), plugin `homebridge-http-webhooks`

## Project Structure
```
├── CLAUDE.md                # This file
├── docker-compose.yml       # Both services: inference server + camera watcher
├── .env.example             # Template — copy to .env with real values
└── watcher/
    ├── watcher.py           # Detection loop: RTSP → Roboflow inference → Homebridge webhook
    ├── Dockerfile           # Python 3.11-slim + OpenCV + ffmpeg
    └── requirements.txt     # inference-sdk, opencv-python-headless, requests
```

## Configuration (.env file)
Required:
- `RTSP_URL` — camera RTSP stream URL
- `ROBOFLOW_API_KEY` — from Roboflow account dashboard

Optional (defaults):
- `MODEL_ID=yolov8n-640` — can swap to yolov8s-640 for better accuracy, or custom models
- `CONFIDENCE_THRESHOLD=0.5`
- `COOLDOWN_SECONDS=30` — min time between notifications
- `FRAME_INTERVAL=1.0` — seconds between inference calls

## Current Progress
- [x] GPU installed, NVIDIA drivers enabled, verified via `nvidia-smi`
- [x] Roboflow GPU container deployed, caching models to NVMe
- [x] UniFi RTSP streams isolated and tested
- [x] End-to-end API test successful (Mac → TrueNAS Roboflow → bounding boxes with yolov8n-640)
- [x] Homebridge virtual motion sensor configured (`ai_person_detector`)
- [x] Camera watcher script and Docker setup created
- [ ] Deploy watcher container to TrueNAS
- [ ] Verify Homebridge webhook port (51828 vs 8581)
- [ ] Test end-to-end: camera → detection → phone notification
- [ ] Explore additional models and custom model training

## Deployment
```bash
# On TrueNAS, in the project directory:
cp .env.example .env
# Edit .env with RTSP token and Roboflow API key
docker compose up -d --build
docker logs -f camera-watcher
```

## Design Principles
- All inference runs locally (privacy, latency) — cloud only for LLM agentic tasks in sandboxed VM
- GPU resources are limited to a single RTX 2000 Ada — be mindful of VRAM when choosing models
- User has a Roboflow account with API key for Universe models and custom training

## Next Steps
- Multi-camera support (multiple RTSP streams)
- Custom model training and labeling via Roboflow
- Experiment with different detection models
