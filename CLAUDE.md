# Roboflow Home Server

See [README.md](README.md) for project overview, architecture, configuration, and deployment instructions.

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

## Software Stack (broader server context)

- **Roboflow Inference Server:** GPU Docker container, model cache at `/mnt/fast/apps/roboflow`
- **Camera Watcher:** CPU-only Docker container (this project's `watcher/` directory)
- **OpenClaw (AI Agent):** CPU-only Ubuntu VM on TrueNAS — sandboxed, uses cloud LLMs only
- **Plex:** Native TrueNAS app using Intel iGPU
- **Homebridge:** Separate device (192.168.20.4), plugin `homebridge-http-webhooks`

## Design Principles

- All inference runs locally (privacy, latency) — cloud only for LLM agentic tasks in sandboxed VM
- GPU resources are limited to a single RTX 2000 Ada — be mindful of VRAM when choosing models
- User has a Roboflow account with API key for Universe models and custom training

## Current Progress

- [x] GPU installed, NVIDIA drivers enabled, verified via `nvidia-smi`
- [x] Roboflow GPU container deployed, caching models to NVMe
- [x] UniFi RTSP streams isolated and tested
- [x] End-to-end API test successful (Mac → TrueNAS Roboflow → bounding boxes with yolov8n-640)
- [x] Homebridge virtual motion sensor configured (`ai_person_detector`)
- [x] Camera watcher script and Docker setup created
- [x] GitHub repo, CI pipeline, and TrueNAS app YAML created
- [ ] Deploy watcher container to TrueNAS
- [ ] Verify Homebridge webhook port (51828 vs 8581)
- [ ] Test end-to-end: camera → detection → phone notification
- [ ] Multi-camera support
- [ ] Custom model training via Roboflow
