#!/usr/bin/env python3
"""
Camera Watcher - Monitors RTSP streams via Roboflow inference for person detection.
Triggers Homebridge webhooks when people are detected.
"""

import cv2
import requests
import time
import os
import logging
from inference_sdk import InferenceHTTPClient

# --- Configuration from environment ---
RTSP_URL = os.environ["RTSP_URL"]
ROBOFLOW_API_URL = os.environ.get("ROBOFLOW_API_URL", "http://192.168.20.10:9001")
ROBOFLOW_API_KEY = os.environ.get("ROBOFLOW_API_KEY", "")
MODEL_ID = os.environ.get("MODEL_ID", "yolov8n-640")
HOMEBRIDGE_WEBHOOK_URL = os.environ["HOMEBRIDGE_WEBHOOK_URL"]
ACCESSORY_ID = os.environ.get("ACCESSORY_ID", "ai_person_detector")
CONFIDENCE_THRESHOLD = float(os.environ.get("CONFIDENCE_THRESHOLD", "0.5"))
COOLDOWN_SECONDS = int(os.environ.get("COOLDOWN_SECONDS", "30"))
FRAME_INTERVAL = float(os.environ.get("FRAME_INTERVAL", "1.0"))
RECONNECT_DELAY = int(os.environ.get("RECONNECT_DELAY", "5"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger("watcher")


def create_client():
    return InferenceHTTPClient(
        api_url=ROBOFLOW_API_URL,
        api_key=ROBOFLOW_API_KEY,
    )


def trigger_homebridge(state: bool):
    """Send a state change to the Homebridge HTTP webhook sensor."""
    url = f"{HOMEBRIDGE_WEBHOOK_URL}?accessoryId={ACCESSORY_ID}&state={'true' if state else 'false'}"
    try:
        resp = requests.get(url, timeout=5)
        log.info("Homebridge %s -> HTTP %d", "ON" if state else "OFF", resp.status_code)
    except Exception as e:
        log.error("Homebridge trigger failed: %s", e)


def process_frame(client, frame):
    """Run inference on a single frame and return person detections."""
    result = client.infer(frame, model_id=MODEL_ID)
    predictions = result.get("predictions", [])
    return [
        p for p in predictions
        if p["class"] == "person" and p["confidence"] >= CONFIDENCE_THRESHOLD
    ]


def run():
    client = create_client()
    last_trigger_time = 0
    sensor_active = False
    consecutive_errors = 0

    log.info("Watcher starting | model=%s | cooldown=%ds | interval=%.1fs",
             MODEL_ID, COOLDOWN_SECONDS, FRAME_INTERVAL)

    while True:
        log.info("Connecting to RTSP stream...")
        cap = cv2.VideoCapture(RTSP_URL, cv2.CAP_FFMPEG)

        if not cap.isOpened():
            log.error("Failed to open stream. Retrying in %ds...", RECONNECT_DELAY)
            time.sleep(RECONNECT_DELAY)
            continue

        log.info("Stream connected. Running detection loop.")
        consecutive_errors = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                log.warning("Lost stream frame. Reconnecting...")
                break

            try:
                persons = process_frame(client, frame)
                now = time.time()
                consecutive_errors = 0

                if persons:
                    best_conf = max(p["confidence"] for p in persons)
                    log.info("Detected %d person(s) (best: %.2f)", len(persons), best_conf)

                    if now - last_trigger_time > COOLDOWN_SECONDS:
                        trigger_homebridge(state=True)
                        last_trigger_time = now
                        sensor_active = True

                elif sensor_active and now - last_trigger_time > COOLDOWN_SECONDS:
                    trigger_homebridge(state=False)
                    sensor_active = False
                    log.info("No persons detected. Sensor reset.")

            except Exception as e:
                consecutive_errors += 1
                log.error("Inference error (%d): %s", consecutive_errors, e)
                if consecutive_errors > 10:
                    log.error("Too many consecutive errors. Reconnecting stream...")
                    break

            time.sleep(FRAME_INTERVAL)

        cap.release()
        time.sleep(RECONNECT_DELAY)


if __name__ == "__main__":
    run()
