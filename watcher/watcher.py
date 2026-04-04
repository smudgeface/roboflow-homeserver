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
import threading
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


class FrameGrabber:
    """Continuously reads frames from an RTSP stream in a background thread.
    The main loop always gets the latest frame instead of stale buffered ones.
    The thread keeps the RTSP connection alive by consuming frames continuously."""

    def __init__(self, rtsp_url):
        os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"
        self._cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
        self._frame = None
        self._ret = False
        self._lock = threading.Lock()
        self._stopped = False
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self):
        frames_read = 0
        consecutive_failures = 0
        while not self._stopped:
            ret, frame = self._cap.read()
            if not ret:
                consecutive_failures += 1
                if consecutive_failures >= 30:
                    log.warning("Grabber thread: %d consecutive read failures after %d frames, giving up",
                                consecutive_failures, frames_read)
                    break
                continue
            consecutive_failures = 0
            with self._lock:
                self._ret = ret
                self._frame = frame
            frames_read += 1
            if frames_read == 1:
                log.info("Grabber thread: first frame received")
            elif frames_read % 300 == 0:
                log.info("Grabber thread: %d frames read", frames_read)

    def read(self):
        with self._lock:
            return self._ret, (self._frame.copy() if self._frame is not None else None)

    def is_opened(self):
        return self._cap.isOpened()

    def is_alive(self):
        return self._thread.is_alive()

    def release(self):
        self._stopped = True
        self._thread.join(timeout=5)
        self._cap.release()


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
        grabber = FrameGrabber(RTSP_URL)

        if not grabber.is_opened():
            log.error("Failed to open stream. Retrying in %ds...", RECONNECT_DELAY)
            grabber.release()
            time.sleep(RECONNECT_DELAY)
            continue

        # Wait for the grabber to read the first frame
        for _ in range(50):
            ret, frame = grabber.read()
            if ret and frame is not None:
                break
            if not grabber.is_alive():
                break
            time.sleep(0.1)

        if not ret or frame is None:
            log.error("No frames received. Retrying in %ds...", RECONNECT_DELAY)
            grabber.release()
            time.sleep(RECONNECT_DELAY)
            continue

        log.info("Stream connected. Running detection loop.")
        consecutive_errors = 0

        while True:
            ret, frame = grabber.read()
            if not ret or frame is None or not grabber.is_alive():
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

        grabber.release()
        time.sleep(RECONNECT_DELAY)


if __name__ == "__main__":
    run()
