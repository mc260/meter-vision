"""
main.py  —  meter-vision backend
FastAPI server exposing:
  POST /detect          — single image inference
  GET  /video_feed      — MJPEG stream with live readings
  GET  /video_stop      — stop the active stream
  GET  /                — health check
"""

import sys
import time
import threading
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from ultralytics import YOLO

# Fix import path so gauge_reader can be found regardless of working directory
sys.path.insert(0, str(Path(__file__).parent))
from gauge_reader import compute_gauge_reading

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

# Resolve model path relative to this file, not the working directory
MODEL_PATH = str(Path(__file__).parent.parent / "models" / "best.pt")
CONF_THRESHOLD = 0.25
IMGSZ = 640
STREAM_FPS = 10          # max frames per second for MJPEG stream
STREAM_JPEG_QUALITY = 80


# ---------------------------------------------------------------------------
# Global model (loaded once at startup)
# ---------------------------------------------------------------------------

model: Optional[YOLO] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model
    model = YOLO(MODEL_PATH)
    print(f"[meter-vision] model loaded: {MODEL_PATH}")
    yield
    stop_stream()


app = FastAPI(
    title="meter-vision",
    description="Pointer gauge reading via YOLO pose estimation",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve frontend static files at root
_frontend_dir = Path(__file__).parent.parent / "frontend"
if _frontend_dir.exists():
    from fastapi.staticfiles import StaticFiles
    app.mount("/ui", StaticFiles(directory=str(_frontend_dir), html=True), name="frontend")


# ---------------------------------------------------------------------------
# Pydantic schemas
# ---------------------------------------------------------------------------

class Keypoint(BaseModel):
    kp_id: int
    x: float
    y: float
    conf: float


class PersonKeypoints(BaseModel):
    person_id: int
    keypoints: list[Keypoint]
    box: Optional[list[float]] = None


class DetectionResponse(BaseModel):
    people: list[PersonKeypoints]
    reading: Optional[dict] = None   # gauge reading if exactly 1 person detected


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def run_inference(image_bytes: bytes) -> dict:
    """Run YOLO pose on raw image bytes; return structured keypoint dict."""
    arr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Failed to decode image")

    results = model.predict(img, conf=CONF_THRESHOLD, imgsz=IMGSZ, verbose=False)

    people = []
    for person_id, kp_data in enumerate(results[0].keypoints.data):
        keypoints = [
            Keypoint(kp_id=i, x=float(x), y=float(y), conf=float(c))
            for i, (x, y, c) in enumerate(kp_data.tolist())
        ]
        box = None
        if results[0].boxes is not None and person_id < len(results[0].boxes.xyxy):
            box = results[0].boxes.xyxy[person_id].tolist()
        people.append(PersonKeypoints(person_id=person_id, keypoints=keypoints, box=box))

    return {"people": [p.model_dump() for p in people]}


def annotate_frame(frame: np.ndarray, api_data: dict) -> tuple[np.ndarray, Optional[dict]]:
    """Draw keypoints, pivot, scale rays, and pointer arrow onto frame."""
    reading = None
    if api_data["people"]:
        reading = compute_gauge_reading(api_data)

    for person in api_data["people"]:
        kps = {kp["kp_id"]: kp for kp in person["keypoints"]}

        # Draw all keypoints
        for kp in person["keypoints"]:
            if kp["conf"] < 0.5:
                continue
            x, y = int(kp["x"]), int(kp["y"])
            cv2.circle(frame, (x, y), 5, (0, 0, 255), -1)

        # Skeleton: pointer arm 0→1→2
        for i, j in ((0, 1), (1, 2)):
            kpi, kpj = kps.get(i), kps.get(j)
            if kpi and kpj and kpi["conf"] >= 0.5 and kpj["conf"] >= 0.5:
                cv2.line(frame,
                         (int(kpi["x"]), int(kpi["y"])),
                         (int(kpj["x"]), int(kpj["y"])),
                         (255, 165, 0), 2, cv2.LINE_AA)

    if reading and "error" not in reading:
        import math
        px, py = reading["pivot"]
        # Pivot marker
        cv2.circle(frame, (int(px), int(py)), 10, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.drawMarker(frame, (int(px), int(py)), (0, 255, 0),
                       cv2.MARKER_CROSS, 20, 2, cv2.LINE_AA)
        # Scale rays
        for sx, sy in reading["scale_points"]:
            cv2.line(frame, (int(px), int(py)), (int(sx), int(sy)),
                     (0, 200, 100), 1, cv2.LINE_AA)
            cv2.circle(frame, (int(sx), int(sy)), 4, (0, 200, 100), -1)
        # Pointer arrow
        mean_r = sum(math.hypot(sx - px, sy - py)
                     for sx, sy in reading["scale_points"]) / len(reading["scale_points"])
        rad = math.radians(reading["pointer_angle_deg"])
        tip = (int(px + mean_r * math.cos(rad)), int(py + mean_r * math.sin(rad)))
        color = (0, 69, 255) if reading["out_of_range"] else (255, 100, 0)
        cv2.arrowedLine(frame, (int(px), int(py)), tip,
                        color, 3, cv2.LINE_AA, tipLength=0.15)
        # Reading label
        label = f"{reading['value']:.2f}" + (" [OOR]" if reading["out_of_range"] else "")
        font = cv2.FONT_HERSHEY_SIMPLEX
        (tw, th), _ = cv2.getTextSize(label, font, 1.0, 2)
        tx, ty = max(tip[0] - tw // 2, 0), tip[1] - 14 if tip[1] > 40 else tip[1] + th + 14
        cv2.rectangle(frame, (tx - 4, ty - th - 4), (tx + tw + 4, ty + 4), (0, 0, 0), -1)
        cv2.putText(frame, label, (tx, ty), font, 1.0, (255, 255, 255), 2, cv2.LINE_AA)

    return frame, reading


# ---------------------------------------------------------------------------
# Stream state
# ---------------------------------------------------------------------------

_stream_lock = threading.Lock()
_stream_source: Optional[str] = None
_stream_active = False
_cap: Optional[cv2.VideoCapture] = None


def _stop_stream_nolock():
    """Stop stream without acquiring the lock (call only while holding it)."""
    global _stream_active, _cap
    _stream_active = False
    if _cap is not None:
        _cap.release()
        _cap = None


def stop_stream():
    global _stream_active, _cap
    with _stream_lock:
        _stop_stream_nolock()


def mjpeg_generator(source: str):
    global _stream_active, _cap, _stream_source

    # Set up new capture under lock, stopping any previous stream first
    with _stream_lock:
        _stop_stream_nolock()
        _stream_source = source
        cap = cv2.VideoCapture(int(source) if source.isdigit() else source)
        if not cap.isOpened():
            cap.release()
            return
        _cap = cap
        _stream_active = True

    interval = 1.0 / STREAM_FPS
    try:
        while True:
            # Check active flag and read frame WITHOUT holding the lock
            # to prevent stop_stream() from deadlocking
            with _stream_lock:
                active = _stream_active
                local_cap = _cap
            if not active or local_cap is None:
                break

            ret, frame = local_cap.read()
            if not ret:
                break

            # Run inference (outside lock — may take time)
            try:
                _, buf = cv2.imencode(".jpg", frame,
                                      [cv2.IMWRITE_JPEG_QUALITY, STREAM_JPEG_QUALITY])
                api_data = run_inference(buf.tobytes())
                frame, _ = annotate_frame(frame, api_data)
            except Exception:
                pass  # keep streaming even if inference fails on one frame

            _, out_buf = cv2.imencode(".jpg", frame,
                                      [cv2.IMWRITE_JPEG_QUALITY, STREAM_JPEG_QUALITY])
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n"
                + out_buf.tobytes()
                + b"\r\n"
            )
            time.sleep(interval)
    finally:
        stop_stream()


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/")
def health():
    return {"status": "ok", "model": MODEL_PATH}


@app.post("/detect", response_model=DetectionResponse)
async def detect(file: UploadFile = File(...)):
    """Run pose inference on an uploaded image and return keypoints + gauge reading."""
    img_bytes = await file.read()
    try:
        api_data = run_inference(img_bytes)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    reading = None
    if len(api_data["people"]) == 1:
        reading = compute_gauge_reading(api_data)

    return JSONResponse({"people": api_data["people"], "reading": reading})


@app.get("/video_feed")
def video_feed(source: str = "0"):
    """
    MJPEG stream with live gauge readings overlaid.
    source: camera index (e.g. 0) or RTSP URL (e.g. rtsp://192.168.1.1/stream)
    """
    return StreamingResponse(
        mjpeg_generator(source),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


@app.get("/video_stop")
def video_stop():
    """Stop the active MJPEG stream."""
    stop_stream()
    return {"status": "stopped"}
