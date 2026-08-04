"""
main.py  —  meter-vision backend
Endpoints:
  GET  /              → serve frontend (index.html)
  GET  /ui            → serve frontend (alias)
  POST /detect        → single image inference + gauge reading
  GET  /video_feed    → MJPEG stream with live readings
  GET  /video_stop    → stop the active stream
  GET  /health        → health check
"""

import asyncio
import math
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
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel
from ultralytics import YOLO

# Ensure gauge_reader is importable regardless of working directory
sys.path.insert(0, str(Path(__file__).parent))
from gauge_reader import compute_gauge_reading

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

MODEL_PATH    = str(Path(__file__).parent.parent / "models" / "best.pt")
FRONTEND_DIR  = Path(__file__).parent.parent / "frontend"
CONF_THRESHOLD   = 0.25
IMGSZ            = 640
STREAM_FPS       = 10
STREAM_JPEG_QUAL = 80

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

model: Optional[YOLO] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model
    print(f"[meter-vision] loading model: {MODEL_PATH}")
    model = YOLO(MODEL_PATH)
    print("[meter-vision] model ready")
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
    reading: Optional[dict] = None


# ---------------------------------------------------------------------------
# Inference helpers
# ---------------------------------------------------------------------------

def run_inference(image_bytes: bytes) -> dict:
    """Run YOLO pose on raw image bytes; return structured keypoint dict."""
    arr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Failed to decode image")

    results = model.predict(img, conf=CONF_THRESHOLD, imgsz=IMGSZ, verbose=False)
    people = []
    for pid, kp_data in enumerate(results[0].keypoints.data):
        keypoints = [
            Keypoint(kp_id=i, x=float(x), y=float(y), conf=float(c))
            for i, (x, y, c) in enumerate(kp_data.tolist())
        ]
        box = None
        if results[0].boxes is not None and pid < len(results[0].boxes.xyxy):
            box = results[0].boxes.xyxy[pid].tolist()
        people.append(PersonKeypoints(person_id=pid, keypoints=keypoints, box=box))

    return {"people": [p.model_dump() for p in people]}


def annotate_frame(frame: np.ndarray, api_data: dict) -> np.ndarray:
    """Draw keypoints, pivot, scale rays, and pointer arrow onto frame in-place."""
    reading = None
    if api_data["people"]:
        reading = compute_gauge_reading(api_data)

    for person in api_data["people"]:
        kps = {kp["kp_id"]: kp for kp in person["keypoints"]}
        for kp in person["keypoints"]:
            if kp["conf"] < 0.5:
                continue
            cv2.circle(frame, (int(kp["x"]), int(kp["y"])), 5, (0, 0, 255), -1)
        for i, j in ((0, 1), (1, 2)):
            ki, kj = kps.get(i), kps.get(j)
            if ki and kj and ki["conf"] >= 0.5 and kj["conf"] >= 0.5:
                cv2.line(frame,
                         (int(ki["x"]), int(ki["y"])),
                         (int(kj["x"]), int(kj["y"])),
                         (255, 165, 0), 2, cv2.LINE_AA)

    if reading and "error" not in reading:
        px, py = reading["pivot"]
        cv2.circle(frame, (int(px), int(py)), 10, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.drawMarker(frame, (int(px), int(py)), (0, 255, 0),
                       cv2.MARKER_CROSS, 20, 2, cv2.LINE_AA)
        for sx, sy in reading["scale_points"]:
            cv2.line(frame, (int(px), int(py)), (int(sx), int(sy)),
                     (0, 200, 100), 1, cv2.LINE_AA)
            cv2.circle(frame, (int(sx), int(sy)), 4, (0, 200, 100), -1)
        mean_r = sum(math.hypot(sx - px, sy - py)
                     for sx, sy in reading["scale_points"]) / len(reading["scale_points"])
        rad = math.radians(reading["pointer_angle_deg"])
        tip = (int(px + mean_r * math.cos(rad)), int(py + mean_r * math.sin(rad)))
        color = (0, 69, 255) if reading["out_of_range"] else (255, 100, 0)
        cv2.arrowedLine(frame, (int(px), int(py)), tip,
                        color, 3, cv2.LINE_AA, tipLength=0.15)
        label = f"{reading['value']:.2f}" + (" [超程]" if reading["out_of_range"] else "")
        font = cv2.FONT_HERSHEY_SIMPLEX
        (tw, th), _ = cv2.getTextSize(label, font, 1.0, 2)
        tx = max(tip[0] - tw // 2, 0)
        ty = tip[1] - 14 if tip[1] > 40 else tip[1] + th + 14
        cv2.rectangle(frame, (tx - 4, ty - th - 4), (tx + tw + 4, ty + 4), (0, 0, 0), -1)
        cv2.putText(frame, label, (tx, ty), font, 1.0, (255, 255, 255), 2, cv2.LINE_AA)

    return frame


# ---------------------------------------------------------------------------
# Video stream (async generator — non-blocking)
# ---------------------------------------------------------------------------

_stream_lock   = threading.Lock()
_stream_active = False
_cap: Optional[cv2.VideoCapture] = None


def _open_cap(source: str) -> Optional[cv2.VideoCapture]:
    """Open VideoCapture; try DirectShow first on Windows to avoid MSMF hangs."""
    if source.isdigit():
        idx = int(source)
        cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
        if cap.isOpened():
            return cap
        cap.release()
        cap = cv2.VideoCapture(idx)
        if cap.isOpened():
            return cap
        cap.release()
        return None
    cap = cv2.VideoCapture(source)
    return cap if cap.isOpened() else (cap.release() or None)


def _release_cap():
    global _stream_active, _cap
    with _stream_lock:
        _stream_active = False
        if _cap is not None:
            _cap.release()
            _cap = None


def stop_stream():
    _release_cap()


def _read_frame_sync(cap: cv2.VideoCapture):
    """Called from thread pool — blocking cap.read()."""
    return cap.read()


def _encode_frame_sync(frame: np.ndarray, api_data: dict) -> bytes:
    """Run inference + annotate + JPEG encode — called from thread pool."""
    try:
        annotated = annotate_frame(frame, api_data)
    except Exception:
        annotated = frame
    _, buf = cv2.imencode(".jpg", annotated,
                          [cv2.IMWRITE_JPEG_QUALITY, STREAM_JPEG_QUAL])
    return buf.tobytes()


async def mjpeg_generator(source: str):
    """
    Async MJPEG generator.
    Blocking calls (cap.open, cap.read, inference) run in a thread-pool executor
    so the event loop stays free.
    """
    global _stream_active, _cap

    # Open camera in thread pool (may block for several seconds on Windows)
    cap = await asyncio.to_thread(_open_cap, source)
    if cap is None:
        print(f"[meter-vision] cannot open camera/stream: {source}")
        return

    with _stream_lock:
        if _cap is not None:
            _cap.release()
        _cap = cap
        _stream_active = True

    interval = 1.0 / STREAM_FPS
    try:
        while True:
            with _stream_lock:
                if not _stream_active or _cap is None:
                    break
                local_cap = _cap

            # Blocking read in thread pool
            ret, frame = await asyncio.to_thread(_read_frame_sync, local_cap)
            if not ret:
                print("[meter-vision] stream: cap.read() returned False, stopping")
                break

            # Encode raw frame to JPEG for inference input
            _, raw_buf = cv2.imencode(".jpg", frame,
                                      [cv2.IMWRITE_JPEG_QUALITY, STREAM_JPEG_QUAL])
            # Run inference + annotation in thread pool
            try:
                api_data = await asyncio.to_thread(run_inference, raw_buf.tobytes())
                out_bytes = await asyncio.to_thread(_encode_frame_sync, frame, api_data)
            except Exception as e:
                print(f"[meter-vision] inference error: {e}")
                _, buf = cv2.imencode(".jpg", frame,
                                      [cv2.IMWRITE_JPEG_QUALITY, STREAM_JPEG_QUAL])
                out_bytes = buf.tobytes()

            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n"
                + out_bytes
                + b"\r\n"
            )
            await asyncio.sleep(interval)
    finally:
        _release_cap()


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/health")
def health():
    return {"status": "ok", "model": MODEL_PATH}


@app.get("/")
@app.get("/ui")
@app.get("/ui/")
async def serve_ui():
    """Serve the frontend single-page app."""
    index = FRONTEND_DIR / "index.html"
    if not index.exists():
        raise HTTPException(status_code=404, detail="Frontend not found")
    return FileResponse(str(index), media_type="text/html")


@app.post("/detect", response_model=DetectionResponse)
async def detect(file: UploadFile = File(...)):
    """Run pose inference on an uploaded image; return keypoints + gauge reading."""
    img_bytes = await file.read()
    try:
        api_data = await asyncio.to_thread(run_inference, img_bytes)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    reading = compute_gauge_reading(api_data) if api_data["people"] else None
    return JSONResponse({"people": api_data["people"], "reading": reading})


@app.get("/video_feed")
async def video_feed(source: str = "0"):
    """
    MJPEG stream with live gauge readings overlaid.
    source: camera index (0, 1, …) or rtsp://… URL
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
