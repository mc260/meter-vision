# meter-vision

**Pointer gauge reader powered by YOLO pose estimation.**

Automatically reads analog pointer gauges (pressure gauges, voltmeters, ammeters, etc.) from images or live video streams using keypoint detection.

> **基于 YOLO 姿态估计的指针仪表自动读数系统**，支持图片上传与实时视频流。

---

## Demo

| Upload an image | Live stream with reading |
|---|---|
| Upload → detect keypoints → get scale value | Connect camera / RTSP → overlaid real-time reading |

---

## How It Works

1. **YOLO Pose model** detects 10 keypoints on the gauge:
   - `kp_id 0` — pointer base (rotation pivot)
   - `kp_id 1–2` — points along the pointer arm
   - `kp_id 3–9` — dial scale marks (mapped to values 0.0 → 6.0)

2. **Perspective-robust algorithm** (`gauge_reader.py`):
   - Uses `kp_id 0` directly as the angular origin (avoids circle-fitting failure under perspective distortion)
   - Computes pointer direction via confidence-weighted average of `kp_id 1/2`
   - Maps pointer angle onto scale via piecewise linear interpolation
   - Returns reading value + out-of-range flag

3. **FastAPI backend** exposes:
   - `POST /detect` — single image inference
   - `GET /video_feed?source=<cam>` — MJPEG stream with overlaid readings
   - `GET /video_stop` — stop active stream

4. **Web UI** (`frontend/index.html`) — zero-dependency single-page app:
   - Image upload with drag & drop
   - Canvas overlay showing keypoints, scale rays, pointer arrow, and reading
   - Video stream tab supporting local camera index or RTSP URL

---

## Project Structure

```
meter-vision/
├── backend/
│   ├── main.py            # FastAPI server (detect + MJPEG stream)
│   ├── gauge_reader.py    # Core reading algorithm
│   ├── requirements.txt
│   └── Dockerfile
├── frontend/
│   └── index.html         # Single-file web UI (no build step)
├── models/
│   └── best.pt            # YOLO pose weights
├── docker-compose.yml
└── README.md
```

---

## Quick Start

### Option 1 — Docker Compose (recommended)

```bash
git clone https://github.com/your-username/meter-vision.git
cd meter-vision
docker-compose up --build
```

- Web UI: http://localhost:8080
- API:    http://localhost:9090/docs

### Option 2 — Run directly

**Backend:**
```bash
cd backend
pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port 9090
```

**Frontend:**

Open `frontend/index.html` directly in your browser, or serve with any static server:
```bash
# Python built-in
python -m http.server 8080 --directory frontend
```

Then open http://localhost:8080 and set the API server address to `http://<backend-ip>:9090`.

---

## API Reference

### `POST /detect`

Upload an image file, returns keypoints and gauge reading.

**Request:** `multipart/form-data` with field `file`

**Response:**
```json
{
  "people": [
    {
      "person_id": 0,
      "keypoints": [
        {"kp_id": 0, "x": 613.2, "y": 976.8, "conf": 0.984},
        ...
      ],
      "box": [483.6, 782.9, 1186.1, 972.5]
    }
  ],
  "reading": {
    "value": 3.45,
    "out_of_range": false,
    "confidence": 0.9872,
    "pointer_angle_deg": -12.3,
    "pivot": [613.2, 976.8],
    "scale_points": [[1195.5, 863.8], ...]
  }
}
```

### `GET /video_feed?source=<source>`

Returns an MJPEG stream with readings overlaid.

| `source` value | Meaning |
|---|---|
| `0`, `1`, `2` | Local camera index |
| `rtsp://192.168.1.100:554/stream` | RTSP IP camera |

### `GET /video_stop`

Stops the active MJPEG stream.

---

## Keypoint Label Convention

| kp_id | Role | Scale value |
|---|---|---|
| 0 | Pointer base / rotation pivot | — |
| 1 | Pointer mid point | — |
| 2 | Pointer tip | — |
| 3 | Scale mark | 0.0 |
| 4 | Scale mark | 1.0 |
| 5 | Scale mark | 2.0 |
| 6 | Scale mark | 3.0 |
| 7 | Scale mark | 4.0 |
| 8 | Scale mark | 5.0 |
| 9 | Scale mark | 6.0 |

To adapt to a different scale range, edit `SCALE_MAP` in `backend/gauge_reader.py`.

---

## Training Your Own Model

This project uses [Ultralytics YOLO](https://github.com/ultralytics/ultralytics) with a custom keypoint configuration.

1. Annotate images in COCO keypoint format (10 keypoints per gauge, order as above)
2. Train:
   ```bash
   yolo pose train data=your_data.yaml model=yolov8n-pose.pt epochs=200 imgsz=640
   ```
3. Replace `models/best.pt` with your trained weights

---

## Requirements

- Python 3.10+
- PyTorch (CPU or CUDA)
- See `backend/requirements.txt` for full list

---

## License

MIT License. See [LICENSE](LICENSE).

---

## Acknowledgements

- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) — pose estimation backbone
- [FastAPI](https://fastapi.tiangolo.com/) — backend framework

---

*If this project helps you, please give it a ⭐*
