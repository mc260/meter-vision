# meter-vision

<div align="right">
  <a href="README.zh-CN.md">中文</a> | <strong>English</strong>
</div>

**Pointer gauge reader powered by YOLO pose estimation.**

Automatically reads analog pointer gauges (pressure gauges, voltmeters, ammeters, etc.) from images or live video streams using keypoint detection.

---

## Demo

| Upload an image | Live stream with reading |
|---|---|
| Drag & drop → detect keypoints → get scale value | Connect camera / RTSP → overlaid real-time reading |

---

## How It Works

1. **YOLO Pose model** detects 10 keypoints on the gauge:
   - `kp_id 0` — pointer base (rotation pivot)
   - `kp_id 1–2` — points along the pointer arm
   - `kp_id 3–9` — dial scale marks (mapped to values 0.0 → 6.0)

2. **Perspective-robust algorithm** (`gauge_reader.py`):
   - Uses `kp_id 0` directly as the angular origin — avoids circle-fitting failure under perspective distortion
   - Computes pointer direction via confidence-weighted average of `kp_id 1/2`
   - Maps pointer angle onto scale via piecewise linear interpolation
   - Returns reading value + out-of-range flag

3. **FastAPI backend** exposes:
   - `POST /detect` — single image inference
   - `GET /video_feed?source=<cam>` — MJPEG stream with overlaid readings
   - `GET /video_stop` — stop active stream

4. **Web UI** (`frontend/index.html`) — zero-dependency single-page app:
   - Image upload with drag & drop
   - Canvas overlay: keypoints, scale rays, pointer arrow, reading value
   - Video stream tab: local camera index or RTSP URL

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
- API docs: http://localhost:9090/docs

### Option 2 — Run directly

**Backend:**
```bash
cd backend
pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port 9090
```

**Frontend:**
```bash
# Serve with Python built-in HTTP server
python -m http.server 8080 --directory frontend
```

Open http://localhost:8080, set the API address to `http://<backend-ip>:9090`.

---

## API Reference

### `POST /detect`

Upload an image, returns keypoints and gauge reading.

**Request:** `multipart/form-data`, field name `file`

**Response:**
```json
{
  "people": [
    {
      "person_id": 0,
      "keypoints": [
        {"kp_id": 0, "x": 613.2, "y": 976.8, "conf": 0.984},
        "..."
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
    "scale_points": [[1195.5, 863.8], "..."]
  }
}
```

### `GET /video_feed?source=<source>`

Returns an MJPEG stream with readings overlaid.

| `source` | Meaning |
|---|---|
| `0`, `1` | Local camera index |
| `rtsp://192.168.1.100:554/stream` | RTSP IP camera |

### `GET /video_stop`

Stops the active MJPEG stream.

---

## Keypoint Convention

| kp_id | Role | Scale value |
|---|---|---|
| 0 | Pointer base / rotation pivot | — |
| 1 | Pointer mid | — |
| 2 | Pointer tip | — |
| 3–9 | Scale marks | 0.0 → 6.0 |

To adapt to a different range, edit `SCALE_MAP` in `backend/gauge_reader.py`.

---

## Training Your Own Model

Uses [Ultralytics YOLO](https://github.com/ultralytics/ultralytics) with a custom keypoint config.

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
- See `backend/requirements.txt`

---

## License

MIT License. See [LICENSE](LICENSE).

---

## Acknowledgements

- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) — pose estimation backbone
- [FastAPI](https://fastapi.tiangolo.com/) — backend framework

---

*If this project helps you, please give it a ⭐*
