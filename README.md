# meter-vision

<div align="right">
  <a href="README.En.md">English</a> | <strong>中文</strong>
</div>

**基于 YOLO 姿态估计的指针仪表自动读数系统。**

从图片或实时视频流中自动识别模拟指针仪表（压力表、电压表、电流表等）的读数。

---

## 工作原理

1. **YOLO Pose 模型**在仪表上检测 10 个关键点：
   - `kp_id 0` — 指针底部（旋转轴）
   - `kp_id 1–2` — 指针上的点，由底部向末端延伸
   - `kp_id 3–9` — 表盘刻度点，固定对应刻度值 0.0 → 6.0

2. **透视鲁棒读数算法**（`gauge_reader.py`）：
   - 直接以 `kp_id 0` 为角度原点，避免透视形变下圆拟合圆心偏移的问题
   - 对 `kp_id 1/2` 按置信度加权平均，得到指针方向向量
   - 对刻度角度序列做分段线性插值，计算最终读数
   - 返回读数值及越量程标记

3. **FastAPI 后端**提供以下接口：
   - `POST /detect` — 单张图片推理
   - `GET /video_feed?source=<来源>` — 叠加读数的 MJPEG 视频流
   - `GET /video_stop` — 停止当前视频流

4. **Web 前端**（`frontend/index.html`）— 零依赖单文件页面：
   - 拖拽上传图片，Canvas 叠加显示关键点、刻度射线、指针箭头和读数
   - 视频流标签页，支持本地摄像头编号或 RTSP 地址

---

## 项目结构

```
meter-vision/
├── backend/
│   ├── main.py            # FastAPI 服务（检测 + MJPEG 流）
│   ├── gauge_reader.py    # 核心读数算法
│   ├── requirements.txt
│   └── Dockerfile
├── frontend/
│   └── index.html         # 单文件 Web UI，无需构建
├── models/
│   └── best.pt            # YOLO Pose 权重文件
├── docker-compose.yml
└── README.md
```

---

## 快速开始

### 方式一 — Docker Compose（推荐）

```bash
git clone https://github.com/your-username/meter-vision.git
cd meter-vision
docker-compose up --build
```

- Web UI：http://localhost:8080
- API 文档：http://localhost:9090/docs

### 方式二 — 直接运行

**启动后端：**
```bash
cd backend
pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port 9090
```

**启动前端：**
```bash
# 使用 Python 内置 HTTP 服务器
python -m http.server 8080 --directory frontend
```

打开 http://localhost:8080，将 API 地址设置为 `http://<后端IP>:9090`。

---

## API 说明

### `POST /detect`

上传图片，返回关键点及仪表读数。

**请求：** `multipart/form-data`，字段名 `file`

**响应示例：**
```json
{
  "people": [
    {
      "person_id": 0,
      "keypoints": [
        {"kp_id": 0, "x": 613.2, "y": 976.8, "conf": 0.984},
        "..."
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
    "scale_points": [[1195.5, 863.8], "..."]
  }
}
```

### `GET /video_feed?source=<来源>`

返回叠加读数的 MJPEG 视频流。

| `source` 值 | 含义 |
|---|---|
| `0`、`1` | 本地摄像头编号 |
| `rtsp://192.168.1.100:554/stream` | RTSP IP 摄像头 |

### `GET /video_stop`

停止当前 MJPEG 视频流。

---

## 关键点约定

| kp_id | 作用 | 刻度值 |
|---|---|---|
| 0 | 指针底部 / 旋转轴 | — |
| 1 | 指针中部 | — |
| 2 | 指针末端 | — |
| 3–9 | 表盘刻度点 | 0.0 → 6.0 |

如需适配不同量程，修改 `backend/gauge_reader.py` 中的 `SCALE_MAP` 即可。

---

## 训练自己的模型

项目基于 [Ultralytics YOLO](https://github.com/ultralytics/ultralytics) 自定义关键点配置。

1. 按 COCO 关键点格式标注图片（每个仪表 10 个关键点，顺序同上）
2. 训练：
   ```bash
   yolo pose train data=your_data.yaml model=yolov8n-pose.pt epochs=200 imgsz=640
   ```
3. 将训练好的权重替换 `models/best.pt`

---

## 环境要求

- Python 3.10+
- PyTorch（CPU 或 CUDA 均可）
- 详见 `backend/requirements.txt`

---

## 开源协议

MIT License，详见 [LICENSE](LICENSE)。

---

## 致谢

- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) — 姿态估计骨干网络
- [FastAPI](https://fastapi.tiangolo.com/) — 后端框架

---

*如果这个项目对你有帮助，欢迎点个 ⭐*
