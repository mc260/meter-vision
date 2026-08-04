# meter-vision

<div align="right">
  <strong>中文</strong> | <a href="README.md">English</a>
</div>

**基于 YOLO 姿态估计的指针仪表自动读数系统。**

从图片或实时视频流中自动识别模拟指针仪表（压力表、电压表、电流表等）的读数。

---

## 功能特性

- 支持单张或多张图片上传，原图与检测结果左右对比显示
- 实时 MJPEG 视频流，帧上叠加关键点标注与仪表读数
- 支持本地 USB/内置摄像头和 RTSP IP 摄像头
- 单命令启动，后端同时托管前端页面，无需额外服务器
- 零依赖单文件 Web UI，无需 npm 或构建步骤

---

## 工作原理

1. **YOLO Pose 模型**在仪表上检测 10 个关键点：

   | kp_id | 作用 | 刻度值 |
   |---|---|---|
   | 0 | 指针底部 / 旋转轴 | — |
   | 1 | 指针中部 | — |
   | 2 | 指针末端 | — |
   | 3–9 | 表盘刻度点 | 0.0 → 6.0 |

2. **透视鲁棒读数算法**（`gauge_reader.py`）：
   - 以 `kp_id 0` 为角度原点，避免透视形变下圆拟合偏移
   - 对 `kp_id 1/2` 按置信度加权平均得到指针方向
   - 在刻度角度序列上做分段线性插值
   - 返回读数值及超量程标记

3. **FastAPI 后端**提供：
   - `GET /` 或 `GET /ui` — 前端页面
   - `POST /detect` — 图片推理
   - `GET /video_feed?source=<来源>` — MJPEG 视频流
   - `GET /video_stop` — 停止视频流
   - `GET /health` — 健康检查

---

## 项目结构

```
meter-vision/
├── backend/
│   ├── main.py            # FastAPI 服务
│   ├── gauge_reader.py    # 核心读数算法
│   ├── requirements.txt
│   └── Dockerfile
├── frontend/
│   └── index.html         # 单文件 Web UI
├── models/
│   └── best.pt            # YOLO Pose 权重
├── docker-compose.yml
└── README.md
```

---

## 快速开始

### 方式一 — 直接运行（推荐）

```bash
git clone https://github.com/your-username/meter-vision.git
cd meter-vision

# 安装依赖
pip install -r backend/requirements.txt

# 启动（API 与前端页面共用 9090 端口）
uvicorn backend.main:app --host 0.0.0.0 --port 9090
```

浏览器打开 **http://localhost:9090**

> **注意：** 必须在 `meter-vision/` 根目录下执行 `uvicorn`，不要在 `backend/` 内执行。

### 方式二 — Docker Compose

```bash
docker-compose up --build
```

- Web UI：http://localhost:8080
- API 文档：http://localhost:9090/docs

---

## 使用方法

### 图片检测

1. 打开 http://localhost:9090
2. 拖拽一张或多张仪表图片到上传区域
3. 自动触发检测，右侧显示标注效果、读数值和关键点列表
4. 悬停缩略图可删除，点击 `+` 可追加图片

### 视频流

1. 点击**视频流**标签页
2. 填写视频来源：
   - 本地摄像头：`0`（内置）或 `1`（外接 USB）
   - IP 摄像头：`rtsp://192.168.1.100:554/stream`
3. 点击**开始推流**，实时标注画面立即显示

---

## API 说明

### `POST /detect`

**请求：** `multipart/form-data`，字段名 `file`

**响应示例：**
```json
{
  "people": [
    {
      "person_id": 0,
      "keypoints": [{"kp_id": 0, "x": 613.2, "y": 976.8, "conf": 0.984}, "..."],
      "box": [483.6, 782.9, 1186.1, 972.5]
    }
  ],
  "reading": {
    "value": 3.45,
    "out_of_range": false,
    "confidence": 0.987,
    "pointer_angle_deg": -12.3,
    "pivot": [613.2, 976.8],
    "scale_points": [[1195.5, 863.8], "..."]
  }
}
```

### `GET /video_feed?source=<来源>`

返回 MJPEG 流（`multipart/x-mixed-replace`）。

| source 值 | 含义 |
|---|---|
| `0`、`1` | 本地摄像头编号 |
| `rtsp://…` | RTSP IP 摄像头地址 |

### `GET /video_stop`

停止当前视频流，返回 `{"status": "stopped"}`。

---

## 训练自己的模型

基于 [Ultralytics YOLO](https://github.com/ultralytics/ultralytics) 自定义关键点配置。

1. 按 COCO 关键点格式标注（每个仪表 10 个关键点，顺序同上）
2. 训练：
   ```bash
   yolo pose train data=your_data.yaml model=yolov8n-pose.pt epochs=200 imgsz=640
   ```
3. 将训练好的权重替换 `models/best.pt`

如需适配不同量程，修改 `backend/gauge_reader.py` 中的 `SCALE_MAP`。

---

## 环境要求

- Python 3.10+
- PyTorch（CPU 或 CUDA）
- 详见 `backend/requirements.txt`

---

## 开源协议

MIT License，详见 [LICENSE](LICENSE)。

---

## 致谢

- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
- [FastAPI](https://fastapi.tiangolo.com/)

---

*如果这个项目对你有帮助，欢迎点个 ⭐*
