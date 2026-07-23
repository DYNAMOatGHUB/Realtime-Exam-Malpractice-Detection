# ExamGuard AI 🛡️

**Real-time exam hall malpractice detection powered by computer vision and AI.**

A fully self-hosted, zero-cloud-cost system that ingests live CCTV RTSP feeds, runs a 12-layer YOLOv11 + Pose + LSTM/GRU inference pipeline on your RTX GPU, and automatically alerts invigilators by email when suspicious behaviour is detected.

---

## Architecture

```
CCTV RTSP ──► OpenCV Capture ──► Redis Queue
                                       │
                              Celery Worker (GPU)
                              ┌───────────────────┐
                              │ YOLOv11 Detection │
                              │ YOLOv11-Pose      │
                              │ LSTM/GRU Classify │
                              └─────────┬─────────┘
                                        │ confidence score
                    ┌───────────────────┼────────────────┐
                 HIGH│               MID│            LOW │
                    ▼                  ▼                 ▼
             Auto Evidence        Review Queue        Discard
             + Email Alert       (HEC Dashboard)
                    │                  │
               MinIO Storage     PostgreSQL Log
                                       │
                              Django Dashboard
                              (Admin + HEC)
```

---

## Tech Stack

| Component | Technology |
|---|---|
| ML Inference | YOLOv11 (Ultralytics), PyTorch LSTM/GRU, TensorRT |
| API Server | FastAPI (async), WebSocket |
| Task Queue | Celery + Redis |
| Dashboard | Django + Channels (ASGI), HTMX |
| Database | PostgreSQL + SQLAlchemy |
| Object Storage | MinIO |
| Email | Gmail SMTP |
| Proxy | Nginx |
| Container | Docker Compose |

---

## Quick Start

### Prerequisites
- Docker + Docker Compose v2
- NVIDIA GPU with CUDA 12.1+ (RTX 5070 recommended)
- NVIDIA Container Toolkit installed

### 1. Clone & configure

```bash
git clone https://github.com/your-org/exam-anomaly-detection-system.git
cd exam-anomaly-detection-system
cp .env.example .env
```

Edit `.env` and fill in:
- `GMAIL_USER` and `GMAIL_APP_PASSWORD` (Gmail App Password, not your account password)
- `SECRET_KEY` — generate with `python -c "import secrets; print(secrets.token_urlsafe(50))"`
- Review all other defaults (ports, MinIO credentials, etc.)

### 2. Start the stack

```bash
docker compose up -d
```

First run will pull images and build containers (~5 min). Watch logs with:
```bash
docker compose logs -f
```

### 3. Create the admin user

```bash
docker compose exec dashboard python manage.py createsuperuser
```

### 4. Run database migrations

```bash
docker compose exec dashboard python manage.py migrate
```

### 5. Access the dashboards

| Service | URL | Credentials |
|---|---|---|
| HEC Dashboard | http://localhost (via Nginx) | Your Django superuser |
| Django Admin | http://localhost/django-admin/ | Superuser |
| FastAPI Docs | http://localhost/api/docs | — |
| MinIO Console | http://localhost:9001 | `minioadmin` / `minioadmin` |

---

## Usage

### Adding a lecture hall mapping

1. Log in as HEC user
2. Navigate to **LH Mapping**
3. Click **Add Mapping** → fill in Lecture Hall, Camera ID (RTSP URL), Invigilator name + email
4. Save

### Starting a camera stream

```bash
curl -X POST http://localhost/api/streams/start \
  -H "Content-Type: application/json" \
  -d '{"camera_id": "cam_01", "rtsp_url": "rtsp://192.168.1.10:554/stream1"}'
```

Or use a local `.mp4` file for testing:
```bash
# ffmpeg re-streams a local file as RTSP
ffmpeg -re -i test_video.mp4 -c copy -f rtsp rtsp://localhost:8554/test
```

### Reviewing detections

- **High confidence** (≥80%): Automatically packaged + email sent to invigilator
- **Mid confidence** (50–79%): Appears in **Review Queue** for manual confirm/dismiss
- **Low confidence** (<50%): Discarded silently

---

## ML Pipeline

### Using pre-trained weights

Place model files in `ml/weights/`:
- `yolo11n.pt` or `yolo11n.engine` (TensorRT) — person/object detection
- `yolo11n-pose.pt` — pose estimation
- `lstm_classifier.pt` — behaviour classification

### Training your own LSTM classifier

```bash
# Prepare your dataset (annotated keypoint sequences)
python ml/training/dataset.py --input /path/to/raw --output /path/to/processed

# Train
python ml/training/train_lstm.py \
  --data /path/to/processed \
  --epochs 50 \
  --batch-size 32 \
  --output ml/weights/lstm_classifier.pt

# Evaluate
python ml/training/evaluate.py --model ml/weights/lstm_classifier.pt --data /path/to/test
```

### TensorRT export (for maximum inference speed)

```bash
python ml/models/trt_export.py \
  --yolo ml/weights/yolo11n.pt \
  --lstm ml/weights/lstm_classifier.pt
```

---

## Running Tests

```bash
# Backend unit tests
docker compose exec fastapi pytest backend/tests/ -v

# Or locally (requires dependencies installed)
cd backend && pytest tests/ -v
```

---

## Configuration Reference

See [`.env.example`](.env.example) for all available settings.

Key settings:

| Variable | Default | Description |
|---|---|---|
| `CONFIDENCE_HIGH_THRESHOLD` | `0.80` | Auto-alert threshold |
| `CONFIDENCE_MID_THRESHOLD` | `0.50` | Review queue threshold |
| `ALERT_COOLDOWN_SECONDS` | `300` | Min seconds between alerts per camera |
| `MAX_EMAILS_PER_HOUR` | `10` | Rate limit for email alerts |
| `FRAME_SAMPLE_RATE` | `5` | Frames per second to sample from RTSP |
| `LSTM_WINDOW_SECONDS` | `10` | Sliding window for behaviour classification |

---

## Project Structure

```
exam-anomaly-detection-system/
├── backend/              # FastAPI inference API
│   ├── app/
│   │   ├── core/         # Config, DB, Redis, MinIO clients
│   │   ├── models/       # SQLAlchemy ORM models
│   │   ├── routers/      # REST API endpoints
│   │   └── services/     # Business logic (RTSP, inference, email, alert)
│   └── tests/
├── dashboard/            # Django HEC + Admin dashboard
│   ├── apps/
│   │   ├── accounts/     # Auth, RBAC
│   │   ├── exam_control/ # LH mapping, live monitor, review queue
│   │   └── admin_panel/  # User management, model upload, health
│   ├── templates/        # Dark-mode glassmorphism HTML
│   └── static/           # CSS + JS
├── ml/                   # Machine learning pipeline
│   ├── models/           # YOLO, Pose, LSTM wrappers
│   ├── training/         # Dataset, train loop, evaluation
│   └── weights/          # .pt / .engine files (gitignored)
├── infra/
│   ├── postgres/         # init.sql
│   ├── minio/            # setup.sh
│   └── nginx/            # nginx.conf
├── Dockerfile.backend
├── Dockerfile.dashboard
├── Dockerfile.celery
├── docker-compose.yml
└── .env.example
```

---

## License

MIT License. See [LICENSE](LICENSE) for details.
