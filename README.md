# YOLOv8 Logo Detection — Training API

REST API for fine-tuning YOLOv8 models on logo detection datasets from Roboflow with a Redis-backed job queue.

## Architecture

```
Client ──POST /train──▶ FastAPI ──enqueue──▶ Redis Queue ──▶ RQ Worker ──▶ YOLOTrainBackend
                           │                                                    │
                     GET /jobs/:id                                    download → train → evaluate
                     GET /queue
                     DELETE /jobs/:id
```

The system separates the API layer from training execution. When a training request comes in, the API validates the payload, enqueues a job into Redis, and immediately returns a `job_id`. A dedicated RQ worker picks jobs off the queue one at a time and runs the full pipeline (dataset download, training, evaluation). This means long-running training doesn't block the API, multiple requests get queued in order, and clients can poll for status at any time.

## Project Structure

```
├── finetune_main_api.py        # FastAPI application — routes and queue setup
├── schemas.py                  # Pydantic models — request/response validation
├── queue_worker.py             # RQ task — bridges API to backend
├── finetune_yolo_backend.py     # Core training backend — YOLO + Roboflow logic
└── requirements.txt             # Python dependencies
```

## Requirements

- Python 3.10+
- Redis server
- CUDA-capable GPU (recommended, falls back to CPU)

## Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Start Redis (if not already running as a service)
sudo systemctl enable redis
sudo systemctl start redis
```

## Running

Three processes need to be running:

```bash
# 1. Redis (should already be running as a systemd service)
sudo systemctl status redis

# 2. RQ Worker — executes training jobs from the queue
rq worker training --with-scheduler

# 3. API Server
uvicorn finetune_main_api:app --host 0.0.0.0 --port 1234 --reload
```

Interactive API docs are available at `http://localhost:1234/docs` once the server is running.

## API Endpoints

### `POST /train` — Submit a training job

Accepts a JSON body and enqueues the job. Returns immediately with a `job_id`.

```bash
curl -X POST http://localhost:1234/train \
  -H "Content-Type: application/json" \
  -d '{
    "roboflow": {
      "api_key": "YOUR_ROBOFLOW_API_KEY",
      "workspace": "jakapong-workspace",
      "project_name": "logo-detection-project-iihu1",
      "version": 1,
      "dataset_format": "yolov8"
    },
    "training": {
      "model": "yolov8m.pt",
      "epochs": 100,
      "img_size": 640,
      "batch_size": 4,
      "patience": 60,
      "optimizer": "AdamW",
      "lr0": 0.005,
      "scale": 0.4,
      "mosaic": 1.0,
      "mixup": 0.2,
      "copy_paste": 0.1,
      "plots": true,
      "cache": true
    },
    "paths": {
      "save_dataset_dir": "datasets",
      "save_model_dir": "runs/train"
    }
  }'
```

Only `roboflow.api_key` is required. Everything else has defaults, so the minimal request is:

```bash
curl -X POST http://localhost:1234/train \
  -H "Content-Type: application/json" \
  -d '{"roboflow": {"api_key": "YOUR_KEY"}}'
```

Response (`202 Accepted`):

```json
{
  "job_id": "a1b2c3d4-...",
  "status": "queued",
  "message": "Training job submitted to queue.",
  "position": 1
}
```

### `GET /jobs/{job_id}` — Check job status

```bash
curl http://localhost:1234/jobs/a1b2c3d4-...
```

Response:

```json
{
  "job_id": "a1b2c3d4-...",
  "status": "finished",
  "message": "Job finished",
  "created_at": "2026-04-21T10:00:00",
  "started_at": "2026-04-21T10:00:05",
  "ended_at": "2026-04-21T12:30:00",
  "result": {
    "save_model_dir": "runs/train-20260421-100005",
    "epochs": 100,
    "device": "cuda",
    "mAP50": 0.8923,
    "mAP50_95": 0.7145
  },
  "error": null
}
```

Status values: `queued`, `started`, `finished`, `failed`, `canceled`.

### `DELETE /jobs/{job_id}` — Cancel a job

Removes a queued job or sends a stop signal to a running job.

```bash
curl -X DELETE http://localhost:1234/jobs/a1b2c3d4-...
```

### `GET /queue` — View queue status

Returns counts and details for all jobs across all states.

```bash
curl http://localhost:1234/queue
```

### `GET /health` — Health check

Verifies Redis connectivity and reports queue size.

```bash
curl http://localhost:1234/health
```

## Request Body Reference

All fields under `training` and `paths` have defaults. Only `roboflow.api_key` is required.

| Section | Field | Type | Default | Description |
|---------|-------|------|---------|-------------|
| **roboflow** | `api_key` | string | — | Roboflow API key (required) |
| | `workspace` | string | `jakapong-workspace` | Roboflow workspace name |
| | `project_name` | string | `logo-detection-project-iihu1` | Roboflow project name |
| | `version` | int | `1` | Dataset version |
| | `dataset_format` | string | `yolov8` | Export format |
| **training** | `model` | string | `yolov8m.pt` | Pretrained model weights |
| | `epochs` | int | `100` | Training epochs |
| | `img_size` | int | `640` | Input image size |
| | `batch_size` | int | `4` | Batch size |
| | `patience` | int | `60` | Early stopping patience |
| | `optimizer` | string | `AdamW` | Optimizer |
| | `lr0` | float | `0.005` | Initial learning rate |
| | `scale` | float | `0.4` | Scale augmentation |
| | `mosaic` | float | `1.0` | Mosaic augmentation |
| | `mixup` | float | `0.2` | MixUp augmentation |
| | `copy_paste` | float | `0.1` | Copy-paste augmentation |
| | `plots` | bool | `true` | Save training plots |
| | `cache` | bool | `true` | Cache images in RAM |
| **paths** | `save_dataset_dir` | string | `datasets` | Dataset download directory |
| | `save_model_dir` | string | `runs/train` | Training output directory |

## Training Pipeline

Each job executes three steps in sequence:

1. **Download** — Fetches the dataset from Roboflow (skips if already present).
2. **Train** — Fine-tunes the YOLOv8 model with the specified hyperparameters. Weights and plots are saved to a timestamped directory under `save_model_dir`.
3. **Evaluate** — Runs validation and returns mAP50 and mAP50-95 metrics.

Results are stored in Redis for 7 days.

## Notes

- The queue processes one job at a time to avoid GPU contention. Jobs are executed in FIFO order.
- The API server and the worker can run on different machines as long as they share the same Redis instance and filesystem.
- If the worker is restarted while a job is running, that job will be marked as failed. Re-submit it via `POST /train`.
- GPU is auto-detected. If CUDA is available, training runs on 