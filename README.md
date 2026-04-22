# YOLOv8 Logo Detection — Training API

REST API for fine-tuning YOLOv8 models on logo detection datasets from Roboflow with a Redis-backed job queue.

## Architecture

```
Client ──POST /train──▶ FastAPI ──enqueue──▶ Redis Queue ──▶ RQ Worker ──▶ YOLOTrainBackend
                           │                                                    │
                     GET /jobs/:id                                    download → train → evaluate
                     GET /queue                                                 │
                     DELETE /jobs/:id                                  datasets/{job_name}/
                     GET /download_dataset/:name                      models/{job_name}/
                     GET /download_model/:name
```

The system separates the API layer from training execution. When a training request comes in, the API validates the payload, enqueues a job into Redis, and immediately returns a `job_id` and `job_name`. A dedicated RQ worker picks jobs off the queue one at a time and runs the full pipeline (dataset download, training, evaluation). All outputs are stored under the `job_name` so they can be downloaded later.

## Project Structure

```
├── finetune_main_api.py        # FastAPI application — routes and queue setup
├── schemas.py                  # Pydantic models — request/response validation
├── queue_worker.py             # RQ task — bridges API to backend
├── finetune_yolo_backend.py    # Core training backend — YOLO + Roboflow logic
├── yolo-api.service            # systemd service — API server
├── yolo-worker.service         # systemd service — RQ worker
└── requirements.txt            # Python dependencies
```

Output directories (created automatically):

```
├── datasets/{job_name}/        # Downloaded dataset for each job
└── models/{job_name}/          # Trained model weights & plots for each job
```

## Requirements

- Python 3.10+
- Redis server
- CUDA-capable GPU (recommended, falls back to CPU)

## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Start Redis

```bash
sudo systemctl enable redis
sudo systemctl start redis
```

### 3. Install services

Create the required service files by using the provided examples as references:
  - yolo-api.service.example
  - yolo-worker.service.example
  > **Note:** You can duplicate and modify these example files to match your system configuration.

Copy the service files to systemd and enable them:

```bash
sudo cp yolo-worker.service /etc/systemd/system/
sudo cp yolo-api.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable yolo-worker yolo-api
```

> **Note:** Before enabling, edit the `.service` files if your paths or username differ from the defaults.
> Check `WorkingDirectory` and `ExecStart` paths match your environment.

## Running

```bash
# Start both services
sudo systemctl start yolo-worker yolo-api

# Check status
sudo systemctl status yolo-worker yolo-api

# View logs
journalctl -u yolo-worker -f    # worker logs
journalctl -u yolo-api -f       # API logs

# Stop
sudo systemctl stop yolo-worker yolo-api

# Restart (e.g. after code changes)
sudo systemctl restart yolo-worker yolo-api
```

Interactive API docs are available at `http://localhost:1234/docs` once the API service is running.

## API Endpoints

### `POST /train` — Submit a training job

Accepts a JSON body and enqueues the job. `job_name` is sent as a query parameter. Returns immediately with a `job_id` and `job_name`.

```bash
curl -X POST "http://localhost:1234/train?job_name=my_logo_v1" \
  -H "Content-Type: application/json" \
  -d '{
    "roboflow": {
      "api_key": "YOUR_ROBOFLOW_API_KEY"
    },
    "training": {
      "epochs": 100,
      "batch_size": 4,
      "device": "cuda"
    }
  }'
```

Only `roboflow.api_key` is required. If `job_name` is omitted, a timestamp-based name is generated automatically (e.g. `job_20260421_143000`).

Response (`202 Accepted`):

```json
{
  "job_id": "a1b2c3d4-...",
  "job_name": "my_logo_v1",
  "status": "queued",
  "message": "Training job 'my_logo_v1' submitted to queue.",
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
  "job_name": "my_logo_v1",
  "status": "finished",
  "message": "Job finished",
  "created_at": "2026-04-21T10:00:00",
  "started_at": "2026-04-21T10:00:05",
  "ended_at": "2026-04-21T12:30:00",
  "result": {
    "job_name": "my_logo_v1",
    "model_dir": "models/my_logo_v1",
    "dataset_dir": "datasets/my_logo_v1",
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

```bash
curl -X DELETE http://localhost:1234/jobs/a1b2c3d4-...
```

### `GET /queue` — View queue status

```bash
curl http://localhost:1234/queue
```

### `GET /download_dataset/{job_name}` — Download dataset

Downloads the dataset folder as a zip file.

```bash
curl -OJ http://localhost:1234/download_dataset/my_logo_v1
# → my_logo_v1_dataset.zip
```

### `GET /download_model/{job_name}` — Download trained model

Downloads the model folder (weights, plots, etc.) as a zip file.

```bash
curl -OJ http://localhost:1234/download_model/my_logo_v1
# → my_logo_v1_model.zip
```

### `GET /health` — Health check

```bash
curl http://localhost:1234/health
```

## Request Body Reference

Only `roboflow.api_key` is required. Everything else has defaults.

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
| | `device` | string | `cpu` | Device: `cpu`, `cuda`, `cuda:0`, `cuda:1` |
| — | `job_name` | query param | auto-generated | Folder name for dataset & model outputs |

## Training Pipeline

Each job executes three steps in sequence:

1. **Download** — Fetches the dataset from Roboflow into `datasets/{job_name}/`.
2. **Train** — Fine-tunes the YOLOv8 model. Weights and plots are saved to `models/{job_name}/`.
3. **Evaluate** — Runs validation and returns mAP50 and mAP50-95 metrics.

Results are stored in Redis for 7 days. Files on disk persist until manually deleted.

## Notes

- The queue processes one job at a time to avoid GPU contention. Jobs are executed in FIFO order.
- Job names must be unique. Submitting a duplicate name returns `409 Conflict`.
- Job names may only contain letters, numbers, underscores, and hyphens.
- If the requested device is unavailable (e.g. `cuda` but no GPU), the backend logs a warning and falls back to CPU.
- Both services auto-restart on failure. After a reboot, they start automatically if enabled.
