# YOLOv8 Logo Detection — Training API

REST API for fine-tuning YOLOv8 models on logo detection datasets, backed by an in-process job queue (no external broker — no Redis, no separate worker service). Datasets come from Roboflow, or from your own raw label exports built into a trainable dataset via `POST /preprocess_data`.

## Contents

- [Architecture](#architecture)
- [Project Structure](#project-structure)
- [Setup](#setup)
- [Running](#running)
- [API Endpoints](#api-endpoints)
- [Request Body Reference](#request-body-reference)
- [Dataset Formats](#dataset-formats)
- [Training Pipeline](#training-pipeline)

## Architecture

```
                     POST /train_roboflow_data ──┐
                                                  ├─▶ FastAPI ──enqueue──▶ JobQueue ──▶ worker subprocess ──▶ YOLOTrainBackend
                     POST /train_local_data ─────┘                                                                │
                                           │                                                          [download] → train → evaluate
                                     GET /jobs/:id                                                                 │
                                     GET /queue                                                           models/{job_name}/
                                     DELETE /jobs/:id
                                     POST /preprocess_data ──▶ builds/updates a dataset at output_path from raw_path
```

The API layer is separate from training execution. When a training request comes in, the API validates the payload, enqueues a job in memory, and immediately returns a `job_id` and `job_name`. A background dispatcher thread (started with the API — nothing extra to run) picks jobs off the queue one at a time and runs the pipeline in its own worker subprocess, so a running job can still be canceled by terminating it.

- `/train_roboflow_data` downloads the dataset into `datasets/{job_name}/` first.
- `/train_local_data` trains **in place** on an already-preprocessed dataset at any filesystem path — nothing is copied.
- Model weights and evaluation results go to `models/{job_name}/` by default, or to `output_path` if given.

> **Note:** job state lives in memory for the lifetime of the API process — restarting the API clears the job list. Datasets and models already written to disk are unaffected.

## Project Structure

```
├── finetune_main_api.py        # FastAPI application — routes and queue setup
├── schemas.py                  # Pydantic models — request/response validation
├── queue_worker.py             # In-process JobQueue — dispatches jobs to worker subprocesses
├── finetune_yolo_backend.py    # Core training backend — YOLO + Roboflow logic + dataset preprocessing
└── requirements.txt            # Python dependencies
```

Output directories (created automatically):

```
├── datasets/{job_name}/        # Roboflow-downloaded dataset for each job
└── models/{job_name}/          # Trained model weights & plots for each job
```

Raw data and preprocessed (output) datasets live wherever you point `raw_path` / `output_path` / `dataset_path` — neither has to be under `datasets/`.

## Setup

Requirements: Python 3.10+, a CUDA-capable GPU (recommended — falls back to CPU).

```bash
pip install -r requirements.txt
```

## Running

```bash
python -m finetune_main_api

# Tail the rotating log file (API + worker share it):
tail -f logs/api.log
```

The job queue and its worker start automatically with the API — there's no separate worker process or service to install or run. Interactive API docs are available at `http://localhost:1234/docs` once the API is running.

## API Endpoints

### `POST /train_roboflow_data` — Submit a training job (dataset from Roboflow)

Accepts a JSON body and enqueues the job. `job_name` is sent as a query parameter. Returns immediately with a `job_id` and `job_name`; the worker downloads the dataset from Roboflow before training.

```bash
curl -X POST "http://localhost:1234/train_roboflow_data?job_name=my_logo_v1" \
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

Only `roboflow.api_key` is required. If `job_name` is omitted, a timestamp-based name is generated automatically (e.g. `job_20260421_143000`). Pass `output_path` (query param) to store the trained model + evaluation results somewhere other than `models/{job_name}/`.

Response (`202 Accepted`):

```json
{
  "job_id": "my_logo_v1",
  "job_name": "my_logo_v1",
  "status": "queued",
  "message": "Training job 'my_logo_v1' submitted to queue.",
  "position": 1
}
```

> `job_id` is the same as `job_name` — uniqueness is enforced, so you can use the human-readable name in every endpoint below.

### `POST /train_local_data` — Submit a training job (already-preprocessed dataset)

Trains **in place** on a dataset built by `POST /preprocess_data` (or any directory with a standard `data.yaml` — see [Dataset Formats](#dataset-formats)) — no download, no copying. `job_name`, `dataset_path`, and `output_path` are all query parameters.

```bash
curl -X POST "http://localhost:1234/train_local_data?job_name=my_logo_v1&dataset_path=/data/my_dataset&output_path=/data/my_model" \
  -H "Content-Type: application/json" \
  -d '{
    "training": {
      "epochs": 100,
      "batch_size": 4,
      "device": "cuda"
    }
  }'
```

`dataset_path` points at the preprocessed dataset to train on. `output_path` is optional — where to store the trained model + evaluation results; defaults to `models/{job_name}/` if omitted. The dataset is validated before the job is queued: `404` if `dataset_path` has no `data.yaml`, `400` if it (or `output_path`) is malformed, `409` if a trained model already exists at the resolved output location. Response shape is the same `202 Accepted` job response as above.

### `GET /jobs/{job_id}` — Check job status

```bash
curl http://localhost:1234/jobs/my_logo_v1
```

Response:

```json
{
  "job_id": "my_logo_v1",
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

`test_mAP50` / `test_mAP50_95` are added to `result` automatically when the dataset's `data.yaml` defines a `test` split — see [Training Pipeline](#training-pipeline).

Status values: `queued`, `started`, `finished`, `failed`, `canceled`.

### `DELETE /jobs/{job_id}` — Cancel a job

Only cancels a job the API still has in memory and that hasn't finished yet (`queued` or `started`) — `400` if it's already `finished`/`failed`/`canceled`, `404` if the job isn't tracked (never submitted, or the API restarted since).

```bash
curl -X DELETE http://localhost:1234/jobs/my_logo_v1
```

### `GET /queue` — View queue status

```bash
curl http://localhost:1234/queue
```

### `POST /preprocess_data` — Build or update a dataset from raw label exports

Builds (or incrementally updates) a trainable dataset at `output_path` from raw label exports at `raw_path`. This is the step that turns your labeling-tool output into something `/train_local_data` can consume — see [Dataset Formats](#dataset-formats) for the expected `raw_path` layout and exactly how the incremental behavior works.

```bash
curl -X POST -G "http://localhost:1234/preprocess_data" \
  --data-urlencode "raw_path=/data/raw_labels" \
  --data-urlencode "output_path=/data/my_dataset" \
  --data-urlencode "val_ratio=0.2" \
  --data-urlencode "test_ratio=0.1"
```

Response:

```json
{
  "data_yaml": "/data/my_dataset/data.yaml",
  "root": "/data/my_dataset",
  "num_classes": 3,
  "class_names": ["cat", "dog", "bird"],
  "splits": {
    "train": {"image_dir": "...", "label_dir": "...", "num_images": 70, "num_missing_labels": 0, "missing_labels_sample": []},
    "val":   {"image_dir": "...", "label_dir": "...", "num_images": 20, "num_missing_labels": 0, "missing_labels_sample": []},
    "test":  {"image_dir": "...", "label_dir": "...", "num_images": 10, "num_missing_labels": 0, "missing_labels_sample": []}
  },
  "added": {
    "sources": ["label3"],
    "train": 8, "val": 2, "test": 0,
    "new_classes": ["bird"],
    "skipped_no_label": []
  }
}
```

`added` describes only what changed in *this* call — which source folders had new images, how many went to each split, any newly-introduced class names, and any images that had no matching label file (and so weren't included). Calling this again with the same `raw_path`/`output_path` and nothing new added is a no-op (`added` comes back all-empty/zero).

`404` if `raw_path` doesn't exist or contains no recognizable source; `400` on validation errors (e.g. `val_ratio + test_ratio >= 1`, or a label referencing a class index with no matching name).

### `GET /health` — Health check

```bash
curl http://localhost:1234/health
```

## Request Body Reference

`roboflow` is only part of the `/train_roboflow_data` body (only `roboflow.api_key` is required there); `/train_local_data` accepts `training` only. Everything in `training` has defaults.

| Section | Field | Type | Default | Description |
|---------|-------|------|---------|-------------|
| **roboflow** (`/train_roboflow_data` only) | `api_key` | string | — | Roboflow API key (required) |
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
| — | `job_name` | query param | auto-generated for `/train_roboflow_data`, required for `/train_local_data` | Model output folder name (used when `output_path` is omitted) |
| — | `dataset_path` | query param | required for `/train_local_data` only | Filesystem path to a preprocessed dataset (see [Dataset Formats](#dataset-formats)) |
| — | `output_path` | query param | optional for both training endpoints | Filesystem path to store the trained model + evaluation results; defaults to `models/{job_name}/` |

## Dataset Formats

There are two dataset shapes in this system: the **raw** layout you hand to `POST /preprocess_data`, and the **standard** layout it produces (and that `/train_local_data` trains on).

### Raw layout (input to `/preprocess_data`)

`raw_path` is either:
- **a single source** — one `images/` folder plus a `labels/` or `label/` folder, directly under `raw_path`, or
- **a folder of sources** — several of the above as immediate subdirectories, e.g.:
  ```
  raw_path/
    label1/images/, label1/label/, label1/classes.txt, label1/notes.json
    label2/images/, label2/label/, label2/classes.txt, label2/notes.json
    ...
  ```

This is the layout produced by many labeling tools (e.g. LabelImg-style exports). Each source needs its own class names, found in priority order (each searched recursively within that source): a `data.yaml`'s `names`, a `classes.txt` (one name per line, index = line number), or a `notes.json`'s `categories` key (a plain list of name strings, or COCO-style `{"id": int, "name": str}` objects sorted by `id`).

**Merging multiple sources.** Class names are unioned across sources into one list (first-seen order); a source's own label indices are remapped to match. So `label1` with `[cat, dog]` and `label2` with `[dog, bird]` produce a merged `[cat, dog, bird]`, with `label2`'s labels rewritten so `dog`/`bird` point at the right indices.

**Splitting.** Each source's images are split into train/val/test independently (`val_ratio`/`test_ratio` fractions, remainder to train) — so a small source can't be accidentally swept entirely into one split by chance, the way one global shuffle might.

**Incremental re-runs.** `output_path` tracks which raw images it has already incorporated (a small `.preprocess_state.json` file). Re-running `/preprocess_data` with the same `raw_path`/`output_path` — after adding a new source folder, or new images to an existing one — only processes what's new: existing train/val/test assignments are left untouched, and only the new images get split in (per-source, as above) and appended. Nothing under `raw_path` is ever modified; images are **symlinked** into `output_path` where possible (not copied) — falling back to real copies if `output_path`'s filesystem doesn't support symlinks (e.g. many CIFS/SMB mounts) — and only their remapped label files are otherwise written there.

### Standard layout (output of `/preprocess_data`, input to `/train_local_data`)

A `data.yaml` (searched for directly at the given path, then recursively in subdirectories) with `names` (a list, or a `{index: name}` dict) and a `train` split (`val`/`test` optional). Each split value can be a directory (walked recursively for images) or a manifest `.txt` file listing image paths one per line. Label files are located the same way ultralytics does — the last `/images/` segment in an image's path is swapped for `/labels/`, then the extension becomes `.txt`. This is exactly what `/preprocess_data` builds at `output_path` (`output_path/{train,val,test}/images/` + `.../labels/`).

> **Caveat:** `YOLOTrainBackend.cleanup_dataset()` deletes `dataset_dir` outright. It isn't called anywhere today, but if it's ever wired up, calling it on a preprocessed `output_path` would delete the symlink farm (harmless — raw images are untouched) but also the state file, losing incremental-update history.

## Training Pipeline

`/train_roboflow_data` jobs execute three steps in sequence; `/train_local_data` jobs skip the download step:

1. **Download** (roboflow only) — Fetches the dataset from Roboflow into `datasets/{job_name}/`.
2. **Train** — Fine-tunes the YOLOv8 model. Weights are saved to `{output_path}/train/weights/` (`best.pt`, `last.pt`).
3. **Evaluate** — Runs validation against the `val` split, saving plots/results to `{output_path}/val/`, and returns `mAP50`/`mAP50_95`. If the dataset's `data.yaml` also defines a `test` split, a second pass runs against it, saving to `{output_path}/test/` and adding `test_mAP50`/`test_mAP50_95` to the result.

`{output_path}` is whatever you passed as `output_path`, or `models/{job_name}/` if you didn't. Job results and status are kept in memory for the lifetime of the API process; trained models and evaluation results persist on disk under `{output_path}`. Datasets persist wherever they live — `datasets/{job_name}/` for Roboflow downloads, or whatever `dataset_path` was given for local jobs.
