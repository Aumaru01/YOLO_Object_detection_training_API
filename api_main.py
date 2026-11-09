"""
YOLOv8 Training API
===================
FastAPI application with an in-process job queue for managing training jobs.
See README.md for the full endpoint reference and dataset format docs.

Start API:    python -m api_main
"""

import json
import logging
import time
import uvicorn

from logging.handlers import RotatingFileHandler
from typing import Optional

from pathlib import Path

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles

from api_schemas import (
    TrainRequest,
    LocalTrainRequest,
    JobResponse,
    JobDetail,
    JobStatus,
    QueueInfo,
    PreprocessResult,
)
from train_queue import JobQueue, JobRecord, job_log_file, job_progress_file
from api_ui_routes import build_router
from train_backend import (
    BASE_DATASET_DIR,
    BASE_MODEL_DIR,
    preprocess_data,
    validate_dataset,
    _reject_windows_path,
)

# ---------------------------------------------------------------------------
# Logging — rotating file + console, shared with the worker (logs/api.log)
# ---------------------------------------------------------------------------
LOG_DIR = Path(__file__).resolve().parent / "logs"
LOG_DIR.mkdir(exist_ok=True)
LOG_FILE = LOG_DIR / "api.log"

_log_format = logging.Formatter(
    fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

_root = logging.getLogger()
_root.setLevel(logging.INFO)

# Avoid duplicate handlers on reload.
if not any(isinstance(h, RotatingFileHandler) for h in _root.handlers):
    _file_handler = RotatingFileHandler(
        LOG_FILE, maxBytes=5 * 1024 * 1024, backupCount=5, encoding="utf-8"
    )
    _file_handler.setFormatter(_log_format)
    _root.addHandler(_file_handler)

if not any(
    isinstance(h, logging.StreamHandler) and not isinstance(h, RotatingFileHandler)
    for h in _root.handlers
):
    _console = logging.StreamHandler()
    _console.setFormatter(_log_format)
    _root.addHandler(_console)

# Forward uvicorn / fastapi logs into the same handlers.
for _name in ("uvicorn", "uvicorn.error", "uvicorn.access", "fastapi"):
    _lg = logging.getLogger(_name)
    _lg.handlers.clear()
    _lg.propagate = True
    _lg.setLevel(logging.INFO)

logger = logging.getLogger("api_log")
logger.info("Logging initialised — file=%s", LOG_FILE)

# ---------------------------------------------------------------------------
# App & Job Queue
# ---------------------------------------------------------------------------
app = FastAPI(
    title="YOLOv8 Training API",
    description="Submit, monitor, and manage YOLOv8 fine-tuning jobs via an in-process job queue.",
    version="1.0.0",
)

job_queue = JobQueue()


def _busy_dataset_paths() -> list[str]:
    return [
        r.dataset_path for r in job_queue.all_jobs()
        if r.status in ("queued", "started") and r.dataset_path
    ]


app.include_router(build_router(_busy_dataset_paths))


@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log method, path, status, duration, and client IP for every request."""
    start = time.perf_counter()
    client = request.client.host if request.client else "-"
    try:
        response = await call_next(request)
    except Exception as exc:
        duration_ms = (time.perf_counter() - start) * 1000
        logger.exception(
            "HTTP %s %s -> 500 in %.1fms (client=%s) | unhandled: %s",
            request.method, request.url.path, duration_ms, client, exc,
        )
        raise
    duration_ms = (time.perf_counter() - start) * 1000
    logger.info(
        "HTTP %s %s -> %d in %.1fms (client=%s)",
        request.method, request.url.path, response.status_code, duration_ms, client,
    )
    return response


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
NAME_PATTERN = r"^[a-zA-Z0-9_\-]+$"


def _generate_job_name() -> str:
    return f"job_{time.strftime('%Y%m%d_%H%M%S')}"


def _job_to_detail(record: JobRecord) -> JobDetail:
    status = JobStatus(record.status)
    return JobDetail(
        job_id=record.job_id,
        job_name=record.job_name,
        status=status,
        message=f"Job {status.value}",
        created_at=record.created_at,
        started_at=record.started_at,
        ended_at=record.ended_at,
        result=record.result if status == JobStatus.FINISHED else None,
        error=record.error,
    )


def _fetch_job(job_id: str) -> JobRecord:
    """Fetch a job record, raising 404 if it doesn't exist."""
    record = job_queue.get(job_id)
    if record is None:
        logger.warning("Job lookup failed — id=%s not found.", job_id)
        raise HTTPException(
            status_code=404,
            detail=f"Job '{job_id}' not found (wrong id or never existed).",
        )
    return record


# ---------------------------------------------------------------------------
# Routes — Dataset preprocessing
# ---------------------------------------------------------------------------
@app.post("/preprocess_data", response_model=PreprocessResult)
def preprocess_dataset(
    raw_path: str = Query(..., description="Filesystem path to raw data — one source, or a folder of sources (see README)."),
    output_path: str = Query(..., description="Filesystem path to build/update the preprocessed dataset at."),
    val_ratio: float = Query(0.1, ge=0, le=1, description="Fraction of new images assigned to val, per source."),
    test_ratio: float = Query(0.0, ge=0, le=1, description="Fraction of new images assigned to test, per source."),
):
    """Build or incrementally update a YOLO dataset at output_path from raw_path."""
    try:
        return preprocess_data(raw_path, output_path, val_ratio=val_ratio, test_ratio=test_ratio)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


# ---------------------------------------------------------------------------
# Routes — Training
# ---------------------------------------------------------------------------
def _submit_job(
    job_name: str,
    training: dict,
    roboflow: Optional[dict],
    dataset_path: Optional[str] = None,
    output_path: Optional[str] = None,
) -> JobResponse:
    """Enqueue a job and build the submission response."""
    if job_queue.get(job_name) is not None:
        raise HTTPException(
            status_code=409,
            detail=f"Job id '{job_name}' already exists. Choose a different name.",
        )

    record = job_queue.enqueue(
        job_id=job_name,
        job_name=job_name,
        training=training,
        roboflow=roboflow,
        dataset_path=dataset_path,
        output_path=output_path,
    )

    position = job_queue.position(job_name) or job_queue.pending_count()
    logger.info("Job %s [%s] queued (position %d).", record.job_id, job_name, position)

    return JobResponse(
        job_id=record.job_id,
        job_name=job_name,
        status=JobStatus.QUEUED,
        message=f"Training job '{job_name}' submitted to queue.",
        position=position,
    )


@app.post("/train_roboflow_data", response_model=JobResponse, status_code=202)
def train_roboflow_data(
    request: TrainRequest,
    job_name: Optional[str] = Query(
        None, description="Job name; auto-generated from a timestamp if omitted.", pattern=NAME_PATTERN,
    ),
    output_path: Optional[str] = Query(
        None, description="Filesystem path under which a {job_name}/ folder is created to store "
                          "the trained model + evaluation results; defaults to models/{job_name}/.",
    ),
):
    """Submit a training job that downloads its dataset from Roboflow."""
    job_name = job_name or _generate_job_name()
    if output_path:
        try:
            _reject_windows_path(output_path, "output_path")
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))

    model_output_dir = Path(output_path, job_name) if output_path else (BASE_MODEL_DIR / job_name)
    if (BASE_DATASET_DIR / job_name).exists() or model_output_dir.exists():
        raise HTTPException(
            status_code=409,
            detail=f"Job name '{job_name}' already exists. Choose a different name.",
        )

    return _submit_job(
        job_name,
        training=request.training.model_dump(),
        roboflow=request.roboflow.model_dump(),
        output_path=output_path,
    )


@app.post("/train_local_data", response_model=JobResponse, status_code=202)
def train_local_data(
    request: LocalTrainRequest,
    job_name: str = Query(
        ..., description="Job name; used as the model output folder (models/{job_name}/).", pattern=NAME_PATTERN,
    ),
    dataset_path: str = Query(
        ..., description="Filesystem path to an already-preprocessed dataset, trained in place (see README).",
    ),
    output_path: Optional[str] = Query(
        None, description="Filesystem path under which a {job_name}/ folder is created to store "
                          "the trained model + evaluation results; defaults to models/{job_name}/.",
    ),
):
    """Submit a training job that trains on an already-local dataset (no download)."""
    dataset_dir = Path(dataset_path)
    try:
        validate_dataset(dataset_dir)
        if output_path:
            _reject_windows_path(output_path, "output_path")
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    model_output_dir = Path(output_path, job_name) if output_path else (BASE_MODEL_DIR / job_name)
    if model_output_dir.exists():
        raise HTTPException(
            status_code=409,
            detail=f"Job name '{job_name}' already has trained model output. Choose a different name.",
        )

    return _submit_job(
        job_name,
        training=request.training.model_dump(),
        roboflow=None,
        dataset_path=str(dataset_dir),
        output_path=output_path,
    )


# ---------------------------------------------------------------------------
# Routes — Job management
# ---------------------------------------------------------------------------
@app.get("/jobs/{job_id}", response_model=JobDetail)
def get_job_status(job_id: str):
    """Check the status of a specific job."""
    record = _fetch_job(job_id)
    detail = _job_to_detail(record)
    logger.info("Job %s [%s] -> %s", job_id, detail.job_name, detail.status.value)
    return detail


@app.delete("/jobs/{job_id}", response_model=JobResponse)
def cancel_job(job_id: str):
    """Cancel a queued or running job."""
    record = _fetch_job(job_id)
    status = record.status

    if status not in ("queued", "started"):
        raise HTTPException(status_code=400, detail=f"Job is already '{status}', cannot cancel.")

    job_queue.cancel(job_id)
    message = (
        "Job canceled (removed from queue)."
        if status == "queued"
        else "Job canceled (worker process terminated)."
    )
    logger.info("Job %s [%s] canceled (was %s).", job_id, record.job_name, status)

    return JobResponse(
        job_id=job_id,
        job_name=record.job_name,
        status=JobStatus.CANCELED,
        message=message,
    )


@app.get("/jobs/{job_id}/monitor")
def get_job_monitor(job_id: str):
    """Everything the UI's training monitor needs for one job: status, process
    info, live training progress (epoch, batch, losses, metric history), and
    the current size of its log (fetch text via GET /jobs/{job_id}/logs)."""
    record = _fetch_job(job_id)
    detail = _job_to_detail(record).model_dump(mode="json")

    progress = None
    progress_path = job_progress_file(job_id)
    if progress_path.exists():
        try:
            progress = json.loads(progress_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            progress = None  # mid-write; the next poll picks it up

    process = record.process
    log_path = job_log_file(job_id)
    detail.update(
        source="roboflow" if record.roboflow else "local",
        training=record.training,
        dataset_path=record.dataset_path,
        output_path=record.output_path,
        position=job_queue.position(job_id),
        process={
            "pid": record.pid,
            "alive": bool(process is not None and process.is_alive()),
            "exitcode": record.exitcode,
        },
        progress=progress,
        log_size=log_path.stat().st_size if log_path.exists() else 0,
    )
    return detail


@app.get("/jobs/{job_id}/logs")
def get_job_logs(
    job_id: str,
    offset: int = Query(-1, description="Byte offset to read from; -1 = the last max_bytes of the log."),
    max_bytes: int = Query(256_000, ge=1, le=5_000_000),
):
    """Incrementally read a job's worker output (stdout/stderr). Poll with the
    returned next_offset to tail it."""
    _fetch_job(job_id)
    path = job_log_file(job_id)
    if not path.exists():
        return {"offset": 0, "next_offset": 0, "size": 0, "text": ""}
    size = path.stat().st_size
    start = max(0, size - max_bytes) if offset < 0 else min(offset, size)
    with open(path, "rb") as f:
        f.seek(start)
        data = f.read(max_bytes)
    return {
        "offset": start,
        "next_offset": start + len(data),
        "size": size,
        "text": data.decode("utf-8", errors="replace"),
    }


@app.get("/queue", response_model=QueueInfo)
def get_queue_info():
    """Get current queue status and all jobs."""
    records = job_queue.all_jobs()
    counts = {"queued": 0, "started": 0, "finished": 0, "failed": 0}
    for record in records:
        if record.status in counts:
            counts[record.status] += 1

    return QueueInfo(jobs=[_job_to_detail(r) for r in records], **counts)


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------
@app.get("/health")
def health_check():
    """Health check — reports the in-process job queue status."""
    return {"status": "healthy", "worker": "running", "queue_size": job_queue.pending_count()}


# ---------------------------------------------------------------------------
# Web UI (static single-page app)
# ---------------------------------------------------------------------------
STATIC_DIR = Path(__file__).resolve().parent / "static"
app.mount("/ui", StaticFiles(directory=STATIC_DIR, html=True), name="ui")


@app.get("/", include_in_schema=False)
def root():
    return RedirectResponse(url="/ui/")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    uvicorn.run(app, host="0.0.0.0", port=1234)
