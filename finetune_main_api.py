"""
YOLOv8 Training API
===================
FastAPI application with Redis Queue for managing training jobs.

Start API:    python -m finetune_main_api
Start Worker: rq worker training --with-scheduler
"""

import logging
import time
import traceback
import zipfile
import uvicorn

from logging.handlers import RotatingFileHandler
from typing import Optional

from io import BytesIO
from pathlib import Path

from rq import Queue
from rq.job import Job
from rq.exceptions import NoSuchJobError
from redis import Redis
from redis.exceptions import RedisError
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import StreamingResponse
from rq.command import send_stop_job_command

from schemas import (
    TrainRequest,
    JobResponse,
    JobDetail,
    JobStatus,
    QueueInfo,
)
from queue_worker import run_training_job
from finetune_yolo_backend import BASE_DATASET_DIR, BASE_MODEL_DIR

# ---------------------------------------------------------------------------
# Logging — rotating file + console, shared across the app
# ---------------------------------------------------------------------------
LOG_DIR = Path(__file__).resolve().parent / "logs"
LOG_DIR.mkdir(exist_ok=True)
LOG_FILE = LOG_DIR / "api.log"

_log_format = logging.Formatter(
    fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

# Root logger gets both console and rotating file handlers.
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
for _name in ("uvicorn", "uvicorn.error", "uvicorn.access", "fastapi", "rq.worker"):
    _lg = logging.getLogger(_name)
    _lg.handlers.clear()
    _lg.propagate = True
    _lg.setLevel(logging.INFO)

logger = logging.getLogger("api_log")
logger.info("Logging initialised — file=%s", LOG_FILE)

# ---------------------------------------------------------------------------
# App & Redis Queue
# ---------------------------------------------------------------------------
app = FastAPI(
    title="YOLOv8 Training API",
    description="Submit, monitor, and manage YOLOv8 fine-tuning jobs via Redis Queue.",
    version="1.0.0",
)

redis_conn = Redis(host="localhost", port=6379, db=0)
task_queue = Queue("training", connection=redis_conn, default_timeout=86400)  # 24h timeout


# ---------------------------------------------------------------------------
# Middleware — log every HTTP request
# ---------------------------------------------------------------------------
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
def _generate_job_name() -> str:
    """Generate a fallback job name from timestamp."""
    return f"job_{time.strftime('%Y%m%d_%H%M%S')}"


def _map_rq_status(rq_status: str) -> JobStatus:
    """Map rq job status string to our JobStatus enum."""
    mapping = {
        "queued": JobStatus.QUEUED,
        "started": JobStatus.STARTED,
        "finished": JobStatus.FINISHED,
        "failed": JobStatus.FAILED,
        "stopped": JobStatus.CANCELED,
        "canceled": JobStatus.CANCELED,
    }
    return mapping.get(rq_status, JobStatus.QUEUED)


def _job_to_detail(job: Job) -> JobDetail:
    """Convert an rq Job to our JobDetail schema."""
    status = _map_rq_status(job.get_status())

    error_msg = None
    if status == JobStatus.FAILED and job.exc_info:
        error_msg = str(job.exc_info)

    return JobDetail(
        job_id=job.id,
        job_name=job.meta.get("job_name"),
        status=status,
        message=f"Job {status.value}",
        created_at=job.enqueued_at,
        started_at=job.started_at,
        ended_at=job.ended_at,
        result=job.result if status == JobStatus.FINISHED else None,
        error=error_msg,
    )


def _fetch_job(job_id: str) -> Job:
    """Fetch a job from Redis, with precise error handling + logging.

    Raises
    ------
    HTTPException(404) — if the job genuinely doesn't exist.
    HTTPException(503) — if Redis is unreachable.
    HTTPException(500) — for any other unexpected failure (e.g. unpickling).
    """
    try:
        return Job.fetch(job_id, connection=redis_conn)
    except NoSuchJobError:
        logger.warning("Job lookup failed — id=%s not in Redis.", job_id)
        raise HTTPException(
            status_code=404,
            detail=f"Job '{job_id}' not found (expired, wrong id, or never existed).",
        )
    except RedisError as exc:
        logger.error("Redis error while fetching job %s: %s", job_id, exc)
        raise HTTPException(status_code=503, detail=f"Redis unavailable: {exc}")
    except Exception as exc:
        # Often a serializer / unpickling mismatch between API and worker.
        logger.error(
            "Unexpected error fetching job %s: %s\n%s",
            job_id, exc, traceback.format_exc(),
        )
        raise HTTPException(
            status_code=500,
            detail=f"Failed to load job '{job_id}': {type(exc).__name__}: {exc}",
        )


def _zip_directory(directory: Path) -> BytesIO:
    """Zip a directory into an in-memory BytesIO buffer."""
    buffer = BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        for file_path in directory.rglob("*"):
            if file_path.is_file():
                zf.write(file_path, file_path.relative_to(directory))
    buffer.seek(0)
    return buffer


# ---------------------------------------------------------------------------
# Routes — Training
# ---------------------------------------------------------------------------
@app.post("/train", response_model=JobResponse, status_code=202)
def submit_training(
    request: TrainRequest,
    job_name: Optional[str] = Query(
        None,
        description="Name for this job — used as folder name for dataset & model. "
                    "Auto-generated from timestamp if not provided.",
        pattern=r"^[a-zA-Z0-9_\-]+$",
    ),
):
    """Submit a new training job to the queue.

    ``job_name`` is sent as a query parameter:

        POST /train?job_name=my_logo_v1

    If not provided, a timestamp-based name is generated.
    """
    job_name = job_name or _generate_job_name()

    # Check if job_name already exists
    dataset_exists = (BASE_DATASET_DIR / job_name).exists()
    model_exists = (BASE_MODEL_DIR / job_name).exists()
    if dataset_exists or model_exists:
        raise HTTPException(
            status_code=409,
            detail=f"Job name '{job_name}' already exists. Choose a different name.",
        )

    job = task_queue.enqueue(
        run_training_job,
        roboflow=request.roboflow.model_dump(),
        training=request.training.model_dump(),
        job_name=job_name,
        job_timeout=86400,
        result_ttl=604800,    # keep result 7 days
        failure_ttl=604800,
        meta={"job_name": job_name},
    )

    position = len(task_queue)
    logger.info("Job %s [%s] queued (position %d).", job.id, job_name, position)

    return JobResponse(
        job_id=job.id,
        job_name=job_name,
        status=JobStatus.QUEUED,
        message=f"Training job '{job_name}' submitted to queue.",
        position=position,
    )


# ---------------------------------------------------------------------------
# Routes — Job management
# ---------------------------------------------------------------------------
@app.get("/jobs/{job_id}", response_model=JobDetail)
def get_job_status(job_id: str):
    """Check the status of a specific job."""
    logger.info("Fetching job status — id=%s", job_id)
    job = _fetch_job(job_id)
    detail = _job_to_detail(job)
    logger.info("Job %s [%s] -> %s", job_id, detail.job_name, detail.status.value)
    return detail


@app.delete("/jobs/{job_id}", response_model=JobResponse)
def cancel_job(job_id: str):
    """Cancel a queued or running job."""
    logger.info("Cancel request — id=%s", job_id)
    job = _fetch_job(job_id)

    job_name = job.meta.get("job_name", "unknown")
    status = job.get_status()
    logger.info("Job %s [%s] current status=%s", job_id, job_name, status)

    if status == "queued":
        job.cancel()
        logger.info("Job %s [%s] canceled (was queued).", job_id, job_name)
        return JobResponse(
            job_id=job_id,
            job_name=job_name,
            status=JobStatus.CANCELED,
            message="Job canceled (removed from queue).",
        )
    elif status == "started":
        try:
            send_stop_job_command(redis_conn, job_id)
        except Exception as exc:
            logger.error("Failed to send stop signal for %s: %s", job_id, exc)
            raise HTTPException(
                status_code=500,
                detail=f"Failed to send stop signal: {exc}",
            )
        logger.info("Job %s [%s] stop signal sent (was running).", job_id, job_name)
        return JobResponse(
            job_id=job_id,
            job_name=job_name,
            status=JobStatus.CANCELED,
            message="Stop signal sent to running job.",
        )
    else:
        logger.info("Job %s [%s] cannot be canceled (status=%s).", job_id, job_name, status)
        raise HTTPException(
            status_code=400,
            detail=f"Job is already '{status}', cannot cancel.",
        )


@app.get("/queue", response_model=QueueInfo)
def get_queue_info():
    """Get current queue status and all jobs."""
    job_details: list[JobDetail] = []
    counts = {"queued": 0, "started": 0, "finished": 0, "failed": 0}

    for job_id in task_queue.job_ids:
        try:
            job = Job.fetch(job_id, connection=redis_conn)
            job_details.append(_job_to_detail(job))
            counts["queued"] += 1
        except NoSuchJobError:
            logger.warning("Queued job id %s missing in Redis (skipped).", job_id)
        except Exception as exc:
            logger.error("Failed to load queued job %s: %s", job_id, exc)

    for registry, status_key in [
        (task_queue.started_job_registry, "started"),
        (task_queue.finished_job_registry, "finished"),
        (task_queue.failed_job_registry, "failed"),
    ]:
        for job_id in registry.get_job_ids():
            try:
                job = Job.fetch(job_id, connection=redis_conn)
                job_details.append(_job_to_detail(job))
                counts[status_key] += 1
            except NoSuchJobError:
                logger.warning(
                    "%s registry references missing job %s (skipped).",
                    status_key, job_id,
                )
            except Exception as exc:
                logger.error(
                    "Failed to load %s job %s: %s", status_key, job_id, exc,
                )

    return QueueInfo(
        queued=counts["queued"],
        started=counts["started"],
        finished=counts["finished"],
        failed=counts["failed"],
        jobs=job_details,
    )


# ---------------------------------------------------------------------------
# Routes — Download
# ---------------------------------------------------------------------------
@app.get("/download_dataset/{job_name}")
def download_dataset(job_name: str):
    """Download the dataset folder for a job as a zip file."""
    dataset_path = BASE_DATASET_DIR / job_name
    if not dataset_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Dataset for job '{job_name}' not found.",
        )

    logger.info("Downloading dataset for job '%s'.", job_name)
    buffer = _zip_directory(dataset_path)

    return StreamingResponse(
        buffer,
        media_type="application/zip",
        headers={"Content-Disposition": f"attachment; filename={job_name}_dataset.zip"},
    )


@app.get("/download_model/{job_name}")
def download_model(job_name: str):
    """Download the trained model folder for a job as a zip file."""
    model_path = BASE_MODEL_DIR / job_name
    if not model_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Model for job '{job_name}' not found.",
        )

    logger.info("Downloading model for job '%s'.", job_name)
    buffer = _zip_directory(model_path)

    return StreamingResponse(
        buffer,
        media_type="application/zip",
        headers={"Content-Disposition": f"attachment; filename={job_name}_model.zip"},
    )


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------
@app.get("/health")
def health_check():
    """Health check — verifies Redis connection."""
    try:
        redis_conn.ping()
        return {"status": "healthy", "redis": "connected", "queue_size": len(task_queue)}
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Redis unavailable: {e}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    uvicorn.run(app, host="0.0.0.0", port=1234)
