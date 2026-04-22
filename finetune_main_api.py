"""
YOLOv8 Training API
===================
FastAPI application with Redis Queue for managing training jobs.

Start API:    python -m finetune_main_api
Start Worker: rq worker training --with-scheduler
"""

import logging
import time
import zipfile
import uvicorn

from io import BytesIO
from pathlib import Path

from rq import Queue
from rq.job import Job
from redis import Redis
from fastapi import FastAPI, HTTPException
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
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("api_log")

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
def submit_training(request: TrainRequest):
    """Submit a new training job to the queue.

    ``job_name`` is used as the folder name for dataset & model.
    If not provided, a timestamp-based name is generated.

    ```json
    {
        "roboflow": { "api_key": "...", ... },
        "training": { "epochs": 100, "lr0": 0.005, ... },
        "job_name": "my_logo_v1"
    }
    ```
    """
    job_name = request.job_name or _generate_job_name()

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
    try:
        job = Job.fetch(job_id, connection=redis_conn)
    except Exception:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found.")

    return _job_to_detail(job)


@app.delete("/jobs/{job_id}", response_model=JobResponse)
def cancel_job(job_id: str):
    """Cancel a queued or running job."""
    try:
        job = Job.fetch(job_id, connection=redis_conn)
    except Exception:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found.")

    job_name = job.meta.get("job_name", "unknown")
    status = job.get_status()

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
        send_stop_job_command(redis_conn, job_id)
        logger.info("Job %s [%s] stop signal sent (was running).", job_id, job_name)
        return JobResponse(
            job_id=job_id,
            job_name=job_name,
            status=JobStatus.CANCELED,
            message="Stop signal sent to running job.",
        )
    else:
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
        except Exception:
            pass

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
            except Exception:
                pass

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
