"""
YOLOv8 Training API
===================
FastAPI application with Redis Queue for managing training jobs.

Start API:    uvicorn api:app --host 0.0.0.0 --port 8000 --reload
Start Worker: rq worker training --with-scheduler
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Optional

from fastapi import FastAPI, HTTPException
from redis import Redis
from rq import Queue
from rq.job import Job
from rq.command import send_stop_job_command

from schemas import (
    TrainRequest,
    JobResponse,
    JobDetail,
    JobStatus,
    QueueInfo,
)
from queue_worker import run_training_job

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("api")

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
        status=status,
        message=f"Job {status.value}",
        created_at=job.enqueued_at,
        started_at=job.started_at,
        ended_at=job.ended_at,
        result=job.result if status == JobStatus.FINISHED else None,
        error=error_msg,
    )


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.post("/train", response_model=JobResponse, status_code=202)
def submit_training(request: TrainRequest):
    """Submit a new training job to the queue.

    Request body มี structure เดียวกับ config.yaml เดิม:

    ```json
    {
        "roboflow": { "api_key": "...", "workspace": "...", ... },
        "training": { "epochs": 100, "lr0": 0.005, ... },
        "paths":    { "dataset_dir": "datasets", "save_dir": "runs/train" }
    }
    ```
    """
    job = task_queue.enqueue(
        run_training_job,
        roboflow=request.roboflow.model_dump(),
        training=request.training.model_dump(),
        paths=request.paths.model_dump(),
        job_timeout=86400,
        result_ttl=604800,    # keep result 7 days
        failure_ttl=604800,
    )

    position = len(task_queue)
    logger.info("Job %s queued (position %d).", job.id, position)

    return JobResponse(
        job_id=job.id,
        status=JobStatus.QUEUED,
        message="Training job submitted to queue.",
        position=position,
    )


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

    status = job.get_status()

    if status == "queued":
        job.cancel()
        logger.info("Job %s canceled (was queued).", job_id)
        return JobResponse(
            job_id=job_id,
            status=JobStatus.CANCELED,
            message="Job canceled (removed from queue).",
        )
    elif status == "started":
        send_stop_job_command(redis_conn, job_id)
        logger.info("Job %s stop signal sent (was running).", job_id)
        return JobResponse(
            job_id=job_id,
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
    # Gather jobs from all registries
    job_details: list[JobDetail] = []
    counts = {"queued": 0, "started": 0, "finished": 0, "failed": 0}

    for job_id in task_queue.job_ids:
        try:
            job = Job.fetch(job_id, connection=redis_conn)
            detail = _job_to_detail(job)
            job_details.append(detail)
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
                detail = _job_to_detail(job)
                job_details.append(detail)
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


@app.get("/health")
def health_check():
    """Health check — verifies Redis connection."""
    try:
        redis_conn.ping()
        return {"status": "healthy", "redis": "connected", "queue_size": len(task_queue)}
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Redis unavailable: {e}")
