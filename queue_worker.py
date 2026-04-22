"""
Redis Queue Worker
==================
Defines the training task that runs in a separate worker process.
Start with:  rq worker training --with-scheduler
"""

import logging

from typing import Any
from finetune_yolo_backend import YOLOTrainBackend

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("queue_worker_log")


def run_training_job(
    roboflow: dict[str, Any],
    training: dict[str, Any],
    job_name: str,
) -> dict[str, Any]:
    """Execute a full training pipeline.

    This function is enqueued by the API and executed by the rq worker.

    Returns
    -------
    dict with training results and metrics.
    """
    logger.info("Job started — name=%s, project=%s, epochs=%s",
                job_name, roboflow["project_name"], training["epochs"])

    backend = YOLOTrainBackend(
        api_key=roboflow["api_key"],
        workspace=roboflow["workspace"],
        project_name=roboflow["project_name"],
        version=roboflow["version"],
        dataset_format=roboflow["dataset_format"],
        model=training["model"],
        job_name=job_name,
    )

    result = backend.run_pipeline(train_params=training)

    logger.info("Job finished [%s] — result: %s", job_name, result)
    return result
