"""
Redis Queue Worker
==================
Defines the training task that runs in a separate worker process.
Start with:  rq worker training --with-scheduler
"""

from __future__ import annotations

import logging
from typing import Any

from finetune_yolov8_backend import YOLOv8Backend

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("queue_worker")


def run_training_job(
    roboflow: dict[str, Any],
    training: dict[str, Any],
    paths: dict[str, Any],
) -> dict[str, Any]:
    """Execute a full training pipeline.

    This function is enqueued by the API and executed by the rq worker.
    Each argument is a dict matching the Pydantic schema sections.

    Returns
    -------
    dict with training results and metrics.
    """
    logger.info("Job started — project=%s, epochs=%s",
                roboflow["project_name"], training["epochs"])

    backend = YOLOv8Backend(
        api_key=roboflow["api_key"],
        workspace=roboflow["workspace"],
        project_name=roboflow["project_name"],
        version=roboflow["version"],
        dataset_format=roboflow["dataset_format"],
        model=training["model"],
        dataset_dir=paths["dataset_dir"],
        save_dir=paths["save_dir"],
    )

    result = backend.run_pipeline(train_params=training)

    logger.info("Job finished — result: %s", result)
    return result
