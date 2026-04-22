"""
Redis Queue Worker
==================
Defines the training task that runs in a separate worker process.
Start with:  rq worker training --with-scheduler
"""

import logging

from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any

from finetune_yolo_backend import YOLOTrainBackend

# ---------------------------------------------------------------------------
# Logging — share the same rotating file as the API so everything is
# visible in one place (logs/api.log).
# ---------------------------------------------------------------------------
_LOG_DIR = Path(__file__).resolve().parent / "logs"
_LOG_DIR.mkdir(exist_ok=True)
_LOG_FILE = _LOG_DIR / "api.log"

_fmt = logging.Formatter(
    fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

_root = logging.getLogger()
_root.setLevel(logging.INFO)

if not any(isinstance(h, RotatingFileHandler) for h in _root.handlers):
    _fh = RotatingFileHandler(
        _LOG_FILE, maxBytes=5 * 1024 * 1024, backupCount=5, encoding="utf-8"
    )
    _fh.setFormatter(_fmt)
    _root.addHandler(_fh)

if not any(
    isinstance(h, logging.StreamHandler) and not isinstance(h, RotatingFileHandler)
    for h in _root.handlers
):
    _ch = logging.StreamHandler()
    _ch.setFormatter(_fmt)
    _root.addHandler(_ch)

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

    try:
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
    except Exception:
        # Let RQ mark the job as failed, but make sure we see the full
        # traceback in the shared log file.
        logger.exception("Job failed [%s]", job_name)
        raise

    logger.info("Job finished [%s] — result: %s", job_name, result)
    return result
