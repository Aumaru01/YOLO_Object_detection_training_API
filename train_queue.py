"""
In-Process Job Queue & Worker
==============================
Runs training jobs sequentially with no external broker (no Redis, no
separate worker process/service to start). A background dispatcher thread
pulls jobs off an internal queue and runs each one in its own worker
subprocess, so a running job can still be canceled by terminating it.

The queue starts automatically with the API — just:  python -m api_main
"""

import atexit
import json
import logging
import multiprocessing
import os
import queue as queue_mod
import sys
import threading
import traceback

from dataclasses import dataclass, field
from datetime import datetime
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any, Optional

from train_backend import YOLOTrainBackend

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

# Per-job artifacts read by the UI's training monitor:
#   logs/jobs/{job_id}/job.json       persisted job record (survives API restarts)
#   logs/jobs/{job_id}/train.log      the worker process's full stdout/stderr
#   logs/jobs/{job_id}/progress.json  live training progress (epoch, losses, metrics)
JOBS_LOG_DIR = _LOG_DIR / "jobs"
JOBS_LOG_DIR.mkdir(exist_ok=True)


def job_dir(job_id: str) -> Path:
    return JOBS_LOG_DIR / job_id


def job_log_file(job_id: str) -> Path:
    return job_dir(job_id) / "train.log"


def job_progress_file(job_id: str) -> Path:
    return job_dir(job_id) / "progress.json"

# "spawn" avoids inheriting locks/threads from the API process (which runs a
# background dispatcher thread) — safer than "fork" in a multi-threaded app.
_MP_CTX = multiprocessing.get_context("spawn")


class _ThrottledProgressStream:
    """Text stream wrapper for the job log that rate-limits carriage-return
    progress-bar redraws (ultralytics redraws ~10x/second with '\\r'), so a
    long run doesn't bloat train.log. Complete lines pass straight through;
    an in-progress '\\r' redraw is written at most once per ``interval``.
    """

    def __init__(self, raw: Any, interval: float = 2.0) -> None:
        self._raw = raw
        self._interval = interval
        self._pending = ""
        self._last = 0.0

    def write(self, s: str) -> int:
        import time
        if "\r" not in s and not self._pending:
            self._raw.write(s)
            return len(s)
        text = self._pending + s
        if "\n" in text:
            head, _, tail = text.rpartition("\n")
            self._raw.write(head + "\n")
            text = tail
        self._pending = text
        now = time.monotonic()
        if text and now - self._last >= self._interval:
            self._raw.write(text if text.startswith("\r") else "\r" + text)
            self._raw.flush()
            self._pending = ""
            self._last = now
        return len(s)

    def flush(self) -> None:
        self._raw.flush()

    def isatty(self) -> bool:
        return False

    def __getattr__(self, name: str) -> Any:
        return getattr(self._raw, name)


def run_training_job(
    roboflow: Optional[dict[str, Any]],
    training: dict[str, Any],
    job_name: str,
    dataset_path: Optional[str] = None,
    output_path: Optional[str] = None,
    progress_file: Optional[str] = None,
) -> dict[str, Any]:
    """Execute a full training pipeline. Runs inside a worker subprocess.

    ``roboflow`` is None for jobs training on an already-local dataset —
    the pipeline then skips the download step entirely. ``dataset_path``,
    when given, points the backend at that filesystem location instead of
    the default datasets/{job_name}/ convention (used by local jobs).
    ``output_path``, when given, stores the trained model + evaluation
    results there instead of the default models/{job_name}/.
    """
    if roboflow is not None:
        logger.info("Job started — name=%s, source=roboflow, project=%s, epochs=%s",
                    job_name, roboflow["project_name"], training["epochs"])
        backend = YOLOTrainBackend(
            model=training["model"],
            job_name=job_name,
            api_key=roboflow["api_key"],
            workspace=roboflow["workspace"],
            project_name=roboflow["project_name"],
            version=roboflow["version"],
            dataset_format=roboflow["dataset_format"],
            output_path=output_path,
            progress_file=progress_file,
        )
        result = backend.run_pipeline_roboflow(train_params=training)
    else:
        logger.info("Job started — name=%s, source=local, dataset_path=%s, epochs=%s",
                    job_name, dataset_path, training["epochs"])
        backend = YOLOTrainBackend(
            model=training["model"], job_name=job_name,
            dataset_path=dataset_path, output_path=output_path,
            progress_file=progress_file,
        )
        result = backend.run_pipeline_local(train_params=training)

    logger.info("Job finished [%s] — result: %s", job_name, result)
    return result


def _worker_entrypoint(
    roboflow: Optional[dict[str, Any]],
    training: dict[str, Any],
    job_name: str,
    dataset_path: Optional[str],
    output_path: Optional[str],
    result_queue: "multiprocessing.Queue",
    job_id: Optional[str] = None,
) -> None:
    """Run inside the spawned worker process and report the outcome back."""
    progress_file = None
    if job_id is not None:
        # Send everything this process prints (ultralytics progress bars,
        # torch warnings, our own log records) to the job's own log file.
        job_dir(job_id).mkdir(parents=True, exist_ok=True)
        log_fd = os.open(job_log_file(job_id), os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
        os.dup2(log_fd, 1)
        os.dup2(log_fd, 2)
        os.close(log_fd)
        for stream in (sys.stdout, sys.stderr):
            try:
                stream.reconfigure(line_buffering=True)
            except (AttributeError, ValueError):
                pass
        sys.stdout = _ThrottledProgressStream(sys.stdout)
        sys.stderr = _ThrottledProgressStream(sys.stderr)
        progress_file = str(job_progress_file(job_id))
    try:
        result = run_training_job(
            roboflow, training, job_name, dataset_path, output_path, progress_file,
        )
    except Exception:
        logger.exception("Job failed [%s]", job_name)
        result_queue.put(("error", traceback.format_exc()))
        return
    result_queue.put(("ok", result))


@dataclass
class JobRecord:
    """In-memory record of a submitted job."""
    job_id: str
    job_name: str
    training: dict[str, Any]
    roboflow: Optional[dict[str, Any]] = None
    dataset_path: Optional[str] = None
    output_path: Optional[str] = None
    status: str = "queued"
    created_at: datetime = field(default_factory=datetime.utcnow)
    started_at: Optional[datetime] = None
    ended_at: Optional[datetime] = None
    result: Optional[dict[str, Any]] = None
    error: Optional[str] = None
    pid: Optional[int] = None
    exitcode: Optional[int] = None
    process: Optional[Any] = field(default=None, repr=False)

    _PERSISTED = (
        "job_id", "job_name", "training", "dataset_path", "output_path", "status",
        "result", "error", "pid", "exitcode",
    )

    def to_json(self) -> dict[str, Any]:
        data = {k: getattr(self, k) for k in self._PERSISTED}
        for k in ("created_at", "started_at", "ended_at"):
            v = getattr(self, k)
            data[k] = v.isoformat() if v else None
        data["source"] = "roboflow" if self.roboflow else "local"
        if self.roboflow:  # never write the API key to disk
            data["roboflow"] = {k: v for k, v in self.roboflow.items() if k != "api_key"}
        return data

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> "JobRecord":
        kwargs = {k: data.get(k) for k in cls._PERSISTED if k in data}
        for k in ("created_at", "started_at", "ended_at"):
            if data.get(k):
                kwargs[k] = datetime.fromisoformat(data[k])
        kwargs.setdefault("training", {})
        record = cls(**kwargs)
        record.roboflow = data.get("roboflow")
        return record

    def save(self) -> None:
        try:
            job_dir(self.job_id).mkdir(parents=True, exist_ok=True)
            path = job_dir(self.job_id) / "job.json"
            tmp = path.with_suffix(".tmp")
            tmp.write_text(json.dumps(self.to_json(), indent=2, default=str), encoding="utf-8")
            os.replace(tmp, path)
        except OSError:
            logger.exception("Could not persist job record %s", self.job_id)


class JobQueue:
    """Single-worker in-process job queue — no Redis, no separate worker service.

    A background dispatcher thread runs jobs one at a time, each inside its
    own worker subprocess so a running job can be canceled via terminate().
    """

    def __init__(self) -> None:
        self._jobs: dict[str, JobRecord] = {}
        self._pending: "queue_mod.Queue[str]" = queue_mod.Queue()
        self._lock = threading.Lock()
        self._load_history()
        self._dispatcher = threading.Thread(target=self._dispatch_loop, daemon=True)
        self._dispatcher.start()
        atexit.register(self._terminate_all)

    def _load_history(self) -> None:
        """Reload job records persisted by earlier API runs (read-only history).
        Jobs that were still queued/running when the API stopped are marked failed.
        """
        for meta in sorted(JOBS_LOG_DIR.glob("*/job.json")):
            try:
                record = JobRecord.from_json(json.loads(meta.read_text(encoding="utf-8")))
            except Exception:
                logger.warning("Skipping unreadable job record %s", meta)
                continue
            if record.status in ("queued", "started"):
                record.status = "failed"
                record.error = "Interrupted: the API was restarted while this job was queued or running."
                record.ended_at = record.ended_at or datetime.utcnow()
                record.save()
            self._jobs[record.job_id] = record
        if self._jobs:
            logger.info("Loaded %d job record(s) from %s", len(self._jobs), JOBS_LOG_DIR)

    def _terminate_all(self) -> None:
        """Terminate any still-running worker process on API shutdown.

        Worker processes are non-daemonic (see _dispatch_loop) — required so
        ultralytics/torch can spawn their own DataLoader worker children,
        since Python forbids daemonic processes from having children — so
        this is the substitute safety net against leaving them orphaned.
        """
        with self._lock:
            processes = [r.process for r in self._jobs.values() if r.process is not None]
        for process in processes:
            if process.is_alive():
                process.terminate()

    def enqueue(
        self,
        job_id: str,
        job_name: str,
        training: dict[str, Any],
        roboflow: Optional[dict[str, Any]] = None,
        dataset_path: Optional[str] = None,
        output_path: Optional[str] = None,
    ) -> JobRecord:
        record = JobRecord(
            job_id=job_id, job_name=job_name, training=training,
            roboflow=roboflow, dataset_path=dataset_path, output_path=output_path,
        )
        with self._lock:
            self._jobs[job_id] = record
        record.save()
        self._pending.put(job_id)
        return record

    def get(self, job_id: str) -> Optional[JobRecord]:
        with self._lock:
            return self._jobs.get(job_id)

    def all_jobs(self) -> list[JobRecord]:
        with self._lock:
            return list(self._jobs.values())

    def position(self, job_id: str) -> Optional[int]:
        """1-based position in the pending queue, or None if not queued."""
        pending_ids = list(self._pending.queue)
        if job_id in pending_ids:
            return pending_ids.index(job_id) + 1
        return None

    def pending_count(self) -> int:
        return self._pending.qsize()

    def cancel(self, job_id: str) -> str:
        """Cancel a queued or running job. Returns the resulting status.

        Raises KeyError if the job is unknown.
        """
        with self._lock:
            record = self._jobs.get(job_id)
            if record is None:
                raise KeyError(job_id)
            if record.status not in ("queued", "started"):
                return record.status
            process = record.process
            record.status = "canceled"
            record.ended_at = datetime.utcnow()
            record.save()
        if process is not None:
            process.terminate()
        return "canceled"

    def _dispatch_loop(self) -> None:
        while True:
            job_id = self._pending.get()
            with self._lock:
                record = self._jobs.get(job_id)
                if record is None or record.status == "canceled":
                    continue
                result_queue = _MP_CTX.Queue()
                # Not daemonic: ultralytics/torch spawn their own DataLoader
                # worker child processes, which Python forbids for daemonic
                # processes (see JobQueue._terminate_all for the tradeoff).
                process = _MP_CTX.Process(
                    target=_worker_entrypoint,
                    args=(
                        record.roboflow, record.training, record.job_name,
                        record.dataset_path, record.output_path, result_queue,
                        record.job_id,
                    ),
                )
                record.process = process
                record.status = "started"
                record.started_at = datetime.utcnow()

            process.start()
            with self._lock:
                record.pid = process.pid
                record.save()
            process.join()

            with self._lock:
                record.exitcode = process.exitcode
                if record.status == "canceled":
                    record.process = None
                    record.save()
                    continue
                if not result_queue.empty():
                    kind, payload = result_queue.get()
                    if kind == "ok":
                        record.result = payload
                        record.status = "finished"
                    else:
                        record.error = payload
                        record.status = "failed"
                else:
                    record.status = "failed"
                    record.error = (
                        f"Worker process exited unexpectedly (code={process.exitcode})."
                    )
                record.ended_at = datetime.utcnow()
                record.process = None
                record.save()
