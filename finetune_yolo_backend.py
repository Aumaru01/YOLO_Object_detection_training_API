"""
YOLO Fine-tuning Backend
==========================
Pure backend — receives all parameters directly (no config file).
Called by the queue worker with values from the API request body.

Files are stored under:
  datasets/{job_name}/    — downloaded dataset
  models/{job_name}/      — trained model weights & plots
"""

import os
import shutil
import logging

from pathlib import Path
from ultralytics import YOLO
from roboflow import Roboflow
from typing import Any, Optional

import torch

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("backend_log")

# ---------------------------------------------------------------------------
# Base directories
# ---------------------------------------------------------------------------
BASE_DATASET_DIR = Path(__file__).resolve().parent / "datasets"
BASE_MODEL_DIR = Path(__file__).resolve().parent / "models"


# ---------------------------------------------------------------------------
# Backend class
# ---------------------------------------------------------------------------
class YOLOTrainBackend:
    """Stateless-ish training backend.

    Every parameter comes from the caller (API -> worker -> here).
    Files are organized by job_name:
      datasets/{job_name}/   and   models/{job_name}/
    """

    def __init__(
        self,
        api_key: str,
        workspace: str,
        project_name: str,
        version: int,
        dataset_format: str,
        model: str,
        job_name: str,
    ) -> None:
        # Roboflow
        self.rf = Roboflow(api_key=api_key)
        self.workspace = workspace
        self.project_name = project_name
        self.version = version
        self.dataset_format = dataset_format

        # Model
        self.model_name = model
        self._model: Optional[YOLO] = None

        # Paths based on job_name
        self.job_name = job_name
        self.dataset_dir = BASE_DATASET_DIR / job_name
        self.model_dir = BASE_MODEL_DIR / job_name
        self._data_yaml_cache: Optional[Path] = None

        self.dataset_dir.mkdir(parents=True, exist_ok=True)
        self.model_dir.mkdir(parents=True, exist_ok=True)

        # Device
        self.device = self._select_device()

        logger.info("Backend ready — %s", self)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------
    @property
    def model(self) -> YOLO:
        if self._model is None:
            logger.info("Loading model '%s' ...", self.model_name)
            self._model = YOLO(self.model_name)
        return self._model

    def _find_data_yaml(self) -> Optional[Path]:
        """Find data.yaml — check root first, then search subdirectories."""
        # Cached
        if self._data_yaml_cache and self._data_yaml_cache.exists():
            return self._data_yaml_cache

        # Direct path
        direct = self.dataset_dir / "data.yaml"
        if direct.exists():
            self._data_yaml_cache = direct
            return direct

        # Recursive search (Roboflow sometimes nests in subfolder)
        found = list(self.dataset_dir.rglob("data.yaml"))
        if found:
            self._data_yaml_cache = found[0]
            logger.info("Found data.yaml at: %s", found[0])
            return found[0]

        return None

    @property
    def data_yaml_path(self) -> Path:
        """Return data.yaml path, or expected path if not yet downloaded."""
        found = self._find_data_yaml()
        return found if found else self.dataset_dir / "data.yaml"

    # ------------------------------------------------------------------
    # Device
    # ------------------------------------------------------------------
    @staticmethod
    def _select_device() -> str:
        if torch.cuda.is_available():
            logger.info("Using CUDA: %s", torch.cuda.get_device_name(0))
            return "cuda"
        logger.info("CUDA not available — using CPU.")
        return "cpu"

    # ------------------------------------------------------------------
    # Dataset
    # ------------------------------------------------------------------
    def download_dataset(self, *, force: bool = False) -> Path:
        """Download dataset from Roboflow."""
        if self._find_data_yaml() and not force:
            logger.info("Dataset already at '%s' — skipping.", self.data_yaml_path)
            return self.dataset_dir

        logger.info(
            "Downloading: %s/%s v%d (%s) ...",
            self.workspace, self.project_name, self.version, self.dataset_format,
        )
        project = self.rf.workspace(self.workspace).project(self.project_name)
        project.version(self.version).download(
            self.dataset_format, location=str(self.dataset_dir),
        )

        # Log what was actually downloaded
        all_files = list(self.dataset_dir.rglob("*"))
        logger.info("Downloaded %d files to '%s'.", len(all_files), self.dataset_dir)
        yaml_files = [f for f in all_files if f.name == "data.yaml"]
        if yaml_files:
            logger.info("Found data.yaml at: %s", yaml_files[0])
            self._data_yaml_cache = yaml_files[0]
        else:
            logger.error("data.yaml NOT FOUND after download! Files in dir:")
            for f in sorted(all_files)[:20]:
                logger.error("  %s", f)

        return self.dataset_dir

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    def train(self, train_params: dict[str, Any]) -> dict[str, Any]:
        """Run training with the given hyperparameters."""
        if not self.data_yaml_path.exists():
            logger.info("Dataset not found — downloading first.")
            self.download_dataset()

        train_args: dict[str, Any] = {
            "data": str(self.data_yaml_path),
            "epochs": train_params["epochs"],
            "imgsz": train_params["img_size"],
            "batch": train_params["batch_size"],
            "patience": train_params["patience"],
            "optimizer": train_params["optimizer"],
            "lr0": train_params["lr0"],
            "scale": train_params["scale"],
            "mosaic": train_params["mosaic"],
            "mixup": train_params["mixup"],
            "copy_paste": train_params["copy_paste"],
            "plots": train_params["plots"],
            "cache": train_params["cache"],
            "device": self.device,
            "project": str(self.model_dir),
            "save": True,
        }

        logger.info(
            "Training — epochs=%s, batch=%s, lr0=%s, save='%s'",
            train_args["epochs"], train_args["batch"], train_args["lr0"], self.model_dir,
        )

        results = self.model.train(**train_args)
        logger.info("Training complete.")

        return {
            "job_name": self.job_name,
            "model_dir": str(self.model_dir),
            "dataset_dir": str(self.dataset_dir),
            "epochs": train_args["epochs"],
            "device": self.device,
        }

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------
    def evaluate(self, img_size: int = 640) -> dict[str, float]:
        """Run validation and return mAP metrics."""
        if not self.data_yaml_path.exists():
            raise FileNotFoundError(
                f"Dataset not found at '{self.data_yaml_path}'. "
                "Call download_dataset() first."
            )

        logger.info("Running evaluation ...")
        metrics = self.model.val(
            data=str(self.data_yaml_path),
            imgsz=img_size,
            device=self.device,
            plots=True,
        )

        result = {
            "mAP50": float(metrics.box.map50),
            "mAP50_95": float(metrics.box.map),
        }
        logger.info("mAP50: %.4f | mAP50-95: %.4f", result["mAP50"], result["mAP50_95"])
        return result

    # ------------------------------------------------------------------
    # Full pipeline (download -> train -> evaluate)
    # ------------------------------------------------------------------
    def run_pipeline(self, train_params: dict[str, Any]) -> dict[str, Any]:
        """Execute the full pipeline: download -> train -> evaluate."""
        logger.info("=== Starting full pipeline [%s] ===", self.job_name)

        self.download_dataset()
        train_result = self.train(train_params)
        eval_result = self.evaluate(img_size=train_params["img_size"])

        result = {**train_result, **eval_result}
        logger.info("=== Pipeline complete [%s] === Result: %s", self.job_name, result)
        return result

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------
    def cleanup_dataset(self) -> None:
        if self.dataset_dir.exists():
            shutil.rmtree(self.dataset_dir)
            logger.info("Cleaned up '%s'.", self.dataset_dir)

    # ------------------------------------------------------------------
    # Repr
    # ------------------------------------------------------------------
    def __repr__(self) -> str:
        return (
            f"YOLOTrainBackend(job={self.job_name!r}, "
            f"project={self.project_name!r}, "
            f"v{self.version}, model={self.model_name!r}, device={self.device!r})"
        )
