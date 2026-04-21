"""
YOLO Fine-tuning Backend
==========================
Pure backend — receives all parameters directly (no config file).
Called by the queue worker with values from the API request body.
"""

import os
import time
import torch
import shutil
import logging

from pathlib import Path
from ultralytics import YOLO
from roboflow import Roboflow
from typing import Any, Optional


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
# Backend class
# ---------------------------------------------------------------------------
class YOLOTrainBackend:
    """Stateless-ish training backend.

    Every parameter comes from the caller (API → worker → here).
    No config file, no hidden defaults.
    """

    def __init__(
        self,
        api_key: str,
        workspace: str,
        project_name: str,
        version: int,
        dataset_format: str,
        model: str,
        save_dataset_dir: str,
        save_model_dir: str,
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

        # Paths
        self.save_dataset_dir = Path(save_dataset_dir)
        self.save_dataset_dir.mkdir(parents=True, exist_ok=True)

        timestamp = time.strftime("%Y%m%d-%H%M%S")
        self.save_model_dir = f"{save_model_dir}-{timestamp}"
        os.makedirs(self.save_model_dir, exist_ok=True)

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

    @property
    def data_yaml_path(self) -> Path:
        return self.save_dataset_dir / "data.yaml"

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
        if self.data_yaml_path.exists() and not force:
            logger.info("Dataset already at '%s' — skipping.", self.save_dataset_dir)
            return self.save_dataset_dir

        logger.info(
            "Downloading: %s/%s v%d (%s) ...",
            self.workspace, self.project_name, self.version, self.dataset_format,
        )
        project = self.rf.workspace(self.workspace).project(self.project_name)
        project.version(self.version).download(
            self.dataset_format, location=str(self.save_dataset_dir),
        )
        logger.info("Dataset saved to '%s'.", self.save_dataset_dir)
        return self.save_dataset_dir

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    def train(self, train_params: dict[str, Any]) -> dict[str, Any]:
        """Run training with the given hyperparameters.

        Parameters
        ----------
        train_params : dict
            Hyperparameters from TrainingConfig (epochs, lr0, etc.)

        Returns
        -------
        dict with save_model_dir and basic metrics.
        """
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
            "project": self.save_model_dir,
            "save": True,
        }

        logger.info(
            "Training — epochs=%s, batch=%s, lr0=%s, save='%s'",
            train_args["epochs"], train_args["batch"], train_args["lr0"], self.save_model_dir,
        )

        results = self.model.train(**train_args)
        logger.info("Training complete.")

        return {
            "save_model_dir": self.save_model_dir,
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
    # Full pipeline (download → train → evaluate)
    # ------------------------------------------------------------------
    def run_pipeline(self, train_params: dict[str, Any]) -> dict[str, Any]:
        """Execute the full pipeline: download → train → evaluate.

        This is the main entry point called by the queue worker.
        """
        logger.info("=== Starting full pipeline ===")

        self.download_dataset()
        train_result = self.train(train_params)
        eval_result = self.evaluate(img_size=train_params["img_size"])

        result = {**train_result, **eval_result}
        logger.info("=== Pipeline complete === Result: %s", result)
        return result

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------
    def cleanup_dataset(self) -> None:
        if self.save_dataset_dir.exists():
            shutil.rmtree(self.save_dataset_dir)
            logger.info("Cleaned up '%s'.", self.save_dataset_dir)

    # ------------------------------------------------------------------
    # Repr
    # ------------------------------------------------------------------
    def __repr__(self) -> str:
        return (
            f"YOLOTrainBackend(project={self.project_name!r}, "
            f"v{self.version}, model={self.model_name!r}, device={self.device!r})"
        )
