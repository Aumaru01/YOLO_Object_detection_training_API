"""
Pydantic schemas — request/response models for the training API.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Request body (job_name is sent as query parameter, NOT in body)
# ---------------------------------------------------------------------------
class RoboflowConfig(BaseModel):
    """Roboflow dataset source settings."""
    api_key: str = Field(..., description="Roboflow API key")
    workspace: str = Field("jakapong-workspace")
    project_name: str = Field("logo-detection-project-iihu1")
    version: int = Field(1, ge=1)
    dataset_format: str = Field("yolov8")


class TrainingConfig(BaseModel):
    """Training hyperparameters."""
    model: str = Field("yolov8m.pt")
    epochs: int = Field(100, ge=1)
    img_size: int = Field(640, ge=32)
    batch_size: int = Field(4, ge=1)
    patience: int = Field(60, ge=1)
    optimizer: str = Field("AdamW")
    lr0: float = Field(0.005, gt=0)
    scale: float = Field(0.4, ge=0)
    mosaic: float = Field(1.0, ge=0, le=1)
    mixup: float = Field(0.2, ge=0, le=1)
    copy_paste: float = Field(0.1, ge=0, le=1)
    plots: bool = Field(True)
    cache: bool = Field(True)
    device: str = Field(
        "cpu",
        description="Device to train on: 'cpu', 'cuda', 'cuda:0', 'cuda:1', etc.",
    )


class TrainRequest(BaseModel):
    """Training request body — only roboflow + training config.
    job_name is sent as a query parameter separately.
    """
    roboflow: RoboflowConfig
    training: TrainingConfig = Field(default_factory=TrainingConfig)

    model_config = {"json_schema_extra": {
        "examples": [{
            "roboflow": {
                "api_key": "YOUR_API_KEY",
                "workspace": "workspace-name",
                "project_name": "projectID",
                "version": 1,
                "dataset_format": "yolov8",
            },
            "training": {
                    "batch_size": 4,
                    "cache": True,
                    "copy_paste": 0.1,
                    "epochs": 100,
                    "img_size": 640,
                    "lr0": 0.005,
                    "mixup": 0.2,
                    "model": "yolov8m.pt",
                    "mosaic": 1,
                    "optimizer": "AdamW",
                    "patience": 60,
                    "plots": True,
                    "scale": 0.4,
                    "device": "cuda",
            },
        }],
    }}


# ---------------------------------------------------------------------------
# Job status
# ---------------------------------------------------------------------------
class JobStatus(str, Enum):
    QUEUED = "queued"
    STARTED = "started"
    FINISHED = "finished"
    FAILED = "failed"
    CANCELED = "canceled"


class JobResponse(BaseModel):
    """Returned when a job is submitted."""
    job_id: str
    job_name: str
    status: JobStatus
    message: str
    position: Optional[int] = None


class JobDetail(BaseModel):
    """Detailed job info for status checks."""
    job_id: str
    job_name: Optional[str] = None
    status: JobStatus
    message: str
    created_at: Optional[datetime] = None
    started_at: Optional[datetime] = None
    ended_at: Optional[datetime] = None
    result: Optional[dict] = None
    error: Optional[str] = None


class QueueInfo(BaseModel):
    """Current queue state."""
    queued: int
    started: int
    finished: int
    failed: int
    jobs: list[JobDetail]
