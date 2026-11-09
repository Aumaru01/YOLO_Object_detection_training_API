"""
Model test routes — try a trained model on images from the UI's Test tab
=======================================================================
Pipeline: 1. give a weights path  2. load the model (kept in memory)
          3. send images and get detections back.

One model is loaded at a time, inside the API process (on the chosen device),
until it is unloaded or replaced. Inference calls are serialized with a lock.
"""

from __future__ import annotations

import io
import logging
import threading
import time

from pathlib import Path
from typing import Any, Optional

from fastapi import APIRouter, HTTPException, Query, Request
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

logger = logging.getLogger("model_test")

PROJECT_DIR = Path(__file__).resolve().parent
MODEL_SUFFIXES = {".pt", ".onnx", ".engine", ".torchscript"}
MAX_IMAGE_BYTES = 30 * 1024 * 1024


class LoadBody(BaseModel):
    path: str = Field(..., description="Weights file, e.g. models/<job>/train/weights/best.pt (absolute or project-relative).")
    device: str = Field("cpu", description="'cpu', 'cuda:0', ...")


class _Loaded:
    def __init__(self) -> None:
        self.model: Any = None
        self.info: Optional[dict[str, Any]] = None
        self.lock = threading.Lock()


_state = _Loaded()


def _resolve_weights(path: str) -> Path:
    path = (path or "").strip()
    if not path:
        raise HTTPException(400, "Enter the path of a model weights file.")
    if "\\" in path:
        raise HTTPException(400, "Windows-style paths aren't supported — use the Linux path on this server.")
    p = Path(path).expanduser()
    if not p.is_absolute():
        p = PROJECT_DIR / p
    if not p.exists():
        raise HTTPException(404, f"'{path}' not found on the server.")
    if p.is_file() and p.suffix.lower() not in MODEL_SUFFIXES:
        raise HTTPException(400, f"Unsupported model file '{p.name}' — expected {', '.join(sorted(MODEL_SUFFIXES))}.")
    return p


def _model_info(model: Any, path: Path, device: str, load_s: float) -> dict[str, Any]:
    names = model.names if isinstance(model.names, dict) else dict(enumerate(model.names or []))
    train_args = {}
    ckpt = getattr(model, "ckpt", None)
    if isinstance(ckpt, dict):
        ta = ckpt.get("train_args") or {}
        train_args = {k: ta.get(k) for k in ("data", "epochs", "imgsz", "batch", "model") if k in ta}
    st = path.stat()
    return {
        "path": str(path),
        "name": path.name,
        "task": getattr(model, "task", None),
        "device": device,
        "class_names": [names[k] for k in sorted(names)],
        "num_classes": len(names),
        "imgsz": train_args.get("imgsz") or 640,
        "train_args": train_args,
        "file_size": st.st_size if path.is_file() else None,
        "modified": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(st.st_mtime)),
        "load_seconds": round(load_s, 2),
        "loaded_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }


def build_router() -> APIRouter:
    router = APIRouter(prefix="/test", tags=["UI: model test"])

    @router.get("/model")
    def current_model():
        """The currently loaded model, or {loaded: false}."""
        return {"loaded": _state.info is not None, "model": _state.info}

    @router.post("/load")
    def load_model(body: LoadBody):
        """Load a weights file (replacing any loaded model)."""
        path = _resolve_weights(body.path)
        device = body.device.strip() or "cpu"
        from ultralytics import YOLO
        import torch

        if device.startswith("cuda") and not torch.cuda.is_available():
            raise HTTPException(400, "CUDA isn't available on this server — choose cpu.")
        with _state.lock:
            _unload_locked()
            t0 = time.perf_counter()
            try:
                model = YOLO(str(path))
                # Warm-up so the first real image isn't slow (also validates the device).
                import numpy as np
                model.predict(np.zeros((64, 64, 3), dtype=np.uint8), device=device, verbose=False)
            except Exception as exc:
                logger.exception("Loading model %s failed", path)
                raise HTTPException(400, f"Could not load '{path.name}': {exc}")
            _state.model = model
            _state.info = _model_info(model, path, device, time.perf_counter() - t0)
        logger.info("Test model loaded: %s on %s", path, device)
        return {"loaded": True, "model": _state.info}

    @router.post("/unload")
    def unload_model():
        """Free the loaded model (and its GPU memory)."""
        with _state.lock:
            _unload_locked()
        return {"loaded": False, "model": None}

    @router.post("/predict")
    async def predict(
        request: Request,
        conf: float = Query(0.25, ge=0.0, le=1.0),
        iou: float = Query(0.7, ge=0.0, le=1.0),
        imgsz: Optional[int] = Query(None, ge=32, le=4096),
    ):
        """Run the loaded model on one image sent as the raw request body."""
        data = await request.body()
        if not data:
            raise HTTPException(400, "Empty image.")
        if len(data) > MAX_IMAGE_BYTES:
            raise HTTPException(413, "Image too large (max 30 MB).")
        return await run_in_threadpool(_predict, data, conf, iou, imgsz)

    return router


def _unload_locked() -> None:
    if _state.model is None:
        return
    _state.model = None
    _state.info = None
    try:
        import gc
        import torch
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def _predict(data: bytes, conf: float, iou: float, imgsz: Optional[int]) -> dict[str, Any]:
    from PIL import Image, ImageOps

    try:
        img = ImageOps.exif_transpose(Image.open(io.BytesIO(data)))
        img = img.convert("RGB")
    except Exception:
        raise HTTPException(400, "Couldn't read the image (expected JPG, PNG, WebP or BMP).")
    with _state.lock:
        if _state.model is None:
            raise HTTPException(409, "No model loaded — load one first.")
        info = _state.info
        t0 = time.perf_counter()
        results = _state.model.predict(
            img, conf=conf, iou=iou, imgsz=imgsz or info["imgsz"], device=info["device"], verbose=False,
        )
        elapsed = (time.perf_counter() - t0) * 1000
    r = results[0]
    names = r.names
    detections = []
    if r.boxes is not None and len(r.boxes):
        for (x, y, w, h), c, p in zip(r.boxes.xywhn.tolist(), r.boxes.cls.tolist(), r.boxes.conf.tolist()):
            detections.append({"cls": int(c), "name": names.get(int(c), str(int(c))),
                               "conf": round(float(p), 4), "x": x, "y": y, "w": w, "h": h})
    detections.sort(key=lambda d: -d["conf"])
    return {
        "width": img.width,
        "height": img.height,
        "detections": detections,
        "speed_ms": {k: round(v, 1) for k, v in (r.speed or {}).items()},
        "total_ms": round(elapsed, 1),
        "model": info["name"],
        "conf": conf,
        "iou": iou,
        "imgsz": imgsz or info["imgsz"],
    }
