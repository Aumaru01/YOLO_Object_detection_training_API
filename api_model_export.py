"""
Model export routes — convert trained .pt weights to ONNX (and optionally OpenVINO)
=================================================================================
Used by the UI's "Export" tab. Pipeline: 1. give a .pt path  2. choose options
3. convert  4. verify the .onnx with onnxruntime  5. download / test it.

The conversion runs in a separate Python process (``python -m api_model_export``)
so it can't leak memory/CUDA state into the API, and its console output is
returned as a log. One export runs at a time.
"""

from __future__ import annotations

import io
import json
import logging
import os
import re
import subprocess
import sys
import threading
import time
import zipfile

from pathlib import Path
from typing import Any, Optional

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

logger = logging.getLogger("model_export")

PROJECT_DIR = Path(__file__).resolve().parent
EXPORT_TIMEOUT_S = 30 * 60
MAX_LOG_CHARS = 60_000
_lock = threading.Lock()
_running: dict[str, Any] = {}


class ExportBody(BaseModel):
    path: str = Field(..., description="Trained weights (.pt), absolute or project-relative.")
    imgsz: Optional[int] = Field(None, ge=32, le=4096, description="Input size; default = the size the model was trained at.")
    dynamic: bool = Field(False, description="Dynamic batch / image size axes.")
    simplify: bool = Field(True, description="Simplify the graph (onnxslim).")
    half: bool = Field(False, description="FP16 weights (needs a CUDA device).")
    opset: Optional[int] = Field(None, ge=7, le=25, description="ONNX opset; default = ultralytics' choice.")
    batch: int = Field(1, ge=1, le=64)
    device: str = Field("cpu", description="'cpu' or 'cuda:0', ...")
    openvino: bool = Field(False, description="Also export OpenVINO (.xml/.bin).")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _resolve_pt(path: str) -> Path:
    path = (path or "").strip()
    if not path:
        raise HTTPException(400, "Enter the path of a .pt weights file.")
    if "\\" in path:
        raise HTTPException(400, "Windows-style paths aren't supported — use the Linux path on this server.")
    p = Path(path).expanduser()
    if not p.is_absolute():
        p = PROJECT_DIR / p
    if not p.is_file():
        raise HTTPException(404, f"'{path}' not found on the server.")
    if p.suffix.lower() != ".pt":
        raise HTTPException(400, f"'{p.name}' is not a .pt file — only PyTorch weights can be exported.")
    return p


def _file_entry(p: Path, source_mtime: float) -> dict[str, Any]:
    if p.is_dir():
        size = sum(f.stat().st_size for f in p.rglob("*") if f.is_file())
    else:
        size = p.stat().st_size
    mtime = p.stat().st_mtime
    return {
        "path": str(p),
        "name": p.name,
        "is_dir": p.is_dir(),
        "size": size,
        "modified": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(mtime)),
        "stale": mtime < source_mtime,  # older than the .pt it came from
    }


def _existing_exports(pt: Path) -> dict[str, Any]:
    src_m = pt.stat().st_mtime
    onnx = pt.with_suffix(".onnx")
    ov = pt.parent / f"{pt.stem}_openvino_model"
    return {
        "onnx": _file_entry(onnx, src_m) if onnx.is_file() else None,
        "openvino": _file_entry(ov, src_m) if ov.is_dir() else None,
    }


def _pt_info(pt: Path) -> dict[str, Any]:
    """Cheap metadata from the checkpoint (no model build)."""
    info: dict[str, Any] = {"path": str(pt), "name": pt.name, "size": pt.stat().st_size,
                            "modified": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(pt.stat().st_mtime))}
    try:
        import torch
        ckpt = torch.load(str(pt), map_location="cpu", weights_only=False)
        ta = ckpt.get("train_args") or {}
        info["imgsz"] = ta.get("imgsz")
        info["train_data"] = ta.get("data")
        model = ckpt.get("ema") or ckpt.get("model")
        names = getattr(model, "names", None)
        if isinstance(names, dict):
            info["class_names"] = [names[k] for k in sorted(names)]
        elif names:
            info["class_names"] = list(names)
        info["task"] = getattr(model, "task", None)
    except Exception as exc:  # metadata is a nice-to-have
        info["error"] = f"Couldn't read checkpoint metadata: {exc}"
    return info


def _verify_onnx(path: Path) -> dict[str, Any]:
    """Open the exported file with onnxruntime and run one dummy inference."""
    out: dict[str, Any] = {"ok": False}
    try:
        import numpy as np
        import onnxruntime as ort

        sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        ins, outs = sess.get_inputs(), sess.get_outputs()
        out["inputs"] = [{"name": i.name, "shape": i.shape, "type": i.type} for i in ins]
        out["outputs"] = [{"name": o.name, "shape": o.shape, "type": o.type} for o in outs]
        meta = sess.get_modelmeta().custom_metadata_map or {}
        out["metadata"] = {k: meta[k] for k in ("description", "task", "imgsz", "batch", "stride", "date", "version") if k in meta}
        names = meta.get("names")
        if names:
            import ast
            try:
                d = ast.literal_eval(names)
                out["class_names"] = [d[k] for k in sorted(d)] if isinstance(d, dict) else list(d)
            except Exception:
                pass
        shape = [d if isinstance(d, int) else None for d in ins[0].shape]
        imgsz = json.loads(meta["imgsz"]) if "imgsz" in meta else [640, 640]
        shape = [shape[0] or 1, shape[1] or 3, shape[2] or imgsz[0], shape[3] or imgsz[1]]
        dtype = np.float16 if "float16" in ins[0].type else np.float32
        x = np.zeros(shape, dtype=dtype)
        sess.run(None, {ins[0].name: x})  # warm-up
        t0 = time.perf_counter()
        sess.run(None, {ins[0].name: x})
        out["cpu_inference_ms"] = round((time.perf_counter() - t0) * 1000, 1)
        out["ok"] = True
    except Exception as exc:
        out["error"] = str(exc)
    return out


def _run_export(pt: Path, body: ExportBody) -> dict[str, Any]:
    args = {
        "path": str(pt), "imgsz": body.imgsz, "dynamic": body.dynamic, "simplify": body.simplify,
        "half": body.half, "opset": body.opset, "batch": body.batch, "device": body.device,
        "openvino": body.openvino,
    }
    t0 = time.perf_counter()
    env = {**os.environ, "PYTHONUNBUFFERED": "1"}
    try:
        proc = subprocess.run(
            [sys.executable, "-m", "api_model_export", json.dumps(args)],
            cwd=str(PROJECT_DIR), env=env, capture_output=True, text=True, timeout=EXPORT_TIMEOUT_S,
        )
    except subprocess.TimeoutExpired:
        raise HTTPException(504, f"Export took longer than {EXPORT_TIMEOUT_S // 60} minutes and was stopped.")
    log = (proc.stdout or "") + (proc.stderr or "")
    result: dict[str, Any] = {}
    for line in reversed(log.splitlines()):
        if line.startswith("__EXPORT_RESULT__"):
            result = json.loads(line[len("__EXPORT_RESULT__"):])
            break
    log = re.sub(r"\x1b\[[0-9;]*m", "", log)
    log = "\n".join(l for l in log.splitlines() if not l.startswith("__EXPORT_RESULT__"))
    if len(log) > MAX_LOG_CHARS:
        log = "…(truncated)…\n" + log[-MAX_LOG_CHARS:]
    elapsed = round(time.perf_counter() - t0, 1)
    if proc.returncode != 0 or not result.get("onnx"):
        err = result.get("error") or f"exit code {proc.returncode}"
        return {"ok": False, "error": err, "log": log, "seconds": elapsed}

    onnx = Path(result["onnx"])
    src_m = pt.stat().st_mtime
    res: dict[str, Any] = {
        "ok": True,
        "seconds": elapsed,
        "source": str(pt),
        "options": args,
        "onnx": _file_entry(onnx, src_m),
        "verify": _verify_onnx(onnx),
        "log": log,
    }
    if result.get("openvino"):
        res["openvino"] = _file_entry(Path(result["openvino"]), src_m)
    if result.get("openvino_error"):
        res["openvino_error"] = result["openvino_error"]
    return res


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
def build_router() -> APIRouter:
    router = APIRouter(prefix="/export", tags=["UI: model export"])

    @router.get("/inspect")
    def inspect(path: str = Query(...)):
        """Checkpoint info + any exports that already sit next to it."""
        pt = _resolve_pt(path)
        return {"model": _pt_info(pt), "existing": _existing_exports(pt), "busy": bool(_running)}

    @router.get("/status")
    def status():
        """Whether an export is running right now."""
        return {"busy": bool(_running), "running": dict(_running) or None}

    @router.post("/onnx")
    async def export_onnx(body: ExportBody):
        """Convert a .pt to ONNX (written next to it as <name>.onnx, replacing an older one)."""
        pt = _resolve_pt(body.path)
        device = body.device.strip() or "cpu"
        if body.half and not device.startswith("cuda"):
            raise HTTPException(400, "FP16 (half) export needs a CUDA device — choose a GPU or turn FP16 off.")
        if body.half and body.dynamic:
            raise HTTPException(400, "FP16 and dynamic axes can't be combined — turn one of them off.")
        if not _lock.acquire(blocking=False):
            raise HTTPException(409, f"Another export is running ({_running.get('path', '?')}). Try again when it finishes.")
        try:
            _running.update(path=str(pt), started=time.strftime("%Y-%m-%d %H:%M:%S"))
            logger.info("Exporting %s to ONNX%s", pt, " + OpenVINO" if body.openvino else "")
            res = await run_in_threadpool(_run_export, pt, body.model_copy(update={"device": device}))
            res["existing"] = _existing_exports(pt)
            return res
        finally:
            _running.clear()
            _lock.release()

    @router.get("/download")
    def download(path: str = Query(...)):
        """Download an exported .onnx file, or an OpenVINO folder as a zip."""
        p = Path(path)
        if not p.is_absolute():
            p = PROJECT_DIR / p
        if p.is_file() and p.suffix.lower() == ".onnx":
            return FileResponse(p, filename=p.name, media_type="application/octet-stream")
        if p.is_dir() and p.name.endswith("_openvino_model"):
            buf = io.BytesIO()
            with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
                for f in sorted(p.rglob("*")):
                    if f.is_file():
                        zf.write(f, f"{p.name}/{f.relative_to(p).as_posix()}")
            buf.seek(0)
            return StreamingResponse(buf, media_type="application/zip",
                                     headers={"Content-Disposition": f'attachment; filename="{p.name}.zip"'})
        raise HTTPException(404, "Only exported .onnx files or *_openvino_model folders can be downloaded.")

    return router


# ---------------------------------------------------------------------------
# Subprocess entry point: python -m api_model_export '<json args>'
# ---------------------------------------------------------------------------
def _main(argv: list[str]) -> int:
    a = json.loads(argv[1])
    from ultralytics import YOLO

    out: dict[str, Any] = {}
    try:
        kw: dict[str, Any] = {"format": "onnx", "dynamic": a["dynamic"], "simplify": a["simplify"],
                              "half": a["half"], "batch": a["batch"], "device": a["device"]}
        if a.get("imgsz"):
            kw["imgsz"] = a["imgsz"]
        if a.get("opset"):
            kw["opset"] = a["opset"]
        # A fresh YOLO instance per format: fusing etc. from one export must not leak into the next.
        out["onnx"] = str(YOLO(a["path"]).export(**kw))
        if a.get("openvino"):
            try:
                ov_kw = {"format": "openvino", "dynamic": a["dynamic"], "half": a["half"], "batch": a["batch"]}
                if a.get("imgsz"):
                    ov_kw["imgsz"] = a["imgsz"]
                out["openvino"] = str(YOLO(a["path"]).export(**ov_kw))
            except Exception as exc:
                out["openvino_error"] = str(exc)
    except Exception as exc:
        import traceback
        traceback.print_exc()
        out["error"] = str(exc)
    print("__EXPORT_RESULT__" + json.dumps(out), flush=True)
    return 0 if out.get("onnx") else 1


if __name__ == "__main__":
    raise SystemExit(_main(sys.argv))
