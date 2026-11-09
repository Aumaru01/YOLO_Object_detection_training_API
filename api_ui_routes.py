"""
UI routes — dataset management, preview, Label Studio, and system info
=====================================================================
JSON endpoints used by the web UI (static/, served at /ui). Dataset ids are
paths relative to the dataset root (e.g. "1.labeled/pea_all"), passed as the ``id``
query/body field because they contain slashes.
"""

from __future__ import annotations

import logging
import os
import re
import shutil
import subprocess
import uuid

from pathlib import Path
from typing import Callable, Iterable, Literal, Optional

from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

import data_manager as dm
import data_label_studio as ls
from train_backend import BASE_MODEL_DIR, preprocess_data

logger = logging.getLogger("ui_routes")

PROJECT_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = dm.CACHE_DIR / "uploads"

# Pretrained checkpoints ultralytics can download by name.
PRETRAINED_SUGGESTIONS = [
    f"{family}{size}.pt"
    for family in ("yolo26", "yolo11", "yolov8")
    for size in ("n", "s", "m", "l", "x")
]


# ---------------------------------------------------------------------------
# Request bodies
# ---------------------------------------------------------------------------
class RenameDatasetBody(BaseModel):
    id: str
    new_name: str


class RenameClassesBody(BaseModel):
    id: str
    names: list[str]


class DatasetIdBody(BaseModel):
    id: str


class ImportFolderBody(BaseModel):
    source_path: str
    name: str
    group: str = ""
    mode: Literal["copy", "symlink"] = "copy"


class RootBody(BaseModel):
    path: str


class ReindexBody(BaseModel):
    sources: list[str] = Field(..., min_length=1, description="Flat labeled datasets to merge in (e.g. 1.labeled/*).")
    output_id: Optional[str] = Field(None, description="Existing re-indexed dataset to extend (e.g. 2.re_index_label).")
    output_name: Optional[str] = None
    output_group: str = ""


class RepointBody(BaseModel):
    target_id: str
    source_id: str


class PreprocessBody(BaseModel):
    sources: list[str] = Field(..., min_length=1, description="Dataset ids of flat sources to merge in.")
    output_id: Optional[str] = Field(None, description="Existing prepared dataset to extend.")
    output_name: Optional[str] = Field(None, description="Name of a new prepared dataset.")
    output_group: str = ""
    val_ratio: float = Field(0.1, ge=0, le=1)
    test_ratio: float = Field(0.0, ge=0, le=1)
    ignore_stale: bool = Field(False, description="Proceed even if records point at moved copies of the source images.")


class LabelStudioExportBody(BaseModel):
    id: str
    folders: Optional[list[str]] = None
    mode: Literal["url", "local_files"] = "url"
    api_base_url: Optional[str] = Field(None, description="Public base URL of this API (url mode).")
    document_root: Optional[str] = Field(None, description="LOCAL_FILES_DOCUMENT_ROOT (local_files mode).")
    as_predictions: bool = False
    include_unlabeled: bool = True


class LabelStudioPushBody(LabelStudioExportBody):
    ls_url: str
    token: str
    title: str = Field(..., min_length=1, max_length=50)
    storage_path: Optional[str] = Field(None, description="Local Files storage path to register (local_files mode).")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _dataset(dataset_id: str, root: Optional[str] = None) -> dm.Dataset:
    try:
        return dm.get_dataset(dataset_id, dm.root_for(root) if root else None)
    except dm.DatasetError as exc:
        raise HTTPException(status_code=exc.status, detail=str(exc))


def _call(fn: Callable, *args, **kwargs):
    try:
        return fn(*args, **kwargs)
    except dm.DatasetError as exc:
        raise HTTPException(status_code=exc.status, detail=str(exc))
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


def _base_url(request: Request, explicit: Optional[str]) -> str:
    return (explicit or str(request.base_url)).rstrip("/")


def _gpu_status() -> list[dict]:
    if not shutil.which("nvidia-smi"):
        return []
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,name,utilization.gpu,memory.used,memory.total,temperature.gpu",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return []
    gpus = []
    for line in out.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 6:
            continue

        def num(v: str) -> Optional[float]:
            try:
                return float(v)
            except ValueError:
                return None

        gpus.append({
            "index": int(parts[0]), "name": parts[1], "util": num(parts[2]),
            "mem_used_mb": num(parts[3]), "mem_total_mb": num(parts[4]), "temp_c": num(parts[5]),
        })
    return gpus


def _meminfo() -> dict:
    info = {}
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            k, v = line.split(":", 1)
            if k in ("MemTotal", "MemAvailable"):
                info[k] = int(v.split()[0]) * 1024
    except OSError:
        pass
    return {"total": info.get("MemTotal"), "available": info.get("MemAvailable")}


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------
def build_router(busy_dataset_paths: Callable[[], Iterable[str]]) -> APIRouter:
    """``busy_dataset_paths`` returns dataset paths of queued/running jobs
    (renaming those datasets is refused)."""
    router = APIRouter(tags=["UI: datasets"])

    # -- dataset root ------------------------------------------------------
    @router.get("/settings")
    def get_settings():
        """Current dataset root and the list of known roots."""
        return dm.get_settings()

    @router.put("/settings/root")
    def set_root(body: RootBody):
        """Switch the folder the UI works in (added to the known roots if new)."""
        return _call(dm.set_root, body.path)

    @router.delete("/settings/root")
    def forget_root(path: str = Query(...)):
        """Remove a root from the known list (files are not touched)."""
        return _call(dm.forget_root, path)

    # -- listing / detail ------------------------------------------------
    @router.get("/datasets")
    def list_datasets():
        """All YOLO datasets found under the current dataset root (any depth)."""
        return {
            "base_dir": str(dm.get_root()),
            "groups": dm.list_groups(),
            "datasets": [dm.summarize(d) for d in dm.discover()],
        }

    @router.get("/datasets/info")
    def dataset_info(id: str = Query(...), refresh: bool = False):
        """Dataset summary plus the class index -> name table with instance counts."""
        ds = _dataset(id)
        if refresh:
            dm.invalidate_cache(ds.path)
        info = dm.summarize(ds)
        info.update(dm.class_stats(ds))
        return info

    # -- management --------------------------------------------------------
    @router.post("/datasets/rename")
    def rename_dataset(body: RenameDatasetBody):
        """Rename a dataset folder. Symlinks / preprocess state / data.yaml 'path' that
        reference it are updated so prepared datasets built from it keep working."""
        ds = _dataset(body.id)
        return _call(dm.rename_dataset, ds, body.new_name, busy_dataset_paths())

    @router.put("/datasets/classes")
    def rename_classes(body: RenameClassesBody):
        """Rename classes (index order unchanged; label files untouched)."""
        ds = _dataset(body.id)
        files = _call(dm.rename_classes, ds, body.names)
        return {"updated_files": files, "names": body.names}

    @router.post("/datasets/repair-yaml")
    def repair_yaml(body: DatasetIdBody):
        """Point data.yaml 'path' at the dataset's current folder."""
        ds = _dataset(body.id)
        return {"path": _call(dm.repair_yaml_path, ds)}

    # -- import --------------------------------------------------------------
    @router.post("/datasets/import/upload")
    async def import_upload(
        request: Request,
        name: str = Query(...),
        group: str = Query(""),
    ):
        """Import a YOLO dataset from a zip file sent as the raw request body
        (Content-Type: application/zip). Accepts Label Studio / Roboflow YOLO exports."""
        _call(dm.validate_name, name, "dataset name")
        _call(dm.validate_group, group)
        UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
        tmp = UPLOAD_DIR / f"{uuid.uuid4().hex}.zip"
        try:
            with open(tmp, "wb") as f:
                async for chunk in request.stream():
                    f.write(chunk)
            if tmp.stat().st_size == 0:
                raise HTTPException(status_code=400, detail="Empty upload.")
            return await run_in_threadpool(_call, dm.import_zip, tmp, name, group)
        finally:
            tmp.unlink(missing_ok=True)

    @router.post("/datasets/import/folder")
    def import_folder(body: ImportFolderBody):
        """Import a dataset folder already on the server (copied, or symlinked)."""
        return _call(dm.import_folder, body.source_path, body.name, body.group, body.mode)

    # -- re-index (1.labeled/* -> 2.re_index_label) -------------------------
    def _new_output(name: Optional[str], group: str) -> Path:
        name = _call(dm.validate_name, name or "", "dataset name")
        group = _call(dm.validate_group, group)
        root = dm.get_root()
        output = root / group / name if group else root / name
        if output.exists():
            raise HTTPException(409, f"'{output.relative_to(root)}' already exists.")
        return output

    @router.post("/datasets/reindex")
    def reindex(body: ReindexBody):
        """Merge labeled exports into one flat dataset with a single class list,
        remapping class ids by name. Re-running skips images already merged."""
        sources = [_dataset(sid) for sid in body.sources]
        if body.output_id:
            target = _dataset(body.output_id)
            if target.kind != "flat":
                raise HTTPException(400, f"'{body.output_id}' is a prepared (split) dataset — pick a flat re-index folder.")
            output = target.path
        elif body.output_name:
            output = _new_output(body.output_name, body.output_group)
        else:
            raise HTTPException(400, "Give output_id (existing) or output_name (new).")
        result = _call(dm.reindex_merge, sources, output)
        dm.record_run(output, "reindexed", {
            "action": "reindex",
            "sources": [r["source"] for r in result["sources"]],
            "added": result["added"],
            "new_classes": result["new_classes"],
        })
        return result

    @router.post("/datasets/repoint-state")
    def repoint_state(body: RepointBody):
        """Point a prepared dataset's preprocess records at this source's images
        (matched by file name), so re-running the split doesn't add duplicates."""
        target, source = _dataset(body.target_id), _dataset(body.source_id)
        return _call(dm.repoint_state, target.path, source.path)

    # -- preprocess (merge flat sources into a trainable split dataset) -----
    @router.post("/datasets/preprocess")
    def preprocess(body: PreprocessBody):
        """Merge one or more flat sources into a new or existing prepared dataset
        (wraps POST /preprocess_data with dataset ids)."""
        sources = []
        for sid in body.sources:
            ds = _dataset(sid)
            if ds.kind != "flat":
                raise HTTPException(400, f"'{sid}' is already a prepared (split) dataset — only flat sources can be merged in.")
            sources.append(ds.path)
        if body.output_id:
            target = _dataset(body.output_id)
            if not (target.path / ".preprocess_state.json").is_file():
                raise HTTPException(
                    400, f"'{body.output_id}' wasn't built by preprocessing — pick a dataset created here, or a new name.",
                )
            output = target.path
            if not body.ignore_stale:
                stale = dm.stale_state_entries(output, sources)
                if stale["count"]:
                    raise HTTPException(409, detail={
                        "code": "stale_state",
                        "message": (
                            f"{stale['count']} images already in '{body.output_id}' were split from another "
                            f"folder (e.g. {stale['examples'][0]}) that has files with the same names as the "
                            "selected source. Splitting now would add them again as duplicates. If they are the "
                            "same images (e.g. a local copy of the share), repoint the records to this source first."
                        ),
                        "count": stale["count"],
                    })
        elif body.output_name:
            output = _new_output(body.output_name, body.output_group)
        else:
            raise HTTPException(400, "Give output_id (existing) or output_name (new).")
        result = _call(preprocess_data, sources, output, val_ratio=body.val_ratio, test_ratio=body.test_ratio)
        dm.invalidate_cache(output)
        a = result["added"]
        dm.record_run(output, "prepared", {
            "action": "split",
            "sources": [str(dm._rel_id(p)) for p in sources],
            "ratio": {"train": round(1 - body.val_ratio - body.test_ratio, 4), "val": body.val_ratio, "test": body.test_ratio},
            "added": {"train": a["train"], "val": a["val"], "test": a["test"]},
            "new_classes": a["new_classes"],
        }, split_ratio={"train": round(1 - body.val_ratio - body.test_ratio, 4), "val": body.val_ratio, "test": body.test_ratio})
        result["id"] = dm._rel_id(output)
        return result

    @router.get("/datasets/details")
    def dataset_details(id: str = Query(...)):
        """How a folder was built: summary, split counts and ratios (configured
        and actual), source folders, and run history from dataset_info.json."""
        return _call(dm.folder_details, _dataset(id))

    # -- preview -------------------------------------------------------------
    @router.get("/datasets/images")
    def dataset_images(
        id: str = Query(...),
        folder: str = Query(...),
        offset: int = Query(0, ge=0),
        limit: int = Query(60, ge=1, le=500),
        class_id: Optional[int] = Query(None),
        q: str = Query(""),
        unlabeled: bool = Query(False, description="Only images without a label file or with no boxes."),
    ):
        """A page of images in a dataset folder, each with its parsed YOLO boxes."""
        ds = _dataset(id)
        return _call(dm.list_images, ds, folder, offset, limit, class_id, q, unlabeled)

    @router.get("/datasets/file")
    def dataset_file(
        ds: str = Query(..., description="Dataset id"),
        folder: str = Query(...),
        path: str = Query(...),
        thumb: bool = Query(False),
        root: Optional[str] = Query(None, description="Dataset root (a configured one); default: current."),
    ):
        """Serve an image (or a cached thumbnail) from a dataset folder.
        Also used as the image URL inside Label Studio tasks."""
        dataset = _dataset(ds, root)
        img = _call(dm.image_path, dataset, folder, path)
        if thumb:
            try:
                return FileResponse(dm.thumbnail(img), media_type="image/jpeg",
                                    headers={"Cache-Control": "max-age=86400"})
            except Exception as exc:
                logger.warning("Thumbnail failed for %s: %s", img, exc)
        return FileResponse(img, headers={"Cache-Control": "max-age=3600"})

    # -- Label Studio ----------------------------------------------------------
    @router.post("/datasets/labelstudio/export")
    def labelstudio_export(body: LabelStudioExportBody, request: Request):
        """Download a zip with tasks.json (pre-annotated) + label_config.xml for Label Studio."""
        ds = _dataset(body.id)
        tasks = _call(
            ls.build_tasks, ds, body.folders, body.mode, _base_url(request, body.api_base_url),
            body.document_root, body.as_predictions, body.include_unlabeled,
        )
        data = ls.export_zip(ds, tasks, body.mode, body.document_root)
        fname = re.sub(r"[^A-Za-z0-9_.-]+", "_", ds.id) + "_labelstudio.zip"
        return Response(
            data, media_type="application/zip",
            headers={"Content-Disposition": f'attachment; filename="{fname}"', "X-Task-Count": str(len(tasks))},
        )

    @router.post("/datasets/labelstudio/push")
    def labelstudio_push(body: LabelStudioPushBody, request: Request):
        """Create a Label Studio project (with matching labeling config) and import the tasks."""
        ds = _dataset(body.id)
        tasks = _call(
            ls.build_tasks, ds, body.folders, body.mode, _base_url(request, body.api_base_url),
            body.document_root, body.as_predictions, body.include_unlabeled,
        )
        return _call(ls.push, ds, tasks, body.ls_url, body.token, body.title, body.mode, body.storage_path)

    @router.get("/datasets/labelstudio/config")
    def labelstudio_config(id: str = Query(...)):
        """The Label Studio labeling-interface XML for a dataset's classes."""
        return Response(ls.label_config(_dataset(id).class_names), media_type="application/xml")

    # -- models & system -------------------------------------------------------
    @router.get("/models/available", tags=["UI: system"])
    def available_models():
        """Local .pt weights (project root + trained models) and downloadable pretrained names."""
        local = [{"value": p.name, "label": p.name, "kind": "local"} for p in sorted(PROJECT_DIR.glob("*.pt"))]
        trained = []
        if BASE_MODEL_DIR.is_dir():
            for p in sorted(BASE_MODEL_DIR.glob("*/**/weights/*.pt"), key=lambda p: p.stat().st_mtime, reverse=True):
                rel = p.relative_to(BASE_MODEL_DIR)
                trained.append({"value": str(p), "label": rel.as_posix(), "kind": "trained"})
        local_names = {m["value"] for m in local}
        pretrained = [{"value": n, "label": n, "kind": "pretrained"} for n in PRETRAINED_SUGGESTIONS if n not in local_names]
        return {"models": local + trained + pretrained}

    @router.get("/system/status", tags=["UI: system"])
    def system_status():
        """GPU / memory / disk snapshot for the monitor."""
        root = dm.get_root()
        disk = shutil.disk_usage(root if root.exists() else PROJECT_DIR)
        try:
            load = os.getloadavg()
        except OSError:
            load = None
        return {
            "gpus": _gpu_status(),
            "cpu_count": os.cpu_count(),
            "load_avg": load,
            "memory": _meminfo(),
            "disk": {"total": disk.total, "free": disk.free},
        }

    return router
