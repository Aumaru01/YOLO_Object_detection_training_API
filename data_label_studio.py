"""
Label Studio integration
========================
Convert a YOLO dataset (as seen by data_manager) into Label Studio tasks
with pre-filled rectangle annotations, so labels can be reviewed / corrected
in Label Studio and exported back as YOLO (which this project imports as a
``flat`` dataset).

Images must be reachable by the browser that opens Label Studio. Two modes:

``url``          tasks point at this API (GET /datasets/file?...) — works with any
                 Label Studio that can reach this server over the network.
``local_files``  tasks use Label Studio's local-file serving
                 (``/data/local-files/?d=<path relative to its document root>``) —
                 for a Label Studio running on a machine that can see the same
                 files (LOCAL_FILES_SERVING_ENABLED=true,
                 LOCAL_FILES_DOCUMENT_ROOT=<document_root>).
"""

from __future__ import annotations

import io
import json
import logging
import os
import zipfile

from pathlib import Path
from typing import Any, Optional
from urllib.parse import quote, urlencode
from xml.sax.saxutils import quoteattr

from data_manager import Dataset, DatasetError, Folder, image_size, list_images

logger = logging.getLogger("label_studio")

# Distinct box colours for the labeling interface (cycled for >24 classes;
# Label Studio shows the class name on every region, so colour isn't the only cue).
_COLORS = [
    "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948",
    "#0e9aa7", "#b5651d", "#8e44ad", "#7f8c8d", "#c0392b", "#16a085", "#d35400", "#2c3e50",
    "#f39c12", "#27ae60", "#9b59b6", "#34495e", "#e67e22", "#1abc9c", "#3498db", "#e74c3c",
]


def label_config(class_names: list[str]) -> str:
    labels = "\n".join(
        f'    <Label value={quoteattr(n)} background="{_COLORS[i % len(_COLORS)]}"/>'
        for i, n in enumerate(class_names)
    )
    return (
        "<View>\n"
        '  <Image name="image" value="$image" zoom="true" zoomControl="true" rotateControl="false"/>\n'
        '  <RectangleLabels name="label" toName="image">\n'
        f"{labels}\n"
        "  </RectangleLabels>\n"
        "</View>\n"
    )


def _image_ref(
    ds: Dataset, folder: Folder, rel: str, mode: str,
    api_base_url: str, document_root: Optional[str],
) -> str:
    if mode == "url":
        from data_manager import get_root
        q = urlencode({"ds": ds.id, "folder": folder.key, "path": rel, "root": str(get_root())})
        return f"{api_base_url.rstrip('/')}/datasets/file?{q}"
    if mode == "local_files":
        if not document_root:
            raise DatasetError("document_root is required for local_files mode.")
        img = Path(os.path.abspath(folder.images_dir / rel))
        root = Path(os.path.abspath(document_root))
        if not img.is_relative_to(root):
            # Symlinked images (preprocessed datasets) may live elsewhere — try their target.
            img = img.resolve()
            if not img.is_relative_to(root.resolve()):
                raise DatasetError(
                    f"Image '{img}' is not under document_root '{document_root}'. Set document_root "
                    "to a folder containing the dataset (Label Studio's LOCAL_FILES_DOCUMENT_ROOT)."
                )
            root = root.resolve()
        return "/data/local-files/?d=" + quote(img.relative_to(root).as_posix())
    raise DatasetError("mode must be 'url' or 'local_files'.")


def build_tasks(
    ds: Dataset,
    folders: Optional[list[str]] = None,
    mode: str = "url",
    api_base_url: str = "",
    document_root: Optional[str] = None,
    as_predictions: bool = False,
    include_unlabeled: bool = True,
) -> list[dict[str, Any]]:
    keys = folders or [f.key for f in ds.folders if "/" not in f.key]
    names = ds.class_names
    tasks: list[dict[str, Any]] = []
    for key in keys:
        folder = ds.folder(key)
        page = list_images(ds, key, 0, 10 ** 9)
        for item in page["items"]:
            rel = item["path"]
            if not item["has_label"] and not include_unlabeled:
                continue
            img = folder.images_dir / rel
            try:
                width, height = image_size(img)
            except Exception as exc:
                logger.warning("Skipping unreadable image %s: %s", img, exc)
                continue
            results = []
            for i, b in enumerate(item["boxes"]):
                cls = b["cls"]
                label = names[cls] if 0 <= cls < len(names) else f"class_{cls}"
                results.append({
                    "id": f"r{i}",
                    "from_name": "label",
                    "to_name": "image",
                    "type": "rectanglelabels",
                    "original_width": width,
                    "original_height": height,
                    "image_rotation": 0,
                    "value": {
                        "x": max(0.0, (b["x"] - b["w"] / 2) * 100),
                        "y": max(0.0, (b["y"] - b["h"] / 2) * 100),
                        "width": b["w"] * 100,
                        "height": b["h"] * 100,
                        "rotation": 0,
                        "rectanglelabels": [label],
                    },
                })
            task: dict[str, Any] = {
                "data": {
                    "image": _image_ref(ds, folder, rel, mode, api_base_url, document_root),
                    "dataset": ds.id,
                    "split": key,
                    "file": rel,
                },
            }
            if item["has_label"]:
                block = {"result": results}
                if as_predictions:
                    task["predictions"] = [{"model_version": f"yolo:{ds.id}", "result": results}]
                else:
                    task["annotations"] = [block]
            tasks.append(task)
    return tasks


def export_zip(ds: Dataset, tasks: list[dict[str, Any]], mode: str, document_root: Optional[str]) -> bytes:
    readme = f"""Label Studio export of '{ds.id}'
{len(tasks)} tasks, {len(ds.class_names)} classes.

How to import into Label Studio
-------------------------------
1. Create a project. In Settings > Labeling Interface > Code, paste label_config.xml.
2. Data Import > Upload Files > choose tasks.json.
"""
    if mode == "url":
        readme += """
Images are referenced by URL on the training API server, so the browser you
use for Label Studio must be able to reach that server (and if Label Studio is
served over HTTPS, the API must be too, or the browser blocks the images).
"""
    else:
        readme += f"""
Images use Label Studio local-file serving. Start Label Studio with:
    LOCAL_FILES_SERVING_ENABLED=true
    LOCAL_FILES_DOCUMENT_ROOT={document_root}
and add a Local Files storage in the project (Settings > Cloud Storage) whose
path is (or contains) the dataset's image folder. Don't sync it — the tasks
from tasks.json already reference the files.
"""
    readme += """
When labeling is done, export from Label Studio as "YOLO" and import that zip in
the training UI (Datasets > Import) to train on the corrected labels.
"""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("tasks.json", json.dumps(tasks, ensure_ascii=False, indent=1))
        zf.writestr("label_config.xml", label_config(ds.class_names))
        zf.writestr("classes.txt", "\n".join(ds.class_names) + "\n")
        zf.writestr("README.txt", readme)
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Push straight into a Label Studio instance via its REST API
# ---------------------------------------------------------------------------
def _auth_header(session: Any, base: str, token: str) -> dict[str, str]:
    """Legacy tokens use 'Token <t>'; personal access tokens (JWT refresh
    tokens, Label Studio >= 1.17) are exchanged for a short-lived access token."""
    token = token.strip()
    if token.startswith("eyJ"):
        r = session.post(f"{base}/api/token/refresh", json={"refresh": token}, timeout=30)
        if r.status_code != 200:
            raise DatasetError(f"Label Studio rejected the personal access token ({r.status_code}): {r.text[:300]}")
        return {"Authorization": f"Bearer {r.json()['access']}"}
    return {"Authorization": f"Token {token}"}


def push(
    ds: Dataset,
    tasks: list[dict[str, Any]],
    ls_url: str,
    token: str,
    title: str,
    mode: str,
    storage_path: Optional[str] = None,
) -> dict[str, Any]:
    import requests

    base = ls_url.rstrip("/")
    if not base.startswith(("http://", "https://")):
        raise DatasetError("Label Studio URL must start with http:// or https://")
    s = requests.Session()
    try:
        s.headers.update(_auth_header(s, base, token))
        r = s.post(
            f"{base}/api/projects",
            json={"title": title[:50], "label_config": label_config(ds.class_names),
                  "description": f"Imported from YOLO dataset '{ds.id}'"},
            timeout=60,
        )
        if r.status_code not in (200, 201):
            raise DatasetError(f"Creating the Label Studio project failed ({r.status_code}): {r.text[:300]}", 502)
        project = r.json()
        pid = project["id"]

        warnings: list[str] = []
        if mode == "local_files" and storage_path:
            r = s.post(
                f"{base}/api/storages/localfiles",
                json={"project": pid, "path": storage_path, "use_blob_urls": True,
                      "title": f"{ds.name} images", "regex_filter": ""},
                timeout=60,
            )
            if r.status_code not in (200, 201):
                warnings.append(
                    f"Couldn't add the Local Files storage ({r.status_code}: {r.text[:200]}). Add it in "
                    "Project Settings > Cloud Storage so Label Studio may serve the images."
                )

        for i in range(0, len(tasks), 500):
            chunk = tasks[i: i + 500]
            r = s.post(f"{base}/api/projects/{pid}/import", json=chunk, timeout=600)
            if r.status_code not in (200, 201):
                raise DatasetError(
                    f"Project {pid} was created but importing tasks {i}-{i + len(chunk)} failed "
                    f"({r.status_code}): {r.text[:300]}", 502,
                )
    except requests.RequestException as exc:
        raise DatasetError(f"Could not reach Label Studio at {base}: {exc}", 502)
    logger.info("Pushed %d tasks of %s to Label Studio project %s", len(tasks), ds.id, pid)
    return {"project_id": pid, "project_url": f"{base}/projects/{pid}", "tasks": len(tasks), "warnings": warnings}
