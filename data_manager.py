"""
Dataset Manager
===============
Filesystem logic behind the web UI's dataset features:

- discovery of YOLO datasets under the dataset root (default ``dataset/``, any depth)
- class-name (index -> name) mapping, instance counts, and class renaming
- dataset folder renaming (keeping symlinks / preprocess state that point at it intact)
- importing datasets from a zip upload or a server-side folder
- image listing, label parsing, and thumbnails for the preview tab

Two dataset layouts are recognised (the same two the training pipeline uses):

``split``  a folder with a ``data.yaml`` (``names`` + ``train``/``val``/``test`` splits)
           — directly trainable via POST /train_local_data.
``flat``   a flat export: ``images/`` + ``labels/`` (or ``label/``) plus class names in
           ``classes.txt`` / ``notes.json`` / ``data.yaml`` (e.g. a Label Studio YOLO
           export) — a *source* that POST /preprocess_data turns into a ``split`` dataset.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import shutil
import threading
import time
import uuid
import zipfile

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Optional

import yaml

from train_backend import (
    BASE_DATASET_DIR,
    IMAGE_EXTENSIONS,
    _img2label_path,
    _names_from_notes_json,
    _names_to_list,
)

logger = logging.getLogger("dataset_manager")

PROJECT_DIR = Path(__file__).resolve().parent
CACHE_DIR = PROJECT_DIR / ".cache"
THUMB_DIR = CACHE_DIR / "thumbs"
SETTINGS_FILE = Path(os.environ.get("YOLO_UI_SETTINGS", PROJECT_DIR / "ui_settings.json"))

PREVIEW_IMAGE_EXTENSIONS = IMAGE_EXTENSIONS
SPLIT_NAMES = ("train", "val", "test")
NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.\-]*$")
MAX_DISCOVERY_DEPTH = 5


class DatasetError(Exception):
    """User-facing error; ``status`` maps to the HTTP status code."""

    def __init__(self, message: str, status: int = 400) -> None:
        super().__init__(message)
        self.status = status


# ---------------------------------------------------------------------------
# Dataset root (chosen in the UI, persisted in ui_settings.json)
# ---------------------------------------------------------------------------
_settings_lock = threading.Lock()


def _load_settings() -> dict[str, Any]:
    default = {"dataset_root": str(BASE_DATASET_DIR), "roots": [str(BASE_DATASET_DIR)]}
    try:
        data = json.loads(SETTINGS_FILE.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return default
    roots = [r for r in data.get("roots", []) if isinstance(r, str)]
    if str(BASE_DATASET_DIR) not in roots:
        roots.insert(0, str(BASE_DATASET_DIR))
    root = data.get("dataset_root") if data.get("dataset_root") in roots else str(BASE_DATASET_DIR)
    return {"dataset_root": root, "roots": roots}


def get_settings() -> dict[str, Any]:
    with _settings_lock:
        s = _load_settings()
    s["roots"] = [{"path": r, "exists": Path(r).is_dir(), "default": r == str(BASE_DATASET_DIR)} for r in s["roots"]]
    return s


def get_root() -> Path:
    """The folder the UI currently scans for datasets (default: <project>/datasets)."""
    with _settings_lock:
        return Path(_load_settings()["dataset_root"])


def known_roots() -> list[Path]:
    with _settings_lock:
        return [Path(r) for r in _load_settings()["roots"]]


def _save_settings(data: dict[str, Any]) -> None:
    tmp = SETTINGS_FILE.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
    os.replace(tmp, SETTINGS_FILE)


def set_root(path: str) -> dict[str, Any]:
    """Switch the dataset root (adding it to the known list if new)."""
    path = (path or "").strip()
    if "\\" in path:
        raise DatasetError("Windows-style paths aren't supported — mount the share and use a Linux path.")
    if not path.startswith("/"):
        raise DatasetError("Enter an absolute path (starting with /).")
    p = _abs(Path(path).expanduser())
    if not p.is_dir():
        raise DatasetError(f"Folder '{p}' doesn't exist or isn't reachable from the server.", 404)
    with _settings_lock:
        data = _load_settings()
        if str(p) not in data["roots"]:
            data["roots"].append(str(p))
        data["dataset_root"] = str(p)
        _save_settings(data)
    invalidate_cache()
    logger.info("Dataset root set to %s", p)
    return get_settings()


def forget_root(path: str) -> dict[str, Any]:
    with _settings_lock:
        data = _load_settings()
        if path == str(BASE_DATASET_DIR):
            raise DatasetError("The project's own dataset/ folder can't be removed.")
        data["roots"] = [r for r in data["roots"] if r != path]
        if data["dataset_root"] == path:
            data["dataset_root"] = str(BASE_DATASET_DIR)
        _save_settings(data)
    return get_settings()


def root_for(path: Optional[str]) -> Path:
    """A known root by path (None -> current root). Unknown paths are refused,
    so request parameters can't widen what the file endpoints serve."""
    if not path:
        return get_root()
    p = _abs(path)
    if p not in known_roots():
        raise DatasetError(f"'{path}' is not a configured dataset root.", 400)
    return p


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------
@dataclass
class Folder:
    """A browsable image folder of a dataset (a split, or a flat images/ dir)."""
    key: str
    images_dir: Path
    labels_dir: Optional[Path]  # None for split datasets: labels derived per image

    def label_for(self, image: Path) -> Path:
        if self.labels_dir is None:
            return _img2label_path(image)
        return (self.labels_dir / image.relative_to(self.images_dir)).with_suffix(".txt")


@dataclass
class Dataset:
    id: str                      # path relative to the dataset root (posix)
    path: Path
    kind: str                    # "split" | "flat"
    class_names: list[str]
    class_source: Optional[Path]
    folders: list[Folder] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    data_yaml: Optional[Path] = None

    @property
    def name(self) -> str:
        return self.path.name

    @property
    def group(self) -> str:
        parent = Path(self.id).parent.as_posix()
        return "" if parent == "." else parent

    def folder(self, key: str) -> Folder:
        for f in self.folders:
            if f.key == key:
                return f
        raise DatasetError(f"Dataset '{self.id}' has no folder '{key}'.", 404)


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------
def _is_hidden(p: Path) -> bool:
    return p.name.startswith(".") or p.name == "__pycache__"


def _flat_labels_dir(d: Path) -> Optional[Path]:
    for name in ("labels", "label"):
        if (d / name).is_dir():
            return d / name
    return None


def is_dataset_dir(d: Path) -> bool:
    return (d / "data.yaml").is_file() or ((d / "images").is_dir() and _flat_labels_dir(d) is not None)


def _list_images(images_dir: Path) -> list[Path]:
    if not images_dir.is_dir():
        return []
    out: list[Path] = []
    for root, dirs, files in os.walk(images_dir, followlinks=True):
        dirs[:] = [d for d in dirs if not d.startswith(".")]
        for f in files:
            if os.path.splitext(f)[1].lower() in PREVIEW_IMAGE_EXTENSIONS:
                out.append(Path(root, f))
    out.sort()
    return out


def _count_images(images_dir: Path) -> int:
    return len(_list_images(images_dir))


def validate_name(name: str, what: str = "name") -> str:
    name = (name or "").strip()
    if not NAME_RE.match(name) or len(name) > 120:
        raise DatasetError(
            f"Invalid {what} '{name}': use letters, digits, '_', '-', '.', starting with a letter or digit."
        )
    return name


def validate_group(group: str) -> str:
    """A relative folder path under the dataset root (e.g. '1.labeled'); '' means the root."""
    group = (group or "").strip().strip("/")
    if not group:
        return ""
    for part in group.split("/"):
        validate_name(part, "group folder")
    return group


def _abs(path: Path | str) -> Path:
    """Absolute, normalized path WITHOUT resolving symlinks (a dataset may be a
    symlink to a folder elsewhere — it still lives at its location under the root)."""
    return Path(os.path.abspath(path))


def _rel_id(path: Path, root: Optional[Path] = None) -> str:
    return _abs(path).relative_to(root or get_root()).as_posix()


def _read_yaml(path: Path) -> dict[str, Any]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except (OSError, yaml.YAMLError) as exc:
        raise DatasetError(f"Could not read '{path}': {exc}")


def _write_yaml(path: Path, data: dict[str, Any]) -> None:
    tmp = path.with_suffix(".yaml.tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True)
    os.replace(tmp, path)


# ---------------------------------------------------------------------------
# Loading a dataset
# ---------------------------------------------------------------------------
def _resolve_split_dir(root: Path, yaml_dir: Path, rel: str) -> Path:
    """Resolve a data.yaml split entry like ultralytics does (incl. Roboflow's '../x')."""
    p = Path(rel)
    if p.is_absolute():
        return p
    candidate = root / p
    if not candidate.exists() and rel.startswith("../"):
        alt = root / rel[3:]
        if alt.exists():
            return alt
    return candidate


def _load_split_dataset(d: Path) -> Dataset:
    data_yaml = d / "data.yaml"
    config = _read_yaml(data_yaml)
    warnings: list[str] = []

    names = _names_to_list(config["names"]) if config.get("names") is not None else []
    if not names:
        warnings.append("data.yaml has no 'names' — class names unknown.")

    root = Path(config["path"]) if config.get("path") else d
    if not root.is_absolute():
        root = d / root
    if not root.exists():
        warnings.append(
            f"data.yaml 'path' points to '{config.get('path')}', which doesn't exist on this "
            "machine — previewing relative to the dataset folder. Use 'Repair data.yaml path' "
            "before training on it."
        )
        root = d
    elif _abs(root) != _abs(d) and not _abs(root).is_relative_to(_abs(d)):
        warnings.append(
            f"data.yaml 'path' points outside this folder, to '{root}' — previews and training "
            "read images from there. Use 'Repair data.yaml path' to use this folder instead."
        )

    folders: list[Folder] = []
    for split in SPLIT_NAMES:
        rel = config.get(split)
        if not rel:
            continue
        if isinstance(rel, list):
            rel = rel[0]
        split_dir = _resolve_split_dir(root, d, str(rel))
        if split_dir.is_file():
            warnings.append(f"Split '{split}' is a manifest file — not browsable in preview.")
            continue
        if not split_dir.exists():
            warnings.append(f"Split '{split}' folder '{split_dir}' not found.")
            continue
        folders.append(Folder(split, split_dir, None))
    if "train" not in config:
        warnings.append("data.yaml defines no 'train' split — not trainable.")

    return Dataset(
        id="", path=d, kind="split", class_names=names,
        class_source=data_yaml, folders=folders, warnings=warnings, data_yaml=data_yaml,
    )


def _load_flat_dataset(d: Path) -> Dataset:
    warnings: list[str] = []
    names: list[str] = []
    source: Optional[Path] = None
    if (d / "classes.txt").is_file():
        source = d / "classes.txt"
        names = [l.strip() for l in source.read_text(encoding="utf-8").splitlines() if l.strip()]
    elif (d / "notes.json").is_file():
        source = d / "notes.json"
        names = _names_from_notes_json(source) or []
    if not names:
        warnings.append("No class names found (expected classes.txt or notes.json).")

    labels_dir = _flat_labels_dir(d)
    folders = [Folder("images", d / "images", labels_dir)]
    for sub in sorted(p for p in (d / "images").iterdir() if p.is_dir() and not _is_hidden(p)):
        folders.append(Folder(f"images/{sub.name}", sub, labels_dir / sub.name if labels_dir else None))
    return Dataset(
        id="", path=d, kind="flat", class_names=names,
        class_source=source, folders=folders, warnings=warnings,
    )


def load_dataset(d: Path, root: Optional[Path] = None) -> Dataset:
    if (d / "data.yaml").is_file():
        ds = _load_split_dataset(d)
    elif (d / "images").is_dir() and _flat_labels_dir(d) is not None:
        ds = _load_flat_dataset(d)
    else:
        raise DatasetError(f"'{d}' is not a YOLO dataset.", 404)
    ds.id = _rel_id(d, root)
    return ds
    raise DatasetError(f"'{d}' is not a YOLO dataset.", 404)


def resolve_id(dataset_id: str, root: Optional[Path] = None) -> Path:
    """Map a dataset id to its folder, refusing anything outside the root."""
    base = _abs(root or get_root())
    target = _abs(base / dataset_id)
    if target == base or not target.is_relative_to(base) or any(
        part.startswith(".") for part in target.relative_to(base).parts
    ):
        raise DatasetError(f"Invalid dataset id '{dataset_id}'.", 400)
    if not target.is_dir():
        raise DatasetError(f"Dataset '{dataset_id}' not found.", 404)
    return target


def get_dataset(dataset_id: str, root: Optional[Path] = None) -> Dataset:
    d = resolve_id(dataset_id, root)
    if not is_dataset_dir(d):
        raise DatasetError(f"'{dataset_id}' is not a YOLO dataset.", 404)
    return load_dataset(d, root)


# ---------------------------------------------------------------------------
# Discovery / listing
# ---------------------------------------------------------------------------
def discover() -> list[Dataset]:
    found: list[Dataset] = []
    root = get_root()

    def walk(d: Path, depth: int) -> None:
        if depth > MAX_DISCOVERY_DEPTH:
            return
        try:
            children = sorted(c for c in d.iterdir() if c.is_dir() and not _is_hidden(c))
        except OSError:
            return
        for c in children:
            if is_dataset_dir(c):
                try:
                    found.append(load_dataset(c, root))
                except DatasetError as exc:
                    logger.warning("Skipping dataset %s: %s", c, exc)
            else:
                walk(c, depth + 1)

    if root.is_dir():
        walk(root, 1)
    return found


def list_groups() -> list[str]:
    """Top-level folders under the dataset root that aren't datasets themselves."""
    root = get_root()
    if not root.is_dir():
        return []
    return sorted(
        c.name for c in root.iterdir()
        if c.is_dir() and not _is_hidden(c) and not is_dataset_dir(c)
    )


def summarize(ds: Dataset, counts: bool = True) -> dict[str, Any]:
    folders = []
    total = 0
    for f in ds.folders:
        n = _count_images(f.images_dir) if counts else None
        folders.append({"key": f.key, "path": str(f.images_dir), "num_images": n})
        # Sub-folders of a flat dataset's images/ are already counted in "images".
        if counts and "/" not in f.key:
            total += n or 0
    return {
        "id": ds.id,
        "name": ds.name,
        "group": ds.group,
        "path": str(ds.path),
        "kind": ds.kind,
        "trainable": ds.kind == "split" and any(f.key == "train" for f in ds.folders),
        "num_classes": len(ds.class_names),
        "num_images": total if counts else None,
        "folders": folders,
        "warnings": ds.warnings,
        "data_yaml": str(ds.data_yaml) if ds.data_yaml else None,
        "class_names": ds.class_names,
        # Built by preprocessing -> more flat sources can be merged in incrementally.
        "preprocess_target": (ds.path / ".preprocess_state.json").is_file(),
    }


# ---------------------------------------------------------------------------
# Labels
# ---------------------------------------------------------------------------
def parse_label_file(path: Path) -> list[dict[str, Any]]:
    """YOLO label -> [{cls, x, y, w, h}] in normalized centre format.
    Segmentation polygons are reduced to their bounding box."""
    boxes: list[dict[str, Any]] = []
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return boxes
    for line in text.splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        try:
            cls = int(float(parts[0]))
            vals = [float(v) for v in parts[1:]]
        except ValueError:
            continue
        if len(vals) == 4:
            x, y, w, h = vals
        else:
            xs, ys = vals[0::2], vals[1::2]
            x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)
            x, y, w, h = (x0 + x1) / 2, (y0 + y1) / 2, x1 - x0, y1 - y0
        boxes.append({"cls": cls, "x": x, "y": y, "w": w, "h": h})
    return boxes


# Cached per-folder index: [(relative image path, [class ids])]
_index_cache: dict[tuple[str, str], tuple[float, list[tuple[str, list[int], bool]]]] = {}
_index_lock = threading.Lock()
_INDEX_TTL = 300.0  # seconds; mutations through this module invalidate it immediately


def _folder_index(ds: Dataset, folder: Folder) -> list[tuple[str, list[int], bool]]:
    key = (str(ds.path), folder.key)
    now = time.time()
    with _index_lock:
        hit = _index_cache.get(key)
        if hit and now - hit[0] < _INDEX_TTL:
            return hit[1]
    index = []
    for img in _list_images(folder.images_dir):
        label = folder.label_for(img)
        has_label = label.exists()
        classes = [b["cls"] for b in parse_label_file(label)] if has_label else []
        index.append((img.relative_to(folder.images_dir).as_posix(), classes, has_label))
    with _index_lock:
        _index_cache[key] = (now, index)
    return index


def invalidate_cache(path: Optional[Path] = None) -> None:
    with _index_lock:
        if path is None:
            _index_cache.clear()
        else:
            for k in [k for k in _index_cache if k[0].startswith(str(path))]:
                del _index_cache[k]


def class_stats(ds: Dataset) -> dict[str, Any]:
    """Per-class instance/image counts across all top-level folders of a dataset."""
    instances = [0] * len(ds.class_names)
    images = [0] * len(ds.class_names)
    unknown: dict[int, int] = {}
    unlabeled = 0
    background = 0
    for f in ds.folders:
        if "/" in f.key:
            continue
        for _, classes, has_label in _folder_index(ds, f):
            if not has_label:
                unlabeled += 1
                continue
            if not classes:
                background += 1
            for c in classes:
                if 0 <= c < len(instances):
                    instances[c] += 1
                else:
                    unknown[c] = unknown.get(c, 0) + 1
            for c in set(classes):
                if 0 <= c < len(images):
                    images[c] += 1
    return {
        "classes": [
            {"index": i, "name": n, "instances": instances[i], "images": images[i]}
            for i, n in enumerate(ds.class_names)
        ],
        "unknown_class_ids": {str(k): v for k, v in sorted(unknown.items())},
        "images_without_label": unlabeled,
        "background_images": background,
    }


def list_images(
    ds: Dataset, folder_key: str, offset: int = 0, limit: int = 60,
    class_id: Optional[int] = None, search: str = "", only_unlabeled: bool = False,
) -> dict[str, Any]:
    folder = ds.folder(folder_key)
    index = _folder_index(ds, folder)
    if class_id is not None:
        index = [e for e in index if class_id in e[1]]
    if only_unlabeled:
        index = [e for e in index if not e[2] or not e[1]]
    if search:
        s = search.lower()
        index = [e for e in index if s in e[0].lower()]
    page = index[offset: offset + limit]
    items = []
    for rel, _, has_label in page:
        img = folder.images_dir / rel
        items.append({
            "path": rel,
            "has_label": has_label,
            "boxes": parse_label_file(folder.label_for(img)) if has_label else [],
        })
    return {"total": len(index), "offset": offset, "limit": limit, "items": items}


def image_path(ds: Dataset, folder_key: str, rel: str) -> Path:
    """Resolve an image inside a dataset folder, refusing path traversal.
    Symlinks are allowed (preprocessed datasets symlink their images)."""
    folder = ds.folder(folder_key)
    base = os.path.normpath(folder.images_dir)
    candidate = os.path.normpath(os.path.join(base, rel))
    if not candidate.startswith(base + os.sep):
        raise DatasetError("Invalid image path.", 400)
    p = Path(candidate)
    if p.suffix.lower() not in PREVIEW_IMAGE_EXTENSIONS or not p.is_file():
        raise DatasetError(f"Image '{rel}' not found.", 404)
    return p


def thumbnail(src: Path, size: int = 360) -> Path:
    """Cached JPEG thumbnail (EXIF orientation applied, like browsers/YOLO do)."""
    from PIL import Image, ImageOps

    st = src.stat()
    key = hashlib.sha1(f"{src}|{st.st_mtime_ns}|{st.st_size}|{size}".encode()).hexdigest()
    out = THUMB_DIR / key[:2] / f"{key}.jpg"
    if out.exists():
        return out
    out.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(src) as im:
        im = ImageOps.exif_transpose(im)
        im.thumbnail((size, size))
        if im.mode not in ("RGB", "L"):
            im = im.convert("RGB")
        tmp = out.with_suffix(f".{uuid.uuid4().hex}.tmp")
        im.save(tmp, "JPEG", quality=82)
        os.replace(tmp, out)
    return out


def image_size(src: Path) -> tuple[int, int]:
    """(width, height) as displayed, i.e. after EXIF orientation."""
    from PIL import Image

    with Image.open(src) as im:
        w, h = im.size
        try:
            orientation = im.getexif().get(0x0112, 1)
        except Exception:
            orientation = 1
    return (h, w) if orientation in (5, 6, 7, 8) else (w, h)


# ---------------------------------------------------------------------------
# Class renaming
# ---------------------------------------------------------------------------
def rename_classes(ds: Dataset, new_names: list[str]) -> list[str]:
    """Rename classes in place (index order unchanged — label files untouched).
    Updates every class-name file present at the dataset root."""
    new_names = [str(n).strip() for n in new_names]
    if len(new_names) != len(ds.class_names):
        raise DatasetError(
            f"Expected {len(ds.class_names)} names (one per class index); got {len(new_names)}. "
            "Adding/removing classes would require re-indexing labels."
        )
    if any(not n for n in new_names):
        raise DatasetError("Class names cannot be empty.")
    if len(set(new_names)) != len(new_names):
        raise DatasetError("Class names must be unique within a dataset.")

    touched: list[str] = []
    d = ds.path
    yaml_path = d / "data.yaml"
    if yaml_path.is_file():
        config = _read_yaml(yaml_path)
        if isinstance(config.get("names"), dict):
            keys = sorted(config["names"], key=int)
            config["names"] = {k: n for k, n in zip(keys, new_names)}
        else:
            config["names"] = new_names
        if "nc" in config:
            config["nc"] = len(new_names)
        _write_yaml(yaml_path, config)
        touched.append("data.yaml")
    if (d / "classes.txt").is_file():
        (d / "classes.txt").write_text("\n".join(new_names) + "\n", encoding="utf-8")
        touched.append("classes.txt")
    notes = d / "notes.json"
    if notes.is_file():
        try:
            data = json.loads(notes.read_text(encoding="utf-8"))
            cats = data.get("categories")
            if isinstance(cats, list) and len(cats) == len(new_names):
                if all(isinstance(c, str) for c in cats):
                    data["categories"] = list(new_names)
                else:
                    ordered = sorted(cats, key=lambda c: c.get("id", 0))
                    for c, n in zip(ordered, new_names):
                        c["name"] = n
                notes.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
                touched.append("notes.json")
        except (json.JSONDecodeError, OSError, AttributeError):
            logger.warning("Could not update %s", notes)
    if not touched:
        raise DatasetError("No class-name file found to update.")
    logger.info("Renamed classes of %s (%s)", ds.id, ", ".join(touched))
    return touched


# ---------------------------------------------------------------------------
# Dataset renaming
# ---------------------------------------------------------------------------
def _retarget_references(old: Path, new: Path) -> dict[str, int]:
    """After moving old -> new, repoint symlinks under the dataset root and the
    .preprocess_state.json entries that referenced files under ``old``."""
    old_s, new_s = str(old), str(new)
    links = states = 0
    for root, dirs, files in os.walk(get_root()):
        dirs[:] = [d for d in dirs if d != ".cache"]
        for name in files + dirs:
            p = Path(root, name)
            if not p.is_symlink():
                continue
            target = os.readlink(p)
            if target == old_s or target.startswith(old_s + os.sep):
                p.unlink()
                p.symlink_to(new_s + target[len(old_s):])
                links += 1
        if ".preprocess_state.json" in files:
            sp = Path(root, ".preprocess_state.json")
            try:
                data = json.loads(sp.read_text(encoding="utf-8"))
                imgs = data.get("processed_images", [])
                changed = [
                    new_s + i[len(old_s):] if (i == old_s or i.startswith(old_s + os.sep)) else i
                    for i in imgs
                ]
                if changed != imgs:
                    data["processed_images"] = sorted(changed)
                    sp.write_text(json.dumps(data, indent=2), encoding="utf-8")
                    states += 1
            except (OSError, json.JSONDecodeError):
                logger.warning("Could not update %s", sp)
    return {"symlinks_updated": links, "state_files_updated": states}


def rename_dataset(ds: Dataset, new_name: str, busy_paths: Iterable[str] = ()) -> dict[str, Any]:
    new_name = validate_name(new_name, "dataset name")
    old = _abs(ds.path)
    new = old.with_name(new_name)
    if new == old:
        raise DatasetError("The new name is the same as the current one.")
    if new.exists():
        raise DatasetError(f"'{new_name}' already exists in '{ds.group or 'datasets'}'.", 409)
    for bp in busy_paths:
        try:
            if _abs(bp).is_relative_to(old) or Path(bp).resolve().is_relative_to(old.resolve()):
                raise DatasetError("A queued or running training job uses this dataset — rename it later.", 409)
        except OSError:
            continue

    os.rename(old, new)
    refs = _retarget_references(old, new)

    yaml_path = new / "data.yaml"
    if yaml_path.is_file():
        config = _read_yaml(yaml_path)
        p = config.get("path")
        if p and Path(p).is_absolute() and (Path(p) == old or str(p).startswith(str(old) + os.sep)):
            config["path"] = str(new) + str(p)[len(str(old)):]
            _write_yaml(yaml_path, config)
            refs["data_yaml_path_updated"] = True

    invalidate_cache()
    logger.info("Renamed dataset %s -> %s (%s)", old, new, refs)
    return {"id": _rel_id(new), **refs}


def repair_yaml_path(ds: Dataset) -> str:
    """Point data.yaml's 'path' at the dataset's current folder."""
    if ds.data_yaml is None:
        raise DatasetError("Only datasets with a data.yaml have a 'path' to repair.")
    config = _read_yaml(ds.data_yaml)
    config["path"] = str(_abs(ds.path))
    _write_yaml(ds.data_yaml, config)
    invalidate_cache(ds.path)
    return config["path"]


# ---------------------------------------------------------------------------
# Import
# ---------------------------------------------------------------------------
def _import_destination(name: str, group: str) -> Path:
    name = validate_name(name, "dataset name")
    group = validate_group(group)
    base = get_root()
    dest = base / group / name if group else base / name
    if dest.exists():
        raise DatasetError(f"'{(Path(group) / name).as_posix()}' already exists.", 409)
    return dest


def _find_dataset_root(staging: Path) -> Path:
    """Unwrap single-folder nesting (zip of a folder) down to the dataset or
    the folder holding several datasets."""
    root = staging
    for _ in range(4):
        if is_dataset_dir(root):
            return root
        children = [c for c in root.iterdir() if not _is_hidden(c) and c.name != "__MACOSX"]
        if len(children) == 1 and children[0].is_dir():
            root = children[0]
        else:
            break
    return root


def _contains_dataset(root: Path) -> bool:
    if is_dataset_dir(root):
        return True
    for dirpath, dirs, _ in os.walk(root):
        if len(Path(dirpath).relative_to(root).parts) > 3:
            dirs[:] = []
            continue
        if is_dataset_dir(Path(dirpath)):
            return True
    return False


def _finish_import(staging: Path, dest: Path) -> Dataset:
    root = _find_dataset_root(staging)
    if not _contains_dataset(root):
        raise DatasetError(
            "No YOLO dataset found in the import. Expected a data.yaml (with train/val splits), "
            "or images/ + labels/ with classes.txt / notes.json (e.g. a Label Studio YOLO export)."
        )
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(root), str(dest))
    invalidate_cache()
    if is_dataset_dir(dest):
        return load_dataset(dest)
    return None  # type: ignore[return-value]  # a folder of several datasets


def import_zip(zip_file: Path, name: str, group: str) -> dict[str, Any]:
    dest = _import_destination(name, group)
    staging = get_root() / f".import_{uuid.uuid4().hex}"
    staging.mkdir(parents=True)
    try:
        try:
            zf = zipfile.ZipFile(zip_file)
        except zipfile.BadZipFile:
            raise DatasetError("The uploaded file is not a valid zip archive.")
        with zf:
            base = staging.resolve()
            for info in zf.infolist():
                target = (staging / info.filename).resolve()
                if not target.is_relative_to(base):
                    raise DatasetError(f"Unsafe path in zip: '{info.filename}'.")
                # Skip symlink entries (unix mode S_IFLNK).
                if (info.external_attr >> 16) & 0o170000 == 0o120000:
                    continue
                if info.is_dir():
                    target.mkdir(parents=True, exist_ok=True)
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                with zf.open(info) as src, open(target, "wb") as out:
                    shutil.copyfileobj(src, out, 1024 * 1024)
        ds = _finish_import(staging, dest)
    finally:
        shutil.rmtree(staging, ignore_errors=True)
    return _import_result(dest, ds)


def import_folder(source: str, name: str, group: str, mode: str = "copy") -> dict[str, Any]:
    if "\\" in source:
        raise DatasetError("Windows-style paths aren't supported — mount the share and use a Linux path.")
    src = Path(source).expanduser()
    if not src.is_dir():
        raise DatasetError(f"Folder '{source}' not found on the server.", 404)
    if src.resolve().is_relative_to(get_root().resolve()):
        raise DatasetError("That folder is already inside the dataset root — it's listed already.")
    dest = _import_destination(name, group)
    root = _find_dataset_root(src)
    if not _contains_dataset(root):
        raise DatasetError(f"No YOLO dataset found under '{source}'.")
    dest.parent.mkdir(parents=True, exist_ok=True)
    if mode == "symlink":
        dest.symlink_to(root.resolve(), target_is_directory=True)
    elif mode == "copy":
        shutil.copytree(root, dest, symlinks=False)
    else:
        raise DatasetError("mode must be 'copy' or 'symlink'.")
    invalidate_cache()
    ds = load_dataset(dest) if is_dataset_dir(dest) else None
    return _import_result(dest, ds)


def _import_result(dest: Path, ds: Optional[Dataset]) -> dict[str, Any]:
    if ds is not None:
        return {"id": ds.id, "datasets": [summarize(ds)]}
    found = [summarize(d) for d in discover() if _abs(d.path).is_relative_to(_abs(dest))]
    return {"id": _rel_id(dest), "datasets": found}


# ---------------------------------------------------------------------------
# Re-index: merge labeled exports into one flat dataset with a single class list
# (the 1.labeled -> 2.re_index_label step; same idea as merge_relabel.py)
# ---------------------------------------------------------------------------
_REINDEX_STATE = ".reindex_state.json"
_COPY_SUFFIX_RE = re.compile(r"^(?P<stem>.+?)(__\d+)?$")


def _read_class_names_file(d: Path) -> list[str]:
    if (d / "classes.txt").is_file():
        return [l.strip() for l in (d / "classes.txt").read_text(encoding="utf-8").splitlines() if l.strip()]
    if (d / "notes.json").is_file():
        return _names_from_notes_json(d / "notes.json") or []
    return []


def _write_class_names(d: Path, names: list[str]) -> None:
    (d / "classes.txt").write_text("\n".join(names) + "\n", encoding="utf-8")
    notes = {
        "categories": [{"id": i, "name": n} for i, n in enumerate(names)],
        "info": {"year": time.localtime().tm_year, "version": "1.0", "contributor": "Label Studio"},
    }
    (d / "notes.json").write_text(json.dumps(notes, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def reindex_merge(sources: list[Dataset], output: Path) -> dict[str, Any]:
    """Merge flat labeled datasets into ``output`` (images/ + labels/ + one
    classes.txt/notes.json), remapping class ids by class name.

    Safe to re-run, unlike a plain copy:
    - existing class indices in ``output`` are kept; new names are appended;
    - images already merged are skipped — tracked in .reindex_state.json, and for
      folders merged before that existed (e.g. by merge_relabel.py), an image
      counts as merged when ``output/images`` has the same file name (or its
      ``name__N`` copy) with the same size.
    Images are copied (``output`` stays self-contained); sources are never modified.
    """
    output = _abs(output)
    out_images, out_labels = output / "images", output / "labels"
    if output.exists() and not is_dataset_dir(output) and any(output.iterdir()):
        raise DatasetError(f"'{output}' exists but isn't a flat dataset (images/ + labels/).")
    for src in sources:
        if src.kind != "flat":
            raise DatasetError(f"'{src.id}' is a prepared (split) dataset — only flat label folders can be re-indexed.")
        if _abs(src.path) == output:
            raise DatasetError(f"'{src.id}' is the re-index target itself — deselect it.")
    out_images.mkdir(parents=True, exist_ok=True)
    out_labels.mkdir(parents=True, exist_ok=True)

    names = _read_class_names_file(output)
    original_count = len(names)

    state_file = output / _REINDEX_STATE
    try:
        state = json.loads(state_file.read_text(encoding="utf-8")) if state_file.exists() else {}
    except (OSError, json.JSONDecodeError):
        state = {}
    merged: dict[str, str] = state.get("merged", {})

    # Existing output images by base name (without merge_relabel's __N suffix) -> sizes
    existing: dict[str, set[int]] = {}
    used_names: set[str] = set()
    for f in out_images.iterdir():
        if f.suffix.lower() not in PREVIEW_IMAGE_EXTENSIONS:
            continue
        used_names.add(f.name)
        base = _COPY_SUFFIX_RE.match(f.stem).group("stem") + f.suffix.lower()
        try:
            existing.setdefault(base, set()).add(f.stat().st_size)
        except OSError:
            pass

    report = []
    for src in sources:
        src_names = src.class_names
        if not src_names:
            raise DatasetError(f"'{src.id}' has no class names (classes.txt / notes.json).")
        id_map: dict[int, int] = {}
        for i, n in enumerate(src_names):
            if n not in names:
                names.append(n)
            id_map[i] = names.index(n)

        folder = src.folders[0]  # the whole images/ tree
        added = skipped = no_label = dropped_lines = 0
        for img in _list_images(folder.images_dir):
            key = str(img.resolve())
            if key in merged:
                skipped += 1
                continue
            label = folder.label_for(img)
            if not label.exists():
                no_label += 1
                continue
            try:
                size = img.stat().st_size
            except OSError:
                continue
            if size in existing.get(img.stem + img.suffix.lower(), set()):
                merged[key] = "(matched existing file)"
                skipped += 1
                continue

            target = out_images / img.name
            i = 2
            while target.name in used_names or target.exists():
                target = out_images / f"{img.stem}__{i}{img.suffix}"
                i += 1
            lines = []
            for line in label.read_text(encoding="utf-8").splitlines():
                parts = line.split()
                if not parts:
                    continue
                try:
                    old = int(float(parts[0]))
                except ValueError:
                    dropped_lines += 1
                    continue
                if old not in id_map:
                    dropped_lines += 1
                    continue
                parts[0] = str(id_map[old])
                lines.append(" ".join(parts))
            shutil.copy2(img, target)
            (out_labels / f"{target.stem}.txt").write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
            used_names.add(target.name)
            existing.setdefault(img.stem + img.suffix.lower(), set()).add(size)
            merged[key] = target.name
            added += 1
        report.append({"source": src.id, "added": added, "already_merged": skipped,
                       "no_label": no_label, "dropped_label_lines": dropped_lines})

    _write_class_names(output, names)
    state_file.write_text(json.dumps({"merged": merged}, indent=1), encoding="utf-8")
    invalidate_cache(output)
    logger.info("Re-indexed %d source(s) into %s: %s", len(sources), output, report)
    return {
        "id": _rel_id(output),
        "class_names": names,
        "new_classes": names[original_count:],
        "sources": report,
        "added": sum(r["added"] for r in report),
    }


# ---------------------------------------------------------------------------
# Preprocess-state consistency (prevents duplicate images when the same data is
# reached through a different path, e.g. a local copy vs the network share)
# ---------------------------------------------------------------------------
def _source_image_names(source: Path) -> dict[str, Path]:
    imgs = _abs(source) / "images"
    return {f.name: f for f in imgs.iterdir()} if imgs.is_dir() else {}


def stale_state_entries(target: Path, sources: list[Path]) -> dict[str, Any]:
    """Records in target/.preprocess_state.json that name the same image files
    as a source, but under a different folder (e.g. the network share vs a local
    copy, or a renamed folder). Preprocessing tracks absolute paths, so it would
    add those images again as duplicates."""
    sf = target / ".preprocess_state.json"
    if not sf.is_file():
        return {"count": 0, "examples": []}
    entries = json.loads(sf.read_text(encoding="utf-8")).get("processed_images", [])
    stale: list[str] = []
    for s in sources:
        names = _source_image_names(s)
        # preprocess_data records resolved paths, so compare against those.
        stale += [e for e in entries if Path(e).name in names and e != str(names[Path(e).name].resolve())]
    return {"count": len(stale), "examples": stale[:3]}


def repoint_state(target: Path, source: Path) -> dict[str, Any]:
    """Rewrite records naming the same file as ``source/images/<name>`` but under
    another folder so they point at the source (a timestamped backup is kept)."""
    sf = target / ".preprocess_state.json"
    if not sf.is_file():
        raise DatasetError("Target has no preprocess records.")
    data = json.loads(sf.read_text(encoding="utf-8"))
    entries = data.get("processed_images", [])
    names = _source_image_names(source)
    out, changed = [], 0
    for e in entries:
        name = Path(e).name
        if name in names and e != str(names[name].resolve()):
            out.append(str(names[name].resolve()))
            changed += 1
        else:
            out.append(e)
    backup = sf.with_name(f".preprocess_state.backup_{time.strftime('%Y%m%d_%H%M%S')}.json")
    shutil.copy2(sf, backup)
    data["processed_images"] = sorted(set(out))
    sf.write_text(json.dumps(data, indent=2), encoding="utf-8")
    logger.info("Repointed %d preprocess records in %s to %s (backup %s)", changed, target, source, backup.name)
    return {"repointed": changed, "backup": str(backup)}


# ---------------------------------------------------------------------------
# Folder details: dataset_info.json (written on every re-index / split run)
# ---------------------------------------------------------------------------
INFO_FILE = "dataset_info.json"


def read_info(path: Path) -> Optional[dict[str, Any]]:
    try:
        data = json.loads((Path(path) / INFO_FILE).read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else None
    except (OSError, json.JSONDecodeError):
        return None


def _now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def _fmt_ratio(r: dict[str, float]) -> str:
    """{'train': .7, 'val': .3, 'test': 0} -> '0.7-0.3-0.0'"""
    def one(v: float) -> str:
        t = f"{v:.2f}".rstrip("0")
        return t + "0" if t.endswith(".") else t
    return "-".join(one(r.get(s, 0)) for s in SPLIT_NAMES)


def _sources_from_paths(paths: Iterable[str]) -> list[dict[str, Any]]:
    """Group recorded image paths by the folder they came from (the part before /images/)."""
    counts: dict[str, int] = {}
    for p in paths:
        src = p.split(f"{os.sep}images{os.sep}")[0] if f"{os.sep}images{os.sep}" in p else str(Path(p).parent)
        counts[src] = counts.get(src, 0) + 1
    return [{"path": k, "name": Path(k).name, "images": v} for k, v in sorted(counts.items())]


def folder_details(ds: Dataset) -> dict[str, Any]:
    """What a folder is and how it was built — from dataset_info.json when
    present, otherwise worked out from its contents and state files."""
    info = read_info(ds.path)
    out: dict[str, Any] = {"id": ds.id, "path": str(ds.path), "kind": ds.kind,
                           "num_classes": len(ds.class_names), "info": info}
    if ds.kind == "split":
        counts = {s: 0 for s in SPLIT_NAMES}
        for f in ds.folders:
            if f.key in counts:
                counts[f.key] = _count_images(f.images_dir)
        total = sum(counts.values())
        actual = {s: round(counts[s] / total, 2) if total else 0.0 for s in SPLIT_NAMES}
        configured = (info or {}).get("split_ratio")
        sf = ds.path / ".preprocess_state.json"
        sources = []
        if sf.is_file():
            try:
                sources = _sources_from_paths(json.loads(sf.read_text(encoding="utf-8")).get("processed_images", []))
            except (OSError, json.JSONDecodeError):
                pass
        out.update(role="prepared", num_images=total, split_counts=counts, actual_ratio=actual,
                   configured_ratio=configured, sources=sources)
        ratio = configured or actual
        summary = (f"Prepared dataset · {total:,} images · {len(ds.class_names)} classes · split train-val-test "
                   f"{_fmt_ratio(ratio)} ({counts['train']:,} / {counts['val']:,} / {counts['test']:,})")
        if configured and any(abs(configured[s] - actual[s]) >= 0.03 for s in SPLIT_NAMES):
            summary += f" · actual {_fmt_ratio(actual)}"
        if sources:
            summary += " · from " + ", ".join(f"{s['name']} ({s['images']:,})" for s in sources)
    else:
        total = _count_images(ds.path / "images")
        sf = ds.path / _REINDEX_STATE
        sources = []
        if sf.is_file():
            try:
                merged = json.loads(sf.read_text(encoding="utf-8")).get("merged", {})
                sources = _sources_from_paths(merged.keys())
            except (OSError, json.JSONDecodeError):
                pass
        role = "reindexed" if (sources or (info or {}).get("kind") == "reindexed") else "labeled"
        out.update(role=role, num_images=total, sources=sources)
        if role == "reindexed":
            summary = (f"Re-indexed folder · {total:,} images · {len(ds.class_names)} classes"
                       + (f" · merged from {len(sources)} labeled folder(s)" if sources else ""))
        else:
            summary = f"Labeled folder · {total:,} images · {len(ds.class_names)} classes"
    if info and info.get("updated_at"):
        summary += f" · last updated {info['updated_at']}"
    out["summary"] = summary
    out["history"] = (info or {}).get("history", [])
    return out


def record_run(path: Path, kind: str, entry: dict[str, Any],
               split_ratio: Optional[dict[str, float]] = None) -> None:
    """Append a run to path/dataset_info.json and refresh its summary.
    Never raises — a details file must not break a successful run."""
    try:
        path = Path(path)
        info = read_info(path) or {"created_at": _now()}
        info["kind"] = kind
        info["updated_at"] = _now()
        if split_ratio is not None:
            info["split_ratio"] = split_ratio
        info.setdefault("history", []).append({"time": _now(), **entry})
        info["history"] = info["history"][-200:]
        try:
            info["summary"] = folder_details(load_dataset(path, get_root() if _abs(path).is_relative_to(get_root()) else path.parent))["summary"]
        except Exception:
            pass
        # Keep "summary" first so the file reads well when opened directly.
        ordered = {"summary": info.pop("summary", ""), **info}
        tmp = path / f".{INFO_FILE}.tmp"
        tmp.write_text(json.dumps(ordered, indent=2, ensure_ascii=False), encoding="utf-8")
        os.replace(tmp, path / INFO_FILE)
    except Exception:
        logger.exception("Could not write %s in %s", INFO_FILE, path)
