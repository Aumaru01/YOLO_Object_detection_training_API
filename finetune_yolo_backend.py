"""
YOLO Fine-tuning Backend
==========================
Pure backend — receives all parameters directly (no config file).
Called by the queue worker with values from the API request body.
See README.md for supported dataset layouts.
"""

import json
import os
import random
import shutil
import logging

from pathlib import Path, PureWindowsPath
from ultralytics import YOLO
from roboflow import Roboflow
from typing import Any, Optional

import torch
import yaml

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

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


# ---------------------------------------------------------------------------
# Dataset preprocessing / validation
# ---------------------------------------------------------------------------
def _find_file_recursive(root: Path, filename: str) -> Optional[Path]:
    """Find filename directly under root, else search subdirectories."""
    direct = root / filename
    if direct.exists():
        return direct

    found = list(root.rglob(filename)) if root.exists() else []
    return found[0] if found else None


def _locate_data_yaml(dataset_dir: Path) -> Optional[Path]:
    return _find_file_recursive(dataset_dir, "data.yaml")


def _names_from_notes_json(notes_path: Path) -> Optional[list[str]]:
    """Parse a notes.json 'categories' list (plain names or COCO-style {id, name} dicts)."""
    try:
        data = json.loads(notes_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None

    categories = data.get("categories") if isinstance(data, dict) else None
    if not categories:
        return None

    if all(isinstance(c, str) for c in categories):
        return list(categories)
    if all(isinstance(c, dict) and "name" in c for c in categories):
        return [c["name"] for c in sorted(categories, key=lambda c: c.get("id", 0))]
    return None


def _names_to_list(names: Any) -> list[str]:
    """Normalize a data.yaml 'names' field (list or {index: name} dict) to a list."""
    if isinstance(names, dict):
        return [names[k] for k in sorted(names, key=int)]
    return list(names)


def _img2label_path(img_path: Path) -> Path:
    """Mirror ultralytics' images/ -> labels/ path convention (last occurrence)."""
    sa, sb = f"{os.sep}images{os.sep}", f"{os.sep}labels{os.sep}"
    img_str = str(img_path)
    if sa not in img_str:
        return img_path.with_suffix(".txt")
    head, tail = img_str.rsplit(sa, 1)
    return Path(f"{head}{sb}{tail}").with_suffix(".txt")


def _images_from_manifest(manifest: Path) -> list[Path]:
    """Read image paths (one per line) from a train.txt/val.txt-style manifest."""
    base = manifest.parent
    lines = [line.strip() for line in manifest.read_text(encoding="utf-8").splitlines() if line.strip()]
    return [(Path(line) if Path(line).is_absolute() else base / line) for line in lines]


def _check_split(split_source: Path) -> dict[str, Any]:
    """Count images in a split (dir or manifest .txt) and how many lack a label file."""
    if split_source.is_file():
        images = _images_from_manifest(split_source)
        label_dir = f"<derived per-image from {split_source.name}>"
    else:
        if not split_source.exists():
            raise FileNotFoundError(f"Split not found: '{split_source}'.")
        images = [p for p in split_source.rglob("*") if p.suffix.lower() in IMAGE_EXTENSIONS]
        label_dir = str(split_source).replace("images", "labels", 1)

    missing_labels = [str(img) for img in images if not _img2label_path(img).exists()]

    return {
        "image_dir": str(split_source),
        "label_dir": label_dir,
        "num_images": len(images),
        "num_missing_labels": len(missing_labels),
        "missing_labels_sample": missing_labels[:10],
    }


def _reject_windows_path(path: str | Path, param_name: str) -> None:
    """Raise a clear error for a Windows/UNC-style path — this API runs on Linux,
    where backslashes are literal filename characters, not separators, so a path
    like '\\\\host\\share\\dir' is treated as one nonexistent filename rather than
    being split into directories.
    """
    if "\\" in str(path):
        raise ValueError(
            f"{param_name} '{path}' looks like a Windows path, but this API runs on "
            "Linux — backslashes aren't path separators here, so the whole string is "
            "treated as a single (nonexistent) filename. Mount the network share on "
            "this host first (e.g. via CIFS/SMB) and pass the mounted path instead, "
            "using forward slashes (e.g. '/mnt/ai-model/...')."
        )


def validate_dataset(dataset_path: str | Path) -> dict[str, Any]:
    """Validate an already-built dataset (a data.yaml, found directly or
    recursively under dataset_path) and summarize its contents.
    """
    _reject_windows_path(dataset_path, "dataset_path")
    dataset_path = Path(dataset_path)

    data_yaml = dataset_path if dataset_path.is_file() else _locate_data_yaml(dataset_path)
    if data_yaml is None:
        raise FileNotFoundError(f"No data.yaml found under '{dataset_path}'.")

    with open(data_yaml, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}

    if config.get("names") is None:
        raise ValueError(f"'{data_yaml}' is missing the required 'names' key.")
    names = _names_to_list(config["names"])
    num_classes = len(names)

    root = Path(config.get("path", data_yaml.parent))
    if not root.is_absolute():
        root = data_yaml.parent / root

    splits: dict[str, Any] = {}
    for split in ("train", "val", "test"):
        rel = config.get(split)
        if rel is None:
            continue
        split_dir = Path(rel)
        if not split_dir.is_absolute():
            split_dir = root / split_dir
        splits[split] = _check_split(split_dir)

    if "train" not in splits:
        raise ValueError(f"'{data_yaml}' does not define a 'train' split.")

    logger.info(
        "Preprocessed dataset '%s' — %d classes, splits=%s",
        data_yaml, num_classes, {k: v["num_images"] for k, v in splits.items()},
    )

    return {
        "data_yaml": str(data_yaml),
        "root": str(root),
        "num_classes": num_classes,
        "class_names": names,
        "splits": splits,
    }


def export_model(model_path: str | Path) -> dict[str, Any]:
    """Export a trained .pt model to ONNX and OpenVINO (.xml/.bin) formats,
    written alongside model_path.
    """
    _reject_windows_path(model_path, "model_path")
    model_path = Path(model_path)

    if not model_path.exists():
        raise FileNotFoundError(f"No model found at '{model_path}'.")
    if model_path.suffix != ".pt":
        raise ValueError(f"model_path '{model_path}' must be a '.pt' file, got '{model_path.suffix}'.")

    # A fresh YOLO instance per export avoids state (e.g. layer fusion) from
    # one export format leaking into the next.
    onnx_path = Path(YOLO(str(model_path)).export(format="onnx"))
    openvino_dir = Path(YOLO(str(model_path)).export(format="openvino"))
    bin_path = next(openvino_dir.glob("*.bin"), None)
    xml_path = next(openvino_dir.glob("*.xml"), None)

    logger.info("Exported '%s' -> onnx='%s', bin='%s'", model_path, onnx_path, bin_path)

    return {
        "model_path": str(model_path),
        "onnx_path": str(onnx_path),
        "openvino_dir": str(openvino_dir),
        "bin_path": str(bin_path) if bin_path else None,
        "xml_path": str(xml_path) if xml_path else None,
    }


def _find_class_names(source_dir: Path) -> Optional[list[str]]:
    """Look for class names via data.yaml, classes.txt, or notes.json (each
    searched recursively, in that priority order). None if none found.
    """
    yaml_path = _find_file_recursive(source_dir, "data.yaml")
    if yaml_path:
        with open(yaml_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f) or {}
        if config.get("names") is not None:
            return _names_to_list(config["names"])

    classes_txt = _find_file_recursive(source_dir, "classes.txt")
    if classes_txt:
        names = [line.strip() for line in classes_txt.read_text(encoding="utf-8").splitlines() if line.strip()]
        if names:
            return names

    notes_json = _find_file_recursive(source_dir, "notes.json")
    if notes_json:
        names = _names_from_notes_json(notes_json)
        if names:
            return names

    return None


def _unique_destination(dst_dir: Path, filename: str) -> Path:
    """Return a non-colliding path for filename inside dst_dir, suffixing on collision."""
    dst = dst_dir / filename
    if not dst.exists():
        return dst

    stem, suffix = Path(filename).stem, Path(filename).suffix
    i = 1
    while (candidate := dst_dir / f"{stem}_{i}{suffix}").exists():
        i += 1
    return candidate


def _remap_label_file(src_label: Path, dst_label: Path, class_remap: dict[int, int]) -> None:
    """Copy a YOLO label file, rewriting class indices per class_remap."""
    lines_out = []
    for line in src_label.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        parts = line.split()
        old_cls = int(parts[0])
        if old_cls not in class_remap:
            raise ValueError(
                f"Label '{src_label}' references class index {old_cls}, which has "
                "no matching entry in the incoming class list."
            )
        parts[0] = str(class_remap[old_cls])
        lines_out.append(" ".join(parts))
    dst_label.write_text("\n".join(lines_out) + ("\n" if lines_out else ""), encoding="utf-8")


def _looks_like_flat_source(source_dir: Path) -> bool:
    """A flat export: an images/ folder plus a labels/ or label/ folder."""
    if not (source_dir / "images").exists():
        return False
    return (source_dir / "labels").exists() or (source_dir / "label").exists()


def _discover_sources(raw_path: Path, output_path: Path) -> list[Path]:
    """Find flat-export sources under raw_path: raw_path itself if it is one,
    else its immediate subdirectories that are. Excludes anything under
    output_path, so a previous run's output is never re-ingested as raw input.
    """
    if not raw_path.exists():
        raise FileNotFoundError(f"raw_path '{raw_path}' does not exist.")

    candidates = (
        [raw_path] if _looks_like_flat_source(raw_path)
        else sorted(d for d in raw_path.iterdir() if d.is_dir() and _looks_like_flat_source(d))
    )
    output_resolved = output_path.resolve()
    return [d for d in candidates if not d.resolve().is_relative_to(output_resolved)]


_STATE_FILENAME = ".preprocess_state.json"


def _load_processed_set(output_path: Path) -> set[str]:
    """Resolved raw image paths already incorporated into output_path, if any."""
    state_file = output_path / _STATE_FILENAME
    if not state_file.exists():
        return set()
    return set(json.loads(state_file.read_text(encoding="utf-8")).get("processed_images", []))


def _save_processed_set(output_path: Path, processed: set[str]) -> None:
    state_file = output_path / _STATE_FILENAME
    state_file.write_text(
        json.dumps({"processed_images": sorted(processed)}, indent=2), encoding="utf-8",
    )


def preprocess_data(
    raw_path: str | Path,
    output_path: str | Path,
    val_ratio: float = 0.1,
    test_ratio: float = 0.0,
) -> dict[str, Any]:
    """Build (or incrementally update) a YOLO dataset at output_path from one or
    more flat-export sources at raw_path.

    raw_path is either a single flat source (images/ + labels/ or label/ +
    class names) or a folder of several such sources as immediate
    subdirectories (e.g. label1/, label2/, ...). Each source's class names
    are merged into one list and its label indices remapped to match.
    Re-running with the same raw_path/output_path only processes images not
    already incorporated (tracked in a state file under output_path), split
    independently per source — so adding a new source, or new images to an
    existing one, extends the existing train/val/test split rather than
    reshuffling it. Nothing under raw_path is ever modified: images are
    symlinked into output_path where possible (not copied), falling back to
    real copies if the output filesystem doesn't support symlinks (e.g. many
    CIFS/SMB mounts) — only their remapped labels are otherwise written
    there. See README.md for details.
    """
    if val_ratio + test_ratio >= 1:
        raise ValueError(
            f"val_ratio ({val_ratio}) + test_ratio ({test_ratio}) must leave room "
            "for a non-empty train split (sum must be < 1)."
        )
    _reject_windows_path(raw_path, "raw_path")
    _reject_windows_path(output_path, "output_path")

    raw_path = Path(raw_path)
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    sources = _discover_sources(raw_path, output_path)
    if not sources:
        raise FileNotFoundError(
            f"No YOLO-format source found under '{raw_path}' (expected images/ + "
            "labels/ or label/, directly there or in immediate subdirectories)."
        )

    existing_yaml = output_path / "data.yaml"
    if existing_yaml.exists():
        with open(existing_yaml, "r", encoding="utf-8") as f:
            names = _names_to_list((yaml.safe_load(f) or {}).get("names", []))
    else:
        names = []

    processed = _load_processed_set(output_path)
    added = {"train": 0, "val": 0, "test": 0}
    new_classes: list[str] = []
    skipped_no_label: list[str] = []
    sources_touched: list[str] = []
    symlinks_supported = True

    for source in sources:
        labels_dir = source / "labels" if (source / "labels").exists() else source / "label"
        images_dir = source / "images"

        source_names = _find_class_names(source)
        if source_names is None:
            raise FileNotFoundError(
                f"No class names found for source '{source}' — expected a data.yaml "
                "with 'names', a classes.txt, or a notes.json."
            )

        original_num_classes = len(names)
        class_remap: dict[int, int] = {}
        for idx, name in enumerate(source_names):
            if name in names:
                class_remap[idx] = names.index(name)
            else:
                names.append(name)
                class_remap[idx] = len(names) - 1
        new_classes.extend(names[original_num_classes:])

        all_images = [p for p in images_dir.rglob("*") if p.suffix.lower() in IMAGE_EXTENSIONS]
        new_images = [p for p in all_images if str(p.resolve()) not in processed]
        if not new_images:
            continue

        random.shuffle(new_images)
        n_val = round(len(new_images) * val_ratio)
        n_test = round(len(new_images) * test_ratio)
        split_assignment = (
            [("val", p) for p in new_images[:n_val]]
            + [("test", p) for p in new_images[n_val:n_val + n_test]]
            + [("train", p) for p in new_images[n_val + n_test:]]
        )

        for split, img in split_assignment:
            label_src = labels_dir / img.relative_to(images_dir).with_suffix(".txt")
            if not label_src.exists():
                skipped_no_label.append(str(img))
                continue

            dst_images_dir = output_path / split / "images"
            dst_labels_dir = output_path / split / "labels"
            dst_images_dir.mkdir(parents=True, exist_ok=True)
            dst_labels_dir.mkdir(parents=True, exist_ok=True)

            dst_image = _unique_destination(dst_images_dir, img.name)
            src_image = img.resolve()
            if symlinks_supported:
                try:
                    dst_image.symlink_to(src_image)
                except OSError:
                    symlinks_supported = False
                    logger.warning(
                        "'%s' doesn't support symlinks (%s) — falling back to copying "
                        "images instead (uses more disk).",
                        output_path, src_image,
                    )
                    if dst_image.is_symlink() or dst_image.exists():
                        dst_image.unlink()
                    shutil.copy2(src_image, dst_image)
            else:
                shutil.copy2(src_image, dst_image)
            _remap_label_file(label_src, dst_labels_dir / dst_image.with_suffix(".txt").name, class_remap)

            processed.add(str(img.resolve()))
            added[split] += 1
            if source.name not in sources_touched:
                sources_touched.append(source.name)

    config: dict[str, Any] = {"path": str(output_path), "names": names, "nc": len(names)}
    for split in ("train", "val", "test"):
        if (output_path / split / "images").exists():
            config[split] = f"{split}/images"

    if "train" not in config:
        raise FileNotFoundError(f"No usable images found across sources under '{raw_path}'.")

    with open(output_path / "data.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, sort_keys=False, allow_unicode=True)

    _save_processed_set(output_path, processed)

    logger.info(
        "Preprocessed '%s' -> '%s' — sources=%s, +train=%d +val=%d +test=%d, "
        "%d new classes, %d skipped (no label).",
        raw_path, output_path, sources_touched, added["train"], added["val"], added["test"],
        len(new_classes), len(skipped_no_label),
    )

    summary = validate_dataset(output_path)
    summary["added"] = {
        "sources": sources_touched,
        "train": added["train"],
        "val": added["val"],
        "test": added["test"],
        "new_classes": new_classes,
        "skipped_no_label": skipped_no_label,
    }
    return summary


# ---------------------------------------------------------------------------
# Backend class
# ---------------------------------------------------------------------------
class YOLOTrainBackend:
    """Training backend — every parameter comes from the caller (API -> worker -> here)."""

    def __init__(
        self,
        model: str,
        job_name: str,
        api_key: Optional[str] = None,
        workspace: Optional[str] = None,
        project_name: Optional[str] = None,
        version: Optional[int] = None,
        dataset_format: Optional[str] = None,
        dataset_path: Optional[str | Path] = None,
        output_path: Optional[str | Path] = None,
    ) -> None:
        # Roboflow — omitted when training on an already-local dataset.
        self.rf = Roboflow(api_key=api_key) if api_key else None
        self.workspace = workspace
        self.project_name = project_name
        self.version = version
        self.dataset_format = dataset_format

        # Model
        self.model_name = model
        self._model: Optional[YOLO] = None

        # dataset_path/output_path override the default datasets/{job_name}/
        # and models/{job_name}/ conventions, letting a job read/write at
        # arbitrary filesystem paths.
        self.job_name = job_name
        self.dataset_dir = Path(dataset_path) if dataset_path else (BASE_DATASET_DIR / job_name)
        self.model_dir = Path(output_path) if output_path else (BASE_MODEL_DIR / job_name)
        self._data_yaml_cache: Optional[Path] = None

        # Only create model dir upfront — Roboflow skips download if
        # dataset_dir already exists, so it must not be pre-created.
        self.model_dir.mkdir(parents=True, exist_ok=True)

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
        if self._data_yaml_cache and self._data_yaml_cache.exists():
            return self._data_yaml_cache

        found = _locate_data_yaml(self.dataset_dir)
        if found:
            self._data_yaml_cache = found
            logger.info("Found data.yaml at: %s", found)
        return found

    @property
    def data_yaml_path(self) -> Path:
        """Return data.yaml path, or expected path if not yet downloaded."""
        found = self._find_data_yaml()
        return found if found else self.dataset_dir / "data.yaml"

    # ------------------------------------------------------------------
    # Device validation
    # ------------------------------------------------------------------
    @staticmethod
    def _validate_device(device: str) -> str:
        """Validate device and log warning if unavailable."""
        if device == "cpu":
            logger.info("Using CPU.")
            return device

        if device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError(
                f"Device '{device}' requested but CUDA is not available. "
                "Please check CUDA installation or use 'cpu'."
            )

        if ":" in device:
            try:
                gpu_index = int(device.split(":")[1])
                gpu_count = torch.cuda.device_count()
                if gpu_index >= gpu_count:
                    logger.warning(
                        "Device '%s' requested but only %d GPU(s) found. Falling back to 'cuda:0'.",
                        device, gpu_count,
                    )
                    return "cuda:0"
            except ValueError:
                logger.warning("Invalid device format '%s'. Falling back to CPU.", device)
                return "cpu"

        logger.info("Using device: %s (%s)", device, torch.cuda.get_device_name(0))
        return device

    # ------------------------------------------------------------------
    # Dataset
    # ------------------------------------------------------------------
    def download_dataset(self, *, force: bool = False) -> Path:
        """Download dataset from Roboflow."""
        if self._find_data_yaml() and not force:
            logger.info("Dataset already at '%s' — skipping.", self.data_yaml_path)
            return self.dataset_dir

        if self.rf is None:
            raise RuntimeError(
                f"No local dataset found at '{self.dataset_dir}' and no Roboflow "
                "credentials were provided to download one."
            )

        logger.info(
            "Downloading: %s/%s v%d (%s) ...",
            self.workspace, self.project_name, self.version, self.dataset_format,
        )
        project = self.rf.workspace(self.workspace).project(self.project_name)
        project.version(self.version).download(
            self.dataset_format, location=str(self.dataset_dir),
        )

        all_files = list(self.dataset_dir.rglob("*")) if self.dataset_dir.exists() else []
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

        device = self._validate_device(train_params["device"])

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
            "device": device,
            "project": str(self.model_dir),
            "save": True,
        }

        logger.info(
            "Training — epochs=%s, batch=%s, lr0=%s, device=%s, save='%s'",
            train_args["epochs"], train_args["batch"], train_args["lr0"],
            device, self.model_dir,
        )

        self.model.train(**train_args)
        logger.info("Training complete.")

        return {
            "job_name": self.job_name,
            "model_dir": str(self.model_dir),
            "dataset_dir": str(self.dataset_dir),
            "epochs": train_args["epochs"],
            "device": device,
        }

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------
    def evaluate(self, train_params: dict[str, Any]) -> dict[str, float]:
        """Run validation (plus a test-set pass, if the dataset defines one)
        and return mAP metrics. Results are saved under model_dir/val/ (and
        model_dir/test/, if applicable), alongside the trained weights.
        """
        if not self.data_yaml_path.exists():
            raise FileNotFoundError(
                f"Dataset not found at '{self.data_yaml_path}'. "
                "Call download_dataset() first."
            )

        device = self._validate_device(train_params["device"])

        logger.info("Running evaluation ...")
        metrics = self.model.val(
            data=str(self.data_yaml_path),
            imgsz=train_params["img_size"],
            device=device,
            plots=True,
            project=str(self.model_dir),
            name="val",
        )
        result = {
            "mAP50": float(metrics.box.map50),
            "mAP50_95": float(metrics.box.map),
        }
        logger.info("mAP50: %.4f | mAP50-95: %.4f", result["mAP50"], result["mAP50_95"])

        with open(self.data_yaml_path, "r", encoding="utf-8") as f:
            has_test_split = bool((yaml.safe_load(f) or {}).get("test"))

        if has_test_split:
            logger.info("Running test-set evaluation ...")
            test_metrics = self.model.val(
                data=str(self.data_yaml_path),
                split="test",
                imgsz=train_params["img_size"],
                device=device,
                plots=True,
                project=str(self.model_dir),
                name="test",
            )
            result["test_mAP50"] = float(test_metrics.box.map50)
            result["test_mAP50_95"] = float(test_metrics.box.map)
            logger.info(
                "test mAP50: %.4f | test mAP50-95: %.4f",
                result["test_mAP50"], result["test_mAP50_95"],
            )

        return result

    # ------------------------------------------------------------------
    # Full pipeline (download -> train -> evaluate)
    # ------------------------------------------------------------------
    def run_pipeline_roboflow(self, train_params: dict[str, Any]) -> dict[str, Any]:
        """Execute the full pipeline: download from Roboflow -> train -> evaluate."""
        logger.info("=== Starting full pipeline [%s] ===", self.job_name)

        self.download_dataset()
        train_result = self.train(train_params)
        eval_result = self.evaluate(train_params)

        result = {**train_result, **eval_result}
        logger.info("=== Pipeline complete [%s] === Result: %s", self.job_name, result)
        return result

    def run_pipeline_local(self, train_params: dict[str, Any]) -> dict[str, Any]:
        """Execute the pipeline on an already-present local dataset: train -> evaluate."""
        logger.info("=== Starting local pipeline [%s] ===", self.job_name)

        if not self.data_yaml_path.exists():
            raise FileNotFoundError(
                f"No local dataset found at '{self.dataset_dir}'. Place a dataset "
                "(with data.yaml) there before submitting a local training job."
            )

        train_result = self.train(train_params)
        eval_result = self.evaluate(train_params)

        result = {**train_result, **eval_result}
        logger.info("=== Local pipeline complete [%s] === Result: %s", self.job_name, result)
        return result

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------
    def cleanup_dataset(self) -> None:
        """Delete dataset_dir. Unused today — dangerous for in-place (non-copied) datasets."""
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
            f"v{self.version}, model={self.model_name!r})"
        )
