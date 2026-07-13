"""Controlled ZS32 YOLO26 parameter sweep with an offline-safe dry run."""

from __future__ import annotations

import argparse
import csv
import fcntl
import hashlib
import json
import math
import re
import sys
from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from types import MappingProxyType
from typing import Any, Callable

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from examples.c789 import zs32_sweep_metrics as metrics

DEFAULT_PROJECT = Path("/home/yunjing/ultralytics-c789/runs/zs32_sweep")
DEFAULT_DATA_YAML = Path("/home/yunjing/anomalib/dataset/zs32_six_view_roi_yolo/data.yaml")
DEFAULT_MANIFEST = Path("/home/yunjing/anomalib/dataset/zs32_six_view_roi_yolo/split_manifest.csv")
DEFAULT_WEIGHTS_DIR = Path("/home/yunjing/ultralytics-c789")
PROTECTED_BASELINE = Path("/home/yunjing/ultralytics-c789/runs/zs32/six_view_yolo26n_1536_v1")

PROTECTED_AUGMENTATIONS = (
    "degrees",
    "fliplr",
    "flipud",
    "mosaic",
    "mixup",
    "cutmix",
    "copy_paste",
    "shear",
    "perspective",
)


@dataclass(frozen=True)
class ResourceBinding:
    """One model, resolution, and resource-safe batch binding."""

    model_family: str
    model_filename: str
    imgsz: int
    batch: int


MODEL_RESOLUTION_BATCH: Mapping[str, ResourceBinding] = MappingProxyType(
    {
        "n640": ResourceBinding("n", "yolo26n.pt", 640, 64),
        "n1280": ResourceBinding("n", "yolo26n.pt", 1280, 24),
        "n1536": ResourceBinding("n", "yolo26n.pt", 1536, 16),
        "m640": ResourceBinding("m", "yolo26m.pt", 640, 32),
        "m1280": ResourceBinding("m", "yolo26m.pt", 1280, 8),
        "m1536": ResourceBinding("m", "yolo26m.pt", 1536, 8),
    }
)


def _profile(**values: object) -> Mapping[str, object]:
    for key in PROTECTED_AUGMENTATIONS:
        if key in values and values[key] != 0 and values[key] != 0.0:
            raise ValueError(f"protected augmentation {key!r} must remain zero")
    return MappingProxyType({**{key: 0.0 for key in PROTECTED_AUGMENTATIONS}, **values})


TRAINING_PROFILES: Mapping[str, Mapping[str, object]] = MappingProxyType(
    {
        "P0_no_aug": _profile(
            optimizer="auto",
            translate=0.0,
            scale=0.0,
            hsv_h=0.0,
            hsv_s=0.0,
            hsv_v=0.0,
        ),
        "P1_conservative": _profile(
            optimizer="auto",
            translate=0.03,
            scale=0.10,
            hsv_h=0.005,
            hsv_s=0.20,
            hsv_v=0.15,
        ),
        "P2_adamw_cosine": _profile(
            optimizer="AdamW",
            translate=0.03,
            scale=0.10,
            hsv_h=0.005,
            hsv_s=0.20,
            hsv_v=0.15,
            lr0=0.001,
            lrf=0.01,
            cos_lr=True,
            weight_decay=0.0005,
        ),
    }
)


@dataclass(frozen=True)
class TrialSpec:
    """Fully resolved immutable definition of one sweep trial."""

    trial_id: str
    stage: str
    resource_id: str
    model_family: str
    model_path: Path
    model_sha256: str
    data_yaml_sha256: str
    manifest_sha256: str
    profile: str
    imgsz: int
    batch: int
    seed: int
    epochs: int
    patience: int
    final_epochs: int
    finalists: int
    conf: float
    iou: float
    data_yaml: Path
    manifest: Path
    project: Path
    overrides: Mapping[str, object]


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be a positive integer")
    return parsed


def _unit_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or not 0.0 <= parsed <= 1.0:
        raise argparse.ArgumentTypeError("value must be finite and in [0, 1]")
    return parsed


def _experiment_component(value: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", value) or value in {".", ".."}:
        raise argparse.ArgumentTypeError("experiment-id must be one safe relative path component")
    return value


def build_parser() -> argparse.ArgumentParser:
    """Build the ZS32 sweep command-line parser."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--stage", choices=("screen", "final", "all"), default="all")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-trials", type=_positive_int)
    parser.add_argument("--experiment-id", type=_experiment_component, default="zs32_sweep")
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--screen-epochs", type=_positive_int, default=50)
    parser.add_argument("--final-epochs", type=_positive_int, default=150)
    parser.add_argument("--finalists", type=_positive_int, default=3)
    parser.add_argument("--conf", type=_unit_float, default=0.25)
    parser.add_argument("--iou", type=_unit_float, default=0.50)
    parser.add_argument("--project", type=Path, default=DEFAULT_PROJECT)
    parser.add_argument("--data-yaml", type=Path, default=DEFAULT_DATA_YAML)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--weights-dir", type=Path, default=DEFAULT_WEIGHTS_DIR)
    parser.add_argument("--oom-retry", action=argparse.BooleanOptionalAction, default=True)
    return parser


def _required_file(path: Path, description: str) -> Path:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"required {description} does not exist: {resolved}")
    return resolved


def _validate_paths(args: argparse.Namespace) -> tuple[Path, Path, Path]:
    data_yaml = _required_file(args.data_yaml, "data YAML")
    manifest = _required_file(args.manifest, "manifest")
    weights_dir = args.weights_dir.expanduser().resolve()
    if not weights_dir.is_dir():
        raise FileNotFoundError(f"required weights directory does not exist: {weights_dir}")
    for filename in ("yolo26n.pt", "yolo26m.pt"):
        _required_file(weights_dir / filename, f"model weight {filename}")
    return data_yaml, manifest, weights_dir


def _experiment_root(project: Path, experiment_id: str) -> Path:
    try:
        safe_id = _experiment_component(experiment_id)
    except argparse.ArgumentTypeError as error:
        raise ValueError(str(error)) from error
    resolved_project = project.expanduser().resolve()
    root = (resolved_project / safe_id).resolve()
    try:
        root.relative_to(resolved_project)
    except ValueError as error:
        raise ValueError(f"experiment root escapes project directory: {root}") from error
    if root == resolved_project:
        raise ValueError("experiment root must be a child of project directory")
    protected = PROTECTED_BASELINE.resolve()
    if root == protected or protected in root.parents:
        raise ValueError(f"experiment root is inside protected baseline: {root}")
    return root


@contextmanager
def _experiment_lock(experiment_root: Path):
    """Hold a non-blocking process lock for one experiment state reader/writer."""
    experiment_root.mkdir(parents=True, exist_ok=True)
    lock_path = experiment_root / ".sweep.lock"
    with lock_path.open("a+", encoding="utf-8") as lock_file:
        try:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(f"experiment is locked by another process: {experiment_root}") from error
        try:
            yield
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


def build_experiment_plan(args: argparse.Namespace) -> list[TrialSpec]:
    """Return the deterministic eighteen-trial screening matrix."""
    data_yaml, manifest, weights_dir = _validate_paths(args)
    data_yaml_sha256 = _sha256_file(data_yaml)
    manifest_sha256 = _sha256_file(manifest)
    model_hashes = {
        filename: _sha256_file((weights_dir / filename).resolve())
        for filename in {binding.model_filename for binding in MODEL_RESOLUTION_BATCH.values()}
    }
    experiment_root = _experiment_root(args.project, args.experiment_id) / "screen"
    trials: list[TrialSpec] = []
    for resource_id, binding in MODEL_RESOLUTION_BATCH.items():
        for profile_index, (profile_name, profile_overrides) in enumerate(TRAINING_PROFILES.items()):
            trials.append(
                TrialSpec(
                    trial_id=f"screen_{resource_id}_p{profile_index}_seed{args.seed}",
                    stage="screen",
                    resource_id=resource_id,
                    model_family=binding.model_family,
                    model_path=(weights_dir / binding.model_filename).resolve(),
                    model_sha256=model_hashes[binding.model_filename],
                    data_yaml_sha256=data_yaml_sha256,
                    manifest_sha256=manifest_sha256,
                    profile=profile_name,
                    imgsz=binding.imgsz,
                    batch=binding.batch,
                    seed=args.seed,
                    epochs=args.screen_epochs,
                    patience=args.screen_epochs,
                    final_epochs=args.final_epochs,
                    finalists=args.finalists,
                    conf=args.conf,
                    iou=args.iou,
                    data_yaml=data_yaml,
                    manifest=manifest,
                    project=experiment_root,
                    overrides=profile_overrides,
                )
            )
    return trials


def build_train_kwargs(trial: TrialSpec, args: argparse.Namespace) -> dict[str, object]:
    """Build fully resolved Ultralytics training arguments for one trial."""
    return {
        "data": str(trial.data_yaml.resolve()),
        "epochs": trial.epochs,
        "patience": trial.patience,
        "imgsz": trial.imgsz,
        "batch": trial.batch,
        "device": 0,
        "workers": 8,
        "project": str(trial.project.resolve()),
        "name": trial.trial_id,
        "exist_ok": False,
        "seed": trial.seed,
        "cache": False,
        "single_cls": True,
        **dict(trial.overrides),
    }


def _trial_record(trial: TrialSpec, *, include_output: bool = True) -> dict[str, object]:
    record: dict[str, object] = {
        "trial_id": trial.trial_id,
        "stage": trial.stage,
        "resource_id": trial.resource_id,
        "model_family": trial.model_family,
        "model_path": str(trial.model_path),
        "model_sha256": trial.model_sha256,
        "data_yaml_sha256": trial.data_yaml_sha256,
        "manifest_sha256": trial.manifest_sha256,
        "profile": trial.profile,
        "imgsz": trial.imgsz,
        "batch": trial.batch,
        "seed": trial.seed,
        "epochs": trial.epochs,
        "patience": trial.patience,
        "final_epochs": trial.final_epochs,
        "finalists": trial.finalists,
        "conf": trial.conf,
        "iou": trial.iou,
        "data_yaml": str(trial.data_yaml),
        "manifest": str(trial.manifest),
        "overrides": dict(sorted(trial.overrides.items())),
    }
    if include_output:
        record["project"] = str(trial.project)
        record["name"] = trial.trial_id
    return record


def plan_hash(trials: Sequence[TrialSpec]) -> str:
    """Hash scientific inputs while excluding invocation stage and output location.

    Excluding stage lets one experiment ID continue from a screen invocation to a later final invocation. Selection and
    evaluation settings remain hashed.
    """
    payload = []
    for trial in trials:
        record = _trial_record(trial, include_output=False)
        record.pop("stage")
        payload.append(record)
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def atomic_write_json(path: str | Path, payload: object) -> None:
    """Atomically replace a JSON file using a sibling temporary file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def _run_save_dir(model: object, train_result: object) -> Path:
    trainer = getattr(model, "trainer", None)
    save_dir = getattr(trainer, "save_dir", None) or getattr(train_result, "save_dir", None)
    if save_dir is None:
        raise RuntimeError("Ultralytics training did not expose a save directory")
    return Path(save_dir).resolve()


def _split_source(data_yaml: Path, split: str) -> Path | list[Path]:
    payload = yaml.safe_load(data_yaml.read_text(encoding="utf-8")) or {}
    raw = payload.get(split)
    if raw is None:
        raise ValueError(f"dataset YAML does not define {split!r}")
    root = Path(payload.get("path") or data_yaml.parent)
    if not root.is_absolute():
        root = data_yaml.parent / root

    def resolve(value: object) -> Path:
        path = Path(str(value))
        return path.resolve() if path.is_absolute() else (root / path).resolve()

    return [resolve(value) for value in raw] if isinstance(raw, list) else resolve(raw)


def _label_path(image_path: Path) -> Path:
    parts = list(image_path.parts)
    try:
        index = len(parts) - 1 - parts[::-1].index("images")
    except ValueError:
        return image_path.with_suffix(".txt")
    parts[index] = "labels"
    return Path(*parts).with_suffix(".txt")


def _empty_label_fpr(predictions: Sequence[object]) -> float:
    empty_count = 0
    false_positive_count = 0
    for prediction in predictions:
        image_path = Path(str(getattr(prediction, "path")))
        label_path = _label_path(image_path)
        if not label_path.is_file():
            raise FileNotFoundError(f"label file is missing for prediction image {image_path}: {label_path}")
        if label_path.read_text(encoding="utf-8").strip():
            continue
        empty_count += 1
        boxes = getattr(prediction, "boxes", ())
        if len(boxes) > 0:
            false_positive_count += 1
    return false_positive_count / empty_count if empty_count else 0.0


def _is_cuda_oom(error: BaseException) -> bool:
    text = str(error).lower()
    return error.__class__.__name__ == "OutOfMemoryError" or "cuda out of memory" in text


def _has_completed_epoch(run_dir: Path) -> bool:
    results_csv = run_dir / "results.csv"
    if not results_csv.is_file():
        return False
    try:
        metrics.read_training_summary(results_csv)
    except (OSError, ValueError):
        return False
    return True


class TrainingOOMError(RuntimeError):
    """CUDA OOM raised by training before a completed epoch was recorded."""


def _box_rows(boxes: object) -> list[tuple[float, float, float, float]]:
    values = getattr(boxes, "xyxy", ())
    if hasattr(values, "tolist"):
        values = values.tolist()
    output = []
    for row in values:
        if hasattr(row, "tolist"):
            row = row.tolist()
        output.append(tuple(float(value) for value in row[:4]))
    return output


def _ground_truth_boxes(image_path: Path, image_shape: Sequence[int]) -> list[tuple[float, float, float, float]]:
    label_path = _label_path(image_path)
    if not label_path.is_file():
        raise FileNotFoundError(f"label file is missing for prediction image {image_path}: {label_path}")
    height, width = int(image_shape[0]), int(image_shape[1])
    boxes = []
    for row_number, line in enumerate(label_path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        parts = line.split()
        if len(parts) != 5:
            raise ValueError(f"invalid YOLO label row {row_number}: {label_path}")
        class_id, cx, cy, box_width, box_height = (float(value) for value in parts)
        if class_id != 0:
            raise ValueError(f"unexpected class id in {label_path}: {class_id}")
        boxes.append(
            (
                (cx - box_width / 2) * width,
                (cy - box_height / 2) * height,
                (cx + box_width / 2) * width,
                (cy + box_height / 2) * height,
            )
        )
    return boxes


def _manifest_rows(path: Path, split: str) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as file:
        rows = [row for row in csv.DictReader(file) if row.get("split") == split]
    if not rows:
        raise ValueError(f"manifest has no {split!r} rows: {path}")
    return rows


def _manifest_root(manifest: Path, rows: Sequence[Mapping[str, str]]) -> Path:
    relative_paths = [Path(str(row.get("output_image", ""))) for row in rows]
    relative_paths = [path for path in relative_paths if path and not path.is_absolute()]
    if not relative_paths:
        return manifest.parent.resolve()
    candidates = list(dict.fromkeys((manifest.parent, *manifest.parents)))
    matches = [
        candidate.resolve() for candidate in candidates if all((candidate / path).exists() for path in relative_paths)
    ]
    if len(matches) != 1:
        raise ValueError(f"cannot resolve one unambiguous manifest path root for {manifest}: {matches}")
    return matches[0]


def _write_final_grouped_metrics(
    run_dir: Path,
    trial: TrialSpec,
    predictions: Sequence[object],
) -> Path:
    rows = _manifest_rows(trial.manifest, "test")
    path_root = _manifest_root(trial.manifest, rows)
    ground_truth: dict[Path, Sequence[Sequence[float]]] = {}
    predicted: dict[Path, Sequence[Sequence[float]]] = {}
    for prediction in predictions:
        image_path = Path(str(getattr(prediction, "path"))).resolve()
        image_shape = getattr(prediction, "orig_shape", None)
        if image_shape is None:
            raise ValueError(f"prediction has no orig_shape: {image_path}")
        ground_truth[image_path] = _ground_truth_boxes(image_path, image_shape)
        predicted[image_path] = _box_rows(getattr(prediction, "boxes", ()))
    grouped = metrics.summarize_grouped_predictions(
        rows,
        ground_truth,
        predicted,
        iou_threshold=trial.iou,
        path_root=path_root,
    )
    output = run_dir / "grouped_metrics.csv"
    metrics.write_grouped_metrics_csv(output, grouped)
    return output


def _validation_payload(result: object) -> dict[str, object]:
    values = getattr(result, "results_dict", None)
    if values is None:
        values = getattr(getattr(result, "metrics", None), "results_dict", None)
    if not isinstance(values, Mapping):
        return {}
    output = {}
    for key, value in values.items():
        if hasattr(value, "item"):
            value = value.item()
        if isinstance(value, (str, int, float, bool)) or value is None:
            output[str(key)] = value
    return output


def _cuda_memory_backend(yolo_factory: Callable[..., object]) -> object | None:
    if not getattr(yolo_factory, "__module__", "").startswith("ultralytics"):
        return None
    try:
        import torch

        return torch.cuda if torch.cuda.is_available() else None
    except (ImportError, RuntimeError):
        return None


def _reset_peak_gpu_memory(cuda_memory: object | None) -> None:
    if cuda_memory is not None:
        cuda_memory.reset_peak_memory_stats()


def _peak_gpu_memory_gb(model: object, cuda_memory: object | None = None) -> float | None:
    if cuda_memory is not None:
        value = float(cuda_memory.max_memory_allocated() / (1024**3))
        return value if math.isfinite(value) and value >= 0 else None
    trainer = getattr(model, "trainer", None)
    recorded = getattr(trainer, "peak_gpu_memory_gb", None)
    if recorded is not None:
        value = float(recorded)
        return value if math.isfinite(value) and value >= 0 else None
    return None


def _write_peak_gpu_metadata(run_dir: Path, model: object, cuda_memory: object | None) -> None:
    peak_gpu_memory_gb = _peak_gpu_memory_gb(model, cuda_memory)
    atomic_write_json(
        run_dir / "execution_metadata.json",
        {
            "peak_gpu_memory_gb": peak_gpu_memory_gb,
            "peak_gpu_memory_status": "recorded" if peak_gpu_memory_gb is not None else "unavailable",
        },
    )


def run_trial(
    trial: TrialSpec,
    args: argparse.Namespace,
    *,
    yolo_factory: Callable[..., object],
    cuda_memory: object | None = None,
) -> metrics.TrialMetrics:
    """Execute one injected trial and collect its validated training artifacts."""
    cuda_memory = cuda_memory if cuda_memory is not None else _cuda_memory_backend(yolo_factory)
    _reset_peak_gpu_memory(cuda_memory)
    model = yolo_factory(str(trial.model_path), task="detect")
    expected_run_dir = trial.project.resolve() / trial.trial_id
    run_dir = expected_run_dir
    try:
        try:
            train_result = model.train(**build_train_kwargs(trial, args))
        except RuntimeError as error:
            if _is_cuda_oom(error) and not _has_completed_epoch(expected_run_dir):
                raise TrainingOOMError(str(error)) from error
            raise
        run_dir = _run_save_dir(model, train_result)
        results_csv = run_dir / "results.csv"
        summary = metrics.read_training_summary(results_csv)
        best_pt = run_dir / "weights" / "best.pt"
        if not best_pt.is_file():
            raise FileNotFoundError(f"training completed without best.pt: {best_pt}")

        val_result = model.val(
            data=str(trial.data_yaml),
            split="val",
            imgsz=trial.imgsz,
            batch=trial.batch,
            device=0,
            conf=0.001,
            iou=0.7,
            project=str(run_dir),
            name="val",
            exist_ok=True,
            plots=False,
            verbose=False,
        )
        val_predictions = model.predict(
            source=_split_source(trial.data_yaml, "val"),
            conf=trial.conf,
            iou=trial.iou,
            imgsz=trial.imgsz,
            device=0,
            save=False,
            verbose=False,
        )
        evaluation = {"val": _validation_payload(val_result)}
        if trial.stage == "final":
            test_result = model.val(
                data=str(trial.data_yaml),
                split="test",
                imgsz=trial.imgsz,
                batch=trial.batch,
                device=0,
                conf=0.001,
                iou=0.7,
                project=str(run_dir),
                name="test",
                exist_ok=True,
                plots=False,
                verbose=False,
            )
            evaluation["test"] = _validation_payload(test_result)
            test_predictions = model.predict(
                source=_split_source(trial.data_yaml, "test"),
                conf=trial.conf,
                iou=trial.iou,
                imgsz=trial.imgsz,
                device=0,
                save=False,
                verbose=False,
            )
            _write_final_grouped_metrics(run_dir, trial, list(test_predictions))
        atomic_write_json(run_dir / "evaluation_metrics.json", evaluation)
        return metrics.TrialMetrics(
            trial_id=trial.trial_id,
            model_family=trial.model_family,
            profile=trial.profile,
            imgsz=trial.imgsz,
            batch=trial.batch,
            best=summary.best,
            mean_epoch_seconds=summary.mean_epoch_seconds,
            empty_label_fpr=_empty_label_fpr(val_predictions),
            run_dir=str(run_dir),
            best_pt=str(best_pt),
        )
    finally:
        trainer_save_dir = getattr(getattr(model, "trainer", None), "save_dir", None)
        metadata_dir = Path(trainer_save_dir).resolve() if trainer_save_dir is not None else run_dir
        metadata_dir.mkdir(parents=True, exist_ok=True)
        _write_peak_gpu_metadata(metadata_dir, model, cuda_memory)


def _final_trial(screen_trial: TrialSpec, *, effective_batch: int) -> TrialSpec:
    """Create a clean finalist trial that starts from the original pretrained weight."""
    suffix = screen_trial.trial_id.removeprefix("screen_")
    return replace(
        screen_trial,
        trial_id=f"final_{suffix}",
        stage="final",
        batch=effective_batch,
        epochs=screen_trial.final_epochs,
        patience=30,
        project=screen_trial.project.parent / "finalists",
    )


def _metric_payload(row: metrics.TrialMetrics) -> dict[str, object]:
    return {
        "trial_id": row.trial_id,
        "model_family": row.model_family,
        "profile": row.profile,
        "imgsz": row.imgsz,
        "batch": row.batch,
        "best": {
            "epoch": row.best.epoch,
            "epoch_seconds": row.best.epoch_seconds,
            "precision": row.best.precision,
            "recall": row.best.recall,
            "map50": row.best.map50,
            "map50_95": row.best.map50_95,
        },
        "mean_epoch_seconds": row.mean_epoch_seconds,
        "empty_label_fpr": row.empty_label_fpr,
        "run_dir": row.run_dir,
        "best_pt": row.best_pt,
        "status": row.status,
    }


def _metric_from_payload(payload: Mapping[str, Any]) -> metrics.TrialMetrics:
    best = payload["best"]
    return metrics.TrialMetrics(
        trial_id=str(payload["trial_id"]),
        model_family=str(payload["model_family"]),
        profile=str(payload["profile"]),
        imgsz=int(payload["imgsz"]),
        batch=int(payload["batch"]),
        best=metrics.EpochMetrics(
            epoch=int(best["epoch"]),
            epoch_seconds=float(best["epoch_seconds"]),
            precision=float(best["precision"]),
            recall=float(best["recall"]),
            map50=float(best["map50"]),
            map50_95=float(best["map50_95"]),
        ),
        mean_epoch_seconds=float(payload["mean_epoch_seconds"]),
        empty_label_fpr=float(payload["empty_label_fpr"]),
        run_dir=str(payload["run_dir"]),
        best_pt=str(payload["best_pt"]),
        status=str(payload.get("status", "complete")),
    )


def _attempt_is_valid(attempt: Mapping[str, Any]) -> bool:
    if attempt.get("status") != "complete" or not attempt.get("metrics"):
        return False
    artifacts = attempt.get("artifacts") or {}
    results_csv = Path(str(artifacts.get("results_csv", "")))
    best_pt = Path(str(artifacts.get("best_pt", "")))
    if not results_csv.is_file() or not best_pt.is_file():
        return False
    if attempt.get("stage") == "final":
        evaluation = Path(str(artifacts.get("evaluation_metrics", "")))
        grouped = Path(str(artifacts.get("grouped_metrics", "")))
        if not evaluation.is_file() or not grouped.is_file():
            return False
    try:
        metrics.read_training_summary(results_csv)
    except (OSError, ValueError):
        return False
    return True


def _initial_state(plan_digest: str, args: argparse.Namespace) -> dict[str, Any]:
    return {
        "plan_hash": plan_digest,
        "experiment_id": args.experiment_id,
        "stage": args.stage,
        "attempts": [],
        "finalists": [],
    }


def _load_state(path: Path, plan_digest: str, args: argparse.Namespace) -> dict[str, Any]:
    if not path.is_file():
        return _initial_state(plan_digest, args)
    state = json.loads(path.read_text(encoding="utf-8"))
    if state.get("plan_hash") != plan_digest:
        raise ValueError("experiment plan hash does not match existing state")
    if state.get("experiment_id") != args.experiment_id:
        raise ValueError("experiment ID does not match existing state")
    state["stage"] = args.stage
    state.setdefault("attempts", [])
    state.setdefault("finalists", [])
    return state


def _completed_attempt(state: Mapping[str, Any], source_trial_id: str) -> Mapping[str, Any] | None:
    candidates = [
        attempt
        for attempt in state.get("attempts", [])
        if attempt.get("source_trial_id") == source_trial_id and attempt.get("status") == "complete"
    ]
    return next((attempt for attempt in reversed(candidates) if _attempt_is_valid(attempt)), None)


def _write_failed_csv(path: Path, attempts: Sequence[Mapping[str, Any]]) -> None:
    columns = (
        "attempt_id",
        "trial_id",
        "stage",
        "status",
        "requested_batch",
        "effective_batch",
        "run_dir",
        "started_at",
        "ended_at",
        "error",
    )
    rows = [attempt for attempt in attempts if attempt.get("status") in {"failed", "oom"}]
    temporary = path.with_name(f".{path.name}.tmp")
    path.parent.mkdir(parents=True, exist_ok=True)
    with temporary.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=columns, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(sorted(rows, key=lambda row: str(row.get("attempt_id", ""))))
    temporary.replace(path)


def _write_aggregate_grouped(path: Path, attempts: Sequence[Mapping[str, Any]]) -> None:
    columns = ("trial_id", "split", *metrics.GroupedMetrics.__dataclass_fields__)
    rows = []
    for attempt in attempts:
        if attempt.get("stage") != "final" or attempt.get("status") != "complete":
            continue
        grouped_path = Path(str((attempt.get("artifacts") or {}).get("grouped_metrics", "")))
        if not grouped_path.is_file():
            continue
        with grouped_path.open(newline="", encoding="utf-8") as file:
            for row in csv.DictReader(file):
                rows.append({"trial_id": attempt["attempt_id"], "split": "test", **row})
    temporary = path.with_name(f".{path.name}.tmp")
    path.parent.mkdir(parents=True, exist_ok=True)
    with temporary.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        writer.writerows(sorted(rows, key=lambda row: (row["trial_id"], row["group_dimension"], row["group_value"])))
    temporary.replace(path)


def _write_reports(experiment_root: Path, state: Mapping[str, Any]) -> None:
    complete = [
        _metric_from_payload(attempt["metrics"])
        for attempt in state.get("attempts", [])
        if attempt.get("status") == "complete" and attempt.get("metrics") and _attempt_is_valid(attempt)
    ]
    metrics.write_leaderboard_csv(experiment_root / "leaderboard.csv", complete)
    metrics.write_leaderboard_markdown(experiment_root / "leaderboard.md", complete)
    _write_failed_csv(experiment_root / "failed_trials.csv", state.get("attempts", []))
    _write_aggregate_grouped(experiment_root / "grouped_metrics.csv", state.get("attempts", []))


def _attempt_record(trial: TrialSpec, source_trial: TrialSpec, requested_batch: int) -> dict[str, Any]:
    run_dir = trial.project.resolve() / trial.trial_id
    return {
        "attempt_id": trial.trial_id,
        "trial_id": trial.trial_id,
        "source_trial_id": source_trial.trial_id,
        "stage": trial.stage,
        "status": "running",
        "started_at": _timestamp(),
        "ended_at": None,
        "requested_batch": requested_batch,
        "effective_batch": trial.batch,
        "run_dir": str(run_dir),
        "error": "",
        "resolved_args": build_train_kwargs(trial, argparse.Namespace()),
        "model_path": str(trial.model_path),
        "model_sha256": trial.model_sha256,
        "data_yaml_sha256": trial.data_yaml_sha256,
        "manifest_sha256": trial.manifest_sha256,
        "data_yaml": str(trial.data_yaml),
        "manifest": str(trial.manifest),
        "metrics": None,
        "peak_gpu_memory_gb": None,
        "artifacts": {},
    }


def _update_attempt_peak(attempt: dict[str, Any]) -> None:
    execution_metadata = Path(str(attempt["run_dir"])) / "execution_metadata.json"
    if execution_metadata.is_file():
        provenance = json.loads(execution_metadata.read_text(encoding="utf-8"))
        attempt["peak_gpu_memory_gb"] = provenance.get("peak_gpu_memory_gb")
        attempt["peak_gpu_memory_status"] = provenance.get("peak_gpu_memory_status", "unavailable")
    else:
        attempt["peak_gpu_memory_status"] = "unavailable"


def _execute_trial(
    trial: TrialSpec,
    args: argparse.Namespace,
    state: dict[str, Any],
    state_path: Path,
    experiment_root: Path,
    yolo_factory: Callable[..., object],
) -> bool:
    existing = [attempt for attempt in state["attempts"] if attempt.get("source_trial_id") == trial.trial_id]
    valid = _completed_attempt(state, trial.trial_id)
    if valid is not None:
        return True
    original_run_dir = trial.project.resolve() / trial.trial_id
    stale_directory = original_run_dir.exists()
    if (existing or stale_directory) and not args.resume:
        raise RuntimeError(f"stale or partial trial requires --resume: {trial.trial_id}")

    requested_batch = trial.batch
    retry_batch = max(1, trial.batch // 2)
    oom_retry_consumed = any(attempt.get("status") == "oom" for attempt in existing)
    half_retry_recorded = any(
        attempt.get("requested_batch") == requested_batch and attempt.get("effective_batch") == retry_batch
        for attempt in existing
    )
    pending_half_retry = (
        args.resume and args.oom_retry and trial.batch > 1 and oom_retry_consumed and not half_retry_recorded
    )
    if pending_half_retry:
        candidates = [replace(trial, trial_id=f"{trial.trial_id}__oom_b{retry_batch}", batch=retry_batch)]
    else:
        primary = trial
        if args.resume and (existing or stale_directory):
            resume_index = len(existing) + 1
            resume_id = f"{trial.trial_id}__resume{resume_index}"
            while (trial.project.resolve() / resume_id).exists():
                resume_index += 1
                resume_id = f"{trial.trial_id}__resume{resume_index}"
            primary = replace(trial, trial_id=resume_id)
        candidates = [primary]
        if args.oom_retry and trial.batch > 1 and not oom_retry_consumed:
            candidates.append(replace(primary, trial_id=f"{primary.trial_id}__oom_b{retry_batch}", batch=retry_batch))

    for index, candidate in enumerate(candidates):
        attempt = _attempt_record(candidate, trial, requested_batch)
        state["attempts"].append(attempt)
        atomic_write_json(state_path, state)
        try:
            result = run_trial(candidate, args, yolo_factory=yolo_factory)
        except Exception as error:
            retryable = index == 0 and len(candidates) > 1 and isinstance(error, TrainingOOMError)
            attempt["status"] = "oom" if retryable else "failed"
            attempt["ended_at"] = _timestamp()
            attempt["error"] = f"{error.__class__.__name__}: {error}"
            _update_attempt_peak(attempt)
            atomic_write_json(state_path, state)
            _write_reports(experiment_root, state)
            if retryable:
                continue
            return False

        run_dir = Path(result.run_dir)
        artifacts = {
            "results_csv": str(run_dir / "results.csv"),
            "best_pt": result.best_pt,
            "last_pt": str(run_dir / "weights" / "last.pt"),
            "evaluation_metrics": str(run_dir / "evaluation_metrics.json"),
            "execution_metadata": str(run_dir / "execution_metadata.json"),
        }
        grouped = run_dir / "grouped_metrics.csv"
        if grouped.is_file():
            artifacts["grouped_metrics"] = str(grouped)
        attempt.update(
            status="complete",
            ended_at=_timestamp(),
            metrics=_metric_payload(result),
            artifacts=artifacts,
        )
        _update_attempt_peak(attempt)
        atomic_write_json(state_path, state)
        _write_reports(experiment_root, state)
        return True
    return False


def _screen_metrics(state: Mapping[str, Any]) -> list[metrics.TrialMetrics]:
    latest: dict[str, Mapping[str, Any]] = {}
    for attempt in state.get("attempts", []):
        if attempt.get("stage") == "screen" and _attempt_is_valid(attempt):
            latest[str(attempt.get("source_trial_id"))] = attempt
    return [_metric_from_payload(attempt["metrics"]) for attempt in latest.values()]


def _selected_final_trials(plan: Sequence[TrialSpec], state: dict[str, Any], count: int) -> list[TrialSpec]:
    completed = _screen_metrics(state)
    if len(completed) != len(plan):
        raise RuntimeError(f"final stage requires {len(plan)} valid completed screening trials, found {len(completed)}")
    selected = metrics.select_finalists(completed, count=count)
    finals = []
    for row in selected:
        attempt = next(attempt for attempt in state["attempts"] if attempt.get("attempt_id") == row.trial_id)
        source = next(trial for trial in plan if trial.trial_id == attempt["source_trial_id"])
        finals.append(_final_trial(source, effective_batch=int(attempt["effective_batch"])))
    state["finalists"] = [trial.trial_id for trial in finals]
    return finals


def _run_workflow(
    args: argparse.Namespace,
    experiment_root: Path,
    yolo_factory: Callable[..., object] | None,
) -> int:
    plan = build_experiment_plan(args)
    digest = plan_hash(plan)
    payload = {
        "plan_hash": digest,
        "experiment_id": args.experiment_id,
        "stage": args.stage,
        "trials": [_trial_record(trial) for trial in plan],
    }
    plan_path = experiment_root / "experiment_plan.json"
    state_path = experiment_root / "state.json"
    if plan_path.is_file():
        existing_plan = json.loads(plan_path.read_text(encoding="utf-8"))
        if existing_plan.get("plan_hash") != digest:
            raise ValueError("experiment plan hash does not match existing plan")
    state = _load_state(state_path, digest, args)
    atomic_write_json(plan_path, payload)
    atomic_write_json(state_path, state)
    selected_plan = plan[: args.max_trials] if args.max_trials is not None else plan
    for trial in selected_plan:
        print(json.dumps(_trial_record(trial), sort_keys=True))
    if args.dry_run:
        return 0
    if yolo_factory is None:
        from ultralytics import YOLO

        yolo_factory = YOLO
    failed = False
    if args.stage in {"screen", "all"}:
        for trial in selected_plan:
            try:
                success = _execute_trial(trial, args, state, state_path, experiment_root, yolo_factory)
            except RuntimeError:
                raise
            failed |= not success
            if not success and args.fail_fast:
                return 1
    if args.stage in {"final", "all"}:
        finals = _selected_final_trials(plan, state, args.finalists)
        atomic_write_json(state_path, state)
        for trial in finals:
            success = _execute_trial(trial, args, state, state_path, experiment_root, yolo_factory)
            failed |= not success
            if not success and args.fail_fast:
                return 1
    _write_reports(experiment_root, state)
    return 1 if failed else 0


def main(argv: Sequence[str] | None = None, yolo_factory: Callable[..., object] | None = None) -> int:
    """Execute a resumable screening/final workflow with atomic state and reports."""
    args = build_parser().parse_args(argv)
    experiment_root = _experiment_root(args.project, args.experiment_id)
    # Fail before lock creation can produce experiment output. The plan is built
    # again while locked so content hashes cannot be reused across the boundary.
    build_experiment_plan(args)
    with _experiment_lock(experiment_root):
        return _run_workflow(args, experiment_root, yolo_factory)


if __name__ == "__main__":
    raise SystemExit(main())
