"""Pure metrics, ranking, and reporting helpers for the ZS32 YOLO sweep."""

from __future__ import annotations

import csv
import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class EpochMetrics:
    """Metrics and duration captured for one completed training epoch."""

    epoch: int
    epoch_seconds: float
    precision: float
    recall: float
    map50: float
    map50_95: float


@dataclass(frozen=True)
class TrainingSummary:
    """Best epoch and full-run mean epoch duration from one results CSV."""

    best: EpochMetrics
    mean_epoch_seconds: float


@dataclass(frozen=True)
class TrialMetrics:
    """Immutable summary of one sweep trial."""

    trial_id: str
    model_family: str
    profile: str
    imgsz: int
    batch: int
    best: EpochMetrics
    mean_epoch_seconds: float
    empty_label_fpr: float
    run_dir: str
    best_pt: str
    status: str = "complete"


@dataclass(frozen=True)
class GroupedMetrics:
    """Detection counts and rates for one deterministic manifest group."""

    group_dimension: str
    group_value: str
    image_count: int
    gt_box_count: int
    predicted_box_count: int
    matched_box_count: int
    box_recall: float
    positive_image_count: int
    recalled_image_count: int
    image_recall: float
    empty_image_count: int
    false_positive_image_count: int
    empty_label_fpr: float


_RESULT_COLUMNS = {
    "epoch",
    "time",
    "metrics/precision(B)",
    "metrics/recall(B)",
    "metrics/mAP50(B)",
    "metrics/mAP50-95(B)",
}


def _finite_float(row: Mapping[str, str], column: str, row_number: int) -> float:
    raw = row.get(column)
    if raw is None or not raw.strip():
        raise ValueError(f"results.csv row {row_number} has missing or empty {column!r}")
    try:
        value = float(raw)
    except ValueError as error:
        raise ValueError(f"results.csv row {row_number} has invalid {column!r}: {raw!r}") from error
    if not math.isfinite(value):
        raise ValueError(f"results.csv row {row_number} has non-finite {column!r}: {raw!r}")
    return value


def read_training_summary(path: str | Path) -> TrainingSummary:
    """Return best metrics and mean epoch seconds parsed from one results CSV."""
    path = Path(path)
    with path.open(newline="", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)
        columns = set(reader.fieldnames or ())
        missing = sorted(_RESULT_COLUMNS - columns)
        if missing:
            raise ValueError(f"results.csv is missing required columns: {', '.join(missing)}")

        epochs: list[EpochMetrics] = []
        previous_cumulative_time = 0.0
        for row_number, row in enumerate(reader, start=2):
            epoch_value = _finite_float(row, "epoch", row_number)
            if not epoch_value.is_integer() or epoch_value < 0:
                raise ValueError(f"results.csv row {row_number} has invalid epoch: {epoch_value}")
            cumulative_time = _finite_float(row, "time", row_number)
            epoch_seconds = cumulative_time - previous_cumulative_time
            if epoch_seconds < 0:
                raise ValueError(f"results.csv row {row_number} has decreasing cumulative time")
            epochs.append(
                EpochMetrics(
                    epoch=int(epoch_value),
                    epoch_seconds=epoch_seconds,
                    precision=_finite_float(row, "metrics/precision(B)", row_number),
                    recall=_finite_float(row, "metrics/recall(B)", row_number),
                    map50=_finite_float(row, "metrics/mAP50(B)", row_number),
                    map50_95=_finite_float(row, "metrics/mAP50-95(B)", row_number),
                )
            )
            previous_cumulative_time = cumulative_time

    if not epochs:
        raise ValueError("results.csv contains no epoch rows")
    best = max(
        epochs,
        key=lambda row: (
            row.map50_95,
            row.recall,
            row.map50,
            -row.epoch_seconds,
            -row.epoch,
        ),
    )
    return TrainingSummary(best=best, mean_epoch_seconds=previous_cumulative_time / len(epochs))


def read_best_epoch(path: str | Path) -> EpochMetrics:
    """Return the best metrics row with its duration derived from cumulative time."""
    return read_training_summary(path).best


def _ranking_key(row: TrialMetrics) -> tuple[float, float, float, float, float, str]:
    return (
        -row.best.map50_95,
        -row.best.recall,
        row.empty_label_fpr,
        -row.best.map50,
        row.mean_epoch_seconds,
        row.trial_id,
    )


def rank_trials(rows: Iterable[TrialMetrics]) -> list[TrialMetrics]:
    """Rank trials by accuracy, recall, false positives, accuracy, then speed."""
    return sorted(rows, key=_ranking_key)


def select_finalists(rows: Iterable[TrialMetrics], count: int = 3) -> list[TrialMetrics]:
    """Select best n, best m, and then the best remaining trials."""
    if count < 0:
        raise ValueError("count must be non-negative")
    ranked = rank_trials(rows)
    selected: list[TrialMetrics] = []
    selected_ids: set[str] = set()

    def add(row: TrialMetrics) -> None:
        if len(selected) < count and row.trial_id not in selected_ids:
            selected.append(row)
            selected_ids.add(row.trial_id)

    for family in ("n", "m"):
        candidate = next((row for row in ranked if row.model_family == family), None)
        if candidate is not None:
            add(candidate)
    for row in ranked:
        add(row)
    return selected


def _validated_box(box: Sequence[float]) -> tuple[float, float, float, float]:
    if len(box) != 4:
        raise ValueError(f"box must contain four xyxy values, got {len(box)}")
    values = tuple(float(value) for value in box)
    if not all(math.isfinite(value) for value in values):
        raise ValueError(f"box contains non-finite values: {box!r}")
    x1, y1, x2, y2 = values
    if x2 <= x1 or y2 <= y1:
        raise ValueError(f"box must have positive xyxy area: {box!r}")
    return x1, y1, x2, y2


def box_iou(box_a: Sequence[float], box_b: Sequence[float]) -> float:
    """Return intersection-over-union for two positive-area xyxy boxes."""
    ax1, ay1, ax2, ay2 = _validated_box(box_a)
    bx1, by1, bx2, by2 = _validated_box(box_b)
    intersection_width = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    intersection_height = max(0.0, min(ay2, by2) - max(ay1, by1))
    intersection = intersection_width * intersection_height
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    return intersection / (area_a + area_b - intersection)


def match_boxes(
    ground_truth: Sequence[Sequence[float]],
    predictions: Sequence[Sequence[float]],
    iou_threshold: float = 0.5,
) -> list[tuple[int, int, float]]:
    """Return deterministic maximum-cardinality one-to-one matches at an inclusive IoU threshold."""
    if not math.isfinite(iou_threshold) or not 0.0 <= iou_threshold <= 1.0:
        raise ValueError("iou_threshold must be finite and in [0, 1]")
    validated_ground_truth = [_validated_box(box) for box in ground_truth]
    validated_predictions = [_validated_box(box) for box in predictions]
    adjacency: list[list[tuple[int, float]]] = [[] for _ in validated_ground_truth]
    for gt_index, gt_box in enumerate(validated_ground_truth):
        for prediction_index, prediction_box in enumerate(validated_predictions):
            iou = box_iou(gt_box, prediction_box)
            if iou >= iou_threshold:
                adjacency[gt_index].append((prediction_index, iou))
    for candidates in adjacency:
        candidates.sort(key=lambda item: (-item[1], item[0]))

    prediction_to_gt: dict[int, int] = {}

    def augment(gt_index: int, seen_predictions: set[int]) -> bool:
        for prediction_index, _ in adjacency[gt_index]:
            if prediction_index in seen_predictions:
                continue
            seen_predictions.add(prediction_index)
            previous_gt = prediction_to_gt.get(prediction_index)
            if previous_gt is None or augment(previous_gt, seen_predictions):
                prediction_to_gt[prediction_index] = gt_index
                return True
        return False

    for gt_index in sorted(range(len(adjacency)), key=lambda index: (len(adjacency[index]), index)):
        augment(gt_index, set())

    iou_by_pair = {
        (gt_index, prediction_index): iou
        for gt_index, candidates in enumerate(adjacency)
        for prediction_index, iou in candidates
    }
    return sorted(
        (gt_index, prediction_index, iou_by_pair[(gt_index, prediction_index)])
        for prediction_index, gt_index in prediction_to_gt.items()
    )


@dataclass(frozen=True)
class _ImageMetrics:
    metadata: Mapping[str, str]
    gt_box_count: int
    predicted_box_count: int
    matched_box_count: int


_GROUP_DIMENSIONS = ("view", "hand", "defect_type", "kind")


def _manifest_image_key(row: Mapping[str, Any]) -> str:
    for key in ("output_image", "image_path", "source_path", "path", "image_id"):
        value = str(row.get(key, "")).strip()
        if value:
            return value
    raise ValueError("manifest row has no image identifier")


def _canonical_image_key(value: str | Path, path_root: str | Path | None) -> str:
    path = Path(value)
    if not path.is_absolute() and path_root is not None:
        path = Path(path_root) / path
    return str(path.resolve(strict=False)) if path.is_absolute() else path.as_posix()


def _canonical_box_mapping(
    rows: Mapping[str | Path, Sequence[Sequence[float]]],
    path_root: str | Path | None,
) -> dict[str, Sequence[Sequence[float]]]:
    canonical: dict[str, Sequence[Sequence[float]]] = {}
    for key, boxes in rows.items():
        canonical_key = _canonical_image_key(key, path_root)
        if canonical_key in canonical:
            raise ValueError(f"duplicate canonical image path: {canonical_key}")
        canonical[canonical_key] = boxes
    return canonical


def _aggregate_group(dimension: str, value: str, images: Sequence[_ImageMetrics]) -> GroupedMetrics:
    gt_box_count = sum(image.gt_box_count for image in images)
    predicted_box_count = sum(image.predicted_box_count for image in images)
    matched_box_count = sum(image.matched_box_count for image in images)
    positive = [image for image in images if image.gt_box_count]
    empty = [image for image in images if not image.gt_box_count]
    recalled_image_count = sum(image.matched_box_count > 0 for image in positive)
    false_positive_image_count = sum(image.predicted_box_count > 0 for image in empty)
    return GroupedMetrics(
        group_dimension=dimension,
        group_value=value,
        image_count=len(images),
        gt_box_count=gt_box_count,
        predicted_box_count=predicted_box_count,
        matched_box_count=matched_box_count,
        box_recall=matched_box_count / gt_box_count if gt_box_count else 0.0,
        positive_image_count=len(positive),
        recalled_image_count=recalled_image_count,
        image_recall=recalled_image_count / len(positive) if positive else 0.0,
        empty_image_count=len(empty),
        false_positive_image_count=false_positive_image_count,
        empty_label_fpr=false_positive_image_count / len(empty) if empty else 0.0,
    )


def summarize_grouped_predictions(
    manifest_rows: Iterable[Mapping[str, Any]],
    ground_truth: Mapping[str | Path, Sequence[Sequence[float]]],
    predictions: Mapping[str | Path, Sequence[Sequence[float]]],
    iou_threshold: float = 0.5,
    *,
    path_root: str | Path | None = None,
) -> list[GroupedMetrics]:
    """Summarize ZS32 metrics using full paths resolved against an optional repository root."""
    ground_truth_by_key = _canonical_box_mapping(ground_truth, path_root)
    predictions_by_key = _canonical_box_mapping(predictions, path_root)
    images: list[_ImageMetrics] = []
    seen: set[str] = set()
    for manifest_row in manifest_rows:
        image_key = _canonical_image_key(_manifest_image_key(manifest_row), path_root)
        if image_key in seen:
            raise ValueError(f"duplicate manifest image identifier: {image_key}")
        seen.add(image_key)
        if image_key not in ground_truth_by_key:
            raise ValueError(f"ground truth is missing manifest image: {image_key}")
        if image_key not in predictions_by_key:
            raise ValueError(f"predictions are missing manifest image: {image_key}")
        gt_boxes = ground_truth_by_key[image_key]
        predicted_boxes = predictions_by_key[image_key]
        matches = match_boxes(gt_boxes, predicted_boxes, iou_threshold=iou_threshold)
        images.append(
            _ImageMetrics(
                metadata={dimension: str(manifest_row.get(dimension, "")).strip() for dimension in _GROUP_DIMENSIONS},
                gt_box_count=len(gt_boxes),
                predicted_box_count=len(predicted_boxes),
                matched_box_count=len(matches),
            )
        )
    if not images:
        raise ValueError("manifest contains no image rows")

    grouped = [_aggregate_group("all", "all", images)]
    for dimension in _GROUP_DIMENSIONS:
        for value in sorted({image.metadata[dimension] for image in images}):
            members = [image for image in images if image.metadata[dimension] == value]
            grouped.append(_aggregate_group(dimension, value, members))
    return grouped


_LEADERBOARD_COLUMNS = (
    "rank",
    "trial_id",
    "model_family",
    "profile",
    "imgsz",
    "batch",
    "status",
    "best_epoch",
    "mean_epoch_seconds",
    "precision",
    "recall",
    "map50",
    "map50_95",
    "empty_label_fpr",
    "run_dir",
    "best_pt",
)

_GROUPED_COLUMNS = tuple(GroupedMetrics.__dataclass_fields__)


def _float_text(value: float) -> str:
    return f"{value:.6f}"


def _leaderboard_records(rows: Iterable[TrialMetrics]) -> list[dict[str, Any]]:
    records = []
    for rank, row in enumerate(rank_trials(rows), start=1):
        records.append(
            {
                "rank": rank,
                "trial_id": row.trial_id,
                "model_family": row.model_family,
                "profile": row.profile,
                "imgsz": row.imgsz,
                "batch": row.batch,
                "status": row.status,
                "best_epoch": row.best.epoch,
                "mean_epoch_seconds": _float_text(row.mean_epoch_seconds),
                "precision": _float_text(row.best.precision),
                "recall": _float_text(row.best.recall),
                "map50": _float_text(row.best.map50),
                "map50_95": _float_text(row.best.map50_95),
                "empty_label_fpr": _float_text(row.empty_label_fpr),
                "run_dir": row.run_dir,
                "best_pt": row.best_pt,
            }
        )
    return records


def _write_csv(path: str | Path, columns: Sequence[str], rows: Iterable[Mapping[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_leaderboard_csv(path: str | Path, rows: Iterable[TrialMetrics]) -> None:
    """Write the ranked sweep leaderboard with a stable CSV schema."""
    _write_csv(path, _LEADERBOARD_COLUMNS, _leaderboard_records(rows))


def _markdown_text(value: Any) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def write_leaderboard_markdown(path: str | Path, rows: Iterable[TrialMetrics]) -> None:
    """Write the ranked sweep leaderboard as deterministic Markdown."""
    records = _leaderboard_records(rows)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# ZS32 YOLO Sweep Leaderboard",
        "",
        "| " + " | ".join(_LEADERBOARD_COLUMNS) + " |",
        "| " + " | ".join("---" for _ in _LEADERBOARD_COLUMNS) + " |",
    ]
    lines.extend(
        "| " + " | ".join(_markdown_text(record[column]) for column in _LEADERBOARD_COLUMNS) + " |"
        for record in records
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_grouped_metrics_csv(path: str | Path, rows: Iterable[GroupedMetrics]) -> None:
    """Write grouped detection metrics with a stable CSV schema."""
    records = []
    dimension_order = {"all": 0, **{dimension: index for index, dimension in enumerate(_GROUP_DIMENSIONS, start=1)}}
    ordered_rows = sorted(
        rows,
        key=lambda row: (dimension_order.get(row.group_dimension, len(dimension_order)), row.group_value),
    )
    for row in ordered_rows:
        record = {column: getattr(row, column) for column in _GROUPED_COLUMNS}
        for column in ("box_recall", "image_recall", "empty_label_fpr"):
            record[column] = _float_text(record[column])
        records.append(record)
    _write_csv(path, _GROUPED_COLUMNS, records)
