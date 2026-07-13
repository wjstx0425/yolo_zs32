# Ultralytics AGPL-3.0 License - https://ultralytics.com/license

"""Run C789 YOLO prediction and export anomalib fusion-compatible CSV rows."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from _common import (
    defect_type_from_class,
    infer_slot_id,
    load_manifest_index,
    metadata_for_path,
    parse_overrides,
    part_id_for_path,
    write_fusion_csv,
)

from ultralytics import YOLO


def build_parser() -> argparse.ArgumentParser:
    """Build the predict/export parser."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--model", type=Path, required=True, help="YOLO weights.")
    parser.add_argument("--source", required=True, help="Slot crop file, directory, glob, or image list.")
    parser.add_argument("--data-yaml", type=Path, help="Dataset YAML used for class names.")
    parser.add_argument(
        "--manifest",
        type=Path,
        help="Optional anomalib part_crop_manifest.csv or export_manifest.csv.",
    )
    parser.add_argument("--side", choices=("top", "bottom"), required=True, help="Inspection side.")
    parser.add_argument("--view", default="uniform", help="Capture view/light label.")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold.")
    parser.add_argument("--iou", type=float, default=0.7, help="NMS IoU threshold.")
    parser.add_argument("--imgsz", type=int, default=1024, help="Prediction image size.")
    parser.add_argument("--device", default=None, help="CUDA device, device list, cpu, or auto.")
    parser.add_argument("--batch", type=int, default=1, help="Prediction batch size.")
    parser.add_argument("--max-det", type=int, default=300, help="Maximum detections per image.")
    parser.add_argument("--classes", nargs="+", type=int, help="Optional class ids to keep.")
    parser.add_argument("--project", default="runs/c789", help="Ultralytics project output directory.")
    parser.add_argument("--name", default="defect_predict", help="Ultralytics run name.")
    parser.add_argument("--exist-ok", action="store_true", help="Allow reusing project/name.")
    parser.add_argument("--save", action="store_true", help="Save annotated prediction images.")
    parser.add_argument("--save-txt", action="store_true", help="Save YOLO txt predictions.")
    parser.add_argument("--save-conf", action="store_true", help="Save confidences in txt predictions.")
    parser.add_argument("--fusion-csv", type=Path, required=True, help="Output anomalib yolo_predictions.csv.")
    parser.add_argument(
        "--skip-predict",
        action="store_true",
        help="Skip prediction and only export the model if requested.",
    )
    parser.add_argument("--skip-export", action="store_true", help="Skip model export.")
    parser.add_argument("--export-format", default="onnx", help="Ultralytics export format.")
    parser.add_argument("--export-imgsz", type=int, default=1024, help="Export image size.")
    parser.add_argument("--dynamic", action="store_true", help="Use dynamic axes for supported exports.")
    parser.add_argument("--nms", action="store_true", help="Include NMS in supported exports.")
    parser.add_argument("--opset", type=int, help="ONNX opset.")
    parser.add_argument("--simplify", action="store_true", help="Simplify supported exports.")
    parser.add_argument("--quantize", action="store_true", help="Quantize supported exports.")
    parser.add_argument("--fraction", type=float, default=1.0, help="Calibration fraction for supported exports.")
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        help="Extra Ultralytics predict override as key=value.",
    )
    return parser


def _max_detection(result: Any) -> tuple[int, float] | tuple[None, float]:
    """Return the class id and confidence for the highest-confidence detection."""
    boxes = getattr(result, "boxes", None)
    if boxes is None or len(boxes) == 0:
        return None, 0.0
    confidences = boxes.conf
    best_index = int(confidences.argmax().item())
    class_id = int(boxes.cls[best_index].item()) if boxes.cls is not None else None
    score = float(confidences[best_index].item())
    return class_id, score


def _evidence_path(result: Any, save_enabled: bool) -> str:
    """Best-effort path to the annotated evidence image."""
    if not save_enabled:
        return ""
    save_dir = getattr(result, "save_dir", None)
    image_path = getattr(result, "path", None)
    if save_dir is None or image_path is None:
        return ""
    return str(Path(save_dir) / Path(image_path).name)


def rows_from_results(
    results: list[Any],
    manifest: Path | None,
    side: str,
    view: str,
    threshold: float,
    save: bool,
) -> list[dict[str, Any]]:
    """Convert Ultralytics results into anomalib fusion branch rows."""
    index = load_manifest_index(manifest)
    rows: list[dict[str, Any]] = []
    for result in results:
        image_path = Path(result.path)
        metadata = metadata_for_path(index, image_path)
        class_id, score = _max_detection(result)
        pred_label = 1 if score >= threshold else 0
        defect_type = defect_type_from_class(class_id, getattr(result, "names", None)) if pred_label else ""
        rows.append(
            {
                "part_id": part_id_for_path(image_path, metadata),
                "side": side,
                "view": view,
                "slot_id": infer_slot_id(image_path, metadata),
                "branch": "yolo",
                "pred_label": pred_label,
                "score": f"{score:.6f}",
                "threshold": f"{threshold:.6f}",
                "defect_type": defect_type,
                "reason": "yolo box confidence above threshold" if pred_label else "no yolo box above threshold",
                "source_path": str(image_path),
                "evidence_path": _evidence_path(result, save),
                "status": "",
            },
        )
    return rows


def main() -> None:
    """Run YOLO predict/export and write fusion CSV."""
    args = build_parser().parse_args()
    overrides = parse_overrides(args.override)
    model = YOLO(str(args.model))
    if not args.skip_predict:
        results = model.predict(
            source=args.source,
            data=str(args.data_yaml) if args.data_yaml else None,
            conf=args.conf,
            iou=args.iou,
            imgsz=args.imgsz,
            device=args.device,
            batch=args.batch,
            max_det=args.max_det,
            classes=args.classes,
            project=args.project,
            name=args.name,
            exist_ok=args.exist_ok,
            save=args.save,
            save_txt=args.save_txt,
            save_conf=args.save_conf,
            **overrides,
        )
        rows = rows_from_results(list(results), args.manifest, args.side, args.view, args.conf, args.save)
        write_fusion_csv(args.fusion_csv, rows)
        print(f"wrote fusion CSV: {args.fusion_csv}")
    if not args.skip_export:
        exported = model.export(
            format=args.export_format,
            imgsz=args.export_imgsz,
            dynamic=args.dynamic,
            nms=args.nms,
            opset=args.opset,
            simplify=args.simplify,
            int8=args.quantize,
            fraction=args.fraction,
        )
        print(f"exported model: {exported}")


if __name__ == "__main__":
    main()
