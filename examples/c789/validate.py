# Ultralytics AGPL-3.0 License - https://ultralytics.com/license

"""Validate a C789 YOLO defect detector."""

from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO

from _common import parse_overrides


def build_parser() -> argparse.ArgumentParser:
    """Build the validation parser."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--model", type=Path, required=True, help="YOLO weights, for example weights/best.pt.")
    parser.add_argument("--data-yaml", type=Path, required=True, help="Ultralytics detection dataset YAML.")
    parser.add_argument("--split", choices=("train", "val", "test"), default="val", help="Dataset split to validate.")
    parser.add_argument("--imgsz", type=int, default=1024, help="Validation image size.")
    parser.add_argument("--batch", type=int, default=8, help="Batch size.")
    parser.add_argument("--device", default=None, help="CUDA device, device list, cpu, or auto.")
    parser.add_argument("--workers", type=int, default=8, help="Data-loader workers.")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold.")
    parser.add_argument("--iou", type=float, default=0.7, help="NMS IoU threshold.")
    parser.add_argument("--max-det", type=int, default=300, help="Maximum detections per image.")
    parser.add_argument("--project", default="runs/c789", help="Ultralytics project output directory.")
    parser.add_argument("--name", default="defect_val", help="Ultralytics run name.")
    parser.add_argument("--exist-ok", action="store_true", help="Allow reusing project/name.")
    parser.add_argument("--save-json", action="store_true", help="Save COCO JSON if supported.")
    parser.add_argument("--save-txt", action="store_true", help="Save prediction txt labels.")
    parser.add_argument("--save-conf", action="store_true", help="Save confidences in txt labels.")
    parser.add_argument(
        "--plots",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Save validation plots.",
    )
    parser.add_argument("--override", action="append", default=[], help="Extra Ultralytics val override as key=value.")
    return parser


def main() -> None:
    """Run YOLO validation."""
    args = build_parser().parse_args()
    overrides = parse_overrides(args.override)
    model = YOLO(str(args.model))
    model.val(
        data=str(args.data_yaml),
        split=args.split,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        conf=args.conf,
        iou=args.iou,
        max_det=args.max_det,
        project=args.project,
        name=args.name,
        exist_ok=args.exist_ok,
        save_json=args.save_json,
        save_txt=args.save_txt,
        save_conf=args.save_conf,
        plots=args.plots,
        **overrides,
    )


if __name__ == "__main__":
    main()
