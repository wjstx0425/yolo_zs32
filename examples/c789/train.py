# Ultralytics AGPL-3.0 License - https://ultralytics.com/license

"""Train a C789 YOLO defect detector from an Ultralytics data.yaml file."""

from __future__ import annotations

import argparse
from pathlib import Path

from _common import parse_overrides

from ultralytics import YOLO


def parse_batch(value: str) -> int | float:
    """Parse an Ultralytics batch value as int or float."""
    try:
        return int(value)
    except ValueError:
        return float(value)


def build_parser() -> argparse.ArgumentParser:
    """Build the training parser."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--data-yaml", type=Path, required=True, help="Ultralytics detection dataset YAML.")
    parser.add_argument("--model", default="yolo26n.pt", help="Pretrained model or model YAML.")
    parser.add_argument("--task", choices=("detect", "segment"), default="detect", help="YOLO task.")
    parser.add_argument("--epochs", type=int, default=100, help="Training epochs.")
    parser.add_argument("--imgsz", type=int, default=1024, help="Training image size.")
    parser.add_argument("--batch", type=parse_batch, default=8, help="Batch size, or Ultralytics auto-batch value.")
    parser.add_argument("--device", default=None, help="CUDA device, device list, cpu, or auto.")
    parser.add_argument("--workers", type=int, default=8, help="Data-loader workers.")
    parser.add_argument("--project", default="runs/c789", help="Ultralytics project output directory.")
    parser.add_argument("--name", default="defect_train", help="Ultralytics run name.")
    parser.add_argument("--exist-ok", action="store_true", help="Allow reusing project/name.")
    parser.add_argument("--seed", type=int, default=0, help="Training seed.")
    parser.add_argument("--resume", action="store_true", help="Resume interrupted training.")
    parser.add_argument("--cache", default=False, help="Ultralytics cache setting.")
    parser.add_argument("--patience", type=int, default=100, help="Early-stopping patience.")
    parser.add_argument(
        "--single-cls",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Train as one defect class.",
    )
    parser.add_argument("--degrees", type=float, default=5.0, help="Built-in random rotation augmentation.")
    parser.add_argument("--translate", type=float, default=0.05, help="Built-in translation augmentation.")
    parser.add_argument("--scale", type=float, default=0.10, help="Built-in scale augmentation.")
    parser.add_argument("--fliplr", type=float, default=0.5, help="Built-in horizontal flip probability.")
    parser.add_argument(
        "--flipud",
        type=float,
        default=0.0,
        help="Vertical flip probability; enable only after review.",
    )
    parser.add_argument(
        "--mosaic",
        type=float,
        default=0.0,
        help="Mosaic probability; default off for small defects.",
    )
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        help="Extra Ultralytics train override as key=value.",
    )
    return parser


def main() -> None:
    """Run YOLO training."""
    args = build_parser().parse_args()
    overrides = parse_overrides(args.override)
    model = YOLO(args.model, task=args.task)
    model.train(
        data=str(args.data_yaml),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        project=args.project,
        name=args.name,
        exist_ok=args.exist_ok,
        seed=args.seed,
        resume=args.resume,
        cache=args.cache,
        patience=args.patience,
        single_cls=args.single_cls,
        degrees=args.degrees,
        translate=args.translate,
        scale=args.scale,
        fliplr=args.fliplr,
        flipud=args.flipud,
        mosaic=args.mosaic,
        **overrides,
    )


if __name__ == "__main__":
    main()
