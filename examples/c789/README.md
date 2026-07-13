# C789 YOLO Defect Detection

These examples keep C789-specific code outside Ultralytics core trainer code.
Use anomalib to collect images, crop slots, export YOLO labels, and fuse branch
results. Use this fork only to run YOLO training, validation, prediction, and
model export.

## Data Contract

Create the dataset from anomalib:

```bash
cd /home/yunjing/anomalib
.venv/bin/python pipeline/21_prepare_yolo_dataset.py \
  --manifest dataset/c789_100_left_top_parts/part_crop_manifest.csv \
  --annotations dataset/c789_100_left_top_parts/bbox_annotations.csv \
  --output-root dataset/c789_yolo/left_top \
  --positive-val-ratio 0.2 \
  --normal-test-split val \
  --preview-dir results/c789_yolo/left_top_bbox_previews \
  --overwrite
```

Annotate bounding boxes on the slot crop images, not on raw 4024x3036 capture
images. Use crop-level identifiers such as `processed_path`, `image_path`, or
`sample_id`; do not use raw `source_path` as a bbox join key. The first training
schema is single-class:

```yaml
names:
  0: defect
```

Do not convert unannotated defect crops into empty negative labels. The anomalib
exporter fails closed by default when a defect crop has no bbox annotation.

## Train

```bash
cd /home/yunjing/ultralytics-c789
python examples/c789/train.py \
  --data-yaml /home/yunjing/anomalib/dataset/c789_yolo/left_top/data.yaml \
  --model yolo26n.pt \
  --epochs 100 \
  --imgsz 1024 \
  --batch 8 \
  --device 0 \
  --project /home/yunjing/anomalib/results/c789_yolo \
  --name left_top_defect
```

The wrapper uses Ultralytics built-in bbox-aware augmentations. Defaults are
conservative for small metal defects: light rotation, translation, scale, and
horizontal mirror; vertical flip and mosaic are off by default.

## Validate

```bash
python examples/c789/validate.py \
  --model /home/yunjing/anomalib/results/c789_yolo/left_top_defect/weights/best.pt \
  --data-yaml /home/yunjing/anomalib/dataset/c789_yolo/left_top/data.yaml \
  --split val \
  --imgsz 1024 \
  --device 0 \
  --project /home/yunjing/anomalib/results/c789_yolo \
  --name left_top_val
```

Use anomalib `export_manifest.csv` to audit per-slot and per-defect-type
coverage. Validation and test images should be real, unaugmented captures.

## Predict And Export Fusion CSV

```bash
python examples/c789/predict_export.py \
  --model /home/yunjing/anomalib/results/c789_yolo/left_top_defect/weights/best.pt \
  --source /home/yunjing/anomalib/dataset/c789_100_left_top_parts/left/top/defect \
  --manifest /home/yunjing/anomalib/dataset/c789_yolo/left_top/export_manifest.csv \
  --side top \
  --view uniform \
  --conf 0.25 \
  --imgsz 1024 \
  --device 0 \
  --save \
  --skip-export \
  --fusion-csv /home/yunjing/anomalib/results/c789_yolo/yolo_predictions.csv
```

Fuse with anomalib stage 18:

```bash
cd /home/yunjing/anomalib
.venv/bin/python pipeline/18_fuse_inspection_results.py \
  --branch-csv yolo=results/c789_yolo/yolo_predictions.csv \
  --output-dir results/c789_yolo/fused
```

The CSV branch name is `yolo`, and a positive crop maps to `NG_YOLO` in anomalib
fusion.

## ZS32 Controlled Parameter Sweep

The ZS32 runner compares eighteen controlled screening configurations: six
fixed model/resolution/batch bindings times three augmentation/optimizer
profiles. Screening uses only `val` for 50 epochs. It promotes the best
YOLO26n configuration, the best YOLO26m configuration, and the best remaining
configuration, then retrains those three from their original pretrained
weights for 150 epochs with `patience=30`.

Start with an offline-safe dry run. It validates the dataset, manifest, and
weight paths and writes `experiment_plan.json` plus a pristine `state.json`,
but does not import or construct YOLO:

```bash
/home/yunjing/miniconda3/envs/yolo/bin/python examples/c789/sweep_zs32.py \
  --dry-run --experiment-id zs32_full_v1
```

Run screening:

```bash
/home/yunjing/miniconda3/envs/yolo/bin/python examples/c789/sweep_zs32.py \
  --stage screen --experiment-id zs32_full_v1
```

If screening was interrupted or left a failed/stale attempt, authorize a new
scheduler attempt explicitly:

```bash
/home/yunjing/miniconda3/envs/yolo/bin/python examples/c789/sweep_zs32.py \
  --stage screen --experiment-id zs32_full_v1 --resume
```

Run the clean finalist stage after all eighteen screening trials are valid:

```bash
/home/yunjing/miniconda3/envs/yolo/bin/python examples/c789/sweep_zs32.py \
  --stage final --experiment-id zs32_full_v1
```

`--resume` is required only when the requested stage contains stale, failed,
or partial work; a clean screen-to-final stage transition does not require it.
It never resumes a YOLO checkpoint. Instead, it authorizes a distinct scheduler
attempt directory, while valid completed attempts are skipped. The single
half-batch OOM retry allowance is persisted across invocations and is not reset
by `--resume`.

The experiment directory contains `state.json`, `leaderboard.csv`,
`leaderboard.md`, `failed_trials.csv`, and `grouped_metrics.csv`. Reports are
refreshed after each completed or failed attempt. Test data is never used for
screening or finalist selection; only a completed clean finalist runs test
evaluation and grouped test reporting. The expected cost for 18x50 screening
epochs plus 3x150 finalist epochs is approximately 10-15 GPU hours, subject to
PNG decode and validation time.

Standard Ultralytics validation metrics use `conf=0.001` and `iou=0.7`.
Operational `conf=0.25` and `iou=0.5` apply only to prediction-based empty-label
FPR and GT matching. The experiment plan records SHA256 fingerprints for both
`data.yaml` and `split_manifest.csv`. Every invocation, including dry-run, holds
a non-blocking experiment-level file lock while it reads or writes plan/state,
so a dry-run cannot overwrite an active scheduler's state and two schedulers
cannot train the same experiment ID concurrently.
