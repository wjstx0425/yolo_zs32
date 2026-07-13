# ZS32 YOLO Parameter Sweep Design

## Goal

Build a local, resumable experiment runner that compares controlled YOLO26n and YOLO26m training configurations on the existing ZS32 six-view dataset, ranks trials without changing the dataset split, promotes three finalists, and writes a reproducible leaderboard.

## Scope and constraints

- Dataset: `/home/yunjing/anomalib/dataset/zs32_six_view_roi_yolo/data.yaml`.
- Manifest: `/home/yunjing/anomalib/dataset/zs32_six_view_roi_yolo/split_manifest.csv`.
- Weights: `/home/yunjing/ultralytics-c789/yolo26n.pt` and `/home/yunjing/ultralytics-c789/yolo26m.pt`.
- Use the current local Ultralytics checkout through `/home/yunjing/miniconda3/envs/yolo/bin/python`.
- Never modify or re-split train, val, or test.
- Empty label files remain valid negatives.
- Disable horizontal and vertical flips in every profile because explicit mirror data already exists and view orientation is meaningful.
- Do not use the test split for screening or finalist selection.
- Do not reuse or overwrite the currently active `runs/zs32/six_view_yolo26n_1536_v1` run.
- Do not perform GitHub, branch, PR, or upload work.

## Approaches considered

### Controlled experiment matrix — selected

Use an explicit matrix with resource-safe model, resolution, and batch bindings. This is transparent, reproducible, and allows ZS32-specific negative-image evaluation.

### Ultralytics built-in tuner — rejected for this stage

The local tuner is available, but its default continuous search includes aggressive rotation, scale, flip, Mosaic, MixUp, and CutMix ranges that conflict with the six-view industrial constraints. It also does not naturally express the model/resolution/batch bindings or the required empty-label false-positive metric.

### Ray Tune or Optuna — rejected for this stage

These tools would add dependencies that are not installed locally. Their scheduling benefits do not justify that extra setup for eighteen controlled screening trials.

## Experiment matrix

### Model, resolution, and batch bindings

| Combination | Model | imgsz | batch |
|---|---|---:|---:|
| `n640` | `yolo26n.pt` | 640 | 64 |
| `n1280` | `yolo26n.pt` | 1280 | 24 |
| `n1536` | `yolo26n.pt` | 1536 | 16 |
| `m640` | `yolo26m.pt` | 640 | 32 |
| `m1280` | `yolo26m.pt` | 1280 | 8 |
| `m1536` | `yolo26m.pt` | 1536 | 8 |

Batch is a resource binding, not an independent Cartesian-product dimension. If a trial raises CUDA OOM before completing its first epoch, the runner records the failure and may retry exactly once at half batch when `--oom-retry` is enabled. The retry becomes a distinct recorded configuration and never silently replaces the requested batch.

### Training profiles

All profiles use `degrees=0`, `fliplr=0`, `flipud=0`, `mosaic=0`, `mixup=0`, `cutmix=0`, `copy_paste=0`, `shear=0`, and `perspective=0`.

| Profile | Optimizer and augmentation |
|---|---|
| `P0_no_aug` | `optimizer=auto`, `translate=0`, `scale=0`, `hsv_h=0`, `hsv_s=0`, `hsv_v=0` |
| `P1_conservative` | `optimizer=auto`, `translate=0.03`, `scale=0.10`, `hsv_h=0.005`, `hsv_s=0.20`, `hsv_v=0.15` |
| `P2_adamw_cosine` | P1 augmentation plus `optimizer=AdamW`, `lr0=0.001`, `lrf=0.01`, `cos_lr=True`, `weight_decay=0.0005` |

The matrix contains `6 × 3 = 18` screening trials.

## Two-stage workflow

### Screening stage

- Train all eighteen trials from their original pretrained weight for 50 epochs.
- Use `patience=50` so screening trials have the same training budget and are not eliminated by different early-stop points.
- Use one fixed seed for all trials, default `seed=42`.
- Evaluate only on val during screening.
- Record the best validation epoch rather than the final CSV row.

### Finalist selection

Promote exactly three configurations:

1. the highest-ranked YOLO26n trial;
2. the highest-ranked YOLO26m trial;
3. the highest-ranked remaining trial overall.

This preserves a direct n-versus-m comparison while still allowing the globally strongest family to occupy the third slot. Ranking is lexicographic and transparent:

1. higher validation `mAP50-95`;
2. higher validation recall;
3. lower empty-label image false-positive rate at `conf=0.25`;
4. higher validation `mAP50`;
5. lower mean epoch time.

The leaderboard contains every raw metric, so users can re-sort without rerunning training.

### Final stage

- Retrain the three promoted configurations from their original pretrained weights for up to 150 epochs.
- Use `patience=30`.
- Do not resume from the 50-epoch screening weights; finalists are clean, independent full runs.
- Select the best epoch by validation `mAP50-95`.
- Run test evaluation only after each finalist has finished.
- Produce overall val/test metrics plus grouped test recall by `view`, `hand`, and `defect_type`, and false positives for `normal_real`, `normal_mirror`, and all empty-label images.

## Components

### Sweep runner

Create `examples/c789/sweep_zs32.py` with these testable boundaries:

```python
def build_parser() -> argparse.ArgumentParser: ...
def build_experiment_plan(args: argparse.Namespace) -> list[TrialSpec]: ...
def build_train_kwargs(trial: TrialSpec, args: argparse.Namespace) -> dict[str, object]: ...
def select_finalists(rows: Sequence[TrialResult], count: int = 3) -> list[TrialResult]: ...
def run_trial(trial: TrialSpec, yolo_factory=YOLO) -> TrialResult: ...
def main(argv: Sequence[str] | None = None) -> int: ...
```

The CLI supports:

```text
--stage screen|final|all
--dry-run
--resume
--max-trials N
--seed 42
--screen-epochs 50
--final-epochs 150
--finalists 3
--conf 0.25
--iou 0.50
--oom-retry / --no-oom-retry
```

`--dry-run` validates paths, prints every resolved trial, and writes the plan without importing or constructing a YOLO model.

### Metrics and grouped evaluation

Create `examples/c789/zs32_sweep_metrics.py` for pure result parsing, ranking, prediction-to-manifest joining, IoU matching, grouped recall, and empty-label false-positive counting. Keeping metrics separate prevents the training scheduler from becoming a large mixed-responsibility file.

### Output layout

Every invocation receives a stable experiment ID or an explicitly supplied `--experiment-id`:

```text
runs/zs32_sweep/<experiment_id>/
├── experiment_plan.json
├── state.json
├── screen/
│   └── <trial_id>/
├── finalists/
│   └── <trial_id>/
├── leaderboard.csv
├── leaderboard.md
├── grouped_metrics.csv
└── failed_trials.csv
```

Trial IDs encode every controlled dimension, for example `screen_n640_p1_seed42`. Each trial also records the fully resolved arguments, source weight checksum, dataset YAML path, manifest path, start/end timestamps, exit status, peak GPU memory, best epoch, best/last weight paths, and failure text.

## Resume and failure behavior

- `state.json` is written atomically after each state transition.
- A completed trial is skipped only when its state is `complete`, its `results.csv` parses, and its recorded `best.pt` exists.
- A stale or partial trial is reported and requires `--resume`; its existing directory is never silently overwritten.
- A trial failure is recorded in `failed_trials.csv`; later trials continue unless `--fail-fast` is passed.
- OOM retry is limited to one half-batch retry and is recorded as a separate attempt.
- `SIGINT` or `SIGTERM` leaves the current state recoverable and does not mark the trial complete.
- The runner rejects an experiment directory that belongs to a different plan hash.

## Testing strategy

Add `tests/test_c789_examples.py` and use test-first development. Tests must not load weights, access CUDA, or train a real model.

Coverage includes:

- the exact eighteen screening trials and bound batch values;
- the three exact training profiles and protected no-flip/no-Mosaic constraints;
- unique deterministic trial IDs and plan hashing;
- train kwargs and override precedence;
- finalist selection including the best n, best m, and best remaining trial;
- best-epoch extraction from synthetic `results.csv` content;
- grouped view/hand/defect-type recall and negative-image FPR from synthetic manifest/predictions;
- dry-run never constructing `YOLO`;
- completed-trial resume checks and partial-trial rejection;
- failed-trial continuation and one-time OOM retry state recording;
- CLI validation and missing-path errors.

After unit tests, verification includes a dry run against the real ZS32 paths and a `--max-trials 1 --screen-epochs 1` GPU smoke only when explicitly requested. Implementing the runner does not automatically start the 10–15+ hour full sweep.

## Estimated cost

The eighteen 50-epoch screening runs plus three 150-epoch finalists are expected to require roughly 10–15 GPU hours, with PNG decode, validation, and grouped prediction potentially increasing wall time. The runner records actual time so the estimate can be replaced after the first trials.
