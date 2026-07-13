# ZS32 YOLO Parameter Sweep Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a local, resumable 18-trial YOLO26n/YOLO26m experiment runner that promotes three finalists and produces overall and grouped comparison reports without modifying the ZS32 split.

**Architecture:** Put pure CSV parsing, ranking, IoU matching, grouped metrics, and report formatting in `zs32_sweep_metrics.py`. Put immutable experiment definitions, CLI validation, state transitions, YOLO training/prediction orchestration, resume/OOM behavior, and atomic artifact writes in `sweep_zs32.py`; all side effects sit behind injectable factories so unit tests never load CUDA or weights.

**Tech Stack:** Python 3.11, argparse, dataclasses, csv/json/pathlib, Ultralytics 8.4.89, pytest.

## Global Constraints

- Dataset is `/home/yunjing/anomalib/dataset/zs32_six_view_roi_yolo/data.yaml`; manifest is `/home/yunjing/anomalib/dataset/zs32_six_view_roi_yolo/split_manifest.csv`.
- Models are `/home/yunjing/ultralytics-c789/yolo26n.pt` and `/home/yunjing/ultralytics-c789/yolo26m.pt`.
- Never modify or re-split train, val, or test; preserve empty labels as valid negatives.
- Every profile fixes `degrees=0`, `fliplr=0`, `flipud=0`, `mosaic=0`, `mixup=0`, `cutmix=0`, `copy_paste=0`, `shear=0`, and `perspective=0`.
- Screening uses val only; test is evaluated only for completed finalists.
- Do not reuse, overwrite, resume, or inspect an active `runs/zs32/six_view_yolo26n_1536_v1` as a sweep trial.
- Do not start the 18-trial sweep or any GPU training during implementation verification.
- Do not perform GitHub, branch, PR, commit, or upload work.

---

### Task 1: Pure metrics and ranking module

**Files:**

- Create: `examples/c789/zs32_sweep_metrics.py`
- Create: `tests/test_c789_examples.py`

**Interfaces:**

- Produces `EpochMetrics`, `TrialMetrics`, `read_best_epoch(path)`, `rank_trials(rows)`, `select_finalists(rows, count=3)`, `box_iou()`, `match_boxes()`, `summarize_grouped_predictions()`, `write_leaderboard_csv()`, `write_leaderboard_markdown()`, and `write_grouped_metrics_csv()`.
- Ranking key is descending mAP50-95, descending recall, ascending empty-label FPR, descending mAP50, ascending mean epoch seconds.
- Finalists are best n, best m, then best remaining, without duplicates.

- [x] **Step 1: Write failing best-epoch and ranking tests**

Add synthetic `results.csv` rows where the last epoch is not best, then assert:

```python
best = metrics.read_best_epoch(results_csv)
assert best.epoch == 2
assert best.map50_95 == pytest.approx(0.31)
assert [row.trial_id for row in metrics.rank_trials(rows)] == ["n_best", "m_best", "n_other"]
assert [row.trial_id for row in metrics.select_finalists(rows)] == ["n_best", "m_best", "n_other"]
```

- [x] **Step 2: Run the focused test and verify RED**

Run:

```bash
/home/yunjing/miniconda3/envs/yolo/bin/python -m pytest tests/test_c789_examples.py -q
```

Expected: import or missing-symbol failure for `zs32_sweep_metrics.py`.

- [x] **Step 3: Implement immutable metric records and CSV parsing**

Use frozen dataclasses with explicit fields:

```python
@dataclass(frozen=True)
class EpochMetrics:
    epoch: int
    epoch_seconds: float
    precision: float
    recall: float
    map50: float
    map50_95: float


@dataclass(frozen=True)
class TrialMetrics:
    trial_id: str
    model_family: str
    profile: str
    imgsz: int
    batch: int
    best: EpochMetrics
    empty_label_fpr: float
    run_dir: str
    best_pt: str
    status: str = "complete"
```

Parse Ultralytics headers by exact names, derive per-epoch duration from cumulative `time`, reject missing/empty/non-finite CSV data, and select maximum `metrics/mAP50-95(B)` with the documented tie-breakers.

- [x] **Step 4: Verify ranking tests GREEN**

Run the focused pytest command and require all current tests to pass.

- [x] **Step 5: Write failing grouped-evaluation tests**

Create synthetic manifest rows for multiple views/hands/defect types, normal real/mirror rows, empty defect views, GT boxes, and predictions. Assert IoU ≥ 0.50 one-to-one matching, per-group box/image recall, and separate normal/empty-image false-positive counters.

- [x] **Step 6: Verify grouped tests RED**

Expected: missing grouped functions or mismatched expected aggregates.

- [x] **Step 7: Implement grouped metrics and report writers**

Implement deterministic CSV/Markdown column order. Match each prediction to at most one GT and each GT to at most one prediction. Report group dimensions `view`, `hand`, `defect_type`, and `kind`; report `normal_real`, `normal_mirror`, and all empty-label FPR separately.

- [x] **Step 8: Verify Task 1 GREEN**

Run the focused pytest command; require no test failures.

### Task 2: Experiment plan, CLI, and dry-run orchestration

**Files:**

- Create: `examples/c789/sweep_zs32.py`
- Modify: `tests/test_c789_examples.py`

**Interfaces:**

- Consumes all Task 1 metric types/functions.
- Produces frozen `TrialSpec`, `MODEL_RESOLUTION_BATCH`, `TRAINING_PROFILES`, `build_parser()`, `build_experiment_plan(args)`, `build_train_kwargs(trial,args)`, `plan_hash()`, `atomic_write_json()`, `run_trial()`, and `main(argv=None,yolo_factory=None)`.
- Exact resource bindings: `n640=64`, `n1280=24`, `n1536=16`, `m640=32`, `m1280=8`, `m1536=8`.

- [x] **Step 1: Write failing 18-trial plan tests**

Assert exact trial count, IDs, model paths, sizes, batches, profiles, seed, protected augmentation values, screening epoch/patience values, unique IDs, and deterministic plan hash.

- [x] **Step 2: Verify plan tests RED**

Expected: missing `sweep_zs32.py` or missing symbols.

- [x] **Step 3: Implement CLI and plan construction**

CLI defaults:

```text
--stage all
--seed 42
--screen-epochs 50
--final-epochs 150
--finalists 3
--conf 0.25
--iou 0.50
--project /home/yunjing/ultralytics-c789/runs/zs32_sweep
--data-yaml /home/yunjing/anomalib/dataset/zs32_six_view_roi_yolo/data.yaml
--manifest /home/yunjing/anomalib/dataset/zs32_six_view_roi_yolo/split_manifest.csv
--weights-dir /home/yunjing/ultralytics-c789
--oom-retry true
```

Support `--dry-run`, `--resume`, `--max-trials`, `--experiment-id`, `--fail-fast`, and BooleanOptionalAction for OOM retry. Validate positive epochs/finalists, confidence/IoU in `[0,1]`, and required paths before creating output.

- [x] **Step 4: Verify 18-trial tests GREEN**

Run focused pytest; require all tests to pass.

- [x] **Step 5: Write failing train-kwargs and injected-run tests**

Use `FakeYOLO` to capture constructor, `.train()`, `.val()`, and `.predict()` calls. Assert original weight initialization, unique absolute project/name, screen vs final epochs/patience, fixed augmentation safety fields, P2 optimizer fields, and no model construction during dry-run.

- [x] **Step 6: Verify orchestration tests RED**

Expected: missing orchestration behavior, not CUDA/model-loading errors.

- [x] **Step 7: Implement trial execution and artifact collection**

Construct `YOLO(str(trial.model_path), task="detect")`, call `.train(**kwargs)`, parse the run `results.csv`, verify `best.pt`, evaluate val empty-label FPR for screen trials, and evaluate val/test plus grouped manifest metrics for finalists. Do not treat a zero metric as missing. Store every resolved argument and artifact path.

- [x] **Step 8: Verify injected runner GREEN**

Run focused pytest; require `FakeYOLO` tests to pass without torch/CUDA access.

### Task 3: Resume, OOM retry, reports, docs, and real dry-run

**Files:**

- Modify: `examples/c789/sweep_zs32.py`
- Modify: `examples/c789/README.md`
- Modify: `tests/test_c789_examples.py`
- Modify: `AGENTS_MEMORY.md`

**Interfaces:**

- State schema includes `plan_hash`, `experiment_id`, `stage`, and per-attempt `trial_id`, `status`, timestamps, requested/effective batch, run directory, error, metrics, and artifacts.
- Completed-trial validity requires status complete, parseable `results.csv`, and existing `best.pt`.

- [x] **Step 1: Write failing state/resume/OOM tests**

Assert atomic state writes, plan-hash mismatch rejection, complete-trial skipping, partial-trial rejection without `--resume`, failed-trial continuation, `--fail-fast`, and exactly one half-batch OOM retry recorded as a distinct attempt.

- [x] **Step 2: Verify state tests RED**

Expected: missing state and retry behavior.

- [x] **Step 3: Implement state machine and reporting**

Write JSON via sibling temporary file plus `Path.replace()`. Catch CUDA OOM by exception type/message without swallowing unrelated errors. Write deterministic `leaderboard.csv`, `leaderboard.md`, `grouped_metrics.csv`, and `failed_trials.csv` after each completed/failed trial so interruption leaves useful output.

- [x] **Step 4: Verify state tests GREEN**

Run focused pytest; require all tests to pass.

- [x] **Step 5: Document exact commands**

Add README commands using the verified interpreter:

```bash
/home/yunjing/miniconda3/envs/yolo/bin/python examples/c789/sweep_zs32.py --dry-run --experiment-id zs32_full_v1
/home/yunjing/miniconda3/envs/yolo/bin/python examples/c789/sweep_zs32.py --stage screen --experiment-id zs32_full_v1
/home/yunjing/miniconda3/envs/yolo/bin/python examples/c789/sweep_zs32.py --stage final --experiment-id zs32_full_v1 --resume
```

Explain 18×50 screening, finalist policy, 3×150 final runs, output files, no-test-selection rule, resume semantics, and estimated cost. Update `AGENTS_MEMORY.md` with implemented file paths and verified commands.

- [x] **Step 6: Run full offline verification**

Run:

```bash
/home/yunjing/miniconda3/envs/yolo/bin/python -m pytest tests/test_c789_examples.py -q
/home/yunjing/miniconda3/envs/yolo/bin/python -m py_compile examples/c789/sweep_zs32.py examples/c789/zs32_sweep_metrics.py
/home/yunjing/miniconda3/envs/yolo/bin/python examples/c789/sweep_zs32.py --help
/home/yunjing/miniconda3/envs/yolo/bin/python examples/c789/sweep_zs32.py --dry-run --experiment-id zs32_full_v1
git diff --check -- examples/c789 tests/test_c789_examples.py docs/superpowers AGENTS_MEMORY.md
```

Expected: tests pass, compile exits zero, help exits zero, dry-run writes an 18-trial plan without constructing YOLO or starting GPU training, and diff check is clean.

- [x] **Step 7: Inspect generated dry-run artifacts**

Confirm `experiment_plan.json` has 18 unique trials, the plan hash matches `state.json`, no `screen/*/weights` exists, and the active `runs/zs32/six_view_yolo26n_1536_v1` timestamps were not changed by verification.
