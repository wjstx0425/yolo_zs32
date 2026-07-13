# AGENTS Memory

## GitHub publishing boundary (2026-07-13)

- Approved target repository: `git@github.com:wjstx0425/yolo_zs32.git`.
- Keep `origin` attached to upstream Ultralytics and use a separate `yolo-zs32` remote for the project repository.
- GitHub is code-only: commit C789/ZS32 source, tests, docs, and this memory; never commit root `dataset/`, `runs/`, weights, checkpoints, or generated training artifacts.
- Publish the current `c789-defect-yolo` work as remote `main`; transfer the approximately 16 GB dataset to the training server separately with `rsync` or equivalent storage.
- Publishing design: `docs/superpowers/specs/2026-07-13-yolo-zs32-github-publish-design.md`.
- This checkout initially had no Git author identity; reuse the existing workspace repository identity `wjstx0425 <lijunlai@sjtu.edu.cn>` as a repository-local setting, not a global setting.

## ZS32 six-view single-class YOLO training (2026-07-12)

- Dataset source of truth: `/home/yunjing/anomalib/dataset/zs32_six_view_roi_yolo` with grouping metadata in `split_manifest.csv`.
- Preserve empty labels as valid negative samples and never re-split individual images; physical `sample_id` grouping is immutable.
- YOLO class mapping is only `0: defect`; `deform`, `less`, and `others` stay as manifest metadata for grouped evaluation.
- Before training, gate on image/label parity, complete label validation, sample leakage, current manifest statistics, mirror-crop inspection, local weights, and measured GPU VRAM.
- Execution plan: `docs/superpowers/plans/2026-07-12-zs32-six-view-yolo-training.md`.
- Verified data audit: 2010 image/label/manifest rows with split counts `1398/306/306`; 488 valid class-0 boxes; 222 unique `sample_id` values and zero cross-split leakage.
- Current ROI conversion totals: 19 clipped annotations and 0 dropped annotations. Kinds are 654 defect, 678 normal_real, and 678 normal_mirror.
- Three labels have only an approximately `5e-9` full-box boundary excess caused by 8-decimal serialization; their five stored YOLO values still satisfy the required normalized-value checks.
- Mirror rebuild evidence: ROI implementation mtime `23:09:55`; target outputs `23:25:24`/`23:29:12`; manifest `23:31:08`. All 678 mirrors match the source-view horizontal ROI formula with zero mismatches.
- Pixel checks passed for the named front-right and back-right mirror samples; both subjects are visually complete. Audit sheets are under `/tmp/zs32_roi_visual_audit/`.
- Runtime: `/home/yunjing/miniconda3/envs/yolo/bin/python`, local Ultralytics 8.4.89, torch 2.12.1+cu130, RTX 4090 with about 47.37 GiB VRAM.
- First smoke: `yolo26m.pt`, 1 epoch, `imgsz=1280`, `batch=8`, peak GPU memory 18.8 GiB, 1398 train and 306 val images scanned with 0 corrupt, exit code 0. Output: `runs/zs32/six_view_yolo26m_smoke_1280`.
- The 1-epoch metrics are not meaningful (`P=0.000558`, `R=0.243`, `mAP50=0.000168`, `mAP50-95=0.0000417`). Its immature single-class head produced reversed low-confidence prediction boxes, causing non-fatal `plots=True` Pillow thread exceptions; pretrained `yolo26m.pt` did not produce such boxes on the same diagnostic batch.
- Second smoke completed with exit code 0: `yolo26n.pt`, 3 epochs, `imgsz=1536`, `batch=16`, peak GPU memory 19.9 GiB; no plotting exception. Output: `runs/zs32/six_view_yolo26n_smoke_1536`.
- Second-smoke val metrics are intentionally non-final: `P=0.0000218`, `R=0.0270`, `mAP50=0.000000592`, `mAP50-95=0.000000118`. Test: `P=0.0000109`, `R=0.0128`, `mAP50=0.000000299`, `mAP50-95=0.0000000594`.
- At operational `conf=0.25`, the 3-epoch smoke produced zero test detections: box/image recall is zero for every view, hand, and defect type, while false positives are `0/204` normal images and `0/237` all empty-label images. This validates the grouping pipeline only and is not a usable trained detector.
- Recommended first formal baseline uses the exact stable combination `yolo26n.pt`, `imgsz=1536`, `batch=16`, `epochs=150`, `patience=30`, conservative HSV/translation/scale, no rotation, no flips, and no mosaic.
- Final verification: ROI unit test `9 passed`; independent full-dataset assertion passed; both smoke `best.pt` files and the 1536 `results.csv` exist; formal run name `runs/zs32/six_view_yolo26n_1536_v1` was free; the existing training wrapper parsed successfully.

## ZS32 YOLO controlled parameter sweep design (2026-07-13)

- Approved scope is the complete mode with `yolo26n/yolo26m`, `imgsz=640/1280/1536`, and three controlled training profiles: 18 screening trials at 50 epochs, followed by 3 clean finalist runs at 150 epochs.
- Bound batches are `n640=64`, `n1280=24`, `n1536=16`, `m640=32`, `m1280=8`, and `m1536=8`; batch is not an independent grid dimension.
- Finalists must include the best n, best m, and best remaining configuration. Test is never used during screening or selection.
- Full design: `docs/superpowers/specs/2026-07-13-zs32-yolo-parameter-sweep-design.md`.
- The active `runs/zs32/six_view_yolo26n_1536_v1` training must not be read as final, reused, overwritten, or interrupted by the sweep runner.
- Task 3 implementation is `examples/c789/sweep_zs32.py`; pure metrics remain in `examples/c789/zs32_sweep_metrics.py`, with offline FakeYOLO/state-machine coverage in `tests/test_c789_examples.py`.
- The runner supports `--stage screen|final|all`, writes atomic `state.json`, validates `plan_hash`, skips only valid complete attempts, requires `--resume` for stale/partial work, and gives resumed/OOM attempts unique run names.
- A CUDA training OOM before any completed epoch may retry exactly once at half batch as a distinct attempt. Ordinary errors continue unless `--fail-fast`; `KeyboardInterrupt` leaves the attempt in recoverable `running` state.
- Finalists are best n, best m, then best remaining. Each finalist starts from the original `yolo26n.pt` or `yolo26m.pt`, trains for 150 epochs with `patience=30`, and only then evaluates test and writes grouped test metrics.
- Persistent outputs under `runs/zs32_sweep/<experiment_id>/` are `experiment_plan.json`, `state.json`, `leaderboard.csv`, `leaderboard.md`, `failed_trials.csv`, and `grouped_metrics.csv`; grouped aggregate rows include finalist `trial_id` and `split=test`.
- Verified command shapes are `/home/yunjing/miniconda3/envs/yolo/bin/python examples/c789/sweep_zs32.py --dry-run --experiment-id zs32_full_v1`, then `--stage screen`, then clean `--stage final`; add `--resume` only when the selected stage has stale/failed/partial work. Resume creates a new scheduler attempt and never resumes `last.pt`.
- Each attempt resets CUDA peak-memory statistics before YOLO construction and records its own `max_memory_allocated` value afterward; offline tests inject the memory backend and never access CUDA.
- Task 3 focused verification is `61 passed`; direct script `--help`, `py_compile`, real-path dry-run, and `git diff --check` also pass. The dry-run artifact has 18 unique trials, matching plan/state hashes, zero attempts, and no weights directory.
- Standard `.val()` calls use `conf=0.001,iou=0.7`; operational `conf=0.25,iou=0.5` remains limited to `.predict()` empty-label FPR and grouped GT matching.
- Plan/state provenance includes SHA256 fingerprints for `data.yaml`, `split_manifest.csv`, and both model weights. The hash functions read current bytes rather than caching by path.
- The active baseline directory and every descendant are rejected as experiment roots before input or output access. Every invocation, including dry-run, holds a non-blocking `fcntl` lock across plan/state reads/writes; dry-run therefore fails closed instead of racing an active scheduler.
- `main()` builds a read-only plan before creating the experiment lock, so missing data YAML, manifest, weights directory, or model weights leave no experiment output. It rebuilds and rehashes the plan inside the lock to avoid carrying pre-lock fingerprints across a TOCTOU boundary.
- OOM half-batch retry allowance is persisted per source trial across `--resume`. If the primary OOM reached state but the half attempt did not, resume runs that pending half attempt directly exactly once; failed/OOM attempts also persist their own peak GPU metadata.
- Real scheduler smoke on 2026-07-13 used the actual conda `yolo` interpreter and RTX 4090: `--stage screen --experiment-id zs32_script_smoke_20260713 --screen-epochs 1 --max-trials 1` completed `screen_n640_p0_seed42` with exit 0.
- The real smoke scanned 1398 train and 306 val images with 0 corrupt, trained `yolo26n.pt` at `imgsz=640`, `batch=64`, produced `best.pt`/`last.pt`, completed standard val plus operational empty-label prediction, and recorded per-attempt peak GPU memory `8.8756 GiB` (`9.51G` displayed during training).
- Real smoke output: `runs/zs32_sweep/zs32_script_smoke_20260713`; state has exactly one `complete` attempt, leaderboard rank 1, best epoch 1, val recall `0.09459`, mAP50 `0.000010`, mAP50-95 `0.000000`, and empty-label FPR `0.0`. These 1-epoch metrics are pipeline evidence only.
- Re-running the same screen command exited 0 and left `attempts_after_rerun=1`, proving a valid complete trial is skipped rather than retrained.
- Full-sweep launch note on 2026-07-13: `runs/zs32_sweep/zs32_full_v1` was an old pre-final-code dry-run with only `.sweep.lock`, `experiment_plan.json`, and `state.json`; it had `attempts=0` and no weights, but its old plan hash intentionally fails against the finalized runner. Do not use `--resume` to bypass this mismatch.
- A fresh finalized dry-run was verified at `runs/zs32_sweep/zs32_full_v2`: 18 unique trials, matching plan/state hash `e7ae9e9e0dd6029c073c472d6751f2f3fbdf11906d6f121505707aad0bd5c19a`, zero attempts, and zero weights. Start the full experiment with `--stage all --experiment-id zs32_full_v2`.

## First formal ZS32 baseline diagnosis (2026-07-13)

- `runs/zs32/six_view_yolo26n_1536_v1` used `yolo26n.pt`, `imgsz=1536`, actual `batch=32`, `epochs=150`, and `patience=30`; it early-stopped after epoch 138 and `best.pt` is epoch 108.
- Best validation metrics are `P=0.55172`, `R=0.37838`, `mAP50=0.31797`, and `mAP50-95=0.11805`. Use `weights/best.pt`, not `last.pt`.
- The result is operationally weak mainly because recall is only 37.8% and localization falls sharply from mAP50 to mAP50-95, but it is a valid converged baseline rather than a failed training run.
- There is a clear late generalization gap: at epoch 108 train/val box loss is `0.87499/2.77645`; by epoch 138 train box improves to `0.72934` while val box worsens to `2.88658`, and mAP50-95 drops to `0.10373`. Increasing epochs or patience is not the priority.
- Validation contains only 74 boxes, so per-epoch P/R/mAP is noisy. Test contains 306 images, 78 boxes, 69 positive images, and 237 empty-label images; grouped test metrics have not yet been generated for this baseline.
- At 1536 letterbox scale, all 488 boxes have median size about `41.9 x 48.4 px`; only 9 have a minimum side below 16 px. `less` is smaller (median about `33.5 x 35.3 px`) than `deform` (about `51.6 x 61.9 px`), so 1536 is reasonable and raw resolution alone is not the dominant remaining bottleneck.
- The active `zs32_full_v2` controlled sweep should be allowed to finish before launching extra GPU evaluation. Its n/m, 640/1280/1536, augmentation, optimizer, and schedule comparisons are the proper next evidence; do not rerun the same 1536 baseline with more epochs.
