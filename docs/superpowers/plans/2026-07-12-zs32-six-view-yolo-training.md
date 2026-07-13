# ZS32 Six-View YOLO Training Execution Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Validate the existing ROI dataset without changing its split, run a short local YOLO smoke test, and provide an evidence-based full-training command.

**Architecture:** Treat `split_manifest.csv` as the grouping and evaluation source of truth, while validating the YOLO directory tree independently. Gate training on label, split-leakage, mirror-crop, environment, and GPU checks; keep all training outputs under the local Ultralytics checkout.

**Tech Stack:** Python, Ultralytics YOLO, PyTorch/CUDA, OpenCV/Pillow, CSV/YAML.

## Global Constraints

- Do not modify or recreate the existing train/val/test split.
- Empty label files are valid negative samples and must remain in the dataset.
- The only YOLO class is `0: defect`; defect subtypes remain manifest metadata.
- Use conservative augmentation and disable uncontrolled horizontal flips.
- Do not perform GitHub, branch, PR, or upload work.

---

### Task 1: Dataset integrity gate

**Files:**

- Read: `/home/yunjing/anomalib/dataset/zs32_six_view_roi_yolo/data.yaml`
- Read: `/home/yunjing/anomalib/dataset/zs32_six_view_roi_yolo/split_manifest.csv`
- Read: `/home/yunjing/anomalib/dataset/zs32_six_view_roi_yolo/{train,val,test}/{images,labels}`

- [x] Confirm image/label stem parity in every split.
- [x] Validate every non-empty label row has five columns, class 0, normalized coordinates, and positive width/height.
- [x] Confirm each `sample_id` occurs in exactly one split.
- [x] Recompute split/view/hand/kind, box, clipped, and dropped totals from the current manifest and labels.

### Task 2: ROI mirror and visual gate

**Files:**

- Read: `/home/yunjing/anomalib/capture_data/zs32_view_roi_dataset.py`
- Read: `/home/yunjing/anomalib/pipeline/29_zs32_fixed_roi.py`
- Read: `/home/yunjing/anomalib/dataset/zs32_six_view_roi_config.json`
- Write: `/tmp/zs32_roi_visual_audit/*`

- [x] Verify the `normal_mirror` source-view ROI transformation in source code and generated manifest evidence.
- [x] Inspect the two named mirror images for truncation.
- [x] Build a contact sheet covering every view and real, mirror, boxed-defect, and empty-label cases.

### Task 3: Runtime and smoke test

**Files:**

- Read: `/home/yunjing/ultralytics-c789/pyproject.toml`
- Read: locally available `*.pt` weights.
- Write: `/home/yunjing/ultralytics-c789/runs/zs32/*`

- [x] Confirm the import/CLI entrypoint, local package version, CUDA device, free VRAM, and candidate pretrained weights.
- [x] Select `imgsz` and `batch` from measured VRAM, preferring 1280 and using conservative augmentation.
- [x] Run 1-3 epochs and require successful train plus validation completion.
- [x] Record the exact command, output directory, metrics, and weight paths.

### Task 4: Full-training handoff

- [x] Provide one copy-paste full-training command with model, image size, batch, epochs, patience, and augmentation values fixed from smoke evidence.
- [x] Explain that final best.pt, val/test metrics, grouped recall, normal false positives, and overfitting conclusions only exist after the full run.
- [x] Provide a concrete post-training evaluation route joined against the unchanged manifest.
