# YOLO ZS32 GitHub Publish Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Publish the current ZS32 training code to `wjstx0425/yolo_zs32` without uploading datasets, run artifacts, or model weights.

**Architecture:** Keep upstream Ultralytics as `origin` and add the project repository as `yolo-zs32`. Enforce a code-only boundary through the root `.gitignore`, explicitly stage reviewed paths, verify tests and staged content, then push the local commit graph as remote `main`.

**Tech Stack:** Git, GitHub SSH, Python, pytest, Ultralytics

## Global Constraints

- Do not delete or modify local datasets, training outputs, or weights.
- Do not stage root `dataset/`, `runs/`, model weights, checkpoints, Python bytecode, or caches.
- Keep `origin` pointing to `https://github.com/ultralytics/ultralytics.git`.
- Publish to `git@github.com:wjstx0425/yolo_zs32.git` as remote branch `main`.

---

### Task 1: Enforce and verify the code-only boundary

**Files:**

- Modify: `.gitignore`
- Modify: `AGENTS_MEMORY.md`

**Interfaces:**

- Consumes: the existing root-level ignore rules for generated artifacts.
- Produces: a root `/dataset/` ignore rule used by Git status and staging.

- [x] **Step 1: Verify the missing ignore behavior**

Run: `git check-ignore -q dataset/.publish-probe`

Expected: exit code `1`, proving root `dataset/` is not currently ignored.

- [x] **Step 2: Add the minimal ignore rule**

Add `/dataset/` beside the existing `/datasets` rule in `.gitignore`, and record the completed publishing preparation in `AGENTS_MEMORY.md`.

- [x] **Step 3: Verify the ignore behavior**

Run: `git check-ignore -v dataset/.publish-probe`

Expected: `.gitignore` reports the `/dataset/` rule.

- [x] **Step 4: Verify no generated Python files are candidates**

Run: `git status --short --untracked-files=all | rg '(__pycache__|\\.pyc$|^\\?\\? (dataset|runs)/)'`

Expected: no output.

### Task 2: Validate and commit the reviewed source scope

**Files:**

- Add: `docs/superpowers/plans/2026-07-12-zs32-six-view-yolo-training.md`
- Add: `docs/superpowers/plans/2026-07-13-zs32-yolo-parameter-sweep.md`
- Add: `docs/superpowers/plans/2026-07-13-yolo-zs32-github-publish.md`
- Add: `docs/superpowers/specs/2026-07-13-zs32-yolo-parameter-sweep-design.md`
- Add: `examples/c789/`
- Add: `tests/test_c789_examples.py`
- Modify: `.gitignore`
- Modify: `AGENTS_MEMORY.md`

**Interfaces:**

- Consumes: the approved untracked C789/ZS32 source, tests, and documents.
- Produces: one reviewed code commit ready for remote publication.

- [x] **Step 1: Run focused verification**

Run: `/home/yunjing/miniconda3/envs/yolo/bin/python -m pytest tests/test_c789_examples.py -q`

Expected: all focused C789 tests pass.

- [x] **Step 2: Run syntax and whitespace checks**

Run: `/home/yunjing/miniconda3/envs/yolo/bin/python -m py_compile examples/c789/*.py tests/test_c789_examples.py`

Expected: exit code `0`.

Run: `git diff --check`

Expected: exit code `0`.

- [x] **Step 3: Stage only reviewed paths**

Run: `git add .gitignore AGENTS_MEMORY.md docs/superpowers examples/c789 tests/test_c789_examples.py`

Expected: only these paths enter the index.

- [x] **Step 4: Audit the index**

Run: `git diff --cached --name-only`

Expected: no path under `dataset/`, `runs/`, or `weights/`, and no model artifact or Python cache.

Run: `git diff --cached --check`

Expected: exit code `0`.

- [x] **Step 5: Commit the source scope**

Run: `git commit -m "Add ZS32 YOLO training workflow"`

Expected: one new commit containing the reviewed code-only scope.

### Task 3: Configure and verify GitHub publication

**Files:**

- Modify: `.git/config` through `git remote add` only.

**Interfaces:**

- Consumes: the clean reviewed local commit and accessible GitHub SSH repository.
- Produces: `yolo-zs32/main` pointing at the local `c789-defect-yolo` commit.

- [x] **Step 1: Add the project remote without changing upstream**

Run: `git remote add yolo-zs32 git@github.com:wjstx0425/yolo_zs32.git`

Expected: `origin` remains upstream and `yolo-zs32` points to the project repository.

- [x] **Step 2: Push the current commit as project main**

Run: `git push -u yolo-zs32 HEAD:main`

Expected: GitHub creates or updates `main`, and the local branch tracks `yolo-zs32/main`.

- [x] **Step 3: Verify remote commit equality**

Run: `test "$(git rev-parse HEAD)" = "$(git ls-remote yolo-zs32 refs/heads/main | cut -f1)"`

Expected: exit code `0`.

- [x] **Step 4: Verify final repository state**

Run: `git status -sb && git remote -v`

Expected: no unintended untracked dataset/run content; upstream and project remotes remain distinct.
