"""Focused offline tests for the C789 example utilities."""

from __future__ import annotations

import csv
import json
import subprocess
import sys
from dataclasses import FrozenInstanceError, replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from examples.c789 import sweep_zs32 as sweep
from examples.c789 import zs32_sweep_metrics as metrics

RESULT_FIELDS = [
    "epoch",
    "time",
    "metrics/precision(B)",
    "metrics/recall(B)",
    "metrics/mAP50(B)",
    "metrics/mAP50-95(B)",
]


def _write_results_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=RESULT_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def _trial(
    trial_id: str,
    model_family: str,
    *,
    map50_95: float,
    recall: float,
    empty_label_fpr: float,
    map50: float,
    epoch_seconds: float,
    mean_epoch_seconds: float | None = None,
) -> metrics.TrialMetrics:
    return metrics.TrialMetrics(
        trial_id=trial_id,
        model_family=model_family,
        profile="safe",
        imgsz=1280,
        batch=8,
        best=metrics.EpochMetrics(
            epoch=2,
            epoch_seconds=epoch_seconds,
            precision=0.7,
            recall=recall,
            map50=map50,
            map50_95=map50_95,
        ),
        mean_epoch_seconds=epoch_seconds if mean_epoch_seconds is None else mean_epoch_seconds,
        empty_label_fpr=empty_label_fpr,
        run_dir=f"/runs/{trial_id}",
        best_pt=f"/runs/{trial_id}/weights/best.pt",
    )


def test_read_best_epoch_uses_best_row_and_mean_epoch_duration(tmp_path):
    results_csv = tmp_path / "results.csv"
    _write_results_csv(
        results_csv,
        [
            {
                "epoch": 1,
                "time": 5.0,
                "metrics/precision(B)": 0.40,
                "metrics/recall(B)": 0.50,
                "metrics/mAP50(B)": 0.30,
                "metrics/mAP50-95(B)": 0.20,
            },
            {
                "epoch": 2,
                "time": 13.0,
                "metrics/precision(B)": 0.65,
                "metrics/recall(B)": 0.70,
                "metrics/mAP50(B)": 0.45,
                "metrics/mAP50-95(B)": 0.31,
            },
            {
                "epoch": 3,
                "time": 24.0,
                "metrics/precision(B)": 0.70,
                "metrics/recall(B)": 0.60,
                "metrics/mAP50(B)": 0.42,
                "metrics/mAP50-95(B)": 0.29,
            },
        ],
    )

    best = metrics.read_best_epoch(results_csv)

    assert best.epoch == 2
    assert best.epoch_seconds == pytest.approx(8.0)
    assert best.map50_95 == pytest.approx(0.31)


def test_read_best_epoch_reports_selected_epoch_duration(tmp_path):
    results_csv = tmp_path / "results.csv"
    _write_results_csv(
        results_csv,
        [
            {
                "epoch": 1,
                "time": 4.0,
                "metrics/precision(B)": 0.4,
                "metrics/recall(B)": 0.5,
                "metrics/mAP50(B)": 0.3,
                "metrics/mAP50-95(B)": 0.2,
            },
            {
                "epoch": 2,
                "time": 14.0,
                "metrics/precision(B)": 0.6,
                "metrics/recall(B)": 0.7,
                "metrics/mAP50(B)": 0.5,
                "metrics/mAP50-95(B)": 0.4,
            },
        ],
    )

    best = metrics.read_best_epoch(results_csv)

    assert best.epoch == 2
    assert best.epoch_seconds == pytest.approx(10.0)


def test_read_training_summary_returns_best_and_full_run_mean_seconds(tmp_path):
    results_csv = tmp_path / "results.csv"
    _write_results_csv(
        results_csv,
        [
            {
                "epoch": 1,
                "time": 4.0,
                "metrics/precision(B)": 0.4,
                "metrics/recall(B)": 0.5,
                "metrics/mAP50(B)": 0.3,
                "metrics/mAP50-95(B)": 0.2,
            },
            {
                "epoch": 2,
                "time": 14.0,
                "metrics/precision(B)": 0.6,
                "metrics/recall(B)": 0.7,
                "metrics/mAP50(B)": 0.5,
                "metrics/mAP50-95(B)": 0.4,
            },
            {
                "epoch": 3,
                "time": 27.0,
                "metrics/precision(B)": 0.5,
                "metrics/recall(B)": 0.6,
                "metrics/mAP50(B)": 0.4,
                "metrics/mAP50-95(B)": 0.3,
            },
        ],
    )

    summary = metrics.read_training_summary(results_csv)

    assert summary.best.epoch == 2
    assert summary.best.epoch_seconds == pytest.approx(10.0)
    assert summary.mean_epoch_seconds == pytest.approx(9.0)


@pytest.mark.parametrize("invalid", ["", "nan", "inf", "not-a-number"])
def test_read_best_epoch_rejects_empty_or_nonfinite_metrics(tmp_path, invalid):
    results_csv = tmp_path / "results.csv"
    _write_results_csv(
        results_csv,
        [
            {
                "epoch": 1,
                "time": 5.0,
                "metrics/precision(B)": 0.4,
                "metrics/recall(B)": 0.5,
                "metrics/mAP50(B)": 0.3,
                "metrics/mAP50-95(B)": invalid,
            }
        ],
    )

    with pytest.raises(ValueError, match=r"metrics/mAP50-95\(B\)"):
        metrics.read_best_epoch(results_csv)


def test_read_best_epoch_rejects_missing_columns(tmp_path):
    results_csv = tmp_path / "results.csv"
    results_csv.write_text("epoch,time\n1,5.0\n", encoding="utf-8")

    with pytest.raises(ValueError, match="missing required columns"):
        metrics.read_best_epoch(results_csv)


def test_metric_records_are_immutable():
    row = _trial(
        "n_best",
        "n",
        map50_95=0.31,
        recall=0.7,
        empty_label_fpr=0.01,
        map50=0.45,
        epoch_seconds=8.0,
    )

    with pytest.raises(FrozenInstanceError):
        row.status = "failed"


def test_rank_trials_uses_documented_metric_order():
    rows = [
        _trial(
            "n_other",
            "n",
            map50_95=0.29,
            recall=0.9,
            empty_label_fpr=0.0,
            map50=0.7,
            epoch_seconds=3.0,
        ),
        _trial(
            "m_best",
            "m",
            map50_95=0.31,
            recall=0.65,
            empty_label_fpr=0.01,
            map50=0.46,
            epoch_seconds=7.0,
        ),
        _trial(
            "n_best",
            "n",
            map50_95=0.31,
            recall=0.70,
            empty_label_fpr=0.02,
            map50=0.45,
            epoch_seconds=8.0,
        ),
    ]

    assert [row.trial_id for row in metrics.rank_trials(rows)] == ["n_best", "m_best", "n_other"]


def test_rank_trials_uses_run_mean_seconds_not_best_epoch_seconds():
    slow_mean = _trial(
        "slow_mean",
        "n",
        map50_95=0.3,
        recall=0.6,
        empty_label_fpr=0.1,
        map50=0.4,
        epoch_seconds=1.0,
        mean_epoch_seconds=10.0,
    )
    fast_mean = _trial(
        "fast_mean",
        "m",
        map50_95=0.3,
        recall=0.6,
        empty_label_fpr=0.1,
        map50=0.4,
        epoch_seconds=9.0,
        mean_epoch_seconds=5.0,
    )

    assert [row.trial_id for row in metrics.rank_trials([slow_mean, fast_mean])] == ["fast_mean", "slow_mean"]


def test_select_finalists_picks_best_n_best_m_then_best_remaining():
    rows = [
        _trial(
            "n_best",
            "n",
            map50_95=0.40,
            recall=0.8,
            empty_label_fpr=0.01,
            map50=0.5,
            epoch_seconds=8.0,
        ),
        _trial(
            "n_other",
            "n",
            map50_95=0.39,
            recall=0.8,
            empty_label_fpr=0.01,
            map50=0.5,
            epoch_seconds=8.0,
        ),
        _trial(
            "m_best",
            "m",
            map50_95=0.31,
            recall=0.7,
            empty_label_fpr=0.01,
            map50=0.45,
            epoch_seconds=9.0,
        ),
    ]

    assert [row.trial_id for row in metrics.select_finalists(rows)] == ["n_best", "m_best", "n_other"]


def test_box_iou_and_matching_are_threshold_inclusive_and_one_to_one():
    ground_truth = [(0.0, 0.0, 2.0, 1.0), (3.0, 3.0, 4.0, 4.0)]
    predictions = [
        (0.0, 0.0, 1.0, 1.0),  # IoU is exactly 0.50 with GT 0.
        (0.0, 0.0, 2.0, 1.0),  # Better duplicate prediction for GT 0.
        (3.0, 3.0, 4.0, 4.0),
    ]

    assert metrics.box_iou(ground_truth[0], predictions[0]) == pytest.approx(0.5)
    matches = metrics.match_boxes(ground_truth, predictions, iou_threshold=0.5)

    assert matches == [(0, 1, pytest.approx(1.0)), (1, 2, pytest.approx(1.0))]
    assert len({match[0] for match in matches}) == len(matches)
    assert len({match[1] for match in matches}) == len(matches)


def test_match_boxes_validates_boxes_even_when_other_side_is_empty():
    with pytest.raises(ValueError, match="positive xyxy area"):
        metrics.match_boxes([(0.0, 0.0, 0.0, 1.0)], [])


def test_match_boxes_maximizes_match_count_before_iou():
    ground_truth = [(0.0, 0.0, 2.0, 1.0), (0.5, 0.0, 2.5, 1.0)]
    predictions = [
        (0.0, 0.0, 2.0, 1.0),  # GT 0 IoU 1.0, but the only match for GT 1.
        (0.0, 0.0, 1.2, 1.0),  # GT 0 IoU 0.6 and no match for GT 1.
    ]

    matches = metrics.match_boxes(ground_truth, predictions, iou_threshold=0.5)

    assert [(gt_index, pred_index) for gt_index, pred_index, _ in matches] == [(0, 1), (1, 0)]


def _grouped_fixture():
    manifest_rows = [
        {
            "output_image": "positive_less.png",
            "view": "front",
            "hand": "left",
            "defect_type": "less",
            "kind": "defect",
        },
        {
            "output_image": "positive_deform.png",
            "view": "back",
            "hand": "left",
            "defect_type": "deform",
            "kind": "defect",
        },
        {
            "output_image": "empty_defect.png",
            "view": "front",
            "hand": "right",
            "defect_type": "less",
            "kind": "defect",
        },
        {
            "output_image": "normal_real_fp.png",
            "view": "front",
            "hand": "right",
            "defect_type": "",
            "kind": "normal_real",
        },
        {
            "output_image": "normal_mirror_tn.png",
            "view": "back",
            "hand": "left",
            "defect_type": "",
            "kind": "normal_mirror",
        },
        {
            "output_image": "normal_real_tn.png",
            "view": "back",
            "hand": "right",
            "defect_type": "",
            "kind": "normal_real",
        },
    ]
    ground_truth = {
        "positive_less.png": [(0.0, 0.0, 1.0, 1.0)],
        "positive_deform.png": [(0.0, 0.0, 2.0, 1.0), (3.0, 3.0, 4.0, 4.0)],
        "empty_defect.png": [],
        "normal_real_fp.png": [],
        "normal_mirror_tn.png": [],
        "normal_real_tn.png": [],
    }
    predictions = {
        "positive_less.png": [(0.0, 0.0, 1.0, 1.0)],
        # Exactly-IoU-0.50 prediction verifies the inclusive matching threshold.
        "positive_deform.png": [(0.0, 0.0, 1.0, 1.0)],
        "empty_defect.png": [(0.0, 0.0, 0.5, 0.5)],
        "normal_real_fp.png": [(0.0, 0.0, 0.5, 0.5)],
        "normal_mirror_tn.png": [],
        "normal_real_tn.png": [],
    }
    return manifest_rows, ground_truth, predictions


def _group(rows, dimension, value):
    return next(row for row in rows if row.group_dimension == dimension and row.group_value == value)


def test_grouped_predictions_report_box_image_and_separate_empty_fprs():
    manifest_rows, ground_truth, predictions = _grouped_fixture()

    rows = metrics.summarize_grouped_predictions(
        manifest_rows,
        ground_truth,
        predictions,
        iou_threshold=0.5,
    )

    overall = _group(rows, "all", "all")
    assert overall.gt_box_count == 3
    assert overall.matched_box_count == 2
    assert overall.box_recall == pytest.approx(2 / 3)
    assert overall.positive_image_count == 2
    assert overall.recalled_image_count == 2
    assert overall.image_recall == pytest.approx(1.0)
    assert overall.empty_image_count == 4
    assert overall.false_positive_image_count == 2
    assert overall.empty_label_fpr == pytest.approx(0.5)

    normal_real = _group(rows, "kind", "normal_real")
    assert normal_real.empty_image_count == 2
    assert normal_real.false_positive_image_count == 1
    assert normal_real.empty_label_fpr == pytest.approx(0.5)

    normal_mirror = _group(rows, "kind", "normal_mirror")
    assert normal_mirror.empty_image_count == 1
    assert normal_mirror.false_positive_image_count == 0
    assert normal_mirror.empty_label_fpr == pytest.approx(0.0)

    front = _group(rows, "view", "front")
    assert front.box_recall == pytest.approx(1.0)
    assert front.empty_label_fpr == pytest.approx(1.0)

    left = _group(rows, "hand", "left")
    assert left.box_recall == pytest.approx(2 / 3)
    assert left.image_recall == pytest.approx(1.0)

    less = _group(rows, "defect_type", "less")
    assert less.image_count == 2
    assert less.empty_image_count == 1
    assert less.empty_label_fpr == pytest.approx(1.0)


def test_grouped_predictions_align_repo_relative_and_absolute_paths_without_basename_aliasing(tmp_path):
    path_root = tmp_path / "anomalib"
    relative_positive = "dataset/positive/shared.png"
    relative_empty = "dataset/empty/shared.png"
    manifest_rows = [
        {
            "output_image": relative_positive,
            "view": "front",
            "hand": "left",
            "defect_type": "less",
            "kind": "defect",
        },
        {
            "output_image": relative_empty,
            "view": "back",
            "hand": "right",
            "defect_type": "",
            "kind": "normal_real",
        },
    ]
    ground_truth = {
        relative_positive: [(0.0, 0.0, 1.0, 1.0)],
        relative_empty: [],
    }
    predictions = {
        path_root / relative_positive: [(0.0, 0.0, 1.0, 1.0)],
        path_root / relative_empty: [(2.0, 2.0, 3.0, 3.0)],
    }

    rows = metrics.summarize_grouped_predictions(
        manifest_rows,
        ground_truth,
        predictions,
        path_root=path_root,
    )

    overall = _group(rows, "all", "all")
    assert overall.matched_box_count == 1
    assert overall.empty_image_count == 1
    assert overall.false_positive_image_count == 1


def test_report_writers_use_deterministic_columns_and_ranking(tmp_path):
    trials = [
        _trial(
            "second",
            "m",
            map50_95=0.20,
            recall=0.5,
            empty_label_fpr=0.1,
            map50=0.3,
            epoch_seconds=9.0,
        ),
        _trial(
            "first",
            "n",
            map50_95=0.30,
            recall=0.6,
            empty_label_fpr=0.0,
            map50=0.4,
            epoch_seconds=8.0,
        ),
    ]
    leaderboard_csv = tmp_path / "leaderboard.csv"
    leaderboard_md = tmp_path / "leaderboard.md"
    grouped_csv = tmp_path / "grouped.csv"
    manifest_rows, ground_truth, predictions = _grouped_fixture()
    grouped = metrics.summarize_grouped_predictions(manifest_rows, ground_truth, predictions)

    metrics.write_leaderboard_csv(leaderboard_csv, trials)
    metrics.write_leaderboard_markdown(leaderboard_md, trials)
    metrics.write_grouped_metrics_csv(grouped_csv, reversed(grouped))

    with leaderboard_csv.open(newline="", encoding="utf-8") as file:
        leaderboard_rows = list(csv.DictReader(file))
    assert list(leaderboard_rows[0]) == [
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
    ]
    assert [row["trial_id"] for row in leaderboard_rows] == ["first", "second"]
    assert leaderboard_rows[0]["map50_95"] == "0.300000"

    markdown = leaderboard_md.read_text(encoding="utf-8")
    assert markdown.startswith("# ZS32 YOLO Sweep Leaderboard\n\n")
    assert markdown.index("| 1 | first |") < markdown.index("| 2 | second |")

    with grouped_csv.open(newline="", encoding="utf-8") as file:
        grouped_rows = list(csv.DictReader(file))
    assert list(grouped_rows[0]) == [
        "group_dimension",
        "group_value",
        "image_count",
        "gt_box_count",
        "predicted_box_count",
        "matched_box_count",
        "box_recall",
        "positive_image_count",
        "recalled_image_count",
        "image_recall",
        "empty_image_count",
        "false_positive_image_count",
        "empty_label_fpr",
    ]
    assert grouped_rows[0]["group_dimension"] == "all"
    assert grouped_rows[0]["group_value"] == "all"


def _sweep_args(tmp_path, *extra):
    return sweep.build_parser().parse_args(
        [
            "--project",
            str(tmp_path / "runs"),
            "--experiment-id",
            "unit_plan",
            *extra,
        ]
    )


def test_zs32_sweep_plan_has_exact_18_trials_and_resource_bindings(tmp_path):
    args = _sweep_args(tmp_path)

    plan = sweep.build_experiment_plan(args)

    assert len(plan) == 18
    assert len({trial.trial_id for trial in plan}) == 18
    expected_resources = {
        "n640": ("n", "yolo26n.pt", 640, 64),
        "n1280": ("n", "yolo26n.pt", 1280, 24),
        "n1536": ("n", "yolo26n.pt", 1536, 16),
        "m640": ("m", "yolo26m.pt", 640, 32),
        "m1280": ("m", "yolo26m.pt", 1280, 8),
        "m1536": ("m", "yolo26m.pt", 1536, 8),
    }
    expected_profiles = {"P0_no_aug", "P1_conservative", "P2_adamw_cosine"}
    for trial in plan:
        family, filename, imgsz, batch = expected_resources[trial.resource_id]
        assert trial.model_family == family
        assert trial.model_path == Path("/home/yunjing/ultralytics-c789") / filename
        assert (trial.imgsz, trial.batch) == (imgsz, batch)
        assert trial.profile in expected_profiles
        assert trial.seed == 42
        assert trial.stage == "screen"
        assert (trial.epochs, trial.patience) == (50, 50)
        assert trial.trial_id == f"screen_{trial.resource_id}_p{trial.profile[1]}_seed42"
        for key in sweep.PROTECTED_AUGMENTATIONS:
            assert trial.overrides[key] == 0

    assert sweep.plan_hash(plan) == sweep.plan_hash(sweep.build_experiment_plan(args))
    assert sweep.plan_hash(plan) != sweep.plan_hash([replace(plan[0], seed=43), *plan[1:]])
    with pytest.raises(FrozenInstanceError):
        plan[0].batch = 1


def test_zs32_sweep_plan_hash_covers_selection_inputs_but_not_invocation_stage(tmp_path):
    base = sweep.build_experiment_plan(_sweep_args(tmp_path, "--stage", "screen"))
    final_invocation = sweep.build_experiment_plan(_sweep_args(tmp_path, "--stage", "final"))

    # One experiment ID must support a screen invocation followed by final without a hash mismatch.
    assert sweep.plan_hash(base) == sweep.plan_hash(final_invocation)
    assert sweep.plan_hash(base) == sweep.plan_hash([replace(base[0], stage="final"), *base[1:]])
    assert sweep.plan_hash(base) != sweep.plan_hash([replace(base[0], conf=0.3), *base[1:]])
    assert sweep.plan_hash(base) != sweep.plan_hash([replace(base[0], iou=0.6), *base[1:]])
    assert sweep.plan_hash(base) != sweep.plan_hash([replace(base[0], final_epochs=200), *base[1:]])
    assert sweep.plan_hash(base) != sweep.plan_hash([replace(base[0], finalists=2), *base[1:]])


def test_zs32_sweep_plan_hash_and_records_cover_dataset_and_manifest_content(tmp_path):
    args, _, _, _ = _fake_trial_fixture(tmp_path)
    first = sweep.build_experiment_plan(args)
    first_hash = sweep.plan_hash(first)
    assert first[0].data_yaml_sha256 == sweep._sha256_file(Path(args.data_yaml))
    assert first[0].manifest_sha256 == sweep._sha256_file(Path(args.manifest))
    assert sweep._trial_record(first[0])["data_yaml_sha256"] == first[0].data_yaml_sha256
    assert sweep._trial_record(first[0])["manifest_sha256"] == first[0].manifest_sha256

    Path(args.manifest).write_text("output_image,split\nchanged.png,val\n", encoding="utf-8")
    second = sweep.build_experiment_plan(args)
    assert second[0].manifest_sha256 != first[0].manifest_sha256
    assert sweep.plan_hash(second) != first_hash


@pytest.mark.parametrize(
    ("project", "experiment_id"),
    [
        (Path("/home/yunjing/ultralytics-c789/runs/zs32"), "six_view_yolo26n_1536_v1"),
        (Path("/home/yunjing/ultralytics-c789/runs/zs32/six_view_yolo26n_1536_v1"), "child"),
    ],
)
def test_zs32_sweep_rejects_protected_baseline_root_before_path_reads(monkeypatch, project, experiment_id):
    monkeypatch.setattr(sweep, "_validate_paths", lambda args: (_ for _ in ()).throw(AssertionError("path read")))
    with pytest.raises(ValueError, match="protected baseline"):
        sweep.main(["--dry-run", "--project", str(project), "--experiment-id", experiment_id])


def test_zs32_sweep_cli_defaults_and_validation(tmp_path):
    args = _sweep_args(tmp_path)

    assert args.stage == "all"
    assert args.screen_epochs == 50
    assert args.final_epochs == 150
    assert args.finalists == 3
    assert args.conf == pytest.approx(0.25)
    assert args.iou == pytest.approx(0.50)
    assert args.oom_retry is True

    parser = sweep.build_parser()
    for argv in (
        ["--screen-epochs", "0"],
        ["--final-epochs", "-1"],
        ["--finalists", "0"],
        ["--conf", "1.01"],
        ["--iou", "-0.01"],
    ):
        with pytest.raises(SystemExit):
            parser.parse_args(argv)

    missing = _sweep_args(tmp_path, "--data-yaml", str(tmp_path / "missing.yaml"))
    with pytest.raises(FileNotFoundError, match="data YAML"):
        sweep.build_experiment_plan(missing)
    assert not (tmp_path / "runs").exists()


def test_zs32_sweep_direct_script_entrypoint_resolves_repo_imports():
    script = Path(__file__).resolve().parents[1] / "examples" / "c789" / "sweep_zs32.py"
    completed = subprocess.run(
        [sys.executable, str(script), "--help"],
        cwd=script.parents[2],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert "--stage" in completed.stdout


@pytest.mark.parametrize("experiment_id", ["/tmp/escape", "../escape", "nested/escape", ".", "..", ""])
def test_zs32_sweep_rejects_unsafe_experiment_id_components(tmp_path, experiment_id):
    parser = sweep.build_parser()

    with pytest.raises(SystemExit):
        parser.parse_args(["--project", str(tmp_path / "runs"), "--experiment-id", experiment_id])


def test_zs32_sweep_profile_rejects_nonzero_protected_augmentation():
    with pytest.raises(ValueError, match="protected augmentation"):
        sweep._profile(degrees=1.0)


def test_zs32_sweep_train_kwargs_preserve_safety_and_p2_optimizer(tmp_path):
    args = _sweep_args(tmp_path)
    plan = sweep.build_experiment_plan(args)
    p2 = next(trial for trial in plan if trial.resource_id == "m1280" and trial.profile == "P2_adamw_cosine")

    kwargs = sweep.build_train_kwargs(p2, args)

    assert kwargs["data"] == str(Path(args.data_yaml).resolve())
    assert kwargs["epochs"] == 50
    assert kwargs["patience"] == 50
    assert kwargs["imgsz"] == 1280
    assert kwargs["batch"] == 8
    assert kwargs["seed"] == 42
    assert Path(kwargs["project"]).is_absolute()
    assert kwargs["name"] == p2.trial_id
    assert kwargs["exist_ok"] is False
    assert kwargs["optimizer"] == "AdamW"
    assert kwargs["lr0"] == pytest.approx(0.001)
    assert kwargs["lrf"] == pytest.approx(0.01)
    assert kwargs["cos_lr"] is True
    assert kwargs["weight_decay"] == pytest.approx(0.0005)
    for key in sweep.PROTECTED_AUGMENTATIONS:
        assert kwargs[key] == 0


def test_zs32_sweep_dry_run_writes_plan_without_importing_or_constructing_yolo(tmp_path):
    constructed = []

    def forbidden_factory(*args, **kwargs):
        constructed.append((args, kwargs))
        raise AssertionError("dry-run constructed YOLO")

    sys.modules.pop("ultralytics", None)
    exit_code = sweep.main(
        [
            "--dry-run",
            "--project",
            str(tmp_path / "runs"),
            "--experiment-id",
            "dry",
        ],
        yolo_factory=forbidden_factory,
    )

    assert exit_code == 0
    assert constructed == []
    assert "ultralytics" not in sys.modules
    plan_path = tmp_path / "runs" / "dry" / "experiment_plan.json"
    payload = json.loads(plan_path.read_text(encoding="utf-8"))
    assert len(payload["trials"]) == 18
    assert payload["plan_hash"] == sweep.plan_hash(sweep.build_experiment_plan(_sweep_args(tmp_path)))


@pytest.mark.parametrize(
    ("option", "missing_name", "error_match"),
    [
        ("--data-yaml", "missing.yaml", "data YAML"),
        ("--manifest", "missing.csv", "manifest"),
        ("--weights-dir", "missing_weights", "weights directory"),
    ],
)
def test_zs32_sweep_main_missing_required_path_creates_no_experiment_output(
    tmp_path, option, missing_name, error_match
):
    args, _, _, _ = _fake_trial_fixture(tmp_path)
    root = Path(args.project).resolve() / args.experiment_id

    with pytest.raises(FileNotFoundError, match=error_match):
        sweep.main(
            [*_state_argv(args), "--dry-run", option, str(tmp_path / missing_name)],
            yolo_factory=object(),
        )

    assert not root.exists()


def test_zs32_sweep_dry_run_uses_experiment_lock_and_cannot_overwrite_active_state(tmp_path):
    args, _, _, _ = _fake_trial_fixture(tmp_path)
    assert sweep.main([*_state_argv(args), "--dry-run"], yolo_factory=object()) == 0
    root = Path(args.project).resolve() / args.experiment_id
    plan_before = (root / "experiment_plan.json").read_bytes()
    state_before = (root / "state.json").read_bytes()

    with sweep._experiment_lock(root), pytest.raises(RuntimeError, match="locked"):
        sweep.main([*_state_argv(args), "--dry-run"], yolo_factory=object())

    assert (root / "experiment_plan.json").read_bytes() == plan_before
    assert (root / "state.json").read_bytes() == state_before


def test_zs32_sweep_non_dry_lock_rejects_same_experiment_and_releases_after_context(tmp_path, monkeypatch):
    args, _, _, _ = _fake_trial_fixture(tmp_path)
    argv = _state_argv(args)
    root = Path(args.project).resolve() / args.experiment_id
    monkeypatch.setattr(sweep, "run_trial", lambda trial, _args, *, yolo_factory: _fake_completed_result(trial))

    with sweep._experiment_lock(root), pytest.raises(RuntimeError, match="locked"):
        sweep.main(argv, yolo_factory=object())

    assert sweep.main(argv, yolo_factory=object()) == 0


def test_zs32_sweep_process_lock_is_nonblocking_and_released_after_crash(tmp_path):
    root = tmp_path / "runs" / "locked_experiment"
    code = (
        "import sys; from pathlib import Path; "
        "from examples.c789.sweep_zs32 import _experiment_lock; "
        "ctx=_experiment_lock(Path(sys.argv[1])); ctx.__enter__(); "
        "print('LOCKED', flush=True); sys.stdin.read()"
    )
    process = subprocess.Popen(
        [sys.executable, "-c", code, str(root)],
        cwd=Path(__file__).resolve().parents[1],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        assert process.stdout.readline().strip() == "LOCKED"
        with pytest.raises(RuntimeError, match="locked"), sweep._experiment_lock(root):
            pass
        process.kill()
        process.wait(timeout=5)
        with sweep._experiment_lock(root):
            pass
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=5)


def _fake_trial_fixture(tmp_path):
    dataset = tmp_path / "dataset"
    for split in ("train", "val", "test"):
        (dataset / "images" / split).mkdir(parents=True)
        (dataset / "labels" / split).mkdir(parents=True)
    positive = dataset / "images" / "val" / "positive.png"
    empty = dataset / "images" / "val" / "empty.png"
    positive.touch()
    empty.touch()
    (dataset / "labels" / "val" / "positive.txt").write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")
    (dataset / "labels" / "val" / "empty.txt").write_text("", encoding="utf-8")
    data_yaml = dataset / "data.yaml"
    data_yaml.write_text(
        "train: images/train\nval: images/val\ntest: images/test\nnames:\n  0: defect\n",
        encoding="utf-8",
    )
    manifest = dataset / "split_manifest.csv"
    manifest.write_text("output_image,split\npositive.png,val\nempty.png,val\n", encoding="utf-8")
    weights = tmp_path / "yolo26n.pt"
    weights.touch()
    (tmp_path / "yolo26m.pt").touch()
    args = _sweep_args(
        tmp_path,
        "--data-yaml",
        str(data_yaml),
        "--manifest",
        str(manifest),
        "--weights-dir",
        str(tmp_path),
    )
    trial = sweep.build_experiment_plan(args)[0]
    return args, trial, positive, empty


def test_zs32_sweep_run_trial_uses_injected_yolo_and_collects_artifacts(tmp_path):
    args, trial, positive, empty = _fake_trial_fixture(tmp_path)
    args.conf = 0.99
    args.iou = 0.01
    instances = []

    class FakeYOLO:
        def __init__(self, model, task):
            self.constructor = (model, task)
            self.train_calls = []
            self.val_calls = []
            self.predict_calls = []
            self.trainer = None
            instances.append(self)

        def train(self, **kwargs):
            self.train_calls.append(kwargs)
            run_dir = Path(kwargs["project"]) / kwargs["name"]
            (run_dir / "weights").mkdir(parents=True)
            _write_results_csv(
                run_dir / "results.csv",
                [
                    {
                        "epoch": 1,
                        "time": 3.0,
                        "metrics/precision(B)": 0.4,
                        "metrics/recall(B)": 0.5,
                        "metrics/mAP50(B)": 0.3,
                        "metrics/mAP50-95(B)": 0.2,
                    }
                ],
            )
            (run_dir / "weights" / "best.pt").touch()
            self.trainer = SimpleNamespace(save_dir=run_dir, peak_gpu_memory_gb=1.25)

        def val(self, **kwargs):
            self.val_calls.append(kwargs)
            return SimpleNamespace()

        def predict(self, **kwargs):
            self.predict_calls.append(kwargs)
            return [
                SimpleNamespace(path=str(positive), boxes=[]),
                SimpleNamespace(path=str(empty), boxes=[object()]),
            ]

    result = sweep.run_trial(trial, args, yolo_factory=FakeYOLO)

    fake = instances[0]
    assert fake.constructor == (str(trial.model_path), "detect")
    assert len(fake.train_calls) == len(fake.val_calls) == len(fake.predict_calls) == 1
    assert fake.val_calls[0]["split"] == "val"
    assert fake.val_calls[0]["project"] == str(Path(result.run_dir))
    assert fake.val_calls[0]["name"] == "val"
    assert fake.val_calls[0]["exist_ok"] is True
    assert fake.val_calls[0]["plots"] is False
    assert fake.val_calls[0]["conf"] == pytest.approx(0.001)
    assert fake.val_calls[0]["iou"] == pytest.approx(0.7)
    assert fake.predict_calls[0]["conf"] == pytest.approx(0.25)
    assert fake.predict_calls[0]["iou"] == pytest.approx(0.50)
    assert result.trial_id == trial.trial_id
    assert result.best.map50_95 == pytest.approx(0.2)
    assert result.empty_label_fpr == pytest.approx(1.0)
    assert Path(result.best_pt).is_file()
    execution = json.loads((Path(result.run_dir) / "execution_metadata.json").read_text())
    assert execution["peak_gpu_memory_gb"] == pytest.approx(1.25)


def test_zs32_sweep_standard_val_thresholds_do_not_change_operational_predict_thresholds(tmp_path):
    args, screen_trial, positive, empty = _fake_trial_fixture(tmp_path)
    calls = []

    class FakeYOLO:
        def __init__(self, model, task):
            self.trainer = None

        def train(self, **kwargs):
            run_dir = Path(kwargs["project"]) / kwargs["name"]
            (run_dir / "weights").mkdir(parents=True)
            _write_results_csv(
                run_dir / "results.csv",
                [
                    {
                        "epoch": 1,
                        "time": 1,
                        "metrics/precision(B)": 0.5,
                        "metrics/recall(B)": 0.5,
                        "metrics/mAP50(B)": 0.5,
                        "metrics/mAP50-95(B)": 0.5,
                    }
                ],
            )
            (run_dir / "weights" / "best.pt").touch()
            self.trainer = SimpleNamespace(save_dir=run_dir, peak_gpu_memory_gb=0.0)

        def val(self, **kwargs):
            calls.append(("val", kwargs["conf"], kwargs["iou"]))
            return SimpleNamespace()

        def predict(self, **kwargs):
            calls.append(("predict", kwargs["conf"], kwargs["iou"]))
            return [SimpleNamespace(path=str(positive), boxes=[]), SimpleNamespace(path=str(empty), boxes=[])]

    sweep.run_trial(screen_trial, args, yolo_factory=FakeYOLO)
    assert ("val", 0.001, 0.7) in calls
    assert ("predict", 0.25, 0.5) in calls


def test_zs32_sweep_peak_gpu_memory_is_reset_and_measured_per_attempt():
    class FakeCudaMemory:
        def __init__(self):
            self.peaks = iter([8 * 1024**3, 2 * 1024**3])
            self.reset_calls = 0

        def reset_peak_memory_stats(self):
            self.reset_calls += 1

        def max_memory_allocated(self):
            return next(self.peaks)

    backend = FakeCudaMemory()
    measured = []
    for _ in range(2):
        sweep._reset_peak_gpu_memory(backend)
        measured.append(sweep._peak_gpu_memory_gb(object(), backend))

    assert backend.reset_calls == 2
    assert measured == [pytest.approx(8.0), pytest.approx(2.0)]


def test_zs32_sweep_run_trial_propagates_injected_training_failure(tmp_path):
    args, trial, _, _ = _fake_trial_fixture(tmp_path)

    class FailingYOLO:
        def __init__(self, model, task):
            self.model = model
            self.task = task

        def train(self, **kwargs):
            raise RuntimeError("synthetic training failure")

        def val(self, **kwargs):
            raise AssertionError("val must not run after training failure")

        def predict(self, **kwargs):
            raise AssertionError("predict must not run after training failure")

    with pytest.raises(RuntimeError, match="synthetic training failure"):
        sweep.run_trial(trial, args, yolo_factory=FailingYOLO)


def test_zs32_sweep_empty_label_fpr_rejects_missing_label_file(tmp_path):
    image = tmp_path / "images" / "val" / "missing.png"
    image.parent.mkdir(parents=True)
    image.touch()

    with pytest.raises(FileNotFoundError, match="label file"):
        sweep._empty_label_fpr([SimpleNamespace(path=str(image), boxes=[])])


def _state_argv(args, *, stage="screen", max_trials=1, extra=()):
    return [
        "--stage",
        stage,
        "--project",
        str(args.project),
        "--experiment-id",
        args.experiment_id,
        "--data-yaml",
        str(args.data_yaml),
        "--manifest",
        str(args.manifest),
        "--weights-dir",
        str(args.weights_dir),
        "--max-trials",
        str(max_trials),
        *extra,
    ]


def _fake_completed_result(trial, *, score=0.2):
    run_dir = trial.project / trial.trial_id
    (run_dir / "weights").mkdir(parents=True, exist_ok=True)
    _write_results_csv(
        run_dir / "results.csv",
        [
            {
                "epoch": 1,
                "time": 2.0,
                "metrics/precision(B)": score,
                "metrics/recall(B)": score,
                "metrics/mAP50(B)": score,
                "metrics/mAP50-95(B)": score,
            }
        ],
    )
    best_pt = run_dir / "weights" / "best.pt"
    best_pt.touch()
    return metrics.TrialMetrics(
        trial_id=trial.trial_id,
        model_family=trial.model_family,
        profile=trial.profile,
        imgsz=trial.imgsz,
        batch=trial.batch,
        best=metrics.EpochMetrics(1, 2.0, score, score, score, score),
        mean_epoch_seconds=2.0,
        empty_label_fpr=0.0,
        run_dir=str(run_dir),
        best_pt=str(best_pt),
    )


def test_zs32_sweep_atomic_state_complete_skip_and_stale_resume(tmp_path, monkeypatch):
    args, _, _, _ = _fake_trial_fixture(tmp_path)
    calls = []

    def fake_run(trial, _args, *, yolo_factory):
        calls.append((trial.trial_id, trial.batch))
        return _fake_completed_result(trial)

    monkeypatch.setattr(sweep, "run_trial", fake_run)
    argv = _state_argv(args)
    assert sweep.main(argv, yolo_factory=object()) == 0
    experiment_root = Path(args.project).resolve() / args.experiment_id
    state_path = experiment_root / "state.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    assert state["plan_hash"] == json.loads((experiment_root / "experiment_plan.json").read_text())["plan_hash"]
    assert state["attempts"][0]["status"] == "complete"
    assert not (experiment_root / ".state.json.tmp").exists()

    assert sweep.main(argv, yolo_factory=object()) == 0
    assert len(calls) == 1

    Path(state["attempts"][0]["artifacts"]["best_pt"]).unlink()
    with pytest.raises(RuntimeError, match="requires --resume"):
        sweep.main(argv, yolo_factory=object())
    assert sweep.main([*argv, "--resume"], yolo_factory=object()) == 0
    assert len(calls) == 2
    assert calls[-1][0].startswith("screen_n640_p0_seed42__resume")


def test_zs32_sweep_partial_run_directory_requires_resume_and_uses_new_name(tmp_path, monkeypatch):
    args, trial, _, _ = _fake_trial_fixture(tmp_path)
    partial = trial.project / trial.trial_id
    partial.mkdir(parents=True)
    calls = []

    def fake_run(candidate, _args, *, yolo_factory):
        calls.append(candidate.trial_id)
        return _fake_completed_result(candidate)

    monkeypatch.setattr(sweep, "run_trial", fake_run)
    argv = _state_argv(args)
    with pytest.raises(RuntimeError, match="requires --resume"):
        sweep.main(argv, yolo_factory=object())
    assert calls == []
    assert sweep.main([*argv, "--resume"], yolo_factory=object()) == 0
    assert calls == [f"{trial.trial_id}__resume1"]


def test_zs32_sweep_keyboard_interrupt_leaves_running_attempt_recoverable(tmp_path, monkeypatch):
    args, trial, _, _ = _fake_trial_fixture(tmp_path)

    def interrupt(*_args, **_kwargs):
        raise KeyboardInterrupt

    monkeypatch.setattr(sweep, "run_trial", interrupt)
    with pytest.raises(KeyboardInterrupt):
        sweep.main(_state_argv(args), yolo_factory=object())
    state = json.loads((Path(args.project).resolve() / args.experiment_id / "state.json").read_text())
    assert state["attempts"][0]["source_trial_id"] == trial.trial_id
    assert state["attempts"][0]["status"] == "running"
    assert state["attempts"][0]["ended_at"] is None


def test_zs32_sweep_rejects_plan_hash_mismatch(tmp_path, monkeypatch):
    args, _, _, _ = _fake_trial_fixture(tmp_path)
    monkeypatch.setattr(sweep, "run_trial", lambda trial, _args, *, yolo_factory: _fake_completed_result(trial))
    argv = _state_argv(args)
    assert sweep.main(argv, yolo_factory=object()) == 0
    state_path = Path(args.project).resolve() / args.experiment_id / "state.json"
    state = json.loads(state_path.read_text())
    state["plan_hash"] = "different-plan"
    sweep.atomic_write_json(state_path, state)

    with pytest.raises(ValueError, match="plan hash"):
        sweep.main([*argv, "--resume"], yolo_factory=object())


def test_zs32_sweep_plan_hash_mismatch_is_rejected_before_any_provenance_write(tmp_path, monkeypatch):
    args, _, _, _ = _fake_trial_fixture(tmp_path)
    monkeypatch.setattr(sweep, "run_trial", lambda trial, _args, *, yolo_factory: _fake_completed_result(trial))
    argv = _state_argv(args)
    assert sweep.main(argv, yolo_factory=object()) == 0
    root = Path(args.project).resolve() / args.experiment_id
    plan_before = (root / "experiment_plan.json").read_bytes()
    state_before = (root / "state.json").read_bytes()

    with pytest.raises(ValueError, match="plan hash"):
        sweep.main([*argv, "--seed", "43", "--resume"], yolo_factory=object())
    assert (root / "experiment_plan.json").read_bytes() == plan_before
    assert (root / "state.json").read_bytes() == state_before


def test_zs32_sweep_final_complete_validity_requires_evaluation_and_grouped_artifacts(tmp_path):
    result = _fake_completed_result(
        SimpleNamespace(
            project=tmp_path,
            trial_id="final_n640_p0_seed42",
            model_family="n",
            profile="P0_no_aug",
            imgsz=640,
            batch=64,
        )
    )
    run_dir = Path(result.run_dir)
    evaluation = run_dir / "evaluation_metrics.json"
    grouped = run_dir / "grouped_metrics.csv"
    evaluation.write_text('{"val": {}, "test": {}}\n', encoding="utf-8")
    grouped.write_text("group_dimension,group_value\nall,all\n", encoding="utf-8")
    attempt = {
        "status": "complete",
        "stage": "final",
        "metrics": _metric_payload_for_test(result),
        "artifacts": {
            "results_csv": str(run_dir / "results.csv"),
            "best_pt": result.best_pt,
            "evaluation_metrics": str(evaluation),
            "grouped_metrics": str(grouped),
        },
    }
    assert sweep._attempt_is_valid(attempt)
    grouped.unlink()
    assert not sweep._attempt_is_valid(attempt)


def _metric_payload_for_test(result):
    return {
        "trial_id": result.trial_id,
        "model_family": result.model_family,
        "profile": result.profile,
        "imgsz": result.imgsz,
        "batch": result.batch,
        "best": {
            "epoch": result.best.epoch,
            "epoch_seconds": result.best.epoch_seconds,
            "precision": result.best.precision,
            "recall": result.best.recall,
            "map50": result.best.map50,
            "map50_95": result.best.map50_95,
        },
        "mean_epoch_seconds": result.mean_epoch_seconds,
        "empty_label_fpr": result.empty_label_fpr,
        "run_dir": result.run_dir,
        "best_pt": result.best_pt,
        "status": result.status,
    }


def test_zs32_sweep_screen_metrics_uses_latest_valid_attempt_per_source(tmp_path):
    source = SimpleNamespace(
        project=tmp_path,
        trial_id="screen_n640_p0_seed42",
        model_family="n",
        profile="P0_no_aug",
        imgsz=640,
        batch=64,
    )
    first = _fake_completed_result(
        SimpleNamespace(**{**source.__dict__, "trial_id": "screen_n640_p0_seed42__resume1"}), score=0.1
    )
    second = _fake_completed_result(
        SimpleNamespace(**{**source.__dict__, "trial_id": "screen_n640_p0_seed42__resume2"}), score=0.2
    )

    def attempt(result):
        return {
            "attempt_id": result.trial_id,
            "source_trial_id": source.trial_id,
            "stage": "screen",
            "status": "complete",
            "metrics": _metric_payload_for_test(result),
            "artifacts": {
                "results_csv": str(Path(result.run_dir) / "results.csv"),
                "best_pt": result.best_pt,
            },
        }

    rows = sweep._screen_metrics({"attempts": [attempt(first), attempt(second)]})
    assert len(rows) == 1
    assert rows[0].trial_id == second.trial_id


def test_zs32_sweep_failure_continues_and_fail_fast_stops_with_reports(tmp_path, monkeypatch):
    args, _, _, _ = _fake_trial_fixture(tmp_path)
    calls = []

    def fail_first(trial, _args, *, yolo_factory):
        calls.append(trial.trial_id)
        if len(calls) == 1:
            raise RuntimeError("synthetic non-OOM failure")
        return _fake_completed_result(trial, score=0.3)

    monkeypatch.setattr(sweep, "run_trial", fail_first)
    argv = _state_argv(args, max_trials=2)
    assert sweep.main(argv, yolo_factory=object()) == 1
    assert len(calls) == 2
    root = Path(args.project).resolve() / args.experiment_id
    assert (root / "failed_trials.csv").read_text().count("synthetic non-OOM failure") == 1
    assert (root / "leaderboard.csv").is_file()
    assert (root / "leaderboard.md").is_file()
    state = json.loads((root / "state.json").read_text())
    assert [attempt["status"] for attempt in state["attempts"]] == ["failed", "complete"]

    args2, _, _, _ = _fake_trial_fixture(tmp_path / "failfast")
    calls.clear()
    argv2 = _state_argv(args2, max_trials=2, extra=("--fail-fast",))
    assert sweep.main(argv2, yolo_factory=object()) == 1
    assert len(calls) == 1


def test_zs32_sweep_oom_retries_once_at_half_batch_as_distinct_attempt(tmp_path, monkeypatch):
    args, trial, _, _ = _fake_trial_fixture(tmp_path)
    calls = []

    def oom_once(candidate, _args, *, yolo_factory):
        calls.append((candidate.trial_id, candidate.batch))
        if len(calls) == 1:
            raise sweep.TrainingOOMError("CUDA out of memory")
        return _fake_completed_result(candidate)

    monkeypatch.setattr(sweep, "run_trial", oom_once)
    assert sweep.main(_state_argv(args), yolo_factory=object()) == 0
    assert calls == [(trial.trial_id, trial.batch), (f"{trial.trial_id}__oom_b{trial.batch // 2}", trial.batch // 2)]
    state = json.loads((Path(args.project).resolve() / args.experiment_id / "state.json").read_text())
    assert [attempt["status"] for attempt in state["attempts"]] == ["oom", "complete"]
    assert [attempt["requested_batch"] for attempt in state["attempts"]] == [trial.batch, trial.batch]
    assert [attempt["effective_batch"] for attempt in state["attempts"]] == [trial.batch, trial.batch // 2]
    assert state["attempts"][0]["attempt_id"] != state["attempts"][1]["attempt_id"]


def test_zs32_sweep_evaluation_oom_is_failed_without_training_retry(tmp_path, monkeypatch):
    args, trial, _, _ = _fake_trial_fixture(tmp_path)
    calls = []

    def evaluation_oom(candidate, _args, *, yolo_factory):
        calls.append(candidate.trial_id)
        raise RuntimeError("CUDA out of memory during val")

    monkeypatch.setattr(sweep, "run_trial", evaluation_oom)
    assert sweep.main(_state_argv(args), yolo_factory=object()) == 1
    assert calls == [trial.trial_id]
    state = json.loads((Path(args.project).resolve() / args.experiment_id / "state.json").read_text())
    assert [attempt["status"] for attempt in state["attempts"]] == ["failed"]


def test_zs32_sweep_oom_half_batch_retry_occurs_only_once_across_resume(tmp_path, monkeypatch):
    args, trial, _, _ = _fake_trial_fixture(tmp_path)
    calls = []

    def always_oom(candidate, _args, *, yolo_factory):
        calls.append(candidate.trial_id)
        raise sweep.TrainingOOMError("CUDA out of memory")

    monkeypatch.setattr(sweep, "run_trial", always_oom)
    argv = _state_argv(args)
    assert sweep.main(argv, yolo_factory=object()) == 1
    assert calls == [trial.trial_id, f"{trial.trial_id}__oom_b{trial.batch // 2}"]
    assert sweep.main([*argv, "--resume"], yolo_factory=object()) == 1
    state = json.loads((Path(args.project).resolve() / args.experiment_id / "state.json").read_text())
    half_batch_attempts = [attempt for attempt in state["attempts"] if attempt["effective_batch"] == trial.batch // 2]
    assert len(half_batch_attempts) == 1
    assert len(calls) == 3


def test_zs32_sweep_resume_runs_pending_half_batch_after_primary_oom_was_persisted(tmp_path, monkeypatch):
    args, trial, _, _ = _fake_trial_fixture(tmp_path)
    argv = _state_argv(args)
    assert sweep.main([*argv, "--dry-run"], yolo_factory=object()) == 0
    root = Path(args.project).resolve() / args.experiment_id
    state_path = root / "state.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    primary = sweep._attempt_record(trial, trial, trial.batch)
    primary.update(status="oom", ended_at="2026-07-13T00:00:00+00:00", error="TrainingOOMError: CUDA out of memory")
    state["attempts"] = [primary]
    sweep.atomic_write_json(state_path, state)
    calls = []

    def complete(candidate, _args, *, yolo_factory):
        calls.append((candidate.trial_id, candidate.batch))
        return _fake_completed_result(candidate)

    monkeypatch.setattr(sweep, "run_trial", complete)
    assert sweep.main([*argv, "--resume"], yolo_factory=object()) == 0
    assert calls == [(f"{trial.trial_id}__oom_b{trial.batch // 2}", trial.batch // 2)]
    assert sweep.main([*argv, "--resume"], yolo_factory=object()) == 0
    assert len(calls) == 1
    persisted = json.loads(state_path.read_text(encoding="utf-8"))
    half_attempts = [attempt for attempt in persisted["attempts"] if attempt["effective_batch"] == trial.batch // 2]
    assert len(half_attempts) == 1


def test_zs32_sweep_failed_attempt_writes_peak_gpu_metadata(tmp_path):
    args, trial, _, _ = _fake_trial_fixture(tmp_path)

    class FailingYOLO:
        def __init__(self, model, task):
            self.trainer = SimpleNamespace(peak_gpu_memory_gb=3.5)

        def train(self, **kwargs):
            run_dir = Path(kwargs["project"]) / kwargs["name"]
            run_dir.mkdir(parents=True)
            self.trainer.save_dir = run_dir
            raise RuntimeError("synthetic failure")

    with pytest.raises(RuntimeError, match="synthetic failure"):
        sweep.run_trial(trial, args, yolo_factory=FailingYOLO)
    metadata = json.loads((trial.project / trial.trial_id / "execution_metadata.json").read_text())
    assert metadata["peak_gpu_memory_gb"] == pytest.approx(3.5)
    assert metadata["peak_gpu_memory_status"] == "recorded"


def test_zs32_sweep_failed_attempt_persists_peak_gpu_in_state(tmp_path):
    args, _, _, _ = _fake_trial_fixture(tmp_path)

    class FailingYOLO:
        def __init__(self, model, task):
            self.trainer = SimpleNamespace(peak_gpu_memory_gb=4.25)

        def train(self, **kwargs):
            run_dir = Path(kwargs["project"]) / kwargs["name"]
            run_dir.mkdir(parents=True)
            self.trainer.save_dir = run_dir
            raise RuntimeError("synthetic failure")

    assert sweep.main(_state_argv(args), yolo_factory=FailingYOLO) == 1
    state = json.loads((Path(args.project).resolve() / args.experiment_id / "state.json").read_text())
    assert state["attempts"][0]["status"] == "failed"
    assert state["attempts"][0]["peak_gpu_memory_gb"] == pytest.approx(4.25)
    assert state["attempts"][0]["peak_gpu_memory_status"] == "recorded"


def test_zs32_sweep_finalists_retrain_original_weights_and_test_only_final(tmp_path, monkeypatch):
    args, _, _, _ = _fake_trial_fixture(tmp_path)
    calls = []

    def fake_run(trial, _args, *, yolo_factory):
        calls.append(trial)
        family_bonus = 0.2 if trial.model_family == "m" else 0.1
        profile_bonus = int(trial.profile[1]) / 100
        result = _fake_completed_result(trial, score=family_bonus + profile_bonus)
        if trial.stage == "final":
            grouped = [metrics.GroupedMetrics("all", "all", 1, 1, 1, 1, 1.0, 1, 1, 1.0, 0, 0, 0.0)]
            metrics.write_grouped_metrics_csv(Path(result.run_dir) / "grouped_metrics.csv", grouped)
        return result

    monkeypatch.setattr(sweep, "run_trial", fake_run)
    # all executes all 18 screen trials, promotes best n, best m, then the best remaining, then runs three clean finals.
    assert sweep.main(_state_argv(args, stage="all", max_trials=18), yolo_factory=object()) == 0
    screen = [trial for trial in calls if trial.stage == "screen"]
    finals = [trial for trial in calls if trial.stage == "final"]
    assert len(screen) == 18
    assert len(finals) == 3
    assert {trial.model_family for trial in finals} == {"n", "m"}
    assert all(trial.epochs == 150 and trial.patience == 30 for trial in finals)
    assert all(trial.project.name == "finalists" for trial in finals)
    assert all(trial.model_path.name in {"yolo26n.pt", "yolo26m.pt"} for trial in finals)
    assert not any("best.pt" in str(trial.model_path) for trial in finals)
    root = Path(args.project).resolve() / args.experiment_id
    assert (root / "grouped_metrics.csv").is_file()
    state = json.loads((root / "state.json").read_text())
    assert state["stage"] == "all"
    assert len(state["finalists"]) == 3


def test_zs32_sweep_run_trial_calls_test_only_for_finalist(tmp_path):
    args, screen_trial, positive, empty = _fake_trial_fixture(tmp_path)
    test_positive = positive.parents[1] / "test" / "test_positive.png"
    test_positive.touch()
    _label_path = test_positive.parents[2] / "labels" / "test" / "test_positive.txt"
    _label_path.write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")
    args.manifest.write_text(
        f"output_image,split,view,hand,defect_type,kind\n{test_positive},test,front,left,less,defect\n",
        encoding="utf-8",
    )
    split_calls = []

    class FakeYOLO:
        def __init__(self, model, task):
            self.trainer = None

        def train(self, **kwargs):
            run_dir = Path(kwargs["project"]) / kwargs["name"]
            (run_dir / "weights").mkdir(parents=True)
            _write_results_csv(
                run_dir / "results.csv",
                [
                    {
                        "epoch": 1,
                        "time": 1,
                        "metrics/precision(B)": 0.5,
                        "metrics/recall(B)": 0.5,
                        "metrics/mAP50(B)": 0.5,
                        "metrics/mAP50-95(B)": 0.5,
                    }
                ],
            )
            (run_dir / "weights" / "best.pt").touch()
            self.trainer = SimpleNamespace(save_dir=run_dir)

        def val(self, **kwargs):
            split_calls.append(("val", kwargs["split"]))

        def predict(self, **kwargs):
            source = kwargs["source"]
            is_test = "test" in str(source)
            split_calls.append(("predict", "test" if is_test else "val"))
            path = test_positive if is_test else empty
            boxes = SimpleNamespace(xyxy=[[0.0, 0.0, 10.0, 10.0]]) if is_test else []
            return [SimpleNamespace(path=str(path), boxes=boxes, orig_shape=(100, 100))]

    sweep.run_trial(screen_trial, args, yolo_factory=FakeYOLO)
    assert ("val", "test") not in split_calls
    final_trial = sweep._final_trial(screen_trial, effective_batch=screen_trial.batch)
    result = sweep.run_trial(final_trial, args, yolo_factory=FakeYOLO)
    assert ("val", "test") in split_calls
    assert (Path(result.run_dir) / "grouped_metrics.csv").is_file()
    evaluation = json.loads((Path(result.run_dir) / "evaluation_metrics.json").read_text())
    assert set(evaluation) == {"val", "test"}
