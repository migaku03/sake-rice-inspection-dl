"""Tests for sake_rice_inspection.experiment."""

from sake_rice_inspection.experiment import (
    create_experiment_dir,
    save_experiment_readme,
    summarize_fold_results,
    update_readme_with_results,
)

SAMPLE_CONFIG = {
    "tag": "test_run",
    "description": "A test run.",
    "crop_dir": "/data/crops",
    "folds_csv": "/data/folds.csv",
    "num_classes": 2,
    "details": "test",
    "class_names": ["A", "B"],
    "folds": 2,
    "arch": "resnet18",
    "img_size": 224,
    "batch_size": 8,
    "epochs_head": 1,
    "lr_head": 1e-3,
    "epochs_ft": 1,
    "lr_ft": 1e-4,
    "seed": 42,
    "augmentation": "none",
    "augmentation_details": "none",
    "scale": 1.0,
}


def test_create_experiment_dir_is_unique(tmp_path):
    dir1 = create_experiment_dir(str(tmp_path), "run", date="20260101")
    dir2 = create_experiment_dir(str(tmp_path), "run", date="20260101")

    assert dir1 != dir2
    assert dir1.endswith("run_20260101_001")
    assert dir2.endswith("run_20260101_002")


def test_save_experiment_readme_contains_config_values(tmp_path):
    exp_dir = create_experiment_dir(str(tmp_path), "run", date="20260101")
    readme_path = save_experiment_readme(exp_dir, SAMPLE_CONFIG)

    content = open(readme_path, encoding="utf-8-sig").read()
    assert "test_run" in content
    assert "resnet18" in content
    assert "A, B" in content


def test_summarize_fold_results_computes_totals():
    fold_results = [
        {"fold": 0, "eval_count": 10, "exact_match_count": 8, "exact_match_acc": 0.8, "hamming_acc": 0.9, "macro_f1": 0.85},
        {"fold": 1, "eval_count": 10, "exact_match_count": 6, "exact_match_acc": 0.6, "hamming_acc": 0.7, "macro_f1": 0.65},
    ]

    summary = summarize_fold_results(fold_results)

    assert summary["total_eval"] == 20
    assert summary["total_exact_match"] == 14
    assert summary["exact_match_acc"] == 0.7
    assert summary["hamming_acc"] == 0.8
    assert summary["macro_f1"] == 0.75


def test_update_readme_with_results_appends_section(tmp_path):
    exp_dir = create_experiment_dir(str(tmp_path), "run", date="20260101")
    save_experiment_readme(exp_dir, SAMPLE_CONFIG)

    fold_results = [
        {"fold": 0, "eval_count": 5, "exact_match_count": 4, "exact_match_acc": 0.8, "hamming_acc": 0.9, "macro_f1": 0.85},
    ]
    summary = summarize_fold_results(fold_results)
    update_readme_with_results(exp_dir, summary)

    content = open(f"{exp_dir}/README.md", encoding="utf-8-sig").read()
    assert "Final Results" in content
    assert "0.8500" in content or "0.85" in content
