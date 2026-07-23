"""Tests for sake_rice_inspection.visualization.

Uses the Agg backend (set in conftest.py) so plotting works headlessly.
"""

import numpy as np

from sake_rice_inspection.visualization import (
    plot_curves,
    plot_multiclass_confusion,
    plot_multilabel_metrics,
    plot_roc_curve,
)


def test_plot_multilabel_metrics_saves_file(tmp_path):
    y_true = np.array([[1, 0], [0, 1], [1, 1], [0, 0]])
    y_pred = np.array([[1, 0], [0, 0], [1, 1], [0, 1]])
    out_path = tmp_path / "multilabel.png"

    plot_multilabel_metrics(y_true, y_pred, ["A", "B"], save_path=str(out_path))

    assert out_path.exists()


def test_plot_multiclass_confusion_saves_file(tmp_path):
    y_true_str = ["A", "B", "A+B", "A"]
    y_pred_str = ["A", "B", "A", "A"]
    out_path = tmp_path / "multiclass.png"

    plot_multiclass_confusion(y_true_str, y_pred_str, save_path=str(out_path))

    assert out_path.exists()


def test_plot_curves_saves_file(tmp_path):
    history = {
        "epoch": [1, 2, 3],
        "train_loss": [1.0, 0.8, 0.6],
        "val_acc": [0.5, 0.6, 0.7],
        "val_f1": [0.4, 0.5, 0.6],
    }
    out_path = tmp_path / "curves.png"

    plot_curves(history, save_path=str(out_path))

    assert out_path.exists()


def test_plot_roc_curve_returns_auc_and_saves_file(tmp_path):
    y_true = np.array([0, 0, 1, 1])
    y_score = np.array([0.1, 0.4, 0.6, 0.9])
    out_path = tmp_path / "roc.png"

    auc_value = plot_roc_curve(y_true, y_score, save_path=str(out_path))

    assert 0.0 <= auc_value <= 1.0
    assert out_path.exists()
