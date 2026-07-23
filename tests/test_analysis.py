"""Tests for sake_rice_inspection.analysis."""

import numpy as np
import pandas as pd

from sake_rice_inspection.analysis import (
    analyze_feature_binary,
    apply_pw_correction,
    collect_binary_vectors,
    compute_classification_metrics,
    format_metrics_markdown,
    hamming_accuracy_label_mean,
    integrate_hierarchical_predictions,
    labels_to_presence_strings,
    order_labels,
    plot_confusion_matrix,
    split_label,
)

ALLOWED_PW_LABELS = [
    "心白", "背白", "腹白", "基白",
    "心白+背白", "心白+腹白", "心白+基白",
    "背白+腹白", "基白+腹白", "心白+背白+腹白",
]


def test_split_label_handles_combinations_and_nan():
    assert split_label("心白+基白") == ["心白", "基白"]
    assert split_label("心白") == ["心白"]
    assert split_label(float("nan")) == []


def test_apply_pw_correction_keeps_allowed_labels_unchanged():
    df = pd.DataFrame({
        "img": ["a.jpg"], "filename": ["a.jpg"],
        "true": ["心白"], "pred": ["心白"],
        "心白_score": [0.9], "基白_score": [0.1], "背白_score": [0.05], "腹白_score": [0.05],
    })
    result = apply_pw_correction(df, ALLOWED_PW_LABELS)
    assert result["pred_corrected"].iloc[0] == "心白"


def test_apply_pw_correction_remaps_invalid_combination_by_score():
    # "心白+基白+背白+腹白" (all four) is not in ALLOWED_PW_LABELS; the corrector
    # must fall back to the best-scoring allowed combination.
    df = pd.DataFrame({
        "img": ["a.jpg"], "filename": ["a.jpg"],
        "true": ["心白+基白"], "pred": ["心白+基白+背白+腹白"],
        "心白_score": [0.9], "基白_score": [0.9], "背白_score": [0.05], "腹白_score": [0.05],
    })
    result = apply_pw_correction(df, ALLOWED_PW_LABELS)
    assert result["pred_corrected"].iloc[0] == "心白+基白"


def test_compute_classification_metrics_perfect_predictions():
    true = ["A", "B", "A", "B"]
    pred = ["A", "B", "A", "B"]

    metrics = compute_classification_metrics(true, pred)

    assert metrics["accuracy"] == 1.0
    assert metrics["f1"] == 1.0
    assert metrics["eval_count"] == 4
    assert metrics["correct_count"] == 4
    assert metrics["n_classes"] == 2


def test_format_metrics_markdown_includes_all_fields():
    metrics = compute_classification_metrics(["A", "A"], ["A", "B"])
    lines = format_metrics_markdown("Test Title", metrics)

    joined = "\n".join(lines)
    assert "Test Title" in joined
    assert "Accuracy" in joined
    assert "eval_count: 2" in joined


def test_order_labels_places_desired_order_first():
    labels = ["Z", "A", "M"]
    ordered = order_labels(labels, desired_order=["M", "A"])
    assert ordered == ["M", "A", "Z"]


def test_order_labels_without_desired_order_is_identity():
    labels = ["Z", "A", "M"]
    assert order_labels(labels) == labels


def test_plot_confusion_matrix_saves_file(tmp_path):
    out_path = tmp_path / "cm.png"
    plot_confusion_matrix(["A", "B", "A"], ["A", "B", "B"], title="Test", save_path=str(out_path))
    assert out_path.exists()


def test_analyze_feature_binary_detects_substring():
    df = pd.DataFrame({"true": ["心白+基白", "背白"], "pred": ["基白", "心白"]})
    y_true, y_pred = analyze_feature_binary(df, "心白")
    np.testing.assert_array_equal(y_true, [1, 0])
    np.testing.assert_array_equal(y_pred, [0, 1])


def test_collect_binary_vectors_builds_multihot_matrix():
    df = pd.DataFrame({"true": ["心白+基白", "背白"], "pred": ["心白", "背白+腹白"]})
    labels = ["心白", "基白", "背白", "腹白"]

    y_true, y_pred = collect_binary_vectors(df, labels, pred_col="pred")

    np.testing.assert_array_equal(y_true, [[1, 1, 0, 0], [0, 0, 1, 0]])
    np.testing.assert_array_equal(y_pred, [[1, 0, 0, 0], [0, 0, 1, 1]])


def test_hamming_accuracy_label_mean_weights_labels_equally():
    y_true = np.array([[1, 0], [1, 0], [1, 0], [1, 0]])
    y_pred = np.array([[1, 0], [1, 0], [1, 0], [0, 1]])  # label0: 3/4 right, label1: 3/4 right

    mean_acc, per_label = hamming_accuracy_label_mean(y_true, y_pred)

    assert mean_acc == 0.75
    assert per_label == [0.75, 0.75]


def test_labels_to_presence_strings_maps_binary_values():
    result = labels_to_presence_strings(np.array([0, 1, 1, 0]))
    assert list(result) == ["無", "有", "有", "無"]


def test_integrate_hierarchical_predictions_uses_layer2_when_layer1_correct():
    df_layer1 = pd.DataFrame({
        "img": ["a.jpg", "b.jpg"],
        "filename": ["a.jpg", "b.jpg"],
        "true": ["Colored", "Perfect Grain"],
        "pred": ["Colored", "Perfect Grain"],
    })
    true_map = {"a.jpg": "Spotted", "b.jpg": "Perfect Grain"}
    layer2_map = {"a.jpg": "Spotted"}  # only layer-1-correct "Colored" grains have a layer-2 result

    result = integrate_hierarchical_predictions(df_layer1, true_map, layer2_map)

    assert result.loc[result["filename"] == "a.jpg", "pred"].iloc[0] == "Spotted"
    assert result.loc[result["filename"] == "b.jpg", "pred"].iloc[0] == "Perfect Grain"


def test_integrate_hierarchical_predictions_falls_back_when_layer1_wrong():
    df_layer1 = pd.DataFrame({
        "img": ["a.jpg"], "filename": ["a.jpg"],
        "true": ["Colored"], "pred": ["Cracked"],  # layer-1 misrouted
    })
    true_map = {"a.jpg": "Colored"}
    layer2_map = {"a.jpg": "Spotted"}

    result = integrate_hierarchical_predictions(df_layer1, true_map, layer2_map)

    assert result["pred"].iloc[0] == "Cracked"
