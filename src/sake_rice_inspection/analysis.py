"""Cross-model, cross-layer evaluation of exported predictions.

Consumes the `all_predictions.xlsx` files produced by the training
notebooks and compares the hierarchical (multi-stage) classification
approach against a single flat 18-class classifier, including the
domain-logic correction for the multi-label "partially clouded" traits.
"""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


def split_label(label: Any) -> List[str]:
    """Splits a "+"-joined trait-combination label into its components.

    Args:
        label (Any): A label string (e.g. "心白+基白"), or NaN.

    Returns:
        List[str]: The individual trait names, or an empty list for NaN.
    """
    if pd.isna(label):
        return []
    return [x.strip() for x in str(label).split("+") if x.strip()]


def apply_pw_correction(
    df: pd.DataFrame, allowed_labels: List[str], true_col: Optional[str] = None, pred_col: Optional[str] = None,
) -> pd.DataFrame:
    """Remaps physically-impossible multi-label predictions to the closest allowed one.

    For predictions not in `allowed_labels`, scores each allowed combination
    as (mean probability of its components) minus a small penalty for the
    mean probability of components *not* in it, and picks the best-scoring
    allowed label. Requires the dataframe to have per-component probability
    columns whose names contain each component's name.

    Args:
        df (pd.DataFrame): Predictions dataframe, with per-component score
            columns (as written by `reporting.create_unified_prediction_excel`).
        allowed_labels (List[str]): The physically valid label strings.
        true_col (Optional[str]): Ground-truth column name; defaults to the 3rd column.
        pred_col (Optional[str]): Predicted-label column name; defaults to the 4th column.

    Returns:
        pd.DataFrame: A copy of `df` with an added `pred_corrected` column.
    """
    df = df.copy()
    true_col = true_col or df.columns[2]
    pred_col = pred_col or df.columns[3]

    components = sorted({c for lbl in allowed_labels for c in split_label(lbl)})
    prob_cols = {
        c: col for c in components for col in df.columns
        if c in str(col) and pd.api.types.is_numeric_dtype(df[col])
    }

    def _map_invalid(row):
        pred = row[pred_col]
        if pred in allowed_labels:
            return pred
        if not prob_cols:
            return allowed_labels[0]

        scores = {}
        for lbl in allowed_labels:
            comps = split_label(lbl)
            vals = [row[prob_cols[c]] for c in comps if c in prob_cols]
            if vals:
                non_vals = [row[prob_cols[c]] for c in prob_cols if c not in comps]
                scores[lbl] = np.mean(vals) - (0.05 * np.mean(non_vals) if non_vals else 0)
        return max(scores, key=scores.get) if scores else allowed_labels[0]

    df["pred_corrected"] = df.apply(_map_invalid, axis=1)
    return df


def compute_classification_metrics(true, pred, labels: Optional[List[str]] = None, average: str = "macro") -> Dict[str, Any]:
    """Computes accuracy/precision/recall/F1 for a classification result.

    Args:
        true: Ground-truth labels (array-like of strings).
        pred: Predicted labels (array-like of strings).
        labels (Optional[List[str]]): Restrict precision/recall/F1 averaging
            to these classes; defaults to the union of `true` and `pred`.
        average (str): Averaging strategy passed to sklearn (default "macro").

    Returns:
        Dict[str, Any]: `accuracy`, `precision`, `recall`, `f1`, `eval_count`,
        `correct_count`, and `n_classes` (number of classes averaged over).
    """
    if labels is None:
        labels = sorted(set(true) | set(pred))

    accuracy = accuracy_score(true, pred)
    precision = precision_score(true, pred, labels=labels, average=average, zero_division=0)
    recall = recall_score(true, pred, labels=labels, average=average, zero_division=0)
    f1 = f1_score(true, pred, labels=labels, average=average, zero_division=0)
    correct_count = int(np.sum(np.array(true) == np.array(pred)))

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "eval_count": len(true),
        "correct_count": correct_count,
        "n_classes": len(labels),
    }


def format_metrics_markdown(title: str, metrics: Dict[str, Any]) -> List[str]:
    """Formats a metrics dict as Markdown report lines.

    Args:
        title (str): Section heading.
        metrics (Dict[str, Any]): Output of `compute_classification_metrics`.

    Returns:
        List[str]: Lines to append to a report (join with "\\n").
    """
    return [
        f"\n## {title}",
        f"- eval_count: {metrics['eval_count']}",
        f"- correct_count: {metrics['correct_count']}",
        f"- Accuracy: {metrics['accuracy']:.4f}",
        f"- Precision: {metrics['precision']:.4f}",
        f"- Recall: {metrics['recall']:.4f}",
        f"- F1-Score: {metrics['f1']:.4f}",
    ]


def order_labels(labels: List[str], desired_order: Optional[List[str]] = None) -> List[str]:
    """Orders a label list, placing entries from `desired_order` first.

    Args:
        labels (List[str]): The labels to order.
        desired_order (Optional[List[str]]): Preferred display order; labels
            not listed here are appended afterward, in their original order.

    Returns:
        List[str]: The ordered labels.
    """
    if not desired_order:
        return list(labels)
    ordered = [lbl for lbl in desired_order if lbl in labels]
    ordered += [lbl for lbl in labels if lbl not in desired_order]
    return ordered


def plot_confusion_matrix(
    true, pred, labels: Optional[List[str]] = None, title: str = "", save_path: Optional[str] = None,
    desired_order: Optional[List[str]] = None,
):
    """Draws a confusion matrix sized and annotated for its class count.

    Args:
        true: Ground-truth labels.
        pred: Predicted labels.
        labels (Optional[List[str]]): Classes to include; defaults to the
            union of `true` and `pred`.
        title (str): Plot title.
        save_path (Optional[str]): If given, saves the figure to this path.
        desired_order (Optional[List[str]]): Preferred label display order
            (see `order_labels`).

    Returns:
        matplotlib.figure.Figure: The created figure.
    """
    import matplotlib.pyplot as plt
    import seaborn as sns

    if labels is None:
        labels = sorted(set(true) | set(pred))
    labels = order_labels(labels, desired_order)
    n_classes = len(labels)

    fig_width = max(8, n_classes * 0.7)
    fig_height = max(6, n_classes * 0.6)
    if n_classes <= 2:
        annot_size, tick_size = 24, 16
    elif n_classes <= 7:
        annot_size, tick_size = 14, 12
    elif n_classes <= 12:
        annot_size, tick_size = 12, 12
    else:
        annot_size, tick_size = 14, 14

    cm = confusion_matrix(true, pred, labels=labels)

    fig = plt.figure(figsize=(fig_width, fig_height))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels,
        annot_kws={"size": annot_size}, cbar=(n_classes <= 12),
    )
    plt.ylabel("True Label", fontsize=tick_size + 2)
    plt.xlabel("Predicted Label", fontsize=tick_size + 2)
    plt.title(title, fontsize=tick_size + 4)
    plt.xticks(rotation=45 if n_classes > 2 else 0, fontsize=tick_size, ha="right" if n_classes > 2 else "center")
    plt.yticks(rotation=0, fontsize=tick_size)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300)
    return fig


def analyze_feature_binary(df: pd.DataFrame, feature_name: str, true_col: str = "true", pred_col: str = "pred") -> Tuple[np.ndarray, np.ndarray]:
    """Binarizes true/predicted labels by whether a given trait is present.

    Args:
        df (pd.DataFrame): Predictions dataframe with joined-label columns.
        feature_name (str): Trait name to check for (substring match).
        true_col (str): Ground-truth column name.
        pred_col (str): Predicted-label column name.

    Returns:
        Tuple[np.ndarray, np.ndarray]: 0/1 arrays for true and predicted presence.
    """
    def has_feature(label):
        if pd.isna(label):
            return 0
        return 1 if feature_name in str(label) else 0

    y_true = df[true_col].apply(has_feature).values
    y_pred = df[pred_col].apply(has_feature).values
    return y_true, y_pred


def collect_binary_vectors(df: pd.DataFrame, labels: List[str], pred_col: str, true_col: Optional[str] = None) -> Tuple[np.ndarray, np.ndarray]:
    """Converts joined-label columns into multi-hot binary matrices.

    Args:
        df (pd.DataFrame): Predictions dataframe.
        labels (List[str]): The trait names to extract, in output column order.
        pred_col (str): Predicted joined-label column name.
        true_col (Optional[str]): Ground-truth joined-label column name;
            defaults to a `true` column if present, else the 3rd column.

    Returns:
        Tuple[np.ndarray, np.ndarray]: (N, len(labels)) 0/1 arrays.
    """
    if true_col is None:
        true_col = "true" if "true" in df.columns else df.columns[2]

    label_to_idx = {label: i for i, label in enumerate(labels)}
    y_true = np.zeros((len(df), len(labels)), dtype=int)
    y_pred = np.zeros((len(df), len(labels)), dtype=int)

    for i, (true_label, pred_label) in enumerate(zip(df[true_col], df[pred_col])):
        for part in split_label(true_label):
            if part in label_to_idx:
                y_true[i, label_to_idx[part]] = 1
        for part in split_label(pred_label):
            if part in label_to_idx:
                y_pred[i, label_to_idx[part]] = 1

    return y_true, y_pred


def hamming_accuracy_label_mean(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[float, List[float]]:
    """Computes per-label accuracy and its mean across labels.

    Unlike element-wise Hamming accuracy, this weights every label equally
    regardless of how many positive examples it has.

    Args:
        y_true (np.ndarray): (N, num_labels) ground-truth binary matrix.
        y_pred (np.ndarray): (N, num_labels) predicted binary matrix.

    Returns:
        Tuple[float, List[float]]: The mean per-label accuracy, and the
        per-label accuracy list.
    """
    per_label_acc = [float(np.mean(y_true[:, i] == y_pred[:, i])) for i in range(y_true.shape[1])]
    return float(np.mean(per_label_acc)), per_label_acc


def labels_to_presence_strings(values: np.ndarray, absent: str = "無", present: str = "有") -> np.ndarray:
    """Converts a 0/1 array to human-readable presence strings.

    Args:
        values (np.ndarray): 0/1 array.
        absent (str): String for 0.
        present (str): String for 1.

    Returns:
        np.ndarray: Array of `absent`/`present` strings.
    """
    return np.array([absent if v == 0 else present for v in values])


def normalize_filename_key(filename: Any) -> str:
    """Normalizes a workbook filename for cross-file matching.

    `all_predictions.xlsx` stores each grain's path relative to its crop
    directory, using whatever OS generated it (backslashes on Windows,
    forward slashes on Linux/macOS). Comparing or joining filenames across
    workbooks produced on different platforms requires a common separator.

    Args:
        filename (Any): A filename/path value, typically from a workbook cell.

    Returns:
        str: The filename with backslashes normalized to forward slashes.
    """
    return str(filename).replace("\\", "/")


def integrate_hierarchical_predictions(df_layer1: pd.DataFrame, true_map: Dict[str, Any], layer2_map: Dict[str, Any]) -> pd.DataFrame:
    """Combines layer-1 (coarse) and layer-2 (fine) predictions into one result.

    A grain's final prediction is its layer-2 prediction if layer 1 correctly
    routed it to a category with a layer-2 model (i.e. layer-1's prediction
    matched the ground truth and a layer-2 result exists for it); otherwise
    it falls back to the layer-1 prediction itself.

    Args:
        df_layer1 (pd.DataFrame): Layer-1 predictions; columns 2/3 (0-indexed:
            1/2) are filename, true label, layer-1 predicted label.
        true_map (Dict[str, Any]): filename -> final ground-truth label.
        layer2_map (Dict[str, Any]): filename -> layer-2 predicted label
            (only present for grains routed to a layer-2 model).

    Returns:
        pd.DataFrame: Columns `filename`, `true`, `pred`.
    """
    rows = []
    for _, row in df_layer1.iterrows():
        filename = normalize_filename_key(row.iloc[1])
        layer1_pred = row.iloc[3]
        final_true = true_map.get(filename, row.iloc[2])
        final_pred = layer2_map[filename] if (row.iloc[2] == layer1_pred and filename in layer2_map) else layer1_pred
        rows.append({"filename": filename, "true": final_true, "pred": final_pred})
    return pd.DataFrame(rows)
