"""Plotting utilities for training curves and confusion matrices."""

from typing import Dict, List, Optional

import numpy as np
from sklearn.metrics import auc, confusion_matrix, roc_curve


def plot_multilabel_metrics(y_true: np.ndarray, y_pred: np.ndarray, label_names: List[str], save_path: Optional[str] = None):
    """Draws a per-label binary confusion matrix panel.

    Args:
        y_true (np.ndarray): (N, num_labels) ground-truth multi-hot array.
        y_pred (np.ndarray): (N, num_labels) predicted multi-hot array.
        label_names (List[str]): Name for each label column.
        save_path (Optional[str]): If given, saves the figure to this path.

    Returns:
        matplotlib.figure.Figure: The created figure.
    """
    import matplotlib.pyplot as plt

    n_labels = len(label_names)
    fig, axes = plt.subplots(1, n_labels, figsize=(4 * n_labels, 3))
    if n_labels == 1:
        axes = [axes]

    for i, label_name in enumerate(label_names):
        cm = confusion_matrix(y_true[:, i], y_pred[:, i], labels=[0, 1])
        ax = axes[i]
        im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
        ax.figure.colorbar(im, ax=ax)
        ax.set(
            xticks=[0, 1], yticks=[0, 1],
            xticklabels=["No", "Yes"], yticklabels=["No", "Yes"],
            title=label_name, ylabel="True", xlabel="Predicted",
        )
        thresh = cm.max() / 2.0
        for ii in range(2):
            for jj in range(2):
                ax.text(jj, ii, format(cm[ii, jj], "d"), ha="center", va="center",
                         color="white" if cm[ii, jj] > thresh else "black")

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_multiclass_confusion(y_true_str: List[str], y_pred_str: List[str], save_path: Optional[str] = None):
    """Draws a confusion matrix over joined trait-combination labels.

    Args:
        y_true_str (List[str]): Ground-truth joined label strings.
        y_pred_str (List[str]): Predicted joined label strings.
        save_path (Optional[str]): If given, saves the figure to this path.

    Returns:
        matplotlib.figure.Figure: The created figure.
    """
    import matplotlib.pyplot as plt

    unique_labels = sorted(set(y_true_str) | set(y_pred_str))
    cm = confusion_matrix(y_true_str, y_pred_str, labels=unique_labels)

    fig, ax = plt.subplots(figsize=(max(8, len(unique_labels)), max(6, len(unique_labels) * 0.8)))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    ax.figure.colorbar(im, ax=ax)
    ax.set(
        xticks=np.arange(len(unique_labels)), yticks=np.arange(len(unique_labels)),
        xticklabels=unique_labels, yticklabels=unique_labels,
        title="Multiclass Confusion Matrix", ylabel="True Label", xlabel="Predicted Label",
    )
    import matplotlib.pyplot as _plt
    _plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    thresh = cm.max() / 2.0
    for i in range(len(unique_labels)):
        for j in range(len(unique_labels)):
            ax.text(j, i, format(cm[i, j], "d"), ha="center", va="center",
                     color="white" if cm[i, j] > thresh else "black")

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_curves(history: Dict[str, List[float]], save_path: Optional[str] = None):
    """Draws training loss / validation accuracy / validation F1 curves.

    Args:
        history (Dict[str, List[float]]): Keys `epoch`, `train_loss`,
            `val_acc`, `val_f1`, each a per-epoch list.
        save_path (Optional[str]): If given, saves the figure to this path.

    Returns:
        matplotlib.figure.Figure: The created figure.
    """
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(history["epoch"], history["train_loss"], label="train loss")
    ax.plot(history["epoch"], history["val_acc"], label="val acc")
    ax.plot(history["epoch"], history["val_f1"], label="val macro-F1")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Score")
    ax.legend()
    ax.grid(alpha=0.3)
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_roc_curve(y_true: np.ndarray, y_score: np.ndarray, save_path: Optional[str] = None, pos_label: int = 1, title: str = "ROC Curve") -> float:
    """Draws an ROC curve for a binary classification head.

    Args:
        y_true (np.ndarray): 0/1 ground-truth labels.
        y_score (np.ndarray): Predicted probability of the positive class.
        save_path (Optional[str]): If given, saves the figure to this path.
        pos_label (int): Which label value counts as positive.
        title (str): Plot title.

    Returns:
        float: The area under the ROC curve.
    """
    import matplotlib.pyplot as plt

    fpr, tpr, _ = roc_curve(y_true, y_score, pos_label=pos_label)
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, label=f"AUC = {roc_auc:.4f}")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", alpha=0.6)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(title)
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return roc_auc
