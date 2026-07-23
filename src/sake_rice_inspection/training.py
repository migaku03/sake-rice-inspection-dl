"""Training and evaluation loops shared across classification heads.

Provides both a single-label (softmax/argmax) and multi-label
(sigmoid/threshold) evaluation path, since the hierarchical system uses
softmax classifiers for most layers but a sigmoid multi-label classifier for
the "partially clouded" trait combinations.
"""

from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix, f1_score
from torch.utils.data import DataLoader


def train_one_epoch(model: nn.Module, loader: DataLoader, device: torch.device, optimizer, criterion) -> float:
    """Runs one training epoch.

    Args:
        model (nn.Module): The model to train (mutated in place).
        loader (DataLoader): Training data loader.
        device (torch.device): Compute device.
        optimizer: A torch optimizer over `model`'s trainable parameters.
        criterion: A loss function taking (logits, targets).

    Returns:
        float: The mean per-sample training loss for the epoch.
    """
    model.train()
    loss_sum = 0.0
    n = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        loss_sum += loss.item() * x.size(0)
        n += x.size(0)
    return loss_sum / max(1, n)


@torch.no_grad()
def eval_metrics(model: nn.Module, loader: DataLoader, device: torch.device, n_classes: int):
    """Evaluates a single-label (softmax) classifier.

    Args:
        model (nn.Module): The model to evaluate.
        loader (DataLoader): Validation data loader.
        device (torch.device): Compute device.
        n_classes (int): Number of classes.

    Returns:
        Tuple[float, float, np.ndarray, np.ndarray, np.ndarray]: accuracy,
        macro-F1, confusion matrix, true labels, and predicted labels.
    """
    model.eval()
    ys, ps = [], []
    for x, y in loader:
        x = x.to(device)
        logits = model(x)
        pred = torch.argmax(logits, 1).cpu().numpy()
        ys.append(y.numpy())
        ps.append(pred)

    y_true = np.concatenate(ys)
    y_pred = np.concatenate(ps)
    acc = (y_true == y_pred).mean()
    macro_f1 = f1_score(y_true, y_pred, average="macro")
    cm = confusion_matrix(y_true, y_pred, labels=list(range(n_classes)))
    return acc, macro_f1, cm, y_true, y_pred


@torch.no_grad()
def eval_metrics_multilabel(model: nn.Module, loader: DataLoader, device: torch.device, threshold: float = 0.5):
    """Evaluates a multi-label (sigmoid) classifier.

    Args:
        model (nn.Module): The model to evaluate.
        loader (DataLoader): Validation data loader.
        device (torch.device): Compute device.
        threshold (float): Probability threshold for a positive prediction.

    Returns:
        Tuple[float, float, float, np.ndarray, np.ndarray, np.ndarray]:
        exact-match accuracy (all labels correct), Hamming accuracy
        (per-label average), macro-F1, true labels, predicted labels, and
        predicted probabilities.
    """
    model.eval()
    all_y_true, all_y_pred, all_y_prob = [], [], []

    for x, y in loader:
        x = x.to(device)
        logits = model(x)
        probs = torch.sigmoid(logits).cpu().numpy()
        preds = (probs >= threshold).astype(int)

        all_y_true.append(y.numpy())
        all_y_pred.append(preds)
        all_y_prob.append(probs)

    y_true = np.concatenate(all_y_true)
    y_pred = np.concatenate(all_y_pred)
    y_prob = np.concatenate(all_y_prob)

    exact_match = (y_true == y_pred).all(axis=1).mean()
    hamming_acc = (y_true == y_pred).mean()
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)

    return exact_match, hamming_acc, macro_f1, y_true, y_pred, y_prob


def compute_pos_weight(labels: np.ndarray, device: torch.device) -> torch.Tensor:
    """Computes per-label positive class weights for `BCEWithLogitsLoss`.

    Args:
        labels (np.ndarray): (N, num_labels) multi-hot training labels.
        device (torch.device): Compute device for the returned tensor.

    Returns:
        torch.Tensor: Per-label weight, `negative_count / positive_count`.
    """
    pos_counts = labels.sum(axis=0)
    neg_counts = len(labels) - pos_counts
    return torch.tensor(neg_counts / (pos_counts + 1e-5), dtype=torch.float32, device=device)


def labels_to_str(labels_array: np.ndarray, label_names: List[str], empty_label: str = "なし") -> List[str]:
    """Converts multi-hot label rows to "+"-joined trait name strings.

    Args:
        labels_array (np.ndarray): (N, num_labels) multi-hot array.
        label_names (List[str]): Name for each label column.
        empty_label (str): String used when no label is active in a row.

    Returns:
        List[str]: One joined string per row.
    """
    result = []
    for labels in labels_array:
        active = [label_names[i] for i, v in enumerate(labels) if v == 1]
        result.append("+".join(active) if active else empty_label)
    return result
