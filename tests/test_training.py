"""Tests for sake_rice_inspection.training.

Uses a tiny dummy linear model on synthetic tensors instead of a real
backbone, so tests run fast and fully offline.
"""

import numpy as np
import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from sake_rice_inspection.training import (
    compute_pos_weight,
    eval_metrics,
    eval_metrics_multilabel,
    labels_to_str,
    train_one_epoch,
)

IMG_SHAPE = (3, 8, 8)


def make_dummy_model(num_outputs):
    return nn.Sequential(nn.Flatten(), nn.Linear(int(np.prod(IMG_SHAPE)), num_outputs))


def make_classification_loader(n_samples, num_classes, batch_size=4, seed=0):
    rng = np.random.default_rng(seed)
    x = torch.tensor(rng.normal(size=(n_samples, *IMG_SHAPE)), dtype=torch.float32)
    y = torch.tensor(rng.integers(0, num_classes, size=n_samples), dtype=torch.long)
    return DataLoader(TensorDataset(x, y), batch_size=batch_size)


def make_multilabel_loader(n_samples, num_labels, batch_size=4, seed=0):
    rng = np.random.default_rng(seed)
    x = torch.tensor(rng.normal(size=(n_samples, *IMG_SHAPE)), dtype=torch.float32)
    y = torch.tensor(rng.integers(0, 2, size=(n_samples, num_labels)), dtype=torch.float32)
    return DataLoader(TensorDataset(x, y), batch_size=batch_size)


def test_train_one_epoch_reduces_loss_over_many_steps():
    torch.manual_seed(0)
    model = make_dummy_model(num_outputs=3)
    loader = make_classification_loader(n_samples=64, num_classes=3)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()

    first_loss = train_one_epoch(model, loader, torch.device("cpu"), optimizer, criterion)
    for _ in range(20):
        last_loss = train_one_epoch(model, loader, torch.device("cpu"), optimizer, criterion)

    assert last_loss < first_loss


def test_eval_metrics_perfect_predictions_give_full_accuracy():
    model = make_dummy_model(num_outputs=2)
    with torch.no_grad():
        # Force the model to strongly separate two known inputs.
        model[1].bias.zero_()
        model[1].weight.zero_()

    x = torch.zeros(4, *IMG_SHAPE)
    y = torch.zeros(4, dtype=torch.long)  # all class 0
    loader = DataLoader(TensorDataset(x, y), batch_size=4)

    acc, macro_f1, cm, y_true, y_pred = eval_metrics(model, loader, torch.device("cpu"), n_classes=2)

    assert acc == 1.0
    np.testing.assert_array_equal(y_pred, y_true)


def test_eval_metrics_multilabel_perfect_predictions():
    model = make_dummy_model(num_outputs=3)
    with torch.no_grad():
        model[1].weight.zero_()
        model[1].bias.fill_(10.0)  # sigmoid(10) ~= 1 for every label

    x = torch.zeros(4, *IMG_SHAPE)
    y = torch.ones(4, 3)
    loader = DataLoader(TensorDataset(x, y), batch_size=4)

    exact_match, hamming_acc, macro_f1, y_true, y_pred, y_prob = eval_metrics_multilabel(
        model, loader, torch.device("cpu")
    )

    assert exact_match == 1.0
    assert hamming_acc == 1.0
    np.testing.assert_array_equal(y_pred, y_true)


def test_compute_pos_weight_balances_rare_positive_label():
    labels = np.array([[1, 0], [0, 0], [0, 0], [0, 0]])  # label 0: 1/3, label 1: 0/4
    weight = compute_pos_weight(labels, torch.device("cpu"))

    assert weight[0] == pytest.approx(3.0, rel=1e-3)
    assert weight[1] > 1000  # near-zero positive count -> huge weight


def test_labels_to_str_joins_active_labels():
    labels = np.array([[1, 0, 1], [0, 0, 0]])
    result = labels_to_str(labels, ["A", "B", "C"], empty_label="None")

    assert result == ["A+C", "None"]
