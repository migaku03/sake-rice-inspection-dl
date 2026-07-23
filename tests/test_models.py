"""Tests for sake_rice_inspection.models.

Uses pretrained=False throughout so tests run offline without fetching
ImageNet weights.
"""

import pytest
import torch

from sake_rice_inspection.models import build_model


def test_build_model_resnet_head_matches_num_classes():
    model = build_model("resnet18", num_classes=4, pretrained=False)
    x = torch.randn(2, 3, 224, 224)
    out = model(x)
    assert out.shape == (2, 4)


def test_build_model_efficientnet_head_matches_num_classes():
    model = build_model("efficientnet_b0", num_classes=3, pretrained=False)
    x = torch.randn(2, 3, 224, 224)
    out = model(x)
    assert out.shape == (2, 3)


def test_build_model_freeze_backbone_disables_grad():
    model = build_model("resnet18", num_classes=4, freeze_backbone=True, pretrained=False)
    backbone_params = list(model.parameters())[:-2]  # all but the new fc layer's weight/bias
    assert all(not p.requires_grad for p in backbone_params)
    assert model.fc.weight.requires_grad


def test_build_model_no_freeze_keeps_grad_enabled():
    model = build_model("resnet18", num_classes=4, freeze_backbone=False, pretrained=False)
    assert all(p.requires_grad for p in model.parameters())


def test_build_model_unsupported_architecture_raises():
    with pytest.raises(ValueError):
        build_model("not_a_real_arch", num_classes=4, pretrained=False)
