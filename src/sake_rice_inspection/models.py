"""Backbone model construction for grain classification.

Supports the architectures explored during the research phase (several
EfficientNet and ResNet variants, plus ViT), swapping in a fresh linear head
sized to the number of output labels.
"""

import torch.nn as nn
from torchvision import models

SUPPORTED_ARCHITECTURES = (
    "efficientnet_b0", "efficientnet_b1", "efficientnet_b2", "efficientnet_b4",
    "resnet18", "resnet34", "resnet50", "resnet101",
    "vit_b_16", "vit_b_32",
)

_EFFICIENTNET_BUILDERS = {
    "efficientnet_b0": (models.efficientnet_b0, models.EfficientNet_B0_Weights.IMAGENET1K_V1),
    "efficientnet_b1": (models.efficientnet_b1, models.EfficientNet_B1_Weights.IMAGENET1K_V2),
    "efficientnet_b2": (models.efficientnet_b2, models.EfficientNet_B2_Weights.IMAGENET1K_V1),
    "efficientnet_b4": (models.efficientnet_b4, models.EfficientNet_B4_Weights.IMAGENET1K_V1),
}

_RESNET_BUILDERS = {
    "resnet18": (models.resnet18, models.ResNet18_Weights.IMAGENET1K_V1),
    "resnet34": (models.resnet34, models.ResNet34_Weights.IMAGENET1K_V1),
    "resnet50": (models.resnet50, models.ResNet50_Weights.IMAGENET1K_V1),
    "resnet101": (models.resnet101, models.ResNet101_Weights.IMAGENET1K_V1),
}

_VIT_BUILDERS = {
    "vit_b_16": (models.vit_b_16, models.ViT_B_16_Weights.IMAGENET1K_V1),
    "vit_b_32": (models.vit_b_32, models.ViT_B_32_Weights.IMAGENET1K_V1),
}


def build_model(arch: str, num_classes: int, freeze_backbone: bool = True, pretrained: bool = True) -> nn.Module:
    """Builds a classification model with a fresh head for `num_classes` outputs.

    Args:
        arch (str): One of `SUPPORTED_ARCHITECTURES`.
        num_classes (int): Number of output logits (one per label, for
            multi-label sigmoid classification, or one per class for softmax).
        freeze_backbone (bool): If True, backbone parameters are frozen
            (`requires_grad=False`), leaving only the new head trainable.
            Typically unfrozen later for a fine-tuning phase.
        pretrained (bool): If True, loads ImageNet-pretrained weights.
            Set to False to build an architecture without a network fetch
            (e.g. in tests).

    Returns:
        nn.Module: The constructed model, with backbone freezing already applied.

    Raises:
        ValueError: If `arch` is not one of `SUPPORTED_ARCHITECTURES`.
    """
    if arch in _EFFICIENTNET_BUILDERS:
        builder, weights = _EFFICIENTNET_BUILDERS[arch]
        model = builder(weights=weights if pretrained else None)
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features, num_classes)
        backbone = model.features
    elif arch in _RESNET_BUILDERS:
        builder, weights = _RESNET_BUILDERS[arch]
        model = builder(weights=weights if pretrained else None)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
        backbone = nn.Sequential(*list(model.children())[:-1])
    elif arch in _VIT_BUILDERS:
        builder, weights = _VIT_BUILDERS[arch]
        model = builder(weights=weights if pretrained else None)
        in_features = model.heads.head.in_features
        model.heads.head = nn.Linear(in_features, num_classes)
        backbone = nn.Sequential(model.conv_proj, model.encoder)
    else:
        raise ValueError(f"arch must be one of {SUPPORTED_ARCHITECTURES}, got {arch!r}")

    if freeze_backbone:
        for p in backbone.parameters():
            p.requires_grad = False

    return model
