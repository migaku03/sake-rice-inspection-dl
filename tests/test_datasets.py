"""Tests for sake_rice_inspection.datasets. Uses tiny synthetic images."""

import numpy as np
import torch
from PIL import Image

from sake_rice_inspection.datasets import (
    MultiLabelGrainDataset,
    apply_flip_augmentation,
    denormalize_for_display,
    preprocess_image,
    resize_and_pad,
)


def make_asymmetric_image(size=(20, 10)):
    """A gradient image so flips are distinguishable, saved as RGB."""
    w, h = size
    arr = np.zeros((h, w, 3), dtype=np.uint8)
    arr[:, :, 0] = np.linspace(0, 255, w, dtype=np.uint8)[None, :]
    return Image.fromarray(arr)


def test_apply_flip_augmentation_variants_are_distinct():
    image = make_asymmetric_image()
    original = np.array(image)
    h_flip = np.array(apply_flip_augmentation(image, 1))
    v_flip = np.array(apply_flip_augmentation(image, 2))
    both_flip = np.array(apply_flip_augmentation(image, 3))

    assert not np.array_equal(original, h_flip)
    np.testing.assert_array_equal(original[::-1, :], v_flip)
    np.testing.assert_array_equal(h_flip[::-1, :], both_flip)


def test_apply_flip_augmentation_zero_is_identity():
    image = make_asymmetric_image()
    result = apply_flip_augmentation(image, 0)
    np.testing.assert_array_equal(np.array(image), np.array(result))


def test_resize_and_pad_produces_target_square():
    image = make_asymmetric_image(size=(40, 20))
    padded = resize_and_pad(image, scale=1.0, out_size=64)
    assert padded.size == (64, 64)


def test_preprocess_image_returns_chw_float_array(tmp_path):
    image = make_asymmetric_image(size=(30, 20))
    path = tmp_path / "grain.png"
    image.save(path)

    result = preprocess_image(str(path), scale=1.0, img_size=64)

    assert result.shape == (3, 64, 64)
    assert result.dtype == np.float32


def test_denormalize_for_display_clips_to_unit_range():
    tensor_image = np.random.uniform(-5, 5, size=(3, 8, 8)).astype(np.float32)
    display = denormalize_for_display(tensor_image)

    assert display.shape == (8, 8, 3)
    assert display.min() >= 0.0
    assert display.max() <= 1.0


def test_multilabel_grain_dataset_without_augmentation(tmp_path):
    image = make_asymmetric_image()
    path = tmp_path / "grain.png"
    image.save(path)

    items = [(str(path), [1, 0, 0, 0])]
    ds = MultiLabelGrainDataset(items, scale=1.0, img_size=32, augment=False)

    assert len(ds) == 1
    x, y = ds[0]
    assert x.shape == (3, 32, 32)
    assert torch.equal(y, torch.tensor([1.0, 0.0, 0.0, 0.0]))


def test_multilabel_grain_dataset_with_augmentation_expands_4x(tmp_path):
    image = make_asymmetric_image()
    path = tmp_path / "grain.png"
    image.save(path)

    items = [(str(path), [1, 0, 0, 0]), (str(path), [0, 1, 0, 0])]
    ds = MultiLabelGrainDataset(items, scale=1.0, img_size=32, augment=True)

    assert len(ds) == 8
