"""Tests for sake_rice_inspection.domain_adaptation.

All tests use synthetic arrays; no real grain photographs are required.
"""

import numpy as np
import pytest

from sake_rice_inspection.domain_adaptation import (
    apply_domain_transform,
    compute_target_statistics,
    get_masked_stats,
    imread_safe,
    imwrite_safe,
    jitter_statistics,
    transform_directory,
)


def make_grain(mean, std, shape=(30, 30, 3), seed=0):
    """Builds a synthetic masked grain: a filled circle on a black background."""
    rng = np.random.default_rng(seed)
    image = np.zeros(shape, dtype=np.float32)
    mask = np.zeros(shape[:2], dtype=np.uint8)

    yy, xx = np.mgrid[0:shape[0], 0:shape[1]]
    center = (shape[0] // 2, shape[1] // 2)
    radius = min(shape[0], shape[1]) // 3
    circle = (yy - center[0]) ** 2 + (xx - center[1]) ** 2 <= radius ** 2
    mask[circle] = 255

    noise = rng.normal(mean, std, size=(circle.sum(), shape[2]))
    image[circle] = np.clip(noise, 0, 255)
    return image.astype(np.uint8), mask


def test_get_masked_stats_matches_expected_distribution():
    image, mask = make_grain(mean=[100, 120, 140], std=[10, 10, 10], seed=1)
    mean, std = get_masked_stats(image, mask)

    assert mean is not None
    np.testing.assert_allclose(mean, [100, 120, 140], atol=3)


def test_get_masked_stats_returns_none_for_empty_mask():
    image = np.zeros((10, 10, 3), dtype=np.uint8)
    mask = np.zeros((10, 10), dtype=np.uint8)

    mean, std = get_masked_stats(image, mask)

    assert mean is None
    assert std is None


def test_compute_target_statistics_averages_across_reference_set():
    pairs = [make_grain(mean=[100, 100, 100], std=[10, 10, 10], seed=i) for i in range(5)]
    mean, std = compute_target_statistics(pairs)

    np.testing.assert_allclose(mean, [100, 100, 100], atol=5)


def test_compute_target_statistics_raises_on_no_valid_pairs():
    empty_mask = np.zeros((10, 10), dtype=np.uint8)
    empty_image = np.zeros((10, 10, 3), dtype=np.uint8)

    with pytest.raises(ValueError):
        compute_target_statistics([(empty_image, empty_mask)])


def test_jitter_statistics_is_deterministic_with_seeded_rng():
    target_mean = np.array([100.0, 100.0, 100.0])
    target_std = np.array([20.0, 20.0, 20.0])

    m1, s1 = jitter_statistics(target_mean, target_std, rng=np.random.default_rng(42))
    m2, s2 = jitter_statistics(target_mean, target_std, rng=np.random.default_rng(42))

    np.testing.assert_array_equal(m1, m2)
    np.testing.assert_array_equal(s1, s2)


def test_jitter_statistics_clips_std_above_minimum():
    target_mean = np.array([0.0])
    target_std = np.array([0.0])

    _, jittered_std = jitter_statistics(target_mean, target_std, rng=np.random.default_rng(0))

    assert (jittered_std >= 0.1).all()


def test_apply_domain_transform_shifts_masked_region_toward_target():
    image, mask = make_grain(mean=[50, 50, 50], std=[5, 5, 5], seed=2)
    target_mean = np.array([200.0, 150.0, 100.0])
    target_std = np.array([5.0, 5.0, 5.0])

    result = apply_domain_transform(image, mask, target_mean, target_std)

    grain_pixels = result[mask > 0]
    np.testing.assert_allclose(grain_pixels.mean(axis=0), target_mean, atol=10)


def test_apply_domain_transform_zeros_background():
    image, mask = make_grain(mean=[50, 50, 50], std=[5, 5, 5], seed=3)
    target_mean = np.array([200.0, 150.0, 100.0])
    target_std = np.array([5.0, 5.0, 5.0])

    result = apply_domain_transform(image, mask, target_mean, target_std)

    assert result[mask == 0].sum() == 0


def test_apply_domain_transform_raises_on_empty_mask():
    image = np.zeros((10, 10, 3), dtype=np.uint8)
    mask = np.zeros((10, 10), dtype=np.uint8)

    with pytest.raises(ValueError):
        apply_domain_transform(image, mask, np.array([100.0]), np.array([10.0]))


def test_imread_imwrite_safe_roundtrip_with_non_ascii_path(tmp_path):
    image = np.full((5, 5, 3), 128, dtype=np.uint8)
    path = tmp_path / "基白_サンプル.png"

    assert imwrite_safe(path, image)
    loaded = imread_safe(path)

    assert loaded is not None
    np.testing.assert_array_equal(loaded, image)


def test_imread_safe_returns_none_for_missing_file(tmp_path):
    assert imread_safe(tmp_path / "does_not_exist.png") is None


def test_transform_directory_writes_one_output_per_matched_pair(tmp_path):
    import cv2

    source_dir = tmp_path / "source"
    mask_dir = tmp_path / "masks"
    output_dir = tmp_path / "out"
    source_dir.mkdir()
    mask_dir.mkdir()

    image, mask = make_grain(mean=[80, 90, 100], std=[8, 8, 8], seed=4)
    cv2.imwrite(str(source_dir / "grain_0001.jpg"), image)
    cv2.imwrite(str(mask_dir / "grain_0001_mask.png"), mask)
    # Second source image has no matching mask and should be skipped.
    cv2.imwrite(str(source_dir / "grain_0002.jpg"), image)

    saved_count, failed = transform_directory(
        source_dir, mask_dir,
        target_mean=np.array([150.0, 150.0, 150.0]),
        target_std=np.array([10.0, 10.0, 10.0]),
        output_dir=output_dir,
        jitter=True,
        rng=np.random.default_rng(0),
    )

    assert saved_count == 1
    assert failed == []
    assert (output_dir / "grain_0001.jpg").exists()
    assert not (output_dir / "grain_0002.jpg").exists()
