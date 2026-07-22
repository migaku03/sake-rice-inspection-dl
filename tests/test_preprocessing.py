"""Tests for sake_rice_inspection.preprocessing.

All tests use synthetic arrays; no real grain photographs are required.
"""

import numpy as np
import pytest
from skimage.measure import regionprops

from sake_rice_inspection.preprocessing import (
    extract_grain,
    filter_mask_by_area,
    process_directory,
    process_image_to_grains,
    render_overlay,
)


def make_two_blob_mask(shape=(60, 60)):
    """Builds a synthetic instance mask with one large and one tiny blob."""
    mask = np.zeros(shape, dtype=np.int32)
    mask[10:30, 10:35] = 1  # large blob: 20*25 = 500 px
    mask[45:48, 45:48] = 2  # tiny blob: 3*3 = 9 px
    return mask


class FakeCellposeModel:
    """Stands in for cellpose.models.CellposeModel in tests."""

    def __init__(self, mask):
        self._mask = mask

    def eval(self, image, channels, diameter):
        return self._mask, None, None


def test_filter_mask_by_area_drops_small_regions():
    mask = make_two_blob_mask()
    filtered, df = filter_mask_by_area(mask, min_area=100)

    assert filtered.max() == 1
    assert len(df) == 1
    assert df.iloc[0]["area"] == 500


def test_filter_mask_by_area_keeps_all_when_threshold_low():
    mask = make_two_blob_mask()
    filtered, df = filter_mask_by_area(mask, min_area=1)

    assert filtered.max() == 2
    assert len(df) == 2


def test_extract_grain_returns_masked_crop_and_binary_mask():
    mask = make_two_blob_mask()
    image = np.zeros((*mask.shape, 3), dtype=np.uint8)
    image[mask == 1] = [200, 150, 100]

    props = regionprops(mask, intensity_image=image)
    prop = next(p for p in props if p.label == 1)

    grain_img, grain_mask = extract_grain(image, prop, mask, margin=5)

    assert grain_img.ndim == 3
    assert grain_img.shape[:2] == grain_mask.shape
    assert set(np.unique(grain_mask)).issubset({0, 255})
    # Background pixels outside the rotated/trimmed grain must be zeroed out.
    assert grain_img[grain_mask == 0].sum() == 0


def test_render_overlay_matches_image_shape():
    mask = make_two_blob_mask()
    image = np.zeros((*mask.shape, 3), dtype=np.uint8)

    overlay = render_overlay(image, mask)

    assert overlay.shape[:2] == mask.shape


def test_process_image_to_grains_uses_area_threshold(tmp_path):
    mask = make_two_blob_mask()
    image = np.zeros((*mask.shape, 3), dtype=np.uint8)
    image[mask == 1] = [200, 150, 100]

    from skimage import io as skio
    img_path = tmp_path / "sample.png"
    skio.imsave(img_path, image)

    model = FakeCellposeModel(mask)
    grains = process_image_to_grains(img_path, model, min_area=100, diameter=50, margin=5)

    assert len(grains) == 1
    assert grains[0]["grain_id"] == 1


def test_process_directory_writes_expected_layout(tmp_path):
    mask = make_two_blob_mask()
    image = np.zeros((*mask.shape, 3), dtype=np.uint8)
    image[mask == 1] = [200, 150, 100]

    curated_root = tmp_path / "curated"
    label_dir = curated_root / "sample_label"
    label_dir.mkdir(parents=True)

    from skimage import io as skio
    skio.imsave(label_dir / "photo1.png", image)

    output_root = tmp_path / "crops"
    mask_output_root = tmp_path / "masks"
    crop_mask_output_root = tmp_path / "crop_masks"

    model = FakeCellposeModel(mask)
    stats = process_directory(
        curated_root, output_root, mask_output_root, crop_mask_output_root,
        model, grouped_by_image=False, min_area=100,
    )

    assert stats == {"sample_label": 1}
    assert (output_root / "sample_label" / "photo1_grain_0001.jpg").exists()
    assert (crop_mask_output_root / "sample_label" / "photo1_grain_0001_mask.png").exists()
    assert (mask_output_root / "photo1_mask.png").exists()


def test_process_directory_groups_by_image_when_requested(tmp_path):
    mask = make_two_blob_mask()
    image = np.zeros((*mask.shape, 3), dtype=np.uint8)
    image[mask == 1] = [200, 150, 100]

    curated_root = tmp_path / "curated"
    label_dir = curated_root / "composite_label"
    label_dir.mkdir(parents=True)

    from skimage import io as skio
    skio.imsave(label_dir / "photoA.png", image)

    output_root = tmp_path / "crops"
    mask_output_root = tmp_path / "masks"
    crop_mask_output_root = tmp_path / "crop_masks"

    model = FakeCellposeModel(mask)
    process_directory(
        curated_root, output_root, mask_output_root, crop_mask_output_root,
        model, grouped_by_image=True, min_area=100,
    )

    assert (output_root / "composite_label" / "photoA" / "photoA_grain_0001.jpg").exists()
