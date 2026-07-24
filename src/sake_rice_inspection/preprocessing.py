"""Grain segmentation and extraction pipeline.

This module segments individual sake rice grains out of a bulk photograph
and extracts each one as a rotation-corrected, background-masked crop,
ready for downstream classification.

It uses Cellpose for instance segmentation and scikit-image for region
property analysis, area filtering, and orientation-based rotation
correction. The core per-grain logic mirrors `preprocess_grain` in
`sake_rice_inspection_system.py`; this module additionally provides the
directory-level batch pipeline used to build the training crops.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from PIL import Image
from skimage import io as skio
from skimage.color import label2rgb
from skimage.measure import regionprops, regionprops_table
from skimage.transform import rotate

logger = logging.getLogger(__name__)

REGION_PROPERTIES = [
    "label", "area", "perimeter", "eccentricity", "orientation",
    "major_axis_length", "minor_axis_length", "centroid", "bbox",
]

DEFAULT_IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".tif", ".tiff")


def filter_mask_by_area(
    mask: np.ndarray, min_area: int, intensity_image: np.ndarray = None
) -> Tuple[np.ndarray, pd.DataFrame]:
    """Removes small regions from a segmentation mask and relabels the rest.

    Args:
        mask (np.ndarray): Instance segmentation mask (e.g. Cellpose output).
        min_area (int): Minimum area (in pixels) for a region to be kept.
        intensity_image (np.ndarray): Original image, used to populate
            intensity-based region properties. Optional.

    Returns:
        Tuple[np.ndarray, pd.DataFrame]: The relabeled, filtered mask and a
        DataFrame of region properties for the kept regions.
    """
    regions = regionprops(mask, intensity_image=intensity_image)
    valid_regions = [r for r in regions if r.area >= min_area]

    filtered_mask = np.zeros_like(mask)
    for new_label, region in enumerate(valid_regions, start=1):
        coords = region.coords
        filtered_mask[coords[:, 0], coords[:, 1]] = new_label

    df = pd.DataFrame(
        regionprops_table(filtered_mask, properties=REGION_PROPERTIES, intensity_image=intensity_image)
    )
    return filtered_mask, df


def extract_grain(
    image: np.ndarray, prop: Any, mask: np.ndarray, margin: int = 20
) -> Tuple[np.ndarray, np.ndarray]:
    """Crops, rotates, and masks a single grain from the source image.

    The grain is rotated so that its major axis is vertical, then trimmed
    to its rotated bounding box and masked to remove background pixels.

    Args:
        image (np.ndarray): The full source image.
        prop (Any): A skimage RegionProperties object for this grain.
        mask (np.ndarray): The full instance segmentation mask.
        margin (int): Pixel margin added around the bounding box before rotation.

    Returns:
        Tuple[np.ndarray, np.ndarray]: The masked, rotated grain image (uint8)
        and its corresponding binary mask (uint8, values 0/255).
    """
    label_id = prop.label

    y1, x1, y2, x2 = prop.bbox
    y1 = max(y1 - margin, 0)
    x1 = max(x1 - margin, 0)
    y2 = min(y2 + margin, mask.shape[0])
    x2 = min(x2 + margin, mask.shape[1])

    img_crop = image[y1:y2, x1:x2]
    label_mask = (mask == label_id).astype(np.uint8)
    mask_crop = label_mask[y1:y2, x1:x2]

    angle_deg = -np.rad2deg(prop.orientation)
    img_rot = rotate(img_crop, angle=angle_deg, resize=True, order=1, preserve_range=True)
    mask_rot = rotate(mask_crop, angle=angle_deg, resize=True, order=0, preserve_range=True)
    mask_rot = (mask_rot > 0.5).astype(np.uint8)

    ys, xs = np.where(mask_rot > 0)
    if len(ys) == 0:
        out = np.clip(img_crop, 0, 255).astype(np.uint8)
        return out, np.zeros(out.shape[:2], dtype=np.uint8)

    min_y, max_y = ys.min(), ys.max() + 1
    min_x, max_x = xs.min(), xs.max() + 1
    img_trim = img_rot[min_y:max_y, min_x:max_x]
    mask_trim = mask_rot[min_y:max_y, min_x:max_x]

    if img_trim.ndim == 3:
        img_masked = img_trim * mask_trim[..., None]
    else:
        img_masked = img_trim * mask_trim

    if np.issubdtype(image.dtype, np.integer):
        out = np.clip(img_masked, 0, 255).astype(np.uint8)
    else:
        out = np.clip(img_masked * 255, 0, 255).astype(np.uint8)

    if out.ndim == 3:
        out = out[:, :, :3]

    return out, (mask_trim * 255).astype(np.uint8)


def segment_image(
    model: Any, image: np.ndarray, min_area: int = 1500, diameter: int = 50
) -> Tuple[np.ndarray, pd.DataFrame]:
    """Runs Cellpose segmentation on an image and filters by area.

    Args:
        model (Any): An initialized `cellpose.models.CellposeModel` instance.
        image (np.ndarray): The source image to segment.
        min_area (int): Minimum region area (pixels) to keep.
        diameter (int): Expected object diameter, passed to Cellpose.

    Returns:
        Tuple[np.ndarray, pd.DataFrame]: The filtered instance mask and a
        DataFrame of region properties.
    """
    masks, _, _ = model.eval(image, channels=[0, 0], diameter=diameter)
    return filter_mask_by_area(masks, min_area, intensity_image=image)


def render_overlay(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Builds an RGB overlay visualizing the segmentation mask on the image.

    Args:
        image (np.ndarray): The source image.
        mask (np.ndarray): The instance segmentation mask.

    Returns:
        np.ndarray: An RGB image with each region tinted for visualization.
    """
    return label2rgb(mask, image=image, bg_label=0, alpha=0.5)


def process_image_to_grains(
    image_path: Path,
    model: Any,
    min_area: int = 1500,
    diameter: int = 50,
    margin: int = 20,
) -> List[Dict[str, Any]]:
    """Segments an image and extracts every grain above the area threshold.

    Args:
        image_path (Path): Path to the source image.
        model (Any): An initialized `cellpose.models.CellposeModel` instance.
        min_area (int): Minimum region area (pixels) to keep.
        diameter (int): Expected object diameter, passed to Cellpose.
        margin (int): Pixel margin added around each grain's bounding box.

    Returns:
        List[Dict[str, Any]]: One entry per extracted grain, each with keys
        `grain_id`, `image` (masked crop, np.ndarray), and `mask` (np.ndarray).
    """
    image = skio.imread(image_path)
    filtered_mask, _ = segment_image(model, image, min_area=min_area, diameter=diameter)

    props = regionprops(filtered_mask, intensity_image=image)
    grains = []
    for prop in props:
        grain_img, grain_mask = extract_grain(image, prop, filtered_mask, margin=margin)
        grains.append({"grain_id": prop.label, "image": grain_img, "mask": grain_mask})
    return grains


def process_directory(
    curated_root: Path,
    output_root: Path,
    mask_output_root: Path,
    crop_mask_output_root: Path,
    model: Any,
    grouped_by_image: bool = False,
    min_area: int = 1500,
    diameter: int = 50,
    margin: int = 20,
    image_suffixes: Tuple[str, ...] = DEFAULT_IMAGE_SUFFIXES,
) -> Dict[str, int]:
    """Batch-processes a directory of labeled grain photographs.

    Walks each label subdirectory under `curated_root`, segments every image,
    saves an overlay visualization, and writes out per-grain crops and masks.

    Args:
        curated_root (Path): Root directory containing one subfolder per class label.
        output_root (Path): Destination for per-grain JPEG crops.
        mask_output_root (Path): Destination for full-image overlay visualizations.
        crop_mask_output_root (Path): Destination for per-grain binary masks.
        model (Any): An initialized `cellpose.models.CellposeModel` instance.
        grouped_by_image (bool): If True, nests each source image's grains in
            their own subfolder. Needed when multiple composite-trait images
            would otherwise collide on overlapping grain IDs within a label.
        min_area (int): Minimum region area (pixels) to keep.
        diameter (int): Expected object diameter, passed to Cellpose.
        margin (int): Pixel margin added around each grain's bounding box.
        image_suffixes (Tuple[str, ...]): File extensions treated as images.

    Returns:
        Dict[str, int]: Number of grains extracted per label subdirectory.
    """
    import matplotlib.pyplot as plt

    output_root.mkdir(parents=True, exist_ok=True)
    mask_output_root.mkdir(parents=True, exist_ok=True)
    crop_mask_output_root.mkdir(parents=True, exist_ok=True)

    label_stats: Dict[str, int] = {}

    for label_dir in sorted(p for p in curated_root.iterdir() if p.is_dir()):
        label_name = label_dir.name
        logger.info(f"Processing label: {label_name}")

        label_output_dir = output_root / label_name
        label_crop_mask_dir = crop_mask_output_root / label_name
        label_output_dir.mkdir(parents=True, exist_ok=True)
        label_crop_mask_dir.mkdir(parents=True, exist_ok=True)

        grain_count = 0
        for img_path in sorted(label_dir.glob("*")):
            if img_path.suffix.lower() not in image_suffixes:
                continue

            try:
                image = skio.imread(img_path)
                filtered_mask, _ = segment_image(model, image, min_area=min_area, diameter=diameter)

                overlay = render_overlay(image, filtered_mask)
                plt.figure(figsize=(8, 8))
                plt.imshow(overlay)
                plt.title(f"{img_path.name} - grains: {filtered_mask.max()}")
                plt.axis("off")
                plt.savefig(mask_output_root / f"{img_path.stem}_mask.png", dpi=150, bbox_inches="tight")
                plt.close()

                if grouped_by_image:
                    image_output_dir = label_output_dir / img_path.stem
                    image_crop_mask_dir = label_crop_mask_dir / img_path.stem
                    image_output_dir.mkdir(parents=True, exist_ok=True)
                    image_crop_mask_dir.mkdir(parents=True, exist_ok=True)
                else:
                    image_output_dir = label_output_dir
                    image_crop_mask_dir = label_crop_mask_dir

                props = regionprops(filtered_mask, intensity_image=image)
                for prop in props:
                    grain_img, grain_mask = extract_grain(image, prop, filtered_mask, margin=margin)
                    stem = f"{img_path.stem}_grain_{prop.label:04d}"
                    # Saved via PIL rather than skio.imsave: skimage's imsave forwards
                    # `quality` as a plugin kwarg, which triggers a FutureWarning about
                    # the deprecated plugin_args mechanism as of scikit-image 0.25.
                    Image.fromarray(grain_img).save(image_output_dir / f"{stem}.jpg", quality=95)
                    skio.imsave(image_crop_mask_dir / f"{stem}_mask.png", grain_mask)
                    grain_count += 1

            except Exception as exc:
                logger.error(f"Failed to process {img_path.name}: {exc}")
                continue

        label_stats[label_name] = grain_count
        logger.info(f"{label_name}: {grain_count} grains extracted")

    return label_stats
