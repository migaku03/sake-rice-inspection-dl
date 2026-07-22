"""Statistical domain adaptation for grain color/texture distributions.

Sake rice grain photographs are scarce for some visual traits, while much
larger datasets of visually similar edible rice exist. This module adapts
an edible-rice grain image so that its masked-region color statistics match
those of a target sake-rice reference set, following:

    P_out = (P_in - Mean_in) * (Std_target / Std_in) + Mean_target

Optionally, small random noise ("jitter") can be added to the target
statistics per source image so that a batch of transformed outputs is not
all pulled toward one identical statistic.
"""

import logging
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)


def imread_safe(path: Path, flags: int = cv2.IMREAD_COLOR) -> Optional[np.ndarray]:
    """Reads an image robustly, including paths with non-ASCII characters.

    `cv2.imread` can silently fail on some platforms when the path contains
    non-ASCII characters. Decoding raw bytes via `cv2.imdecode` avoids that.

    Args:
        path (Path): Path to the image file.
        flags (int): OpenCV imread flags (e.g. `cv2.IMREAD_GRAYSCALE`).

    Returns:
        Optional[np.ndarray]: The decoded image (BGR order), or None on failure.
    """
    try:
        data = np.fromfile(str(path), dtype=np.uint8)
        if data.size == 0:
            return None
        return cv2.imdecode(data, flags)
    except Exception:
        return None


def imwrite_safe(path: Path, image: np.ndarray) -> bool:
    """Writes an image robustly, including paths with non-ASCII characters.

    Args:
        path (Path): Destination path. The file extension determines the
            encoding format (e.g. `.jpg`, `.png`).
        image (np.ndarray): The image to write (BGR order).

    Returns:
        bool: True if the file was written successfully.
    """
    try:
        ext = Path(path).suffix
        ok, encoded = cv2.imencode(ext, image)
        if not ok:
            return False
        encoded.tofile(str(path))
        return True
    except Exception:
        return False


def get_masked_stats(image: np.ndarray, mask: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Computes per-channel mean and standard deviation within a mask.

    Args:
        image (np.ndarray): Source image (H, W, C).
        mask (np.ndarray): Single-channel mask; pixels where `mask > 0` are
            included in the statistics.

    Returns:
        Tuple[Optional[np.ndarray], Optional[np.ndarray]]: Per-channel mean
        and standard deviation arrays, or (None, None) if the mask is empty.
    """
    pixels = image[mask > 0]
    if len(pixels) == 0:
        return None, None

    return np.mean(pixels, axis=0), np.std(pixels, axis=0)


def compute_target_statistics(image_mask_pairs: Iterable[Tuple[np.ndarray, np.ndarray]]) -> Tuple[np.ndarray, np.ndarray]:
    """Averages masked-region statistics across a reference set of images.

    Args:
        image_mask_pairs (Iterable[Tuple[np.ndarray, np.ndarray]]): Pairs of
            (image, mask) arrays for the reference (target-domain) grains.

    Returns:
        Tuple[np.ndarray, np.ndarray]: The mean-of-means and mean-of-stds
        per channel across the reference set.

    Raises:
        ValueError: If no valid (non-empty-mask) pairs were provided.
    """
    means: List[np.ndarray] = []
    stds: List[np.ndarray] = []

    for image, mask in image_mask_pairs:
        mean, std = get_masked_stats(image, mask)
        if mean is not None:
            means.append(mean)
            stds.append(std)

    if not means:
        raise ValueError("No valid image/mask pairs to compute target statistics from.")

    return np.mean(means, axis=0), np.mean(stds, axis=0)


def jitter_statistics(
    target_mean: np.ndarray,
    target_std: np.ndarray,
    mean_jitter_scale: float = 0.08,
    std_jitter_scale: float = 0.02,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Perturbs target statistics with small Gaussian noise.

    Applying a jittered target to each source image (rather than one fixed
    target) means a batch of transformed outputs ends up with slightly
    different statistics from one another instead of being identical.

    Args:
        target_mean (np.ndarray): Per-channel target mean.
        target_std (np.ndarray): Per-channel target standard deviation.
        mean_jitter_scale (float): Noise std for the mean, as a fraction of `target_std`.
        std_jitter_scale (float): Noise std for the std, as a fraction of `target_std`.
        rng (Optional[np.random.Generator]): Random generator; a fresh default
            generator is used if not provided.

    Returns:
        Tuple[np.ndarray, np.ndarray]: The jittered (mean, std).
    """
    if rng is None:
        rng = np.random.default_rng()

    jitter_mean = target_mean + rng.normal(0, target_std * mean_jitter_scale, size=target_mean.shape)
    jitter_std = target_std + rng.normal(0, target_std * std_jitter_scale, size=target_std.shape)
    jitter_std = np.clip(jitter_std, 0.1, None)

    return jitter_mean, jitter_std


def apply_domain_transform(
    image: np.ndarray,
    mask: np.ndarray,
    target_mean: np.ndarray,
    target_std: np.ndarray,
) -> np.ndarray:
    """Shifts a masked image's per-channel distribution to match a target.

    Background pixels (where `mask == 0`) are zeroed out in the output.

    Args:
        image (np.ndarray): Source image (H, W, C), any numeric dtype.
        mask (np.ndarray): Single-channel mask; `mask > 0` marks the grain.
        target_mean (np.ndarray): Per-channel target mean.
        target_std (np.ndarray): Per-channel target standard deviation.

    Returns:
        np.ndarray: The transformed image, dtype uint8, values in [0, 255].
    """
    image = image.astype(np.float32)
    curr_mean, curr_std = get_masked_stats(image, mask)
    if curr_mean is None:
        raise ValueError("Mask is empty; cannot compute source statistics.")

    transformed = np.zeros_like(image)
    for c in range(image.shape[2]):
        channel = image[:, :, c]
        transformed[:, :, c] = (channel - curr_mean[c]) * (target_std[c] / curr_std[c]) + target_mean[c]

    result = np.where(mask[:, :, np.newaxis] > 0, transformed, 0)
    return np.clip(result, 0, 255).astype(np.uint8)


def transform_directory(
    source_dir: Path,
    source_mask_dir: Path,
    target_mean: np.ndarray,
    target_std: np.ndarray,
    output_dir: Path,
    jitter: bool = True,
    mean_jitter_scale: float = 0.08,
    std_jitter_scale: float = 0.02,
    rng: Optional[np.random.Generator] = None,
    mask_suffix: str = "_mask.png",
) -> Tuple[int, List[str]]:
    """Applies the domain transform to every grain crop in a directory.

    Args:
        source_dir (Path): Directory of source (e.g. edible-rice) grain crops (`.jpg`).
        source_mask_dir (Path): Directory of matching per-grain binary masks.
        target_mean (np.ndarray): Per-channel target mean.
        target_std (np.ndarray): Per-channel target standard deviation.
        output_dir (Path): Destination directory for transformed crops.
        jitter (bool): Whether to jitter the target per image (see `jitter_statistics`).
        mean_jitter_scale (float): Passed to `jitter_statistics` when `jitter=True`.
        std_jitter_scale (float): Passed to `jitter_statistics` when `jitter=True`.
        rng (Optional[np.random.Generator]): Random generator for jitter.
        mask_suffix (str): Suffix appended to an image's stem to find its mask file.

    Returns:
        Tuple[int, List[str]]: Number of images saved successfully, and paths
        of any images that failed to write.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    if rng is None:
        rng = np.random.default_rng()

    saved_count = 0
    failed_writes: List[str] = []

    for img_path in sorted(source_dir.glob("*.jpg")):
        mask_path = source_mask_dir / (img_path.stem + mask_suffix)
        if not mask_path.exists():
            continue

        image = imread_safe(img_path)
        mask = imread_safe(mask_path, cv2.IMREAD_GRAYSCALE)
        if image is None or mask is None:
            logger.warning(f"Skipping {img_path.name}: failed to load image or mask.")
            continue

        this_mean, this_std = (
            jitter_statistics(target_mean, target_std, mean_jitter_scale, std_jitter_scale, rng)
            if jitter
            else (target_mean, target_std)
        )

        try:
            result = apply_domain_transform(image, mask, this_mean, this_std)
        except ValueError:
            logger.warning(f"Skipping {img_path.name}: empty mask.")
            continue

        out_path = output_dir / img_path.name
        if imwrite_safe(out_path, result):
            saved_count += 1
        else:
            failed_writes.append(str(out_path))

    return saved_count, failed_writes
