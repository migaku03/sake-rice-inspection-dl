"""Dataset and image preprocessing for multi-label grain classification.

Grain crops vary in size (they are trimmed to their own bounding box during
preprocessing), so images are resized by a fixed scale factor and then
zero-padded to a square before being fed to a CNN/ViT backbone. Training-time
flip augmentation is applied as a 4x multiplier (original, horizontal flip,
vertical flip, both).
"""

from typing import List, Tuple

import numpy as np
import torch
from PIL import Image, ImageOps
from torch.utils.data import Dataset
from torchvision import transforms

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def apply_flip_augmentation(image: Image.Image, aug_idx: int) -> Image.Image:
    """Applies one of four flip augmentations to a PIL image.

    Args:
        image (Image.Image): The source image.
        aug_idx (int): 0 = original, 1 = horizontal flip, 2 = vertical flip,
            3 = both (equivalent to a 180-degree rotation).

    Returns:
        Image.Image: The augmented image.
    """
    if aug_idx == 1:
        return image.transpose(Image.FLIP_LEFT_RIGHT)
    if aug_idx == 2:
        return image.transpose(Image.FLIP_TOP_BOTTOM)
    if aug_idx == 3:
        return image.transpose(Image.FLIP_LEFT_RIGHT).transpose(Image.FLIP_TOP_BOTTOM)
    return image


def resize_and_pad(image: Image.Image, scale: float, out_size: int = 224) -> Image.Image:
    """Scales an image and zero-pads it to a square of side `out_size`.

    Args:
        image (Image.Image): The source image.
        scale (float): Uniform scale factor applied before padding.
        out_size (int): Target square side length.

    Returns:
        Image.Image: The resized, padded image.
    """
    new_w = int(image.width * scale)
    new_h = int(image.height * scale)
    image = image.resize((new_w, new_h), Image.LANCZOS)

    delta_w = out_size - new_w
    delta_h = out_size - new_h
    padding = (delta_w // 2, delta_h // 2, delta_w - delta_w // 2, delta_h - delta_h // 2)
    return ImageOps.expand(image, padding, fill=(0, 0, 0))


def to_normalized_tensor(image: Image.Image) -> torch.Tensor:
    """Converts a PIL image to a normalized (ImageNet stats) tensor.

    Args:
        image (Image.Image): The source image (RGB).

    Returns:
        torch.Tensor: A (C, H, W) float tensor.
    """
    tfm = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    return tfm(image)


def preprocess_image(image_path: str, scale: float, img_size: int = 224) -> np.ndarray:
    """Loads, scales, pads, and normalizes an image for inference.

    Args:
        image_path (str): Path to the grain crop.
        scale (float): Uniform scale factor applied before padding.
        img_size (int): Target square side length.

    Returns:
        np.ndarray: A (C, H, W) float32 array.
    """
    image = Image.open(image_path).convert("RGB")
    image = resize_and_pad(image, scale, img_size)
    return to_normalized_tensor(image).numpy()


def denormalize_for_display(tensor_image: np.ndarray) -> np.ndarray:
    """Reverses ImageNet normalization for visualization.

    Args:
        tensor_image (np.ndarray): A (C, H, W) normalized array.

    Returns:
        np.ndarray: An (H, W, C) array with values clipped to [0, 1].
    """
    image = tensor_image.transpose(1, 2, 0)
    image = np.array(IMAGENET_STD) * image + np.array(IMAGENET_MEAN)
    return np.clip(image, 0, 1)


class MultiLabelGrainDataset(Dataset):
    """Grain crop dataset with multi-hot labels and optional flip augmentation.

    Attributes:
        items (List[Tuple[str, List[int], int]]): Expanded (path, labels, aug_idx) entries.
        scale (float): Uniform scale factor applied before padding.
        img_size (int): Target square side length.
    """

    def __init__(
        self,
        items: List[Tuple[str, List[int]]],
        scale: float,
        img_size: int = 224,
        augment: bool = False,
    ):
        """Initializes the dataset.

        Args:
            items (List[Tuple[str, List[int]]]): (path, multi-hot label) pairs.
            scale (float): Uniform scale factor applied before padding.
            img_size (int): Target square side length.
            augment (bool): If True, expands each item into 4 flip variants.
        """
        if augment:
            self.items = [(p, lab, aug_idx) for p, lab in items for aug_idx in range(4)]
        else:
            self.items = [(p, lab, 0) for p, lab in items]

        self.scale = scale
        self.img_size = img_size

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        path, labels, aug_idx = self.items[index]

        image = Image.open(path).convert("RGB")
        image = apply_flip_augmentation(image, aug_idx)
        image = resize_and_pad(image, self.scale, self.img_size)
        tensor = to_normalized_tensor(image)

        return tensor, torch.tensor(labels, dtype=torch.float32)
