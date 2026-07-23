"""Cross-validation fold construction for the multi-label "partially clouded"
classification task.

A grain can simultaneously exhibit up to four co-occurring traits (Shinpaku,
Base White, Back White, Belly White). This module collects grain crops from
both the single-trait and composite-trait crop directories, derives a
multi-hot label vector for each, and writes a stratified k-fold assignment
CSV (stratified on the joined label string, so rare trait combinations are
still spread across folds).
"""

import csv
import glob
import os
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

from sklearn.model_selection import StratifiedKFold

DEFAULT_LABEL_COLUMNS = ["心白", "基白", "背白", "腹白"]
DEFAULT_IMAGE_EXTENSIONS = (".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp")


def get_multilabel_flags(dir_name: str, label_columns: List[str] = DEFAULT_LABEL_COLUMNS) -> Dict[str, int]:
    """Infers which traits are present from a composite-crop directory name.

    Directory names encode which traits co-occur (e.g. a folder for grains
    with both Shinpaku and Base White). This maps Japanese substrings in the
    name to a multi-hot flag dict over `label_columns`.

    Args:
        dir_name (str): The composite-trait subdirectory name.
        label_columns (List[str]): The four trait names, in output order.

    Returns:
        Dict[str, int]: 1/0 flag for each entry in `label_columns`.
    """
    flags = {col: 0 for col in label_columns}

    if "心白" in dir_name or "腹一体" in dir_name:
        flags["心白"] = 1
    if "基" in dir_name:
        flags["基白"] = 1
    if "背" in dir_name:
        flags["背白"] = 1
    if "腹" in dir_name and "腹寄り" not in dir_name:
        flags["腹白"] = 1

    return flags


def collect_single_trait_samples(
    root: str, allowed_classes: List[str], label_columns: List[str] = DEFAULT_LABEL_COLUMNS,
    extensions: Tuple[str, ...] = DEFAULT_IMAGE_EXTENSIONS,
) -> List[dict]:
    """Collects samples from single-trait class subdirectories.

    Each subdirectory under `root` is assumed to represent exactly one trait;
    every image in it gets a one-hot label for that trait.

    Args:
        root (str): Root directory containing one subfolder per single trait.
        allowed_classes (List[str]): Subdirectory names to include; others are skipped.
        label_columns (List[str]): The four trait names, in output order.
        extensions (Tuple[str, ...]): Image file extensions to collect.

    Returns:
        List[dict]: Each entry has `path`, `label_str`, and `flags` keys.
    """
    if not os.path.isdir(root):
        return []

    classes = sorted(d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d)))
    samples = []
    for cls in classes:
        if cls not in allowed_classes:
            continue

        flags = {col: (1 if col == cls else 0) for col in label_columns}
        for ext in extensions:
            for p in glob.glob(os.path.join(root, cls, "**", f"*{ext}"), recursive=True):
                samples.append({"path": p, "label_str": cls, "flags": flags})
    return samples


def collect_composite_trait_samples(
    root: str, label_columns: List[str] = DEFAULT_LABEL_COLUMNS,
    extensions: Tuple[str, ...] = DEFAULT_IMAGE_EXTENSIONS,
) -> List[dict]:
    """Collects samples from composite-trait directories.

    Composite crops are nested two levels deep (category, then a specific
    combination directory); the combination directory name is decoded via
    `get_multilabel_flags`.

    Args:
        root (str): Root directory of composite-trait crops.
        label_columns (List[str]): The four trait names, in output order.
        extensions (Tuple[str, ...]): Image file extensions to collect.

    Returns:
        List[dict]: Each entry has `path`, `label_str`, and `flags` keys.
    """
    root_path = Path(root)
    if not root_path.exists():
        return []

    samples = []
    for category_dir in root_path.iterdir():
        if not category_dir.is_dir():
            continue
        for specific_dir in category_dir.iterdir():
            if not specific_dir.is_dir():
                continue

            flags = get_multilabel_flags(specific_dir.name, label_columns)
            active_labels = [k for k, v in flags.items() if v == 1]
            label_str = "+".join(active_labels) if active_labels else "その他"

            for ext in extensions:
                for p in specific_dir.glob(f"*{ext}"):
                    samples.append({"path": str(p), "label_str": label_str, "flags": flags})
    return samples


def filter_by_minimum_class_count(samples: List[dict], min_count: int) -> List[dict]:
    """Drops samples belonging to a label class smaller than `min_count`.

    `StratifiedKFold` requires every class to have at least as many members
    as the number of folds; classes below that threshold are dropped.

    Args:
        samples (List[dict]): Samples as produced by the `collect_*` functions.
        min_count (int): Minimum number of samples a label class must have.

    Returns:
        List[dict]: The filtered sample list.
    """
    counts = Counter(s["label_str"] for s in samples)
    return [s for s in samples if counts[s["label_str"]] >= min_count]


def make_multilabel_folds_csv(
    samples: List[dict], out_csv: str, label_columns: List[str] = DEFAULT_LABEL_COLUMNS,
    folds: int = 4, seed: int = 42,
) -> None:
    """Writes a stratified k-fold assignment CSV for multi-label samples.

    Stratification is performed on the joined label string (e.g.
    "Shinpaku+Belly White"), so rare trait combinations are still distributed
    evenly across folds rather than being clustered in a single one.

    Args:
        samples (List[dict]): Samples as produced by the `collect_*` functions.
        out_csv (str): Destination CSV path.
        label_columns (List[str]): The four trait names, in output order.
        folds (int): Number of cross-validation folds.
        seed (int): Random seed for the fold split.
    """
    paths = [s["path"] for s in samples]
    labels = [s["label_str"] for s in samples]

    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)

    header = ["path", "label_str", "fold"] + label_columns
    with open(out_csv, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(header)

        for fold, (_, va_idx) in enumerate(skf.split(paths, labels)):
            for i in va_idx:
                sample = samples[i]
                flag_values = [sample["flags"][col] for col in label_columns]
                writer.writerow([sample["path"], sample["label_str"], fold] + flag_values)


def read_folds_csv(csv_path: str) -> Tuple[List[Tuple[str, List[int], str, int]], List[str]]:
    """Reads back a fold-assignment CSV written by `make_multilabel_folds_csv`.

    Args:
        csv_path (str): Path to the fold CSV.

    Returns:
        Tuple[List[Tuple[str, List[int], str, int]], List[str]]: A list of
        `(path, label_flags, label_str, fold)` rows, and the label column names.
    """
    rows = []
    label_columns: List[str] = []
    with open(csv_path, encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for d in reader:
            if not label_columns:
                label_columns = [k for k in d.keys() if k not in ("path", "label_str", "fold")]
            labels = [int(d[col]) for col in label_columns]
            rows.append((d["path"], labels, d["label_str"], int(d["fold"])))
    return rows, label_columns
