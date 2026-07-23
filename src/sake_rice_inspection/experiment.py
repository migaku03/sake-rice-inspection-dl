"""Experiment directory bookkeeping.

Each training run gets its own timestamped directory and a `README.md`
recording its configuration up front, with final cross-validation results
appended once training completes. This keeps every run's settings and
outcome self-documenting without a separate experiment-tracking service.
"""

import os
from datetime import datetime
from typing import Any, Dict, List


def create_experiment_dir(base_dir: str, tag: str, date: str = None) -> str:
    """Creates a fresh, uniquely-numbered experiment directory.

    Args:
        base_dir (str): Parent directory for all experiments of this kind.
        tag (str): Short experiment name, used as a directory name prefix.
        date (str): Date string to embed in the directory name (`%Y%m%d`).
            Defaults to today's date.

    Returns:
        str: Path to the newly created experiment directory.
    """
    date = date or datetime.now().strftime("%Y%m%d")
    idx = 1
    while True:
        exp_name = f"{tag}_{date}_{idx:03d}"
        exp_dir = os.path.join(base_dir, exp_name)
        if not os.path.exists(exp_dir):
            os.makedirs(exp_dir)
            return exp_dir
        idx += 1


def save_experiment_readme(exp_dir: str, config: Dict[str, Any]) -> str:
    """Writes a `README.md` recording a run's configuration.

    Args:
        exp_dir (str): The experiment directory (see `create_experiment_dir`).
        config (Dict[str, Any]): Run configuration. Expected keys: `tag`,
            `description`, `crop_dir`, `folds_csv`, `num_classes`, `details`,
            `class_names`, `folds`, `arch`, `img_size`, `batch_size`,
            `epochs_head`, `lr_head`, `epochs_ft`, `lr_ft`, `seed`,
            `augmentation`, `augmentation_details`, `scale`.

    Returns:
        str: Path to the written README.
    """
    readme_path = os.path.join(exp_dir, "README.md")

    with open(readme_path, "w", encoding="utf-8-sig") as f:
        f.write(f"# Experiment: {config['tag']}\n\n")
        f.write(f"**Created:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        f.write("## Overview\n\n")
        f.write(config.get("description", "") + "\n\n")

        f.write("## Data\n\n")
        f.write(f"- **Source:** `{config['crop_dir']}`\n")
        f.write(f"- **Folds CSV:** `{config['folds_csv']}`\n")
        f.write(f"- **Number of classes:** {config['num_classes']}\n")
        f.write(f"- **Details:** `{config['details']}`\n")
        f.write(f"- **Class names:** {', '.join(config['class_names'])}\n")
        f.write(f"- **Folds:** {config['folds']}\n\n")

        f.write("## Model\n\n")
        f.write(f"- **Architecture:** {config['arch']}\n")
        f.write(f"- **Image size:** {config['img_size']}x{config['img_size']}\n")
        f.write(f"- **Batch size:** {config['batch_size']}\n\n")

        f.write("## Training\n\n")
        f.write(f"- **Head epochs:** {config['epochs_head']}\n")
        f.write(f"- **Head LR:** {config['lr_head']}\n")
        f.write(f"- **Finetune epochs:** {config['epochs_ft']}\n")
        f.write(f"- **Finetune LR:** {config['lr_ft']}\n")
        f.write(f"- **Seed:** {config['seed']}\n\n")

        f.write("## Augmentation & Preprocessing\n\n")
        f.write(f"- **Training augmentation:** {config['augmentation']}\n")
        f.write(f"- **Details:** {config['augmentation_details']}\n")
        f.write(f"- **Scale factor:** {config['scale']}\n\n")

    return readme_path


def update_readme_with_results(exp_dir: str, results: Dict[str, Any]) -> None:
    """Appends final cross-validation results to an experiment's README.

    Args:
        exp_dir (str): The experiment directory.
        results (Dict[str, Any]): Expected keys: `total_eval`,
            `total_exact_match`, `exact_match_acc`, `hamming_acc`, `macro_f1`,
            and `fold_results` (a list of per-fold dicts with the same
            metric keys plus `fold` and `eval_count`).
    """
    readme_path = os.path.join(exp_dir, "README.md")

    with open(readme_path, "a", encoding="utf-8-sig") as f:
        f.write(f"\n## Final Results (updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')})\n\n")

        f.write("### Overall\n\n")
        f.write(f"- **Total evaluated:** {results['total_eval']}\n")
        f.write(f"- **Exact matches:** {results['total_exact_match']}\n")
        f.write(f"- **Exact match accuracy:** {results['exact_match_acc']:.4f}\n")
        f.write(f"- **Hamming accuracy:** {results['hamming_acc']:.4f}\n")
        f.write(f"- **Macro F1:** {results['macro_f1']:.4f}\n\n")

        f.write("### Per Fold\n\n")
        f.write("| Fold | Eval count | Exact matches | Exact match acc | Hamming acc | Macro F1 |\n")
        f.write("|------|------------|----------------|------------------|-------------|----------|\n")
        for fr in results["fold_results"]:
            f.write(
                f"| {fr['fold']} | {fr['eval_count']} | {fr['exact_match_count']} | "
                f"{fr['exact_match_acc']:.4f} | {fr['hamming_acc']:.4f} | {fr['macro_f1']:.4f} |\n"
            )


def summarize_fold_results(fold_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Aggregates per-fold metrics into overall totals.

    Args:
        fold_results (List[Dict[str, Any]]): Per-fold dicts with keys
            `fold`, `eval_count`, `exact_match_count`, `exact_match_acc`,
            `hamming_acc`, `macro_f1`.

    Returns:
        Dict[str, Any]: Overall totals plus the original `fold_results`,
        ready to pass to `update_readme_with_results`.
    """
    total_eval = sum(fr["eval_count"] for fr in fold_results)
    total_exact_match = sum(fr["exact_match_count"] for fr in fold_results)
    total_exact_match_acc = total_exact_match / total_eval if total_eval > 0 else 0.0
    total_hamming_acc = sum(fr["hamming_acc"] for fr in fold_results) / len(fold_results)
    total_macro_f1 = sum(fr["macro_f1"] for fr in fold_results) / len(fold_results)

    return {
        "total_eval": total_eval,
        "total_exact_match": total_exact_match,
        "exact_match_acc": total_exact_match_acc,
        "hamming_acc": total_hamming_acc,
        "macro_f1": total_macro_f1,
        "fold_results": fold_results,
    }
