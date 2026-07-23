"""Tests for sake_rice_inspection.folds. Uses only empty placeholder files
(fold construction operates on paths and directory names, not pixel data).
"""

from sake_rice_inspection.folds import (
    collect_composite_trait_samples,
    collect_single_trait_samples,
    filter_by_minimum_class_count,
    get_multilabel_flags,
    make_multilabel_folds_csv,
    read_folds_csv,
)

LABEL_COLUMNS = ["心白", "基白", "背白", "腹白"]


def test_get_multilabel_flags_detects_each_trait():
    assert get_multilabel_flags("心白", LABEL_COLUMNS) == {"心白": 1, "基白": 0, "背白": 0, "腹白": 0}
    assert get_multilabel_flags("基白", LABEL_COLUMNS) == {"心白": 0, "基白": 1, "背白": 0, "腹白": 0}
    assert get_multilabel_flags("背白", LABEL_COLUMNS) == {"心白": 0, "基白": 0, "背白": 1, "腹白": 0}
    assert get_multilabel_flags("腹白", LABEL_COLUMNS) == {"心白": 0, "基白": 0, "背白": 0, "腹白": 1}


def test_get_multilabel_flags_detects_combination():
    flags = get_multilabel_flags("心白_基白_複合", LABEL_COLUMNS)
    assert flags == {"心白": 1, "基白": 1, "背白": 0, "腹白": 0}


def test_get_multilabel_flags_excludes_belly_leaning_variant():
    flags = get_multilabel_flags("腹寄り", LABEL_COLUMNS)
    assert flags["腹白"] == 0


def test_collect_single_trait_samples_only_allowed_classes(tmp_path):
    root = tmp_path / "single"
    (root / "心白").mkdir(parents=True)
    (root / "心白" / "a.jpg").touch()
    (root / "その他_食用").mkdir(parents=True)
    (root / "その他_食用" / "b.jpg").touch()

    samples = collect_single_trait_samples(str(root), allowed_classes=["心白"], label_columns=LABEL_COLUMNS)

    assert len(samples) == 1
    assert samples[0]["label_str"] == "心白"
    assert samples[0]["flags"] == {"心白": 1, "基白": 0, "背白": 0, "腹白": 0}


def test_collect_single_trait_samples_missing_root_returns_empty(tmp_path):
    samples = collect_single_trait_samples(str(tmp_path / "missing"), allowed_classes=["心白"])
    assert samples == []


def test_collect_composite_trait_samples_nested_layout(tmp_path):
    root = tmp_path / "composite"
    specific_dir = root / "category" / "心白_基白"
    specific_dir.mkdir(parents=True)
    (specific_dir / "img.jpg").touch()

    samples = collect_composite_trait_samples(str(root), label_columns=LABEL_COLUMNS)

    assert len(samples) == 1
    assert samples[0]["label_str"] == "心白+基白"
    assert samples[0]["flags"]["心白"] == 1
    assert samples[0]["flags"]["基白"] == 1


def test_filter_by_minimum_class_count_drops_rare_classes():
    samples = [
        {"label_str": "A"}, {"label_str": "A"}, {"label_str": "A"},
        {"label_str": "B"},
    ]
    filtered = filter_by_minimum_class_count(samples, min_count=2)
    assert all(s["label_str"] == "A" for s in filtered)
    assert len(filtered) == 3


def test_make_and_read_folds_csv_roundtrip(tmp_path):
    samples = [
        {"path": f"/data/img_{i}.jpg", "label_str": "心白", "flags": {"心白": 1, "基白": 0, "背白": 0, "腹白": 0}}
        for i in range(4)
    ] + [
        {"path": f"/data/img_b_{i}.jpg", "label_str": "基白", "flags": {"心白": 0, "基白": 1, "背白": 0, "腹白": 0}}
        for i in range(4)
    ]
    out_csv = tmp_path / "folds.csv"

    make_multilabel_folds_csv(samples, str(out_csv), label_columns=LABEL_COLUMNS, folds=2, seed=0)
    rows, label_columns = read_folds_csv(str(out_csv))

    assert label_columns == LABEL_COLUMNS
    assert len(rows) == 8
    folds_seen = {row[3] for row in rows}
    assert folds_seen == {0, 1}
