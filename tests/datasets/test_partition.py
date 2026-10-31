"""Tests for partition_dataset (train/val/test as a dataset artifact)."""

import csv

import pytest

from bioamla.datasets._metadata import read_metadata_csv, write_metadata_csv
from bioamla.datasets.partition import partition_dataset
from bioamla.exceptions import DatasetError, NotFoundError


def _write_dataset(tmp_path, n_per_label=20, labels=("call", "chorus"), with_source=True):
    """Build a flat dataset.csv + dummy clip files, one row per clip."""
    rows = []
    for label in labels:
        for i in range(n_per_label):
            rel = f"{label}/{label}_{i:03d}.wav"
            (tmp_path / label).mkdir(parents=True, exist_ok=True)
            (tmp_path / rel).write_bytes(b"RIFF")  # placeholder file
            row = {"file_name": rel, "label": label}
            if with_source:
                # Two clips share each recording, to exercise group_by.
                row["source_file"] = f"{label}_rec_{i // 2}.wav"
            rows.append(row)
    write_metadata_csv(tmp_path / "metadata.csv", rows, merge_existing=False)
    return tmp_path


def _read_split_map(dataset_dir):
    rows, _ = read_metadata_csv(dataset_dir / "metadata.csv")
    return rows


class TestPartitionColumn:
    def test_ratios_within_tolerance(self, tmp_path):
        d = _write_dataset(tmp_path, n_per_label=50, with_source=False)
        result = partition_dataset(str(d), splits=(0.7, 0.15, 0.15), mode="column")
        total = sum(result["splits"].values())
        assert total == 100
        assert abs(result["splits"]["train"] / total - 0.7) < 0.05
        assert abs(result["splits"]["val"] / total - 0.15) < 0.05
        assert abs(result["splits"]["test"] / total - 0.15) < 0.05

    def test_deterministic_across_runs(self, tmp_path):
        d1 = _write_dataset(tmp_path / "a", n_per_label=30)
        d2 = _write_dataset(tmp_path / "b", n_per_label=30)
        partition_dataset(str(d1), seed=7, mode="column")
        partition_dataset(str(d2), seed=7, mode="column")
        m1 = {r["file_name"]: r["split"] for r in _read_split_map(d1)}
        m2 = {r["file_name"]: r["split"] for r in _read_split_map(d2)}
        assert m1 == m2

    def test_stratify_keeps_each_label_in_train(self, tmp_path):
        d = _write_dataset(tmp_path, n_per_label=30, with_source=False)
        partition_dataset(str(d), mode="column", stratify=True)
        by_label_split = {(r["label"], r["split"]) for r in _read_split_map(d)}
        # Each label should appear in train, val, and test.
        for label in ("call", "chorus"):
            assert (label, "train") in by_label_split
            assert (label, "val") in by_label_split
            assert (label, "test") in by_label_split


class TestGroupingNoLeakage:
    def test_recording_does_not_span_splits(self, tmp_path):
        d = _write_dataset(tmp_path, n_per_label=40, with_source=True)
        partition_dataset(str(d), group_by="source_file", mode="column")
        rows = _read_split_map(d)
        source_to_splits = {}
        for r in rows:
            source_to_splits.setdefault(r["source_file"], set()).add(r["split"])
        assert all(len(splits) == 1 for splits in source_to_splits.values())


class TestEmptySplits:
    def test_too_few_groups_reported(self, tmp_path, caplog):
        d = _write_dataset(tmp_path, n_per_label=2, labels=("call",))  # one recording
        result = partition_dataset(str(d), mode="column")
        assert result["splits"] == {"train": 2}
        assert result["empty_splits"] == ["val", "test"]
        assert "Empty split(s) val, test: only 1 group(s)" in caplog.text
        assert "source_file" in caplog.text

    def test_stratified_shortfall_names_the_strata(self, tmp_path, caplog):
        # 3 labels x 4 recordings: 12 groups overall, but 4 per label at 10% leaves one
        # leftover group per stratum, so one of val/test gets nothing anywhere.
        d = _write_dataset(tmp_path, n_per_label=8, labels=("a", "b", "c"))
        result = partition_dataset(str(d), splits=(0.8, 0.1, 0.1), stratify=True, mode="column")
        assert result["empty_splits"] == ["test"]
        assert "too few per stratum" in caplog.text

    def test_zero_fraction_split_is_not_reported(self, tmp_path):
        d = _write_dataset(tmp_path, n_per_label=20, with_source=False)
        result = partition_dataset(str(d), splits=(0.8, 0.2, 0.0), mode="column")
        assert result["empty_splits"] == []

    def test_filled_splits(self, tmp_path):
        d = _write_dataset(tmp_path, n_per_label=30)
        assert partition_dataset(str(d), mode="column")["empty_splits"] == []


class TestAllocation:
    @pytest.mark.parametrize(
        ("n", "fractions", "expected"),
        [
            (5, (0.5, 0.5, 0.0), (3, 2, 0)),  # round-half-even used to give test a group
            (1, (0.5, 0.5, 0.0), (1, 0, 0)),
            (10, (0.8, 0.1, 0.1), (8, 1, 1)),
            (7, (0.6, 0.2, 0.2), (4, 2, 1)),  # 4.2, 1.4, 1.4: the leftover goes to val
            (3, (1.0, 0.0, 0.0), (3, 0, 0)),
            (0, (0.8, 0.1, 0.1), (0, 0, 0)),
        ],
    )
    def test_allocate(self, n, fractions, expected):
        from bioamla.datasets.partition import _allocate

        counts = _allocate(n, fractions)
        assert counts == expected and sum(counts) == n

    def test_zero_test_fraction_gets_no_rows(self, tmp_path):
        d = _write_dataset(tmp_path, n_per_label=5, labels=("call",), with_source=False)
        result = partition_dataset(str(d), splits=(0.5, 0.5, 0.0), mode="column")
        assert "test" not in result["splits"]

    def test_accepts_the_metadata_csv_path(self, tmp_path):
        d = _write_dataset(tmp_path, n_per_label=10)
        result = partition_dataset(str(d / "metadata.csv"), mode="column")
        assert result["dataset_dir"] == str(d)
        assert sum(result["splits"].values()) == 20


class TestBackgroundLabel:
    def test_background_present_in_all_splits(self, tmp_path):
        d = _write_dataset(
            tmp_path, n_per_label=20, labels=("call", "background"), with_source=False
        )
        partition_dataset(str(d), mode="column", stratify=False, background_label="background")
        bg_splits = {r["split"] for r in _read_split_map(d) if r["label"] == "background"}
        assert bg_splits == {"train", "val", "test"}


class TestPartitionSubdirs:
    def test_reorganizes_into_split_label_dirs(self, tmp_path):
        d = _write_dataset(tmp_path, n_per_label=20, with_source=False)
        partition_dataset(str(d), mode="subdirs")
        # Files now live under <split>/<label>/...
        for split in ("train", "val", "test"):
            assert (d / split).is_dir()
        rows = _read_split_map(d)
        for r in rows:
            assert r["file_name"].startswith(f"{r['split']}/{r['label']}/")
            assert (d / r["file_name"]).exists()
        # Original flat label dirs were cleaned up.
        assert not (d / "call").exists()


class TestErrors:
    def test_missing_metadata_raises(self, tmp_path):
        with pytest.raises(NotFoundError):
            partition_dataset(str(tmp_path))

    def test_bad_ratios_raise(self, tmp_path):
        d = _write_dataset(tmp_path, n_per_label=5, with_source=False)
        with pytest.raises(DatasetError):
            partition_dataset(str(d), splits=(0.5, 0.3, 0.3))

    def test_bad_mode_raises(self, tmp_path):
        d = _write_dataset(tmp_path, n_per_label=5, with_source=False)
        with pytest.raises(DatasetError):
            partition_dataset(str(d), mode="bogus")


def test_csv_has_split_column(tmp_path):
    d = _write_dataset(tmp_path, n_per_label=10, with_source=False)
    partition_dataset(str(d), mode="column")
    with (d / "metadata.csv").open() as f:
        header = next(csv.reader(f))
    assert "split" in header
