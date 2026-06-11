"""Coverage tests for :mod:`bioamla.datasets.merge`.

Targets the branches not hit by ``test_datasets.py``: ``find_species_name``,
target-format fallback copy, skip-existing against pre-existing output metadata,
missing source files, and the no-metadata-dataset warning.
"""

from __future__ import annotations

import csv

import pytest

from bioamla.datasets._metadata import write_metadata_csv
from bioamla.datasets.merge import find_species_name, merge_datasets
from bioamla.exceptions import InvalidInputError, NotFoundError


def _make_dataset(root, rows: list[dict]) -> None:
    """Create a dataset dir with metadata.csv and the referenced audio files."""
    root.mkdir(parents=True, exist_ok=True)
    fieldnames = set()
    for r in rows:
        fieldnames.update(r.keys())
    write_metadata_csv(root / "metadata.csv", rows, fieldnames, merge_existing=False)
    for r in rows:
        fn = r.get("file_name")
        if fn:
            fpath = root / fn
            fpath.parent.mkdir(parents=True, exist_ok=True)
            fpath.write_bytes(b"RIFFfake")


class TestFindSpeciesName:
    def test_empty_returns_empty(self) -> None:
        assert find_species_name("", {"a", "b"}) == ""

    def test_subspecies_maps_to_shortest_prefix(self) -> None:
        cats = {"Genus species", "Genus species subspecies"}
        assert find_species_name("Genus species subspecies", cats) == "Genus species"

    def test_no_match_returns_unchanged(self) -> None:
        assert find_species_name("Unique name", {"Other"}) == "Unique name"


class TestMergeDatasets:
    def test_no_paths_raises(self, tmp_path) -> None:
        with pytest.raises(InvalidInputError, match="At least one dataset"):
            merge_datasets([], str(tmp_path / "out"), verbose=False)

    def test_unsupported_format_raises(self, tmp_path) -> None:
        with pytest.raises(InvalidInputError, match="Unsupported target format"):
            merge_datasets(
                [str(tmp_path)], str(tmp_path / "out"), target_format="xyz", verbose=False
            )

    def test_missing_source_raises(self, tmp_path) -> None:
        with pytest.raises(NotFoundError, match="does not exist"):
            merge_datasets(
                [str(tmp_path / "nope")], str(tmp_path / "out"), verbose=False
            )

    def test_basic_merge_organized_by_category(self, tmp_path) -> None:
        ds = tmp_path / "ds1"
        _make_dataset(
            ds,
            [
                {"file_name": "a.wav", "label": "Robin"},
                {"file_name": "b.wav", "label": "Robin"},
            ],
        )
        out = tmp_path / "out"
        stats = merge_datasets(
            [str(ds)], str(out), organize_by_category=True, verbose=True
        )
        assert stats["datasets_merged"] == 1
        assert stats["files_copied"] == 2
        assert (out / "robin" / "a.wav").exists()  # sanitize_filename lowercases

    def test_merge_without_organize(self, tmp_path) -> None:
        ds = tmp_path / "ds1"
        _make_dataset(ds, [{"file_name": "a.wav", "label": "Robin"}])
        out = tmp_path / "out"
        stats = merge_datasets(
            [str(ds)], str(out), organize_by_category=False, verbose=False
        )
        assert (out / "a.wav").exists()
        assert stats["files_copied"] == 1

    def test_target_format_fallback_copy(self, tmp_path) -> None:
        # No real converter exists, so a format change falls back to copy with the
        # new extension; the row is recorded with an attr_note.
        ds = tmp_path / "ds1"
        _make_dataset(ds, [{"file_name": "a.mp3", "label": "Robin"}])
        out = tmp_path / "out"
        stats = merge_datasets(
            [str(ds)],
            str(out),
            organize_by_category=False,
            target_format="wav",
            verbose=True,
        )
        # Falls back to a plain copy (no converter), counted as copied.
        assert stats["files_copied"] == 1
        assert (out / "a.wav").exists()

    def test_skip_existing_against_prior_output(self, tmp_path) -> None:
        ds = tmp_path / "ds1"
        _make_dataset(ds, [{"file_name": "a.wav", "label": "Robin"}])
        out = tmp_path / "out"
        # First merge populates the output.
        merge_datasets([str(ds)], str(out), organize_by_category=False, verbose=False)
        # Second merge should skip the already-present file.
        stats = merge_datasets(
            [str(ds)], str(out), organize_by_category=False, skip_existing=True, verbose=True
        )
        assert stats["files_skipped"] == 1
        assert stats["files_copied"] == 0

    def test_missing_metadata_dataset_warns_and_skips(self, tmp_path) -> None:
        empty = tmp_path / "empty"
        empty.mkdir()
        out = tmp_path / "out"
        stats = merge_datasets([str(empty)], str(out), verbose=True)
        assert stats["datasets_merged"] == 0

    def test_missing_source_file_warns(self, tmp_path) -> None:
        ds = tmp_path / "ds1"
        ds.mkdir()
        # Metadata references a file that doesn't exist on disk.
        write_metadata_csv(
            ds / "metadata.csv",
            [{"file_name": "ghost.wav", "label": "Robin"}],
            {"file_name", "label"},
            merge_existing=False,
        )
        out = tmp_path / "out"
        stats = merge_datasets(
            [str(ds)], str(out), organize_by_category=False, verbose=True
        )
        # Nothing copied because the source file is missing.
        assert stats["files_copied"] == 0

    def test_metadata_written_to_output(self, tmp_path) -> None:
        ds = tmp_path / "ds1"
        _make_dataset(ds, [{"file_name": "a.wav", "label": "Robin"}])
        out = tmp_path / "out"
        merge_datasets([str(ds)], str(out), organize_by_category=False, verbose=False)
        meta = out / "metadata.csv"
        assert meta.exists()
        rows = list(csv.DictReader(meta.read_text().splitlines()))
        assert any(r["file_name"] == "a.wav" for r in rows)
