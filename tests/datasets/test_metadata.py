"""Tests for the canonical metadata-CSV schema, normalization, and I/O."""

from __future__ import annotations

import csv

import pytest

from bioamla.datasets._metadata import (
    ATTRIBUTION_FIELDS,
    CORE_FIELDS,
    REQUIRED_FIELDS,
    get_existing_observation_ids,
    normalize_catalog_row,
    read_metadata_csv,
    write_metadata_csv,
)


class TestNormalizeCatalogRow:
    def test_xeno_canto_maps_to_canonical(self) -> None:
        row = {
            "file_name": "Turdus_migratorius/XC1_turdus.mp3",
            "xc_id": "1",
            "scientific_name": "Turdus migratorius",
            "common_name": "American Robin",
            "recordist": "Jane Doe",
            "url": "https://xeno-canto.org/1",
            "license": "cc-by",
            "quality": "A",
        }
        out = normalize_catalog_row(row, "xeno_canto")
        assert out["source"] == "xeno_canto"
        assert out["label"] == "Turdus migratorius"  # derived from scientific_name
        assert out["attribution"] == "Jane Doe"  # recordist -> attribution
        assert out["attr_url"] == "https://xeno-canto.org/1"  # url -> attr_url
        assert "recordist" not in out and "url" not in out  # remapped keys removed
        assert out["quality"] == "A"  # source-specific extra preserved
        assert out["license"] == "cc-by"  # core field untouched

    def test_macaulay_contributor_to_attribution(self) -> None:
        out = normalize_catalog_row(
            {"file_name": "f.wav", "scientific_name": "Corvus corax", "contributor": "Bob"},
            "macaulay",
        )
        assert out["source"] == "macaulay"
        assert out["attribution"] == "Bob"
        assert out["label"] == "Corvus corax"

    def test_existing_label_not_overwritten(self) -> None:
        out = normalize_catalog_row(
            {"file_name": "f.wav", "label": "explicit", "scientific_name": "X y"}, "inaturalist"
        )
        assert out["label"] == "explicit"


class TestWriteOrdering:
    def test_core_fields_lead_then_extras_sorted(self, tmp_path) -> None:
        path = tmp_path / "metadata.csv"
        rows = [
            normalize_catalog_row(
                {
                    "file_name": "a.wav",
                    "scientific_name": "Turdus migratorius",
                    "recordist": "Jane",
                    "xc_id": "1",
                    "quality": "A",
                },
                "xeno_canto",
            )
        ]
        write_metadata_csv(path, rows, merge_existing=False)
        header = path.read_text(encoding="utf-8").splitlines()[0].split(",")

        assert header[0] == "file_name"
        # Every present core/attribution column precedes every extra column.
        present_core = [c for c in CORE_FIELDS + ATTRIBUTION_FIELDS if c in header]
        extras = [c for c in header if c not in CORE_FIELDS + ATTRIBUTION_FIELDS]
        assert header[: len(present_core)] == present_core
        assert extras == sorted(extras)  # remainder alphabetical
        assert "xc_id" in extras and "quality" in extras

    def test_roundtrip_preserves_values(self, tmp_path) -> None:
        path = tmp_path / "metadata.csv"
        rows = [normalize_catalog_row({"file_name": "a.wav", "scientific_name": "X y"}, "macaulay")]
        write_metadata_csv(path, rows, merge_existing=False)
        read_rows, fieldnames = read_metadata_csv(path)
        assert read_rows[0]["file_name"] == "a.wav"
        assert read_rows[0]["source"] == "macaulay"
        assert read_rows[0]["label"] == "X y"


class TestReadMetadataCsv:
    def test_missing_file_returns_empty(self, tmp_path) -> None:
        rows, fieldnames = read_metadata_csv(tmp_path / "nope.csv")
        assert rows == []
        assert fieldnames == set()

    def test_read_error_logged_returns_empty(self, tmp_path, monkeypatch) -> None:
        path = tmp_path / "m.csv"
        path.write_text("file_name,label\na.wav,x\n")

        def _boom(*a, **k):
            raise OSError("disk gone")

        monkeypatch.setattr("pathlib.Path.open", _boom)
        rows, fieldnames = read_metadata_csv(path)
        assert rows == []
        assert fieldnames == set()


class TestWriteMetadataCsv:
    def test_empty_rows_merge_existing_returns_zero(self, tmp_path) -> None:
        path = tmp_path / "m.csv"
        assert write_metadata_csv(path, [], merge_existing=True) == 0
        assert not path.exists()  # nothing written

    def test_empty_rows_no_merge_writes_header(self, tmp_path) -> None:
        path = tmp_path / "m.csv"
        assert write_metadata_csv(path, [], merge_existing=False) == 0
        header = path.read_text().splitlines()[0].split(",")
        assert header == REQUIRED_FIELDS

    def test_fieldnames_inferred_when_none(self, tmp_path) -> None:
        path = tmp_path / "m.csv"
        rows = [{"file_name": "a.wav", "label": "x"}]
        n = write_metadata_csv(path, rows, fieldnames=None, merge_existing=False)
        assert n == 1
        read_rows, _ = read_metadata_csv(path)
        assert read_rows[0]["file_name"] == "a.wav"

    def test_merge_dedup_by_file_name(self, tmp_path) -> None:
        path = tmp_path / "m.csv"
        write_metadata_csv(
            path,
            [{"file_name": "a.wav", "label": "x"}],
            {"file_name", "label"},
            merge_existing=False,
        )
        # Merge: one duplicate (a.wav) + one new (b.wav).
        n = write_metadata_csv(
            path,
            [
                {"file_name": "a.wav", "label": "x"},
                {"file_name": "b.wav", "label": "y"},
            ],
            {"file_name", "label"},
            merge_existing=True,
        )
        assert n == 2  # a.wav deduped, b.wav added
        rows, _ = read_metadata_csv(path)
        names = sorted(r["file_name"] for r in rows)
        assert names == ["a.wav", "b.wav"]

    def test_optional_field_mismatch_warns_and_drops(self, tmp_path) -> None:
        path = tmp_path / "m.csv"
        # Existing rows carry an optional iNat field.
        write_metadata_csv(
            path,
            [{"file_name": "a.wav", "label": "x", "observation_id": "111"}],
            {"file_name", "label", "observation_id"},
            merge_existing=False,
        )
        # New rows lack the optional field -> mismatch -> warning + column drop.
        with pytest.warns(UserWarning, match="Optional metadata mismatch"):
            write_metadata_csv(
                path,
                [{"file_name": "b.wav", "label": "y"}],
                {"file_name", "label"},
                merge_existing=True,
            )
        _, fieldnames = read_metadata_csv(path)
        assert "observation_id" not in fieldnames


class TestGetExistingObservationIds:
    def test_missing_file_returns_empty(self, tmp_path) -> None:
        assert get_existing_observation_ids(tmp_path / "nope.csv") == set()

    def test_parses_inat_filenames(self, tmp_path) -> None:
        path = tmp_path / "m.csv"
        with path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["file_name"])
            writer.writeheader()
            writer.writerow({"file_name": "inat_123_sound_456.wav"})
            writer.writerow({"file_name": "inat_789_sound_1011.mp3"})
            writer.writerow({"file_name": "random.wav"})  # ignored
            writer.writerow({"file_name": "inat_bad_sound_xx.wav"})  # parse error, skipped
        ids = get_existing_observation_ids(path)
        assert (123, 456) in ids
        assert (789, 1011) in ids
        assert len(ids) == 2
