"""Tests for iNaturalist catalog (helpers, validation, errors; no network)."""

import csv

import pytest

from bioamla.catalogs import inat
from bioamla.catalogs._metadata import (
    get_existing_observation_ids,
    read_metadata_csv,
    write_metadata_csv,
)
from bioamla.catalogs._models import INaturalistSearchResult
from bioamla.exceptions import CatalogError, InvalidInputError


class TestLoadTaxonIdsFromCsv:
    def test_valid_csv(self, tmp_path) -> None:
        csv_path = tmp_path / "taxa.csv"
        with csv_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["taxon_id"])
            writer.writeheader()
            writer.writerow({"taxon_id": "3"})
            writer.writerow({"taxon_id": "19893"})
        assert inat._load_taxon_ids_from_csv(csv_path) == [3, 19893]

    def test_missing_file_raises(self, tmp_path) -> None:
        with pytest.raises(InvalidInputError):
            inat._load_taxon_ids_from_csv(tmp_path / "nope.csv")

    def test_missing_column_raises(self, tmp_path) -> None:
        csv_path = tmp_path / "bad.csv"
        csv_path.write_text("name\nfoo\n")
        with pytest.raises(InvalidInputError):
            inat._load_taxon_ids_from_csv(csv_path)

    def test_no_valid_ids_raises(self, tmp_path) -> None:
        csv_path = tmp_path / "empty.csv"
        with csv_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["taxon_id"])
            writer.writeheader()
            writer.writerow({"taxon_id": ""})
        with pytest.raises(InvalidInputError):
            inat._load_taxon_ids_from_csv(csv_path)


class TestSearch:
    def test_search_parses_results(self, monkeypatch) -> None:
        monkeypatch.setattr(
            inat, "get_observations", lambda **k: {"results": [{"id": 1}, {"id": 2}]}
        )
        result = inat.search(taxon_name="Strix varia")
        assert isinstance(result, INaturalistSearchResult)
        assert result.total_results == 2

    def test_search_failure_raises_catalog_error(self, monkeypatch) -> None:
        def boom(**k):
            raise RuntimeError("api down")

        monkeypatch.setattr(inat, "get_observations", boom)
        with pytest.raises(CatalogError):
            inat.search(taxon_name="Strix varia")


class TestGetTaxa:
    def test_requires_place_or_project(self) -> None:
        with pytest.raises(InvalidInputError):
            inat.get_taxa()


class TestProjectStats:
    def test_project_not_found_raises(self, monkeypatch) -> None:
        class FakeResp:
            def raise_for_status(self):
                pass

            def json(self):
                return {"results": []}

        monkeypatch.setattr(inat.requests, "get", lambda *a, **k: FakeResp())
        with pytest.raises(CatalogError):
            inat.get_project_stats("missing-project")


class TestMetadataHelpers:
    def test_write_then_read_roundtrip(self, tmp_path) -> None:
        path = tmp_path / "metadata.csv"
        rows = [
            {
                "file_name": "inat_1_sound_2.mp3",
                "split": "train",
                "target": "",
                "label": "strix_varia",
                "attr_id": "user",
                "attr_lic": "cc-by",
                "attr_url": "http://x",
                "attr_note": "",
            }
        ]
        written = write_metadata_csv(path, rows, merge_existing=False)
        assert written == 1
        read_rows, fieldnames = read_metadata_csv(path)
        assert len(read_rows) == 1
        assert "file_name" in fieldnames
        # Required fields should come first.
        assert read_rows[0]["file_name"] == "inat_1_sound_2.mp3"

    def test_get_existing_observation_ids(self, tmp_path) -> None:
        path = tmp_path / "metadata.csv"
        write_metadata_csv(
            path,
            [
                {
                    "file_name": "strix_varia/inat_111_sound_222.mp3",
                    "split": "train",
                    "target": "",
                    "label": "strix_varia",
                    "attr_id": "u",
                    "attr_lic": "cc0",
                    "attr_url": "x",
                    "attr_note": "",
                }
            ],
            merge_existing=False,
        )
        existing = get_existing_observation_ids(path)
        assert (111, 222) in existing

    def test_read_missing_file_returns_empty(self, tmp_path) -> None:
        rows, fieldnames = read_metadata_csv(tmp_path / "nope.csv")
        assert rows == []
        assert fieldnames == set()
