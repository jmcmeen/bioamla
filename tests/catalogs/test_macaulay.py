"""Tests for Macaulay Library catalog (param validation, errors; no network)."""

import pytest

from bioamla.catalogs import macaulay as ml
from bioamla.catalogs._models import MacaulaySearchResult
from bioamla.exceptions import CatalogError, InvalidInputError


class TestSearchValidation:
    def test_no_filter_raises_invalid_input(self) -> None:
        with pytest.raises(InvalidInputError):
            ml.search()

    def test_search_with_filter_parses_results(self, monkeypatch) -> None:
        fake = {"results": {"content": [{"assetId": "1", "rating": "4"}]}}
        monkeypatch.setattr(ml._client, "get", lambda *a, **k: fake)
        result = ml.search(species_code="amerob")
        assert isinstance(result, MacaulaySearchResult)
        assert result.total_results == 1
        assert result.recordings[0].asset_id == "1"

    def test_search_api_failure_raises_catalog_error(self, monkeypatch) -> None:
        def boom(*a, **k):
            raise RuntimeError("down")

        monkeypatch.setattr(ml._client, "get", boom)
        with pytest.raises(CatalogError):
            ml.search(species_code="amerob")


class TestGetRecording:
    def test_not_found_raises_catalog_error(self, monkeypatch) -> None:
        monkeypatch.setattr(ml._client, "get", lambda *a, **k: {"results": {"content": []}})
        with pytest.raises(CatalogError):
            ml.get_recording("999")

    def test_found_returns_recording(self, monkeypatch) -> None:
        fake = {"results": {"content": [{"assetId": "7", "sciName": "Turdus migratorius"}]}}
        monkeypatch.setattr(ml._client, "get", lambda *a, **k: fake)
        rec = ml.get_recording("7")
        assert rec.asset_id == "7"


class TestGetSpeciesCount:
    def test_returns_count(self, monkeypatch) -> None:
        monkeypatch.setattr(ml._client, "get", lambda *a, **k: {"results": {"count": 42}})
        assert ml.get_species_count("amerob") == 42

    def test_failure_raises_catalog_error(self, monkeypatch) -> None:
        monkeypatch.setattr(
            ml._client, "get", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("x"))
        )
        with pytest.raises(CatalogError):
            ml.get_species_count("amerob")
