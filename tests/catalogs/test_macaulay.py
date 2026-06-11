"""Tests for Macaulay Library catalog (param validation, errors; no network)."""

import pytest

from bioamla.catalogs import macaulay as ml
from bioamla.catalogs._models import (
    MacaulayDownloadResult,
    MacaulaySearchResult,
    MLRecording,
)
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


def _rec(asset_id="100", *, catalog_id="100", sci="Turdus migratorius", url="http://x/a.mp3"):
    return MLRecording(
        asset_id=asset_id,
        catalog_id=catalog_id,
        species_code="amerob",
        common_name="American Robin",
        scientific_name=sci,
        rating=4,
        duration=10.0,
        location="loc",
        country="US",
        user_display_name="alice",
        download_url=url,
    )


class TestSearchParamBranches:
    def test_all_filters_set(self, monkeypatch) -> None:
        captured = {}

        def fake_get(url, params=None):
            captured.update(params or {})
            return {"results": {"content": []}}

        monkeypatch.setattr(ml._client, "get", fake_get)
        ml.search(
            species_code="amerob",
            scientific_name="Turdus migratorius",
            common_name="American Robin",
            region="US-CA",
            country="US",
            taxon_code="amerob",
            hotspot_code="L99",
            min_rating=3,
            year=2024,
            month=5,
            media_type="all",
        )
        assert captured["sciName"] == "Turdus migratorius"
        assert captured["commonName"] == "American Robin"
        assert captured["region"] == "US-CA"
        assert captured["country"] == "US"
        assert captured["hotspotCode"] == "L99"
        assert captured["rating"] == 3
        assert captured["year"] == 2024
        assert captured["month"] == 5
        # media_type "all" -> mediaType dropped
        assert "mediaType" not in captured


class TestDownloadRecording:
    def test_downloads_and_organizes_by_species(self, tmp_path, monkeypatch) -> None:
        monkeypatch.setattr(ml._client, "download", lambda url, fp: None)
        path = ml.download_recording(_rec(), tmp_path)
        assert path.parent.name == "turdus_migratorius"
        assert path.name.startswith("ML100_")

    def test_custom_filename_and_no_organize(self, tmp_path, monkeypatch) -> None:
        monkeypatch.setattr(ml._client, "download", lambda url, fp: None)
        path = ml.download_recording(
            _rec(), tmp_path, filename="custom.mp3", organize_by_species=False
        )
        assert path.name == "custom.mp3"
        assert path.parent == tmp_path

    def test_string_id_fetches_recording(self, tmp_path, monkeypatch) -> None:
        monkeypatch.setattr(
            ml._client,
            "get",
            lambda *a, **k: {"results": {"content": [{"assetId": "55", "sciName": "Strix varia"}]}},
        )
        monkeypatch.setattr(ml._client, "download", lambda url, fp: None)
        path = ml.download_recording("55", tmp_path)
        assert "ML" in path.name

    def test_no_download_url_raises(self, tmp_path) -> None:
        # Empty download_url AND empty asset_id so get_download_url() yields nothing.
        with pytest.raises(CatalogError):
            ml.download_recording(_rec(asset_id="", catalog_id="", url=""), tmp_path)

    def test_download_failure_raises(self, tmp_path, monkeypatch) -> None:
        def boom(url, fp):
            raise RuntimeError("net")

        monkeypatch.setattr(ml._client, "download", boom)
        with pytest.raises(CatalogError):
            ml.download_recording(_rec(), tmp_path)


class TestDownload:
    def test_no_recordings_returns_empty_result(self, tmp_path, monkeypatch) -> None:
        monkeypatch.setattr(ml._client, "get", lambda *a, **k: {"results": {"content": []}})
        result = ml.download(species_code="amerob", output_dir=str(tmp_path))
        assert isinstance(result, MacaulayDownloadResult)
        assert result.total == 0

    def test_full_download_writes_metadata(self, tmp_path, monkeypatch) -> None:
        content = [
            {"assetId": "1", "catalogId": "1", "sciName": "Turdus migratorius", "rating": "4"},
            {"assetId": "2", "catalogId": "2", "sciName": "Turdus migratorius", "rating": "3"},
        ]
        monkeypatch.setattr(ml._client, "get", lambda *a, **k: {"results": {"content": content}})
        monkeypatch.setattr(ml._client, "download", lambda url, fp: None)
        monkeypatch.setattr(ml.time, "sleep", lambda s: None)
        result = ml.download(
            species_code="amerob", max_recordings=2, output_dir=str(tmp_path), create_metadata=True
        )
        assert result.downloaded == 2
        assert (tmp_path / "metadata.csv").exists()

    def test_per_recording_failure_recorded(self, tmp_path, monkeypatch) -> None:
        content = [{"assetId": "1", "catalogId": "1", "sciName": "T m", "rating": "4"}]
        monkeypatch.setattr(ml._client, "get", lambda *a, **k: {"results": {"content": content}})

        def boom(url, fp):
            raise RuntimeError("net")

        monkeypatch.setattr(ml._client, "download", boom)
        monkeypatch.setattr(ml.time, "sleep", lambda s: None)
        result = ml.download(species_code="amerob", output_dir=str(tmp_path))
        assert result.failed == 1
        assert result.errors

    def test_download_search_failure_raises(self, tmp_path, monkeypatch) -> None:
        def boom(*a, **k):
            raise RuntimeError("api")

        monkeypatch.setattr(ml._client, "get", boom)
        with pytest.raises(CatalogError):
            ml.download(species_code="amerob", output_dir=str(tmp_path))


class TestSearchAudio:
    def test_search_audio_delegates(self, monkeypatch) -> None:
        monkeypatch.setattr(
            ml._client, "get", lambda *a, **k: {"results": {"content": [{"assetId": "9"}]}}
        )
        result = ml.search_audio(species_code="amerob")
        assert result.total_results == 1
