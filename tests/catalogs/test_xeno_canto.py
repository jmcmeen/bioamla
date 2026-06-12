"""Tests for Xeno-canto catalog (query building, API key, errors; no network)."""

import pytest

from bioamla.catalogs import xeno_canto as xc
from bioamla.catalogs._models import (
    XCRecording,
    XenoCantoDownloadResult,
    XenoCantoSearchResult,
)
from bioamla.exceptions import CatalogError, InvalidInputError


class TestQueryBuilding:
    def test_scientific_name_becomes_tagged_query(self) -> None:
        q = xc._build_query_string(species="Turdus migratorius")
        assert "gen:Turdus" in q
        assert "sp:migratorius" in q

    def test_single_name_uses_en_tag(self) -> None:
        q = xc._build_query_string(species="robin")
        assert q == "en:robin"

    def test_filters_combine(self) -> None:
        q = xc._build_query_string(genus="Turdus", country="United States", quality="A")
        assert "gen:Turdus" in q
        # Multi-word values must be double-quoted for the v3 API (else HTTP 400).
        assert 'cnt:"United States"' in q
        assert "q:A" in q

    def test_multiword_value_is_quoted(self) -> None:
        assert xc._tag("cnt", "United States") == 'cnt:"United States"'

    def test_single_word_value_is_unquoted(self) -> None:
        assert xc._tag("q", "A") == "q:A"

    def test_raw_query_overrides(self) -> None:
        assert xc._build_query_string(species="x", query="raw:value") == "raw:value"

    def test_box_formatting(self) -> None:
        q = xc._build_query_string(box=(1.0, 2.0, 3.0, 4.0))
        # box:lat_min,lon_min,lat_max,lon_max
        assert "box:1.0,3.0,2.0,4.0" in q

    def test_empty_returns_empty_string(self) -> None:
        assert xc._build_query_string() == ""


class TestApiKeyResolution:
    def test_runtime_key_highest_priority(self, monkeypatch) -> None:
        monkeypatch.delenv("XC_API_KEY", raising=False)
        xc.set_xc_api_key("runtime")
        try:
            assert xc.get_xc_api_key() == "runtime"
        finally:
            xc.set_xc_api_key(None)  # reset

    def test_env_key_used_when_no_runtime(self, monkeypatch) -> None:
        xc.set_xc_api_key(None)
        monkeypatch.setenv("XC_API_KEY", "env-key")
        assert xc.get_xc_api_key() == "env-key"

    def test_none_when_unset(self, monkeypatch) -> None:
        xc.set_xc_api_key(None)
        monkeypatch.delenv("XC_API_KEY", raising=False)
        assert xc.get_xc_api_key() is None


class TestSearchValidation:
    def test_no_params_raises_invalid_input(self, monkeypatch) -> None:
        monkeypatch.setattr(xc, "get_xc_api_key", lambda: "key")
        with pytest.raises(InvalidInputError):
            xc.search()

    def test_missing_api_key_raises_invalid_input(self, monkeypatch) -> None:
        monkeypatch.setattr(xc, "get_xc_api_key", lambda: None)
        with pytest.raises(InvalidInputError):
            xc.search(species="Turdus migratorius")

    def test_get_recording_missing_key_raises(self, monkeypatch) -> None:
        monkeypatch.setattr(xc, "get_xc_api_key", lambda: None)
        with pytest.raises(InvalidInputError):
            xc.get_recording("12345")

    def test_count_missing_key_raises(self, monkeypatch) -> None:
        monkeypatch.setattr(xc, "get_xc_api_key", lambda: None)
        with pytest.raises(InvalidInputError):
            xc.get_species_recordings_count("Turdus migratorius")


class TestSearchSuccess:
    def test_search_parses_recordings(self, monkeypatch) -> None:
        monkeypatch.setattr(xc, "get_xc_api_key", lambda: "key")
        fake_response = {
            "numPages": 1,
            "recordings": [
                {"id": "1", "gen": "Turdus", "sp": "migratorius", "file": "u"},
            ],
        }
        monkeypatch.setattr(xc._client, "get", lambda *a, **k: fake_response)
        result = xc.search(species="Turdus migratorius", max_results=10)
        assert isinstance(result, XenoCantoSearchResult)
        assert result.total_results == 1
        assert result.recordings[0].scientific_name == "Turdus migratorius"

    def test_search_api_failure_raises_catalog_error(self, monkeypatch) -> None:
        monkeypatch.setattr(xc, "get_xc_api_key", lambda: "key")

        def boom(*a, **k):
            raise RuntimeError("boom")

        monkeypatch.setattr(xc._client, "get", boom)
        with pytest.raises(CatalogError):
            xc.search(species="Turdus migratorius")


@pytest.fixture(autouse=True)
def _reset_runtime_key():
    xc.set_xc_api_key(None)
    yield
    xc.set_xc_api_key(None)


def _rec(rid="1", sci="Turdus migratorius", url="http://x/a.mp3"):
    return XCRecording(
        id=rid,
        scientific_name=sci,
        common_name="American Robin",
        quality="A",
        sound_type="song",
        length="0:30",
        location="loc",
        country="US",
        recordist="alice",
        url="http://xc/1",
        download_url=url,
        license="cc-by",
    )


class TestQueryBuildingBranches:
    def test_all_tags(self) -> None:
        q = xc._build_query_string(
            genus="Turdus",
            recordist="alice",
            location="park",
            sound_type="song",
            latitude=1.0,
            longitude=2.0,
            since="2024-01-01",
            year=2024,
            month=5,
        )
        assert "gen:Turdus" in q
        assert "rec:alice" in q
        assert "loc:park" in q
        assert "type:song" in q
        assert "lat:1.0" in q
        assert "lon:2.0" in q
        assert "since:2024-01-01" in q
        assert "year:2024" in q
        assert "month:5" in q


class TestSearchPagination:
    def test_paginates_until_max_results(self, monkeypatch) -> None:
        monkeypatch.setattr(xc, "get_xc_api_key", lambda: "key")
        monkeypatch.setattr(xc.time, "sleep", lambda s: None)

        pages = {
            1: {"numPages": 2, "recordings": [{"id": "1", "gen": "T", "sp": "a", "file": "u"}]},
            2: {"numPages": 2, "recordings": [{"id": "2", "gen": "T", "sp": "b", "file": "u"}]},
        }
        monkeypatch.setattr(xc._client, "get", lambda url, params=None: pages[params["page"]])
        result = xc.search(species="Turdus migratorius")
        assert result.total_results == 2

    def test_max_results_truncates(self, monkeypatch) -> None:
        monkeypatch.setattr(xc, "get_xc_api_key", lambda: "key")
        resp = {
            "numPages": 1,
            "recordings": [
                {"id": "1", "gen": "T", "sp": "a", "file": "u"},
                {"id": "2", "gen": "T", "sp": "b", "file": "u"},
                {"id": "3", "gen": "T", "sp": "c", "file": "u"},
            ],
        }
        monkeypatch.setattr(xc._client, "get", lambda *a, **k: resp)
        result = xc.search(species="Turdus migratorius", max_results=2)
        assert result.total_results == 2


class TestGetRecording:
    def test_found(self, monkeypatch) -> None:
        monkeypatch.setattr(xc, "get_xc_api_key", lambda: "key")
        monkeypatch.setattr(
            xc._client,
            "get",
            lambda *a, **k: {"recordings": [{"id": "12345", "gen": "T", "sp": "m"}]},
        )
        rec = xc.get_recording("12345")
        assert rec.id == "12345"

    def test_not_found_raises(self, monkeypatch) -> None:
        monkeypatch.setattr(xc, "get_xc_api_key", lambda: "key")
        monkeypatch.setattr(xc._client, "get", lambda *a, **k: {"recordings": []})
        with pytest.raises(CatalogError):
            xc.get_recording("12345")

    def test_api_failure_raises(self, monkeypatch) -> None:
        monkeypatch.setattr(xc, "get_xc_api_key", lambda: "key")

        def boom(*a, **k):
            raise RuntimeError("net")

        monkeypatch.setattr(xc._client, "get", boom)
        with pytest.raises(CatalogError):
            xc.get_recording("12345")


class TestDownloadRecording:
    def test_downloads_organized(self, tmp_path, monkeypatch) -> None:
        monkeypatch.setattr(xc._client, "download", lambda url, fp: None)
        path = xc.download_recording(_rec(), tmp_path)
        assert path.parent.name == "turdus_migratorius"
        assert path.name.startswith("XC1_")

    def test_custom_filename_no_organize(self, tmp_path, monkeypatch) -> None:
        monkeypatch.setattr(xc._client, "download", lambda url, fp: None)
        path = xc.download_recording(_rec(), tmp_path, filename="c.mp3", organize_by_species=False)
        assert path.name == "c.mp3"

    def test_string_id_fetches(self, tmp_path, monkeypatch) -> None:
        monkeypatch.setattr(xc, "get_xc_api_key", lambda: "key")
        monkeypatch.setattr(
            xc._client, "get", lambda *a, **k: {"recordings": [{"id": "5", "gen": "T", "sp": "m"}]}
        )
        # record has no file -> download_url empty -> raises
        with pytest.raises(CatalogError):
            xc.download_recording("5", tmp_path)

    def test_no_url_raises(self, tmp_path) -> None:
        with pytest.raises(CatalogError):
            xc.download_recording(_rec(url=""), tmp_path)

    def test_download_failure_raises(self, tmp_path, monkeypatch) -> None:
        def boom(url, fp):
            raise RuntimeError("net")

        monkeypatch.setattr(xc._client, "download", boom)
        with pytest.raises(CatalogError):
            xc.download_recording(_rec(), tmp_path)


class TestDownload:
    def test_no_recordings_returns_empty(self, tmp_path, monkeypatch) -> None:
        monkeypatch.setattr(xc, "get_xc_api_key", lambda: "key")
        monkeypatch.setattr(xc._client, "get", lambda *a, **k: {"numPages": 1, "recordings": []})
        result = xc.download(species="Turdus migratorius", output_dir=str(tmp_path))
        assert isinstance(result, XenoCantoDownloadResult)
        assert result.total == 0

    def test_full_download_writes_metadata(self, tmp_path, monkeypatch) -> None:
        monkeypatch.setattr(xc, "get_xc_api_key", lambda: "key")
        resp = {
            "numPages": 1,
            "recordings": [
                {"id": "1", "gen": "Turdus", "sp": "migratorius", "file": "http://x/1.mp3"},
            ],
        }
        monkeypatch.setattr(xc._client, "get", lambda *a, **k: resp)
        monkeypatch.setattr(xc._client, "download", lambda url, fp: None)
        monkeypatch.setattr(xc.time, "sleep", lambda s: None)
        result = xc.download(species="Turdus migratorius", output_dir=str(tmp_path))
        assert result.downloaded == 1
        assert (tmp_path / "metadata.csv").exists()

    def test_per_recording_failure(self, tmp_path, monkeypatch) -> None:
        monkeypatch.setattr(xc, "get_xc_api_key", lambda: "key")
        resp = {
            "numPages": 1,
            "recordings": [
                {"id": "1", "gen": "Turdus", "sp": "migratorius", "file": "http://x/1.mp3"}
            ],
        }
        monkeypatch.setattr(xc._client, "get", lambda *a, **k: resp)

        def boom(url, fp):
            raise RuntimeError("net")

        monkeypatch.setattr(xc._client, "download", boom)
        monkeypatch.setattr(xc.time, "sleep", lambda s: None)
        result = xc.download(species="Turdus migratorius", output_dir=str(tmp_path))
        assert result.failed == 1

    def test_download_invalid_input_propagates(self, tmp_path, monkeypatch) -> None:
        monkeypatch.setattr(xc, "get_xc_api_key", lambda: None)
        with pytest.raises(InvalidInputError):
            xc.download(species="Turdus migratorius", output_dir=str(tmp_path))


class TestSpeciesCount:
    def test_two_word_query(self, monkeypatch) -> None:
        monkeypatch.setattr(xc, "get_xc_api_key", lambda: "key")
        monkeypatch.setattr(xc._client, "get", lambda *a, **k: {"numRecordings": "37"})
        assert xc.get_species_recordings_count("Turdus migratorius") == 37

    def test_single_word_query(self, monkeypatch) -> None:
        monkeypatch.setattr(xc, "get_xc_api_key", lambda: "key")
        monkeypatch.setattr(xc._client, "get", lambda *a, **k: {"numRecordings": 5})
        assert xc.get_species_recordings_count("robin") == 5

    def test_failure_raises(self, monkeypatch) -> None:
        monkeypatch.setattr(xc, "get_xc_api_key", lambda: "key")

        def boom(*a, **k):
            raise RuntimeError("x")

        monkeypatch.setattr(xc._client, "get", boom)
        with pytest.raises(CatalogError):
            xc.get_species_recordings_count("robin")


class TestSearchByLocation:
    def test_builds_box_query(self, monkeypatch) -> None:
        monkeypatch.setattr(xc, "get_xc_api_key", lambda: "key")
        captured = {}

        def fake_get(url, params=None):
            captured.update(params or {})
            return {"numPages": 1, "recordings": []}

        monkeypatch.setattr(xc._client, "get", fake_get)
        result = xc.search_by_location(40.0, -100.0, radius_km=50, max_results=10)
        assert result.total_results == 0
        assert "box:" in captured["query"]
