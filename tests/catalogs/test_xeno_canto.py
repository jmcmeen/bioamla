"""Tests for Xeno-canto catalog (query building, API key, errors; no network)."""

import pytest

from bioamla.catalogs import xeno_canto as xc
from bioamla.catalogs._models import XenoCantoSearchResult
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
        monkeypatch.setattr(
            "bioamla.common.config.get_config",
            lambda: type("C", (), {"get": lambda self, s, k, d=None: None})(),
        )
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
