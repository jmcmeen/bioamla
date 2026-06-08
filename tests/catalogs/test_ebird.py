"""Tests for eBird catalog (param validation, error mapping, no network)."""

import pytest

from bioamla.catalogs._models import NearbyResult
from bioamla.catalogs.ebird import EBirdService, match_detections_to_ebird
from bioamla.exceptions import CatalogError, InvalidInputError


class TestApiKeyResolution:
    def test_missing_api_key_raises_invalid_input(self, monkeypatch) -> None:
        monkeypatch.delenv("EBIRD_API_KEY", raising=False)
        service = EBirdService()
        with pytest.raises(InvalidInputError):
            service._get_api_key()

    def test_explicit_key_used(self) -> None:
        service = EBirdService(api_key="explicit")
        assert service._get_api_key() == "explicit"

    def test_env_key_used(self, monkeypatch) -> None:
        monkeypatch.setenv("EBIRD_API_KEY", "from_env")
        service = EBirdService()
        assert service._get_api_key() == "from_env"


class TestErrorMapping:
    def test_request_failure_raises_catalog_error(self, monkeypatch) -> None:
        service = EBirdService(api_key="k")

        def boom(endpoint, params=None):
            raise RuntimeError("network down")

        monkeypatch.setattr(service, "_request", boom)
        with pytest.raises(CatalogError):
            service.get_nearby(lat=1.0, lng=2.0)

    def test_get_recent_observations_maps_error(self, monkeypatch) -> None:
        service = EBirdService(api_key="k")
        monkeypatch.setattr(
            service, "_request", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("x"))
        )
        with pytest.raises(CatalogError):
            service.get_recent_observations("US-CA")


class TestParsing:
    def test_get_nearby_parses_observations(self, monkeypatch) -> None:
        service = EBirdService(api_key="k")
        fake = [
            {"speciesCode": "amerob", "comName": "American Robin"},
            {"speciesCode": "barswa", "comName": "Barn Swallow"},
        ]
        monkeypatch.setattr(service, "_request", lambda *a, **k: fake)
        result = service.get_nearby(lat=1.0, lng=2.0)
        assert isinstance(result, NearbyResult)
        assert result.total_count == 2
        assert result.observations[0].species_code == "amerob"

    def test_validate_species_uses_nearby(self, monkeypatch) -> None:
        service = EBirdService(api_key="k")
        fake = [{"speciesCode": "amerob", "obsDt": "2024-05-01"}]
        monkeypatch.setattr(service, "_request", lambda *a, **k: fake)
        validation = service.validate_species("amerob", lat=1.0, lng=2.0)
        assert validation.is_valid is True
        assert validation.nearby_observations == 1

    def test_validate_species_not_present(self, monkeypatch) -> None:
        service = EBirdService(api_key="k")
        monkeypatch.setattr(service, "_request", lambda *a, **k: [{"speciesCode": "other"}])
        validation = service.validate_species("amerob", lat=1.0, lng=2.0)
        assert validation.is_valid is False


class TestMatchDetections:
    def test_match_detections_validates_against_nearby(self, monkeypatch) -> None:
        service = EBirdService(api_key="k")
        monkeypatch.setattr(service, "_request", lambda *a, **k: [{"speciesCode": "amerob"}])
        detections = [{"label": "American Robin", "confidence": 0.9}]
        out = match_detections_to_ebird(
            detections,
            service,
            latitude=1.0,
            longitude=2.0,
            species_mapping={"American Robin": "amerob"},
        )
        assert out[0]["ebird_validated"] is True
        assert out[0]["ebird_species_code"] == "amerob"
