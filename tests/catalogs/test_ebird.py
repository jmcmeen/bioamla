"""Tests for eBird catalog (param validation, error mapping, no network)."""

import pytest

from bioamla.catalogs._models import (
    EBirdChecklist,
    EBirdHotspot,
    NearbyResult,
    RegionResult,
)
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


class _FakeSessionResponse:
    def __init__(self, data, raise_exc=None):
        self._data = data
        self._raise_exc = raise_exc

    def raise_for_status(self):
        if self._raise_exc is not None:
            raise self._raise_exc

    def json(self):
        return self._data


class _FakeSession:
    """Stand-in for requests.Session capturing headers and returning canned JSON."""

    def __init__(self, data, raise_exc=None):
        self.headers: dict[str, str] = {}
        self._data = data
        self._raise_exc = raise_exc
        self.last_url = None
        self.last_params = None

    def get(self, url, params=None):
        self.last_url = url
        self.last_params = params
        return _FakeSessionResponse(self._data, self._raise_exc)


def _service_with_session(monkeypatch, data, raise_exc=None):
    service = EBirdService(api_key="k")
    session = _FakeSession(data, raise_exc)
    monkeypatch.setattr(service, "_get_session", lambda: session)
    return service, session


class TestRequestWiring:
    def test_session_sets_token_header(self) -> None:
        service = EBirdService(api_key="secret")
        session = service._get_session()
        assert session.headers["X-eBirdApiToken"] == "secret"
        # Subsequent call reuses the same session.
        assert service._get_session() is session

    def test_request_raises_on_http_error(self, monkeypatch) -> None:
        service, _ = _service_with_session(monkeypatch, {}, raise_exc=RuntimeError("500"))
        with pytest.raises(CatalogError):
            service.get_species_list("US-CA")


class TestRecentObservations:
    def test_parses_results(self, monkeypatch) -> None:
        data = [
            {"speciesCode": "amerob", "comName": "American Robin"},
            {"speciesCode": "barswa", "comName": "Barn Swallow"},
        ]
        service, _ = _service_with_session(monkeypatch, data)
        result = service.get_recent_observations("US-CA")
        assert isinstance(result, RegionResult)
        assert result.total_count == 2

    def test_species_filtered_endpoint(self, monkeypatch) -> None:
        service, session = _service_with_session(monkeypatch, [])
        service.get_recent_observations("US-CA", species_code="amerob")
        assert session.last_url.endswith("data/obs/US-CA/recent/amerob")


class TestSpeciesList:
    def test_returns_codes(self, monkeypatch) -> None:
        service, _ = _service_with_session(monkeypatch, ["amerob", "barswa"])
        assert service.get_species_list("US-CA") == ["amerob", "barswa"]


class TestTaxonomy:
    def test_with_species_filter(self, monkeypatch) -> None:
        service, session = _service_with_session(monkeypatch, [{"sciName": "Turdus migratorius"}])
        out = service.get_taxonomy(species_codes=["amerob", "barswa"])
        assert out[0]["sciName"] == "Turdus migratorius"
        assert session.last_params["species"] == "amerob,barswa"

    def test_failure_maps_error(self, monkeypatch) -> None:
        service, _ = _service_with_session(monkeypatch, {}, raise_exc=RuntimeError("x"))
        with pytest.raises(CatalogError):
            service.get_taxonomy()


class TestHotspots:
    def test_parses_hotspots(self, monkeypatch) -> None:
        data = [{"locId": "L1", "locName": "Park", "lat": 1.0, "lng": 2.0}]
        service, _ = _service_with_session(monkeypatch, data)
        hotspots = service.get_hotspots("US-CA")
        assert isinstance(hotspots[0], EBirdHotspot)
        assert hotspots[0].loc_id == "L1"

    def test_failure_maps_error(self, monkeypatch) -> None:
        service, _ = _service_with_session(monkeypatch, {}, raise_exc=RuntimeError("x"))
        with pytest.raises(CatalogError):
            service.get_hotspots("US-CA")


class TestChecklist:
    def test_parses_checklist(self, monkeypatch) -> None:
        data = {
            "subId": "S123",
            "locId": "L1",
            "loc": {"name": "Park", "lat": 1.0, "lng": 2.0},
            "obsDt": "2024-05-01 08:00",
            "obsTime": "08:00",
            "durationHrs": 1.5,
            "effortDistanceKm": 2.0,
            "numObservers": 3,
            "obs": [
                {
                    "speciesCode": "amerob",
                    "species": {"comName": "American Robin", "sciName": "Turdus migratorius"},
                    "howManyStr": "2",
                }
            ],
        }
        service, _ = _service_with_session(monkeypatch, data)
        checklist = service.get_checklist("S123")
        assert isinstance(checklist, EBirdChecklist)
        assert checklist.submission_id == "S123"
        assert checklist.duration_minutes == 90
        assert checklist.species_count == 1
        assert checklist.observations[0].species_code == "amerob"

    def test_failure_maps_error(self, monkeypatch) -> None:
        service, _ = _service_with_session(monkeypatch, {}, raise_exc=RuntimeError("x"))
        with pytest.raises(CatalogError):
            service.get_checklist("S123")
