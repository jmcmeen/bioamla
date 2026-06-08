"""Tests for catalog data models (API-response parsing, no network)."""

from bioamla.catalogs import (
    EBirdObservation,
    MLRecording,
    SpeciesInfo,
    XCRecording,
)
from bioamla.catalogs._models import ML_ASSET_URL, EBirdHotspot


class TestEBirdObservation:
    def test_from_api_response_maps_fields(self) -> None:
        obs = EBirdObservation.from_api_response({
            "speciesCode": "amerob",
            "comName": "American Robin",
            "sciName": "Turdus migratorius",
            "locId": "L99",
            "locName": "Central Park",
            "obsDt": "2024-05-01",
            "howMany": 3,
            "lat": 40.7,
            "lng": -74.0,
        })
        assert obs.species_code == "amerob"
        assert obs.common_name == "American Robin"
        assert obs.scientific_name == "Turdus migratorius"
        assert obs.how_many == 3
        assert obs.latitude == 40.7

    def test_from_api_response_defaults(self) -> None:
        obs = EBirdObservation.from_api_response({})
        assert obs.species_code == ""
        assert obs.observation_valid is True
        assert obs.how_many is None

    def test_to_dict_roundtrip(self) -> None:
        obs = EBirdObservation.from_api_response({"speciesCode": "x", "comName": "Y"})
        d = obs.to_dict()
        assert d["species_code"] == "x"
        assert d["common_name"] == "Y"


class TestEBirdHotspot:
    def test_from_api_response(self) -> None:
        hs = EBirdHotspot.from_api_response({
            "locId": "L1",
            "locName": "Marsh",
            "countryCode": "US",
            "subnational1Code": "US-NY",
            "lat": 1.0,
            "lng": 2.0,
        })
        assert hs.loc_id == "L1"
        assert hs.latitude == 1.0
        assert hs.longitude == 2.0


class TestSpeciesInfo:
    def test_from_ebird_response_splits_name(self) -> None:
        info = SpeciesInfo.from_ebird_response({
            "sciName": "Turdus migratorius",
            "comName": "American Robin",
            "speciesCode": "amerob",
            "familyComName": "Thrushes",
            "order": "Passeriformes",
        })
        assert info.genus == "Turdus"
        assert info.species == "migratorius"
        assert info.source == "ebird"

    def test_from_inat_response(self) -> None:
        info = SpeciesInfo.from_inat_response({
            "name": "Strix varia",
            "preferred_common_name": "Barred Owl",
            "id": 19893,
            "rank": "species",
        })
        assert info.scientific_name == "Strix varia"
        assert info.taxon_id == 19893
        assert info.source == "inat"


class TestMLRecording:
    def test_from_api_response_builds_download_url(self) -> None:
        rec = MLRecording.from_api_response({"assetId": "12345", "rating": "4.0"})
        assert rec.asset_id == "12345"
        assert rec.rating == 4
        assert rec.download_url == f"{ML_ASSET_URL}/12345"

    def test_get_download_url_prefers_explicit(self) -> None:
        rec = MLRecording.from_api_response(
            {"assetId": "1", "downloadUrl": "https://example.com/a.mp3"}
        )
        assert rec.get_download_url() == "https://example.com/a.mp3"

    def test_rating_handles_none(self) -> None:
        rec = MLRecording.from_api_response({"assetId": "1", "rating": None})
        assert rec.rating == 0


class TestXCRecording:
    def test_from_api_response_builds_scientific_name(self) -> None:
        rec = XCRecording.from_api_response({
            "id": "777",
            "gen": "Turdus",
            "sp": "migratorius",
            "ssp": "",
            "en": "American Robin",
            "lat": "40.7",
            "lng": "-74.0",
            "file": "https://example.com/XC777.mp3",
        })
        assert rec.scientific_name == "Turdus migratorius"
        assert rec.latitude == 40.7
        assert rec.download_url == "https://example.com/XC777.mp3"

    def test_subspecies_included(self) -> None:
        rec = XCRecording.from_api_response({"gen": "A", "sp": "b", "ssp": "c"})
        assert rec.scientific_name == "A b c"

    def test_missing_coords_are_none(self) -> None:
        rec = XCRecording.from_api_response({"gen": "A", "sp": "b"})
        assert rec.latitude is None
        assert rec.longitude is None
