"""Tests for species lookup/conversion (taxonomy cache, no network)."""

import pytest

from bioamla.catalogs import species
from bioamla.catalogs._models import SearchMatch, SpeciesInfo
from bioamla.exceptions import SpeciesError


@pytest.fixture
def loaded_taxonomy(monkeypatch):
    """Pre-populate the module taxonomy cache and disable network loading."""
    entry = {
        "scientific_name": "Turdus migratorius",
        "common_name": "American Robin",
        "species_code": "amerob",
        "family": "Thrushes",
        "order": "Passeriformes",
        "category": "species",
    }
    cache = {
        species._normalize_name("Turdus migratorius"): entry,
        species._normalize_name("American Robin"): entry,
        "amerob": entry,
    }
    monkeypatch.setattr(species, "_taxonomy_cache", cache)
    monkeypatch.setattr(species, "_taxonomy_loaded", True)
    # Avoid any iNaturalist network fallback.
    monkeypatch.setattr(species, "_search_inat_taxon", lambda name: None)
    yield


class TestNormalize:
    def test_strips_punctuation_and_lowercases(self) -> None:
        assert species._normalize_name("  Turdus, Migratorius!  ") == "turdus migratorius"


class TestLookup:
    def test_lookup_by_common_name(self, loaded_taxonomy) -> None:
        info = species.lookup("American Robin")
        assert isinstance(info, SpeciesInfo)
        assert info.scientific_name == "Turdus migratorius"
        assert info.genus == "Turdus"

    def test_lookup_by_code(self, loaded_taxonomy) -> None:
        info = species.lookup("amerob")
        assert info.species_code == "amerob"

    def test_lookup_missing_raises(self, loaded_taxonomy) -> None:
        with pytest.raises(SpeciesError):
            species.lookup("Nonexistent Bird")


class TestConversions:
    def test_scientific_to_common(self, loaded_taxonomy) -> None:
        assert species.scientific_to_common("Turdus migratorius") == "American Robin"

    def test_common_to_scientific(self, loaded_taxonomy) -> None:
        assert species.common_to_scientific("American Robin") == "Turdus migratorius"

    def test_scientific_to_common_missing_raises(self, loaded_taxonomy) -> None:
        with pytest.raises(SpeciesError):
            species.scientific_to_common("Unknown species", fallback_inat=False)

    def test_get_species_code(self, loaded_taxonomy) -> None:
        assert species.get_species_code("American Robin") == "amerob"

    def test_get_species_code_missing_raises(self, loaded_taxonomy) -> None:
        with pytest.raises(SpeciesError):
            species.get_species_code("Unknown")

    def test_code_to_name(self, loaded_taxonomy) -> None:
        sci, common = species.code_to_name("amerob")
        assert sci == "Turdus migratorius"
        assert common == "American Robin"

    def test_code_to_name_missing_raises(self, loaded_taxonomy) -> None:
        with pytest.raises(SpeciesError):
            species.code_to_name("zzzzzz")


class TestSearchAndValidate:
    def test_search_returns_matches(self, loaded_taxonomy) -> None:
        matches = species.search("robin", min_score=0.3)
        assert matches
        assert isinstance(matches[0], SearchMatch)
        assert matches[0].scientific_name == "Turdus migratorius"

    def test_validate_name_true(self, loaded_taxonomy) -> None:
        assert species.validate_name("American Robin") is True

    def test_validate_name_false(self, loaded_taxonomy) -> None:
        assert species.validate_name("Nope") is False


class TestBatchConvert:
    def test_batch_convert_returns_none_for_misses(self, loaded_taxonomy) -> None:
        result = species.batch_convert(["Turdus migratorius", "Unknown sp"])
        assert result["Turdus migratorius"] == "American Robin"
        assert result["Unknown sp"] is None


class TestExportTaxonomy:
    def test_export_csv(self, loaded_taxonomy, tmp_path) -> None:
        out = tmp_path / "taxa.csv"
        result = species.export_taxonomy(out, format="csv")
        assert result == out
        assert out.exists()
        contents = out.read_text()
        assert "Turdus migratorius" in contents

    def test_export_json(self, loaded_taxonomy, tmp_path) -> None:
        import json

        out = tmp_path / "taxa.json"
        species.export_taxonomy(out, format="json")
        data = json.loads(out.read_text())
        assert any(r["scientific_name"] == "Turdus migratorius" for r in data)


class TestFindSpeciesName:
    def test_subspecies_resolves_to_species(self) -> None:
        all_cats = {"Lithobates sphenocephalus", "Lithobates sphenocephalus utricularius"}
        result = species.find_species_name(
            "Lithobates sphenocephalus utricularius", all_cats
        )
        assert result == "Lithobates sphenocephalus"

    def test_no_match_returns_original(self) -> None:
        assert species.find_species_name("Strix varia", {"Strix varia"}) == "Strix varia"

    def test_empty_returns_empty(self) -> None:
        assert species.find_species_name("", set()) == ""
