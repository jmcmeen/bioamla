# tests/controllers/test_inaturalist.py
"""
Tests for INaturalistController.
"""


import pytest

from bioamla.controllers.inaturalist import (
    INaturalistController,
    SearchResult,
    TaxonInfo,
)


class TestINaturalistController:
    """Tests for INaturalistController."""

    @pytest.fixture
    def controller(self):
        return INaturalistController()

    def test_search_success(self, controller, mocker):
        """Test that searching for observations succeeds."""
        mock_search = mocker.patch("bioamla.core.services.inaturalist.search_inat_sounds")
        mock_search.return_value = [
            {"id": 1, "species": "Strix varia"},
            {"id": 2, "species": "Strix varia"},
        ]

        result = controller.search(taxon_name="Strix varia", per_page=10)

        assert result.success is True
        assert result.data is not None
        assert isinstance(result.data, SearchResult)
        assert result.data.total_results == 2

    def test_search_no_taxon_succeeds(self, controller, mocker):
        """Test that search without specific taxon succeeds."""
        mock_search = mocker.patch("bioamla.core.services.inaturalist.search_inat_sounds")
        mock_search.return_value = []

        result = controller.search(per_page=10)

        assert result.success is True
        assert result.data.total_results == 0


class TestTaxaOperations:
    """Tests for taxa-related operations."""

    @pytest.fixture
    def controller(self):
        return INaturalistController()

    def test_get_taxa_requires_place_or_project(self, controller):
        """Test that get_taxa requires at least place_id or project_id."""
        result = controller.get_taxa()

        assert result.success is False
        assert "place_id or project_id" in result.error

    def test_get_taxa_with_project_success(self, controller, mocker):
        """Test that getting taxa with project_id succeeds."""
        mock_get_taxa = mocker.patch("bioamla.core.services.inaturalist.get_taxa")
        mock_get_taxa.return_value = [
            {
                "taxon_id": 1,
                "name": "Strix varia",
                "common_name": "Barred Owl",
                "observation_count": 100,
            },
        ]

        result = controller.get_taxa(project_id="test-project")

        assert result.success is True
        assert result.data is not None
        assert len(result.data) == 1
        assert isinstance(result.data[0], TaxonInfo)

    def test_get_common_taxa_returns_dict(self, controller):
        """Test that get_common_taxa returns a dictionary."""
        result = controller.get_common_taxa()

        assert result.success is True
        assert result.data is not None
        assert "Aves" in result.data
        assert "Amphibia" in result.data


class TestExportOperations:
    """Tests for export operations."""

    @pytest.fixture
    def controller(self):
        return INaturalistController()

    def test_export_taxa_csv_success(self, controller, tmp_path):
        """Test that exporting taxa to CSV succeeds."""
        taxa = [
            TaxonInfo(
                taxon_id=1, name="Strix varia", common_name="Barred Owl", observation_count=100
            ),
            TaxonInfo(
                taxon_id=2,
                name="Bubo virginianus",
                common_name="Great Horned Owl",
                observation_count=50,
            ),
        ]

        output_path = str(tmp_path / "taxa.csv")
        result = controller.export_taxa_csv(taxa, output_path)

        assert result.success is True
        assert result.data == output_path
        assert (tmp_path / "taxa.csv").exists()


class TestLoadOperations:
    """Tests for loading operations."""

    @pytest.fixture
    def controller(self):
        return INaturalistController()

    def test_load_taxon_ids_success(self, controller, tmp_path, mocker):
        """Test that loading taxon IDs from CSV succeeds."""
        csv_file = tmp_path / "taxon_ids.csv"
        csv_file.write_text("taxon_id\n1\n2\n3\n")

        mock_load = mocker.patch("bioamla.core.services.inaturalist.load_taxon_ids_from_csv")
        mock_load.return_value = [1, 2, 3]

        result = controller.load_taxon_ids(str(csv_file))

        assert result.success is True
        assert result.data == [1, 2, 3]
