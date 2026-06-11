"""CLI tests for `bioamla catalogs` commands."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner

from bioamla.cli.cli import cli
from bioamla.exceptions import CatalogError


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


def test_catalogs_group_help(runner: CliRunner) -> None:
    result = runner.invoke(cli, ["catalogs", "--help"])
    assert result.exit_code == 0
    for sub in ["inat", "hf", "xc", "ml", "ebird"]:
        assert sub in result.output


# --- iNaturalist ---------------------------------------------------------


def _obs() -> dict:
    return {
        "id": 123,
        "taxon": {"name": "Rana sphenocephala", "preferred_common_name": "Leopard Frog"},
        "sounds": [{"id": 1}],
        "observed_on": "2024-01-01",
        "place_guess": "Tennessee",
    }


def test_inat_search_requires_filter(runner: CliRunner) -> None:
    result = runner.invoke(cli, ["catalogs", "inat", "search"])
    assert result.exit_code != 0
    assert "filter" in result.output.lower()


def test_inat_search_table(runner: CliRunner) -> None:
    result_obj = SimpleNamespace(observations=[_obs()])
    with patch("bioamla.catalogs.inat.search", return_value=result_obj):
        result = runner.invoke(cli, ["catalogs", "inat", "search", "--species", "Rana"])
    assert result.exit_code == 0
    assert "123" in result.output
    assert "Rana" in result.output


def test_inat_search_no_results(runner: CliRunner) -> None:
    with patch("bioamla.catalogs.inat.search", return_value=SimpleNamespace(observations=[])):
        result = runner.invoke(cli, ["catalogs", "inat", "search", "--taxon-id", "5"])
    assert result.exit_code == 0
    assert "No observations" in result.output


def test_inat_search_output_csv(runner: CliRunner, tmp_path) -> None:
    out = tmp_path / "obs.csv"
    result_obj = SimpleNamespace(observations=[_obs()])
    with patch("bioamla.catalogs.inat.search", return_value=result_obj):
        result = runner.invoke(
            cli, ["catalogs", "inat", "search", "--species", "Rana", "-o", str(out)]
        )
    assert result.exit_code == 0
    assert out.exists()
    assert "Rana sphenocephala" in out.read_text()


def test_inat_search_error(runner: CliRunner) -> None:
    with patch("bioamla.catalogs.inat.search", side_effect=CatalogError("api down")):
        result = runner.invoke(cli, ["catalogs", "inat", "search", "--species", "Rana"])
    assert result.exit_code != 0


def test_inat_stats(runner: CliRunner) -> None:
    stats = SimpleNamespace(
        title="My Project",
        url="http://x",
        project_type="collection",
        place="TN",
        created_at="2024",
        observation_count=10,
        species_count=5,
        observers_count=3,
        to_dict=lambda: {"title": "My Project"},
    )
    with patch("bioamla.catalogs.inat.get_project_stats", return_value=stats):
        result = runner.invoke(cli, ["catalogs", "inat", "stats", "myproj"])
    assert result.exit_code == 0
    assert "My Project" in result.output


def test_inat_stats_output(runner: CliRunner, tmp_path) -> None:
    out = tmp_path / "stats.json"
    stats = SimpleNamespace(to_dict=lambda: {"title": "P"})
    with patch("bioamla.catalogs.inat.get_project_stats", return_value=stats):
        result = runner.invoke(cli, ["catalogs", "inat", "stats", "p", "-o", str(out)])
    assert result.exit_code == 0
    assert out.exists()


def test_inat_stats_error(runner: CliRunner) -> None:
    with patch("bioamla.catalogs.inat.get_project_stats", side_effect=CatalogError("x")):
        result = runner.invoke(cli, ["catalogs", "inat", "stats", "p"])
    assert result.exit_code != 0


def test_inat_download(runner: CliRunner, tmp_path) -> None:
    dl = SimpleNamespace(
        total_observations=5,
        total_sounds=8,
        observations_with_multiple_sounds=2,
        skipped_existing=1,
        failed_downloads=0,
        output_dir=str(tmp_path),
        metadata_file=str(tmp_path / "metadata.csv"),
    )
    with patch("bioamla.catalogs.inat.download", return_value=dl):
        result = runner.invoke(
            cli,
            ["catalogs", "inat", "download", str(tmp_path), "-t", "1,2", "-l", "cc0"],
        )
    assert result.exit_code == 0
    assert "Download complete" in result.output


def test_inat_download_error(runner: CliRunner, tmp_path) -> None:
    with patch("bioamla.catalogs.inat.download", side_effect=CatalogError("fail")):
        result = runner.invoke(cli, ["catalogs", "inat", "download", str(tmp_path), "-n", "Rana"])
    assert result.exit_code != 0


# --- HuggingFace ---------------------------------------------------------


def test_hf_push_model(runner: CliRunner, tmp_path) -> None:
    with patch(
        "bioamla.catalogs.huggingface.push_model",
        return_value=SimpleNamespace(url="http://hub/model"),
    ):
        result = runner.invoke(
            cli, ["catalogs", "hf", "push-model", str(tmp_path), "me/model"]
        )
    assert result.exit_code == 0
    assert "Successfully pushed" in result.output


def test_hf_push_model_error(runner: CliRunner, tmp_path) -> None:
    with patch(
        "bioamla.catalogs.huggingface.push_model",
        side_effect=CatalogError("not logged in"),
    ):
        result = runner.invoke(
            cli, ["catalogs", "hf", "push-model", str(tmp_path), "me/model"]
        )
    assert result.exit_code != 0
    assert "huggingface-cli login" in result.output


def test_hf_push_dataset_no_card(runner: CliRunner, tmp_path) -> None:
    with patch(
        "bioamla.catalogs.huggingface.push_dataset",
        return_value=SimpleNamespace(url="http://hub/ds"),
    ):
        result = runner.invoke(
            cli, ["catalogs", "hf", "push-dataset", str(tmp_path), "me/ds", "--no-card"]
        )
    assert result.exit_code == 0
    assert "Successfully pushed dataset" in result.output


def test_hf_push_dataset_with_card(runner: CliRunner, tmp_path) -> None:
    with patch(
        "bioamla.catalogs.huggingface.push_dataset",
        return_value=SimpleNamespace(url="http://hub/ds"),
    ), patch(
        "bioamla.datasets.write_dataset_card", return_value=str(tmp_path / "README.md")
    ):
        result = runner.invoke(
            cli, ["catalogs", "hf", "push-dataset", str(tmp_path), "me/ds"]
        )
    assert result.exit_code == 0
    assert "dataset card" in result.output


def test_hf_pull_dataset(runner: CliRunner, tmp_path) -> None:
    pulled = SimpleNamespace(
        files_written=10,
        labels=["a", "b"],
        dest=str(tmp_path),
        metadata_file=str(tmp_path / "metadata.csv"),
    )
    with patch("bioamla.catalogs.huggingface.pull_dataset", return_value=pulled):
        result = runner.invoke(
            cli, ["catalogs", "hf", "pull-dataset", "me/ds", str(tmp_path)]
        )
    assert result.exit_code == 0
    assert "Wrote 10 clips" in result.output


def test_hf_pull_dataset_error(runner: CliRunner, tmp_path) -> None:
    with patch(
        "bioamla.catalogs.huggingface.pull_dataset",
        side_effect=CatalogError("not found"),
    ):
        result = runner.invoke(
            cli, ["catalogs", "hf", "pull-dataset", "me/ds", str(tmp_path)]
        )
    assert result.exit_code != 0


def test_hf_cache_list_empty(runner: CliRunner) -> None:
    with patch("bioamla.catalogs.huggingface.scan_cache", return_value=[]):
        result = runner.invoke(cli, ["catalogs", "hf", "cache"])
    assert result.exit_code == 0
    assert "No cached data" in result.output


def test_hf_cache_list(runner: CliRunner) -> None:
    repos = [SimpleNamespace(repo_type="dataset", repo_id="me/ds", size_bytes=2048)]
    with patch("bioamla.catalogs.huggingface.scan_cache", return_value=repos):
        result = runner.invoke(cli, ["catalogs", "hf", "cache"])
    assert result.exit_code == 0
    assert "me/ds" in result.output
    assert "--purge" in result.output


def test_hf_cache_purge(runner: CliRunner) -> None:
    repos = [SimpleNamespace(repo_type="model", repo_id="me/m", size_bytes=1024)]
    purge = SimpleNamespace(deleted=1, freed_bytes=1024, failures=[])
    with patch(
        "bioamla.catalogs.huggingface.scan_cache", return_value=repos
    ), patch("bioamla.catalogs.huggingface.purge_cache", return_value=purge):
        result = runner.invoke(cli, ["catalogs", "hf", "cache", "--purge", "-y"])
    assert result.exit_code == 0
    assert "Purged 1" in result.output


def test_hf_cache_error(runner: CliRunner) -> None:
    with patch(
        "bioamla.catalogs.huggingface.scan_cache",
        side_effect=CatalogError("cache fail"),
    ):
        result = runner.invoke(cli, ["catalogs", "hf", "cache"])
    assert result.exit_code != 0


# --- Xeno-canto ----------------------------------------------------------


def _xc_rec():
    return SimpleNamespace(
        id="555",
        scientific_name="Turdus migratorius",
        common_name="American Robin",
        quality="A",
        sound_type="song",
        length="0:30",
        location="NY",
        country="USA",
        recordist="Someone",
        url="http://xc/555",
        to_dict=lambda: {"id": "555", "scientific_name": "Turdus migratorius"},
    )


def test_xc_search_table(runner: CliRunner) -> None:
    res = SimpleNamespace(recordings=[_xc_rec()])
    with patch("bioamla.catalogs.xeno_canto.search", return_value=res):
        result = runner.invoke(cli, ["catalogs", "xc", "search", "-s", "robin"])
    assert result.exit_code == 0
    assert "XC555" in result.output


def test_xc_search_json(runner: CliRunner) -> None:
    res = SimpleNamespace(recordings=[_xc_rec()])
    with patch("bioamla.catalogs.xeno_canto.search", return_value=res):
        result = runner.invoke(
            cli, ["catalogs", "xc", "search", "-s", "robin", "--format", "json"]
        )
    assert result.exit_code == 0
    assert "Turdus migratorius" in result.output


def test_xc_search_csv(runner: CliRunner) -> None:
    res = SimpleNamespace(recordings=[_xc_rec()])
    with patch("bioamla.catalogs.xeno_canto.search", return_value=res):
        result = runner.invoke(
            cli, ["catalogs", "xc", "search", "-s", "robin", "--format", "csv"]
        )
    assert result.exit_code == 0
    assert "scientific_name" in result.output


def test_xc_search_none(runner: CliRunner) -> None:
    with patch("bioamla.catalogs.xeno_canto.search", return_value=SimpleNamespace(recordings=[])):
        result = runner.invoke(cli, ["catalogs", "xc", "search", "-s", "robin"])
    assert result.exit_code == 0
    assert "No recordings" in result.output


def test_xc_search_error(runner: CliRunner) -> None:
    with patch("bioamla.catalogs.xeno_canto.search", side_effect=CatalogError("x")):
        result = runner.invoke(cli, ["catalogs", "xc", "search", "-s", "robin"])
    assert result.exit_code != 0


def test_xc_download(runner: CliRunner, tmp_path) -> None:
    dl = SimpleNamespace(total=3, downloaded=3)
    with patch("bioamla.catalogs.xeno_canto.download", return_value=dl):
        result = runner.invoke(
            cli, ["catalogs", "xc", "download", "-s", "robin", "-o", str(tmp_path)]
        )
    assert result.exit_code == 0
    assert "Download complete" in result.output


def test_xc_download_none(runner: CliRunner, tmp_path) -> None:
    dl = SimpleNamespace(total=0, downloaded=0)
    with patch("bioamla.catalogs.xeno_canto.download", return_value=dl):
        result = runner.invoke(
            cli, ["catalogs", "xc", "download", "-s", "robin", "-o", str(tmp_path)]
        )
    assert result.exit_code == 0
    assert "No recordings" in result.output


def test_xc_download_error(runner: CliRunner, tmp_path) -> None:
    with patch("bioamla.catalogs.xeno_canto.download", side_effect=CatalogError("x")):
        result = runner.invoke(
            cli, ["catalogs", "xc", "download", "-s", "robin", "-o", str(tmp_path)]
        )
    assert result.exit_code != 0


# --- Macaulay ------------------------------------------------------------


def _ml_rec():
    return SimpleNamespace(
        catalog_id="999",
        scientific_name="Turdus migratorius",
        common_name="American Robin",
        rating=4,
        duration=30,
        location="NY",
        country="USA",
        user_display_name="Birder",
        to_dict=lambda: {"catalog_id": "999"},
    )


def test_ml_search_table(runner: CliRunner) -> None:
    res = SimpleNamespace(recordings=[_ml_rec()])
    with patch("bioamla.catalogs.macaulay.search", return_value=res):
        result = runner.invoke(cli, ["catalogs", "ml", "search", "-s", "amerob"])
    assert result.exit_code == 0
    assert "ML999" in result.output


def test_ml_search_json(runner: CliRunner) -> None:
    res = SimpleNamespace(recordings=[_ml_rec()])
    with patch("bioamla.catalogs.macaulay.search", return_value=res):
        result = runner.invoke(
            cli, ["catalogs", "ml", "search", "-s", "amerob", "--format", "json"]
        )
    assert result.exit_code == 0
    assert "999" in result.output


def test_ml_search_none(runner: CliRunner) -> None:
    with patch("bioamla.catalogs.macaulay.search", return_value=SimpleNamespace(recordings=[])):
        result = runner.invoke(cli, ["catalogs", "ml", "search", "-s", "amerob"])
    assert result.exit_code == 0
    assert "No recordings" in result.output


def test_ml_search_error(runner: CliRunner) -> None:
    with patch("bioamla.catalogs.macaulay.search", side_effect=CatalogError("x")):
        result = runner.invoke(cli, ["catalogs", "ml", "search", "-s", "amerob"])
    assert result.exit_code != 0


def test_ml_download(runner: CliRunner, tmp_path) -> None:
    dl = SimpleNamespace(total=2, downloaded=2)
    with patch("bioamla.catalogs.macaulay.download", return_value=dl):
        result = runner.invoke(
            cli, ["catalogs", "ml", "download", "-s", "amerob", "-o", str(tmp_path)]
        )
    assert result.exit_code == 0
    assert "Download complete" in result.output


def test_ml_download_none(runner: CliRunner, tmp_path) -> None:
    dl = SimpleNamespace(total=0, downloaded=0)
    with patch("bioamla.catalogs.macaulay.download", return_value=dl):
        result = runner.invoke(
            cli, ["catalogs", "ml", "download", "-s", "amerob", "-o", str(tmp_path)]
        )
    assert result.exit_code == 0
    assert "No recordings" in result.output


# --- eBird ---------------------------------------------------------------


def test_ebird_species(runner: CliRunner) -> None:
    info = SimpleNamespace(
        scientific_name="Turdus migratorius",
        common_name="American Robin",
        species_code="amerob",
        family="Turdidae",
        order="Passeriformes",
    )
    with patch("bioamla.catalogs.species.lookup", return_value=info):
        result = runner.invoke(cli, ["catalogs", "ebird", "species", "robin"])
    assert result.exit_code == 0
    assert "amerob" in result.output


def test_ebird_species_error(runner: CliRunner) -> None:
    with patch("bioamla.catalogs.species.lookup", side_effect=CatalogError("not found")):
        result = runner.invoke(cli, ["catalogs", "ebird", "species", "xyz"])
    assert result.exit_code != 0


def test_ebird_search(runner: CliRunner) -> None:
    matches = [
        SimpleNamespace(
            scientific_name="Turdus migratorius",
            common_name="American Robin",
            species_code="amerob",
            family="Turdidae",
            score=0.95,
        )
    ]
    with patch("bioamla.catalogs.species.search", return_value=matches):
        result = runner.invoke(cli, ["catalogs", "ebird", "search", "robin"])
    assert result.exit_code == 0
    assert "amerob" in result.output


def test_ebird_search_none(runner: CliRunner) -> None:
    with patch("bioamla.catalogs.species.search", return_value=[]):
        result = runner.invoke(cli, ["catalogs", "ebird", "search", "zzz"])
    assert result.exit_code == 0
    assert "No species found" in result.output


def test_ebird_search_error(runner: CliRunner) -> None:
    with patch("bioamla.catalogs.species.search", side_effect=CatalogError("x")):
        result = runner.invoke(cli, ["catalogs", "ebird", "search", "robin"])
    assert result.exit_code != 0


def test_ebird_validate(runner: CliRunner) -> None:
    mock_mod = MagicMock()
    service = mock_mod.EBirdService.return_value
    service.validate_species.return_value = SimpleNamespace(
        is_valid=True,
        nearby_observations=12,
        most_recent_observation="2024-01-01",
        total_species_in_area=50,
    )
    with patch.dict("sys.modules", {"bioamla.catalogs.ebird": mock_mod}):
        result = runner.invoke(
            cli,
            ["catalogs", "ebird", "validate", "amerob", "--lat", "40", "--lng", "-73",
             "--api-key", "KEY"],
        )
    assert result.exit_code == 0
    assert "expected at this location" in result.output


def test_ebird_validate_invalid(runner: CliRunner) -> None:
    mock_mod = MagicMock()
    service = mock_mod.EBirdService.return_value
    service.validate_species.return_value = SimpleNamespace(
        is_valid=False,
        nearby_observations=0,
        most_recent_observation=None,
        total_species_in_area=50,
    )
    with patch.dict("sys.modules", {"bioamla.catalogs.ebird": mock_mod}):
        result = runner.invoke(
            cli,
            ["catalogs", "ebird", "validate", "amerob", "--lat", "40", "--lng", "-73",
             "--api-key", "KEY"],
        )
    assert result.exit_code == 0
    assert "not recently observed" in result.output


def test_ebird_validate_error(runner: CliRunner) -> None:
    mock_mod = MagicMock()
    mock_mod.EBirdService.side_effect = CatalogError("bad key")
    with patch.dict("sys.modules", {"bioamla.catalogs.ebird": mock_mod}):
        result = runner.invoke(
            cli,
            ["catalogs", "ebird", "validate", "amerob", "--lat", "40", "--lng", "-73",
             "--api-key", "KEY"],
        )
    assert result.exit_code != 0


def test_ebird_nearby(runner: CliRunner, tmp_path) -> None:
    obs = [
        SimpleNamespace(
            common_name="American Robin",
            how_many=2,
            location_name="Central Park",
            to_dict=lambda: {
                "species_code": "amerob",
                "common_name": "American Robin",
                "scientific_name": "Turdus migratorius",
                "location_name": "Central Park",
                "observation_date": "2024-01-01",
                "how_many": 2,
            },
        )
    ]
    mock_mod = MagicMock()
    service = mock_mod.EBirdService.return_value
    service.get_nearby.return_value = SimpleNamespace(observations=obs)
    out = tmp_path / "nearby.csv"
    with patch.dict("sys.modules", {"bioamla.catalogs.ebird": mock_mod}):
        result = runner.invoke(
            cli,
            ["catalogs", "ebird", "nearby", "--lat", "40", "--lng", "-73",
             "--api-key", "KEY", "-o", str(out)],
        )
    assert result.exit_code == 0
    assert "American Robin" in result.output
    assert out.exists()


def test_ebird_nearby_error(runner: CliRunner) -> None:
    mock_mod = MagicMock()
    service = mock_mod.EBirdService.return_value
    service.get_nearby.side_effect = CatalogError("x")
    with patch.dict("sys.modules", {"bioamla.catalogs.ebird": mock_mod}):
        result = runner.invoke(
            cli,
            ["catalogs", "ebird", "nearby", "--lat", "40", "--lng", "-73", "--api-key", "KEY"],
        )
    assert result.exit_code != 0
