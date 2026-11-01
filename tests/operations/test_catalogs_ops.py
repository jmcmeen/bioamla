"""catalogs.* operations and the library calls behind them (network mocked)."""

import csv
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from click.testing import CliRunner

from bioamla import catalogs as lib
from bioamla import operations
from bioamla.cli.cli import cli
from bioamla.exceptions import OptionConflictError


@pytest.fixture
def inat_results(mocker):
    observations = [
        {"id": 1, "taxon": {"name": "Rana", "preferred_common_name": "frog"}, "sounds": [1, 2],
         "observed_on": "2024-05-01", "place_guess": "Pond"},
        {"id": 2, "taxon": None, "sounds": None, "observed_on": None, "place_guess": None},
    ]  # fmt: skip
    return mocker.patch(
        "bioamla.catalogs.inat.search", return_value=SimpleNamespace(observations=observations)
    )


def test_inat_search_rows_and_csv(inat_results, tmp_path):
    found = lib.inat_search(taxon_name="Rana", output=tmp_path / "o.csv")
    assert found.count == 2
    assert found.observations[0]["sound_count"] == 2
    assert found.observations[1]["scientific_name"] == ""
    with open(tmp_path / "o.csv", newline="") as f:
        assert [r["observation_id"] for r in csv.DictReader(f)] == ["1", "2"]
    got = json.loads(
        CliRunner()
        .invoke(cli, ["--json", "catalogs", "inat", "search", "--species", "Rana"])
        .stdout
    )["result"]
    assert got["observations"] == found.to_dict()["observations"]


def test_inat_search_needs_a_filter():
    with pytest.raises(OptionConflictError, match="At least one search filter"):
        lib.inat_search()
    assert CliRunner().invoke(cli, ["catalogs", "inat", "search"]).exit_code == 2


def test_push_dataset_writes_card_and_reports_warnings(mocker, tmp_path):
    push = mocker.patch(
        "bioamla.catalogs.huggingface.push_dataset", return_value=SimpleNamespace(url="u")
    )
    mocker.patch("bioamla.datasets.write_dataset_card", return_value=str(tmp_path / "README.md"))
    mocker.patch("bioamla.datasets.has_location_columns", return_value=True)
    mocker.patch("bioamla.datasets.wavs_with_location", return_value=[])
    pushed = lib.push_dataset_folder(tmp_path, "me/ds")
    assert pushed.card == str(tmp_path / "README.md")
    assert "will be published as they are" in pushed.warnings[0]
    assert push.call_args.kwargs["coordinates"] == "keep"


def test_ebird_nearby_csv(mocker, tmp_path):
    obs = SimpleNamespace(
        to_dict=lambda: {"species_code": "amerob", "common_name": "Robin", "extra": 1}
    )
    service = mocker.patch("bioamla.catalogs.ebird.EBirdService")
    service.return_value.get_nearby.return_value = SimpleNamespace(observations=[obs])
    found = lib.ebird_nearby(1.0, 2.0, output=tmp_path / "n.csv", api_key="k")
    assert found.count == 1
    assert (tmp_path / "n.csv").read_text().splitlines()[0] == ",".join(lib.NEARBY_COLUMNS)
    service.assert_called_once_with(api_key="k")


def test_requests_never_carry_api_keys():
    for op in operations.all_operations()[0]:
        if op.name.startswith("catalogs."):
            fields = set(op.request.__dataclass_fields__)
            assert not {f for f in fields if "key" in f or "token" in f}, op.name


def test_cache_purge_is_local_only_and_outputs_are_checked(inat_results, tmp_path):
    assert operations.get("catalogs.hf.cache-purge").remote is False
    assert operations.get("catalogs.hf.cache").remote is True
    seen = []
    ctx = operations.OperationContext(path_check=lambda p: seen.append(Path(p).name) or p)
    operations.run(
        "catalogs.inat.search", {"species": "Rana", "output": str(tmp_path / "x.csv")}, ctx
    )
    assert seen == ["x.csv"]
    assert operations.get("catalogs.xc.download").kind == "job"
    assert "output_dir" in operations.get("catalogs.ml.download").path_fields()


def test_cli_api_key_reaches_the_operation_as_a_secret(mocker, monkeypatch):
    import os

    monkeypatch.setenv("EBIRD_API_KEY", "from-env")
    seen = {}

    class Service:
        def __init__(self, api_key=None):
            seen["key"] = api_key

        def validate_species(self, **kwargs):
            return SimpleNamespace(
                is_valid=True, nearby_observations=1, most_recent_observation=None
            )

    mocker.patch("bioamla.catalogs.ebird.EBirdService", Service)
    run = CliRunner().invoke(
        cli,
        ["catalogs", "ebird", "validate", "amerob", "--lat", "1", "--lng", "2", "--api-key", "K"],
    )
    assert run.exit_code == 0, run.output
    assert seen["key"] == "K"  # the flag, passed as a secret
    assert os.environ["EBIRD_API_KEY"] == "from-env"  # the environment is untouched


def test_operation_falls_back_to_the_environment_key(mocker, monkeypatch):
    monkeypatch.setenv("EBIRD_API_KEY", "from-env")
    service = mocker.patch("bioamla.catalogs.ebird.EBirdService")
    operations.run("catalogs.ebird.validate", {"species_code": "x", "lat": 1, "lng": 2})
    service.assert_called_once_with(api_key="from-env")


def test_push_dataset_notices_come_before_the_upload(mocker, tmp_path):
    order = []
    mocker.patch(
        "bioamla.catalogs.huggingface.push_dataset",
        side_effect=lambda *a, **k: order.append("push") or SimpleNamespace(url="u"),
    )
    mocker.patch("bioamla.datasets.write_dataset_card", return_value="README.md")
    mocker.patch("bioamla.datasets.has_location_columns", return_value=True)
    mocker.patch("bioamla.datasets.wavs_with_location", return_value=[])
    ctx = operations.OperationContext(on_notice=lambda level, msg: order.append(level))
    operations.run("catalogs.hf.push-dataset", {"path": str(tmp_path), "repo_id": "me/ds"}, ctx)
    assert order == ["info", "warning", "push"]
