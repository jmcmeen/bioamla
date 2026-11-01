"""The committed OpenAPI document matches the API."""

import json
from pathlib import Path

import pytest

pytest.importorskip("fastapi")

from bioamla import operations  # noqa: E402
from bioamla.server.openapi import openapi_spec, write_openapi_spec  # noqa: E402

COMMITTED = Path(__file__).parents[2] / "docs" / "api" / "openapi.json"


def test_committed_openapi_document_is_current():
    _, unavailable = operations.all_operations()
    if unavailable:
        pytest.skip(f"operations need extras that aren't installed: {sorted(unavailable)}")
    committed = json.loads(COMMITTED.read_text(encoding="utf-8"))
    assert committed == openapi_spec(), (
        "docs/api/openapi.json is out of date; run: bioamla serve --openapi docs/api/openapi.json"
    )


def test_the_document_names_the_api_version_not_the_package_version(tmp_path):
    spec = json.loads(write_openapi_spec(tmp_path / "api" / "openapi.json").read_text())
    assert spec["info"]["version"] == "v1"
    assert "/v1/jobs/{job_id}/events" in spec["paths"]
    assert "/v1/workspaces/{ws}/files/{path}" in spec["paths"]


def test_serve_writes_it_without_a_root(tmp_path):
    from click.testing import CliRunner

    from bioamla.cli.cli import cli

    out = tmp_path / "openapi.json"
    result = CliRunner().invoke(cli, ["serve", "--openapi", str(out)])
    assert result.exit_code == 0, result.output
    assert json.loads(out.read_text())["openapi"].startswith("3.")
    assert CliRunner().invoke(cli, ["serve"]).exit_code == 2  # --root is required to serve
    as_json = CliRunner().invoke(cli, ["--json", "serve", "--openapi", str(out)])
    assert json.loads(as_json.stdout)["result"]["path"] == str(out)
    blocked = tmp_path / "file"
    blocked.write_text("")
    failed = CliRunner().invoke(cli, ["serve", "--openapi", str(blocked / "openapi.json")])
    assert failed.exit_code == 1 and "Can't write" in failed.output


def test_a_missing_extra_is_an_error_not_a_partial_document(monkeypatch):
    from bioamla.exceptions import DependencyError

    monkeypatch.setattr(operations, "all_operations", lambda: ([], {"ml": "no torch"}))
    with pytest.raises(DependencyError, match="every extra"):
        openapi_spec()
