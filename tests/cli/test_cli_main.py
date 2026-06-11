"""Tests for bioamla.cli.cli root group and main() error handling."""

from __future__ import annotations

import click
import pytest
from click.testing import CliRunner

from bioamla import __version__
from bioamla.cli.cli import cli, main
from bioamla.exceptions import ProcessingError


def test_cli_help():
    result = CliRunner().invoke(cli, ["--help"])
    assert result.exit_code == 0
    assert "BioAMLA" in result.output


def test_cli_version():
    result = CliRunner().invoke(cli, ["--version"])
    assert result.exit_code == 0
    assert __version__ in result.output


def test_cli_registers_command_groups():
    result = CliRunner().invoke(cli, ["--help"])
    for name in ("audio", "config", "indices", "detect", "cluster", "dataset"):
        assert name in result.output


def test_main_success(monkeypatch):
    called = {}
    monkeypatch.setattr(cli, "main", lambda **kw: called.setdefault("ok", True))
    main()
    assert called["ok"] is True


def test_main_handles_bioamla_error(monkeypatch):
    def raise_err(**kw):
        raise ProcessingError("domain failure")

    monkeypatch.setattr(cli, "main", raise_err)
    with pytest.raises(SystemExit) as exc:
        main()
    assert exc.value.code == 1


def test_main_handles_click_exception(monkeypatch):
    def raise_err(**kw):
        raise click.ClickException("bad usage")

    monkeypatch.setattr(cli, "main", raise_err)
    with pytest.raises(SystemExit) as exc:
        main()
    # click.ClickException default exit code is 1
    assert exc.value.code == 1


def test_main_handles_abort(monkeypatch):
    def raise_err(**kw):
        raise click.exceptions.Abort()

    monkeypatch.setattr(cli, "main", raise_err)
    with pytest.raises(SystemExit) as exc:
        main()
    assert exc.value.code == 1


def test_main_handles_keyboard_interrupt(monkeypatch):
    def raise_err(**kw):
        raise KeyboardInterrupt()

    monkeypatch.setattr(cli, "main", raise_err)
    with pytest.raises(SystemExit) as exc:
        main()
    assert exc.value.code == 1
