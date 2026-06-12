"""CLI tests for `bioamla config` commands."""

from __future__ import annotations

from unittest.mock import patch

import pytest
from click.testing import CliRunner

from bioamla.cli.cli import cli
from bioamla.exceptions import ConfigError
from bioamla.system.dependency import DependencyInfo, DependencyReport
from bioamla.system.util import DeviceInfo, DevicesData, VersionData


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


def test_config_group_help(runner: CliRunner) -> None:
    result = runner.invoke(cli, ["config", "--help"])
    assert result.exit_code == 0
    assert "Configuration" in result.output


# --- version -------------------------------------------------------------


def test_config_version(runner: CliRunner) -> None:
    data = VersionData(
        bioamla_version="0.2.0",
        python_version="3.11.0 (main)",
        platform="Linux",
        pytorch_version="2.1.0",
        cuda_version="12.1",
    )
    with patch("bioamla.system.util.get_version", return_value=data):
        result = runner.invoke(cli, ["config", "version"])
    assert result.exit_code == 0
    assert "0.2.0" in result.output
    assert "PyTorch" in result.output


def test_config_version_no_torch(runner: CliRunner) -> None:
    data = VersionData(
        bioamla_version="0.2.0",
        python_version="3.11.0 (main)",
        platform="Linux",
    )
    with patch("bioamla.system.util.get_version", return_value=data):
        result = runner.invoke(cli, ["config", "version"])
    assert result.exit_code == 0
    assert "PyTorch" not in result.output


def test_config_version_error(runner: CliRunner) -> None:
    with patch("bioamla.system.util.get_version", side_effect=ConfigError("boom")):
        result = runner.invoke(cli, ["config", "version"])
    assert result.exit_code != 0


# --- devices -------------------------------------------------------------


def test_config_devices_cuda(runner: CliRunner) -> None:
    data = DevicesData(
        devices=[
            DeviceInfo(name="NVIDIA A100", device_type="cuda", device_id="cuda:0", memory_gb=40.0),
        ],
        cuda_available=True,
    )
    with patch("bioamla.system.util.get_device_info", return_value=data):
        result = runner.invoke(cli, ["config", "devices"])
    assert result.exit_code == 0
    assert "CUDA available" in result.output


def test_config_devices_mps(runner: CliRunner) -> None:
    data = DevicesData(
        devices=[DeviceInfo(name="Apple MPS", device_type="mps", device_id="mps")],
        mps_available=True,
    )
    with patch("bioamla.system.util.get_device_info", return_value=data):
        result = runner.invoke(cli, ["config", "devices"])
    assert result.exit_code == 0
    assert "MPS" in result.output


def test_config_devices_cpu(runner: CliRunner) -> None:
    data = DevicesData(
        devices=[DeviceInfo(name="CPU", device_type="cpu", device_id="cpu")],
    )
    with patch("bioamla.system.util.get_device_info", return_value=data):
        result = runner.invoke(cli, ["config", "devices"])
    assert result.exit_code == 0
    assert "CPU" in result.output


def test_config_devices_error(runner: CliRunner) -> None:
    with patch("bioamla.system.util.get_device_info", side_effect=ConfigError("nope")):
        result = runner.invoke(cli, ["config", "devices"])
    assert result.exit_code != 0


# --- deps ----------------------------------------------------------------


def _report(all_installed: bool, install_cmd: str | None = "apt install x") -> DependencyReport:
    deps = [
        DependencyInfo(
            name="FFmpeg",
            description="Audio conversion",
            required_for="conversion",
            installed=all_installed,
            version="6.0" if all_installed else None,
            install_hint=None if all_installed else "apt install ffmpeg",
        )
    ]
    return DependencyReport(
        os_type="linux",
        all_installed=all_installed,
        dependencies=deps,
        install_command=install_cmd,
    )


def test_config_deps_all_installed(runner: CliRunner) -> None:
    with patch("bioamla.system.dependency.check_all", return_value=_report(True)):
        result = runner.invoke(cli, ["config", "deps"])
    assert result.exit_code == 0
    assert "All system dependencies are installed" in result.output


def test_config_deps_missing_list(runner: CliRunner) -> None:
    with patch("bioamla.system.dependency.check_all", return_value=_report(False)):
        result = runner.invoke(cli, ["config", "deps"])
    assert result.exit_code == 0
    assert "not installed" in result.output
    assert "To install" in result.output


def test_config_deps_install(runner: CliRunner) -> None:
    with (
        patch(
            "bioamla.system.dependency.check_all",
            side_effect=[_report(False), _report(True)],
        ),
        patch("bioamla.system.dependency.install", return_value="Installed!"),
    ):
        result = runner.invoke(cli, ["config", "deps", "--install", "-y"])
    assert result.exit_code == 0
    assert "Installing" in result.output


def test_config_deps_install_abort(runner: CliRunner) -> None:
    with patch("bioamla.system.dependency.check_all", return_value=_report(False)):
        result = runner.invoke(cli, ["config", "deps", "--install"], input="n\n")
    assert result.exit_code == 0
    assert "Aborted" in result.output


def test_config_deps_install_fails(runner: CliRunner) -> None:
    with (
        patch("bioamla.system.dependency.check_all", return_value=_report(False)),
        patch("bioamla.system.dependency.install", side_effect=ConfigError("no sudo")),
    ):
        result = runner.invoke(cli, ["config", "deps", "--install", "-y"])
    assert result.exit_code == 1


def test_config_deps_check_error(runner: CliRunner) -> None:
    with patch("bioamla.system.dependency.check_all", side_effect=ConfigError("fail")):
        result = runner.invoke(cli, ["config", "deps"])
    assert result.exit_code != 0
