"""CLI tests for `bioamla indices` single-file commands.

Domain functions are imported lazily from `bioamla.audio` and `bioamla.indices`
inside the command bodies, so we patch the symbols there.
"""

from unittest.mock import MagicMock

import pytest
from click.testing import CliRunner

from bioamla.cli.cli import cli
from bioamla.exceptions import ProcessingError
from bioamla.indices.compute import AcousticIndices


@pytest.fixture
def runner():
    return CliRunner()


@pytest.fixture
def mock_audio(mocker):
    """Patch load_audio_data to return a lightweight fake AudioData."""
    fake = MagicMock()
    fake.samples = MagicMock()
    fake.sample_rate = 16000
    return mocker.patch("bioamla.audio.load_audio_data", return_value=fake)


def _indices(with_entropy=True):
    return AcousticIndices(
        aci=12.34,
        adi=0.5,
        aei=0.6,
        bio=7.8,
        ndsi=0.1,
        h_spectral=0.9 if with_entropy else None,
        h_temporal=0.8 if with_entropy else None,
    )


# --------------------------------------------------------------------------- #
# help / registration
# --------------------------------------------------------------------------- #


def test_indices_group_help(runner):
    result = runner.invoke(cli, ["indices", "--help"])
    assert result.exit_code == 0
    assert "compute" in result.output


@pytest.mark.parametrize(
    "cmd", ["compute", "temporal", "aci", "adi", "aei", "bio", "ndsi", "entropy"]
)
def test_indices_subcommand_help(runner, cmd):
    result = runner.invoke(cli, ["indices", cmd, "--help"])
    assert result.exit_code == 0


# --------------------------------------------------------------------------- #
# compute
# --------------------------------------------------------------------------- #


def test_compute_table(runner, test_audio_path, mock_audio, mocker):
    mocker.patch("bioamla.indices.compute_all_indices", return_value=_indices())
    result = runner.invoke(cli, ["indices", "compute", test_audio_path])
    assert result.exit_code == 0
    assert "ACI:" in result.output
    assert "H (spectral)" in result.output


def test_compute_table_no_entropy(runner, test_audio_path, mock_audio, mocker):
    mocker.patch("bioamla.indices.compute_all_indices", return_value=_indices(with_entropy=False))
    result = runner.invoke(cli, ["indices", "compute", test_audio_path])
    assert result.exit_code == 0
    assert "H (spectral)" not in result.output


def test_compute_json(runner, test_audio_path, mock_audio, mocker):
    mocker.patch("bioamla.indices.compute_all_indices", return_value=_indices())
    result = runner.invoke(cli, ["indices", "compute", test_audio_path, "--format", "json"])
    assert result.exit_code == 0
    assert '"aci"' in result.output
    assert '"filepath"' in result.output


def test_compute_csv(runner, test_audio_path, mock_audio, mocker):
    mocker.patch("bioamla.indices.compute_all_indices", return_value=_indices())
    result = runner.invoke(cli, ["indices", "compute", test_audio_path, "--format", "csv"])
    assert result.exit_code == 0
    assert "aci" in result.output


def test_compute_output_file(runner, test_audio_path, tmp_path, mock_audio, mocker):
    mocker.patch("bioamla.indices.compute_all_indices", return_value=_indices())
    out = tmp_path / "sub" / "indices.json"
    result = runner.invoke(cli, ["indices", "compute", test_audio_path, "-o", str(out)])
    assert result.exit_code == 0
    assert out.exists()
    assert "Results saved to" in result.output


def test_compute_with_aci_max_freq(runner, test_audio_path, mock_audio, mocker):
    m = mocker.patch("bioamla.indices.compute_all_indices", return_value=_indices())
    result = runner.invoke(cli, ["indices", "compute", test_audio_path, "--aci-max-freq", "8000"])
    assert result.exit_code == 0
    assert m.call_args.kwargs["aci_max_freq"] == 8000.0


def test_compute_error(runner, test_audio_path, mock_audio, mocker):
    mocker.patch("bioamla.indices.compute_all_indices", side_effect=ProcessingError("bad"))
    result = runner.invoke(cli, ["indices", "compute", test_audio_path])
    assert result.exit_code != 0
    assert "bad" in result.output


def test_compute_missing_file(runner):
    result = runner.invoke(cli, ["indices", "compute", "/no/file.wav"])
    assert result.exit_code != 0


# --------------------------------------------------------------------------- #
# temporal
# --------------------------------------------------------------------------- #


def _windows(n=2):
    return [
        {
            "start_time": i * 60.0,
            "aci": 1.0,
            "adi": 0.2,
            "aei": 0.3,
            "bio": 4.0,
            "ndsi": 0.1,
        }
        for i in range(n)
    ]


def test_temporal_table(runner, test_audio_path, mock_audio, mocker):
    mocker.patch("bioamla.indices.temporal_indices", return_value=_windows())
    result = runner.invoke(cli, ["indices", "temporal", test_audio_path])
    assert result.exit_code == 0
    assert "Temporal indices" in result.output
    assert "Total segments: 2" in result.output


def test_temporal_table_truncates(runner, test_audio_path, mock_audio, mocker):
    mocker.patch("bioamla.indices.temporal_indices", return_value=_windows(15))
    result = runner.invoke(cli, ["indices", "temporal", test_audio_path])
    assert result.exit_code == 0
    assert "5 more segments" in result.output


def test_temporal_output_file(runner, test_audio_path, tmp_path, mock_audio, mocker):
    mocker.patch("bioamla.indices.temporal_indices", return_value=_windows())
    out = tmp_path / "temporal.json"
    result = runner.invoke(cli, ["indices", "temporal", test_audio_path, "-o", str(out)])
    assert result.exit_code == 0
    assert out.exists()
    assert "Temporal indices saved to" in result.output


def test_temporal_error(runner, test_audio_path, mock_audio, mocker):
    mocker.patch("bioamla.indices.temporal_indices", side_effect=ProcessingError("tfail"))
    result = runner.invoke(cli, ["indices", "temporal", test_audio_path])
    assert result.exit_code != 0


# --------------------------------------------------------------------------- #
# single-index commands
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("cmd", "label"),
    [
        ("aci", "ACI:"),
        ("adi", "ADI:"),
        ("aei", "AEI:"),
        ("bio", "BIO:"),
        ("ndsi", "NDSI:"),
    ],
)
def test_single_index(runner, test_audio_path, mock_audio, mocker, cmd, label):
    mocker.patch("bioamla.indices.compute_index", return_value=1.234)
    result = runner.invoke(cli, ["indices", cmd, test_audio_path])
    assert result.exit_code == 0
    assert label in result.output


@pytest.mark.parametrize("cmd", ["aci", "adi", "aei", "bio", "ndsi"])
def test_single_index_error(runner, test_audio_path, mock_audio, mocker, cmd):
    mocker.patch("bioamla.indices.compute_index", side_effect=ProcessingError("x"))
    result = runner.invoke(cli, ["indices", cmd, test_audio_path])
    assert result.exit_code != 0


def test_entropy(runner, test_audio_path, mock_audio, mocker):
    mocker.patch("bioamla.indices.compute_index", side_effect=[0.9, 0.8])
    result = runner.invoke(cli, ["indices", "entropy", test_audio_path])
    assert result.exit_code == 0
    assert "H (spectral): 0.900" in result.output
    assert "H (temporal): 0.800" in result.output


def test_entropy_error(runner, test_audio_path, mock_audio, mocker):
    mocker.patch("bioamla.indices.compute_index", side_effect=ProcessingError("e"))
    result = runner.invoke(cli, ["indices", "entropy", test_audio_path])
    assert result.exit_code != 0
