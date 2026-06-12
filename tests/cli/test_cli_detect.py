"""CLI tests for `bioamla detect` single-file commands.

Domain detectors are imported lazily inside the command bodies from
`bioamla.detect`, so we patch the symbols there.
"""

from unittest.mock import MagicMock

import pytest
from click.testing import CliRunner

from bioamla.cli.cli import cli
from bioamla.detect.core import Detection, PeakDetection
from bioamla.exceptions import DetectionError


@pytest.fixture
def runner():
    return CliRunner()


def _detection(start=0.1, end=0.5, conf=0.9, **meta):
    return Detection(start_time=start, end_time=end, confidence=conf, metadata=meta)


def _mock_detector(detections):
    """Return a detector class whose instances yield `detections`."""
    instance = MagicMock()
    instance.detect_from_file.return_value = detections
    cls = MagicMock(return_value=instance)
    return cls, instance


# --------------------------------------------------------------------------- #
# help / registration
# --------------------------------------------------------------------------- #


def test_detect_group_help(runner):
    result = runner.invoke(cli, ["detect", "--help"])
    assert result.exit_code == 0
    assert "energy" in result.output
    assert "ribbit" in result.output


@pytest.mark.parametrize("cmd", ["energy", "ribbit", "peaks", "accelerating"])
def test_detect_subcommand_help(runner, cmd):
    result = runner.invoke(cli, ["detect", cmd, "--help"])
    assert result.exit_code == 0


# --------------------------------------------------------------------------- #
# energy
# --------------------------------------------------------------------------- #


def test_energy_table(runner, test_audio_path, mocker):
    cls, _ = _mock_detector([_detection()])
    mocker.patch("bioamla.detect.BandLimitedEnergyDetector", cls)
    mocker.patch("bioamla.detect.export_detections")
    result = runner.invoke(cli, ["detect", "energy", test_audio_path])
    assert result.exit_code == 0
    assert "Found 1 detections" in result.output
    assert "Total: 1 detections" in result.output


def test_energy_json(runner, test_audio_path, mocker):
    cls, _ = _mock_detector([_detection()])
    mocker.patch("bioamla.detect.BandLimitedEnergyDetector", cls)
    mocker.patch("bioamla.detect.export_detections")
    result = runner.invoke(cli, ["detect", "energy", test_audio_path, "--format", "json"])
    assert result.exit_code == 0
    assert '"start_time"' in result.output


def test_energy_csv(runner, test_audio_path, mocker):
    cls, _ = _mock_detector([_detection()])
    mocker.patch("bioamla.detect.BandLimitedEnergyDetector", cls)
    mocker.patch("bioamla.detect.export_detections")
    result = runner.invoke(cli, ["detect", "energy", test_audio_path, "--format", "csv"])
    assert result.exit_code == 0
    assert "start_time" in result.output


def test_energy_csv_empty(runner, test_audio_path, mocker):
    cls, _ = _mock_detector([])
    mocker.patch("bioamla.detect.BandLimitedEnergyDetector", cls)
    mocker.patch("bioamla.detect.export_detections")
    result = runner.invoke(cli, ["detect", "energy", test_audio_path, "--format", "csv"])
    assert result.exit_code == 0
    assert "No detections found." in result.output


def test_energy_output_file(runner, test_audio_path, tmp_path, mocker):
    cls, _ = _mock_detector([_detection()])
    mocker.patch("bioamla.detect.BandLimitedEnergyDetector", cls)
    export = mocker.patch("bioamla.detect.export_detections")
    out = tmp_path / "dets.json"
    result = runner.invoke(cli, ["detect", "energy", test_audio_path, "-o", str(out)])
    assert result.exit_code == 0
    assert "Saved 1 detections" in result.output
    export.assert_called_once()
    assert export.call_args.kwargs["format"] == "json"


def test_energy_output_csv_format(runner, test_audio_path, tmp_path, mocker):
    cls, _ = _mock_detector([_detection()])
    mocker.patch("bioamla.detect.BandLimitedEnergyDetector", cls)
    export = mocker.patch("bioamla.detect.export_detections")
    out = tmp_path / "dets.csv"
    result = runner.invoke(cli, ["detect", "energy", test_audio_path, "-o", str(out)])
    assert result.exit_code == 0
    assert export.call_args.kwargs["format"] == "csv"


def test_energy_error(runner, test_audio_path, mocker):
    instance = MagicMock()
    instance.detect_from_file.side_effect = DetectionError("boom")
    mocker.patch("bioamla.detect.BandLimitedEnergyDetector", MagicMock(return_value=instance))
    result = runner.invoke(cli, ["detect", "energy", test_audio_path])
    assert result.exit_code != 0
    assert "boom" in result.output


def test_energy_missing_file(runner):
    result = runner.invoke(cli, ["detect", "energy", "/no/such/file.wav"])
    assert result.exit_code != 0


# --------------------------------------------------------------------------- #
# ribbit
# --------------------------------------------------------------------------- #


def test_ribbit_table(runner, test_audio_path, mocker):
    cls, _ = _mock_detector([_detection(pulse_rate_hz=10.0)])
    mocker.patch("bioamla.detect.RibbitDetector", cls)
    mocker.patch("bioamla.detect.export_detections")
    result = runner.invoke(cli, ["detect", "ribbit", test_audio_path])
    assert result.exit_code == 0
    assert "periodic call detections" in result.output


def test_ribbit_json(runner, test_audio_path, mocker):
    cls, _ = _mock_detector([_detection(pulse_rate_hz=10.0)])
    mocker.patch("bioamla.detect.RibbitDetector", cls)
    mocker.patch("bioamla.detect.export_detections")
    result = runner.invoke(cli, ["detect", "ribbit", test_audio_path, "--format", "json"])
    assert result.exit_code == 0


def test_ribbit_csv_empty(runner, test_audio_path, mocker):
    cls, _ = _mock_detector([])
    mocker.patch("bioamla.detect.RibbitDetector", cls)
    mocker.patch("bioamla.detect.export_detections")
    result = runner.invoke(cli, ["detect", "ribbit", test_audio_path, "--format", "csv"])
    assert result.exit_code == 0
    assert "No detections found." in result.output


def test_ribbit_output_file(runner, test_audio_path, tmp_path, mocker):
    cls, _ = _mock_detector([_detection(pulse_rate_hz=10.0)])
    mocker.patch("bioamla.detect.RibbitDetector", cls)
    export = mocker.patch("bioamla.detect.export_detections")
    out = tmp_path / "r.json"
    result = runner.invoke(cli, ["detect", "ribbit", test_audio_path, "-o", str(out)])
    assert result.exit_code == 0
    export.assert_called_once()


def test_ribbit_error(runner, test_audio_path, mocker):
    instance = MagicMock()
    instance.detect_from_file.side_effect = DetectionError("nope")
    mocker.patch("bioamla.detect.RibbitDetector", MagicMock(return_value=instance))
    result = runner.invoke(cli, ["detect", "ribbit", test_audio_path])
    assert result.exit_code != 0


# --------------------------------------------------------------------------- #
# peaks
# --------------------------------------------------------------------------- #


def _peak(time=0.2, amp=0.8, width=0.01, prom=0.5):
    return PeakDetection(time=time, amplitude=amp, width=width, prominence=prom)


def test_peaks_table(runner, test_audio_path, mocker):
    cls, _ = _mock_detector([_peak()])
    mocker.patch("bioamla.detect.CWTPeakDetector", cls)
    result = runner.invoke(cli, ["detect", "peaks", test_audio_path])
    assert result.exit_code == 0
    assert "Found 1 peaks" in result.output


def test_peaks_table_truncates(runner, test_audio_path, mocker):
    peaks = [_peak(time=i * 0.01) for i in range(25)]
    cls, _ = _mock_detector(peaks)
    mocker.patch("bioamla.detect.CWTPeakDetector", cls)
    result = runner.invoke(cli, ["detect", "peaks", test_audio_path])
    assert result.exit_code == 0
    assert "and 5 more peaks" in result.output


def test_peaks_json(runner, test_audio_path, mocker):
    cls, _ = _mock_detector([_peak()])
    mocker.patch("bioamla.detect.CWTPeakDetector", cls)
    result = runner.invoke(cli, ["detect", "peaks", test_audio_path, "--format", "json"])
    assert result.exit_code == 0
    assert '"time"' in result.output


def test_peaks_csv(runner, test_audio_path, mocker):
    cls, _ = _mock_detector([_peak()])
    mocker.patch("bioamla.detect.CWTPeakDetector", cls)
    result = runner.invoke(cli, ["detect", "peaks", test_audio_path, "--format", "csv"])
    assert result.exit_code == 0
    assert "amplitude" in result.output


def test_peaks_csv_empty(runner, test_audio_path, mocker):
    cls, _ = _mock_detector([])
    mocker.patch("bioamla.detect.CWTPeakDetector", cls)
    result = runner.invoke(cli, ["detect", "peaks", test_audio_path, "--format", "csv"])
    assert result.exit_code == 0
    assert "No peaks found." in result.output


def test_peaks_output_file(runner, test_audio_path, tmp_path, mocker):
    cls, _ = _mock_detector([_peak()])
    mocker.patch("bioamla.detect.CWTPeakDetector", cls)
    out = tmp_path / "sub" / "peaks.csv"
    result = runner.invoke(cli, ["detect", "peaks", test_audio_path, "-o", str(out)])
    assert result.exit_code == 0
    assert out.exists()
    assert "Saved 1 peaks" in result.output


def test_peaks_error(runner, test_audio_path, mocker):
    instance = MagicMock()
    instance.detect_from_file.side_effect = DetectionError("peakfail")
    mocker.patch("bioamla.detect.CWTPeakDetector", MagicMock(return_value=instance))
    result = runner.invoke(cli, ["detect", "peaks", test_audio_path])
    assert result.exit_code != 0


# --------------------------------------------------------------------------- #
# accelerating
# --------------------------------------------------------------------------- #


def _accel_detection():
    return _detection(
        pattern_type="accelerating",
        acceleration_ratio=2.0,
        initial_rate=5.0,
        final_rate=10.0,
    )


def test_accelerating_table(runner, test_audio_path, mocker):
    cls, _ = _mock_detector([_accel_detection()])
    mocker.patch("bioamla.detect.AcceleratingPatternDetector", cls)
    mocker.patch("bioamla.detect.export_detections")
    result = runner.invoke(cli, ["detect", "accelerating", test_audio_path])
    assert result.exit_code == 0
    assert "pattern detections" in result.output
    assert "accelerating" in result.output


def test_accelerating_json(runner, test_audio_path, mocker):
    cls, _ = _mock_detector([_accel_detection()])
    mocker.patch("bioamla.detect.AcceleratingPatternDetector", cls)
    mocker.patch("bioamla.detect.export_detections")
    result = runner.invoke(cli, ["detect", "accelerating", test_audio_path, "--format", "json"])
    assert result.exit_code == 0


def test_accelerating_csv_empty(runner, test_audio_path, mocker):
    cls, _ = _mock_detector([])
    mocker.patch("bioamla.detect.AcceleratingPatternDetector", cls)
    mocker.patch("bioamla.detect.export_detections")
    result = runner.invoke(cli, ["detect", "accelerating", test_audio_path, "--format", "csv"])
    assert result.exit_code == 0
    assert "No detections found." in result.output


def test_accelerating_output_file(runner, test_audio_path, tmp_path, mocker):
    cls, _ = _mock_detector([_accel_detection()])
    mocker.patch("bioamla.detect.AcceleratingPatternDetector", cls)
    export = mocker.patch("bioamla.detect.export_detections")
    out = tmp_path / "a.csv"
    result = runner.invoke(cli, ["detect", "accelerating", test_audio_path, "-o", str(out)])
    assert result.exit_code == 0
    export.assert_called_once()


def test_accelerating_error(runner, test_audio_path, mocker):
    instance = MagicMock()
    instance.detect_from_file.side_effect = DetectionError("accfail")
    mocker.patch("bioamla.detect.AcceleratingPatternDetector", MagicMock(return_value=instance))
    result = runner.invoke(cli, ["detect", "accelerating", test_audio_path])
    assert result.exit_code != 0
