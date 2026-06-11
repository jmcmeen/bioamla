"""CLI tests for `bioamla audio` single-file commands.

Domain functions are imported lazily from `bioamla.audio` (and `bioamla.viz`)
inside the command bodies, so we patch the symbols there. Heavy audio/ML work is
mocked; tests exercise arg-parsing, branching and output formatting only.
"""

from unittest.mock import MagicMock

import numpy as np
import pytest
from click.testing import CliRunner

from bioamla.cli.cli import cli
from bioamla.exceptions import AudioLoadError


@pytest.fixture
def runner():
    return CliRunner()


def _fake_audio_data(channels=1):
    fake = MagicMock()
    fake.duration = 1.0
    fake.sample_rate = 16000
    fake.channels = channels
    fake.num_samples = 16000
    if channels == 2:
        fake.samples = np.zeros((16000, 2), dtype=np.float32)
    else:
        fake.samples = np.zeros(16000, dtype=np.float32)
    return fake


@pytest.fixture
def mock_load_save(mocker):
    """Patch load_audio/save_audio used by the processing commands."""
    audio = np.zeros(16000, dtype=np.float32)
    load = mocker.patch("bioamla.audio.load_audio", return_value=(audio, 16000))
    save = mocker.patch("bioamla.audio.save_audio")
    return load, save


# --------------------------------------------------------------------------- #
# help / registration
# --------------------------------------------------------------------------- #


def test_audio_group_help(runner):
    result = runner.invoke(cli, ["audio", "--help"])
    assert result.exit_code == 0
    assert "info" in result.output
    assert "convert" in result.output


@pytest.mark.parametrize(
    "cmd",
    [
        "info",
        "list",
        "convert",
        "segment",
        "trim",
        "normalize",
        "resample",
        "filter",
        "denoise",
        "pitch-shift",
        "time-stretch",
        "add-noise",
        "gain",
        "visualize",
    ],
)
def test_audio_subcommand_help(runner, cmd):
    result = runner.invoke(cli, ["audio", cmd, "--help"])
    assert result.exit_code == 0


# --------------------------------------------------------------------------- #
# info
# --------------------------------------------------------------------------- #


def test_info(runner, mocker):
    mocker.patch("bioamla.audio.load_audio_data", return_value=_fake_audio_data())
    result = runner.invoke(cli, ["audio", "info", "file.wav"])
    assert result.exit_code == 0
    assert "Duration: 1.00s" in result.output
    assert "Sample rate: 16000 Hz" in result.output


def test_info_error(runner, mocker):
    mocker.patch("bioamla.audio.load_audio_data", side_effect=AudioLoadError("bad"))
    result = runner.invoke(cli, ["audio", "info", "file.wav"])
    assert result.exit_code != 0
    assert "bad" in result.output


# --------------------------------------------------------------------------- #
# list
# --------------------------------------------------------------------------- #


def test_list_found(runner, mocker):
    mocker.patch("bioamla.audio.list_audio_files", return_value=["a.wav", "b.wav"])
    result = runner.invoke(cli, ["audio", "list", "somedir"])
    assert result.exit_code == 0
    assert "Found 2 audio file(s)" in result.output
    assert "a.wav" in result.output


def test_list_empty(runner, mocker):
    mocker.patch("bioamla.audio.list_audio_files", return_value=[])
    result = runner.invoke(cli, ["audio", "list", "somedir", "--no-recursive"])
    assert result.exit_code == 0
    assert "No audio files found" in result.output


def test_list_error(runner, mocker):
    from bioamla.exceptions import NotFoundError

    mocker.patch("bioamla.audio.list_audio_files", side_effect=NotFoundError("nope"))
    result = runner.invoke(cli, ["audio", "list", "somedir"])
    assert result.exit_code != 0


# --------------------------------------------------------------------------- #
# convert
# --------------------------------------------------------------------------- #


def test_convert(runner, tmp_path, mocker):
    mocker.patch("bioamla.audio.load_audio_data", return_value=_fake_audio_data())
    save = mocker.patch("bioamla.audio.save_audio_data_as")
    out = tmp_path / "out.flac"
    result = runner.invoke(
        cli, ["audio", "convert", "in.wav", str(out), "-r", "8000", "-f", "flac"]
    )
    assert result.exit_code == 0
    assert "Converted" in result.output
    save.assert_called_once()


def test_convert_stereo_to_mono(runner, tmp_path, mocker):
    mocker.patch("bioamla.audio.load_audio_data", return_value=_fake_audio_data(channels=2))
    mocker.patch("bioamla.audio.save_audio_data_as")
    out = tmp_path / "out.wav"
    result = runner.invoke(cli, ["audio", "convert", "in.wav", str(out), "-c", "1"])
    assert result.exit_code == 0


def test_convert_mono_to_stereo(runner, tmp_path, mocker):
    mocker.patch("bioamla.audio.load_audio_data", return_value=_fake_audio_data(channels=1))
    mocker.patch("bioamla.audio.save_audio_data_as")
    out = tmp_path / "out.wav"
    result = runner.invoke(cli, ["audio", "convert", "in.wav", str(out), "-c", "2"])
    assert result.exit_code == 0


def test_convert_error(runner, tmp_path, mocker):
    mocker.patch("bioamla.audio.load_audio_data", side_effect=AudioLoadError("x"))
    result = runner.invoke(cli, ["audio", "convert", "in.wav", str(tmp_path / "o.wav")])
    assert result.exit_code != 0


# --------------------------------------------------------------------------- #
# segment
# --------------------------------------------------------------------------- #


def test_segment(runner, tmp_path, mocker):
    audio = np.zeros(16000 * 3, dtype=np.float32)
    mocker.patch("bioamla.audio.load_audio", return_value=(audio, 16000))
    mocker.patch("bioamla.audio.save_audio")
    out_dir = tmp_path / "segs"
    result = runner.invoke(cli, ["audio", "segment", "in.wav", str(out_dir), "-d", "1.0"])
    assert result.exit_code == 0
    assert "Created 3 segments" in result.output


def test_segment_overlap_too_large(runner, tmp_path, mocker):
    audio = np.zeros(16000 * 3, dtype=np.float32)
    mocker.patch("bioamla.audio.load_audio", return_value=(audio, 16000))
    mocker.patch("bioamla.audio.save_audio")
    out_dir = tmp_path / "segs"
    result = runner.invoke(
        cli, ["audio", "segment", "in.wav", str(out_dir), "-d", "1.0", "-o", "2.0"]
    )
    assert result.exit_code != 0


# --------------------------------------------------------------------------- #
# trim
# --------------------------------------------------------------------------- #


def test_trim(runner, tmp_path, mock_load_save, mocker):
    mocker.patch("bioamla.audio.trim_audio", return_value=np.zeros(8000, dtype=np.float32))
    out = tmp_path / "t.wav"
    result = runner.invoke(cli, ["audio", "trim", "in.wav", str(out), "-s", "0.1", "-e", "0.5"])
    assert result.exit_code == 0
    assert "Trimmed audio saved" in result.output


def test_trim_with_duration(runner, tmp_path, mock_load_save, mocker):
    mocker.patch("bioamla.audio.trim_audio", return_value=np.zeros(8000, dtype=np.float32))
    out = tmp_path / "t.wav"
    result = runner.invoke(cli, ["audio", "trim", "in.wav", str(out), "-s", "0.0", "-d", "0.5"])
    assert result.exit_code == 0


def test_trim_end_and_duration_conflict(runner, tmp_path):
    out = tmp_path / "t.wav"
    result = runner.invoke(cli, ["audio", "trim", "in.wav", str(out), "-e", "0.5", "-d", "0.5"])
    assert result.exit_code != 0
    assert "Cannot specify both" in result.output


# --------------------------------------------------------------------------- #
# normalize
# --------------------------------------------------------------------------- #


def test_normalize_peak(runner, tmp_path, mock_load_save, mocker):
    mocker.patch("bioamla.audio.peak_normalize", return_value=np.zeros(16000, dtype=np.float32))
    out = tmp_path / "n.wav"
    result = runner.invoke(cli, ["audio", "normalize", "in.wav", str(out), "-m", "peak"])
    assert result.exit_code == 0
    assert "Normalized audio saved" in result.output


def test_normalize_rms(runner, tmp_path, mock_load_save, mocker):
    mocker.patch(
        "bioamla.audio.normalize_loudness",
        return_value=np.zeros(16000, dtype=np.float32),
    )
    out = tmp_path / "n.wav"
    result = runner.invoke(cli, ["audio", "normalize", "in.wav", str(out), "-m", "rms"])
    assert result.exit_code == 0


# --------------------------------------------------------------------------- #
# resample / denoise / pitch-shift / time-stretch / add-noise / gain
# --------------------------------------------------------------------------- #


def test_resample(runner, tmp_path, mock_load_save, mocker):
    mocker.patch("bioamla.audio.resample_audio", return_value=np.zeros(8000, dtype=np.float32))
    out = tmp_path / "r.wav"
    result = runner.invoke(cli, ["audio", "resample", "in.wav", str(out), "-r", "8000"])
    assert result.exit_code == 0
    assert "Resampled audio saved" in result.output


def test_denoise(runner, tmp_path, mock_load_save, mocker):
    mocker.patch(
        "bioamla.audio.spectral_denoise",
        return_value=np.zeros(16000, dtype=np.float32),
    )
    out = tmp_path / "d.wav"
    result = runner.invoke(cli, ["audio", "denoise", "in.wav", str(out)])
    assert result.exit_code == 0
    assert "Denoised audio saved" in result.output


def test_pitch_shift(runner, tmp_path, mock_load_save, mocker):
    mocker.patch("bioamla.audio.pitch_shift", return_value=np.zeros(16000, dtype=np.float32))
    out = tmp_path / "p.wav"
    result = runner.invoke(cli, ["audio", "pitch-shift", "in.wav", str(out), "-n", "2"])
    assert result.exit_code == 0
    assert "Pitch-shifted audio" in result.output


def test_time_stretch(runner, tmp_path, mock_load_save, mocker):
    mocker.patch("bioamla.audio.time_stretch", return_value=np.zeros(16000, dtype=np.float32))
    out = tmp_path / "s.wav"
    result = runner.invoke(cli, ["audio", "time-stretch", "in.wav", str(out), "-r", "1.5"])
    assert result.exit_code == 0
    assert "Time-stretched audio" in result.output


def test_add_noise(runner, tmp_path, mock_load_save, mocker):
    mocker.patch("bioamla.audio.add_noise", return_value=np.zeros(16000, dtype=np.float32))
    out = tmp_path / "an.wav"
    result = runner.invoke(
        cli, ["audio", "add-noise", "in.wav", str(out), "--snr-db", "10", "--seed", "1"]
    )
    assert result.exit_code == 0
    assert "Noisy audio" in result.output


def test_gain(runner, tmp_path, mock_load_save, mocker):
    mocker.patch("bioamla.audio.apply_gain", return_value=np.zeros(16000, dtype=np.float32))
    out = tmp_path / "g.wav"
    result = runner.invoke(cli, ["audio", "gain", "in.wav", str(out), "--gain-db", "3"])
    assert result.exit_code == 0
    assert "Gain-adjusted audio" in result.output


def test_resample_error(runner, tmp_path, mocker):
    mocker.patch("bioamla.audio.load_audio", side_effect=AudioLoadError("boom"))
    out = tmp_path / "r.wav"
    result = runner.invoke(cli, ["audio", "resample", "in.wav", str(out), "-r", "8000"])
    assert result.exit_code != 0


# --------------------------------------------------------------------------- #
# filter
# --------------------------------------------------------------------------- #


def test_filter_lowpass(runner, tmp_path, mock_load_save, mocker):
    mocker.patch("bioamla.audio.lowpass_filter", return_value=np.zeros(16000, dtype=np.float32))
    out = tmp_path / "f.wav"
    result = runner.invoke(cli, ["audio", "filter", "in.wav", str(out), "--lowpass", "4000"])
    assert result.exit_code == 0
    assert "lowpass filter" in result.output


def test_filter_highpass(runner, tmp_path, mock_load_save, mocker):
    mocker.patch("bioamla.audio.highpass_filter", return_value=np.zeros(16000, dtype=np.float32))
    out = tmp_path / "f.wav"
    result = runner.invoke(cli, ["audio", "filter", "in.wav", str(out), "--highpass", "200"])
    assert result.exit_code == 0
    assert "highpass filter" in result.output


def test_filter_bandpass(runner, tmp_path, mock_load_save, mocker):
    mocker.patch("bioamla.audio.bandpass_filter", return_value=np.zeros(16000, dtype=np.float32))
    out = tmp_path / "f.wav"
    result = runner.invoke(
        cli,
        [
            "audio",
            "filter",
            "in.wav",
            str(out),
            "--bandpass-low",
            "500",
            "--bandpass-high",
            "4000",
        ],
    )
    assert result.exit_code == 0
    assert "bandpass filter" in result.output


def test_filter_no_option(runner, tmp_path):
    out = tmp_path / "f.wav"
    result = runner.invoke(cli, ["audio", "filter", "in.wav", str(out)])
    assert result.exit_code != 0
    assert "Must specify" in result.output


def test_filter_bandpass_incomplete(runner, tmp_path):
    out = tmp_path / "f.wav"
    result = runner.invoke(cli, ["audio", "filter", "in.wav", str(out), "--bandpass-low", "500"])
    assert result.exit_code != 0
    assert "must be specified together" in result.output


# --------------------------------------------------------------------------- #
# visualize
# --------------------------------------------------------------------------- #


def test_visualize_default_output(runner, mocker):
    gen = mocker.patch("bioamla.viz.generate_spectrogram")
    result = runner.invoke(cli, ["audio", "visualize", "/some/song.wav"])
    assert result.exit_code == 0
    assert "Visualization saved to" in result.output
    gen.assert_called_once()
    assert gen.call_args.kwargs["output_path"] == "song_mel.png"


def test_visualize_explicit_output(runner, tmp_path, mocker):
    gen = mocker.patch("bioamla.viz.generate_spectrogram")
    out = tmp_path / "viz.png"
    result = runner.invoke(cli, ["audio", "visualize", "in.wav", "-o", str(out), "-t", "waveform"])
    assert result.exit_code == 0
    assert gen.call_args.kwargs["viz_type"] == "waveform"


def test_visualize_error(runner, mocker):
    from bioamla.exceptions import ProcessingError

    mocker.patch("bioamla.viz.generate_spectrogram", side_effect=ProcessingError("v"))
    result = runner.invoke(cli, ["audio", "visualize", "in.wav"])
    assert result.exit_code != 0
