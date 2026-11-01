"""CLI tests for `bioamla audio` single-file commands.

Domain functions are imported lazily from `bioamla.audio` (and `bioamla.viz`)
inside the command bodies, so we patch the symbols there. Heavy audio/ML work is
mocked; tests exercise arg-parsing, branching and output formatting only.
"""

from datetime import UTC
from unittest.mock import MagicMock, PropertyMock

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
    load = mocker.patch("bioamla.audio.files.load_audio", return_value=(audio, 16000))
    save = mocker.patch("bioamla.audio.files.save_audio")
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
        "play",
    ],
)
def test_audio_subcommand_help(runner, cmd):
    result = runner.invoke(cli, ["audio", cmd, "--help"])
    assert result.exit_code == 0


# --------------------------------------------------------------------------- #
# info
# --------------------------------------------------------------------------- #


def test_info(runner, mocker):
    from bioamla.audio import AudioInfo

    info = AudioInfo(
        duration=1.0,
        sample_rate=16000,
        channels=1,
        samples=16000,
        bit_depth=16,
        format="WAV",
        subtype="PCM_16",
    )
    get_info = mocker.patch("bioamla.audio.files.get_audio_info", return_value=info)
    load = mocker.patch("bioamla.audio.load_audio_data")
    decode = mocker.patch("bioamla.audio.files.load_audio")
    result = runner.invoke(cli, ["audio", "info", "file.wav"])
    assert result.exit_code == 0
    assert "Format: WAV" in result.output
    assert "Duration: 1.00s" in result.output
    assert "Sample rate: 16000 Hz" in result.output
    assert "Bit depth: 16" in result.output
    assert "Subtype: PCM_16" in result.output
    get_info.assert_called_once_with("file.wav")
    load.assert_not_called()
    decode.assert_not_called()


def _fake_info():
    from bioamla.audio import AudioInfo

    return AudioInfo(
        duration=1.0,
        sample_rate=16000,
        channels=1,
        samples=16000,
        bit_depth=16,
        format="WAV",
        subtype="PCM_16",
    )


def test_info_metadata_flag(runner, mocker):
    from datetime import datetime

    from bioamla.audio import RecordingMetadata

    mocker.patch("bioamla.audio.files.get_audio_info", return_value=_fake_info())
    mocker.patch(
        "bioamla.audio.metadata.get_recording_metadata",
        return_value=RecordingMetadata(
            recorded_at=datetime(2021, 11, 12, 19, 30, tzinfo=UTC),
            device_model="AudioMoth",
            device_id="248D9B045EC9EE79",
            gain="medium",
            temperature_c=14.0,
            battery_voltage=4.1,
            sources=("info", "audiomoth"),
        ),
    )
    result = runner.invoke(cli, ["audio", "info", "file.wav", "--metadata"])
    assert result.exit_code == 0, result.output
    assert "Recorder metadata" in result.output
    assert "Recorded at: 2021-11-12T19:30:00+00:00" in result.output
    assert "Device ID: 248D9B045EC9EE79" in result.output
    assert "Temperature: 14.0 °C" in result.output
    assert "Battery: 4.1 V" in result.output
    assert "Sources: info+audiomoth" in result.output
    assert "Latitude" not in result.output


def test_info_metadata_flag_empty(runner, mocker):
    from bioamla.audio import RecordingMetadata

    mocker.patch("bioamla.audio.files.get_audio_info", return_value=_fake_info())
    mocker.patch("bioamla.audio.metadata.get_recording_metadata", return_value=RecordingMetadata())
    result = runner.invoke(cli, ["audio", "info", "file.wav", "--metadata"])
    assert result.exit_code == 0
    assert "Recorder metadata" not in result.output


def test_info_metadata_flag_non_wav(runner, mocker):
    from bioamla.exceptions import InvalidInputError

    mocker.patch("bioamla.audio.files.get_audio_info", return_value=_fake_info())
    mocker.patch(
        "bioamla.audio.metadata.get_recording_metadata", side_effect=InvalidInputError("no")
    )
    result = runner.invoke(cli, ["audio", "info", "file.flac", "--metadata"])
    assert result.exit_code == 0
    assert "Duration: 1.00s" in result.output


def test_info_error(runner, mocker):
    mocker.patch("bioamla.audio.files.get_audio_info", side_effect=AudioLoadError("bad"))
    result = runner.invoke(cli, ["audio", "info", "file.wav"])
    assert result.exit_code != 0
    assert "bad" in result.output


# --------------------------------------------------------------------------- #
# metadata
# --------------------------------------------------------------------------- #


def test_metadata(runner, mocker):
    mocker.patch(
        "bioamla.audio.metadata.dump_recording_metadata", return_value={"file": "file.wav"}
    )
    from bioamla.audio import RecordingMetadata

    mocker.patch(
        "bioamla.audio.metadata.get_recording_metadata",
        return_value=RecordingMetadata(
            latitude=36.52, longitude=-82.21, device_make="Wildlife Acoustics", sources=("guano",)
        ),
    )
    result = runner.invoke(cli, ["audio", "metadata", "file.wav"])
    assert result.exit_code == 0, result.output
    assert "Latitude: 36.52" in result.output
    assert "Longitude: -82.21" in result.output
    assert "Device make: Wildlife Acoustics" in result.output
    assert "Sources: guano" in result.output


def test_metadata_empty(runner, mocker):
    mocker.patch(
        "bioamla.audio.metadata.dump_recording_metadata", return_value={"file": "file.wav"}
    )
    from bioamla.audio import RecordingMetadata

    mocker.patch("bioamla.audio.metadata.get_recording_metadata", return_value=RecordingMetadata())
    result = runner.invoke(cli, ["audio", "metadata", "file.wav"])
    assert result.exit_code == 0
    assert "No recorder metadata" in result.output


def test_metadata_json(runner, mocker):
    import json

    from bioamla.audio import RecordingMetadata

    mocker.patch("bioamla.audio.metadata.get_recording_metadata", return_value=RecordingMetadata())
    dump = mocker.patch(
        "bioamla.audio.metadata.dump_recording_metadata",
        return_value={"file": "file.wav", "sources": ["guano"], "guano": {"Make": "X"}},
    )
    result = runner.invoke(cli, ["--json", "audio", "metadata", "file.wav"])
    assert result.exit_code == 0, result.output
    assert json.loads(result.stdout)["result"]["standards"]["guano"] == {"Make": "X"}
    dump.assert_called_once_with("file.wav")


def test_metadata_error(runner, mocker):
    mocker.patch(
        "bioamla.audio.metadata.dump_recording_metadata", return_value={"file": "file.wav"}
    )
    mocker.patch("bioamla.audio.metadata.get_recording_metadata", side_effect=AudioLoadError("bad"))
    result = runner.invoke(cli, ["audio", "metadata", "file.wav"])
    assert result.exit_code != 0
    assert "bad" in result.output


# --------------------------------------------------------------------------- #
# list
# --------------------------------------------------------------------------- #


def test_list_found(runner, mocker):
    mocker.patch("bioamla.audio.operations.list_audio_files", return_value=["a.wav", "b.wav"])
    result = runner.invoke(cli, ["audio", "list", "somedir"])
    assert result.exit_code == 0
    assert "Found 2 audio file(s)" in result.output
    assert "a.wav" in result.output


def test_list_empty(runner, mocker):
    mocker.patch("bioamla.audio.operations.list_audio_files", return_value=[])
    result = runner.invoke(cli, ["audio", "list", "somedir", "--no-recursive"])
    assert result.exit_code == 0
    assert "No audio files found" in result.output


def test_list_error(runner, mocker):
    from bioamla.exceptions import NotFoundError

    mocker.patch("bioamla.audio.operations.list_audio_files", side_effect=NotFoundError("nope"))
    result = runner.invoke(cli, ["audio", "list", "somedir"])
    assert result.exit_code != 0


# --------------------------------------------------------------------------- #
# play
# --------------------------------------------------------------------------- #


def _fake_player():
    """A player mock that reports one is_playing poll then finishes."""
    player = MagicMock()
    player.duration = 1.4
    type(player).is_playing = PropertyMock(side_effect=[True, False])
    return player


def test_play(runner, mocker):
    player = _fake_player()
    play = mocker.patch("bioamla.audio.playback.play_audio", return_value=player)
    result = runner.invoke(cli, ["audio", "play", "file.wav"])
    assert result.exit_code == 0
    assert "Playing file.wav (1.40s)" in result.output
    assert "Done." in result.output
    play.assert_called_once_with("file.wav", loop=False, block=False)


def test_play_loop_flag(runner, mocker):
    player = _fake_player()
    play = mocker.patch("bioamla.audio.playback.play_audio", return_value=player)
    result = runner.invoke(cli, ["audio", "play", "file.wav", "--loop"])
    assert result.exit_code == 0
    assert "[loop]" in result.output
    assert play.call_args.kwargs["loop"] is True


def test_play_interrupt(runner, mocker):
    player = MagicMock()
    player.duration = 1.4
    type(player).is_playing = PropertyMock(side_effect=KeyboardInterrupt)
    mocker.patch("bioamla.audio.playback.play_audio", return_value=player)
    result = runner.invoke(cli, ["audio", "play", "file.wav"])
    assert result.exit_code == 0
    assert "Stopped." in result.output
    player.stop.assert_called_once()


def test_play_error(runner, mocker):
    mocker.patch("bioamla.audio.playback.play_audio", side_effect=AudioLoadError("bad"))
    result = runner.invoke(cli, ["audio", "play", "missing.wav"])
    assert result.exit_code != 0
    assert "bad" in result.output


# --------------------------------------------------------------------------- #
# convert
# --------------------------------------------------------------------------- #


def test_convert(runner, tmp_path, mocker):
    mocker.patch(
        "bioamla.audio.convert.load_audio_channels",
        return_value=(np.zeros(16000, np.float32), 16000),
    )
    save = mocker.patch("bioamla.audio.convert.save_audio", return_value="out.flac")
    out = tmp_path / "out.flac"
    result = runner.invoke(
        cli, ["audio", "convert", "in.wav", str(out), "-r", "8000", "-f", "flac"]
    )
    assert result.exit_code == 0, result.output
    assert "Converted" in result.output
    save.assert_called_once()
    samples, rate = save.call_args.args[1:3]
    assert rate == 8000 and len(samples) == 8000
    assert save.call_args.kwargs["format"] == "flac"


def test_convert_stereo_to_mono(runner, tmp_path, mocker):
    mocker.patch(
        "bioamla.audio.convert.load_audio_channels",
        return_value=(np.zeros((16000, 2), np.float32), 16000),
    )
    save = mocker.patch("bioamla.audio.convert.save_audio", return_value="out.wav")
    out = tmp_path / "out.wav"
    result = runner.invoke(cli, ["audio", "convert", "in.wav", str(out), "-c", "1"])
    assert result.exit_code == 0, result.output
    assert save.call_args.args[1].ndim == 1


def test_convert_mono_to_stereo(runner, tmp_path, mocker):
    mocker.patch(
        "bioamla.audio.convert.load_audio_channels",
        return_value=(np.zeros(16000, np.float32), 16000),
    )
    save = mocker.patch("bioamla.audio.convert.save_audio", return_value="out.wav")
    out = tmp_path / "out.wav"
    result = runner.invoke(cli, ["audio", "convert", "in.wav", str(out), "-c", "2"])
    assert result.exit_code == 0, result.output
    assert save.call_args.args[1].shape == (16000, 2)
    assert save.call_args.kwargs["format"] is None  # from the file name


def test_convert_bit_depth_is_ignored_with_a_warning(runner, tmp_path, mocker):
    mocker.patch(
        "bioamla.audio.convert.load_audio_channels",
        return_value=(np.zeros(16000, np.float32), 16000),
    )
    mocker.patch("bioamla.audio.convert.save_audio", return_value="out.wav")
    result = runner.invoke(cli, ["audio", "convert", "in.wav", str(tmp_path / "o.wav"), "-b", "24"])
    assert result.exit_code == 0
    assert "--bit-depth is not supported" in result.stderr


def test_convert_error(runner, tmp_path, mocker):
    mocker.patch("bioamla.audio.convert.load_audio_channels", side_effect=AudioLoadError("broken"))
    result = runner.invoke(cli, ["audio", "convert", "in.wav", str(tmp_path / "o.wav")])
    assert result.exit_code != 0
    assert "broken" in result.output


# --------------------------------------------------------------------------- #
# segment
# --------------------------------------------------------------------------- #


def test_segment(runner, tmp_path, mocker):
    audio = np.zeros(16000 * 3, dtype=np.float32)
    mocker.patch("bioamla.audio.batch.load_audio", return_value=(audio, 16000))
    mocker.patch("bioamla.audio.batch.save_audio")
    out_dir = tmp_path / "segs"
    result = runner.invoke(cli, ["audio", "segment", "in.wav", str(out_dir), "-d", "1.0"])
    assert result.exit_code == 0
    assert "Created 3 segments" in result.output


def test_segment_overlap_too_large(runner, tmp_path, mocker):
    audio = np.zeros(16000 * 3, dtype=np.float32)
    mocker.patch("bioamla.audio.batch.load_audio", return_value=(audio, 16000))
    mocker.patch("bioamla.audio.batch.save_audio")
    out_dir = tmp_path / "segs"
    result = runner.invoke(
        cli, ["audio", "segment", "in.wav", str(out_dir), "-d", "1.0", "-o", "2.0"]
    )
    assert result.exit_code != 0


# --------------------------------------------------------------------------- #
# trim
# --------------------------------------------------------------------------- #


def test_trim(runner, tmp_path, mock_load_save, mocker):
    mocker.patch(
        "bioamla.audio.processing.trim_audio", return_value=np.zeros(8000, dtype=np.float32)
    )
    out = tmp_path / "t.wav"
    result = runner.invoke(cli, ["audio", "trim", "in.wav", str(out), "-s", "0.1", "-e", "0.5"])
    assert result.exit_code == 0
    assert "Trimmed audio saved" in result.output


def test_trim_with_duration(runner, tmp_path, mock_load_save, mocker):
    mocker.patch(
        "bioamla.audio.processing.trim_audio", return_value=np.zeros(8000, dtype=np.float32)
    )
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
    mocker.patch(
        "bioamla.audio.processing.peak_normalize", return_value=np.zeros(16000, dtype=np.float32)
    )
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
    mocker.patch(
        "bioamla.audio.processing.resample_audio", return_value=np.zeros(8000, dtype=np.float32)
    )
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
    mocker.patch(
        "bioamla.audio.processing.pitch_shift", return_value=np.zeros(16000, dtype=np.float32)
    )
    out = tmp_path / "p.wav"
    result = runner.invoke(cli, ["audio", "pitch-shift", "in.wav", str(out), "-n", "2"])
    assert result.exit_code == 0
    assert "Pitch-shifted audio" in result.output


def test_time_stretch(runner, tmp_path, mock_load_save, mocker):
    mocker.patch(
        "bioamla.audio.processing.time_stretch", return_value=np.zeros(16000, dtype=np.float32)
    )
    out = tmp_path / "s.wav"
    result = runner.invoke(cli, ["audio", "time-stretch", "in.wav", str(out), "-r", "1.5"])
    assert result.exit_code == 0
    assert "Time-stretched audio" in result.output


def test_add_noise(runner, tmp_path, mock_load_save, mocker):
    mocker.patch(
        "bioamla.audio.processing.add_noise", return_value=np.zeros(16000, dtype=np.float32)
    )
    out = tmp_path / "an.wav"
    result = runner.invoke(
        cli, ["audio", "add-noise", "in.wav", str(out), "--snr-db", "10", "--seed", "1"]
    )
    assert result.exit_code == 0
    assert "Noisy audio" in result.output


def test_gain(runner, tmp_path, mock_load_save, mocker):
    mocker.patch(
        "bioamla.audio.processing.apply_gain", return_value=np.zeros(16000, dtype=np.float32)
    )
    out = tmp_path / "g.wav"
    result = runner.invoke(cli, ["audio", "gain", "in.wav", str(out), "--gain-db", "3"])
    assert result.exit_code == 0
    assert "Gain-adjusted audio" in result.output


def test_resample_error(runner, tmp_path, mocker):
    mocker.patch("bioamla.audio.files.load_audio", side_effect=AudioLoadError("boom"))
    out = tmp_path / "r.wav"
    result = runner.invoke(cli, ["audio", "resample", "in.wav", str(out), "-r", "8000"])
    assert result.exit_code != 0


# --------------------------------------------------------------------------- #
# filter
# --------------------------------------------------------------------------- #


def test_filter_lowpass(runner, tmp_path, mock_load_save, mocker):
    mocker.patch(
        "bioamla.audio.processing.lowpass_filter", return_value=np.zeros(16000, dtype=np.float32)
    )
    out = tmp_path / "f.wav"
    result = runner.invoke(cli, ["audio", "filter", "in.wav", str(out), "--lowpass", "4000"])
    assert result.exit_code == 0
    assert "lowpass filter" in result.output


def test_filter_highpass(runner, tmp_path, mock_load_save, mocker):
    mocker.patch(
        "bioamla.audio.processing.highpass_filter", return_value=np.zeros(16000, dtype=np.float32)
    )
    out = tmp_path / "f.wav"
    result = runner.invoke(cli, ["audio", "filter", "in.wav", str(out), "--highpass", "200"])
    assert result.exit_code == 0
    assert "highpass filter" in result.output


def test_filter_bandpass(runner, tmp_path, mock_load_save, mocker):
    mocker.patch(
        "bioamla.audio.processing.bandpass_filter", return_value=np.zeros(16000, dtype=np.float32)
    )
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


# ---------------------------------------------------------------------------
# video
# ---------------------------------------------------------------------------


def _video_result(**overrides):
    from bioamla.viz import SpectrogramVideoResult

    kwargs = {
        "output_path": "song.mp4",
        "duration": 2.0,
        "start": 0.0,
        "end": 2.0,
        "fps": 30,
        "frames": 60,
        "width": 1280,
        "height": 720,
        "has_audio": True,
    }
    kwargs.update(overrides)
    return SpectrogramVideoResult(**kwargs)


def test_video_default_output(runner, mocker):
    gen = mocker.patch("bioamla.viz.generate_spectrogram_video", return_value=_video_result())
    result = runner.invoke(cli, ["audio", "video", "/some/song.wav", "-q"])
    assert result.exit_code == 0, result.output
    assert "Video saved to" in result.output
    kwargs = gen.call_args.kwargs
    assert kwargs["output_path"] == "song.mp4"
    assert (kwargs["width"], kwargs["height"]) == (1280, 720)
    assert kwargs["viz_type"] == "mel" and kwargs["include_audio"] is True
    assert kwargs["motion"] == "scroll" and kwargs["window_seconds"] == 4.0


def test_video_page_mode_whole_recording(runner, mocker):
    gen = mocker.patch("bioamla.viz.generate_spectrogram_video", return_value=_video_result())
    result = runner.invoke(
        cli, ["audio", "video", "in.wav", "-t", "waveform", "-m", "page", "-w", "0"]
    )
    assert result.exit_code == 0, result.output
    kwargs = gen.call_args.kwargs
    assert kwargs["viz_type"] == "waveform"
    assert kwargs["motion"] == "page" and kwargs["window_seconds"] is None


def test_video_start_end(runner, mocker):
    gen = mocker.patch("bioamla.viz.generate_spectrogram_video", return_value=_video_result())
    result = runner.invoke(cli, ["audio", "video", "in.wav", "-s", "90", "-e", "120"])
    assert result.exit_code == 0, result.output
    assert gen.call_args.kwargs["start"] == 90.0 and gen.call_args.kwargs["end"] == 120.0


def test_video_options(runner, mocker):
    gen = mocker.patch("bioamla.viz.generate_spectrogram_video", return_value=_video_result())
    result = runner.invoke(
        cli,
        [
            "audio",
            "video",
            "in.wav",
            "-o",
            "out.mp4",
            "-t",
            "stft",
            "--size",
            "640x360",
            "--fps",
            "24",
            "-w",
            "8",
            "--no-waveform",
            "--no-axes",
            "--no-audio",
            "--playhead",
            "1",
        ],
    )
    assert result.exit_code == 0, result.output
    kwargs = gen.call_args.kwargs
    assert kwargs["output_path"] == "out.mp4"
    assert kwargs["viz_type"] == "stft"
    assert (kwargs["width"], kwargs["height"]) == (640, 360)
    assert kwargs["fps"] == 24 and kwargs["window_seconds"] == 8.0
    assert kwargs["waveform"] is False and kwargs["show_axes"] is False
    assert kwargs["include_audio"] is False and kwargs["playhead"] == 1.0


def test_video_progress_callback_drives_bar(runner, mocker):
    def fake(**kwargs):
        for i in range(1, 4):
            kwargs["on_progress"](i, 3)
        return _video_result(frames=3)

    mocker.patch("bioamla.viz.generate_spectrogram_video", side_effect=fake)
    result = runner.invoke(cli, ["audio", "video", "in.wav"])
    assert result.exit_code == 0, result.output
    assert "3 frames" in result.output


def test_video_bad_size(runner, mocker):
    mocker.patch("bioamla.viz.generate_spectrogram_video")
    result = runner.invoke(cli, ["audio", "video", "in.wav", "--size", "big"])
    assert result.exit_code != 0
    assert "WIDTHxHEIGHT" in result.output


def test_video_error(runner, mocker):
    from bioamla.exceptions import DependencyError

    mocker.patch("bioamla.viz.generate_spectrogram_video", side_effect=DependencyError("no ffmpeg"))
    result = runner.invoke(cli, ["audio", "video", "in.wav"])
    assert result.exit_code != 0
    assert "no ffmpeg" in result.output


def test_visualize_boxes_and_annotations(runner, tmp_path, mocker):
    gen = mocker.patch("bioamla.viz.generate_spectrogram")
    ann = tmp_path / "det.csv"
    ann.write_text("start_time,end_time,low_freq,high_freq,label\n0.5,1.0,500,5000,call\n")
    result = runner.invoke(cli, ["audio", "visualize", "in.wav", "--box", "1,2",
                                 "--box", "2.5, 3, 100, 900, a frog, really",
                                 "--annotations", str(ann)])  # fmt: skip
    assert result.exit_code == 0, result.output
    from bioamla.viz import SpectrogramBox

    assert gen.call_args.kwargs["boxes"] == [
        SpectrogramBox(1, 2), SpectrogramBox(2.5, 3, 100, 900, "a frog, really"),
        SpectrogramBox(0.5, 1.0, 500, 5000, "call"),
    ]  # fmt: skip


@pytest.mark.parametrize("box", ["1", "a,b", "2,1", "1,2,900,100", "1,2,x,100"])
def test_visualize_bad_box(runner, mocker, box):
    mocker.patch("bioamla.viz.generate_spectrogram")
    result = runner.invoke(cli, ["audio", "visualize", "in.wav", "--box", box])
    assert result.exit_code != 0
    assert "box" in result.output.lower()


def test_visualize_labelled_full_height_boxes(runner, mocker):
    gen = mocker.patch("bioamla.viz.generate_spectrogram")
    result = runner.invoke(cli, ["audio", "visualize", "in.wav", "--box", "1,2,frog",
                                 "--box", "3,4,,,toad"])  # fmt: skip
    assert result.exit_code == 0, result.output
    from bioamla.viz import SpectrogramBox

    assert gen.call_args.kwargs["boxes"] == [SpectrogramBox(1, 2, label="frog"),
                                             SpectrogramBox(3, 4, label="toad")]  # fmt: skip
