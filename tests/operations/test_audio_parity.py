"""Every `audio` command (but `play`) can be reproduced with library calls alone.

Each test runs a command with `--json`, then does the same with bioamla's public
library functions (no CLI, no operations registry) and checks the results and the
files written agree.
"""

import json
import shutil

import numpy as np
import pytest
import soundfile as sf
from click.testing import CliRunner

from bioamla import audio as lib
from bioamla.cli.cli import cli


def _cli(*args: str) -> dict:
    result = CliRunner().invoke(cli, ["--json", *args])
    assert result.exit_code == 0, result.output
    return json.loads(result.stdout)["result"]


def _same_audio(a, b) -> None:
    da, ra = sf.read(str(a), dtype="float32")
    db, rb = sf.read(str(b), dtype="float32")
    assert ra == rb
    np.testing.assert_array_equal(da, db)


def test_info(test_audio_path_3s):
    got = _cli("audio", "info", test_audio_path_3s, "--metadata")
    want = lib.describe_audio_file(test_audio_path_3s, recording_metadata=True).to_dict()
    assert got["recording_metadata"] == want["recording_metadata"]
    assert got["info"]["duration"] == want["info"].duration


def test_metadata(test_audio_path_3s):
    got = _cli("audio", "metadata", test_audio_path_3s)
    report = lib.recording_metadata_report(test_audio_path_3s)
    assert got["standards"] == json.loads(json.dumps(report.standards, default=str))
    assert got["summary"] == json.loads(json.dumps(report.summary.to_dict(), default=str))


def test_list(test_audio_dir):
    got = _cli("audio", "list", test_audio_dir)
    files = lib.list_audio_files(test_audio_dir, recursive=True)
    assert got["files"] == [str(f) for f in files] and got["count"] == len(files)


TRANSFORMS = [
    ("trim", ["-s", "0.5", "-d", "1"], lambda f, o: lib.trim_audio_file(f, o, start=0.5, duration=1.0)),
    ("normalize", ["-t", "-6", "-m", "rms"], lambda f, o: lib.normalize_audio_file(f, o, target_db=-6.0, method="rms")),
    ("resample", ["-r", "8000"], lambda f, o: lib.resample_audio_file(f, o, 8000)),
    ("filter", ["--highpass", "300"], lambda f, o: lib.filter_audio_file(f, o, highpass=300.0)),
    ("denoise", ["--strength", "0.5"], lambda f, o: lib.denoise_audio_file(f, o, strength=0.5)),
    ("add-noise", ["--snr-db", "10", "--seed", "1"], lambda f, o: lib.add_noise_audio_file(f, o, 10.0, seed=1)),
    ("gain", ["--gain-db", "-3"], lambda f, o: lib.gain_audio_file(f, o, -3.0)),
    ("convert", ["-r", "8000", "-c", "2"], lambda f, o: lib.convert_audio_file(f, o, target_format=None, target_sample_rate=8000, target_channels=2)),
]  # fmt: skip

LIBROSA_TRANSFORMS = [
    ("pitch-shift", ["--steps", "2"], lambda f, o: lib.pitch_shift_audio_file(f, o, 2.0)),
    ("time-stretch", ["--rate", "1.5"], lambda f, o: lib.time_stretch_audio_file(f, o, 1.5)),
]


def _check_transform(command, args, call, file, tmp_path):
    got = _cli("audio", command, file, str(tmp_path / "cli.wav"), *args)
    assert got["output"] == str(tmp_path / "cli.wav")
    call(file, tmp_path / "lib.wav")
    _same_audio(tmp_path / "cli.wav", tmp_path / "lib.wav")


@pytest.mark.parametrize(("command", "args", "call"), TRANSFORMS)
def test_transforms(command, args, call, test_audio_path_3s, tmp_path):
    _check_transform(command, args, call, test_audio_path_3s, tmp_path)


@pytest.mark.parametrize(("command", "args", "call"), LIBROSA_TRANSFORMS)
def test_librosa_transforms(command, args, call, test_audio_path_3s, tmp_path):
    pytest.importorskip("librosa")
    _check_transform(command, args, call, test_audio_path_3s, tmp_path)


def test_segment(test_audio_path_3s, tmp_path):
    got = _cli("audio", "segment", test_audio_path_3s, str(tmp_path / "cli"), "-d", "1.25")
    segments = lib.segment_audio_file(
        test_audio_path_3s,
        str(tmp_path / "lib"),
        duration=1.25,
        include_partial=False,
        label="",
    )
    assert got["segments"] == len(segments) == 2
    for seg in segments:
        _same_audio(tmp_path / "cli" / seg.segment_path.name, seg.segment_path)


def test_features(test_audio_path_3s, tmp_path):
    pytest.importorskip("librosa")
    out = tmp_path / "cli.csv"
    got = _cli("audio", "features", test_audio_path_3s, "-o", str(out), "--n-mfcc", "8")
    result = lib.frame_features_file(
        test_audio_path_3s, tmp_path / "lib.csv", config=lib.FrameFeatureConfig(n_mfcc=8)
    )
    assert got["n_frames"] == result.to_dict()["n_frames"]
    assert out.read_text() == (tmp_path / "lib.csv").read_text()


def test_visualize(test_audio_path_3s, tmp_path):
    pytest.importorskip("matplotlib")
    from bioamla.viz import generate_spectrogram

    out = tmp_path / "cli.png"
    _cli("audio", "visualize", test_audio_path_3s, "-o", str(out), "-t", "waveform")
    generate_spectrogram(
        audio_path=test_audio_path_3s, output_path=str(tmp_path / "lib.png"), viz_type="waveform"
    )
    assert out.stat().st_size > 0 and (tmp_path / "lib.png").stat().st_size > 0


@pytest.mark.slow
def test_video(test_audio_path_3s, tmp_path):
    pytest.importorskip("matplotlib")
    if shutil.which("ffmpeg") is None:
        pytest.skip("needs ffmpeg")
    from bioamla.viz import generate_spectrogram_video

    args = ["--fps", "5", "--size", "160x90", "--no-audio"]
    got = _cli("audio", "video", test_audio_path_3s, "-o", str(tmp_path / "cli.mp4"), *args)
    result = generate_spectrogram_video(
        audio_path=test_audio_path_3s,
        output_path=str(tmp_path / "lib.mp4"),
        fps=5,
        width=160,
        height=90,
        include_audio=False,
    )
    assert got["frames"] == result.frames and got["duration"] == result.duration
