"""Writing formats soundfile can't (MP3, M4A, OGG) through ffmpeg, under a time limit."""

import os
import shutil
import time

import numpy as np
import pytest
import soundfile as sf

from bioamla.audio import AudioData, load_audio_data, save_audio_data
from bioamla.exceptions import AudioSaveError, ConfigError, DependencyError

SR = 16000


def tone(channels: int = 1) -> AudioData:
    t = np.arange(SR) / SR
    mono = (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    samples = mono if channels == 1 else np.stack([mono, np.zeros_like(mono)], axis=1)
    return AudioData(samples=samples, sample_rate=SR, channels=channels)


@pytest.mark.usefixtures("requires_ffmpeg_cli")
@pytest.mark.parametrize("name", ["out.mp3", "out.m4a", "out.ogg"])
def test_lossy_formats_round_trip(tmp_path, name):
    out = save_audio_data(tone(channels=2), tmp_path / name)
    back = load_audio_data(out, keep_channels=True)
    assert back.channels == 2 and abs(back.duration - 1.0) < 0.1
    assert np.abs(back.samples[:, 0]).max() > 0.3 > np.abs(back.samples[:, 1]).max()
    assert [p.name for p in tmp_path.iterdir()] == [name]  # no temporary file left


def test_a_stalled_encoder_is_stopped_and_leaves_the_old_file(
    tmp_path, monkeypatch, stalled_ffmpeg
):
    out = tmp_path / "out.mp3"
    out.write_bytes(b"the previous file")
    monkeypatch.setenv("BIOAMLA_ENCODE_TIMEOUT", "0.5")

    began = time.monotonic()
    with (
        stalled_ffmpeg.expect_killed(),
        pytest.raises(AudioSaveError, match=r"still encoding after 0\.5 s.*BIOAMLA_ENCODE_TIMEOUT"),
    ):
        save_audio_data(tone(), out)
    assert time.monotonic() - began < 10
    assert out.read_bytes() == b"the previous file"
    left = {p.name for p in tmp_path.iterdir()} - {"pids"}  # no temporary file
    assert left == {"bin", "out.mp3"}


@pytest.mark.skipif(os.name == "nt", reason="a shell-script ffmpeg")
def test_a_failed_encode_leaves_nothing(tmp_path, monkeypatch):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    (bin_dir / "ffmpeg").write_text("#!/bin/sh\necho 'no such encoder' >&2\nexit 1\n")
    (bin_dir / "ffmpeg").chmod(0o755)
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ['PATH']}")
    with pytest.raises(AudioSaveError, match="no such encoder"):
        save_audio_data(tone(), tmp_path / "out.mp3")
    assert sorted(p.name for p in tmp_path.iterdir()) == ["bin"]


def test_a_bad_encode_timeout_is_a_config_error(tmp_path, monkeypatch):
    if shutil.which("ffmpeg") is None:
        pytest.skip("needs ffmpeg on the PATH to reach the limit")
    for bad in ("soon", "0", "-1", "inf", "nan"):
        monkeypatch.setenv("BIOAMLA_ENCODE_TIMEOUT", bad)
        with pytest.raises(ConfigError, match="BIOAMLA_ENCODE_TIMEOUT must be a positive"):
            save_audio_data(tone(), tmp_path / "out.mp3")


def test_wav_and_flac_need_no_ffmpeg(tmp_path, monkeypatch):
    monkeypatch.setattr(shutil, "which", lambda name: None)
    for name in ("out.wav", "out.flac"):
        save_audio_data(tone(), tmp_path / name)
        assert sf.info(tmp_path / name).frames == SR
    with pytest.raises(DependencyError, match="ffmpeg is required to write MP3"):
        save_audio_data(tone(), tmp_path / "out.mp3")


@pytest.mark.usefixtures("requires_ffmpeg_cli")
@pytest.mark.skipif(os.name == "nt", reason="symlinks and POSIX modes")
def test_saving_over_a_symlink_writes_its_target_and_keeps_the_mode(tmp_path):
    target = tmp_path / "library" / "song.mp3"
    target.parent.mkdir()
    target.write_bytes(b"old")
    target.chmod(0o664)
    link = tmp_path / "song.mp3"
    link.symlink_to(target)
    save_audio_data(tone(), link)
    assert link.is_symlink() and link.resolve() == target
    assert target.stat().st_size > 3 and (target.stat().st_mode & 0o777) == 0o664


@pytest.mark.usefixtures("requires_ffmpeg_cli")
def test_nan_and_infinite_samples_are_written_as_finite_audio(tmp_path):
    samples = tone().samples.copy()
    samples[:100] = np.nan
    samples[100:200] = np.inf
    out = save_audio_data(AudioData(samples=samples, sample_rate=SR), tmp_path / "x.mp3")
    assert np.isfinite(load_audio_data(out).samples).all()


def test_input_errors_are_not_save_errors(tmp_path):
    from bioamla.exceptions import InvalidInputError

    three = AudioData(samples=np.zeros((SR, 3), dtype=np.float32), sample_rate=SR, channels=3)
    with pytest.raises(InvalidInputError, match="MP3 holds at most 2 channels"):
        save_audio_data(three, tmp_path / "x.mp3")
