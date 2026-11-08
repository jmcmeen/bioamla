"""Tests for windowed loading (load_audio_data start/duration) and stream_audio_data."""

import subprocess

import numpy as np
import pytest
import soundfile as sf

from bioamla.audio import load_audio_data, stream_audio_data
from bioamla.exceptions import InvalidInputError, NotFoundError

SR = 16000


@pytest.fixture
def ramp_wav(tmp_path) -> str:
    """3 s of a ramp, so every sample's value gives away its position."""
    samples = np.linspace(-0.9, 0.9, 3 * SR, dtype=np.float32)
    path = tmp_path / "ramp.wav"
    sf.write(path, samples, SR, subtype="FLOAT")
    return str(path)


@pytest.fixture
def noise_mp3(tmp_path) -> str:
    """3 s of noise as MP3 (noise exposes any misalignment between decodes)."""
    wav = tmp_path / "noise.wav"
    rng = np.random.default_rng(0)
    sf.write(wav, rng.normal(0, 0.1, 3 * SR).astype(np.float32), SR)
    mp3 = tmp_path / "noise.mp3"
    # Encoded by the ffmpeg CLI rather than bioamla's own encoder, so the decoder is
    # tested against a file it didn't write.
    subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(wav), "-f", "mp3", "-y", str(mp3)],
        stdin=subprocess.DEVNULL,
        check=True,
        timeout=60,
    )
    return str(mp3)


@pytest.fixture
def ramp_flac(tmp_path) -> str:
    samples = np.linspace(-0.9, 0.9, 3 * SR, dtype=np.float32)
    path = tmp_path / "ramp.flac"
    sf.write(path, samples, SR)
    return str(path)


class TestLoadWindow:
    def test_window_matches_slice_of_full_load(self, ramp_wav: str) -> None:
        full = load_audio_data(ramp_wav).samples
        window = load_audio_data(ramp_wav, start=0.5, duration=1.0)
        assert window.duration == pytest.approx(1.0)
        assert window.metadata["start_time"] == 0.5
        np.testing.assert_array_equal(window.samples, full[SR // 2 : SR // 2 + SR])

    def test_start_only_reads_to_the_end(self, ramp_wav: str) -> None:
        assert load_audio_data(ramp_wav, start=2.5).duration == pytest.approx(0.5)

    def test_duration_only_reads_from_the_start(self, ramp_wav: str) -> None:
        window = load_audio_data(ramp_wav, duration=0.25)
        assert window.duration == pytest.approx(0.25)
        assert window.metadata["start_time"] == 0.0

    def test_window_past_the_end_is_short(self, ramp_wav: str) -> None:
        assert load_audio_data(ramp_wav, start=2.0, duration=5.0).duration == pytest.approx(1.0)
        assert load_audio_data(ramp_wav, start=10.0).num_samples == 0

    def test_window_with_resample(self, ramp_flac: str) -> None:
        window = load_audio_data(ramp_flac, start=1.0, duration=1.0, sample_rate=8000)
        assert window.sample_rate == 8000
        assert window.num_samples == 8000

    def test_full_load_has_no_start_time(self, ramp_wav: str) -> None:
        assert "start_time" not in load_audio_data(ramp_wav).metadata

    @pytest.mark.parametrize(("start", "duration"), [(-1.0, None), (0.0, 0.0), (0.0, -2.0)])
    def test_invalid_window_rejected(self, ramp_wav: str, start, duration) -> None:
        with pytest.raises(InvalidInputError):
            load_audio_data(ramp_wav, start=start, duration=duration)

    @pytest.mark.usefixtures("requires_ffmpeg_cli")
    @pytest.mark.parametrize("start", [0.05, 1.0, 2.2])
    def test_mp3_window_matches_full_decode(self, noise_mp3: str, start: float) -> None:
        # The pre-roll lets the decoder settle, so even a window starting just
        # after a seek point is sample-identical to the same span of a full decode.
        full = load_audio_data(noise_mp3).samples
        window = load_audio_data(noise_mp3, start=start, duration=0.5)
        first = round(start * SR)
        assert window.num_samples == SR // 2
        np.testing.assert_allclose(window.samples, full[first : first + SR // 2], atol=1e-4)

    def test_missing_ffmpeg_is_a_dependency_error(self, tmp_path, monkeypatch) -> None:
        import shutil

        from bioamla.exceptions import DependencyError

        m4a = tmp_path / "clip.m4a"
        m4a.write_bytes(b"not decoded")
        real_which = shutil.which
        monkeypatch.setattr(
            shutil, "which", lambda name: None if name == "ffmpeg" else real_which(name)
        )
        with pytest.raises(DependencyError, match="ffmpeg"):
            load_audio_data(m4a, start=0.0, duration=1.0)


class TestStream:
    def test_blocks_cover_the_file_exactly(self, ramp_wav: str) -> None:
        full = load_audio_data(ramp_wav).samples
        blocks = list(stream_audio_data(ramp_wav, block_duration=1.0))
        assert [b.metadata["start_time"] for b in blocks] == [0.0, 1.0, 2.0]
        np.testing.assert_array_equal(np.concatenate([b.samples for b in blocks]), full)

    def test_overlapping_blocks(self, ramp_flac: str) -> None:
        full = load_audio_data(ramp_flac).samples
        blocks = list(stream_audio_data(ramp_flac, block_duration=1.0, overlap=0.5))
        starts = [b.metadata["start_time"] for b in blocks]
        assert starts == [0.0, 0.5, 1.0, 1.5, 2.0]
        for block, start in zip(blocks, starts, strict=True):
            first = round(start * SR)
            np.testing.assert_array_equal(block.samples, full[first : first + SR])

    def test_last_block_may_be_short(self, ramp_wav: str) -> None:
        blocks = list(stream_audio_data(ramp_wav, block_duration=2.0))
        assert [b.duration for b in blocks] == pytest.approx([2.0, 1.0])

    def test_resampled_blocks(self, ramp_wav: str) -> None:
        blocks = list(stream_audio_data(ramp_wav, block_duration=1.5, sample_rate=8000))
        assert [b.num_samples for b in blocks] == [12000, 12000]
        assert all(b.sample_rate == 8000 for b in blocks)

    @pytest.mark.usefixtures("requires_ffmpeg_cli")
    def test_mp3_stream_reassembles_the_full_decode(self, noise_mp3: str) -> None:
        full = load_audio_data(noise_mp3).samples
        blocks = list(stream_audio_data(noise_mp3, block_duration=0.7))
        assert [b.metadata["start_time"] for b in blocks] == pytest.approx(
            [0.0, 0.7, 1.4, 2.1, 2.8]
        )
        joined = np.concatenate([b.samples for b in blocks])
        np.testing.assert_allclose(joined, full, atol=1e-4)

    def test_step_shorter_than_a_sample_rejected(self, ramp_wav: str) -> None:
        # Passes the seconds check, but block - overlap rounds to 0 samples.
        with pytest.raises(InvalidInputError, match="one sample"):
            next(stream_audio_data(ramp_wav, block_duration=1.0, overlap=0.99999))

    def test_bad_arguments_fail_on_the_call(self, tmp_path) -> None:
        # Not deferred to the first next().
        with pytest.raises(InvalidInputError):
            stream_audio_data(tmp_path / "x.wav", block_duration=0)
        with pytest.raises(NotFoundError):
            stream_audio_data(tmp_path / "nope.wav", block_duration=1.0)

    @pytest.mark.parametrize(("block", "overlap"), [(0.0, 0.0), (1.0, 1.0), (1.0, -0.1)])
    def test_invalid_blocks_rejected(self, ramp_wav: str, block, overlap) -> None:
        with pytest.raises(InvalidInputError):
            next(stream_audio_data(ramp_wav, block_duration=block, overlap=overlap))

    def test_missing_file(self, tmp_path) -> None:
        with pytest.raises(NotFoundError):
            next(stream_audio_data(tmp_path / "nope.wav", block_duration=1.0))


@pytest.mark.usefixtures("requires_ffmpeg_cli")
@pytest.mark.skipif(__import__("os").name == "nt", reason="a shell-script ffmpeg")
def test_a_stalled_ffmpeg_is_stopped_on_every_decode_path(noise_mp3, tmp_path, monkeypatch):
    """Whole-file loads, windows and streamed blocks all run ffmpeg under
    BIOAMLA_DECODE_TIMEOUT: a decoder that stalls is killed and reported, not waited on."""
    import os
    import shutil
    import time

    from bioamla.exceptions import AudioLoadError

    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    pids = tmp_path / "pids"
    (bin_dir / "ffmpeg").write_text(f"#!/bin/sh\necho $$ >> {pids}\nexec sleep 60\n")
    (bin_dir / "ffmpeg").chmod(0o755)
    (bin_dir / "ffprobe").symlink_to(shutil.which("ffprobe"))  # the header is read as usual
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ['PATH']}")
    monkeypatch.setenv("BIOAMLA_DECODE_TIMEOUT", "0.5")

    for decode in (
        lambda: load_audio_data(noise_mp3),
        lambda: load_audio_data(noise_mp3, start=1.0, duration=0.5),
        lambda: list(stream_audio_data(noise_mp3, block_duration=1.0)),
    ):
        began = time.monotonic()
        with pytest.raises(
            AudioLoadError, match=r"still decoding after 0\.5 s.*BIOAMLA_DECODE_TIMEOUT"
        ):
            decode()
        assert time.monotonic() - began < 10
        pid = int(pids.read_text().split()[-1])
        with pytest.raises(ProcessLookupError):  # the stalled decoder was killed, not left
            os.kill(pid, 0)


@pytest.mark.usefixtures("requires_ffmpeg_cli")
def test_a_bad_decode_timeout_is_a_config_error(noise_mp3, monkeypatch):
    from bioamla.exceptions import ConfigError

    for bad in ("soon", "0", "-1", "inf", "nan"):
        monkeypatch.setenv("BIOAMLA_DECODE_TIMEOUT", bad)
        with pytest.raises(ConfigError, match="BIOAMLA_DECODE_TIMEOUT must be a positive"):
            load_audio_data(noise_mp3)
    monkeypatch.setenv("BIOAMLA_DECODE_TIMEOUT", " ")  # blank: the default
    assert load_audio_data(noise_mp3).sample_rate == SR


@pytest.fixture
def three_channels(tmp_path) -> tuple[str, np.ndarray]:
    """3 s with a different ramp on each of three channels, so a sample's value says
    which channel and position it came from."""
    ramp = np.linspace(0.0, 0.3, 3 * SR, dtype=np.float32)
    samples = np.stack([ramp, -ramp, ramp + 0.5], axis=1)
    path = tmp_path / "three.wav"
    sf.write(path, samples, SR, subtype="FLOAT")
    return str(path), samples


class TestKeepChannels:
    def test_window_keeps_every_channel(self, three_channels) -> None:
        path, samples = three_channels
        window = load_audio_data(path, start=1.0, duration=0.5, keep_channels=True)
        assert window.channels == 3 and window.samples.shape == (SR // 2, 3)
        np.testing.assert_array_equal(window.samples, samples[SR : SR + SR // 2])
        mixed = load_audio_data(path, start=1.0, duration=0.5)  # the default stays mono
        assert mixed.channels == 1 and mixed.samples.ndim == 1
        np.testing.assert_allclose(mixed.samples, samples[SR : SR + SR // 2].mean(axis=1))

    def test_whole_file_and_resampled_windows_keep_channels(self, three_channels) -> None:
        path, samples = three_channels
        whole = load_audio_data(path, keep_channels=True)
        assert whole.channels == 3 and whole.samples.shape == samples.shape
        halved = load_audio_data(path, duration=1.0, sample_rate=8000, keep_channels=True)
        assert halved.channels == 3 and halved.samples.shape == (8000, 3)
        assert halved.samples[:, 1].max() <= 1e-3 < halved.samples[:, 2].min()  # not mixed

    def test_blocks_keep_channels(self, three_channels) -> None:
        path, samples = three_channels
        blocks = list(stream_audio_data(path, block_duration=1.0, keep_channels=True))
        assert [b.samples.shape for b in blocks] == [(SR, 3)] * 3
        np.testing.assert_array_equal(np.concatenate([b.samples for b in blocks]), samples)

    def test_a_mono_file_stays_one_dimensional(self, ramp_wav) -> None:
        window = load_audio_data(ramp_wav, start=0.5, duration=0.5, keep_channels=True)
        assert window.channels == 1 and window.samples.shape == (SR // 2,)

    @pytest.mark.usefixtures("requires_ffmpeg_cli")
    def test_an_ffmpeg_decoded_window_keeps_its_channels(self, tmp_path) -> None:
        """MP3 goes through ffmpeg rather than soundfile; its window keeps both channels
        and lines up with the whole decode."""
        import subprocess

        t = np.arange(3 * SR) / SR
        wav = tmp_path / "st.wav"
        sf.write(wav, np.stack([0.4 * np.sin(2 * np.pi * 440 * t), np.zeros_like(t)], 1), SR)
        mp3 = tmp_path / "st.mp3"
        subprocess.run(["ffmpeg", "-v", "error", "-i", str(wav), str(mp3)], check=True)
        whole = load_audio_data(mp3, keep_channels=True).samples
        window = load_audio_data(mp3, start=1.0, duration=0.5, keep_channels=True)
        assert window.channels == 2 and window.samples.shape == (SR // 2, 2)
        np.testing.assert_allclose(window.samples, whole[SR : SR + SR // 2], atol=1e-4)
        assert np.abs(window.samples[:, 0]).max() > 0.1 > np.abs(window.samples[:, 1]).max()
        blocks = list(stream_audio_data(mp3, block_duration=1.0, keep_channels=True))
        assert all(b.samples.ndim == 2 and b.samples.shape[1] == 2 for b in blocks)


@pytest.mark.usefixtures("requires_ffmpeg_cli")
def test_ffmpeg_decodes_are_writable_arrays(noise_mp3):
    """Like every other loader's, an ffmpeg-decoded array belongs to the caller."""
    whole = load_audio_data(noise_mp3).samples
    window = load_audio_data(noise_mp3, start=1.0, duration=0.5).samples
    block = next(iter(stream_audio_data(noise_mp3, block_duration=1.0))).samples
    for samples in (whole, window, block):
        assert samples.flags.writeable
        samples[:10] *= 0.5  # an in-place step works


def test_a_missing_ffprobe_is_a_dependency_error_from_get_audio_info(tmp_path, monkeypatch):
    import shutil

    from bioamla.audio import _codec, get_audio_info
    from bioamla.exceptions import DependencyError

    path = tmp_path / "x.mp3"
    path.write_bytes(b"fake")
    monkeypatch.setattr(_codec, "_get_metadata_ffprobe", lambda p: None)
    monkeypatch.setattr(shutil, "which", lambda name: None)
    with pytest.raises(DependencyError, match="ffprobe is required"):
        get_audio_info(str(path))


def test_ffmpeg_without_ffprobe_is_a_dependency_error(tmp_path, monkeypatch):
    import shutil

    from bioamla.audio import _codec
    from bioamla.exceptions import DependencyError

    path = tmp_path / "x.mp3"
    path.write_bytes(b"fake")
    real_which = shutil.which
    monkeypatch.setattr(shutil, "which", lambda n: None if n == "ffprobe" else real_which(n))
    monkeypatch.setattr(_codec, "_get_metadata_ffprobe", lambda p: None)
    with pytest.raises(DependencyError, match="ffprobe is required to read .mp3"):
        load_audio_data(path)


@pytest.mark.usefixtures("requires_ffmpeg_cli")
def test_overshoots_are_clipped_to_full_scale(tmp_path):
    """A lossy decode can overshoot full scale; loaders keep to [-1, 1]."""
    import subprocess

    t = np.arange(2 * SR) / SR
    wav = tmp_path / "loud.wav"
    sf.write(wav, np.sign(np.sin(2 * np.pi * 300 * t)).astype(np.float32), SR, subtype="FLOAT")
    mp3 = tmp_path / "loud.mp3"
    subprocess.run(["ffmpeg", "-v", "error", "-i", str(wav), str(mp3)], check=True)
    assert np.abs(load_audio_data(mp3).samples).max() <= 1.0


@pytest.mark.usefixtures("requires_ffmpeg_cli")
def test_load_audio_passes_config_errors_through(noise_mp3, monkeypatch):
    from bioamla.audio import load_audio
    from bioamla.exceptions import ConfigError

    monkeypatch.setenv("BIOAMLA_DECODE_TIMEOUT", "soon")
    with pytest.raises(ConfigError):
        load_audio(noise_mp3)
