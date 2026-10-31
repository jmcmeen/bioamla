"""Tests for windowed loading (load_audio_data start/duration) and stream_audio_data."""

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
    from pydub import AudioSegment

    wav = tmp_path / "noise.wav"
    rng = np.random.default_rng(0)
    sf.write(wav, rng.normal(0, 0.1, 3 * SR).astype(np.float32), SR)
    mp3 = tmp_path / "noise.mp3"
    AudioSegment.from_wav(wav).export(mp3, format="mp3")
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
