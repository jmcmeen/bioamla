"""Tests for bioamla.viz.video: frame composition without ffmpeg, encoding with it."""

import shutil
import subprocess

import numpy as np
import pytest

from bioamla.exceptions import DependencyError, InvalidInputError, NotFoundError
from bioamla.viz.video import (
    SpectrogramVideoResult,
    _fmt_time,
    _FrameRenderer,
    generate_spectrogram_video,
)

SR = 16000


def _tone(seconds: float = 2.0, freq: float = 2000.0) -> np.ndarray:
    t = np.arange(int(SR * seconds)) / SR
    return (0.5 * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def _renderer_params(**overrides) -> dict:
    params = {
        "viz_type": "mel",
        "n_fft": 512,
        "hop_length": 128,
        "n_mels": 32,
        "window": "hann",
        "fmin": 0.0,
        "fmax": None,
        "db_range": 80.0,
        "cmap": "magma",
        "width": 320,
        "height": 180,
        "window_seconds": 1.0,
        "playhead": 0.5,
        "waveform": True,
        "show_axes": True,
        "backend": "librosa",
    }
    params.update(overrides)
    return params


def _renderer(**overrides) -> _FrameRenderer:
    return _FrameRenderer(_tone(), SR, **_renderer_params(**overrides))


class TestFrameRenderer:
    def test_frame_shape_and_dtype(self) -> None:
        r = _renderer()
        frame = r.frame(1.0)
        assert frame.shape == (180, 320, 3)
        assert frame.dtype == np.uint8
        assert frame.flags["C_CONTIGUOUS"]

    def test_odd_sizes_rounded_to_even(self) -> None:
        r = _renderer(width=321, height=181)
        assert (r.width, r.height) == (320, 180)
        assert r.frame(0.5).shape == (180, 320, 3)

    def test_playhead_column_is_white(self) -> None:
        r = _renderer(show_axes=False)
        frame = r.frame(1.0)
        x = r.playhead_x
        assert np.all(frame[:, x : x + 2] == 255)

    def test_before_start_is_background(self) -> None:
        """At t=0 with a centred playhead, the left half precedes the recording."""
        r = _renderer(show_axes=False, waveform=False)
        frame = r.frame(0.0)
        left = frame[:, : r.playhead_x - 1]
        assert np.all(left == r.background)
        right = frame[:, r.playhead_x + 2 :]
        assert np.any(right != r.background)

    def test_tone_lands_on_expected_row(self) -> None:
        """A 2 kHz tone should be brightest around the 2 kHz row, not elsewhere."""
        r = _renderer(viz_type="stft", show_axes=False, waveform=False)
        frame = r.frame(1.0).astype(int)
        brightness = frame.sum(axis=2).mean(axis=1)  # per display row
        peak_row = int(np.argmax(brightness))
        # Display rows run top (Nyquist) to bottom (0 Hz): 2 kHz of 8 kHz is 1/4 up.
        expected = int((1 - 2000 / 8000) * (r.spec_h - 1))
        assert abs(peak_row - expected) <= r.spec_h // 16

    def test_waveform_strip_present_and_absent(self) -> None:
        with_wave = _renderer(show_axes=False)
        without = _renderer(show_axes=False, waveform=False)
        assert with_wave.wave_h > 0 and without.wave_h == 0
        assert with_wave.spec_h + with_wave.wave_h == with_wave.height
        strip = with_wave.frame(1.0)[with_wave.spec_h :]
        assert np.any(np.all(strip == with_wave.wave_color, axis=2))

    def test_axes_labels_change_pixels(self) -> None:
        plain = _renderer(show_axes=False).frame(1.0)
        labelled = _renderer(show_axes=True).frame(1.0)
        assert not np.array_equal(plain, labelled)

    def test_frequency_ticks_within_spectrogram(self) -> None:
        r = _renderer()
        assert r._ticks, "expected some frequency ticks"
        for row, label in r._ticks:
            assert 0 <= row < r.spec_h
            assert label.endswith("Hz")

    def test_frequency_ticks_do_not_overlap(self) -> None:
        """On a linear axis the low ticks crowd together; they must be thinned."""
        r = _renderer(viz_type="stft", height=360)
        rows = [row for row, _ in r._ticks]
        assert rows == sorted(rows)
        assert all(b - a >= 14 for a, b in zip(rows, rows[1:], strict=False))

    def test_time_readout_uses_source_offset(self) -> None:
        offset = _FrameRenderer(_tone(), SR, time_offset=90.0, **_renderer_params())
        plain = _renderer()
        assert not np.array_equal(offset.frame(0.0), plain.frame(0.0))

    def test_unknown_colormap(self) -> None:
        with pytest.raises(InvalidInputError, match="colormap"):
            _renderer(cmap="no-such-map")

    def test_playhead_at_right_edge_shows_only_past(self) -> None:
        r = _renderer(playhead=1.0, show_axes=False, waveform=False)
        frame = r.frame(0.0)
        assert np.all(frame[:, : r.playhead_x] == r.background)


class TestPageMotion:
    def test_playhead_moves_and_picture_holds(self) -> None:
        r = _renderer(motion="page", show_axes=False, window_seconds=1.0)
        a, b = r.frame(0.25), r.frame(0.5)
        # Same page: everything but the two playhead columns is identical.
        xa, xb = int(0.25 * (r.width - 2)), int(0.5 * (r.width - 2))
        assert np.all(a[:, xa : xa + 2] == 255) and np.all(b[:, xb : xb + 2] == 255)
        keep = np.ones(r.width, dtype=bool)
        keep[[xa, xa + 1, xb, xb + 1]] = False
        assert np.array_equal(a[:, keep], b[:, keep])
        assert r._page is not None and r._page[0] == 0.0

    def test_turns_the_page(self) -> None:
        r = _renderer(motion="page", show_axes=False, window_seconds=1.0)
        r.frame(0.5)
        r.frame(1.5)
        assert r._page is not None and r._page[0] == 1.0

    def test_whole_recording_is_one_page(self) -> None:
        r = _renderer(motion="page", show_axes=False, window_seconds=None)
        assert r.window_seconds == pytest.approx(2.0)
        first, last = r.frame(0.0), r.frame(1.99)
        # The picture spans the whole recording, so no column is background-only
        # padding at either end; only the playhead differs between the frames.
        assert not np.all(first[:, -3:] == r.background)
        assert r._page is not None and r._page[0] == 0.0
        assert last.shape == first.shape


class TestWaveformOnly:
    def test_fills_the_frame(self) -> None:
        r = _renderer(viz_type="waveform", show_axes=False)
        assert r.spec_h == 0 and r.wave_h == r.height
        frame = r.frame(1.0)
        assert frame.shape == (180, 320, 3)
        assert np.any(np.all(frame == r.wave_color, axis=2))
        assert not hasattr(r, "levels")  # no spectrogram was computed

    def test_axes_show_only_the_time_readout(self) -> None:
        r = _renderer(viz_type="waveform", show_axes=True)
        assert r._ticks == []
        assert not np.array_equal(
            r.frame(1.0), _renderer(viz_type="waveform", show_axes=False).frame(1.0)
        )


class TestValidation:
    def test_missing_file(self, tmp_path) -> None:
        with pytest.raises(NotFoundError):
            generate_spectrogram_video(tmp_path / "nope.wav", tmp_path / "out.mp4")

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"viz_type": "mfcc"},
            {"motion": "slide"},
            {"window_seconds": None},
            {"fps": 0},
            {"width": 10},
            {"window_seconds": 0},
            {"playhead": 1.5},
            {"db_range": -1},
        ],
    )
    def test_bad_arguments(self, test_audio_path: str, tmp_path, kwargs) -> None:
        with pytest.raises(InvalidInputError):
            generate_spectrogram_video(test_audio_path, tmp_path / "out.mp4", **kwargs)

    def test_bad_start_end(self, test_audio_path: str, tmp_path) -> None:
        with pytest.raises(InvalidInputError, match="negative"):
            generate_spectrogram_video(test_audio_path, tmp_path / "o.mp4", start=-1)
        with pytest.raises(InvalidInputError, match="after start"):
            generate_spectrogram_video(test_audio_path, tmp_path / "o.mp4", start=0.5, end=0.5)

    def test_start_beyond_file(self, test_audio_path: str, tmp_path) -> None:
        """The 1 s fixture has no audio at 5 s; needs ffmpeg present to get that far."""
        if not shutil.which("ffmpeg"):
            pytest.skip("needs ffmpeg on PATH")
        with pytest.raises(InvalidInputError, match="beyond the end"):
            generate_spectrogram_video(test_audio_path, tmp_path / "o.mp4", start=5)

    def test_missing_ffmpeg(self, test_audio_path: str, tmp_path, mocker) -> None:
        mocker.patch("bioamla.viz.video.shutil.which", return_value=None)
        with pytest.raises(DependencyError, match="ffmpeg"):
            generate_spectrogram_video(test_audio_path, tmp_path / "out.mp4")


@pytest.mark.usefixtures("requires_ffmpeg_cli")
class TestEncoding:
    def test_writes_playable_video_with_audio(self, test_audio_path: str, tmp_path) -> None:
        out = tmp_path / "out" / "tone.mp4"
        progress: list[tuple[int, int]] = []
        result = generate_spectrogram_video(
            test_audio_path,
            out,
            width=160,
            height=90,
            fps=10,
            window_seconds=0.5,
            on_progress=lambda done, total: progress.append((done, total)),
        )
        assert isinstance(result, SpectrogramVideoResult)
        assert out.is_file() and out.stat().st_size > 0
        assert result.frames == 10 and result.fps == 10
        assert result.duration == pytest.approx(1.0, abs=0.01)
        assert (result.width, result.height) == (160, 90)
        assert result.has_audio
        assert progress[-1] == (10, 10) and len(progress) == 10
        assert result.to_dict()["frames"] == 10

        if shutil.which("ffprobe"):
            probe = subprocess.run(
                [
                    "ffprobe",
                    "-v",
                    "error",
                    "-show_entries",
                    "stream=codec_type",
                    "-of",
                    "csv=p=0",
                    str(out),
                ],
                capture_output=True,
                text=True,
                check=True,
            )
            assert sorted(probe.stdout.split()) == ["audio", "video"]

    def test_start_end_clip(self, test_audio_path_3s: str, tmp_path) -> None:
        """1.0-2.5 s of a 3 s file: 1.5 s of video, end past the file is clipped."""
        out = tmp_path / "clip.mp4"
        result = generate_spectrogram_video(
            test_audio_path_3s, out, start=1.0, end=2.5, width=160, height=90, fps=10
        )
        assert result.duration == pytest.approx(1.5, abs=0.01)
        assert (result.start, result.end) == (1.0, 2.5)
        assert result.frames == 15
        clipped = generate_spectrogram_video(
            test_audio_path_3s,
            tmp_path / "tail.mp4",
            start=2.0,
            end=99,
            width=160,
            height=90,
            fps=10,
        )
        assert clipped.end == pytest.approx(3.0, abs=0.01)
        assert clipped.frames == 10

    def test_without_audio(self, test_audio_path: str, tmp_path) -> None:
        out = tmp_path / "silent.mp4"
        result = generate_spectrogram_video(
            test_audio_path, out, width=160, height=90, fps=5, include_audio=False
        )
        assert out.is_file() and not result.has_audio


def test_fmt_time() -> None:
    assert _fmt_time(0) == "00:00.0"
    assert _fmt_time(65.25) == "01:05.2"
    assert _fmt_time(-3) == "00:00.0"
