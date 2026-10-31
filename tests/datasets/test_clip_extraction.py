"""Tests for bioamla.datasets.extract_audio_clips."""

from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from bioamla.datasets import extract_audio_clips
from bioamla.datasets.annotations import Annotation
from bioamla.exceptions import AnnotationError, NotFoundError

SR = 16000


@pytest.fixture
def two_tone_wav(make_tone_wav) -> str:
    """3 s of 500 Hz + 3 kHz tones, so band-pass filtering is measurable."""
    return make_tone_wav("rec.wav", [(500.0, 0.3), (3000.0, 0.3)], seconds=3.0)


def _band_power(clip: np.ndarray, sr: int, freq: float) -> float:
    spectrum = np.abs(np.fft.rfft(clip))
    freqs = np.fft.rfftfreq(len(clip), 1 / sr)
    return float(spectrum[np.argmin(np.abs(freqs - freq))])


class TestExtractAudioClips:
    def test_clips_records_and_names(self, two_tone_wav, tmp_path):
        anns = [
            Annotation(start_time=0.5, end_time=1.0, label="spring peeper", confidence=0.9),
            Annotation(start_time=1.0, end_time=2.0, label="bullfrog"),
        ]
        out = tmp_path / "clips"
        result = extract_audio_clips(anns, two_tone_wav, str(out))

        assert result["total_clips"] == 2
        assert [Path(p).name for p in result["extracted_clips"]] == [
            "rec_0000_spring_peeper.wav",
            "rec_0001_bullfrog.wav",
        ]
        assert result["failed_clips"] == [] and result["skipped_clips"] == []
        first = result["clips"][0]
        assert first["file_name"] == "rec_0000_spring_peeper.wav"
        assert (first["label"], first["source_file"], first["sample_rate"]) == (
            "spring peeper",
            "rec.wav",
            SR,
        )
        assert first["confidence"] == 0.9 and first["low_freq"] == ""
        assert sf.info(result["extracted_clips"][1]).duration == pytest.approx(1.0)

    def test_padding_is_clamped_at_the_edges(self, two_tone_wav, tmp_path):
        anns = [Annotation(start_time=0.05, end_time=2.9, label="x")]
        result = extract_audio_clips(anns, two_tone_wav, str(tmp_path), padding_ms=200)
        # 200 ms each side, cut off at 0 s and 3 s.
        assert sf.info(result["extracted_clips"][0]).duration == pytest.approx(3.0)

    def test_annotation_outside_the_audio_is_skipped(self, two_tone_wav, tmp_path):
        anns = [Annotation(start_time=5.0, end_time=6.0, label="late")]
        result = extract_audio_clips(anns, two_tone_wav, str(tmp_path))
        assert result["extracted_clips"] == []
        assert "outside the audio" in result["skipped_clips"][0]

    def test_audiofolder_layout(self, two_tone_wav, tmp_path):
        anns = [
            Annotation(start_time=0.0, end_time=0.5, label="Spring Peeper"),
            Annotation(start_time=0.5, end_time=1.0, label=""),
        ]
        result = extract_audio_clips(anns, two_tone_wav, str(tmp_path), subdir_by_label=True)
        assert [c["file_name"] for c in result["clips"]] == [
            "spring_peeper/rec_0000.wav",
            "unknown/rec_0001.wav",
        ]

    def test_label_left_out_of_filename(self, two_tone_wav, tmp_path):
        anns = [Annotation(start_time=0.0, end_time=0.5, label="frog")]
        result = extract_audio_clips(
            anns, two_tone_wav, str(tmp_path), include_label_in_filename=False
        )
        assert result["clips"][0]["file_name"] == "rec_0000.wav"

    def test_label_cleaned_for_filenames(self, two_tone_wav, tmp_path):
        anns = [
            Annotation(start_time=0.0, end_time=0.5, label='Rana sp.? "A/B": x*y'),
            Annotation(start_time=0.5, end_time=1.0, label="frog\tcall\n"),
            Annotation(start_time=1.0, end_time=1.5, label="???"),
        ]
        result = extract_audio_clips(anns, two_tone_wav, str(tmp_path))
        assert [c["file_name"] for c in result["clips"]] == [
            "rec_0000_Rana_sp.___A_B___x_y.wav",  # case kept, invalid characters replaced
            "rec_0001_frog_call.wav",  # control characters too
            "rec_0002.wav",  # nothing usable left: no label in the name
        ]

    def test_bandpass_keeps_the_annotated_band(self, two_tone_wav, tmp_path):
        anns = [Annotation(start_time=0.5, end_time=2.5, low_freq=2000, high_freq=4000)]
        plain = extract_audio_clips(anns, two_tone_wav, str(tmp_path / "plain"))
        banded = extract_audio_clips(anns, two_tone_wav, str(tmp_path / "band"), bandpass=True)
        raw, _ = sf.read(plain["extracted_clips"][0])
        filt, _ = sf.read(banded["extracted_clips"][0])
        assert banded["clips"][0]["low_freq"] == 2000
        assert _band_power(filt, SR, 500) < 0.05 * _band_power(raw, SR, 500)
        assert _band_power(filt, SR, 3000) > 0.5 * _band_power(raw, SR, 3000)

    def test_too_short_for_bandpass_is_kept_unfiltered(self, two_tone_wav, tmp_path):
        anns = [Annotation(start_time=1.0, end_time=1.001, low_freq=2000, high_freq=4000)]
        result = extract_audio_clips(anns, two_tone_wav, str(tmp_path), bandpass=True)
        assert len(result["extracted_clips"]) == 1
        assert result["failed_clips"] == []

    def test_resampled_clips(self, two_tone_wav, tmp_path):
        anns = [Annotation(start_time=0.0, end_time=1.0, label="x")]
        result = extract_audio_clips(anns, two_tone_wav, str(tmp_path), target_sample_rate=8000)
        info = sf.info(result["extracted_clips"][0])
        assert (info.samplerate, info.frames) == (8000, 8000)
        assert result["clips"][0]["sample_rate"] == 8000

    def test_a_failing_clip_is_recorded_and_the_rest_continue(
        self, two_tone_wav, tmp_path, monkeypatch
    ):
        import bioamla.audio as audio

        real_save = audio.save_audio

        def flaky(path, clip, sr, *args, **kwargs):
            if "0000" in Path(path).name:
                raise OSError("disk full")
            return real_save(path, clip, sr, *args, **kwargs)

        monkeypatch.setattr(audio, "save_audio", flaky)
        anns = [Annotation(start_time=0.0, end_time=0.5), Annotation(start_time=0.5, end_time=1.0)]
        result = extract_audio_clips(anns, two_tone_wav, str(tmp_path / "out"))
        assert len(result["extracted_clips"]) == 1
        assert "disk full" in result["failed_clips"][0]

    def test_missing_audio(self, tmp_path):
        with pytest.raises(NotFoundError):
            extract_audio_clips([], str(tmp_path / "nope.wav"), str(tmp_path))

    def test_unreadable_audio(self, tmp_path):
        bad = tmp_path / "bad.wav"
        bad.write_bytes(b"not audio")
        with pytest.raises(AnnotationError, match="Failed to load audio"):
            extract_audio_clips([], str(bad), str(tmp_path / "out"))
