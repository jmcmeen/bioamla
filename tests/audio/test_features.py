"""Frame-level descriptors, log-mel spectrogram and MFCCs."""

import json
from pathlib import Path

import numpy as np
import pytest

from bioamla.audio import DESCRIPTORS, FrameFeatureConfig, frame_features
from bioamla.exceptions import ConfigValidationError, InvalidInputError

FIXTURE = Path(__file__).parent / "fixtures" / "frame_features_reference.json"


def reference_signal(sr: int = 22050, seconds: float = 2.0) -> tuple[np.ndarray, int]:
    """Rising chirps, tone bursts with a harmonic, gaps and a low noise floor.

    The noise is a linear congruential sequence, so the signal is the same on every
    platform and numpy version.
    """
    n = int(sr * seconds)
    t = np.arange(n) / sr
    state = (np.arange(n, dtype=np.int64) * 1103515245 + 12345) % (2**31)
    x = 0.01 * (state / 2**31 - 0.5)
    for start in (0.1, 0.6, 1.1):  # 2 -> 5 kHz chirps, 0.3 s
        m = (t >= start) & (t < start + 0.3)
        tt = t[m] - start
        x[m] += 0.5 * np.sin(2 * np.pi * (2000 * tt + 0.5 * 10000 * tt**2)) * np.hanning(m.sum())
    for start in (1.5, 1.7):  # 3 kHz bursts with a 6 kHz harmonic, 0.1 s
        m = (t >= start) & (t < start + 0.1)
        tt = t[m]
        burst = 0.4 * np.sin(2 * np.pi * 3000 * tt) + 0.1 * np.sin(2 * np.pi * 6000 * tt)
        x[m] += burst * np.hanning(m.sum())
    return x.astype(np.float32), sr


@pytest.fixture(scope="module")
def reference() -> dict:
    return json.loads(FIXTURE.read_text())


@pytest.fixture(scope="module")
def features():
    return frame_features(*reference_signal())


def tone(freq: float, sr: int = 22050, seconds: float = 1.0, amp: float = 0.5) -> np.ndarray:
    t = np.arange(int(sr * seconds)) / sr
    return amp * np.sin(2 * np.pi * freq * t)


class TestReference:
    """Agreement with an independent implementation of the same analysis."""

    def test_frame_layout(self, features, reference) -> None:
        assert features.n_frames == reference["frames"]
        assert features.hop == reference["hop"]
        assert features.n_fft == 1024

    @pytest.mark.parametrize("name", DESCRIPTORS)
    def test_descriptor_matches(self, features, reference, name: str) -> None:
        # flux at frame 0 is defined as 0 here; the reference differences against silence
        skip = 1 if name == "flux" else 0
        np.testing.assert_allclose(
            features.descriptors[name][skip:],
            reference["descriptors"][name][skip:],
            rtol=1e-4,
            atol=1e-3,
        )

    def test_mfcc_matches(self, features, reference) -> None:
        np.testing.assert_allclose(features.mfcc, reference["mfcc"], rtol=1e-4, atol=5e-3)


class TestDescriptors:
    def test_pure_tone(self) -> None:
        f = frame_features(tone(3000), 22050, FrameFeatureConfig(denoise=False))
        mid = slice(2, -2)
        np.testing.assert_allclose(f.descriptors["peak_hz"][mid], 3000, atol=22050 / 1024)
        np.testing.assert_allclose(f.descriptors["centroid_hz"][mid], 3000, atol=50)
        assert (f.descriptors["crest"][mid] > 20).all()
        assert (f.descriptors["flatness"][mid] < 0.1).all()
        assert (f.descriptors["low_hz"][mid] <= 3000).all()
        assert (f.descriptors["high_hz"][mid] >= 3000).all()

    def test_noise_is_flat(self) -> None:
        rng = np.random.default_rng(0)
        f = frame_features(rng.normal(size=22050), 22050, FrameFeatureConfig(denoise=False))
        assert np.median(f.descriptors["flatness"]) > 0.5
        assert np.median(f.descriptors["crest"]) < 10

    def test_level_is_relative_to_loudest_frame(self) -> None:
        x = np.concatenate([tone(2000, amp=0.5, seconds=0.5), tone(2000, amp=0.05, seconds=0.5)])
        f = frame_features(x, 22050, FrameFeatureConfig(denoise=False))
        level = f.descriptors["level_db"]
        assert level.max() == pytest.approx(0.0, abs=1e-6)
        assert level[-3] == pytest.approx(-20.0, abs=0.5)

    def test_flux_peaks_at_onset(self) -> None:
        x = np.concatenate([np.zeros(11025), tone(2000, seconds=0.5)])
        f = frame_features(x, 22050, FrameFeatureConfig(denoise=False))
        onset = np.argmax(f.descriptors["flux"])
        assert abs(f.times[onset] - 0.5) < 0.05

    def test_denoise_removes_stationary_hum(self) -> None:
        sr = 22050
        hum = tone(1000, amp=0.1, seconds=2.0)
        call = np.zeros_like(hum)
        call[sr // 2 : sr] = tone(5000, amp=0.5, seconds=0.5)
        f = frame_features(hum + call, sr)
        quiet = f.times < 0.4
        assert f.descriptors["level_db"][quiet].max() < -40  # the hum is gone
        assert np.median(f.descriptors["peak_hz"][~quiet & (f.times < 0.95)]) == pytest.approx(
            5000, abs=30
        )

    def test_silence_gives_finite_values(self) -> None:
        f = frame_features(np.zeros(22050), 22050)
        for name in DESCRIPTORS:
            assert np.isfinite(f.descriptors[name]).all(), name
        assert np.isfinite(f.mfcc).all()


class TestShapesAndOptions:
    def test_shapes_and_times(self, features) -> None:
        n = features.n_frames
        assert features.mel_db.shape == (n, 40)
        assert features.mfcc.shape == (n, 13)
        assert features.times[0] == pytest.approx(512 / 22050)
        np.testing.assert_allclose(np.diff(features.times), 256 / 22050)

    def test_derived_defaults(self) -> None:
        f = frame_features(tone(5000, sr=96000), 96000)
        assert (f.n_fft, f.hop, f.f_max) == (2048, 512, 16000.0)
        g = frame_features(tone(3000, sr=16000), 16000)
        assert g.f_max == pytest.approx(16000 / 2 * 0.92)

    def test_custom_settings(self) -> None:
        cfg = FrameFeatureConfig(n_fft=512, hop=256, n_mels=20, n_mfcc=8, f_min=500, f_max=8000)
        f = frame_features(tone(3000), 22050, cfg)
        assert f.mel_db.shape[1] == 20
        assert f.mfcc.shape[1] == 8
        assert f.config.f_max == 8000

    def test_stereo_is_downmixed(self) -> None:
        mono = tone(3000)
        stereo = np.stack([mono, mono], axis=1)
        a = frame_features(mono, 22050)
        b = frame_features(stereo, 22050)
        np.testing.assert_allclose(a.mfcc, b.mfcc)

    def test_channels_first_is_downmixed(self) -> None:
        mono = tone(3000)
        a = frame_features(mono, 22050)
        b = frame_features(np.stack([mono, mono]), 22050)
        np.testing.assert_allclose(a.mfcc, b.mfcc)

    def test_first_frame_has_no_flux(self, features) -> None:
        assert features.descriptors["flux"][0] == 0

    def test_chunking_does_not_change_results(self, monkeypatch) -> None:
        import bioamla.audio.features as mod

        x, sr = reference_signal()
        whole = frame_features(x, sr)
        monkeypatch.setattr(mod, "_CHUNK_FRAMES", 7)
        chunked = frame_features(x, sr)
        for name in DESCRIPTORS:
            np.testing.assert_allclose(chunked.descriptors[name], whole.descriptors[name])
        np.testing.assert_allclose(chunked.mfcc, whole.mfcc)

    def test_to_dataframe(self, features) -> None:
        df = features.to_dataframe()
        assert list(df.columns) == ["time", *DESCRIPTORS, *(f"mfcc_{k}" for k in range(13))]
        assert len(df) == features.n_frames

    def test_to_dict_is_json_ready(self, features) -> None:
        d = features.to_dict()
        json.dumps(d)
        assert d["n_frames"] == features.n_frames
        assert set(d["descriptors"]) == set(DESCRIPTORS)
        assert d["config"]["n_fft"] == 1024
        assert d["config"]["denoise"] is True


class TestErrors:
    def test_too_short(self) -> None:
        with pytest.raises(InvalidInputError, match="too short"):
            frame_features(np.zeros(2000), 22050)

    def test_bad_shape(self) -> None:
        with pytest.raises(InvalidInputError, match="1-D"):
            frame_features(np.zeros((2, 2, 2000)), 22050)

    @pytest.mark.parametrize(
        ("kwargs", "match"),
        [
            ({"n_mfcc": 50}, "n_mfcc"),
            ({"f_min": 9000, "f_max": 8000}, "f_max must be above f_min"),
            ({"f_max": 20000}, "Nyquist"),
            ({"noise_percentile": 1.5}, "noise_percentile"),
        ],
    )
    def test_bad_settings(self, kwargs: dict, match: str) -> None:
        with pytest.raises(InvalidInputError, match=match):
            frame_features(tone(3000), 22050, FrameFeatureConfig(**kwargs))

    @pytest.mark.parametrize(
        "kwargs", [{"hop": 0}, {"n_fft": 0}, {"n_mels": 1}, {"noise_scale": -1}]
    )
    def test_config_rejects_out_of_range(self, kwargs: dict) -> None:
        with pytest.raises(InvalidInputError):
            FrameFeatureConfig(**kwargs)

    @pytest.mark.parametrize("sr", [0, -1, float("nan")])
    def test_bad_sample_rate(self, sr: float) -> None:
        with pytest.raises(InvalidInputError, match="sample_rate"):
            frame_features(tone(3000), sr)

    def test_config_validates_types(self) -> None:
        with pytest.raises(ConfigValidationError, match="n_mels"):
            FrameFeatureConfig(n_mels="many")
