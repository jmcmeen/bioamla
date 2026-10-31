"""Frame-level descriptors, log-mel spectrogram and MFCCs."""

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from bioamla.audio import DESCRIPTORS, FrameFeatureConfig, frame_features, mfcc_trajectory
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


class TestTrajectory:
    def test_matches_reference(self, features, reference) -> None:
        traj = mfcc_trajectory(features, gate_db=reference["gate_db"], scale="equalize")
        np.testing.assert_array_equal(traj.active, np.array(reference["active"], dtype=bool))
        ref = np.array(reference["trajectory"])[traj.active]
        ours = traj.coords[traj.active]
        for k in range(3):  # each axis may come out with either sign
            sign = np.sign(ref[:, k] @ ours[:, k])
            np.testing.assert_allclose(sign * ours[:, k], ref[:, k], atol=1e-4)

    def test_shapes_and_gating(self, features) -> None:
        traj = mfcc_trajectory(features)
        assert traj.coords.shape == (features.n_frames, 3)
        assert traj.components.shape == (3, 12)
        assert np.isnan(traj.coords[~traj.active]).all()
        assert np.isfinite(traj.coords[traj.active]).all()
        assert 0 < traj.active.sum() < features.n_frames  # the gaps are gated out

    def test_axes_are_ordered_and_orthonormal(self, features) -> None:
        traj = mfcc_trajectory(features, n_components=5)
        ratio = traj.explained_variance_ratio
        assert (np.diff(ratio) <= 0).all()
        assert 0 < ratio.sum() <= 1
        np.testing.assert_allclose(traj.components @ traj.components.T, np.eye(5), atol=1e-10)

    def test_raw_coords_have_axis_variance(self, features) -> None:
        traj = mfcc_trajectory(features)
        var = traj.coords[traj.active].var(axis=0)
        assert (np.diff(var) <= 0).all()

    def test_deterministic_signs(self, features) -> None:
        traj = mfcc_trajectory(features)
        for row in traj.components:
            assert row[np.abs(row).argmax()] > 0
        # Flipping the input's MFCCs flips the data, not the convention.
        flipped = mfcc_trajectory(replace(features, mfcc=-features.mfcc))
        for row in flipped.components:
            assert row[np.abs(row).argmax()] > 0

    def test_times_are_a_copy(self, features) -> None:
        traj = mfcc_trajectory(features)
        traj.times += 1
        assert features.times[0] < 1

    def test_rank_deficient_input_has_no_negative_variance(self) -> None:
        f = frame_features(tone(3000, seconds=2), 22050, FrameFeatureConfig(denoise=False))
        traj = mfcc_trajectory(f, n_components=5)
        assert (traj.explained_variance_ratio >= 0).all()

    def test_similar_sounds_land_together(self) -> None:
        sr = 22050
        a, b = tone(2000, seconds=0.5), tone(6000, seconds=0.5)
        f = frame_features(np.concatenate([a, b, a, b]), sr, FrameFeatureConfig(denoise=False))
        traj = mfcc_trajectory(f, n_components=1)
        pc1 = traj.coords[:, 0]
        first = pc1[(f.times > 0.1) & (f.times < 0.4)].mean()
        third = pc1[(f.times > 1.1) & (f.times < 1.4)].mean()
        second = pc1[(f.times > 0.6) & (f.times < 0.9)].mean()
        assert abs(first - third) < abs(first - second) / 10

    def test_to_dataframe_and_dict(self, features) -> None:
        traj = mfcc_trajectory(features)
        df = traj.to_dataframe()
        assert list(df.columns) == ["time", "active", "pc1", "pc2", "pc3"]
        d = traj.to_dict()
        json.dumps(d)  # NaN becomes None
        assert (~traj.active).any()
        gated = int(np.argmin(traj.active))
        assert d["coords"][gated] == [None, None, None]

    @pytest.mark.parametrize(
        ("kwargs", "match"),
        [
            ({"n_components": 0}, "n_components"),
            ({"n_components": 13}, "n_components"),
            ({"scale": "loud"}, "scale"),
            ({"gate_db": 0}, "gate_db"),
            ({"gate_db": float("nan")}, "gate_db"),
            ({"n_components": 2.5}, "integer"),
            ({"n_components": True}, "integer"),
        ],
    )
    def test_bad_arguments(self, features, kwargs: dict, match: str) -> None:
        with pytest.raises(InvalidInputError, match=match):
            mfcc_trajectory(features, **kwargs)

    def test_too_few_loud_frames(self) -> None:
        x = np.zeros(22050)
        x[5000:5100] = 0.5  # one click
        f = frame_features(x, 22050, FrameFeatureConfig(denoise=False))
        with pytest.raises(InvalidInputError, match="frames are within"):
            mfcc_trajectory(f, gate_db=3)


class TestFiles:
    @pytest.fixture
    def wav(self, tmp_path) -> Path:
        import soundfile as sf

        x, sr = reference_signal()
        path = tmp_path / "ref.wav"
        sf.write(path, x, sr)
        return path

    def test_frame_features_file(self, wav, tmp_path) -> None:
        from bioamla.audio import frame_features_file

        out = tmp_path / "deep" / "t.csv"
        result = frame_features_file(wav, out, trajectory=True)
        assert result.output_path == str(out) and out.exists()
        assert list(result.table.columns[-3:]) == ["pc1", "pc2", "pc3"]
        summary = result.to_dict()
        json.dumps(summary)
        assert summary["n_frames"] == result.features.n_frames
        assert len(summary["explained_variance_ratio"]) == 3

    def test_no_output_writes_nothing(self, wav, tmp_path) -> None:
        from bioamla.audio import frame_features_file

        result = frame_features_file(wav)
        assert result.output_path is None and result.trajectory is None
        assert "pc1" not in result.table.columns

    def test_bad_extension_fails_before_analysis(self, wav, tmp_path, mocker) -> None:
        from bioamla.audio import frame_features_file

        load = mocker.patch("bioamla.audio.io.load_audio")
        with pytest.raises(InvalidInputError, match=".csv or .parquet"):
            frame_features_file(wav, tmp_path / "t.xlsx")
        load.assert_not_called()

    def test_missing_file(self, tmp_path) -> None:
        from bioamla.audio import frame_features_file
        from bioamla.exceptions import NotFoundError

        with pytest.raises(NotFoundError):
            frame_features_file(tmp_path / "nope.wav")

    def test_batch(self, wav, tmp_path) -> None:
        from bioamla.audio import batch_frame_features_files

        out = tmp_path / "out"
        result = batch_frame_features_files(wav.parent, out, file_format="parquet")
        assert result.successful == 1
        assert result.output_files == [
            str(out / "ref.features.parquet"),
            str(out / "frame_features.csv"),
        ]
        assert result.metadata["rows"][0]["n_frames"] > 0

    def test_batch_bad_format(self, wav, tmp_path) -> None:
        from bioamla.audio import batch_frame_features_files

        with pytest.raises(InvalidInputError, match="file_format"):
            batch_frame_features_files(wav.parent, tmp_path, file_format="xlsx")
