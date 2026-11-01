"""The file-level audio functions: load one file, process it, save the result."""

import numpy as np
import pytest
import soundfile as sf

from bioamla.audio import (
    ProcessedFile,
    add_noise_audio_file,
    convert_audio_file,
    describe_audio_file,
    filter_audio_file,
    gain_audio_file,
    load_audio,
    normalize_audio_file,
    resample_audio_file,
    segment_audio_file,
    trim_audio_file,
)
from bioamla.exceptions import InvalidInputError


def _read(path):
    data, sr = sf.read(str(path), dtype="float32")
    return data, sr


def test_trim_by_end_and_by_duration(test_audio_path_3s, tmp_path):
    a = trim_audio_file(test_audio_path_3s, tmp_path / "a.wav", start=0.5, end=1.5)
    b = trim_audio_file(test_audio_path_3s, tmp_path / "b.wav", start=0.5, duration=1.0)
    assert isinstance(a, ProcessedFile) and a.output == str(tmp_path / "a.wav")
    (da, sr), (db, _) = _read(a.output), _read(b.output)
    assert len(da) == len(db) == sr


def test_trim_rejects_end_and_duration(test_audio_path_3s, tmp_path):
    with pytest.raises(InvalidInputError, match="both"):
        trim_audio_file(test_audio_path_3s, tmp_path / "x.wav", end=1.0, duration=1.0)


def test_normalize_peak_caps_below_full_scale(test_audio_path_3s, tmp_path):
    out = normalize_audio_file(test_audio_path_3s, tmp_path / "n.wav", target_db=0.0).output
    data, _ = _read(out)
    assert np.abs(data).max() == pytest.approx(0.99, abs=1e-3)
    with pytest.raises(InvalidInputError, match="method"):
        normalize_audio_file(test_audio_path_3s, tmp_path / "m.wav", method="loud")


def test_resample(test_audio_path_3s, tmp_path):
    data, sr = _read(resample_audio_file(test_audio_path_3s, tmp_path / "r.wav", 8000).output)
    assert sr == 8000 and len(data) == 3 * 8000


@pytest.mark.parametrize(
    ("kwargs", "description"),
    [
        ({"highpass": 200.0}, "highpass filter (200.0 Hz)"),
        ({"lowpass": 1000.0}, "lowpass filter (1000.0 Hz)"),
        ({"bandpass_low": 300.0, "bandpass_high": 600.0, "lowpass": 9.0}, "bandpass filter"),
    ],
)
def test_filter_picks_one_filter(test_audio_path_3s, tmp_path, kwargs, description):
    result = filter_audio_file(test_audio_path_3s, tmp_path / "f.wav", **kwargs)
    assert result.description.startswith(description)
    assert (tmp_path / "f.wav").exists()


def test_filter_validation(test_audio_path_3s, tmp_path):
    with pytest.raises(InvalidInputError, match="Must specify"):
        filter_audio_file(test_audio_path_3s, tmp_path / "f.wav")
    with pytest.raises(InvalidInputError, match="together"):
        filter_audio_file(test_audio_path_3s, tmp_path / "f.wav", bandpass_low=300.0)
    assert not (tmp_path / "f.wav").exists()


def test_gain_and_noise(test_audio_path_3s, tmp_path):
    original, _ = load_audio(test_audio_path_3s)
    louder, _ = _read(gain_audio_file(test_audio_path_3s, tmp_path / "g.wav", -6.0).output)
    assert np.abs(louder).max() == pytest.approx(np.abs(original).max() / 2, rel=0.01)
    a = add_noise_audio_file(test_audio_path_3s, tmp_path / "n1.wav", 10.0, seed=3).output
    b = add_noise_audio_file(test_audio_path_3s, tmp_path / "n2.wav", 10.0, seed=3).output
    assert np.array_equal(_read(a)[0], _read(b)[0])


def test_describe_audio_file(test_audio_path_3s):
    plain = describe_audio_file(test_audio_path_3s)
    assert plain.info.duration == pytest.approx(3.0)
    assert "recording_metadata" not in plain.to_dict()
    with_meta = describe_audio_file(test_audio_path_3s, recording_metadata=True).to_dict()
    assert with_meta["recording_metadata"] == {}  # a WAV without recorder metadata


def test_describe_non_wav_has_no_recording_metadata(test_audio_path_3s, tmp_path):
    flac = convert_audio_file(test_audio_path_3s, tmp_path / "x.flac", target_format="flac")
    assert (
        describe_audio_file(flac, recording_metadata=True).to_dict()["recording_metadata"] is None
    )


def test_convert_format_from_file_name(test_audio_path_3s, tmp_path):
    out = convert_audio_file(test_audio_path_3s, tmp_path / "x.flac", target_format=None)
    assert sf.info(out).format == "FLAC"


def test_segment_options(test_audio_path_3s, tmp_path):
    # 3 s in 1.25 s clips: two whole clips and a 0.5 s remainder.
    default = segment_audio_file(test_audio_path_3s, str(tmp_path / "d"), duration=1.25)
    assert [s.segment_path.name for s in default] == [
        "test_audio_3s_seg0000.wav", "test_audio_3s_seg0001.wav",
    ]  # fmt: skip
    kept = segment_audio_file(test_audio_path_3s, str(tmp_path / "k"), duration=1.25,
                              prefix="rec", format="flac", include_partial=True)  # fmt: skip
    assert [s.segment_path.name for s in kept][-1] == "rec_seg0002.flac"
    assert kept[-1].duration == pytest.approx(0.5)
    plain = segment_audio_file(test_audio_path_3s, str(tmp_path / "p"), duration=1.25, label="")
    assert plain[0].segment_path.name == "test_audio_3s_0000.wav"


def test_single_and_batch_segment_cut_the_same_clips(test_audio_path_3s, tmp_path):
    """`audio segment` and `batch audio segment` give the same names, clips and formats."""
    import shutil

    from click.testing import CliRunner

    from bioamla.cli.cli import cli

    folder = tmp_path / "in"
    folder.mkdir()
    shutil.copy(test_audio_path_3s, folder / "rec.wav")
    runner = CliRunner()
    for keep in ([], ["--keep-short"]):
        one, many = tmp_path / f"one{len(keep)}", tmp_path / f"many{len(keep)}"
        common = ["-d", "1.25", "-f", "flac", *keep]
        r1 = runner.invoke(cli, ["audio", "segment", str(folder / "rec.wav"), str(one), *common])
        r2 = runner.invoke(cli, ["batch", "audio", "segment", "--input-dir", str(folder),
                                 "--output-dir", str(many), *common])  # fmt: skip
        assert r1.exit_code == 0 and r2.exit_code == 0, (r1.output, r2.output)
        names = sorted(p.name for p in one.iterdir())
        assert names == sorted(p.name for p in many.iterdir())
        assert names == ["rec_seg0000.flac", "rec_seg0001.flac"] + (
            ["rec_seg0002.flac"] if keep else []
        )


@pytest.mark.parametrize("prefix", ["../up", "a/b", "..", "", "c\\d"])
def test_segment_prefix_must_be_a_plain_name(test_audio_path_3s, tmp_path, prefix):
    with pytest.raises(InvalidInputError, match="prefix"):
        segment_audio_file(test_audio_path_3s, str(tmp_path), duration=1.0, prefix=prefix)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"start": -1.0}, "negative"),
        ({"duration": 0.0}, "positive"),
        ({"duration": -2.0}, "positive"),
    ],
)
def test_trim_rejects_bad_ranges(test_audio_path_3s, tmp_path, kwargs, match):
    with pytest.raises(InvalidInputError, match=match):
        trim_audio_file(test_audio_path_3s, tmp_path / "x.wav", **kwargs)


def test_filter_zero_and_half_band(test_audio_path_3s, tmp_path):
    with pytest.raises(InvalidInputError, match="together"):
        filter_audio_file(test_audio_path_3s, tmp_path / "f.wav", bandpass_high=4000.0)
    # A zero edge counts as given (the filters clamp cutoffs into (0, Nyquist)).
    result = filter_audio_file(test_audio_path_3s, tmp_path / "f.wav", lowpass=0.0)
    assert result.description == "lowpass filter (0.0 Hz)"


def test_segment_makes_no_folder_when_loading_fails(tmp_path):
    from bioamla.exceptions import BioamlaError

    with pytest.raises(BioamlaError):
        segment_audio_file(str(tmp_path / "missing.wav"), str(tmp_path / "out"), duration=1.0)
    assert not (tmp_path / "out").exists()


def test_listing_a_missing_folder_is_an_error(tmp_path):
    from bioamla.audio import list_audio_files
    from bioamla.exceptions import NotFoundError

    with pytest.raises(NotFoundError, match="Directory not found"):
        list_audio_files(tmp_path / "missing")
    assert list_audio_files(tmp_path) == []  # an empty folder is fine
