"""The file-level indices functions: load a file, compute, optionally save."""

import json

import pytest

pytest.importorskip("librosa")

from bioamla.exceptions import InvalidInputError, NotFoundError, ProcessingError  # noqa: E402
from bioamla.indices import (  # noqa: E402
    SEGMENT_COLUMNS,
    FileIndices,
    compute_all_indices,
    index_file,
    index_values_file,
    temporal_index_file,
)


def test_index_file_matches_compute_all_indices(test_audio_path_3s):
    from bioamla.audio import load_audio_data

    found = index_file(test_audio_path_3s)
    audio = load_audio_data(test_audio_path_3s)
    direct = compute_all_indices(audio.samples, audio.sample_rate, include_entropy=True)
    assert isinstance(found, FileIndices)
    assert found.indices == direct
    assert found.output is None
    assert found.to_dict()["indices"]["h_spectral"] == direct.h_spectral


def test_index_file_writes_json(test_audio_path_3s, tmp_path):
    out = tmp_path / "sub" / "idx.json"
    found = index_file(test_audio_path_3s, output=out)
    saved = json.loads(out.read_text())
    assert found.output == str(out)
    assert saved["filepath"] == test_audio_path_3s
    assert saved["aci"] == pytest.approx(found.indices.aci)


def test_index_file_zero_aci_max_freq_means_nyquist(test_audio_path_3s):
    assert (
        index_file(test_audio_path_3s, aci_max_freq=0).indices
        == index_file(test_audio_path_3s).indices
    )


def test_write_failure_is_a_processing_error(test_audio_path_3s, tmp_path):
    blocker = tmp_path / "file"
    blocker.write_text("")
    with pytest.raises(ProcessingError, match="Failed to write"):
        index_file(test_audio_path_3s, output=blocker / "idx.json")


def test_temporal_index_file_rows(test_audio_path_3s, tmp_path):
    out = tmp_path / "t.json"
    found = temporal_index_file(test_audio_path_3s, window_duration=1.0, output=out)
    assert [r["segment"] for r in found.segments] == [0, 1, 2]
    assert [r["start_time"] for r in found.segments] == [0.0, 1.0, 2.0]
    assert all(tuple(r) == SEGMENT_COLUMNS for r in found.segments)
    assert json.loads(out.read_text()) == found.segments
    assert found.to_dict()["window_duration"] == 1.0


def test_index_values_file(test_audio_path_3s):
    from bioamla.audio import load_audio_data
    from bioamla.indices import compute_index

    audio = load_audio_data(test_audio_path_3s)
    aci = index_values_file(test_audio_path_3s, ["aci"], min_freq=500.0).values
    assert aci == {"aci": compute_index(audio.samples, audio.sample_rate, "aci", min_freq=500.0)}
    both = index_values_file(test_audio_path_3s, ["bio", "ndsi"]).values
    assert list(both) == ["bio", "ndsi"]
    assert both["bio"] == compute_index(audio.samples, audio.sample_rate, "bio")


def test_index_values_file_params_need_a_single_index(test_audio_path_3s):
    with pytest.raises(InvalidInputError, match="apply to one index"):
        index_values_file(test_audio_path_3s, ["aci", "adi"], max_freq=4000.0)


def test_ndsi_honours_n_fft_and_keeps_its_default(test_audio_path_3s):
    from bioamla.audio import load_audio_data
    from bioamla.indices import compute_index, compute_ndsi

    audio = load_audio_data(test_audio_path_3s)
    default = index_values_file(test_audio_path_3s, ["ndsi"]).values["ndsi"]
    assert default == compute_ndsi(audio.samples, audio.sample_rate, n_fft=1024)[0]
    assert (
        compute_index(audio.samples, audio.sample_rate, "ndsi", n_fft=256)
        == (compute_ndsi(audio.samples, audio.sample_rate, n_fft=256)[0])
    )


def test_temporal_window_must_be_at_least_one_sample(test_audio_path_3s):
    for window in (0, 1e-9, -1):
        with pytest.raises(InvalidInputError, match="at least one sample"):
            temporal_index_file(test_audio_path_3s, window_duration=window)


def test_empty_output_saves_nothing(test_audio_path_3s, tmp_path, monkeypatch):
    work = tmp_path / "work"
    work.mkdir()
    monkeypatch.chdir(work)
    assert index_file(test_audio_path_3s, output="").output is None
    assert temporal_index_file(test_audio_path_3s, window_duration=1.0, output="").output is None
    assert list(work.iterdir()) == []


def test_index_values_file_rejects_unknown_names(test_audio_path_3s):
    with pytest.raises(InvalidInputError):
        index_values_file(test_audio_path_3s, ["nope"])


def test_missing_file(tmp_path):
    with pytest.raises(NotFoundError):
        index_file(tmp_path / "missing.wav")
