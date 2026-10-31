"""Tests for the indices domain (flattened, exception-based API)."""

from pathlib import Path

import numpy as np
import pytest

from bioamla.audio import AudioData
from bioamla.exceptions import InvalidInputError
from bioamla.indices import (
    AVAILABLE_INDICES,
    AcousticIndices,
    batch_compute_indices,
    batch_index_files,
    compute_all_indices,
    compute_index,
    describe_index,
    temporal_indices,
)


class TestComputeAllIndices:
    def test_compute_all_indices(self, sample_audio_data: AudioData) -> None:
        result = compute_all_indices(sample_audio_data.samples, sample_audio_data.sample_rate)
        assert isinstance(result, AcousticIndices)
        d = result.to_dict()
        for key in ("aci", "adi", "aei", "bio", "ndsi"):
            assert key in d

    def test_compute_indices_with_entropy(self, sample_audio_data: AudioData) -> None:
        result = compute_all_indices(
            sample_audio_data.samples, sample_audio_data.sample_rate, include_entropy=True
        )
        assert result.h_spectral is not None
        assert result.h_temporal is not None

    def test_compute_indices_returns_valid_values(self, sample_audio_data: AudioData) -> None:
        result = compute_all_indices(sample_audio_data.samples, sample_audio_data.sample_rate)
        for key, value in result.to_dict().items():
            if isinstance(value, (int, float)):
                assert not np.isnan(value), f"{key} is NaN"
                assert not np.isinf(value), f"{key} is infinite"


class TestComputeSingleIndex:
    @pytest.mark.parametrize(
        "index_name",
        ["aci", "adi", "aei", "bio", "ndsi", "h_spectral", "h_temporal"],
    )
    def test_compute_index(self, sample_audio_data: AudioData, index_name: str) -> None:
        value = compute_index(sample_audio_data.samples, sample_audio_data.sample_rate, index_name)
        assert isinstance(value, float)
        assert not np.isnan(value)

    def test_compute_index_invalid_name(self, sample_audio_data: AudioData) -> None:
        with pytest.raises(InvalidInputError, match="Unknown index"):
            compute_index(sample_audio_data.samples, sample_audio_data.sample_rate, "invalid_index")


class TestTemporalIndices:
    def test_temporal_indices(self, sample_audio_3s: AudioData) -> None:
        windows = temporal_indices(
            sample_audio_3s.samples,
            sample_audio_3s.sample_rate,
            window_duration=1.0,
            hop_duration=1.0,
        )
        # 3-second audio with 1-second windows => 3 windows
        assert len(windows) == 3


class TestIndicesHelpers:
    def test_available_indices(self) -> None:
        assert "aci" in AVAILABLE_INDICES
        assert "ndsi" in AVAILABLE_INDICES

    def test_describe_index(self) -> None:
        description = describe_index("aci")
        assert isinstance(description, str)
        assert len(description) > 0

    def test_describe_index_invalid(self) -> None:
        assert describe_index("invalid") is None


class TestBatchComputeIndices:
    def test_parallel_keeps_input_order(self, test_audio_dir: str, tmp_path) -> None:
        files = sorted(str(p) for p in Path(test_audio_dir).glob("*.wav"))
        missing = str(tmp_path / "missing.wav")
        paths = [files[2], missing, files[0], files[1]]

        results = batch_compute_indices(paths, verbose=False, max_workers=4)

        assert [r["filepath"] for r in results] == paths
        assert [r["success"] for r in results] == [True, False, True, True]
        assert results[1]["error"]
        assert "ndsi" in results[0]

    def test_repeated_path_computed_once(self, test_audio_dir: str) -> None:
        path = str(sorted(Path(test_audio_dir).glob("*.wav"))[0])
        calls: list[tuple[int, int]] = []
        results = batch_compute_indices(
            [path, path], verbose=True, on_progress=lambda d, t: calls.append((d, t))
        )
        assert results[0] == results[1]
        assert calls == [(1, 1)]


class TestBatchIndexFiles:
    def test_rows_csv_progress_and_failures(self, test_audio_dir: str, tmp_path) -> None:
        (Path(test_audio_dir) / "broken.wav").write_bytes(b"not audio")
        calls: list[tuple[int, int]] = []

        result = batch_index_files(
            test_audio_dir,
            tmp_path / "out",
            max_workers=4,
            on_progress=lambda done, total: calls.append((done, total)),
        )

        rows = result.metadata["rows"]
        assert [Path(r["filepath"]).name for r in rows] == [
            "audio_0.wav",
            "audio_1.wav",
            "audio_2.wav",
            "broken.wav",
        ]
        assert [r["success"] for r in rows] == [True, True, True, False]
        assert result.failed == 1
        assert calls[-1] == (4, 4)
        assert (tmp_path / "out" / "indices.csv").exists()


class TestIndicesRow:
    def test_selection_keeps_companions_and_context(self, test_audio_dir: str) -> None:
        from bioamla.indices import indices_row

        path = sorted(Path(test_audio_dir).glob("*.wav"))[0]
        row = indices_row(path, ["ndsi", "h_temporal"])
        assert list(row) == [
            "filepath",
            "ndsi",
            "anthrophony",
            "biophony",
            "sample_rate",
            "duration",
            "h_temporal",
            "success",
        ]

    def test_unknown_index_rejected(self, test_audio_dir: str) -> None:
        from bioamla.indices import indices_row

        path = sorted(Path(test_audio_dir).glob("*.wav"))[0]
        with pytest.raises(InvalidInputError, match="Unknown indices: nope"):
            indices_row(path, ["aci", "nope"])
