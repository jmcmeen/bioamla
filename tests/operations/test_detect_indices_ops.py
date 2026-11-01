"""The detect and indices operations, run through the registry without the CLI."""

import pytest

pytest.importorskip("librosa")

from bioamla import operations  # noqa: E402

CASES = [
    ("detect.energy", {}, "FileDetections"),
    ("detect.ribbit", {}, "FileDetections"),
    ("detect.peaks", {}, "FilePeaks"),
    ("detect.accelerating", {}, "FileDetections"),
    ("indices.compute", {}, "FileIndices"),
    ("indices.temporal", {"window_duration": 1.0}, "FileTemporalIndices"),
    ("indices.aci", {}, "FileIndexValues"),
    ("indices.adi", {}, "FileIndexValues"),
    ("indices.aei", {}, "FileIndexValues"),
    ("indices.bio", {}, "FileIndexValues"),
    ("indices.ndsi", {}, "FileIndexValues"),
    ("indices.entropy", {}, "FileIndexValues"),
]


@pytest.mark.parametrize(("name", "extra", "result_type"), CASES)
def test_runs_through_the_path_check(name, extra, result_type, test_audio_path_3s):
    checked = []

    def check(path):
        checked.append(path)
        return path

    ctx = operations.OperationContext(path_check=check)
    result = operations.run(name, {"file": test_audio_path_3s, **extra}, ctx)
    assert type(result).__name__ == result_type
    assert result.file == test_audio_path_3s
    assert checked == [test_audio_path_3s]
    assert "file" in result.to_dict()


@pytest.mark.parametrize("name", [c[0] for c in CASES])
def test_declared_as_remote_jobs(name):
    op = operations.get(name)
    assert (op.kind, op.remote) == ("job", True)
    assert op.summary
    assert "file" in op.path_fields()


@pytest.mark.parametrize(
    "name", ["detect.energy", "detect.peaks", "indices.compute", "indices.temporal"]
)
def test_output_paths_are_checked(name, test_audio_path_3s, tmp_path):
    def refuse_outputs(path):
        if path.endswith(".out"):
            raise PermissionError(path)
        return path

    ctx = operations.OperationContext(path_check=refuse_outputs)
    with pytest.raises(PermissionError):
        operations.run(name, {"file": test_audio_path_3s, "output": str(tmp_path / "x.out")}, ctx)


def test_entropy_returns_both_values(test_audio_path_3s):
    values = operations.run("indices.entropy", {"file": test_audio_path_3s}).values
    assert list(values) == ["h_spectral", "h_temporal"]


def test_request_defaults():
    from bioamla.exceptions import ConfigValidationError
    from bioamla.indices.operations import BioIndexRequest, IndexBandRequest, NdsiRequest

    bio = BioIndexRequest(file="a")
    assert (bio.min_freq, bio.max_freq) == (2000.0, 8000.0)
    with pytest.raises(ConfigValidationError):  # compute_bio needs a real upper bound
        BioIndexRequest(file="a", max_freq=None)
    assert IndexBandRequest(file="a").max_freq is None  # ACI: up to Nyquist
    assert NdsiRequest(file="a").n_fft == 1024  # NDSI's own default


def test_temporal_rejects_a_zero_window(test_audio_path_3s):
    from bioamla.exceptions import InvalidInputError

    with pytest.raises(InvalidInputError):
        operations.run("indices.temporal", {"file": test_audio_path_3s, "window_duration": 0})
