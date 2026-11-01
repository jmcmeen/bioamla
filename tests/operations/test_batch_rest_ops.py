"""batch detect|index|cluster|models operations, and parity with the CLI.

Each parity test runs the command with `--json`, then the same batch with library
calls alone, and compares the results and files. The model commands' parity tests
are in tests/integration (they need a model).
"""

import csv
import json
from pathlib import Path

import numpy as np
import pytest
from click.testing import CliRunner

from bioamla import operations
from bioamla.batch import BatchConfig
from bioamla.cli.cli import cli
from bioamla.exceptions import ConfigValidationError, InvalidInputError


def _cli(*args: str) -> dict:
    result = CliRunner().invoke(cli, ["--json", "batch", *args])
    assert result.exit_code == 0, result.output
    return json.loads(result.stdout)["result"]


def _csv(path) -> list[dict]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


@pytest.fixture
def manifest(test_audio_dir) -> Path:
    path = Path(test_audio_dir) / "files.csv"
    names = sorted(p.name for p in Path(test_audio_dir).glob("*.wav"))
    path.write_text("file_name\n" + "".join(f"{n}\n" for n in names))
    return path


@pytest.mark.parametrize("method", ["energy", "peaks"])
def test_detect_parity_both_modes(method, test_audio_dir, manifest, tmp_path):
    pytest.importorskip("librosa")
    from bioamla.detect import batch_detection

    got = _cli("detect", method, "--input-dir", test_audio_dir, "--output-dir", str(tmp_path / "c"))
    want = batch_detection(
        BatchConfig(input_dir=test_audio_dir, output_dir=str(tmp_path / "l")), method
    )
    assert got["successful"] == want.successful == 3
    name = f"detections_{method}.json"
    assert Path(got["results_file"]).name == name == Path(want.metadata["output_file"]).name
    cli_json = json.loads((tmp_path / "c" / name).read_text())
    lib_json = json.loads((tmp_path / "l" / name).read_text())
    assert json.dumps(cli_json, sort_keys=True) == json.dumps(lib_json, sort_keys=True)

    got = _cli(
        "detect", method, "--input-file", str(manifest), "--output-dir", str(tmp_path / "cc")
    )
    want = batch_detection(
        BatchConfig(input_file=str(manifest), output_dir=str(tmp_path / "lc")), method
    )
    cli_counts = [r["num_detections"] for r in _csv(got["metadata_csv"])]
    assert cli_counts == [r["num_detections"] for r in _csv(want.metadata["metadata_csv"])]


def test_detect_default_output_is_the_input_folder(test_audio_dir):
    pytest.importorskip("librosa")
    from bioamla.detect import batch_detection

    result = batch_detection(BatchConfig(input_dir=test_audio_dir), "energy")
    assert Path(result.metadata["output_file"]).parent == Path(test_audio_dir)


def test_index_parity_both_modes(test_audio_dir, manifest, tmp_path):
    pytest.importorskip("librosa")
    from bioamla.indices import batch_indices

    got = _cli(
        "index",
        "--input-dir",
        test_audio_dir,
        "--output-dir",
        str(tmp_path / "c"),
        "--indices",
        "aci,ndsi",
    )
    batch_indices(
        BatchConfig(input_dir=test_audio_dir, output_dir=str(tmp_path / "l")), ["aci", "ndsi"]
    )
    assert Path(got["results_file"]).name == "indices.csv"
    assert _csv(tmp_path / "c" / "indices.csv") == _csv(tmp_path / "l" / "indices.csv")

    got = _cli(
        "index",
        "--input-file",
        str(manifest),
        "--output-dir",
        str(tmp_path / "cc"),
        "--indices",
        "bio",
    )
    want = batch_indices(
        BatchConfig(input_file=str(manifest), output_dir=str(tmp_path / "lc")), ["bio"]
    )
    assert _csv(got["metadata_csv"]) == _csv(want.metadata["metadata_csv"])


def test_index_rejects_an_empty_selection(test_audio_dir):
    from bioamla.indices import batch_indices

    with pytest.raises(InvalidInputError):
        batch_indices(BatchConfig(input_dir=test_audio_dir), [])


@pytest.fixture
def embeddings(tmp_path) -> Path:
    folder = tmp_path / "emb"
    folder.mkdir()
    rng = np.random.default_rng(0)
    for i in range(8):
        center = 0.0 if i < 4 else 5.0
        np.save(folder / f"e{i}.npy", rng.normal(center, 0.1, size=(3, 4)).astype(np.float32))
    return folder


def test_cluster_parity_both_modes(embeddings, tmp_path):
    pytest.importorskip("sklearn")
    from bioamla.cluster import batch_cluster

    args = ["--method", "kmeans", "--n-clusters", "2"]
    got = _cli(
        "cluster", "--input-dir", str(embeddings), "--output-dir", str(tmp_path / "c"), *args
    )
    want = batch_cluster(
        BatchConfig(input_dir=str(embeddings), output_dir=str(tmp_path / "l")),
        method="kmeans",
        n_clusters=2,
    )
    assert got["successful"] == want.successful == 8
    a = json.loads((tmp_path / "c" / "cluster_assignments.json").read_text())
    b = json.loads((tmp_path / "l" / "cluster_assignments.json").read_text())
    assert a == b

    manifest = embeddings / "files.csv"
    manifest.write_text("file_name\n" + "".join(f"e{i}.npy\n" for i in range(8)))
    got = _cli(
        "cluster", "--input-file", str(manifest), "--output-dir", str(tmp_path / "cc"), *args
    )
    want = batch_cluster(
        BatchConfig(input_file=str(manifest), output_dir=str(tmp_path / "lc")),
        method="kmeans",
        n_clusters=2,
    )
    cli_rows, lib_rows = _csv(got["metadata_csv"]), _csv(want.metadata["metadata_csv"])
    assert [r["cluster"] for r in cli_rows] == [r["cluster"] for r in lib_rows]
    assert len({r["cluster"] for r in cli_rows}) == 2


def test_cluster_needs_an_output_dir(embeddings):
    from bioamla.cluster import batch_cluster

    with pytest.raises(InvalidInputError, match="output_dir"):
        batch_cluster(BatchConfig(input_dir=str(embeddings)))


@pytest.mark.parametrize("name", ["batch.models.predict", "batch.models.embed"])
def test_model_requests_need_a_model(name, test_audio_dir):
    with pytest.raises((InvalidInputError, ConfigValidationError), match="model"):
        operations.run(name, {"input_dir": test_audio_dir, "output_dir": "/tmp/x"})


@pytest.mark.parametrize(
    "name",
    [
        "batch.detect.energy",
        "batch.detect.ribbit",
        "batch.detect.peaks",
        "batch.detect.accelerating",
        "batch.index",
        "batch.models.predict",
        "batch.models.embed",
        "batch.cluster",
    ],
)
def test_every_batch_operation_checks_its_paths(name, mocker):
    op = operations.get(name)
    assert {"input_dir", "input_file", "output_dir"} <= set(op.path_fields())
    assert (op.kind, op.remote) == ("job", True)

    def refuse(path):
        raise PermissionError(path)

    request = (
        {"input_dir": "/nowhere", "model": "m"} if "models" in name else {"input_dir": "/nowhere"}
    )
    with pytest.raises(PermissionError):
        operations.run(name, request, operations.OperationContext(path_check=refuse))
