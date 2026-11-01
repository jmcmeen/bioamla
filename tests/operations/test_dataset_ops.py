"""dataset.* operations, and parity with the CLI.

Each parity test runs a `dataset` command with `--json`, then does the same with
bioamla's library functions alone (no CLI, no registry), and compares results and files.
"""

import csv
import json
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
from click.testing import CliRunner

from bioamla import datasets as lib
from bioamla import operations
from bioamla.cli.cli import cli
from bioamla.exceptions import InvalidInputError, OptionConflictError


def _cli(*args: str) -> dict:
    result = CliRunner().invoke(cli, ["--json", "dataset", *args])
    assert result.exit_code == 0, result.output
    return json.loads(result.stdout)["result"]


def _rows(path) -> list[dict]:
    with open(path, newline="") as f:
        return sorted(csv.DictReader(f), key=lambda r: json.dumps(r, sort_keys=True))


@pytest.fixture
def recordings(tmp_path) -> Path:
    """Three 4 s recordings, each with a sibling CSV of two frog and two bird calls."""
    folder = tmp_path / "rec"
    folder.mkdir()
    t = np.arange(16000 * 4) / 16000
    for i in range(3):
        sf.write(folder / f"r{i}.wav", 0.3 * np.sin(2 * np.pi * (800 + 200 * i) * t), 16000)
        (folder / f"r{i}.csv").write_text(
            "start_time,end_time,label\n0.2,0.8,frog\n1.0,1.5,bird\n2.0,2.6,frog\n3.0,3.4,bird\n"
        )
    return folder


def test_extract_clips(recordings, tmp_path):
    got = _cli("extract-clips", str(recordings), str(tmp_path / "c"))
    want = lib.extract_labeled_dataset(str(recordings), str(tmp_path / "l"), verbose=False)
    assert got["clips_written"] == want["clips_written"] == 12
    assert got["labels"] == want["labels"]
    assert len(_rows(tmp_path / "c" / "metadata.csv")) == len(
        _rows(tmp_path / "l" / "metadata.csv")
    )


def test_build_stats_manifest(recordings, tmp_path):
    got = _cli("build", str(recordings), str(tmp_path / "c"), "--seed", "1", "--no-attributions")
    want = lib.build_dataset(recordings, tmp_path / "l", seed=1, attributions=False, verbose=False)
    assert got["extract"]["clips_written"] == want["extract"]["clips_written"]
    assert got["partition"]["splits"] == want["partition"]["splits"]
    assert got["manifest_path"] == str(tmp_path / "c" / "dataset.json")
    a = json.loads((tmp_path / "c" / "dataset.json").read_text())
    b = json.loads((tmp_path / "l" / "dataset.json").read_text())
    for m in (a, b):
        m.pop("created"), m.pop("name")
    assert a == b

    assert _cli("stats", str(tmp_path / "c")) == lib.get_dataset_stats(str(tmp_path / "l"))
    got = _cli("manifest", str(tmp_path / "c"), "-o", str(tmp_path / "m1.json"))
    lib.write_dataset_manifest(tmp_path / "l", output=tmp_path / "m2.json")
    assert got["output"] == str(tmp_path / "m1.json")
    a, b = (json.loads((tmp_path / n).read_text()) for n in ("m1.json", "m2.json"))
    for m in (a, b):
        m.pop("created"), m.pop("name")
    assert a == b


def test_partition_column_mode(recordings, tmp_path):
    for name in ("c", "l"):
        lib.extract_labeled_dataset(str(recordings), str(tmp_path / name), verbose=False)
    got = _cli("partition", str(tmp_path / "c"), "--mode", "column", "--seed", "3")
    want = lib.partition_dataset(str(tmp_path / "l"), mode="column", seed=3, verbose=False)
    assert got["splits"] == want["splits"]
    assert [r["split"] for r in _rows(tmp_path / "c" / "metadata.csv")] == [
        r["split"] for r in _rows(tmp_path / "l" / "metadata.csv")
    ]


def test_augment(recordings, tmp_path):
    got = _cli("augment", str(recordings), "-o", str(tmp_path / "c"), "--gain", "-3,-3")
    want = lib.augment_dataset(recordings, tmp_path / "l", gain=(-3.0, -3.0), verbose=False)
    assert got["files_created"] == want["files_created"] == 3
    for a in sorted((tmp_path / "c").rglob("*.wav")):
        b = tmp_path / "l" / a.relative_to(tmp_path / "c")
        np.testing.assert_allclose(sf.read(a)[0], sf.read(b)[0], atol=1e-4)


def test_augment_needs_a_transform(recordings, tmp_path):
    with pytest.raises(InvalidInputError, match="At least one"):
        lib.augment_dataset(recordings, tmp_path / "x")


def test_remix_and_merge(recordings, tmp_path):
    for name in ("src",):
        lib.extract_labeled_dataset(str(recordings), str(tmp_path / name), verbose=False)
    got = _cli("remix", str(tmp_path / "c"), "--source", str(tmp_path / "src"), "--keep", "frog")
    want = lib.remix_from(tmp_path / "l", source=tmp_path / "src", keep=["frog"], verbose=False)
    assert got["labels"] == want["labels"] == ["frog"]
    assert got["total_files"] == want["total_files"]

    got = _cli("merge", str(tmp_path / "mc"), str(tmp_path / "c"), str(tmp_path / "l"))
    want = lib.merge_datasets(
        [str(tmp_path / "c"), str(tmp_path / "l")], str(tmp_path / "ml"), verbose=False
    )
    assert got["total_files"] == want["total_files"]


def test_remix_option_conflicts_are_usage_errors(tmp_path):
    with pytest.raises(OptionConflictError):
        lib.remix_from(tmp_path / "x", source=tmp_path, keep=["a"], drop=["b"])
    with pytest.raises(OptionConflictError):
        lib.remix_from(tmp_path / "x")
    run = CliRunner().invoke(cli, ["dataset", "remix", str(tmp_path / "x")])
    assert run.exit_code == 2


def test_license_checks(tmp_path):
    from bioamla.exceptions import DatasetError, NotFoundError

    with pytest.raises(NotFoundError, match="not a directory"):
        lib.write_dataset_license(tmp_path / "missing")
    with pytest.raises(NotFoundError, match="Metadata file"):
        lib.write_dataset_license(tmp_path)
    with pytest.raises(DatasetError, match="No datasets found"):
        lib.write_dataset_license(tmp_path, batch=True)


def test_operations_check_paths_and_kinds(recordings, tmp_path):
    seen = []

    def check(path):
        seen.append(Path(path).name)
        return path

    operations.run(
        "dataset.merge",
        {"output_dir": str(tmp_path / "m"), "dataset_paths": [str(recordings)], "verbose": False},
        operations.OperationContext(path_check=check),
    )
    assert sorted(seen) == ["m", "rec"]  # every listed dataset is checked
    assert operations.get("dataset.stats").kind == "quick"
    assert operations.get("dataset.build").kind == "job"


def test_remix_checks_the_paths_inside_its_recipe(recordings, tmp_path):
    lib.extract_labeled_dataset(str(recordings), str(tmp_path / "outside"), verbose=False)
    recipe = tmp_path / "allowed" / "r.toml"
    recipe.parent.mkdir()
    recipe.write_text('[remix]\nsource = "../outside"\n')
    allowed = tmp_path / "allowed"

    def confine(path):
        if not str(Path(path).resolve()).startswith(str(allowed)):
            raise PermissionError(path)
        return path

    with pytest.raises(PermissionError, match="outside"):
        operations.run(
            "dataset.remix",
            {"output_dir": str(allowed / "out"), "recipe": str(recipe), "verbose": False},
            operations.OperationContext(path_check=confine),
        )


@pytest.mark.parametrize("name", ["../x", "/etc/x", "a/b", "..", "~/x"])
def test_file_names_in_requests_must_be_plain(name, tmp_path):
    with pytest.raises(InvalidInputError, match="plain file name"):
        operations.run("dataset.license", {"path": str(tmp_path), "output": name})
    with pytest.raises(InvalidInputError, match="plain file name"):
        operations.run("dataset.stats", {"dataset_dir": str(tmp_path), "metadata_filename": name})
