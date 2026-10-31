"""`bioamla --json` results for the annotation and dataset commands."""

import json
from unittest.mock import patch

import pytest
from click.testing import CliRunner

from bioamla.cli.cli import cli


def _run(*args: str):
    result = CliRunner().invoke(cli, ["--json", *args])
    return result, json.loads(result.stdout)


def _ok(*args: str) -> dict:
    result, envelope = _run(*args)
    assert result.exit_code == 0, result.output
    assert envelope["ok"] is True
    return envelope["result"]


@pytest.fixture
def labels_csv(tmp_path):
    path = tmp_path / "a.csv"
    path.write_text("start_time,end_time,label\n0.0,1.0,frog\n1.5,1.7,bird\n2.0,3.0,frog\n")
    return path


def test_annotation_commands(test_audio_path, labels_csv, tmp_path) -> None:
    template = _ok(
        "annotation", "template", test_audio_path, str(tmp_path / "t.json"), "--label", "x"
    )
    assert template["rows"] == 1 and template["format"] == "bioamla"
    assert template["audio"]["sample_rate"] > 0

    converted = _ok("annotation", "convert", str(labels_csv), str(tmp_path / "a.txt"))
    assert (converted["from_format"], converted["to_format"], converted["count"]) == (
        "csv",
        "raven",
        3,
    )

    mapping = tmp_path / "map.csv"
    mapping.write_text("source,target\nfrog,anuran\n")
    remapped = _ok(
        "annotation", "remap", str(labels_csv), str(tmp_path / "r.csv"), "-m", str(mapping)
    )
    assert remapped["input_count"] == 3

    filtered = _ok("annotation", "filter", str(labels_csv), str(tmp_path / "f.csv"), "-i", "frog")
    assert (filtered["input_count"], filtered["output_count"]) == (3, 2)

    clips = _ok(
        "annotation",
        "generate-labels",
        str(labels_csv),
        str(tmp_path / "l.csv"),
        "--clip-duration",
        "1.0",
        "--audio-duration",
        "3.0",
    )
    assert clips["clips"] == 3 and clips["labels"] == ["bird", "frog"]

    frames = _ok(
        "annotation",
        "generate-frame-labels",
        str(labels_csv),
        str(tmp_path / "fl.csv"),
        "--frame-size",
        "0.5",
        "--audio-duration",
        "3.0",
    )
    assert frames["frames"] == 6 and frames["classes"] == 2


def test_missing_input_is_a_usage_error(tmp_path) -> None:
    result, envelope = _run(
        "annotation", "convert", str(tmp_path / "nope.csv"), str(tmp_path / "o.csv")
    )
    assert result.exit_code == 2
    assert envelope["error"]["type"] == "BadParameter"
    assert "does not exist" in envelope["error"]["message"]


@pytest.mark.parametrize("fmt", ["csv", "numpy"])
def test_generate_labels_reports_the_files_it_wrote(labels_csv, tmp_path, fmt) -> None:
    from pathlib import Path

    result = _ok(
        "annotation",
        "generate-labels",
        str(labels_csv),
        str(tmp_path / "out" / "labels"),
        "--clip-duration",
        "0.1",
        "--audio-duration",
        "0.3",
        "--format",
        fmt,
    )
    assert result["clips"] == 3  # not 2: 0.2 / 0.1 is 1.9999999999999998 in floats
    assert Path(result["output"]).exists()
    if fmt == "numpy":
        assert result["output"].endswith(".npy") and Path(result["label_map"]).exists()
    else:
        assert result["label_map"] is None


@pytest.mark.parametrize(
    ("extra", "message"),
    [
        (["--clip-duration", "5", "--audio-duration", "3"], "longer than the audio"),
        (
            ["--clip-duration", "1", "--hop-length", "0", "--audio-duration", "3"],
            "must be positive",
        ),
    ],
)
def test_generate_labels_rejects_bad_durations(labels_csv, tmp_path, extra, message) -> None:
    result, envelope = _run(
        "annotation", "generate-labels", str(labels_csv), str(tmp_path / "l.csv"), *extra
    )
    assert result.exit_code == 1
    assert message in envelope["error"]["message"]


def test_augment_range_errors_are_usage_errors(tmp_path) -> None:
    result, envelope = _run("dataset", "augment", str(tmp_path), "-o", "out", "--gain", "abc")
    assert result.exit_code == 2
    assert envelope["error"]["type"] == "BadParameter"


@pytest.mark.parametrize(
    ("command", "target", "stats"),
    [
        (
            ["dataset", "merge", "OUT", "A", "B"],
            "merge_datasets",
            {"datasets_merged": 2, "total_files": 9},
        ),
        (
            ["dataset", "partition", "DIR"],
            "partition_dataset",
            {"groups": 3, "splits": {"train": 2, "test": 1}},
        ),
        (
            ["dataset", "remix", "OUT", "--source", "SRC"],
            "remix_dataset",
            {"total_files": 4, "output_dir": "OUT"},
        ),
    ],
)
def test_dataset_commands_report_their_stats(tmp_path, command, target, stats) -> None:
    args = [str(tmp_path / a) if a.isupper() else a for a in command]
    with patch(f"bioamla.datasets.{target}", return_value=stats):
        assert _ok(*args) == stats


def test_license_batch_partial_failure_is_an_error(tmp_path) -> None:
    stats = {
        "datasets_found": 2,
        "datasets_processed": 1,
        "datasets_failed": 1,
        "results": [
            {"status": "success", "dataset_name": "a", "attributions_count": 5},
            {"status": "failed", "dataset_name": "b", "error": "bad csv"},
        ],
    }
    with patch("bioamla.datasets.generate_licenses_for_directory", return_value=stats):
        result, envelope = _run("dataset", "license", str(tmp_path), "--batch")
    assert result.exit_code == 1
    assert envelope["ok"] is False
    assert "1 of 2" in envelope["error"]["message"]
    # The per-dataset results come along, so a script needn't parse the message.
    assert envelope["error"]["details"]["results"][1] == {
        "status": "failed",
        "dataset_name": "b",
        "error": "bad csv",
    }


def test_license_batch_success_reports_stats(tmp_path) -> None:
    stats = {
        "datasets_found": 1,
        "datasets_processed": 1,
        "datasets_failed": 0,
        "results": [{"status": "success", "dataset_name": "a", "attributions_count": 5}],
    }
    with patch("bioamla.datasets.generate_licenses_for_directory", return_value=stats):
        assert _ok("dataset", "license", str(tmp_path), "--batch") == stats


def test_extract_clips_and_augment_report_stats(tmp_path) -> None:
    extract = {"clips_written": 5, "files_processed": 1, "labels": ["frog"], "failed": []}
    with patch("bioamla.datasets.extract_labeled_dataset", return_value=extract):
        assert (
            _ok("dataset", "extract-clips", str(tmp_path / "s.wav"), str(tmp_path / "out"))
            == extract
        )

    stats = {"files_created": 20, "files_processed": 10, "files_failed": 1, "output_dir": "out"}
    with patch("bioamla.datasets.batch_augment", return_value=stats):
        # A partial failure is a warning for people and a count in the result, as before.
        assert _ok("dataset", "augment", str(tmp_path), "-o", "out", "--add-noise", "3-30") == stats


def test_manifest_and_build(tmp_path) -> None:
    from bioamla.datasets import DatasetManifest

    manifest = DatasetManifest(label2id={"a": 0}, class_counts={"a": 5}, splits={"train": 5})
    with (
        patch("bioamla.datasets.build_manifest_from_metadata", return_value=manifest),
        patch("bioamla.datasets.save_dataset_manifest"),
    ):
        result = _ok("dataset", "manifest", str(tmp_path))
    assert result["manifest"]["label2id"] == {"a": 0}

    with (
        patch("bioamla.datasets.extract_labeled_dataset", return_value={"clips_written": 10}),
        patch("bioamla.datasets.partition_dataset", return_value={"splits": {"train": 10}}),
        patch("bioamla.datasets.build_manifest_from_metadata", return_value=manifest),
        patch("bioamla.datasets.save_dataset_manifest"),
        patch(
            "bioamla.datasets.generate_license_for_dataset",
            return_value={"output_path": str(tmp_path / "ATTRIBUTIONS.md")},
        ),
    ):
        built = _ok("dataset", "build", str(tmp_path / "s.wav"), str(tmp_path / "out"))
    assert built["extract"] == {"clips_written": 10}
    assert built["attributions"] == str(tmp_path / "ATTRIBUTIONS.md")
    assert built["manifest_path"] == str(tmp_path / "out" / "dataset.json")
