"""CLI tests for `bioamla dataset` commands."""

from __future__ import annotations

from unittest.mock import patch

import pytest
from click.testing import CliRunner

from bioamla.cli.cli import cli
from bioamla.exceptions import DatasetError, InvalidInputError, MergeError


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


def test_dataset_group_help(runner: CliRunner) -> None:
    result = runner.invoke(cli, ["dataset", "--help"])
    assert result.exit_code == 0
    for sub in [
        "merge",
        "extract-clips",
        "stats",
        "manifest",
        "partition",
        "build",
        "license",
        "augment",
    ]:
        assert sub in result.output


# --- merge ---------------------------------------------------------------


def test_dataset_merge_quiet(runner: CliRunner, tmp_path) -> None:
    stats = {"datasets_merged": 2, "total_files": 100, "files_converted": 0}
    with patch("bioamla.datasets.merge.merge_datasets", return_value=stats):
        result = runner.invoke(
            cli,
            [
                "dataset",
                "merge",
                str(tmp_path / "out"),
                str(tmp_path / "a"),
                str(tmp_path / "b"),
                "--quiet",
            ],
        )
    assert result.exit_code == 0, result.output
    assert "Merged 2 datasets" in result.output


def test_dataset_merge_error(runner: CliRunner, tmp_path) -> None:
    with patch("bioamla.datasets.merge.merge_datasets", side_effect=MergeError("conflict")):
        result = runner.invoke(
            cli, ["dataset", "merge", str(tmp_path / "out"), str(tmp_path / "a")]
        )
    assert result.exit_code != 0


# --- extract-clips -------------------------------------------------------


def test_dataset_extract_clips(runner: CliRunner, tmp_path) -> None:
    result_dict = {
        "clips_written": 5,
        "files_processed": 1,
        "output_dir": str(tmp_path / "out"),
        "labels": ["frog", "bird"],
        "metadata_file": str(tmp_path / "out" / "metadata.csv"),
        "provenance": {"joined": True, "matched": 5, "columns": ["license"], "unmatched": 0},
        "skipped": [],
        "failed": [],
    }
    with patch(
        "bioamla.datasets.labeled_dataset.extract_labeled_dataset", return_value=result_dict
    ):
        result = runner.invoke(
            cli, ["dataset", "extract-clips", str(tmp_path / "src.wav"), str(tmp_path / "out")]
        )
    assert result.exit_code == 0, result.output
    assert "Extracted 5 clips" in result.output


def test_dataset_extract_clips_error(runner: CliRunner, tmp_path) -> None:
    with patch(
        "bioamla.datasets.extract_labeled_dataset", side_effect=DatasetError("no annotations")
    ):
        result = runner.invoke(
            cli, ["dataset", "extract-clips", str(tmp_path / "src.wav"), str(tmp_path / "out")]
        )
    assert result.exit_code != 0


# --- stats ---------------------------------------------------------------


def test_dataset_stats(runner: CliRunner, tmp_path) -> None:
    stats = {
        "total_files": 50,
        "num_categories": 2,
        "splits": {"train": 40, "test": 10},
        "categories": {"frog": 30, "bird": 20},
        "licenses": {"cc0": 50},
    }
    with patch("bioamla.datasets.stats.get_dataset_stats", return_value=stats):
        result = runner.invoke(cli, ["dataset", "stats", str(tmp_path)])
    assert result.exit_code == 0, result.output
    assert "Total files: 50" in result.output


def test_dataset_stats_json(runner: CliRunner, tmp_path) -> None:
    stats = {"total_files": 1, "num_categories": 1, "categories": {"frog": 1}}
    with patch("bioamla.datasets.stats.get_dataset_stats", return_value=stats):
        result = runner.invoke(cli, ["dataset", "stats", str(tmp_path), "--json"])
    assert result.exit_code == 0
    assert '"total_files": 1' in result.output


def test_dataset_stats_error(runner: CliRunner, tmp_path) -> None:
    with patch("bioamla.datasets.stats.get_dataset_stats", side_effect=DatasetError("no csv")):
        result = runner.invoke(cli, ["dataset", "stats", str(tmp_path)])
    assert result.exit_code != 0


# --- manifest ------------------------------------------------------------


def test_dataset_manifest(runner: CliRunner, tmp_path) -> None:
    from bioamla.datasets import DatasetManifest

    manifest = DatasetManifest(
        label2id={"a": 0, "b": 1}, class_counts={"a": 5, "b": 3}, splits={"train": 8}
    )
    with (
        patch("bioamla.datasets.manifest.build_manifest_from_metadata", return_value=manifest),
        patch("bioamla.datasets.manifest.save_dataset_manifest"),
    ):
        result = runner.invoke(cli, ["dataset", "manifest", str(tmp_path)])
    assert result.exit_code == 0, result.output
    assert "Wrote manifest" in result.output
    assert "recorded:" not in result.output


def test_dataset_manifest_prints_recording_summary(runner: CliRunner, tmp_path) -> None:
    from bioamla.datasets import DatasetManifest

    manifest = DatasetManifest(
        label2id={"a": 0},
        class_counts={"a": 2},
        recorded_from="2023-05-01T22:00:00-04:00",
        recorded_to="2023-06-15T05:00:00-04:00",
        devices=[{"device_model": "Song Meter Micro", "device_id": "SMU00001", "files": 2}],
        sites=[{"latitude": 36.52, "longitude": -82.21, "files": 2}],
    )
    with (
        patch("bioamla.datasets.manifest.build_manifest_from_metadata", return_value=manifest),
        patch("bioamla.datasets.manifest.save_dataset_manifest"),
    ):
        result = runner.invoke(cli, ["dataset", "manifest", str(tmp_path)])
    assert result.exit_code == 0, result.output
    assert "recorded: 2023-05-01T22:00:00-04:00 to 2023-06-15T05:00:00-04:00" in result.output
    assert "recorders: 1  sites: 1" in result.output


def test_dataset_manifest_error(runner: CliRunner, tmp_path) -> None:
    with patch(
        "bioamla.datasets.manifest.build_manifest_from_metadata", side_effect=DatasetError("bad")
    ):
        result = runner.invoke(cli, ["dataset", "manifest", str(tmp_path)])
    assert result.exit_code != 0


# --- partition / split ---------------------------------------------------


def test_dataset_partition(runner: CliRunner, tmp_path) -> None:
    result_dict = {
        "groups": 30,
        "mode": "subdirs",
        "splits": {"train": 21, "val": 5, "test": 4},
        "metadata_file": str(tmp_path / "metadata.csv"),
    }
    with patch("bioamla.datasets.partition.partition_dataset", return_value=result_dict):
        result = runner.invoke(cli, ["dataset", "partition", str(tmp_path)])
    assert result.exit_code == 0, result.output
    assert "Partitioned" in result.output


def test_dataset_partition_error(runner: CliRunner, tmp_path) -> None:
    with patch("bioamla.datasets.partition.partition_dataset", side_effect=DatasetError("x")):
        result = runner.invoke(cli, ["dataset", "partition", str(tmp_path)])
    assert result.exit_code != 0


# --- build ---------------------------------------------------------------


def test_dataset_build(runner: CliRunner, tmp_path) -> None:
    from types import SimpleNamespace

    extract = {"clips_written": 10}
    partition = {"splits": {"train": 7, "val": 2, "test": 1}}
    manifest = SimpleNamespace(label2id={"a": 0})
    with (
        patch("bioamla.datasets.labeled_dataset.extract_labeled_dataset", return_value=extract),
        patch("bioamla.datasets.partition.partition_dataset", return_value=partition),
        patch("bioamla.datasets.manifest.build_manifest_from_metadata", return_value=manifest),
        patch("bioamla.datasets.manifest.save_dataset_manifest"),
        patch(
            "bioamla.datasets.generate_license_for_dataset",
            return_value={"output_path": str(tmp_path / "ATTRIBUTIONS.md")},
        ),
    ):
        result = runner.invoke(
            cli, ["dataset", "build", str(tmp_path / "src.wav"), str(tmp_path / "out")]
        )
    assert result.exit_code == 0, result.output
    assert "Built dataset" in result.output


def test_dataset_build_no_partition(runner: CliRunner, tmp_path) -> None:
    from types import SimpleNamespace

    extract = {"clips_written": 10}
    manifest = SimpleNamespace(label2id={"a": 0})
    with (
        patch("bioamla.datasets.labeled_dataset.extract_labeled_dataset", return_value=extract),
        patch("bioamla.datasets.partition.partition_dataset") as mpart,
        patch("bioamla.datasets.manifest.build_manifest_from_metadata", return_value=manifest),
        patch("bioamla.datasets.manifest.save_dataset_manifest"),
        patch(
            "bioamla.datasets.licenses.generate_license_for_dataset",
            side_effect=DatasetError("none"),
        ),
    ):
        result = runner.invoke(
            cli,
            [
                "dataset",
                "build",
                str(tmp_path / "src.wav"),
                str(tmp_path / "out"),
                "--no-partition",
                "--no-attributions",
            ],
        )
    assert result.exit_code == 0, result.output
    mpart.assert_not_called()


def test_dataset_build_error(runner: CliRunner, tmp_path) -> None:
    with patch(
        "bioamla.datasets.labeled_dataset.extract_labeled_dataset", side_effect=DatasetError("fail")
    ):
        result = runner.invoke(
            cli, ["dataset", "build", str(tmp_path / "src.wav"), str(tmp_path / "out")]
        )
    assert result.exit_code != 0


# --- license -------------------------------------------------------------


def test_dataset_license_single(runner: CliRunner, tmp_path) -> None:
    ds = tmp_path / "ds"
    ds.mkdir()
    (ds / "metadata.csv").write_text("file_name,label\na.wav,frog\n")
    stats = {
        "output_path": str(ds / "LICENSE"),
        "attributions_count": 3,
        "file_size": 1234,
    }
    with patch("bioamla.datasets.licenses.generate_license_for_dataset", return_value=stats):
        result = runner.invoke(cli, ["dataset", "license", str(ds)])
    assert result.exit_code == 0, result.output
    assert "License file generated" in result.output


def test_dataset_license_not_dir(runner: CliRunner, tmp_path) -> None:
    f = tmp_path / "file.txt"
    f.write_text("x")
    result = runner.invoke(cli, ["dataset", "license", str(f)])
    assert result.exit_code != 0
    assert "not a directory" in result.output


def test_dataset_license_no_metadata(runner: CliRunner, tmp_path) -> None:
    ds = tmp_path / "ds"
    ds.mkdir()
    result = runner.invoke(cli, ["dataset", "license", str(ds)])
    assert result.exit_code != 0
    assert "not found" in result.output


def test_dataset_license_batch(runner: CliRunner, tmp_path) -> None:
    stats = {
        "datasets_found": 2,
        "datasets_processed": 2,
        "datasets_failed": 0,
        "results": [
            {"status": "success", "dataset_name": "a", "attributions_count": 5},
            {"status": "success", "dataset_name": "b", "attributions_count": 3},
        ],
    }
    with patch("bioamla.datasets.licenses.generate_licenses_for_directory", return_value=stats):
        result = runner.invoke(cli, ["dataset", "license", str(tmp_path), "--batch"])
    assert result.exit_code == 0, result.output
    assert "Processed 2 dataset(s)" in result.output


def test_dataset_license_batch_none(runner: CliRunner, tmp_path) -> None:
    stats = {"datasets_found": 0, "datasets_processed": 0, "datasets_failed": 0, "results": []}
    with patch("bioamla.datasets.licenses.generate_licenses_for_directory", return_value=stats):
        result = runner.invoke(cli, ["dataset", "license", str(tmp_path), "--batch"])
    assert result.exit_code != 0
    assert "No datasets found" in result.output


# --- augment -------------------------------------------------------------


def test_dataset_augment(runner: CliRunner, tmp_path) -> None:
    stats = {
        "files_created": 20,
        "files_processed": 10,
        "files_failed": 0,
        "output_dir": str(tmp_path / "out"),
    }
    with patch("bioamla.datasets.augmentation.batch_augment", return_value=stats):
        result = runner.invoke(
            cli,
            [
                "dataset",
                "augment",
                str(tmp_path),
                "-o",
                str(tmp_path / "out"),
                "--add-noise",
                "3-30",
                "--quiet",
            ],
        )
    assert result.exit_code == 0, result.output
    assert "Created 20 augmented files" in result.output


def test_dataset_augment_summary_reports_failures(runner: CliRunner, tmp_path) -> None:
    stats = {"files_created": 4, "files_processed": 2, "files_failed": 1, "output_dir": "o"}
    with patch("bioamla.datasets.augmentation.batch_augment", return_value=stats):
        result = runner.invoke(
            cli,
            ["dataset", "augment", str(tmp_path), "-o", str(tmp_path / "out"), "--gain", "-6,6"],
        )
    assert result.exit_code == 0, result.output
    assert result.output.count("Created 4 augmented files") == 1
    assert "1 failed" in result.output


def test_dataset_augment_no_options(runner: CliRunner, tmp_path) -> None:
    result = runner.invoke(cli, ["dataset", "augment", str(tmp_path), "-o", str(tmp_path / "out")])
    assert result.exit_code != 0
    assert "At least one augmentation" in result.output


def test_dataset_augment_ranges(runner: CliRunner, tmp_path) -> None:
    stats = {"files_created": 1, "files_processed": 1, "files_failed": 0, "output_dir": "o"}
    with patch("bioamla.datasets.augmentation.batch_augment", return_value=stats):
        result = runner.invoke(
            cli,
            [
                "dataset",
                "augment",
                str(tmp_path),
                "-o",
                str(tmp_path / "out"),
                "--time-stretch",
                "0.8-1.2",
                "--pitch-shift",
                "-2,2",
                "--gain",
                "-12,12",
                "--quiet",
            ],
        )
    assert result.exit_code == 0, result.output


def test_dataset_augment_error(runner: CliRunner, tmp_path) -> None:
    with patch("bioamla.datasets.augmentation.batch_augment", side_effect=DatasetError("bad")):
        result = runner.invoke(
            cli,
            [
                "dataset",
                "augment",
                str(tmp_path),
                "-o",
                str(tmp_path / "out"),
                "--add-noise",
                "3-30",
            ],
        )
    assert result.exit_code != 0


# --- remix ---------------------------------------------------------------


def _remix_stats(tmp_path) -> dict:
    return {
        "name": "animals",
        "output_dir": str(tmp_path / "out"),
        "metadata_file": str(tmp_path / "out" / "metadata.csv"),
        "label_column": "category",
        "labels": ["cat", "dog"],
        "total_files": 4,
        "files_copied": 4,
        "files_missing": 0,
        "files_duplicate": 0,
        "kept": 2,
        "dropped": 2,
        "added": 0,
        "replaced": 0,
        "license_file": str(tmp_path / "out" / "LICENSE"),
    }


def test_dataset_remix_flags(runner: CliRunner, tmp_path) -> None:
    with patch("bioamla.datasets.remix.remix_dataset", return_value=_remix_stats(tmp_path)) as m:
        result = runner.invoke(
            cli,
            [
                "dataset",
                "remix",
                str(tmp_path / "out"),
                "--source",
                str(tmp_path / "src"),
                "--keep",
                "dog",
                "--keep",
                "cat",
            ],
        )
    assert result.exit_code == 0, result.output
    recipe = m.call_args.args[0]
    assert recipe.keep == ["dog", "cat"] and recipe.drop is None
    assert "Labels (2): cat, dog" in result.output
    assert "Wrote 4 files" in result.output


def test_dataset_remix_recipe_with_overrides(runner: CliRunner, tmp_path) -> None:
    recipe_path = tmp_path / "r.toml"
    recipe_path.write_text('[remix]\nsource = "orig"\nkeep = ["dog"]\n')
    with patch("bioamla.datasets.remix.remix_dataset", return_value=_remix_stats(tmp_path)) as m:
        result = runner.invoke(
            cli,
            [
                "dataset",
                "remix",
                str(tmp_path / "out"),
                "--recipe",
                str(recipe_path),
                "--drop",
                "airplane",
                "--quiet",
            ],
        )
    assert result.exit_code == 0, result.output
    recipe = m.call_args.args[0]
    assert recipe.source == str(tmp_path / "orig")
    assert recipe.drop == ["airplane"] and recipe.keep is None
    assert "Labels" not in result.output


def test_dataset_remix_keep_and_drop_rejected(runner: CliRunner, tmp_path) -> None:
    recipe_path = tmp_path / "r.toml"
    recipe_path.write_text('[remix]\nsource = "orig"\n')
    for args in (["--recipe", str(recipe_path)], ["--source", str(tmp_path)]):
        with patch("bioamla.datasets.remix.remix_dataset") as m:
            result = runner.invoke(
                cli, ["dataset", "remix", str(tmp_path / "out"), *args, "-k", "dog", "-d", "cat"]
            )
        assert result.exit_code != 0
        assert "not both" in result.output
        m.assert_not_called()


def test_dataset_remix_needs_recipe_or_source(runner: CliRunner, tmp_path) -> None:
    result = runner.invoke(cli, ["dataset", "remix", str(tmp_path / "out")])
    assert result.exit_code != 0
    assert "--recipe" in result.output


def test_dataset_remix_error(runner: CliRunner, tmp_path) -> None:
    with patch("bioamla.datasets.remix.remix_dataset", side_effect=InvalidInputError("bad label")):
        result = runner.invoke(
            cli, ["dataset", "remix", str(tmp_path / "out"), "--source", str(tmp_path)]
        )
    assert result.exit_code != 0
    assert "bad label" in result.output


def test_partition_help_describes_the_default_mode():
    from click.testing import CliRunner

    from bioamla.cli.cli import cli

    text = CliRunner().invoke(cli, ["dataset", "partition", "--help"]).output
    assert "moved into train/val/test/<label>/" in text
    assert "--mode column" in text
    assert "writing a `split` column" not in text  # the old, wrong description


def test_train_config_help_names_init_config():
    from click.testing import CliRunner

    from bioamla.cli.cli import cli

    text = " ".join(CliRunner().invoke(cli, ["models", "ast", "train", "--help"]).output.split())
    assert "bioamla models ast init-config" in text
    assert "bioamla config init" not in text
    assert "[augmentation]" in text
