"""CLI tests for `bioamla annotation` commands."""

from __future__ import annotations

import importlib
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
from click.testing import CliRunner

from bioamla.cli.cli import cli
from bioamla.datasets import Annotation
from bioamla.exceptions import AnnotationError

# The command group `annotation` shadows the submodule attribute on the
# `bioamla.cli.commands` package (the package does `from .annotation import
# annotation`), so attribute-based access (incl. `import ... as`) resolves to
# the Click Group, not the module. Fetch the real module from sys.modules.
ann_mod = importlib.import_module("bioamla.cli.commands.annotation")


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


def _ann(start=0.0, end=2.0, label="frog"):
    return Annotation(start_time=start, end_time=end, label=label)


def test_annotation_group_help(runner: CliRunner) -> None:
    result = runner.invoke(cli, ["annotation", "--help"])
    assert result.exit_code == 0
    for sub in [
        "template",
        "convert",
        "summary",
        "remap",
        "filter",
        "generate-labels",
        "generate-frame-labels",
    ]:
        assert sub in result.output


# --- template ------------------------------------------------------------


def test_annotation_template(runner: CliRunner, test_audio_path, tmp_path) -> None:
    out = tmp_path / "ann.json"
    info = SimpleNamespace(duration=3.0, sample_rate=16000, channels=1)
    with (
        patch("bioamla.audio.get_audio_info", return_value=info),
        patch.object(ann_mod, "_save") as msave,
        patch.object(ann_mod, "_detect_format", return_value="bioamla"),
    ):
        result = runner.invoke(cli, ["annotation", "template", test_audio_path, str(out)])
    assert result.exit_code == 0, result.output
    assert "annotation template" in result.output
    msave.assert_called_once()


def test_annotation_template_empty(runner: CliRunner, test_audio_path, tmp_path) -> None:
    out = tmp_path / "ann.csv"
    info = SimpleNamespace(duration=3.0, sample_rate=16000, channels=1)
    with (
        patch("bioamla.audio.get_audio_info", return_value=info),
        patch.object(ann_mod, "_save"),
        patch.object(ann_mod, "_detect_format", return_value="csv"),
    ):
        result = runner.invoke(
            cli, ["annotation", "template", test_audio_path, str(out), "--empty"]
        )
    assert result.exit_code == 0, result.output
    assert "rows: 0" in result.output


def test_annotation_template_missing_audio(runner: CliRunner, tmp_path) -> None:
    result = runner.invoke(
        cli, ["annotation", "template", str(tmp_path / "no.wav"), str(tmp_path / "out.json")]
    )
    assert result.exit_code != 0
    assert "not found" in result.output


def test_annotation_template_error(runner: CliRunner, test_audio_path, tmp_path) -> None:
    with (
        patch("bioamla.audio.get_audio_info", side_effect=AnnotationError("bad audio")),
        patch.object(ann_mod, "_detect_format", return_value="bioamla"),
    ):
        result = runner.invoke(
            cli, ["annotation", "template", test_audio_path, str(tmp_path / "out.json")]
        )
    assert result.exit_code != 0


# --- convert -------------------------------------------------------------


def test_annotation_convert(runner: CliRunner, tmp_path) -> None:
    src = tmp_path / "in.csv"
    src.write_text("data")
    out = tmp_path / "out.json"
    with (
        patch.object(ann_mod, "_detect_format", side_effect=["csv", "bioamla"]),
        patch.object(ann_mod, "_load", return_value=([_ann()], {})),
        patch.object(ann_mod, "_save") as msave,
    ):
        result = runner.invoke(cli, ["annotation", "convert", str(src), str(out)])
    assert result.exit_code == 0, result.output
    assert "Converted 1 annotations" in result.output
    msave.assert_called_once()


def test_annotation_convert_missing(runner: CliRunner, tmp_path) -> None:
    result = runner.invoke(
        cli, ["annotation", "convert", str(tmp_path / "no.csv"), str(tmp_path / "o.json")]
    )
    assert result.exit_code != 0
    assert "not found" in result.output


def test_annotation_convert_error(runner: CliRunner, tmp_path) -> None:
    src = tmp_path / "in.csv"
    src.write_text("data")
    with (
        patch.object(ann_mod, "_detect_format", side_effect=["csv", "bioamla"]),
        patch.object(ann_mod, "_load", side_effect=AnnotationError("parse error")),
    ):
        result = runner.invoke(cli, ["annotation", "convert", str(src), str(tmp_path / "o.json")])
    assert result.exit_code != 0


# --- summary -------------------------------------------------------------


def _summary():
    return {
        "total_annotations": 2,
        "unique_labels": 1,
        "total_duration": 4.0,
        "min_duration": 2.0,
        "max_duration": 2.0,
        "mean_duration": 2.0,
        "labels": {"frog": 2},
    }


def test_annotation_summary(runner: CliRunner, tmp_path) -> None:
    src = tmp_path / "ann.json"
    src.write_text("{}")
    with (
        patch.object(ann_mod, "_detect_format", return_value="bioamla"),
        patch.object(
            ann_mod,
            "_load",
            return_value=([_ann(), _ann()], {"audio_file": "a.wav", "duration": 10.0}),
        ),
        patch("bioamla.datasets.summarize_annotations", return_value=_summary()),
    ):
        result = runner.invoke(cli, ["annotation", "summary", str(src)])
    assert result.exit_code == 0, result.output
    assert "Total annotations: 2" in result.output


def test_annotation_summary_json(runner: CliRunner, tmp_path) -> None:
    src = tmp_path / "ann.json"
    src.write_text("{}")
    with (
        patch.object(ann_mod, "_detect_format", return_value="bioamla"),
        patch.object(ann_mod, "_load", return_value=([_ann()], {})),
        patch("bioamla.datasets.summarize_annotations", return_value=_summary()),
    ):
        result = runner.invoke(cli, ["annotation", "summary", str(src), "--json"])
    assert result.exit_code == 0
    assert '"total_annotations": 2' in result.output


def test_annotation_summary_missing(runner: CliRunner, tmp_path) -> None:
    result = runner.invoke(cli, ["annotation", "summary", str(tmp_path / "no.json")])
    assert result.exit_code != 0
    assert "not found" in result.output


def test_annotation_summary_error(runner: CliRunner, tmp_path) -> None:
    src = tmp_path / "ann.json"
    src.write_text("{}")
    with (
        patch.object(ann_mod, "_detect_format", return_value="bioamla"),
        patch.object(ann_mod, "_load", side_effect=AnnotationError("bad")),
    ):
        result = runner.invoke(cli, ["annotation", "summary", str(src)])
    assert result.exit_code != 0


# --- remap ---------------------------------------------------------------


def test_annotation_remap(runner: CliRunner, tmp_path) -> None:
    src = tmp_path / "in.csv"
    src.write_text("data")
    mapping = tmp_path / "map.csv"
    mapping.write_text("source,target\nfrog,amphibian\n")
    out = tmp_path / "out.csv"
    with (
        patch.object(ann_mod, "_detect_format", return_value="csv"),
        patch.object(ann_mod, "_load", return_value=([_ann()], {})),
        patch.object(ann_mod, "_save"),
        patch("bioamla.datasets.load_label_mapping", return_value={"frog": "amphibian"}),
        patch("bioamla.datasets.remap_labels", return_value=[_ann(label="amphibian")]),
    ):
        result = runner.invoke(cli, ["annotation", "remap", str(src), str(out), "-m", str(mapping)])
    assert result.exit_code == 0, result.output
    assert "Remapped" in result.output


def test_annotation_remap_missing(runner: CliRunner, tmp_path) -> None:
    mapping = tmp_path / "map.csv"
    mapping.write_text("source,target\n")
    result = runner.invoke(
        cli,
        [
            "annotation",
            "remap",
            str(tmp_path / "no.csv"),
            str(tmp_path / "o.csv"),
            "-m",
            str(mapping),
        ],
    )
    assert result.exit_code != 0
    assert "not found" in result.output


def test_annotation_remap_error(runner: CliRunner, tmp_path) -> None:
    src = tmp_path / "in.csv"
    src.write_text("data")
    mapping = tmp_path / "map.csv"
    mapping.write_text("source,target\n")
    with (
        patch.object(ann_mod, "_detect_format", return_value="csv"),
        patch("bioamla.datasets.load_label_mapping", side_effect=AnnotationError("bad map")),
    ):
        result = runner.invoke(
            cli, ["annotation", "remap", str(src), str(tmp_path / "o.csv"), "-m", str(mapping)]
        )
    assert result.exit_code != 0


# --- filter --------------------------------------------------------------


def test_annotation_filter(runner: CliRunner, tmp_path) -> None:
    src = tmp_path / "in.csv"
    src.write_text("data")
    out = tmp_path / "out.csv"
    anns = [_ann(end=2.0, label="frog"), _ann(end=5.0, label="bird")]
    with (
        patch.object(ann_mod, "_detect_format", return_value="csv"),
        patch.object(ann_mod, "_load", return_value=(anns, {})),
        patch.object(ann_mod, "_save"),
        patch("bioamla.datasets.filter_labels", return_value=anns),
    ):
        result = runner.invoke(
            cli,
            [
                "annotation",
                "filter",
                str(src),
                str(out),
                "-i",
                "frog",
                "--min-duration",
                "1.0",
                "--max-duration",
                "3.0",
            ],
        )
    assert result.exit_code == 0, result.output
    assert "Filtered" in result.output


def test_annotation_filter_missing(runner: CliRunner, tmp_path) -> None:
    result = runner.invoke(
        cli, ["annotation", "filter", str(tmp_path / "no.csv"), str(tmp_path / "o.csv")]
    )
    assert result.exit_code != 0
    assert "not found" in result.output


def test_annotation_filter_error(runner: CliRunner, tmp_path) -> None:
    src = tmp_path / "in.csv"
    src.write_text("data")
    with (
        patch.object(ann_mod, "_detect_format", return_value="csv"),
        patch.object(ann_mod, "_load", side_effect=AnnotationError("bad")),
    ):
        result = runner.invoke(cli, ["annotation", "filter", str(src), str(tmp_path / "o.csv")])
    assert result.exit_code != 0


# --- generate-labels -----------------------------------------------------


def test_annotation_generate_labels_csv(runner: CliRunner, tmp_path) -> None:
    src = tmp_path / "ann.json"
    src.write_text("{}")
    out = tmp_path / "labels.csv"
    with (
        patch.object(ann_mod, "_detect_format", return_value="bioamla"),
        patch.object(
            ann_mod,
            "_load",
            return_value=([_ann()], {"duration": 10.0}),
        ),
        patch("bioamla.datasets.get_unique_labels", return_value=["frog"]),
        patch("bioamla.datasets.create_label_map", return_value={"frog": 0}),
        patch("bioamla.datasets.generate_clip_labels", return_value=np.array([1])),
    ):
        result = runner.invoke(
            cli,
            ["annotation", "generate-labels", str(src), str(out), "--clip-duration", "2.0"],
        )
    assert result.exit_code == 0, result.output
    assert out.exists()
    assert "Generated labels" in result.output


def test_annotation_generate_labels_numpy(runner: CliRunner, tmp_path) -> None:
    src = tmp_path / "ann.json"
    src.write_text("{}")
    out = tmp_path / "labels.npy"
    with (
        patch.object(ann_mod, "_detect_format", return_value="bioamla"),
        patch.object(
            ann_mod,
            "_load",
            return_value=([_ann()], {"duration": 10.0}),
        ),
        patch("bioamla.datasets.get_unique_labels", return_value=["frog"]),
        patch("bioamla.datasets.create_label_map", return_value={"frog": 0}),
        patch("bioamla.datasets.generate_clip_labels", return_value=np.array([1])),
    ):
        result = runner.invoke(
            cli,
            [
                "annotation",
                "generate-labels",
                str(src),
                str(out),
                "--clip-duration",
                "2.0",
                "--format",
                "numpy",
            ],
        )
    assert result.exit_code == 0, result.output
    assert out.exists()


def test_annotation_generate_labels_missing(runner: CliRunner, tmp_path) -> None:
    result = runner.invoke(
        cli,
        [
            "annotation",
            "generate-labels",
            str(tmp_path / "no.json"),
            str(tmp_path / "o.csv"),
            "--clip-duration",
            "2.0",
        ],
    )
    assert result.exit_code != 0
    assert "not found" in result.output


def test_annotation_generate_labels_no_duration(runner: CliRunner, tmp_path) -> None:
    src = tmp_path / "ann.json"
    src.write_text("{}")
    with (
        patch.object(ann_mod, "_detect_format", return_value="bioamla"),
        patch.object(ann_mod, "_load", return_value=([_ann()], {})),
    ):
        result = runner.invoke(
            cli,
            [
                "annotation",
                "generate-labels",
                str(src),
                str(tmp_path / "o.csv"),
                "--clip-duration",
                "2.0",
            ],
        )
    assert result.exit_code != 0
    assert "audio-duration" in result.output


def test_annotation_generate_labels_empty(runner: CliRunner, tmp_path) -> None:
    src = tmp_path / "ann.json"
    src.write_text("{}")
    with (
        patch.object(ann_mod, "_detect_format", return_value="bioamla"),
        patch.object(ann_mod, "_load", return_value=([], {"duration": 10.0})),
    ):
        result = runner.invoke(
            cli,
            [
                "annotation",
                "generate-labels",
                str(src),
                str(tmp_path / "o.csv"),
                "--clip-duration",
                "2.0",
            ],
        )
    assert result.exit_code != 0
    assert "No annotations" in result.output


# --- generate-frame-labels -----------------------------------------------


def test_annotation_generate_frame_labels_csv(runner: CliRunner, tmp_path) -> None:
    src = tmp_path / "ann.json"
    src.write_text("{}")
    out = tmp_path / "frames.csv"
    frame_labels = np.array([[1, 0, 1], [0, 1, 0]])  # (num_classes=2, num_frames=3)
    with (
        patch.object(ann_mod, "_detect_format", return_value="bioamla"),
        patch.object(
            ann_mod,
            "_load",
            return_value=([_ann()], {"duration": 6.0}),
        ),
        patch("bioamla.datasets.get_unique_labels", return_value=["a", "b"]),
        patch("bioamla.datasets.create_label_map", return_value={"a": 0, "b": 1}),
        patch("bioamla.datasets.generate_frame_labels", return_value=frame_labels),
    ):
        result = runner.invoke(
            cli,
            ["annotation", "generate-frame-labels", str(src), str(out), "--frame-size", "2.0"],
        )
    assert result.exit_code == 0, result.output
    assert out.exists()
    assert "Generated frame labels" in result.output


def test_annotation_generate_frame_labels_numpy(runner: CliRunner, tmp_path) -> None:
    src = tmp_path / "ann.json"
    src.write_text("{}")
    out = tmp_path / "frames.npy"
    frame_labels = np.array([[1, 0], [0, 1]])
    with (
        patch.object(ann_mod, "_detect_format", return_value="bioamla"),
        patch.object(
            ann_mod,
            "_load",
            return_value=([_ann()], {"duration": 4.0}),
        ),
        patch("bioamla.datasets.get_unique_labels", return_value=["a", "b"]),
        patch("bioamla.datasets.create_label_map", return_value={"a": 0, "b": 1}),
        patch("bioamla.datasets.generate_frame_labels", return_value=frame_labels),
    ):
        result = runner.invoke(
            cli,
            [
                "annotation",
                "generate-frame-labels",
                str(src),
                str(out),
                "--frame-size",
                "2.0",
                "--format",
                "numpy",
            ],
        )
    assert result.exit_code == 0, result.output
    assert out.exists()


def test_annotation_generate_frame_labels_no_duration(runner: CliRunner, tmp_path) -> None:
    src = tmp_path / "ann.json"
    src.write_text("{}")
    with (
        patch.object(ann_mod, "_detect_format", return_value="bioamla"),
        patch.object(ann_mod, "_load", return_value=([_ann()], {})),
    ):
        result = runner.invoke(
            cli,
            [
                "annotation",
                "generate-frame-labels",
                str(src),
                str(tmp_path / "o.csv"),
                "--frame-size",
                "2.0",
            ],
        )
    assert result.exit_code != 0
    assert "audio-duration" in result.output
