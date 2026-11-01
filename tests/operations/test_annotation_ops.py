"""annotation.* operations, and parity with the CLI.

Each parity test runs an `annotation` command with `--json`, then does the same with
bioamla's library functions alone (no CLI, no registry), and compares results and files.
"""

import json
from pathlib import Path

import pytest
from click.testing import CliRunner

from bioamla import datasets as lib
from bioamla import operations
from bioamla.cli.cli import cli
from bioamla.exceptions import InvalidInputError


def _cli(*args: str) -> dict:
    result = CliRunner().invoke(cli, ["--json", "annotation", *args])
    assert result.exit_code == 0, result.output
    return json.loads(result.stdout)["result"]


@pytest.fixture
def table(tmp_path) -> Path:
    path = tmp_path / "calls.csv"
    path.write_text(
        "start_time,end_time,label\n0.0,1.0,frog\n1.5,2.0,bird\n2.5,4.0,frog\n4.2,4.4,insect\n"
    )
    return path


def test_template(test_audio_path_3s, tmp_path):
    got = _cli("template", test_audio_path_3s, str(tmp_path / "c.json"), "--label", "x")
    want = lib.annotation_template(test_audio_path_3s, tmp_path / "l.json", label="x")
    assert {k: v for k, v in got.items() if k != "output"} == {
        k: v for k, v in want.to_dict().items() if k != "output"
    }
    a = json.loads((tmp_path / "c.json").read_text())
    b = json.loads((tmp_path / "l.json").read_text())
    a.pop("created"), b.pop("created")  # a timestamp
    assert a == b


@pytest.mark.parametrize("to", ["c.txt", "c.json"])
def test_convert(table, tmp_path, to):
    got = _cli("convert", str(table), str(tmp_path / to))
    want = lib.convert_annotation_file(table, tmp_path / ("l" + Path(to).suffix))
    assert got["count"] == want.count == 4
    assert got["to_format"] == want.to_format
    cli_text = (tmp_path / to).read_text()
    assert cli_text == (tmp_path / ("l" + Path(to).suffix)).read_text()


def test_summary(table):
    assert _cli("summary", str(table)) == lib.summarize_annotation_file(table).to_dict()


def test_remap(table, tmp_path):
    mapping = tmp_path / "map.csv"
    mapping.write_text("source,target\nfrog,anuran\n")
    got = _cli("remap", str(table), str(tmp_path / "c.csv"), "-m", str(mapping), "--drop-unmapped")
    want = lib.remap_annotation_file(table, tmp_path / "l.csv", mapping, keep_unmapped=False)
    assert (
        (got["input_count"], got["output_count"]) == (want.input_count, want.output_count) == (4, 2)
    )
    assert (tmp_path / "c.csv").read_text() == (tmp_path / "l.csv").read_text()


def test_filter(table, tmp_path):
    got = _cli(
        "filter", str(table), str(tmp_path / "c.csv"), "-e", "insect", "--min-duration", "0.6"
    )
    want = lib.filter_annotation_file(
        table, tmp_path / "l.csv", exclude=["insect"], min_duration=0.6
    )
    assert got["output_count"] == want.output_count == 2
    assert (tmp_path / "c.csv").read_text() == (tmp_path / "l.csv").read_text()


@pytest.mark.parametrize("fmt", ["csv", "numpy"])
def test_generate_labels(table, tmp_path, fmt):
    args = ["--clip-duration", "1", "--audio-duration", "5", "--format", fmt]
    got = _cli("generate-labels", str(table), str(tmp_path / "c.csv"), *args)
    want = lib.clip_labels_file(
        table, tmp_path / "l.csv", clip_duration=1.0, audio_duration=5.0, output_format=fmt
    )
    assert got["clips"] == want.clips == 5
    assert got["labels"] == sorted(want.labels)
    assert Path(got["output"]).read_bytes() == Path(want.output).read_bytes()


def test_generate_frame_labels(table, tmp_path):
    args = ["--frame-size", "0.5", "--audio-duration", "5"]
    got = _cli("generate-frame-labels", str(table), str(tmp_path / "c.csv"), *args)
    want = lib.frame_labels_file(table, tmp_path / "l.csv", frame_size=0.5, audio_duration=5.0)
    assert (got["classes"], got["frames"]) == (want.classes, want.frames)
    assert (tmp_path / "c.csv").read_text() == (tmp_path / "l.csv").read_text()


def test_labels_need_a_duration_and_one_recording(table, tmp_path):
    with pytest.raises(InvalidInputError, match="audio_duration"):
        lib.clip_labels_file(table, tmp_path / "x.csv", clip_duration=1.0)
    with pytest.raises(InvalidInputError, match="longer than the audio"):
        lib.clip_labels_file(table, tmp_path / "x.csv", clip_duration=9.0, audio_duration=5.0)
    many = tmp_path / "many.csv"
    many.write_text("start_time,end_time,label,recording\n0,1,a,r1.wav\n0,1,b,r2.wav\n")
    with pytest.raises(InvalidInputError, match="covers 2 recordings"):
        lib.frame_labels_file(many, tmp_path / "y.csv", frame_size=0.5, audio_duration=2.0)
    one = lib.frame_labels_file(
        many, tmp_path / "y.csv", frame_size=0.5, audio_duration=2.0, recording="r2.wav"
    )
    assert one.labels == ["b"]


def test_convert_recording_needs_arbimon(table, tmp_path):
    from bioamla.exceptions import OptionConflictError

    with pytest.raises(OptionConflictError, match="arbimon"):
        lib.convert_annotation_file(table, tmp_path / "x.txt", recording="r.wav")
    run = CliRunner().invoke(
        cli, ["annotation", "convert", str(table), str(tmp_path / "x.txt"), "--recording", "r"]
    )
    assert run.exit_code == 2  # still a usage error, with --to inferred from the name


def test_operations_check_every_path(table, tmp_path):
    mapping = tmp_path / "map.csv"
    mapping.write_text("source,target\nfrog,anuran\n")
    seen = []

    def check(path):
        seen.append(Path(path).name)
        return path

    ctx = operations.OperationContext(path_check=check)
    operations.run(
        "annotation.remap",
        {"input": str(table), "output": str(tmp_path / "o.csv"), "mapping": str(mapping)},
        ctx,
    )
    assert sorted(seen) == ["calls.csv", "map.csv", "o.csv"]
    for name in ("annotation.summary", "annotation.convert", "annotation.filter"):
        assert operations.get(name).kind == "quick"
    for name in ("annotation.generate-labels", "annotation.generate-frame-labels"):
        assert operations.get(name).kind == "job"  # grows with duration / step


def test_frame_labels_keep_the_last_frame_and_reject_long_frames(table, tmp_path):
    three = lib.frame_labels_file(table, tmp_path / "f.csv", frame_size=0.1, audio_duration=0.3)
    assert three.frames == 3  # (0.3 - 0.1) / 0.1 is 1.9999999999999998 in floats
    with pytest.raises(InvalidInputError, match="longer than the audio"):
        lib.frame_labels_file(table, tmp_path / "g.csv", frame_size=9.0, audio_duration=5.0)
    with pytest.raises(InvalidInputError, match="positive"):
        lib.frame_labels_file(table, tmp_path / "h.csv", frame_size=0.0, audio_duration=5.0)


def test_numpy_output_reports_the_file_written(table, tmp_path):
    run = CliRunner().invoke(
        cli,
        ["annotation", "generate-labels", str(table), str(tmp_path / "labels.csv"),
         "--clip-duration", "1", "--audio-duration", "5", "--format", "numpy"],
    )  # fmt: skip
    assert run.exit_code == 0, run.output
    assert f"Output: {tmp_path / 'labels.npy'}" in run.output  # a .csv name becomes .npy
    assert f"Label map: {tmp_path / 'labels.labels.csv'}" in run.output
