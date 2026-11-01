"""batch.audio.* operations through the registry, and parity with the CLI.

Each parity test runs a `batch audio` command with `--json`, then the same batch with
bioamla's library functions alone (no CLI, no registry), and compares the results and
the files written.
"""

import csv
import json
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
from click.testing import CliRunner

from bioamla import operations
from bioamla.audio import batch as lib
from bioamla.batch import BatchConfig, BatchRequest
from bioamla.cli.cli import cli
from bioamla.exceptions import InvalidInputError


def _cli(*args: str) -> dict:
    result = CliRunner().invoke(cli, ["--json", "batch", "audio", *args])
    assert result.exit_code == 0, result.output
    return json.loads(result.stdout)["result"]


def _audio(path) -> np.ndarray:
    return sf.read(str(path), dtype="float32")[0]


def _same_tree(a: Path, b: Path) -> None:
    files_a = sorted(p.relative_to(a) for p in a.rglob("*.wav"))
    assert files_a and files_a == sorted(p.relative_to(b) for p in b.rglob("*.wav"))
    for rel in files_a:
        np.testing.assert_array_equal(_audio(a / rel), _audio(b / rel))


TRANSFORMS = [
    ("resample", ["-r", "8000"], lambda c: lib.batch_resample(c, 8000)),
    ("normalize", ["-d", "-18"], lambda c: lib.batch_transform(c, lib.normalize_processor(target_db=-18))),
    ("normalize", ["--peak"], lambda c: lib.batch_transform(c, lib.normalize_processor(peak=True))),
    ("trim", ["-s", "0.2", "-e", "0.7"], lambda c: lib.batch_transform(c, lib.trim_processor(start=0.2, end=0.7))),
    ("trim", ["--trim-silence"], lambda c: lib.batch_transform(c, lib.trim_processor(silence_threshold_db=-40))),
    ("filter", ["--lowpass", "1000"], lambda c: lib.batch_transform(c, lib.filter_processor(lowpass=1000))),
    ("denoise", ["--strength", "0.5"], lambda c: lib.batch_transform(c, lib.denoise_processor(strength=0.5))),
    ("convert", ["-f", "wav", "-r", "8000"], lambda c: lib.batch_convert(c, target_format="wav", sample_rate=8000)),
]  # fmt: skip


@pytest.mark.parametrize(("command", "args", "call"), TRANSFORMS)
def test_transform_parity_directory_mode(command, args, call, test_audio_dir, tmp_path):
    got = _cli(command, "--input-dir", test_audio_dir, "--output-dir", str(tmp_path / "cli"), *args)
    want = call(BatchConfig(input_dir=test_audio_dir, output_dir=str(tmp_path / "lib")))
    assert (got["successful"], got["failed"]) == (want.successful, want.failed) == (3, 0)
    _same_tree(tmp_path / "cli", tmp_path / "lib")


@pytest.fixture
def manifest(test_audio_dir, tmp_path) -> Path:
    path = Path(test_audio_dir) / "files.csv"
    names = sorted(p.name for p in Path(test_audio_dir).glob("*.wav"))
    path.write_text("file_name,site\n" + "".join(f"{n},s{i}\n" for i, n in enumerate(names)))
    return path


def _csv(path) -> list[dict]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


@pytest.mark.parametrize(
    ("command", "args", "call"),
    [
        ("info", [], lambda c: lib.batch_audio_info(c)),
        ("metadata", [], lambda c: lib.batch_recording_metadata(c)),
        ("validate", [], lambda c: lib.batch_validate_wavs(c)),
        ("segment", ["-d", "0.4"], lambda c: lib.batch_segment(c, duration=0.4)),
        ("normalize", [], lambda c: lib.batch_transform(c, lib.normalize_processor())),
    ],
)
def test_csv_mode_parity(command, args, call, manifest, tmp_path):
    got = _cli(command, "--input-file", str(manifest), "--output-dir", str(tmp_path / "cli"), *args)
    want = call(BatchConfig(input_file=str(manifest), output_dir=str(tmp_path / "lib")))
    cli_rows, lib_rows = _csv(got["metadata_csv"]), _csv(want.metadata["metadata_csv"])
    assert len(cli_rows) == len(lib_rows) > 0
    assert [sorted(r) for r in cli_rows] == [sorted(r) for r in lib_rows]


def test_info_and_validate_directory_mode(test_audio_dir, tmp_path):
    got = _cli("info", "--input-dir", test_audio_dir, "--output-dir", str(tmp_path / "cli"))
    lib.batch_audio_info(BatchConfig(input_dir=test_audio_dir, output_dir=str(tmp_path / "lib")))
    assert got["results_file"].endswith("audio_info.csv")
    assert len(_csv(tmp_path / "cli" / "audio_info.csv")) == 3

    got = _cli("validate", "--input-dir", test_audio_dir, "--output-dir", str(tmp_path / "v"))
    want = lib.batch_validate_wavs(BatchConfig(input_dir=test_audio_dir))
    assert got["successful"] == want.successful == 3
    assert want.metadata["invalid_files"] == []
    assert Path(got["results_file"]).name == "validation_report.csv"


def test_validate_reports_invalid_files_in_both_modes(tmp_path):
    folder = tmp_path / "in"
    folder.mkdir()
    (folder / "bad.wav").write_bytes(b"RIFF\x00\x00\x00\x00WAVEnope")
    sf.write(folder / "good.wav", np.zeros(800, np.float32), 8000)
    result = lib.batch_validate_wavs(BatchConfig(input_dir=str(folder)))
    assert [Path(p).name for p in result.metadata["invalid_files"]] == ["bad.wav"]
    (folder / "files.csv").write_text("file_name\nbad.wav\ngood.wav\n")
    result = lib.batch_validate_wavs(
        BatchConfig(input_file=str(folder / "files.csv"), output_dir=str(tmp_path / "o"))
    )
    assert [Path(p).name for p in result.metadata["invalid_files"]] == ["bad.wav"]
    run = CliRunner().invoke(cli, ["batch", "audio", "validate", "--input-dir", str(folder)])
    assert run.exit_code == 1


def test_operations_check_every_batch_path(test_audio_dir, manifest, tmp_path):
    seen = []

    def check(path):
        seen.append(path)
        return path

    ctx = operations.OperationContext(path_check=check)
    operations.run(
        "batch.audio.normalize",
        {"input_dir": test_audio_dir, "output_dir": str(tmp_path / "o")},
        ctx,
    )
    assert seen == [test_audio_dir, str(tmp_path / "o")]
    seen.clear()
    operations.run(
        "batch.audio.info", {"input_file": str(manifest), "output_dir": str(tmp_path / "p")}, ctx
    )
    assert seen == [str(manifest), str(tmp_path / "p")]


def test_operations_report_progress(test_audio_dir, tmp_path):
    progress = []
    ctx = operations.OperationContext(on_progress=lambda d, t: progress.append((d, t)))
    operations.run(
        "batch.audio.segment",
        {"input_dir": test_audio_dir, "output_dir": str(tmp_path / "s"), "duration": 0.5},
        ctx,
    )
    assert progress[-1] == (3, 3)
    assert all(operations.get(n).kind == "job" for n in ["batch.audio.info", "batch.audio.trim"])


def test_batch_request_needs_exactly_one_input():
    with pytest.raises(InvalidInputError):
        BatchRequest().to_config()
    with pytest.raises(InvalidInputError):
        BatchRequest(input_dir="a", input_file="b.csv").to_config()


def test_filter_processor_validation():
    with pytest.raises(InvalidInputError, match="Must specify"):
        lib.filter_processor()
    with pytest.raises(InvalidInputError, match="together"):
        lib.filter_processor(bandpass_low=100.0)


def test_directory_mode_needs_an_output_dir(test_audio_dir):
    with pytest.raises(InvalidInputError, match="output-dir"):
        lib.batch_transform(BatchConfig(input_dir=test_audio_dir), lib.normalize_processor())


def test_visualize_is_directory_mode_only(manifest, tmp_path):
    pytest.importorskip("matplotlib")
    with pytest.raises(InvalidInputError, match="input-dir"):
        lib.batch_visualize(BatchConfig(input_file=str(manifest), output_dir=str(tmp_path)))


def test_no_output_dir_writes_no_none_folder(manifest, test_audio_dir, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    config = BatchConfig(input_file=str(manifest), _skip_validation=True)
    config.output_dir = None
    lib.batch_audio_info(config)
    dir_config = BatchConfig(input_dir=test_audio_dir, _skip_validation=True)
    dir_config.output_dir = None
    lib.batch_audio_info(dir_config)
    assert not (tmp_path / "None").exists()


@pytest.mark.parametrize(
    "name", ["batch.audio.info", "batch.audio.metadata", "batch.audio.segment"]
)
def test_directory_mode_honours_continue_on_error(name, tmp_path):
    from bioamla.exceptions import BioamlaError

    folder = tmp_path / "in"
    folder.mkdir()
    for n in ("a.wav", "b.wav"):
        (folder / n).write_bytes(b"RIFF\x00\x00\x00\x00WAVEbroken")
    request = {"input_dir": str(folder), "output_dir": str(tmp_path / "o")}
    if name.endswith("segment"):
        request["duration"] = 1.0
    keep_going = operations.run(name, request)
    assert keep_going.failed == 2
    with pytest.raises(BioamlaError):
        operations.run(name, {**request, "continue_on_error": False})


def test_filter_prints_no_header_for_a_bad_combination(test_audio_dir, tmp_path):
    run = CliRunner().invoke(
        cli,
        ["batch", "audio", "filter", "--input-dir", test_audio_dir, "--output-dir", str(tmp_path)],
    )
    assert run.exit_code != 0
    assert "Applying" not in run.output and "Must specify" in run.output


def test_library_exports_the_batch_functions():
    import bioamla.audio as audio

    for name in ("batch_transform", "batch_convert", "batch_validate_wavs", "filter_processor"):
        assert name in audio.__all__ and callable(getattr(audio, name))
