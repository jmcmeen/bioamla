"""The `bioamla --json` contract: envelope, serializer, stdout/stderr separation."""

import json
import math
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path

import numpy as np
import pytest
from click.testing import CliRunner

from bioamla.cli.cli import cli
from bioamla.cli.output import command_from_args, to_jsonable


class _Color(Enum):
    RED = "red"


@dataclass
class _Plain:
    path: Path
    score: float


@dataclass
class _WithToDict:
    x: int

    def to_dict(self):
        return {"x_from_to_dict": self.x}


def test_to_jsonable() -> None:
    value = {
        "path": Path("a/b.wav"),
        "nan": math.nan,
        "inf": float("inf"),
        "np_float": np.float32(0.5),
        "np_int": np.int64(3),
        "array": np.array([[1, 2], [3, 4]]),
        "when": datetime(2026, 10, 31, 12, 0),
        "enum": _Color.RED,
        "plain": _Plain(Path("c.wav"), np.float64(0.25)),
        "custom": _WithToDict(7),
        "tuple": (1, Path("d")),
        3: "int key",
    }
    assert to_jsonable(value) == {
        "path": "a/b.wav",
        "nan": None,
        "inf": None,
        "np_float": 0.5,
        "np_int": 3,
        "array": [[1, 2], [3, 4]],
        "when": "2026-10-31T12:00:00",
        "enum": "red",
        "plain": {"path": "c.wav", "score": 0.25},
        "custom": {"x_from_to_dict": 7},
        "tuple": [1, "d"],
        "3": "int key",
    }
    json.dumps(to_jsonable(value))  # all JSON types


def test_command_from_args() -> None:
    args = ["--log-format", "json", "--json", "detect", "energy", "f.wav", "--threshold", "3"]
    assert command_from_args(cli, args) == "detect energy"
    assert command_from_args(cli, ["--json"]) is None


def test_envelope_on_stdout(tmp_path) -> None:
    ann = tmp_path / "a.csv"
    ann.write_text("start_time,end_time,label\n0.0,1.0,frog\n1.0,2.5,bird\n")
    result = CliRunner().invoke(cli, ["--json", "annotation", "summary", str(ann)])
    assert result.exit_code == 0, result.output
    envelope = json.loads(result.stdout)  # nothing else on stdout
    assert envelope["ok"] is True
    assert envelope["command"] == "annotation summary"
    assert envelope["result"]["total_annotations"] == 2


def test_old_per_command_flag_keeps_bare_json(tmp_path) -> None:
    ann = tmp_path / "a.csv"
    ann.write_text("start_time,end_time,label\n0.0,1.0,frog\n")
    result = CliRunner().invoke(cli, ["annotation", "summary", str(ann), "--json"])
    assert json.loads(result.stdout)["total_annotations"] == 1
    assert "deprecated" in result.stderr


def _run(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-c", "from bioamla.cli.cli import main; main()", *args],
        capture_output=True,
        text=True,
    )


@pytest.mark.parametrize(
    ("args", "error_type", "code"),
    [
        (["--json", "annotation", "summary", "missing.csv"], "BadParameter", 2),
        (["--json", "dataset", "stats", "no_such_dir"], "NotFoundError", 1),
        (["--json", "detect", "energy"], "MissingParameter", 2),
    ],
)
def test_failure_envelope(args, error_type, code) -> None:
    proc = _run(*args)
    assert proc.returncode == code
    envelope = json.loads(proc.stdout)
    assert envelope["ok"] is False
    assert envelope["error"]["type"] == error_type
    assert envelope["error"]["message"]
    assert envelope["command"] == " ".join(a for a in args[1:3])


def test_human_output_goes_to_stderr_and_missing_result_is_null() -> None:
    import click

    @click.command("no-result")
    def no_result() -> None:
        click.echo("human output")
        print("printed too")

    cli.add_command(no_result)
    try:
        result = CliRunner().invoke(cli, ["--json", "no-result"])
    finally:
        cli.commands.pop("no-result")
    assert result.exit_code == 0
    envelope = json.loads(result.stdout)  # only the envelope reached stdout
    assert envelope["result"] is None and envelope["command"] == "no-result"
    assert "human output" in result.stderr and "printed too" in result.stderr


# --- Regressions from review -------------------------------------------------


def test_per_command_flag_through_main_stays_bare(tmp_path) -> None:
    # A --json after the subcommand is the command's own (deprecated) flag.
    ann = tmp_path / "a.csv"
    ann.write_text("start_time,end_time,label\n0.0,1.0,frog\n")
    proc = _run("annotation", "summary", str(ann), "--json")
    assert proc.returncode == 0
    assert json.loads(proc.stdout)["total_annotations"] == 1
    assert "deprecated" in proc.stderr


def test_bare_system_exit_still_gets_an_envelope() -> None:
    import click

    @click.command("bare-exit")
    def bare_exit() -> None:
        click.echo("Error: something", err=True)
        raise SystemExit(1)

    cli.add_command(bare_exit)
    try:
        result = CliRunner().invoke(cli, ["--json", "bare-exit"])
    finally:
        cli.commands.pop("bare-exit")
    assert result.exit_code == 1
    envelope = json.loads(result.stdout)
    assert envelope["ok"] is False and envelope["error"]["type"] == "Exit"


def test_json_mode_is_switched_off_after_each_in_process_run(tmp_path) -> None:
    ann = tmp_path / "a.csv"
    ann.write_text("start_time,end_time,label\n0.0,1.0,frog\n")
    runner = CliRunner()
    first = runner.invoke(cli, ["--json", "annotation", "summary", str(ann)])
    second = runner.invoke(cli, ["--json", "annotation", "summary", str(ann)])
    assert json.loads(first.stdout) == json.loads(second.stdout)
    plain = runner.invoke(cli, ["annotation", "summary", str(ann), "--json"])
    assert json.loads(plain.stdout)["total_annotations"] == 1  # back to non-JSON mode


def test_unexpected_exception_gets_an_envelope(monkeypatch, tmp_path) -> None:
    def boom(*args, **kwargs):
        raise RuntimeError("domain bug")

    monkeypatch.setattr("bioamla.datasets.get_dataset_stats", boom)
    result = CliRunner().invoke(cli, ["--json", "dataset", "stats", str(tmp_path)])
    assert result.exit_code == 1
    envelope = json.loads(result.stdout)
    assert envelope["error"] == {"type": "RuntimeError", "message": "domain bug"}


def test_version_in_json_mode() -> None:
    proc = _run("--json", "--version")
    assert proc.returncode == 0
    assert json.loads(proc.stdout)["result"] == {"version": json.loads(proc.stdout)["bioamla"]}


def test_nested_dataclass_keeps_its_to_dict() -> None:
    @dataclass
    class Outer:
        inner: _WithToDict

    assert to_jsonable(Outer(_WithToDict(2))) == {"inner": {"x_from_to_dict": 2}}
