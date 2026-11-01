"""Run history: each command run inside a workspace is recorded in history.jsonl."""

import json

import pytest
from click.testing import CliRunner

from bioamla.cli.cli import cli
from bioamla.cli.history import redact, should_record


def run(*args: str):
    return CliRunner().invoke(cli, list(args))


@pytest.fixture
def ws(tmp_path, monkeypatch):
    from bioamla.workspace import create_workspace

    ws = create_workspace(tmp_path / "ws")
    monkeypatch.chdir(ws.root)
    return ws


def records(ws) -> list[dict]:
    return ws.history()


def test_runs_inside_a_workspace_are_recorded(ws) -> None:
    run("system", "version")
    (record,) = records(ws)
    assert record["v"] == 1 and record["command"] == "system version"
    assert record["argv"] == ["system", "version"]
    assert record["ok"] is True and record["exit_code"] == 0 and record["error"] is None
    assert record["cwd"] == "." and record["host"] and "bioamla" in record


def test_cwd_is_relative_to_the_workspace(ws, monkeypatch) -> None:
    (ws.root / "sub").mkdir()
    monkeypatch.chdir(ws.root / "sub")
    run("system", "version")
    assert records(ws)[-1]["cwd"] == "sub"


def test_nothing_recorded_outside_a_workspace(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    run("system", "version")
    assert not list(tmp_path.rglob("history.jsonl"))


def test_init_records_itself(tmp_path, monkeypatch) -> None:
    from bioamla.workspace import open_workspace

    monkeypatch.chdir(tmp_path)
    run("workspace", "init", ".")
    assert [r["command"] for r in open_workspace(tmp_path).history()] == ["workspace init"]


@pytest.mark.parametrize(
    "args",
    [["--help"], ["detect", "--help"], ["--version"], [],
     ["workspace", "history"], ["workspace", "status"], ["workspace", "list"]],
)  # fmt: skip
def test_not_recorded(ws, args) -> None:
    run(*args)
    assert records(ws) == []


def test_failures_are_recorded(ws) -> None:
    run("--json", "workspace", "remove", "nope")
    (record,) = records(ws)
    assert record["ok"] is False and record["exit_code"] == 1
    assert record["error"]["type"] and "No artifact" in record["error"]["message"]


def test_outputs_come_from_the_result(ws) -> None:
    (ws.root / "a.wav").write_bytes(b"RIFF")
    run("workspace", "import", "a.wav")
    assert records(ws)[-1]["outputs"] == []  # import reports artifacts, not output files
    assert ws.output_paths({"output": ws.root / "d.csv", "output_files": ["/elsewhere/x"]}) == [
        "d.csv",
        "/elsewhere/x",
    ]


def test_history_command(ws) -> None:
    run("system", "version")
    run("workspace", "remove", "nope")
    run("system", "schema")
    result = json.loads(run("--json", "workspace", "history").stdout)["result"]
    assert [r["command"] for r in result["runs"]] == [
        "system version", "workspace remove", "system schema"
    ]  # fmt: skip
    failed = json.loads(run("--json", "workspace", "history", "--failed").stdout)["result"]
    assert [r["command"] for r in failed["runs"]] == ["workspace remove"]
    last = json.loads(run("--json", "workspace", "history", "-n", "1").stdout)["result"]
    assert [r["command"] for r in last["runs"]] == ["system schema"]
    assert "system version" in run("workspace", "history").output


def test_bad_lines_are_skipped(ws) -> None:
    run("system", "version")
    with open(ws.history_path, "a") as f:
        f.write("{half a record\n[1, 2]\n")
    run("system", "schema")
    assert [r["command"] for r in ws.history()] == ["system version", "system schema"]


@pytest.mark.parametrize(
    ("argv", "expected"),
    [
        (["catalogs", "x", "--api-key", "abc", "-n", "3"],
         ["catalogs", "x", "--api-key", "***", "-n", "3"]),
        (["x", "--token=abc"], ["x", "--token=***"]),
        (["x", "--hf-password", "p", "--output", "o"], ["x", "--hf-password", "***", "--output", "o"]),
        (["x", "--keep", "1"], ["x", "--keep", "1"]),  # not a secret
    ],
)  # fmt: skip
def test_redact(argv, expected) -> None:
    assert redact(argv) == expected


def test_should_record() -> None:
    assert should_record(cli, ["detect", "energy", "a.wav"])
    assert not should_record(cli, ["detect", "energy", "--help"])
    assert not should_record(cli, ["workspace", "history"])
    assert not should_record(cli, ["workspace", "-C", "x", "status"])
    assert not should_record(cli, ["--json"])


def test_dash_c_and_init_from_outside_record_in_that_workspace(tmp_path, monkeypatch) -> None:
    from bioamla.workspace import open_workspace

    monkeypatch.chdir(tmp_path)
    run("workspace", "init", "survey", "--name", "Survey")
    run("workspace", "-C", "survey", "remove", "nope")
    commands = [r["command"] for r in open_workspace(tmp_path / "survey").history()]
    assert commands == ["workspace init", "workspace remove"]


def test_failures_without_json_keep_their_error(ws) -> None:
    run("workspace", "remove", "nope")  # standalone, human output
    (record,) = records(ws)
    assert record["ok"] is False and record["exit_code"] == 1
    assert "No artifact" in record["error"]["message"]


def test_dash_h_is_high_freq_not_help() -> None:
    assert should_record(cli, ["detect", "energy", "a.wav", "-h", "8000"])


def test_attached_dash_c(tmp_path, monkeypatch) -> None:
    from bioamla.workspace import create_workspace

    ws = create_workspace(tmp_path / "survey")
    monkeypatch.chdir(tmp_path)
    run("workspace", "-Csurvey", "remove", "nope")
    assert [r["command"] for r in ws.history()] == ["workspace remove"]


def test_flags_never_hide_the_next_argument() -> None:
    import click

    @click.group()
    def root() -> None: ...

    @root.command()
    @click.option("--save-token", is_flag=True)
    @click.option("--api-key")
    @click.argument("path")
    def cmd(save_token, api_key, path) -> None: ...

    argv = ["cmd", "--save-token", "data.csv", "--api-key", "-abc123"]
    assert redact(argv, root) == ["cmd", "--save-token", "data.csv", "--api-key", "***"]


def test_odd_records_still_show(ws) -> None:
    ws.history_path.write_text('{"argv": [1, "x"], "duration": null, "ok": true}\n')
    result = run("workspace", "history")
    assert result.exit_code == 0 and "1 x" in result.output


def test_bool_exit_codes_are_numbers() -> None:
    from bioamla.cli.cli import _describe_exit

    assert _describe_exit(SystemExit(True)) == (1, None)
    assert _describe_exit(SystemExit(None)) == (0, None)
    assert _describe_exit(SystemExit(3)) == (3, None)


def test_recording_problems_never_fail_the_command(ws, monkeypatch) -> None:
    def boom(*a, **k):
        raise RuntimeError("disk on fire")

    monkeypatch.setattr("bioamla.cli.history.record_run", boom)
    result = run("system", "version")
    assert result.exit_code == 0


def test_status_shows_the_last_runs(ws) -> None:
    first = run("workspace", "status")
    assert "Last runs" in first.output and "none" in first.output

    for _ in range(6):
        run("system", "version")
    run("workspace", "config", "get", "nope.x")  # fails
    result = CliRunner().invoke(cli, ["--json", "workspace", "status"])
    last = json.loads(result.stdout)["result"]["last_runs"]
    assert len(last) == 5
    assert last == records(ws)[-5:]  # status itself isn't recorded
    assert last[-1]["ok"] is False and last[0]["command"] == "system version"

    human = run("workspace", "status").output
    assert "Last runs" in human and "workspace config get nope.x" in human
    assert "exit 1" in human
