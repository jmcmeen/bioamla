"""A pipeline on a server: upload, detect -> annotation summary -> clips -> stats, then
download, each step reading the previous step's result, through the client and through
``bioamla --remote``."""

import json

import pytest
from click.testing import CliRunner

from bioamla.cli.cli import cli
from bioamla.client import Client
from bioamla.workspace import open_workspace
from tests.integration.test_pipelines import read_csv, recording  # noqa: F401 - fixture

pytestmark = pytest.mark.integration


@pytest.fixture(autouse=True)
def _needs_audio_extra() -> None:
    pytest.importorskip("librosa")


def test_pipeline_through_the_client(live_server, recording, tmp_path):  # noqa: F811
    url, token, root = live_server
    lab = Client(url, token)
    lab.create_workspace("field")
    lab.upload("field", recording, "raw/calls.wav")

    detections = lab.run("field", "detect.energy",
                         {"file": "raw/calls.wav", "output": "det.csv"})  # fmt: skip
    assert detections["count"] == 7
    summary = lab.run("field", "annotation.summary", {"file": detections["output"]})
    assert summary["total_annotations"] == 7

    progress = []
    clips = lab.run(
        "field", "dataset.extract-clips",
        {"source": "raw/calls.wav", "output_dir": "clips", "annotations": detections["output"]},
        on_progress=lambda done, total: progress.append((done, total)),
    )  # fmt: skip
    assert clips["clips_written"] == 7
    stats = lab.run("field", "dataset.stats", {"dataset_dir": clips["metadata_file"]})
    assert stats["total_files"] == 7

    listed = lab.files("field", "clips", recursive=True)["files"]  # label folders
    wavs = [f["path"] for f in listed if f["path"].endswith(".wav")]
    assert len(wavs) == 7
    got = lab.download("field", wavs[0], tmp_path / "clip.wav")
    assert got.stat().st_size > 0

    history = open_workspace(root / "field").history()
    assert [r["operation"] for r in history] == [
        "detect.energy", "annotation.summary", "dataset.extract-clips", "dataset.stats",
    ]  # fmt: skip
    assert all(r["via"] == "server" and r["ok"] for r in history)
    assert history[2]["job"]  # extract-clips ran as a job


def bioamla_remote(*args: str) -> dict:
    run = CliRunner().invoke(cli, ["--json", "--remote", "lab:field", *args])
    envelope = json.loads(run.stdout)
    assert envelope["ok"], envelope.get("error")
    return envelope["result"]


def test_pipeline_through_the_cli(live_server, recording, tmp_path, monkeypatch):  # noqa: F811
    url, token, root = live_server
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))
    monkeypatch.setenv("BIOAMLA_LAB_TOKEN", token)
    monkeypatch.delenv("BIOAMLA_REMOTE", raising=False)
    runner = CliRunner()
    assert runner.invoke(cli, ["remote", "add", "lab", url]).exit_code == 0
    assert runner.invoke(cli, ["remote", "workspaces", "lab", "--create", "field"]).exit_code == 0
    up = runner.invoke(cli, ["remote", "upload", "lab", "field", str(recording), "--to", "raw"])
    assert up.exit_code == 0, up.output

    detections = bioamla_remote("detect", "energy", "raw/calls.wav", "-o", "det.csv")
    assert detections["count"] == 7
    assert detections["output"] == "det.csv"  # as given, like a local run: relative to the
    assert (root / "field" / "det.csv").is_file()  # remote workspace
    summary = bioamla_remote("annotation", "summary", detections["output"])
    assert summary["total_annotations"] == 7
    clips = bioamla_remote("dataset", "extract-clips", "raw/calls.wav", "clips",
                           "--annotations", detections["output"])  # fmt: skip
    assert clips["clips_written"] == 7
    rows = read_csv(clips["metadata_file"])  # same machine in the test: read it directly
    assert {r["low_freq"] for r in rows} == {"500.0"}  # the detector's band survives
    stats = bioamla_remote("dataset", "stats", clips["metadata_file"])
    assert stats["total_files"] == 7

    out = tmp_path / "local"
    down = runner.invoke(cli, ["--json", "remote", "download", "lab", "field",
                               "clips/metadata.csv", "-o", str(out)])  # fmt: skip
    assert down.exit_code == 0, down.output
    assert read_csv(out / "metadata.csv") == rows
