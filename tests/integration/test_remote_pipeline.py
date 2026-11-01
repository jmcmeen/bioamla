"""A pipeline on a server: upload, detect -> annotation summary -> clips -> stats, then
download, each step reading the previous step's result, through the client and through
``bioamla --remote``."""

import pytest
from click.testing import CliRunner

from bioamla.cli.cli import cli
from bioamla.client import Client
from bioamla.workspace import open_workspace
from tests.integration.test_pipelines import bioamla, read_csv

pytestmark = pytest.mark.integration


def test_pipeline_through_the_client(live_server, recording, tmp_path):
    url, token, root = live_server
    lab = Client(url, token)
    lab.create_workspace("field")
    lab.upload("field", recording, "raw/calls.wav")

    detections = lab.run("field", "detect.energy",
                         {"file": "raw/calls.wav", "output": "det.csv"})  # fmt: skip
    assert detections["count"] == 7
    summary = lab.run("field", "annotation.summary", {"file": detections["output"]})
    assert summary["total_annotations"] == 7

    clips = lab.run(
        "field", "dataset.extract-clips",
        {"source": "raw/calls.wav", "output_dir": "clips", "annotations": detections["output"]},
    )  # fmt: skip
    assert clips["clips_written"] == 7
    progress = []
    headers = lab.run("field", "batch.audio.info", {"input_dir": clips["output_dir"]},
                      on_progress=lambda done, total: progress.append((done, total)))  # fmt: skip
    assert headers["successful"] == 7
    assert progress[-1] == (7, 7)  # streamed from the job
    stats = lab.run("field", "dataset.stats", {"dataset_dir": clips["metadata_file"]})
    assert stats["total_files"] == 7

    listed = lab.files("field", "clips", recursive=True)["files"]  # label folders
    wavs = [f["path"] for f in listed if f["path"].endswith(".wav")]
    assert len(wavs) == 7
    got = lab.download("field", wavs[0], tmp_path / "clip.wav")
    assert got.stat().st_size > 0

    history = open_workspace(root / "field").history()
    assert [r["operation"] for r in history] == [
        "detect.energy", "annotation.summary", "dataset.extract-clips", "batch.audio.info",
        "dataset.stats",
    ]  # fmt: skip
    assert all(r["via"] == "server" and r["ok"] for r in history)
    assert history[2]["job"]  # extract-clips ran as a job


def bioamla_remote(*args: str) -> dict:
    return bioamla("--remote", "lab:field", *args)


def test_pipeline_through_the_cli(lab, recording, tmp_path, monkeypatch):
    _, _, root = lab
    here = tmp_path / "laptop"
    here.mkdir()
    monkeypatch.chdir(here)  # nothing below may need a local file but the upload
    runner = CliRunner()
    assert runner.invoke(cli, ["remote", "workspaces", "lab", "--create", "field"]).exit_code == 0
    up = runner.invoke(cli, ["remote", "upload", "lab", "field", str(recording), "--to", "raw"])
    assert up.exit_code == 0, up.output

    # Each step reads the paths the previous one reported, relative to the workspace.
    detections = bioamla_remote("detect", "energy", "raw/calls.wav", "-o", "det.csv")
    assert detections["count"] == 7
    summary = bioamla_remote("annotation", "summary", detections["output"])
    assert summary["total_annotations"] == 7
    clips = bioamla_remote("dataset", "extract-clips", "raw/calls.wav", "clips",
                           "--annotations", detections["output"])  # fmt: skip
    assert clips["clips_written"] == 7
    assert clips["metadata_file"] == "clips/metadata.csv"
    stats = bioamla_remote("dataset", "stats", clips["metadata_file"])
    assert stats["total_files"] == 7

    down = runner.invoke(cli, ["--json", "remote", "download", "lab", "field",
                               clips["metadata_file"], "-o", "got"])  # fmt: skip
    assert down.exit_code == 0, down.output
    rows = read_csv(here / "got" / "metadata.csv")
    assert {r["low_freq"] for r in rows} == {"500.0"}  # the detector's band survives
    assert len(rows) == 7

    history = open_workspace(root / "field").history()
    assert [r["operation"] for r in history] == [
        "detect.energy", "annotation.summary", "dataset.extract-clips", "dataset.stats",
    ]  # fmt: skip
    assert all(r["via"] == "server" and r["ok"] for r in history)
    assert all(r["job"] for r in history if r["operation"] in ("detect.energy",
                                                               "dataset.extract-clips"))  # fmt: skip
    assert not (here / ".bioamla").exists()  # nothing recorded on the laptop
