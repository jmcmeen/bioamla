"""`audio visualize` and `batch audio visualize` draw the same picture of a file."""

import shutil
from dataclasses import fields

import numpy as np
import pytest
from click.testing import CliRunner

from bioamla.cli.cli import cli

pytest.importorskip("matplotlib")
pytest.importorskip("librosa")


def pixels(path):
    from PIL import Image

    return np.asarray(Image.open(path).convert("RGB"))


@pytest.mark.parametrize("args", [["--no-legend"], [], ["-t", "mfcc", "--no-legend"],
                                  ["--cmap", "viridis", "--dpi", "80", "--no-legend"]])  # fmt: skip
def test_same_picture_from_both_commands(test_audio_path, tmp_path, args):
    folder = tmp_path / "in"
    folder.mkdir()
    shutil.copy(test_audio_path, folder / "rec.wav")
    run = CliRunner().invoke
    single = run(cli, ["audio", "visualize", str(folder / "rec.wav"), "-o",
                       str(tmp_path / "one.png"), *args])  # fmt: skip
    assert single.exit_code == 0, single.output
    batch_args = ["--plot-type" if a == "-t" else a for a in args]
    batch = run(cli, ["batch", "audio", "visualize", "--input-dir", str(folder), "--output-dir",
                      str(tmp_path / "many"), "-q", *batch_args])  # fmt: skip
    assert batch.exit_code == 0, batch.output
    (made,) = (tmp_path / "many").rglob("*.png")
    one, many = pixels(tmp_path / "one.png"), pixels(made)
    assert one.shape == many.shape
    assert np.array_equal(one, many)


def test_the_requests_share_their_defaults():
    from bioamla.audio.batch_operations import BatchVisualizeRequest
    from bioamla.audio.operations import AudioVisualizeRequest

    single = {f.name: f.default for f in fields(AudioVisualizeRequest)}
    batch = {f.name: f.default for f in fields(BatchVisualizeRequest)}
    for name in ("n_fft", "hop_length", "n_mels", "n_mfcc", "cmap", "dpi", "legend"):
        assert single[name] == batch[name], name
