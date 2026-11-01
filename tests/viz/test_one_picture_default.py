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


@pytest.mark.parametrize("args", [
    ["--no-legend"], [], ["-t", "mfcc", "--no-legend"], ["-t", "stft"], ["-t", "waveform"],
    ["--cmap", "viridis", "--dpi", "80", "--no-legend"],
    ["--n-fft", "1024", "--hop-length", "256", "--n-mels", "64", "--no-legend"],
    ["-t", "mfcc", "--n-mels", "32", "--n-mfcc", "12", "--no-legend"],
])  # fmt: skip
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


def test_n_mels_shapes_the_mfcc_picture(test_audio_path, tmp_path):
    from bioamla.viz import generate_spectrogram

    a = generate_spectrogram(test_audio_path, str(tmp_path / "a.png"), viz_type="mfcc",
                             n_mels=32, n_mfcc=12, show_colorbar=False)  # fmt: skip
    b = generate_spectrogram(test_audio_path, str(tmp_path / "b.png"), viz_type="mfcc",
                             n_mels=128, n_mfcc=12, show_colorbar=False)  # fmt: skip
    assert not np.array_equal(pixels(a), pixels(b))


def test_bad_settings_are_refused_before_any_file_is_drawn(test_audio_path, tmp_path):
    folder = tmp_path / "in"
    folder.mkdir()
    shutil.copy(test_audio_path, folder / "rec.wav")
    run = CliRunner().invoke
    for bad in (["--cmap", "magama"], ["--n-mfcc", "200"]):
        got = run(cli, ["--json", "batch", "audio", "visualize", "--input-dir", str(folder),
                        "--output-dir", str(tmp_path / "out"), "-q", *bad])  # fmt: skip
        assert got.exit_code == 1, (bad, got.output)
        import json

        error = json.loads(got.stdout)["error"]
        assert error["type"] == "InvalidInputError", (bad, error)
        assert not list((tmp_path / "out").rglob("*.png"))
    assert run(cli, ["audio", "visualize", str(folder / "rec.wav"), "--dpi", "0"]).exit_code == 2
