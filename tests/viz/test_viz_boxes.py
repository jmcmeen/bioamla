"""Boxes outlined on spectrograms (generate_spectrogram(boxes=...), annotation_boxes)."""

import matplotlib

matplotlib.use("Agg")

import numpy as np  # noqa: E402
import pytest  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402

from bioamla.datasets import Annotation  # noqa: E402
from bioamla.exceptions import InvalidInputError, NotFoundError  # noqa: E402
from bioamla.viz import SpectrogramBox, annotation_boxes, generate_spectrogram  # noqa: E402
from bioamla.viz.core import _box_color, _BoxGeometry, _draw_boxes  # noqa: E402


def test_box_from_tuples_annotations_and_boxes():
    assert SpectrogramBox.of((1, 2)) == SpectrogramBox(1, 2)
    assert SpectrogramBox.of([1, 2, 100, 200, "frog"]).label == "frog"
    ann = Annotation(start_time=0.5, end_time=1.5, low_freq=500, high_freq=900, label="toad")
    assert SpectrogramBox.of(ann) == SpectrogramBox(0.5, 1.5, 500, 900, "toad")
    box = SpectrogramBox(0, 1)
    assert SpectrogramBox.of(box) is box and not box.banded


@pytest.mark.parametrize(
    "bad", [(2, 1), (1, 2, 100), (1, 2, 300, 200), (1, 2, 100, None), "nope", (1, 2, 3)]
)
def test_bad_boxes_are_refused(bad):
    with pytest.raises(InvalidInputError):
        SpectrogramBox.of(bad)


def test_draw_boxes_in_seconds_and_hz_with_a_legend():
    ax = Figure().subplots()
    ax.set_xlim(0, 3)
    ax.set_ylim(0, 8000)
    geometry = _BoxGeometry("stft", 16000, 512, 2048, 128, legend=True)
    _draw_boxes(ax, [SpectrogramBox(0.5, 1, 1000, 3000, "frog"), SpectrogramBox(2, 10),
                     SpectrogramBox(1, 2, 9000, 12000, "above")], geometry)  # fmt: skip
    band, full = ax.patches  # the box above Nyquist is left out, label and all
    assert (band.get_x(), band.get_y(), band.get_width(), band.get_height()) == (
        0.5,
        1000,
        0.5,
        2000,
    )
    assert full.get_transform() is not ax.transData and full.get_height() == 1  # axes height
    assert [t.get_text() for t in ax.texts] == [" frog"]
    assert ax.get_xlim() == (0, 3)  # a box past the end doesn't rescale the picture


def test_without_a_legend_boxes_map_to_frames_and_bins():
    stft = _BoxGeometry("stft", 16000, 512, 2048, 128, legend=False)
    assert stft.x(1.0) == 16000 / 512 - 0.5
    assert stft.y(4000) == 512 and stft.y(20000) == 1024  # clamped at Nyquist
    mel = _BoxGeometry("mel", 16000, 512, 2048, 128, legend=False)
    assert 0 < mel.y(1000) < mel.y(4000) < 127 and mel.y(8000) == 127
    wave = _BoxGeometry("waveform", 16000, 512, 2048, 128, legend=False)
    assert wave.x(1.0) == 1.0 and not wave.banded


def test_box_colour_shows_on_the_colormap():
    assert _box_color("mel", "magma") == "white"
    assert _box_color("mel", "gray_r") == "black"
    assert _box_color("waveform", "magma") == "tab:red"


@pytest.mark.parametrize("viz_type", ["mel", "stft", "mfcc", "waveform"])
@pytest.mark.parametrize("legend", [True, False])
def test_boxes_change_the_picture(test_audio_path_3s, tmp_path, viz_type, legend):
    from PIL import Image

    plain = generate_spectrogram(test_audio_path_3s, str(tmp_path / "plain.png"),
                                 viz_type=viz_type, show_colorbar=legend)  # fmt: skip
    boxed = generate_spectrogram(test_audio_path_3s, str(tmp_path / "boxed.png"),
                                 viz_type=viz_type, show_colorbar=legend,
                                 boxes=[(0.5, 1.0, 1000, 3000, "frog"), (2.0, 2.5)])  # fmt: skip
    a, b = np.asarray(Image.open(plain)), np.asarray(Image.open(boxed))
    assert a.shape == b.shape  # same layout, with or without boxes
    changed = (a != b).any(axis=-1)
    assert changed.any()
    # Only the outlines (and labels) differ: the picture underneath is the same.
    assert changed.mean() < 0.05


def test_annotation_boxes_from_any_format(tmp_path):
    csv = tmp_path / "det.csv"
    csv.write_text("start_time,end_time,low_freq,high_freq,label\n0.5,1.0,500,5000,call\n"
                   "2.0,2.4,,,\n")  # fmt: skip
    assert annotation_boxes(csv) == [SpectrogramBox(0.5, 1.0, 500, 5000, "call"),
                                     SpectrogramBox(2.0, 2.4)]  # fmt: skip
    raven = tmp_path / "sel.txt"
    raven.write_text("Selection\tView\tChannel\tBegin Time (s)\tEnd Time (s)\tLow Freq (Hz)\t"
                     "High Freq (Hz)\tAnnotation\n1\tSpectrogram 1\t1\t0.1\t0.4\t200\t800\tfrog\n")  # fmt: skip
    assert annotation_boxes(raven) == [SpectrogramBox(0.1, 0.4, 200, 800, "frog")]
    with pytest.raises(NotFoundError):
        annotation_boxes(tmp_path / "missing.csv")


def test_annotation_boxes_for_one_recording_skip_bad_rows(tmp_path, caplog):
    csv = tmp_path / "export.csv"
    csv.write_text("recording,start_time,end_time,low_freq,high_freq,label\n"
                   "rec_01.wav,0.5,1.0,500,5000,a\nrec_02.wav,1,2,500,5000,b\n"
                   "rec_01.wav,3,3,,,point\nrec_01.wav,4,5,700,,half\n")  # fmt: skip
    assert annotation_boxes(csv, recording="/data/REC_01.wav") == [
        SpectrogramBox(0.5, 1.0, 500, 5000, "a")
    ]
    assert "skipping annotation" in caplog.text
    assert len(annotation_boxes(csv)) == 2  # every recording's valid rows
