"""Trim, resample and gain keep a file's channels (single files and batches)."""

import numpy as np
import pytest
import soundfile as sf
from click.testing import CliRunner

from bioamla.cli.cli import cli

pytest.importorskip("librosa")

SR = 16000


@pytest.fixture
def stereo(tmp_path):
    """Two seconds: a tone on the left, silence on the right."""
    t = np.arange(SR * 2) / SR
    left = 0.5 * np.sin(2 * np.pi * 440 * t)
    path = tmp_path / "in" / "stereo.wav"
    path.parent.mkdir()
    sf.write(path, np.stack([left, np.zeros_like(left)], axis=1), SR)
    return path


def check_stereo(path, *, rate=SR, frames=None):
    data, sr = sf.read(path, always_2d=True)
    assert data.shape[1] == 2, f"{path.name} has {data.shape[1]} channel(s)"
    assert sr == rate
    if frames is not None:
        assert data.shape[0] == frames
    assert np.abs(data[:, 0]).max() > 0.1  # the tone stays left
    assert np.abs(data[:, 1]).max() < 1e-3  # and the right stays silent


def run(*args):
    result = CliRunner().invoke(cli, list(args))
    assert result.exit_code == 0, result.output
    return result


def test_library_keeps_channels(stereo, tmp_path):
    from bioamla.audio import gain_audio_file, resample_audio_file, trim_audio_file

    trim_audio_file(stereo, tmp_path / "t.wav", start=0.5, duration=1.0)
    check_stereo(tmp_path / "t.wav", frames=SR)
    resample_audio_file(stereo, tmp_path / "r.wav", 8000)
    check_stereo(tmp_path / "r.wav", rate=8000, frames=8000 * 2)
    gain_audio_file(stereo, tmp_path / "g.wav", -6.0)
    check_stereo(tmp_path / "g.wav")
    left = sf.read(tmp_path / "g.wav")[0][:, 0]
    assert np.abs(left).max() == pytest.approx(0.5 * 10 ** (-6 / 20), rel=1e-2)


def test_commands_keep_channels(stereo, tmp_path):
    run("audio", "trim", str(stereo), str(tmp_path / "t.wav"), "-s", "0.5", "-e", "1.5")
    check_stereo(tmp_path / "t.wav", frames=SR)
    run("audio", "resample", str(stereo), str(tmp_path / "r.wav"), "-r", "22050")
    check_stereo(tmp_path / "r.wav", rate=22050)
    run("audio", "gain", str(stereo), str(tmp_path / "g.wav"), "--gain-db", "3")
    check_stereo(tmp_path / "g.wav")


def test_batch_commands_keep_channels(stereo, tmp_path):
    folder = stereo.parent
    run("batch", "audio", "resample", "--input-dir", str(folder), "--output-dir",
        str(tmp_path / "r"), "-r", "8000", "-q")  # fmt: skip
    check_stereo(tmp_path / "r" / "stereo.wav", rate=8000)
    run("batch", "audio", "trim", "--input-dir", str(folder), "--output-dir",
        str(tmp_path / "t"), "-s", "0", "-e", "1", "-q")  # fmt: skip
    check_stereo(tmp_path / "t" / "stereo.wav", frames=SR)
    # CSV mode too.
    manifest = folder / "files.csv"
    manifest.write_text("file_name\nstereo.wav\n")
    run("batch", "audio", "resample", "--input-file", str(manifest), "--output-dir",
        str(tmp_path / "csv"), "-r", "8000", "-q")  # fmt: skip
    check_stereo(tmp_path / "csv" / "stereo.wav", rate=8000)


def test_silence_trimming_still_mixes_down(stereo, tmp_path):
    run("batch", "audio", "trim", "--input-dir", str(stereo.parent), "--output-dir",
        str(tmp_path / "s"), "--trim-silence", "-q")  # fmt: skip
    assert sf.info(tmp_path / "s" / "stereo.wav").channels == 1


def test_each_channel():
    from bioamla.audio import each_channel

    mono = np.arange(4.0)
    assert np.array_equal(each_channel(mono, lambda c: c * 2), mono * 2)
    two = np.stack([mono, -mono], axis=1)
    assert np.array_equal(each_channel(two, lambda c: c[:2]), two[:2])


def test_channels_of_different_lengths_are_a_processing_error():
    from bioamla.audio import each_channel
    from bioamla.exceptions import ProcessingError

    two = np.stack([np.zeros(10), np.ones(10)], axis=1)
    with pytest.raises(ProcessingError):
        each_channel(two, lambda c: c[:5] if c[0] == 0 else c)  # trims only the first


def test_batch_convert_refuses_a_bad_channel_count_before_writing(stereo, tmp_path):
    from bioamla.audio import batch_convert_files
    from bioamla.exceptions import InvalidInputError

    with pytest.raises(InvalidInputError, match="channel count"):
        batch_convert_files(str(stereo.parent), str(tmp_path / "out"), channels=3)
    assert not (tmp_path / "out").exists()


def test_mp3_holds_at_most_two_channels(tmp_path):
    from bioamla.audio import save_audio
    from bioamla.exceptions import InvalidInputError

    with pytest.raises(InvalidInputError, match="MP3 holds at most 2"):
        save_audio(str(tmp_path / "x.mp3"), np.zeros((100, 3), dtype=np.float32), SR)


@pytest.mark.parametrize(("subtype", "fmt", "want"), [
    ("FLOAT", "wav", "PCM_32"), ("PCM_32", "flac", "PCM_24"), ("PCM_24", "wav", "PCM_24"),
])  # fmt: skip
def test_convert_keeps_the_most_bits_the_output_holds(tmp_path, subtype, fmt, want):
    from bioamla.audio import convert_audio_file

    src = tmp_path / "src.wav"
    sf.write(src, 0.1 * np.ones(SR, dtype=np.float32), SR, subtype=subtype)
    out = convert_audio_file(src, tmp_path / f"out.{fmt}")
    assert sf.info(out).subtype == want


def test_segment_edges(tmp_path):
    from bioamla.audio import segment_audio_file
    from bioamla.exceptions import InvalidInputError

    empty = tmp_path / "empty.wav"
    sf.write(empty, np.zeros(0, dtype=np.float32), SR)
    with pytest.raises(InvalidInputError, match="no audio"):
        segment_audio_file(empty, tmp_path / "e", duration=1, include_partial=True)
    rec = tmp_path / "rec.wav"
    sf.write(rec, np.zeros(SR * 10, dtype=np.float32), SR)
    clips = segment_audio_file(rec, tmp_path / "c", duration=3, overlap=2.5, include_partial=True)
    ends = [s.end_time for s in clips]
    assert ends[-1] == pytest.approx(10.0) and ends.count(ends[-1]) == 1  # one shorter last clip
    assert all(s.duration == pytest.approx(3.0) for s in clips[:-1])
