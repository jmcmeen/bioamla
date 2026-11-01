"""The audio operations, run through the registry without the CLI."""

import pytest

from bioamla import operations

OUTPUT_OPS = {
    "audio.trim": {"start": 0.5},
    "audio.normalize": {},
    "audio.resample": {"sample_rate": 8000},
    "audio.filter": {"highpass": 200.0},
    "audio.denoise": {},
    "audio.add-noise": {"snr_db": 20.0, "seed": 0},
    "audio.gain": {"gain_db": 3.0},
    "audio.convert": {"channels": 1},
}


def _counting():
    seen = []

    def check(path):
        seen.append(path)
        return path

    return seen, operations.OperationContext(path_check=check)


@pytest.mark.parametrize(("name", "extra"), OUTPUT_OPS.items())
def test_file_in_file_out(name, extra, test_audio_path_3s, tmp_path):
    out = str(tmp_path / "out.wav")
    seen, ctx = _counting()
    result = operations.run(name, {"file": test_audio_path_3s, "output": out, **extra}, ctx)
    assert result.to_dict()["output"] == out
    assert sorted(seen) == sorted([test_audio_path_3s, out])
    op = operations.get(name)
    assert (op.kind, op.remote) == ("job", True)


def test_reading_operations_are_quick(test_audio_path_3s, test_audio_dir):
    for name, request in [
        ("audio.info", {"file": test_audio_path_3s, "metadata": True}),
        ("audio.metadata", {"file": test_audio_path_3s}),
        ("audio.list", {"path": test_audio_dir}),
    ]:
        seen, ctx = _counting()
        result = operations.run(name, request, ctx)
        assert operations.get(name).kind == "quick"
        assert seen == [request.get("file", request.get("path"))]
        assert set(result.to_dict()) & {"file", "path"}


def test_segment_checks_its_output_folder(test_audio_path_3s, tmp_path):
    seen, ctx = _counting()
    result = operations.run(
        "audio.segment",
        {"file": test_audio_path_3s, "output_dir": str(tmp_path / "clips"), "duration": 1.0},
        ctx,
    )
    assert result.count == 3
    assert str(tmp_path / "clips") in seen
    assert [s["path"].rsplit("/", 1)[1] for s in result.to_dict()["segments"]][0].endswith(
        "_seg0000.wav"
    )


def test_segment_rejects_a_prefix_that_escapes(test_audio_path_3s, tmp_path):
    from bioamla.exceptions import InvalidInputError

    with pytest.raises(InvalidInputError):
        operations.run(
            "audio.segment",
            {"file": test_audio_path_3s, "output_dir": str(tmp_path), "prefix": "../escape"},
        )


def test_play_is_local_only(mocker):
    play = mocker.patch("bioamla.audio.playback.play_audio", return_value="player")
    assert operations.run("audio.play", {"file": "x.wav", "loop": True}) == "player"
    play.assert_called_once_with("x.wav", loop=True, block=False)
    assert operations.get("audio.play").remote is False


def test_video_reports_progress_and_honours_cancel(mocker, test_audio_path_3s, tmp_path):
    def render(**kwargs):
        for i in range(1, 4):
            kwargs["on_progress"](i, 3)
        return "done"

    mocker.patch("bioamla.viz.generate_spectrogram_video", side_effect=render)
    seen = []
    ctx = operations.OperationContext(on_progress=lambda d, t: seen.append(d))
    request = {"file": test_audio_path_3s, "output": str(tmp_path / "v.mp4")}
    assert operations.run("audio.video", request, ctx) == "done"
    assert seen == [1, 2, 3]

    stop = operations.OperationContext(cancelled=lambda: True)
    with pytest.raises(operations.OperationCancelled):
        operations.run("audio.video", request, stop)


def test_video_leaves_the_default_cmap_to_the_renderer(mocker, test_audio_path_3s, tmp_path):
    spec = mocker.patch("bioamla.viz.generate_spectrogram_video")
    traj = mocker.patch("bioamla.viz.generate_trajectory_video")
    base = {"file": test_audio_path_3s, "output": str(tmp_path / "v.mp4")}
    operations.run("audio.video", base)
    operations.run("audio.video", {**base, "viz_type": "trajectory", "cmap": "gray"})
    assert "cmap" not in spec.call_args.kwargs
    assert traj.call_args.kwargs["cmap"] == "gray"


def test_features_takes_a_nested_config(test_audio_path_3s):
    pytest.importorskip("librosa")
    result = operations.run("audio.features", {"file": test_audio_path_3s, "config": {"n_mfcc": 5}})
    assert "mfcc_4" in result.table.columns
    assert "mfcc_5" not in result.table.columns
