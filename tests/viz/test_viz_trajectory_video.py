"""Tests for bioamla.viz.trajectory_video: frames without ffmpeg, encoding with it."""

import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
from click.testing import CliRunner

pytest.importorskip("matplotlib")

from bioamla.audio import frame_features, mfcc_trajectory  # noqa: E402
from bioamla.cli.cli import cli  # noqa: E402
from bioamla.exceptions import DependencyError, InvalidInputError, NotFoundError  # noqa: E402
from bioamla.viz import TrajectoryVideoResult, generate_trajectory_video  # noqa: E402
from bioamla.viz.trajectory_video import _BACKGROUND, _TrajectoryRenderer  # noqa: E402

SR = 22050


def calls(seconds: float = 3.0) -> np.ndarray:
    """Alternating low and high chirps with silent gaps."""
    t = np.arange(int(SR * seconds)) / SR
    x = np.zeros_like(t)
    for i, start in enumerate(np.arange(0.2, seconds - 0.3, 0.5)):
        m = (t >= start) & (t < start + 0.25)
        tt = t[m] - start
        f0 = 1500 if i % 2 else 4000
        x[m] = 0.5 * np.sin(2 * np.pi * (f0 * tt + 3000 * tt**2)) * np.hanning(m.sum())
    return x


@pytest.fixture
def wav(tmp_path: Path) -> Path:
    path = tmp_path / "calls.wav"
    sf.write(path, calls(), SR)
    return path


def renderer(**kwargs) -> _TrajectoryRenderer:
    audio = calls()
    features = frame_features(audio, SR)
    traj = mfcc_trajectory(features, scale="equalize")
    options = {
        "audio": audio,
        "sample_rate": SR,
        "time_offset": 0.0,
        "width": 320,
        "height": 240,
        "cmap": "plasma",
        "trail_seconds": None,
        "orbit_seconds": 40.0,
        "spectrogram": True,
        "window_seconds": 2.0,
        "show_axes": False,
    }
    options.update(kwargs)
    return _TrajectoryRenderer(features, traj, **options)


def lit(frame: np.ndarray) -> int:
    """Pixels brighter than the background."""
    return int((frame.max(axis=2) > _BACKGROUND.max() + 10).sum())


class TestFrames:
    def test_shape_and_dtype(self) -> None:
        frame = renderer().frame(1.0)
        assert frame.shape == (240, 320, 3)
        assert frame.dtype == np.uint8

    def test_odd_sizes_rounded_to_even(self) -> None:
        assert renderer(width=321, height=241).frame(1.0).shape == (240, 320, 3)

    def test_nothing_drawn_before_the_first_call(self) -> None:
        r = renderer(spectrogram=False)
        assert lit(r.frame(0.05)) == 0

    def test_path_grows_over_time(self) -> None:
        r = renderer(spectrogram=False, orbit_seconds=None)
        assert 0 < lit(r.frame(0.6)) < lit(r.frame(2.9))

    def test_trail_fades_old_path(self) -> None:
        full = renderer(spectrogram=False, orbit_seconds=None)
        trail = renderer(spectrogram=False, orbit_seconds=None, trail_seconds=0.3)
        assert lit(trail.frame(2.9)) < lit(full.frame(2.9))

    def test_orbit_turns_the_view(self) -> None:
        r = renderer(spectrogram=False, orbit_seconds=2.0)
        fixed = renderer(spectrogram=False, orbit_seconds=None)
        assert not np.array_equal(r.frame(2.9), fixed.frame(2.9))

    def test_spectrogram_strip(self) -> None:
        with_strip = renderer().frame(1.0)
        without = renderer(spectrogram=False).frame(1.0)
        strip = with_strip[-40:]
        assert lit(strip) > lit(without[-40:])

    def test_labels_change_pixels(self) -> None:
        plain = renderer(spectrogram=False).frame(1.0)
        labelled = renderer(spectrogram=False, show_axes=True).frame(1.0)
        assert not np.array_equal(plain, labelled)

    def test_path_stays_in_view_as_it_turns(self) -> None:
        r = renderer(spectrogram=False, width=320, height=240, orbit_seconds=4.0)
        for t in np.linspace(0, 4, 9):
            x, y = r._project(r.points, t)  # the sampled path, as drawn
            inside = (x >= 0) & (x < r.width) & (y >= 0) & (y < r.plot_h)
            assert inside.mean() >= 0.95

    @pytest.mark.parametrize("size", [(1280, 720), (1080, 1920), (320, 240), (96, 400)])
    def test_title_never_overlaps_the_time_or_leaves_the_frame(self, size) -> None:
        from PIL import Image, ImageDraw

        from bioamla.viz._video_common import MARGIN, fmt_time, label_font

        width, height = size
        r = renderer(spectrogram=False, show_axes=True, width=width, height=height)
        r.frame(0.0)  # lays the title out
        text, y = r._title
        draw = ImageDraw.Draw(Image.new("RGB", (width, height)))
        font = label_font(r.height)
        total = r.time_offset + r.duration
        stamp = draw.textbbox(
            (r.width - MARGIN, MARGIN), f"{fmt_time(total)} / {fmt_time(total)}",
            font=font, anchor="rt",
        )  # fmt: skip
        if not text:
            return
        title = draw.textbbox((MARGIN, y), text, font=font, anchor="lt")
        assert title[2] <= r.width - MARGIN  # inside the frame
        overlaps = (
            title[0] < stamp[2]
            and stamp[0] < title[2]
            and title[1] < stamp[3]
            and stamp[1] < title[3]
        )
        assert not overlaps

    def test_long_recordings_measure_the_wider_time(self) -> None:
        short = renderer(spectrogram=False, show_axes=True, width=640, height=360)
        long = renderer(
            spectrogram=False, show_axes=True, width=640, height=360, time_offset=6000.0
        )
        short.frame(0.0), long.frame(0.0)
        assert long._title[1] >= short._title[1]  # "100:00.0" is wider; the title may move down

    def test_unknown_colormap(self) -> None:
        with pytest.raises(InvalidInputError, match="colormap"):
            renderer(cmap="no-such-map")


class TestArguments:
    def test_missing_file(self, tmp_path) -> None:
        with pytest.raises(NotFoundError):
            generate_trajectory_video(tmp_path / "nope.wav", tmp_path / "o.mp4")

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"width": 10},
            {"fps": 0},
            {"start": -1},
            {"start": 2, "end": 1},
            {"trail_seconds": 0},
            {"orbit_seconds": 0},
            {"window_seconds": 0},
        ],
    )
    def test_bad_arguments(self, wav, tmp_path, kwargs) -> None:
        with pytest.raises(InvalidInputError):
            generate_trajectory_video(wav, tmp_path / "o.mp4", **kwargs)

    def test_missing_ffmpeg(self, wav, tmp_path, mocker) -> None:
        mocker.patch("bioamla.viz._video_common.shutil.which", return_value=None)
        with pytest.raises(DependencyError, match="ffmpeg"):
            generate_trajectory_video(wav, tmp_path / "o.mp4")


@pytest.mark.usefixtures("requires_ffmpeg_cli")
class TestEncoding:
    def test_writes_video_with_audio(self, wav, tmp_path) -> None:
        out = tmp_path / "out" / "t.mp4"
        progress: list[tuple[int, int]] = []
        result = generate_trajectory_video(
            wav,
            out,
            width=160,
            height=120,
            fps=10,
            on_progress=lambda done, total: progress.append((done, total)),
        )
        assert isinstance(result, TrajectoryVideoResult)
        assert out.is_file() and out.stat().st_size > 0
        assert result.frames == 30 and progress[-1] == (30, 30)
        assert result.duration == pytest.approx(3.0, abs=0.01)
        assert 0 < result.active_frames < result.analysis_frames
        assert len(result.explained_variance_ratio) == 3
        if shutil.which("ffprobe"):
            probe = subprocess.run(
                ["ffprobe", "-v", "error", "-show_entries", "stream=codec_type",
                 "-of", "csv=p=0", str(out)],
                capture_output=True, text=True, check=True,
            )  # fmt: skip
            assert sorted(probe.stdout.split()) == ["audio", "video"]

    def test_clip_without_audio(self, wav, tmp_path) -> None:
        result = generate_trajectory_video(
            wav, tmp_path / "t.mp4", start=1.0, end=2.5, width=96, height=96, fps=4,
            include_audio=False,
        )  # fmt: skip
        assert result.frames == 6 and not result.has_audio
        assert (result.start, result.end) == (1.0, 2.5)


class TestCli:
    def test_spectrogram_options_rejected(self, wav, tmp_path) -> None:
        result = CliRunner().invoke(
            cli, ["audio", "video", str(wav), "-t", "trajectory", "--n-mels", "64"]
        )
        assert result.exit_code == 2
        assert "--n-mels doesn't apply to -t trajectory" in result.output

    def test_trajectory_options_rejected_elsewhere(self, wav) -> None:
        result = CliRunner().invoke(cli, ["audio", "video", str(wav), "--trail", "3"])
        assert result.exit_code == 2
        assert "--trail only applies to -t trajectory" in result.output

    def test_several_options_rejected(self, wav) -> None:
        result = CliRunner().invoke(
            cli, ["audio", "video", str(wav), "-t", "trajectory", "--n-mels", "64", "-m", "page"]
        )
        assert result.exit_code == 2
        assert "--motion, --n-mels don't apply to -t trajectory" in result.output

    def test_zero_window_rejected(self, wav) -> None:
        result = CliRunner().invoke(
            cli, ["audio", "video", str(wav), "-t", "trajectory", "-w", "0"]
        )
        assert result.exit_code == 2
        assert "positive span" in result.output

    @pytest.mark.usefixtures("requires_ffmpeg_cli")
    def test_json(self, wav, tmp_path) -> None:
        out = tmp_path / "t.mp4"
        result = CliRunner().invoke(
            cli,
            ["--json", "audio", "video", str(wav), "-t", "trajectory", "-o", str(out),
             "--size", "128x96", "--fps", "5", "--trail", "1", "--no-orbit"],
        )  # fmt: skip
        assert result.exit_code == 0, result.output
        envelope = json.loads(result.stdout)
        assert envelope["ok"] is True
        assert envelope["result"]["output"] == str(out)
        assert envelope["result"]["frames"] == 15
        assert len(envelope["result"]["explained_variance_ratio"]) == 3
