"""Shared plumbing for the video renderers: clip loading, ffmpeg encoding, labels.

Internal to :mod:`bioamla.viz`. A renderer is anything with ``width``, ``height``
(both even), ``duration`` and ``frame(t)`` returning an ``(height, width, 3)``
uint8 array; :func:`encode_video` pipes one frame per ``1/fps`` of audio into the
``ffmpeg`` command and muxes the audio.
"""

from __future__ import annotations

import logging
import shutil
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

import numpy as np

from bioamla.exceptions import DependencyError, InvalidInputError, ProcessingError
from bioamla.protocols import OnProgress

if TYPE_CHECKING:
    from PIL import ImageDraw

logger = logging.getLogger("bioamla.viz.video")

MARGIN = 8  # px between labels and the frame edge


class Renderer(Protocol):
    """What :func:`encode_video` needs from a frame source."""

    width: int  # even
    height: int  # even
    duration: float  # seconds of audio covered

    def frame(self, t: float) -> np.ndarray: ...  # (height, width, 3) uint8


def require_ffmpeg() -> str:
    """The ``ffmpeg`` executable, or DependencyError."""
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise DependencyError(
            "ffmpeg is required to encode video but was not found on PATH. "
            "Install it (e.g. brew install ffmpeg / apt install ffmpeg) and retry."
        )
    return ffmpeg


def load_clip(
    audio_path: Path, start: float, end: float | None, sample_rate: int | None
) -> tuple[np.ndarray, int, float]:
    """Load ``start``-``end`` of ``audio_path`` as mono, resampled when ``sample_rate`` is set.

    At the file's own rate only the window is decoded, so a short clip of a long
    recording stays cheap. Returns the clip, its sample rate and the resolved
    ``end`` (clipped to the file).

    Raises:
        InvalidInputError: If ``start`` is at or beyond the end of the file.
    """
    if sample_rate is None:
        from bioamla.audio import load_audio_data

        windowed = start > 0 or end is not None
        data = load_audio_data(
            audio_path,
            start=start if windowed else None,
            duration=None if end is None else end - start,
        )
        clip, sample_rate = data.samples, data.sample_rate
        if clip.ndim > 1:
            clip = clip.mean(axis=1)
        if len(clip) == 0:
            raise InvalidInputError(f"start ({start} s) is beyond the end of the file")
        return clip, sample_rate, start + len(clip) / sample_rate

    from bioamla.viz.core import _load_audio_for_viz

    audio = _load_audio_for_viz(str(audio_path), sample_rate)
    file_duration = len(audio) / sample_rate
    if start >= file_duration:
        raise InvalidInputError(
            f"start ({start} s) is beyond the end of the file ({file_duration:.2f} s)"
        )
    end = file_duration if end is None else min(end, file_duration)
    clip = audio[int(round(start * sample_rate)) : int(round(end * sample_rate))]
    return clip, sample_rate, end


def encode_video(
    renderer: Renderer,
    *,
    ffmpeg: str,
    output_path: Path,
    fps: int,
    audio_path: Path | None,
    start: float,
    on_progress: OnProgress | None,
) -> int:
    """Pipe one frame per ``1/fps`` of audio into ffmpeg; mux ``audio_path`` from ``start``.

    Returns the number of frames written.

    Raises:
        ProcessingError: If ffmpeg fails.
    """
    total_frames = max(1, int(np.ceil(renderer.duration * fps)))
    cmd = [
        ffmpeg,
        "-y",
        "-loglevel",
        "error",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        f"{renderer.width}x{renderer.height}",
        "-r",
        str(fps),
        "-i",
        "-",
    ]
    if audio_path is not None:
        cmd += ["-ss", f"{start:.6f}", "-t", f"{renderer.duration:.6f}", "-i", str(audio_path)]
        cmd += ["-map", "0:v", "-map", "1:a", "-c:a", "aac", "-shortest"]
    cmd += ["-c:v", "libx264", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(output_path)]

    logger.info("Rendering %d frames at %d fps to %s", total_frames, fps, output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    assert proc.stdin is not None
    try:
        for i in range(total_frames):
            frame = renderer.frame(i / fps)
            try:
                proc.stdin.write(frame.tobytes())
            except BrokenPipeError:
                break  # ffmpeg died; its stderr is reported below
            if on_progress is not None:
                on_progress(i + 1, total_frames)
        proc.stdin.close()
        stderr = proc.stderr.read().decode(errors="replace") if proc.stderr else ""
        returncode = proc.wait()
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait()
    if returncode != 0:
        raise ProcessingError(f"ffmpeg failed (exit {returncode}): {stderr.strip()[-2000:]}")
    return total_frames


def check_frame_args(width: int, height: int, fps: int, start: float, end: float | None) -> None:
    """The frame size, rate and clip checks every video shares."""
    if width < 64 or height < 64:
        raise InvalidInputError(f"width and height must be at least 64 px, got {width}x{height}")
    if fps < 1:
        raise InvalidInputError(f"fps must be at least 1, got {fps}")
    if start < 0:
        raise InvalidInputError(f"start must not be negative, got {start}")
    if end is not None and end <= start:
        raise InvalidInputError(f"end ({end}) must be after start ({start})")


def label_font(height: int) -> Any:
    """Pillow's default font at a size that suits the frame height."""
    from PIL import ImageFont

    size = max(10, height // 45)
    try:
        return ImageFont.load_default(size=size)
    except TypeError:  # Pillow < 10.1 has no size argument
        return ImageFont.load_default()


def draw_time_stamp(
    draw: ImageDraw.ImageDraw, font: Any, width: int, t: float, total: float
) -> None:
    """``elapsed / total`` in the top-right corner, white with a shadow."""
    stamp = f"{fmt_time(t)} / {fmt_time(total)}"
    x, y = width - MARGIN, MARGIN
    draw.text((x + 1, y + 1), stamp, font=font, fill=(0, 0, 0), anchor="rt")
    draw.text((x, y), stamp, font=font, fill=(255, 255, 255), anchor="rt")


def fmt_time(seconds: float) -> str:
    seconds = max(0.0, seconds)
    minutes, secs = divmod(seconds, 60)
    return f"{int(minutes):02d}:{secs:04.1f}"
