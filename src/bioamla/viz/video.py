"""
Spectrogram Video
=================

Render an audio file as a video of its spectrogram or waveform, in sync with
the audio. Two kinds of motion:

- ``scroll``: the picture slides right-to-left past a fixed playhead, like a
  tape passing a head. ``window_seconds`` sets how much audio spans the frame.
- ``page``: the picture holds still and the playhead sweeps across it; when it
  reaches the right edge the next page is shown. With ``window_seconds=None``
  the single page is the whole recording.

One video frame is produced per 1/fps of audio, so the video is exactly as
long as the recording and the audio track lines up with what is on screen.

Frames are composed in numpy (the spectrogram is coloured once, then each frame
is a window into it), so rendering runs at hundreds of frames per second; the
cost is dominated by the encoder. Encoding and audio muxing go through the
``ffmpeg`` command-line tool, piped raw RGB frames, so there is no video
library dependency.

Example:
    >>> from bioamla.viz import generate_spectrogram_video
    >>> generate_spectrogram_video("wren.wav", "wren.mp4")
    >>> generate_spectrogram_video("wren.wav", "wave.mp4", viz_type="waveform",
    ...                            motion="page", window_seconds=None)
"""

from __future__ import annotations

import logging
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import librosa
import numpy as np

from bioamla.audio.data import ToDictMixin
from bioamla.exceptions import DependencyError, InvalidInputError, NotFoundError, ProcessingError
from bioamla.protocols import OnProgress
from bioamla.viz.core import WindowType, compute_mel_spectrogram, compute_stft, spectrogram_to_db

logger = logging.getLogger(__name__)

VideoVizType = Literal["mel", "stft", "waveform"]
VideoMotion = Literal["scroll", "page"]

# Layout, in pixels. Everything else scales with the frame size.
_WAVEFORM_FRACTION = 0.18  # share of the height given to the waveform strip under a spectrogram
_PLAYHEAD_WIDTH = 2
_AXIS_TICKS = 6
_MARGIN = 8


@dataclass
class SpectrogramVideoResult(ToDictMixin):
    """What :func:`generate_spectrogram_video` wrote.

    Attributes:
        output_path: The video file.
        duration: Length of the video (and the audio it covers) in seconds.
        start: Where in the source file the video starts, in seconds.
        end: Where in the source file it ends, in seconds.
        fps: Frames per second.
        frames: Number of frames rendered.
        width: Frame width in pixels.
        height: Frame height in pixels.
        has_audio: Whether the audio track was muxed in.
    """

    output_path: str
    duration: float
    start: float
    end: float
    fps: int
    frames: int
    width: int
    height: int
    has_audio: bool


def generate_spectrogram_video(
    audio_path: str | Path,
    output_path: str | Path,
    viz_type: VideoVizType = "mel",
    motion: VideoMotion = "scroll",
    start: float = 0.0,
    end: float | None = None,
    sample_rate: int | None = None,
    n_fft: int = 2048,
    hop_length: int = 512,
    n_mels: int = 128,
    window: WindowType = "hann",
    fmin: float = 0.0,
    fmax: float | None = None,
    db_range: float = 80.0,
    cmap: str = "magma",
    width: int = 1280,
    height: int = 720,
    fps: int = 30,
    window_seconds: float | None = 4.0,
    playhead: float = 0.5,
    waveform: bool = True,
    show_axes: bool = True,
    include_audio: bool = True,
    backend: str = "auto",
    on_progress: OnProgress | None = None,
) -> SpectrogramVideoResult:
    """
    Render an audio file as a spectrogram or waveform video in sync with its audio.

    Args:
        audio_path: Path to the input audio file (any format bioamla can load).
        output_path: Path of the video to write. The container is taken from the
            extension; ``.mp4`` (H.264) is the safe default.
        viz_type: ``"mel"`` (default) or ``"stft"`` spectrogram, or ``"waveform"``
            for a waveform-only video.
        motion: ``"scroll"`` (default) moves the picture past a fixed playhead;
            ``"page"`` holds the picture and moves the playhead across it, page
            by page.
        start: Render from this many seconds into the file.
        end: Render up to this many seconds into the file (default: the end).
            An ``end`` past the end of the file is clipped to it.
        sample_rate: Resample the audio to this rate first. ``None`` keeps the
            file's own rate, so the frequency axis reaches its full Nyquist.
        n_fft: FFT window size.
        hop_length: Samples between successive spectrogram frames.
        n_mels: Number of mel bands (mel only).
        window: Window function name.
        fmin: Lowest frequency shown.
        fmax: Highest frequency shown (default: Nyquist).
        db_range: Dynamic range in dB below the loudest point.
        cmap: Matplotlib colormap name.
        width: Frame width in pixels (rounded down to even for the encoder).
        height: Frame height in pixels (rounded down to even).
        fps: Frames per second.
        window_seconds: How many seconds of audio span the frame width. ``None``
            means the whole recording, which only makes sense with
            ``motion="page"``.
        playhead: In scroll mode, the playhead's position as a fraction of the
            width: 0.5 centres it, 1.0 puts it at the right edge so only the
            past is shown. Ignored in page mode, where the playhead moves.
        waveform: Draw a waveform strip under the spectrogram (spectrogram
            types only).
        show_axes: Draw frequency tick labels and the elapsed-time readout.
        include_audio: Mux the audio file into the video.
        backend: Spectrogram compute backend, as in :func:`compute_stft`.
        on_progress: Called as ``on_progress(frames_done, frames_total)``
            after each frame is handed to the encoder.

    Returns:
        A :class:`SpectrogramVideoResult` describing the written video.

    Raises:
        NotFoundError: If the audio file does not exist.
        InvalidInputError: If an argument is out of range, or ``start``/``end``
            select no audio.
        DependencyError: If the ``ffmpeg`` command is not installed.
        AudioLoadError: If the audio cannot be loaded.
        ProcessingError: If encoding fails.
    """
    audio_path = Path(audio_path)
    output_path = Path(output_path)
    if not audio_path.is_file():
        raise NotFoundError(f"Audio file not found: {audio_path}")
    _validate(viz_type, motion, width, height, fps, window_seconds, playhead, db_range)
    if start < 0:
        raise InvalidInputError(f"start must not be negative, got {start}")
    if end is not None and end <= start:
        raise InvalidInputError(f"end ({end}) must be after start ({start})")
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise DependencyError(
            "ffmpeg is required to encode video but was not found on PATH. "
            "Install it (e.g. brew install ffmpeg / apt install ffmpeg) and retry."
        )

    from bioamla.viz.core import _load_audio_for_viz

    if sample_rate is None:
        from bioamla.audio import load_audio

        audio, sample_rate = load_audio(str(audio_path))
    else:
        audio = _load_audio_for_viz(str(audio_path), sample_rate)

    file_duration = len(audio) / sample_rate
    if start >= file_duration:
        raise InvalidInputError(
            f"start ({start} s) is beyond the end of the file ({file_duration:.2f} s)"
        )
    end = file_duration if end is None else min(end, file_duration)
    audio = audio[int(round(start * sample_rate)) : int(round(end * sample_rate))]

    renderer = _FrameRenderer(
        audio,
        sample_rate,
        time_offset=start,
        viz_type=viz_type,
        motion=motion,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
        window=window,
        fmin=fmin,
        fmax=fmax,
        db_range=db_range,
        cmap=cmap,
        width=width,
        height=height,
        window_seconds=window_seconds,
        playhead=playhead,
        waveform=waveform,
        show_axes=show_axes,
        backend=backend,
    )
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
    if include_audio:
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

    return SpectrogramVideoResult(
        output_path=str(output_path),
        duration=renderer.duration,
        start=start,
        end=end,
        fps=fps,
        frames=total_frames,
        width=renderer.width,
        height=renderer.height,
        has_audio=include_audio,
    )


def _validate(
    viz_type: str,
    motion: str,
    width: int,
    height: int,
    fps: int,
    window_seconds: float | None,
    playhead: float,
    db_range: float,
) -> None:
    if viz_type not in ("mel", "stft", "waveform"):
        raise InvalidInputError(f"viz_type must be 'mel', 'stft' or 'waveform', got {viz_type!r}")
    if motion not in ("scroll", "page"):
        raise InvalidInputError(f"motion must be 'scroll' or 'page', got {motion!r}")
    if width < 64 or height < 64:
        raise InvalidInputError(f"width and height must be at least 64 px, got {width}x{height}")
    if fps < 1:
        raise InvalidInputError(f"fps must be at least 1, got {fps}")
    if window_seconds is None:
        if motion != "page":
            raise InvalidInputError("window_seconds=None (whole recording) needs motion='page'")
    elif window_seconds <= 0:
        raise InvalidInputError(f"window_seconds must be positive, got {window_seconds}")
    if not 0.0 <= playhead <= 1.0:
        raise InvalidInputError(f"playhead must be between 0 and 1, got {playhead}")
    if db_range <= 0:
        raise InvalidInputError(f"db_range must be positive, got {db_range}")


class _FrameRenderer:
    """Composes one RGB frame per time position from a precomputed spectrogram.

    The spectrogram is computed and scaled to dB once. For a frame at time
    ``t`` the visible span (``[t - w*playhead, t + w*(1-playhead)]`` when
    scrolling, the current page otherwise) is mapped column-by-column onto
    spectrogram frames (nearest), rows are interpolated to the display height,
    and the result is coloured through the colormap's lookup table. Columns
    outside the recording are the colormap's floor colour, so a scrolling
    picture enters from the right at the start and leaves to the left at the
    end. In page mode the page image is cached and only the playhead moves.
    """

    def __init__(
        self,
        audio: np.ndarray,
        sample_rate: int,
        *,
        time_offset: float = 0.0,
        viz_type: VideoVizType,
        motion: VideoMotion = "scroll",
        n_fft: int,
        hop_length: int,
        n_mels: int,
        window: WindowType,
        fmin: float,
        fmax: float | None,
        db_range: float,
        cmap: str,
        width: int,
        height: int,
        window_seconds: float | None,
        playhead: float,
        waveform: bool,
        show_axes: bool,
        backend: str,
    ) -> None:
        self.sample_rate = sample_rate
        self.duration = len(audio) / sample_rate
        self.time_offset = time_offset  # where this audio sits in the source file
        self.motion = motion
        # yuv420p needs even dimensions.
        self.width = width - width % 2
        self.height = height - height % 2
        self.window_seconds = window_seconds if window_seconds is not None else self.duration
        self.playhead = playhead
        self.show_axes = show_axes
        self.px_per_second = self.width / self.window_seconds
        self.playhead_x = int(round(playhead * (self.width - _PLAYHEAD_WIDTH)))

        self.waveform_only = viz_type == "waveform"
        if self.waveform_only:
            self.wave_h = self.height
        else:
            self.wave_h = int(self.height * _WAVEFORM_FRACTION) if waveform else 0
        self.spec_h = self.height - self.wave_h

        # --- colormap lookup table (matplotlib imported lazily) ---
        from matplotlib import colormaps

        try:
            lut = colormaps[cmap](np.linspace(0, 1, 256))
        except KeyError as e:
            raise InvalidInputError(f"Unknown colormap: {cmap!r}") from e
        self.lut = (lut[:, :3] * 255).astype(np.uint8)
        self.background = self.lut[0]

        if fmax is None:
            fmax = sample_rate / 2
        self.fmin, self.fmax = fmin, fmax
        self._ticks: list[tuple[int, str]] = []
        if not self.waveform_only:
            self._prepare_spectrogram(
                audio, viz_type, n_fft, hop_length, n_mels, window, fmin, fmax, db_range, backend
            )
            if show_axes:
                self._ticks = self._frequency_ticks()

        if self.wave_h:
            self._prepare_waveform(audio)

        self.playhead_color = np.array([255, 255, 255], dtype=np.uint8)
        self._font = None
        self._page: tuple[float, np.ndarray] | None = None

    # -- precomputation ----------------------------------------------------

    def _prepare_spectrogram(
        self,
        audio: np.ndarray,
        viz_type: str,
        n_fft: int,
        hop_length: int,
        n_mels: int,
        window: WindowType,
        fmin: float,
        fmax: float,
        db_range: float,
        backend: str,
    ) -> None:
        """Compute the spectrogram once, as 0..255 indices into the colormap."""
        if viz_type == "mel":
            _, power = compute_mel_spectrogram(
                audio,
                self.sample_rate,
                n_fft=n_fft,
                hop_length=hop_length,
                n_mels=n_mels,
                window=window,
                fmin=fmin,
                fmax=fmax,
                backend=backend,
            )
            self.bin_freqs = librosa.mel_frequencies(n_mels=n_mels, fmin=fmin, fmax=fmax)
        else:
            freqs, _, mag = compute_stft(
                audio,
                self.sample_rate,
                n_fft=n_fft,
                hop_length=hop_length,
                window=window,
                backend=backend,
            )
            keep = (freqs >= fmin) & (freqs <= fmax)
            power = mag[keep] ** 2
            self.bin_freqs = freqs[keep]
        db = spectrogram_to_db(power, ref="max", top_db=db_range)
        self.levels = ((db + db_range) / db_range * 255).clip(0, 255).astype(np.uint8)
        self.n_bins, self.n_frames = self.levels.shape
        self.frames_per_second = self.sample_rate / hop_length

        # Row interpolation: display row -> (lower bin, upper bin, weight).
        # Rows run top-to-bottom, frequency runs bottom-to-top.
        pos = np.linspace(self.n_bins - 1, 0, self.spec_h)
        self.row_lo = np.floor(pos).astype(int)
        self.row_hi = np.minimum(self.row_lo + 1, self.n_bins - 1)
        self.row_w = (pos - self.row_lo).astype(np.float32)[:, None]

    def _prepare_waveform(self, audio: np.ndarray) -> None:
        """Min/max envelope of the signal at one value per display column."""
        n_cols = max(1, int(np.ceil(self.duration * self.px_per_second)))
        per_col = self.sample_rate / self.px_per_second
        starts = (np.arange(n_cols) * per_col).astype(int).clip(0, len(audio) - 1)
        self.env_lo = np.minimum.reduceat(audio, starts).astype(np.float32)
        self.env_hi = np.maximum.reduceat(audio, starts).astype(np.float32)
        peak = max(abs(self.env_lo.min()), abs(self.env_hi.max()), 1e-9)
        self.env_lo /= peak
        self.env_hi /= peak
        self.wave_color = self.lut[200]
        self.wave_background = (self.background * 0.6).astype(np.uint8)

    # -- frame composition -------------------------------------------------

    def frame(self, t: float) -> np.ndarray:
        """Render the frame for time ``t`` (seconds) as an (H, W, 3) uint8 array."""
        if self.motion == "scroll":
            t0 = t - self.playhead * self.window_seconds
            x = self.playhead_x
            picture = self._picture(t0)
        else:
            t0 = np.floor(t / self.window_seconds) * self.window_seconds
            x = int(((t - t0) / self.window_seconds) * (self.width - _PLAYHEAD_WIDTH))
            if self._page is None or self._page[0] != t0:
                self._page = (t0, self._picture(t0))
            picture = self._page[1]

        frame = picture.copy()
        frame[:, x : x + _PLAYHEAD_WIDTH] = self.playhead_color
        if self.show_axes:
            frame = self._draw_labels(frame, t)
        return np.ascontiguousarray(frame)

    def _picture(self, t0: float) -> np.ndarray:
        """The spectrogram and/or waveform for the span starting at ``t0``, no overlay."""
        col_times = t0 + np.arange(self.width) / self.px_per_second
        parts = []
        if self.spec_h:
            parts.append(self._spectrogram_band(col_times))
        if self.wave_h:
            parts.append(self._waveform_strip(col_times))
        return np.concatenate(parts, axis=0)

    def _spectrogram_band(self, col_times: np.ndarray) -> np.ndarray:
        src = np.rint(col_times * self.frames_per_second).astype(int)
        valid = (src >= 0) & (src < self.n_frames)
        window = self.levels[:, src.clip(0, self.n_frames - 1)].astype(np.float32)
        rows = window[self.row_lo] * (1 - self.row_w) + window[self.row_hi] * self.row_w
        band = self.lut[rows.astype(np.uint8)]
        band[:, ~valid] = self.background
        return band

    def _waveform_strip(self, col_times: np.ndarray) -> np.ndarray:
        cols = np.floor(col_times * self.px_per_second).astype(int)
        valid = (cols >= 0) & (cols < len(self.env_lo))
        idx = cols.clip(0, len(self.env_lo) - 1)
        lo = np.where(valid, self.env_lo[idx], 0.0)
        hi = np.where(valid, self.env_hi[idx], 0.0)
        # Row 0 is the top (+1), the last row is the bottom (-1).
        mid = (self.wave_h - 1) / 2
        y = (mid - np.arange(self.wave_h)) / max(mid, 1e-9)
        y = y[:, None]  # (wave_h, 1) in -1..1, top to bottom
        half_px = 1 / max(mid, 1e-9)  # always draw at least one pixel row of signal
        mask = (y <= hi + half_px) & (y >= lo - half_px)
        strip = np.empty((self.wave_h, self.width, 3), dtype=np.uint8)
        strip[:] = self.wave_background
        strip[mask] = self.wave_color
        return strip

    # -- labels (Pillow, which matplotlib already depends on) ----------------

    def _frequency_ticks(self) -> list[tuple[int, str]]:
        """(row, label) pairs for a handful of round frequencies on the spectrogram."""
        lo, hi = self.bin_freqs[0], self.bin_freqs[-1]
        if hi <= lo:
            return []
        candidates = np.array(
            [50, 100, 200, 500, 1000, 2000, 3000, 4000, 5000, 8000, 10000, 15000, 20000, 40000]
        )
        candidates = candidates[(candidates > lo) & (candidates < hi)]
        # Thin to at most _AXIS_TICKS, keeping them spread over the axis.
        if len(candidates) > _AXIS_TICKS:
            idx = np.linspace(0, len(candidates) - 1, _AXIS_TICKS).round().astype(int)
            candidates = candidates[np.unique(idx)]
        ticks: list[tuple[int, str]] = []
        min_gap = max(14, self.height // 40)  # roughly one label height
        for f in candidates[::-1]:  # top of the picture first
            # The bin whose centre is closest, then that bin's display row.
            b = int(np.argmin(np.abs(self.bin_freqs - f)))
            row = int(round((self.n_bins - 1 - b) / max(self.n_bins - 1, 1) * (self.spec_h - 1)))
            if ticks and row - ticks[-1][0] < min_gap:
                continue  # would overprint the label above it (linear axes crowd the bottom)
            label = f"{f / 1000:g} kHz" if f >= 1000 else f"{f:g} Hz"
            ticks.append((row, label))
        return ticks

    def _draw_labels(self, frame: np.ndarray, t: float) -> np.ndarray:
        from PIL import Image, ImageDraw, ImageFont

        if self._font is None:
            size = max(10, self.height // 45)
            try:
                self._font = ImageFont.load_default(size=size)
            except TypeError:  # Pillow < 10.1 has no size argument
                self._font = ImageFont.load_default()
        image = Image.fromarray(frame)
        draw = ImageDraw.Draw(image)
        fg, shadow = (255, 255, 255), (0, 0, 0)

        for row, label in self._ticks:
            y = min(max(row, _MARGIN), self.spec_h - _MARGIN)
            draw.line([(0, row), (6, row)], fill=fg, width=1)
            draw.text((_MARGIN + 2, y + 1), label, font=self._font, fill=shadow, anchor="lm")
            draw.text((_MARGIN + 1, y), label, font=self._font, fill=fg, anchor="lm")

        stamp = f"{_fmt_time(self.time_offset + t)} / {_fmt_time(self.time_offset + self.duration)}"
        x, y = self.width - _MARGIN, _MARGIN
        draw.text((x + 1, y + 1), stamp, font=self._font, fill=shadow, anchor="rt")
        draw.text((x, y), stamp, font=self._font, fill=fg, anchor="rt")
        return np.asarray(image)


def _fmt_time(seconds: float) -> str:
    seconds = max(0.0, seconds)
    minutes, secs = divmod(seconds, 60)
    return f"{int(minutes):02d}:{secs:04.1f}"
