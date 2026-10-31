"""
Trajectory video: a recording's path through timbre space, in sync with its audio.

Each frame of the recording is placed by its MFCC-PCA coordinates
(:func:`bioamla.audio.mfcc_trajectory`) and joined to the next, so the song draws
its own shape as it plays. The view is an orthographic projection that can turn
slowly around the vertical axis. The path is coloured by spectral centroid (low to
high pitch through the colormap) and weighted by level; silent stretches break the
line, and the current frame is a bright dot. An optional scrolling mel spectrogram
along the bottom ties the shape to the sound.

Frames are drawn with numpy (the path is sampled along each segment and added up,
so overlapping strokes glow) and encoded through the same ``ffmpeg`` pipe as
:func:`bioamla.viz.generate_spectrogram_video`.

Example:
    >>> from bioamla.viz import generate_trajectory_video
    >>> generate_trajectory_video("wren.wav", "wren-trajectory.mp4", trail_seconds=4)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from bioamla.audio.data import ToDictMixin
from bioamla.exceptions import InvalidInputError, NotFoundError
from bioamla.protocols import OnProgress
from bioamla.viz._video_common import (
    MARGIN,
    check_frame_args,
    draw_time_stamp,
    encode_video,
    label_font,
    load_clip,
    require_ffmpeg,
)
from bioamla.viz.video import _FrameRenderer  # the spectrogram strip

if TYPE_CHECKING:
    from bioamla.audio.features import FrameFeatureConfig, FrameFeatures, Trajectory

logger = logging.getLogger(__name__)

_STRIP_FRACTION = 0.22  # share of the height given to the spectrogram strip
_TILT_DEGREES = 18.0  # camera elevation above the PC1-PC3 plane
_RECENT_SECONDS = 1.5  # the last moments are drawn brighter than older history
_BACKGROUND = np.array([6, 6, 10], dtype=np.uint8)


@dataclass
class TrajectoryVideoResult(ToDictMixin):
    """A trajectory video written by :func:`generate_trajectory_video`.

    Attributes:
        output_path: Path of the video.
        duration: Length of the video (and the audio it covers) in seconds.
        start: Where the video starts in the audio file, in seconds.
        end: Where it ends, in seconds.
        fps: Frames per second.
        frames: Number of video frames.
        width: Frame width in pixels.
        height: Frame height in pixels.
        has_audio: Whether the audio track was muxed in.
        analysis_frames: Number of analysis frames in the clip.
        active_frames: How many of them passed the level gate (drawn).
        explained_variance_ratio: Share of the MFCC variance on each axis.
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
    analysis_frames: int
    active_frames: int
    explained_variance_ratio: list[float] = field(default_factory=list)


def generate_trajectory_video(
    audio_path: str | Path,
    output_path: str | Path,
    *,
    start: float = 0.0,
    end: float | None = None,
    config: FrameFeatureConfig | None = None,
    gate_db: float = 38.0,
    width: int = 1280,
    height: int = 720,
    fps: int = 30,
    trail_seconds: float | None = None,
    orbit: bool = True,
    orbit_seconds: float = 40.0,
    spectrogram: bool = True,
    window_seconds: float = 4.0,
    cmap: str = "plasma",
    show_axes: bool = True,
    include_audio: bool = True,
    on_progress: OnProgress | None = None,
) -> TrajectoryVideoResult:
    """
    Render a recording's MFCC-PCA timbre trajectory as a video, in sync with its audio.

    The trajectory is fitted on the rendered clip (``start``-``end``), so its axes
    describe that stretch of the recording.

    Args:
        audio_path: Path to the input audio file (any format bioamla can load).
        output_path: Path of the video to write; ``.mp4`` (H.264) is the safe choice.
        start: Render from this many seconds into the file.
        end: Render up to this many seconds into the file (default: the end).
        config: Analysis settings for :func:`bioamla.audio.frame_features`.
        gate_db: Frames more than this many dB below the loudest are not drawn.
        width: Frame width in pixels (rounded down to even).
        height: Frame height in pixels (rounded down to even).
        fps: Frames per second.
        trail_seconds: Fade out path older than this. ``None`` keeps the whole
            history (older parts dimmer than the last moments).
        orbit: Turn the view slowly around the vertical axis.
        orbit_seconds: Seconds per full turn.
        spectrogram: Draw a scrolling mel spectrogram strip along the bottom.
        window_seconds: Seconds of audio across the spectrogram strip.
        cmap: Matplotlib colormap for the path (low to high spectral centroid).
        show_axes: Draw the PC axes with their explained variance, and the time.
        include_audio: Mux the audio file into the video.
        on_progress: Called as ``on_progress(frames_done, frames_total)``.

    Returns:
        A :class:`TrajectoryVideoResult`.

    Raises:
        NotFoundError: If the audio file does not exist.
        InvalidInputError: If an argument is out of range, ``start``/``end`` select
            no audio, or too few frames pass the gate.
        DependencyError: If the ``ffmpeg`` command is not installed.
        AudioLoadError: If the audio cannot be loaded.
        ProcessingError: If encoding fails.
    """
    from bioamla.audio.features import frame_features, mfcc_trajectory

    audio_path = Path(audio_path)
    output_path = Path(output_path)
    if not audio_path.is_file():
        raise NotFoundError(f"Audio file not found: {audio_path}")
    check_frame_args(width, height, fps, start, end)
    if trail_seconds is not None and trail_seconds <= 0:
        raise InvalidInputError(f"trail_seconds must be positive, got {trail_seconds}")
    if orbit_seconds <= 0:
        raise InvalidInputError(f"orbit_seconds must be positive, got {orbit_seconds}")
    if window_seconds <= 0:
        raise InvalidInputError(f"window_seconds must be positive, got {window_seconds}")

    ffmpeg = require_ffmpeg()
    audio, sample_rate, end = load_clip(audio_path, start, end, None)
    features = frame_features(audio, sample_rate, config)
    # "equalize" evens the axes out, so a dominant first axis doesn't flatten the view.
    trajectory = mfcc_trajectory(features, gate_db=gate_db, n_components=3, scale="equalize")

    renderer = _TrajectoryRenderer(
        features,
        trajectory,
        audio=audio,
        sample_rate=sample_rate,
        time_offset=start,
        width=width,
        height=height,
        cmap=cmap,
        trail_seconds=trail_seconds,
        orbit_seconds=orbit_seconds if orbit else None,
        spectrogram=spectrogram,
        window_seconds=window_seconds,
        show_axes=show_axes,
    )
    total_frames = encode_video(
        renderer,
        ffmpeg=ffmpeg,
        output_path=output_path,
        fps=fps,
        audio_path=audio_path if include_audio else None,
        start=start,
        on_progress=on_progress,
    )
    return TrajectoryVideoResult(
        output_path=str(output_path),
        duration=renderer.duration,
        start=start,
        end=end,
        fps=fps,
        frames=total_frames,
        width=renderer.width,
        height=renderer.height,
        has_audio=include_audio,
        analysis_frames=features.n_frames,
        active_frames=int(trajectory.active.sum()),
        explained_variance_ratio=trajectory.explained_variance_ratio.tolist(),
    )


class _TrajectoryRenderer:
    """Draws the trajectory up to time ``t`` (seconds into the clip) as an RGB frame."""

    def __init__(
        self,
        features: FrameFeatures,
        trajectory: Trajectory,
        *,
        audio: np.ndarray,
        sample_rate: int,
        time_offset: float,
        width: int,
        height: int,
        cmap: str,
        trail_seconds: float | None,
        orbit_seconds: float | None,
        spectrogram: bool,
        window_seconds: float,
        show_axes: bool,
    ) -> None:
        from matplotlib import colormaps

        self.width = width - width % 2
        self.height = height - height % 2
        self.duration = len(audio) / sample_rate
        self.time_offset = time_offset
        self.trail_seconds = trail_seconds
        self.orbit_seconds = orbit_seconds
        self.show_axes = show_axes
        self.times = features.times
        self.active = trajectory.active
        self.variance = trajectory.explained_variance_ratio

        strip_h = int(self.height * _STRIP_FRACTION) if spectrogram else 0
        strip_h -= strip_h % 2
        self.plot_h = self.height - strip_h
        self.strip = None
        if strip_h:
            self.strip = _FrameRenderer(
                audio,
                sample_rate,
                time_offset=time_offset,
                viz_type="mel",
                motion="scroll",
                n_fft=2048,
                hop_length=512,
                n_mels=128,
                window="hann",
                fmin=0.0,
                fmax=None,
                db_range=80.0,
                cmap="magma",
                width=self.width,
                height=strip_h,
                window_seconds=window_seconds,
                playhead=0.5,
                waveform=False,
                show_axes=False,
                backend="auto",
            )

        # Centre on the middle of each axis's 2-98th percentile range and scale so
        # that range fills the view; a few outlying frames may leave it.
        xyz = np.nan_to_num(trajectory.coords)
        lo, hi = np.percentile(xyz[self.active], [2, 98], axis=0)
        span = float(((hi - lo) / 2).max()) or 1.0
        self.xyz = (xyz - (lo + hi) / 2) / span
        # A turning unit cube reaches sqrt(3) from the centre at its corners; keep
        # them inside the plot.
        self.scale = 0.28 * min(self.width, self.plot_h)

        try:
            lut = colormaps[cmap](np.linspace(0, 1, 256))[:, :3]
        except KeyError as e:
            raise InvalidInputError(f"Unknown colormap: {cmap!r}") from e
        log_c = np.log2(np.maximum(features.descriptors["centroid_hz"], 1.0))
        lo, hi = np.percentile(log_c[self.active], [3, 97])
        u = np.clip((log_c - lo) / max(hi - lo, 1e-9), 0, 1)
        self.rgb = lut[(u * 255).astype(int)]
        level = features.descriptors["level_db"]
        self.weight = 0.25 + 0.75 * np.clip(1 + level / trajectory.gate_db, 0, 1)
        self.dot_radius = max(3, self.plot_h // 140)
        self.stroke = 1 if self.plot_h < 900 else 2
        self._font = None
        self._sample_path()

    # -- projection ----------------------------------------------------------

    def _angle(self, t: float) -> float:
        if self.orbit_seconds is None:
            return 0.6  # a fixed three-quarter view
        return 0.6 + 2 * np.pi * t / self.orbit_seconds

    def _project(self, xyz: np.ndarray, t: float) -> tuple[np.ndarray, np.ndarray]:
        """Screen x, y for points in the unit cube; PC2 is up, the view turns about it."""
        a, tilt = self._angle(t), np.radians(_TILT_DEGREES)
        x = xyz[:, 0] * np.cos(a) + xyz[:, 2] * np.sin(a)
        z = -xyz[:, 0] * np.sin(a) + xyz[:, 2] * np.cos(a)
        y = xyz[:, 1] * np.cos(tilt) - z * np.sin(tilt)
        return self.width / 2 + x * self.scale, self.plot_h / 2 - y * self.scale

    # -- frame composition ---------------------------------------------------

    def frame(self, t: float) -> np.ndarray:
        """The frame at ``t`` seconds into the clip, shape ``(height, width, 3)`` uint8."""
        plot = self._plot(t)
        if self.strip is not None:
            plot = np.concatenate([plot, self.strip.frame(t)], axis=0)
        if self.show_axes:
            plot = self._draw_labels(plot, t)
        return np.ascontiguousarray(plot)

    def _sample_path(self) -> None:
        """Sample every segment once, in 3-D, about once per pixel it can cover.

        Projection never lengthens a segment, so its 3-D length times the scale
        bounds its length on screen. Points are in time order, so the ones heard by
        ``t`` (and, with a trail, not yet faded) are a contiguous slice.
        """
        seg = np.flatnonzero(self.active[:-1] & self.active[1:])  # i joins i + 1
        a, b = self.xyz[seg], self.xyz[seg + 1]
        steps = np.clip(np.ceil(np.linalg.norm(b - a, axis=1) * self.scale), 1, 400).astype(int)
        owner = np.repeat(np.arange(len(seg)), steps)
        frac = (np.arange(steps.sum()) - np.repeat(np.cumsum(steps) - steps, steps)) / (
            np.repeat(steps, steps)
        )
        self.points = a[owner] + (b - a)[owner] * frac[:, None]
        src = seg[owner]
        self.point_start = self.times[src]  # when the segment's first frame sounds
        self.point_end = self.times[src + 1]  # drawn once its second frame has sounded
        self.point_rgb = self.rgb[src] * self.weight[src][:, None]
        offsets = [(dx, dy) for dx in range(self.stroke) for dy in range(self.stroke)]
        self.offsets = np.array(offsets)

    def _plot(self, t: float) -> np.ndarray:
        w, h = self.width, self.plot_h
        hi = int(np.searchsorted(self.point_end, t, side="right"))
        lo = 0
        if self.trail_seconds is not None:
            lo = int(np.searchsorted(self.point_start, t - self.trail_seconds, side="left"))
        sx, sy = self._project(self.xyz, t)
        buf = np.zeros(h * w * 3)
        if hi > lo:
            px, py = self._project(self.points[lo:hi], t)
            age = t - self.point_start[lo:hi]
            alpha = 0.45 + 0.55 * np.exp(-age / _RECENT_SECONDS)
            if self.trail_seconds is not None:
                alpha = alpha * np.clip(1 - age / self.trail_seconds, 0, 1)
            colour = self.point_rgb[lo:hi] * alpha[:, None]  # (n, 3)
            ix = np.rint(px).astype(int)[:, None] + self.offsets[:, 0]
            iy = np.rint(py).astype(int)[:, None] + self.offsets[:, 1]
            ok = (ix >= 0) & (ix < w) & (iy >= 0) & (iy < h)
            pixel = (iy * w + ix)[ok]  # one row of `ok` per point, one column per offset
            weights = np.broadcast_to(colour[:, None, :], (*ok.shape, 3))[ok]  # (m, 3)
            index = (pixel[:, None] * 3 + np.arange(3)).ravel()
            buf += np.bincount(index, weights=weights.ravel(), minlength=h * w * 3)
        glow = 1 - np.exp(-1.6 * buf)  # additive strokes saturate softly
        img = (glow.reshape(h, w, 3) * 255).astype(np.uint8)
        img = np.maximum(img, _BACKGROUND)

        current = int(np.searchsorted(self.times, t, side="right")) - 1
        if current >= 0 and self.active[current]:
            self._dot(img, int(sx[current]), int(sy[current]))
        return img

    def _dot(self, img: np.ndarray, cx: int, cy: int) -> None:
        r = self.dot_radius
        yy, xx = np.ogrid[-r : r + 1, -r : r + 1]
        disk = xx**2 + yy**2 <= r * r
        y0, y1 = max(cy - r, 0), min(cy + r + 1, img.shape[0])
        x0, x1 = max(cx - r, 0), min(cx + r + 1, img.shape[1])
        if y0 >= y1 or x0 >= x1:
            return
        mask = disk[y0 - (cy - r) : y1 - (cy - r), x0 - (cx - r) : x1 - (cx - r)]
        img[y0:y1, x0:x1][mask] = 255

    # -- labels (Pillow) -----------------------------------------------------

    def _draw_labels(self, frame: np.ndarray, t: float) -> np.ndarray:
        from PIL import Image, ImageDraw

        if self._font is None:
            self._font = label_font(self.height)
        image = Image.fromarray(frame)
        draw = ImageDraw.Draw(image)
        dim, shadow = (150, 150, 165), (0, 0, 0)

        # The three axes, from the origin, labelled with their share of the variance.
        ends = np.vstack([np.zeros(3), np.eye(3) * 0.9])
        ex, ey = self._project(ends, t)
        for k in range(3):
            draw.line([(ex[0], ey[0]), (ex[k + 1], ey[k + 1])], fill=(70, 70, 85), width=1)
            share = self.variance[k] if k < len(self.variance) else 0.0
            label = f"PC{k + 1} {share:.0%}"
            at = (ex[k + 1] + 4, ey[k + 1])
            draw.text((at[0] + 1, at[1] + 1), label, font=self._font, fill=shadow, anchor="lm")
            draw.text(at, label, font=self._font, fill=dim, anchor="lm")

        draw_time_stamp(
            draw, self._font, self.width, self.time_offset + t, self.time_offset + self.duration
        )
        title = "MFCC-PCA timbre trajectory · colour: spectral centroid"
        draw.text((MARGIN, MARGIN), title, font=self._font, fill=dim, anchor="lt")
        return np.asarray(image)


__all__ = ["TrajectoryVideoResult", "generate_trajectory_video"]
