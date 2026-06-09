"""Extract audio clips from time-frequency annotations."""

from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

from bioamla.common.files import sanitize_filename
from bioamla.datasets.annotations import Annotation
from bioamla.exceptions import AnnotationError, NotFoundError

logger = logging.getLogger(__name__)


def _apply_per_channel(clip: np.ndarray, fn: Callable[..., np.ndarray], *args: Any) -> np.ndarray:
    """Apply a 1-D transform to each channel column of a ``(samples, channels)`` clip."""
    channels = [fn(clip[:, c], *args) for c in range(clip.shape[1])]
    return np.stack(channels, axis=1)


def extract_audio_clips(
    annotations: list[Annotation],
    audio_path: str,
    output_dir: str,
    padding_ms: float = 0.0,
    format: str = "wav",
    include_label_in_filename: bool = True,
    subdir_by_label: bool = False,
    bandpass: bool = False,
    target_sample_rate: int | None = None,
) -> dict[str, Any]:
    """Extract one audio clip per annotation from a source audio file.

    Args:
        annotations: Annotations to extract.
        audio_path: Path to the source audio file.
        output_dir: Directory for output clips (created if missing).
        padding_ms: Padding in milliseconds added before/after each clip.
        format: Output audio file extension (wav, flac, ...).
        include_label_in_filename: Include the annotation label in the filename
            (ignored when ``subdir_by_label`` is set).
        subdir_by_label: Place each clip in a ``<label>/`` subdirectory
            (AudioFolder layout).
        bandpass: When an annotation has both ``low_freq`` and ``high_freq``,
            bandpass-filter the clip to that band.
        target_sample_rate: Resample each clip to this rate (e.g. 16000 for AST).

    Returns:
        Dict with ``total_clips``, ``extracted_clips`` (list of paths),
        ``failed_clips`` (list of error strings), ``output_directory``, and
        ``clips`` — a list of per-clip record dicts (file_name relative to
        ``output_dir``, label, source_file, start_time, end_time, low_freq,
        high_freq, confidence, channel, sample_rate, duration).

    Raises:
        NotFoundError: If the source audio file doesn't exist.
        AnnotationError: If the audio file cannot be loaded.
    """
    if not Path(audio_path).exists():
        raise NotFoundError(f"Audio file not found: {audio_path}")

    from bioamla.audio import bandpass_filter, load_audio, resample_audio, save_audio

    try:
        audio_data, sample_rate = load_audio(audio_path)
    except Exception as e:
        raise AnnotationError(f"Failed to load audio {audio_path}: {e}") from e

    if audio_data.ndim == 1:
        audio_data = audio_data.reshape(-1, 1)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    padding_samples = int(padding_ms * sample_rate / 1000)
    total_samples = len(audio_data)
    audio_stem = Path(audio_path).stem
    source_name = Path(audio_path).name

    extracted_clips: list[str] = []
    failed_clips: list[str] = []
    clips: list[dict[str, Any]] = []

    for i, ann in enumerate(annotations):
        try:
            start_sample = max(0, int(ann.start_time * sample_rate) - padding_samples)
            end_sample = min(total_samples, int(ann.end_time * sample_rate) + padding_samples)

            clip = audio_data[start_sample:end_sample]
            clip_sr = sample_rate

            if bandpass and ann.low_freq is not None and ann.high_freq is not None:
                clip = _apply_per_channel(
                    clip, bandpass_filter, clip_sr, ann.low_freq, ann.high_freq
                )

            if target_sample_rate is not None and target_sample_rate != clip_sr:
                clip = _apply_per_channel(clip, resample_audio, clip_sr, target_sample_rate)
                clip_sr = target_sample_rate

            label_dir = ""
            if subdir_by_label:
                label_dir = sanitize_filename(ann.label) if ann.label else "unknown"
                dest_dir = output_path / label_dir
                dest_dir.mkdir(parents=True, exist_ok=True)
                filename = f"{audio_stem}_{i:04d}.{format}"
            else:
                dest_dir = output_path
                if include_label_in_filename and ann.label:
                    safe_label = ann.label.replace(" ", "_").replace("/", "-")
                    filename = f"{audio_stem}_{i:04d}_{safe_label}.{format}"
                else:
                    filename = f"{audio_stem}_{i:04d}.{format}"

            clip_path = dest_dir / filename
            save_audio(str(clip_path), clip, clip_sr)
            extracted_clips.append(str(clip_path))

            rel_name = f"{label_dir}/{filename}" if label_dir else filename
            clips.append(
                {
                    "file_name": rel_name,
                    "label": ann.label,
                    "source_file": source_name,
                    "start_time": ann.start_time,
                    "end_time": ann.end_time,
                    "low_freq": ann.low_freq if ann.low_freq is not None else "",
                    "high_freq": ann.high_freq if ann.high_freq is not None else "",
                    "confidence": ann.confidence if ann.confidence is not None else "",
                    "channel": ann.channel,
                    "sample_rate": clip_sr,
                    "duration": round(ann.duration, 6),
                }
            )
        except Exception as e:
            logger.warning(f"Failed to extract clip {i}: {e}")
            failed_clips.append(f"Clip {i}: {e}")

    return {
        "total_clips": len(annotations),
        "extracted_clips": extracted_clips,
        "failed_clips": failed_clips,
        "output_directory": str(output_path),
        "clips": clips,
    }
