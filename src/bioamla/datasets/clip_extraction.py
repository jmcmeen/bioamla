"""Extract audio clips from time-frequency annotations."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from bioamla.datasets.annotations import Annotation
from bioamla.exceptions import AnnotationError, NotFoundError

logger = logging.getLogger(__name__)


def extract_audio_clips(
    annotations: list[Annotation],
    audio_path: str,
    output_dir: str,
    padding_ms: float = 0.0,
    format: str = "wav",
    include_label_in_filename: bool = True,
) -> dict[str, Any]:
    """Extract one audio clip per annotation from a source audio file.

    Args:
        annotations: Annotations to extract.
        audio_path: Path to the source audio file.
        output_dir: Directory for output clips (created if missing).
        padding_ms: Padding in milliseconds added before/after each clip.
        format: Output audio file extension (wav, flac, ...).
        include_label_in_filename: Include the annotation label in the filename.

    Returns:
        Dict with keys ``total_clips``, ``extracted_clips`` (list of paths),
        ``failed_clips`` (list of error strings), and ``output_directory``.

    Raises:
        NotFoundError: If the source audio file doesn't exist.
        AnnotationError: If the audio file cannot be loaded.
    """
    if not Path(audio_path).exists():
        raise NotFoundError(f"Audio file not found: {audio_path}")

    from bioamla.audio import load_audio, save_audio

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

    extracted_clips: list[str] = []
    failed_clips: list[str] = []

    for i, ann in enumerate(annotations):
        try:
            start_sample = max(0, int(ann.start_time * sample_rate) - padding_samples)
            end_sample = min(total_samples, int(ann.end_time * sample_rate) + padding_samples)

            clip = audio_data[start_sample:end_sample]

            if include_label_in_filename and ann.label:
                safe_label = ann.label.replace(" ", "_").replace("/", "-")
                filename = f"{audio_stem}_{i:04d}_{safe_label}.{format}"
            else:
                filename = f"{audio_stem}_{i:04d}.{format}"

            clip_path = output_path / filename
            save_audio(str(clip_path), clip, sample_rate)
            extracted_clips.append(str(clip_path))
        except Exception as e:
            logger.warning(f"Failed to extract clip {i}: {e}")
            failed_clips.append(f"Clip {i}: {e}")

    return {
        "total_clips": len(annotations),
        "extracted_clips": extracted_clips,
        "failed_clips": failed_clips,
        "output_directory": str(output_path),
    }
