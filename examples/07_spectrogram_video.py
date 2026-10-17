"""Workflow 7, from Python: render spectrogram videos with the library API.

The CLI (see 07_spectrogram_video.sh) covers the common cases; call the
function directly when you want a progress callback, the result as data, or
to render many files in a loop.

Needs: a recording (RECORDING below) and the ffmpeg command on PATH.

    uv run python examples/07_spectrogram_video.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from bioamla.exceptions import BioamlaError
from bioamla.viz import generate_spectrogram_video

OUT = Path("out/07_spectrogram_video")
RECORDING = Path(os.environ.get("RECORDING", OUT / "recording.wav"))  # <-- point at your file


def progress(done: int, total: int) -> None:
    """Called after every frame; here a one-line counter."""
    if done % 30 == 0 or done == total:
        print(f"\r  {done}/{total} frames", end="", flush=True)


def main() -> int:
    if not RECORDING.is_file():
        print(f"Set RECORDING to a real audio file (got {RECORDING}).", file=sys.stderr)
        return 1
    OUT.mkdir(parents=True, exist_ok=True)

    # One video per style. Each entry is just keyword arguments to the function.
    styles = {
        "mel_scroll": {},
        "stft_pages": {
            "viz_type": "stft",
            "motion": "page",
            "window_seconds": 8,
            "cmap": "viridis",
        },
        "waveform_whole": {"viz_type": "waveform", "motion": "page", "window_seconds": None},
        "portrait": {"width": 1080, "height": 1920, "window_seconds": 3, "playhead": 1.0},
    }

    for name, kwargs in styles.items():
        out = OUT / f"{RECORDING.stem}_{name}.mp4"
        print(f"{name}:")
        try:
            result = generate_spectrogram_video(RECORDING, out, on_progress=progress, **kwargs)
        except BioamlaError as e:  # missing ffmpeg, unreadable audio, bad argument
            print(f"\n  failed: {e}", file=sys.stderr)
            return 1
        print(
            f"\n  wrote {result.output_path}: {result.frames} frames, "
            f"{result.width}x{result.height}, {result.duration:.1f}s"
        )

    # The result is a dataclass; to_dict() gives something you can log or JSON-dump.
    print(result.to_dict())
    return 0


if __name__ == "__main__":
    sys.exit(main())
