"""Spectrogram compute backend selection (librosa CPU / torch GPU).

Visualization computes on the CPU with librosa by default. When a CUDA GPU is
available, the STFT can be computed on the GPU for a speedup — useful for batch
spectrogram generation. This is purely an accelerator: ``select_backend("auto")``
silently falls back to librosa when no GPU is present and never raises.

The torch path mirrors the librosa computation (``torch.stft`` magnitude, then
the same ``librosa.filters.mel`` filterbank) so output matches the CPU path
closely enough that callers can treat the two interchangeably.
"""

from __future__ import annotations

import logging

import librosa
import numpy as np

logger = logging.getLogger(__name__)

Backend = str  # "librosa" | "torch"


def _torch_gpu_available() -> bool:
    """True only if torch is importable and a CUDA device is present."""
    try:
        import torch
    except ImportError:
        return False
    try:
        return bool(torch.cuda.is_available())
    except Exception:  # pragma: no cover - defensive: torch present but CUDA query fails
        return False


def select_backend(prefer: str = "auto") -> Backend:
    """Resolve the spectrogram backend.

    Args:
        prefer: ``"auto"`` (torch iff a CUDA GPU is available, else librosa),
            ``"librosa"`` (always CPU), or ``"torch"`` (force GPU/torch path).

    Returns:
        ``"librosa"`` or ``"torch"``.

    """
    if prefer == "librosa":
        return "librosa"
    if prefer == "torch":
        import torch  # noqa: F401
        return "torch"
    # auto
    return "torch" if _torch_gpu_available() else "librosa"


def stft_magnitude(
    audio: np.ndarray,
    n_fft: int,
    hop_length: int,
    window: np.ndarray,
    backend: Backend = "librosa",
) -> np.ndarray:
    """Magnitude STFT, shape ``(n_fft // 2 + 1, frames)``.

    The torch path runs ``torch.stft`` on the GPU when available; on any failure
    it logs and falls back to librosa so visualization never breaks.
    """
    if backend == "torch":
        try:
            return _torch_stft_magnitude(audio, n_fft, hop_length, window)
        except Exception as e:  # pragma: no cover - exercised only with torch installed
            logger.debug("torch STFT failed (%s); falling back to librosa", e)
    return np.abs(librosa.stft(y=audio, n_fft=n_fft, hop_length=hop_length, window=window))


def mel_power(
    audio: np.ndarray,
    sample_rate: int,
    n_fft: int,
    hop_length: int,
    n_mels: int,
    window: np.ndarray,
    fmin: float = 0.0,
    fmax: float | None = None,
    backend: Backend = "librosa",
) -> np.ndarray:
    """Mel power spectrogram, shape ``(n_mels, frames)``.

    Uses the backend-selected STFT, then the librosa mel filterbank (identical
    on both backends, so only the STFT numerics differ).
    """
    if fmax is None:
        fmax = sample_rate / 2
    power = stft_magnitude(audio, n_fft, hop_length, window, backend) ** 2
    mel_filter = librosa.filters.mel(
        sr=sample_rate, n_fft=n_fft, n_mels=n_mels, fmin=fmin, fmax=fmax
    )
    return mel_filter @ power


def _torch_stft_magnitude(
    audio: np.ndarray, n_fft: int, hop_length: int, window: np.ndarray
) -> np.ndarray:
    """Compute |STFT| with torch, on CUDA when available."""
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    y = torch.from_numpy(np.ascontiguousarray(audio)).float().to(device)
    win = torch.from_numpy(np.ascontiguousarray(window)).float().to(device)
    spec = torch.stft(
        y,
        n_fft=n_fft,
        hop_length=hop_length,
        window=win,
        center=True,
        return_complex=True,
    )
    return spec.abs().cpu().numpy()
