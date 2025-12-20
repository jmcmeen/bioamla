"""
Audio Package
=============

Audio processing domain for bioacoustic analysis.

This package provides core audio functionality including:
- Audio analysis and metadata extraction
- Signal processing operations
- Audio augmentation for training
- Audio I/O via torchaudio
- Audio playback utilities
"""

from bioamla.core.audio.audio import (
    AmplitudeStats,
    AudioAnalysis,
    AudioInfo,
    FrequencyStats,
    SilenceInfo,
    analyze_audio,
    detect_silence,
    get_audio_info,
    summarize_analysis,
)

__all__ = [
    # audio.py
    "AudioInfo",
    "AmplitudeStats",
    "FrequencyStats",
    "SilenceInfo",
    "AudioAnalysis",
    "get_audio_info",
    "analyze_audio",
    "detect_silence",
    "summarize_analysis",
]
