"""
bioamla.audio — the foundational audio domain.

Data container (:class:`AudioData`), file I/O, discovery, analysis, signal
processing, and playback. Heavy/optional backends are imported lazily inside
the functions that need them (torchaudio for waveform tensors, sounddevice for
playback), so this package is importable on a slim install.
"""

from bioamla.audio.analysis import (
    AmplitudeStats,
    FrequencyStats,
    SilenceInfo,
    calculate_dbfs,
    calculate_peak,
    calculate_rms,
    detect_silence,
    get_amplitude_stats,
    get_frequency_stats,
    get_peak_frequency,
    is_silent,
)
from bioamla.audio.batch import (
    batch_convert_files,
    batch_resample_files,
    batch_transform_files,
    segment_audio_file,
)
from bioamla.audio.convert import convert_audio_file
from bioamla.audio.data import (
    AudioData,
    AudioMetadata,
    ProcessedAudio,
)
from bioamla.audio.discovery import (
    get_audio_files,
    get_wav_metadata,
    list_audio_files,
)
from bioamla.audio.info import (
    AudioAnalysis,
    AudioInfo,
    analyze_audio,
    analyze_audio_batch,
    get_audio_info,
    get_channels,
    get_duration,
    get_sample_rate,
    summarize_analysis,
)
from bioamla.audio.io import (
    batch_process,
    create_temp_audio_file,
    load_audio,
    load_audio_data,
    load_waveform_tensor,
    process_file,
    save_audio,
    save_audio_data,
    save_audio_data_as,
)
from bioamla.audio.playback import (
    AudioPlayer,
    PlaybackPosition,
    PlaybackState,
    play_audio,
    stop_audio,
)
from bioamla.audio.processing import (
    AudioEvent,
    AudioSegment,
    bandpass_filter,
    detect_onsets,
    highpass_filter,
    lowpass_filter,
    normalize_loudness,
    peak_normalize,
    resample_audio,
    segment_on_silence,
    spectral_denoise,
    split_audio_on_silence,
    trim_audio,
    trim_silence,
)
from bioamla.audio.torchaudio import (
    resample_waveform_tensor,
    split_waveform_tensor,
)

__all__ = [
    # Data containers
    "AudioData",
    "AudioMetadata",
    "ProcessedAudio",
    # File I/O (AudioData-level)
    "load_audio_data",
    "save_audio_data",
    "save_audio_data_as",
    "create_temp_audio_file",
    # File I/O (array-level)
    "load_audio",
    "save_audio",
    "load_waveform_tensor",
    "resample_waveform_tensor",
    "split_waveform_tensor",
    "process_file",
    "batch_process",
    # Discovery
    "get_audio_files",
    "list_audio_files",
    "get_wav_metadata",
    # Info / analysis (file-level)
    "AudioInfo",
    "AudioAnalysis",
    "get_audio_info",
    "get_duration",
    "get_sample_rate",
    "get_channels",
    "analyze_audio",
    "analyze_audio_batch",
    "summarize_analysis",
    # Analysis (array-level)
    "AmplitudeStats",
    "FrequencyStats",
    "SilenceInfo",
    "calculate_rms",
    "calculate_dbfs",
    "calculate_peak",
    "get_amplitude_stats",
    "get_peak_frequency",
    "get_frequency_stats",
    "detect_silence",
    "is_silent",
    # Signal processing
    "AudioSegment",
    "AudioEvent",
    "bandpass_filter",
    "lowpass_filter",
    "highpass_filter",
    "spectral_denoise",
    "segment_on_silence",
    "split_audio_on_silence",
    "detect_onsets",
    "normalize_loudness",
    "peak_normalize",
    "resample_audio",
    "trim_audio",
    "trim_silence",
    # Playback
    "PlaybackState",
    "PlaybackPosition",
    "AudioPlayer",
    "play_audio",
    "stop_audio",
    # Batch
    "batch_transform_files",
    "batch_resample_files",
    "batch_convert_files",
    "segment_audio_file",
    # Conversion
    "convert_audio_file",
]
