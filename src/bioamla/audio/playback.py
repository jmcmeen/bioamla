"""
Audio Playback
==============

Real-time audio playback via ``sounddevice``, folded from
``services/audio_playback.py``. The :class:`AudioPlayer` state machine,
threading, and position logic are unchanged.

``sounddevice`` is an optional dependency. It is imported lazily inside the
methods that need it; if it is missing a :class:`DependencyError` is raised
telling the user to install ``bioamla[playback]``.
"""

import logging
import threading
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from typing import Any

import numpy as np

from bioamla.exceptions import DependencyError

logger = logging.getLogger(__name__)


class PlaybackState(Enum):
    """Enumeration of playback states."""

    STOPPED = "stopped"
    PLAYING = "playing"
    PAUSED = "paused"


@dataclass
class PlaybackPosition:
    """Current playback position information."""

    current_sample: int
    current_time: float
    total_samples: int
    total_time: float
    progress: float  # 0.0 to 1.0


def _import_sounddevice() -> Any:
    """Import sounddevice lazily, raising DependencyError if it is missing."""
    try:
        import sounddevice as sd
    except ImportError as err:
        raise DependencyError(
            "audio playback requires sounddevice — install bioamla[playback]"
        ) from err
    return sd


class AudioPlayer:
    """
    Audio player with play, pause, stop, and seek functionality.

    Provides a simple interface for playing audio through the system's audio
    output using ``sounddevice``. Supports play/pause/stop controls, seeking
    (by time or sample), looping, and playback-event callbacks.

    Note:
        Requires the optional ``sounddevice`` dependency
        (``pip install bioamla[playback]``).
    """

    def __init__(self) -> None:
        """Initialize the audio player."""
        self._audio: np.ndarray | None = None
        self._sample_rate: int = 44100
        self._position: int = 0
        self._state: PlaybackState = PlaybackState.STOPPED
        self._stream = None
        self._loop: bool = False
        self._lock = threading.Lock()
        self._on_complete: Callable[[], None] | None = None
        self._on_position_change: Callable[[PlaybackPosition], None] | None = None

    @property
    def state(self) -> PlaybackState:
        """Get the current playback state."""
        return self._state

    @property
    def is_playing(self) -> bool:
        """Check if audio is currently playing."""
        return self._state == PlaybackState.PLAYING

    @property
    def is_paused(self) -> bool:
        """Check if playback is paused."""
        return self._state == PlaybackState.PAUSED

    @property
    def is_stopped(self) -> bool:
        """Check if playback is stopped."""
        return self._state == PlaybackState.STOPPED

    @property
    def position(self) -> PlaybackPosition:
        """Get the current playback position."""
        with self._lock:
            total_samples = len(self._audio) if self._audio is not None else 0
            total_time = total_samples / self._sample_rate if self._sample_rate > 0 else 0
            current_time = self._position / self._sample_rate if self._sample_rate > 0 else 0
            progress = self._position / total_samples if total_samples > 0 else 0

            return PlaybackPosition(
                current_sample=self._position,
                current_time=current_time,
                total_samples=total_samples,
                total_time=total_time,
                progress=progress,
            )

    @property
    def duration(self) -> float:
        """Get the total duration in seconds."""
        if self._audio is None:
            return 0.0
        return len(self._audio) / self._sample_rate

    def load(
        self,
        audio: np.ndarray,
        sample_rate: int,
        on_complete: Callable[[], None] | None = None,
        on_position_change: Callable[[PlaybackPosition], None] | None = None,
    ) -> None:
        """
        Load audio data for playback.

        Args:
            audio: Audio data as numpy array (mono or stereo).
            sample_rate: Sample rate in Hz.
            on_complete: Optional callback when playback completes.
            on_position_change: Optional callback for position updates.

        Raises:
            ValueError: If the audio array has an unexpected shape.
        """
        self.stop()

        with self._lock:
            # Ensure audio is 1D (mono) or 2D with shape (samples, channels)
            if audio.ndim == 1:
                self._audio = audio.astype(np.float32)
            elif audio.ndim == 2:
                # If shape is (channels, samples), transpose
                if audio.shape[0] < audio.shape[1]:
                    audio = audio.T
                self._audio = audio.astype(np.float32)
            else:
                raise ValueError(f"Unexpected audio shape: {audio.shape}")

            self._sample_rate = sample_rate
            self._position = 0
            self._on_complete = on_complete
            self._on_position_change = on_position_change

    def load_file(
        self,
        filepath: str,
        on_complete: Callable[[], None] | None = None,
        on_position_change: Callable[[PlaybackPosition], None] | None = None,
    ) -> None:
        """
        Load audio from a file for playback.

        Args:
            filepath: Path to the audio file.
            on_complete: Optional callback when playback completes.
            on_position_change: Optional callback for position updates.
        """
        from bioamla.adapters.pydub import load_audio

        audio, sr = load_audio(filepath)
        self.load(audio, sr, on_complete, on_position_change)

    def play(self, loop: bool = False) -> None:
        """
        Start or resume playback.

        Args:
            loop: If True, loop the audio continuously.

        Raises:
            RuntimeError: If no audio has been loaded.
            DependencyError: If ``sounddevice`` is not installed.
        """
        if self._audio is None:
            raise RuntimeError("No audio loaded. Call load() first.")

        sd = _import_sounddevice()

        with self._lock:
            if self._state == PlaybackState.PLAYING:
                return

            self._loop = loop
            self._state = PlaybackState.PLAYING

            # Determine number of channels
            if self._audio.ndim == 1:
                channels = 1
            else:
                channels = self._audio.shape[1]

            # Create output stream with callback
            self._stream = sd.OutputStream(
                samplerate=self._sample_rate,
                channels=channels,
                callback=self._audio_callback,
                finished_callback=self._finished_callback,
                dtype=np.float32,
            )
            self._stream.start()

    def pause(self) -> None:
        """Pause playback. Can be resumed with play()."""
        with self._lock:
            if self._state == PlaybackState.PLAYING:
                self._state = PlaybackState.PAUSED
                if self._stream is not None:
                    self._stream.stop()
                    self._stream.close()
                    self._stream = None

    def stop(self) -> None:
        """Stop playback and reset position to the beginning."""
        with self._lock:
            self._state = PlaybackState.STOPPED
            self._position = 0
            if self._stream is not None:
                self._stream.stop()
                self._stream.close()
                self._stream = None

    def seek(self, time_or_sample: float | int, by_sample: bool = False) -> None:
        """
        Seek to a specific position.

        Args:
            time_or_sample: Position to seek to (seconds or sample index).
            by_sample: If True, interpret position as sample index. If False
                (default), interpret as time in seconds.
        """
        if self._audio is None:
            return

        with self._lock:
            if by_sample:
                new_position = int(time_or_sample)
            else:
                new_position = int(time_or_sample * self._sample_rate)

            # Clamp to valid range
            self._position = max(0, min(new_position, len(self._audio)))

    def _audio_callback(self, outdata: np.ndarray, frames: int, time_info: Any, status: Any) -> None:
        """Callback for sounddevice output stream."""
        import sounddevice as sd

        if status:
            logger.warning(f"Playback status: {status}")

        with self._lock:
            if self._state != PlaybackState.PLAYING or self._audio is None:
                outdata.fill(0)
                return

            # Get audio chunk
            start = self._position
            end = start + frames
            total_samples = len(self._audio)

            if start >= total_samples:
                if self._loop:
                    self._position = 0
                    start = 0
                    end = frames
                else:
                    outdata.fill(0)
                    raise sd.CallbackStop()

            # Handle end of audio
            if end > total_samples:
                if self._loop:
                    # Wrap around for looping
                    first_part = self._audio[start:total_samples]
                    second_part_len = end - total_samples
                    second_part = self._audio[:second_part_len]

                    if self._audio.ndim == 1:
                        chunk = np.concatenate([first_part, second_part])
                        outdata[:, 0] = chunk[:frames]
                    else:
                        chunk = np.vstack([first_part, second_part])
                        outdata[:] = chunk[:frames]

                    self._position = second_part_len
                else:
                    # Fill with what we have, pad with zeros
                    available = total_samples - start
                    if self._audio.ndim == 1:
                        outdata[:available, 0] = self._audio[start:total_samples]
                        outdata[available:] = 0
                    else:
                        outdata[:available] = self._audio[start:total_samples]
                        outdata[available:] = 0

                    self._position = total_samples
                    raise sd.CallbackStop()
            else:
                # Normal case: enough audio available
                if self._audio.ndim == 1:
                    outdata[:, 0] = self._audio[start:end]
                else:
                    outdata[:] = self._audio[start:end]

                self._position = end

        # Notify position change
        if self._on_position_change is not None:
            try:
                self._on_position_change(self.position)
            except Exception as e:
                logger.warning(f"Position callback error: {e}")

    def _finished_callback(self) -> None:
        """Callback when playback completes."""
        with self._lock:
            if not self._loop:
                self._state = PlaybackState.STOPPED
                self._position = 0

        if self._on_complete is not None and not self._loop:
            try:
                self._on_complete()
            except Exception as e:
                logger.warning(f"Completion callback error: {e}")


# =============================================================================
# Convenience Functions
# =============================================================================

# Global player instance for simple playback
_global_player: AudioPlayer | None = None


def play_audio(
    audio_or_filepath: np.ndarray | str,
    sample_rate: int | None = None,
    loop: bool = False,
    block: bool = False,
) -> AudioPlayer:
    """
    Play audio data or a file.

    Convenience function that creates or reuses a global :class:`AudioPlayer`.
    For more control, create your own :class:`AudioPlayer` instance.

    Args:
        audio_or_filepath: Either a numpy array of audio data or a path to an
            audio file.
        sample_rate: Sample rate (required if ``audio_or_filepath`` is an array).
        loop: If True, loop the audio continuously.
        block: If True, block until playback completes.

    Returns:
        The :class:`AudioPlayer` instance (can be used to control playback).

    Raises:
        ValueError: If a numpy array is given without a sample rate.
        DependencyError: If ``sounddevice`` is not installed.
    """
    global _global_player

    if _global_player is None:
        _global_player = AudioPlayer()
    else:
        _global_player.stop()

    if isinstance(audio_or_filepath, str):
        _global_player.load_file(audio_or_filepath)
    else:
        if sample_rate is None:
            raise ValueError("sample_rate is required when playing numpy array")
        _global_player.load(audio_or_filepath, sample_rate)

    _global_player.play(loop=loop)

    if block:
        sd = _import_sounddevice()
        try:
            while _global_player.is_playing:
                sd.sleep(100)
        except KeyboardInterrupt:
            _global_player.stop()

    return _global_player


def stop_audio() -> None:
    """Stop the global audio player started via :func:`play_audio`."""
    global _global_player
    if _global_player is not None:
        _global_player.stop()


__all__ = [
    "PlaybackState",
    "PlaybackPosition",
    "AudioPlayer",
    "play_audio",
    "stop_audio",
]
