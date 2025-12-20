"""
Audio Playback
==============

Audio playback functionality using sounddevice for real-time audio output.
Provides play, pause, stop, and seek capabilities.

Note: Requires sounddevice to be installed: pip install sounddevice
"""

import logging
import threading
from dataclasses import dataclass
from enum import Enum
from typing import Callable, Optional, Union

import numpy as np

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


class AudioPlayer:
    """
    Audio player with play, pause, stop, and seek functionality.

    This class provides a simple interface for playing audio through the
    system's audio output using sounddevice. It supports:

    - Play/pause/stop controls
    - Seeking to specific positions (by time or sample)
    - Looping
    - Callbacks for playback events

    Example:
        >>> player = AudioPlayer()
        >>> audio, sr = load_audio("audio.wav")
        >>> player.load(audio, sr)
        >>> player.play()
        >>> player.pause()
        >>> player.seek(2.5)  # Seek to 2.5 seconds
        >>> player.play()
        >>> player.stop()

    Note:
        Requires sounddevice to be installed: pip install sounddevice
    """

    def __init__(self):
        """Initialize the audio player."""
        self._audio: Optional[np.ndarray] = None
        self._sample_rate: int = 44100
        self._position: int = 0
        self._state: PlaybackState = PlaybackState.STOPPED
        self._stream = None
        self._loop: bool = False
        self._lock = threading.Lock()
        self._on_complete: Optional[Callable[[], None]] = None
        self._on_position_change: Optional[Callable[[PlaybackPosition], None]] = None

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
        on_complete: Optional[Callable[[], None]] = None,
        on_position_change: Optional[Callable[[PlaybackPosition], None]] = None,
    ) -> None:
        """
        Load audio data for playback.

        Args:
            audio: Audio data as numpy array (mono or stereo)
            sample_rate: Sample rate in Hz
            on_complete: Optional callback when playback completes
            on_position_change: Optional callback for position updates
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
        on_complete: Optional[Callable[[], None]] = None,
        on_position_change: Optional[Callable[[PlaybackPosition], None]] = None,
    ) -> None:
        """
        Load audio from a file for playback.

        Args:
            filepath: Path to the audio file
            on_complete: Optional callback when playback completes
            on_position_change: Optional callback for position updates
        """
        from bioamla.utils.audio_utils import load_audio

        audio, sr = load_audio(filepath, mono=False)
        self.load(audio, sr, on_complete, on_position_change)

    def play(self, loop: bool = False) -> None:
        """
        Start or resume playback.

        Args:
            loop: If True, loop the audio continuously
        """
        if self._audio is None:
            raise RuntimeError("No audio loaded. Call load() first.")

        try:
            import sounddevice as sd
        except ImportError:
            raise ImportError(
                "sounddevice is required for audio playback. "
                "Install it with: pip install sounddevice"
            )

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

    def seek(self, time_or_sample: Union[float, int], by_sample: bool = False) -> None:
        """
        Seek to a specific position.

        Args:
            time_or_sample: Position to seek to (seconds or sample index)
            by_sample: If True, interpret position as sample index.
                       If False (default), interpret as time in seconds.
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

    def _audio_callback(self, outdata, frames, time_info, status):
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

    def _finished_callback(self):
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
_global_player: Optional[AudioPlayer] = None


def play_audio(
    audio_or_filepath: Union[np.ndarray, str],
    sample_rate: Optional[int] = None,
    loop: bool = False,
    block: bool = False,
) -> AudioPlayer:
    """
    Play audio data or file.

    This is a convenience function that creates or reuses a global AudioPlayer.
    For more control, create your own AudioPlayer instance.

    Args:
        audio_or_filepath: Either a numpy array of audio data or path to audio file
        sample_rate: Sample rate (required if audio_or_filepath is numpy array)
        loop: If True, loop the audio continuously
        block: If True, block until playback completes

    Returns:
        The AudioPlayer instance (can be used to control playback)

    Example:
        >>> # Play a file
        >>> player = play_audio("audio.wav")
        >>> # Later, stop it
        >>> player.stop()

        >>> # Play numpy array
        >>> audio, sr = load_audio("audio.wav")
        >>> play_audio(audio, sr)
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
        try:
            import sounddevice as sd

            while _global_player.is_playing:
                sd.sleep(100)
        except KeyboardInterrupt:
            _global_player.stop()

    return _global_player


def stop_audio() -> None:
    """
    Stop the global audio player.

    This stops any audio playing via the play_audio() convenience function.
    """
    global _global_player
    if _global_player is not None:
        _global_player.stop()
