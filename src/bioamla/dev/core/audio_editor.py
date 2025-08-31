"""
Core Audio Editor Module

This module provides comprehensive audio processing, editing, and analysis functionality
for the Bioamla audio editor application. It includes classes for managing audio data,
applying various processing operations, filtering, spectrogram generation, annotation
management, and audio playback.

The module is designed with a modular architecture:
- AudioData: Container class for audio data with history tracking and metadata
- AudioProcessor: Core processing operations (load, save, trim, gain, normalize, fade)
- AudioFilters: Digital signal processing filters (low-pass, high-pass, band-pass, notch)
- SpectrogramGenerator: Visualization tools for frequency domain analysis
- AnnotationManager: Tools for managing time-based audio annotations
- AudioPlayback: Real-time audio playback functionality

All processing operations maintain audio quality and provide undo functionality
through the AudioData history system. The module supports both mono and stereo
audio files and uses industry-standard libraries for audio processing.
"""
import numpy as np
import librosa
import soundfile as sf
from scipy import signal
from typing import Dict, List, Tuple, Optional, Any


class AudioData:
    """
    Container for audio data and metadata.
    
    This class encapsulates audio data along with its sample rate, file path,
    annotations, and processing history. It provides automatic state saving
    for undo functionality and convenient properties for audio metadata.
    
    Attributes:
        data (np.ndarray): Audio samples as numpy array
        sample_rate (int): Audio sample rate in Hz
        filepath (Optional[str]): Path to source audio file
        annotations (List[Dict[str, Any]]): List of time-based annotations
        history (List[np.ndarray]): History of audio data states for undo
    """
    
    def __init__(self, data: np.ndarray, sample_rate: int, filepath: Optional[str] = None):
        """
        Initialize AudioData container.
        
        Args:
            data (np.ndarray): Audio samples array
            sample_rate (int): Sample rate in Hz
            filepath (Optional[str]): Path to source file
        """
        self.data = data
        self.sample_rate = sample_rate
        self.filepath = filepath
        self.annotations: List[Dict[str, Any]] = []
        self.history: List[np.ndarray] = []
        self._save_state()
    
    def _save_state(self) -> None:
        """
        Save current audio state for undo functionality.
        
        Maintains a history of the last 20 audio states to enable
        multiple levels of undo operations.
        """
        self.history.append(self.data.copy())
        if len(self.history) > 20:  # Keep last 20 states
            self.history.pop(0)
    
    @property
    def duration(self) -> float:
        """Get duration in seconds."""
        return len(self.data) / self.sample_rate
    
    @property
    def channels(self) -> int:
        """Get number of channels."""
        return 1 if self.data.ndim == 1 else self.data.shape[1]
    
    def undo(self) -> bool:
        """
        Undo the last audio processing operation.
        
        Returns:
            bool: True if undo was successful, False if no history available
        """
        if len(self.history) > 1:
            self.history.pop()  # Remove current state
            self.data = self.history[-1].copy()
            return True
        return False


class AudioProcessor:
    """
    Main audio processing class providing core audio operations.
    
    This class contains static methods for fundamental audio processing tasks
    including file I/O, format conversion, trimming, gain adjustment, normalization,
    and fade effects. All methods are designed to work with AudioData objects
    and maintain audio quality throughout processing.
    """
    
    @staticmethod
    def load_audio(filepath: str, target_sr: Optional[int] = None) -> AudioData:
        """
        Load audio file and create AudioData object.
        
        Args:
            filepath (str): Path to audio file
            target_sr (Optional[int]): Target sample rate for resampling
        
        Returns:
            AudioData: Loaded audio data object
        
        Raises:
            ValueError: If file cannot be loaded
        """
        try:
            data, sr = librosa.load(filepath, sr=target_sr, mono=False)
            # Ensure 2D array for stereo handling
            if data.ndim == 1:
                data = data.reshape(-1, 1)
            else:
                data = data.T  # librosa returns (channels, samples), we want (samples, channels)
            
            return AudioData(data, sr, filepath)
        except Exception as e:
            raise ValueError(f"Could not load audio file {filepath}: {str(e)}")
    
    @staticmethod
    def save_audio(audio_data: AudioData, filepath: str, format: Optional[str] = None) -> None:
        """
        Save AudioData to file.
        
        Args:
            audio_data (AudioData): Audio data to save
            filepath (str): Output file path
            format (Optional[str]): Audio format (inferred from extension if None)
        
        Raises:
            ValueError: If file cannot be saved
        """
        try:
            # Handle mono/stereo
            data = audio_data.data
            if data.shape[1] == 1:
                data = data.flatten()
            else:
                data = data.T  # soundfile expects (channels, samples)
            
            sf.write(filepath, data, audio_data.sample_rate, format=format)
        except Exception as e:
            raise ValueError(f"Could not save audio file {filepath}: {str(e)}")
    
    @staticmethod
    def convert_format(input_path: str, output_path: str, format: str) -> None:
        """
        Convert audio file from one format to another.
        
        Args:
            input_path (str): Input file path
            output_path (str): Output file path
            format (str): Target audio format
        """
        audio_data = AudioProcessor.load_audio(input_path)
        AudioProcessor.save_audio(audio_data, output_path, format)
    
    @staticmethod
    def trim_audio(audio_data: AudioData, start_time: float, end_time: float) -> AudioData:
        """
        Trim audio to specified time range.
        
        Args:
            audio_data (AudioData): Source audio data
            start_time (float): Start time in seconds
            end_time (float): End time in seconds
        
        Returns:
            AudioData: New AudioData object with trimmed audio
        """
        start_sample = int(start_time * audio_data.sample_rate)
        end_sample = int(end_time * audio_data.sample_rate)
        
        start_sample = max(0, start_sample)
        end_sample = min(len(audio_data.data), end_sample)
        
        new_data = audio_data.data[start_sample:end_sample]
        result = AudioData(new_data, audio_data.sample_rate, audio_data.filepath)
        result.annotations = audio_data.annotations.copy()
        return result
    
    @staticmethod
    def apply_gain(audio_data: AudioData, gain_db: float) -> None:
        """
        Apply gain adjustment to audio data.
        
        Args:
            audio_data (AudioData): Audio data to modify (modified in-place)
            gain_db (float): Gain adjustment in decibels
        """
        audio_data._save_state()
        gain_linear = 10 ** (gain_db / 20)
        audio_data.data = audio_data.data * gain_linear
    
    @staticmethod
    def normalize(audio_data: AudioData, target_level: float = -3.0) -> None:
        """
        Normalize audio to target peak level.
        
        Args:
            audio_data (AudioData): Audio data to normalize (modified in-place)
            target_level (float): Target peak level in dB (default: -3.0)
        """
        audio_data._save_state()
        current_max = np.max(np.abs(audio_data.data))
        if current_max > 0:
            target_linear = 10 ** (target_level / 20)
            gain = target_linear / current_max
            audio_data.data = audio_data.data * gain
    
    @staticmethod
    def apply_fade(audio_data: AudioData, fade_in: float = 0.0, fade_out: float = 0.0) -> None:
        """
        Apply fade in and/or fade out effects.
        
        Args:
            audio_data (AudioData): Audio data to modify (modified in-place)
            fade_in (float): Fade in duration in seconds (default: 0.0)
            fade_out (float): Fade out duration in seconds (default: 0.0)
        """
        audio_data._save_state()
        samples_in = int(fade_in * audio_data.sample_rate)
        samples_out = int(fade_out * audio_data.sample_rate)
        
        if samples_in > 0:
            fade_curve = np.linspace(0, 1, samples_in)
            for ch in range(audio_data.channels):
                if audio_data.channels == 1:
                    audio_data.data[:samples_in] *= fade_curve
                else:
                    audio_data.data[:samples_in, ch] *= fade_curve
        
        if samples_out > 0:
            fade_curve = np.linspace(1, 0, samples_out)
            for ch in range(audio_data.channels):
                if audio_data.channels == 1:
                    audio_data.data[-samples_out:] *= fade_curve
                else:
                    audio_data.data[-samples_out:, ch] *= fade_curve


class AudioFilters:
    """
    Audio filtering and digital signal processing effects.
    
    This class provides various digital filters for audio processing including
    low-pass, high-pass, band-pass, and notch filters. All filters use scipy's
    signal processing functions and support both mono and stereo audio.
    """
    
    @staticmethod
    def low_pass_filter(audio_data: AudioData, cutoff_freq: float, order: int = 5) -> None:
        """
        Apply low-pass filter to remove high frequencies.
        
        Args:
            audio_data (AudioData): Audio data to filter (modified in-place)
            cutoff_freq (float): Cutoff frequency in Hz
            order (int): Filter order (default: 5)
        """
        audio_data._save_state()
        nyquist = audio_data.sample_rate / 2
        normalized_cutoff = cutoff_freq / nyquist
        b, a = signal.butter(order, normalized_cutoff, btype='low')
        
        for ch in range(audio_data.channels):
            if audio_data.channels == 1:
                audio_data.data = signal.filtfilt(b, a, audio_data.data)
            else:
                audio_data.data[:, ch] = signal.filtfilt(b, a, audio_data.data[:, ch])
    
    @staticmethod
    def high_pass_filter(audio_data: AudioData, cutoff_freq: float, order: int = 5) -> None:
        """
        Apply high-pass filter to remove low frequencies.
        
        Args:
            audio_data (AudioData): Audio data to filter (modified in-place)
            cutoff_freq (float): Cutoff frequency in Hz
            order (int): Filter order (default: 5)
        """
        audio_data._save_state()
        nyquist = audio_data.sample_rate / 2
        normalized_cutoff = cutoff_freq / nyquist
        b, a = signal.butter(order, normalized_cutoff, btype='high')
        
        for ch in range(audio_data.channels):
            if audio_data.channels == 1:
                audio_data.data = signal.filtfilt(b, a, audio_data.data)
            else:
                audio_data.data[:, ch] = signal.filtfilt(b, a, audio_data.data[:, ch])
    
    @staticmethod
    def band_pass_filter(audio_data: AudioData, low_freq: float, high_freq: float, order: int = 5) -> None:
        """
        Apply band-pass filter to keep frequencies within specified range.
        
        Args:
            audio_data (AudioData): Audio data to filter (modified in-place)
            low_freq (float): Low cutoff frequency in Hz
            high_freq (float): High cutoff frequency in Hz
            order (int): Filter order (default: 5)
        """
        audio_data._save_state()
        nyquist = audio_data.sample_rate / 2
        low_norm = low_freq / nyquist
        high_norm = high_freq / nyquist
        b, a = signal.butter(order, [low_norm, high_norm], btype='band')
        
        for ch in range(audio_data.channels):
            if audio_data.channels == 1:
                audio_data.data = signal.filtfilt(b, a, audio_data.data)
            else:
                audio_data.data[:, ch] = signal.filtfilt(b, a, audio_data.data[:, ch])
    
    @staticmethod
    def notch_filter(audio_data: AudioData, freq: float, quality: float = 30) -> None:
        """
        Apply notch filter to remove a specific frequency.
        
        Args:
            audio_data (AudioData): Audio data to filter (modified in-place)
            freq (float): Frequency to remove in Hz
            quality (float): Quality factor controlling filter sharpness (default: 30)
        """
        audio_data._save_state()
        nyquist = audio_data.sample_rate / 2
        freq_norm = freq / nyquist
        b, a = signal.iirnotch(freq_norm, quality)
        
        for ch in range(audio_data.channels):
            if audio_data.channels == 1:
                audio_data.data = signal.filtfilt(b, a, audio_data.data)
            else:
                audio_data.data[:, ch] = signal.filtfilt(b, a, audio_data.data[:, ch])


class SpectrogramGenerator:
    """
    Generate spectrograms for audio visualization and analysis.
    
    This class provides methods to compute both linear and mel-scale spectrograms
    for frequency domain analysis of audio signals. Spectrograms are useful for
    visualizing audio content and identifying frequency components over time.
    """
    
    @staticmethod
    def compute_spectrogram(audio_data: AudioData, n_fft: int = 2048, 
                          hop_length: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute linear frequency spectrogram.
        
        Args:
            audio_data (AudioData): Input audio data
            n_fft (int): FFT window size (default: 2048)
            hop_length (Optional[int]): Hop length between frames (default: n_fft//4)
        
        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]: Magnitude spectrogram in dB, time axis, frequency axis
        """
        if hop_length is None:
            hop_length = n_fft // 4
        
        # Use first channel for mono or stereo
        data = audio_data.data[:, 0] if audio_data.channels > 1 else audio_data.data.flatten()
        
        stft = librosa.stft(data, n_fft=n_fft, hop_length=hop_length)
        magnitude = np.abs(stft)
        magnitude_db = librosa.amplitude_to_db(magnitude)
        
        # Time and frequency axes
        times = librosa.frames_to_time(np.arange(magnitude.shape[1]), 
                                     sr=audio_data.sample_rate, hop_length=hop_length)
        freqs = librosa.fft_frequencies(sr=audio_data.sample_rate, n_fft=n_fft)
        
        return magnitude_db, times, freqs
    
    @staticmethod
    def compute_mel_spectrogram(audio_data: AudioData, n_mels: int = 128, 
                              n_fft: int = 2048) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute mel-scale spectrogram.
        
        Args:
            audio_data (AudioData): Input audio data
            n_mels (int): Number of mel frequency bins (default: 128)
            n_fft (int): FFT window size (default: 2048)
        
        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]: Mel spectrogram in dB, time axis, mel frequency axis
        """
        data = audio_data.data[:, 0] if audio_data.channels > 1 else audio_data.data.flatten()
        
        mel_spec = librosa.feature.melspectrogram(y=data, sr=audio_data.sample_rate, 
                                                n_mels=n_mels, n_fft=n_fft)
        mel_spec_db = librosa.power_to_db(mel_spec)
        
        times = librosa.frames_to_time(np.arange(mel_spec.shape[1]), sr=audio_data.sample_rate)
        freqs = librosa.mel_frequencies(n_mels=n_mels, fmin=0.0, fmax=audio_data.sample_rate/2)
        
        return mel_spec_db, times, freqs


class AnnotationManager:
    """
    Manage time-based audio annotations.
    
    This class provides functionality for adding, removing, and managing
    time-based annotations on audio files. Annotations can be used for
    marking regions of interest, labeling content, or creating metadata
    for audio analysis workflows.
    """
    
    @staticmethod
    def add_annotation(audio_data: AudioData, start_time: float, end_time: float, 
                      label: str, description: str = "") -> None:
        """
        Add a new annotation to audio data.
        
        Args:
            audio_data (AudioData): Audio data to annotate
            start_time (float): Start time of annotation in seconds
            end_time (float): End time of annotation in seconds
            label (str): Annotation label
            description (str): Optional description (default: "")
        """
        annotation = {
            'start_time': start_time,
            'end_time': end_time,
            'label': label,
            'description': description,
            'id': len(audio_data.annotations)
        }
        audio_data.annotations.append(annotation)
    
    @staticmethod
    def remove_annotation(audio_data: AudioData, annotation_id: int) -> bool:
        """
        Remove annotation by ID.
        
        Args:
            audio_data (AudioData): Audio data containing annotations
            annotation_id (int): ID of annotation to remove
        
        Returns:
            bool: True if annotation was removed
        """
        audio_data.annotations = [ann for ann in audio_data.annotations 
                                if ann['id'] != annotation_id]
        return True
    
    @staticmethod
    def get_annotations_in_range(audio_data: AudioData, start_time: float, 
                               end_time: float) -> List[Dict[str, Any]]:
        """
        Get annotations that overlap with specified time range.
        
        Args:
            audio_data (AudioData): Audio data containing annotations
            start_time (float): Start time of range in seconds
            end_time (float): End time of range in seconds
        
        Returns:
            List[Dict[str, Any]]: List of annotations within time range
        """
        return [ann for ann in audio_data.annotations 
                if not (ann['end_time'] < start_time or ann['start_time'] > end_time)]
    
    @staticmethod
    def export_annotations(audio_data: AudioData, filepath: str) -> None:
        """
        Export annotations to JSON file.
        
        Args:
            audio_data (AudioData): Audio data containing annotations
            filepath (str): Output file path
        """
        import json
        with open(filepath, 'w') as f:
            json.dump(audio_data.annotations, f, indent=2)
    
    @staticmethod
    def import_annotations(audio_data: AudioData, filepath: str) -> None:
        """
        Import annotations from JSON file.
        
        Args:
            audio_data (AudioData): Audio data to add annotations to
            filepath (str): Input file path
        """
        import json
        with open(filepath, 'r') as f:
            audio_data.annotations = json.load(f)


class AudioPlayback:
    """
    Handle real-time audio playback functionality.
    
    This class provides audio playback capabilities using the simpleaudio library.
    It supports playing audio from specific time positions and includes controls
    for starting, stopping, and monitoring playback status.
    
    Attributes:
        is_playing (bool): Current playback status
        current_position (float): Current playback position in seconds
    """
    
    def __init__(self) -> None:
        """
        Initialize audio playback manager.
        """
        self.is_playing = False
        self.current_position = 0.0
    
    def play(self, audio_data: AudioData, start_time: float = 0.0) -> None:
        """
        Play audio data from specified start time.
        
        Args:
            audio_data (AudioData): Audio data to play
            start_time (float): Start time in seconds (default: 0.0)
        
        Raises:
            RuntimeError: If simpleaudio is not available or playback fails
        """
        try:
            import simpleaudio as sa
            
            start_sample = int(start_time * audio_data.sample_rate)
            data_to_play = audio_data.data[start_sample:]
            
            # Convert to format suitable for playback
            if audio_data.channels == 1:
                audio_array = data_to_play.flatten()
            else:
                audio_array = data_to_play
            
            # Convert to 16-bit PCM
            audio_array = (audio_array * 32767).astype(np.int16)
            
            self.play_obj = sa.play_buffer(audio_array, audio_data.channels, 
                                         2, audio_data.sample_rate)
            self.is_playing = True
            
        except ImportError:
            raise RuntimeError("simpleaudio not available for playback")
        except Exception as e:
            raise RuntimeError(f"Playback error: {str(e)}")
    
    def stop(self) -> None:
        """
        Stop current audio playback.
        """
        if hasattr(self, 'play_obj') and self.play_obj.is_playing():
            self.play_obj.stop()
        self.is_playing = False
    
    def is_playing_audio(self) -> bool:
        """
        Check if audio is currently playing.
        
        Returns:
            bool: True if audio is playing, False otherwise
        """
        if hasattr(self, 'play_obj'):
            return self.play_obj.is_playing()
        return False