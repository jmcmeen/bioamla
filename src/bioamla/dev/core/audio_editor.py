"""
Core audio editing functionality for the audio editor application.
"""
import numpy as np
import librosa
import soundfile as sf
from scipy import signal
from typing import Dict, List, Tuple, Optional, Any


class AudioData:
    """Container for audio data and metadata."""
    
    def __init__(self, data: np.ndarray, sample_rate: int, filepath: Optional[str] = None):
        self.data = data
        self.sample_rate = sample_rate
        self.filepath = filepath
        self.annotations: List[Dict[str, Any]] = []
        self.history: List[np.ndarray] = []
        self._save_state()
    
    def _save_state(self):
        """Save current state for undo functionality."""
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
        """Undo last operation."""
        if len(self.history) > 1:
            self.history.pop()  # Remove current state
            self.data = self.history[-1].copy()
            return True
        return False


class AudioProcessor:
    """Main audio processing class."""
    
    @staticmethod
    def load_audio(filepath: str, target_sr: Optional[int] = None) -> AudioData:
        """Load audio file."""
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
        """Save audio file."""
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
        """Convert audio file format."""
        audio_data = AudioProcessor.load_audio(input_path)
        AudioProcessor.save_audio(audio_data, output_path, format)
    
    @staticmethod
    def trim_audio(audio_data: AudioData, start_time: float, end_time: float) -> AudioData:
        """Trim audio to specified time range."""
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
        """Apply gain in decibels."""
        audio_data._save_state()
        gain_linear = 10 ** (gain_db / 20)
        audio_data.data = audio_data.data * gain_linear
    
    @staticmethod
    def normalize(audio_data: AudioData, target_level: float = -3.0) -> None:
        """Normalize audio to target level in dB."""
        audio_data._save_state()
        current_max = np.max(np.abs(audio_data.data))
        if current_max > 0:
            target_linear = 10 ** (target_level / 20)
            gain = target_linear / current_max
            audio_data.data = audio_data.data * gain
    
    @staticmethod
    def apply_fade(audio_data: AudioData, fade_in: float = 0.0, fade_out: float = 0.0) -> None:
        """Apply fade in/out."""
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
    """Audio filtering and effects."""
    
    @staticmethod
    def low_pass_filter(audio_data: AudioData, cutoff_freq: float, order: int = 5) -> None:
        """Apply low-pass filter."""
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
        """Apply high-pass filter."""
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
        """Apply band-pass filter."""
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
        """Apply notch filter to remove specific frequency."""
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
    """Generate spectrograms for visualization."""
    
    @staticmethod
    def compute_spectrogram(audio_data: AudioData, n_fft: int = 2048, 
                          hop_length: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute spectrogram."""
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
        """Compute mel spectrogram."""
        data = audio_data.data[:, 0] if audio_data.channels > 1 else audio_data.data.flatten()
        
        mel_spec = librosa.feature.melspectrogram(y=data, sr=audio_data.sample_rate, 
                                                n_mels=n_mels, n_fft=n_fft)
        mel_spec_db = librosa.power_to_db(mel_spec)
        
        times = librosa.frames_to_time(np.arange(mel_spec.shape[1]), sr=audio_data.sample_rate)
        freqs = librosa.mel_frequencies(n_mels=n_mels, fmin=0.0, fmax=audio_data.sample_rate/2)
        
        return mel_spec_db, times, freqs


class AnnotationManager:
    """Manage audio annotations."""
    
    @staticmethod
    def add_annotation(audio_data: AudioData, start_time: float, end_time: float, 
                      label: str, description: str = "") -> None:
        """Add annotation to audio."""
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
        """Remove annotation by ID."""
        audio_data.annotations = [ann for ann in audio_data.annotations 
                                if ann['id'] != annotation_id]
        return True
    
    @staticmethod
    def get_annotations_in_range(audio_data: AudioData, start_time: float, 
                               end_time: float) -> List[Dict[str, Any]]:
        """Get annotations within time range."""
        return [ann for ann in audio_data.annotations 
                if not (ann['end_time'] < start_time or ann['start_time'] > end_time)]
    
    @staticmethod
    def export_annotations(audio_data: AudioData, filepath: str) -> None:
        """Export annotations to JSON file."""
        import json
        with open(filepath, 'w') as f:
            json.dump(audio_data.annotations, f, indent=2)
    
    @staticmethod
    def import_annotations(audio_data: AudioData, filepath: str) -> None:
        """Import annotations from JSON file."""
        import json
        with open(filepath, 'r') as f:
            audio_data.annotations = json.load(f)


class AudioPlayback:
    """Handle audio playback."""
    
    def __init__(self):
        self.is_playing = False
        self.current_position = 0.0
    
    def play(self, audio_data: AudioData, start_time: float = 0.0) -> None:
        """Play audio from specified time."""
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
        """Stop playback."""
        if hasattr(self, 'play_obj') and self.play_obj.is_playing():
            self.play_obj.stop()
        self.is_playing = False
    
    def is_playing_audio(self) -> bool:
        """Check if audio is currently playing."""
        if hasattr(self, 'play_obj'):
            return self.play_obj.is_playing()
        return False