#!/usr/bin/env python3
"""
Simple test script for the audio editor functionality.
"""
import sys
import tempfile
import numpy as np
from pathlib import Path

# Add the src directory to the path so we can import our modules
sys.path.insert(0, str(Path(__file__).parent / "src"))

from bioamla.core.audio_editor import (
    AudioData, AudioProcessor, AudioFilters, 
    AnnotationManager, SpectrogramGenerator
)


def create_test_audio():
    """Create a simple test audio signal."""
    # Generate a 2-second sine wave at 440 Hz
    sample_rate = 16000
    duration = 2.0
    frequency = 440.0
    
    t = np.linspace(0, duration, int(sample_rate * duration))
    audio_signal = np.sin(2 * np.pi * frequency * t).reshape(-1, 1)  # Make it 2D
    
    return AudioData(audio_signal, sample_rate)


def test_audio_processing():
    """Test basic audio processing functions."""
    print("Testing audio processing functions...")
    
    # Create test audio
    audio_data = create_test_audio()
    print(f"✓ Created test audio: {audio_data.duration:.1f}s, {audio_data.sample_rate}Hz")
    
    # Test saving and loading
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
        temp_path = tmp_file.name
    
    try:
        AudioProcessor.save_audio(audio_data, temp_path)
        loaded_audio = AudioProcessor.load_audio(temp_path)
        print(f"✓ Save/load test passed: {loaded_audio.duration:.1f}s")
    except Exception as e:
        print(f"✗ Save/load test failed: {e}")
        return False
    finally:
        Path(temp_path).unlink(missing_ok=True)
    
    # Test trimming
    try:
        trimmed = AudioProcessor.trim_audio(audio_data, 0.5, 1.5)
        expected_duration = 1.0
        if abs(trimmed.duration - expected_duration) < 0.1:
            print(f"✓ Trim test passed: {trimmed.duration:.1f}s")
        else:
            print(f"✗ Trim test failed: expected {expected_duration}s, got {trimmed.duration:.1f}s")
    except Exception as e:
        print(f"✗ Trim test failed: {e}")
    
    # Test gain
    try:
        original_max = np.max(np.abs(audio_data.data))
        AudioProcessor.apply_gain(audio_data, 6.0)  # +6dB
        new_max = np.max(np.abs(audio_data.data))
        if new_max > original_max:
            print("✓ Gain test passed")
        else:
            print("✗ Gain test failed")
    except Exception as e:
        print(f"✗ Gain test failed: {e}")
    
    return True


def test_filters():
    """Test audio filters."""
    print("\nTesting audio filters...")
    
    audio_data = create_test_audio()
    
    # Test low-pass filter
    try:
        AudioFilters.low_pass_filter(audio_data, 1000.0)
        print("✓ Low-pass filter test passed")
    except Exception as e:
        print(f"✗ Low-pass filter test failed: {e}")
        return False
    
    # Test high-pass filter  
    audio_data = create_test_audio()  # Reset audio
    try:
        AudioFilters.high_pass_filter(audio_data, 100.0)
        print("✓ High-pass filter test passed")
    except Exception as e:
        print(f"✗ High-pass filter test failed: {e}")
        return False
    
    return True


def test_spectrogram():
    """Test spectrogram generation."""
    print("\nTesting spectrogram generation...")
    
    audio_data = create_test_audio()
    
    try:
        spec_data, times, freqs = SpectrogramGenerator.compute_spectrogram(audio_data)
        print(f"✓ Linear spectrogram: {spec_data.shape}")
    except Exception as e:
        print(f"✗ Linear spectrogram test failed: {e}")
        return False
    
    try:
        mel_data, times, freqs = SpectrogramGenerator.compute_mel_spectrogram(audio_data)
        print(f"✓ Mel spectrogram: {mel_data.shape}")
    except Exception as e:
        print(f"✗ Mel spectrogram test failed: {e}")
        return False
    
    return True


def test_annotations():
    """Test annotation system."""
    print("\nTesting annotations...")
    
    audio_data = create_test_audio()
    
    try:
        AnnotationManager.add_annotation(audio_data, 0.5, 1.0, "Test annotation", "Test description")
        if len(audio_data.annotations) == 1:
            print("✓ Add annotation test passed")
        else:
            print("✗ Add annotation test failed")
            return False
    except Exception as e:
        print(f"✗ Add annotation test failed: {e}")
        return False
    
    try:
        annotations = AnnotationManager.get_annotations_in_range(audio_data, 0.3, 0.8)
        if len(annotations) == 1:
            print("✓ Get annotations in range test passed")
        else:
            print("✗ Get annotations in range test failed")
    except Exception as e:
        print(f"✗ Get annotations in range test failed: {e}")
    
    return True


def test_undo_functionality():
    """Test undo functionality."""
    print("\nTesting undo functionality...")
    
    audio_data = create_test_audio()
    original_max = np.max(np.abs(audio_data.data))
    
    try:
        # Apply gain
        AudioProcessor.apply_gain(audio_data, 6.0)
        modified_max = np.max(np.abs(audio_data.data))
        
        # Undo
        if audio_data.undo():
            undone_max = np.max(np.abs(audio_data.data))
            if abs(undone_max - original_max) < 0.001:
                print("✓ Undo test passed")
                return True
            else:
                print("✗ Undo test failed: values don't match")
        else:
            print("✗ Undo test failed: undo returned False")
    except Exception as e:
        print(f"✗ Undo test failed: {e}")
    
    return False


def main():
    """Run all tests."""
    print("Running Audio Editor Tests")
    print("=" * 50)
    
    tests = [
        test_audio_processing,
        test_filters,
        test_spectrogram,
        test_annotations,
        test_undo_functionality
    ]
    
    passed = 0
    total = len(tests)
    
    for test_func in tests:
        try:
            if test_func():
                passed += 1
        except Exception as e:
            print(f"✗ {test_func.__name__} failed with exception: {e}")
    
    print("\n" + "=" * 50)
    print(f"Tests completed: {passed}/{total} passed")
    
    if passed == total:
        print("🎉 All tests passed!")
        return 0
    else:
        print("❌ Some tests failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())