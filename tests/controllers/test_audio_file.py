# tests/controllers/test_audio_file.py
"""
Tests for AudioFileController.
"""

import numpy as np
import pytest

from bioamla.controllers.audio_file import AudioData, AudioFileController


class TestAudioFileController:
    """Tests for AudioFileController."""

    @pytest.fixture
    def controller(self):
        return AudioFileController()

    def test_open_valid_file_success(self, controller, tmp_audio_file):
        """Test that opening a valid audio file succeeds."""
        result = controller.open(tmp_audio_file)

        assert result.success is True
        assert result.data is not None
        assert isinstance(result.data, AudioData)
        assert result.data.sample_rate == 16000

    def test_open_nonexistent_file_fails(self, controller):
        """Test that opening a nonexistent file fails with error."""
        result = controller.open("/nonexistent/path/audio.wav")

        assert result.success is False
        assert result.error is not None
        assert "does not exist" in result.error

    def test_save_creates_file(self, controller, sample_audio_data, tmp_path):
        """Test that save creates a new audio file."""
        output_path = str(tmp_path / "output.wav")
        result = controller.save(sample_audio_data, output_path)

        assert result.success is True
        assert (tmp_path / "output.wav").exists()


class TestAudioData:
    """Tests for AudioData dataclass."""

    def test_duration_calculated_correctly(self):
        """Test that duration is calculated from samples and sample rate."""
        samples = np.zeros(16000, dtype=np.float32)  # 1 second at 16kHz
        audio = AudioData(samples=samples, sample_rate=16000)

        assert audio.duration == 1.0

    def test_channels_is_one_for_mono(self):
        """Test that channels is 1 for mono audio."""
        samples = np.zeros(16000, dtype=np.float32)
        audio = AudioData(samples=samples, sample_rate=16000, channels=1)

        assert audio.channels == 1

    def test_channels_is_two_for_stereo(self):
        """Test that channels is 2 for stereo audio."""
        samples = np.zeros((16000, 2), dtype=np.float32)
        audio = AudioData(samples=samples, sample_rate=16000, channels=2)

        assert audio.channels == 2

    def test_copy_creates_independent_copy(self):
        """Test that copy creates an independent copy."""
        samples = np.random.randn(16000).astype(np.float32)
        original = AudioData(samples=samples, sample_rate=16000)

        copied = original.copy()
        copied.samples[0] = 999.0

        assert original.samples[0] != 999.0
        assert copied.samples[0] == 999.0

    def test_mark_modified_returns_copy(self):
        """Test that mark_modified returns a modified copy."""
        audio = AudioData(samples=np.zeros(100, dtype=np.float32), sample_rate=16000)

        modified = audio.mark_modified()

        assert modified.is_modified is True
        assert audio.is_modified is False

    def test_num_samples_property(self):
        """Test that num_samples returns the correct count."""
        samples = np.zeros(16000, dtype=np.float32)
        audio = AudioData(samples=samples, sample_rate=16000)

        assert audio.num_samples == 16000


class TestAudioFileControllerSaveAs:
    """Tests for save_as method."""

    @pytest.fixture
    def controller(self):
        return AudioFileController()

    def test_save_as_success(self, controller, sample_audio_data, tmp_path):
        """Test that save_as creates a file."""
        output_path = str(tmp_path / "output.wav")
        result = controller.save_as(sample_audio_data, output_path)

        assert result.success is True
        assert (tmp_path / "output.wav").exists()

    def test_save_as_with_resample(self, controller, sample_audio_data, tmp_path, mocker):
        """Test that save_as can resample audio."""
        mock_resample = mocker.patch("bioamla.core.audio.signal.resample_audio")
        mock_resample.return_value = np.zeros(8000, dtype=np.float32)

        output_path = str(tmp_path / "output.wav")
        result = controller.save_as(sample_audio_data, output_path, target_sample_rate=8000)

        assert result.success is True
        mock_resample.assert_called_once()

    def test_save_as_same_sample_rate_no_resample(self, controller, sample_audio_data, tmp_path, mocker):
        """Test that save_as skips resampling if sample rate matches."""
        mock_resample = mocker.patch("bioamla.core.audio.signal.resample_audio")

        output_path = str(tmp_path / "output.wav")
        # sample_audio_data is 16000 Hz
        result = controller.save_as(sample_audio_data, output_path, target_sample_rate=16000)

        assert result.success is True
        mock_resample.assert_not_called()


class TestAudioFileControllerOverwrite:
    """Tests for overwrite method."""

    @pytest.fixture
    def controller(self):
        return AudioFileController()

    def test_overwrite_success(self, controller, tmp_audio_file):
        """Test that overwrite replaces an existing file."""
        # Load original
        result = controller.open(tmp_audio_file)
        audio_data = result.data

        # Modify and overwrite
        audio_data.samples = audio_data.samples * 0.5
        audio_data = audio_data.mark_modified()

        result = controller.overwrite(audio_data, tmp_audio_file)

        assert result.success is True

    def test_overwrite_no_target_fails(self, controller):
        """Test that overwrite fails without a target path."""
        audio_data = AudioData(
            samples=np.zeros(100, dtype=np.float32),
            sample_rate=16000,
            source_path=None,
        )

        result = controller.overwrite(audio_data)

        assert result.success is False
        assert "No target path" in result.error or "target path" in result.error.lower()

    def test_overwrite_nonexistent_file_fails(self, controller):
        """Test that overwrite fails on nonexistent file."""
        audio_data = AudioData(
            samples=np.zeros(100, dtype=np.float32),
            sample_rate=16000,
            source_path="/nonexistent/file.wav",
        )

        result = controller.overwrite(audio_data)

        assert result.success is False


class TestAudioFileControllerUndoRedo:
    """Tests for undo/redo functionality."""

    @pytest.fixture
    def controller(self, tmp_path):
        # Undo/redo requires stateful mode (project_path)
        return AudioFileController(project_path=str(tmp_path))

    def test_undo_after_save(self, controller, sample_audio_data, tmp_path):
        """Test that undo removes a saved file."""
        output_path = tmp_path / "output.wav"
        controller.save(sample_audio_data, str(output_path))

        assert output_path.exists()

        result = controller.undo()

        assert result.success is True
        assert not output_path.exists()

    def test_redo_after_undo(self, controller, sample_audio_data, tmp_path):
        """Test that redo restores a file after undo."""
        output_path = tmp_path / "output.wav"
        controller.save(sample_audio_data, str(output_path))
        controller.undo()

        assert not output_path.exists()

        result = controller.redo()

        assert result.success is True
        assert output_path.exists()

    def test_undo_nothing_fails(self, controller):
        """Test that undo fails when nothing to undo."""
        result = controller.undo()

        assert result.success is False
        assert "Nothing to undo" in result.error

    def test_redo_nothing_fails(self, controller):
        """Test that redo fails when nothing to redo."""
        result = controller.redo()

        assert result.success is False
        assert "Nothing to redo" in result.error

    def test_can_undo_property(self, controller, sample_audio_data, tmp_path):
        """Test can_undo property."""
        assert controller.can_undo is False

        controller.save(sample_audio_data, str(tmp_path / "out.wav"))

        assert controller.can_undo is True

    def test_can_redo_property(self, controller, sample_audio_data, tmp_path):
        """Test can_redo property."""
        assert controller.can_redo is False

        controller.save(sample_audio_data, str(tmp_path / "out.wav"))
        controller.undo()

        assert controller.can_redo is True

    def test_undo_history_contains_descriptions(self, controller, sample_audio_data, tmp_path):
        """Test that undo_history contains operation descriptions."""
        controller.save(sample_audio_data, str(tmp_path / "out1.wav"))
        controller.save(sample_audio_data, str(tmp_path / "out2.wav"))

        history = controller.undo_history

        assert len(history) == 2
        assert all("Save" in h or "out" in h for h in history)

    def test_clear_history(self, controller, sample_audio_data, tmp_path):
        """Test that clear_history removes all history."""
        controller.save(sample_audio_data, str(tmp_path / "out.wav"))

        controller.clear_history()

        assert controller.can_undo is False


class TestAudioFileControllerTempFiles:
    """Tests for temporary file functionality."""

    @pytest.fixture
    def controller(self):
        return AudioFileController()

    def test_create_temp_file_success(self, controller, sample_audio_data):
        """Test that create_temp_file creates a temporary file."""
        result = controller.create_temp_file(sample_audio_data)

        assert result.success is True
        assert result.data is not None
        import os

        assert os.path.exists(result.data)

        # Cleanup
        controller.cleanup()

    def test_create_temp_file_custom_suffix(self, controller, sample_audio_data):
        """Test that create_temp_file respects custom suffix."""
        result = controller.create_temp_file(sample_audio_data, suffix=".flac")

        assert result.success is True
        assert result.data.endswith(".flac")

        controller.cleanup()

    def test_cleanup_removes_temp_files(self, controller, sample_audio_data):
        """Test that cleanup removes temporary files."""
        result = controller.create_temp_file(sample_audio_data)
        temp_path = result.data

        controller.cleanup()

        import os

        assert not os.path.exists(temp_path)


class TestWriteWithTransform:
    """Tests for write_with_transform method."""

    @pytest.fixture
    def controller(self):
        return AudioFileController()

    def test_write_with_transform_success(self, controller, tmp_audio_file, tmp_path):
        """Test that write_with_transform applies transform and saves."""

        def double_volume(audio, sr):
            return audio * 2, sr

        output_path = str(tmp_path / "output.wav")
        result = controller.write_with_transform(
            tmp_audio_file,
            output_path,
            double_volume,
            "double_volume",
        )

        assert result.success is True
        assert (tmp_path / "output.wav").exists()

    def test_write_with_transform_nonexistent_input_fails(self, controller, tmp_path):
        """Test that write_with_transform fails for nonexistent input."""

        def noop(audio, sr):
            return audio, sr

        result = controller.write_with_transform(
            "/nonexistent/file.wav",
            str(tmp_path / "output.wav"),
            noop,
        )

        assert result.success is False
        assert "does not exist" in result.error


class TestAudioFileControllerStatelessMode:
    """Tests for stateless mode behavior (no project_path)."""

    @pytest.fixture
    def controller(self):
        # Stateless controller (no project_path)
        return AudioFileController()

    def test_undo_fails_in_stateless_mode(self, controller, sample_audio_data, tmp_path):
        """Test that undo fails when not in stateful mode."""
        # Save works in stateless mode
        output_path = tmp_path / "output.wav"
        save_result = controller.save(sample_audio_data, str(output_path))
        assert save_result.success is True
        assert output_path.exists()

        # But undo fails
        result = controller.undo()
        assert result.success is False
        assert "--project" in result.error

    def test_redo_fails_in_stateless_mode(self, controller):
        """Test that redo fails when not in stateful mode."""
        result = controller.redo()
        assert result.success is False
        assert "--project" in result.error

    def test_is_stateful_false(self, controller):
        """Test that is_stateful returns False without project_path."""
        assert controller.is_stateful is False

    def test_is_stateful_true_with_project(self, tmp_path):
        """Test that is_stateful returns True with project_path."""
        controller = AudioFileController(project_path=str(tmp_path))
        assert controller.is_stateful is True
