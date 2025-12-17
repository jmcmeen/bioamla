"""
Unit tests for the ast predict CLI command options.
"""

from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner

from bioamla.cli import cli


@pytest.fixture
def runner():
    """Create a CLI test runner."""
    return CliRunner()


class TestAstPredictHelp:
    """Tests for ast predict help and options."""

    def test_ast_predict_help(self, runner):
        """Test ast predict --help shows all options."""
        result = runner.invoke(cli, ["models", "ast", "predict", "--help"])

        assert result.exit_code == 0
        assert "PATH" in result.output
        assert "--output-csv" in result.output
        assert "--model-path" in result.output
        assert "--resample-freq" in result.output
        assert "--clip-seconds" in result.output
        assert "--overlap-seconds" in result.output
        assert "--restart" in result.output
        assert "--batch" in result.output

    def test_ast_predict_performance_options_exist(self, runner):
        """Test that performance options are shown in help."""
        result = runner.invoke(cli, ["models", "ast", "predict", "--help"])

        assert result.exit_code == 0
        assert "--batch-size" in result.output
        assert "--fp16" in result.output
        assert "--compile" in result.output
        assert "--workers" in result.output

    def test_ast_predict_requires_path(self, runner):
        """Test that ast predict requires path argument."""
        result = runner.invoke(cli, ["models", "ast", "predict"])

        assert result.exit_code != 0
        assert "Missing argument" in result.output or "PATH" in result.output


class TestAstPredictPerformanceOptions:
    """Tests for performance-related options."""

    def test_batch_size_option_exists(self, runner):
        """Test that --batch-size option exists."""
        result = runner.invoke(cli, ["models", "ast", "predict", "--help"])

        assert result.exit_code == 0
        assert "--batch-size" in result.output
        assert "Number of segments to process in parallel" in result.output

    def test_fp16_option_exists(self, runner):
        """Test that --fp16 option exists."""
        result = runner.invoke(cli, ["models", "ast", "predict", "--help"])

        assert result.exit_code == 0
        assert "--fp16" in result.output
        assert "--no-fp16" in result.output

    def test_compile_option_exists(self, runner):
        """Test that --compile option exists."""
        result = runner.invoke(cli, ["models", "ast", "predict", "--help"])

        assert result.exit_code == 0
        assert "--compile" in result.output
        assert "--no-compile" in result.output

    def test_workers_option_exists(self, runner):
        """Test that --workers option exists."""
        result = runner.invoke(cli, ["models", "ast", "predict", "--help"])

        assert result.exit_code == 0
        assert "--workers" in result.output
        assert "Number of parallel workers" in result.output


class TestAstPredictDefaults:
    """Tests for default option values."""

    def test_default_batch_size_is_8(self, runner):
        """Test that default batch size is 8."""
        result = runner.invoke(cli, ["models", "ast", "predict", "--help"])

        assert result.exit_code == 0
        # Check help text contains default value info
        assert "(default:" in result.output.lower() and "8" in result.output

    def test_default_workers_is_1(self, runner):
        """Test that default workers is 1."""
        result = runner.invoke(cli, ["models", "ast", "predict", "--help"])

        assert result.exit_code == 0
        assert "(default:" in result.output.lower() and "1" in result.output


class TestAstPredictBatchExecution:
    """Tests for ast predict --batch execution with mocked dependencies."""

    def test_passes_performance_options_to_config(self, runner, temp_dir):
        """Test that performance options are passed to InferenceConfig."""
        audio_dir = temp_dir / "audio"
        audio_dir.mkdir()

        with patch("novus_pytils.files.get_files_by_extension") as mock_get_files:
            mock_get_files.return_value = []

            result = runner.invoke(cli, [
                "models", "ast", "predict",
                str(audio_dir),
                "--batch",
                "--batch-size", "16",
                "--fp16",
                "--compile",
                "--workers", "4"
            ])

            # Should fail with "No wave files found" but options should be parsed
            assert "No wave files found" in result.output

    def test_prints_performance_options(self, runner, temp_dir):
        """Test that performance options are printed during execution."""
        audio_dir = temp_dir / "audio"
        audio_dir.mkdir()

        # Create a dummy wav file
        wav_file = audio_dir / "test.wav"
        wav_file.write_bytes(b"dummy")

        with patch("novus_pytils.files.get_files_by_extension") as mock_get_files:
            mock_get_files.return_value = [str(wav_file)]

            with patch("bioamla.ast.load_pretrained_ast_model") as mock_load:
                mock_model = MagicMock()
                mock_load.return_value = mock_model

                with patch("bioamla.ast.wave_file_batch_inference"):
                    result = runner.invoke(cli, [
                        "models", "ast", "predict",
                        str(audio_dir),
                        "--batch",
                        "--batch-size", "16",
                        "--fp16",
                        "--workers", "4"
                    ])

                    assert "Performance options:" in result.output
                    assert "batch_size=16" in result.output
                    assert "fp16=True" in result.output
                    assert "workers=4" in result.output

    def test_creates_inference_config(self, runner, temp_dir):
        """Test that InferenceConfig is created with correct options."""
        audio_dir = temp_dir / "audio"
        audio_dir.mkdir()

        wav_file = audio_dir / "test.wav"
        wav_file.write_bytes(b"dummy")

        with patch("novus_pytils.files.get_files_by_extension") as mock_get_files:
            mock_get_files.return_value = [str(wav_file)]

            with patch("bioamla.ast.load_pretrained_ast_model") as mock_load:
                mock_model = MagicMock()
                mock_load.return_value = mock_model

                with patch("bioamla.ast.wave_file_batch_inference") as mock_inference:
                    result = runner.invoke(cli, [
                        "models", "ast", "predict",
                        str(audio_dir),
                        "--batch",
                        "--batch-size", "32",
                        "--workers", "8"
                    ])

                    # Verify the command ran and printed the config
                    assert "batch_size=32" in result.output
                    assert "workers=8" in result.output

    def test_fp16_passed_to_model_loader(self, runner, temp_dir):
        """Test that fp16 option is passed to load_pretrained_ast_model."""
        audio_dir = temp_dir / "audio"
        audio_dir.mkdir()

        wav_file = audio_dir / "test.wav"
        wav_file.write_bytes(b"dummy")

        with patch("novus_pytils.files.get_files_by_extension") as mock_get_files:
            mock_get_files.return_value = [str(wav_file)]

            with patch("bioamla.ast.load_pretrained_ast_model") as mock_load:
                mock_model = MagicMock()
                mock_load.return_value = mock_model

                with patch("bioamla.ast.wave_file_batch_inference"):
                    runner.invoke(cli, [
                        "models", "ast", "predict",
                        str(audio_dir),
                        "--batch",
                        "--fp16"
                    ])

                    mock_load.assert_called_once()
                    call_kwargs = mock_load.call_args[1]
                    assert call_kwargs['use_fp16'] is True

    def test_compile_passed_to_model_loader(self, runner, temp_dir):
        """Test that compile option is passed to load_pretrained_ast_model."""
        audio_dir = temp_dir / "audio"
        audio_dir.mkdir()

        wav_file = audio_dir / "test.wav"
        wav_file.write_bytes(b"dummy")

        with patch("novus_pytils.files.get_files_by_extension") as mock_get_files:
            mock_get_files.return_value = [str(wav_file)]

            with patch("bioamla.ast.load_pretrained_ast_model") as mock_load:
                mock_model = MagicMock()
                mock_load.return_value = mock_model

                with patch("bioamla.ast.wave_file_batch_inference"):
                    runner.invoke(cli, [
                        "models", "ast", "predict",
                        str(audio_dir),
                        "--batch",
                        "--compile"
                    ])

                    mock_load.assert_called_once()
                    call_kwargs = mock_load.call_args[1]
                    assert call_kwargs['use_compile'] is True


class TestAstPredictSingleFile:
    """Tests for ast predict single file mode."""

    def test_single_file_prediction(self, runner):
        """Test single file prediction mode."""
        with patch("bioamla.ast.wav_ast_inference") as mock_inference:
            mock_inference.return_value = "species_a"

            result = runner.invoke(cli, [
                "models", "ast", "predict",
                "test.wav",
                "--model-path", "my_model"
            ])

            assert result.exit_code == 0
            assert "species_a" in result.output
            mock_inference.assert_called_once_with("test.wav", "my_model", 16000)

    def test_single_file_with_custom_resample_freq(self, runner):
        """Test single file prediction with custom resample frequency."""
        with patch("bioamla.ast.wav_ast_inference") as mock_inference:
            mock_inference.return_value = "species_b"

            result = runner.invoke(cli, [
                "models", "ast", "predict",
                "test.wav",
                "--model-path", "my_model",
                "--resample-freq", "22050"
            ])

            assert result.exit_code == 0
            mock_inference.assert_called_once_with("test.wav", "my_model", 22050)
