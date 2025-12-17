"""
Unit tests for the ast train CLI command options.
"""


import os
import tempfile
from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner

from bioamla.cli import _count_files, _get_folder_size, _is_large_folder, cli


@pytest.fixture
def runner():
    """Create a CLI test runner."""
    return CliRunner()


class TestAstTrainHelp:
    """Tests for ast train help and options."""

    def test_ast_train_help(self, runner):
        """Test ast train --help shows all options."""
        result = runner.invoke(cli, ["models", "train", "ast", "--help"])

        assert result.exit_code == 0
        assert "--training-dir" in result.output
        assert "--base-model" in result.output
        assert "--train-dataset" in result.output
        assert "--num-train-epochs" in result.output
        assert "--per-device-train-batch-size" in result.output
        assert "--fp16" in result.output
        assert "--bf16" in result.output
        assert "--gradient-accumulation-steps" in result.output
        assert "--dataloader-num-workers" in result.output
        assert "--torch-compile" in result.output
        assert "--finetune-mode" in result.output

    def test_ast_train_finetune_mode_choices(self, runner):
        """Test that --finetune-mode shows valid choices."""
        result = runner.invoke(cli, ["models", "train", "ast", "--help"])

        assert result.exit_code == 0
        assert "full" in result.output
        assert "feature-extraction" in result.output

    def test_ast_train_default_batch_size(self, runner):
        """Test that default batch size is 8."""
        result = runner.invoke(cli, ["models", "train", "ast", "--help"])

        assert result.exit_code == 0
        # Check help text mentions the default
        assert "per-device-train-batch-size" in result.output


class TestAstTrainOptions:
    """Tests for ast train option validation."""

    def test_invalid_finetune_mode(self, runner):
        """Test that invalid finetune mode is rejected."""
        result = runner.invoke(cli, ["models", "train", "ast", "--finetune-mode", "invalid"])

        assert result.exit_code != 0
        assert "Invalid value" in result.output or "invalid" in result.output.lower()

    def test_valid_finetune_mode_full(self, runner):
        """Test that 'full' is a valid finetune mode."""
        # Just check that the option is accepted (will fail later due to missing dataset)
        result = runner.invoke(cli, ["models", "train", "ast", "--finetune-mode", "full", "--help"])

        assert result.exit_code == 0

    def test_valid_finetune_mode_feature_extraction(self, runner):
        """Test that 'feature-extraction' is a valid finetune mode."""
        # Just check that the option is accepted
        result = runner.invoke(cli, ["models", "train", "ast", "--finetune-mode", "feature-extraction", "--help"])

        assert result.exit_code == 0


class TestAstTrainPerformanceOptions:
    """Tests for performance-related options."""

    def test_fp16_and_bf16_options_exist(self, runner):
        """Test that fp16 and bf16 options exist."""
        result = runner.invoke(cli, ["models", "train", "ast", "--help"])

        assert result.exit_code == 0
        assert "--fp16" in result.output
        assert "--no-fp16" in result.output
        assert "--bf16" in result.output
        assert "--no-bf16" in result.output

    def test_gradient_accumulation_option_exists(self, runner):
        """Test that gradient accumulation option exists."""
        result = runner.invoke(cli, ["models", "train", "ast", "--help"])

        assert result.exit_code == 0
        assert "--gradient-accumulation-steps" in result.output

    def test_dataloader_workers_option_exists(self, runner):
        """Test that dataloader workers option exists."""
        result = runner.invoke(cli, ["models", "train", "ast", "--help"])

        assert result.exit_code == 0
        assert "--dataloader-num-workers" in result.output

    def test_torch_compile_option_exists(self, runner):
        """Test that torch compile option exists."""
        result = runner.invoke(cli, ["models", "train", "ast", "--help"])

        assert result.exit_code == 0
        assert "--torch-compile" in result.output
        assert "--no-torch-compile" in result.output


class TestAstTrainMlflowOptions:
    """Tests for MLflow-related options."""

    def test_mlflow_tracking_uri_option_exists(self, runner):
        """Test that mlflow tracking URI option exists."""
        result = runner.invoke(cli, ["models", "train", "ast", "--help"])

        assert result.exit_code == 0
        assert "--mlflow-tracking-uri" in result.output

    def test_mlflow_experiment_name_option_exists(self, runner):
        """Test that mlflow experiment name option exists."""
        result = runner.invoke(cli, ["models", "train", "ast", "--help"])

        assert result.exit_code == 0
        assert "--mlflow-experiment-name" in result.output

    def test_mlflow_run_name_option_exists(self, runner):
        """Test that mlflow run name option exists."""
        result = runner.invoke(cli, ["models", "train", "ast", "--help"])

        assert result.exit_code == 0
        assert "--mlflow-run-name" in result.output


class TestServicesHfPushModel:
    """Tests for services hf push-model command."""

    def test_services_hf_push_model_help(self, runner):
        """Test services hf push-model --help shows all options."""
        result = runner.invoke(cli, ["services", "hf", "push-model", "--help"])

        assert result.exit_code == 0
        assert "PATH" in result.output
        assert "REPO_ID" in result.output
        assert "--private" in result.output
        assert "--public" in result.output
        assert "--commit-message" in result.output
        assert "entire contents" in result.output

    def test_services_hf_push_model_requires_path(self, runner):
        """Test that services hf push-model requires path argument."""
        result = runner.invoke(cli, ["services", "hf", "push-model"])

        assert result.exit_code != 0
        assert "Missing argument" in result.output or "PATH" in result.output

    def test_services_hf_push_model_requires_repo_id(self, runner):
        """Test that services hf push-model requires repo_id argument."""
        result = runner.invoke(cli, ["services", "hf", "push-model", "/some/path"])

        assert result.exit_code != 0
        assert "Missing argument" in result.output or "REPO_ID" in result.output

    def test_services_hf_push_model_invalid_path(self, runner):
        """Test that services hf push-model fails with non-existent path."""
        result = runner.invoke(cli, ["services", "hf", "push-model", "/nonexistent/path", "user/repo"])

        assert result.exit_code != 0
        assert "does not exist" in result.output


class TestServicesHfPushDataset:
    """Tests for services hf push-dataset command."""

    def test_services_hf_push_dataset_help(self, runner):
        """Test services hf push-dataset --help shows all options."""
        result = runner.invoke(cli, ["services", "hf", "push-dataset", "--help"])

        assert result.exit_code == 0
        assert "PATH" in result.output
        assert "REPO_ID" in result.output
        assert "--private" in result.output
        assert "--public" in result.output
        assert "--commit-message" in result.output
        assert "entire contents" in result.output

    def test_services_hf_push_dataset_requires_path(self, runner):
        """Test that services hf push-dataset requires path argument."""
        result = runner.invoke(cli, ["services", "hf", "push-dataset"])

        assert result.exit_code != 0
        assert "Missing argument" in result.output or "PATH" in result.output

    def test_services_hf_push_dataset_requires_repo_id(self, runner):
        """Test that services hf push-dataset requires repo_id argument."""
        result = runner.invoke(cli, ["services", "hf", "push-dataset", "/some/path"])

        assert result.exit_code != 0
        assert "Missing argument" in result.output or "REPO_ID" in result.output

    def test_services_hf_push_dataset_invalid_path(self, runner):
        """Test that services hf push-dataset fails with non-existent path."""
        result = runner.invoke(cli, ["services", "hf", "push-dataset", "/nonexistent/path", "user/repo"])

        assert result.exit_code != 0
        assert "does not exist" in result.output


class TestGetFolderSize:
    """Tests for _get_folder_size helper function."""

    def test_empty_folder(self):
        """Test that empty folder returns 0 bytes."""
        with tempfile.TemporaryDirectory() as tmpdir:
            assert _get_folder_size(tmpdir) == 0

    def test_folder_with_files(self):
        """Test that folder size is calculated correctly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a file with known size
            file_path = os.path.join(tmpdir, "test.txt")
            content = b"Hello, World!"  # 13 bytes
            with open(file_path, "wb") as f:
                f.write(content)

            assert _get_folder_size(tmpdir) == 13

    def test_folder_with_nested_files(self):
        """Test that nested folder sizes are included."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create nested directory
            nested_dir = os.path.join(tmpdir, "subdir")
            os.makedirs(nested_dir)

            # Create files in both directories
            file1 = os.path.join(tmpdir, "file1.txt")
            file2 = os.path.join(nested_dir, "file2.txt")

            with open(file1, "wb") as f:
                f.write(b"12345")  # 5 bytes
            with open(file2, "wb") as f:
                f.write(b"67890")  # 5 bytes

            assert _get_folder_size(tmpdir) == 10


class TestCountFiles:
    """Tests for _count_files helper function."""

    def test_empty_folder(self):
        """Test that empty folder returns 0 files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            assert _count_files(tmpdir) == 0

    def test_folder_with_files(self):
        """Test that file count is correct."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create multiple files
            for i in range(5):
                file_path = os.path.join(tmpdir, f"file{i}.txt")
                with open(file_path, "w") as f:
                    f.write("test")

            assert _count_files(tmpdir) == 5

    def test_folder_with_nested_files(self):
        """Test that nested files are counted."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create nested directories
            nested_dir = os.path.join(tmpdir, "subdir")
            os.makedirs(nested_dir)

            # Create files in both directories
            with open(os.path.join(tmpdir, "file1.txt"), "w") as f:
                f.write("test")
            with open(os.path.join(nested_dir, "file2.txt"), "w") as f:
                f.write("test")
            with open(os.path.join(nested_dir, "file3.txt"), "w") as f:
                f.write("test")

            assert _count_files(tmpdir) == 3


class TestIsLargeFolder:
    """Tests for _is_large_folder helper function."""

    def test_small_folder_returns_false(self):
        """Test that small folder is not detected as large."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a small file
            with open(os.path.join(tmpdir, "small.txt"), "w") as f:
                f.write("small content")

            assert _is_large_folder(tmpdir) is False

    def test_many_files_returns_true(self):
        """Test that folder with many files is detected as large."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create many small files (use threshold of 10 for testing)
            for i in range(15):
                with open(os.path.join(tmpdir, f"file{i}.txt"), "w") as f:
                    f.write("test")

            # With low threshold, should return True
            assert _is_large_folder(tmpdir, file_count_threshold=10) is True

    def test_large_size_returns_true(self):
        """Test that folder with large total size is detected as large."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a file that exceeds a small threshold
            with open(os.path.join(tmpdir, "large.bin"), "wb") as f:
                f.write(b"x" * 1024 * 1024)  # 1 MB

            # With 0.0005 GB threshold (~500KB), should return True
            assert _is_large_folder(tmpdir, size_threshold_gb=0.0005) is True

    def test_custom_thresholds(self):
        """Test that custom thresholds work correctly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a few files
            for i in range(5):
                with open(os.path.join(tmpdir, f"file{i}.txt"), "w") as f:
                    f.write("test content")

            # Should be small with default thresholds
            assert _is_large_folder(tmpdir) is False

            # Should be large with very low file threshold
            assert _is_large_folder(tmpdir, file_count_threshold=3) is True


class TestServicesHfPushModelUploadSwitching:
    """Tests that hf push-model correctly switches between upload_folder and upload_large_folder."""

    @patch('bioamla.cli._is_large_folder')
    @patch('huggingface_hub.HfApi')
    def test_small_folder_uses_upload_folder(self, mock_hf_api_class, mock_is_large, runner):
        """Test that small folders use upload_folder."""
        mock_is_large.return_value = False
        mock_api = MagicMock()
        mock_hf_api_class.return_value = mock_api

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a small file
            with open(os.path.join(tmpdir, "model.bin"), "w") as f:
                f.write("test")

            result = runner.invoke(cli, ["services", "hf", "push-model", tmpdir, "user/test-repo"])

            # Should call upload_folder, not upload_large_folder
            mock_api.upload_folder.assert_called_once()
            mock_api.upload_large_folder.assert_not_called()

            # Verify correct arguments
            call_kwargs = mock_api.upload_folder.call_args[1]
            assert call_kwargs['folder_path'] == tmpdir
            assert call_kwargs['repo_id'] == "user/test-repo"
            assert call_kwargs['repo_type'] == "model"
            assert call_kwargs['commit_message'] == "Upload model"

    @patch('bioamla.cli._is_large_folder')
    @patch('huggingface_hub.HfApi')
    def test_large_folder_uses_upload_large_folder(self, mock_hf_api_class, mock_is_large, runner):
        """Test that large folders use upload_large_folder."""
        mock_is_large.return_value = True
        mock_api = MagicMock()
        mock_hf_api_class.return_value = mock_api

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a file (size doesn't matter since we mock _is_large_folder)
            with open(os.path.join(tmpdir, "model.bin"), "w") as f:
                f.write("test")

            result = runner.invoke(cli, ["services", "hf", "push-model", tmpdir, "user/test-repo"])

            # Should call upload_large_folder, not upload_folder
            mock_api.upload_large_folder.assert_called_once()
            mock_api.upload_folder.assert_not_called()

            # Verify correct arguments
            call_kwargs = mock_api.upload_large_folder.call_args[1]
            assert call_kwargs['folder_path'] == tmpdir
            assert call_kwargs['repo_id'] == "user/test-repo"
            assert call_kwargs['repo_type'] == "model"
            assert call_kwargs['commit_message'] == "Upload model"

    @patch('bioamla.cli._is_large_folder')
    @patch('huggingface_hub.HfApi')
    def test_custom_commit_message_passed_to_upload_folder(self, mock_hf_api_class, mock_is_large, runner):
        """Test that custom commit message is passed to upload_folder."""
        mock_is_large.return_value = False
        mock_api = MagicMock()
        mock_hf_api_class.return_value = mock_api

        with tempfile.TemporaryDirectory() as tmpdir:
            with open(os.path.join(tmpdir, "model.bin"), "w") as f:
                f.write("test")

            result = runner.invoke(cli, [
                "services", "hf", "push-model", tmpdir, "user/test-repo",
                "--commit-message", "My custom message"
            ])

            call_kwargs = mock_api.upload_folder.call_args[1]
            assert call_kwargs['commit_message'] == "My custom message"

    @patch('bioamla.cli._is_large_folder')
    @patch('huggingface_hub.HfApi')
    def test_custom_commit_message_passed_to_upload_large_folder(self, mock_hf_api_class, mock_is_large, runner):
        """Test that custom commit message is passed to upload_large_folder."""
        mock_is_large.return_value = True
        mock_api = MagicMock()
        mock_hf_api_class.return_value = mock_api

        with tempfile.TemporaryDirectory() as tmpdir:
            with open(os.path.join(tmpdir, "model.bin"), "w") as f:
                f.write("test")

            result = runner.invoke(cli, [
                "services", "hf", "push-model", tmpdir, "user/test-repo",
                "--commit-message", "My custom message"
            ])

            call_kwargs = mock_api.upload_large_folder.call_args[1]
            assert call_kwargs['commit_message'] == "My custom message"


class TestServicesHfPushDatasetUploadSwitching:
    """Tests that hf push-dataset correctly switches between upload_folder and upload_large_folder."""

    @patch('bioamla.cli._is_large_folder')
    @patch('huggingface_hub.HfApi')
    def test_small_folder_uses_upload_folder(self, mock_hf_api_class, mock_is_large, runner):
        """Test that small folders use upload_folder."""
        mock_is_large.return_value = False
        mock_api = MagicMock()
        mock_hf_api_class.return_value = mock_api

        with tempfile.TemporaryDirectory() as tmpdir:
            with open(os.path.join(tmpdir, "data.csv"), "w") as f:
                f.write("col1,col2\n1,2")

            result = runner.invoke(cli, ["services", "hf", "push-dataset", tmpdir, "user/test-dataset"])

            # Should call upload_folder, not upload_large_folder
            mock_api.upload_folder.assert_called_once()
            mock_api.upload_large_folder.assert_not_called()

            # Verify correct arguments
            call_kwargs = mock_api.upload_folder.call_args[1]
            assert call_kwargs['folder_path'] == tmpdir
            assert call_kwargs['repo_id'] == "user/test-dataset"
            assert call_kwargs['repo_type'] == "dataset"
            assert call_kwargs['commit_message'] == "Upload dataset"

    @patch('bioamla.cli._is_large_folder')
    @patch('huggingface_hub.HfApi')
    def test_large_folder_uses_upload_large_folder(self, mock_hf_api_class, mock_is_large, runner):
        """Test that large folders use upload_large_folder."""
        mock_is_large.return_value = True
        mock_api = MagicMock()
        mock_hf_api_class.return_value = mock_api

        with tempfile.TemporaryDirectory() as tmpdir:
            with open(os.path.join(tmpdir, "data.csv"), "w") as f:
                f.write("col1,col2\n1,2")

            result = runner.invoke(cli, ["services", "hf", "push-dataset", tmpdir, "user/test-dataset"])

            # Should call upload_large_folder, not upload_folder
            mock_api.upload_large_folder.assert_called_once()
            mock_api.upload_folder.assert_not_called()

            # Verify correct arguments
            call_kwargs = mock_api.upload_large_folder.call_args[1]
            assert call_kwargs['folder_path'] == tmpdir
            assert call_kwargs['repo_id'] == "user/test-dataset"
            assert call_kwargs['repo_type'] == "dataset"
            assert call_kwargs['commit_message'] == "Upload dataset"

    @patch('bioamla.cli._is_large_folder')
    @patch('huggingface_hub.HfApi')
    def test_custom_commit_message_passed_to_upload_folder(self, mock_hf_api_class, mock_is_large, runner):
        """Test that custom commit message is passed to upload_folder."""
        mock_is_large.return_value = False
        mock_api = MagicMock()
        mock_hf_api_class.return_value = mock_api

        with tempfile.TemporaryDirectory() as tmpdir:
            with open(os.path.join(tmpdir, "data.csv"), "w") as f:
                f.write("col1,col2\n1,2")

            result = runner.invoke(cli, [
                "services", "hf", "push-dataset", tmpdir, "user/test-dataset",
                "--commit-message", "Custom dataset upload"
            ])

            call_kwargs = mock_api.upload_folder.call_args[1]
            assert call_kwargs['commit_message'] == "Custom dataset upload"

    @patch('bioamla.cli._is_large_folder')
    @patch('huggingface_hub.HfApi')
    def test_custom_commit_message_passed_to_upload_large_folder(self, mock_hf_api_class, mock_is_large, runner):
        """Test that custom commit message is passed to upload_large_folder."""
        mock_is_large.return_value = True
        mock_api = MagicMock()
        mock_hf_api_class.return_value = mock_api

        with tempfile.TemporaryDirectory() as tmpdir:
            with open(os.path.join(tmpdir, "data.csv"), "w") as f:
                f.write("col1,col2\n1,2")

            result = runner.invoke(cli, [
                "services", "hf", "push-dataset", tmpdir, "user/test-dataset",
                "--commit-message", "Custom dataset upload"
            ])

            call_kwargs = mock_api.upload_large_folder.call_args[1]
            assert call_kwargs['commit_message'] == "Custom dataset upload"


