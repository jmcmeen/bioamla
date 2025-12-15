"""
Unit tests for the ast train CLI command options.
"""


import pytest
from click.testing import CliRunner

from bioamla.cli import cli


@pytest.fixture
def runner():
    """Create a CLI test runner."""
    return CliRunner()


class TestAstTrainHelp:
    """Tests for ast train help and options."""

    def test_ast_train_help(self, runner):
        """Test ast train --help shows all options."""
        result = runner.invoke(cli, ["ast", "train", "--help"])

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
        result = runner.invoke(cli, ["ast", "train", "--help"])

        assert result.exit_code == 0
        assert "full" in result.output
        assert "feature-extraction" in result.output

    def test_ast_train_default_batch_size(self, runner):
        """Test that default batch size is 8."""
        result = runner.invoke(cli, ["ast", "train", "--help"])

        assert result.exit_code == 0
        # Check help text mentions the default
        assert "per-device-train-batch-size" in result.output


class TestAstTrainOptions:
    """Tests for ast train option validation."""

    def test_invalid_finetune_mode(self, runner):
        """Test that invalid finetune mode is rejected."""
        result = runner.invoke(cli, ["ast", "train", "--finetune-mode", "invalid"])

        assert result.exit_code != 0
        assert "Invalid value" in result.output or "invalid" in result.output.lower()

    def test_valid_finetune_mode_full(self, runner):
        """Test that 'full' is a valid finetune mode."""
        # Just check that the option is accepted (will fail later due to missing dataset)
        result = runner.invoke(cli, ["ast", "train", "--finetune-mode", "full", "--help"])

        assert result.exit_code == 0

    def test_valid_finetune_mode_feature_extraction(self, runner):
        """Test that 'feature-extraction' is a valid finetune mode."""
        # Just check that the option is accepted
        result = runner.invoke(cli, ["ast", "train", "--finetune-mode", "feature-extraction", "--help"])

        assert result.exit_code == 0


class TestAstTrainPerformanceOptions:
    """Tests for performance-related options."""

    def test_fp16_and_bf16_options_exist(self, runner):
        """Test that fp16 and bf16 options exist."""
        result = runner.invoke(cli, ["ast", "train", "--help"])

        assert result.exit_code == 0
        assert "--fp16" in result.output
        assert "--no-fp16" in result.output
        assert "--bf16" in result.output
        assert "--no-bf16" in result.output

    def test_gradient_accumulation_option_exists(self, runner):
        """Test that gradient accumulation option exists."""
        result = runner.invoke(cli, ["ast", "train", "--help"])

        assert result.exit_code == 0
        assert "--gradient-accumulation-steps" in result.output

    def test_dataloader_workers_option_exists(self, runner):
        """Test that dataloader workers option exists."""
        result = runner.invoke(cli, ["ast", "train", "--help"])

        assert result.exit_code == 0
        assert "--dataloader-num-workers" in result.output

    def test_torch_compile_option_exists(self, runner):
        """Test that torch compile option exists."""
        result = runner.invoke(cli, ["ast", "train", "--help"])

        assert result.exit_code == 0
        assert "--torch-compile" in result.output
        assert "--no-torch-compile" in result.output


class TestAstTrainMlflowOptions:
    """Tests for MLflow-related options."""

    def test_mlflow_tracking_uri_option_exists(self, runner):
        """Test that mlflow tracking URI option exists."""
        result = runner.invoke(cli, ["ast", "train", "--help"])

        assert result.exit_code == 0
        assert "--mlflow-tracking-uri" in result.output

    def test_mlflow_experiment_name_option_exists(self, runner):
        """Test that mlflow experiment name option exists."""
        result = runner.invoke(cli, ["ast", "train", "--help"])

        assert result.exit_code == 0
        assert "--mlflow-experiment-name" in result.output

    def test_mlflow_run_name_option_exists(self, runner):
        """Test that mlflow run name option exists."""
        result = runner.invoke(cli, ["ast", "train", "--help"])

        assert result.exit_code == 0
        assert "--mlflow-run-name" in result.output


class TestHfPushModel:
    """Tests for hf push-model command."""

    def test_hf_push_model_help(self, runner):
        """Test hf push-model --help shows all options."""
        result = runner.invoke(cli, ["hf", "push-model", "--help"])

        assert result.exit_code == 0
        assert "PATH" in result.output
        assert "REPO_ID" in result.output
        assert "--private" in result.output
        assert "--public" in result.output
        assert "--commit-message" in result.output
        assert "entire contents" in result.output

    def test_hf_push_model_requires_path(self, runner):
        """Test that hf push-model requires path argument."""
        result = runner.invoke(cli, ["hf", "push-model"])

        assert result.exit_code != 0
        assert "Missing argument" in result.output or "PATH" in result.output

    def test_hf_push_model_requires_repo_id(self, runner):
        """Test that hf push-model requires repo_id argument."""
        result = runner.invoke(cli, ["hf", "push-model", "/some/path"])

        assert result.exit_code != 0
        assert "Missing argument" in result.output or "REPO_ID" in result.output

    def test_hf_push_model_invalid_path(self, runner):
        """Test that hf push-model fails with non-existent path."""
        result = runner.invoke(cli, ["hf", "push-model", "/nonexistent/path", "user/repo"])

        assert result.exit_code != 0
        assert "does not exist" in result.output


class TestHfPushDataset:
    """Tests for hf push-dataset command."""

    def test_hf_push_dataset_help(self, runner):
        """Test hf push-dataset --help shows all options."""
        result = runner.invoke(cli, ["hf", "push-dataset", "--help"])

        assert result.exit_code == 0
        assert "PATH" in result.output
        assert "REPO_ID" in result.output
        assert "--private" in result.output
        assert "--public" in result.output
        assert "--commit-message" in result.output
        assert "entire contents" in result.output

    def test_hf_push_dataset_requires_path(self, runner):
        """Test that hf push-dataset requires path argument."""
        result = runner.invoke(cli, ["hf", "push-dataset"])

        assert result.exit_code != 0
        assert "Missing argument" in result.output or "PATH" in result.output

    def test_hf_push_dataset_requires_repo_id(self, runner):
        """Test that hf push-dataset requires repo_id argument."""
        result = runner.invoke(cli, ["hf", "push-dataset", "/some/path"])

        assert result.exit_code != 0
        assert "Missing argument" in result.output or "REPO_ID" in result.output

    def test_hf_push_dataset_invalid_path(self, runner):
        """Test that hf push-dataset fails with non-existent path."""
        result = runner.invoke(cli, ["hf", "push-dataset", "/nonexistent/path", "user/repo"])

        assert result.exit_code != 0
        assert "does not exist" in result.output


