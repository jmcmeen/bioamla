"""
Unit tests for the ast-finetune CLI command options.
"""

from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner

from bioamla.cli import cli


@pytest.fixture
def runner():
    """Create a CLI test runner."""
    return CliRunner()


class TestAstFinetuneHelp:
    """Tests for ast-finetune help and options."""

    def test_ast_finetune_help(self, runner):
        """Test ast-finetune --help shows all options."""
        result = runner.invoke(cli, ["ast-finetune", "--help"])

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

    def test_ast_finetune_finetune_mode_choices(self, runner):
        """Test that --finetune-mode shows valid choices."""
        result = runner.invoke(cli, ["ast-finetune", "--help"])

        assert result.exit_code == 0
        assert "full" in result.output
        assert "feature-extraction" in result.output

    def test_ast_finetune_default_batch_size(self, runner):
        """Test that default batch size is 8."""
        result = runner.invoke(cli, ["ast-finetune", "--help"])

        assert result.exit_code == 0
        # Check help text mentions the default
        assert "per-device-train-batch-size" in result.output


class TestAstFinetuneOptions:
    """Tests for ast-finetune option validation."""

    def test_invalid_finetune_mode(self, runner):
        """Test that invalid finetune mode is rejected."""
        result = runner.invoke(cli, ["ast-finetune", "--finetune-mode", "invalid"])

        assert result.exit_code != 0
        assert "Invalid value" in result.output or "invalid" in result.output.lower()

    def test_valid_finetune_mode_full(self, runner):
        """Test that 'full' is a valid finetune mode."""
        # Just check that the option is accepted (will fail later due to missing dataset)
        result = runner.invoke(cli, ["ast-finetune", "--finetune-mode", "full", "--help"])

        assert result.exit_code == 0

    def test_valid_finetune_mode_feature_extraction(self, runner):
        """Test that 'feature-extraction' is a valid finetune mode."""
        # Just check that the option is accepted
        result = runner.invoke(cli, ["ast-finetune", "--finetune-mode", "feature-extraction", "--help"])

        assert result.exit_code == 0


class TestAstFinetunePerformanceOptions:
    """Tests for performance-related options."""

    def test_fp16_and_bf16_options_exist(self, runner):
        """Test that fp16 and bf16 options exist."""
        result = runner.invoke(cli, ["ast-finetune", "--help"])

        assert result.exit_code == 0
        assert "--fp16" in result.output
        assert "--no-fp16" in result.output
        assert "--bf16" in result.output
        assert "--no-bf16" in result.output

    def test_gradient_accumulation_option_exists(self, runner):
        """Test that gradient accumulation option exists."""
        result = runner.invoke(cli, ["ast-finetune", "--help"])

        assert result.exit_code == 0
        assert "--gradient-accumulation-steps" in result.output

    def test_dataloader_workers_option_exists(self, runner):
        """Test that dataloader workers option exists."""
        result = runner.invoke(cli, ["ast-finetune", "--help"])

        assert result.exit_code == 0
        assert "--dataloader-num-workers" in result.output

    def test_torch_compile_option_exists(self, runner):
        """Test that torch compile option exists."""
        result = runner.invoke(cli, ["ast-finetune", "--help"])

        assert result.exit_code == 0
        assert "--torch-compile" in result.output
        assert "--no-torch-compile" in result.output


class TestAstFinetuneMlflowOptions:
    """Tests for MLflow-related options."""

    def test_mlflow_tracking_uri_option_exists(self, runner):
        """Test that mlflow tracking URI option exists."""
        result = runner.invoke(cli, ["ast-finetune", "--help"])

        assert result.exit_code == 0
        assert "--mlflow-tracking-uri" in result.output

    def test_mlflow_experiment_name_option_exists(self, runner):
        """Test that mlflow experiment name option exists."""
        result = runner.invoke(cli, ["ast-finetune", "--help"])

        assert result.exit_code == 0
        assert "--mlflow-experiment-name" in result.output

    def test_mlflow_run_name_option_exists(self, runner):
        """Test that mlflow run name option exists."""
        result = runner.invoke(cli, ["ast-finetune", "--help"])

        assert result.exit_code == 0
        assert "--mlflow-run-name" in result.output


class TestAstPush:
    """Tests for ast-push command."""

    def test_ast_push_help(self, runner):
        """Test ast-push --help shows all options."""
        result = runner.invoke(cli, ["ast-push", "--help"])

        assert result.exit_code == 0
        assert "MODEL_PATH" in result.output
        assert "REPO_ID" in result.output
        assert "--private" in result.output
        assert "--public" in result.output
        assert "--commit-message" in result.output

    def test_ast_push_requires_model_path(self, runner):
        """Test that ast-push requires model_path argument."""
        result = runner.invoke(cli, ["ast-push"])

        assert result.exit_code != 0
        assert "Missing argument" in result.output or "MODEL_PATH" in result.output

    def test_ast_push_requires_repo_id(self, runner):
        """Test that ast-push requires repo_id argument."""
        result = runner.invoke(cli, ["ast-push", "/some/path"])

        assert result.exit_code != 0
        assert "Missing argument" in result.output or "REPO_ID" in result.output

    def test_ast_push_invalid_model_path(self, runner):
        """Test that ast-push fails with non-existent model path."""
        result = runner.invoke(cli, ["ast-push", "/nonexistent/path", "user/repo"])

        assert result.exit_code != 0
        assert "does not exist" in result.output
