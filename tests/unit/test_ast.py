"""
Unit tests for bioamla.core.ast module.
"""

import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock


class TestLoadPretrainedAstModel:
    """Tests for load_pretrained_ast_model function."""

    def test_local_path_exists_uses_local_files_only(self, temp_dir):
        """Test that existing local paths use local_files_only=True."""
        # Create a fake model directory
        model_dir = temp_dir / "my_model"
        model_dir.mkdir()

        with patch("bioamla.core.ast.AutoModelForAudioClassification") as mock_model:
            mock_model.from_pretrained.return_value = MagicMock()

            from bioamla.core.ast import load_pretrained_ast_model
            load_pretrained_ast_model(str(model_dir))

            mock_model.from_pretrained.assert_called_once_with(
                str(model_dir),
                device_map="auto",
                local_files_only=True
            )

    def test_relative_path_starting_with_dot_uses_local_files_only(self):
        """Test that paths starting with ./ use local_files_only=True."""
        with patch("bioamla.core.ast.AutoModelForAudioClassification") as mock_model:
            mock_model.from_pretrained.return_value = MagicMock()

            from bioamla.core.ast import load_pretrained_ast_model
            load_pretrained_ast_model("./models/my_ast")

            mock_model.from_pretrained.assert_called_once_with(
                "./models/my_ast",
                device_map="auto",
                local_files_only=True
            )

    def test_relative_parent_path_uses_local_files_only(self):
        """Test that paths starting with ../ use local_files_only=True."""
        with patch("bioamla.core.ast.AutoModelForAudioClassification") as mock_model:
            mock_model.from_pretrained.return_value = MagicMock()

            from bioamla.core.ast import load_pretrained_ast_model
            load_pretrained_ast_model("../models/my_ast")

            mock_model.from_pretrained.assert_called_once_with(
                "../models/my_ast",
                device_map="auto",
                local_files_only=True
            )

    def test_huggingface_repo_id_does_not_use_local_files_only(self):
        """Test that HuggingFace repo IDs like 'org/model' don't use local_files_only."""
        with patch("bioamla.core.ast.AutoModelForAudioClassification") as mock_model:
            mock_model.from_pretrained.return_value = MagicMock()

            from bioamla.core.ast import load_pretrained_ast_model
            load_pretrained_ast_model("MIT/ast-finetuned-audioset-10-10-0.4593")

            mock_model.from_pretrained.assert_called_once_with(
                "MIT/ast-finetuned-audioset-10-10-0.4593",
                device_map="auto"
            )

    def test_huggingface_repo_with_org_does_not_use_local_files_only(self):
        """Test that org/model format is recognized as HuggingFace repo ID."""
        with patch("bioamla.core.ast.AutoModelForAudioClassification") as mock_model:
            mock_model.from_pretrained.return_value = MagicMock()

            from bioamla.core.ast import load_pretrained_ast_model
            load_pretrained_ast_model("facebook/wav2vec2-base")

            mock_model.from_pretrained.assert_called_once_with(
                "facebook/wav2vec2-base",
                device_map="auto"
            )

    def test_simple_model_name_does_not_use_local_files_only(self):
        """Test that simple model names without paths don't use local_files_only."""
        with patch("bioamla.core.ast.AutoModelForAudioClassification") as mock_model:
            mock_model.from_pretrained.return_value = MagicMock()

            from bioamla.core.ast import load_pretrained_ast_model
            load_pretrained_ast_model("bert-base-uncased")

            mock_model.from_pretrained.assert_called_once_with(
                "bert-base-uncased",
                device_map="auto"
            )
