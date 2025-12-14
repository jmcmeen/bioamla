"""
Unit tests for bioamla.core.ast module.
"""

import pytest
from unittest.mock import patch, MagicMock
import torch


class TestLoadPretrainedAstModel:
    """Tests for load_pretrained_ast_model function."""

    def test_local_path_exists_uses_local_files_only(self, temp_dir):
        """Test that existing local paths use local_files_only=True."""
        model_dir = temp_dir / "my_model"
        model_dir.mkdir()

        with patch("bioamla.core.ast.AutoModelForAudioClassification") as mock_model:
            mock_model.from_pretrained.return_value = MagicMock()

            from bioamla.core.ast import load_pretrained_ast_model
            load_pretrained_ast_model(str(model_dir))

            mock_model.from_pretrained.assert_called_once_with(
                str(model_dir),
                device_map="auto",
                local_files_only=True,
                torch_dtype=None
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
                local_files_only=True,
                torch_dtype=None
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
                local_files_only=True,
                torch_dtype=None
            )

    def test_huggingface_repo_id_does_not_use_local_files_only(self):
        """Test that HuggingFace repo IDs like 'org/model' don't use local_files_only."""
        with patch("bioamla.core.ast.AutoModelForAudioClassification") as mock_model:
            mock_model.from_pretrained.return_value = MagicMock()

            from bioamla.core.ast import load_pretrained_ast_model
            load_pretrained_ast_model("MIT/ast-finetuned-audioset-10-10-0.4593")

            mock_model.from_pretrained.assert_called_once_with(
                "MIT/ast-finetuned-audioset-10-10-0.4593",
                device_map="auto",
                torch_dtype=None
            )

    def test_huggingface_repo_with_org_does_not_use_local_files_only(self):
        """Test that org/model format is recognized as HuggingFace repo ID."""
        with patch("bioamla.core.ast.AutoModelForAudioClassification") as mock_model:
            mock_model.from_pretrained.return_value = MagicMock()

            from bioamla.core.ast import load_pretrained_ast_model
            load_pretrained_ast_model("facebook/wav2vec2-base")

            mock_model.from_pretrained.assert_called_once_with(
                "facebook/wav2vec2-base",
                device_map="auto",
                torch_dtype=None
            )

    def test_simple_model_name_does_not_use_local_files_only(self):
        """Test that simple model names without paths don't use local_files_only."""
        with patch("bioamla.core.ast.AutoModelForAudioClassification") as mock_model:
            mock_model.from_pretrained.return_value = MagicMock()

            from bioamla.core.ast import load_pretrained_ast_model
            load_pretrained_ast_model("bert-base-uncased")

            mock_model.from_pretrained.assert_called_once_with(
                "bert-base-uncased",
                device_map="auto",
                torch_dtype=None
            )

    def test_fp16_sets_torch_dtype(self):
        """Test that use_fp16=True sets torch_dtype to float16."""
        with patch("bioamla.core.ast.AutoModelForAudioClassification") as mock_model:
            mock_model.from_pretrained.return_value = MagicMock()

            from bioamla.core.ast import load_pretrained_ast_model
            load_pretrained_ast_model("bert-base-uncased", use_fp16=True)

            mock_model.from_pretrained.assert_called_once_with(
                "bert-base-uncased",
                device_map="auto",
                torch_dtype=torch.float16
            )

    def test_compile_calls_torch_compile(self):
        """Test that use_compile=True calls torch.compile on the model."""
        mock_model_instance = MagicMock()

        with patch("bioamla.core.ast.AutoModelForAudioClassification") as mock_model:
            mock_model.from_pretrained.return_value = mock_model_instance

            with patch("torch.compile") as mock_compile:
                mock_compile.return_value = mock_model_instance

                from bioamla.core.ast import load_pretrained_ast_model
                load_pretrained_ast_model("bert-base-uncased", use_compile=True)

                mock_compile.assert_called_once_with(mock_model_instance)


class TestInferenceConfig:
    """Tests for InferenceConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        from bioamla.core.ast import InferenceConfig

        config = InferenceConfig()
        assert config.batch_size == 8
        assert config.use_fp16 is False
        assert config.use_compile is False
        assert config.num_workers == 1

    def test_custom_values(self):
        """Test custom configuration values."""
        from bioamla.core.ast import InferenceConfig

        config = InferenceConfig(
            batch_size=16,
            use_fp16=True,
            use_compile=True,
            num_workers=4
        )
        assert config.batch_size == 16
        assert config.use_fp16 is True
        assert config.use_compile is True
        assert config.num_workers == 4


class TestCachedFeatureExtractor:
    """Tests for get_cached_feature_extractor function."""

    def test_returns_ast_feature_extractor(self):
        """Test that it returns an ASTFeatureExtractor instance."""
        with patch("bioamla.core.ast.ASTFeatureExtractor") as mock_extractor:
            mock_instance = MagicMock()
            mock_extractor.return_value = mock_instance

            from bioamla.core.ast import get_cached_feature_extractor
            get_cached_feature_extractor.cache_clear()

            result = get_cached_feature_extractor()

            mock_extractor.assert_called_once()
            assert result == mock_instance

    def test_caches_result(self):
        """Test that repeated calls return cached instance."""
        with patch("bioamla.core.ast.ASTFeatureExtractor") as mock_extractor:
            mock_instance = MagicMock()
            mock_extractor.return_value = mock_instance

            from bioamla.core.ast import get_cached_feature_extractor
            get_cached_feature_extractor.cache_clear()

            result1 = get_cached_feature_extractor()
            result2 = get_cached_feature_extractor()

            assert mock_extractor.call_count == 1
            assert result1 is result2

    def test_loads_from_model_path(self):
        """Test loading feature extractor from a model path."""
        with patch("bioamla.core.ast.ASTFeatureExtractor") as mock_extractor:
            mock_instance = MagicMock()
            mock_extractor.from_pretrained.return_value = mock_instance

            from bioamla.core.ast import get_cached_feature_extractor
            get_cached_feature_extractor.cache_clear()

            result = get_cached_feature_extractor("my/model")

            mock_extractor.from_pretrained.assert_called_once_with("my/model")
            assert result == mock_instance


class TestExtractFeatures:
    """Tests for extract_features function."""

    def test_extracts_features_from_waveform(self):
        """Test feature extraction from a waveform tensor."""
        mock_extractor = MagicMock()
        mock_inputs = MagicMock()
        mock_inputs.input_values = torch.zeros(1, 100)
        mock_extractor.return_value = mock_inputs

        waveform = torch.randn(1, 16000)

        from bioamla.core.ast import extract_features
        result = extract_features(
            waveform,
            sample_rate=16000,
            feature_extractor=mock_extractor,
            device=torch.device('cpu')
        )

        mock_extractor.assert_called_once()
        assert result is not None


class TestAstPredict:
    """Tests for ast_predict function."""

    def test_returns_predicted_label(self):
        """Test that ast_predict returns the predicted label."""
        mock_model = MagicMock()
        mock_outputs = MagicMock()
        mock_outputs.logits = torch.tensor([[0.1, 0.9, 0.2]])
        mock_model.return_value = mock_outputs
        mock_model.config.id2label = {0: "class_a", 1: "class_b", 2: "class_c"}

        input_values = torch.randn(1, 100)

        from bioamla.core.ast import ast_predict
        result = ast_predict(input_values, mock_model)

        assert result == "class_b"


class TestAstPredictBatch:
    """Tests for ast_predict_batch function."""

    def test_returns_list_of_predictions(self):
        """Test that ast_predict_batch returns a list of labels."""
        mock_model = MagicMock()
        mock_outputs = MagicMock()
        mock_outputs.logits = torch.tensor([
            [0.9, 0.1, 0.2],
            [0.1, 0.8, 0.2],
            [0.1, 0.2, 0.9]
        ])
        mock_model.return_value = mock_outputs
        mock_model.config.id2label = {0: "class_a", 1: "class_b", 2: "class_c"}

        input_values = torch.randn(3, 100)

        from bioamla.core.ast import ast_predict_batch
        result = ast_predict_batch(input_values, mock_model)

        assert result == ["class_a", "class_b", "class_c"]
        assert len(result) == 3


class TestProcessSegmentsBatched:
    """Tests for _process_segments_batched function."""

    def test_batches_segments_correctly(self):
        """Test that segments are batched and processed correctly."""
        mock_model = MagicMock()
        mock_outputs = MagicMock()
        mock_outputs.logits = torch.tensor([[0.9, 0.1], [0.1, 0.9]])
        mock_model.return_value = mock_outputs
        mock_model.config.id2label = {0: "species_a", 1: "species_b"}

        mock_extractor = MagicMock()
        mock_inputs = MagicMock()
        mock_inputs.input_values = torch.zeros(1, 100)
        mock_extractor.return_value = mock_inputs

        from bioamla.core.ast import _process_segments_batched, InferenceConfig

        segments = [
            (torch.randn(1, 16000), 0, 16000),
            (torch.randn(1, 16000), 16000, 32000),
        ]

        config = InferenceConfig(batch_size=2)

        rows = _process_segments_batched(
            filepath="test.wav",
            segments=segments,
            model=mock_model,
            freq=16000,
            config=config,
            feature_extractor=mock_extractor,
            device=torch.device('cpu')
        )

        assert len(rows) == 2
        assert rows[0]['filepath'] == "test.wav"
        assert rows[0]['start'] == 0
        assert rows[0]['stop'] == 16000
        assert rows[1]['start'] == 16000
        assert rows[1]['stop'] == 32000


class TestWaveFileBatchInference:
    """Tests for wave_file_batch_inference function."""

    def test_uses_config_for_batching(self, temp_dir):
        """Test that inference config is used correctly."""
        from bioamla.core.ast import InferenceConfig

        config = InferenceConfig(batch_size=4, num_workers=1)

        mock_model = MagicMock()
        mock_model.parameters.return_value = iter([torch.zeros(1)])

        output_csv = temp_dir / "output.csv"

        with patch("bioamla.core.ast.segmented_wave_file_inference") as mock_inference:
            import pandas as pd
            mock_inference.return_value = pd.DataFrame(
                columns=['filepath', 'start', 'stop', 'prediction']
            )

            from bioamla.core.ast import wave_file_batch_inference
            wave_file_batch_inference(
                wave_files=["test1.wav", "test2.wav"],
                model=mock_model,
                freq=16000,
                clip_seconds=1,
                overlap_seconds=0,
                output_csv=str(output_csv),
                config=config
            )

            assert mock_inference.call_count == 2

    def test_parallel_inference_uses_threadpool(self, temp_dir):
        """Test that parallel inference uses ThreadPoolExecutor."""
        from bioamla.core.ast import InferenceConfig

        config = InferenceConfig(batch_size=4, num_workers=4)

        mock_model = MagicMock()
        mock_model.parameters.return_value = iter([torch.zeros(1)])

        output_csv = temp_dir / "output.csv"

        with patch("bioamla.core.ast.ThreadPoolExecutor") as mock_executor:
            mock_executor_instance = MagicMock()
            mock_executor.return_value.__enter__.return_value = mock_executor_instance
            mock_executor_instance.map.return_value = iter([])

            from bioamla.core.ast import wave_file_batch_inference
            wave_file_batch_inference(
                wave_files=["test1.wav"],
                model=mock_model,
                freq=16000,
                clip_seconds=1,
                overlap_seconds=0,
                output_csv=str(output_csv),
                config=config
            )

            mock_executor.assert_called_once_with(max_workers=4)
