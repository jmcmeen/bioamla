# services/ast.py
"""
Service for Audio Spectrogram Transformer (AST) model operations.

This service provides a high-level interface for AST model operations including
prediction, evaluation, and embedding extraction. All AST-specific logic is
contained within this service layer.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional

from bioamla.models.ast import EvaluationResult, PredictionResult, TrainResult
from bioamla.repository.protocol import FileRepositoryProtocol

from .base import BaseService, ServiceResult


class ASTService(BaseService):
    """
    Service for AST (Audio Spectrogram Transformer) model operations.

    Provides ServiceResult-wrapped methods for AST model operations.
    """

    def __init__(self, file_repository: FileRepositoryProtocol) -> None:
        """Initialize AST service.

        Args:
            file_repository: Repository for file operations (required)
        """
        super().__init__(file_repository=file_repository)
        self._model = None
        self._model_path = None

    def predict(
        self,
        filepath: str,
        model_path: str = "bioamla/scp-frogs",
        resample_freq: int = 16000,
    ) -> ServiceResult[List[PredictionResult]]:
        """
        Run prediction on a single audio file.

        Args:
            filepath: Path to audio file
            model_path: Path to model or HuggingFace identifier
            resample_freq: Target sample rate

        Returns:
            ServiceResult containing list of PredictionResult
        """
        error = self._validate_input_path(filepath)
        if error:
            return ServiceResult.fail(error)

        try:
            from bioamla.services.inference import InferenceService

            service = InferenceService(
                file_repository=self.file_repository,
                model_path=model_path,
            )
            result = service.predict(filepath=filepath)

            if not result.success:
                return ServiceResult.fail(result.error)

            predictions = [
                PredictionResult(
                    filepath=filepath,
                    label=pred.predicted_label,
                    confidence=pred.confidence,
                )
                for pred in result.data
            ]

            return ServiceResult.ok(
                data=predictions,
                message=f"Generated {len(predictions)} predictions",
            )
        except Exception as e:
            return ServiceResult.fail(f"Prediction failed: {e}")

    def train(
        self,
        train_dataset: str,
        output_dir: str,
        base_model: str = "MIT/ast-finetuned-audioset-10-10-0.4593",
        split: str = "train",
        category_label_column: str = "category",
        learning_rate: float = 5e-5,
        num_epochs: int = 1,
        batch_size: int = 8,
        fp16: bool = False,
        bf16: bool = False,
        finetune_mode: str = "full",
        push_to_hub: bool = False,
    ) -> ServiceResult[TrainResult]:
        """
        Fine-tune an AST model on a custom dataset.

        Args:
            train_dataset: Training dataset from HuggingFace Hub
            output_dir: Output directory for training outputs
            base_model: Base model to fine-tune
            split: Dataset split to use
            category_label_column: Column name for category labels
            learning_rate: Learning rate
            num_epochs: Number of training epochs
            batch_size: Training batch size
            fp16: Use FP16 mixed precision
            bf16: Use BF16 mixed precision
            finetune_mode: Training mode (full or feature-extraction)
            push_to_hub: Push model to HuggingFace Hub

        Returns:
            ServiceResult containing TrainResult
        """
        try:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)

            # Note: Full training implementation is complex and should be
            # delegated to the CLI command which uses transformers Trainer
            # This is a placeholder for the service interface

            return ServiceResult.fail(
                "Full training not yet implemented in service layer. "
                "Use CLI command 'bioamla models ast train' directly."
            )
        except Exception as e:
            return ServiceResult.fail(f"Training failed: {e}")

    def evaluate(
        self,
        audio_dir: str,
        model_path: str,
        ground_truth_csv: str,
        file_column: str = "file_name",
        label_column: str = "label",
        resample_freq: int = 16000,
        batch_size: int = 8,
        fp16: bool = False,
    ) -> ServiceResult[EvaluationResult]:
        """
        Evaluate an AST model on a directory of audio files.

        Args:
            audio_dir: Directory containing audio files
            model_path: Path to model
            ground_truth_csv: Path to CSV with ground truth labels
            file_column: Column name for file names
            label_column: Column name for labels
            resample_freq: Target sample rate
            batch_size: Batch size for inference
            fp16: Use half-precision inference

        Returns:
            ServiceResult containing EvaluationResult
        """
        error = self._validate_input_path(audio_dir)
        if error:
            return ServiceResult.fail(error)

        error = self._validate_input_path(ground_truth_csv)
        if error:
            return ServiceResult.fail(error)

        try:
            result = self._evaluate_directory(
                audio_dir=audio_dir,
                model_path=model_path,
                ground_truth_csv=ground_truth_csv,
                gt_file_column=file_column,
                gt_label_column=label_column,
                resample_freq=resample_freq,
                use_fp16=fp16,
            )

            eval_result = EvaluationResult(
                accuracy=result["accuracy"],
                precision=result["precision"],
                recall=result["recall"],
                f1_score=result["f1_score"],
                total_samples=result["total_samples"],
            )

            return ServiceResult.ok(
                data=eval_result,
                message=f"Accuracy: {result['accuracy']:.4f}, F1: {result['f1_score']:.4f}",
            )
        except Exception as e:
            return ServiceResult.fail(f"Evaluation failed: {e}")

    def _evaluate_directory(
        self,
        audio_dir: str,
        model_path: str,
        ground_truth_csv: str,
        gt_file_column: str = "file_name",
        gt_label_column: str = "label",
        resample_freq: int = 16000,
        use_fp16: bool = False,
    ) -> Dict[str, Any]:
        """
        Internal method to evaluate a model on a directory of audio files.

        Returns dict with accuracy, precision, recall, f1_score, total_samples.
        """
        import torch

        # Lazy imports for AST functionality
        from bioamla.core.torchaudio import load_waveform_tensor, resample_waveform_tensor
        from bioamla.core.ast import (
            ast_predict,
            extract_features,
            get_cached_feature_extractor,
            load_pretrained_ast_model,
        )

        audio_path = Path(audio_dir)
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio directory not found: {audio_dir}")

        # Load ground truth
        ground_truth = self._load_ground_truth(ground_truth_csv, gt_file_column, gt_label_column)

        # Find audio files that have ground truth
        audio_extensions = [".wav", ".mp3", ".flac", ".ogg", ".m4a"]
        audio_files = []
        for ext in audio_extensions:
            audio_files.extend(audio_path.rglob(f"*{ext}"))

        # Filter to only files with ground truth
        matched_files = [(f, ground_truth[f.name]) for f in audio_files if f.name in ground_truth]

        if not matched_files:
            raise ValueError("No audio files found that match ground truth labels")

        # Load model
        model = load_pretrained_ast_model(model_path, use_fp16=use_fp16)
        feature_extractor = get_cached_feature_extractor(model_path)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Run inference
        y_true = []
        y_pred = []

        for audio_file, true_label in matched_files:
            try:
                waveform, orig_freq = load_waveform_tensor(str(audio_file))
                waveform = resample_waveform_tensor(waveform, orig_freq, resample_freq)
                input_values = extract_features(waveform, resample_freq, feature_extractor, device)

                if use_fp16 and device.type == "cuda":
                    input_values = input_values.half()

                prediction = ast_predict(input_values, model)

                y_true.append(true_label)
                y_pred.append(prediction)

            except Exception:
                # Skip files that fail to process
                pass

        if not y_true:
            raise ValueError("No predictions were generated")

        return self._compute_metrics(y_true, y_pred)

    def _load_ground_truth(
        self,
        csv_path: str,
        file_column: str,
        label_column: str,
    ) -> Dict[str, str]:
        """Load ground truth labels from a CSV file."""
        import pandas as pd

        df = pd.read_csv(csv_path)

        if file_column not in df.columns:
            raise ValueError(f"Column '{file_column}' not found in CSV. Available: {list(df.columns)}")
        if label_column not in df.columns:
            raise ValueError(f"Column '{label_column}' not found in CSV. Available: {list(df.columns)}")

        ground_truth = {}
        for _, row in df.iterrows():
            file_name = Path(row[file_column]).name
            ground_truth[file_name] = str(row[label_column])

        return ground_truth

    def _compute_metrics(
        self,
        y_true: List[str],
        y_pred: List[str],
    ) -> Dict[str, Any]:
        """Compute classification metrics from predictions and ground truth."""
        import numpy as np

        if len(y_true) != len(y_pred):
            raise ValueError(f"Length mismatch: y_true ({len(y_true)}) != y_pred ({len(y_pred)})")

        labels = sorted(set(y_true) | set(y_pred))
        label_to_idx = {label: idx for idx, label in enumerate(labels)}
        n_classes = len(labels)

        # Build confusion matrix
        confusion = np.zeros((n_classes, n_classes), dtype=np.int64)
        for true_label, pred_label in zip(y_true, y_pred):
            if true_label in label_to_idx and pred_label in label_to_idx:
                true_idx = label_to_idx[true_label]
                pred_idx = label_to_idx[pred_label]
                confusion[true_idx, pred_idx] += 1

        # Compute overall accuracy
        total_samples = len(y_true)
        correct = sum(1 for t, p in zip(y_true, y_pred) if t == p)
        accuracy = correct / total_samples if total_samples > 0 else 0.0

        # Compute per-class metrics for macro average
        precisions = []
        recalls = []
        f1_scores = []

        for idx in range(n_classes):
            tp = confusion[idx, idx]
            fp = confusion[:, idx].sum() - tp
            fn = confusion[idx, :].sum() - tp

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

            support = confusion[idx, :].sum()
            if support > 0:
                precisions.append(precision)
                recalls.append(recall)
                f1_scores.append(f1)

        macro_precision = float(np.mean(precisions)) if precisions else 0.0
        macro_recall = float(np.mean(recalls)) if recalls else 0.0
        macro_f1 = float(np.mean(f1_scores)) if f1_scores else 0.0

        return {
            "accuracy": accuracy,
            "precision": macro_precision,
            "recall": macro_recall,
            "f1_score": macro_f1,
            "total_samples": total_samples,
            "correct_predictions": correct,
        }

    def list_models(self) -> ServiceResult[List[str]]:
        """
        List available AST models.

        Returns:
            ServiceResult containing list of model identifiers
        """
        try:
            models = [
                "MIT/ast-finetuned-audioset-10-10-0.4593",
                "bioamla/scp-frogs",
            ]

            return ServiceResult.ok(
                data=models,
                message=f"Found {len(models)} AST models",
            )
        except Exception as e:
            return ServiceResult.fail(f"Failed to list models: {e}")

    def extract_embeddings(
        self,
        filepath: str,
        model_path: str,
        layer: Optional[str] = None,
        sample_rate: int = 16000,
    ) -> ServiceResult[Dict[str, Any]]:
        """
        Extract embeddings from audio using an AST model.

        Args:
            filepath: Path to audio file
            model_path: Path to model or HuggingFace identifier
            layer: Layer to extract embeddings from
            sample_rate: Target sample rate

        Returns:
            ServiceResult containing embeddings info
        """
        error = self._validate_input_path(filepath)
        if error:
            return ServiceResult.fail(error)

        try:
            import torch
            from transformers import ASTFeatureExtractor, AutoModel

            from bioamla.core.torchaudio import load_waveform_tensor, resample_waveform_tensor

            # Load audio
            waveform, orig_sr = load_waveform_tensor(filepath)
            if orig_sr != sample_rate:
                waveform = resample_waveform_tensor(waveform, orig_sr, sample_rate)

            # Load model and feature extractor
            try:
                feature_extractor = ASTFeatureExtractor.from_pretrained(model_path)
            except OSError:
                # Model doesn't have preprocessor_config.json, use default
                feature_extractor = ASTFeatureExtractor()

            model = AutoModel.from_pretrained(model_path)
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model = model.to(device)
            model.eval()

            # Extract features
            inputs = feature_extractor(
                waveform.squeeze().numpy(),
                sampling_rate=sample_rate,
                padding="max_length",
                return_tensors="pt",
            ).to(device)

            # Get embeddings from last hidden state
            with torch.inference_mode():
                outputs = model(**inputs)
                # Use CLS token or mean pool over sequence
                embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()

            return ServiceResult.ok(
                data={
                    "filepath": filepath,
                    "shape": embeddings.shape,
                    "dtype": str(embeddings.dtype),
                },
                message=f"Extracted embeddings with shape {embeddings.shape}",
                embeddings=embeddings,
            )
        except Exception as e:
            return ServiceResult.fail(f"Embedding extraction failed: {e}")

    def get_model_info(
        self,
        model_path: str,
    ) -> ServiceResult[Dict[str, Any]]:
        """
        Get information about an AST model.

        Args:
            model_path: Path to model or HuggingFace identifier

        Returns:
            ServiceResult containing model info dict
        """
        try:
            from transformers import AutoConfig

            # Load config first (lighter than full model)
            config = AutoConfig.from_pretrained(model_path)

            # Get id2label mapping if available
            id2label = getattr(config, "id2label", {})
            classes = list(id2label.values()) if id2label else []

            info = {
                "path": model_path,
                "model_type": getattr(config, "model_type", "unknown"),
                "num_classes": len(classes) if classes else getattr(config, "num_labels", 0),
                "classes": classes[:10] if classes else [],
                "has_more_classes": len(classes) > 10 if classes else False,
                "hidden_size": getattr(config, "hidden_size", None),
            }

            return ServiceResult.ok(
                data=info,
                message=f"Model has {info['num_classes']} classes",
            )
        except Exception as e:
            return ServiceResult.fail(f"Failed to get model info: {e}")
